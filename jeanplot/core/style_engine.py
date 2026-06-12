import inspect
import logging
import types
from contextlib import contextmanager
from typing import Any, Sequence, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from dracon import make_locator_cascade_strategy, register_cascade_strategy
from dracon.symbols import CallableSymbol

from jeanplot.core.style_dialect import parse_jstyle_rule_tree, parse_selector_key
from jeanplot.core.tree_adapter import COMPONENT_ADAPTER

logger = logging.getLogger(__name__)


def _model_type_of(annotation: Any) -> type[BaseModel] | None:
    """The BaseModel subclass in a field annotation (unwrapping `X | None`), else None."""
    if get_origin(annotation) is Union or isinstance(annotation, types.UnionType):
        return next(
            (a for a in get_args(annotation) if inspect.isclass(a) and issubclass(a, BaseModel)),
            None,
        )
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    return None


# selection + specificity + merge are all dracon's now (the locator's CSS specificity
# then snug-chain tiebreak: fewer ancestor-skips, closer MRO). Only the parse stays
# local, so `_looks_like_selector` decides selector-vs-config keys.
_JSTYLE_STRATEGY = make_locator_cascade_strategy(
    "jstyle", adapter=COMPONENT_ADAPTER, input_param="component", parse=parse_selector_key
)
_JSTYLE_FILL_STRATEGY = make_locator_cascade_strategy(
    "jstyle_fill", adapter=COMPONENT_ADAPTER, input_param="component", parse=parse_selector_key
)
register_cascade_strategy(_JSTYLE_STRATEGY)
register_cascade_strategy(_JSTYLE_FILL_STRATEGY)


def _strategy_is_fill(cascade: CallableSymbol | None) -> bool:
    if cascade is None:
        return False
    strat = getattr(cascade, "_cascade_strategy", None)
    return getattr(strat, "name", None) == "jstyle_fill"


def _as_cascade(value: Any) -> CallableSymbol | None:
    """Coerce a value into a jstyle CallableSymbol with a flat rule_tree.

    Nested-rule trees ``{Sel: {Sel2: {prop: v}}}`` flatten into descendant selectors
    ``{Selector("Sel Sel2"): {prop: v}}``.
    """
    from dracon.utils import dict_like

    if value is None:
        return None
    if isinstance(value, CallableSymbol):
        value._rule_tree = parse_jstyle_rule_tree(_resolve_lazies(value._rule_tree))
        return value
    if dict_like(value):
        flat = parse_jstyle_rule_tree(_resolve_lazies(value))
        return CallableSymbol.from_match(flat, _JSTYLE_FILL_STRATEGY, name="jstyle_fill")
    raise TypeError(f"unsupported jstyle value type: {type(value).__name__}")


def merge_jstyle_rules(base: Any, overrides: Any) -> Any:
    """Deep-merge a jstyle rule tree with ``overrides``; ``overrides`` wins at the leaves.

    ``JStyle.update`` *replaces* the active cascade, so overrides must be merged into the
    base first, then applied with a single ``update``. ``base`` may be a raw dict OR an
    already-parsed cascade ``CallableSymbol`` (what every ``load_*_theme`` produces); for a
    cascade, overrides are layered onto its flat ``{Selector: props}`` tree.
    """
    from dracon.utils import dict_like, raw_items

    if isinstance(base, CallableSymbol):
        # Flatten BOTH trees to Locator keys before merging. The override is parsed below;
        # the base rule_tree may still be raw (string keys, nested selectors) if the cascade
        # hasn't been applied yet. Without parsing the base too, a string `"CubeStackPanel"`
        # base key never matches the override's `Locator("CubeStackPanel")`, so the override
        # lands as a SEPARATE same-specificity rule and (under fill) is shadowed by the base
        # whenever they set the same field. Parsing both makes equal selectors collide and
        # deep-merge. `parse_jstyle_rule_tree` is idempotent on already-flat trees.
        merged: dict[Any, Any] = dict(
            parse_jstyle_rule_tree(_resolve_lazies(dict(base._rule_tree or {})))
        )
        ov_tree = parse_jstyle_rule_tree(_resolve_lazies(overrides)) if overrides else {}
        for sel, props in ov_tree.items():
            if sel in merged and dict_like(merged[sel]) and dict_like(props):
                merged[sel] = {**dict(raw_items(merged[sel])), **dict(raw_items(props))}
            else:
                merged[sel] = props
        return CallableSymbol.from_match(merged, _JSTYLE_FILL_STRATEGY, name="jstyle_fill")

    out: dict = dict(raw_items(base)) if dict_like(base) else {}
    if dict_like(overrides):
        for k, v in raw_items(overrides):
            if k in out and dict_like(out[k]) and dict_like(v):
                out[k] = merge_jstyle_rules(out[k], v)
            else:
                out[k] = v
    return out


def _resolve_lazies(value: Any) -> Any:
    """Force-resolve any LazyInterpolable in dict keys / values, leaving live-scope
    (component-bound) lazies untouched.

    dracon's ``resolve_all_lazy`` skips private attributes, so ``CallableSymbol._rule_tree``
    is never walked; we mirror its semantics here against the flat rule_tree on update.
    """
    from dracon.lazy import LazyInterpolable
    from dracon.utils import dict_like, raw_items

    if isinstance(value, LazyInterpolable):
        return value if value._scope_params else value.resolve()
    if dict_like(value):
        out: dict[Any, Any] = {}
        for k, v in raw_items(value):
            if isinstance(k, LazyInterpolable) and not k._scope_params:
                k = k.resolve()
            out[k] = _resolve_lazies(v)
        return out
    if isinstance(value, list):
        return [_resolve_lazies(v) for v in value]
    return value


class JStyle:
    def __init__(self, value: Any = None):
        self._cascade: CallableSymbol | None = None
        if value is not None:
            self.update(value)

    def update(self, value: Any):
        """Replace the active cascade. Accepts a CallableSymbol or a nested dict."""
        self._cascade = _as_cascade(value)

    def apply_one(self, component: Any):
        if self._cascade is None or component is None:
            return component
        props = self._cascade.invoke(component=component)
        clobber = not _strategy_is_fill(self._cascade)
        for path, value in props.items():
            if isinstance(value, BaseModel):
                value = value.model_copy(deep=True)  # cascade rule is a spec, not a singleton
            self._set_property(component, path, value, clobber=clobber)
        self._apply_leaves(component)
        return component

    def _apply_leaves(self, component: Any):
        """Resolve a component's `CascadeLeaf` config fields (e.g. `SmoothGrid`) against
        the cascade with the component as parent, so a bare `LeafType:` rule reaches them.
        Component-typed fields (parent/children/anchors) are tree, not config -- `apply`
        owns those; skip them to avoid walking back up the tree."""
        from jeanplot.core.component import Component
        from jeanplot.core.models import CascadeLeaf

        if not isinstance(component, BaseModel):
            return
        for name in type(component).model_fields:
            if name == "parent":  # back-reference, not a config child -- would recurse forever
                continue
            leaf = getattr(component, name, None)
            if isinstance(leaf, CascadeLeaf) and not isinstance(leaf, Component):
                leaf.parent = component
                self.apply_one(leaf)

    def apply(self, component: Any):
        if component is None:
            return None
        self.apply_one(component)
        for child in getattr(component, "children", None) or []:
            if getattr(child, "parent", None) is not component:
                child.parent = component
            self.apply(child)
        return component

    def clear(self):
        self._cascade = None

    @contextmanager
    def context(self, value: Any):
        old = self._cascade
        self._cascade = _as_cascade(value)
        try:
            yield
        finally:
            self._cascade = old

    __call__ = context

    def _set_property(
        self, component: Any, property_name: str, value: Any, *, clobber: bool = True
    ):
        comp_id = getattr(component, "id", "N/A")
        comp_cls = component.__class__.__name__

        parts = property_name.split(".")
        target_obj = component
        attr_to_set = parts[-1]

        fill_field_set_by_user = False

        try:
            for part in parts[:-1]:
                current_intermediate = getattr(target_obj, part)
                if current_intermediate is None:
                    field_info = getattr(type(target_obj), "model_fields", {}).get(part)
                    if (
                        field_info
                        and hasattr(field_info, "default_factory")
                        and field_info.default_factory
                    ):
                        current_intermediate = field_info.default_factory()
                        setattr(target_obj, part, current_intermediate)
                        target_obj = current_intermediate
                    else:
                        logger.warning(
                            "intermediate attribute '%s' is None in '%s' for %s(id=%s).",
                            part,
                            property_name,
                            comp_cls,
                            comp_id,
                        )
                        return
                else:
                    target_obj = current_intermediate

            current_val = getattr(target_obj, attr_to_set, None)

            if (
                isinstance(current_val, BaseModel)
                and isinstance(value, BaseModel)
                and isinstance(value, type(current_val))
            ):
                # full same-type instance = complete value: replace, don't merge the
                # field default_factory back in. fill mode still keeps user-set fields.
                user_set = getattr(target_obj, "_user_set_fields", None) or set()
                if clobber or attr_to_set not in user_set:
                    setattr(target_obj, attr_to_set, value)
                    return
                # exclude_defaults, not exclude_unset: Size always fills model_fields_set
                value = value.model_dump(exclude_defaults=True)

            final_value = value

            if not clobber:
                user_set = getattr(target_obj, "_user_set_fields", None)
                if user_set and attr_to_set in user_set:
                    if isinstance(current_val, dict) and isinstance(value, dict):
                        fill_field_set_by_user = True
                    elif isinstance(current_val, BaseModel) and isinstance(value, dict):
                        pass
                    else:
                        return

            if fill_field_set_by_user:
                assert isinstance(value, dict) and isinstance(current_val, dict)
                merged: dict[Any, Any] = dict(value)
                merged.update(current_val)
                final_value = merged
            elif isinstance(current_val, BaseModel) and isinstance(value, dict):
                final_value = self._update_pydantic_model(
                    current_val, value, property_name, clobber=clobber
                )
            elif (
                current_val is None
                and isinstance(value, dict)
                and isinstance(target_obj, BaseModel)
            ):
                field_info = type(target_obj).model_fields.get(attr_to_set)
                if field_info and hasattr(field_info, "annotation"):
                    model_type = _model_type_of(field_info.annotation)
                    if model_type:
                        try:
                            final_value = model_type(**value)
                        except ValidationError as exc:
                            logger.error(
                                "validation error creating %s for %s: %s",
                                model_type.__name__,
                                property_name,
                                exc,
                            )
                            return
                        except Exception as exc:
                            logger.error(
                                "error creating %s for %s: %s",
                                model_type.__name__,
                                property_name,
                                exc,
                                exc_info=True,
                            )
                            return

            setattr(target_obj, attr_to_set, final_value)

        except AttributeError:
            logger.warning(
                "cannot set '%s': attribute not found on %s for %s(id=%s).",
                property_name,
                type(target_obj).__name__,
                comp_cls,
                comp_id,
            )
        except ValidationError as exc:
            logger.error(
                "pydantic validation error setting '%s' on %s with value %r: %s",
                property_name,
                type(target_obj).__name__,
                value,
                exc,
            )
        except Exception as exc:
            logger.error(
                "general error setting property '%s' on %s to value %r: %s",
                property_name,
                type(target_obj).__name__,
                value,
                exc,
                exc_info=True,
            )

    def _update_pydantic_model(
        self,
        current_model: BaseModel,
        update_dict: dict,
        prop_name: str,
        *,
        clobber: bool = True,
    ) -> BaseModel:
        current_dict = current_model.model_dump()
        merged_update_dict = {}

        # Read user-set fields from `_user_set_fields` (Component, MarginPadding,
        # BoxInset opt in via model_post_init) or fall back to `model_fields_set`.
        # Models like Size now use sentinel-init so `model_fields_set` only
        # contains fields the caller actually passed.
        user_set_nested = (
            getattr(current_model, "_user_set_fields", None) or current_model.model_fields_set
        )

        model_fields = type(current_model).model_fields
        for key, update_val in update_dict.items():
            if key not in model_fields:
                logger.warning(
                    "skipping key '%s' in update for %s: not a field in %s.",
                    key,
                    prop_name,
                    current_model.__class__.__name__,
                )
                continue

            if not clobber and key in user_set_nested:
                continue

            current_field_val = current_dict.get(key)

            if isinstance(current_field_val, (list, tuple)) and isinstance(update_val, list):
                try:
                    merged_update_dict[key] = self._merge_sequence(
                        current_field_val,
                        update_val,
                        model_fields[key],
                    )
                except Exception as exc:
                    logger.error("error merging sequence for key '%s': %s.", key, exc)
                    merged_update_dict[key] = current_field_val
            elif isinstance(current_field_val, BaseModel) and isinstance(update_val, dict):
                merged_update_dict[key] = self._update_pydantic_model(
                    current_field_val,
                    update_val,
                    f"{prop_name}.{key}",
                )
            else:
                # model_copy(update=...) skips validation, so a submodel field set from a
                # raw list/dict (e.g. padding: [..] -> BoxInset) must be coerced here.
                model_type = _model_type_of(model_fields[key].annotation)
                if model_type is not None and not isinstance(update_val, BaseModel):
                    merged_update_dict[key] = model_type.model_validate(update_val)
                else:
                    merged_update_dict[key] = update_val

        # Identity-preserve: no actual updates means the input is unchanged. Avoid
        # constructing a new instance so `panel.rescaler is explicit` holds for the
        # user-set / theme-no-op case.
        if not merged_update_dict:
            return current_model
        return current_model.model_copy(update=merged_update_dict)

    def _merge_sequence(
        self,
        current_seq: Sequence,
        update_list: list,
        field_info: Any,
    ) -> list | tuple:
        updated_list = list(current_seq)
        len_update = min(len(update_list), len(updated_list))

        expected_type = None
        try:
            if hasattr(field_info, "annotation") and hasattr(field_info.annotation, "__args__"):
                args = getattr(field_info.annotation, "__args__", [])
                if args and all(isinstance(a, type) for a in args):
                    is_list_hint = (
                        hasattr(field_info.annotation, "__origin__")
                        and field_info.annotation.__origin__ is list
                    )
                    expected_type = args[0] if is_list_hint or len(args) == 1 else args
        except Exception:
            pass

        for i in range(len_update):
            if update_list[i] is not None:
                val_to_set = update_list[i]
                target_type = float
                if isinstance(expected_type, (list, tuple)) and i < len(expected_type):
                    target_type = expected_type[i]
                elif isinstance(expected_type, type):
                    target_type = expected_type

                current_element_type = (
                    type(updated_list[i])
                    if i < len(updated_list) and updated_list[i] is not None
                    else None
                )
                if not isinstance(val_to_set, target_type) and current_element_type != target_type:
                    try:
                        val_to_set = target_type(val_to_set)
                    except (ValueError, TypeError):
                        continue

                updated_list[i] = val_to_set

        return tuple(updated_list) if isinstance(current_seq, tuple) else updated_list


jstyle = JStyle()
