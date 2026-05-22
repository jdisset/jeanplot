import inspect
import logging
import types
from contextlib import contextmanager
from typing import Any, Sequence, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from dracon import CascadeStrategy, register_cascade_strategy
from dracon.symbols import CallableSymbol

from jeanplot.core.style_dialect import parse_jstyle_rule_tree, parse_selector_key

logger = logging.getLogger(__name__)


_JSTYLE_STRATEGY = CascadeStrategy(
    name="jstyle",
    input_params=("component",),
    parse=parse_selector_key,
    matches=lambda sel, component: sel.matches(component),
    specificity=lambda sel: tuple(sel.specificity),
)
register_cascade_strategy(_JSTYLE_STRATEGY)

_JSTYLE_FILL_STRATEGY = CascadeStrategy(
    name="jstyle_fill",
    input_params=("component",),
    parse=parse_selector_key,
    matches=lambda sel, component: sel.matches(component),
    specificity=lambda sel: tuple(sel.specificity),
)
register_cascade_strategy(_JSTYLE_FILL_STRATEGY)


def _strategy_is_fill(cascade: CallableSymbol | None) -> bool:
    if cascade is None:
        return False
    strat = getattr(cascade, "_cascade_strategy", None)
    return getattr(strat, "name", None) == "jstyle_fill"


def _as_cascade(value: Any) -> CallableSymbol | None:
    """Coerce a value into a jstyle CallableSymbol with a flat rule_tree.

    Accepts a CallableSymbol (rule_tree flattened in place) or a nested dict
    (flattened then wrapped). Nested-rule trees ``{Sel: {Sel2: {prop: v}}}``
    are flattened into descendant selectors ``{Selector("Sel Sel2"): {prop: v}}``.
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


def _resolve_lazies(value: Any) -> Any:
    """Recursively force-resolve any LazyInterpolable in dict keys / values,
    leaving live-scope (component-bound) lazies untouched.

    Needed because dracon's ``resolve_all_lazy`` skips private attributes on
    objects, so ``CallableSymbol._rule_tree`` is never walked. We mirror its
    semantics here against the flat rule_tree on update.
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
        return component

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
            final_value = value

            if not clobber:
                user_set = getattr(target_obj, "_user_set_fields", None)
                if user_set and attr_to_set in user_set:
                    if isinstance(current_val, dict) and isinstance(value, dict):
                        fill_field_set_by_user = True
                    elif isinstance(current_val, BaseModel) and isinstance(value, dict):
                        pass  # fall through to per-key model merge under fill semantics
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
                    target_type = field_info.annotation
                    origin = get_origin(target_type)
                    if origin is Union or isinstance(target_type, types.UnionType):
                        args = get_args(target_type)
                        model_type = next(
                            (
                                arg
                                for arg in args
                                if inspect.isclass(arg) and issubclass(arg, BaseModel)
                            ),
                            None,
                        )
                    elif inspect.isclass(target_type) and issubclass(target_type, BaseModel):
                        model_type = target_type
                    else:
                        model_type = None

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
                merged_update_dict[key] = update_val

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
