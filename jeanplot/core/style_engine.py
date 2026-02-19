"""Engine implementation for CSS-like style resolution."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import inspect
import logging
import types
from typing import Any, Sequence, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from jeanplot.core.style_models import PropertyApplication, StyleRule
from jeanplot.core.style_selector import Selector

logger = logging.getLogger(__name__)


class JStyle:
    def __init__(self, style_dict: dict[str, Any] | None = None):
        self._raw_styles: dict[str, Any] = {}
        self.styles: list[StyleRule] = []
        if style_dict:
            self.update(style_dict)

    def update(self, style_dict: dict[str, Any]):
        self._raw_styles = self._deep_merge_dicts(self._raw_styles, style_dict)
        self.styles = self._parse_style_dict(self._raw_styles, is_context=False)
        self.styles.sort(key=lambda r: (r.selector.specificity, r.source_index))

    def apply(self, component: Any):
        """Apply styles recursively to a component tree."""
        if component is None:
            return

        self._apply_to_component(component)

        if hasattr(component, "children") and component.children:
            for child in component.children:
                if getattr(child, "parent", None) != component:
                    child.parent = component
                self.apply(child)

        return component

    def apply_one(self, component: Any):
        """Apply styles to one component without recursing into children."""
        if component is None:
            return
        self._apply_to_component(component)
        return component

    def _apply_to_component(self, component: Any):
        comp_id = getattr(component, "id", "N/A")
        comp_cls = component.__class__.__name__
        logger.debug("\n--- applying styles to %s(id=%s) ---", comp_cls, comp_id)

        applicable_rules = self._get_applicable_rules(component)
        properties_to_set = self._resolve_properties(applicable_rules)

        logger.debug("  applying properties to %s(id=%s): %s", comp_cls, comp_id, properties_to_set)
        for prop_path, value in properties_to_set.items():
            self._set_property(component, prop_path, value)

    def _get_applicable_rules(self, component: Any) -> list[StyleRule]:
        applicable_rules: list[StyleRule] = []

        for rule in self.styles:
            if not rule.is_context_rule and rule.selector.matches(component):
                rule.match_level = rule.selector.get_mro_level(component)
                applicable_rules.append(rule)
                logger.debug(
                    "  matched global rule: '%s' (spec: %s, level: %s, order: %s)",
                    rule.selector.raw_selector,
                    rule.selector.specificity,
                    rule.match_level,
                    rule.source_index,
                )

        context_rules = self._discover_context_rules(component)
        context_rules.sort(key=lambda r: (r.selector.specificity, r.source_index))
        for rule in context_rules:
            if rule.selector.matches(component):
                rule.match_level = rule.selector.get_mro_level(component)
                applicable_rules.append(rule)
                logger.debug(
                    "  matched context rule: '%s' (spec: %s, level: %s, order: %s, context: True)",
                    rule.selector.raw_selector,
                    rule.selector.specificity,
                    rule.match_level,
                    rule.source_index,
                )

        return applicable_rules

    def _discover_context_rules(self, component: Any) -> list[StyleRule]:
        effective_nested_rules: dict[str, Any] = {}
        ancestors = []
        parent = getattr(component, "parent", None)
        while parent is not None:
            ancestors.append(parent)
            parent = getattr(parent, "parent", None)

        for ancestor in reversed(ancestors):
            for rule in self.styles:
                if not rule.is_context_rule and rule.selector.matches(ancestor):
                    effective_nested_rules = self._deep_merge_dicts(
                        effective_nested_rules, rule.nested_rules
                    )

        return self._parse_style_dict(effective_nested_rules, is_context=True)

    def _resolve_properties(self, applicable_rules: list[StyleRule]) -> dict[str, Any]:
        declarations_by_prop: dict[str, list[PropertyApplication]] = {}

        for rule in applicable_rules:
            for prop_path, value in rule.properties.items():
                app = PropertyApplication(
                    specificity=rule.selector.specificity,
                    is_context=rule.is_context_rule,
                    mro_level=rule.match_level if rule.match_level is not None else float("inf"),
                    source_order=rule.source_index,
                    value=value,
                )
                declarations_by_prop.setdefault(prop_path, []).append(app)

        winning_properties = {}
        for prop_path, declarations in declarations_by_prop.items():
            if not declarations:
                continue
            declarations.sort(
                key=lambda x: (x.specificity, x.is_context, -x.mro_level, x.source_order),
                reverse=True,
            )
            winner = declarations[0]
            winning_properties[prop_path] = winner.value
            logger.debug(
                "    => winner for '%s': spec=%s, context=%s, level=%s, order=%s -> value=%r",
                prop_path,
                winner.specificity,
                winner.is_context,
                winner.mro_level,
                winner.source_order,
                winner.value,
            )

        return winning_properties

    def _is_key_likely_property(self, key: str) -> bool:
        if not isinstance(key, str):
            return False
        if "." in key:
            return True
        if key.startswith("[") or key == "*":
            return False
        if key[0].islower():
            return True
        if key[0].isupper() and "[" not in key:
            return False
        if key[0].isupper() and "[" in key:
            return False

        logger.warning("ambiguous style key '%s', assuming property.", key)
        return True

    def _parse_style_dict(self, style_dict: dict[str, Any], is_context: bool) -> list[StyleRule]:
        rules_list = []
        for i, (selector_str, declarations) in enumerate(style_dict.items()):
            if not isinstance(declarations, dict):
                continue
            try:
                selector = Selector(selector_str)
                rule = StyleRule(selector=selector, is_context_rule=is_context, source_index=i)
                for key, value in declarations.items():
                    if self._is_key_likely_property(key):
                        rule.properties[key] = value
                    else:
                        rule.nested_rules[key] = value
                rules_list.append(rule)
            except Exception as exc:
                logger.error(
                    "failed to parse rule '%s': %s",
                    selector_str,
                    exc,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
        return rules_list

    def _set_property(self, component: Any, property_name: str, value: Any):
        comp_id = getattr(component, "id", "N/A")
        comp_cls = component.__class__.__name__

        parts = property_name.split(".")
        target_obj = component
        attr_to_set = parts[-1]

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
                        logger.debug(
                            "      instantiating intermediate model '%s' using default factory",
                            part,
                        )
                        current_intermediate = field_info.default_factory()
                        setattr(target_obj, part, current_intermediate)
                        target_obj = current_intermediate
                    else:
                        logger.warning(
                            "intermediate attribute '%s' is None in '%s' for %s(id=%s) and cannot be auto-created.",
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

            if isinstance(current_val, BaseModel) and isinstance(value, dict):
                logger.debug("      handling basemodel update for existing %s", property_name)
                final_value = self._update_pydantic_model(current_val, value, property_name)
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
                            logger.debug(
                                "      instantiating new '%s' for %s from dict",
                                model_type.__name__,
                                property_name,
                            )
                            final_value = model_type(**value)
                        except ValidationError as exc:
                            logger.error(
                                "      validation error creating %s for %s: %s",
                                model_type.__name__,
                                property_name,
                                exc,
                            )
                            return
                        except Exception as exc:
                            logger.error(
                                "      error creating %s for %s: %s",
                                model_type.__name__,
                                property_name,
                                exc,
                                exc_info=True,
                            )
                            return

            setattr(target_obj, attr_to_set, final_value)

            if logger.isEnabledFor(logging.DEBUG):
                target_after = getattr(target_obj, attr_to_set, "ATTR_NOT_FOUND_AFTER")
                logger.debug(
                    "    set %s(id=%s).%s: value=%r, result=%r",
                    comp_cls,
                    comp_id,
                    property_name,
                    final_value,
                    target_after,
                )

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
                "    pydantic validation error setting '%s' on %s with value %r: %s",
                property_name,
                type(target_obj).__name__,
                value,
                exc,
            )
        except Exception as exc:
            logger.error(
                "    general error setting property '%s' on %s to value %r: %s",
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
    ) -> BaseModel:
        current_dict = current_model.model_dump()
        merged_update_dict = {}

        model_fields = type(current_model).model_fields
        for key, update_val in update_dict.items():
            if key not in model_fields:
                logger.warning(
                    "      skipping key '%s' in update for %s: not a field in %s.",
                    key,
                    prop_name,
                    current_model.__class__.__name__,
                )
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
                    logger.error(
                        "      error merging sequence for key '%s': %s. using original.",
                        key,
                        exc,
                    )
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
        logger.debug(
            "        merging sequence. current: %s, update: %s",
            current_seq,
            update_list[:len_update],
        )

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
                        logger.debug(
                            "          converted index %s value %s to type %s",
                            i,
                            update_list[i],
                            target_type,
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            "          type conversion failed for index %s value %r to type %s. skipping update.",
                            i,
                            update_list[i],
                            target_type,
                        )
                        continue

                logger.debug("          updating index %s: %s -> %s", i, updated_list[i], val_to_set)
                updated_list[i] = val_to_set

        return tuple(updated_list) if isinstance(current_seq, tuple) else updated_list

    def _deep_merge_dicts(self, base: dict, overlay: dict) -> dict:
        merged = copy.deepcopy(base)
        for key, value in overlay.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @contextmanager
    def context(self, style_dict):
        old_raw_styles = copy.deepcopy(self._raw_styles)
        old_parsed_rules = copy.deepcopy(self.styles)
        try:
            context_raw_styles = self._deep_merge_dicts(self._raw_styles, style_dict)
            self.styles = self._parse_style_dict(context_raw_styles, is_context=False)
            self._raw_styles = context_raw_styles
            yield
        finally:
            self.styles = old_parsed_rules
            self._raw_styles = old_raw_styles

    def clear(self):
        self._raw_styles = {}
        self.styles = []

    __call__ = context


jstyle = JStyle()
