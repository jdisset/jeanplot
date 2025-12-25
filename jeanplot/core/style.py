"""CSS-like styling system for jeanplot components.

See docs/STYLE_GUIDE.md for full documentation on selectors, specificity,
context rules, and property setting.
"""

from contextlib import contextmanager
import copy
import re
from pydantic import BaseModel, Field, ValidationError
from typing import Any, ClassVar, NamedTuple, Callable, Sequence, Union, get_origin, get_args
import types
import inspect
import logging


logger = logging.getLogger(__name__)


class Specificity(NamedTuple):
    """represents selector specificity (higher values are more specific)"""

    id_count: int = 0
    attr_class_count: int = 0
    type_count: int = 0


class Selector:
    """parses and evaluates style selectors."""

    # pre-compile regex for efficiency
    _ATTRIBUTE_SELECTOR_RE: ClassVar = re.compile(r"\[([^\]]+)\]")
    _CONDITION_RE: ClassVar = re.compile(
        # group 1: name, group 2: operator, group 3: value
        r"^\s*([^=~!<>*^$!]+)\s*(=[/~]?|!=|<=?|>=?|\*=|\^=|\$=)?\s*(.*)\s*$"  # added '!' to excluded chars for name start
    )
    _REGEX_RE: ClassVar = re.compile(r"^(.*)/([ism]*)$")  # pattern, flags
    _PRESENCE_RE: ClassVar = re.compile(
        r"^\s*(!)?([\w.-]+)\s*$"
    )  # group 1: negation, group 2: attr name

    raw_selector: str
    type_selector: str | None = None
    attributes: list[tuple[str, str | None, str | None]]
    specificity: Specificity

    def __init__(self, selector_str: str):
        self.raw_selector = selector_str.strip()
        self.attributes = []
        self._parse()
        self.specificity = self._calculate_specificity()
        logger.debug(
            f"parsed selector: '{self.raw_selector}' -> type='{self.type_selector}', attrs={self.attributes}, spec={self.specificity}"
        )

    def _parse(self):
        """extracts type and attribute parts from the raw selector."""
        selector = self.raw_selector
        matches = list(self._ATTRIBUTE_SELECTOR_RE.finditer(selector))  # find all attribute parts

        if not matches:
            # no attribute selectors, just type or wildcard
            if selector != "*":
                self.type_selector = selector
            return

        # handle potentially multiple attribute selectors like Type[attr1][attr2]
        last_match_end = 0
        combined_attr_content = ""
        if not selector.startswith("["):  # check if there's a type selector before the first attr
            first_match_start = matches[0].start()
            type_part = selector[:first_match_start].strip()
            if type_part and type_part != "*":
                self.type_selector = type_part
            last_match_end = matches[0].end()
            combined_attr_content += matches[0].group(1)  # content of first bracket
        else:  # starts directly with attribute selector
            last_match_end = matches[0].end()
            combined_attr_content += matches[0].group(1)

        # append content from subsequent adjacent brackets if any
        for i in range(1, len(matches)):
            if matches[i].start() == last_match_end:  # check if brackets are adjacent
                # use comma as separator for distinct selectors within merged content
                combined_attr_content += "," + matches[i].group(1)
                last_match_end = matches[i].end()
            else:
                # non-adjacent bracket means invalid syntax between them or end of sequence
                logger.warning(
                    f"ignoring attribute content after non-adjacent bracket in selector: '{self.raw_selector}'"
                )
                break

        self._parse_attributes(combined_attr_content)

    def _parse_attributes(self, attr_content: str):
        """parses comma-separated attribute conditions."""
        conditions = [
            cond.strip()
            for cond in re.split(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", attr_content)
        ]
        for cond in conditions:
            if not cond:
                continue

            match_presence = self._PRESENCE_RE.match(cond)
            if match_presence:
                negation, name = match_presence.groups()
                op = "not_exists" if negation == "!" else "exists"
                self.attributes.append((name.strip(), op, None))
                continue  # condition parsed, move to next

            match_cond = self._CONDITION_RE.match(cond)
            if match_cond:
                name, op, val = match_cond.groups()
                # strip quotes from value if present
                if val and len(val) >= 2 and val.startswith(("'", '"')) and val.endswith(val[0]):
                    val = val[1:-1]
                # default op is '=' if present but group is None (e.g., [attr=val] which is caught by _CONDITION_RE)
                self.attributes.append((name.strip(), op or "=", val.strip()))
                continue  # condition parsed, move to next

            # if neither matched, log warning
            logger.warning(
                f"could not parse attribute condition: '{cond}' in '{self.raw_selector}'"
            )

    def _calculate_specificity(self) -> Specificity:
        """calculates specificity based on parsed components."""
        ids = sum(1 for name, _, _ in self.attributes if name == "id")
        attrs_classes = sum(1 for name, _, _ in self.attributes if name != "id")
        types = 1 if self.type_selector else 0
        return Specificity(ids, attrs_classes, types)

    def matches(self, component: Any) -> bool:
        """checks if the component matches this selector."""
        if not self._matches_type(component):
            return False
        if not self._matches_attributes(component):
            return False
        return True

    def _matches_type(self, component: Any) -> bool:
        """checks the type part of the selector."""
        if not self.type_selector:
            return True  # wildcard or attribute-only selector
        # check against component's class name and its bases
        return any(
            cls.__name__ == self.type_selector for cls in inspect.getmro(component.__class__)
        )

    def _matches_attributes(self, component: Any) -> bool:
        """checks all attribute conditions."""
        return all(
            self._check_condition(component, name, op, val) for name, op, val in self.attributes
        )

    def _check_condition(
        self, component: Any, name: str, op: str | None, value_pattern: str | None
    ) -> bool:
        """checks a single attribute condition."""
        actual_value = self._get_attribute_value(component, name)
        attr_exists = (
            actual_value is not None
        )  # need to know if the attribute exists at all for exists/not_exists

        op_map: dict[str, Callable[[Any, str | None], bool]] = {
            "exists": lambda v, p: attr_exists and bool(v),  # exists and truthy
            "not_exists": lambda v, p: not attr_exists or not bool(v),  # does not exist or is falsy
            "=": lambda v, p: attr_exists and self._value_matches(v, p, operator="="),
            "!=": lambda v, p: not attr_exists
            or self._value_matches(v, p, operator="!="),  # true if not exists OR value mismatch
            "=~": lambda v, p: attr_exists and self._value_matches(v, p, operator="=~"),
            "^=": lambda v, p: attr_exists and str(v).startswith(p) if p is not None else False,
            "$=": lambda v, p: attr_exists and str(v).endswith(p) if p is not None else False,
            "*=": lambda v, p: attr_exists and p in str(v) if p is not None else False,
            "=/": lambda v, p: attr_exists and self._regex_matches(v, p),
            "<": lambda v, p: attr_exists and self._numeric_compare(v, p, operator="<"),
            "<=": lambda v, p: attr_exists and self._numeric_compare(v, p, operator="<="),
            ">": lambda v, p: attr_exists and self._numeric_compare(v, p, operator=">"),
            ">=": lambda v, p: attr_exists and self._numeric_compare(v, p, operator=">="),
        }

        # handle list attributes: match if *any* item matches (only for comparison ops, not exists/not_exists)
        list_ops = {"=", "!=", "=~", "^=", "$=", "*=", "=/", "<", "<=", ">", ">="}
        if isinstance(actual_value, (list, tuple, set)) and op in list_ops:
            matcher = op_map.get(op)
            if matcher and value_pattern is not None:
                # note: list matching against != might be unintuitive, e.g., [a!=foo] is true if *any* element is not foo.
                return any(matcher(item, value_pattern) for item in actual_value)
            return False  # invalid op for list or no pattern

        # handle non-list attributes or exists/not_exists
        matcher = op_map.get(op) if op else None
        if matcher:
            return matcher(actual_value, value_pattern)
        else:
            logger.warning(f"unknown operator '{op}' for attribute '{name}'")
            return False

    def _get_attribute_value(self, component: Any, attr_name: str) -> Any:
        """safely retrieves potentially nested attribute values."""
        try:
            obj = component
            for part in attr_name.split("."):
                if obj is None:
                    return None
                # handle list/tuple indexing
                if part.isdigit() and isinstance(obj, (list, tuple)):
                    idx = int(part)
                    obj = obj[idx] if 0 <= idx < len(obj) else None
                else:
                    obj = getattr(obj, part, None)  # use getattr default
            return obj
        except (AttributeError, IndexError, TypeError, ValueError):
            return None  # return none if any part fails

    def _regex_matches(self, actual_value: Any, pattern_details: str | None) -> bool:
        """performs regex matching."""
        if actual_value is None or pattern_details is None:
            return False
        pattern, flags_str = pattern_details, ""
        match = self._REGEX_RE.match(pattern_details)
        if match:
            pattern, flags_str = match.groups()

        re_flags = 0
        if flags_str:
            if "i" in flags_str:
                re_flags |= re.IGNORECASE
            if "m" in flags_str:
                re_flags |= re.MULTILINE
            if "s" in flags_str:
                re_flags |= re.DOTALL
        try:
            return bool(re.search(pattern, str(actual_value), re_flags))
        except re.error as e:
            logger.warning(f"regex error in selector '{self.raw_selector}': {e}")
            return False

    def _value_matches(self, actual_value: Any, pattern_value: str | None, operator: str) -> bool:
        """performs basic value comparisons (equality, inequality, case-insensitive)."""
        # this check assumes the attribute exists (done in _check_condition)
        if pattern_value is None:
            # this case should ideally not be reached if op requires a pattern
            logger.warning(f"pattern_value is None for operator '{operator}'")
            return False

        # handle 'none' keyword for checking against None
        pattern_is_none_keyword = pattern_value.lower() == "none"
        if operator == "=":
            return (actual_value is None and pattern_is_none_keyword) or (
                actual_value is not None
                and not pattern_is_none_keyword
                and str(actual_value) == pattern_value
            )
        elif operator == "!=":
            # actual value is not None AND (pattern is not 'none' keyword OR actual value differs)
            # OR actual value is None AND pattern is not 'none' keyword
            return (
                actual_value is not None
                and (not pattern_is_none_keyword or str(actual_value) != pattern_value)
            ) or (actual_value is None and not pattern_is_none_keyword)
        elif operator == "=~":
            # case-insensitive comparison only makes sense if both are not None and pattern is not 'none'
            return (
                actual_value is not None
                and not pattern_is_none_keyword
                and str(actual_value).lower() == pattern_value.lower()
            )
        else:
            # other operators handled by specific methods (_numeric_compare, _regex_matches etc)
            logger.warning(f"unexpected operator '{operator}' in _value_matches")
            return False

    def _numeric_compare(self, actual_value: Any, pattern_value: str | None, operator: str) -> bool:
        """performs numeric comparisons."""
        # this check assumes the attribute exists (done in _check_condition)
        if pattern_value is None:
            return False  # cannot compare numerically with None
        try:
            num_actual, num_pattern = float(actual_value), float(pattern_value)
            op_map = {
                "<": lambda a, b: a < b,
                "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b,
                ">=": lambda a, b: a >= b,
            }
            return op_map[operator](num_actual, num_pattern)
        except (ValueError, TypeError, KeyError):
            return False  # cannot compare numerically

    def get_mro_level(self, component: Any) -> int:
        """returns the MRO level index if type selector matches, else infinity."""
        if self.type_selector:
            for i, cls in enumerate(inspect.getmro(component.__class__)):
                if cls.__name__ == self.type_selector:
                    return i
        return float("inf")  # handles wildcard or non-matching types

    def __repr__(self):
        return f"<Selector('{self.raw_selector}', spec={self.specificity})>"


class StyleRule(BaseModel):
    selector: Selector
    properties: dict[str, Any] = Field(default_factory=dict)
    nested_rules: dict[str, Any] = Field(default_factory=dict)
    is_context_rule: bool = False
    source_index: int = 0
    match_level: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class PropertyApplication(NamedTuple):
    """data needed to decide the winning value for a property."""

    specificity: Specificity
    is_context: bool
    mro_level: int
    source_order: int  # tie-breaker based on definition order
    value: Any


class JStyle:
    def __init__(self, style_dict: dict[str, Any] | None = None):
        self._raw_styles: dict[str, Any] = {}
        self.styles: list[StyleRule] = []
        if style_dict:
            self.update(style_dict)

    def update(self, style_dict: dict[str, Any]):
        """adds or updates style rules from a dictionary."""
        self._raw_styles = self._deep_merge_dicts(self._raw_styles, style_dict)
        self.styles = self._parse_style_dict(self._raw_styles, is_context=False)
        # sort global rules once by specificity (asc) then source order (asc)
        # this ensures consistent tie-breaking later
        self.styles.sort(key=lambda r: (r.selector.specificity, r.source_index))

    def apply(self, component: Any):
        """applies styles recursively to a component and its children."""
        if component is None:
            return

        self._apply_to_component(component)

        # recurse
        if hasattr(component, "children") and component.children:
            for child in component.children:
                # ensure parent link is set before styling child
                if getattr(child, "parent", None) != component:
                    child.parent = component
                self.apply(child)  # recursive call

        return component

    def _apply_to_component(self, component: Any):
        """applies styles to a single component, considering context."""
        comp_id = getattr(component, "id", "N/A")
        comp_cls = component.__class__.__name__
        logger.debug(f"\n--- applying styles to {comp_cls}(id={comp_id}) ---")

        # 1. find all matching rules (global + context)
        applicable_rules = self._get_applicable_rules(component)

        # 2. determine winning value for each property
        properties_to_set = self._resolve_properties(applicable_rules)

        # 3. apply the winning properties
        logger.debug(f"  applying properties to {comp_cls}(id={comp_id}): {properties_to_set}")
        for prop_path, value in properties_to_set.items():
            self._set_property(component, prop_path, value)

    def _get_applicable_rules(self, component: Any) -> list[StyleRule]:
        applicable_rules: list[StyleRule] = []

        # global rules
        for rule in self.styles:  # self.styles is already sorted by spec/source
            if not rule.is_context_rule and rule.selector.matches(component):
                rule.match_level = rule.selector.get_mro_level(component)
                applicable_rules.append(rule)
                logger.debug(
                    f"  matched global rule: '{rule.selector.raw_selector}' (spec: {rule.selector.specificity}, level: {rule.match_level}, order: {rule.source_index})"
                )

        # context rules (discovered via parent hierarchy)
        context_rules = self._discover_context_rules(component)
        # sort context rules like global rules for consistent processing
        context_rules.sort(key=lambda r: (r.selector.specificity, r.source_index))
        for rule in context_rules:
            if rule.selector.matches(component):
                rule.match_level = rule.selector.get_mro_level(component)
                applicable_rules.append(rule)
                logger.debug(
                    f"  matched context rule: '{rule.selector.raw_selector}' (spec: {rule.selector.specificity}, level: {rule.match_level}, order: {rule.source_index}, context: True)"
                )

        return applicable_rules

    def _discover_context_rules(self, component: Any) -> list[StyleRule]:
        effective_nested_rules: dict[str, Any] = {}
        ancestors = []
        parent = getattr(component, "parent", None)
        while parent is not None:
            ancestors.append(parent)
            parent = getattr(parent, "parent", None)

        # apply rules from root down to immediate parent
        for ancestor in reversed(ancestors):
            # consider only global rules that match the ancestor
            for rule in self.styles:
                if not rule.is_context_rule and rule.selector.matches(ancestor):
                    effective_nested_rules = self._deep_merge_dicts(
                        effective_nested_rules, rule.nested_rules
                    )

        # parse the final merged nested dict into context rules
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
            # sort declarations: specificity (desc), context (desc), mro_level (asc), source_order (asc)
            declarations.sort(
                key=lambda x: (x.specificity, x.is_context, -x.mro_level, x.source_order),
                reverse=True,
            )
            winner = declarations[0]
            winning_properties[prop_path] = winner.value
            logger.debug(
                f"    => winner for '{prop_path}': spec={winner.specificity}, context={winner.is_context}, level={winner.mro_level}, order={winner.source_order} -> value={repr(winner.value)}"
            )

        return winning_properties

    def _is_key_likely_property(self, key: str) -> bool:
        """determines if a key in a style declaration dict is likely a property path or a nested selector."""
        if not isinstance(key, str):
            return False
        # definitely a property if it contains '.'
        if "." in key:
            return True
        # definitely a nested selector if it starts with '[' or is '*'
        if key.startswith("[") or key == "*":
            return False
        # likely a property if it starts lowercase
        if key[0].islower():
            return True
        # likely a nested selector if it starts uppercase and doesn't contain '['
        # (handles simple Type selectors like "Text" or "Container")
        if key[0].isupper() and "[" not in key:
            return False
        # if it starts uppercase and *does* contain '[', it's a combined nested selector like "Text[id=foo]"
        if key[0].isupper() and "[" in key:
            return False

        # fallback: if we're unsure, assume property (e.g., an uppercase attribute name)
        logger.warning(f"ambiguous style key '{key}', assuming property.")
        return True

    def _parse_style_dict(self, style_dict: dict[str, Any], is_context: bool) -> list[StyleRule]:
        """parses a style dictionary, adding source order and distinguishing properties vs nested rules."""
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
            except Exception as e:
                logger.error(
                    f"failed to parse rule '{selector_str}': {e}",
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
        return rules_list

    def _set_property(self, component: Any, property_name: str, value: Any):
        """sets a property, potentially nested, handling model updates and partial list merges."""
        comp_id = getattr(component, "id", "N/A")
        comp_cls = component.__class__.__name__

        parts = property_name.split(".")
        target_obj = component
        attr_to_set = parts[-1]

        try:
            # traverse to the parent object
            for part in parts[:-1]:
                # if intermediate part is a pydantic model but None, try to create it? - might be too complex/risky
                # for now, assume intermediate parts must exist
                current_intermediate = getattr(target_obj, part)
                if current_intermediate is None:
                    # Check if it's supposed to be a model and has a default factory
                    field_info = getattr(type(target_obj), "model_fields", {}).get(part)
                    if (
                        field_info
                        and hasattr(field_info, "default_factory")
                        and field_info.default_factory
                    ):
                        logger.debug(
                            f"      instantiating intermediate model '{part}' using default factory"
                        )
                        current_intermediate = field_info.default_factory()
                        setattr(
                            target_obj, part, current_intermediate
                        )  # set the new intermediate model back
                        target_obj = current_intermediate
                    else:
                        logger.warning(
                            f"intermediate attribute '{part}' is None in '{property_name}' for {comp_cls}(id={comp_id}) and cannot be auto-created."
                        )
                        return
                else:
                    target_obj = current_intermediate

            current_val = getattr(target_obj, attr_to_set, None)
            final_value = value

            # handle pydantic model update with dict value
            if isinstance(current_val, BaseModel) and isinstance(value, dict):
                logger.debug(f"      handling basemodel update for existing {property_name}")
                final_value = self._update_pydantic_model(current_val, value, property_name)
            elif (
                current_val is None
                and isinstance(value, dict)
                and isinstance(target_obj, BaseModel)
            ):
                # check if the target attribute is annotated as a pydantic model
                field_info = type(target_obj).model_fields.get(attr_to_set)
                if field_info and hasattr(field_info, "annotation"):
                    target_type = field_info.annotation
                    origin = get_origin(target_type)
                    # handle Union (typing.Union or X | Y syntax)
                    if origin is Union or isinstance(target_type, types.UnionType):
                        args = get_args(target_type)
                        # find the basemodel type among the union args
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
                                f"      instantiating new '{model_type.__name__}' for {property_name} from dict"
                            )
                            # instantiate the model from the dict value
                            final_value = model_type(**value)
                        except ValidationError as e_create:
                            logger.error(
                                f"      validation error creating {model_type.__name__} for {property_name}: {e_create}"
                            )
                            # fallback to setting the dict if model creation fails? or just skip? skip for now.
                            return
                        except Exception as e_create_other:
                            logger.error(
                                f"      error creating {model_type.__name__} for {property_name}: {e_create_other}",
                                exc_info=True,
                            )
                            return

            # perform assignment
            setattr(target_obj, attr_to_set, final_value)

            # --- verification logging ---
            if logger.isEnabledFor(logging.DEBUG):
                target_after = getattr(target_obj, attr_to_set, "ATTR_NOT_FOUND_AFTER")
                logger.debug(
                    f"    set {comp_cls}(id={comp_id}).{property_name}: value={repr(final_value)}, result={repr(target_after)}"
                )

        except AttributeError:
            logger.warning(
                f"cannot set '{property_name}': attribute not found on {type(target_obj).__name__} for {comp_cls}(id={comp_id})."
            )
        except ValidationError as e:
            logger.error(
                f"    pydantic validation error setting '{property_name}' on {type(target_obj).__name__} with value {repr(value)}: {e}"
            )
        except Exception as e:
            logger.error(
                f"    general error setting property '{property_name}' on {type(target_obj).__name__} to value {repr(value)}: {e}",
                exc_info=True,
            )

    def _update_pydantic_model(
        self, current_model: BaseModel, update_dict: dict, prop_name: str
    ) -> BaseModel:
        """handles partial updates for pydantic models, including lists/tuples."""
        current_dict = current_model.model_dump()
        merged_update_dict = {}

        model_fields = type(current_model).model_fields
        for key, update_val in update_dict.items():
            if key not in model_fields:
                logger.warning(
                    f"      skipping key '{key}' in update for {prop_name}: not a field in {current_model.__class__.__name__}."
                )
                continue

            current_field_val = current_dict.get(key)

            if isinstance(current_field_val, (list, tuple)) and isinstance(update_val, list):
                # handle partial list/tuple update
                try:
                    merged_update_dict[key] = self._merge_sequence(
                        current_field_val, update_val, model_fields[key]
                    )
                except Exception as e_seq:
                    logger.error(
                        f"      error merging sequence for key '{key}': {e_seq}. using original."
                    )
                    merged_update_dict[key] = current_field_val  # fallback
            elif isinstance(current_field_val, BaseModel) and isinstance(update_val, dict):
                # handle nested model update recursively
                merged_update_dict[key] = self._update_pydantic_model(
                    current_field_val, update_val, f"{prop_name}.{key}"
                )
            else:
                # direct value update
                merged_update_dict[key] = update_val

        # use pydantic's update mechanism for robustness
        return current_model.model_copy(update=merged_update_dict)

    def _merge_sequence(
        self, current_seq: Sequence, update_list: list, field_info: Any
    ) -> list | tuple:
        """merges an update list into a current sequence, respecting Nones and types."""
        updated_list = list(current_seq)
        len_update = min(len(update_list), len(updated_list))
        logger.debug(
            f"        merging sequence. current: {current_seq}, update: {update_list[:len_update]}"
        )

        # try to get element type hint
        expected_type = None
        try:
            # handles Tuple[type, ...]
            if hasattr(field_info, "annotation") and hasattr(field_info.annotation, "__args__"):
                args = getattr(field_info.annotation, "__args__", [])
                if args and all(isinstance(a, type) for a in args):  # basic check
                    # simple case: assume all elements have same type as first arg for List, or use tuple args
                    is_list_hint = (
                        hasattr(field_info.annotation, "__origin__")
                        and field_info.annotation.__origin__ is list
                    )
                    expected_type = (
                        args[0] if is_list_hint or len(args) == 1 else args
                    )  # use tuple types if available
        except Exception:
            pass  # ignore errors getting type hint

        for i in range(len_update):
            if update_list[i] is not None:
                val_to_set = update_list[i]
                # determine target type for this index
                target_type = float  # default fallback
                if isinstance(expected_type, (list, tuple)) and i < len(expected_type):
                    target_type = expected_type[i]
                elif isinstance(expected_type, type):
                    target_type = expected_type

                # attempt type conversion if needed
                current_element_type = (
                    type(updated_list[i])
                    if i < len(updated_list) and updated_list[i] is not None
                    else None
                )
                if not isinstance(val_to_set, target_type) and current_element_type != target_type:
                    try:
                        val_to_set = target_type(val_to_set)
                        logger.debug(
                            f"          converted index {i} value {update_list[i]} to type {target_type}"
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"          type conversion failed for index {i} value {repr(update_list[i])} to type {target_type}. skipping update."
                        )
                        continue  # skip if conversion fails

                logger.debug(f"          updating index {i}: {updated_list[i]} -> {val_to_set}")
                updated_list[i] = val_to_set

        return tuple(updated_list) if isinstance(current_seq, tuple) else updated_list

    def _deep_merge_dicts(self, base: dict, overlay: dict) -> dict:
        """recursively merges overlay dict into base dict."""
        merged = copy.deepcopy(base)
        for key, value in overlay.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge_dicts(merged[key], value)
            else:
                # overlay always wins for non-dict values
                merged[key] = value
        return merged

    @contextmanager
    def context(self, style_dict):
        """temporary style context manager."""
        old_raw_styles = copy.deepcopy(self._raw_styles)
        old_parsed_rules = copy.deepcopy(self.styles)
        try:
            # create temporary merged styles for the context
            context_raw_styles = self._deep_merge_dicts(self._raw_styles, style_dict)
            self.styles = self._parse_style_dict(context_raw_styles, is_context=False)
            self._raw_styles = context_raw_styles  # also update raw for consistency within context
            yield
        finally:
            # restore original styles
            self.styles = old_parsed_rules
            self._raw_styles = old_raw_styles

    def clear(self):
        """clears all styles."""
        self._raw_styles = {}
        self.styles = []

    __call__ = context


# global style instance
jstyle = JStyle()
