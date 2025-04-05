"""
Manages and applies styles to jeanplot components using CSS-like selectors.

This module provides a styling system (`jstyle`) inspired by CSS, allowing
declarative styling of components based on their type, attributes, and
position within the component hierarchy.

Core Features:
-------------

1.  **Selectors:** Target components for styling.
    *   **Type Selectors:** Match component class names (e.g., `Container`, `Text`).
        Inheritance is respected; a rule for `Component` applies to `Container`.
        ```python
        jstyle.update({
            "Container": { "style.background_color": "#f0f0f0" },
            "Text": { "font_size": 10 }
        })
        ```
    *   **ID Selectors:** Match the specific `id` attribute (e.g., `[id=my-button]`).
        ```python
        jstyle.update({ "[id=special]": { "color": "red" } })
        ```
    *   **Style Class Selectors:** Match components containing a specific class in
        their `style_class` list (e.g., `[style_class=highlight]`).
        ```python
        jstyle.update({ "[style_class=primary]": { "style.border_color": "blue" } })
        ```
    *   **Attribute Selectors:** Match components based on any attribute value using
        various operators:
        - `=`: Exact match (`[name=foo]`, `[size=10]`)
        - `!=`: Not equal (`[status!=error]`)
        - `~=`: Case-insensitive match (`[label=~submit]`)
        - `^=`: Starts with (`[id^=item-]`)
        - `$=`: Ends with (`[filename$=.png]`)
        - `*=`: Contains substring (`[text*='important']`)
        - `=/regex/flags`: Regular expression match (e.g., `[id=/^user-\d+$/i]`)
        - `[attr]`: Presence check (attribute exists and is truthy) (`[disabled]`)
        - `<`, `<=`, `>`, `>=`: Numeric comparison (`[value>100]`)
        ```python
        jstyle.update({
            "Button[status=active]": { "opacity": 1.0 },
            "Image[filename$=.jpg]": { "style.border_radius": 4 },
            "Component[debug]": { "style.border_style": "dotted" }
        })
        ```
    *   **Combined Selectors:** Combine type and attribute selectors (e.g., `Text[style_class=warning]`).
        ```python
        jstyle.update({
            "Text[style_class=error]": { "color": "red", "font_weight": "bold" }
        })
        ```
    *   **Wildcard Selector:** `*` matches any component (lowest specificity).
        ```python
        jstyle.update({ "*": { "debug": False } })
        ```

2.  **Specificity:** Determines which rule applies when multiple selectors match.
    The order is: ID > Attribute/Class > Type > Wildcard. Within the same
    specificity level, rules defined later in the style dictionary (or added via
    `update()`) take precedence. Contextual rules (see below) override global
    rules of the same specificity.

3.  **Context (Nested Rules):** Apply styles to descendants based on their ancestor.
    Rules nested inside another rule's dictionary only apply if the outer selector
    matches an ancestor and the inner selector matches the descendant.
    ```python
    jstyle.update({
        "Container": { # applies to all containers
            "style.padding": (10,10,10,10),
            "Text": { # applies to Text inside any Container
                "color": "darkgray"
            }
        },
        "Container[id=sidebar]": { # specific container
            "style.background_color": "#eee",
            "Text": { # applies to Text inside the sidebar ONLY
                "color": "black",
                "font_size": 9
            },
            "Button[style_class=primary]": { # button inside sidebar
                 "style.background_color": "blue"
            }
        }
    })
    ```

4.  **Inheritance (MRO):** Rules targeting a base class (e.g., `Component`)
    apply to subclasses (e.g., `Container`), unless overridden by a more specific
    rule for the subclass.

5.  **Property Setting:**
    *   **Direct Attributes:** Set top-level component attributes (`color: "red"`).
    *   **Nested Attributes:** Use dot notation to set attributes on nested Pydantic
        models like `style` or `transform` (`style.background_color: "#fff"`).
    *   **Partial Updates:** Update nested models or sequences (lists/tuples) partially:
        - For models: Provide a dictionary with only the keys to change.
        - For lists/tuples: Provide a list where `None` preserves the original value
          at that index. Type conversion is attempted.
        ```python
        jstyle.update({
            "Container[id=main]": {
                "style": { # partially update style model
                    "background_color": "lightblue",
                    "padding": [20, None, 20, None] # update top/bottom padding only
                 },
                 "transform": { "rotate": 15 } # partially update transform
            }
        })
        ```

6.  **Applying Styles:**
    *   `jstyle.apply(component)`: Applies all currently defined styles
        recursively to the component and its children, respecting context and
        specificity. This is the standard way to apply styles during layout/render.
    *   `jstyle.update(new_styles)`: Merges new style rules into the global stylesheet.
    *   `with jstyle(temporary_styles): ...`: Creates a temporary context where
        `temporary_styles` are merged with and override global styles. Styles
        revert upon exiting the `with` block.

"""

from contextlib import contextmanager
import copy
import re
from pydantic import BaseModel, Field, ValidationError
from typing import List, Tuple, Dict, Any, Optional, ClassVar, Union, NamedTuple, Callable, Sequence
import inspect
import logging


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # uncomment for detailed style debugging


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
        r"^\s*([^=~!<>*^$]+)\s*(=[/~]?|!=|<=?|>=?|\*=|\^=|\$=)?\s*(.*)\s*$"
    )
    _REGEX_RE: ClassVar = re.compile(r"^(.*)/([ism]*)$")  # pattern, flags

    raw_selector: str
    type_selector: Optional[str] = None
    attributes: List[Tuple[str, Optional[str], Optional[str]]]  # (name, operator, value)
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
        match = self._ATTRIBUTE_SELECTOR_RE.search(selector)
        if match:
            attr_part = match.group(0)
            type_part = selector.split(attr_part)[0].strip()
            if type_part and type_part != "*":
                self.type_selector = type_part
            self._parse_attributes(attr_part[1:-1])  # content inside brackets
        elif selector != "*":
            self.type_selector = selector

    def _parse_attributes(self, attr_content: str):
        """parses comma-separated attribute conditions."""
        # split by comma, but not inside quotes or brackets (basic handling)
        conditions = [
            cond.strip()
            for cond in re.split(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", attr_content)
        ]
        for cond in conditions:
            if not cond:
                continue
            match = self._CONDITION_RE.match(cond)
            if match:
                name, op, val = match.groups()
                # strip quotes from value if present
                if val and val.startswith(("'", '"')) and val.endswith(val[0]):
                    val = val[1:-1]
                self.attributes.append((name.strip(), op, val.strip()))
            # handle simple attribute presence check like '[disabled]'
            elif re.match(r"^\s*[\w.-]+\s*$", cond):
                self.attributes.append((cond.strip(), "exists", None))
            else:
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
        self, component: Any, name: str, op: Optional[str], value_pattern: Optional[str]
    ) -> bool:
        """checks a single attribute condition."""
        actual_value = self._get_attribute_value(component, name)

        op_map: Dict[str, Callable[[Any, Optional[str]], bool]] = {
            "exists": lambda v, p: bool(v),
            "=": lambda v, p: self._value_matches(v, p, operator="="),
            "!=": lambda v, p: self._value_matches(v, p, operator="!="),
            "=~": lambda v, p: self._value_matches(v, p, operator="=~"),
            "^=": lambda v, p: str(v).startswith(p) if v is not None and p is not None else False,
            "$=": lambda v, p: str(v).endswith(p) if v is not None and p is not None else False,
            "*=": lambda v, p: p in str(v) if v is not None and p is not None else False,
            "=/": self._regex_matches,
            "<": lambda v, p: self._numeric_compare(v, p, operator="<"),
            "<=": lambda v, p: self._numeric_compare(v, p, operator="<="),
            ">": lambda v, p: self._numeric_compare(v, p, operator=">"),
            ">=": lambda v, p: self._numeric_compare(v, p, operator=">="),
        }

        # handle list attributes: match if *any* item matches
        if isinstance(actual_value, (list, tuple, set)) and op != "exists":
            # use the specific operator's function for list items
            matcher = op_map.get(op) if op else None
            if matcher and value_pattern is not None:
                return any(matcher(item, value_pattern) for item in actual_value)
            return False  # invalid op for list or no pattern

        # handle non-list attributes
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

    def _regex_matches(self, actual_value: Any, pattern_details: Optional[str]) -> bool:
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

    def _value_matches(
        self, actual_value: Any, pattern_value: Optional[str], operator: str
    ) -> bool:
        """performs basic value comparisons (equality, inequality, case-insensitive)."""
        if pattern_value is None:
            # only makes sense for != check against None/non-None
            return operator == "!=" and actual_value is not None

        # handle 'none' keyword for checking against None
        pattern_is_none_keyword = pattern_value.lower() == "none"
        if operator == "=":
            return (actual_value is None and pattern_is_none_keyword) or (
                actual_value is not None
                and not pattern_is_none_keyword
                and str(actual_value) == pattern_value
            )
        elif operator == "!=":
            return (actual_value is not None or not pattern_is_none_keyword) and (
                actual_value is None
                or pattern_is_none_keyword
                or str(actual_value) != pattern_value
            )
        elif operator == "=~":
            return (
                actual_value is not None
                and not pattern_is_none_keyword
                and str(actual_value).lower() == pattern_value.lower()
            )
        else:
            # other operators handled by specific methods (_numeric_compare, _regex_matches etc)
            logger.warning(f"unexpected operator '{operator}' in _value_matches")
            return False

    def _numeric_compare(
        self, actual_value: Any, pattern_value: Optional[str], operator: str
    ) -> bool:
        """performs numeric comparisons."""
        if actual_value is None or pattern_value is None:
            return False
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
    """holds a parsed selector and its associated declarations."""

    selector: Selector
    properties: Dict[str, Any] = Field(default_factory=dict)
    nested_rules: Dict[str, Any] = Field(default_factory=dict)
    is_context_rule: bool = False
    source_index: int = 0  # order in the original definition

    # store calculated match level during application phase
    match_level: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}


class PropertyApplication(NamedTuple):
    """data needed to decide the winning value for a property."""

    specificity: Specificity
    is_context: bool
    mro_level: int
    source_order: int  # tie-breaker based on definition order
    value: Any


class JStyle:
    """manages style rules and applies them to components."""

    def __init__(self, style_dict: Optional[Dict[str, Any]] = None):
        self._raw_styles: Dict[str, Any] = {}
        self.styles: List[StyleRule] = []  # master list of parsed global rules
        if style_dict:
            self.update(style_dict)

    def update(self, style_dict: Dict[str, Any]):
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

    def _get_applicable_rules(self, component: Any) -> List[StyleRule]:
        """finds all global and context rules matching the component."""
        applicable_rules: List[StyleRule] = []

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

    def _discover_context_rules(self, component: Any) -> List[StyleRule]:
        """walks up the parent chain, finding matching rules and merging their nested contexts."""
        effective_nested_rules: Dict[str, Any] = {}
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

    def _resolve_properties(self, applicable_rules: List[StyleRule]) -> Dict[str, Any]:
        """determines the winning value for each property based on rule priorities."""
        declarations_by_prop: Dict[str, List[PropertyApplication]] = {}

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

    def _parse_style_dict(self, style_dict: Dict[str, Any], is_context: bool) -> List[StyleRule]:
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
                target_obj = getattr(target_obj, part)
                if target_obj is None:
                    logger.warning(
                        f"intermediate attribute '{part}' is None in '{property_name}' for {comp_cls}(id={comp_id})"
                    )
                    return

            current_val = getattr(target_obj, attr_to_set, None)
            final_value = value

            # handle pydantic model update with dict value
            if isinstance(current_val, BaseModel) and isinstance(value, dict):
                logger.debug(f"      handling basemodel update for {property_name}")
                final_value = self._update_pydantic_model(current_val, value, property_name)

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
        self, current_model: BaseModel, update_dict: Dict, prop_name: str
    ) -> BaseModel:
        """handles partial updates for pydantic models, including lists/tuples."""
        current_dict = current_model.model_dump()
        merged_update_dict = {}

        for key, update_val in update_dict.items():
            if key not in current_model.model_fields:
                logger.warning(
                    f"      skipping key '{key}' in update for {prop_name}: not a field in {current_model.__class__.__name__}."
                )
                continue

            current_field_val = current_dict.get(key)

            if isinstance(current_field_val, (list, tuple)) and isinstance(update_val, list):
                # handle partial list/tuple update
                try:
                    merged_update_dict[key] = self._merge_sequence(
                        current_field_val, update_val, current_model.model_fields[key]
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
        self, current_seq: Sequence, update_list: List, field_info: Any
    ) -> Union[List, Tuple]:
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

    def _deep_merge_dicts(self, base: Dict, overlay: Dict) -> Dict:
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
            # fix: pass is_context=False for temporary global override
            self.styles = self._parse_style_dict(context_raw_styles, is_context=False)
            self._raw_styles = context_raw_styles  # also update raw for consistency within context
            yield
        finally:
            # restore original styles
            self.styles = old_parsed_rules
            self._raw_styles = old_raw_styles

    __call__ = context


# global style instance
jstyle = JStyle()
