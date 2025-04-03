from contextlib import contextmanager
import copy
import re

"""
style system for jeanplot components that supports nested dictionary syntax:

selectors:
- type selectors: "Container", "ERN" - match component class name
- attribute selectors: "[id=foo]", "[style_class=primary]" - match attribute values
- multi-attribute: "[id=foo, style_class=primary]" - AND condition
- wildcard: "*" - matches any component
- nested selectors: dict nesting targets children of matching components

property paths:
- direct props: "main_color": "red" - set component.main_color
- nested props: "style.background_color": "#fff" - set component.style.background_color

specificity:
- more specific selectors (by ID, attributes) override less specific ones
- attribute selector specificity: [id=x] > [style_class=y] > [part_name=z]
- selectors are applied in specificity order to prevent conflicts

basic usage:
    jstyle.styles = {
        "ERN": {"main_color": "#AAAAAA"},
        "Container": {
            "style.background_color": "white",
            "ERN": {"main_color": "red"}  # ERNs inside Containers
        },
        "[id=special]": {"main_color": "blue"},  # target by ID
        "[style_class=primary]": {"style.border_width": 2}  # utility class
        "ERN[style_class=primary]": {"style.border_width": 2}  # selct by type and attribute
    }
    
    jstyle.apply(component)  # apply to specific component (and children)
    
    # temporary context override
    with jstyle({"ERN": {"main_color": "green"}}):
        # components created here get temporary styles
        ern = ERN()
"""


class JStyle:
    """nested dictionary-based style system"""

    def __init__(self, style_dict=None):
        self.styles = style_dict or {}

    def apply(self, component):
        """apply styles to component and its children"""
        self._apply_styles_to_component(component, self.styles)
        return component

    def _is_property_key(self, key, component):
        if "[" in key:
            return False
        if "." in key:
            return True
        if key in ("style", "transform", "offset") or hasattr(component, key):
            return True
        return False

    def _merge_styles(self, target, source):
        """Recursively merge two style dictionaries. The source values override target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                target[key] = self._merge_styles(copy.deepcopy(target[key]), value)
            else:
                target[key] = value
        return target

    def _apply_styles_to_component(self, component, style_dict):
        """
        Apply styles to a component by first splitting the style dictionary into
        base properties (which set component attributes) and nested selectors
        (which target either the current component or its children).

        Matching selectors are merged (in order of increasing specificity) into the base
        properties. Then, for each child component, any nested selectors that match are merged
        into a child style dictionary which is applied recursively.
        """
        # separate into base properties and nested selectors.
        base_props = {}
        nested_selectors = {}
        for key, value in style_dict.items():
            if isinstance(value, dict):
                # TODO: more robust selector/property detection

                # if key contains '[', it is definitely a selector.
                if "[" in key:
                    nested_selectors[key] = value
                elif (
                    "." in key or key in ("style", "transform", "offset") or hasattr(component, key)
                ):  # most likely a property
                    base_props[key] = value
                else:  # assume it's a selector by default
                    nested_selectors[key] = value
            else:
                base_props[key] = value

        # merge properties from any selectors that match this component.
        merged_props = copy.deepcopy(base_props)
        matching_selectors = []
        for selector, sub_dict in nested_selectors.items():
            if self._selector_matches(selector, component):
                spec = self._calculate_specificity(selector)
                matching_selectors.append((spec, selector, sub_dict))
        matching_selectors.sort(key=lambda x: x[0])
        for spec, selector, sub_dict in matching_selectors:
            # from the sub-dict, only extract keys that qualify as properties for this component.
            sub_props = {}
            for key, value in sub_dict.items():
                if self._is_property_key(key, component):
                    sub_props[key] = value
            merged_props = self._merge_styles(merged_props, sub_props)

        # apply the merged properties to the component.
        for key, value in merged_props.items():
            self._set_property(component, key, value)

        # process children: gather nested selectors that match the child.
        if hasattr(component, "children") and component.children:
            for child in component.children:
                child_style = {}
                # first, check all top-level nested selectors from the parent's style dict.
                for sel, sub_dict in nested_selectors.items():
                    if self._selector_matches(sel, child):
                        child_style = self._merge_styles(child_style, sub_dict)
                # next, for each selector that matched the parent, look for nested selectors (i.e. keys
                # in the matching sub-dict that are not properties) that might match the child.
                for spec, parent_sel, parent_sub in matching_selectors:
                    for child_sel, child_sub in parent_sub.items():
                        if isinstance(child_sub, dict) and not self._is_property_key(
                            child_sel, child
                        ):
                            if self._selector_matches(child_sel, child):
                                child_style = self._merge_styles(child_style, child_sub)
                if child_style:
                    self._apply_styles_to_component(child, child_style)

    def _selector_matches(self, selector, component):
        """check if selector matches component"""
        # handle combined type and attribute selector (ERN[attr=val])
        if "[" in selector and not selector.startswith("["):
            # extract type and attribute parts
            type_part = selector.split("[")[0]
            attr_part = "[" + selector.split("[", 1)[1]

            # both parts must match
            return type_part == component.__class__.__name__ and self._selector_matches(
                attr_part, component
            )

        # handle type selector (ERN, Container, etc)
        if selector == component.__class__.__name__:
            return True

        # handle multi-attribute selector [attr1=value1, attr2=value2]
        if selector.startswith("[") and selector.endswith("]"):
            attr_expr = selector[1:-1]

            # split multiple conditions by comma
            if "," in attr_expr:
                conditions = [cond.strip() for cond in attr_expr.split(",")]
                return all(self._check_attribute_condition(cond, component) for cond in conditions)

            # single attribute condition
            return self._check_attribute_condition(attr_expr, component)

        # handle wildcard
        if selector == "*":
            return True

        return False

    def _calculate_specificity(self, selector):
        """calculate selector specificity for ordering"""
        specificity = 0

        # combined selectors have higher specificity
        if "[" in selector and not selector.startswith("["):
            # extract type and attribute parts
            type_part = selector.split("[")[0]
            attr_part = "[" + selector.split("[", 1)[1]

            # calculate specificity for each part
            type_specificity = 10 if type_part != "*" else 0
            attr_specificity = self._calculate_specificity(attr_part)

            return type_specificity + attr_specificity

        # type selectors have base specificity
        if selector not in ("*", "") and not selector.startswith("["):
            specificity += 10

        # each attribute condition adds to specificity
        if selector.startswith("[") and selector.endswith("]"):
            attr_expr = selector[1:-1]
            conditions = [cond.strip() for cond in attr_expr.split(",")]
            specificity += len(conditions) * 100

            # id selectors have highest specificity
            if any(cond.startswith("id=") for cond in conditions):
                specificity += 1000

        return specificity

    def _check_attribute_condition(self, condition, component):
        """check a single attribute condition (e.g., 'id=foo', 'name=~bar', 'class=/item-\\d+/i')"""
        # determine operator and split
        attr_name = ""
        op = ""
        value_pattern = ""

        # order matters: check for regex first, then case-insensitive, then exact
        if "=/" in condition and condition.endswith("/"):  # naive check for regex start/end
            parts = condition.split("=/", 1)
            if len(parts) == 2:
                # try to split pattern and flags
                pattern_flags = parts[1].rsplit("/", 1)
                if (
                    len(pattern_flags) == 2 and len(pattern_flags[1]) <= 3
                ):  # allow up to 3 flags (e.g., 'i', 'm', 's')
                    attr_name = parts[0].strip()
                    op = "=/"
                    value_pattern = pattern_flags[0]  # the regex pattern
                    flags_str = pattern_flags[1]  # the flags (e.g., 'i')
                else:  # doesn't look like pattern/flags, treat as normal string
                    pass  # fallback to other checks

        if not op:  # if regex didn't match format
            if "=~" in condition:
                parts = condition.split("=~", 1)
                if len(parts) == 2:
                    attr_name = parts[0].strip()
                    op = "=~"
                    value_pattern = parts[1].strip()
            elif "=" in condition:
                parts = condition.split("=", 1)
                if len(parts) == 2:
                    attr_name = parts[0].strip()
                    op = "="
                    value_pattern = parts[1].strip()
            else:
                return False  # invalid condition format

        # clean up value pattern if it's not regex
        if (
            op != "=/"
            and value_pattern.startswith(("'", '"'))
            and value_pattern.endswith(value_pattern[0])
        ):
            value_pattern = value_pattern[1:-1]

        # get component's actual value
        actual_value = self._get_attribute_value(component, attr_name)
        if actual_value is None:
            return False

        # handle list values: check if *any* item matches
        if isinstance(actual_value, (list, tuple, set)) and not isinstance(actual_value, str):
            if op == "=/":
                return any(
                    self._regex_matches(item, value_pattern, flags_str) for item in actual_value
                )
            else:
                return any(self._value_matches(item, value_pattern, op) for item in actual_value)
        else:
            # handle single value
            if op == "=/":
                return self._regex_matches(actual_value, value_pattern, flags_str)
            else:
                return self._value_matches(actual_value, value_pattern, op)

    def _regex_matches(self, actual_value, pattern, flags_str):
        """perform regex matching"""
        actual_str = str(actual_value)
        re_flags = 0
        if "i" in flags_str:
            re_flags |= re.IGNORECASE  # case-insensitive (eq to using =~)
        if "m" in flags_str:
            re_flags |= re.MULTILINE  # multiline means '^' and '$' match start/end of lines
        if "s" in flags_str:
            re_flags |= re.DOTALL  # dotall means '.' matches any character, including newline
        # using re.search to find pattern anywhere in string, use re.match for start only
        try:
            return bool(re.search(pattern, actual_str, re_flags))
        except re.error as e:
            # handle invalid regex patterns gracefully in styles
            print(f"Warning: Invalid regex pattern '{pattern}' in style selector: {e}")
            return False

    def _value_matches(self, actual_value, pattern_value, operator):
        """check if a single value matches using =, =~ operators"""
        actual_str = str(actual_value)

        has_wildcard = "*" in pattern_value or "?" in pattern_value
        if has_wildcard:
            # convert wildcard pattern to regex
            regex_pattern = pattern_value.replace(".", "\\.").replace("*", ".*").replace("?", ".")
            regex_pattern = f"^{regex_pattern}$"  # anchor wildcard match

            # case sensitivity depends on original operator
            flags = re.IGNORECASE if operator == "=~" else 0
            try:
                return bool(re.match(regex_pattern, actual_str, flags))
            except re.error as e:
                print(f"Warning: Invalid wildcard pattern '{pattern_value}' in style selector: {e}")
                return False
        else:
            # standard comparison without wildcards
            if operator == "=":  # exact match
                return actual_str == pattern_value
            elif operator == "=~":  # case-insensitive match
                # check type to avoid errors comparing non-strings case-insensitively
                return actual_str.lower() == pattern_value.lower()
            else:
                return False  # should not happen

    def _get_attribute_value(self, component, attr_name):
        """get attribute value, handling nested paths"""
        # handle nested attributes (style.background_color)
        if "." in attr_name:
            parts = attr_name.split(".")
            obj = component
            for part in parts[:-1]:
                # handle potential None values during traversal
                if obj is None or not hasattr(obj, part):
                    return None
                obj = getattr(obj, part)

            # check final part on the potentially None object
            return getattr(obj, parts[-1]) if obj is not None and hasattr(obj, parts[-1]) else None

        # direct attribute
        return getattr(component, attr_name, None)  # use default=None

    def _set_property(self, component, property_name, value):
        """set a property on a component"""
        # handle nested properties (style.background_color)
        if "." in property_name:
            parts = property_name.split(".")
            obj = component

            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return

            if hasattr(obj, parts[-1]):
                setattr(obj, parts[-1], value)

        elif hasattr(component, property_name):  # direct property
            setattr(component, property_name, value)

    @contextmanager
    def context(self, style_dict):
        """temporary style context"""
        old_styles = copy.deepcopy(self.styles)

        try:
            # create a merged style dictionary
            merged = copy.deepcopy(old_styles)
            self._merge_styles(merged, style_dict)
            self.styles = merged
            yield
        finally:
            self.styles = old_styles

    # shorthand for context
    __call__ = context


# global style instance
jstyle = JStyle()
