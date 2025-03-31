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

    def _apply_styles_to_component(self, component, style_dict):
        """apply styles from dict to component and recursively process children"""
        # separate properties and selectors
        properties = {}
        selectors = {}

        for key, value in style_dict.items():
            if isinstance(value, dict):
                # Check if this is a property path or a selector
                if "." in key or key in ("style", "transform", "offset") or hasattr(component, key):
                    # This is likely a nested property (like style.background_color)
                    properties[key] = value
                else:
                    # This is a selector for child components
                    selectors[key] = value
            else:
                # Direct property
                properties[key] = value

        for key, value in properties.items():
            self._set_property(component, key, value)

        # find matching child selectors
        matching_selectors = []
        for selector, sub_dict in selectors.items():
            if self._selector_matches(selector, component):
                specificity = self._calculate_specificity(selector)
                matching_selectors.append((selector, sub_dict, specificity))

        matching_selectors.sort(key=lambda x: x[2])

        for selector, sub_dict, _ in matching_selectors:
            for key, value in sub_dict.items():
                if (
                    not isinstance(value, dict)
                    or "." in key
                    or key in ("style", "transform", "offset")
                    or hasattr(component, key)
                ):
                    self._set_property(component, key, value)

        # process children
        if hasattr(component, "children") and component.children:
            for child in component.children:
                # apply all selectors that might match children
                for selector, sub_dict in selectors.items():
                    # if child type matches selector, apply entire subdictionary
                    if self._selector_matches(selector, child):
                        self._apply_styles_to_component(child, sub_dict)

                # also apply from matching parent selectors
                for _, matched_dict, _ in matching_selectors:
                    # recursively apply children selectors from matched parent
                    for child_selector, child_dict in matched_dict.items():
                        if (
                            isinstance(child_dict, dict)
                            and not "." in child_selector
                            and not child_selector in ("style", "transform", "offset")
                        ):
                            if self._selector_matches(child_selector, child):
                                self._apply_styles_to_component(child, child_dict)

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
        """check a single attribute=value condition"""
        if "=" not in condition:
            return False

        attr_name, attr_value = condition.split("=", 1)
        attr_name = attr_name.strip()
        attr_value = attr_value.strip()

        # strip quotes if present
        if attr_value.startswith(("'", '"')) and attr_value.endswith(attr_value[0]):
            attr_value = attr_value[1:-1]

        # handle nested attributes (style.background_color)
        if "." in attr_name:
            parts = attr_name.split(".")
            obj = component

            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return False

            return hasattr(obj, parts[-1]) and str(getattr(obj, parts[-1])) == attr_value

        # simple attribute
        return hasattr(component, attr_name) and str(getattr(component, attr_name)) == attr_value

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

        # direct property
        elif hasattr(component, property_name):
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

    def _merge_styles(self, target, source):
        """recursively merge style dictionaries"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_styles(target[key], value)
            else:
                target[key] = value

    # shorthand for context
    __call__ = context


# global style instance
jstyle = JStyle()
