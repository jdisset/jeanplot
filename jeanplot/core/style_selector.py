import inspect
import logging
import re
from typing import Any, Callable, ClassVar, NamedTuple

logger = logging.getLogger(__name__)


class Specificity(NamedTuple):
    """Selector specificity; higher values win."""

    id_count: int = 0
    attr_class_count: int = 0
    type_count: int = 0


class _SimpleSelector:
    """One segment of a (possibly compound) selector: type + attribute conditions."""

    _ATTRIBUTE_SELECTOR_RE: ClassVar = re.compile(r"\[([^\]]+)\]")
    _CONDITION_RE: ClassVar = re.compile(
        r"^\s*([^=~!<>*^$!]+)\s*(=[/~]?|!=|<=?|>=?|\*=|\^=|\$=)?\s*(.*)\s*$"
    )
    _REGEX_RE: ClassVar = re.compile(r"^(.*)/([ism]*)$")
    _PRESENCE_RE: ClassVar = re.compile(r"^\s*(!)?([\w.-]+)\s*$")

    def __init__(self, segment: str):
        self.raw = segment.strip()
        self.type_selector: str | None = None
        self.attributes: list[tuple[str, str | None, str | None]] = []
        self._parse()
        self.specificity = self._calculate_specificity()

    def _parse(self):
        selector = self.raw
        matches = list(self._ATTRIBUTE_SELECTOR_RE.finditer(selector))

        if not matches:
            if selector and selector != "*":
                self.type_selector = selector
            return

        last_match_end = 0
        combined_attr_content = ""
        if not selector.startswith("["):
            first_match_start = matches[0].start()
            type_part = selector[:first_match_start].strip()
            if type_part and type_part != "*":
                self.type_selector = type_part
            last_match_end = matches[0].end()
            combined_attr_content += matches[0].group(1)
        else:
            last_match_end = matches[0].end()
            combined_attr_content += matches[0].group(1)

        for i in range(1, len(matches)):
            if matches[i].start() == last_match_end:
                combined_attr_content += "," + matches[i].group(1)
                last_match_end = matches[i].end()
            else:
                logger.warning(
                    "ignoring attribute content after non-adjacent bracket in selector: '%s'",
                    self.raw,
                )
                break

        self._parse_attributes(combined_attr_content)

    def _parse_attributes(self, attr_content: str):
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
                continue

            match_cond = self._CONDITION_RE.match(cond)
            if match_cond:
                name, op, val = match_cond.groups()
                if val and len(val) >= 2 and val.startswith(("'", '"')) and val.endswith(val[0]):
                    val = val[1:-1]
                self.attributes.append((name.strip(), op or "=", val.strip()))
                continue

            logger.warning("could not parse attribute condition: '%s' in '%s'", cond, self.raw)

    def _calculate_specificity(self) -> Specificity:
        ids = sum(1 for name, _, _ in self.attributes if name == "id")
        attrs_classes = sum(1 for name, _, _ in self.attributes if name != "id")
        types = 1 if self.type_selector else 0
        return Specificity(ids, attrs_classes, types)

    def matches(self, component: Any) -> bool:
        return self._matches_type(component) and self._matches_attributes(component)

    def _matches_type(self, component: Any) -> bool:
        if not self.type_selector:
            return True
        return any(cls.__name__ == self.type_selector for cls in inspect.getmro(component.__class__))

    def _matches_attributes(self, component: Any) -> bool:
        return all(
            self._check_condition(component, name, op, val) for name, op, val in self.attributes
        )

    def _check_condition(
        self,
        component: Any,
        name: str,
        op: str | None,
        value_pattern: str | None,
    ) -> bool:
        actual_value = self._get_attribute_value(component, name)
        attr_exists = actual_value is not None

        op_map: dict[str, Callable[[Any, str | None], bool]] = {
            "exists": lambda v, p: attr_exists and bool(v),
            "not_exists": lambda v, p: not attr_exists or not bool(v),
            "=": lambda v, p: attr_exists and self._value_matches(v, p, operator="="),
            "!=": lambda v, p: not attr_exists or self._value_matches(v, p, operator="!="),
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

        list_ops = {"=", "!=", "=~", "^=", "$=", "*=", "=/", "<", "<=", ">", ">="}
        if isinstance(actual_value, (list, tuple, set)) and op in list_ops:
            matcher = op_map.get(op)
            if matcher and value_pattern is not None:
                return any(matcher(item, value_pattern) for item in actual_value)
            return False

        matcher = op_map.get(op) if op else None
        if matcher:
            return matcher(actual_value, value_pattern)

        logger.warning("unknown operator '%s' for attribute '%s'", op, name)
        return False

    def _get_attribute_value(self, component: Any, attr_name: str) -> Any:
        try:
            obj = component
            for part in attr_name.split("."):
                if obj is None:
                    return None
                if part.isdigit() and isinstance(obj, (list, tuple)):
                    idx = int(part)
                    obj = obj[idx] if 0 <= idx < len(obj) else None
                else:
                    obj = getattr(obj, part, None)
            return obj
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    def _regex_matches(self, actual_value: Any, pattern_details: str | None) -> bool:
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
        except re.error as exc:
            logger.warning("regex error in selector '%s': %s", self.raw, exc)
            return False

    def _value_matches(self, actual_value: Any, pattern_value: str | None, operator: str) -> bool:
        if pattern_value is None:
            return False

        pattern_is_none_keyword = pattern_value.lower() == "none"
        if operator == "=":
            return (actual_value is None and pattern_is_none_keyword) or (
                actual_value is not None
                and not pattern_is_none_keyword
                and str(actual_value) == pattern_value
            )
        if operator == "!=":
            return (
                actual_value is not None
                and (not pattern_is_none_keyword or str(actual_value) != pattern_value)
            ) or (actual_value is None and not pattern_is_none_keyword)
        if operator == "=~":
            return (
                actual_value is not None
                and not pattern_is_none_keyword
                and str(actual_value).lower() == pattern_value.lower()
            )

        return False

    def _numeric_compare(self, actual_value: Any, pattern_value: str | None, operator: str) -> bool:
        if pattern_value is None:
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
            return False

    def get_mro_level(self, component: Any) -> int:
        if self.type_selector:
            for i, cls in enumerate(inspect.getmro(component.__class__)):
                if cls.__name__ == self.type_selector:
                    return i
        return 1 << 30


def _split_descendant(selector_str: str) -> list[str]:
    """Split a compound selector on whitespace, but only outside `[...]` brackets."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in selector_str:
        if ch == "[":
            depth += 1
            buf.append(ch)
        elif ch == "]":
            depth -= 1
            buf.append(ch)
        elif ch.isspace() and depth == 0:
            if buf:
                parts.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


class Selector:
    """Compound CSS-like selector. Whitespace separates descendant segments;
    the rightmost segment matches the component itself, earlier segments match
    ancestors (in order, not necessarily adjacent)."""

    raw_selector: str
    segments: list[_SimpleSelector]
    specificity: Specificity

    def __init__(self, selector_str: str):
        self.raw_selector = selector_str.strip()
        parts = _split_descendant(self.raw_selector)
        if not parts:
            parts = ["*"]
        self.segments = [_SimpleSelector(p) for p in parts]
        self.specificity = self._combine_specificity()

    def _combine_specificity(self) -> Specificity:
        ids = sum(s.specificity.id_count for s in self.segments)
        attrs = sum(s.specificity.attr_class_count for s in self.segments)
        types = sum(s.specificity.type_count for s in self.segments)
        return Specificity(ids, attrs, types)

    def matches(self, component: Any) -> bool:
        if not self.segments[-1].matches(component):
            return False
        if len(self.segments) == 1:
            return True

        ancestors_needed = list(reversed(self.segments[:-1]))
        parent = getattr(component, "parent", None)
        for needed in ancestors_needed:
            while parent is not None and not needed.matches(parent):
                parent = getattr(parent, "parent", None)
            if parent is None:
                return False
            parent = getattr(parent, "parent", None)
        return True

    def get_mro_level(self, component: Any) -> int:
        return self.segments[-1].get_mro_level(component)

    def __repr__(self):
        return f"<Selector('{self.raw_selector}', spec={self.specificity})>"

    def __hash__(self):
        return hash(self.raw_selector)

    def __eq__(self, other):
        return isinstance(other, Selector) and self.raw_selector == other.raw_selector
