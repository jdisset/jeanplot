from typing import Any

from dracon import Locator, compose_nested_locators, parse_locator


def parse_jstyle_rule_tree(body: dict[Any, Any]) -> dict[Locator, Any]:
    """Flatten `{Sel: {Sel2: {prop: val}}}` into `{Locator("Sel Sel2"): {prop: val}}`
    (nesting == descendant combinator), so the cascade sees flat locator → props.

    Idempotent: an already-flat (Locator-keyed) tree passes through unchanged.
    `_as_cascade` mutates the cached cascade's rule_tree in place, so a second theme
    apply re-runs this; re-composing would stringify the Locator keys into garbage."""
    if body and all(isinstance(k, Locator) for k in body):
        return dict(body)
    return compose_nested_locators(body)


def parse_selector_key(key: Any) -> Locator | None:
    """Per-key parser for the jstyle cascade. Returns a Locator for keys that look
    like selectors, None to skip plain config keys."""
    if isinstance(key, Locator):
        return key
    if isinstance(key, str) and _looks_like_selector(key):
        return parse_locator(key)
    return None


def _looks_like_selector(key: Any) -> bool:
    if isinstance(key, Locator):
        return True
    if not isinstance(key, str) or not key or "." in key:
        return False
    head = key.lstrip()
    if not head:
        return False
    first = head[0]
    return first.isupper() or first in ("[", "#", ".", "*")
