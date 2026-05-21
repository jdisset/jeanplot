from typing import Any

from jeanplot.core.style_selector import Selector


def parse_jstyle_rule_tree(body: dict[str, Any]) -> dict[Selector, dict[str, Any]]:
    """Flatten `{Sel: {Sel2: {prop: val}}}` into descendant-selector form.

    Returns `{Selector("Sel Sel2"): {prop: val}}` so dracon's select-mode
    cascade sees a flat list of selector → properties pairs.
    """
    out: dict[Selector, dict[str, Any]] = {}

    def walk(prefix: list[str], node: dict[str, Any]):
        leaf: dict[str, Any] = {}
        for key, value in node.items():
            if isinstance(value, dict) and _looks_like_selector(key):
                walk(prefix + [_key_str(key)], value)
            else:
                leaf[key] = value
        if leaf:
            sel = Selector(" ".join(prefix))
            existing = out.get(sel)
            if existing is None:
                out[sel] = leaf
            else:
                existing.update(leaf)

    for selector_key, decls in body.items():
        if isinstance(decls, dict):
            walk([_key_str(selector_key)], decls)

    return out


def _key_str(key: Any) -> str:
    if isinstance(key, Selector):
        return key.raw_selector
    return str(key)


def parse_selector_key(key: Any) -> Selector | None:
    """Per-key parser for dracon `!cascade:jstyle`. Returns a Selector for
    keys that look like selectors, None to skip non-selector keys."""
    if isinstance(key, Selector):
        return key
    if not isinstance(key, str):
        return None
    if not _looks_like_selector(key):
        return None
    return Selector(key)


def _looks_like_selector(key: Any) -> bool:
    if isinstance(key, Selector):
        return True
    if not isinstance(key, str):
        return False
    if not key:
        return False
    if "." in key:
        return False
    head = key.lstrip()
    if not head:
        return False
    first = head[0]
    return first.isupper() or first in ("[", "#", ".", "*")
