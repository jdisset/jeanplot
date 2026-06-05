from jeanplot.core.style_engine import JStyle, jstyle


def closest_attr(component, attrs=("marker", "part_name"), default=None):
    """First non-falsy value of any of ``attrs`` walking ancestors via ``.parent``."""
    cur = component
    while cur is not None:
        for n in attrs:
            v = getattr(cur, n, None)
            if v:
                return v
        cur = getattr(cur, "parent", None)
    return default


def lookup_by_attr(
    component, table, attrs=("marker", "part_name"), default=None, normalize=str.upper
):
    """Look up ``table`` by closest-ancestor ``attrs`` value (normalized, default upper-case)."""
    key = closest_attr(component, attrs)
    if key is None:
        return default
    k = normalize(str(key)) if normalize else str(key)
    return table.get(k, default)


def lookup_by_classes(component, table, default=None, normalize=str.upper):
    """Look up ``table`` by first matching entry in ``component.style_class``."""
    for cls in getattr(component, "style_class", None) or ():
        k = normalize(str(cls)) if normalize else str(cls)
        if k in table:
            return table[k]
    return default


def palette_value(component, table, attrs=None, default=None, channel=None):
    """Unified palette lookup. `attrs=None` → look up by style_class; else by attr.
    `channel=None` → return the whole swatch (or scalar); else index into it."""
    s = (
        lookup_by_classes(component, table, default)
        if attrs is None
        else lookup_by_attr(component, table, attrs, default)
    )
    return s[channel] if channel and isinstance(s, dict) else s


__all__ = [
    "JStyle",
    "jstyle",
    "closest_attr",
    "lookup_by_attr",
    "lookup_by_classes",
    "palette_value",
]
