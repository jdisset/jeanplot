# Step 00 — jstyle becomes a thin layer over `!cascade:jstyle`

## Goal

Rewrite jeanplot's style engine to delegate selector matching,
specificity ordering, and multi-rule merging to dracon's
`!cascade:jstyle` select-mode dialect. This is a **self-contained
engine refactor of the existing scene graph's style code** — no plot
panels, no themes, no figures touched. The plot refactor (steps 01-05)
consumes the result; sequencing this first means none of those steps
ever has to think about jstyle internals.

After this step:
- `jeanplot/core/style_engine.py` shrinks from ~460 LOC to ~120 LOC.
- `JStyle.update(theme_value)` takes the `CallableSymbol` from a
  `!cascade:jstyle` document and stores it. `JStyle.apply_one(component)`
  invokes that symbol with `component=...` to get a flat
  `{path: value}` mapping, then writes those into the component via the
  existing `_set_property` / `_update_pydantic_model` helpers.
- Theme YAML moves from raw nested dicts to `!cascade:jstyle` documents.
  The existing `themes/default.yaml` is rewritten in the new dialect.
- All existing gene/ schematic tests pass against the new engine.

## Why now

Two reasons.

**(1) Orthogonality.** The cascade rewrite is purely about *how rules
are parsed, matched, and merged*. It doesn't depend on plot panels
existing. Doing it first lets the plot refactor target a working,
tested dialect without bundling engine-rewrite risk into the same
step as new panel classes.

**(2) Live-scoped lazies are load-bearing for the bio theme.**
Select-mode `!cascade` auto-opens a `!live` scope for each
`input_params` name. Themes can write `${component.part_name}` and
have it stay lazy until apply-time. The current jstyle engine can't
express this without manual `!each` enumeration. The plan's §4.7 (bio
theme collapses from ~180 LOC to ~30) only works if the dialect is
already in place.

## Prerequisites

None inside this repo. **Dracon dependency:** `register_cascade_strategy`,
`!cascade:NAME(arg)?` (select-mode), and the `!live` scope mechanism
must all be available in the installed dracon. The dracon skill
documents all three.

## What changes

### 0.1 Register the dialect at package init

Add to `jeanplot/__init__.py`, before any theme is loaded:

```python
from dracon import CascadeStrategy, register_cascade_strategy
from jeanplot.core.style_selector import Selector
from jeanplot.core.style_dialect import parse_jstyle_rule_tree

register_cascade_strategy(CascadeStrategy(
    name="jstyle",
    input_params=("component",),
    parse=parse_jstyle_rule_tree,
    matches=lambda sel, c: sel.matches(c),
    specificity=lambda sel: sel.specificity,
))
```

### 0.2 New file: `jeanplot/core/style_dialect.py`

Tiny helper (~40 LOC). Walks the cascade body and flattens
nested-rule trees into descendant selectors:

```python
# style_dialect.py
from jeanplot.core.style_selector import Selector


def parse_jstyle_rule_tree(body: dict) -> dict[Selector, dict]:
    """Flatten `{Sel: {Sel2: {prop: val}}}` into descendant-selector form.

    Returns `{Selector("Sel Sel2"): {prop: val}}` so dracon's select-mode
    cascade sees a flat list of selector → properties pairs.
    """
    out: dict[Selector, dict] = {}

    def walk(prefix: list[str], node: dict):
        leaf: dict[str, object] = {}
        for key, value in node.items():
            if isinstance(value, dict) and _looks_like_selector(key):
                walk(prefix + [key], value)
            else:
                leaf[key] = value
        if leaf:
            sel = Selector(" ".join(prefix)) if len(prefix) > 1 else Selector(prefix[0])
            out[sel] = leaf

    for selector_str, decls in body.items():
        walk([selector_str], decls if isinstance(decls, dict) else {selector_str: decls})

    return out


def _looks_like_selector(key: str) -> bool:
    # Capitalized → class name; "[...]" → attribute selector; "#..." → id;
    # ".cls" → class. Lowercase non-dotted strings are properties.
    if not isinstance(key, str):
        return False
    if "." in key:                    # dotted property path
        return False
    return key[0].isupper() or key.startswith(("[", "#", "."))
```

The `_looks_like_selector` heuristic replaces the current engine's
`_is_key_likely_property`; it inverts the polarity (selectors look
like CSS selectors; everything else is a property path). Cleaner.

### 0.3 Rewrite `jeanplot/core/style_engine.py`

Reduce to ~120 LOC. Keep verbatim:
- `_set_property` (~80 LOC) — sets a dotted property path on a
  Pydantic model with deep-merge for nested model fields.
- `_update_pydantic_model` + `_merge_sequence` (~40 LOC) — the
  Pydantic-aware deep-merge helpers.

Replace everything else:

```python
class JStyle:
    def __init__(self):
        self._cascade = None       # CallableSymbol of kind 'match', or None

    def update(self, cascade_value):
        """cascade_value is a CallableSymbol from a !cascade:jstyle document.

        Replaces, not merges. Layer multiple themes at the dracon level
        via `<<(<): !include` -- the merged tree feeds one cascade.
        """
        self._cascade = cascade_value

    def apply_one(self, component):
        if self._cascade is None or component is None:
            return component
        props = self._cascade.invoke(component=component)
        for path, value in props.items():
            self._set_property(component, path, value)
        return component

    def apply(self, component):
        if component is None:
            return
        self.apply_one(component)
        for child in getattr(component, "children", []) or []:
            if getattr(child, "parent", None) is not component:
                child.parent = component
            self.apply(child)
        return component

    def clear(self):
        self._cascade = None

    @contextmanager
    def context(self, cascade_value):
        old = self._cascade
        self._cascade = cascade_value
        try:
            yield
        finally:
            self._cascade = old

    __call__ = context

    # _set_property, _update_pydantic_model, _merge_sequence -- kept verbatim.
```

What disappears (delete from `style_engine.py` entirely):

| Removed | Why |
|---|---|
| `_parse_style_dict` | Replaced by `parse_jstyle_rule_tree` in the dialect file |
| `_is_key_likely_property` | Replaced by inverted-polarity `_looks_like_selector` |
| `_get_applicable_rules` | dracon's select-mode walks the rule tree |
| `_discover_context_rules` | Descendant pattern handled by parse-time flattening |
| `_resolve_properties` | Specificity-ordered merge handled by dracon's `_cascade_select` |
| `_deep_merge_dicts` | dracon's `<<{+<}[~<]` merge is canonical |
| `StyleRule`, `PropertyApplication` dataclasses | Engine doesn't materialise them anymore |

### 0.4 Rewrite `themes/default.yaml` in the cascade dialect

Existing scene-graph theme moves from raw nested-dict form to a
`!cascade:jstyle` document. The body is structurally similar — same
selector keys, same property paths — but wrapped under
`rules: !cascade:jstyle` so the engine recognizes it.

Before (current):
```yaml
SVGElement:
  fill: ${default_fill}
Text:
  font_size: 12
```

After:
```yaml
rules: !cascade:jstyle
  SVGElement:
    fill: ${default_fill}
  Text:
    font_size: 12
```

`load_default_theme()` in `jeanplot/__init__.py`:

```python
def load_default_theme(force: bool = False):
    global _DEFAULT_THEME_CACHE
    if _DEFAULT_THEME_CACHE is None or force:
        cfg = dr.load(
            "pkg:jeanplot:resources/themes/default.yaml",
            enable_interpolation=True,
            context=make_context_from_types(DEFAULT_TYPES),
        )
        dr.resolve_all_lazy(cfg, except_for={"component"})  # leave live lazies
        _DEFAULT_THEME_CACHE = cfg["rules"]
    jstyle.clear()
    jstyle.update(_DEFAULT_THEME_CACHE)
```

Note `except_for={"component"}` — leaves the `!live`-scoped lazies
that reference `${component.X}` un-resolved so they bind per-component
at apply time.

### 0.5 Tests

- `tests/test_jstyle_cascade_basic.py` — single-rule cascade applies
  a property to a matching component, leaves a non-matching one alone.
- `tests/test_jstyle_specificity.py` — class selector beats type
  selector; id selector beats class selector. Parity with old engine.
- `tests/test_jstyle_descendant.py` — nested-rule form
  `Container: { Text: { color: red } }` only matches a `Text` whose
  ancestor chain includes a `Container`. Parity with old engine.
- `tests/test_jstyle_live_scope.py` — a rule body containing
  `${component.font_size_override or 12}` resolves per-component
  against each component's actual attribute.
- All existing gene/ schematic tests must pass unchanged.

## Verification

```bash
pytest jeanplot/tests/ -v
dracon show pkg:jeanplot:resources/themes/default.yaml -c -r
```

The full existing jeanplot test suite is green and the rewritten
theme document resolves cleanly.

## Out of scope

- No plot panels (step 02).
- No `themes/plots.yaml` (step 04).
- No biocomp / biocomp-tools changes — ever.

## Estimate

~-300 LOC in `style_engine.py` (460 → ~120). +40 LOC in
`style_dialect.py`. Minor edits to `__init__.py` and `themes/default.yaml`.
Tests: ~150 LOC. **Net delta for step 00: ~-100 LOC.** Already
net-negative before the plot refactor begins.
