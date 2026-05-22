# 01 — Foundation: cascade-fill + Python/YAML ergonomics

## What

One architectural pass that touches `Component` / `Container` / `JStyle` and lands the primitive every later step depends on: **cascade now fills defaults, never clobbers**. Plus two small ergonomic finishers that ride along because they touch the same construction path: positional `*children` on `Container`, and a `"row gap=1"` string DSL on `LayoutConstraints`.

## Why

Three failures of the current architecture all trace back to the same root: `JStyle._set_property` unconditionally `setattr`s onto the component, so theme rules overwrite explicit per-panel YAML. This already bit us once (the `PlotPanel: rescaler: ${IdentityRescaler()}` rule clobbered explicit panel rescalers and silently broke fig1). It forces every paper-jobs panel to repeat `rescaler: ${_rescaler}`, `knn_grid_params: ${_shared_knn_grid}`, etc. — because the theme cannot safely carry those.

CSS-shaped semantics ("theme as defaults, instance overrides win, `!important` available if you need clobber") is what users expect from anything that calls itself a cascade. Once this lands, *every later step* leans on the theme as a true default supplier:

- Step 02's `axes_size` / `colorbar_pad` / `legend_pad` become cascade-fillable knobs.
- Step 04's paper-domain defaults move from per-panel YAML into theme rules; only genuinely-parametric compositions remain as `!fn:target` aliases.

The two ergonomic finishers (`Container(a, b)` positional children, layout-string DSL) are bundled here because they all touch `Component` / `Container` construction. Landing them together avoids retesting the construction path three separate times.

## Current state

- `jeanplot/core/style_engine.py:116` — `JStyle._set_property` does `setattr(target_obj, attr_to_set, final_value)` unconditionally.
- `jeanplot/core/style_engine.py:17` — `_JSTYLE_STRATEGY` registered once; one strategy total.
- `Component` (`jeanplot/core/component.py:36`) is a Pydantic `BaseModel` — `self.model_fields_set` is populated by construction-time kwargs, but **also by every subsequent `setattr`** (`validate_assignment=True`). So we cannot read `model_fields_set` at apply time and trust it.
- `Container` (`jeanplot/core/container.py:13`) inherits Pydantic's keyword-only `__init__`; positional `Container(a, b)` raises `TypeError`. A `_bare_list_is_children` `model_validator(mode="before")` exists, so `Container.model_validate([a, b])` works but `Container(a, b)` does not.
- `LayoutConstraints` (`jeanplot/core/models.py:218`) takes a typed model body. Every `!Container` / `!Figure` / `!PlotPanel` declares `layout: !LayoutConstraints { direction: row, gap: 1.0 }` — 53 characters of noise per Container, multiplied by every container in every figure file.

## Design

Four moves. Each is a few lines.

### (a) Snapshot user-set fields at construction

```python
# Component
_user_set_fields: set[str] = PrivateAttr(default_factory=set)

def model_post_init(self, _context):
    # snapshot now; later setattr (incl. jstyle apply) won't touch this set
    object.__setattr__(self, "_user_set_fields", set(self.model_fields_set))
```

`model_post_init` runs after `__init__` completes; `model_fields_set` at that point contains exactly the kwargs the user supplied. Freezing it into a private attr means later jstyle assignments don't pollute it.

### (b) Cascade skips fields the user set

```python
# JStyle._set_property — at the very top, before walking parts/setattr
if attr_to_set in getattr(target_obj, "_user_set_fields", set()):
    return
```

Nested property paths (`heatmap_params.cmap`) target `attr_to_set` on a *child* object (the `heatmap_params` dict). The skip is per-leaf — if the user set `heatmap_params: { cmap: x }` explicitly, the `cmap` key in their dict is what survives; theme `heatmap_params.cmap` rules fill missing keys in the same dict (already the existing dict-merge path in `_set_property`, just confirm it still works).

### (c) Register `!cascade:jstyle_fill` strategy alias

Keep `!cascade:jstyle` for callers who *want* clobber semantics (rare but legitimate — e.g., forced theme reset). The fill variant becomes the recommended default.

```python
# style_engine.py
_JSTYLE_FILL_STRATEGY = CascadeStrategy(
    name="jstyle_fill",
    input_params=("component",),
    parse=parse_selector_key,
    matches=lambda sel, component: sel.matches(component),
    specificity=lambda sel: tuple(sel.specificity),
)
register_cascade_strategy(_JSTYLE_FILL_STRATEGY)
```

Then `_as_cascade` and the `JStyle.apply_one` path read the strategy name off the `CallableSymbol` and thread `clobber: bool` into `_set_property`. Jeanplot's in-tree themes (`default.yaml`, `plots.yaml`) switch to `!cascade:jstyle_fill`. (The paper theme lives in `paper-jobs/common/plot_config_paper.yaml` and is migrated in step 04, not here.)

Restore the `PlotPanel: rescaler: ${IdentityRescaler()}` rule in `plots.yaml` — under fill semantics it's now a safe default.

### (d) Positional `*children` on `Container`

```python
class Container(Component):
    children: list[Component] = Field(default_factory=list)
    layout: LayoutConstraints = Field(default_factory=LayoutConstraints)

    def __init__(self, *args, **kwargs):
        if args:
            if "children" in kwargs:
                raise TypeError(
                    f"{type(self).__name__}: positional children and `children=` are mutually exclusive"
                )
            kwargs["children"] = list(args)
        super().__init__(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def _bare_list_is_children(cls, v):
        return {"children": v} if isinstance(v, list) else v
```

`Figure(p1, p2, theme=t)` and `SmoothPanel2D(overlay, plot_data=…)` now work. `Pydantic`'s field validation still runs (we only intercept `args → children` then delegate). `model_fields_set` correctly records `children` as user-set, so the snapshot from (a) keeps cascade-fill working.

### (e) Layout-string DSL

```python
# jeanplot/core/models.py
def _parse_layout_string(v):
    if not isinstance(v, str):
        return v
    parts = v.split()
    if not parts:
        return v
    direction, *kvs = parts
    if direction not in ("row", "column", "col"):
        raise ValueError(f"layout: first token must be row|column, got {direction!r}")
    kwargs = {"direction": "column" if direction == "col" else direction}
    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"layout: expected key=value, got {kv!r}")
        k, _, v = kv.partition("=")
        kwargs[_resolve_alias(k)] = _coerce(v)
    return kwargs

_ALIASES = {"align": "align_items", "justify": "justify_content"}

def _coerce(s):
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return s

LayoutConstraintsField = Annotated[
    LayoutConstraints,
    BeforeValidator(_parse_layout_string),
]
```

`Container.layout` annotation switches to `LayoutConstraintsField`. Three input shapes are now accepted: typed `!LayoutConstraints { … }`, mapping `{ direction: row, gap: 1 }`, string `"row gap=1.0 align=stretch"`.

## Implementation steps

1. Add `_user_set_fields` private attr + `model_post_init` snapshot on `Component`. `Component.model_rebuild(force=True)` at the bottom of `component.py` to refresh subclasses.
2. Add `_JSTYLE_FILL_STRATEGY` registration in `style_engine.py`. Thread strategy name through `_as_cascade` and `JStyle.apply_one` to `_set_property` as a `clobber: bool` flag.
3. In `JStyle._set_property` (and `_update_pydantic_model` for nested dicts), gate the `setattr` / dict-key write on the clobber flag. Skip when `attr_to_set in target_obj._user_set_fields` (top-level) or when the dict already has that key (nested dict default merge).
4. Migrate jeanplot in-tree themes: `jeanplot/resources/themes/default.yaml`, `plots.yaml` — change `!cascade:jstyle` → `!cascade:jstyle_fill`. Restore `PlotPanel: rescaler: IdentityRescaler()`.
5. Add `__init__` override to `Container` for positional `*children`.
6. Add `_parse_layout_string` + `LayoutConstraintsField` to `jeanplot/core/models.py`; switch `Container.layout` type annotation.
7. Smoke tests: `Container(Text(text="a"), Text(text="b")).children` has 2 elements; `Container(layout="row gap=1").layout.gap == 1.0`; `Figure(panel, theme=t)` works.

## Tests

`tests/test_cascade_fill.py`:
- Explicit panel value wins over theme rule (the regression that drove this refactor).
- Theme rule fills missing field when panel didn't set it.
- Nested dict (`heatmap_params.cmap`) merges per-key: user's `cmap` stays, theme's `bad_color` fills.
- Descendant selectors (`SmoothPanel3D SmoothPanel2D`) still work under fill semantics.
- Reapplying jstyle multiple times doesn't escalate.
- `_user_set_fields` excludes defaults; bare-list children path records `children` as set.

`tests/test_container_positional.py` (extending `tests/test_container_bare_list.py`):
- `Container(Text(text="a"), Text(text="b"))` has 2 children.
- `Figure(Text(text="x"), theme=None)` works.
- `Container(Text(text="a"), children=[Text(text="b")])` raises `TypeError(... mutually exclusive ...)`.
- `SmoothPanel2D(SliceOverlay(...), plot_data=…)` puts the overlay in children.
- `_user_set_fields` records `children` from the positional path.

`tests/test_layout_dsl.py`:
- `LayoutConstraints` typed instance still works.
- Mapping input still works.
- String `"row gap=1.0"` parses; `"col align=stretch"` parses (alias).
- Invalid first token raises a clear error.
- YAML integration: `Container(layout="row gap=1")` round-trips through dracon.

Existing `tests/test_panel_defaults_from_theme.py` updated to assert the fill semantic.

Render `paper-jobs/plot/fig1_matrix_gradient.yaml` and verify the output stays visually correct.

## Risks

- The `_user_set_fields` snapshot in `model_post_init` must not capture *defaults*. Pydantic v2's `model_fields_set` already excludes defaults — verify with a quick assertion.
- Validator-mutated fields (e.g., the bare-list `children` validator) — `model_fields_set` after the `before` validator should include `children`. Confirm with a test using `Container.model_validate([...])` and `Container(text_a, text_b)`.
- Re-applying jstyle on the same tree (e.g., after a panel is reparented) should still cascade. The "user-set" snapshot is from construction, not from a previous jstyle pass — themes applied later by `setattr` don't enter `_user_set_fields`.
- Audit `default.yaml` / `plots.yaml` rules briefly for any that *intentionally* clobber. None found in a quick scan, but the migration step is the right time to confirm.
- Pydantic v2 allows `__init__` override but may warn in some configs. Test under the project's actual Pydantic version.
- dracon's tag-construction path resolves `!Container` tags via Pydantic's validation, not direct `__init__` — so the YAML side keeps working through `model_validator(mode="before")`. Confirm with existing YAML tests.
- The `Annotated[…, BeforeValidator]` pattern doesn't always survive Pydantic subclassing cleanly. Test with at least one `Container` subclass that re-declares `layout`.

## Acceptance

- All existing tests pass plus the new cascade-fill / positional-children / layout-DSL tests.
- `paper-jobs/plot/fig1_matrix_gradient.yaml` still renders correctly. The `rescaler: ${_rescaler}` lines on each panel are now redundant (theme carries it), but **don't delete them in this step** — that cleanup belongs to step 04.
- `!cascade:jstyle` still works for callers who explicitly opt into clobber.
- `Figure(p1, p2)` and `Container(layout="row gap=1")` work in Python REPL.
- Net diff: ~50 lines added across `component.py` / `container.py` / `style_engine.py` / `models.py`; ~10 lines deleted across themes; ~0 lines deleted from `fig1_matrix_gradient.yaml` (that comes in step 04). The architectural enabler — this step's value is in what it unblocks, not in what it deletes today.
