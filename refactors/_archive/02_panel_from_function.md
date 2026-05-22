# 02 — Panel = function (+ content-aware sizing)

## What

A decorator `@panel_from` that takes a drawing function and produces a Pydantic Panel class whose fields mirror the function's signature and whose `draw(self, ax)` calls the function with `{field: getattr(self, field) for field in fields}`. Plus a content-aware sizing mechanism on `PlotPanel`: `axes_size`, `label_pad`, `colorbar_pad`, `legend_pad`, and a computed `min_dimensions` derived from those. Both fields and computed sizing are cascade-fillable.

```python
# jeanplot/plots/smooth_2d.py — drawing function, unchanged
def smooth_2d(X, Y, ax, rescaler, *, xlims=(0,1), vlims=(None,None), …):
    "2D KNN-smoothed heatmap."
    ...

# jeanplot/panels/smooth_2d.py — replaces the hand-written class
SmoothPanel2D = panel_from(
    smooth_2d,
    txt_fn=smooth_2d_txt,            # optional, drives render_txt
    field_overrides={"colorbar_pad": 0.6},   # type-specific sizing
)
```

## Why

Every drawing function in `jeanplot/plots/*.py` has a hand-written companion class in `jeanplot/panels/*.py`. Each class declares 10–25 fields that mirror the function signature, plus a `draw()` that calls the function with each field. ~1200 LOC of panels, of which ~80% is pure boilerplate. Signatures drift between function and panel; the `arrow_scale` bug already once propagated only halfway.

Once `panel_from` exists, every panel becomes a one-liner. New plot authors write the function, the Panel class falls out. biocomp-plot's "function = plot" ergonomics, with the panel-system structure and type safety.

The content-aware sizing fold-in is mandatory because of step 01. With cascade-fill, the theme can carry sizing rules like `SmoothPanel2D: { colorbar_pad: 0.6 }` — but the *fields* have to exist on the class. We add them once on `PlotPanel`; subclasses inherit. The figure auto-sizes from its children's computed `min_dimensions`. No more per-file inch arithmetic in `min_dimensions: [3, 2.5]` and `style.padding: [0.1, 1.5, 0.6, 0.7]`.

## Why this order

Depends on Step 01 — uses `_user_set_fields` to gate computed `min_dimensions` against explicit user override.

Unblocks Step 04 — once `axes_size` / `colorbar_pad` are cascade-fillable, the paper theme owns them, and per-panel YAML stops declaring sizing.

## Current state

### Panel side
- Drawing functions live in `jeanplot/plots/{smooth_1d,smooth_2d,smooth_3d,density,mvp,particle,scatter,stacked_poly,violin,ascii_heatmap}.py`.
- Their hand-written Panel companions live in `jeanplot/panels/*.py` and inherit `PlotPanel` (`jeanplot/panels/base.py:9`).
- `PlotPanel` provides `plot_data`, `rescaler`, `title`, `xtitle`, `ytitle`, `vtitle`, `is_drawable`.
- dracon's `register_template` (`dracon/symbols.py:649`) already does the signature introspection we need for the YAML side, but it produces a `CallableSymbol`, not a Pydantic class. We reuse the introspection idea, not the output.

### Sizing side
- Default `min_dimensions = Size(0.0, 0.0)` from `Component`. Means panels with no explicit `min_dimensions` collapse — `autofig_pred_combined.yaml` works only because `gap=4` forces non-zero spacing.
- `BoxStyle.padding` exists but is per-component and only affects the *inside* of containers, not bbox for label space outside the axes.
- `_figure_render.py` uses `panel._dimensions` (set by layout pass) to compute axes bbox via `_panel_bbox`. Labels/titles/legends drawn *inside* axes coordinates that may extend past the bbox.
- No mechanism today to say "this panel needs N inches of extra space for its colorbar."
- `fig1_matrix_gradient.yaml:174-188` shows the pattern: `min_dimensions: ${_panel_size}` on every panel, hand-tuned figure padding `[0.1, 1.5, 0.6, 0.7]` to leave room for labels.

## Design

### `panel_from` factory

```python
# jeanplot/panels/from_function.py
import inspect
from typing import Any, get_type_hints
import pydantic
from jeanplot.panels.base import PlotPanel
from jeanplot.data import PlotFunctionResult, IdentityRescaler

_SENTINEL_PARAMS = {"self", "ax"}  # not panel fields

def panel_from(
    fn,
    *,
    name: str | None = None,
    base: type = PlotPanel,
    field_overrides: dict[str, Any] | None = None,
    txt_fn=None,
) -> type:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, tuple[type, Any]] = {}
    for pname, param in sig.parameters.items():
        if pname in _SENTINEL_PARAMS:
            continue
        if pname in base.model_fields:
            # inherited from PlotPanel (plot_data, rescaler, title, ...)
            continue
        annot = hints.get(pname, Any)
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[pname] = (annot, default)
    if field_overrides:
        fields.update(field_overrides)

    cls = pydantic.create_model(
        name or _derive_name(fn.__name__),
        __base__=base,
        **fields,
    )

    own_fields = set(fields)
    base_fields = set(base.model_fields)

    def draw(self, ax) -> PlotFunctionResult | None:
        kwargs = {f: getattr(self, f) for f in own_fields | base_fields if f in fn_param_names}
        if "rescaler" in fn_param_names and kwargs.get("rescaler") is None:
            kwargs["rescaler"] = IdentityRescaler()
        result = fn(ax=ax, **kwargs)
        if result is not None and getattr(result, "mappable", None) is not None:
            self._mappable = result.mappable
        return result

    fn_param_names = set(sig.parameters) - _SENTINEL_PARAMS
    cls.draw = draw

    if txt_fn is not None:
        txt_sig = inspect.signature(txt_fn)
        txt_params = set(txt_sig.parameters) - _SENTINEL_PARAMS

        def render_txt(self):
            kwargs = {f: getattr(self, f) for f in txt_params if hasattr(self, f)}
            return str(txt_fn(**kwargs))

        cls.render_txt = render_txt

    cls.model_rebuild(force=True)
    return cls
```

Decorator form (for define-function-and-panel-together):

```python
@panel_from
def smooth_2d(X, Y, ax, rescaler, *, xlims=(0,1), …):
    ...
# `smooth_2d` is now the Panel class; the original callable is `smooth_2d.__panel_fn__`.
```

Or call form:
```python
SmoothPanel2D = panel_from(smooth_2d)
```

Both work. Call form is preferred when the function lives in one module and the panel in another (current jeanplot layout).

The **rescaler-fallback** (`None → IdentityRescaler()`) is centralized here; today it's duplicated in every hand-written `draw()`. **One SSOT point removed.**

### Content-aware sizing on `PlotPanel`

```python
# jeanplot/panels/base.py
class PlotPanel(Container):
    plot_data: PlotData | LazyPlotData | None = None
    rescaler: Any | None = None
    title: str | None = None
    title_kwargs: dict = Field(default_factory=dict)
    xtitle: str | None = None
    ytitle: str | None = None
    vtitle: str | None = None
    is_drawable: bool = True

    # NEW: content-aware sizing — cascade-fillable on every subclass
    axes_size: Size = Field(default_factory=lambda: Size(2.5, 2.0))
    label_pad: float = 0.5         # inches reserved for x/y labels
    title_pad: float = 0.0         # set automatically when self.title
    colorbar_pad: float = 0.0      # tuned per subclass; theme can override
    legend_pad: float = 0.0        # tuned per subclass; theme can override

    @model_validator(mode="after")
    def _compute_min_dimensions(self):
        if "min_dimensions" in self._user_set_fields:
            return self  # respect explicit override (interop with step 01)
        title_room = 0.3 if self.title else 0.0
        self.min_dimensions = Size(
            width=self.axes_size.width + self.label_pad + self.colorbar_pad + self.legend_pad,
            height=self.axes_size.height + self.label_pad + title_room + self.title_pad,
        )
        return self
```

Subclass-specific defaults are not hardcoded — they live in the theme as cascade rules:

```yaml
# jeanplot/resources/themes/plots.yaml
SmoothPanel2D:
  colorbar_pad: 0.6
SmoothPanel1D:
  legend_pad: 1.2
```

With cascade-fill, these are defaults that user-set values still override.

### `Figure` auto-sizing

If `Figure.min_dimensions` is unset, the existing `_layout_children` path computes natural size from children + gap; expose this to `render_figure` so the matplotlib `figsize` follows. Most paper-jobs files lose their `min_dimensions:` and `style.padding:` blocks entirely.

## Implementation steps

1. Create `jeanplot/panels/from_function.py` with `panel_from`, `_derive_name` helper (e.g., `smooth_2d` → `SmoothPanel2D`), and the `txt_fn=` kwarg.
2. Add `axes_size`, `label_pad`, `title_pad`, `colorbar_pad`, `legend_pad` fields + `_compute_min_dimensions` model-validator to `PlotPanel`. Gate computed `min_dimensions` on `min_dimensions not in self._user_set_fields`.
3. Update `jeanplot/resources/themes/plots.yaml` with per-subclass `colorbar_pad` / `legend_pad` defaults.
4. Migrate one panel as proof-of-concept: replace `SmoothPanel2D` in `jeanplot/panels/smooth_2d.py` with `SmoothPanel2D = panel_from(smooth_2d, txt_fn=smooth_2d_txt)`. Confirm rendering.
5. Update `Figure` to derive its own `min_dimensions` from children + gap when not explicitly set (the layout pass already computes this — wire it to figsize).
6. Migrate the remaining panels — `SmoothGradMagnitudePanel2D`, `GradientFieldPanel2D`, `SmoothPanel1D`, `SmoothPanel3D`, `MVPPanel`, `DensityPanel1D`, `ViolinPanel`, `ScatterPanel3D`, `GridHistogramPanel`, `ParticlePanel`, `StackedPolyPanel`, `AsciiHeatmapPanel`. Each migration deletes ~30-80 lines.
7. After all panels migrated, **remove** the `min_dimensions: ${figure_size}` lines and `style.padding: [...]` block from `paper-jobs/plot/fig1_matrix_gradient.yaml`. Confirm fig1 still renders correctly.

## Tests

`tests/test_panel_from_function.py`:
- Generated class has fields matching the function signature (kwargs only, excluding `ax`).
- Default values propagate from function defaults.
- Drawing produces the same `PlotFunctionResult` as calling the function directly (pixel parity for one simple case).
- Inherited PlotPanel fields (`plot_data`, `rescaler`, `title`) work as expected.
- `txt_fn=` produces a `render_txt` method that takes the right subset of fields.
- Class is a valid YAML tag (loads via dracon with `make_plot_context()`).
- Functions with `*args` / `**kwargs` raise a clear error (same constraint as dracon's `register_template`).

`tests/test_panel_min_dimensions.py`:
- `SmoothPanel2D(plot_data=…)` with no explicit `min_dimensions` has computed `min_dimensions` ~= (axes + colorbar + labels).
- Explicit `min_dimensions=Size(5,5)` overrides the computation.
- Theme rule `SmoothPanel2D: { colorbar_pad: 0.3 }` is reflected in computed `min_dimensions`.
- User-set `colorbar_pad: 0.8` survives the cascade (step 01 interop).
- `Figure` containing 3 panels auto-sizes to ~3·panel_width + 2·gap.

Render `fig1_matrix_gradient.yaml` without `min_dimensions:` / `style.padding:`; check the SVG dimensions are reasonable. Existing panel tests stay green after each migration step.

## Risks

- **Type-hint introspection edge cases.** `get_type_hints` resolves `from __future__ import annotations`, but generic aliases (`Sequence[str]`, `Literal["raw","latent"]`) need to round-trip through Pydantic. Test these first.
- **Positional-only params** (`def fn(X, /, …)`). `pydantic.create_model` requires keyword fields. Add a check + clear error message.
- **`*args` / `**kwargs`** in function signatures. Reject with a clear error.
- **Hand-written panels with bespoke `draw` logic.** Any panel that does extra work in `draw` beyond forwarding to the function stays hand-written; `panel_from` is opt-in. Audit before migration.
- **Computed `min_dimensions` + writable field conflict.** Pydantic v2 won't let a `@computed_field` shadow a regular field directly. We use a `model_validator` workaround; verify the pattern survives `validate_assignment=True`.
- **Heuristic padding values.** `colorbar_pad=0.6` is a guess. Tune per panel after rendering. Worst case: matplotlib clips a label and the test catches it.
- **Existing files with explicit `min_dimensions`** — don't auto-rewrite them; they were tuned. Just stop *requiring* them.
- **`Figure` auto-size and `dpi`.** Figure has `dpi: 300`. Make sure auto-size produces same px output as today for at least one regression test.

## Acceptance

- `SmoothPanel2D`, `SmoothPanel1D`, `SmoothPanel3D`, `SmoothGradMagnitudePanel2D`, `GradientFieldPanel2D`, `MVPPanel` are one-liner factory calls. Other panels migrated opportunistically.
- All existing tests pass plus new `panel_from` and min-dimensions tests.
- `paper-jobs/plot/fig1_matrix_gradient.yaml` renders pixel-equivalently (within fp noise) to pre-refactor, even after `min_dimensions` and `style.padding` lines are removed.
- Net lines deleted: aim for ~800 across migrated panels, ~50 in `fig1_matrix_gradient.yaml` (sizing block removal).
