# Step 03 — Panel implementations (one Component per drawing function)

## Goal

Copy every general-purpose matplotlib drawing function from
`biocomp/biocomp/plotting/` into a `PlotPanel` subclass under
`jeanplot/panels/`. **Drawing code is duplicated verbatim**; biocomp's
originals stay in place and keep serving biocomp-plot and existing
scripts. Jeanplot grows an independent implementation, and the
function-with-many-kwargs becomes a Pydantic-typed Component wrapped
around the same code. Pixel parity (against the biocomp baseline) is
the acceptance criterion.

After this step:
- `jeanplot.panels` exports one Component subclass per plot kind:
  `SmoothPanel1D`, `SmoothPanel2D`, `SmoothPanel3D`, `MVPPanel`,
  `DensityPanel`, `ScatterPanel`, `ViolinPanel`, `ParticlePanel`,
  `StackedPolyPanel`, `AsciiHeatmapPanel`.
- `AutoPanel` dispatches to the right subclass based on
  `plot_data.dimensions.input` — collapses the four
  `tasks/{auto,1D,2D,3D}.yaml` templates into one Python class.
- `SmoothPanel3D` is itself a `Container` whose `children` are a
  cube-view sub-panel + a sub-Container holding the slice grid.
  This replaces `FigAx.subdivide` and the `axnum + subax_role +
  subax_spec` indirection entirely — the slice grid emerges from
  nested Containers.
- Each subclass implements ONE matplotlib method, `draw(self, ax)`,
  plus optional `render_txt(self) -> str | None` on the kinds that
  have ASCII variants.
- The shared draw modules (`jeanplot.plots.colorbar`,
  `jeanplot.plots.ticks`, `jeanplot.plots.overlays`,
  `jeanplot.plots.smooth_kernel`) carry the pure drawing functions
  that panels call.
- Tests render each panel kind against a fixture `PlotData` and
  pixel-compare against a snapshot stored in `jeanplot/tests/snapshots/`.

## Why now

Step 02 nailed the base class. Step 04 needs panels to exist before it
can write `themes/plots.yaml` rules targeting them by name. This is
the *mechanical* heart of the refactor — repetitive but
straightforward.

## Prerequisites

Steps 00, 01, and 02 complete:
- `!cascade:jstyle` dialect registered (step 00).
- `jeanplot.data.PlotData`, `Rescaler`, `IdentityRescaler` (step 01).
- `jeanplot.knn.*` (step 01).
- `jeanplot.color` palettes registered (step 01).
- `jeanplot.panels.PlotPanel` (with `is_drawable` field), `Colorbar` (step 02).
- `jeanplot.panels.figure.Figure` (with `theme` field + one-line `render()`
  forwarder; step 02).

## What changes

### 3.1 Shared draw modules first (the function bodies)

These are pure functions, no Component wrapping — they're called by
panel `draw()` methods, and many of them are also called by
non-panel code (the existing biocomp draw functions during the
back-compat phase).

| New file | Source | Notes |
|---|---|---|
| `jeanplot/plots/colorbar.py` | `biocomp/biocomp/plotting/plotting_core.py:267-435` (`colorbar`) | Verbatim; depends on `Rescaler` (replace `DataRescaler` annotation) |
| `jeanplot/plots/ticks.py` | `biocomp/biocomp/plotting/plotting_core.py:189-340` (`powers_of_ten`, `format_powers`, `PowerFormatter`, `get_transformed_ticks_and_labels`, `_install_overlap_skip`) + `biocomp/biocomp/plotutils.py:1232-1262` (`ShortScientificFormatter`) | Verbatim |
| `jeanplot/plots/smooth_kernel.py` | `biocomp/biocomp/plotting/plotting_smooth_2d.py:`_resolve_lims, _finite_xy, _resolve_vlims, _render_smooth_heatmap, plus the `knn_grid` function from `plotting_core.py:139-265` | These are the shared internals across smooth_1d/2d/3d |
| `jeanplot/plots/overlays.py` | `biocomp/biocomp/plotutils.py:952-1203` (`diagonal_*`, `plot_diagonal_paths`, `slice_panel_args`, `plot_slice_overlay`, `plot_slice_chords`, `plot_addition_vs_removal_overlay`) | Verbatim; these become callable both standalone and from overlay Components |

For each module: copy the source into the new jeanplot location,
replace `from biocomp...` imports with the new jeanplot locations
(`from jeanplot.data import Rescaler, PlotFunctionResult`,
`from jeanplot.knn import ...`), and **drop the `@configurable`
decorator** wherever it appears — it's biocomp-internal machinery
and irrelevant in jeanplot (the dialect/jstyle path replaces it).
The biocomp original keeps its `@configurable` markers; jeanplot's
copy doesn't.

### 3.2 Panel subclasses

Each is a small Pydantic class. **Pattern**: declare the kwargs as
typed fields, delegate `draw()` to a free function that takes arrays.
`PlotFunctionResult` gained a `mappable` field in step 02 — panels
that produce a colorbar-eligible mappable return it there.

`jeanplot/panels/smooth_2d.py`:

```python
from typing import Any
from pydantic import Field
from jeanplot.panels.base import PlotPanel
from jeanplot.data import PlotData, PlotFunctionResult
from jeanplot.plots.smooth_kernel import _resolve_lims, _finite_xy, _render_smooth_heatmap
from jeanplot.knn import knn_grid


class SmoothPanel2D(PlotPanel):
    plot_data: PlotData
    zslice: Any | None = None
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    ylims: tuple[float | None, float | None] = (None, None)
    vlims: tuple[float | None, float | None] = (None, None)
    vlim_quantiles: tuple[float | None, float | None] | None = (0.01, 0.99)
    vlim_min_floor: float | None = None
    vlim_min_range: float | None = None
    draw_xlabel: bool = True
    draw_ylabel: bool = True
    xaxis_labelpad: float | None = None
    yaxis_labelpad: float | None = None
    cmap: str = "bc_blues"
    bad_color: str = "#ffffff"
    contours: int = 0
    image_interpolation: str | None = None
    knn_grid_params: dict = Field(default_factory=dict)
    # Colorbar (if present) is a child overlay; this panel doesn't
    # need to know whether one was declared — it always sets
    # `_mappable` and the overlay reads it.

    def draw(self, ax) -> PlotFunctionResult:
        X, Y = _finite_xy(self.plot_data.x, self.plot_data.y)
        xlims, ylims = _resolve_lims(X, self.xlims, self.ylims)
        resolution = self.knn_grid_params.get("grid_resolution", 200)
        input_coords, output_values = knn_grid(
            X, Y, xlims, ylims, **{**self.knn_grid_params, "zslice": self.zslice},
        )
        return _render_smooth_heatmap(
            ax, input_coords, output_values,
            self.plot_data.input_names, self.plot_data.output_name,
            self.rescaler, self.rescaler,
            xlims, ylims, resolution,
            title=self.title, title_kwargs=self.title_kwargs,
            xtitle=self.xtitle, ytitle=self.ytitle, vtitle=self.vtitle,
            vlims=self.vlims, vlim_quantiles=self.vlim_quantiles,
            vlim_min_floor=self.vlim_min_floor, vlim_min_range=self.vlim_min_range,
            draw_xlabel=self.draw_xlabel, draw_ylabel=self.draw_ylabel,
            xaxis_labelpad=self.xaxis_labelpad, yaxis_labelpad=self.yaxis_labelpad,
            cmap=self.cmap, bad_color=self.bad_color, contours=self.contours,
            image_interpolation=self.image_interpolation,
        )
        # _render_smooth_heatmap returns PlotFunctionResult(rendering=..., metadata=..., mappable=quadmesh).
        # The renderer (step 02 §2.5) picks up the mappable and stores it on self._mappable
        # so any Colorbar overlay child can read it.

    def render_txt(self) -> str:
        from jeanplot.plots.txt import smooth_2d_txt
        return smooth_2d_txt(self.plot_data.x, self.plot_data.y, ...)
```

Same pattern for every other panel kind. Each panel file is ~80-150
lines of declarations + delegation. The matplotlib *logic* stays in
`jeanplot/plots/<kernel>.py` modules unchanged.

**Why `draw(ax)` and not `render_mpl(ax)`.** Per step 02 §2.2: panels
have one matplotlib method, not three. SVG output comes from the
existing svg-export pipeline that already turns matplotlib figures into
SVG; panels do not need a `render_svg`. ASCII is the one exception —
it produces bytes, not an axes — so it gets its own optional
`render_txt() -> str | None`.

### 3.3 Panel list (one file each under `jeanplot/panels/`)

| File | Class | Wraps | Source |
|---|---|---|---|
| `smooth_1d.py` | `SmoothPanel1D` | `smooth_1d` | `biocomp/biocomp/plotting/plotting_smooth_1d.py` |
| `smooth_2d.py` | `SmoothPanel2D` + `SmoothGradMagnitudePanel2D` + `GradientFieldPanel2D` | `smooth_2d`, `smooth_grad_magnitude_2d`, `gradient_field_2d` | `plotting_smooth_2d.py` |
| `smooth_3d.py` | `SmoothPanel3D` | `smooth_3d`, `plot_3d_stack` | `plotting_3d.py` |
| `mvp.py` | `MVPPanel` | `measured_vs_predicted`, `noise_floor_panel` | `plotting_mvp.py` |
| `density.py` | `DensityPanel1D` | `density_plot_1d` | `plotting_density.py` |
| `scatter.py` | `ScatterPanel3D` + `GridHistogramPanel` | `scatter_3d`, `grid_histogram` | `plotting_scatter.py` |
| `violin.py` | `ViolinPanel` | `smooth_voxel_conditioned_violin` | `plotting_violin.py` |
| `particle.py` | `ParticlePanel` | `particle_plot` | `plotting_particle.py` |
| `stacked_poly.py` | `StackedPolyPanel` | (the stacked-poly drawing) | `plotting_mvp.py:166-...` + `stacked_poly.py` |
| `ascii_heatmap.py` | `AsciiHeatmapPanel` | the `░▒▓█` heatmap | `ascii_heatmap.py` |

Plus `jeanplot/panels/auto.py` — the `AutoPanel` dim-dispatcher (§3.3a).

### 3.3a `AutoPanel` — a `!fn` template + Python helper

The existing `tasks/{auto,1D,2D,3D}.yaml` templates exist for one
purpose: pick a panel kind from `plot_data.dimensions.input`. That's
the **dracon "Constructor Slots" pattern** — a single tag chosen by a
table lookup. Express it the dracon-native way (a `!fn` template
using `!$(...)` for dynamic tag) and add a tiny Python helper for
code-side use. Both are SSOT-equivalent — the template IS the
dispatcher; the Python helper just wraps the same dict.

**The YAML form** — lives in `jeanplot/resources/templates/auto_panel.yaml`,
auto-included by `themes/plots.yaml`:

```yaml
# jeanplot/resources/templates/auto_panel.yaml
!define _PanelByDim:
  1: SmoothPanel1D
  2: SmoothPanel2D
  3: SmoothPanel3D

!define AutoPanel: !fn
  !require plot_data: "PlotData (or LazyPlotData)"
  !set_default force_dim: null
  !set_default title: null
  !set_default rescaler: null
  !fn : !$(_PanelByDim[force_dim or plot_data.dimensions.input])
    plot_data: ${plot_data}
    title: ${title}
    rescaler: ${rescaler}
```

`!$(expr)` evaluates the expression and uses the result as a tag name.
`_PanelByDim[1]` → `"SmoothPanel1D"` → `!SmoothPanel1D`, which constructs
the right Pydantic model. No `model_post_init`, no `model_dump`
roundtrip, no delegate-attribute-copy fragility — pure dracon
composition.

**The Python helper** — `jeanplot/panels/auto.py`:

```python
# jeanplot/panels/auto.py
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.smooth_1d import SmoothPanel1D
from jeanplot.panels.smooth_2d import SmoothPanel2D
from jeanplot.panels.smooth_3d import SmoothPanel3D

_PANEL_BY_DIM: dict[int, type[PlotPanel]] = {
    1: SmoothPanel1D, 2: SmoothPanel2D, 3: SmoothPanel3D,
}


def auto_panel(plot_data, *, force_dim: int | None = None, **kwargs) -> PlotPanel:
    """Pick the right SmoothPanel{1,2,3}D by data dim. The Python twin of
    the `!AutoPanel` !fn template."""
    dim = force_dim or plot_data.dimensions.input
    if dim not in _PANEL_BY_DIM:
        raise ValueError(f"auto_panel: unsupported input dim={dim}")
    return _PANEL_BY_DIM[dim](plot_data=plot_data, **kwargs)
```

Ten lines. Same dispatch as the YAML template. SSOT lives in
`_PANEL_BY_DIM` / `_PanelByDim` — keep both in sync (a single short
test asserts the dicts match key-for-key).

**YAML usage** — one line at the call site:

```yaml
- !AutoPanel { plot_data: ${D} }
```

**Python usage**:

```python
from jeanplot.panels.auto import auto_panel
panel = auto_panel(plot_data=d)
```

The four `tasks/{auto,1D,2D,3D}.yaml` templates are **deleted** in
step 04. They have no jeanplot replacement.

### 3.3b `SmoothPanel3D` is a Container with cube + slice grid children

The biocomp `smooth_3d` plot draws a cube view + an R×C grid of 2D
slice heatmaps. Today that requires `FigAx.subdivide(axnum, spec)` to
carve sub-axes inside one Container cell. In the new world,
`SmoothPanel3D` IS a `Container` whose children are exactly those
sub-panels:

```python
# jeanplot/panels/smooth_3d.py (sketch)
from jeanplot.core.container import Container
from jeanplot.core.models import LayoutConstraints
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.smooth_2d import SmoothPanel2D


class CubeView(PlotPanel):
    plot_data: PlotData
    projection_angle: float = 45.0
    projection_diag_coef: float = 0.5
    # ... cube_edge_props from the theme cascade ...

    def draw(self, ax):
        from jeanplot.plots.cube import draw_cube_view
        return draw_cube_view(ax, self.plot_data, ...)


class SmoothPanel3D(PlotPanel):
    plot_data: PlotData
    zslices: list[float] = [0.05, 0.25, 0.4, 0.55]
    slice_grid: tuple[int, int] = (3, 3)
    cube_frac_w: float = 0.57
    # ... shared 2D/3D knobs ...

    def model_post_init(self, __context):
        # Compose the inner tree once; the renderer's measure/layout pass
        # handles everything else.
        import numpy as np
        R, C = self.slice_grid
        zs = np.linspace(self.zslices[0], self.zslices[-1], R*C)
        cube = CubeView(plot_data=self.plot_data, ...)
        slice_panels = [
            SmoothPanel2D(plot_data=self.plot_data, zslice=float(z), ...)
            for z in zs
        ]
        slice_rows = [
            Container(
                layout=LayoutConstraints(direction="row", gap=4),
                children=slice_panels[r*C:(r+1)*C],
            ) for r in range(R)
        ]
        slice_grid = Container(
            layout=LayoutConstraints(direction="column", gap=4),
            children=slice_rows,
        )
        # this panel becomes a row-Container of [cube, slice_grid] with weighted widths
        self.layout = LayoutConstraints(
            direction="row", gap=8,
            main_axis_weights=[self.cube_frac_w, 1.0 - self.cube_frac_w],
        )
        self.add_children([cube, slice_grid])

    is_drawable: bool = False     # outer shell is layout-only; children draw
    # no draw() override needed — base class returns None when is_drawable=False
```

The `is_drawable` field lives on `PlotPanel` base (step 02 §2.2);
the renderer (step 02 §2.5) skips axes allocation and draw-dispatch
for any panel with `is_drawable=False`. So `CubeView` gets its own
axes, each slice `SmoothPanel2D` gets its own, and the outer
`SmoothPanel3D` shell is just layout. No wasted axes, no special
casing in the renderer beyond the one-line guard already in §2.5.

**What this collapses.** The old machinery had:
- `expand_panel_atomics` (3D special case yielding `cube` + R*C `slice` atomics)
- `FigAx.subdivide(axnum, spec)` (carve a parent axes into sub-axes)
- `axnum + subax_role + subax_spec` indirection in every plot task dict
- a `cube_frac_w` / `cube_slice_gap_frac` / `slice_vgap_frac` / `slice_hgap_frac` knob soup threaded through 4 layers

All of it is gone. `SmoothPanel3D(plot_data=d)` lays itself out via
nested `Container`s, the renderer hands each child panel its own
matplotlib Axes from its laid-out bbox, and the existing `LayoutConstraints`
gap + flex-weight machinery does the spacing.

### 3.4 Overlay panels

Overlays are also Components (they're just panels with `is_overlay=True`):

| File | Class | Wraps |
|---|---|---|
| `jeanplot/panels/overlays.py` | `DiagonalPathOverlay`, `SliceOverlay`, `SliceChordOverlay`, `AdditionVsRemovalOverlay`, `IdentityLineOverlay`, `DensityContourOverlay` | the existing overlay primitives from `plotutils.py:952-1203` |

The renderer calls each overlay's `draw(parent_ax)` after the parent
panel has rendered (step 02 §2.5). Overlays read `self.parent._axes`
and `self.parent._mappable` to draw against the parent's chart. Same
uniform mechanism `Colorbar` uses — overlays are not a separate
protocol, just `PlotPanel` subclasses with `is_overlay=True`.

### 3.5 The `txt` rendering polymorphism

Subclasses with ASCII variants override `render_txt(self) -> str | None`.
The renderer for `.txt` output (a small `txt_renderer` module, ~30 LOC)
walks the tree, calls `render_txt()` on every `PlotPanel`, joins with
`\n\n`. Panels that don't override return `None` and are skipped.



Source: `biocomp/biocomp/plotting/plotting_txt.py` and
`biocomp-tools/biocomptools/toollib/plot.py:224-228` (the
`TXT_PLOT_FUNC_MAP` dict that maps function names to txt variants).

Replace the dict with `PlotPanel.render_txt()` overrides on the
relevant subclasses (`SmoothPanel1D`, `SmoothPanel2D`, `SmoothPanel3D`,
plus `AsciiHeatmapPanel` which is *only* a txt renderer). The
text-mode path in `jeanplot.render()` (step 02 §2.6 — extend the
dispatch to route to `render_txt` when output extension is `.txt`)
walks panels and joins their text outputs.

The `TXT_PLOT_FUNC_MAP` dict and its dispatch logic in
`toollib/plot.py:_run_txt` are biocomp-internal; jeanplot's panel-side
`render_txt()` polymorphism is the equivalent for jeanplot code. The
biocomp dispatch continues to serve biocomp-plot — biocomp isn't
touched.

### 3.6 Tests — pixel parity

For each panel kind, build a fixture under `jeanplot/tests/fixtures/`:
a small `PlotData` (deterministic random) appropriate to the kind
(1D / 2D / 3D / mvp pairs / scatter / violin).

For each kind:

```python
# tests/test_panel_pixel_parity.py
def test_smooth_2d_pixel_parity(tmp_path):
    from jeanplot import Figure, PlotPanel
    from jeanplot.panels import SmoothPanel2D

    data = load_fixture("plot_data_2d.json")
    fig = Figure(
        output_dir=str(tmp_path),
        output_file="smooth_2d.png",
        children=[SmoothPanel2D(plot_data=data)],
    )
    fig.render()       # one-line forwarder; equivalent to jeanplot.render(fig)

    expected = load_snapshot("smooth_2d.png")
    actual = Image.open(tmp_path / "smooth_2d.png")
    assert pixel_diff(actual, expected) < TOL_PER_PIXEL
```

For ASCII outputs: byte-for-byte equal (`render_txt()` deterministic).

A bootstrapping run on a clean checkout generates `snapshots/*.png`
from the *current biocomp* outputs (run a script that uses
`biocomp.plotutils.smooth` to produce the snapshot). Once the snapshot
exists, the new code must match.

### 3.7 No biocomp imports inside `jeanplot/panels/` or `jeanplot/plots/`

Single rule, easy to grep for. CI check:

```bash
! grep -rn "from biocomp\|import biocomp" jeanplot/jeanplot/panels jeanplot/jeanplot/plots
```

Any hit fails the build.

## Code sketches

The MVP panel (most independent — doesn't take a `PlotData` directly):

```python
# jeanplot/panels/mvp.py
class MVPPanel(PlotPanel):
    measured: Any        # NDArray
    predicted: Any
    model_samples: Any | None = None
    plot_data: None = None    # MVP doesn't use the inherited slot

    show_density: bool = True
    show_identity: bool = True
    show_stats: bool = True
    show_trendline: bool = False
    show_bias: bool = False
    show_calibration_rms: bool = False
    show_spread: bool = False
    show_noise_floor: bool = False
    noise_local: bool = False
    vlims: tuple = (None, None)
    margins: float = 0.05
    # ... about 20 more knobs that mvp.measured_vs_predicted takes

    def draw(self, ax) -> PlotFunctionResult:
        from jeanplot.plots.mvp import measured_vs_predicted
        return measured_vs_predicted(
            measured=self.measured, predicted=self.predicted,
            ax=ax,
            model_samples=self.model_samples,
            show_density=self.show_density,
            # ... pass through every field
            rescaler=self.rescaler,
        )
```

The drawing module:

```python
# jeanplot/plots/mvp.py
# verbatim copy of biocomp/biocomp/plotting/plotting_mvp.py with:
#   from biocomp.metric_utils import rmse, r_squared
#     -> from jeanplot.stats import rmse, r_squared   (NEW small module
#        or inline these two functions; they're 6 lines each)
#   from biocomp.datautils import DataRescaler
#     -> from jeanplot.data import Rescaler
#   from biocomp.plotutils import PlotFunctionResult
#     -> from jeanplot.data import PlotFunctionResult
#   from biocomp.plotting.stacked_poly import (...)
#     -> from jeanplot.plots.stacked_poly import (...)
```

(`jeanplot/stats.py` — a 30-line module with the small statistical
helpers used across plot kinds: `rmse`, `r_squared`, `pearson_r`,
`mae`. Keep biocomp's `metric_utils.py` as canonical for biocomp
purposes; jeanplot's `stats.py` carries only the bits drawing needs.
Both implement the same formulas; duplication is acceptable because
the formulas are tiny and SSOT across repos is impractical without a
shared package. If jeanplot grows further, consider extracting them
later.)

## Verification

```bash
pytest jeanplot/tests/test_panel_pixel_parity.py -v
```

Every panel kind must match its snapshot within tolerance.

```bash
pytest jeanplot/tests/ -v
```

Existing tests still green.

CI grep check:

```bash
! grep -rn "from biocomp\|import biocomp" jeanplot/jeanplot/panels jeanplot/jeanplot/plots
```

## Out of scope

- Themes (step 04).
- Compose engine / dim-dispatch (step 04).
- Biocomp-specific panels — `NetworkDiagramPanel`, `CircuitPanel`,
  `MVPNetworkPanel`. Those stay in biocomp-tools and continue to be
  used from there (whether by biocomp-plot or by a migrated jeanplot
  script that mixes the two). Strictly general-purpose panels here.
- Any change to biocomp / biocomp-tools. They remain untouched in
  this step (and in every step).

## Estimate

~3000 LOC copied into `jeanplot/panels/` and `jeanplot/plots/`. Net
new file count: ~10 panel modules + ~4 shared draw modules + 1 stats
module. Zero LOC removed from biocomp.

Net diff across steps 01-03: ~+5000 LOC into jeanplot. The whole
refactor is net-positive on disk — biocomp keeps its plotting code
forever (or until the user opportunistically cleans up after the
last script migrates). The win is structural (clean SSOT inside
jeanplot, no biocomp coupling), not lines-of-code.
