# Step 02 — `PlotPanel` base + `Figure(Container)` + renderer extension

## Goal

Establish the two Component classes every subsequent panel and every
figure depends on, plus the renderer extension that lets the existing
scene-graph render pipeline allocate one matplotlib `Axes` per leaf
`PlotPanel` from its laid-out bbox.

The architectural commitment in this step is **no new orchestrator**.
`MatplotlibRenderer.render_component` already does
`measure_and_layout` → `compute_world_matrix` → tree-walk `render()`.
We extend it with one more thing: "when you encounter a PlotPanel
during the tree walk, allocate a sub-axes for it and call
`panel.draw(ax)`". Nothing else needs orchestration. `Figure` is
literally a `Container` with a handful of file-output attributes.

After this step:
- `PlotPanel(Container)` is the base class for every plot kind.
  Subclasses implement `draw(self, ax) -> PlotFunctionResult | None`.
  Optional `render_txt(self) -> str | None` for ASCII output.
- `Figure(Container)` is a thin Container that owns
  `output_path`, `dpi`, `metadata`, `rc_context`, `extra_output_paths`,
  and `theme` (the `!cascade:jstyle` CallableSymbol that styles the
  subtree). The only method is a one-line `render(**kw)` forwarder to
  `jeanplot.render()` — no parallel lifecycle.
- `MatplotlibRenderer` knows how to:
  1. detect when its target tree contains plot panels;
  2. create a real `plt.Figure` sized to the laid-out root;
  3. for each leaf `PlotPanel` allocate an `Axes` from its bbox
     (figure-relative coordinates);
  4. call `panel.draw(ax)` inside `mpl.rc_context(rc=fig.rc_context)`;
  5. let overlays render via the existing tree-walk after the parent
     panel sets `panel._mappable`.
- `Colorbar` is a normal overlay Component (`is_overlay=True`) that
  reads `parent._mappable` during its `draw()` — the same pattern as
  every other overlay.
- `jeanplot.render(component, output=...)` saves to disk when
  `component` is a `Figure` with an `output_file`.

## Why now

Step 01 gave us the data and KNN substrate. Step 03 will create one
`PlotPanel` subclass per drawing function — but those classes need a
base class to subclass. Step 02 is the *architecturally* hardest piece
because every downstream Component inherits from what we settle here.
The goal is to settle as little as possible, then let everything else
emerge.

## Prerequisites

Steps 00-01 complete:
- jstyle engine backed by `!cascade:jstyle`.
- `jeanplot.data.PlotData` available.
- `jeanplot.data.Rescaler` protocol available.
- `jeanplot.color` palettes registered.

## What changes

### 2.1 `jeanplot/panels/__init__.py` — new package

Empty for now (concrete panel subclasses arrive in step 03). Re-export
the base class:

```python
from jeanplot.panels.base import PlotPanel, Colorbar
__all__ = ["PlotPanel", "Colorbar"]
```

### 2.2 `jeanplot/panels/base.py` — `PlotPanel`

The base class. Inherits from `Container`, so children / layout /
jstyle all work for free. Adds:

```python
from typing import Any
from pydantic import Field, PrivateAttr
import matplotlib.axes

from jeanplot.core.container import Container
from jeanplot.data import PlotData, LazyPlotData, PlotFunctionResult


class PlotPanel(Container):
    plot_data: PlotData | LazyPlotData | None = None
    rescaler: Any | None = None            # runtime-checked via isinstance(x, Rescaler)
    title: str | None = None
    title_kwargs: dict = Field(default_factory=dict)
    xtitle: str | None = None
    ytitle: str | None = None
    vtitle: str | None = None
    is_drawable: bool = True               # False for layout-only PlotPanels (e.g. the SmoothPanel3D outer shell that just holds cube + slice-grid children)

    _axes: matplotlib.axes.Axes | None = PrivateAttr(default=None)
    _mappable: Any | None = PrivateAttr(default=None)      # for Colorbar / overlays
    _last_metadata: dict = PrivateAttr(default_factory=dict)

    def draw(self, ax) -> PlotFunctionResult | None:
        """Render into a matplotlib Axes. Subclasses MUST override unless `is_drawable=False`."""
        if not self.is_drawable:
            return None
        raise NotImplementedError(f"{type(self).__name__} must implement draw()")

    def render_txt(self) -> str | None:
        """ASCII / kitty-graphics representation. Subclasses MAY override."""
        return None
```

**Why one `draw()` method, not three.** The matplotlib drawing operation
is the load-bearing thing. SVG output goes through the existing scene
graph's matplotlib-to-SVG export path (a Figure rendered to mpl is
already savable as SVG via `fig.savefig(..., format='svg')`); panels
do not need a separate `render_svg()`. ASCII output is fundamentally
different — bytes, no axes — and gets its own optional method.

**Why expose `_mappable` on the panel, not a method.** Overlays
(notably `Colorbar`) need to read it; they're separate Components.
A `_mappable` private attr lets the overlay grab `parent._mappable`
during its own `draw()` without coupling to a method name.

### 2.3 `Colorbar` and the uniform overlay protocol

`Colorbar` is **one overlay among many** — same mechanism as
`IdentityLineOverlay`, `SliceChordOverlay`, `DiagonalPathOverlay`,
`DensityContourOverlay` (all introduced in step 03).

```python
class Colorbar(PlotPanel):              # PlotPanel, not bare Component
    is_overlay: bool = True             # render after parent panel
    plot_data: None = None              # overlay has no own data
    size: tuple[float, float] = (0.06, 0.85)
    position: tuple[float, float] = (1.05, 0.075)
    tick_props: dict = Field(default_factory=lambda: {"labelsize": 9, "pad": 3, "length": 4})
    label: str | None = None

    def draw(self, ax) -> None:
        from jeanplot.plots.colorbar import draw_colorbar
        mappable = getattr(self.parent, "_mappable", None)
        if mappable is None:
            return
        draw_colorbar(ax, mappable, size=self.size, position=self.position,
                      tick_props=self.tick_props, label=self.label)
```

The uniform rule: **every overlay is a `PlotPanel` (or subclass) with
`is_overlay=True`. Its `draw(ax)` receives the *parent's* axes** (set
during the renderer's tree walk — see §2.5). It MAY read `parent._mappable`
or `parent._axes` or any other parent state. No special "overlay
protocol" — just Component mechanics.

### 2.4 `jeanplot/panels/figure.py` — `Figure(Container)`

```python
from pathlib import Path
from typing import Any
from pydantic import Field, PrivateAttr

from jeanplot.core.container import Container


class Figure(Container):
    output_dir: str = "./"
    output_file: str | None = "unnamed.png"
    extra_output_paths: list[str] = Field(default_factory=list)
    dpi: int = 300
    rc_context: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    svg_id_prefix: str = "jp_"
    subtitle: str | None = None
    subtitle_kwargs: dict = Field(default_factory=dict)

    # The cascade that styles this figure (and its subtree). Typically a
    # CallableSymbol from a `!cascade:jstyle` document. When set, the
    # renderer calls `jstyle.update(self.theme)` before applying styles
    # to the tree. Making this a *field on the Figure* removes the need
    # for a hidden `load_plot_theme()` Python prelude: a Figure YAML is
    # self-contained ("I declare my theme, my layout, my children").
    theme: Any | None = None

    _mpl_figure: Any = PrivateAttr(default=None)

    @property
    def output_path(self) -> Path | None:
        if not self.output_file:
            return None
        return Path(self.output_dir) / self.output_file

    def render(self, **kwargs):
        """Convenience: equivalent to `jeanplot.render(self, **kwargs)`."""
        from jeanplot.render import render
        return render(self, **kwargs)
```

That's the entire class — `render()` is a one-line forwarder so
`fig.render()` and `jeanplot.render(fig)` are interchangeable. There
is no parallel lifecycle, no `_apply_styles()` /
`_make_mpl_figure_and_axes()` / `_render_tree()` / `_save()` orchestrator.
The behaviour lives in the renderer (§2.5) and in
`jeanplot.render(component)` (§2.6).

### 2.5 `MatplotlibRenderer` extension

Today `MatplotlibRenderer.render_component(context, component)` runs
`measure_and_layout` then walks the tree, dispatching each Component's
`render(self, context, matrix)`. Extend `render_component` (and
`Component.render` dispatch) so that **a `PlotPanel` claims its own
axes during the tree walk** instead of drawing into the parent
context.

The minimal additions to `MatplotlibRenderer`:

```python
def render_component(self, context, component, adjust_lims=True):
    # Existing entry path unchanged when `context` is a bare Axes and
    # `component` has no PlotPanels. The only new branch:
    if isinstance(component, Figure) or _tree_contains_plot_panel(component):
        return self._render_figure(component)
    return self._render_into_axes(context, component, adjust_lims)

def _render_figure(self, fig: Figure):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    with mpl.rc_context(rc=fig.rc_context):
        # 1. self-contained theme: if the figure carries a cascade, swap
        #    jstyle to it before walking. No hidden Python prelude.
        if fig.theme is not None:
            jstyle.update(fig.theme)
        jstyle.apply(fig)
        fig.measure_and_layout(self)

        # 2. create the mpl figure sized from laid-out root
        figsize = (fig._dimensions.width, fig._dimensions.height)  # inches
        mfig = plt.figure(figsize=figsize, dpi=fig.dpi)
        fig._mpl_figure = mfig

        # 3. allocate one Axes per *drawable* leaf PlotPanel from its bbox.
        #    Non-drawable panels (e.g. the SmoothPanel3D outer shell that
        #    only carries layout) are skipped here.
        for panel, bbox in _iter_panel_bboxes(fig):
            if panel.is_drawable:
                panel._axes = mfig.add_axes(bbox)

        # 4. tree walk: panels draw, then overlays draw against parent axes
        for panel in _iter_panels(fig):
            if not panel.is_drawable:
                continue
            res = panel.draw(panel._axes)
            if res is not None:
                if getattr(res, "mappable", None) is not None:
                    panel._mappable = res.mappable
                if getattr(res, "metadata", None):
                    fig.metadata.update(res.metadata)
        for overlay in _iter_overlays(fig):
            overlay.draw(overlay.parent._axes)

    return mfig
```

The bbox helper computes figure-relative `(x, y, w, h)` from each
panel's laid-out world position. ~30 LOC, pure:

```python
def _iter_panel_bboxes(root: Container):
    rw = root._dimensions.width
    rh = root._dimensions.height
    for panel in _iter_panels(root):
        ox, oy = _absolute_origin(panel)
        w, h = panel._dimensions.width, panel._dimensions.height
        # mpl axes are bottom-up; jeanplot layout is top-down
        yield panel, (ox / rw, 1.0 - (oy + h) / rh, w / rw, h / rh)
```

`_iter_panels` / `_iter_overlays` are 5-line tree walks filtering by
`isinstance(c, PlotPanel)` + `is_overlay`.

**`PlotFunctionResult` gains a `mappable: Any | None = None` field.**
This is the one structural change to step 01's `PlotFunctionResult`
— a sibling to `rendering` and `metadata`. Panels that produce a
colorbar-eligible mappable (`SmoothPanel2D`, `DensityPanel1D`, etc.)
set it; everyone else leaves it `None`. Colorbar children read
`parent._mappable`. Done.

### 2.6 `jeanplot/render.py` — extension

Today `jeanplot.render(component, ...)` calls
`MatplotlibRenderer.render_component(context, component)`. Extend:

```python
def render(component, *, output_path=None, backend="matplotlib",
           overwrite=True, **kwargs):
    if isinstance(component, Figure):
        if output_path is not None:
            component.output_file = Path(output_path).name
            component.output_dir = str(Path(output_path).parent)
        if not overwrite and component.output_path and component.output_path.exists():
            return None
        renderer = _build_renderer(backend, **kwargs.get("renderer_kwargs", {}))
        mfig = renderer.render_component(None, component)
        _save_figure(component, mfig)
        return mfig
    # ... existing code unchanged
```

`_save_figure` does what the old `FigureSpec._save_to_path` /
`_postprocess_svg` did (~150 LOC ported), parameterised on
`fig.svg_id_prefix`. The biocomp original is untouched; this is a
fresh implementation in `jeanplot/render.py` or a new
`jeanplot/_save.py` helper module.

That's the **entire** save / output lifecycle. No `Figure.render()`.
No `_apply_styles()`. No `_make_mpl_figure_and_axes()`. The lifecycle
is the renderer's existing one, with one new dispatch branch for
`Figure`.

### 2.7 `LayoutConstraints` flex weights (minor `core/` extension)

`biocomp.plotutils.GridLayout` supports `col_widths: list[float]` and
`row_heights: list[float]` summing to 1.0. To express the same with
`Container`, extend `LayoutConstraints` (in `jeanplot/core/models.py`)
with two optional fields:

```python
class LayoutConstraints(BaseModel):
    # ... existing ...
    main_axis_weights: list[float] | None = None     # one entry per child along main axis
    cross_axis_weights: list[float] | None = None    # used for wrapped grids
```

If `main_axis_weights` is set and its length matches the child count,
the layout engine distributes available space along the main axis by
those weights (after subtracting fixed gaps and margins). Otherwise
the existing equal-share behaviour stands. ~30 LOC including tests.

This is the **only** change to `jeanplot/core/` in this step. (Step 00
already did the bigger one — the cascade rewrite.)

### 2.8 `Container` and the figure-bbox cascade

A subtle point worth pinning down: `Figure(Container)` and every
intermediate `Container` lay out children in *inches* (or some other
absolute unit consistent across the tree). Leaf `PlotPanel`s declare
their natural / min / max dimensions in inches. The renderer converts
to figure-relative coordinates at axes-allocation time.

No new code is needed for this — `Container.measure_and_layout`
already works in unit-free numbers. The convention is just "for plot
trees, treat the numbers as inches". This is documented in
`PlotPanel`'s docstring; no machinery change.

### 2.9 Tests

- `tests/test_plot_panel_base.py` — instantiate a minimal subclass that
  overrides `draw()` with a recorder; assert it's called with an Axes.
- `tests/test_figure_render.py` — make a `Figure` with two minimal
  `PlotPanel` subclasses, render to a temp PNG, assert the file
  exists, is non-empty, and has the expected metadata `Subject` field.
  Also assert `fig.render()` and `jeanplot.render(fig)` are
  interchangeable (the former is a thin forwarder).
- `tests/test_figure_theme_field.py` — set `fig.theme` to a fixture
  `!cascade:jstyle` value; render; assert the cascade was applied
  (panel properties reflect the rules). With `fig.theme = None`, the
  ambient `jstyle` state stays untouched.
- `tests/test_figure_layout_to_axes.py` — `Figure` with
  `LayoutConstraints(direction="row")` and three children; assert
  axes are allocated at the expected figure-relative fractions.
- `tests/test_layout_flex_weights.py` — `Container` with
  `main_axis_weights=[2, 1, 1]`; assert child widths are 50% / 25% / 25%.
- `tests/test_colorbar_overlay.py` — `Colorbar` as an overlay child
  of a recorder-panel: assert `Colorbar.draw` is called with the
  parent's axes AND can read `parent._mappable`.
- `tests/test_nested_panels.py` — a `Container` holding two
  `Container`s each holding two `PlotPanel`s; assert all four axes
  are allocated at the correct nested-bbox positions. (This is the
  test that pins down "3D cube + slice grid" working out of nested
  Containers in step 03.)

## Out of scope

- No actual plot kinds (step 03). Tests use recorder subclasses.
- No themes / `themes/plots.yaml` (step 04).
- No biocomp / biocomp-tools changes (never, in any step).
- No new jstyle engine code (step 00 owned that).

## Estimate

~250 LOC added (`panels/base.py` ~80, `panels/figure.py` ~30,
`MatplotlibRenderer` extensions ~80, `core/models.py` flex-weight
extension ~30, `render.py` extension + `_save_figure` ~150 — note
`_save_figure` is ported drawing-code, not new design). Tests: ~150
LOC.

The key reduction vs the original plan: **no `Figure.render()`
method, no parallel render lifecycle.** The renderer already had a
lifecycle; we extended it with one branch. That cuts ~150 LOC of
duplicated orchestration.
