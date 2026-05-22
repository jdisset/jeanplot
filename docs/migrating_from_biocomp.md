# Migrating a script from biocomp-plot to jeanplot-plot

Reference for porting an existing biocomp / biocomp-tools plotting script to
jeanplot. Organised as **before / after** pairs by what the script is doing.

The constraint behind the refactor: biocomp keeps rendering unchanged. Port
scripts one at a time, at your own pace. The `biocomp-plot` CLI and every
existing job YAML in `paper-jobs/` continue to work today.

Under the hood, `biocomp.plotting.*` is now a thin shim that re-exports the
canonical jeanplot symbols (drawing functions, KNN kernels, axis helpers,
`PlotData`, `Rescaler`). The orchestration layer it sits on
(`BiocompPlotFigure`, `PlotConfig`, `PartialFunction`, `@configurable`,
`SimpleLayout` / `GridLayout` / `MultiRowGridLayout` / `MergeSpec` / `FigureSpec`,
`BiocompFigureAdapter`) is **deprecated**: constructing any of them emits a
`DeprecationWarning` pointing at the jeanplot replacement. Old jobs still
render; new jobs should target jeanplot directly.

---

## Section 1 — Python imports

| biocomp / biocomptools import | jeanplot replacement |
|---|---|
| `from biocomp.plotutils import PlotData, LazyPlotData, FigureSpec, FigAx, SimpleLayout, GridLayout, MultiRowGridLayout, MergeSpec, PlotFunctionResult, IdentityRescaler` | `from jeanplot import PlotData, LazyPlotData, Figure, PlotFunctionResult, IdentityRescaler` — `FigAx` / `SimpleLayout` / `GridLayout` / `MultiRowGridLayout` / `MergeSpec` are no longer needed: nested `Container` + `LayoutConstraints` (with `main_axis_weights`) covers row, grid, multi-row, and merge-pages cases. `FigAx.subdivide(axnum, spec)` is replaced by nested Containers (the parent panel holds the sub-panels as `children`). |
| `from biocomp.plotting.plotting_smooth_2d import smooth_2d` | `from jeanplot.panels import SmoothPanel2D` (Component form) or `from jeanplot.plots.smooth_2d import smooth_2d` (free-function form) |
| `from biocomp.plotting.plotting_3d import smooth_3d` | `from jeanplot.panels import SmoothPanel3D` |
| `from biocomp.plotting.plotting_mvp import measured_vs_predicted` | `from jeanplot.panels import MVPPanel` |
| `from biocomp.plotting.knn_utils_np import make_tree, get_gaussian_weighted_knn` | `from jeanplot.knn import make_tree, get_gaussian_weighted_knn` |
| `from biocomp.plotutils import diagonal_xy, plot_slice_overlay, plot_slice_chords` | `from jeanplot.plots.overlays import diagonal_xy, plot_slice_overlay, plot_slice_chords` |
| `from biocomptools.toollib.plot import Figure, PlotTask, PlotConfig` | `from jeanplot import Figure` — `PlotTask` / `PlotConfig` dissolve into Figure's `children` + jstyle theme |
| `from biocomp.datautils import DataRescaler` | unchanged — biocomp keeps its `DataRescaler`; jeanplot's `Rescaler` protocol accepts it structurally (no adapter needed) |
| `from biocomp.plotutils import extract_plot_data_from_network` | unchanged — biocomp-specific, stays in biocomp |
| `from biocomptools.toollib.figuremakers.networkdiagram import ...` | unchanged — biocomp-specific panels stay in biocomptools (see §3) |

---

## Section 2 — YAML

### Before (biocomp-plot style)

```yaml
plot_config: !include pkg:biocomptools:configs/plot_config/default_plotconf_v2

figure_spec:
  output_dir: ./
  output_file: out.pdf
  layout: !SimpleLayout { cols: 2 }

plot_tasks:
  - !deferred
    !define D: ${ground_truth_data}
    !define axnum: 0
    <<: !include pkg:biocomptools:configs/plots/tasks/auto
```

### After (jeanplot-plot style)

```yaml
<<(<): !include pkg:jeanplot:resources/themes/plots

figure: !Figure
  theme: !include pkg:jeanplot:resources/themes/plots.yaml@rules
  output_dir: ./
  output_file: out.pdf
  layout: !LayoutConstraints { direction: row, gap: 8 }
  children:
    - !AutoPanel { plot_data: ${ground_truth_data} }
    - !AutoPanel { plot_data: ${predicted_data} }
```

Same figure. Less than half the YAML. No `plot_config` / `figure_spec` /
`plot_tasks` / `plot_method` / `axnum` / `task_file` indirection.
`AutoPanel` dispatches to the right `SmoothPanel{1,2,3}D` based on data
input dim — replaces the `tasks/auto.yaml` template entirely.

### Explicit kind (no dim-dispatch)

```yaml
children:
  - !SmoothPanel2D { plot_data: ${D1} }
  - !MVPPanel      { measured: ${m}, predicted: ${p} }
```

### Iterated case — a row of panels, one per dataset

```yaml
children:
  !each(d) ${datasets}:
    - !AutoPanel { plot_data: ${d} }
```

---

## Section 3 — biocomp-specific panels inside a jeanplot Figure

`NetworkDiagramPanel`, `CircuitPanel`, `MVPNetworkPanel` are
biocomp-specific. They stay in biocomp-tools. Three ways to use them
inside a jeanplot Figure:

### Side-by-side composites (no jeanplot wrapping)

Render the biocomp piece and the jeanplot piece as two separate
matplotlib figures, then composite post-hoc with PDF/PNG concatenation.
Useful when the two halves don't need to share theming.

### Hybrid: import biocomp panel as a Component

Most paper-jobs scripts will live here. Import the biocomp panel class
and drop it into the Figure tree alongside jeanplot panels:

```python
from jeanplot import Figure
from jeanplot.panels import SmoothPanel2D
from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagramFigure

fig = Figure(
    output_file="combo.pdf",
    children=[
        SmoothPanel2D(plot_data=data),
        # NetworkDiagramFigure / your own adapter, etc.
    ],
)
fig.render()
```

### Full Component-tree integration via a local adapter

If you want a biocomp panel to participate in jeanplot's layout, theming,
and overlay machinery, write a small adapter in your own script (not in
jeanplot, not in biocomp):

```python
# in your migration script
from typing import Any

from jeanplot.panels.base import PlotPanel
from biocomptools.toollib.figuremakers.networkdiagram import draw_network_diagram


class NetworkDiagramPanel(PlotPanel):
    network: Any                    # biocomp.network.Network — jeanplot doesn't import it
    layout_spec: Any | None = None

    def draw(self, ax):
        draw_network_diagram(self.network, ax=ax, layout_spec=self.layout_spec)
        return None
```

Twenty lines. Local to your script. Doesn't pollute jeanplot's
general-purpose surface.

---

## Section 4 — callstack params → jstyle

### Before

```yaml
plot_config:
  callstack_params:
    smooth_2d_params:
      vlims: [-1, 1]
      colorbar_params:
        size: [0.05, 0.7]
```

### After — override at the Panel call site

```yaml
children:
  - !SmoothPanel2D
    plot_data: ${D}
    vlims: [-1, 1]
    Colorbar:
      size: [0.05, 0.7]
```

### After — global via a theme override

```yaml
<<(<): !include pkg:jeanplot:resources/themes/plots
<<(<):
  rules: !cascade:jstyle
    SmoothPanel2D:
      vlims: [-1, 1]
      Colorbar:
        size: [0.05, 0.7]

figure: !Figure
  children:
    - !SmoothPanel2D { plot_data: ${D} }
```

Selector specificity gives you per-panel / per-class / per-id overrides
without restating the rest of the config.

---

## Section 5 — Rescaler

`DataRescaler` from biocomp satisfies jeanplot's `Rescaler` protocol
structurally. Pass it as-is:

```python
from biocomp.datautils import DataRescaler
from jeanplot.panels import SmoothPanel2D

panel = SmoothPanel2D(plot_data=data, rescaler=DataRescaler(...))
```

No adapter needed. No biocomp-side change required.

---

## Section 6 — CLI

### Before

```bash
biocomp-plot +paper-jobs/plot/figures/autofig_pred_combined.yaml \
    ++ground_truth_data=$GT ++predicted_data=$PRED \
    ++output_dir=/tmp/out
```

### After

```bash
jeanplot-plot +my_figure.yaml ++plot_data=... -o /tmp/out
```

`jeanplot-plot` is the jeanplot-native counterpart to `biocomp-plot`. It's
a `@dracon_program` wrapping a `figure: Figure` field — any `!set_default`
declared in your YAML surfaces as a `--name` CLI flag, exactly like
`biocomp-plot`'s vocab-as-CLI.

---

## Section 7 — Verification checklist

After porting a script, verify with:

```bash
# 1. The jeanplot job composes and renders
jeanplot-plot +my_new_figure.yaml -o /tmp/jeanplot-out

# 2. The original biocomp job still works (it is unchanged)
biocomp-plot +path/to/original.yaml ++output_dir=/tmp/biocomp-out

# 3. (Optional) Compare outputs visually or via pixel-tolerance diff
#    See jeanplot/tests/parity/ for the snapshot-test pattern.
```

---

## What this migration does **not** change

- Network-aware panels stay in `biocomptools.jeanplot_panels` (CircuitPanel,
  NetworkDiagramPanel, MVPNetworkPanel, BlurbPanel, smooth-voxel /
  benchmark / quantile-coverage panels, biocomp-aware data holders).
- `DataRescaler` keeps working as-is; it already satisfies jeanplot's
  `Rescaler` protocol structurally.
- Existing job YAMLs in `paper-jobs/` / `biocomp-jobs/` keep rendering;
  the only difference is that any construction of the legacy
  `Figure` / `PlotConfig` / `SimpleLayout` / etc. now emits a
  `DeprecationWarning`.
- `biocomp-plot` CLI keeps working.

## What this migration **does** change

- `biocomp.plotting.*` is no longer a fork. Every drawing function, KNN
  kernel, axis helper, `PlotData`, `Rescaler` lives once in jeanplot;
  `biocomp.plotting` re-exports those symbols. Bug fixes happen in one
  place.
- The orchestration legacy (`BiocompPlotFigure`, `PlotConfig`,
  `PartialFunction`, `@configurable`, `SimpleLayout` / `GridLayout` /
  `MultiRowGridLayout` / `MergeSpec` / `FigureSpec`, `BiocompFigureAdapter`)
  is deprecated. It will be deleted once the remaining paper-jobs YAMLs
  migrate to the jeanplot `Figure` skeleton.
- The canonical paper figure is now a one-line skeleton include
  (`pkg:jeanplot:resources/themes/paper`) plus the figure-specific shape.
  See `paper-jobs/plot/fig1_matrix_gradient.yaml` for the reference port.

Mix biocomp and jeanplot freely in a single script if that's the simplest
path; the deprecation warning is the migration prompt, not a wall.
