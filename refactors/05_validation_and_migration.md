# Step 05 — Validation + migration guide

## Goal

Prove that jeanplot now stands on its own as a general-purpose plotting
library, and document the path for migrating biocomp / biocomp-tools
plotting scripts to use it.

**Crucial constraint: biocomp and biocomp-tools are not touched in this
refactor.** The existing `biocomp.plotutils`, `biocomp.plotting`,
`biocomptools.toollib.plot`, `biocomptools.toollib.figuremakers`,
`biocomptools.plot` CLI (i.e. `biocomp-plot`), and all the existing
`biocomp-tools/biocomptools/configs/plot...` YAML files keep working
exactly as they do today. Jeanplot is a **parallel** library that
grows from the source material; migration happens later, one script
at a time, on the user's schedule.

After this step:
- Jeanplot is a self-contained library with no biocomp imports anywhere.
- `jeanplot-plot` CLI exists, the jeanplot-native counterpart to
  `biocomp-plot`, taking jeanplot-native YAML (`!Figure` /
  `!SmoothPanel2D` / `!cascade:jstyle`) without any biocomp tags.
- Parity tests confirm jeanplot can render figures equivalent to
  what biocomp-plot produces for the representative job set.
- A migration cheatsheet documents the import / YAML mapping for when
  a user decides to port a specific plotting script.

## Why now

Steps 01-04 built jeanplot's plotting capability. This step proves it
works end-to-end against real fixtures, ships the CLI, and writes the
"how to port a script" doc so future migrations are mechanical.

## Prerequisites

Steps 01-04 complete. All jeanplot tests green.

## What changes

### 5.1 `jeanplot/cli.py` — the `jeanplot-plot` CLI entry point

A small `@dracon_program` that loads a jeanplot Figure YAML and runs
its `render()`. The counterpart to `biocomp-plot` but consuming the
new YAML shape (`!Figure` with `children`, `!cascade:jstyle` themes,
etc.).

```python
# jeanplot/cli.py
from typing import Annotated, Optional
from pathlib import Path
from pydantic import BaseModel
from dracon.commandline import Arg, dracon_program

from jeanplot import Figure


@dracon_program(
    name="jeanplot-plot",
    description="Render a jeanplot Figure YAML to PNG/PDF/SVG.",
)
class PlotJob(BaseModel):
    figure: Figure                                                  # the Figure tree (typed)
    overwrite: bool = True

    output_dir: Annotated[
        Optional[Path],
        Arg(short="o", help="Override the figure's output_dir."),
    ] = None

    def run(self) -> None:
        if self.output_dir is not None:
            self.figure.output_dir = str(self.output_dir)
        self.figure.render(overwrite=self.overwrite)


if __name__ == "__main__":
    PlotJob.run_cli()
```

Register the entry point in `jeanplot/pyproject.toml`:

```toml
[project.scripts]
jeanplot-plot = "jeanplot.cli:PlotJob.run_cli"
```

Usage:

```bash
jeanplot-plot +my_figure.yaml
```

The CLI uses dracon's vocab-as-CLI machinery — any `!set_default`
flags in the loaded YAML surface as `--name` options on the command
line. See `paper-jobs/` patterns for how user YAML drives this from
the top.

### 5.2 Parity test harness

A small `tests/parity/` directory under `jeanplot/tests/` that runs
the **same input fixtures** through both biocomp-plot and
jeanplot-plot, then compares outputs.

**Layout:**

```
jeanplot/tests/parity/
├── fixtures/                     # PlotData JSON snapshots + small Network pickles
│   ├── 1d_smooth.json
│   ├── 2d_smooth.json
│   ├── 3d_smooth.json
│   └── mvp_pair.json
├── biocomp_baselines/            # biocomp-plot output PNGs (refreshed only when biocomp changes)
│   ├── 1d_smooth.png
│   ├── 2d_smooth.png
│   ├── 3d_smooth.png
│   └── mvp_pair.png
├── jeanplot_jobs/                # jeanplot-native YAML for each fixture
│   ├── 1d_smooth.yaml
│   ├── 2d_smooth.yaml
│   ├── 3d_smooth.yaml
│   └── mvp_pair.yaml
└── test_parity.py
```

The test:

```python
# jeanplot/tests/parity/test_parity.py
import pytest
from PIL import Image, ImageChops
from pathlib import Path

PARITY_FIXTURES = ["1d_smooth", "2d_smooth", "3d_smooth", "mvp_pair"]
PIXEL_TOLERANCE = 0.02         # 2% allowable per-pixel divergence

@pytest.mark.parametrize("fixture", PARITY_FIXTURES)
def test_parity(fixture, tmp_path):
    baseline = Image.open(Path(__file__).parent / "biocomp_baselines" / f"{fixture}.png")
    out = tmp_path / f"{fixture}.png"
    run_jeanplot_plot(
        config=f"jeanplot/tests/parity/jeanplot_jobs/{fixture}.yaml",
        output=out,
    )
    actual = Image.open(out)
    diff = ImageChops.difference(baseline, actual)
    n_diff = sum(1 for px in diff.getdata()
                 if any(c > 0 for c in (px if isinstance(px, tuple) else (px,))))
    total = baseline.size[0] * baseline.size[1]
    assert n_diff / total < PIXEL_TOLERANCE
```

The `biocomp_baselines/` directory is regenerated once, manually, by
running `biocomp-plot` against the same fixture data; checked into
the repo and only refreshed when biocomp's drawing code intentionally
changes. This is *not* a unit test of biocomp — it's a snapshot of
"what biocomp produced when we forked the drawing code at refactor
time."

The fixture data (`fixtures/*.json`) is hand-built to exercise each
panel kind. No biocomp imports in the test or fixtures.

**Tolerance discussion.** A 2% per-pixel tolerance accommodates
matplotlib's font-hinting jitter across versions and SVG
rasterization nondeterminism but catches structural regressions (axes
flipped, colour mapping wrong, legend missing). For SVG outputs, use
`svg_hash` from `jeanplot.testing` for byte-stable comparison on a
normalised form.

### 5.3 Migration cheatsheet

A reference doc at `jeanplot/docs/migrating_from_biocomp.md` capturing
the common patterns a user will hit when porting a script. Structured
as **before / after** pairs, organised by what the user is doing.

**Section 1: Python imports.**

| biocomp / biocomptools import | jeanplot replacement |
|---|---|
| `from biocomp.plotutils import PlotData, LazyPlotData, FigureSpec, FigAx, SimpleLayout, GridLayout, MultiRowGridLayout, MergeSpec, PlotFunctionResult, IdentityRescaler` | `from jeanplot import PlotData, LazyPlotData, Figure, PlotFunctionResult, IdentityRescaler` — FigAx / `SimpleLayout` / `GridLayout` / `MultiRowGridLayout` / `MergeSpec` are no longer needed: nested `Container` + `LayoutConstraints` (with `main_axis_weights`) covers row, grid, multi-row, and merge-pages cases. `FigAx.subdivide(axnum, spec)` is replaced by nested Containers (the parent panel holds the sub-panels as `children`). |
| `from biocomp.plotting.plotting_smooth_2d import smooth_2d` | `from jeanplot.panels import SmoothPanel2D` — use as a Component, or call `from jeanplot.plots.smooth_kernel import smooth_2d_render` for the free-function form |
| `from biocomp.plotting.plotting_3d import smooth_3d` | `from jeanplot.panels import SmoothPanel3D` |
| `from biocomp.plotting.plotting_mvp import measured_vs_predicted` | `from jeanplot.panels import MVPPanel` |
| `from biocomp.plotting.knn_utils_np import make_tree, get_gaussian_weighted_knn` | `from jeanplot.knn import make_tree, get_gaussian_weighted_knn` |
| `from biocomp.plotutils import diagonal_xy, plot_slice_overlay, plot_slice_chords` | `from jeanplot.plots.overlays import diagonal_xy, plot_slice_overlay, plot_slice_chords` |
| `from biocomptools.toollib.plot import Figure, PlotTask, PlotConfig` | `from jeanplot import Figure` — PlotTask / PlotConfig dissolve into Figure's children + jstyle theme |
| `from biocomp.datautils import DataRescaler` | unchanged — biocomp keeps its `DataRescaler`; jeanplot's `Rescaler` protocol accepts it structurally (`isinstance(dr, jeanplot.Rescaler)` is True) |
| `from biocomp.plotutils import extract_plot_data_from_network` | unchanged — this is biocomp-specific and stays in biocomp |
| `from biocomptools.toollib.figuremakers.networkdiagram import ...` | unchanged — biocomp-specific panels stay in biocomptools; see §5.3.3 for using them inside a jeanplot Figure |

**Section 2: YAML.**

Before (biocomp-plot style):

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

After (jeanplot-plot style):

```yaml
<<(<): !include pkg:jeanplot:resources/themes/plots

!Figure
theme: ${@/rules}     # cascade picked up from the include above
output_dir: ./
output_file: out.pdf
layout: !LayoutConstraints { direction: row, gap: 8 }
children:
  - !AutoPanel { plot_data: ${ground_truth_data} }
  - !AutoPanel { plot_data: ${predicted_data} }
```

Same figure. Less than half the YAML. No `plot_config` / `figure_spec` /
`plot_tasks` / `plot_method` / `axnum` / `task_file` indirection.
`AutoPanel` dispatches to the right `SmoothPanel{1,2,3}D` based on
the data's input dim — replaces the `tasks/auto.yml` template
entirely.

For an explicit kind (no dim-dispatch needed):

```yaml
children:
  - !SmoothPanel2D { plot_data: ${D1} }
  - !MVPPanel      { measured: ${m}, predicted: ${p} }
```

For an iterated case (a row of panels, one per dataset):

```yaml
children:
  !each(d) ${datasets}:
    - !AutoPanel { plot_data: ${d} }
```

**Section 3: biocomp-specific panels in a jeanplot Figure.**

If your script needs `NetworkDiagramPanel` or `CircuitPanel`, just
import them from biocomptools and drop them into the Figure tree —
they're Pydantic models like any other Component:

```python
from jeanplot import Figure
from jeanplot.panels import SmoothPanel2D
from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagramFigure
# (NetworkDiagramFigure is the existing biocomptools class; wrap it
# in a thin PlotPanel subclass in your own code if you want it to
# slot into the children tree directly. Or render side-by-side via
# two separate figures and a post-hoc composite.)

fig = Figure(
    output_file="combo.pdf",
    children=[
        SmoothPanel2D(plot_data=data),
    ],
)
fig.render()
```

This is the "hybrid" migration mode — the script uses jeanplot for
the general figure / layout / theming and biocomptools for the
network-aware panels. Most paper-jobs scripts will probably stay in
this mode permanently since the network panels are genuinely
biocomp-specific and biocomp-tools already implements them.

If you want full Component-tree integration of biocomp panels (so
they style and layout alongside jeanplot panels), write a small local
adapter in your own script:

```python
# in your own migration script — not in jeanplot, not in biocomp
from jeanplot.panels import PlotPanel
from biocomptools.toollib.figuremakers.networkdiagram import draw_network_diagram

class NetworkDiagramPanel(PlotPanel):
    network: Any                    # biocomp.network.Network — but jeanplot doesn't import it
    layout_spec: Any | None = None

    def draw(self, ax):
        draw_network_diagram(self.network, ax=ax, layout_spec=self.layout_spec)
        return None
```

Twenty lines. Local to your script. Doesn't pollute jeanplot's
general-purpose surface.

**Section 4: callstack params → jstyle.**

Before:

```yaml
plot_config:
  callstack_params:
    smooth_2d_params:
      vlims: [-1, 1]
      colorbar_params:
        size: [0.05, 0.7]
```

After (override at the Panel call site):

```yaml
children:
  - !SmoothPanel2D
    plot_data: ${D}
    vlims: [-1, 1]
    Colorbar:
      size: [0.05, 0.7]
```

Or globally via a small theme override:

```yaml
<<(<): !include pkg:jeanplot:resources/themes/plots
<<(<):
  rules: !cascade:jstyle
    SmoothPanel2D:
      vlims: [-1, 1]
      Colorbar:
        size: [0.05, 0.7]

!Figure
children:
  - !SmoothPanel2D { plot_data: ${D} }
```

**Section 5: rescaler.**

`DataRescaler` from biocomp satisfies jeanplot's `Rescaler` protocol
structurally. Pass it as-is:

```python
from biocomp.datautils import DataRescaler
from jeanplot.panels import SmoothPanel2D

panel = SmoothPanel2D(plot_data=data, rescaler=DataRescaler(...))
```

No adapter needed; no biocomp-side change required.

### 5.4 Verification

```bash
# 1. Jeanplot tests still pass (including the new parity suite)
cd jeanplot && pytest -v

# 2. biocomp-plot still works exactly as before (sanity-check a representative job)
biocomp-plot +paper-jobs/plot/figures/autofig_pred_combined.yaml \
    ++ground_truth_data=$FIXTURE_GT ++predicted_data=$FIXTURE_PRED \
    ++output_dir=/tmp/biocomp-baseline

# 3. jeanplot-plot can produce equivalent output from a jeanplot-native job
jeanplot-plot +jeanplot/tests/parity/jeanplot_jobs/2d_smooth.yaml \
    -o /tmp/jeanplot-trial

# 4. CI grep gate — confirm no biocomp imports in jeanplot
! grep -rn "from biocomp\|import biocomp" jeanplot/jeanplot/
```

## Out of scope

- Modifying anything in `biocomp/`, `biocomp-tools/`, or `paper-jobs/`.
  They keep working exactly as they do today.
- Deleting biocomp's plotting modules. They keep working forever (or
  until the user decides, on their own schedule, that all downstream
  scripts have migrated and the originals can go).
- Auto-translating biocomp YAML to jeanplot YAML. The migration
  cheatsheet documents the patterns; mechanical translation tooling
  is out of scope.
- A unified `biocomp-plot` / `jeanplot-plot` CLI. Two CLIs coexist;
  each understands its own YAML dialect.
- Biocomp-aware panel classes inside jeanplot. None. Jeanplot has zero
  biocomp imports. If a user's migrating script needs them, the
  adapter pattern in §5.3 keeps them in user code.

## Estimate

- `jeanplot/cli.py`: ~50 LOC.
- `jeanplot/tests/parity/`: ~150 LOC of test infrastructure +
  4-8 fixture YAML files (~20 LOC each) + checked-in baseline PNGs.
- `jeanplot/docs/migrating_from_biocomp.md`: ~400 LOC of markdown.

**Net jeanplot diff at end of step 05: ~+600 LOC** on top of steps
01-04. **Zero LOC removed from biocomp or biocomp-tools.**

This is a *net-positive* refactor on disk — we're growing jeanplot
into a self-contained library without shrinking biocomp. The win is
not lines-of-code; it's:
- A clean general-purpose plotting library now exists.
- Future scripts that don't need network-aware viz can be written
  entirely against jeanplot, with no biocomp coupling.
- biocomp / biocomp-tools / paper-jobs / all existing scripts keep
  working unchanged — zero migration pressure.
- When a user does want to migrate a script, the cheatsheet makes it
  mechanical and the parity tests confirm output equivalence.

The eventual deletion of biocomp's plotting modules — if it ever
happens — is a future, opportunistic cleanup. Not part of this
refactor.
