# Step 01 — Foundations

## Goal

Add to `jeanplot/jeanplot/` the foundational pieces that every subsequent
step depends on, with **no behaviour change anywhere**. Biocomp / biocomp-tools
keep using their existing classes from their current locations and are
not touched by this or any subsequent step in the refactor.

All "move from biocomp" instructions below are actually **copies** —
biocomp's originals stay in place; jeanplot grows an independent
implementation that other code can opt into later, one script at a
time.

After this step:
- `jeanplot.data` exists with `PlotData`, `LazyPlotData`, `DataDimensions`, `PlotFunctionResult`.
- `jeanplot.knn` exists with the tree, KNN density, and Gaussian-weighted KNN utilities.
- `jeanplot.color` exists with palette loading, cmap registration, and name-matching.
- `jeanplot.data.rescaler` defines the `Rescaler` protocol (and `IdentityRescaler`).
- `biocomp.datautils.DataRescaler` is verified to satisfy the protocol structurally — no biocomp code change.
- Tests cover round-trip Pydantic loading, KNN parity vs the biocomp source (which keeps producing baseline arrays from its untouched implementation), and palette registration.

## Why now

Every Component-based panel in step 03 takes a `plot_data: PlotData` and
optionally a `rescaler: Rescaler | None`. Every 2D heatmap calls
`jeanplot.knn.grid`. Every colourful render references a cmap name.
None of that can come second. Foundations land first, in a single
self-contained step.

## Prerequisites

Step 00 (jstyle → `!cascade:jstyle` engine rewrite) complete and green.
That step is orthogonal to this one — neither would block the other in
principle — but landing 00 first keeps every subsequent step targeting
a working dialect. This step itself touches only `jeanplot/`, adding
files.

## What changes

### 1.1 `jeanplot/data/__init__.py` — new package

Re-exports `PlotData`, `LazyPlotData`, `DataDimensions`, `PlotFunctionResult`,
`Rescaler`, `IdentityRescaler`, `GridData`, `extract_grid_data`,
`grid_data_to_b64`, `grid_data_from_b64`.

### 1.2 `jeanplot/data/plot_data.py` — copy from biocomp

Source: `biocomp/biocomp/plotutils.py:80-218` (`DataDimensions`, `PlotData`, `LazyPlotData`).
**The biocomp original is unchanged**; this is a fresh copy in jeanplot.

Changes vs source:

- Replace `from biocomp.utils import ArbitraryModel` with a local minimal
  base (use `pydantic.BaseModel` with `model_config = ConfigDict(arbitrary_types_allowed=True)`).
- Rename `column_proteins: list[str] | None = None` →
  `column_names: list[str] | None = None`. Keep `column_proteins` as a
  back-compat alias property:
  ```python
  @property
  def column_proteins(self) -> list[str] | None:
      return self.column_names
  ```
  (Setter ditto, writing to `column_names`.)
- `PlotData` is intentionally *not* a Component yet. It's a data
  container that `PlotPanel`s reference. (Treating it as an
  `AnchorComponent` was tempting; it's cleaner to keep it a plain
  Pydantic value, because panels copy/derive arrays from it. The
  Anchor pattern fits inert visual nodes, not arrays. Revisit if a
  concrete need surfaces.)
- Keep `check_shapes()` assertions and the `metadata: dict[str, Any] = {}`.

### 1.3 `jeanplot/data/result.py`

Source: `biocomp/biocomp/plotutils.py:47-66` (`PlotFunctionResult`).
Verbatim copy.

### 1.4 `jeanplot/data/rescaler.py` — new

```python
# jeanplot/data/rescaler.py
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Rescaler(Protocol):
    """Bijective transform between a raw value space and a display space.

    `fwd(raw) -> display` projects into plot coordinates.
    `inv(display) -> raw` labels ticks with raw values.

    Implementations must accept numpy arrays of any shape and broadcast.
    """
    def fwd(self, x: np.ndarray) -> np.ndarray: ...
    def inv(self, x: np.ndarray) -> np.ndarray: ...


class IdentityRescaler:
    def fwd(self, x): return np.asarray(x)
    def inv(self, x): return np.asarray(x)
```

`biocomp.datautils.DataRescaler` (and its subclasses
`LogPolyLogRescaler`, `LogPlusOneRescaler`, etc.) already expose
`fwd` and `inv`. They satisfy the protocol structurally with zero
biocomp edits. Confirm with a small test that imports both and asserts
`isinstance(biocomp_dr, jeanplot.Rescaler)` is True.

### 1.5 `jeanplot/data/grid.py` — port from biocomp

Source: `biocomp/biocomp/plotting/plotting_smooth.py` (search for `class GridData`,
`extract_grid_data`, `grid_data_to_b64`, `grid_data_from_b64`).

This is the per-plot grid-stats blob carried in figure metadata. Verbatim
copy, no biocomp imports needed.

### 1.6 `jeanplot/knn/` — new package

Source: `biocomp/biocomp/plotting/knn_utils_np.py` (434 LOC) and
`knn_utils_jax.py` (82 LOC).

Split into three modules to make the API readable; the public surface
is one `from jeanplot.knn import ...`:

- `jeanplot/knn/tree.py` — `make_tree`, `_query`, `_UsearchTree`, backend resolution helpers.
- `jeanplot/knn/density.py` — `knn_density`, `knn_density_chunked`.
- `jeanplot/knn/gaussian.py` — `get_gaussian_weighted_knn`, `get_knn_mean_and_variance`, `_knn_mean_from_indices_weights`, `get_knn_mean_only`.
- `jeanplot/knn/jax_kernel.py` — verbatim from `knn_utils_jax.py`.

Replace `from biocomp.logging_config import get_logger` with the
existing jeanplot logger (`from jeanplot.core.debug import get_logger`).

Public re-exports via `jeanplot/knn/__init__.py`.

### 1.7 `jeanplot/color/` — new package

Source: `biocomp/biocomp/plotting/plotting_core.py:43-66` (the colour-map
loading + cmap registration block) and `biocomp/biocomp/config/biocomp_colors.yaml`
(the palette data).

Files:

- `jeanplot/color/palettes.py` — `load_palettes(path)` loads a YAML and
  returns the cmap dict; `register_palettes(cmaps)` registers with
  matplotlib (idempotent, unregisters-then-re-registers).
- `jeanplot/color/matching.py` — `closest_name(name, candidates, default=None)`
  using `difflib.get_close_matches`. Pure generic utility; the
  biology-flavoured `get_bio_color` (with the fluorophore alias
  dict) stays where it lives in biocomp today —
  `biocomp/biocomp/plotting/plotting_core.py:69-80`, untouched.
  When a user migrates a script that needs fluorophore colour
  lookup, they keep importing `from biocomp.plotting.plotting_core
  import get_bio_color` and pass the result into a jeanplot panel.
- `jeanplot/resources/colors/bio_palettes.yaml` — the `bc_blues`,
  `bc_diverging`, etc. colour-map data extracted from
  `biocomp_colors.yaml`. The `default_color_map` key stays.

`jeanplot/__init__.py` calls `register_palettes(load_palettes("pkg:jeanplot:resources/colors/bio_palettes.yaml"))`
at import time, so `plt.cm.get_cmap('bc_blues')` works for any caller.

### 1.8 Tests

Under `jeanplot/tests/`:

- `test_plot_data.py` — round-trip `PlotData` through `dr.dump`/`dr.loads`;
  assert `column_proteins` alias reads from `column_names`; assert
  `dimensions.input` / `.output` for 1D/2D/3D fixtures.
- `test_rescaler.py` — protocol satisfaction by `IdentityRescaler` and
  (gated on biocomp installed) by `biocomp.datautils.DataRescaler`
  and `LogPolyLogRescaler`.
- `test_knn_parity.py` — for a fixed random seed, assert
  `jeanplot.knn.get_gaussian_weighted_knn(...)` produces arrays
  identical to `biocomp.plotting.knn_utils_np.get_gaussian_weighted_knn(...)`.
  (This catches accidental refactor drift during the move.)
- `test_color_palettes.py` — load `bio_palettes.yaml`, register, assert
  `'bc_blues' in plt.colormaps()` and `not 'bc_blues' in plt.colormaps()`
  before re-registration.

## Code sketches

`jeanplot/data/__init__.py`:

```python
from jeanplot.data.plot_data import PlotData, LazyPlotData, DataDimensions
from jeanplot.data.result import PlotFunctionResult
from jeanplot.data.rescaler import Rescaler, IdentityRescaler
from jeanplot.data.grid import GridData, extract_grid_data, grid_data_to_b64, grid_data_from_b64

__all__ = [
    "PlotData", "LazyPlotData", "DataDimensions",
    "PlotFunctionResult",
    "Rescaler", "IdentityRescaler",
    "GridData", "extract_grid_data", "grid_data_to_b64", "grid_data_from_b64",
]
```

`jeanplot/__init__.py` additions (top-level convenience):

```python
from jeanplot.data import (
    PlotData, LazyPlotData, DataDimensions,
    PlotFunctionResult,
    Rescaler, IdentityRescaler,
    GridData,
)
from jeanplot.color.palettes import load_palettes, register_palettes
# At import time, register the bio-flavoured palettes.
register_palettes(load_palettes("pkg:jeanplot:resources/colors/bio_palettes.yaml"))

DEFAULT_TYPES.extend([PlotData, LazyPlotData, DataDimensions, PlotFunctionResult])
```

(Add the new types to `DEFAULT_TYPES` so dracon's `!PlotData { ... }`
tag works.)

## Verification

```bash
cd /Users/jeandisset/Code/Weiss/biocompiler/jeanplot
pytest jeanplot/tests/test_plot_data.py jeanplot/tests/test_rescaler.py \
       jeanplot/tests/test_knn_parity.py jeanplot/tests/test_color_palettes.py -v
```

All four must pass. Plus the existing jeanplot test suite must still pass:

```bash
pytest jeanplot/tests/ -v
```

Plus a quick smoke import from biocomp:

```bash
python -c "import jeanplot; from biocomp.datautils import DataRescaler; \
           import numpy as np; \
           dr = DataRescaler(...); \
           assert isinstance(dr, jeanplot.Rescaler)"
```

## Out of scope

- No `PlotPanel` class yet (step 02).
- No `Figure(Container)` (step 02).
- No themes (step 04). jstyle engine changes already happened in step 00.
- No biocomp / biocomp-tools changes — ever. They keep working as-is.
- **No `from __future__ import annotations` in any new file.** Per
  CLAUDE.md, breaks Pydantic introspection. Use `X | None` syntax
  directly.

## Estimate

~200 LOC added in jeanplot. Zero LOC removed (anywhere). Net positive.
Every subsequent step is also net positive on disk; the whole refactor
grows jeanplot, it does not shrink biocomp.
