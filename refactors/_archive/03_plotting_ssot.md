# 03 — Plotting SSOT in jeanplot (port + shim)

## What

Every drawing function, KNN helper, axis-setup utility, ASCII variant, rescaler class, and `PlotData` type currently duplicated between `biocomp/` and `jeanplot/` lands in jeanplot exactly once. `biocomp.plotting` / `biocomp.plotutils` become thin re-export shims pointing at `jeanplot.plots`. After this step there is one implementation of `smooth_2d`, one of `knn_grid`, one of `setup_transformed_axis`, one of `DataRescaler` — all in jeanplot. Step 05 removes the shims; this step gets us to "one source, multiple aliases".

## Why

Today we have a fork, not a refactor. ~1900 LOC in `biocomp.plotting.*` plus ~1100 LOC in `biocomp.plotutils` overlap with ~600 LOC in `jeanplot.plots.*`. Bug fixes have to be applied twice and we already failed at it once this session (`arrow_scale` in `gradient_field_2d` was fixed in jeanplot only; biocomp's copy still has the inverted formula). Every drawing function that exists in both places is a future divergence waiting to happen.

The whole refactor is about SSOT. This is the SSOT step.

## Why this order

Step 02 generates panels from drawing functions; the panels are agnostic to *which side* the function lives on, so the panel migration can happen against either biocomp or jeanplot functions. Currently jeanplot already has the smaller, more-recent copies — step 02 uses those. This step then ports the biocomp *superset* knobs over (so the larger biocomp implementation wins on features), and biocomp.plotting becomes a shim.

If a function gains new knobs during the port, regenerate the corresponding panel via `panel_from` (cheap — one-line factory call from step 02).

Step 05 (legacy retirement) lands after this and after the paper-jobs migration is complete.

## Current state — inventory

**Pure plotting functions duplicated** (both sides, must consolidate):

| biocomp | jeanplot | function |
|---|---|---|
| `biocomp/plotting/plotting_smooth_1d.py` | `jeanplot/plots/smooth_1d.py` | `smooth_1d` |
| `biocomp/plotting/plotting_smooth_2d.py` | `jeanplot/plots/smooth_2d.py` | `smooth_2d`, `smooth_grad_magnitude_2d`, `gradient_field_2d`, `knn_grid`, `knn_gradient_grid` |
| `biocomp/plotting/plotting_3d.py` | `jeanplot/plots/cube.py` (partial) | `smooth_3d`, cube-face rendering, projection helpers |
| `biocomp/plotting/plotting_density.py` | `jeanplot/plots/density.py` | density variants |
| `biocomp/plotting/plotting_scatter.py` | `jeanplot/plots/scatter.py` | `scatter_*`, `grid_histogram`, `make_density_cmap` |
| `biocomp/plotting/plotting_violin.py` | `jeanplot/plots/violin.py` | `smooth_voxel_conditioned_violin` |
| `biocomp/plotting/plotting_mvp.py` | `jeanplot/plots/mvp.py` | mvp scatter + diagnostics |
| `biocomp/plotting/plotting_particle.py` | `jeanplot/plots/particle.py` | `particle_plot` |
| `biocomp/plotting/plotting_txt.py` | `jeanplot/plots/txt.py` | ASCII variants |
| `biocomp/plotting/stacked_poly.py` | `jeanplot/plots/stacked_poly.py` | poly fitting |
| `biocomp/plotting/ascii_heatmap.py` | `jeanplot/plots/ascii_heatmap.py` | ASCII heatmap |
| `biocomp/plotting/plotting_core.py` | partial in `jeanplot/plots/ticks.py` + `smooth_kernel.py` | `setup_transformed_axis*`, `format_powers`, `network_ticks_and_labels`, `knn_stats` |
| `biocomp/plotting/plotting_smooth.py` | needs new `jeanplot/plots/dispatch.py` | `smooth_line_plot`, `smooth_line_slices` |
| `biocomp/plotting/knn_utils_np.py` | needs new `jeanplot/plots/knn_np.py` | `make_tree`, `knn_density`, `get_gaussian_weighted_knn`, `get_knn_mean_and_variance` |
| `biocomp/plotting/knn_utils_jax.py` | needs new `jeanplot/plots/knn_jax.py` | jax variants of the above + `weighted_quantile` |
| `biocomp/plotutils.py` (most of it) | `jeanplot/plots/overlays.py` + others | `smooth` dispatcher, `slice_panel_args`, `plot_diagonal_paths`, `histogram`, `make_xy_grid`, `compute_shared_vlims`, `combine_dicts`, `violin_style`, `normalized_violin` |

**Data abstractions that need a single home** (jeanplot, with biocomp re-export):

- `DataRescaler` base + `CompressedSymLogRescaler` + `IdentityRescaler` + any others. Currently in `biocomp/datautils.py`. Generic class — no biology dependency.
- `PlotData`. biocomp has `biocomp.datautils.PlotData` (fields: `x`, `y`, `input_names`, `output_name`, `metadata`, `column_names`, `dimensions`); jeanplot has `jeanplot.data.plot_data.PlotData` (fields: `xval`, `yval`, …). The two are near-identical except for field names. Consolidate to ONE type with both name aliases (`x`/`y` as `@property` for back-compat).

**Stays in biocomp** (genuinely biology-specific):

- `biocomp.plotutils.extract_plot_data_from_network` / `extract_lazy_plot_data_from_network` — read a `Network` and produce a `PlotData`. Needs biocomp's `Network` type.
- `biocomp.plotting.plotting_core.get_bio_color`, `get_reordered_protein_names`, `network_ticks_and_labels` — protein-name aware.
- Anything else that imports `biocomp.recipe` / `biocomp.network` / etc.

## Design

Three groups of work, executed in order. Each group lands as one PR to keep diffs reviewable.

### 03.A — Migrate the kernels and primitives

Low-level first, so higher-level functions can be ported on top.

1. **`DataRescaler` + concretes.** Move `biocomp.datautils.DataRescaler` (and `CompressedSymLogRescaler`, `IdentityRescaler`, any others) into `jeanplot/data/rescaler.py`. Keep / merge with the existing `jeanplot.data.IdentityRescaler`. In `biocomp.datautils`, replace the class bodies with `from jeanplot.data.rescaler import DataRescaler, CompressedSymLogRescaler, …`.
2. **`PlotData`.** Decide field names — recommend keeping jeanplot's `xval/yval` going forward (active development is on this side). Add `x` / `y` as `@property` aliases for back-compat. Move the full type to `jeanplot/data/plot_data.py` if not already there; `biocomp.datautils.PlotData = jeanplot.data.PlotData` shim.
3. **KNN kernels.** `biocomp/plotting/knn_utils_np.py` and `biocomp/plotting/knn_utils_jax.py` → `jeanplot/plots/knn_np.py` and `jeanplot/plots/knn_jax.py`. Verify nothing in biocomp's versions depends on biocomp-specific types (they're numpy / jax — should be clean). Re-export from biocomp.
4. **`knn_grid`, `knn_gradient_grid`, `knn_stats`.** Copies in both biocomp (`plotting_core.py`, `plotting_smooth_2d.py`) and jeanplot (`smooth_kernel.py`). Reconcile signatures — biocomp likely has more knobs (`radius`, `min_points`, `max_centroid_offset_frac`). Take the superset.
5. **Axis-setup utilities.** `setup_transformed_axis`, `setup_transformed_axis_generic`, `setup_xaxis`, `setup_yaxis`, `setup_symlog_*`, `get_transformed_ticks_and_labels`, `format_powers`, `powers_of_ten` — port to `jeanplot/plots/ticks.py`. Heavily used.
6. **Heatmap primitives.** `heatmap()` and `_render_smooth_heatmap` already exist in jeanplot; biocomp has its own copies. Reconcile features — biocomp's `heatmap` accepts more keyword args (`axtransform`, opacities, contours, `bad_color`). Take the superset in jeanplot.

### 03.B — Migrate the drawing functions

For each function: port to jeanplot with the superset of knobs, write a regression test that confirms output matches biocomp's for the same inputs, replace biocomp's implementation with `from jeanplot.plots.X import Y`. If the function's signature changes during the port, regenerate its panel with `panel_from` (one-line update from step 02).

1. `smooth_1d` — confirm biocomp's 350 LOC has features jeanplot's 368 doesn't (or vice versa). Port the missing knobs.
2. `smooth_2d`, `smooth_grad_magnitude_2d`, `gradient_field_2d` — biocomp's `smooth_2d` is 785 LOC including all helpers; jeanplot's is 228. Biocomp has more knobs (arrow width, normalize_arrows, dot threshold/size, color_by). Port to jeanplot.
3. `smooth_3d` — **not in jeanplot yet.** Biocomp's `plotting_3d.py` is 806 LOC of cube rendering, cabinet projection, edge styling, slice machinery. Single biggest piece of work. Port to `jeanplot/plots/cube.py`. Regenerate `SmoothPanel3D` via `panel_from` after.
4. ASCII variants (`smooth_1d_txt`, `smooth_2d_txt`, `smooth_3d_txt`) — `biocomp/plotting/plotting_txt.py` → `jeanplot/plots/txt.py`.
5. `density`, `scatter`, `violin`, `mvp`, `particle`, `stacked_poly` — port each. Most are ~50-200 LOC. Check feature parity per function.
6. Overlay variants (`plot_slice_overlay`, `plot_slice_chords`, `plot_addition_vs_removal_overlay`, `plot_diagonal_paths`, `slice_panel_args`, `diagonal_xy*`, `diagonal_slice_path_latent`) — already in `jeanplot/plots/overlays.py`. Compare signatures, port any missing knobs.

### 03.C — Migrate helpers and dispatch

1. `smooth` dispatcher — `biocomp/plotutils.py:smooth` dispatches by input dim. jeanplot doesn't have a single dispatcher (panels handle that structurally via `auto_panel`). Decide: keep the function-level dispatch as a REPL convenience — port to `jeanplot/plots/dispatch.py`.
2. Generic helpers — `compute_shared_vlims`, `make_xy_grid`, `histogram`, `combine_dicts`, `ax_to_list`, `get_figsize_default`. Move to `jeanplot/plots/utils.py` (or distribute into the relevant modules). Shim from biocomp.
3. Audit `biocomp.plotutils` once more — anything left? If yes, it's either biology-specific (stays) or unused (delete in step 05).

## Implementation steps

Per the three groups above. After each function migration:

- Run the full jeanplot test suite. Stays green.
- Run biocomp's test suite. Stays green (via shim re-exports).
- Render `paper-jobs/plot/fig1_matrix_gradient.yaml` and confirm visual parity.
- If the function gained new knobs, regenerate its panel: `SmoothPanel2D = panel_from(smooth_2d, txt_fn=smooth_2d_txt)`.

Migration commits should be small — one function or one related group per commit. Reviewable diffs matter here.

## Tests

For each function ported in 03.B:

- A pixel-parity test: call the biocomp version and the new jeanplot version with the same inputs; assert output arrays equal (or visual diff ≤ ε for rendered figures).
- After shim is in place, the test imports from biocomp and confirms it now goes through jeanplot (smoke check via `inspect.getsourcefile`).

For the data abstractions (rescaler, PlotData):

- Round-trip tests: construct via biocomp path, construct via jeanplot path, assert identical attributes and behavior.
- `PlotData` `x`/`y` aliases return the same arrays as `xval`/`yval`.
- Existing biocomp tests (`biocomp/tests/test_*.py`) stay green — they're the back-compat guard.

## Risks

- **Hidden feature drift.** Biocomp's `smooth_2d` is 3.4× larger than jeanplot's. Most of that is extra knobs that paper-jobs never exercises but other consumers might. Audit per function — diff the signatures, port every keyword. Do not delete biocomp's body until parity tests pass.
- **JAX vs numpy KNN.** `knn_utils_jax.py` adds a JIT-compiled fast path. Make sure jeanplot's port preserves it; some users rely on the jax fast path for training-time visualization.
- **Import cycles.** Biocomp internals (`designutils.py`, `datautils.py`) currently import from `biocomp.plotting`. After porting rescaler and `PlotData` to jeanplot, `biocomp.datautils` will import from jeanplot, but biocomp.plotting still imports from biocomp.datautils. Check the dependency graph: biocomp → jeanplot only, never the reverse.
- **Panel-side regeneration.** If signatures change during 03.B, regenerate panels via `panel_from`. Affects test snapshots; update them in the same commit as the function migration.
- **`plotting_3d.py` size.** 806 LOC. May need its own sub-step if it can't be ported cleanly in one go. Acceptable to split off "port smooth_3d" as a substep with its own test pass.

## Acceptance

- One implementation of every drawing function and KNN helper, living in jeanplot.
- `biocomp.plotting.*` and `biocomp.plotutils` become re-export shims (no logic, just `from jeanplot.plots.X import Y`).
- All jeanplot tests pass; all biocomp tests pass (via shims).
- `paper-jobs/plot/fig1_matrix_gradient.yaml` visually unchanged.
- Net LOC: ~1500 lines deleted from biocomp; ~500-700 added to jeanplot (most of biocomp's volume was duplicate helpers). Negative diff substantial.
