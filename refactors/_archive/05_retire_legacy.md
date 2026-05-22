# 05 — Retire legacy

## What

Remove the shims and obsolete machinery left over after step 03. Every consumer that imported from `biocomp.plotting.*`, `biocomp.plotutils`, or `biocomp-tools.toollib.plot`'s legacy `Figure` now imports from `jeanplot.plots` directly. The shim layer is deleted. Obsolete legacy modules (`biocomptools/toollib/plot.py`'s `Figure` / `PlotConfig` / `PlotTask` / `PartialFunction`) are deleted if no consumers remain.

End state: there is *no* `biocomp.plotting` module. `biocomp.plotutils` either doesn't exist or contains only biology-specific helpers (`extract_plot_data_from_network` and friends). The legacy biocomp-plot `Figure` machinery is gone.

## Why

Step 03 stops the bleeding (one source of truth). Step 05 finishes the job (one *import path*). Until the shims are gone, drift can still creep in — someone "fixes a bug in biocomp" without realizing it's now a thin re-export, gets confused, splits the implementation again. Until the legacy `Figure` machinery is gone, paper-jobs files that should be on native panels can be quietly rewritten to use it. Removal closes the door.

This also unlocks deletion of a chunk of dead infrastructure: `PartialFunction` (used only by legacy `Figure`), `@configurable` / `generate_full_nested_config` (used only by legacy `Figure`'s `callstack_params`), `FigureSpec` / `SimpleLayout` / `GridLayout` / `MultiRowGridLayout` (replaced by `jeanplot.Figure` + `Container` + `LayoutConstraints`).

## Why last

Depends on:
- **Step 03** — must land first or nothing works.
- **Migration of the remaining ~21 paper-jobs files** that still use `BiocompFigureAdapter`. Steps 02 + 04 give us the templates and aliases to migrate; the actual migration of all 21 files happens as a follow-up. If we can't get them all migrated before this step, the 05.C "delete" pass is deferred and the legacy `Figure` stays in place, marked deprecated.

Does not block anything later.

## Current state

After step 03:

- `biocomp.plotting.*` — all modules are shim re-exports (`from jeanplot.plots.X import Y`).
- `biocomp.plotutils` — most functions are shim re-exports; a handful (network-aware extractors, biology-specific palette helpers) remain as real implementations.
- Internal biocomp consumers (`biocomp/designutils.py`, `biocomp/datautils.py`, `biocomp/plotting/*` cross-deps) still import from `biocomp.plotting.*` (going through shims to jeanplot).
- `biocomp-tools` consumers (`biocomptools/plot.py`, `biocomptools/run_eval.py`, `biocomptools/hyperopt_analysis.py`, `biocomptools/circuitplot.py`, multiple `biocomptools/toollib/*`) similarly import from the shim layer.
- `biocomp-tools/biocomptools/toollib/plot.py` still contains the legacy `Figure` machinery (`Figure`, `PlotConfig`, `PlotTask`, `PartialFunction` import). May still be used by paper-jobs files that haven't migrated, plus biocomp-jobs (frozen but possibly still referenced).
- The 21 remaining paper-jobs files using `BiocompFigureAdapter` are the gating consumers.

## Design

Three passes, each its own commit.

### 05.A — Rewrite imports across biocomp + biocomp-tools

Mechanical search-and-replace: every `from biocomp.plotting.X import Y` → `from jeanplot.plots.X import Y`. Every `from biocomp.plotutils import Y` likewise (excluding the biology-specific functions that stay in biocomp).

Files affected:
- `biocomp/biocomp/designutils.py`
- `biocomp/biocomp/datautils.py`
- `biocomp/biocomp/plotting/*.py` (the cross-deps — these become uninvolved once their hosts are deleted)
- `biocomp/tests/test_mvp.py`, `biocomp/tests/test_text_plotting.py`
- `biocomp-tools/biocomptools/plot.py`
- `biocomp-tools/biocomptools/run_eval.py`
- `biocomp-tools/biocomptools/hyperopt_analysis.py`
- `biocomp-tools/biocomptools/circuitplot.py`
- `biocomp-tools/biocomptools/toollib/design_eval.py`
- `biocomp-tools/biocomptools/toollib/interactive_link.py`
- `biocomp-tools/biocomptools/toollib/pickle_utils.py`
- `biocomp-tools/biocomptools/toollib/plot.py`
- `biocomp-tools/biocomptools/toollib/overlays.py`
- `biocomp-tools/biocomptools/toollib/design_data.py`

After this pass:
- Run jeanplot, biocomp, biocomp-tools tests. All pass.
- Render fig1 — visual parity.

### 05.B — Delete the shim layer

Now that nothing imports from `biocomp.plotting.*` or `biocomp.plotutils.*` (for the migrated functions):

- Delete every shim file under `biocomp/biocomp/plotting/`. The directory should end up either empty (delete it) or holding only the biology-aware files that stay (e.g., if `plotting_core.py` keeps `get_bio_color`, `network_ticks_and_labels`, etc. — rename it `biocomp/plot_bio.py` to make the new scope explicit).
- Trim `biocomp/biocomp/plotutils.py` to the biology-specific functions only (`extract_plot_data_from_network`, `extract_lazy_plot_data_from_network`). Rename to `biocomp/network_plot_data.py` to reflect new scope. Or fold into `biocomp/datautils.py`.
- Update `biocomp/biocomp/__init__.py` exports.

After this pass:
- A grep for `biocomp.plotting` or `biocomp.plotutils.smooth` across the workspace returns *zero* hits.
- Tests still pass.

### 05.C — Remove the legacy Figure machinery (conditional)

`biocomp-tools/biocomptools/toollib/plot.py` contains the legacy `Figure(BaseModel)` / `PlotConfig` / `PlotTask` machinery — replaced by `jeanplot.Figure` + `Container` + Panel tree. Plus `biocomp.utils.PartialFunction` (used only by `PlotTask.plot_method`) and the `@configurable` / `generate_full_nested_config` system (used only by `PlotConfig.callstack_params`).

**Precondition**: all paper-jobs `plot/*.yaml` files migrated off `BiocompFigureAdapter` to native panels (the pattern was established in step 04 — this step just verifies completion).

If precondition met:
- Delete `Figure`, `PlotConfig`, `PlotTask`, `prepare_func`, `load_default_plotconf`, `load_default_rescaler` from `biocomp-tools/biocomptools/toollib/plot.py`. The file may end up empty (delete it) or hold only utility helpers.
- Delete `PartialFunction` (and `PartialFunctionResult`, `ExecuteFunction`) from `biocomp/biocomp/utils.py`.
- Delete the `@configurable`, `generate_full_nested_config`, `generate_base_nested_config` machinery from `biocomp/biocomp/utils.py`.
- Delete `FigureSpec`, `FigAx`, `SimpleLayout`, `GridLayout`, `MultiRowGridLayout`, `MergeSpec` if no remaining consumers. (Quick grep before deleting.)
- Delete `biocomp-tools/biocomptools/jeanplot_panels/biocomp_figure_adapter.py` — no longer needed.
- Update `biocomp-tools/biocomptools/configs/plot_config/default_plotconf_v2.yaml` — either delete or trim to a jeanplot-compatible theme.

If precondition NOT met:
- Mark `BiocompPlotFigure`, `PlotConfig.prepare_func`, `PartialFunction` with `@deprecated` and a `DeprecationWarning` on use.
- Document the unmigrated paper-jobs files; track in an issue for follow-up.

After this pass:
- Workspace contains zero references to `callstack_params` outside of frozen biocomp-jobs YAML files.
- Workspace contains zero references to `PartialFunction` (or its consumers).
- biocomp-tools tests still pass.
- All paper-jobs plot YAML files render correctly via native jeanplot.

## Tests

- Most tests carry over from step 03 — the same parity tests now verify "import from jeanplot directly" produces identical results.
- New: a "no-broken-imports" smoke test that imports every module in biocomp and biocomp-tools and asserts nothing throws on import. Catches stale `from biocomp.plotting.X import Y` references we missed.
- Grep-based regression in CI: `! grep -rn 'from biocomp.plotting' biocomp/ biocomp-tools/ paper-jobs/` returns nothing. Same for `biocomp.plotutils.smooth`. Same for `PartialFunction` (if precondition met).

## Risks

- **Frozen biocomp-jobs files.** `biocomp-jobs/` is preserved for provenance lookups but should not be active. If anyone runs a biocomp-jobs script post-step-05, it will break. Acceptable per the documented policy ("OBSOLETE legacy job repo — read-only reference"), but worth a note in the project README so it's not surprising.
- **External consumers.** If anything outside this workspace (the biocomp-devbook, scratch scripts, notebooks) imports `biocomp.plotting.X`, those break. Easy fix — change imports to jeanplot. Track via a quick `grep -r` in the dev-book repo before landing.
- **Test fixtures with pickled `PartialFunction` instances.** Some training-history pickle files may contain serialized `PartialFunction` objects. Verify the pickle paths don't try to deserialize these after the class is gone; if they do, leave a deprecation stub that emits a warning.
- **biocomp-tools' `load_default_plotconf` / `load_default_rescaler`.** Currently used by the `BiocompFigureAdapter` path. Once the adapter is deleted, these helpers go too. But if any paper-jobs migration still depends on them, this becomes precondition for step 05.C.

## Acceptance

- `find biocomp/ biocomp-tools/ -name "plotting_*.py"` returns nothing (or only the biology-specific renamed files).
- `grep -r "from biocomp.plotting" biocomp/ biocomp-tools/ paper-jobs/` returns nothing.
- `grep -r "PartialFunction" biocomp/ biocomp-tools/` returns nothing if 05.C complete; only deprecation stubs if deferred.
- All tests pass (jeanplot, biocomp, biocomp-tools, paper-jobs render checks).
- `paper-jobs/plot/fig1_matrix_gradient.yaml` (and any other migrated files) render correctly.
- Net diff: another large negative (estimate ~1000-2000 lines deleted across the legacy `Figure` machinery and shim layer).
