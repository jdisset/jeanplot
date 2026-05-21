# Step 04 — Themes + autofig figures + slim compose

## Goal

Collapse the entire defaults configuration of plotting into a single
jstyle theme file (`jeanplot/resources/themes/plots.yaml`), wire up
the autofig figure templates, and port the row-construction helpers
that build Container trees. After this step the new code is *usable*
end-to-end from YAML — biocomp / paper-jobs still use the old paths
until step 05.

After this step:
- `jeanplot/resources/themes/plots.yaml` is the SSOT for every
  plot-panel default. It is a `!cascade:jstyle` document:
  `SmoothPanel2D: { vlims: ..., Colorbar: { ... } }`. This file
  replaces the entire `callstack_params` tree from
  `biocomp-tools/biocomptools/configs/plot_config/default_plotconf_v2.yaml`.
- `jeanplot/resources/themes/{default,bio,paper}.yaml` are layered presets.
- `jeanplot/resources/figures/{data,pred_combined,combined}.yaml` are
  the autofig figure templates (jeanplot-native Component trees).
- `jeanplot/compose.py` exposes ~150 LOC of tree-construction helpers
  (`build_row`, `build_figure_metadata`, etc.). The `expand_panel_atomics`
  / `compose_atomics` / `layout_dimensions` / `gap_mask` / `subdivide`
  machinery is **not ported** — nested `Container`s with
  `LayoutConstraints` + flex weights cover that ground.
- `dracon show pkg:jeanplot:resources/themes/plots.yaml -c -r` resolves cleanly.
- A YAML-driven test runs a `Figure` end-to-end from
  `pkg:jeanplot:resources/figures/data.yaml` against a fixture
  `PlotData` and produces a snapshot-stable PNG.

## Why now

Step 03 created the panel classes; they have hard-coded Pydantic
defaults today. This step shifts those defaults out of the Python
classes and into the theme YAML where they belong. Once that's done,
panel classes carry minimal in-code defaults (the safe ones —
`title: str | None = None`, etc.) and the *configuration* of those
defaults is a single editable YAML.

## Prerequisites

Steps 00-03 complete. In particular:
- Step 00's `!cascade:jstyle` engine is in place.
- Step 03's `AutoPanel` is in place — themes target it by name; the
  autofig figure templates instantiate it directly.

## What changes

### 4.1 No jstyle engine work — step 00 owned that

The single load-bearing prerequisite for theme work is the
`!cascade:jstyle` dialect. Step 00 registered it and rewrote
`style_engine.py`. Step 04 just *uses* it.

If you're reading 04 ahead of 00: do not start theme work until 00 is
landed. Themes target `!cascade:jstyle`; the dialect must exist.

### 4.2 `jeanplot/resources/themes/plots.yaml` — the SSOT

Source: `biocomp-tools/biocomptools/configs/plot_config/default_plotconf_v2.yaml`
(280 lines), specifically the `callstack_params:` block.

Rewrite as a `!cascade:jstyle` document. The top of the file declares
the dialect once; the rest is selector → properties pairs. Example
excerpt (the source uses
`callstack_params.smooth_2d_params.colorbar_params.tick_props.labelsize: 9`,
the target uses descendant-selector cascade — step 00's dialect
flattens nested mappings into descendant selectors at strategy-parse
time):

```yaml
# jeanplot/resources/themes/plots.yaml — a !cascade:jstyle document.

!set_default base_range: [0.0, 0.65]
!set_default xlims: ${base_range}
!set_default ylims: ${base_range}
!set_default zlims: ${base_range}
!set_default vlims: [null, null]
!set_default vlim_quantiles: [0.02, 0.98]
!set_default cmap: bc_blues
!set_default colors_grey: "#000000"

# --- the cascade body: one rule block per panel class ---------------------
rules: !cascade:jstyle
  # Default for every plot panel — IdentityRescaler. Domain-specific
  # themes (or user jobs) override with a custom rescaler. Jeanplot's
  # default theme has no biocomp imports of any kind.
  PlotPanel:
    rescaler: !IdentityRescaler

  SmoothPanel2D:
    xlims: ${xlims}
    ylims: ${ylims}
    vlims: ${vlims}
    vlim_quantiles: ${vlim_quantiles}
    cmap: ${cmap}
    bad_color: "#ffffff"
    contours: 0
    draw_xlabel: true
    draw_ylabel: true
    xaxis_labelpad: null
    yaxis_labelpad: null
    knn_grid_params:
      grid_resolution: 250
      is_density_plot: false
      max_centroid_offset_frac: 1.0
    Colorbar:                         # descendant: Colorbar overlay inside SmoothPanel2D
      size: [0.06, 0.85]
      position: [1.05, 0.075]
      tick_props: { labelsize: 9, pad: 3, length: 4 }

  SmoothPanel3D:
    xlims: ${xlims}
    ylims: ${ylims}
    zlims: ${zlims}
    vlims: ${vlims}
    zslices: [0.05, 0.25, 0.4, 0.55]
    cube_frac_w: 0.57
    projection_angle: 45.0
    projection_diag_coef: 0.5
    xaxis_labelpad: 2
    yaxis_labelpad: -2
    SmoothPanel2D:                    # descendant: slice panels inside SmoothPanel3D
      bad_color: "#ffffff"
      image_interpolation: bilinear
    # Colorbar is declared by the user (or in figures/*.yaml) as a child;
    # SmoothPanel3D doesn't auto-declare one. The theme styles it if present.
    Colorbar:
      position: [1.05, 0.4]
      size: [0.045, 0.5]

  MVPPanel:
    show_density: true
    show_identity: true
    show_stats: true
    show_trendline: false
    show_bias: false
    show_calibration_rms: false
    show_spread: false
    show_noise_floor: false
    noise_local: false

  DensityPanel1D:
    cmap: ${cmap}
    use_log_density: true

  # ... and so on, one block per panel kind.

  Figure:
    rc_context: !include pkg:jeanplot:resources/themes/rcparams
```

Key migration notes:

- The biocomp `default_plotconf_v2.yaml`'s `cube_edge_props` block
  (lines ~165-280) becomes `SmoothPanel3D > CubeView` nested style;
  the per-edge dicts (`bottom_left`, `bottom_right`, `top_left`,
  `top_right`) are nested into a single `cube_edge_props: {...}`
  mapping field on `CubeView`. jstyle's deep merge handles overrides
  at any depth.
- `rc_context` (the matplotlib rcParams) is a regular attribute on
  `Figure` set by a jstyle rule.
- `rescaler:` defaults to `IdentityRescaler` on the `PlotPanel` base
  class. User jobs that want a biocomp `DataRescaler` set it
  explicitly — either on the Panel call site or via a small
  user-side jstyle override. Jeanplot ships no biocomp-aware default.

The file is **~180 lines** total — net **~100 lines smaller** than the
original `default_plotconf_v2.yaml` because the `_params` suffix
boilerplate is gone.

### 4.3 `jeanplot/resources/themes/{default,bio,paper}.yaml`

`default.yaml` — the existing jeanplot theme, rewritten in step 00's
`!cascade:jstyle` dialect. Keep semantics; layered on top of `plots.yaml`
in user configs.

`bio.yaml` — fluorophore + ERN styling, isolated into a bio-only theme.
The current default theme has a hand-rolled `!each` over all known
fluorophores generating ~26 rules. With `!live component:` auto-opened
by step 00's select-mode cascade, this collapses to:

```yaml
# jeanplot/resources/themes/bio.yaml — the WHOLE per-fluorophore styling.

!define marker_colors:
  EYFP:       { dark: "#9b6600", base: "#fad26d", light: "#fff888", bright: "#FFC83Caa" }
  MNEONGREEN: { dark: "#0e633a", base: "#6ccb83", light: "#efffdd", bright: "#0EFF7377" }
  EBFP2:      { dark: "#006394", base: "#6cafc3", light: "#f3fafd", bright: "#1AD5FF60" }
  # ... one row per fluorophore family

!define normalize_marker: !fn
  !require name: "..."
  !fn : ${name.upper().replace('1X', '').replace('L0_G_', '')}

rules: !cascade:jstyle
  FluoMarker:
    SVGElement:
      color_remap:
        "#0000ff": ${marker_colors[normalize_marker(component.part_name)].light}
        "#00ff00": ${marker_colors[normalize_marker(component.part_name)].base}
    Text:
      color: ${marker_colors[normalize_marker(component.part_name)].dark}

  SourceAnnotation:
    style.border_color: ${marker_colors[normalize_marker(component.marker)].base}
    Container[style_class=source_tag]:
      style.background_color: ${marker_colors[normalize_marker(component.marker)].light}
      style.border_color: ${marker_colors[normalize_marker(component.marker)].dark}
      SVGElement:
        color_remap:
          "#0000ff": ${marker_colors[normalize_marker(component.marker)].dark}
```

That's the whole bio theme — ~30 LOC vs the current ~180. Adding a new
fluorophore = one row in `marker_colors`. No theme rule changes.

`paper.yaml` — paper-mode preset (smaller fonts, thinner lines,
no debug colours). New file; minimal content for now.

Composition pattern (the user's entry-point YAML):

```yaml
<<(<): !include pkg:jeanplot:resources/themes/default
<<(<): !include pkg:jeanplot:resources/themes/bio
<<(<): !include pkg:jeanplot:resources/themes/plots
<<(<): !include pkg:jeanplot:resources/themes/paper        # optional
```

### 4.4 `jeanplot/resources/themes/rcparams.yaml`

Source: `biocomp-tools/biocomptools/configs/plot_config/default_rcparams.yaml`.
Verbatim copy. The `Figure: { rc_context: ... }` jstyle rule (§4.2) includes it.

### 4.5 Task templates — gone

The biocomp `configs/plots/tasks/{auto,1D,2D,3D}.yml` quartet is
**not ported**. Step 03's `!AutoPanel` `!fn` template (+ the
`auto_panel()` Python helper) does the dim-dispatch. YAML at the
call site becomes one tag:

```yaml
- !AutoPanel { plot_data: ${D} }
```

If a user *needs* to keep YAML-level dim-dispatch for some reason
(e.g. they want compose-time branching for `force_dim` to skip
loading 3D KNN dependencies), they can write it themselves with `!if`
— but the default path is `AutoPanel` and there's no jeanplot-side
templates to compete with it.

### 4.6 `jeanplot/resources/figures/{data,pred_combined,combined}.yaml`

Source: `biocomp-tools/biocomptools/configs/plots/autofig_*.yml` and
`paper-jobs/plot/figures/autofig_pred_combined.yaml`.

Rewrite each as a `Figure` Component tree. The big simplification vs
the original plan: no `<<: !include pkg:jeanplot:resources/tasks/auto`
gymnastics — just use `!AutoPanel`.

Example for `pred_combined`:

```yaml
# jeanplot/resources/figures/pred_combined.yaml

# One include does three things at once (the dracon idiom):
#   1. Brings the !AutoPanel template into scope.
#   2. Propagates the theme's !set_default vars (xlims, vlims, ...)
#      into this file's scope so the cascade's ${xlims} refs resolve.
#   3. Merges the cascade itself as `rules:` into our tree, so we can
#      reference it as `${@/rules}` from the Figure below.
<<(<): !include pkg:jeanplot:resources/themes/plots

!require ground_truth_data: "PlotData (ground truth)"
!require predicted_data: "PlotData (predictions)"
!set_default file_prefix: ""
!set_default file_suffix: "_combined_pred.pdf"
!set_default output_dir: "./"

!Figure
# Self-contained: the figure carries its own theme. No hidden Python
# `load_plot_theme()` prelude. `${@/rules}` references the cascade
# brought in by the include above.
theme: ${@/rules}
output_dir: ${output_dir}
output_file: "${file_prefix}${predicted_data.metadata['network_name']}${file_suffix}"
layout: !LayoutConstraints { direction: row, gap: 8 }
children:
  - !AutoPanel
    plot_data: ${ground_truth_data}
    title: "Ground Truth"

  - !AutoPanel
    plot_data: ${predicted_data}
    title: "Predictions (rmse: ${round(predicted_data.metadata['prediction_stats']['rmse'], 4)})"
```

That's it. No `!deferred` per task, no `!define D / axnum`, no
task-file include, no duplicated theme path. **One** `<<(<): !include`
pulls in the vocabulary, the variables, and the cascade; **one**
`${@/rules}` hands the cascade to the Figure. The renderer (step 02
§2.5) calls `jstyle.update(fig.theme)` before styling.

**Higher-order config: `!fn` template figures.** Once a figure
*shape* is reused, lift it into a `!fn` template via the same
"curried vocabularies" pattern dracon supports:

```yaml
# jeanplot/resources/figures/templates.yaml
<<(<): !include pkg:jeanplot:resources/themes/plots    # vocab + cascade in scope

!define ComparePair: !fn
  !require a: "left PlotData"
  !require b: "right PlotData"
  !set_default title_a: "Ground Truth"
  !set_default title_b: "Predictions"
  !set_default output_dir: "./"
  !set_default file_prefix: ""
  !set_default file_suffix: "_compare.pdf"
  !fn : !Figure
    theme: ${@/rules}
    output_dir: ${output_dir}
    output_file: "${file_prefix}${b.metadata['network_name']}${file_suffix}"
    layout: !LayoutConstraints { direction: row, gap: 8 }
    children:
      - !AutoPanel { plot_data: ${a}, title: ${title_a} }
      - !AutoPanel { plot_data: ${b}, title: ${title_b} }
```

User-side YAML then collapses to:

```yaml
<<(<): !include pkg:jeanplot:resources/figures/templates
figure: !ComparePair { a: ${gt}, b: ${pred} }
```

Adding a new figure shape = one `!fn` block in `templates.yaml`. No
Python change.

### 4.7 `jeanplot/compose.py` — slim row helpers

Source: `biocomp-tools/biocomptools/toollib/figuremakers/datasetsummary.py`
(1132 LOC), specifically the *general* parts.

**What ports.** A small set of helpers (~150 LOC) for building
Container trees from data, plus the figure-metadata extractor. Each
helper is also registered as a dracon `!fn` template (via
`register_template`) so it doubles as a YAML tag — Python users call
`panel_row(...)`, YAML users write `!panel_row { ... }`, both
dispatch through the same function:

```python
# jeanplot/compose.py
from dracon import register_template
from jeanplot.core.container import Container
from jeanplot.core.models import LayoutConstraints
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.auto import auto_panel


def panel_row(panels: list[PlotPanel], gap: float = 8.0,
              weights: list[float] | None = None) -> Container:
    """One row of panels with optional flex weights."""
    return Container(
        layout=LayoutConstraints(direction="row", gap=gap, main_axis_weights=weights),
        children=panels,
    )


def panel_grid(rows: list[list[PlotPanel]], *, gap: float = 8.0,
               col_weights: list[list[float]] | None = None,
               row_weights: list[float] | None = None) -> Container:
    """A multi-row grid. col_weights[i] is the per-column weights for row i."""
    row_containers = [
        panel_row(row, gap=gap, weights=col_weights[i] if col_weights else None)
        for i, row in enumerate(rows)
    ]
    return Container(
        layout=LayoutConstraints(direction="column", gap=gap, main_axis_weights=row_weights),
        children=row_containers,
    )


def panels_from_datas(datas, **kwargs) -> list[PlotPanel]:
    """Map a list of PlotDatas to a list of AutoPanels (dim-dispatched)."""
    return [auto_panel(plot_data=d, **kwargs) for d in datas]


def build_figure_metadata(...) -> dict:
    """Ported from datasetsummary.build_figure_metadata (~80 LOC)."""
    ...


# Make every helper available as a dracon tag. `register_template`
# introspects the signature, so the YAML form's kwargs match the
# Python signature 1:1 with no parallel registry.
register_template(panel_row)
register_template(panel_grid)
register_template(panels_from_datas)
```

That is the **entire** `jeanplot/compose.py` (plus the metadata
helper). ~150 LOC. From YAML:

```yaml
children:
  - !panel_row
    gap: 4
    panels: !panels_from_datas { datas: ${datasets} }
```

— same composition, declarative, no Python wrapper code.

**What does NOT port.** The following biocomp machinery is *not*
copied because the new model makes it unnecessary:

| Original | Why not ported |
|---|---|
| `expand_panel_atomics` (3D-special-case yielding cube+slice atomics) | `SmoothPanel3D` already builds its own cube+slice tree in `model_post_init` (step 03 §3.3b). No expansion step. |
| `compose_atomics` (flatten rows-of-panels into atomic plot tasks with row-major axnums) | Container tree is already flat / nested as needed; renderer iterates leaves directly. No `axnum`. |
| `layout_dimensions` (derive `MultiRowGridLayout` inputs from rows) | `LayoutConstraints` with `main_axis_weights` does this directly; no `MultiRowGridLayout` exists in jeanplot. |
| `gap_mask` (mark cells as layout-only spacers) | A "gap" is just a `Container(layout=LayoutConstraints(direction="row"), min_dimensions=Size(width=W, height=0))` with no children. No mask needed. |
| `FigAx.subdivide(axnum, spec)` | Nested Containers, see §3.3b. |
| `_panel_width` (per-kind width resolver with input-dim-aware kind_widths) | `panel_row(weights=[...])` or per-panel `min_dimensions.width`. Theme sets defaults via jstyle. |

If the migration of a specific script reveals a need for something in
this list that genuinely can't be expressed via nested Containers,
add it then — not now. The whole point of the refactor is to find out
which of these existed because they were the easiest path, not
because they were necessary.

**The biocomp-specific** panel kinds (`diagram`, `circuit`,
`mvp_network`) are NOT moved — they stay in
`biocomp-tools/biocomptools/toollib/figuremakers/datasetsummary.py`
as biocomp-side extensions. After step 05 those bits become small
extension files that register their Component subclasses; the helper
functions in `jeanplot.compose` work on any `PlotPanel`, including
biocomp's.

### 4.8 jstyle theme loader extension (escape hatch only)

The **primary** path is `fig.theme: !include ...@rules` (step 02 §2.4):
each figure carries its own cascade and the renderer applies it. No
hidden Python prelude. This is the recommended idiom for all user
code and all `jeanplot-plot` jobs.

A `load_plot_theme(*extras)` Python helper is kept as an *escape
hatch* for two cases: notebook sessions that want a global default,
and tests that want to set ambient `jstyle` once and render many
Figures. It is **not** the default path:

```python
def load_plot_theme(*extras: str):
    """Notebook/test escape hatch: set the ambient jstyle to the
    layered cascade. Production code should set `Figure.theme` instead."""
    import dracon as dr
    layers = [
        "pkg:jeanplot:resources/themes/default.yaml",
        "pkg:jeanplot:resources/themes/plots.yaml",
        *extras,
    ]
    loader = dr.DraconLoader(
        enable_interpolation=True,
        context=make_context_from_types(DEFAULT_TYPES + PLOT_TYPES),
    )
    cfg = loader.stack(*layers).construct()
    dr.resolve_all_lazy(cfg, except_for={"component"})
    jstyle.update(cfg["rules"])
```

`PLOT_TYPES` is the new tuple of plot-panel classes added in step 03,
registered in `jeanplot/__init__.py`. Note the `except_for={"component"}`
on `resolve_all_lazy` — leaves the live-scoped lazies (those
referencing `${component.X}`) un-resolved so they can bind
per-component at apply time.

### 4.9 Tests

- `tests/test_themes_plots_loads.py` — `dr.load("pkg:jeanplot:resources/themes/plots")`
  resolves cleanly, no missing types or unresolved interpolations.
- `tests/test_panel_defaults_from_theme.py` — `jstyle.apply(SmoothPanel2D())`,
  assert `vlims`, `cmap`, etc. are set from theme.
- `tests/test_autofig_data.py` — load `pkg:jeanplot:resources/figures/data.yaml`
  with a fixture `PlotData`, render, snapshot-compare PNG.
- `tests/test_compose.py` — `panel_row` and `panel_grid` produce the
  expected Container trees for a synthetic input.
- `tests/test_autopanel_dispatch.py` — for 1D / 2D / 3D fixtures,
  both `auto_panel(plot_data=d)` (Python helper) and
  `dr.loads("!AutoPanel ...", context=...)` (YAML tag) produce an
  instance of the expected `SmoothPanel{N}D` subclass. Same dispatch
  table on both sides.

## Verification

```bash
dracon show pkg:jeanplot:resources/themes/plots.yaml -c -r          # no warnings
dracon show pkg:jeanplot:resources/figures/pred_combined.yaml -c -r \
    ++ground_truth_data="@fixture" ++predicted_data="@fixture"
pytest jeanplot/tests/test_themes_plots_loads.py \
       jeanplot/tests/test_panel_defaults_from_theme.py \
       jeanplot/tests/test_autofig_data.py \
       jeanplot/tests/test_compose.py \
       jeanplot/tests/test_autopanel_dispatch.py -v

pytest jeanplot/tests/ -v   # full suite
```

## Out of scope

- Any change to biocomp / biocomp-tools / paper-jobs YAML files.
  They keep working unchanged. (Step 05 ships a migration cheatsheet
  for when a user voluntarily ports a script.)
- `jeanplot-plot` CLI (step 05).
- Parity tests against biocomp baselines (step 05).
- Re-implementing `expand_panel_atomics` / `subdivide` / `gap_mask` /
  `layout_dimensions` (§4.7 explains why — nested Containers replace
  the lot).

## Estimate

YAML added in jeanplot: ~180 LOC (`themes/plots.yaml`) + ~30
(`themes/bio.yaml`) + ~10 (`themes/paper.yaml`) + ~60 across the
figure templates. The biocomp `default_plotconf_v2.yaml` (~280 LOC)
is *not* touched — its jeanplot counterpart is a fresh write covering
the same configuration surface in jstyle-rule form, ~100 LOC smaller
because `_params` suffix boilerplate is gone.

Python added in jeanplot: ~150 LOC in `jeanplot/compose.py` (down
from the original plan's ~1000 LOC because `expand_panel_atomics`,
`compose_atomics`, `layout_dimensions`, etc. are not ported). The
biocomp original stays.

Net jeanplot diff after step 04: roughly **~+2200 LOC across steps
01-04** (down from the original plan's +6000 estimate). The big
savings come from:
- compose.py collapse (~-850 LOC vs the original plan's 1000-LOC port)
- task-template removal (~-150 LOC of YAML eliminated by AutoPanel)
- single-draw-method per panel (~-500 LOC across step 03's panel files)
- Figure-as-Container with no custom render() (~-150 LOC vs step 02 original)

Zero LOC removed from biocomp.
