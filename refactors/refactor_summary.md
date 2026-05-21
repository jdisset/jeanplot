# Refactor: Grow jeanplot into a self-contained plotting library

**Origin:** see `biocompiler/plot-unification.md` for the full audit and
the design rationale. This document is the actionable plan.

**Scope discipline (load-bearing).** This refactor does not touch
`biocomp/`, `biocomp-tools/`, or `paper-jobs/`. They keep working
exactly as today: `biocomp-plot` runs, every existing job YAML
resolves, every Python import that worked yesterday works tomorrow.
Jeanplot grows in parallel as a self-contained library. Migration of
specific plotting scripts to jeanplot happens later, one at a time,
on the user's schedule — step 05 ships the cheatsheet and parity
tests to support that.

**Dracon dependencies.** Dracon ships `!live`, `!cascade:NAME(arg)?`
(inherit + select modes), the built-in `strip_suffix:params`
parametric strategy, and `register_template`. The refactor uses
these directly; they are documented in the dracon skill, not
duplicated here.

**Scope:** make jeanplot the single home for all general-purpose
scientific-data visualization. Move plot functions, figure orchestration,
data shapes, KNN utilities, colour palettes, and the dracon plot
vocabulary out of `biocomp/biocomp/plotting/`, `biocomp/biocomp/plotutils.py`,
and `biocomp-tools/biocomptools/toollib/plot.py`. Keep biocomp-specific
domain bits (network adapters, NetworkPrediction-aware data holders,
fluorophore aliases) on the biocomp side, talking to jeanplot through
the same Component / jstyle interfaces every other plot uses.

---

## 1. The migration model

Jeanplot is a parallel library. Biocomp stays. **Migration is opt-in,
per-script, at the user's pace.** Concretely:

- biocomp-plot keeps running every job in `paper-jobs/`, `biocomp-jobs/`,
  every user notebook that imports `biocomp.plotutils` or
  `biocomp.plotting`. No deprecation warnings, no shim layer, no
  forced upgrade path.
- jeanplot ships its own `jeanplot-plot` CLI consuming its own
  YAML dialect (Component tree + `!cascade:jstyle` themes). For
  *new* plotting scripts that don't need biocomp-specific viz, write
  against jeanplot directly.
- When porting an existing script: follow the cheatsheet in step 05's
  `migrating_from_biocomp.md`. Network-aware panels
  (`NetworkDiagramPanel` etc.) can keep being imported from
  biocomp-tools and dropped into a jeanplot Figure — hybrid mode is
  fully supported.

No code is deleted from biocomp by this refactor. None. The eventual
cleanup of biocomp originals (if it ever happens) is a separate,
optional, opportunistic project that runs after all consumers have
migrated voluntarily.

## 2. The unifying idea

**Three primitives. N Component subclasses. Everything else emerges.**

1. **`Component`** — a thing that can be placed, sized, styled, and
   rendered. The existing jeanplot scene graph already provides this,
   including `Container` for nested layout, `LayoutConstraints` for
   row/column/flex behaviour, and the `MatplotlibRenderer` /
   `SVGRenderer` backends.
2. **`jstyle`** — selector-keyed property cascade. Step 00 rewrites
   the engine as a thin layer over dracon's `!cascade:jstyle`
   select-mode dialect.
3. **`Rescaler`** — a tiny protocol with `fwd(x)` and `inv(x)`. Today
   `biocomp.datautils.DataRescaler` already satisfies it. Lifting the
   protocol into jeanplot decouples plot functions from biocomp
   without any code change on biocomp's side.

Then everything else collapses. The mapping:

| Old concept | Replaced by |
|---|---|
| `PlotConfig` (rc_context + callstack_params + rescaler) | `Figure(Container)` attrs + jstyle rules |
| `PlotTask` (plot_method + overlays + ax) | a `PlotPanel(Container)` Component with overlay children |
| `Figure` orchestrator class in `toollib/plot.py` | a 5-field `Figure(Container)` Component; the existing `jeanplot.render()` is the lifecycle |
| `FigureSpec`, `FigAx`, `subdivide`, `SimpleLayout`, `GridLayout`, `MultiRowGridLayout`, `MergeSpec` | nested `Container` + `LayoutConstraints` (+ flex weights). 3D-panel cube + slice grid = a `Container` of children, not `subdivide` |
| `Overlay` protocol + `apply(ax, plot_data, plot_config)` | child Components with `is_overlay=True`; uniform mechanism for `Colorbar`, identity line, slice chord, density contour |
| `@configurable` + `_CONFIGURABLE_FUNCTIONS` namespace + `_params` suffix walk + `generate_full_nested_config` | jstyle rules on `PlotPanel` subclasses |
| `default_plotconf_v2.yaml` `callstack_params` tree | one `themes/plots.yaml` (`!cascade:jstyle` rules) |
| `tasks/{auto,1D,2D,3D}.yaml` dim-dispatch YAMLs | one `!AutoPanel` `!fn` template (dracon "Constructor Slots" pattern: `!$(table[dim])`) + a 10-line `auto_panel()` Python helper for code use |
| `expand_panel_atomics`, `compose_rows`, `compose_atomics`, `layout_dimensions`, `gap_mask`, axnum-indirected `FigAx.subdivide` (~1000 LOC) | **nested Container trees.** Rows = `Container(direction=row)`. Multi-row = `Container(direction=column)`. Gap columns = empty fixed-width Components. Per-kind widths = `min_dimensions.width` / `main_axis_weights`. `~150 LOC` of tree-construction helpers replace the lot. |
| `TXT_PLOT_FUNC_MAP` string → function map | `PlotPanel.render_txt()` polymorphism |
| Three `render_{mpl,svg,txt}` methods per panel | one `draw(ax)` + optional `render_txt()`. SVG falls back to mpl → SVG export via existing svg machinery. |
| panel-kind registry (`register_panel_kind`) | a `PlotPanel` subclass *is* the registration |
| `PartialFunction` | dracon `CallableSymbol` for the rare remaining wrapped-callable need |
| `Rescaler` confined to biocomp | one Protocol used by axis transforms, colour transforms, unit transforms |

The mental model collapses from ~15 classes / protocols / decorators /
registries to **three primitives + N Component subclasses**. The
compose engine, the task templates, the figure orchestrator, the
axes-allocation pipeline — *gone, replaced by tree shape*.

---

## 3. New capabilities unlocked

Things you cannot easily do today, that fall out of the unification for free:

1. **Theme-level plot configuration.** Override `SmoothPanel2D.vlims`,
   `Colorbar.tick_props.labelsize`, etc. in a paper theme. Selector
   specificity gives per-panel / per-class / per-id overrides without
   restating the rest of the config.

2. **Composable plot panels in any Container.** A `SmoothPanel2D` can sit
   inside a `Container` next to a `Text` and a `Connection` — no
   special "figure" mode. You can drop plot panels into a circuit
   schematic or a flowchart and they lay out and style consistently.

3. **One rendering pipeline for everything.** `jeanplot.render(component)`
   handles scene graphs, plot figures, and mixed trees the same way.
   No `Figure.render()` orchestrator separate from `MatplotlibRenderer.render_component`.

4. **3D cube + slice grid is just tree shape.** `SmoothPanel3D` is a
   `Container` holding a cube view + a child `Container` of slice
   panels. No `FigAx.subdivide`, no `axnum`, no `subax_spec`. The
   layout falls out of the existing measure/layout cascade.

5. **Live theme reload.** `JStyle.update(...)` swaps the plot
   defaults at runtime without rebuilding the figure. Enables A/B
   theming and notebook-driven design exploration.

6. **CLI-discoverable plot knobs.** Each `!set_default` with a mapping
   body in a theme file surfaces as a `--name` CLI flag (dracon's
   vocab-as-CLI). `jeanplot-plot +mytheme --vlim-low -1 --vlim-high 1`
   works without writing any new argparse.

7. **Theme introspection.** `jstyle._cascade.invoke(component=...)`
   returns the merged rule set for any component. Documentation
   generation for free.

8. **One overlay model.** Colorbar, slice chords, diagonal paths,
   density contours, identity lines on MVP — all are children of the
   panel they overlay. They get layout, styling, and
   connection-attachment semantics from the existing Component machinery.

9. **Trivial extension.** Adding a new plot kind = one Python class +
   one jstyle rule block. No decorator, no defaults pack, no umbrella
   YAML edit, no registry call.

10. **Biocomp-specific panels are first-class.** `NetworkDiagramPanel`,
    `CircuitPanel`, `MVPNetworkPanel` are jeanplot Components that live
    in biocomp-tools but participate in every jeanplot mechanism
    (jstyle, layout, render dispatch, autofig dim-dispatch, overlays).

11. **Pixel-stable migration.** Each plot function's matplotlib drawing
    code is unchanged — Component shells wrap them. SVG-hash and
    PNG-pixel tests confirm zero rendering drift.

---

## 4. Codestyle reminders

The biocompiler `CLAUDE.md` enumerates these in detail. The ones that
bite hardest on this refactor:

- **SSOT.** One canonical definition per concept. Defaults live in
  `themes/plots.yaml`, not also in Python signatures, not also in
  `callstack_params`, not also in per-function packs. Plot functions
  live in `jeanplot.panels.<kind>`, not also re-exported from
  `biocomp.plotting`. Import shims are *thin* — they re-export, they
  don't reimplement.

- **DRY.** If two panels share an axis-formatting helper, lift it. If a
  panel and an overlay share a KNN kernel, both call the one in
  `jeanplot.knn`. No "local copy" of `colorbar()` in `mvp.py`.

- **Composability over orchestration.** A `Figure` does not know what
  panels are inside it; the renderer walks `children` and asks each
  one to render. Adding a new panel kind does not touch any orchestrator
  because there is no orchestrator.

- **No premature abstraction.** `PlotPanel` exists because every plot
  kind needs the same base behaviour (claim an axes from its bbox,
  optionally render-text). If a panel kind only ever has one
  implementation, do not invent a registry for it.

- **No `from __future__ import annotations` in new code.** Per CLAUDE.md
  — breaks Pydantic introspection. New panel modules use `X | None`
  syntax directly. Existing core files that use it stay as-is for now
  (touching them is out of scope for this refactor).

- **Types are the first line of defense.** Each `PlotPanel` subclass
  declares its kwargs as typed Pydantic fields. No `**kwargs` opacity.

- **Assert shape invariants at function boundaries.** When a panel
  pulls `(X, Y)` arrays from a `PlotData`, assert shape and finite-ness
  loudly, not silently.

- **No comments unless absolutely necessary.** Per CLAUDE.md. Renaming
  for clarity beats explaining.

- **The diff sign is conscious.** Step 00 is net-negative on disk by
  design (~-100 LOC in the style engine). The plot refactor (steps
  01-05) is net-positive (~+3000-3500 LOC of new panel code), but the
  whole-file delta is far smaller than the original 5500-LOC estimate
  because the compose engine, task templates, and figure orchestrator
  collapse. Inside jeanplot itself, prefer the smaller solution; but
  don't sacrifice the "no biocomp imports anywhere in jeanplot"
  invariant to save LOC.

---

## 5. Sequencing rationale

Six steps. Step 00 is the orthogonal engine refactor; steps 01-05 are
the plot refactor proper. Each is green-on-tests before the next.

```
00 jstyle cascade        ── style_engine.py thinned to ~120 LOC via
                            dracon's !cascade:jstyle. Self-contained;
                            no plot panels touched. Tests cover
                            existing gene/ schematic.

01 foundations           ── Rescaler protocol, PlotData, KNN, colour.
                            Copies of biocomp's general-purpose
                            building blocks; nothing depends on this
                            not existing.

02 panel base + figure   ── PlotPanel(Container) with draw(ax) +
                            optional render_txt(). Figure is a 5-field
                            Container (no custom render()). Renderer
                            knows how to allocate sub-axes per leaf
                            PlotPanel from its laid-out bbox.

03 panel implementations ── one Component subclass per drawing
                            function, plus AutoPanel for dim-dispatch.
                            Heavy but mechanical.

04 themes + compose      ── themes/plots.yaml (SSOT for defaults via
                            !cascade:jstyle), autofig figure
                            templates, slim compose.py (~150 LOC of
                            tree-construction helpers).

05 validation +          ── jeanplot-plot CLI, parity test harness,
   migration guide         migration cheatsheet documenting the
                           biocomp → jeanplot mapping for when a user
                           ports a specific script. Depends on all
                           of the above.
```

**Why step 00 first.** The cascade rewrite is purely about how rules
parse, match, and merge. It doesn't depend on plot panels. Sequencing
it first means steps 01-05 consume a working dialect without bundling
engine risk. It's also the only step in the refactor that lands as a
net-negative diff inside jeanplot.

**Why this order across 01-05?** Themes target panel classes by name
— they're useless until the classes exist. `Figure(Container)` is
trivial; pinning the panel API first means we know what the renderer
has to do when it encounters a `PlotPanel`.

**Every step leaves biocomp / biocomp-tools / paper-jobs untouched by
construction.** Each step lives entirely inside `jeanplot/` (and in
step 05's case, also adds a few tests and doc pages).

---

## 6. The six steps

0. **`00_jstyle_cascade.md`** — Style engine becomes a thin layer over
   `!cascade:jstyle`. `style_engine.py` 460 → ~120 LOC. New tiny
   `style_dialect.py` (~40 LOC). Existing scene-graph theme rewritten
   in the new dialect. All gene/ tests green.

1. **`01_foundations.md`** — `Rescaler` protocol, `PlotData` (and its
   `LazyPlotData` variant) as plain Pydantic models, `PlotFunctionResult`,
   `GridData`, KNN utilities, colour palettes and matching.

2. **`02_panel_base_and_figure.md`** — `PlotPanel(Container)` base
   class with `draw(ax)` + optional `render_txt() -> str | None`;
   `Figure(Container)` as 5-field Container; `MatplotlibRenderer`
   extended to allocate one sub-axes per leaf `PlotPanel` from its
   laid-out bbox; `Colorbar` and other overlays as uniform
   `is_overlay=True` children.

3. **`03_panel_implementations.md`** — one Component subclass per
   existing drawing function: `SmoothPanel1D`, `SmoothPanel2D`,
   `SmoothPanel3D` (a Container with cube + slice-grid children),
   `MVPPanel`, `DensityPanel`, `ScatterPanel`, `ViolinPanel`,
   `ParticlePanel`, `StackedPolyPanel`, plus `AsciiHeatmapPanel`.
   `AutoPanel` dispatches by data dim. The matplotlib drawing code
   is copied unchanged.

4. **`04_themes_and_compose.md`** — `jeanplot/resources/themes/plots.yaml`
   carries every plot default as `!cascade:jstyle` rules; the autofig
   figure templates (`figures/{data,pred_combined,combined}.yaml`)
   are jeanplot-native Component trees; `jeanplot/compose.py` carries
   ~150 LOC of tree-construction helpers (build a row of panels,
   build cube-with-slices, build figure metadata) — no
   `expand_panel_atomics`, no `compose_atomics`, no `subdivide`.

5. **`05_validation_and_migration.md`** — `jeanplot-plot` CLI entry
   point, parity test harness, migration cheatsheet.

---

## 7. Reading order for the implementer

- Start with this summary.
- Read `00_jstyle_cascade.md` and execute. Verify gene/ tests pass.
- Read `01_foundations.md` and execute. Verify tests pass.
- Read `02_panel_base_and_figure.md` with §5 of
  `biocompiler/plot-unification.md` open in another window — the
  design rationale is dense and the step doc is intentionally lean.
- The remaining steps are increasingly mechanical; the design
  questions all live in 00, 01, and 02.

If a step looks ambiguous, prefer the simpler choice that fits SSOT.
You can stop after any step without breaking biocomp — it isn't
touched by any of them.

---

## 8. What this refactor does *not* do

- It does not modify anything in `biocomp/`, `biocomp-tools/`, or
  `paper-jobs/`. Every existing import, CLI invocation, and YAML
  reference continues to work exactly as today.
- It does not provide automated migration tooling. The cheatsheet
  in step 05 is the migration path; users port scripts at their own
  pace, one at a time.
- It does not change matplotlib drawing logic. Each plot function is
  copied verbatim into its Component shell. Pixel parity (against
  biocomp-plot baselines) is the acceptance criterion.
- It does not touch jeanplot's `gene/` package — the genetic
  circuit primitives are already in good shape.
- It does not touch the existing jeanplot `core/` apart from
  step 00's `style_engine.py` rewrite + step 02's `LayoutConstraints`
  flex-weight extension. No other `core/` edits.
- It does not ship biocomp-aware panel classes inside jeanplot.
  None. Jeanplot has zero biocomp imports. If a user's migrating
  script needs `NetworkDiagramPanel` or `CircuitPanel`, those still
  live in biocomp-tools and can be used alongside jeanplot panels
  in the same Figure.

---

## 9. Acceptance criteria for the whole refactor

- `pytest jeanplot/tests` green, including the new parity suite.
- `biocomp-plot +paper-jobs/plot/figures/autofig_pred_combined.yaml ...`
  runs and produces identical output to its pre-refactor baseline
  (biocomp wasn't touched, so this is essentially a regression
  guard on environmental drift, not on this refactor).
- `jeanplot-plot +jeanplot/tests/parity/jeanplot_jobs/<fixture>.yaml`
  produces output within pixel-tolerance of the corresponding
  biocomp-plot baseline for every fixture in the parity suite.
- `dracon show pkg:jeanplot:resources/themes/plots.yaml -c -r`
  resolves cleanly with no warnings.
- `! grep -rn "from biocomp\|import biocomp" jeanplot/jeanplot/` —
  the CI gate that jeanplot has no biocomp imports anywhere.
- Net jeanplot LOC delta:
  - Step 00: **~-100 LOC** (style engine thinned).
  - Steps 01-05: **~+3000-3500 LOC** of new panel code, themes,
    compose helpers, CLI, tests.
  - Compared to the pre-collapse estimate of +5500-6000, the
    compose engine collapse (~-850 LOC), task-template collapse
    (~-150 LOC), and single-draw-method-per-panel (~-500 LOC) buy
    back ~1500 LOC of duplicated machinery.
  - Net biocomp / biocomp-tools LOC delta: **0** — by construction,
    no changes.

The win is not lines-of-code; it's:
- A clean general-purpose plotting library now exists.
- Future scripts that don't need network-aware viz can be written
  entirely against jeanplot, with no biocomp coupling.
- biocomp / biocomp-tools / paper-jobs / all existing scripts keep
  working unchanged — zero migration pressure.
- When a user does want to migrate a script, the cheatsheet makes it
  mechanical and the parity tests confirm output equivalence.
- The mental model is one diagram: a Component tree, traversed by a
  renderer, styled by a cascade. Plot panels are Components that
  happen to claim a matplotlib axes.

---

## 10. Dracon idioms in use

The plan leans on these dracon features deliberately. If you find
yourself reaching for a parallel Python machine for one of these,
stop and use the dracon-native form instead.

| Idiom | Where | Why |
|---|---|---|
| `!cascade:NAME` (select-mode) | step 00 — jstyle engine | Selector matching + specificity + merge handled by dracon. ~340 LOC of engine code disappears. |
| `!live component:` (auto-opened by select-mode) | `themes/bio.yaml` (step 04 §4.3) | `${component.part_name}` resolves per-component at apply time. Replaces the hand-rolled `!each` over 26 fluorophore rules — one rule does what 26 did. |
| `register_template(fn)` | `jeanplot/compose.py` (step 04 §4.7) | `panel_row`, `panel_grid`, `panels_from_datas` register once and are usable as `!panel_row { ... }` tags. No parallel registry; signature IS the tag interface. |
| `!fn` template + `!$(table[key])` (Constructor Slots pattern) | `templates/auto_panel.yaml` (step 03 §3.3a) | Dim-dispatch is one `!define` lookup + dynamic tag. Replaces a brittle Pydantic `model_post_init` that did `model_dump` + reparse. |
| `!fn` template figures (Higher-Order Config) | `figures/templates.yaml` (step 04 §4.6) | Repeated figure shapes (`ComparePair`, `Triple`, etc.) lift to one-line invocations. Adding a shape = one `!fn` block, no Python. |
| `!include path@selector` | `Figure.theme: !include themes/plots.yaml@rules` (step 02 §2.4, step 04 §4.6) | Pulls just the cascade subtree out of a theme file. Figure carries its own theme — no hidden `load_plot_theme()` prelude. |
| `<<(<): !include` (export-propagating layer merge) | user entry YAMLs | Layered theme presets (`default` → `bio` → `plots` → `paper`). Each layer's `!define`s and tag bindings propagate to the next. |
| `!require` with mapping body (Vocab-as-CLI) | `themes/plots.yaml` `!set_default xlims: { default: ..., help: ..., short: -x }` | Top-level theme vars surface as `--xlims` CLI flags on `jeanplot-plot +mytheme` without writing any argparse. |
| `!cascade:jstyle` document round-trip via `CallableSymbol` | `fig.theme: Any` typed field | The cascade is a real Python value (a `CallableSymbol` of kind 'match'), passable across functions, callable as `cascade.invoke(component=c)`. Same `dump`/`loads` symmetry as any dracon value. |
| `DEFAULT_TYPES.extend([...])` in `jeanplot/__init__.py` | step 01, step 03 | Every new Pydantic Component / data class is added so dracon's `!Foo { ... }` tag works automatically. No per-class registration. |
| `model_validator(mode='after')` lazy fields on data holders | (kept from existing pattern) | Plot-data holders that load arrays on first access are plain Pydantic with `__model_post_init__` doing the heavy lift; nothing dracon-specific needed. |

**Anti-patterns we explicitly avoid:**

- **No `${func(a, b, ...)}` for invocation when a `!func { a: ..., b: ... }` tag works.** Tag form keeps the body as YAML; `${...}` flattens kwargs into one line.
- **No multi-line `${[... for ... in ...]}` blocks.** When the comprehension is more than one line, restructure with `!each` + `!if`, or push to a registered Python helper called via tag form.
- **No `<<: ${some_dict}`.** Merge keys want compose-time content (`!include`, anchor, inline mapping). Interpolating a Python dict into a merge key downgrades dracon's deep-merge to `dict.update`-style overlay and loses anchors + provenance.
- **No taking-a-path-when-we-mean-an-object.** Programs declare `figure: Figure`, `dataset: NetworkSet`, `theme: Any` — the YAML composes via `!include`. We never accept `figure_file: str` and open a private `DraconLoader` inside `run()`.

These are dracon-skill style rules §1-§10. Following them mechanically gives the YAML configs in the plan the brevity they show.
