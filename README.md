# jeanplot

A 2D plotting library that thinks the way matplotlib should have. Build a tree of components, let layout figure out where they go, style them with CSS-ish rules, render to matplotlib or SVG. Comes with a gene-schematic module because that's what I needed it for, and a scientific-plot module on top because I needed that too.

```bash
pip install -e .
```

That's it. No conda, no extras, no platform shenanigans. Bundled fonts ship in the wheel.

## What you actually do with it

Three things, in order of how often you'll touch them:

1. **Build a tree.** A `Container` holds children. Children can be `Container`, `Text`, `Connection`, gene parts, plot panels, tables — anything that subclasses `Component`. Each child has a `min_dimensions`, an optional `offset`, and a `layout` that says how its own children are arranged (row, column, gap, align). It's flexbox-ish.
2. **Render it.** `from jeanplot import render; render(root)`. Matplotlib by default, SVG if you ask for it. You get back an axes or a string.
3. **Style it.** `jstyle` is the global style engine. Selectors look like `Container[style_class=card]`. Properties look like `style.background_color`. You either feed it a dict or a `!cascade:jstyle` YAML document, your call.

Hello world:

```python
from jeanplot import Container, Text, Size, LayoutConstraints, render

root = Container(
    id="root",
    min_dimensions=Size(300, 80),
    layout=LayoutConstraints(direction="row", gap=10, align_items="center"),
    children=[
        Container(id="left", min_dimensions=Size(80, 40)),
        Text(text="hi", font_size=14),
    ],
)
render(root, output="hello.png")
```

`example/hello_jeanplot.py` is a slightly nicer version with colors.

## Dracon: the config engine

jeanplot leans hard on [dracon](../dracon) for YAML. Every component is a Pydantic model, every model is a dracon tag, every CLI flag comes from a `!set_default` in a YAML file. Worth knowing the basics:

- **Tag form is the default invocation.** `!Figure { panels: [...] }` constructs a `Figure`. `!SmoothPanel2D { data: ... }` constructs a panel. Mapping bodies are kwargs.
- **`!include`** pulls in another file: `!include pkg:jeanplot:resources/themes/plots.yaml`, `!include file:./data.json`, `!include path@subkey` to extract a subtree.
- **`<<:` / `<<{+<}:`** merge mappings (deep-merge for the second form). Use these for overrides, not `${dict}`.
- **`${...}`** interpolates values: arithmetic, references via `${@/path}`, lazy resolution.
- **`!set_default`** declares a variable with a default; mapping body (`{default: X, help: "...", short: -X}`) makes it a CLI flag.
- **`!fn`** is a parametric template. `!cascade:jstyle` is the predicate-keyed style mapping dialect.

If you've never seen dracon: scroll its README first. The rest of this doc assumes you can read a YAML config and recognize tags.

## The tree

Everything is a `Component`. The interesting subclasses live in `jeanplot/core/`:

- `Container` — children + `layout` + optional `style` (`BoxStyle`: background, border, corner radius, padding, shadow). 90% of what you'll instantiate.
- `Text` — string with font size, weight, color, optional `TextHalo`. Two sizing modes: `data` (scales with the world) and `points` (fixed visual size). Use `points` for labels that should stay legible at any zoom.
- `Connection` — line between two components by id (`start_component="a", end_component="b"`). Curves: `StraightCurve`, `OrthogonalCurve` (right angles + rounded corners), `SimpleBezierCurve`. Endpoints: `LineEndFlat`, `LineEndCircle`, `LineEndArrow`. Endpoint position defaults to the component centre (`Offset(reference_relative=(0.5, 0.5))`); override with `start_offset` / `end_offset`. If both endpoints have `AnchorComponent` children (gene parts do), `auto_route=True` picks the best anchor pair automatically — set `auto_route=False` to use raw centres.
- `ConnectionLabel` — text attached to a `Connection` at a fractional position along its path.
- `Table` — `TableRow`s of `TableCell`s, column widths in `%` or pixels. Borders are **one collapsed grid** (`GridStyle` on the Table: `frame` / `header` / `inner` line roles + `corner_radius` / `header_fill`), drawn once per line — cells/rows carry no grid borders, so shared edges are never doubled.
- `SVGElement` — drop arbitrary SVG into the tree for things matplotlib doesn't have a primitive for.
- `Overlay` / `AnchorComponent` — base classes for things that read from a parent (overlays for panels, anchors for connections).

Components have an `id`, optional `style_class` (list of strings — CSS-class-ish), and a parent that's set when you add them as children. The parent link is what makes `jstyle` descendant selectors work.

## Layout

`LayoutConstraints` is the boring part you can mostly ignore. Fields that matter:

- `direction`: `row` or `column` (alias: `col`).
- `gap`: pixels between children.
- `align_items`: `start`, `center`, `end`, `stretch`.
- `justify_content`: same vocabulary.
- `main_axis_weights` / `cross_axis_weights`: optional float list for flex-style distribution.

Long-form mapping or string DSL — either works:

```python
Container(layout="row gap=8 align=center justify=center", children=[...])
```

```yaml
layout: "row gap=1.0 align=stretch"
```

`align` aliases `align_items`, `justify` aliases `justify_content`. Mapping and string are interchangeable everywhere.

Children can be positional: `Container(panel1, panel2, ...)` is sugar for `children=[...]`. Works in YAML too: `!Container [a, b]` is shorthand for `!Container { children: [a, b] }`.

Nest containers to get grids — there's no grid primitive because you don't need one. Absolute positioning: pass `offset=Offset(absolute=(x, y))` on a child and layout leaves it alone.

## jstyle

`jstyle` is CSS minus the inheritance soup. You write rules keyed by selectors; the engine matches them against components during layout.

```python
from jeanplot import jstyle

jstyle.update({
    "Container[style_class=card]": {
        "style.background_color": "#ffffff",
        "style.corner_radius": 8,
        "style.padding": (10, 10, 10, 10),
    },
    "[id=header] Text": {
        "color": "#102a43",
        "font_weight": "bold",
    },
})
```

Selectors support:
- type name (`Container`, inherits — a `Component` rule applies to `Container`)
- id (`[id=foo]`)
- class (`[style_class=card]`)
- attribute (`[attr=value]`, `[attr~=value]`, `[attr*=value]`, `[attr^=value]`, `[attr$=value]`, `[attr=/regex/]`, numeric `[attr>0]`, presence `[attr]` / absence `[!attr]`)
- combinators: descendant (space, like CSS), child (`>`), sibling (`~`)

Specificity is the standard `(ids, classes, types)` tuple, ordered `id > class/attr > type > *`. More specific wins; newer wins ties.

The selector engine is the shared **dracon locator** (`dracon.parse_locator`), so this is exactly the grammar behind dracon's `!ref` and `attached_to` (which now takes a locator string, resolved relative to the component). One grammar for styling, references, and attachment; jeanplot keeps only a ~40-line `ComponentTreeAdapter`. See `docs/STYLE_GUIDE.md` and dracon's `reference/locators.md`.

`jstyle.update(...)` defaults to **fill** semantics — cascade values fill fields the user didn't explicitly set on the component, explicit values always win. Every `Component` snapshots `model_fields_set` at construction (`_user_set_fields`), and the cascade walks around those. For the old clobber-everything behaviour, use `!cascade:jstyle` instead of `!cascade:jstyle_fill` (which is the default). `jstyle.clear()` for a true reset.

### Nested rules

Rules nested inside another rule's mapping apply to descendants — same shape as CSS descendant selectors, but indented:

```python
jstyle.update({
    "Container[id=sidebar]": {
        "style.background_color": "#eee",
        "Text": {                       # Text inside #sidebar only
            "color": "black",
            "font_size": 9,
        },
        "Button[style_class=primary]": {  # Button.primary inside #sidebar
            "style.background_color": "blue",
        },
    },
})
```

This flattens internally to descendant selectors (`Container[id=sidebar] Text`, etc.) — same as writing them flat, just nicer to read when rules cluster.

### Partial updates

Nested Pydantic models can be updated partially — drop just the keys you want to change:

```python
jstyle.update({
    "Container[id=main]": {
        "style": {                       # partial BoxStyle update
            "background_color": "lightblue",
            "padding": [20, None, 20, None],  # only top/bottom; None preserves
        },
    },
})
```

Lists and tuples take `None` at each index to mean "leave that one alone".

### Scoped overrides

`jstyle` is a context manager. Use it to apply temporary rules in tests, notebooks, or one-off blocks — the old cascade restores on exit:

```python
with jstyle({"Text": {"color": "red"}}):
    render(root, output="red_text.png")
# back to whatever was set before
```

`jstyle(...)` is sugar for `jstyle.context(...)`. Both work.

YAML form:

```yaml
rules: !cascade:jstyle_fill
  Container[style_class=card]:
    style.background_color: "#ffffff"
    style.corner_radius: 8
  Text:
    font_size: 12
```

Bootstrap the default theme with `load_default_theme()` (reads `pkg:jeanplot:resources/themes/default.yaml`), or load plot defaults on top with `load_plot_theme(*extra_files)`. In production code, prefer setting `Figure.theme` so the theme travels with the figure.

For deeper docs on selectors and the cascade engine, see `docs/STYLE_GUIDE.md`.

## Everything is a selectable primitive (the philosophy)

This is the one idea the whole library is built on. Read it before adding any
"configurable" feature.

**A figure is a tree of small, composable primitives, and every primitive's
*parameters* — not just its colors — are addressable, layerable, and SSOT through
the cascade.** You select by type / `style_class` / `id`, you override by layering a
delta, and the cascade deep-merges across specificity. Aim to make as much as possible
reachable this way, as deeply and freely as possible.

Concrete rules:

1. **Style and parameters are the same thing: cascade data.** A panel's *parameters* —
   smoothing, axis lims, heatmap options — live in the cascade keyed by selector, the same
   as borders and colors. Do **not** invent a flat `!set_default` / `build(**kwargs)`
   surface for something the cascade can own. One base rule + nested specializations beats
   N parallel knobs. A shared config dict becomes its **own selectable type** — a
   `CascadeLeaf` (see rule 3) — so even nested parameters are addressable.

2. **Deep-merge across specificity is the layering primitive.** A more-specific rule
   carrying a *partial* dict merges onto the base, keeping every sibling key:

   ```yaml
   SmoothKernel: { min_points: 5, radius: 0.1 }   # the splat kernel, shared everywhere
   SmoothGrid:   { grid_resolution: 250 }          # grid sampling, every 2D surface
   SmoothPanel3D:                                   # nested = descendant
     SmoothGrid: { grid_resolution: 180 }          # inherits the kernel, overrides only res
   ```

   Prefer **nested** selectors (`Outer:\n  Inner:`) over flat space-separated
   (`Outer Inner:`) — they compose with dracon's merge/override on the nested structure.

3. **A shared config dict becomes its own selectable primitive — `CascadeLeaf`.** When a
   parameter bag (smoothing, a heatmap style) is buried as a dict field, the cascade can
   only reach it *per-panel* and can't share one definition across peer panel types. Promote
   it: make it a `CascadeLeaf` subclass (class-name + parent chain + user-set tracking, no
   layout weight). Then a **bare `LeafType:` rule is the cross-peer SSOT**, and
   `Panel LeafType:` specializes per context. The smoothing config is the worked example:
   `SmoothGrid` (grid) nests a `SmoothKernel` (per-point kernel), and a bare `SmoothKernel:`
   rule reaches every 2D grid *and* `SmoothPanel1D` at once — something no panel selector
   could express. Three small pieces wire it:
   - `panel_from(fn, cascade_leaf_params={"smooth_grid_params": SmoothGrid})` replaces the
     dict field with a typed leaf and bridges `leaf.params` back into the dict-driven fn.
   - `JStyle.apply_one` recurses into any `CascadeLeaf`-typed field (sets `.parent`, applies
     the cascade), so leaves resolve wherever the panel does — including a procedurally-drawn
     `CubeStackPanel` face (`smooth_3d.py:_face_smooth_grid_params`).
   - the leaf's `.params` flattens itself (and nested leaves) back to the dict the plot fn
     wants — the fn stays dict-driven; only the cascade-facing surface is typed.

   **A field path the cascade can't address is a smell — make it a selectable leaf.** A
   `!define` SSOT var is only the right tool when the consumers are *peer* selectors with
   mismatched slots; once the value is a leaf, a bare type rule replaces the var.

4. **Per-figure deltas → the `Figure.theme_overrides` attribute**, deep-merged onto the
   base theme at render (`merge_jstyle_rules`). Never a sentinel `!define` that a distant
   file reaches back for. (See `Figure.theme` / `theme_overrides`.)

The test of an idiomatic addition: can a downstream figure retune it by adding **one
selector rule**, with no new knob, no new field path, no Python? If not, push the config
into the cascade and expose the primitive.

## Gene schematics

`jeanplot/gene/` is the original reason this library exists. It draws genetic circuits: promoters, terminators, ERNs (with optional 5p recognition sites), fluorophore markers, uORF groups, transcription units, sources (plasmids), and the interactions between them.

```python
from jeanplot.gene import GeneticSchematic, CircuitData

circuit = CircuitData(transcription_units=[...], sources=[...], interactions=[...])
schematic = GeneticSchematic.from_circuit(circuit)
render(schematic, output="circuit.svg")
```

Parts are SVG-backed (`Promoter`, `Terminator`, `ERN`, `ERN5pRecog`, `FluoMarker`, `UorfGroup`); raw assets live in `resources/parts/`. They lay out left-to-right inside a `TranscriptionUnit` row. Sources sit on the side. Interactions become `Connection`s with arrow heads.

The grammar of "what valid circuit shapes look like" lives in `jeanplot/gene/data.py` as plain Pydantic models — `CircuitData`, `TUData`, `PartData`, `SourceData`, `InteractionData`. The visual layer just reads those.

## Scientific plots (the panels layer)

This part is newer. It exists because matplotlib's API is fine but everyone reinvents the same wrapper to coordinate figure size, dpi, multi-panel layout, colorbars, themes, and IO. So jeanplot has a thin wrapper that uses the same Container tree.

```python
import numpy as np
from jeanplot import Figure, SmoothPanel2D, PlotData, render

x = np.random.randn(500)
y = np.random.randn(500)
z = np.exp(-(x**2 + y**2))

data = PlotData(
    xval=np.column_stack([x, y]),  # (n_samples, n_inputs)
    yval=z[:, None],                # (n_samples, n_outputs)
    input_names=["x", "y"],
    output_name="z",
)
fig = Figure(panels=[SmoothPanel2D(data=data)])
render(fig, output="smooth.png")
```

`Figure` is a `Container` with figure-level fields (`output_dir`, `output_file`, `dpi`, `rc_context`, `theme`, `metadata`, `subtitle`, `svg_id_prefix`). It doesn't orchestrate anything — the renderer walks the tree, sees `PlotPanel` leaves, and allocates a real matplotlib axes for each one. Layout decides where the axes go.

### Panels

`jeanplot.panels` ships:

- **`SmoothPanel1D`** — KNN-smoothed 1D curve with optional std band, legend.
- **`SmoothPanel2D`** — KNN-smoothed 2D surface, heatmap + optional contours.
- **`SmoothGradMagnitudePanel2D`** — gradient magnitude of a 2D smooth.
- **`GradientFieldPanel2D`** — quiver of the 2D gradient.
- **`SmoothPanel3D`** + **`CubeView`** + **`CubeStackPanel`** — 3D surfaces. Cube view (rotatable wireframe with face heatmaps) plus a grid of 2D slices.
- **`DataBlockPanel`** — a **uniform-face block** that dispatches on data dim, so a row of mixed-dimension circuits reads consistently: 1D → smooth curve (+ 2D density histogram); 2D → value heatmap (+ gradient-magnitude heatmap with the gradient field); 3D → data cube + R×C z-slice grid. Sub-plots' **plotted axes boxes are pinned** so tick labels + titles render at one physical size: the block is **always one row tall**, so a single-row `[1,N]` z-slice strip gives full-size slices identical to a 2D value face, while a grid `[R,C]` packs into that same height (each slice `~h/R` tall). The cube is full-height, `cube_cell_units` face-widths wide. The block **derives its own width** from the face count (rigid `min == max`, so it never overflows) — the legacy `aspect` field is vestigial. Face chrome (`FACE_PAD` / `FACE_COLORBAR`) + the `pin_axes_box` / `pin_to_cell` / `face_cell_size` helpers live in `panels/smooth_3d.py`, shared with the slice grid (`SmoothPanel3D.uniform_height`, which self-pins to its derived footprint). Sub-view look is the cascade keyed under `DataBlockPanel`. `data_block(plot_data, …)` is the code-side twin (mirrors `auto_panel`).
- **`MVPPanel`** — measured-vs-predicted scatter with an identity line.
- **`DensityPanel1D`** — kernel density estimate.
- **`GridHistogramPanel`** / **`ScatterPanel3D`** — gridded histograms and 3D scatter.
- **`ViolinPanel`**, **`ParticlePanel`**, **`StackedPolyPanel`** — distribution viz.
- **`AsciiHeatmapPanel`** — terminal-friendly output (Kitty graphics if your terminal supports it, else `░▒▓█` characters).
- **`Colorbar`** — overlay child that reads `parent._mappable`. Drop it next to any panel that produces one.
- **`AutoPanel`** (via `auto_panel(...)`) — dispatches to the right Smooth* panel based on `data.X.shape[1]`.

Each panel is a typed Pydantic model with the parameters you'd pass to the equivalent matplotlib call as fields. No `**kwargs`. The fields you don't set come from the theme (cascade-fill).

The matplotlib drawing code itself lives in `jeanplot/plots/` as plain functions. Panels are thin shells that call them. If you want to draw without the panel system, just call the function.

### Overlays

Panels accept overlay children that draw on the same axes after the panel itself. `jeanplot.panels.overlays` ships:

- **`IdentityLineOverlay`** — y=x reference line. Default for MVP.
- **`DiagonalPathOverlay`** — a diagonal path through the 2D plot region.
- **`SliceOverlay`** / **`SliceChordOverlay`** — show a 2D slice plane / chord on a 3D plot.
- **`AdditionVsRemovalOverlay`** — directional decomposition viz.
- **`DensityContourOverlay`** — density contours over a scatter or smooth.

Roll your own by subclassing `Overlay` and implementing `draw(self, ax, parent)`.

### `@panel_from` — function = panel

For most panels you don't write the class by hand. Decorate the drawing function:

```python
from jeanplot.panels import panel_from

@panel_from
def my_plot(plot_data, *, ax, vlims=(0.0, 1.0), cmap="viridis"):
    ax.imshow(plot_data.X, vmin=vlims[0], vmax=vlims[1], cmap=cmap)
```

The decorator introspects the signature and synthesises a `PlotPanel` subclass with `vlims` and `cmap` as Pydantic fields, registers it as the YAML tag `!my_plot`, and keeps the original function callable as a plain function (REPL-friendly). Panels expose a cascade-fillable `axes_size` and a computed `min_dimensions` so the figure auto-sizes around its contents. A panel that draws **outside** its axes box (e.g. an out-of-axes colorbar at axes-fraction > 1) overrides `_right_overflow(self) -> float`; `PlotPanel.effective_padding` folds that into the right inset, so the layout reserves the space automatically — no hand-tuned `style.padding.right`. (`SmoothPanel2D` computes it from the colorbar band geometry + a `label_reserve` allowance; see `panels/smooth_2d.py`.)

**Plot data routing.** Four parameter names are special: `X`, `Y`, `input_names`, `output_name`. If your function declares any of them, they're not exposed as Panel fields — they're auto-wired from `self.plot_data.{x, y, input_names, output_name}` at draw time. So you write:

```python
@panel_from
def my_plot(X, Y, *, ax, cmap="viridis"):
    ax.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap=cmap)
```

…and the panel only exposes `cmap` (and inherits `plot_data`, `axes_size`, etc. from `PlotPanel`). Use `plot_data_keys=(...)` on the decorator to customise the routed set. `ax` and `self` are skipped. `*args` / `**kwargs` are rejected — name your params.

Parameter names that collide with `PlotPanel`'s own fields (`title`, `xlims`, `vlims`, ...) are *inherited* rather than re-declared; your function gets the panel's value at draw time.

## Themes

`pkg:jeanplot:resources/themes/plots.yaml` is the single source of truth for plot defaults. Selector-keyed, same dialect as `jstyle`:

```yaml
rules: !cascade:jstyle
  SmoothPanel2D:
    vlims: [0.0, 1.0]
    cmap: viridis
  Colorbar:
    tick_props:
      labelsize: 8
```

Override per-panel-class, per-instance (via id), or per-class (via `style_class`). Specificity does the right thing.

Themes that ship in `resources/themes/`:
- `defaults.yaml` — **the discovery surface + opt-in floor**: every selectable type's own
  stylable fields at their pydantic defaults, GENERATED by `python -m jeanplot.style_schema`
  (no args dumps the whole floor; pass `Type …` to inspect specific selectors). This is how
  you see *what you can override on any type and its default*. It's includable as a base
  (`rules: !cascade:jstyle_fill` → `<<: !include …/defaults.yaml@_body` → your overrides), but
  it is **not** auto-wired under the curated themes: its bare type rules (e.g. `SVGElement:`)
  are less specific than the descendant rules those themes rely on (`GeneticPart SVGElement:`),
  and layering them in shifts specificity. Use it as a reference, and as a base for *new*
  flat-rule themes. Regenerate whenever you add a selectable type or field.
- `default.yaml` — base styles for everything (gene schematics, containers, text).
- `plots.yaml` — defaults for plot panels (xlims/ylims/vlims, cmap, smoothing params).
- `paper.yaml` — print-friendly preset.
- `rcparams.yaml` — pure matplotlib rcParams.
- `_figure_defaults.yaml` — shared baseline `Figure.rc_context` (font chain Poppins → Roboto → DejaVu Sans, spines off on top + right). Both `plots.yaml` and `default.yaml` pull this in via `@Figure` selector-include so the cascade inherits it regardless of which theme you load.

Bio palettes (~hundreds of named colors I use for fluorophores) ship in `resources/colors/bio_palettes.yaml` and load at import time.

### Layering recipe

Themes compose via `!cascade:jstyle` and `<<{+<}:` deep-merge. A typical stack:

```yaml
# my_theme.yaml
rules: !cascade:jstyle
  <<{+<}: !include pkg:jeanplot:resources/themes/default.yaml@rules
  <<{+<}: !include pkg:jeanplot:resources/themes/plots.yaml@rules
  # your overrides on top — explicit fields keep winning at every level
  SmoothPanel2D:
    cmap: magma
    vlims: [0.0, 2.0]
  "[style_class=highlight] Text":
    color: "#ff0066"
```

Load it with `Figure(theme=...)` (preferred — theme travels with the figure), or imperatively via `load_plot_theme("my_theme.yaml")` for notebook/test use. The `@rules` selector-include is important: it pulls just the cascade out of each theme file so they merge cleanly.

### Bundled fonts

Poppins and Roboto (both SIL OFL 1.1) ship in `resources/fonts/` and register with matplotlib at import time via `_fonts.register_bundled_fonts()`. The default font chain resolves to a real font without requiring system-level installs. Disable by setting `font.family` in your theme's `rc_context`.

## PlotData

`PlotData` is what panels consume. Shape conventions:

- **`xval`** — `(n_samples, n_inputs)`. 1D arrays are auto-reshaped to `(n, 1)`.
- **`yval`** — `(n_samples, n_outputs)`. Same auto-reshape.
- **`input_names`** — `list[str]` of length `n_inputs`. Used as axis labels.
- **`output_name`** — `str` (or `list[str]` if `force_single_output=False`). Used as the dependent-axis label.
- **`column_names`** — optional names for the `xval` columns. Defaults to `input_names`.
- **`metadata`** — free-form dict. `default_output_name` reads `metadata['network_name']`; compose helpers aggregate it into figure metadata.

`AutoPanel` dispatches on `xval.shape[1]`: 1 → `SmoothPanel1D`, 2 → `SmoothPanel2D`, 3 → `SmoothPanel3D`. So the shape *is* the panel choice.

`LazyPlotData` has the same fields but loads arrays on first access (useful when you build a figure tree that references files you don't want to read until render time). `GridData` is a gridded summary that round-trips through base64 — handy when shipping precomputed surfaces through YAML.

## Writing a figure YAML

The shape of a typical figure config — same primitives the bundled templates use:

```yaml
# my_figure.yaml
<<(<): !include pkg:jeanplot:resources/templates/auto_panel   # AutoPanel template
<<(<): !include pkg:jeanplot:resources/themes/plots           # propagates xlims, vlims, ...

!require plot_data: "PlotData to render"
!set_default title: null
!set_default output_dir: "./"
!set_default output_file: null

figure: !Figure
  theme: !include pkg:jeanplot:resources/themes/plots.yaml@rules
  output_dir: ${output_dir}
  output_file: !default_output_name
    plot_data: ${plot_data}
    fallback: "fig"
    override: ${output_file}
  layout: !LayoutConstraints { direction: row, gap: 8 }
  children:
    - !AutoPanel
      plot_data: ${plot_data}
      title: ${title}
```

Then either:

```bash
jeanplot +my_figure.yaml ++plot_data='!include file:data.json' --output-dir out/
```

or from Python:

```python
from jeanplot.cli import PlotJob
PlotJob.invoke("my_figure.yaml", plot_data=my_plot_data).run()
```

Things to notice:
- `<<(<):` propagates `!define`/`!set_default` from the included files. That's how `${xlims}` resolves inside `!AutoPanel` even though you didn't declare it here — `plots.yaml` did.
- `theme:` uses a **selector-include** (`@rules`) to pull just the cascade out of the theme file. The Figure carries it; the renderer applies it during layout.
- `!default_output_name` is a registered compose helper called as a tag — kwargs in the mapping body.
- Every `!set_default` becomes a CLI flag automatically. `output_dir` → `--output-dir`. Mapping bodies (`{default: X, help: "..."}`) add help text and shorts.

For multi-panel figures, swap `children:` with `!panel_row` / `!panel_grid`, or use the higher-order templates in `resources/figures/templates.yaml` (`!ComparePair`, `!Triple`).

## Compose helpers

`jeanplot/compose.py` is ~100 LOC of "build a row of panels" / "build a grid" / "build a figure with metadata" helpers. They're registered as dracon `!fn` templates so you can use them from YAML too:

- `panel_row(panels, gap, weights)` / `!panel_row` — one row of panels.
- `panel_grid(rows, gap, col_weights, row_weights)` / `!panel_grid` — multi-row grid.
- `panels_from_datas(datas, **kwargs)` / `!panels_from_datas` — map a list of `PlotData` to a list of `AutoPanel`.
- `build_figure_metadata(panels, extra)` / `!build_figure_metadata` — aggregate per-panel + per-data metadata into one dict (useful for filename templates).
- `default_output_name(plot_data, fallback, prefix, suffix, override)` / `!default_output_name` — pick a sensible output filename from `plot_data.metadata['network_name']` with overrides.

Reusable figure templates live in `resources/figures/`:
- `data.yaml`, `pred_combined.yaml`, `combined.yaml` — concrete figures.
- `templates.yaml` — higher-order `!fn`s like `!ComparePair { a, b }` and `!Triple { a, b, c }`.

```yaml
<<(<): !include pkg:jeanplot:resources/figures/templates
figure: !ComparePair { a: ${gt}, b: ${pred} }
```

## Loading YAML from Python

The package exposes context helpers so dracon can find all the jeanplot types:

```python
from jeanplot import make_plot_context
import dracon as dr

ctx = make_plot_context(extra_types=[MyCustomPanel], extra={"my_var": 42})
cfg = dr.load("path/to/figure.yaml", context=ctx)
```

- `make_context_from_types(types)` — dict of `{TypeName: type}` for dracon tag resolution.
- `make_plot_context(extra_types=None, extra=None)` — adds `COMPOSE_HELPERS` on top so `!panel_row` etc. resolve.
- `DEFAULT_TYPES` — the full list of jeanplot types pre-registered (Components, panels, gene parts, data models).

## The CLI

`jeanplot` is installed by `pip install -e .` and reads a Figure-typed YAML:

```bash
jeanplot +path/to/figure.yaml --output-dir out/
jeanplot +mytheme --vlim-low -1 --vlim-high 1
jeanplot +fig.yaml ++network_name=ABCD12  # context override
```

CLI surface (model fields):
- `-o`/`--output-dir` — override the figure's `output_dir`.
- `--output-file` — override the figure's `output_file`.
- `-v`/`--verbose` — show component tree + full span tree.
- `-q`/`--quiet` — suppress terminal output.
- `--preview {auto,on,off}` — inline image preview in graphics-capable terminals.

Any extra flags come from `!set_default`/`!require` declared in the YAML (or its includes). That's a dracon thing — see "Vocabulary-as-CLI" in dracon's docs.

From Python: `PlotJob.invoke('path.yaml', **overrides)`, or `PlotJob.from_config('path.yaml').run()` if you want to poke at the job before running it.

### The TUI

The CLI uses a rich-based live TUI by default: spinner with current dracon span, rolling history of recent steps, total wall-time receipt, and inline image preview at the end (Kitty graphics protocol if your terminal supports it). It plugs into dracon's progress event system as a `Subscriber`. `-q` disables it; `--preview off` keeps the receipt but skips the image.

## Rendering

`render(component, *, backend="matplotlib", output=None, ...)` is the function you'll use. Returns axes/context, or an SVG string for `backend="svg"`, or just writes the file if you pass `output=`.

`render_to_string(component, *, backend="svg")` is the explicit string variant.

Need backend-specific control? `MatplotlibRenderer().render_component(ax, root)` and `SVGRenderer().render_to_string(root)` are the underlying calls. Both subclass `BaseRenderer`.

For testing layouts, `jeanplot.testing` ships `MockRenderer`, `render_to_svg`, `parse_svg`, `get_element_bounds`, `svg_hash`, and `assert_element_position` / `assert_element_size`. These keep snapshot tests boring.

## Foundations module

`jeanplot/data/`:
- `PlotData` — `X`, `Y`, `column_names`, `output_names`, `metadata`. Plain Pydantic.
- `LazyPlotData` — same shape, arrays load on first access.
- `GridData` — gridded summary, base64-roundtrippable via `grid_data_to_b64` / `grid_data_from_b64` / `extract_grid_data`.
- `DataDimensions` — shape descriptors.
- `Rescaler` — `fwd(x)` / `inv(x)` protocol. `IdentityRescaler` is the default; biocomp's `DataRescaler` already satisfies it without changes.
- `PlotFunctionResult` — return type for drawing functions that want to surface a `_mappable` etc.

`jeanplot/knn/`:
- KNN tree backends — picks `usearch`, `pykdtree`, or `scipy` based on what's installed.
- Density estimators, Gaussian-weighted KNN with optional numba acceleration, optional JAX kernels for differentiable use.
- **Shared neighbour-query cache.** `knn_stats` caches the y-independent prep (query +
  density rebalancing) by (tree, query points, params), so panels over the same grid
  (value heatmap, gradient map, quiver) share *one* query. LRU-bounded (arrays are large);
  `smooth_kernel.clear_knn_caches()` releases them between batch figures. Results are
  exact/deterministic on the `pykdtree`/`scipy` backends; `usearch` is approximate (HNSW).
- **Splat fit cache.** `SplatField.fit` is a pure function with a read-only result, so it
  carries an exact content-keyed LRU memo (`splat.core._FIT_CACHE`, on the full arg set incl.
  `stats`/`zslice`): a surface re-smoothed for both metrics and render, or by peer sub-views,
  reuses one fit. Also released by `clear_knn_caches()` (via `clear_fit_cache`).

`jeanplot/color/`:
- `load_palettes(path)`, `register_palettes(palettes)`, `closest_name(name)` for fuzzy name matching.
- Bio palettes register on import.

`jeanplot/stats.py`:
- `rmse`, `mse`, `mae`, `r_squared`, `pearson_r`. Just numpy, no deps.

## Debugging utilities

`from jeanplot import set_debug, get_logger, debug_print, DebugMixin`. Toggle `set_debug(True)` to surface internal layout / style decisions. `DebugMixin` is the base if you want to add debug output to your own components.

## Migration from biocomp-plot

There's a cheatsheet at `docs/migrating_from_biocomp.md` if you're porting an existing biocomp plotting script. TL;DR mapping:

| biocomp | jeanplot |
|---|---|
| `PlotConfig` | `Figure(Container)` + jstyle rules |
| `PlotTask` | `PlotPanel(Container)` with overlay children |
| `tasks/{auto,1D,2D,3D}.yaml` | `AutoPanel` |
| `default_plotconf_v2.yaml` callstack | `themes/plots.yaml` |
| `Overlay` protocol | child component with `is_overlay=True` |
| `FigAx.subdivide` / `subax_spec` | nested `Container` |

Every drawing function, KNN kernel, axis helper, `PlotData`, and `Rescaler` lives in jeanplot exactly once. `biocomp.plotting.*` is now a shim layer that re-exports jeanplot symbols; the legacy `BiocompPlotFigure` / `PlotConfig` / `PartialFunction` / `@configurable` / `SimpleLayout` / `GridLayout` / `FigureSpec` machinery is deprecated (still renders un-migrated paper-jobs YAMLs, but emits `DeprecationWarning` at construction).

## Repo layout

```
jeanplot/
├── core/         scene graph: Component, Container, Text, Connection, Table, SVGElement, style engine
├── gene/         gene schematic data + visual parts
├── panels/       PlotPanel subclasses + Figure + overlays + panel_from
├── plots/        plain matplotlib drawing functions panels delegate to
├── data/         PlotData, LazyPlotData, GridData, Rescaler
├── knn/          KNN trees + density estimators
├── color/        palettes, name matching
├── resources/    themes, figure templates, color palettes, SVG parts, bundled fonts
├── compose.py    tree-construction helpers (also dracon !fn templates)
├── render.py     top-level render() / render_to_string()
├── testing.py    test helpers (MockRenderer, svg_hash, assert_element_*)
├── _fonts.py     bundled-font registration with matplotlib
├── _tui.py       CLI TUI (dracon progress subscriber + image preview)
├── cli.py        jeanplot entry point
└── tests/        ~460 tests, pytest
docs/             STYLE_GUIDE, migrating_from_biocomp
example/          hello_jeanplot.py
```

## When stuff breaks

- **Nothing renders.** Check `min_dimensions` isn't zero, check `show=True` if you're using matplotlib interactive.
- **Style doesn't apply.** Check the selector — `[style_class=foo]` not `.foo`. Check specificity if you're being overridden. Check the cascade was actually loaded (`jstyle._cascade is None` means nothing's set).
- **Text too big or too small.** It's `font_size_mode`. `data` scales with the world (good for in-world labels). `points` stays a constant visual size (good for everything else).
- **Connection invisible.** Check the endpoint ids resolve and the components have non-zero size.
- **Panel draws but axes empty.** Probably an empty `PlotData` or a shape mismatch — panels assert at the boundary.
- **Font shows up as DejaVu instead of Poppins.** Either the bundled font failed to register (rare; check warnings at import), or your theme's `rc_context` overrode the chain.
- **CLI flag doesn't exist that you expected.** It's declared with `!set_default name:` (or `!require name:`) somewhere in the loaded YAML or an include. Run `jeanplot show +your.yaml --show-vars` (dracon's `show` form) to see what's actually declared.

Reach for `jeanplot.testing.svg_hash` when you want to know if your output changed. Pixel-equality on matplotlib output is a losing game; SVG hashing is cheap and stable.

## What jeanplot is not

It's not a charting library. There's no `bar()`, no `histogram()` convenience function. If you want quick plots, use matplotlib directly. jeanplot's value is when you want the same plot to look right at any size, embedded inside a larger composition, with consistent theming — i.e. when you're making figures for papers, not when you're poking at data in a REPL.

It's also not a UI framework. The scene graph is retained but inert. Nothing animates, nothing responds to clicks, there's no event loop. Render once, get an image out, move on.

## License

MIT. See `LICENSE`. Bundled fonts (Poppins, Roboto) are SIL OFL 1.1 — license files ship in `resources/fonts/<family>/OFL.txt`.
