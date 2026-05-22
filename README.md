# jeanplot

A 2D plotting library that thinks the way matplotlib should have. Build a tree of components, let layout figure out where they go, style them with CSS-ish rules, render to matplotlib or SVG. Comes with a gene-schematic module because that's what I needed it for, and a scientific-plot module on top because I needed that too.

```bash
pip install -e .
```

That's it. No conda, no extras, no platform shenanigans.

## What you actually do with it

Three things, in order of how often you'll touch them:

1. **Build a tree.** A `Container` holds children. Children can be `Container`, `Text`, `Connection`, gene parts, plot panels, tables — anything that subclasses `Component`. Each child has a `min_dimensions`, an optional `offset`, and a `layout` that says how its own children are arranged (row, column, gap, align). It's flexbox-ish.
2. **Render it.** `from jeanplot import render; render(root)`. Matplotlib by default, SVG if you ask for it. You get back an axes or a string.
3. **Style it.** `jstyle` is the global style engine. Selectors look like `Container[style_class=card]`. Properties look like `style.background_color`. You either feed it a dict or a `!cascade:jstyle` YAML document, your call.

Hello world is the same shape as every other example in the repo:

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

Open `example/hello_jeanplot.py` for a slightly nicer version with colors.

## The tree

Everything is a `Component`. The interesting subclasses live in `jeanplot/core/`:

- `Container` — has children, has a `layout`, optionally has a `style` (`BoxStyle` — background, border, corner radius, padding). This is 90% of what you'll instantiate.
- `Text` — a string with a font size, weight, color. Has two sizing modes: `data` (scales with the world) and `points` (fixed visual size). Pick `points` if you want labels that stay legible at any zoom.
- `Connection` — line from one component's anchor to another's. Pass `start_component="id1", end_component="id2"`. Curve type can be straight, orthogonal (right angles with rounded corners), or bezier. Endpoints can be arrows or circles.
- `Table` — rows of cells with column widths in `%` or pixels. Behaves how you'd expect.
- `SVGElement` / `LineEnd*` — for when you want to draw something matplotlib doesn't have a primitive for.

Components have an `id`, optional `style_class` (list of strings, like CSS classes), and a parent set automatically when you add them as children. The parent link is what makes `jstyle` selectors work — "Text inside a Container that has class=warning" actually parses.

## Layout

`LayoutConstraints` is the boring part you can mostly ignore. The fields that matter:

- `direction`: `row` or `column`.
- `gap`: pixels between children.
- `align_items`: `start`, `center`, `end`, `stretch`.
- `justify_content`: same vocabulary.
- `main_axis_weights` / `cross_axis_weights`: optional list of floats for flex-style distribution when there's leftover space. If you don't set them, children get their `min_dimensions` and the rest is padding.

You can write a `LayoutConstraints` long-form mapping or use the string DSL:

```python
Container(layout="row gap=8 align=center", children=[...])
```

```yaml
layout: "row gap=1.0 align=stretch"
```

`align` aliases `align_items`, `justify` aliases `justify_content`, `col` aliases `column`. Mapping and string forms are interchangeable.

Children can be passed positionally: `Container(panel1, panel2, ...)` is sugar for `Container(children=[panel1, panel2, ...])`. Works for `Figure` too, in Python and via the YAML `!Container [a, b]` bare-list shortcut.

Nest containers to get grids. There's no grid primitive because you don't need one — `Container(layout="column", children=[Container(layout="row", ...), ...])` is a grid.

Absolute positioning works too: pass `offset=Offset(absolute=(x, y))` on a child and the layout pass leaves it alone.

## jstyle

`jstyle` is CSS minus the inheritance soup. You write rules keyed by selectors and the engine matches them against components during layout.

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
- type name (`Container`)
- id (`[id=foo]`)
- class (`[style_class=card]`)
- descendant (space-separated, like CSS)
- combinations of all of the above

Specificity is the usual `(ids, classes, types)` tuple. More specific wins. Newer wins ties.

`jstyle.update(value)` defaults to **fill** semantics — cascade values fill fields the user didn't set, explicit values on a component always win. Under the hood every `Component` snapshots `model_fields_set` at construction (`_user_set_fields`), and the cascade walks around those. To get the old clobber-everything behaviour, write the cascade as `!cascade:jstyle` (legacy) instead of `!cascade:jstyle_fill` (the default). `jstyle.clear()` exists if you want a true reset.

YAML form looks like:

```yaml
rules: !cascade:jstyle_fill
  Container[style_class=card]:
    style.background_color: "#ffffff"
    style.corner_radius: 8
  Text:
    font_size: 12
```

Load with `load_default_theme()` (which reads `pkg:jeanplot:resources/themes/default.yaml`) or with `jstyle.update(your_cascade_symbol)`. Layer themes by including one cascade as the base of another — explicit fields keep winning at every level.

## Gene schematics

`jeanplot/gene/` is the original reason this library exists. It draws genetic circuits: promoters, terminators, ERNs, fluorophore markers, transcription units, sources (plasmids), and the interactions between them.

The shape is:

```python
from jeanplot.gene import GeneticSchematic, CircuitData

circuit = CircuitData(transcription_units=[...], sources=[...], interactions=[...])
schematic = GeneticSchematic.from_circuit(circuit)
render(schematic, output="circuit.svg")
```

Parts are SVG-backed (`Promoter`, `Terminator`, `ERN`, `ERN5pRecog`, `FluoMarker`, `UorfGroup`). They lay out left-to-right inside a `TranscriptionUnit` row. Sources sit on the side. Interactions are `Connection` objects with arrow heads.

The grammar of "what valid circuit shapes look like" lives in `jeanplot/gene/data.py` as plain Pydantic models. The visual layer just reads those.

## Scientific plots (the panels layer)

This part is newer. It exists because matplotlib's API is fine but everyone reinvents the same wrapper to coordinate figure size, dpi, multi-panel layout, colorbars, themes, and IO. So jeanplot has a thin wrapper that uses the same Container tree.

```python
import numpy as np
from jeanplot import Figure, SmoothPanel2D, PlotData, render

x = np.random.randn(500)
y = np.random.randn(500)
z = np.exp(-(x**2 + y**2))

data = PlotData(
    X=np.column_stack([x, y]),
    Y=z[:, None],
    column_names=["x", "y"],
    output_names=["z"],
)
fig = Figure(panels=[SmoothPanel2D(data=data)])
render(fig, output="smooth.png")
```

`Figure` is a `Container` with figure-level fields (`dpi`, `output_file`, `rc_context`, `theme`). It doesn't orchestrate anything — the renderer walks the tree, sees `PlotPanel` leaves, and allocates a real matplotlib axes for each one. Layout decides where the axes go.

Panels you get out of the box (`jeanplot.panels`):

- `SmoothPanel1D` / `SmoothPanel2D` / `SmoothPanel3D` — KNN-smoothed surfaces. 3D is a cube view + a grid of slices.
- `MVPPanel` — measured-vs-predicted scatter with an identity line.
- `DensityPanel` — kernel density estimate.
- `ScatterPanel` — what it says.
- `ViolinPanel`, `ParticlePanel`, `StackedPolyPanel` — distribution viz.
- `AsciiHeatmapPanel` — terminal-friendly output (Kitty graphics if you have it, else `░▒▓█` characters).
- `Colorbar` — overlay child that reads `parent._mappable`. Drop it next to any panel that produces one.
- `AutoPanel` — dispatches to the right Smooth* panel based on `data.X.shape[1]`.

Each panel is a typed Pydantic model with the parameters you'd pass to the equivalent matplotlib call as fields. No `**kwargs`. The fields you don't set come from the theme (cascade-fill).

The matplotlib drawing code itself lives in `jeanplot/plots/` as plain functions. Panels are thin shells that call those. If you want to draw without the panel system, just call the function.

### `@panel_from` — function = panel

For most panels you don't write the class by hand. Decorate the drawing function:

```python
from jeanplot.panels import panel_from

@panel_from
def my_plot(plot_data, *, ax, vlims=(0.0, 1.0), cmap="viridis"):
    ax.imshow(plot_data.X, vmin=vlims[0], vmax=vlims[1], cmap=cmap)
```

The decorator introspects the signature and synthesises a `PlotPanel` subclass with `vlims` and `cmap` as Pydantic fields, registers it as the YAML tag `!my_plot`, and keeps the original function callable as a plain function (REPL-friendly). Panels also expose cascade-fillable `axes_size`, `colorbar_pad`, `legend_pad`, and a computed `min_dimensions` so the figure auto-sizes around its contents.

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

You override per-panel-class, per-instance (via id), or per-class (via `style_class`). Specificity does the right thing. The bio palettes (a couple hundred named colors I use for fluorophores) ship in `resources/colors/bio_palettes.yaml`; they're loaded at import time.

There's also `themes/paper.yaml` for a print-friendly preset and `themes/rcparams.yaml` for matplotlib rcParams.

## Compose helpers

`jeanplot/compose.py` is ~100 LOC of "build a row of panels" / "build a grid" / "build a figure with metadata" helpers. They're registered as dracon `!fn` templates so you can use them from YAML:

```yaml
figure: !Figure
  panels: !panel_row
    panels:
      - !SmoothPanel1D { data: !include data_1.json }
      - !SmoothPanel1D { data: !include data_2.json }
```

The reusable figure templates live in `resources/figures/` — `data.yaml`, `pred_combined.yaml`, `combined.yaml`, and `templates.yaml` for higher-order things like `ComparePair`.

## The CLI

`jeanplot` is installed by `pip install -e .` and reads a Figure-typed YAML:

```bash
jeanplot +path/to/figure.yaml --output-dir out/
jeanplot +mytheme --vlim-low -1 --vlim-high 1
```

The flags that appear depend on the YAML you load — any `!set_default`/`!require` declared in the file or its includes surfaces as a CLI flag. This is a dracon thing, not a jeanplot thing, but it's why the CLI is small.

From Python: `PlotJob.invoke('path.yaml', **overrides)` or `PlotJob.from_config('path.yaml').run()` if you want to poke at the job before running it.

## Rendering

`render(component, *, backend="matplotlib", output=None, ...)` is the function you'll use. Returns axes/context, or an SVG string for `backend="svg"`, or just writes the file if you pass `output=`.

If you need backend-specific control: `MatplotlibRenderer().render_component(ax, root)` and `SVGRenderer().render_to_string(root)` are the underlying calls.

For testing layouts: `jeanplot.testing` ships `MockRenderer`, `render_to_svg`, `svg_hash`, and `assert_element_position` / `assert_element_size`. These keep snapshot tests boring.

## Foundations module

`jeanplot/data/`:
- `PlotData` — `X`, `Y`, `column_names`, `output_names`, metadata. Plain Pydantic.
- `LazyPlotData` — same shape, arrays load on first access.
- `GridData` — gridded summary, base64-roundtrippable.
- `Rescaler` — `fwd(x)` / `inv(x)` protocol. `IdentityRescaler` is the default; biocomp's `DataRescaler` already satisfies it without changes.

`jeanplot/knn/`:
- KNN tree backends — picks `usearch`, `pykdtree`, or `scipy` based on what's installed.
- Density estimators, Gaussian-weighted KNN with optional numba acceleration, optional JAX kernels for differentiable use.

`jeanplot/color/`:
- `load_palette`, `closest_name` for fuzzy name matching.
- Bio palettes register on import.

`jeanplot/stats.py`:
- `rmse`, `mse`, `mae`, `r_squared`, `pearson_r`. Just numpy, no deps.

## Migration from biocomp-plot

There's a cheatsheet at `docs/migrating_from_biocomp.md` if you're porting an existing biocomp plotting script. The TL;DR mapping:

| biocomp | jeanplot |
|---|---|
| `PlotConfig` | `Figure(Container)` + jstyle rules |
| `PlotTask` | `PlotPanel(Container)` with overlay children |
| `tasks/{auto,1D,2D,3D}.yaml` | `AutoPanel` |
| `default_plotconf_v2.yaml` callstack | `themes/plots.yaml` |
| `Overlay` protocol | child component with `is_overlay=True` |
| `FigAx.subdivide` / `subax_spec` | nested `Container` |

Every drawing function, KNN kernel, axis helper, `PlotData`, and `Rescaler` lives in jeanplot exactly once. `biocomp.plotting.*` is now a shim layer that re-exports jeanplot symbols; the legacy `BiocompPlotFigure` / `PlotConfig` / `PartialFunction` / `@configurable` / `SimpleLayout` / `GridLayout` / `FigureSpec` machinery is marked deprecated (it still renders the un-migrated paper-jobs YAMLs, but emits `DeprecationWarning` at construction). Port when you feel like it; the deprecation flag tells you when something still uses the old path.

## Repo layout

```
jeanplot/
├── core/         scene graph: Component, Container, Text, Connection, ...
├── gene/         gene schematic data + visual parts
├── panels/       PlotPanel subclasses + Figure
├── plots/        plain matplotlib drawing functions panels delegate to
├── data/         PlotData, GridData, Rescaler
├── knn/          KNN trees + density
├── color/        palettes, name matching
├── resources/    themes, figure templates, color palette YAML
├── compose.py    tree-construction helpers
├── render.py     top-level render() / render_to_string()
├── testing.py    test helpers (MockRenderer, svg_hash, ...)
├── cli.py        jeanplot entry point
└── tests/        ~460 tests, pytest
docs/             this README is the main entry; STYLE_GUIDE, migration
example/          hello_jeanplot.py
```

## When stuff breaks

- **Nothing renders.** Check `min_dimensions` isn't zero, check `show=True` if you're using matplotlib interactive.
- **Style doesn't apply.** Check the selector — `[style_class=foo]` not `.foo`. Check specificity if you're being overridden. Check the cascade was actually loaded (`jstyle._cascade is None` means nothing's set).
- **Text too big or too small.** It's `font_size_mode`. `data` scales with the world (good for in-world labels). `points` stays a constant visual size (good for everything else).
- **Connection invisible.** Check the endpoint ids resolve and the components have non-zero size.
- **Panel draws but axes empty.** Probably an empty `PlotData` or a shape mismatch — panels assert at the boundary.

Reach for `jeanplot.testing.svg_hash` when you want to know if your output changed. Pixel-equality on matplotlib output is a losing game; SVG hashing is cheap and stable.

## What jeanplot is not

It's not a charting library. There's no `bar()`, no `histogram()` convenience function. If you want quick plots, use matplotlib directly. jeanplot's value is when you want the same plot to look right at any size, embedded inside a larger composition, with consistent theming — i.e. when you're making figures for papers, not when you're poking at data in a REPL.

It's also not a UI framework. The scene graph is retained but inert. Nothing animates, nothing responds to clicks, there's no event loop. Render once, get an image out, move on.

## License

MIT. See `LICENSE`.
