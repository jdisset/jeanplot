# Jeanplot

A declarative, scene-graph, component-based, stylable 2D drawing library for rendering complex nested layouts in Python. Currently supports matplotlib as the rendering backend.

Initial use-case is for synthetic biology schematics and diagrams, but should be expressive enough for many other domains.

## Installation

```bash
pip install -e .
```

## Core Concepts

### Component Model

All visual elements inherit from `Component`:
- **Positioning**: `offset` (relative/absolute), `transform` (translate, rotate, scale, skew)
- **Sizing**: `min_dimensions`, `max_dimensions`, constraint application
- **Styling**: `style` (BoxStyle with border, background, shadow, margin, padding)
- **Layout**: flexbox-like `LayoutConstraints` (direction, align_items, justify_content, gap)
- **Hierarchy**: parent-child relationships, attachment points

### Layout System

Uses a flexbox-inspired layout system:
- `direction`: "row" | "column"
- `align_items`: "start" | "center" | "end" | "stretch"
- `justify_content`: "start" | "center" | "end" | "space-between" | "space-around" | "space-evenly"
- `gap`: spacing between children

### Styling (jstyle)

CSS-inspired selector and cascade system:

```python
from jeanplot.style import jstyle

jstyle.update({
    "Container[id=main]": {
        "style.background_color": "lightblue",
        "style.padding": (10, 10, 10, 10)
    },
    "Text[style_class=title]": {
        "font_size": 14,
        "color": "darkblue"
    }
})
```

**Selector types:**
- Type: `Container`, `Text`
- ID: `[id=my-button]`
- Style class: `[style_class=primary]`
- Attribute: `[status=active]`, `[id^=item-]`, `[name*=button]`
- Wildcard: `*`
- Combined: `Text[style_class=error]`

## Quick Start

### Basic Container Layout

```python
from jeanplot.container import Container
from jeanplot.models import Size, BoxStyle, LayoutConstraints
from jeanplot.matplotlib_renderer import MatplotlibRenderer
import matplotlib.pyplot as plt

box1 = Container(
    id="box1",
    min_dimensions=Size(width=100, height=50),
    style=BoxStyle(background_color="red"),
)
box2 = Container(
    id="box2",
    min_dimensions=Size(width=100, height=50),
    style=BoxStyle(background_color="blue"),
)

root = Container(
    children=[box1, box2],
    layout=LayoutConstraints(direction="row", gap=10),
    style=BoxStyle(padding=(20, 20, 20, 20))
)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect("equal")
ax.axis("off")

renderer = MatplotlibRenderer()
renderer.render_component(ax, root, adjust_lims=True)
plt.savefig("output.png")
```

### Genetic Circuit Schematic

```python
from jeanplot.network_schematic_v2 import NetworkGeneticSchematicV2
from jeanplot.container import Container
from jeanplot.models import LayoutConstraints
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.style import jstyle

# network: a biocomp Network object
schematic = NetworkGeneticSchematicV2(
    network=network,
    hide_marker_tus=True,
    grid_gap=(40.0, 20.0),
    connection_style="orthogonal"  # or "bezier", "straight"
)

root = Container(
    children=[schematic],
    layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
)
jstyle.apply(root)

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_aspect("equal")
ax.axis("off")

renderer = MatplotlibRenderer()
renderer.render_component(ax, root, adjust_lims=True)
plt.savefig("circuit.png")
```

### Network Compute Diagram

```python
from jeanplot.network_diagram_v2 import NetworkDiagramV2

diagram = NetworkDiagramV2(
    network=network,
    simplified=True  # Hide inverse chains
)

root = Container(children=[diagram])
jstyle.apply(root)

# Render as above
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `component.py` | Base visual element with position, size, style, transformation |
| `container.py` | Container managing/laying out child components |
| `models.py` | Core data models: Size, Transform, Offset, BoxStyle, LayoutConstraints |
| `renderer.py` | Abstract renderer interface |
| `matplotlib_renderer.py` | Matplotlib implementation |
| `style.py` | CSS-like styling system |
| `text.py` | Text rendering |
| `connector.py` | Connection lines/curves between components |
| `genetic_elements.py` | Biology-specific elements: TranscriptionUnit, Source, Promoter, Terminator, ERN, UorfGroup |
| `network_diagram.py` | Compute graph visualization |
| `network_schematic.py` | Genetic circuit visualization |
| `network_utils.py` | Utility functions for network analysis |

## Biological Components

### Genetic Elements

- `TranscriptionUnit`: Container with connecting line, parts layout
- `Source`: Represents plasmids or co-transfection sources
- `Promoter`, `Terminator`: Regulatory elements
- `ERN`: Endonuclease recognition (with marker)
- `UorfGroup`: uORF groups with labels
- `FluoMarker`: Fluorescent protein markers

### Network Diagram Nodes

- `ComputeNode`: Base for compute graph nodes
- `TranscriptionNode`: DNA→RNA
- `TranslationNode`: RNA→Protein
- `ERNNode`: ERN inhibition
- `AggregationNode`: Co-transfection mixing
- `FluoNode`, `TUNode`, `DeadEndNode`, `InputNode`

## Transform System

Full affine transformations with proper composition:

```python
from jeanplot.models import Transform, Offset

transform = Transform(
    translate=(10, 20),
    rotate=45,           # degrees
    scale=(1.5, 1.0),
    rotation_center=(0, 0)
)
```

Transforms compose via matrix multiplication, supporting nested hierarchies.

## BoxStyle

```python
from jeanplot.models import BoxStyle, Shadow

style = BoxStyle(
    background_color="white",
    border_color="black",
    border_width=2,
    border_style="solid",  # or "dashed", "dotted"
    margin=(5, 5, 5, 5),   # (top, right, bottom, left)
    padding=(10, 10, 10, 10),
    corner_radius=5,
    shadow=Shadow(offset=(3, 3), blur=5, spread=0, color="gray")
)
```

## Connections

```python
from jeanplot.connector import Connection, OrthogonalCurve, SimpleBezierCurve
from jeanplot.svg import LineEndArrow

connection = Connection(
    start_component="source_id",
    end_component="target_id",
    curve_type=OrthogonalCurve(),  # or SimpleBezierCurve(), StraightCurve()
    line_style="solid",
    end_cap=LineEndArrow(size=10, angle=30),
    auto_route=True  # Use anchor points if available
)
```

## Theming

Default theme loaded from `resources/themes/default.yaml`:

```python
from jeanplot import load_default_theme, jstyle

# Reset to default theme
load_default_theme()

# Override specific styles
jstyle.update({
    "TranscriptionUnit": {
        "style.background_color": "#f0f0f0"
    }
})
```

## Integration with biocomp-tools

Used via `biocomptools.toollib.figuremakers`:

```python
from biocomptools.toollib.figuremakers.geneticcircuit import GeneticCircuitFigure
from biocomp.plotutils import FigureSpec

figure = GeneticCircuitFigure(
    figure_spec=FigureSpec(output_dir="./", output_file="circuit.pdf"),
    network=network,
    hide_marker_tus=True,
    connection_style="orthogonal"
)
figure.run()
```

CLI command:
```bash
biocomp-circuitplot -r recipe.json5 -o circuit.pdf -t circuit
biocomp-circuitplot -r recipe.json5 -o diagram.pdf -t diagram
```

## Debug Mode

```python
from jeanplot.debug import set_debug
set_debug(True)  # Shows bounding boxes and origins
```

## Dependencies

- dracon (YAML configuration)
- matplotlib (rendering backend)
- numpy (matrix transformations)
- lxml (SVG parsing)
- pydantic>=2.0 (data validation)
- svgpath2mpl (SVG path conversion)
