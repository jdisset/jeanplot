# Jeanplot

A declarative, component-based 2D visualization library for Python with CSS-like styling, flexbox layout, and multiple rendering backends.

Jeanplot uses a **retained-mode scene graph** architecture where you build a tree of components, apply styles, and render to matplotlib or SVG. Originally designed for synthetic biology schematics, it's expressive enough for many visualization domains.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
  - [Component](#component)
  - [Container](#container)
  - [Text](#text)
  - [SVGElement](#svgelement)
  - [Table](#table)
- [Layout System](#layout-system)
  - [Direction](#direction)
  - [Alignment](#alignment)
  - [Justify Content](#justify-content)
  - [Gap and Spacing](#gap-and-spacing)
  - [Padding and Margin](#padding-and-margin)
- [Styling with jstyle](#styling-with-jstyle)
  - [Basic Selectors](#basic-selectors)
  - [Attribute Selectors](#attribute-selectors)
  - [Selector Operators](#selector-operators)
  - [Combined Selectors](#combined-selectors)
  - [Nested (Contextual) Selectors](#nested-contextual-selectors)
  - [Specificity](#specificity)
  - [Property Setting](#property-setting)
  - [Partial Updates](#partial-updates)
  - [Temporary Style Contexts](#temporary-style-contexts)
- [BoxStyle Reference](#boxstyle-reference)
- [Transforms and Offsets](#transforms-and-offsets)
  - [Transform](#transform)
  - [Offset](#offset)
  - [Attachments](#attachments)
- [Connections and Curves](#connections-and-curves)
  - [Connection](#connection)
  - [Curve Types](#curve-types)
  - [Line End Caps](#line-end-caps)
  - [Anchor Points](#anchor-points)
- [Gene Visualization](#gene-visualization)
  - [Data Models](#data-models)
  - [GeneticSchematic](#geneticschematic)
  - [TranscriptionUnit](#transcriptionunit)
  - [Genetic Parts](#genetic-parts)
- [Rendering](#rendering)
  - [MatplotlibRenderer](#matplotlibrenderer)
  - [SVGRenderer](#svgrenderer)
- [Testing Utilities](#testing-utilities)
- [Theming](#theming)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Installation

```bash
pip install -e .
```

### Dependencies

- **Python 3.11+**
- **pydantic>=2.0** - Data validation and models
- **numpy** - Matrix transformations
- **matplotlib** - Primary rendering backend
- **lxml** - SVG parsing
- **svgpath2mpl** - SVG path conversion
- **dracon** - YAML configuration loading

---

## Quick Start

### Basic Container Layout

```python
import matplotlib.pyplot as plt
import numpy as np
from jeanplot import (
    Container, Text, Size, BoxStyle, LayoutConstraints,
    MatplotlibRenderer, jstyle
)

# Create child components
box1 = Container(
    id="box1",
    min_dimensions=Size(width=80, height=40),
    style=BoxStyle(background_color="#e74c3c", corner_radius=4),
)
box2 = Container(
    id="box2",
    min_dimensions=Size(width=80, height=40),
    style=BoxStyle(background_color="#3498db", corner_radius=4),
)
label = Text(text="Hello Jeanplot!", font_size=12, color="#2c3e50")

# Create root container with flexbox layout
root = Container(
    id="root",
    children=[box1, box2, label],
    layout=LayoutConstraints(
        direction="row",
        gap=15,
        align_items="center",
        justify_content="center",
    ),
    style=BoxStyle(
        padding=(20, 20, 20, 20),
        background_color="#ecf0f1",
    ),
)

# Render
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_aspect("equal")
ax.axis("off")

renderer = MatplotlibRenderer()
renderer.render_component(ax, root, adjust_lims=True)
plt.savefig("hello_jeanplot.png", dpi=150, bbox_inches="tight")
```

### Styling with jstyle

```python
from jeanplot import Container, Text, jstyle

# Define styles using CSS-like selectors
jstyle.update({
    # Type selector - applies to all Containers
    "Container": {
        "style.corner_radius": 4,
    },
    # ID selector
    "[id=header]": {
        "style.background_color": "#2c3e50",
        "style.padding": (10, 15, 10, 15),
    },
    # Class selector
    "[style_class=primary]": {
        "style.background_color": "#3498db",
        "style.border_color": "#2980b9",
        "style.border_width": 2,
    },
    # Nested selector - Text inside header
    "[id=header]": {
        "Text": {
            "color": "white",
            "font_size": 14,
        },
    },
})

# Components pick up styles automatically
header = Container(
    id="header",
    children=[Text(text="Dashboard")],
)
button = Container(
    id="btn1",
    style_class=["primary"],
    children=[Text(text="Click me")],
)
```

### Render to SVG

```python
from jeanplot import Container, SVGRenderer

root = Container(id="root", children=[...])

renderer = SVGRenderer()
svg_string = renderer.render_to_string(root)

# Or save to file
renderer.render_to_file(root, "output.svg")
```

---

## Architecture Overview

Jeanplot follows a **three-phase rendering pipeline**:

```
Component Tree → Style Application → Measure & Layout → Render
     ↓                  ↓                   ↓              ↓
  (Pydantic         (jstyle CSS         (Bottom-up      (Backend-
   models)          cascade)            then top-down)   specific)
```

1. **Build**: Create a tree of `Component` objects
2. **Style**: Apply styles via `jstyle.apply(root)` (automatic during layout)
3. **Layout**: Call `root.measure_and_layout(renderer)` to compute sizes and positions
4. **Render**: Call `renderer.render_component(context, root)` to draw

### Component Hierarchy

```
Component (base)
├── Container (layout node)
│   ├── Table
│   ├── TranscriptionUnit
│   └── GeneticSchematic
├── Text (leaf node)
├── SVGElement (leaf node)
├── AnchorComponent (connection point)
├── Overlay (out-of-flow positioning)
└── Connection (lines between components)
```

---

## Core Components

### Component

Base class for all visual elements.

```python
from jeanplot import Component, Size, BoxStyle, Transform, Offset

component = Component(
    # Identity
    id="my-component",
    style_class=["primary", "card"],

    # Visibility
    show=True,
    debug=False,  # Show bounding box
    z_index=0,

    # Dimensions
    min_dimensions=Size(width=100, height=50),
    max_dimensions=Size(width=200, height=100),

    # Positioning
    offset=Offset(absolute=(10, 20)),
    transform=Transform(rotate=15, scale=(1.2, 1.0)),

    # Styling
    style=BoxStyle(
        background_color="white",
        border_color="black",
        border_width=1,
    ),

    # Layout behavior
    is_overlay=False,  # True to exclude from layout flow
    attached_to=None,  # Component ID or reference for attachment
)
```

**Key Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique identifier for styling/connections |
| `style_class` | `list[str]` | CSS-like class names |
| `min_dimensions` | `Size` | Minimum width/height |
| `max_dimensions` | `Size` | Maximum width/height |
| `offset` | `Offset` | Position offset (absolute/relative) |
| `transform` | `Transform` | Affine transformation |
| `style` | `BoxStyle` | Visual styling (border, background, etc.) |
| `is_overlay` | `bool` | Exclude from parent layout |
| `z_index` | `int` | Render order (higher = on top) |

**Key Methods:**

| Method | Description |
|--------|-------------|
| `measure_and_layout(renderer)` | Compute sizes and positions |
| `render(renderer, context, matrix)` | Draw the component |
| `compute_world_matrix()` | Get world-space transform |
| `get_world_bounds()` | Get bounding box in world coords |
| `add_child(child)` | Add a child component |

### Container

A component that manages child layout using flexbox-like rules.

```python
from jeanplot import Container, LayoutConstraints, BoxStyle

container = Container(
    id="main",
    children=[child1, child2, child3],
    layout=LayoutConstraints(
        direction="row",        # "row" or "column"
        align_items="center",   # "start", "center", "end", "stretch"
        justify_content="space-between",  # "start", "center", "end",
                                          # "space-between", "space-around", "space-evenly"
        gap=10,                 # Spacing between children
    ),
    style=BoxStyle(
        padding=(10, 15, 10, 15),  # (top, right, bottom, left)
        background_color="#ffffff",
    ),
)
```

### Text

Text rendering with font control and alignment.

```python
from jeanplot import Text

text = Text(
    id="title",
    text="Hello World",

    # Font properties
    font_name="Arial",
    font_size=14,
    font_weight="bold",     # "normal", "bold"
    font_style="normal",    # "normal", "italic"

    # Color
    color="#333333",

    # Alignment
    align="center",         # "left", "center", "right"
    vertical_align="middle", # "top", "middle", "bottom"

    # Multi-line
    line_spacing=1.2,

    # Rendering mode
    render_as_path="auto",  # True for complex transforms, False for editable text
)
```

### SVGElement

Renders SVG content with optional color remapping.

```python
from jeanplot import SVGElement

svg = SVGElement(
    id="icon",
    svg_content="path/to/icon.svg",  # Path, string, bytes, or SVGContent

    # Remap colors in the SVG
    color_remap={
        "#000000": "#3498db",  # Black to blue
        "#ff0000": None,       # Red to transparent
    },

    # Line width handling
    line_width_mode="data",  # "data" (scales) or "point" (fixed)
)
```

### Table

Grid-based table layout.

```python
from jeanplot import Table, TableRow, TableCell, Text

table = Table(
    id="data-table",
    data=[
        ["Header 1", "Header 2", "Header 3"],
        ["Cell 1", "Cell 2", "Cell 3"],
        ["Cell 4", "Cell 5", "Cell 6"],
    ],
    header_rows=1,
    border_collapse="collapse",  # "collapse" or "separate"
    border_spacing=0,
)

# Or build manually
table = Table(
    children=[
        TableRow(
            _is_header=True,
            children=[
                TableCell(children=[Text(text="Name")]),
                TableCell(children=[Text(text="Value")]),
            ],
        ),
        TableRow(children=[
            TableCell(children=[Text(text="Alpha")]),
            TableCell(children=[Text(text="100")]),
        ]),
    ],
)
```

---

## Layout System

Jeanplot uses a **flexbox-inspired** layout system for positioning children within containers.

### Direction

```python
# Horizontal layout (children side by side)
layout = LayoutConstraints(direction="row")

# Vertical layout (children stacked)
layout = LayoutConstraints(direction="column")
```

### Alignment

`align_items` controls positioning on the **cross axis** (perpendicular to direction):

```python
# Row direction: align_items controls vertical position
layout = LayoutConstraints(direction="row", align_items="start")   # Top
layout = LayoutConstraints(direction="row", align_items="center")  # Middle
layout = LayoutConstraints(direction="row", align_items="end")     # Bottom
layout = LayoutConstraints(direction="row", align_items="stretch") # Fill height

# Column direction: align_items controls horizontal position
layout = LayoutConstraints(direction="column", align_items="start")  # Left
layout = LayoutConstraints(direction="column", align_items="center") # Center
layout = LayoutConstraints(direction="column", align_items="end")    # Right
```

### Justify Content

`justify_content` controls positioning on the **main axis**:

```python
layout = LayoutConstraints(justify_content="start")        # Pack at start
layout = LayoutConstraints(justify_content="center")       # Center
layout = LayoutConstraints(justify_content="end")          # Pack at end
layout = LayoutConstraints(justify_content="space-between") # Space between items
layout = LayoutConstraints(justify_content="space-around")  # Space around items
layout = LayoutConstraints(justify_content="space-evenly")  # Equal spacing
```

### Gap and Spacing

```python
layout = LayoutConstraints(
    direction="row",
    gap=20,  # 20 units between each child
)
```

### Padding and Margin

```python
from jeanplot import BoxStyle

# Padding: space inside the container (between border and content)
style = BoxStyle(padding=(10, 15, 10, 15))  # (top, right, bottom, left)

# Margin: space outside the component (affects layout positioning)
style = BoxStyle(margin=(5, 5, 5, 5))
```

---

## Styling with jstyle

The `jstyle` system provides **CSS-like declarative styling** with selectors, specificity, and cascade.

### Basic Selectors

```python
from jeanplot import jstyle

jstyle.update({
    # Type selector - matches component class name
    "Container": {
        "style.background_color": "#f0f0f0",
    },

    # Matches Text, but also subclasses
    "Text": {
        "font_size": 12,
        "color": "#333",
    },

    # Wildcard - matches everything (lowest specificity)
    "*": {
        "debug": False,
    },
})
```

### Attribute Selectors

```python
jstyle.update({
    # ID selector
    "[id=main-header]": {
        "style.background_color": "#2c3e50",
    },

    # Style class selector (matches if class is in list)
    "[style_class=primary]": {
        "style.background_color": "#3498db",
    },

    # Any attribute
    "[font_size=14]": {
        "font_weight": "bold",
    },
})
```

### Selector Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Exact match | `[status=active]` |
| `!=` | Not equal | `[status!=error]` |
| `~=` | Case-insensitive | `[name=~john]` |
| `^=` | Starts with | `[id^=btn-]` |
| `$=` | Ends with | `[filename$=.svg]` |
| `*=` | Contains | `[text*=warning]` |
| `=/regex/flags` | Regex match | `[id=/^item-\d+/i]` |
| `[attr]` | Exists and truthy | `[disabled]` |
| `[!attr]` | Not exists or falsy | `[!hidden]` |
| `<`, `<=`, `>`, `>=` | Numeric comparison | `[z_index>10]` |

```python
jstyle.update({
    # Regex: IDs starting with "cell-" followed by numbers
    "[id=/^cell-\\d+$/]": {
        "style.border_width": 1,
    },

    # Numeric: high z-index components
    "[z_index>=100]": {
        "style.shadow": {"offset_x": 2, "offset_y": 2, "blur_radius": 5},
    },

    # Starts with
    "[id^=header-]": {
        "style.background_color": "#ecf0f1",
    },

    # Presence check
    "[debug]": {
        "style.border_style": "dotted",
        "style.border_color": "red",
    },
})
```

### Combined Selectors

```python
jstyle.update({
    # Type + attribute
    "Container[style_class=card]": {
        "style.corner_radius": 8,
        "style.shadow": {"offset_x": 2, "offset_y": 2, "blur_radius": 4},
    },

    # Type + ID
    "Text[id=title]": {
        "font_size": 24,
        "font_weight": "bold",
    },

    # Multiple attributes
    "Container[style_class=button][id^=submit-]": {
        "style.background_color": "#27ae60",
    },
})
```

### Nested (Contextual) Selectors

Apply styles to descendants based on ancestor context:

```python
jstyle.update({
    # Styles for sidebar container
    "Container[id=sidebar]": {
        "style.background_color": "#2c3e50",
        "style.padding": (20, 15, 20, 15),

        # Nested: Text inside sidebar
        "Text": {
            "color": "white",
            "font_size": 11,
        },

        # Nested: Buttons inside sidebar
        "Container[style_class=button]": {
            "style.background_color": "#34495e",
            "style.border_color": "#4a6278",
        },
    },

    # Header text is different
    "Container[id=header]": {
        "Text": {
            "color": "#2c3e50",
            "font_size": 16,
            "font_weight": "bold",
        },
    },
})
```

### Specificity

When multiple selectors match, **specificity** determines which wins:

**Order (highest to lowest):**
1. ID selectors (`[id=foo]`)
2. Attribute/class selectors (`[style_class=bar]`, `[attr=val]`)
3. Type selectors (`Container`, `Text`)
4. Wildcard (`*`)

Within the same specificity:
- Later rules override earlier ones
- Contextual (nested) rules override global rules

```python
jstyle.update({
    # Specificity: 0 (type only)
    "Text": {"color": "black"},

    # Specificity: 1 (one attribute)
    "[style_class=highlight]": {"color": "yellow"},

    # Specificity: 1 (one ID - same as attribute for counting, but IDs win)
    "[id=warning]": {"color": "red"},

    # This Text with id=warning and style_class=[highlight]
    # will be RED because ID > attribute
})
```

### Property Setting

Properties can target nested model attributes using dot notation:

```python
jstyle.update({
    "Container": {
        # Direct attribute
        "debug": True,

        # Nested style attribute
        "style.background_color": "#fff",
        "style.border_width": 2,
        "style.padding": (10, 10, 10, 10),

        # Nested transform attribute
        "transform.rotate": 45,
        "transform.scale": (1.5, 1.0),

        # Nested offset attribute
        "offset.absolute": (10, 20),
    },
})
```

### Partial Updates

Update only specific fields without replacing entire objects:

```python
jstyle.update({
    "[id=card]": {
        # Partial style update - only changes specified fields
        "style": {
            "background_color": "white",
            "corner_radius": 8,
            # Other style properties remain unchanged
        },

        # Partial tuple update using None for unchanged indices
        "style.padding": [20, None, 20, None],  # Update top/bottom only
        "style.margin": [5, 10, None, None],    # Update top/right only
    },
})
```

### Temporary Style Contexts

Override styles temporarily using context manager:

```python
# Permanent update
jstyle.update({"Text": {"color": "black"}})

# Temporary override
with jstyle({"Text": {"color": "red"}}):
    root.measure_and_layout(renderer)  # Text will be red
    renderer.render_component(ax, root)

# After context: Text is black again
```

---

## BoxStyle Reference

Complete styling for component backgrounds, borders, and effects.

```python
from jeanplot import BoxStyle, Shadow

style = BoxStyle(
    # Background
    background_color="#ffffff",  # Hex, name, or None

    # Border
    border_color="#333333",
    border_width=2.0,
    border_style="solid",        # "solid", "dashed", "dotted", "custom"
    border_width_mode="data",    # "data" (scales) or "point" (fixed)

    # Custom dash pattern (when border_style="custom")
    dash_sequence=(5.0, 3.0),    # (dash, gap, dash, gap, ...)
    dash_offset=0.0,

    # Spacing (top, right, bottom, left)
    padding=(10, 15, 10, 15),
    margin=(5, 5, 5, 5),

    # Corners
    corner_radius=8.0,           # Rounded corners

    # Shadow
    shadow=Shadow(
        offset_x=3.0,
        offset_y=3.0,
        blur_radius=5.0,
        spread=0.0,
        color="#00000033",       # With alpha
    ),
)
```

**Helper Methods:**

```python
style = BoxStyle(padding=(10, 15, 10, 15))

# Get content box dimensions
content_w, content_h = style.content_box(Size(width=200, height=100))

# Get insets (padding + border)
top, right, bottom, left = style.content_inset()
```

---

## Transforms and Offsets

### Transform

Affine transformations applied after layout:

```python
from jeanplot import Transform

transform = Transform(
    translate=(10, 20),      # Translation in data units
    rotate=45,               # Rotation in degrees
    scale=(1.5, 1.0),        # Scale factors (x, y)
    skew_x=10,               # X-axis skew in degrees
    skew_y=0,                # Y-axis skew in degrees
    rotation_center=(0, 0),  # Center of rotation (relative to component)
)

# Transforms compose via matrix multiplication in parent-child hierarchies
```

### Offset

Position adjustment relative to layout-computed position:

```python
from jeanplot import Offset

# Absolute offset (in data units)
offset = Offset(absolute=(10, 20))

# Relative offset (fraction of own size)
offset = Offset(relative=(0.5, 0.5))  # Center of self

# Reference-relative offset (fraction of parent/target size)
offset = Offset(reference_relative=(1.0, 0.5))  # Right-center of parent

# Combined
offset = Offset(
    absolute=(5, 0),
    relative=(0.5, 0),
    reference_relative=(1.0, 0.5),
)
```

### Attachments

Position a component relative to another:

```python
from jeanplot import Container, AnchorComponent, Offset

# Attach by ID path
tooltip = Container(
    id="tooltip",
    attached_to="//panel/button",  # Path from root
    attachment_offset=Offset(reference_relative=(0.5, 0), relative=(0.5, 1)),
)

# Attach by reference
button = Container(id="button")
popup = Container(
    attached_to=button,
    attachment_offset=Offset(reference_relative=(1.0, 0.5)),
)
```

---

## Connections and Curves

Draw lines and curves between components.

### Connection

```python
from jeanplot import Connection, OrthogonalCurve, Offset, LineEndArrow

connection = Connection(
    id="conn1",

    # Endpoints (by ID path)
    start_component="//source/output",
    end_component="//target/input",

    # Offset from component origin
    start_offset=Offset(reference_relative=(1.0, 0.5)),
    end_offset=Offset(reference_relative=(0.0, 0.5)),

    # Curve type
    curve_type=OrthogonalCurve(
        start_direction="right",
        end_direction="left",
        corner_radius=5,
    ),

    # Styling
    color="#333333",
    line_width=1.5,
    line_style="solid",  # "solid", "dashed", "dotted", "custom"

    # End caps
    start_cap=None,
    end_cap=LineEndArrow(length=8, angle=30, closed=True),

    # Auto-routing (use anchor points if available)
    auto_route=True,
)
```

### Curve Types

```python
from jeanplot import StraightCurve, SimpleBezierCurve, OrthogonalCurve

# Straight line
curve = StraightCurve()

# Cubic Bezier
curve = SimpleBezierCurve(
    start_mode="auto",       # "auto" or "vector"
    end_mode="auto",
    start_vector=(30, 0),    # Control point offset (when mode="vector")
    end_vector=(-30, 0),
    control_strength=0.4,    # For auto mode
)

# Orthogonal (right-angle) path
curve = OrthogonalCurve(
    start_direction="right",  # "up", "down", "left", "right", "auto"
    end_direction="left",
    start_length=15,          # Minimum segment before first turn
    end_length=15,
    corner_radius=5,          # Rounded corners
)
```

### Line End Caps

```python
from jeanplot import LineEndArrow, LineEndCircle, LineEndFlat

# Arrow
arrow = LineEndArrow(
    length=10,
    angle=30,           # Angle in degrees
    closed=True,        # Filled arrow
    fill_color="#333",
    stroke_color="#333",
    stroke_width=1,
)

# Circle
circle = LineEndCircle(
    radius=4,
    fill_color="white",
    stroke_color="#333",
    stroke_width=1,
)

# Flat (T-shaped)
flat = LineEndFlat(
    length=8,
    stroke_color="#333",
    stroke_width=1,
)
```

### Anchor Points

Define connection attachment points on components:

```python
from jeanplot import Container, AnchorComponent, Offset

container = Container(
    id="node",
    anchor_points=[
        AnchorComponent(
            id="input",
            offset=Offset(reference_relative=(0.0, 0.5)),  # Left center
            direction=(-1, 0),  # Points left
            min_segment=10,     # Minimum line length before turning
        ),
        AnchorComponent(
            id="output",
            offset=Offset(reference_relative=(1.0, 0.5)),  # Right center
            direction=(1, 0),   # Points right
            min_segment=10,
        ),
    ],
)
```

---

## Gene Visualization

Jeanplot includes specialized components for synthetic biology schematics.

### Data Models

SBOL-inspired data structures for genetic circuits:

```python
from jeanplot.gene import (
    PartData, TUData, SourceData, InteractionData, CircuitData
)

# Define a genetic part
part = PartData(
    id="pTet",
    name="pTet",
    role="promoter",  # promoter, terminator, cds, regulator, reporter, etc.
    orientation="forward",
)

# Define a transcription unit
tu = TUData(
    id="tu1",
    name="TetR Cassette",
    parts=[
        PartData(id="p1", name="pTet", role="promoter"),
        PartData(id="ern1", name="CasE", role="regulator"),
        PartData(id="t1", name="T1", role="terminator"),
    ],
    source_id="plasmid1",
)

# Define a source (plasmid)
source = SourceData(
    id="plasmid1",
    name="pTetR",
    source_type="plasmid",
    tu_ids=["tu1"],
)

# Define an interaction
interaction = InteractionData(
    id="i1",
    source_tu="tu1",
    source_part="ern1",
    target_tu="tu2",
    target_part="site1",
    interaction_type="inhibition",  # inhibition, activation, cleavage
)

# Complete circuit
circuit = CircuitData(
    transcription_units=[tu1, tu2],
    sources=[source1],
    interactions=[interaction1],
)
```

### GeneticSchematic

Data-driven layout for genetic circuits:

```python
from jeanplot.gene import GeneticSchematic, CircuitData

circuit = CircuitData(...)

schematic = GeneticSchematic.from_circuit(
    circuit,
    grid_gap=(40.0, 20.0),
    orientation="column",
    connection_style="orthogonal",  # "orthogonal", "bezier", "straight"
    show_sources=True,
    show_interactions=True,
)

# Render
renderer = MatplotlibRenderer()
fig, ax = plt.subplots()
renderer.render_component(ax, schematic, adjust_lims=True)
```

### TranscriptionUnit

Container for genetic parts with connecting line:

```python
from jeanplot.gene import TranscriptionUnit, Promoter, ERN, Terminator

tu = TranscriptionUnit(
    id="tu1",
    name="TetR",
    children=[
        Promoter(id="p1", part_name="pTet"),
        ERN(id="ern1", part_name="CasE"),
        Terminator(id="t1", part_name="T1"),
    ],
    line_thickness=1.5,
    line_color="#333",
)
```

### Genetic Parts

| Part | Description |
|------|-------------|
| `Promoter` | Transcription start site |
| `Terminator` | Transcription stop signal |
| `ERN` | Endonuclease recognition element (with anchors) |
| `ERN5pRecog` | 5' recognition site (with anchors) |
| `FluoMarker` | Fluorescent reporter (auto-labeled) |
| `UorfGroup` | Upstream ORF group (auto-labeled) |

```python
from jeanplot.gene import Promoter, ERN, FluoMarker

# Basic part (loads SVG from resources/parts/Promoter.svg)
promoter = Promoter(id="p1", part_name="hEF1a")

# ERN with auto-generated label and vertical anchors
ern = ERN(id="ern1", part_name="CasE")

# Fluorescent marker with color (via jstyle)
fluo = FluoMarker(id="f1", part_name="EYFP")
```

---

## Rendering

### MatplotlibRenderer

Full-featured renderer for matplotlib axes:

```python
from jeanplot import MatplotlibRenderer
import matplotlib.pyplot as plt

renderer = MatplotlibRenderer()

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect("equal")
ax.axis("off")

# Measure, layout, and render
renderer.render_component(ax, root, adjust_lims=True)

# Access after rendering
print(f"Final bounds: {renderer.get_bounds()}")

plt.savefig("output.png", dpi=150, bbox_inches="tight")
```

### SVGRenderer

Generates clean SVG output:

```python
from jeanplot import SVGRenderer

renderer = SVGRenderer()

# Render to string
svg_string = renderer.render_to_string(root)

# Render to file
renderer.render_to_file(root, "output.svg")

# Get SVG element tree
svg_tree = renderer.render_to_element(root)
```

---

## Testing Utilities

Jeanplot includes utilities for testing and visual regression:

```python
from jeanplot import (
    MockRenderer,
    render_to_svg,
    parse_svg,
    get_element_bounds,
    assert_element_position,
    assert_element_size,
    svg_hash,
)

# MockRenderer for geometry tests (no matplotlib)
renderer = MockRenderer()
root.measure_and_layout(renderer)
assert root._dimensions.width > 0

# Render to SVG string for inspection
svg = render_to_svg(root)

# Parse SVG and query elements
tree = parse_svg(svg)
bounds = get_element_bounds(tree, "component-id")

# Assertions
assert_element_position(tree, "header", x=0, y=0, tolerance=0.1)
assert_element_size(tree, "box", width=100, height=50, tolerance=0.1)

# Visual regression with hash
hash1 = svg_hash(render_to_svg(root_v1))
hash2 = svg_hash(render_to_svg(root_v2))
assert hash1 == hash2, "Visual output changed"
```

---

## Theming

Jeanplot loads a default theme from `resources/themes/default.yaml`:

```python
from jeanplot import load_default_theme, jstyle

# Reset to default theme
load_default_theme()

# Check current styles
print(jstyle._raw_styles)

# Override specific styles
jstyle.update({
    "TranscriptionUnit": {
        "line_color": "#666",
        "style.padding": (10, 5, 10, 5),
    },
})
```

### Theme Colors

The default theme defines colors for:

- **Marker colors**: EYFP, EGFP, mNeonGreen, eBFP, mKate, iRFP, mMaroon
- **ERN colors**: CasE (green), Csy4 (orange), PgU (purple)
- **Connection colors**: Standard, ERN interactions

---

## API Reference

### Core Exports

```python
from jeanplot import (
    # Components
    Component, Container, Text, Table, TableRow, TableCell,
    AnchorComponent, Overlay, SVGElement,

    # Layout & Style
    Size, BoxStyle, LayoutConstraints, Offset, Transform, Shadow,
    jstyle,

    # Connections
    Connection, StraightCurve, SimpleBezierCurve, OrthogonalCurve,
    LineEndArrow, LineEndCircle, LineEndFlat,

    # Rendering
    MatplotlibRenderer, SVGRenderer, BaseRenderer,

    # Gene visualization
    GeneticPart, Promoter, Terminator, ERN, ERN5pRecog,
    FluoMarker, UorfGroup, TranscriptionUnit, Source,

    # Testing
    MockRenderer, render_to_svg, parse_svg,

    # Utilities
    load_default_theme, set_debug,
)
```

### Gene Module Exports

```python
from jeanplot.gene import (
    # Data models
    PartData, TUData, SourceData, InteractionData, CircuitData,

    # Components
    GeneticSchematic, TranscriptionUnitRow,
    GeneticPart, Promoter, Terminator, ERN, ERN5pRecog,
    FluoMarker, UorfGroup, TranscriptionUnit, Source,
)
```

---

## Examples

### Dashboard Layout

```python
from jeanplot import Container, Text, BoxStyle, LayoutConstraints, jstyle

jstyle.update({
    "[style_class=card]": {
        "style.background_color": "white",
        "style.corner_radius": 8,
        "style.shadow": {"offset_x": 2, "offset_y": 2, "blur_radius": 6},
        "style.padding": (15, 20, 15, 20),
    },
    "[style_class=card] Text[style_class=title]": {
        "font_size": 14,
        "font_weight": "bold",
        "color": "#2c3e50",
    },
})

card1 = Container(
    style_class=["card"],
    children=[
        Text(text="Revenue", style_class=["title"]),
        Text(text="$12,345", font_size=24, color="#27ae60"),
    ],
    layout=LayoutConstraints(direction="column", gap=8),
)

card2 = Container(
    style_class=["card"],
    children=[
        Text(text="Users", style_class=["title"]),
        Text(text="1,234", font_size=24, color="#3498db"),
    ],
    layout=LayoutConstraints(direction="column", gap=8),
)

dashboard = Container(
    children=[card1, card2],
    layout=LayoutConstraints(direction="row", gap=20),
    style=BoxStyle(padding=(20, 20, 20, 20), background_color="#ecf0f1"),
)
```

### Flow Diagram with Connections

```python
from jeanplot import (
    Container, Text, Connection, OrthogonalCurve,
    AnchorComponent, Offset, LineEndArrow, jstyle
)

jstyle.update({
    "[style_class=node]": {
        "style.background_color": "#3498db",
        "style.corner_radius": 4,
        "style.padding": (10, 20, 10, 20),
    },
    "[style_class=node] Text": {
        "color": "white",
    },
})

def make_node(id, label):
    return Container(
        id=id,
        style_class=["node"],
        children=[Text(text=label)],
        anchor_points=[
            AnchorComponent(
                id="out",
                offset=Offset(reference_relative=(1.0, 0.5)),
                direction=(1, 0),
            ),
            AnchorComponent(
                id="in",
                offset=Offset(reference_relative=(0.0, 0.5)),
                direction=(-1, 0),
            ),
        ],
    )

node_a = make_node("a", "Start")
node_b = make_node("b", "Process")
node_c = make_node("c", "End")

connection1 = Connection(
    start_component="//a/out",
    end_component="//b/in",
    curve_type=OrthogonalCurve(),
    end_cap=LineEndArrow(length=8, angle=30, closed=True),
    auto_route=True,
)

connection2 = Connection(
    start_component="//b/out",
    end_component="//c/in",
    curve_type=OrthogonalCurve(),
    end_cap=LineEndArrow(length=8, angle=30, closed=True),
    auto_route=True,
)

diagram = Container(
    children=[node_a, node_b, node_c, connection1, connection2],
    layout=LayoutConstraints(direction="row", gap=60, align_items="center"),
    style=BoxStyle(padding=(30, 30, 30, 30)),
)
```

### Genetic Circuit

```python
from jeanplot.gene import (
    CircuitData, TUData, PartData, SourceData,
    InteractionData, GeneticSchematic
)

circuit = CircuitData(
    transcription_units=[
        TUData(
            id="tu1", name="Sensor",
            parts=[
                PartData(id="p1", name="pTet", role="promoter"),
                PartData(id="ern1", name="CasE", role="regulator"),
                PartData(id="t1", name="T1", role="terminator"),
            ],
            source_id="src1",
        ),
        TUData(
            id="tu2", name="Reporter",
            parts=[
                PartData(id="p2", name="pConst", role="promoter"),
                PartData(id="site1", name="CasE_site", role="recognition_site"),
                PartData(id="f1", name="EYFP", role="reporter"),
                PartData(id="t2", name="T2", role="terminator"),
            ],
            source_id="src1",
        ),
    ],
    sources=[
        SourceData(id="src1", name="pCircuit", tu_ids=["tu1", "tu2"]),
    ],
    interactions=[
        InteractionData(
            id="i1",
            source_tu="tu1", source_part="ern1",
            target_tu="tu2", target_part="site1",
            interaction_type="cleavage",
        ),
    ],
)

schematic = GeneticSchematic.from_circuit(circuit)

renderer = MatplotlibRenderer()
fig, ax = plt.subplots(figsize=(12, 6))
renderer.render_component(ax, schematic, adjust_lims=True)
plt.savefig("circuit.png", dpi=150, bbox_inches="tight")
```

---

## License

MIT License - see LICENSE file for details.
