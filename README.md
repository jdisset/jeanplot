# Jeanplot
Jeanplot is a declarative, component-based 2D visualization library for Python.
It gives you:
- A retained scene graph (`Component` tree).
- Layout via `Container` and `LayoutConstraints`.
- CSS-like styling via `jstyle` (`JStyle` engine).
- Multiple rendering backends (`matplotlib`, `svg`).
- Domain layer for genetic circuit schematics.
This README is written for beginners first, then gradually becomes a full API reference.
---
## Table of Contents
- [1. Who This Guide Is For](#1-who-this-guide-is-for)
- [2. Installation](#2-installation)
- [3. Quick Start](#3-quick-start)
- [4. Mental Model](#4-mental-model)
- [5. Core Concepts](#5-core-concepts)
- [6. Layout Basics](#6-layout-basics)
- [7. Geometry and Transforms](#7-geometry-and-transforms)
- [8. Rendering Basics](#8-rendering-basics)
- [9. JStyle Deep Dive](#9-jstyle-deep-dive)
- [10. Selector Reference](#10-selector-reference)
- [11. JStyle Recipes](#11-jstyle-recipes)
- [12. API Reference Overview](#12-api-reference-overview)
- [13. API: Data Models](#13-api-data-models)
- [14. API: Components](#14-api-components)
- [15. API: Text](#15-api-text)
- [16. API: SVG](#16-api-svg)
- [17. API: Connections and Curves](#17-api-connections-and-curves)
- [18. API: Table System](#18-api-table-system)
- [19. API: Renderers](#19-api-renderers)
- [20. API: High-Level Render Functions](#20-api-high-level-render-functions)
- [21. API: Testing Utilities](#21-api-testing-utilities)
- [22. API: Gene Data and Components](#22-api-gene-data-and-components)
- [23. Theming](#23-theming)
- [24. End-to-End Examples](#24-end-to-end-examples)
- [25. Troubleshooting](#25-troubleshooting)
- [26. Best Practices](#26-best-practices)
- [27. FAQ](#27-faq)
- [28. Plot Panels and Figures](#28-plot-panels-and-figures)
- [29. API: Plot Data and Foundations](#29-api-plot-data-and-foundations)
- [30. Plot Themes and the `plots.yaml` cascade](#30-plot-themes-and-the-plotsyaml-cascade)
- [31. Compose Helpers and Autofig Templates](#31-compose-helpers-and-autofig-templates)
- [32. `jeanplot-plot` CLI](#32-jeanplot-plot-cli)
- [33. Migration from biocomp-plot](#33-migration-from-biocomp-plot)
---
## 1. Who This Guide Is For
Use this guide if you are:
- New to Jeanplot.
- Comfortable with Python basics.
- Looking for a maintainable way to build visual diagrams.
You do **not** need prior scene graph experience.
---
## 2. Installation
Install from this repository in editable mode:
```bash
pip install -e .
```
Quick import check:
```python
import jeanplot
print("ok")
```
---
## 3. Quick Start
This example builds a tiny layout and renders it using matplotlib.
```python
import matplotlib.pyplot as plt
from jeanplot import (
    Container,
    Text,
    Size,
    BoxStyle,
    LayoutConstraints,
    MatplotlibRenderer,
)
root = Container(
    id="root",
    min_dimensions=Size(280, 120),
    style=BoxStyle(
        background_color="#f4f6f9",
        border_color="#cbd2d9",
        border_width=1,
        padding=(14, 14, 14, 14),
        corner_radius=8,
    ),
    layout=LayoutConstraints(direction="row", gap=10, align_items="center"),
)
left = Container(
    id="left",
    min_dimensions=Size(90, 50),
    style=BoxStyle(background_color="#dbeafe", corner_radius=6),
)
right = Container(
    id="right",
    min_dimensions=Size(130, 50),
    style=BoxStyle(background_color="#dcfce7", corner_radius=6),
    layout=LayoutConstraints(align_items="center", justify_content="center"),
    children=[Text(text="Hello Jeanplot", font_size=10)],
)
root.add_children([left, right])
fig, ax = plt.subplots(figsize=(6, 3), dpi=140)
ax.set_aspect("equal")
ax.axis("off")
renderer = MatplotlibRenderer()
renderer.render_component(ax, root, adjust_lims=True)
plt.show()
```
What happened:
- You created a component tree.
- You gave nodes constraints and style.
- Container layout placed children.
- Renderer measured, laid out, and drew.
---
## 4. Mental Model
Jeanplot uses a retained scene graph pipeline:
1. Build a tree of components.
2. Resolve style rules.
3. Measure natural sizes.
4. Apply constraints and layout.
5. Render with selected backend.
Useful split in your codebase:
- Builders define structure.
- `jstyle` defines look.
- Render wrappers define output flow.
---
## 5. Core Concepts
### 5.1 Component identity
Use `id` whenever possible.
Why:
- Direct selector targeting (`[id=...]`).
- Connection and attachment path references.
- Easier debug and tests.
### 5.2 Style classes
`style_class` is a list of tags:
```python
Container(style_class=["card", "warning"])
```
Use classes for reusable variants.
### 5.3 Overlay nodes
`is_overlay=True` means node does not consume normal layout slot.
Use for labels, markers, annotation icons, and connectors.
### 5.4 Attachments
Any component may attach to another using:
- `attached_to` (component or path string)
- `attachment_offset`
- plus standard `offset`
Attachment is resolved lazily from the scene root.
---
## 6. Layout Basics
Layout is controlled by `LayoutConstraints`:
```python
LayoutConstraints(
    direction="row",
    align_items="start",
    justify_content="start",
    gap=0.0,
    wrap=False,
)
```
### 6.1 `direction`
- `row`: main axis is horizontal.
- `column`: main axis is vertical.
### 6.2 `align_items` (cross axis)
- `start`
- `center`
- `end`
- `stretch`
### 6.3 `justify_content` (main axis)
- `start`
- `center`
- `end`
- `space-between`
- `space-around`
- `space-evenly`
### 6.4 `gap`
Uniform space between layout children.
### 6.5 Padding and margin
Both are `(top, right, bottom, left)` tuples.
- Parent `style.padding` shrinks content box.
- Child `style.margin` participates in layout spacing.
---
## 7. Geometry and Transforms
Geometry is a composition of:
- size constraints,
- offsets,
- transforms.
### 7.1 Size constraints
Each component has:
- `min_dimensions`
- `max_dimensions`
Natural size is clamped between those bounds.
### 7.2 Offsets
`Offset` has 3 additive terms per axis:
`delta = self_size * relative + ref_size * reference_relative + absolute`
Fields:
- `relative`
- `reference_relative`
- `absolute`
### 7.3 Transform
`Transform` fields:
- `translate`
- `rotate` (degrees)
- `scale`
- `skew_x`
- `skew_y`
- `rotation_center`
Result is an affine 3x3 matrix.
---
## 8. Rendering Basics
Two backend classes:
- `MatplotlibRenderer`
- `SVGRenderer`
Two high-level helpers:
- `render(...)`
- `render_to_string(...)`
Quick SVG string render:
```python
from jeanplot import Container, Size
from jeanplot.render import render
root = Container(id="root", min_dimensions=Size(100, 50))
svg = render(root, backend="svg")
print(type(svg))  # str
```
Quick matplotlib render:
```python
import matplotlib.pyplot as plt
from jeanplot import MatplotlibRenderer
fig, ax = plt.subplots()
MatplotlibRenderer().render_component(ax, root)
plt.show()
```
---
## 9. JStyle Deep Dive
This section is intentionally detailed.
`jstyle` is a global instance of `JStyle`.
It gives you:
- Selector parsing and matching.
- Rule storage and deep merge updates.
- Property resolution precedence.
- Nested contextual rules.
- Recursive apply (`apply`) and single-node apply (`apply_one`).
- Temporary scoped styles (`context`).
### 9.1 Why JStyle matters
Without style engine, constructors get noisy and duplicated.
With JStyle:
- You define style once.
- You avoid repeated literals.
- You can retheme whole graphs fast.
- You enforce visual SSOT.
### 9.2 Style dictionary shape
Top-level keys are selectors.
Values are declaration dictionaries.
Declarations can contain:
- property entries (like `"style.border_width": 1`)
- nested selector blocks
- or both
Example:
```python
from jeanplot import jstyle
jstyle.clear()
jstyle.update({
    "Container": {
        "style.corner_radius": 6,
        "style.padding": (8, 8, 8, 8),
    },
    "[id=header]": {
        "style.background_color": "#243b53",
        "Text": {
            "color": "#ffffff",
            "font_weight": "bold",
        },
    },
})
```
### 9.3 `update` semantics
`jstyle.update(value)` **replaces** the active cascade.
Two input forms are accepted:
- a nested style dict (flattened and wrapped internally),
- a `CallableSymbol` from a `!cascade:jstyle` YAML document.
To clear:
```python
jstyle.clear()
```
Under the hood the engine is a thin layer over dracon's
`!cascade:jstyle` select-mode dialect. Selector matching, specificity
ordering, and multi-rule merging are delegated to dracon. The same
selector grammar (id, class, type, descendant, nested) applies in both
forms.
### 9.4 `apply` vs `apply_one`
- `apply(component)`: style component + descendants recursively.
- `apply_one(component)`: style one component only.
Current component layout pipeline uses `apply_one` per node during measure/layout.
### 9.5 How winners are chosen
For each property path, JStyle gathers matching candidates and sorts by:
1. selector specificity,
2. context-rule status,
3. MRO match level,
4. source order.
More specific and later declarations win.
### 9.6 Specificity tuple
Specificity is:
`(id_count, attr_class_count, type_count)`
Examples:
- `Container` -> `(0, 0, 1)`
- `[style_class=card]` -> `(0, 1, 0)`
- `[id=hero]` -> `(1, 0, 0)`
- `Text[id=title]` -> `(1, 0, 1)`
### 9.7 Nested (contextual) rules
Nested selector keys inside a rule create context-dependent descendants rules.
Example:
```python
jstyle.update({
    "[id=dialog]": {
        "style.background_color": "#ffffff",
        "Text": {
            "color": "#102a43",
        },
        "[style_class=danger]": {
            "style.border_color": "#d64545",
        },
    }
})
```
Interpretation:
- If ancestor matches `[id=dialog]`, descendants may pick nested rules from this block.
### 9.8 Property paths
Property entries use dotted paths:
- `style.background_color`
- `layout.gap`
- `transform.scale`
- `offset.absolute`
JStyle traverses attributes and sets terminal field.
### 9.9 Updating nested models
If target field is a Pydantic model and incoming value is dict, JStyle merges that model.
Example:
```python
jstyle.update({
    "Container": {
        "style": {
            "background_color": "#f5f7fa",
            "border_width": 1,
        }
    }
})
```
### 9.10 Sequence partial updates
For tuple/list fields, update with list values.
`None` means keep existing index.
```python
jstyle.update({
    "Container": {
        "style.padding": [12, None, 12, None]
    }
})
```
### 9.11 Temporary style context
```python
with jstyle.context({
    "Text": {"color": "#d64545"}
}):
    ...
```
State is restored after block.
`jstyle(...)` is alias to same context manager.
### 9.12 Typical style lifecycle in app code
```python
jstyle.clear()
jstyle.update(BASE_THEME)
jstyle.update(FEATURE_THEME)
jstyle.update(PAGE_OVERRIDES)
```
### 9.13 JStyle best practices
Do:
- Keep style dictionaries centralized.
- Use class selectors for variants.
- Use id selectors only for one-off nodes.
- Reset styles in tests.
Do not:
- Scatter `update()` calls across unrelated modules.
- Hide critical layout logic in deeply nested style blocks.
- Depend on accidental selector conflicts.
---
## 10. Selector Reference
### 10.1 Type selector
```python
"Container"
"Text"
"SVGElement"
```
Matches class name through MRO.
### 10.2 Wildcard
```python
"*"
```
Matches any component.
### 10.3 Attribute selector
General form:
`[attribute operator value]`
Presence forms:
- `[attribute]`
- `[!attribute]`
### 10.4 Operators
- `=` exact
- `!=` not exact
- `=~` case-insensitive exact
- `^=` starts with
- `$=` ends with
- `*=` contains
- `=/` regex
- `<` numeric less
- `<=` numeric less/equal
- `>` numeric greater
- `>=` numeric greater/equal
### 10.5 Dot-path attributes
```python
"[style.border_width>0]"
"[layout.direction=row]"
```
### 10.6 Regex selector example
```python
"[id=/^node_\\d+$/]"
```
### 10.7 List field behavior
When actual field is list/tuple/set, value operators test each item.
Example:
```python
"[style_class=warning]"
```
---
## 11. JStyle Recipes
### 11.1 App defaults
```python
jstyle.clear()
jstyle.update({
    "Container": {
        "style.background_color": "#ffffff",
        "style.border_color": "#d9e2ec",
        "style.border_width": 1,
        "style.corner_radius": 6,
    },
    "Text": {
        "color": "#102a43",
        "font_size": 10,
    },
})
```
### 11.2 Variant classes
```python
jstyle.update({
    "[style_class=card]": {
        "style.padding": (12, 12, 12, 12),
    },
    "[style_class=danger]": {
        "style.border_color": "#d64545",
        "Text": {"color": "#8a1c1c"},
    },
})
```
### 11.3 Context-specific descendants
```python
jstyle.update({
    "[id=sidebar]": {
        "Text": {"color": "#f0f4f8"},
        "[style_class=menu-item]": {
            "font_size": 9,
        },
    }
})
```
### 11.4 Temporary export mode
```python
with jstyle.context({
    "Text": {
        "font_size_mode": "points",
        "font_size": 11,
    }
}):
    svg = render(root, backend="svg")
```
---
## 12. API Reference Overview
Top-level imports from `jeanplot` include:
- Core: `Component`, `Container`, `Overlay`, `AnchorComponent`
- Text and SVG: `Text`, `SVGElement`
- Connectors: `Connection`, `StraightCurve`, `SimpleBezierCurve`, `OrthogonalCurve`
- Geometry/style models: `Size`, `Offset`, `Transform`, `BoxStyle`, `LayoutConstraints`, `Shadow`
- Table types: `Table`, `TableRow`, `TableCell`
- Caps: `LineEndArrow`, `LineEndCircle`, `LineEndFlat`
- Renderers: `BaseRenderer`, `MatplotlibRenderer`, `SVGRenderer`
- Styling: `jstyle`
- Rendering helpers: `render`, `render_to_string`
- Testing helpers: `MockRenderer`, `render_to_svg`, `parse_svg`, `get_element_bounds`, `assert_element_position`, `assert_element_size`, `svg_hash`
- Gene layer: `GeneticSchematic`, `TranscriptionUnit`, `Promoter`, `Terminator`, `ERN`, `ERN5pRecog`, `FluoMarker`, `UorfGroup`, `Source`, `CircuitData`, `TUData`, `PartData`, `SourceData`, `InteractionData`
- Theme loader: `load_default_theme`
---
## 13. API: Data Models
### 13.1 `Size`
Fields:
- `width: float`
- `height: float`
Methods:
- `union(other)`
- `Size.min(a, b)`
- `Size.max(a, b)`
### 13.2 `Offset`
Fields:
- `relative`
- `reference_relative`
- `absolute`
Method:
- `compute(self_dims, reference_dims=None) -> (dx, dy)`
Notes:
- Extra fields are forbidden.
- Legacy alias `parent_relative` is not supported.
### 13.3 `Transform`
Fields:
- `translate`
- `rotate`
- `scale`
- `skew_x`
- `skew_y`
- `rotation_center`
Method:
- `to_matrix(dimensions)`
### 13.4 `Shadow`
Fields:
- `offset_x`
- `offset_y`
- `blur_radius`
- `spread`
- `color`
- `resolution`
### 13.5 `BoxStyle`
Main fields:
- `background_color`
- `border_color`
- `border_width`
- `border_width_mode` (`point|data`)
- `border_style`
- `dash_sequence`
- `dash_offset`
- `corner_radius`
- `margin` `(t, r, b, l)`
- `padding` `(t, r, b, l)`
- `shadow`
Methods:
- `content_inset()`
- `content_box(bounds)`
### 13.6 `LayoutConstraints`
Fields:
- `direction`
- `align_items`
- `justify_content`
- `gap`
- `wrap`
### 13.7 `TextMetrics`
Internal measurement cache model for text rendering.
---
## 14. API: Components
### 14.1 `Component`
Important fields:
- `id`
- `style_class`
- `show`
- `debug`
- `is_overlay`
- `z_index`
- `min_dimensions`
- `max_dimensions`
- `transform`
- `offset`
- `attached_to`
- `attachment_offset`
- `anchor_points`
- `style`
- `opacity`
- `parent`
Important methods:
- `measure_and_layout(renderer=None)`
- `render(renderer, context, matrix)`
- `compute_local_matrix()`
- `compute_world_matrix(parent_world_matrix=None)`
- `get_world_origin()`
- `get_world_bounds()`
- `add_child(...)`
- `add_children(...)`
- `add_renderer_option(...)`
- `get_renderer_options(...)`
### 14.2 `Container`
Adds:
- `children: list[Component]`
- `layout: LayoutConstraints`
Behavior:
- Computes natural dimensions from layout children.
- Handles overlays and attached children separately.
- Positions layout children using constraints + margins + alignment.
- Renders visible children by `z_index`.
### 14.3 `Overlay`
`Component` subclass with default `is_overlay=True`.
### 14.4 `AnchorComponent`
Small anchor for connection routing.
Useful fields:
- `direction`
- `min_segment`
- tiny fixed dimensions
---
## 15. API: Text
### 15.1 `Text`
Fields:
- `text`
- `font_name`
- `font_size`
- `font_size_mode` (`points|data`)
- `font_weight` (`normal|bold`)
- `font_style` (`normal|italic`)
- `color`
- `align` (`left|center|right`)
- `vertical_align` (`top|middle|bottom|baseline`)
- `line_spacing`
- `render_as_path` (`auto|True|False`)
Default note:
- `font_size_mode` default is `"data"`.
Interpretation:
- `data`: text size scales with world transform and axis scale.
- `points`: text size is fixed visual point size.
---
## 16. API: SVG
### 16.1 `SVGPathData`
Immutable path model with fields:
- `d`
- `fill`
- `stroke`
- `stroke_width`
- `transform`
- `line_style`
- `dash_array`
- `dash_offset`
### 16.2 `SVGContent`
Immutable parsed/generated SVG content:
- `width`
- `height`
- `viewBox`
- `paths`
### 16.3 Line end cap models
- `LineEndArrow`
- `LineEndCircle`
- `LineEndFlat`
### 16.4 `SVGElement`
Fields:
- `svg_content: str | Path | bytes | SVGContent | None`
- `color_remap: dict[str, str | None]`
- `line_width_mode: "point" | "data"`
Behavior:
- Parses and validates SVG lazily.
- Uses parsed dimensions when dimensions are not set.
### 16.5 SVG helper functions
In `jeanplot.core.svg`:
- `get_svg_data(source, ppi=72.0)`
- `make_svg_line(width, thickness, color)`
- `create_arrow_cap(...)`
- `create_circle_cap(...)`
- `create_flat_cap(...)`
---
## 17. API: Connections and Curves
### 17.1 `Connection`
Required fields:
- `start_component: str | Component`
- `end_component: str | Component`
Common fields:
- `start_offset`, `end_offset`
- `color`
- `line_width`
- `linewidth_mode`
- `curve_type`
- `line_style`
- `dash_array`
- `dash_offset`
- `start_cap`, `end_cap`
- `auto_route`
Behavior:
- Resolves endpoint references.
- Optionally auto-selects best anchor pair.
- Computes path in render phase.
### 17.2 Curves
All implement `CurveDefinition`.
`CurveDefinition` API:
- `get_path(start, end, local_checkpoints=None)`
- `get_directions(start, end, control_points)`
Available curve classes:
- `StraightCurve`
- `SimpleBezierCurve`
- `OrthogonalCurve`
#### `StraightCurve`
Simple line segment.
#### `SimpleBezierCurve`
Single cubic bezier.
Fields:
- `start_mode`, `end_mode`
- `start_vector`, `end_vector`
- `auto_direction_strength`
#### `OrthogonalCurve`
Axis-aligned segmented route.
Fields:
- `start_direction`, `end_direction` (`up|down|left|right|auto`)
- `start_length`, `end_length`
- `corner_radius`
- `auto_simplify`
Utility:
- `get_direction_from_vector(vector)`
---
## 18. API: Table System
Main classes:
- `CellStyle`
- `ColumnStyle`
- `TableCell`
- `TableRow`
- `Table`
### 18.1 `CellStyle`
Extends `BoxStyle` with:
- side border toggles (`border_top`, etc.)
- width/height bounds (`min_width`, etc.)
- optional layout overrides (`align_items`, `justify_content`)
### 18.2 `ColumnStyle`
Fields:
- `width`: float, `"auto"`, or percentage string like `"30%"`
- `cell_style`
### 18.3 `TableCell`
Fields:
- `style: CellStyle`
- `colspan`
Behavior:
- Uses parent-computed column widths.
- Supports partial side border rendering.
- Renders children with normal world-matrix child flow.
### 18.4 `Table`
Fields:
- `data`
- `column_styles`
- `header_rows`
- `border_collapse` (`collapse|separate`)
- `border_spacing`
Behavior summary:
- Builds rows/cells from data.
- Computes natural widths.
- Calculates final per-column widths.
- Applies widths and relayout.
---
## 19. API: Renderers
### 19.1 `BaseRenderer`
Abstract interface methods:
- `create_context(...)`
- `render_component(...)`
- `render_to_output(...)`
- `render_rectangle(...)`
- `render_svg(...)`
- `render_path(...)`
- `render_text(...)`
- `render_debug(...)`
- `measure_text(...)`
Hooks:
- `add_pre_render_callback(cb)`
- `add_post_render_callback(cb)`
### 19.2 `MatplotlibRenderer`
Constructor:
```python
MatplotlibRenderer(debug=False, force_native_text=False)
```
`create_context` supports:
- `width`, `height`, `dpi`
- existing `ax` reuse
Highlights:
- tracks data-unit linewidths and text font sizes,
- refreshes on draw events,
- can auto-adjust limits to content bounds,
- supports native text and path text fallback.
### 19.3 `SVGRenderer`
Constructor:
```python
SVGRenderer(debug=False)
```
Methods:
- `create_context(width=800, height=600)`
- `render_component(...)`
- `render_to_string(component)`
- `render_to_output(context, output=None)`
Output targets:
- `None` -> return SVG string
- file path string -> write file
- text or binary stream -> write accordingly
---
## 20. API: High-Level Render Functions
Module `jeanplot.render`:
### 20.1 `render(...)`
```python
render(
    component,
    *,
    backend="matplotlib",    # "matplotlib" | "svg"
    context=None,
    width=800,
    height=600,
    adjust_lims=True,
    output=None,
    renderer_kwargs=None,
    context_kwargs=None,
)
```
Return shape:
- matplotlib backend without `output`: axes/context
- svg backend without `output`: SVG string
- with `output`: context
### 20.2 `render_to_string(component, width=800, height=600)`
Returns SVG string.
---
## 21. API: Testing Utilities
Module `jeanplot.testing` provides:
- `MockRenderer`
- `render_to_svg(component)`
- `parse_svg(svg_str)`
- `get_element_bounds(svg_elem, element_id)`
- `get_element_position(svg_elem, element_id)`
- `assert_element_position(svg_str, element_id, x, y, tol=0.1)`
- `assert_element_size(svg_str, element_id, width, height, tol=0.1)`
- `assert_elements_connected(svg_str, start_id, end_id)`
- `normalize_svg(svg_str)`
- `svg_hash(svg_str)`
Use these helpers to keep output tests stable and readable.
---
## 22. API: Gene Data and Components
### 22.1 Data models (`jeanplot.gene.data`)
#### `PartData`
Fields:
- `id`
- `name`
- `role`
- `orientation`
- `sequence`
#### `TUData`
Fields:
- `id`
- `name`
- `parts`
- `source_id`
- `position`
- `ratio_percent`
- `disabled`
#### `SourceData`
Fields:
- `id`
- `name`
- `source_type`
- `tu_ids`
- `ratios`
- `marker`
#### `InteractionData`
Fields:
- `id`
- `source_tu`
- `source_part`
- `target_tu`
- `target_part`
- `interaction_type`
#### `CircuitData`
Fields:
- `transcription_units`
- `sources`
- `interactions`
- `metadata`
### 22.2 Gene visual components (`jeanplot.gene.elements`)
#### `TranscriptionUnit`
Container for parts with optional name/ratio labels and line backbone.
#### `Source`
Source marker/tag visual container.
#### `GeneticPart`
Base class for SVG-backed part visuals.
Factory:
- `GeneticPart.from_data(part_data)`
#### Concrete parts
- `Promoter`
- `Terminator`
- `ERN`
- `ERN5pRecog`
- `FluoMarker`
- `UorfGroup`
### 22.3 Schematic (`jeanplot.gene.schematic`)
#### `GeneticSchematic`
Fields:
- `data`
- `grid_gap`
- `orientation`
- `connection_style`
- `show_sources`
- `show_interactions`
Factory:
- `GeneticSchematic.from_circuit(circuit, **kwargs)`
Helpers:
- `SourceAnnotation`
- `TranscriptionUnitRow`
---
## 23. Theming
Load built-in default theme:
```python
from jeanplot import load_default_theme
load_default_theme()
```
Current behavior:
- loads `pkg:jeanplot:resources/themes/default.yaml`
- resolves lazy values
- clears existing `jstyle`
- updates `jstyle` with default rules
CLI check:
```bash
python -m jeanplot.cli theme-check
```
---
## 24. End-to-End Examples
### 24.1 Styled cards with JStyle
```python
import matplotlib.pyplot as plt
from jeanplot import (
    Container,
    Text,
    Size,
    BoxStyle,
    LayoutConstraints,
    MatplotlibRenderer,
    jstyle,
)
jstyle.clear()
jstyle.update({
    "Container[style_class=card]": {
        "style.background_color": "#ffffff",
        "style.border_color": "#d9e2ec",
        "style.border_width": 1,
        "style.corner_radius": 8,
        "style.padding": (10, 10, 10, 10),
    },
    "Text[style_class=title]": {
        "font_size": 10,
        "font_weight": "bold",
        "color": "#102a43",
    },
    "Text[style_class=value]": {
        "font_size": 14,
        "font_weight": "bold",
        "color": "#334e68",
    },
})
def card(cid, title, value):
    return Container(
        id=cid,
        style_class=["card"],
        min_dimensions=Size(140, 90),
        layout=LayoutConstraints(direction="column", gap=6),
        children=[
            Text(text=title, style_class=["title"]),
            Text(text=value, style_class=["value"]),
        ],
    )
root = Container(
    id="dashboard",
    min_dimensions=Size(500, 150),
    style=BoxStyle(background_color="#f5f7fa", padding=(12, 12, 12, 12)),
    layout=LayoutConstraints(direction="row", gap=12, align_items="center"),
    children=[
        card("c1", "Users", "12,304"),
        card("c2", "Latency", "84 ms"),
        card("c3", "Errors", "0.12%"),
    ],
)
fig, ax = plt.subplots(figsize=(10, 3), dpi=140)
ax.set_aspect("equal")
ax.axis("off")
MatplotlibRenderer().render_component(ax, root)
plt.show()
```
### 24.2 Connection routing
```python
import matplotlib.pyplot as plt
from jeanplot import (
    Container,
    Connection,
    Size,
    Offset,
    BoxStyle,
    OrthogonalCurve,
    LineEndArrow,
    MatplotlibRenderer,
)
n1 = Container(
    id="n1",
    min_dimensions=Size(100, 50),
    offset=Offset(absolute=(20, 40)),
    style=BoxStyle(background_color="#dbeafe", border_color="#93c5fd", border_width=1),
)
n2 = Container(
    id="n2",
    min_dimensions=Size(100, 50),
    offset=Offset(absolute=(260, 140)),
    style=BoxStyle(background_color="#dcfce7", border_color="#86efac", border_width=1),
)
conn = Connection(
    id="flow",
    start_component="n1",
    end_component="n2",
    curve_type=OrthogonalCurve(corner_radius=8),
    line_width=1.5,
    color="#334155",
    end_cap=LineEndArrow(closed=True, fill_color="#334155", stroke_color="#334155"),
)
root = Container(id="root", min_dimensions=Size(420, 260), children=[n1, n2, conn])
fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
ax.set_aspect("equal")
ax.axis("off")
MatplotlibRenderer().render_component(ax, root)
plt.show()
```
### 24.3 High-level SVG export
```python
from jeanplot import Container, Size
from jeanplot.render import render
root = Container(id="root", min_dimensions=Size(120, 60))
svg = render(root, backend="svg")
with open("out.svg", "w", encoding="utf-8") as f:
    f.write(svg)
```
## 25. Troubleshooting
### 25.1 Nothing renders
Check:
- `show=True`.
- non-zero dimensions.
- correct context passed to renderer.
### 25.2 Style not applied
Check:
- selector syntax.
- selector specificity conflicts.
- rule exists after updates.
- `jstyle.clear()` not called unexpectedly.
### 25.3 Text appears too big/small
Check `font_size_mode`:
- `data`: tied to world/axis scale.
- `points`: visual point size constant.
### 25.4 Connection not visible
Check:
- endpoint references resolve,
- line width and color not zero/none,
- endpoint nodes have valid dimensions.
### 25.5 SVG stream output issues
`SVGRenderer.render_to_output` supports both text and binary streams.
### 25.6 Table width surprises
Check:
- `column_styles` and data shape,
- valid percent strings,
- colspans and header setup.
---
## 26. Best Practices
### 26.1 Separate structure from style
- Builders create trees.
- `jstyle` controls look.
### 26.2 Use classes for variants
Prefer class selectors (`style_class`) over many id-specific rules.
### 26.3 Reset style state in tests
Use `jstyle.clear()` at test boundaries.
### 26.4 Be explicit about text sizing mode
Do not rely on implicit defaults when visual consistency matters.
### 26.5 Keep ids stable
Stable ids improve:
- styling,
- connectors,
- tests.
### 26.6 Prefer high-level render API in app code
Use backend classes directly only when backend-specific controls are needed.
---
## 27. FAQ
### Is Jeanplot retained mode?
Yes. You keep a component tree and render from it.
### Does JStyle mutate components?
Yes, by setting fields on matched components during apply/layout.
### Can I render same tree to matplotlib and SVG?
Yes.
### Can I theme using YAML?
Yes, via `load_default_theme()`.
### Is there a built-in query language for tree traversal?
Not full DOM query. Use your object references and path-based helpers.
### What is the default text sizing mode?
`Text.font_size_mode` defaults to `"data"`.
### How do I fully reset styles?
`jstyle.clear()`.
---
## 28. Plot Panels and Figures
Jeanplot ships a scientific-plotting layer built on the same Component
machinery as the scene graph. A **plot panel** is a Component that
claims a matplotlib axes from its laid-out bbox and draws into it. A
**Figure** is a Container whose tree contains panels; the renderer
walks the tree, allocates sub-axes per leaf panel, and styles
everything via `jstyle`.
### 28.1 Mental model
- `PlotPanel(Container)` — base class. Subclasses implement
  `draw(ax)`; an optional `render_txt() -> str | None` enables
  terminal-friendly ASCII output.
- `Figure(Container)` — 5-field Container with `theme`, `size`,
  `dpi`, `panels`, `metadata`. No custom render method; the renderer
  walks `children`.
- `Colorbar`, identity lines, slice chords — uniform overlay
  children with `is_overlay=True`. Same layout/style machinery.
- One drawing method per panel (`draw(ax)`); SVG/PNG export reuses
  the matplotlib path.
### 28.2 Concrete panels
Module `jeanplot.panels`:
- `SmoothPanel1D`, `SmoothPanel2D`, `SmoothPanel3D` — KNN-smoothed
  surfaces. 3D is a Container holding a cube view + slice grid.
- `MVPPanel` — measured-vs-predicted scatter.
- `DensityPanel` — kernel density estimate.
- `ScatterPanel` — plain scatter.
- `ViolinPanel`, `ParticlePanel`, `StackedPolyPanel`.
- `AsciiHeatmapPanel` — terminal output.
- `AutoPanel` — dim-dispatch wrapper.
Each subclass declares its kwargs as typed Pydantic fields (no
`**kwargs` opacity) and is registered in `DEFAULT_TYPES` so dracon
`!SmoothPanel2D { ... }` tags resolve.
### 28.3 Minimal example
```python
import numpy as np
from jeanplot import Figure, SmoothPanel2D, PlotData, render
x = np.random.randn(500)
y = np.random.randn(500)
z = np.exp(-(x**2 + y**2))
data = PlotData(X=np.column_stack([x, y]), Y=z[:, None],
                column_names=["x", "y"], output_names=["z"])
fig = Figure(panels=[SmoothPanel2D(data=data)])
render(fig, output="smooth.png")
```
### 28.4 AutoPanel dim-dispatch
`AutoPanel(data=...)` selects `SmoothPanel1D` / `SmoothPanel2D` /
`SmoothPanel3D` from `data.X.shape[1]`. The YAML form is the dracon
Constructor Slots `!fn` template at
`pkg:jeanplot:resources/templates/auto_panel.yaml`. The Python form is
the `auto_panel(data, **kwargs)` helper in `jeanplot.panels`.
---
## 29. API: Plot Data and Foundations
Module `jeanplot.data`:
- `PlotData` — typed Pydantic model with `X`, `Y`, `column_names`,
  `output_names`, `metadata`. `column_proteins` is a back-compat
  alias for `column_names`.
- `LazyPlotData` — same shape, arrays loaded on first access via
  `model_validator(mode='after')`.
- `PlotFunctionResult` — return shape for `draw(ax)` paths that
  need to communicate `mappable` (colorbar source) upward.
- `GridData` — gridded summary with base64 round-trip.
- `Rescaler` — Protocol with `fwd(x)` / `inv(x)`. `IdentityRescaler`
  is the default; biocomp's `DataRescaler` already satisfies it.
Module `jeanplot.knn`:
- KNN tree backends (usearch / pykdtree / scipy auto-selected).
- Density estimators, Gaussian-weighted KNN with optional numba
  acceleration, optional JAX kernels.
Module `jeanplot.color`:
- `load_palette(name)` / palette registration.
- `closest_name(query)` for fuzzy color-name matching.
- Bio palettes ship in `pkg:jeanplot:resources/colors/bio_palettes.yaml`.
Module `jeanplot.stats`:
- `rmse`, `mse`, `mae`, `r_squared`, `pearson_r` (numpy).
---
## 30. Plot Themes and the `plots.yaml` cascade
Module-resource path: `pkg:jeanplot:resources/themes/plots.yaml`.
This is the SSOT for every plot default. It's a `!cascade:jstyle`
document — each rule key is a selector targeting a panel class or
class+attribute, and the body sets typed properties:
```yaml
rules: !cascade:jstyle
  SmoothPanel2D:
    vlims: [0.0, 1.0]
    cmap: viridis
  SmoothPanel2D[style_class=overlay]:
    alpha: 0.5
  Colorbar:
    tick_props:
      labelsize: 8
```
Two presets layer on top:
- `themes/paper.yaml` — paper-formatted preset.
- `themes/rcparams.yaml` — matplotlib rcParams snapshot.
Load via `load_plot_theme()` (in `jeanplot`); compose with the figure
template via `Figure.theme: !include pkg:jeanplot:resources/themes/plots.yaml@rules`
inside a YAML job (selector form pulls just the cascade subtree).
Top-level theme vars surface as `--name` CLI flags on `jeanplot-plot`
via dracon's `!require` + mapping-body / `!set_default` pattern.
---
## 31. Compose Helpers and Autofig Templates
Module `jeanplot.compose` carries small tree-construction helpers,
all registered as dracon `!fn` templates so they're usable from YAML:
- `panel_row(panels, **kwargs)` — Container with row direction.
- `panel_grid(panels, ncols, **kwargs)` — grid layout.
- `panels_from_datas(datas, kind=AutoPanel, **kwargs)`.
- `build_figure_metadata(...)`.
- `default_output_name(...)`.
YAML examples and autofig figure templates live under
`pkg:jeanplot:resources/figures/`:
- `data.yaml` — data-only multi-panel figure.
- `pred_combined.yaml` — measured-vs-predicted combined figure.
- `combined.yaml` — combined data + prediction.
- `templates.yaml` — higher-order figure templates (`ComparePair`,
  `Triple`, ...) as `!fn` blocks. Adding a shape = one `!fn` entry.
The compose helpers replace ~1000 LOC of `expand_panel_atomics` /
`compose_atomics` / `subdivide` / `axnum` machinery — nested
Containers express the same shapes.
---
## 32. `jeanplot-plot` CLI
Entry point: `jeanplot-plot` (installed by `pip install -e .`).
Backed by `PlotJob` in `jeanplot.cli` — a `@dracon_program`-shaped
Pydantic Job that takes `figure: Figure` (typed object, not a path)
and renders. Programs accept objects, not paths — `!include` does
the wiring.
```bash
jeanplot-plot +path/to/figure.yaml --output-dir out/
jeanplot-plot +mytheme --vlim-low -1 --vlim-high 1   # vocab-as-CLI
```
Run from Python via `PlotJob.invoke(...)` or
`PlotJob.from_config(...).run()` — same shape as biocomp's program
classes. The `from_config` form returns the constructed job for
inspection before run.
---
## 33. Migration from biocomp-plot
See `docs/migrating_from_biocomp.md` for the full cheatsheet. Key
mappings:
- `PlotConfig` → `Figure(Container)` attrs + jstyle rules.
- `PlotTask` → a `PlotPanel(Container)` with overlay children.
- `tasks/{auto,1D,2D,3D}.yaml` → `AutoPanel` + Constructor Slots.
- `default_plotconf_v2.yaml` `callstack_params` → `themes/plots.yaml`.
- `Overlay` protocol → child Components with `is_overlay=True`.
Biocomp itself is untouched by the refactor: `biocomp-plot` keeps
running every existing job. Migration is opt-in, per script.
---
