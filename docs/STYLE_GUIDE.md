# Jeanplot Style System Guide

This module provides a styling system (`jstyle`) inspired by CSS, allowing
declarative styling of components based on their type, attributes, and
position within the component hierarchy.

## Core Features

### 1. Selectors

Target components for styling.

#### Type Selectors
Match component class names (e.g., `Container`, `Text`).
Inheritance is respected; a rule for `Component` applies to `Container`.

```python
jstyle.update({
    "Container": { "style.background_color": "#f0f0f0" },
    "Text": { "font_size": 10 }
})
```

#### ID Selectors
Match the specific `id` attribute (e.g., `[id=my-button]`).

```python
jstyle.update({ "[id=special]": { "color": "red" } })
```

#### Style Class Selectors
Match components containing a specific class in their `style_class` list.

```python
jstyle.update({ "[style_class=primary]": { "style.border_color": "blue" } })
```

#### Attribute Selectors
Match components based on any attribute value using various operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Exact match | `[name=foo]`, `[size=10]` |
| `!=` | Not equal | `[status!=error]` |
| `~=` | Case-insensitive match | `[label=~submit]` |
| `^=` | Starts with | `[id^=item-]` |
| `$=` | Ends with | `[filename$=.png]` |
| `*=` | Contains substring | `[text*='important']` |
| `=/regex/flags` | Regular expression match | `[name=/^foo/i]` |
| `[attr]` | Presence check (attribute exists and is truthy) | `[disabled]` |
| `<`, `<=`, `>`, `>=` | Numeric comparison | `[value>100]` |

```python
jstyle.update({
    "Button[status=active]": { "opacity": 1.0 },
    "Image[filename$=.jpg]": { "style.border_radius": 4 },
    "Component[debug]": { "style.border_style": "dotted" }
})
```

#### Combined Selectors
Combine type and attribute selectors.

```python
jstyle.update({
    "Text[style_class=error]": { "color": "red", "font_weight": "bold" }
})
```

#### Wildcard Selector
`*` matches any component (lowest specificity).

```python
jstyle.update({ "*": { "debug": False } })
```

### 2. Specificity

Determines which rule applies when multiple selectors match.

**Order:** ID > Attribute/Class > Type > Wildcard

Within the same specificity level, rules defined later (or added via `update()`)
take precedence. Contextual rules override global rules of the same specificity.

### 3. Context (Nested Rules)

Apply styles to descendants based on their ancestor. Rules nested inside another
rule's dictionary only apply if the outer selector matches an ancestor and the
inner selector matches the descendant.

```python
jstyle.update({
    "Container": {  # applies to all containers
        "style.padding": (10, 10, 10, 10),
        "Text": {  # applies to Text inside any Container
            "color": "darkgray"
        }
    },
    "Container[id=sidebar]": {  # specific container
        "style.background_color": "#eee",
        "Text": {  # applies to Text inside the sidebar ONLY
            "color": "black",
            "font_size": 9
        },
        "Button[style_class=primary]": {  # button inside sidebar
            "style.background_color": "blue"
        }
    }
})
```

### 4. Inheritance (MRO)

Rules targeting a base class (e.g., `Component`) apply to subclasses
(e.g., `Container`), unless overridden by a more specific rule for the subclass.

### 5. Property Setting

#### Direct Attributes
Set top-level component attributes.

```python
{ "color": "red" }
```

#### Nested Attributes
Use dot notation to set attributes on nested Pydantic models like `style` or `transform`.

```python
{ "style.background_color": "#fff" }
```

#### Partial Updates
Update nested models or sequences (lists/tuples) partially:

- For models: Provide a dictionary with only the keys to change.
- For lists/tuples: Provide a list where `None` preserves the original value at that index.

```python
jstyle.update({
    "Container[id=main]": {
        "style": {  # partially update style model
            "background_color": "lightblue",
            "padding": [20, None, 20, None]  # update top/bottom padding only
        },
        "transform": { "rotate": 15 }  # partially update transform
    }
})
```

### 6. Applying Styles

#### `jstyle.apply(component)`
Applies all currently defined styles recursively to the component and its
children, respecting context and specificity. This is the standard way to
apply styles during layout/render.

#### `jstyle.update(new_styles)`
Merges new style rules into the global stylesheet.

#### `with jstyle(temporary_styles): ...`
Creates a temporary context where `temporary_styles` are merged with and
override global styles. Styles revert upon exiting the `with` block.

#### `jstyle.clear()`
Clears all styles, resetting to the default state.
