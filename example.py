from jeanplot.component import Component
import numpy as np
import matplotlib.pyplot as plt
from jeanplot.models import Transform, Size, BoxStyle, LayoutConstraints, Offset
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.container import Container
from jeanplot.text import Text
from jeanplot.svg import SVGElement


def test_simple_container():
    """test a simple container with a single child"""
    container = Container(
        id="root",
        min_dimensions=Size(width=200, height=100),
        style=BoxStyle(background_color="white", border_color="black", border_width=1),
    )

    rect = Container(
        id="rect",
        min_dimensions=Size(width=50, height=50),
        style=BoxStyle(background_color="blue", corner_radius=5),
    )

    container.children = [rect]

    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=300, height=200)
    container.measure(renderer)
    renderer.render_component(ax, container)

    # display
    plt.show()

    return container, renderer


def test_row_layout():
    """test row layout with different alignment options"""
    # create a container with row layout
    container = Container(
        id="row-container",
        min_dimensions=Size(width=400, height=200),
        layout=LayoutConstraints(
            direction="row",
            align_items="start",  # try: "start", "center", "end", "stretch"
            justify_content="space-between",  # try: "start", "center", "end", "space-between", etc.
            gap=10,
        ),
        style=BoxStyle(
            background_color="#f0f0f0",
            border_color="black",
            border_width=1,
            padding=(10, 10, 10, 10),
        ),
    )

    # create three boxes with different heights
    box1 = Container(
        id="box1",
        min_dimensions=Size(width=80, height=50),
        style=BoxStyle(background_color="red"),
    )

    box2 = Container(
        id="box2",
        min_dimensions=Size(width=100, height=80),
        style=BoxStyle(background_color="green"),
    )

    box3 = Container(
        id="box3",
        min_dimensions=Size(width=120, height=30),
        style=BoxStyle(background_color="blue"),
    )

    # add children to container
    container.children = [box1, box2, box3]

    # create renderer and context
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=500, height=300)

    # measure and render
    container.measure(renderer)
    renderer.render_component(ax, container)

    # display
    plt.title(
        f"Row Layout - align: {container.layout.align_items}, justify: {container.layout.justify_content}"
    )
    plt.show()

    return container, renderer


def test_nested_layout():
    """test nested containers with both row and column layouts"""
    # create a root container with column layout
    root = Container(
        id="root",
        min_dimensions=Size(width=500, height=400),
        layout=LayoutConstraints(direction="column", gap=10),
        style=BoxStyle(
            background_color="white", border_color="black", border_width=1, padding=(10, 10, 10, 10)
        ),
    )

    # create a header with row layout
    header = Container(
        id="header",
        min_dimensions=Size(width=0, height=60),
        layout=LayoutConstraints(
            direction="row", justify_content="space-between", align_items="center"
        ),
        style=BoxStyle(background_color="#e0e0e0", padding=(5, 10, 5, 10)),
    )

    # create header items
    logo = Container(
        id="logo",
        min_dimensions=Size(width=40, height=40),
        style=BoxStyle(background_color="blue", corner_radius=20),
    )

    nav = Container(
        id="nav",
        min_dimensions=Size(width=200, height=30),
        layout=LayoutConstraints(direction="row", justify_content="space-evenly", gap=5),
        style=BoxStyle(background_color="#d0d0d0"),
    )

    # create nav items
    for i, color in enumerate(["#ff9999", "#99ff99", "#9999ff"]):
        nav.children.append(
            Container(
                id=f"nav-item-{i+1}",
                min_dimensions=Size(width=60, height=20),
                style=BoxStyle(background_color=color, corner_radius=3),
            )
        )

    # assemble header
    header.children = [logo, nav]

    # create content area
    content = Container(
        id="content",
        min_dimensions=Size(width=0, height=300),
        layout=LayoutConstraints(direction="row", gap=10),
        style=BoxStyle(background_color="#f5f5f5"),
    )

    # create sidebar
    sidebar = Container(
        id="sidebar",
        min_dimensions=Size(width=100, height=0),
        layout=LayoutConstraints(direction="column", gap=5),
        style=BoxStyle(background_color="#e5e5e5", padding=(5, 5, 5, 5)),
    )

    # add sidebar items
    for i in range(4):
        sidebar.children.append(
            Container(
                id=f"sidebar-item-{i+1}",
                min_dimensions=Size(width=0, height=30),
                style=BoxStyle(background_color="#c0c0c0", corner_radius=3),
            )
        )

    # create main content
    main = Container(
        id="main",
        min_dimensions=Size(width=0, height=0),
        style=BoxStyle(background_color="white", border_color="#d0d0d0", border_width=1),
    )

    # assemble content
    content.children = [sidebar, main]

    # create footer
    footer = Container(
        id="footer",
        min_dimensions=Size(width=0, height=40),
        layout=LayoutConstraints(direction="row", justify_content="center", align_items="center"),
        style=BoxStyle(background_color="#e0e0e0"),
    )

    footer_text = Container(
        id="footer-text",
        min_dimensions=Size(width=200, height=20),
        style=BoxStyle(background_color="#d0d0d0"),
    )

    footer.children = [footer_text]

    # assemble root
    root.children = [header, content, footer]

    # create renderer and context
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=1600, height=1500)

    # enable debug mode for all components
    def enable_debug(container):
        container.debug = True
        for child in container.children:
            enable_debug(child)

    # uncomment to enable debug
    # enable_debug(root)

    # measure and render
    root.measure(renderer)
    renderer.render_component(ax, root)

    # display
    plt.show()

    return root, renderer


def test_interdependent_layout():
    """demonstrate a case where measurement and layout are interdependent"""

    # create a parent container with row layout that adapts to children
    parent = Container(
        id="parent",
        min_dimensions=Size(width=400, height=200),
        max_dimensions=Size(width=600, height=400),
        layout=LayoutConstraints(
            direction="row",
            align_items="stretch",  # this is key - it will stretch children
            justify_content="space-evenly",
            gap=10,
        ),
        style=BoxStyle(
            background_color="#f0f0f0",
            border_color="black",
            border_width=1,
            padding=(10, 10, 10, 10),
        ),
    )

    # create a child container that should stretch vertically
    child1 = Container(
        id="child1",
        min_dimensions=Size(width=100, height=50),
        max_dimensions=Size(width=100, height=float("inf")),  # can stretch vertically
        style=BoxStyle(background_color="red", border_color="darkred", border_width=1),
    )

    # create a nested container with children that will affect its height
    nested = Container(
        id="nested",
        min_dimensions=Size(width=200, height=0),  # height will be determined by children
        layout=LayoutConstraints(
            direction="column", align_items="stretch", justify_content="space-evenly", gap=2
        ),
        style=BoxStyle(
            background_color="#e0e0e0",
            border_color="darkblue",
            border_width=1,
            corner_radius=5,
            padding=(15, 15, 15, 15),
        ),
    )

    # add items to the nested container that will determine its height
    for i in range(3):
        nested.children.append(
            Container(
                id=f"nested-child-{i+1}",
                min_dimensions=Size(width=0, height=40),
                style=BoxStyle(background_color="lightblue", border_color="blue", border_width=1),
                layout=LayoutConstraints(
                    direction="column",  # Changed from row to column
                    justify_content="start",  # Center vertically
                ),
                children=[
                    Text(
                        text=f"Item {i+1}",
                        align="center",
                    )
                ],
            )
        )

    # add the children to the parent
    parent.children = [child1, nested]

    # create renderer and context
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=600, height=400)

    # measure and render
    parent.measure_and_layout(renderer)  # Using measure_and_layout to ensure proper layout
    renderer.render_component(ax, parent)

    # display
    plt.title("Interdependent Layout Test")
    plt.show()

    return parent, renderer


def test_svg_integration():
    """test SVG elements with layout"""
    # paths to SVG assets
    test_circle_path = "test_circle.svg"
    aggregation_path = "jeanplot/resources/parts/aggregation.svg"
    color_path = "jeanplot/resources/aprts/color.svg"
    l2_path = "jeanplot/resources/parts/l2.svg"

    # create a container with row layout
    container = Container(
        id="svg-container",
        min_dimensions=Size(width=800, height=400),
        layout=LayoutConstraints(
            direction="row", align_items="center", justify_content="space-around", gap=20
        ),
        style=BoxStyle(
            background_color="#f0f0f0",
            border_color="black",
            border_width=1,
            padding=(20, 20, 20, 20),
        ),
    )

    # create SVG elements
    circle = SVGElement(
        id="circle",
        svg_content=test_circle_path,
        main_color="darkblue",  # customize main color
    )

    aggregation = SVGElement(
        id="aggregation",
        svg_content=aggregation_path,
        transform=Transform(scale=(2.0, 2.0)),  # scale up
        main_color="red",
    )

    color = SVGElement(id="color", svg_content=color_path, main_color="green")

    l2s = (
        SVGElement(
            id=f"l2-{i}",
            svg_content=l2_path,
            offset=Offset(relative=(i, 0)),
            main_color="purple",
            min_dimensions=Size(width=40, height=40),
            transform=Transform(scale=(i + 1, i + 1), rotate=-70),
        )
        for i in range(2)
    )

    # create a nested container for one of the SVGs
    nested = Container(
        id="nested",
        min_dimensions=Size(width=80, height=80),
        layout=LayoutConstraints(
            direction="column", align_items="center", justify_content="space-evenly"
        ),
        style=BoxStyle(
            background_color="white",
            border_color="gray",
            border_width=5,
            corner_radius=10,
            border_style="dotted",
            padding=(5, 5, 5, 5),
        ),
        children=l2s,
    )

    # add SVG elements to container
    container.children = [circle, aggregation, color, nested]

    # create renderer and context
    renderer = MatplotlibRenderer()
    # fig, ax = plt.subplots(figsize=(5, 2.5), dpi=300)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    # equal
    ax.set_aspect("equal")

    # measure and render
    container.measure(renderer)
    renderer.render_component(ax, container)

    # display
    plt.title("SVG Integration Test")
    plt.show()

    return container, renderer


def test_text_component():
    """test text component with various configurations"""

    # create a root container
    root = Container(
        id="text-examples",
        min_dimensions=Size(width=500, height=400),
        style=BoxStyle(
            background_color="#f5f5f5",
            border_color="#333333",
            border_width=1,
            padding=(20, 20, 20, 20),
        ),
        layout=LayoutConstraints(direction="column", gap=20, justify_content="start"),
    )

    # basic text example
    basic_text = Text(
        id="basic-text",
        text="Hello, this is a Text component!",
        font_size=20,
        color="black",
    )

    # styled text
    styled_text = Text(
        id="styled-text",
        text="Bold and blue text",
        font_size=20,
        font_weight="bold",
        color="blue",
    )

    # multiline text
    multiline_text = Text(
        id="multiline-text",
        text="This is a multi-line\ntext component example\nwith three lines",
        font_size=20,
        color="#333",
    )

    # create a container with text alignment examples
    alignment_container = Container(
        id="alignment-examples",
        min_dimensions=Size(width=450, height=150),
        style=BoxStyle(
            background_color="white",
            border_color="#999",
            border_width=1,
            corner_radius=5,
            padding=(10, 10, 10, 10),
        ),
        layout=LayoutConstraints(direction="row", gap=15, justify_content="space-between"),
    )

    # add differently aligned text components
    for align, color in [("left", "red"), ("center", "green"), ("right", "blue")]:
        text_box = Container(
            id=f"{align}-aligned",
            min_dimensions=Size(width=140, height=130),
            style=BoxStyle(
                background_color="#f0f0f0",
                border_color="#ddd",
                border_width=1,
                padding=(5, 5, 5, 5),
            ),
        )

        # add the example text
        aligned_text = Text(
            id=f"{align}-example",
            text=f"This text is\n{align} aligned\non multiple lines",
            font_size=20,
            color=color,
            align=align,
        )

        text_box.children = [aligned_text]
        alignment_container.children.append(text_box)

    # text with vertical alignment demonstration
    vertical_container = Container(
        id="vertical-examples",
        min_dimensions=Size(width=450, height=150),
        style=BoxStyle(
            background_color="white",
            border_color="#999",
            border_width=1,
            corner_radius=5,
            padding=(10, 10, 10, 10),
        ),
        layout=LayoutConstraints(direction="row", gap=15, justify_content="space-between"),
    )

    # add differently vertically aligned text components
    for valign, color in [("top", "purple"), ("middle", "orange"), ("bottom", "teal")]:
        text_box = Container(
            id=f"{valign}-aligned",
            min_dimensions=Size(width=140, height=140),
            style=BoxStyle(
                background_color="#f0f0f0",
                border_color="#d0d",
                border_width=1,
                padding=(5, 5, 5, 5),
            ),
        )

        # add the example text
        aligned_text = Text(
            id=f"{valign}-example",
            text=f"Vertically\n{valign}\naligned",
            font_size=10,
            color=color,
            align="center",
            vertical_align=valign,
        )

        text_box.children = [aligned_text]
        vertical_container.children.append(text_box)

    # add all examples to root container
    root.children = [
        basic_text,
        styled_text,
        multiline_text,
        alignment_container,
        vertical_container,
    ]

    # create renderer and draw
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=600, height=600)

    # uncomment to help visualize the layout
    # alignment_container.debug = True
    # vertical_container.debug = True

    # measure and render with auto-adjusted limits
    renderer.render_component(ax, root, adjust_lims=True)

    # show result
    plt.title("Text Component Examples")
    plt.show()

    return root, renderer


def test_offset_examples():
    """Demonstrate the unified offset system"""
    # create a parent container
    parent = Container(
        id="offset-examples",
        min_dimensions=Size(width=500, height=300),
        style=BoxStyle(
            background_color="#f0f0f0",
            border_color="black",
            border_width=1,
            padding=(20, 20, 20, 20),
        ),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="space-around",
            gap=20,
        ),
    )

    # create boxes with different offset configurations
    boxes = []

    # 1. no offset (default)
    box1 = Container(
        id="no-offset",
        min_dimensions=Size(width=80, height=80),
        style=BoxStyle(
            background_color="lightblue",
            border_color="blue",
            border_width=2,
        ),
    )
    box1.children = [
        Text(
            id="no-offset-text",
            text="No Offset",
            font_size=10,
            color="blue",
            align="center",
            vertical_align="middle",
        )
    ]

    # 2. relative offset only (0.5, 0.5) - centers the origin
    box2 = Container(
        id="relative-offset",
        min_dimensions=Size(width=80, height=80),
        style=BoxStyle(
            background_color="lightgreen",
            border_color="green",
            border_width=2,
        ),
        offset=Offset(relative=(0.5, 0.5)),  # center origin
    )
    box2.children = [
        Text(
            id="relative-offset-text",
            text="Relative\nOffset\n(0.5, 0.5)",
            font_size=10,
            color="green",
            align="center",
            vertical_align="middle",
        )
    ]

    # 3. absolute offset only (20, 20) - shifts by fixed amount
    box3 = Container(
        id="absolute-offset",
        min_dimensions=Size(width=80, height=80),
        style=BoxStyle(
            background_color="lightyellow",
            border_color="orange",
            border_width=2,
        ),
        offset=Offset(absolute=(20, 20)),  # absolute shift
    )
    box3.children = [
        Text(
            id="absolute-offset-text",
            text="Absolute\nOffset\n(20, 20)",
            font_size=10,
            color="orange",
            align="center",
            vertical_align="middle",
        )
    ]

    # 4. combined offset - both relative and absolute
    box4 = Container(
        id="combined-offset",
        min_dimensions=Size(width=80, height=80),
        style=BoxStyle(
            background_color="lightpink",
            border_color="red",
            border_width=2,
        ),
        offset=Offset(relative=(0.5, 0.5), absolute=(20, -20)),  # combined
    )
    box4.children = [
        Text(
            id="combined-offset-text",
            text="Combined\nOffset",
            font_size=10,
            color="red",
            align="center",
            vertical_align="middle",
        )
    ]

    # Add all boxes to parent container
    parent.children = [box1, box2, box3, box4]

    # Enable debug to see origin markers
    for child in parent.children:
        child.debug = True

    # Create renderer and context
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=600, height=400)
    # Measure, layout and render
    parent.measure_and_layout(renderer)
    renderer.render_component(ax, parent)
    # Add title and display
    plt.title("Unified Offset Examples")
    plt.show()


def test_svg_with_offset():
    """Test SVG elements with the new offset system"""
    # Create a container
    container = Container(
        id="svg-container",
        min_dimensions=Size(width=500, height=300),
        style=BoxStyle(
            background_color="#f5f5f5",
            border_color="#333",
            border_width=1,
            padding=(20, 20, 20, 20),
        ),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="space-evenly",
        ),
    )

    # Create SVG elements with different offsets
    svg_paths = [
        "test_circle.svg",
        "jeanplot/resources/symbols/l2.svg",
    ]

    offset_configs = [
        Offset(),  # default
        Offset(relative=(0.5, 0.5)),  # centered origin
        Offset(absolute=(10, 10)),  # small shift
        Offset(relative=(0.5, 0.5), absolute=(15, -15)),  # combined
    ]

    colors = ["darkblue", "purple", "green", "red"]

    # Create SVG elements
    svg_elements = []
    for i, (path, offset, color) in enumerate(zip(svg_paths * 2, offset_configs, colors)):
        svg = SVGElement(
            id=f"svg-{i}",
            svg_content=path,
            main_color=color,
            offset=offset,
            transform=Transform(scale=(3, 3)),
        )
        svg.debug = True
        svg_elements.append(svg)

    # Add to container
    container.children = svg_elements

    # Create renderer and context
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=600, height=400)

    # Measure, layout and render
    container.measure_and_layout(renderer)
    renderer.render_component(ax, container)

    # Add title and display
    plt.title("SVG Elements with Unified Offset")
    plt.show()


def test_text_with_offset():
    """Test text components with the new offset system"""
    # Create a container
    container = Container(
        id="text-container",
        min_dimensions=Size(width=500, height=300),
        style=BoxStyle(
            background_color="#f8f8f8",
            border_color="#999",
            border_width=1,
            padding=(20, 20, 20, 20),
        ),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="space-evenly",
        ),
    )

    # Create text elements with different offsets
    offset_configs = [
        Offset(),  # default
        Offset(relative=(0.5, 0.5)),  # centered origin
        Offset(absolute=(5, 5)),  # small shift
        Offset(relative=(0.5, 0), absolute=(0, 10)),  # top center + shift
    ]

    colors = ["navy", "purple", "forestgreen", "maroon"]

    # Create text elements
    text_elements = []
    for i, (offset, color) in enumerate(zip(offset_configs, colors)):
        bg_container = Container(
            id=f"text-bg-{i}",
            min_dimensions=Size(width=100, height=80),
            style=BoxStyle(
                background_color="#e0e0e0",
                border_color="#999",
                border_width=1,
                corner_radius=5,
            ),
        )

        text = Text(
            id=f"text-{i}",
            text=f"Offset\nExample\n{i+1}",
            font_size=10,
            color=color,
            align="center",
            vertical_align="middle",
            offset=offset,
        )
        text.debug = True

        bg_container.children = [text]
        text_elements.append(bg_container)

    # Add to container
    container.children = text_elements

    # Create renderer and context
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=600, height=400)

    # Measure, layout and render
    container.measure_and_layout(renderer)
    renderer.render_component(ax, container)

    # Add title and display
    plt.title("Text Components with Unified Offset")
    plt.show()


if __name__ == "__main__":
    test_simple_container()
    test_row_layout()
    test_nested_layout()
    test_interdependent_layout()
    test_text_component()
    test_offset_examples()
    test_svg_integration()
