import numpy as np
import matplotlib.pyplot as plt
from jeanplot.components import Container, SVGElement, Text
from jeanplot.models import Transform, Size, VisualStyle, LayoutConstraints, Offset
from jeanplot.renderer import MatplotlibRenderer


def test_simple_container():
    """test a simple container with a single child"""
    # create a container
    container = Container(
        id="root",
        min_dimensions=Size(width=200, height=100),
        style=VisualStyle(background_color="white", border_color="black", border_width=1),
    )

    # create a child rectangle
    rect = Container(
        id="rect",
        min_dimensions=Size(width=50, height=50),
        style=VisualStyle(background_color="blue", corner_radius=5),
    )

    # add child to container
    container.children = [rect]

    # create renderer and context
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=300, height=200)

    # measure and render
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
        style=VisualStyle(
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
        style=VisualStyle(background_color="red"),
    )

    box2 = Container(
        id="box2",
        min_dimensions=Size(width=100, height=80),
        style=VisualStyle(background_color="green"),
    )

    box3 = Container(
        id="box3",
        min_dimensions=Size(width=120, height=30),
        style=VisualStyle(background_color="blue"),
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
        style=VisualStyle(
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
        style=VisualStyle(background_color="#e0e0e0", padding=(5, 10, 5, 10)),
    )

    # create header items
    logo = Container(
        id="logo",
        min_dimensions=Size(width=40, height=40),
        style=VisualStyle(background_color="blue", corner_radius=20),
    )

    nav = Container(
        id="nav",
        min_dimensions=Size(width=200, height=30),
        layout=LayoutConstraints(direction="row", justify_content="space-evenly", gap=5),
        style=VisualStyle(background_color="#d0d0d0"),
    )

    # create nav items
    for i, color in enumerate(["#ff9999", "#99ff99", "#9999ff"]):
        nav.children.append(
            Container(
                id=f"nav-item-{i+1}",
                min_dimensions=Size(width=60, height=20),
                style=VisualStyle(background_color=color, corner_radius=3),
            )
        )

    # assemble header
    header.children = [logo, nav]

    # create content area
    content = Container(
        id="content",
        min_dimensions=Size(width=0, height=300),
        layout=LayoutConstraints(direction="row", gap=10),
        style=VisualStyle(background_color="#f5f5f5"),
    )

    # create sidebar
    sidebar = Container(
        id="sidebar",
        min_dimensions=Size(width=100, height=0),
        layout=LayoutConstraints(direction="column", gap=5),
        style=VisualStyle(background_color="#e5e5e5", padding=(5, 5, 5, 5)),
    )

    # add sidebar items
    for i in range(4):
        sidebar.children.append(
            Container(
                id=f"sidebar-item-{i+1}",
                min_dimensions=Size(width=0, height=30),
                style=VisualStyle(background_color="#c0c0c0", corner_radius=3),
            )
        )

    # create main content
    main = Container(
        id="main",
        min_dimensions=Size(width=0, height=0),
        style=VisualStyle(background_color="white", border_color="#d0d0d0", border_width=1),
    )

    # assemble content
    content.children = [sidebar, main]

    # create footer
    footer = Container(
        id="footer",
        min_dimensions=Size(width=0, height=40),
        layout=LayoutConstraints(direction="row", justify_content="center", align_items="center"),
        style=VisualStyle(background_color="#e0e0e0"),
    )

    footer_text = Container(
        id="footer-text",
        min_dimensions=Size(width=200, height=20),
        style=VisualStyle(background_color="#d0d0d0"),
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
    from jeanplot.components import Container
    from jeanplot.models import Size, VisualStyle, LayoutConstraints
    from jeanplot.renderer import MatplotlibRenderer
    import matplotlib.pyplot as plt

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
        style=VisualStyle(
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
        style=VisualStyle(background_color="red", border_color="darkred", border_width=1),
    )

    # create a nested container with children that will affect its height
    nested = Container(
        id="nested",
        min_dimensions=Size(width=200, height=0),  # height will be determined by children
        layout=LayoutConstraints(
            direction="column", align_items="stretch", justify_content="space-evenly", gap=2
        ),
        style=VisualStyle(
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
                style=VisualStyle(
                    background_color="lightblue", border_color="blue", border_width=1
                ),
                children=[Text(text=f"Item {i+1}")],  # add text to each child
            )
        )

    # add the children to the parent
    parent.children = [child1, nested]

    # create renderer and context
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=600, height=400)

    # measure and render
    parent.measure(renderer)
    renderer.render_component(ax, parent)

    # display
    plt.title("Interdependent Layout Test")
    plt.show()

    return parent, renderer


def test_svg_integration():
    """test SVG elements with layout"""
    # paths to SVG assets
    test_circle_path = "test_circle.svg"
    aggregation_path = "jeanplot/resources/symbols/aggregation.svg"
    color_path = "jeanplot/resources/symbols/color.svg"
    l2_path = "jeanplot/resources/symbols/l2.svg"

    # create a container with row layout
    container = Container(
        id="svg-container",
        min_dimensions=Size(width=800, height=400),
        layout=LayoutConstraints(
            direction="row", align_items="center", justify_content="space-around", gap=20
        ),
        style=VisualStyle(
            background_color="#f0f0f0",
            border_color="black",
            border_width=1,
            padding=(20, 20, 20, 20),
        ),
    )

    # create SVG elements
    circle = SVGElement(
        id="circle",
        file_path=test_circle_path,
        main_color="darkblue",  # customize main color
    )

    aggregation = SVGElement(
        id="aggregation",
        file_path=aggregation_path,
        transform=Transform(scale=(2.0, 2.0)),  # scale up
        main_color="red",
    )

    color = SVGElement(id="color", file_path=color_path, main_color="green")

    l2s = (
        SVGElement(
            id=f"l2-{i}",
            file_path=l2_path,
            offset=Offset(relative=(i, 0)),
            main_color="purple",
            min_dimensions=Size(width=40, height=40),
            transform=Transform(scale=(5, 5), rotate=90),
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
        style=VisualStyle(
            background_color="red",
            border_color="gray",
            border_width=5,
            corner_radius=10,
            # border_style="dotted",
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

    # measure and render
    container.measure(renderer)
    renderer.render_component(ax, container)

    # display
    plt.title("SVG Integration Test")
    plt.show()

    return container, renderer


def test_text_component():
    """test text component with various configurations"""
    from jeanplot.components import Container, Text
    from jeanplot.models import Size, VisualStyle, LayoutConstraints
    from jeanplot.renderer import MatplotlibRenderer
    import matplotlib.pyplot as plt

    # create a root container
    root = Container(
        id="text-examples",
        min_dimensions=Size(width=500, height=400),
        style=VisualStyle(
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
        style=VisualStyle(
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
            style=VisualStyle(
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
        style=VisualStyle(
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
            style=VisualStyle(
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
        style=VisualStyle(
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
        style=VisualStyle(
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
        style=VisualStyle(
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
        style=VisualStyle(
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
        style=VisualStyle(
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
        style=VisualStyle(
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
            file_path=path,
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
        style=VisualStyle(
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
            style=VisualStyle(
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


def test_rotation_with_offset():
    """Test how rotation and offset interact"""
    # Create a container
    container = Container(
        id="rotation-container",
        min_dimensions=Size(width=600, height=400),
        style=VisualStyle(
            background_color="#f8f8f8",
            border_color="#999",
            border_width=1,
            padding=(30, 30, 30, 30),
        ),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="space-evenly",
        ),
    )

    # Rotation angles
    rotations = [0, 45, 90, 180]

    # Create rectangles with different offsets and rotations
    elements = []
    for i, rotation in enumerate(rotations):
        # Create a container for each example
        example = Container(
            id=f"rotation-example-{i}",
            min_dimensions=Size(width=120, height=120),
            style=VisualStyle(
                background_color="#f0f0f0",
                border_color="#ccc",
                border_width=1,
                corner_radius=5,
            ),
        )

        # Create a rotated rectangle with centered origin
        rect = Container(
            id=f"rect-{i}",
            min_dimensions=Size(width=80, height=40),
            style=VisualStyle(
                background_color="lightblue",
                border_color="blue",
                border_width=2,
            ),
            offset=Offset(relative=(0.5, 0.5)),  # center the rotation
            transform=Transform(rotate=rotation),
        )

        # Add text label to the rectangle
        label = Text(
            id=f"label-{i}",
            text=f"{rotation}°",
            font_size=12,
            color="navy",
            align="center",
            vertical_align="middle",
        )
        rect.children = [label]
        rect.debug = True

        # Add explanation text
        explanation = Text(
            id=f"explanation-{i}",
            text=f"Rotation: {rotation}°\nOffset: (0.5, 0.5)",
            font_size=8,
            color="black",
            align="center",
            vertical_align="bottom",
        )

        # Add elements to the example container
        example.children = [rect, explanation]

        # Add to elements list
        elements.append(example)

    # Add to main container
    container.children = elements

    # Create renderer and context
    renderer = MatplotlibRenderer()
    ax = renderer.create_context(width=700, height=500)

    # Measure, layout and render
    container.measure_and_layout(renderer)
    renderer.render_component(ax, container)

    # Add title and display
    plt.title("Rotation with Centered Offset")
    plt.show()


def test_simple_line_width():
    """test line width handling with different modes"""
    # create figure with side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # set the same data range for both axes
    ax1.set_xlim(0, 400)
    ax1.set_ylim(0, 300)
    ax2.set_xlim(0, 400)
    ax2.set_ylim(0, 300)

    # use equal aspect ratio for both
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    # create and render content with point-based line widths
    point_content = create_test_content(mode="point", title="Point-Based Width")
    renderer1 = MatplotlibRenderer()
    point_content.measure_and_layout(renderer1)
    renderer1.render_component(ax1, point_content, adjust_lims=False)
    ax1.set_title("Point Mode (constant visual width)")

    # create and render content with data-based line widths
    data_content = create_test_content(mode="data", title="Data-Based Width")
    renderer2 = MatplotlibRenderer()
    data_content.measure_and_layout(renderer2)
    renderer2.render_component(ax2, data_content, adjust_lims=False)
    ax2.set_title("Data Mode (scales with data units)")

    # add figure title
    plt.suptitle("Line Width Comparison", fontsize=16)

    # add explanation text
    plt.figtext(
        0.5,
        0.01,
        "Point mode uses constant visual widths regardless of scale.\n"
        "Data mode uses widths in data units that scale with figure size and zoom.",
        ha="center",
        fontsize=10,
    )

    # adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("linewidth_basic_comparison.png", dpi=100)
    plt.show()

    # now test with different figure sizes
    for figsize in [(6, 5), (12, 10)]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # set same data range
        for ax in (ax1, ax2):
            ax.set_xlim(0, 400)
            ax.set_ylim(0, 300)
            ax.set_aspect("equal")

        # render the same content in both modes
        point_content.measure_and_layout(renderer1)
        renderer1.render_component(ax1, point_content, adjust_lims=False)
        ax1.set_title("Point Mode")

        data_content.measure_and_layout(renderer2)
        renderer2.render_component(ax2, data_content, adjust_lims=False)
        ax2.set_title("Data Mode")

        # add figure title with size info
        plt.suptitle(f"Figure Size: {figsize[0]}x{figsize[1]} inches", fontsize=16)

        # adjust and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"linewidth_size_{figsize[0]}x{figsize[1]}.png", dpi=100)
        plt.show()

    # finally test with zoom levels
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # full view - top row
    for ax in (ax1, ax2):
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 300)
        ax.set_aspect("equal")

    # zoomed view - bottom row
    for ax in (ax3, ax4):
        ax.set_xlim(100, 300)  # zoomed in
        ax.set_ylim(50, 200)
        ax.set_aspect("equal")

    # render content in all views
    point_content.measure_and_layout(renderer1)
    renderer1.render_component(ax1, point_content, adjust_lims=False)
    ax1.set_title("Point Mode - Full View")

    data_content.measure_and_layout(renderer2)
    renderer2.render_component(ax2, data_content, adjust_lims=False)
    ax2.set_title("Data Mode - Full View")

    point_content.measure_and_layout(renderer1)
    renderer1.render_component(ax3, point_content, adjust_lims=False)
    ax3.set_title("Point Mode - Zoomed In")

    data_content.measure_and_layout(renderer2)
    renderer2.render_component(ax4, data_content, adjust_lims=False)
    ax4.set_title("Data Mode - Zoomed In")

    # add title
    plt.suptitle("Line Width Modes with Different Zoom Levels", fontsize=16)

    # adjust and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("linewidth_zoom_comparison.png", dpi=100)
    plt.show()


def create_test_content(mode="point", title="Line Width Test"):
    """create test content with rectangles of different line widths"""
    # create main container
    container = Container(
        id="main-container",
        min_dimensions=Size(width=400, height=300),
        style=VisualStyle(
            background_color="#f8f8f8",
            border_color="#333",
            border_width=2,
            border_width_mode=mode,
            corner_radius=5,
            padding=(20, 20, 20, 20),
        ),
        layout=LayoutConstraints(
            direction="column",
            align_items="stretch",
            justify_content="start",
            gap=15,
        ),
    )

    # add title
    header = Container(
        id="header",
        min_dimensions=Size(width=0, height=40),
        style=VisualStyle(
            background_color="#e0e0e0",
            border_color="#999",
            border_width=1,
            border_width_mode=mode,
            padding=(5, 5, 5, 5),
        ),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="center",
        ),
    )

    header.children = [
        Text(
            id="title-text",
            text=title,
            font_size=18,
            color="black",
            align="center",
        )
    ]

    # create boxes with different line widths
    boxes_row = Container(
        id="boxes-row",
        min_dimensions=Size(width=0, height=80),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="space-around",
            gap=5,
        ),
    )

    # add boxes with a range of line widths
    widths = [1, 2, 4, 8]
    boxes = []
    for width in widths:
        box = Container(
            id=f"box-{width}",
            min_dimensions=Size(width=80, height=60),
            style=VisualStyle(
                background_color="white",
                border_color="blue",
                border_width=width,
                border_width_mode=mode,
                corner_radius=5,
            ),
            layout=LayoutConstraints(
                direction="row",
                align_items="center",
                justify_content="center",
            ),
        )

        box.children = [
            Text(
                id=f"text-{width}",
                text=f"Width: {width}",
                font_size=12,
                color="black",
                align="center",
            )
        ]

        boxes.append(box)

    boxes_row.children = boxes

    # create row with SVG elements
    svg_row = Container(
        id="svg-row",
        min_dimensions=Size(width=0, height=150),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="space-around",
            gap=10,
        ),
    )

    # add SVG elements with different colors
    colors = ["purple", "red", "green"]
    svg_containers = []

    for i, color in enumerate(colors):
        svg_container = Container(
            id=f"svg-container-{i}",
            min_dimensions=Size(width=110, height=130),
            style=VisualStyle(
                background_color="white",
                border_color="#999",
                border_width=1,
                border_width_mode=mode,
                corner_radius=5,
                padding=(5, 5, 5, 5),
            ),
            layout=LayoutConstraints(
                direction="column",
                align_items="center",
                justify_content="center",
                gap=5,
            ),
        )

        # create SVG element
        svg = SVGElement(
            id=f"svg-{i}",
            file_path="test_circle.svg",
            main_color=color,
            transform=Transform(scale=(1.5, 1.5)),
        )

        # set line width mode for SVG
        svg.add_renderer_option("matplotlib", "line_width_mode", mode)

        # add label
        label = Text(
            id=f"svg-label-{i}",
            text=f"SVG {i+1}",
            font_size=10,
            color="black",
            align="center",
        )

        svg_container.children = [svg, label]
        svg_containers.append(svg_container)

    svg_row.children = svg_containers

    # row for line styles
    styles_row = Container(
        id="styles-row",
        min_dimensions=Size(width=0, height=60),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="space-around",
            gap=5,
        ),
    )

    # add different line styles
    line_styles = [
        {"name": "Solid", "style": "solid"},
        {"name": "Dashed", "style": "dashed"},
        {"name": "Dotted", "style": "dotted"},
    ]

    style_boxes = []
    for style_info in line_styles:
        style_box = Container(
            id=f"style-{style_info['name']}",
            min_dimensions=Size(width=110, height=40),
            style=VisualStyle(
                background_color="white",
                border_color="orange",
                border_width=3,
                border_width_mode=mode,
                border_style=style_info["style"],
            ),
            layout=LayoutConstraints(
                direction="row",
                align_items="center",
                justify_content="center",
            ),
        )

        style_box.children = [
            Text(
                id=f"style-text-{style_info['name']}",
                text=style_info["name"],
                font_size=10,
                color="black",
                align="center",
            )
        ]

        style_boxes.append(style_box)

    styles_row.children = style_boxes

    # add all elements to the main container
    container.children = [header, boxes_row, svg_row, styles_row]

    return container


def test_zoom_linewidth():
    """demonstrate data-unit line widths with different zoom levels"""

    # create a figure with side-by-side comparisons at different zoom levels
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # normal views (top row)
    for ax in axes[0]:
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 300)
        ax.set_aspect("equal")

    # zoomed views (bottom row)
    for ax in axes[1]:
        ax.set_xlim(150, 250)  # zoomed in
        ax.set_ylim(100, 200)
        ax.set_aspect("equal")

    # point mode container (left column)
    point_content = create_content("point", "Point Mode")
    point_renderer = MatplotlibRenderer()
    point_content.measure_and_layout(point_renderer)

    # data mode container (right column)
    data_content = create_content("data", "Data Mode")
    data_renderer = MatplotlibRenderer()
    data_content.measure_and_layout(data_renderer)

    # render normal views
    point_renderer.render_component(axes[0, 0], point_content, adjust_lims=False)
    axes[0, 0].set_title("Point Mode - Normal View")

    data_renderer.render_component(axes[0, 1], data_content, adjust_lims=False)
    axes[0, 1].set_title("Data Mode - Normal View")

    # render zoomed views
    point_renderer.render_component(axes[1, 0], point_content, adjust_lims=False)
    axes[1, 0].set_title("Point Mode - Zoomed In")

    data_renderer.render_component(axes[1, 1], data_content, adjust_lims=False)
    axes[1, 1].set_title("Data Mode - Zoomed In")

    # add title
    plt.suptitle("Line Width Modes with Different Zoom Levels", fontsize=16)

    # add explanatory text
    plt.figtext(
        0.5,
        0.02,
        "Point mode (left): Line widths remain constant in points regardless of zoom\n"
        "Data mode (right): Line widths defined in data units, scaling with the view",
        ha="center",
        fontsize=12,
    )

    # adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("dynamic_linewidth_zoom.png", dpi=100)
    plt.show()

    # now test with two different figures/sizes to demonstrate physical size difference
    sizes = [(6, 6), (12, 12)]
    for figsize in sizes:
        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.xlim(0, 400)
        plt.ylim(0, 300)
        plt.gca().set_aspect("equal")
        point_renderer.render_component(plt.gca(), point_content, adjust_lims=False)
        plt.title("Point Mode")

        plt.subplot(122)
        plt.xlim(0, 400)
        plt.ylim(0, 300)
        plt.gca().set_aspect("equal")
        data_renderer.render_component(plt.gca(), data_content, adjust_lims=False)
        data_renderer.refresh_linewidths(plt.gca())
        plt.title("Data Mode")

        plt.suptitle(f"Figure Size: {figsize[0]}x{figsize[1]} inches", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"dynamic_linewidth_size_{figsize[0]}x{figsize[1]}.png", dpi=100)
        plt.show()


def create_content(mode, title):
    """create test content with boxes of different line widths"""
    # create a container
    container = Container(
        id=f"{mode}-container",
        min_dimensions=Size(width=400, height=300),
        style=VisualStyle(
            background_color="#f8f8f8",
            border_color="#333",
            border_width=5,
            border_width_mode=mode,
            corner_radius=10,
            padding=(20, 20, 20, 20),
        ),
        layout=LayoutConstraints(
            direction="column",
            align_items="stretch",
            justify_content="space-around",
            gap=15,
        ),
    )

    # add title
    header = Container(
        id=f"{mode}-header",
        min_dimensions=Size(width=0, height=40),
        style=VisualStyle(
            background_color="#e0e0e0",
            border_color="#666",
            border_width=2,
            border_width_mode=mode,
            padding=(5, 5, 5, 5),
        ),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="center",
        ),
    )

    header.children = [
        Text(
            id=f"{mode}-title",
            text=title,
            font_size=18,
            color="black",
            align="center",
        )
    ]

    # create rows of boxes with different line widths
    # we'll use two rows to spread them out more
    rows = []

    for row_idx in range(2):
        row = Container(
            id=f"{mode}-row-{row_idx}",
            min_dimensions=Size(width=0, height=100),
            layout=LayoutConstraints(
                direction="row",
                align_items="center",
                justify_content="space-around",
                gap=10,
            ),
        )

        # create boxes with different line widths
        widths = [1, 2, 5, 10] if row_idx == 0 else [3, 6, 9, 12]
        boxes = []

        for width in widths:
            box = Container(
                id=f"{mode}-box-{width}",
                min_dimensions=Size(width=80, height=60),
                style=VisualStyle(
                    background_color="white",
                    border_color="blue",
                    border_width=width,
                    border_width_mode=mode,
                    corner_radius=5,
                ),
                layout=LayoutConstraints(
                    direction="row",
                    align_items="center",
                    justify_content="center",
                ),
            )

            box.children = [
                Text(
                    id=f"{mode}-box-text-{width}",
                    text=f"Width: {width}",
                    font_size=12,
                    color="black",
                    align="center",
                )
            ]

            boxes.append(box)

        row.children = boxes
        rows.append(row)

    # create a row for line styles
    styles_row = Container(
        id=f"{mode}-styles",
        min_dimensions=Size(width=0, height=60),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="space-around",
            gap=10,
        ),
    )

    # add different line styles
    styles = []
    for style_info in [
        {"name": "Solid", "style": "solid"},
        {"name": "Dashed", "style": "dashed"},
        {"name": "Dotted", "style": "dotted"},
    ]:
        style_box = Container(
            id=f"{mode}-style-{style_info['name']}",
            min_dimensions=Size(width=100, height=40),
            style=VisualStyle(
                background_color="white",
                border_color="red",
                border_width=4,
                border_width_mode=mode,
                border_style=style_info["style"],
                corner_radius=5,
            ),
            layout=LayoutConstraints(
                direction="row",
                align_items="center",
                justify_content="center",
            ),
        )

        style_box.children = [
            Text(
                id=f"{mode}-style-label-{style_info['name']}",
                text=style_info["name"],
                font_size=12,
                color="black",
                align="center",
            )
        ]

        styles.append(style_box)

    styles_row.children = styles

    # add all elements to container
    container.children = [header] + rows + [styles_row]

    return container


def test_tracking():
    """test that the line width tracking and updating works"""
    # create a simple container with rectangles
    container = create_container()

    # create a renderer
    renderer = MatplotlibRenderer()
    print("=== Initial rendering ===")

    # create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # set data limits for both axes
    for ax in axes:
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 300)
        ax.set_aspect("equal")

    # render on first axis (normal view)
    container.measure_and_layout(renderer)
    renderer.render_component(axes[0], container, adjust_lims=False)
    axes[0].set_title("Normal View")

    # now set zoomed limits on second axis
    print("\n=== Zoomed rendering ===")
    axes[1].set_xlim(100, 300)
    axes[1].set_ylim(80, 220)

    # render again on second axis (zoomed view)
    container.measure_and_layout(renderer)
    renderer.render_component(axes[1], container, adjust_lims=False)
    axes[1].set_title("Zoomed View (2x)")

    # add a title
    plt.suptitle("Line Width Tracking Test - 'data' mode", fontsize=16)

    # save and show
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    print("\n=== Saving figure ===")
    plt.savefig("tracking_test.png", dpi=100)
    plt.show()


def create_container():
    """create a simple container for testing line width tracking"""
    container = Container(
        id="main-container",
        min_dimensions=Size(width=400, height=300),
        style=VisualStyle(
            background_color="#f8f8f8",
            border_color="black",
            border_width=2,
            border_width_mode="data",  # use data units
            corner_radius=5,
            padding=(20, 20, 20, 20),
        ),
        layout=LayoutConstraints(
            direction="column",
            align_items="stretch",
            justify_content="space-around",
            gap=10,
        ),
    )

    # add a title
    header = Container(
        id="header",
        min_dimensions=Size(width=0, height=40),
        style=VisualStyle(
            background_color="#e0e0e0",
            border_color="#666666",
            border_width=1,
            border_width_mode="data",
            padding=(5, 5, 5, 5),
        ),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="center",
        ),
    )

    header.children = [
        Text(
            id="title",
            text="Data Unit Line Width Test",
            font_size=20,
            color="black",
            align="center",
        )
    ]

    # create boxes with different line widths
    boxes_row = Container(
        id="boxes-row",
        min_dimensions=Size(width=0, height=200),
        layout=LayoutConstraints(
            direction="row",
            align_items="center",
            justify_content="space-around",
            gap=10,
        ),
    )

    # create boxes with varying line widths in data units
    widths = [2, 5, 10]
    boxes = []
    for width in widths:
        box = Container(
            id=f"box-{width}",
            min_dimensions=Size(width=100, height=150),
            style=VisualStyle(
                background_color="white",
                border_color="blue",
                border_width=width,  # key - data units!
                border_width_mode="data",
                corner_radius=10,
            ),
            layout=LayoutConstraints(
                direction="column",
                align_items="center",
                justify_content="center",
                gap=5,
            ),
        )

        # add label
        box.children = [
            Text(
                id=f"width-label-{width}",
                text=f"Width: {width} data units",
                font_size=14,
                color="black",
                align="center",
            ),
            Text(
                id=f"info-{width}",
                text="Line width should scale\nwith zoom level",
                font_size=10,
                color="#666666",
                align="center",
            ),
        ]

        boxes.append(box)

    boxes_row.children = boxes

    # add all elements to container
    container.children = [header, boxes_row]

    return container


if __name__ == "__main__":
    test_simple_container()
    test_row_layout()
    test_nested_layout()
    test_interdependent_layout()
    test_text_component()
    test_offset_examples()
    test_svg_integration()
    test_simple_line_width()
    test_zoom_linewidth()
    test_tracking()
