import numpy as np
import matplotlib.pyplot as plt
from jeanplot.models import Size, BoxStyle, LayoutConstraints, Offset
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.container import Container
from jeanplot.text import Text
from jeanplot.svg import SVGElement
from jeanplot.table import Table, ColumnStyle, CellStyle  # Import Table components
from jeanplot.style import jstyle, JStyle
from jeanplot.debug import set_debug
from jeanplot.table import Table, ColumnStyle, CellStyle, TableCell
from jeanplot.svg import SVGElement, make_svg_line
from jeanplot.models import Transform

# Basic theme setup (or load from file)
jstyle.styles = {
    "Table": {
        "style.border_width": 0.5,
    },
    "TableCell": {
        "style.padding": [7, 10, 7, 10],
        "style.border_color": "#ccc",
        "style.border_width": 0.5,
        "style.background_color": "#ffffffbb",
        "Text": {"align": "center", "vertical_align": "middle"},
    },
    "TableCell[style_class=table-header-cell]": {
        "style.background_color": "#eee",
        "style.margin": [0, 0, 7, 0],
        "style.padding": [10, 10, 10, 10],
        "Text": {"font_weight": "bold"},
    },
    "*[style_class=col-2": {
        "style.background_color": "red",
    },
}

set_debug(False)


def test_simple_table():
    """Demonstrate a basic table with default settings."""

    table_data = [
        ["Header 1", "Header 2", "Header 3"],
        ["Row 1, Col 1", "R1C2", 123.45],
        ["Row 2, Col 1", "Longer text in R2C2", -5],
        ["Row 3, Col 1", "R3C2", 99],
    ]

    simple_table = Table(
        id="simple-table",
        data=table_data,
        header_rows=1,  # first row is a header
    )

    root = Container(
        id="table-root",
        style=BoxStyle(padding=(20, 20, 20, 20), background_color="#ffffff"),
        layout=LayoutConstraints(align_items="center"),  # Center table horizontally
        children=[simple_table],
    )

    renderer = MatplotlibRenderer(debug=False)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.set_aspect("equal")
    ax.set_title("Simple Table Example")
    ax.axis("off")  # Hide plot axes

    renderer.render_component(ax, root, adjust_lims=True)

    plt.tight_layout()
    plt.show()

    return root, renderer


def test_advanced_table():
    """Demonstrate advanced table features and styling."""

    table_data = [
        ["Item Name", "Value", "Status"],  # header row
        ["Component A", 105.2, "Active"],
        [
            "Component B",
            -20,
            TableCell(children=[Text(text="Inactive")], style_class=["inact"]),
        ],
        [  # row with an SVG element
            TableCell(
                children=[
                    SVGElement(
                        svg_content=make_svg_line(40, 5, "green"),
                    )
                ],
            ),
            5.5,
            "Pending",
        ],
        ["Component D", 10, "Active"],
    ]

    column_styles = [
        ColumnStyle(width="auto"),
        ColumnStyle(width=100),
        ColumnStyle(width="auto"),
    ]

    advanced_table = Table(
        id="advanced-table",
        data=table_data,
        header_rows=1,
        column_styles=column_styles,
        border_collapse="separate",
        is_overlay=True,
    )

    tcopies = []

    for i in range(4):
        huemapcolors = [
            "#ff000088",
            "#ff800088",
            "#ffff0088",
            "#80ff0088",
        ]
        local_style = JStyle(
            {
                "Table": {
                    "style.background_color": huemapcolors[i],
                },
            }
        )
        from copy import deepcopy

        t = deepcopy(advanced_table)
        local_style.apply(t)
        t.transform = Transform(
            rotate=-20,
            skew_x=30,
            scale=(1, 0.8),
            translate=(0, -50 * i),
        )
        tcopies.append(t)
        for j in range(1, 5):
            t.data[j][1] = 10 * i + j
        for j in range(1, 5):
            t.data[j][0] = f"Component {i}-{j}"
        for j in range(1, 5):
            t.data[j][2] = f"Status {i}-{j}"

        # # add i extra rows
        # t.data.extend(
        #     [
        #         [f"Extra {i}-{j}", 0, "Inactive"]
        #         for j in range(i*3)
        #     ]
        # )

        t.build_table()

    root = Container(children=tcopies)

    renderer = MatplotlibRenderer(debug=False)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=250)
    # ax.set_title("Advanced Table Example")
    renderer.render_component(ax, root, adjust_lims=True)
    # ax.axis("off")
    # ax.set_xlim(0, 300)
    ax.set_ylim(-150, 200)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    return root, renderer


if __name__ == "__main__":
    test_simple_table()
    test_advanced_table()
