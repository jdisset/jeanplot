import json
from typing import (
    Any,
)

from matplotlib.axes import Axes

from jeanplot.network_diagram import NetworkDiagram
from jeanplot.network_schematic import NetworkGeneticSchematic
from jeanplot.container import Container
from jeanplot.text import Text
from jeanplot.models import LayoutConstraints, BoxStyle, Size, Shadow
from jeanplot.matplotlib_renderer import MatplotlibRenderer


def render_network_card(mnet: Any, lib: Any, ax: Axes, show_recipe: bool = False):
    """
    mnet: The biocomp ManagedNetwork object to render.
    lib: The biocomp Library object needed to build the network.
    ax: The matplotlib Axes object to render onto.
    show_recipe: If True, includes a box displaying the network recipe JSON.
    """

    mnet.build(lib)
    net = mnet._network

    diagram = NetworkDiagram(network=net)
    schematic = NetworkGeneticSchematic(network=net)

    # --- Extract Metadata ---
    infos = {
        "recipe": mnet.recipe.content.get("name", "N/A").rstrip(),
        "experiment": mnet.recipe.experiment.content.get("name", "N/A").rstrip(),
        "operator": mnet.recipe.experiment.content.get("tx_operator", "N/A"),
        "machine": mnet.recipe.experiment.content.get("machine", "N/A"),
        "cell line": mnet.recipe.experiment.content.get("cell_line", "N/A"),
        "protocol": mnet.recipe.experiment.content.get("transfection_protocol", "N/A"),
    }
    recipe_txt = mnet.recipe.content

    # --- Build Info Title ---
    info_title = Container(
        children=[
            Text(
                text=mnet.name,
                font_size=8,
                color="#333",
                style_class=["info_title"],
                vertical_align="middle",
            ),
        ],
        style_class=["info_title_box"],
        layout=LayoutConstraints(
            direction="column",
            gap=5,
            justify_content="center",
            align_items="center",
        ),
        style=BoxStyle(
            padding=(5, 5, 5, 5),
            corner_radius=3,
        ),
    )

    # --- Build Info Box (Metadata Table) ---
    def make_line(k, v):
        return Container(
            children=[
                Text(
                    text=f"{k}:",
                    color="#999",
                    font_size=6,
                    style_class=["info_key"],
                    vertical_align="middle",
                    align="right",
                ),
                Text(
                    min_dimensions=Size(width=50, height=0),
                    text=f"{v}",
                    color="black",
                    font_size=6,
                    style_class=["info_value"],
                    vertical_align="middle",
                    align="left",
                ),
            ],
            layout=LayoutConstraints(
                direction="row",
                gap=10,
                justify_content="start",
            ),
        )

    info_text = [make_line(k, v) for k, v in infos.items()]

    def make_col(row):
        return Container(
            children=row,
            layout=LayoutConstraints(
                direction="column",
                gap=5,
                justify_content="start",
                align_items="stretch",
            ),
            style_class=["info_box"],
            style=BoxStyle(
                padding=(10, 10, 10, 10),
                corner_radius=3,
                border_color="#aaa",
                border_width=0.2,
            ),
        )

    NROWS = 3
    info_box = Container(
        children=[make_col(info_text[i : i + NROWS]) for i in range(0, len(info_text), NROWS)],
        layout=LayoutConstraints(
            direction="row",
            gap=15,
            justify_content="center",
            align_items="start",
        ),
        style=BoxStyle(
            margin=(0, 10, 0, 0),
        ),
    )

    maincard = Container(
        children=[diagram, schematic, info_box],
        layout=LayoutConstraints(
            direction="column",
            gap=40,
            justify_content="space-around",
            align_items="center",
        ),
        style_class=["maincard"],
        style=BoxStyle(
            padding=(20, 20, 20, 20),
        ),
    )

    # --- (Optional) Recipe Box ---
    if show_recipe:
        recipebox = Container(
            children=[
                Text(
                    text=json.dumps(recipe_txt, indent=2),
                    font_size=5,
                    color="#787471",
                    style_class=["info_box"],
                    vertical_align="middle",
                ),
                Text(
                    text="Recipe",
                    font_size=8,
                    color="#787471",
                    style_class=["info_title"],
                    vertical_align="middle",
                ),
            ],
            layout=LayoutConstraints(
                direction="column",
                gap=10,
                justify_content="center",
                align_items="center",
            ),
            style_class=["recipebox"],
            style=BoxStyle(
                padding=(30, 50, 30, 30),
                background_color="#FEF9F2",
                corner_radius=3,
                border_color="#555",
                border_width=0.25,
            ),
        )

        body = Container(
            children=[recipebox, maincard],
            layout=LayoutConstraints(
                direction="row",
                gap=30,
                justify_content="start",
                align_items="stretch",
            ),
            z_index=-100,
            style=BoxStyle(
                padding=(10, 10, 10, 10),
                corner_radius=3,
                background_color="#fff",
                border_color="#222",
                border_width=0.25,
                shadow=Shadow(
                    color="#aaa4",
                    blur_radius=25,
                ),
            ),
        )
    else:
        body = Container(
            children=[maincard],
            layout=LayoutConstraints(
                direction="row",
                justify_content="center",
                align_items="stretch",
            ),
            z_index=-100,
            style=BoxStyle(
                padding=(10, 10, 10, 10),
                corner_radius=3,
                background_color="#fff",
                border_color="#222",
                border_width=0.25,
                shadow=Shadow(
                    color="#aaa4",
                    blur_radius=25,
                ),
            ),
        )

    root = Container(
        children=[body, info_title],
        layout=LayoutConstraints(
            direction="column",
            gap=10,
            justify_content="center",
            align_items="stretch",
        ),
    )

    # --- Render ---
    ax.set_aspect("equal")
    ax.axis("off")
    renderer = MatplotlibRenderer()
    renderer.render_component(ax, root, adjust_lims=True)


def render_network_diagram(mnet: Any, lib: Any, ax: Axes):
    mnet.build(lib)
    net = mnet.network
    diagram = NetworkDiagram(network=net)

    root = Container(
        children=[diagram],
        layout=LayoutConstraints(
            direction="row",
            justify_content="center",
            align_items="stretch",
        ),
    )
    ax.set_aspect("equal")
    ax.axis("off")
    renderer = MatplotlibRenderer()
    renderer.render_component(ax, root, adjust_lims=True)


def render_circuit_schematic(mnet: Any, lib: Any, ax: Axes):
    mnet.build(lib)
    net = mnet.network
    schema = NetworkGeneticSchematic(network=net)

    root = Container(
        children=[schema],
        layout=LayoutConstraints(
            direction="row",
            justify_content="center",
            align_items="stretch",
        ),
    )
    ax.set_aspect("equal")
    ax.axis("off")
    # transparent bg
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    ax.patch.set_visible(False)
    renderer = MatplotlibRenderer()
    renderer.render_component(ax, root, adjust_lims=True)
