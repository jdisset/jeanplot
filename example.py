import matplotlib.pyplot as plt
from jeanplot.models import Bounds, ContainerStyle
from jeanplot.components import Container, Card


def create_style_examples():
    """creates a demo of different card styles with various border and background options"""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    container = Container(
        id="main", direction="row", bounds=Bounds(x=10, y=10), spacing=5, debug=True
    )

    card1 = Card(
        id="card1",
        title="Regular Dashed",
        bounds=Bounds(width=100, height=150),
        title_align="center",
        style=ContainerStyle(
            background_color="lightyellow",
            border_color="navy",
            border_width=2.0,
            corner_radius=10.0,
            border_style="dashed",
        ),
    )

    card2 = Card(
        id="card2",
        title="Custom Pattern",
        bounds=Bounds(width=100, height=150),
        title_align="right",
        style=ContainerStyle(
            background_color="lightblue",
            border_color="green",
            border_width=2.0,
            corner_radius=5.0,
            border_style="custom",
            dash_sequence=(6.0, 3.0, 1.5, 3.0),
        ),
    )

    card3 = Card(
        id="card3",
        title="Fine Dotted",
        bounds=Bounds(width=100, height=150),
        title_align="left",
        style=ContainerStyle(
            background_color="lightpink",
            border_color="darkred",
            border_width=1.5,
            corner_radius=5.0,
            border_style="custom",
            dash_sequence=(1.0, 1.0),
        ),
    )

    container.children = [card1, card2, card3]
    container.render(ax)

    ax.set_xlim(-10, container.bounds.width + 40)
    ax.set_ylim(-10, container.bounds.height + 40)
    ax.set_aspect("equal")
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    create_style_examples()
