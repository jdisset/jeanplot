"""Minimal jeanplot example: a tiny two-box layout rendered with matplotlib."""

import matplotlib.pyplot as plt

from jeanplot import (
    BoxStyle,
    Container,
    LayoutConstraints,
    MatplotlibRenderer,
    Size,
    Text,
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
MatplotlibRenderer().render_component(ax, root, adjust_lims=True)
plt.show()
