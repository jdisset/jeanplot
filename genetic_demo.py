from jeanplot.component import Component
import numpy as np
import matplotlib.pyplot as plt
from jeanplot.models import Transform, Size, VisualStyle, LayoutConstraints, Offset
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.container import Container
from jeanplot.genetic_elements import GeneticPart
from jeanplot.text import Text
from jeanplot.svg import SVGElement

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
        background_color="#00000000",
        border_color="#ccc",
        border_width=1,
        corner_radius=5,
        padding=(10, 10, 10, 10),
    ),
)

container.add_child(
    GeneticPart(
        part_type="fluo_marker",
        id="genetic-part",
    )
)


# container.children = [box1, box2, box3]

# create renderer and context
renderer = MatplotlibRenderer()
fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
ax.set_aspect("equal")

# measure and render
container.measure(renderer)
renderer.render_component(ax, container)

plt.show()
