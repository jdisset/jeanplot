# file: genetic_demo.py
import numpy as np
import matplotlib.pyplot as plt
from jeanplot.models import Size, VisualStyle, LayoutConstraints, Offset
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.container import Container
from jeanplot.connector import Connection, StraightCurve, OrthogonalCurve, SimpleBezierCurve
from jeanplot.genetic_elements import (
    ERN,
    Promoter,
    TranscriptionUnit,
    Terminator,
    UorfGroup,
)
from jeanplot.svg import LineEndFlat
from jeanplot.debug import set_debug

set_debug(False)

renderer = MatplotlibRenderer()
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
ax.set_aspect("equal")

scene = Container(
    id="scene",
    debug=False,  # enable debugging for the scene
    layout=LayoutConstraints(direction="column", gap=20),  # TUs stacked vertically
)

tu1 = TranscriptionUnit(id="tu1", debug=False)
ern0 = ERN(part_name="Csy4", id="ern0", debug=False)  # disable debug on parts if TU is debugged
promoter1 = Promoter(id="promoter1")
uorf1 = UorfGroup(id="uorf1")
term1 = Terminator(id="term1")
tu1.children = [promoter1, uorf1, ern0, term1]

tu2 = TranscriptionUnit(id="tu2", debug=False)
ern1 = ERN(part_name="CasE", id="ern1", debug=False)
promoter2 = Promoter(id="promoter2")
uorf2 = UorfGroup(id="uorf2")
term2 = Terminator(id="term2")
tu2.children = [promoter2, uorf2, ern1, term2]

arrow_conn = Connection(
    start_component=ern0,
    end_component=uorf2,
    start_offset=Offset(relative=(0.77, 0.9)),
    end_offset=Offset(relative=(0.5, -0.2)),
    # curve_type=StraightCurve(),
    # curve_type=SimpleBezierCurve(
    #     start_vector=(0, 20),
    #     end_vector=(0, -20),
    # ),
    curve_type=OrthogonalCurve(
        start_direction="down",
        end_direction="down",
        end_length=15,
        corner_radius=4.0,
    ),
    color="red",
    line_width=1,
    end_cap=LineEndFlat(stroke_width=1, stroke_color="red"),
)

scene.add_children([tu1, tu2, arrow_conn])
scene.measure_and_layout(renderer)
renderer.render_component(ax, scene, adjust_lims=True)

plt.grid(True, alpha=0.5, linestyle=":")
plt.tight_layout()
plt.show()

print("\nDone.")
