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

# enable global debugging
set_debug(True)

# create renderer
renderer = MatplotlibRenderer(debug=False)  # disable renderer debug if scene/components have it
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
ax.set_aspect("equal")

# create main container (scene)
scene = Container(
    id="scene",
    debug=True,  # enable debugging for the scene
    layout=LayoutConstraints(direction="column", gap=20),  # TUs stacked vertically
)

# --- create first transcription unit ---
tu1 = TranscriptionUnit(id="tu1", debug=False)
ern0 = ERN(part_name="Csy4", id="ern0", debug=False)  # disable debug on parts if TU is debugged
promoter1 = Promoter(id="promoter1")
uorf1 = UorfGroup(id="uorf1")
term1 = Terminator(id="term1")
tu1.children = [promoter1, uorf1, ern0, term1]

# --- create second transcription unit ---
tu2 = TranscriptionUnit(id="tu2", debug=False)
ern1 = ERN(part_name="CasE", id="ern1", debug=False)
promoter2 = Promoter(id="promoter2")
uorf2 = UorfGroup(id="uorf2")
term2 = Terminator(id="term2")
tu2.children = [promoter2, uorf2, ern1, term2]

# add transcription units to scene
scene.add_child(tu1)
scene.add_child(tu2)

# create connection between ERN components
# connection points are calculated dynamically during measure/layout
arrow_conn = Connection(
    id="connection",
    start_component=ern0,
    end_component=uorf2,
    start_offset=Offset(relative=(0.77, 0.9)),  # center of start component
    end_offset=Offset(relative=(0.5, -0.1)),  # center of end component
    color="red",
    width=1,
    # curve_type=OrthogonalCurve(start_direction="down", end_direction="up"),
    curve_type=SimpleBezierCurve(start_vec=(0, 20), end_vec=(0, -20)),
    end_cap=LineEndFlat(stroke_color="red", length=8.0, stroke_width=1),
)

# add connection to scene (as an overlay)
scene.add_child(arrow_conn)

# --- measure, layout, and render ---
# this single call processes the entire hierarchy
print("Measuring and Laying out Scene...")
scene.measure_and_layout(renderer)

print(f"\nFinal scene dimensions: {scene._dimensions.width:.1f}x{scene._dimensions.height:.1f}")
if hasattr(arrow_conn, "_dimensions"):
    print(
        f"Connection dimensions: {arrow_conn._dimensions.width:.1f}x{arrow_conn._dimensions.height:.1f}"
    )
    print(
        f"Connection position: ({arrow_conn.transform.translate[0]:.1f}, {arrow_conn.transform.translate[1]:.1f})"
    )
else:
    print("Connection not measured.")


# render the scene
print("\nRendering Scene...")
renderer.render_component(ax, scene, adjust_lims=True)  # adjust limits automatically

# --- display ---
# adjust view slightly for better framing if needed (adjust_lims should handle most)
# plt.ylim(-10, 80)
# plt.xlim(-20, 120)
plt.grid(True, alpha=0.3, linestyle=":")
plt.title("Genetic Circuit with Connection")
plt.tight_layout()
plt.show()

print("\nDone.")
