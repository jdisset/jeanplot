import numpy as np
import matplotlib.pyplot as plt
from jeanplot.models import Size, VisualStyle, LayoutConstraints, Offset
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.container import Container
from jeanplot.connector import Connection, StraightCurve, OrthogonalCurve, SimpleBezierCurve
from jeanplot.style import jstyle
import json
from jeanplot.genetic_elements import (
    ERN,
    Promoter,
    TranscriptionUnit,
    Terminator,
    UorfGroup,
    Source,
    CoTX,
    Plasmid,
    ERN5pRecog,
)
from jeanplot.svg import LineEndFlat
from jeanplot.debug import set_debug
from pydantic import BaseModel, Field
import dracon as dr


theme = dr.load("pkg:jeanplot:resources/themes/default", enable_interpolation=True, raw_dict=True)
dr.resolve_all_lazy(theme)

print("Theme:")
print(json.dumps(theme, indent=2))
print("\n")



jstyle.styles = theme

set_debug(False)

renderer = MatplotlibRenderer()
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
ax.set_aspect("equal")

scene = Container(
    id="scene",
    debug=False,  # enable debugging for the scene
    layout=LayoutConstraints(direction="column"),
)

tu1 = TranscriptionUnit(id="tu1", debug=False)
ern0 = ERN(part_name="Csy4", id="ern0", debug=False)  # disable debug on parts if TU is debugged
promoter1 = Promoter(id="promoter1")
uorf1 = UorfGroup(id="uorf1")
term1 = Terminator(id="term1")
tu1.children = [promoter1, uorf1, ern0, term1]

src1 = Source(
    children=[tu1], id="src1", debug=False, multi_type=CoTX(marker="eBFP2", ratios=[1, 1])
)

tu2 = TranscriptionUnit(id="tu2", debug=False)
ern1 = ERN(part_name="CasE", id="ern1", debug=False)
promoter2 = Promoter(id="promoter2")
recog = ERN5pRecog(part_name="Csy4_rec", id="recog", debug=False)
term2 = Terminator(id="term2")
tu2.children = [promoter2, recog, ern1, term2]

src2 = Source(children=[tu2], id="src2", debug=False, multi_type=Plasmid(marker="mKate"))

arrow_conn = Connection(
    start_component="src1/tu1/ern0",
    end_component="src2/tu2/recog",
    curve_type=OrthogonalCurve(),
    color="black",
    line_width=0.8,
    end_cap=LineEndFlat(stroke_width=0.8, stroke_color="black", length=8),
)

scene.add_children([src1, src2, arrow_conn])
scene.measure_and_layout(renderer)
renderer.render_component(ax, scene, adjust_lims=True)

plt.grid(True, alpha=0.5, linestyle=":")
plt.tight_layout()
plt.show()

print("\nDone.")
