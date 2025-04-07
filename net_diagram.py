import matplotlib.pyplot as plt
import copy
from typing import Any, Optional, List, Dict, Tuple, Annotated
from pydantic import BaseModel, Field, PrivateAttr, BeforeValidator, model_validator
from biocomp.utils import load_lib
from sqlmodel import Session
import biocomptools.toollib.common as cm
import biocomptools.toollib.models as md
from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector
import pandas as pd
from jeanplot.debug import set_debug
from jeanplot.style import jstyle
import dracon as dr
import random
import json
from jeanplot.component import Component
from jeanplot.container import Container
from jeanplot.models import Size, BoxStyle, LayoutConstraints, Offset, Shadow
from jeanplot.text import Text
from jeanplot.svg import SVGElement  # Keep import
from jeanplot.connector import (
    Connection,
    StraightCurve,
    LineEndArrow,
    SimpleBezierCurve,
    OrthogonalCurve,
)
from jeanplot.network_utils import get_tu_informations, TUInfo
from jeanplot.debug import debug_print
from jeanplot.component import Component
import numpy as np
import matplotlib.pyplot as plt
from jeanplot.models import Transform, Size, BoxStyle, LayoutConstraints, Offset
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.container import Container
from jeanplot.component import Component, Overlay, AnchorComponent, get_world_origin
from jeanplot.text import Text
from jeanplot.svg import (
    SVGElement,
    SVGContent,
    SVGPathData,
    LineEndArrow,
    LineEndType,
    LineStyle,
    LineEndCircle,
    LineEndFlat,
    create_arrow_cap,
    create_circle_cap,
    create_flat_cap,
)

# Import the new diagram class
from jeanplot.network_diagram import draw_network_diagram

# theme loading remains the same
theme = dr.load("pkg:jeanplot:resources/themes/default", enable_interpolation=True, raw_dict=True)
dr.resolve_all_lazy(theme)
jstyle.styles = theme

# only enable for debugging
# set_debug(True)


lib = load_lib()
engine = md.get_biocompdb_sqlite_engine(cm.config.db.sqlite.path)


netset = NetworkSet(
    content=[NetworkSelector(experiment_name="2023-11-17_PguConstraints1_BP_DR", recipe_name="")]
)

with Session(engine) as session:
    netset.run_selectors(session)
    data = netset.get_networks_and_data(session)
    networks, datafiles = zip(*data)

# random shuffle of networks
networks = list(networks)
# random.shuffle(networks)


# for mnet in networks[1:2]:
#     # ensure network is built
#     mnet.build(lib)
#     network = mnet._network
#
#     fig, ax, diagram = draw_network_diagram(
#         network,
#         figsize=(15, 10),  # Adjust figsize as needed
#         dpi=150,
#         debug=False,  # Enable component debug boxes if needed
#     )
#
#     ax.set_title(mnet.name, fontsize=10)
#     plt.show()  # Show plot interactively
#     fig.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for title/text
#     # save as pdf:
#     fig.savefig(f"{mnet.name}.pdf", bbox_inches="tight", dpi=100)
#
# print("Diagram generation complete.")


# TODO:
# start with a function to annotate erns layer
# something that spits out ERN node number in topological order (e.g. [[0,1],[2]] for a bandpass)
# that will also be used to annotate the ERN node themselves in their extra field (which will be used by the comp nodes)
# then make a function that draws an ERN node cleanly
# then an aggregation that can be collapsed (a single TU dash) or expanded (the full vertical rounded rectangle with all TUs in it)
# then from there it should be pretty easy to draw the whole network

mnet = networks[23]
mnet.build(lib)
net = mnet._network
ntypes = net.compute_graph["type"].unique()
node_counts = net.compute_graph["type"].value_counts()
node_counts["sequestron_ERN"]
ern_nodes = net.compute_graph[net.compute_graph["type"] == "sequestron_ERN"]
ern_indices = ern_nodes.index
topo = net.topological_order(ern_indices)


ERN_SIZE = Size(width=70, height=70)

style = {
    # --- Keep other ComputeNode, TranscriptionNode etc. styles ---
    "ComputeNode": {
        "min_dimensions": Size(width=18, height=18),  # or 20x20 as in net_diagram2
        "style.corner_radius": 1e6,
        "style.background_color": "#aaa",
        "style.border_color": "#111",
        "style.border_width": 1,
        "style.margin": (10, 10, 10, 10),
        "layout.align_items": "center",
        "layout.justify_content": "center",
        "Text": {
            "font_size": 8,
            "color": "#fff",
            "vertical_align": "middle",
            "align": "center",
        },
    },
    "TranscriptionNode": {
        "style.background_color": "#777",
    },
    "TranslationNode": {  # Global style for Tl
        "style.background_color": "#333",
    },
    "ERNNode": {
        "min_dimensions": ERN_SIZE,
        "max_dimensions": ERN_SIZE,
        "style.background_color": "#eee",
        "style.border_color": "#111",
        "style.border_style": "dashed",
        "style.dash_sequence": (4, 3),
        "style.border_width": 2,  # or same
        "TranslationNode": {  # Override global Tl style for children of ERNNode
            "style.background_color": "#ee3350",
            "offset": Offset(parent_relative=(0, 0.9), relative=(0, -1)),
            "is_overlay": True,
        },
        "TranscriptionNode": {  # Style for Tx children of ERNNode
            "style.background_color": "#555",
            "offset": Offset(parent_relative=(0, 0.1), relative=(0, 0)),
            "is_overlay": True,
        },
        "Connection[style_class=tlconn]": {
            "color": "#ee3350",
            "line_width": 3,
            "start_offset": Offset(parent_relative=(0, 0), relative=(1, 0.5)),
            "end_offset": Offset(absolute=(12, 5)),
            "end_cap": LineEndFlat(
                stroke_color="#ee3350",
                stroke_width=3,
                length=8,
            ),
        },
        "Connection[style_class=txconn]": {
            "color": "#111",
            "line_width": 2,
            "start_offset": Offset(parent_relative=(0, 0), relative=(0.8, 0.88)),
            "curve_type.start_vector": (15, 15),
            "curve_type.end_vector": (-35, 0),
        },
    },
    "FluoNode": {
        "style.background_color": "#FFBB79",
        "style.border_color": "#00000030",
        "style.shadow": Shadow(color="#FF8006", blur_radius=15, resolution=0.01),
    },
    "InvNode": {
        "style.background_color": "#eee",
        "style.border_color": "#111",
        "Text": {"color": "#222"},
    },
    "TUNode": {  # Base style for all TUs
        "min_dimensions": Size(width=17, height=4.5),
        "style.background_color": "#111",
        "style.border_width": 0,
        "style.margin": (0, 0, 0, 0),  # Ensure no default margins interfere with centering
        "offset": Offset(),
    },
    "AggregationNode": {  # Base style for Aggregation (applies to both states unless overridden)
        "style.shadow": Shadow(color="#FFA068aa", blur_radius=15, resolution=1),
    },
    "AggregationNode[!collapsed]": {  # Expanded state
        "style.background_color": "#FFE2D1",
        "style.border_color": "#FFA068",
        "style.border_width": 1.5,
        "style.padding": (12, 4, 12, 4),
        "layout": LayoutConstraints(
            direction="column",
            justify_content="center",
            gap=5,
        ),
        "TUNode": {
            "is_overlay": False,
        },
    },
    "AggregationNode[collapsed]": {  # Collapsed state
        "min_dimensions": Size(width=17, height=4.5),
        "max_dimensions": Size(width=17, height=4.5),
        "style.border_width": 0,
        "style.padding": (0, 0, 0, 0),
        "style.shadow": {  # Use dict for partial update
            "color": "#FFA068",
            "blur_radius": 20,
            "resolution": 0.1,
        },
        "TUNode": {
            "is_overlay": True,
            "offset": Offset(absolute=(0, 0)),
        },
    },
}


jstyle.clear()
jstyle.update(style)


class ComputeNode(Container):
    node_type: str = "unknown"
    node_label: Optional[str] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.style_class.append(f"node-type-{self.node_type}")
        if self.node_label:
            self.add_child(Text(text=self.node_label, id=f"lbl_{self.id}"))


class TranscriptionNode(ComputeNode):
    node_type: str = "transcription"
    node_label: Optional[str] = "Tx"


class TranslationNode(ComputeNode):
    node_type: str = "translation"
    node_label: Optional[str] = "Tl"


class ERNNode(ComputeNode):
    node_type: str = "sequestron_ERN"

    _tx_node: TranscriptionNode = PrivateAttr()
    _tl_node: TranslationNode = PrivateAttr()
    _out: AnchorComponent = PrivateAttr()
    _center: AnchorComponent = PrivateAttr()

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._tx_node = TranscriptionNode(id=f"tx_{self.id}", is_overlay=True)
        self._tl_node = TranslationNode(id=f"tl_{self.id}", is_overlay=True)
        self._out = AnchorComponent(
            style_class=["ernout"],
            id=f"ernout_{self.id}",
            offset=Offset(parent_relative=(1.0, 0.5)),
        )

        self._center = AnchorComponent(
            style_class=["erncenter"],
            id=f"erncenter_{self.id}",
        )
        self._tx_connector = Connection(
            start_component=self._tx_node,
            end_component=self._out,
            style_class=["txconn"],
            curve_type=SimpleBezierCurve(start_vector=(10, 15), end_vector=(-25, 0)),
            auto_route=False,
        )

        self._tl_connector = Connection(
            start_component=self._tl_node,
            end_component=self._center,
            style_class=["tlconn"],
            curve_type=OrthogonalCurve(
                start_direction="right", start_length=15, end_direction="down", end_length=15
            ),
            end_cap=LineEndFlat(),
            auto_route=False,
        )

        self.add_child(self._tx_node)
        self.add_child(self._tl_node)
        self.add_child(self._out)
        self.add_child(self._center)
        self.add_child(self._tx_connector)
        self.add_child(self._tl_connector)


class FluoNode(ComputeNode):
    node_type: str = "fluorescent"
    node_label: Optional[str] = "Y"


class InvNode(ComputeNode):
    node_type: str = "inverted"
    node_label: Optional[str] = "Inv"


class TUNode(ComputeNode):
    node_type: str = "tu"


class AggregationNode(ComputeNode):
    node_type: str = "aggregation"
    collapsed: bool = False


n1 = TranslationNode()
n2 = TranscriptionNode()
ern = ERNNode()
fl = FluoNode()
inv = InvNode()

agg = AggregationNode()
agg.children = [
    TUNode(id="A"),
    TUNode(id="B"),
    TUNode(id="C"),
]


agg2 = copy.deepcopy(agg)
agg2.collapsed = True

root = Container(
    children=[n1, n2, ern, fl, inv, agg, agg2],
    layout=LayoutConstraints(direction="row", gap=10, align_items="center"),
)


renderer = MatplotlibRenderer()
fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
ax.set_aspect("equal")
root.measure_and_layout(renderer)
ern._tl_connector._world_start
renderer.render_component(ax, root)
