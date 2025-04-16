import matplotlib.pyplot as plt
import copy
from typing import Any, Optional, List, Dict, Tuple, Annotated
from pydantic import BaseModel, Field, PrivateAttr, BeforeValidator, model_validator
from biocomp.utils import load_lib
from sqlmodel import Session
import biocomptools.toollib.common as cm
import biocomptools.toollib.models as md
from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector
from jeanplot.debug import set_debug
from jeanplot.style import jstyle
import dracon as dr
from jeanplot.container import Container
from jeanplot.models import Size, BoxStyle, LayoutConstraints, Offset, Shadow, Transform
from jeanplot.text import Text
from jeanplot.connector import (
    Connection,
    SimpleBezierCurve,
    OrthogonalCurve,
)
import numpy as np
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.component import AnchorComponent
from jeanplot.svg import LineEndFlat, LineEndCircle, LineEndArrow


# theme loading remains the same
theme = dr.load("pkg:jeanplot:resources/themes/default", enable_interpolation=True, raw_dict=True)
dr.resolve_all_lazy(theme)
jstyle.styles = theme

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
            "font_size": 7,
            "color": "#fff",
            "vertical_align": "middle",
            "align": "center",
        },
    },
    "TranscriptionNode": {
        "style.background_color": "#555",
    },
    "TranslationNode": {  # Global style for Tl
        "style.background_color": "#333",
    },
    "ERNNode": {
        "min_dimensions": ERN_SIZE,
        "max_dimensions": ERN_SIZE,
        "style.background_color": "#eee",
        "style.border_color": "#111",
        "style.border_style": "custom",
        "style.dash_sequence": (5, 5),
        "style.border_width": 1,
        "TranslationNode": {
            "style.background_color": "#ee3350",
            "offset": Offset(reference_relative=(0, 0.9), relative=(0, -1)),
        },
        "TranscriptionNode": {
            "offset": Offset(reference_relative=(0, 0.1)),
        },
        "Connection[style_class=tlconn]": {
            "color": "#ee3350",
            "line_width": 2,
            "start_offset": Offset(relative=(1, 0.5)),
            "end_offset": Offset(absolute=(15, 4)),
            "curve_type.start_direction": "right",
            "curve_type.end_direction": "up",
            "curve_type.start_length": 5,
            "curve_type.end_length": 5,
            "curve_type.corner_radius": 10,
            "z_index": -1,
            "end_cap.stroke_color": "#ee3350",
            "end_cap.stroke_width": 2,
            "end_cap.length": 10,
        },
        "Connection[style_class=txconn]": {
            "color": "#111",
            "line_width": 2,
            "z_index": -1,
            "curve_type.start_mode": "vector",
            "curve_type.end_mode": "vector",
            "curve_type.start_vector": (15, 15),
            "curve_type.end_vector": (-40, 0),
            "start_offset": Offset(relative=(0.9, 0.8)),
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
        "min_dimensions": Size(width=15, height=4),
        "style.background_color": "#111",
        "style.border_width": 0,
        "style.margin": (0, 0, 0, 0),  # Ensure no default margins interfere with centering
        "offset": Offset(),
    },
    "AggregationNode": {  # Base style for Aggregation (applies to both states unless overridden)
        "style.shadow": Shadow(color="#FFA068aa", blur_radius=15, resolution=1),
    },
    "AggregationNode[!collapsed]": {  # expanded state
        "style.background_color": "#FFFFFFAA",
        "style.border_color": "#FFA068",
        "style.border_width": 1.5,
        "style.padding": (12, 4, 12, 4),
        "layout": LayoutConstraints(
            direction="column",
            justify_content="space-around",
            gap=5,
        ),
        "TUNode": {
            "is_overlay": False,
        },
    },
    "AggregationNode[collapsed]": {  # collapsed state
        "min_dimensions": Size(width=15, height=4),
        "max_dimensions": Size(width=15, height=4),
        "style.border_width": 0,
        "style.padding": (0, 0, 0, 0),
        "style.shadow": {
            "color": "#FFA068",
            "blur_radius": 20,
            "resolution": 0.1,
        },
        "TUNode": {
            "is_overlay": True,
            "offset": Offset(absolute=(0, 0)),
        },
    },
    "DeadEndNode": {
        "style.background_color": "#FF000000",
        "style.border_width": 0,
        "Text": {
            "color": "000000",
            "font_size": 15,
            "vertical_align": "middle",
            "align": "center",
        },
    },
}


jstyle.clear()
jstyle.update(style)


class ComputeNode(Container):
    node_type: str = "unknown"
    node_label: Optional[str] = None

    layout: LayoutConstraints = LayoutConstraints(align_items="center", justify_content="center")

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.style_class.append(f"node-type-{self.node_type}")
        if self.node_label:
            self.add_child(
                Text(
                    text=self.node_label,
                    id=f"lbl_{self.id}",
                    style_class=["label"],
                    vertical_align="middle",
                    align="center",
                )
            )


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
            offset=Offset(reference_relative=(1.0, 0.5)),
        )

        self._center = AnchorComponent(
            style_class=["erncenter"],
            offset=Offset(reference_relative=(0.5, 0.5)),
        )
        self._tx_connector = Connection(
            start_component=self._tx_node,
            end_component=self._out,
            style_class=["txconn"],
            curve_type=SimpleBezierCurve(),
            auto_route=False,
        )

        self._tl_connector = Connection(
            start_component=self._tl_node,
            end_component=self._center,
            style_class=["tlconn"],
            curve_type=OrthogonalCurve(corner_radius=50, start_length=5, end_length=5),
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


class DeadEndNode(ComputeNode):
    node_type: str = "deadend"
    node_label: Optional[str] = "X"


ern = ERNNode()

n1 = TranslationNode()
n2 = TranscriptionNode()
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
de = DeadEndNode()

root = Container(
    children=[n1, n2, ern, fl, inv, agg, agg2, de],
    layout=LayoutConstraints(direction="row", gap=10, align_items="center"),
)


renderer = MatplotlibRenderer()
fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
ax.set_aspect("equal")
root.measure_and_layout(renderer)
renderer.render_component(ax, root, adjust_lims=True)
