"""
Schematic drawing logic.
"""

from typing import Dict, List, Optional, Any, Tuple, Literal, Annotated
from pydantic import Field, PrivateAttr, model_validator, BeforeValidator, BaseModel
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from jeanplot.component import Component, AnchorComponent
from jeanplot.container import Container
from jeanplot.models import Size, BoxStyle, LayoutConstraints, Offset, Transform
from jeanplot.connector import Connection, OrthogonalCurve, SimpleBezierCurve, StraightCurve
from jeanplot.svg import LineEndFlat, LineEndCircle, LineEndArrow
from jeanplot.network_utils import (
    get_tu_informations,
    get_tu_grid_layout,
    get_interactions,
    optimize_grid_for_source_adjacency,
    _get_source_id,
    TUInfo,
    Interaction,
)
from jeanplot.style import jstyle
from jeanplot.debug import debug_print, get_logger
from jeanplot.renderer import BaseRenderer
from jeanplot.text import Text

logger = logging.getLogger(__name__)


class ComputeNode(Container):
    node_type: str = "unknown"
    node_label: Optional[str] = None
    node_id: Optional[int] = None

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
