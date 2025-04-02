from typing import Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator, PrivateAttr, computed_field
from jeanplot.utils import load_file_if_exists
from pathlib import Path
from functools import partial
import numpy as np

from jeanplot.svg import SVGElement, get_svg_data_from_string, make_svg_line, SVGContent
from jeanplot.container import Container
from jeanplot.component import Component
from jeanplot.text import Text
from jeanplot.models import Transform, Size, VisualStyle, LayoutConstraints, Offset, AnchorPoint


MARKER_ALIAS = {
    "NeonGreen": "mNeonGreen",
    "L0.G_mNeonGreen": "mNeonGreen",
    "L0.G_iRFPmystery": "iRFPmystery",
    "eBFP": "eBFP2",
}
MARKER_ALIAS = {k.upper(): v for k, v in MARKER_ALIAS.items()}

DEFAULT_RESOURCE_PATH = "pkg:jeanplot:resources"

AGGREGATION_LOGO = DEFAULT_RESOURCE_PATH + "/parts/aggregation.svg"
PLASMID_LOGO = DEFAULT_RESOURCE_PATH + "/parts/l2.svg"


class TranscriptionUnit(Container):
    """render an SVG line representing the transcription unit, in the middle"""

    line_thickness: float = 1
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row", align_items="center", justify_content="space-between", gap=5
        )
    )

    def render(self, renderer, context, matrix: np.ndarray):
        svg_line_content = make_svg_line(self._dimensions.width, self.line_thickness, "#000000")
        svg_line = SVGElement(
            svg_content=svg_line_content,
            transform=Transform(translate=(0, (self._dimensions.height - self.line_thickness) / 2)),
        )
        svg_line._dimensions = Size(width=self._dimensions.width, height=2)

        svg_line_matrix = svg_line.compute_world_matrix(matrix)
        svg_line.render(renderer, context, svg_line_matrix)

        Container.render(self, renderer, context, matrix)


class CoTX(BaseModel):
    stype: str = "cotx"
    marker: Optional[str] = None
    ratios: list[float] = []


class Plasmid(BaseModel):
    stype: str = "plasmid"
    marker: Optional[str] = None


class Source(Container):
    multi_type: Optional[CoTX | Plasmid] = None
    label: Optional[Container] = None

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="column", align_items="center", justify_content="center", gap=5
        )
    )

    @model_validator(mode="before")
    def style_source(cls, values):
        mtype = values.get("multi_type")
        if not mtype:
            return values

        if not values.get("label"):
            if mtype.marker:
                txt = MARKER_ALIAS.get(mtype.marker.upper(), mtype.marker)
                if isinstance(mtype, CoTX):
                    ratios_txt = ":".join([f"{r:.2g}" for r in mtype.ratios])
                    txt += f"  {ratios_txt}"
                svgpath = AGGREGATION_LOGO if isinstance(mtype, CoTX) else PLASMID_LOGO
                values["label"] = Container(
                    style_class="source_tag",
                    is_overlay=True,
                    children=[
                        SVGElement(svg_content=svgpath),
                        Text(text=txt),
                    ],
                )

                if "children" not in values:
                    values["children"] = []

                values["children"].append(values["label"])

        return values

    def render(self, renderer, context, matrix: np.ndarray):
        Container.render(self, renderer, context, matrix)


class GeneticPart(SVGElement, Container):
    """base class for genetic parts, extends SVGElement with biological features.
    Is also a container - most of the time will only contain one Text element, the label."""

    part_type: str
    part_name: Optional[str] = None
    label: Optional[Text] = None

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(justify_content="center", align_items="start")
    )
    auto_resource_path: str = DEFAULT_RESOURCE_PATH

    # init can take part name as positional argument
    def __init__(self, part_name: Optional[str] = None, **kwargs):
        BaseModel.__init__(self, part_name=part_name, **kwargs)

    def model_post_init(self, *args, **kwargs):
        # load SVG content from file if not already provided
        if self.svg_content is None:
            if self.part_name:
                svg_path = f"{self.auto_resource_path}/parts/{self.part_type}.{self.part_name}.svg"
                svg_content = load_file_if_exists(svg_path)
                if svg_content:
                    self.svg_content = get_svg_data_from_string(svg_content)
            else:
                svg_path = f"{self.auto_resource_path}/parts/{self.part_type}.svg"
                svg_content = load_file_if_exists(svg_path)
                if svg_content:
                    self.svg_content = get_svg_data_from_string(svg_content)

    @model_validator(mode="after")
    def set_label(self):
        """set label text if provided"""
        if self.label and not len(self.children):
            self.add_child(self.label)
        return self

    def render(self, renderer, context, matrix: np.ndarray):
        """render genetic part and optional label"""
        SVGElement.render(self, renderer, context, matrix)
        Container.render(self, renderer, context, matrix)


class ERN(GeneticPart):
    part_type: str = "ERN"
    main_color: str = "#AAAAAA"
    secondary_color: str = "#111111"

    anchor_points: list[AnchorPoint] = [
        AnchorPoint(offset=Offset(relative=(0.77, -0.1)), direction=(0, -1)),
        AnchorPoint(offset=Offset(relative=(0.77, 1.1)), direction=(0, 1)),
    ]

    @model_validator(mode="before")
    def set_label_to_name(cls, values):
        """set label to part_name if no label is provided"""
        if not values.get("label") and values.get("part_name"):
            values["label"] = Text(
                text=values["part_name"],
                align="center",
                vertical_align="middle",
                font_size=9,
                offset=Offset(relative=(-0.35, 0)),
            )
        return values


class Promoter(GeneticPart):
    part_type: str = "Promoter"
    offset: Offset = Offset(relative=(0, -0.5), absolute=(0, 0.5))


class Terminator(GeneticPart):
    part_type: str = "Terminator"
    offset: Offset = Offset(relative=(0.5, -0.5), absolute=(-0.5, 0.5))
    style: VisualStyle = VisualStyle(margin=(0, 0, 0, -10))


class UorfGroup(GeneticPart):
    part_type: str = "uORF_group"
    style: VisualStyle = VisualStyle(margin=(0, 0, 0, -10))
    offset: Offset = Offset(relative=(0, 0.1))


class ERN5pRecog(GeneticPart):
    part_type: str = "ERN_recog_site_5p"
    offset: Offset = Offset(relative=(0, -0.4), absolute=(0, 0.5))

    anchor_points: list[AnchorPoint] = [
        AnchorPoint(offset=Offset(relative=(0.5, -0.1)), direction=(0, 1), min_segment=12),
        AnchorPoint(offset=Offset(relative=(0.5, 1.1)), direction=(0, -1), min_segment=20),
    ]
