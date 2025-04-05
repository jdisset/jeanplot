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
from jeanplot.models import Transform, Size, BoxStyle, LayoutConstraints, Offset, AnchorPoint


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

    name: Optional[str] = None
    label: Optional[Text] = None
    line_thickness: float = 1
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row", align_items="center", justify_content="space-between", gap=5
        )
    )

    @model_validator(mode="after")
    def set_label_from_name(self):
        """create label text component from name if not explicitly provided"""
        if self.name and not self.label and not any(isinstance(c, Text) for c in self.children):
            self.label = Text(
                text=self.name,
                font_size=4,
                offset=Offset(parent_relative=(0, 1.1)),  # position above
                style_class=["tu-label"],
                is_overlay=True,
                color="#aaaaaa",
            )
            self.add_child(self.label)
        elif self.label and self.label not in self.children:
            self.add_child(self.label)
        return self

    def render(self, renderer, context, matrix: np.ndarray):
        # draw main TU line in the middle vertically
        svg_line_content = make_svg_line(self._dimensions.width, self.line_thickness, "#000000")
        svg_line = SVGElement(
            svg_content=svg_line_content,
            transform=Transform(translate=(0, (self._dimensions.height - self.line_thickness) / 2)),
        )
        svg_line._dimensions = Size(width=self._dimensions.width, height=self.line_thickness)

        # render the line using its *own* local matrix relative to the parent (TU) matrix
        svg_line_matrix = matrix @ svg_line.compute_local_matrix()
        svg_line.render(renderer, context, svg_line_matrix)

        # render children (parts and the label) using the standard container logic
        Container.render(self, renderer, context, matrix)


class Source(Container):
    source_type: Optional[Literal["plasmid", "cotx"]] = "cotx"

    marker: Optional[str] = None
    tag_label: Optional[str] = None

    tag_content: Optional[Container] = None

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="column", align_items="center", justify_content="center", gap=5
        )
    )

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if self.source_type and self.tag_content is None:
            svgpath = {"plasmid": PLASMID_LOGO, "cotx": AGGREGATION_LOGO}[self.source_type]
            elmts = [SVGElement(svg_content=svgpath)]
            if self.tag_label is not None:
                elmts.append(Text(text=self.tag_label))  # type: ignore
            self.tag_content = Container(
                style_class=["source_tag"],
                is_overlay=True,
                children=elmts,  # type: ignore
            )

        if self.tag_content:
            self.add_child(self.tag_content)


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
                    print(f"Found SVG content for {self.part_name} at {svg_path}")
                    self.svg_content = get_svg_data_from_string(svg_content)
                    return

            svg_path = f"{self.auto_resource_path}/parts/{self.part_type}.svg"
            svg_content = load_file_if_exists(svg_path)
            if svg_content:
                print(f"Found SVG content for {self.part_name} at {svg_path}")
                self.svg_content = get_svg_data_from_string(svg_content)
            else:
                print(f"SVG content not found for {self.part_name} of type {self.part_type}")

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
        AnchorPoint(offset=Offset(relative=(0.77, -0.1)), direction=(0, -1), min_segment=20),
        AnchorPoint(offset=Offset(relative=(0.77, 1.1)), direction=(0, 1), min_segment=20),
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


class FluoMarker(GeneticPart):
    part_type: str = "fluo_marker"
    main_color: str = "#BBBBBB"

    @model_validator(mode="before")
    def set_label_to_name(cls, values):
        """set label to part_name if no label is provided"""
        if not values.get("label") and values.get("part_name"):
            values["label"] = Text(
                text=values["part_name"],
                align="center",
                vertical_align="middle",
                font_size=5,
                offset=Offset(relative=(-0.15, 0)),
            )
        return values


class Promoter(GeneticPart):
    part_type: str = "Promoter"
    offset: Offset = Offset(relative=(0, -0.5), absolute=(0, 0.5))


class Terminator(GeneticPart):
    part_type: str = "Terminator"
    offset: Offset = Offset(relative=(0.5, -0.5), absolute=(-0.5, 0.5))
    style: BoxStyle = BoxStyle(margin=(0, 0, 0, -10))


class UorfGroup(GeneticPart):
    part_type: str = "uORF_group"
    style: BoxStyle = BoxStyle(margin=(0, 0, 0, -10))
    offset: Offset = Offset(relative=(0, 0.1))

    @model_validator(mode="before")
    def set_label_to_name(cls, values):
        """set label to part_name if no label is provided"""
        if not values.get("label") and values.get("part_name"):
            values["label"] = Text(
                text=values["part_name"].split("_")[0],
                align="center",
                vertical_align="middle",
                font_size=5,
                offset=Offset(relative=(0, 1.3)),
            )
        return values


class ERN5pRecog(GeneticPart):
    part_type: str = "ERN_recog_site_5p"
    offset: Offset = Offset(relative=(0, -0.4), absolute=(0, 0.5))
    style: BoxStyle = BoxStyle(margin=(0, 2, 0, -4))

    anchor_points: list[AnchorPoint] = [
        AnchorPoint(offset=Offset(relative=(0.5, -0.1)), direction=(0, 1), min_segment=30),
        AnchorPoint(offset=Offset(relative=(0.5, 1.1)), direction=(0, -1), min_segment=30),
    ]

    # @model_validator(mode="before")
    # def set_label_to_name(cls, values):
    # """set label to part_name if no label is provided"""
    # if not values.get("label") and values.get("part_name"):
    #     values["label"] = Text(
    #         text=values["part_name"].split("_")[0],
    #         align="center",
    #         vertical_align="middle",
    #         font_size=5,
    #         offset=Offset(relative=(0, 2)),
    #     )
    # return values
