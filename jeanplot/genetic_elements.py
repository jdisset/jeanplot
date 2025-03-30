from typing import Optional
from pydantic import BaseModel, Field, model_validator, PrivateAttr, computed_field
from jeanplot.utils import load_file_if_exists
from pathlib import Path
from functools import partial
import numpy as np

from jeanplot.svg import SVGElement, get_svg_data_from_string, make_svg_line, SVGContent
from jeanplot.container import Container
from jeanplot.component import Component
from jeanplot.text import Text
from jeanplot.models import Transform, Size, VisualStyle, LayoutConstraints, Offset

BASE_FLUO_COLORS = {
    "red": {"base": "#ef957d", "light": "#ffe5de", "dark": "#840137"},
    "green": {"base": "#6CCB83", "light": "#EFFFDD", "dark": "#0E633A"},
    "blue": {"base": "#6cafc3", "light": "#F3FAFD", "dark": "#006394"},
    "yellow": {"base": "#FAD26D", "light": "#FFF8B8", "dark": "#9B6600"},
    "ir": {"base": "#df9ae4", "light": "#ffe7f4", "dark": "#6c1772"},
    "maroon": {"base": "#D3A888", "light": "#F4DECD", "dark": "#734727"},
}

MARKER_COLORS = {
    "NeonGreen": BASE_FLUO_COLORS["green"],
    "eYFP": BASE_FLUO_COLORS["yellow"],
    "eBFP": BASE_FLUO_COLORS["blue"],
    "mKate": BASE_FLUO_COLORS["red"],
    "iRFP720": BASE_FLUO_COLORS["ir"],
    "1xiRFP720": BASE_FLUO_COLORS["ir"],
    "L0.G_mNeonGreen": BASE_FLUO_COLORS["green"],
    "L0.G_iRFPmystery": BASE_FLUO_COLORS["ir"],
    "eYFPG5A": BASE_FLUO_COLORS["yellow"],
    "tagBFP": BASE_FLUO_COLORS["blue"],
    "tdTomato": BASE_FLUO_COLORS["red"],
    "mKO2": BASE_FLUO_COLORS["red"],
    "mMaroon1": BASE_FLUO_COLORS["maroon"],
}

MARKER_ALIAS = {
    "NeonGreen": "mNeonGreen",
    "L0.G_mNeonGreen": "mNeonGreen",
    "L0.G_iRFPmystery": "iRFPmystery",
    "eBFP": "eBFP2",
}

ERN_COLORS = {
    "Csy4": "#AAAAAA",
    "CasE": "#CCCCCC",
    "PgU": "#EEEEEE",
}

BORDER_COLOR = "#222222"
TEXT_COLOR = BORDER_COLOR

DEFAULT_RESOURCE_PATH = "pkg:jeanplot:resources"


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


class GeneticPart(SVGElement, Container):
    """base class for genetic parts, extends SVGElement with biological features.
    Is also a container - most of the time will only contain one Text element, the label."""

    part_type: str
    part_name: Optional[str] = None
    label: Optional[Text] = None
    main_color: str = BORDER_COLOR
    secondary_color: str = BORDER_COLOR

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

    @model_validator(mode="before")
    def preset_colors(cls, values):
        """set main and secondary colors based on part_type"""
        if values.get("part_type") in ERN_COLORS:
            values["main_color"] = ERN_COLORS[values["part_type"]]
            values["secondary_color"] = BORDER_COLOR
        return values

    @model_validator(mode="before")
    def set_label_to_name(cls, values):
        """set label to part_name if no label is provided"""
        if not values.get("label") and values.get("part_name"):
            values["label"] = Text(
                text=values["part_name"],
                align="center",
                vertical_align="middle",
                color=TEXT_COLOR,
                font_size=9,
                offset=Offset(relative=(0.35, -0.1)),
            )
        return values


class Promoter(GeneticPart):
    part_type: str = "Promoter"
    offset: Offset = Offset(relative=(0, -0.5))


class Terminator(GeneticPart):
    part_type: str = "Terminator"
    offset: Offset = Offset(relative=(-0.5, -0.5))
    style: VisualStyle = VisualStyle(margin=(0, 0, 0, -10))


class UorfGroup(GeneticPart):
    part_type: str = "uORF_group"
    style: VisualStyle = VisualStyle(margin=(0, 0, 0, -10))
    offset: Offset = Offset(relative=(0, -0.1))
