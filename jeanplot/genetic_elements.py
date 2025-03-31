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


BASE_FLUO_COLORS = {
    "red": {"base": "#ef957d", "light": "#ffe5de", "dark": "#840137"},
    "green": {"base": "#6CCB83", "light": "#EFFFDD", "dark": "#0E633A"},
    "blue": {"base": "#6cafc3", "light": "#F3FAFD", "dark": "#006394"},
    "yellow": {"base": "#FAD26D", "light": "#FFF8B8", "dark": "#9B6600"},
    "ir": {"base": "#df9ae4", "light": "#ffe7f4", "dark": "#6c1772"},
    "maroon": {"base": "#D3A888", "light": "#F4DECD", "dark": "#734727"},
}

MARKER_COLORS = {
    "eYFPG5A": BASE_FLUO_COLORS["yellow"],
    "eYFP": BASE_FLUO_COLORS["yellow"],
    "NeonGreen": BASE_FLUO_COLORS["green"],
    "mNeonGreen": BASE_FLUO_COLORS["green"],
    "L0.G_mNeonGreen": BASE_FLUO_COLORS["green"],
    "eBFP": BASE_FLUO_COLORS["blue"],
    "eBFP2": BASE_FLUO_COLORS["blue"],
    "tagBFP": BASE_FLUO_COLORS["blue"],
    "mKate": BASE_FLUO_COLORS["red"],
    "mKO2": BASE_FLUO_COLORS["red"],
    "tdTomato": BASE_FLUO_COLORS["red"],
    "iRFP720": BASE_FLUO_COLORS["ir"],
    "1xiRFP720": BASE_FLUO_COLORS["ir"],
    "L0.G_iRFPmystery": BASE_FLUO_COLORS["ir"],
    "iRFP": BASE_FLUO_COLORS["ir"],
    "mMaroon": BASE_FLUO_COLORS["maroon"],
    "mMaroon1": BASE_FLUO_COLORS["maroon"],
}
MARKER_COLORS = {k.upper(): v for k, v in MARKER_COLORS.items()}

MARKER_ALIAS = {
    "NeonGreen": "mNeonGreen",
    "L0.G_mNeonGreen": "mNeonGreen",
    "L0.G_iRFPmystery": "iRFPmystery",
    "eBFP": "eBFP2",
}
MARKER_ALIAS = {k.upper(): v for k, v in MARKER_ALIAS.items()}

ERN_COLORS = {
    "Csy4": "#AAAAAA",
    "CasE": "#CCCCCC",
    "PgU": "#EEEEEE",
}
ERN_COLORS = {k.upper(): v for k, v in ERN_COLORS.items()}

BORDER_COLOR = "#222222"
TEXT_COLOR = BORDER_COLOR

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
    marker: Optional[str] = None
    ratios: list[float] = []


class Plasmid(BaseModel):
    marker: Optional[str] = None


class Source(Container):
    multi_type: Optional[CoTX | Plasmid] = None
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="column", align_items="center", justify_content="center", gap=5
        )
    )

    label: Optional[Container] = None

    @model_validator(mode="before")
    def style_source(cls, values):
        # if no style is provided, set auto style
        mtype = values.get("multi_type")
        if not mtype:
            return values
        if not values.get("style"):
            if mtype.marker:
                values["style"] = VisualStyle(
                    border_color=MARKER_COLORS.get(mtype.marker.upper(), {}).get(
                        "base", BORDER_COLOR
                    ),
                    border_width=0.5,
                    corner_radius=5,
                    border_style="dashed",
                    padding=(10, 10, 10, 10),
                )

        if not values.get("label"):
            if mtype.marker:
                bgcol = MARKER_COLORS.get(mtype.marker.upper(), {}).get("light", "#DDDDDD")
                maincol = MARKER_COLORS.get(mtype.marker.upper(), {}).get("dark", BORDER_COLOR)
                txt = MARKER_ALIAS.get(mtype.marker.upper(), mtype.marker)
                if isinstance(mtype, CoTX):
                    ratios_txt = ":".join([f"{r:.2g}" for r in mtype.ratios])
                    txt += f"  {ratios_txt}"
                svgpath = AGGREGATION_LOGO if isinstance(mtype, CoTX) else PLASMID_LOGO
                svgtransform = (
                    Transform(scale=(0.7, 0.7))
                    if isinstance(mtype, Plasmid)
                    else Transform(scale=(0.9, 0.9))
                )

                values["label"] = Container(
                    style=VisualStyle(
                        background_color=bgcol,
                        border_color=maincol,
                        border_width=0.5,
                        corner_radius=50,
                        padding=(1.25, 5, 1.25, 2),
                    ),
                    is_overlay=True,
                    offset=Offset(
                        relative=(0, -0.5),
                        parent_relative=(0.1, 1),
                    ),
                    layout=LayoutConstraints(
                        direction="row",
                        align_items="center",
                        justify_content="start",
                        gap=3,
                    ),
                    children=[
                        SVGElement(
                            svg_content=svgpath,
                            transform=svgtransform,
                            main_color=maincol,
                            offset=Offset(absolute=(0, 0.25)),
                        ),
                        Text(
                            text=txt,
                            color=maincol,
                            font_size=5,
                        ),
                    ],
                )

                # Make sure children exists before appending
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

    anchor_points: list[AnchorPoint] = [
        AnchorPoint(offset=Offset(relative=(0.77, -0.1)), direction=(0, -1)),
        AnchorPoint(offset=Offset(relative=(0.77, 1.1)), direction=(0, 1)),
    ]

    @model_validator(mode="before")
    def preset_colors(cls, values):
        """set main and secondary colors based on part_type"""
        pname = values.get("part_name").upper()
        if pname in ERN_COLORS:
            values["main_color"] = ERN_COLORS[pname]
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

    @model_validator(mode="before")
    def preset_colors(cls, values):
        """set main and secondary colors based on part_type"""
        pname = values.get("part_name").upper().split("_")[0]
        if pname in ERN_COLORS:
            values["main_color"] = ERN_COLORS[pname]
            values["secondary_color"] = BORDER_COLOR
        return values
