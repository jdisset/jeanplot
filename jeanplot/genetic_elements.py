from typing import Optional
from pydantic import BaseModel, Field, model_validator, PrivateAttr, computed_field
from jeanplot.utils import load_file_if_exists
from pathlib import Path
import numpy as np

from jeanplot.svg import SVGElement, get_svg_data_from_string
from jeanplot.container import Container
from jeanplot.text import Text
from jeanplot.models import Transform, Size, VisualStyle, LayoutConstraints, Offset

# color definitions for markers
BASE_FLUO_COLORS = {
    "red": {"base": "#ef957d", "light": "#ffe5de", "dark": "#840137"},
    "green": {"base": "#6CCB83", "light": "#EFFFDD", "dark": "#0E633A"},
    "blue": {"base": "#6cafc3", "light": "#F3FAFD", "dark": "#006394"},
    "yellow": {"base": "#FAD26D", "light": "#FFF8B8", "dark": "#9B6600"},
    "ir": {"base": "#df9ae4", "light": "#ffe7f4", "dark": "#6c1772"},
    "maroon": {"base": "#D3A888", "light": "#F4DECD", "dark": "#734727"},
}

# mapping of marker names to color schemes
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

# display name aliases for markers
MARKER_ALIAS = {
    "NeonGreen": "mNeonGreen",
    "L0.G_mNeonGreen": "mNeonGreen",
    "L0.G_iRFPmystery": "iRFPmystery",
    "eBFP": "eBFP2",
}

# ern type colors
ERN_COLORS = {
    "Csy4": "#AAAAAA",
    "CasE": "#CCCCCC",
    "PgU": "#EEEEEE",
}

DEFAULT_RESOURCE_PATH = "pkg:jeanplot:resources"


class GeneticPart(SVGElement, Container):
    """base class for genetic parts, extends SVGElement with biological features.
    Is also a container - most of the time will only contain one Text element, the label."""

    part_type: str
    part_name: Optional[str] = None
    main_color: str = "#EEEEEE"
    secondary_color: str = "#EEEEEE"
    label: Optional[str] = None

    auto_resource_path: str = DEFAULT_RESOURCE_PATH

    @model_validator(mode="before")
    def find_svg_path(cls, values):
        part_type = values.get("part_type")
        part_name = values.get("part_name")

        if "svg_content" in values:
            return values

        # try to find the SVG file based on part_name or part_type
        if part_name:
            svg_path = f"{DEFAULT_RESOURCE_PATH}/parts/{part_type}.{part_name}.svg"
            svg_content = load_file_if_exists(svg_path)
            if svg_content:
                values["svg_content"] = get_svg_data_from_string(svg_content)
                return values

        if part_type:
            svg_path = f"{DEFAULT_RESOURCE_PATH}/parts/{part_type}.svg"
            svg_content = load_file_if_exists(svg_path)
            if svg_content:
                values["svg_content"] = get_svg_data_from_string(svg_content)
                return values

        raise ValueError(
            f"No SVG content found for part_type '{part_type}' or part_name '{part_name}'"
        )

    @model_validator(mode="after")
    def set_label(self):
        """set label text if provided"""
        if self.label:
            self.add_child(Text(text=self.label))
        return self

    def render(self, renderer, context, matrix: np.ndarray):
        """render genetic part and optional label"""
        # super(SVGElement, self).render(renderer, context, matrix)
        # super(Container, self).render(renderer, context, matrix)
        super().render(renderer, context, matrix)
