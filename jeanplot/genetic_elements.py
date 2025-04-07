# jeanplot/genetic_elements.py
from typing import Optional, Union, Literal, List  # Added List
from pydantic import BaseModel, Field, model_validator, PrivateAttr, computed_field
from jeanplot.utils import load_file_if_exists
from pathlib import Path
from functools import partial
import numpy as np

from jeanplot.svg import SVGElement, get_svg_data_from_string, make_svg_line, SVGContent
from jeanplot.container import Container

# Updated import: Import AnchorComponent
from jeanplot.component import Component, AnchorComponent
from jeanplot.text import Text
from jeanplot.models import (
    Transform,
    Size,
    BoxStyle,
    LayoutConstraints,
    Offset,
)  # Removed AnchorPoint
from jeanplot.network_utils import Interaction  # Added Interaction for part_info type hint


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
            # Ensure label is added even if manually provided
            if not any(c.id == self.label.id for c in self.children if c.id and self.label.id):
                self.add_child(self.label)
        return self

    def render(self, renderer, context, matrix: np.ndarray):
        if not self.show:
            return

        # draw main TU line in the middle vertically
        svg_line_content = make_svg_line(self._dimensions.width, self.line_thickness, "#000000")
        svg_line = SVGElement(
            svg_content=svg_line_content,
            offset=Offset(absolute=(0, (self._dimensions.height - self.line_thickness) / 2)),
            is_overlay=True,  # Render the line as an overlay within the TU container
            parent=self,  # Explicitly set parent for matrix calculation
        )
        # Measure the line element if it hasn't been
        if not hasattr(svg_line, "_dimensions"):
            svg_line.measure_and_layout(renderer)

        # Render the line using its *own* world matrix relative to the parent (TU) matrix
        svg_line_matrix = svg_line.compute_world_matrix(parent_matrix=matrix)
        svg_line.render(renderer, context, svg_line_matrix)

        # Render children (parts and the label) using the standard container logic
        Container.render(self, renderer, context, matrix)


class Source(Container):
    source_type: Optional[Literal["plasmid", "cotx"]] = "cotx"

    marker: Optional[str] = None
    tag_label: Optional[str] = None

    # tag_content is now managed internally by the validator
    # tag_content: Optional[Container] = None

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="column", align_items="center", justify_content="center", gap=5
        )
    )

    @model_validator(mode="after")
    def setup_source_tag(self):
        # Find existing tag or create a new one
        existing_tag = next((c for c in self.children if "source_tag" in c.style_class), None)

        if self.source_type and existing_tag is None:
            svgpath = {"plasmid": PLASMID_LOGO, "cotx": AGGREGATION_LOGO}.get(self.source_type)
            elmts = []
            if svgpath:
                elmts.append(SVGElement(svg_content=svgpath, id=f"{self.id}_tag_icon"))
            if self.tag_label is not None:
                elmts.append(Text(text=self.tag_label, id=f"{self.id}_tag_label"))

            tag_content = Container(
                id=f"{self.id}_tag",
                style_class=["source_tag"],
                is_overlay=True,
                children=elmts,
            )
            # Apply style rules to the tag here so offsets etc. are loaded
            jstyle.apply(tag_content)
            self.add_child(tag_content)
        elif existing_tag and not self.source_type:
            # Remove tag if source_type is cleared
            self.children = [c for c in self.children if c != existing_tag]

        return self


# GeneticPart remains largely the same, but anchor points change type
class GeneticPart(SVGElement):
    """base class for genetic parts, extends SVGElement with biological features."""

    part_type: str
    part_name: Optional[str] = None
    label: Optional[Text] = None
    # anchor_points defined in Component base class (List[Component])

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(justify_content="center", align_items="start")
    )
    auto_resource_path: str = DEFAULT_RESOURCE_PATH

    # init can take part name as positional argument
    def __init__(self, part_name: Optional[str] = None, **kwargs):
        # Ensure 'children' is initialized if needed, although SVGElement doesn't usually have them
        if "children" not in kwargs:
            kwargs["children"] = []
        BaseModel.__init__(self, part_name=part_name, **kwargs)

    def model_post_init(self, *args, **kwargs):
        # load SVG content from file if not already provided
        if self.svg_content is None:
            if self.part_name:
                svg_path = f"{self.auto_resource_path}/parts/{self.part_type}.{self.part_name}.svg"
                svg_content = load_file_if_exists(svg_path)
                if svg_content:
                    # print(f"found svg content for {self.part_name} at {svg_path}") # debug
                    self.svg_content = get_svg_data_from_string(svg_content)

            if not isinstance(self.svg_content, SVGContent):  # Check if loaded above or still None
                svg_path = f"{self.auto_resource_path}/parts/{self.part_type}.svg"
                svg_content = load_file_if_exists(svg_path)
                if svg_content:
                    # print(f"found svg content for {self.part_name} at {svg_path}") # debug
                    self.svg_content = get_svg_data_from_string(svg_content)
                else:
                    # print(f"svg content not found for {self.part_name} of type {self.part_type}") # debug
                    self.svg_content = SVGContent()  # Ensure it's valid SVGContent

            # Set dimensions based on loaded SVG content
            self._dimensions = Size(width=self.svg_content.width, height=self.svg_content.height)

    @model_validator(mode="after")
    def set_label_and_anchors(self):
        """set label text if provided and ensure anchors are children"""
        if self.label and not any(isinstance(c, Text) for c in self.children):
            self.add_child(self.label)

        # Ensure anchor points (which are now components) are also in children
        for anchor in self.anchor_points:
            if anchor not in self.children:
                self.children.append(anchor)  # Add anchor to children
                anchor.parent = self  # Ensure parent link

        return self

    def render(self, renderer, context, matrix: np.ndarray):
        """render genetic part and optional label"""
        if not self.show:
            return
        # Render the SVG part itself
        SVGElement.render(self, renderer, context, matrix)
        # Render children (label, anchors if shown) using Container logic
        # This assumes GeneticPart might need to render children like label/visible anchors
        for child in self.children:
            child_matrix = child.compute_world_matrix(parent_matrix=matrix)
            child.render(renderer, context, child_matrix)


# --- Update specific genetic parts with AnchorComponent ---


class ERN(GeneticPart):
    part_type: str = "ERN"
    main_color: str = "#AAAAAA"
    secondary_color: str = "#111111"

    # Define anchors as AnchorComponent instances
    anchor_points: List[AnchorComponent] = Field(
        default_factory=lambda: [
            AnchorComponent(
                id="anchor_top",
                offset=Offset(relative=(0.77, -0.1)),
                direction=(0, -1),
                min_segment=20,
            ),
            AnchorComponent(
                id="anchor_bottom",
                offset=Offset(relative=(0.77, 1.1)),
                direction=(0, 1),
                min_segment=20,
            ),
        ]
    )

    @model_validator(mode="before")
    def set_label_to_name(cls, values):
        """set label to part_name if no label is provided"""
        if not values.get("label") and values.get("part_name"):
            values["label"] = Text(
                id=f"{values.get('id', 'ern')}_label",  # Use part ID for label ID
                text=values["part_name"],
                align="center",
                vertical_align="middle",
                font_size=9,
                offset=Offset(relative=(-0.35, 0)),  # Label is positioned via offset
                is_overlay=True,  # Render label as overlay
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
                id=f"{values.get('id', 'fluo')}_label",
                text=values["part_name"],
                align="center",
                vertical_align="middle",
                font_size=5,
                offset=Offset(relative=(-0.15, 0)),
                is_overlay=True,
            )
        return values


class Promoter(GeneticPart):
    part_type: str = "Promoter"
    # Offset is now the standard component offset (for layout)
    offset: Offset = Offset(relative=(0, -0.5), absolute=(0, 0.5))


class Terminator(GeneticPart):
    part_type: str = "Terminator"
    offset: Offset = Offset(relative=(0.5, -0.5), absolute=(-0.5, 0.5))
    style: BoxStyle = Field(default_factory=lambda: BoxStyle(margin=(0, 0, 0, -10)))


class UorfGroup(GeneticPart):
    part_type: str = "uORF_group"
    style: BoxStyle = Field(default_factory=lambda: BoxStyle(margin=(0, 0, 0, -10)))
    offset: Offset = Offset(relative=(0, 0.1))

    @model_validator(mode="before")
    def set_label_to_name(cls, values):
        """set label to part_name if no label is provided"""
        if not values.get("label") and values.get("part_name"):
            part_name = values["part_name"]
            label_text = part_name.split("_")[0] if "_" in part_name else part_name
            values["label"] = Text(
                id=f"{values.get('id', 'uorf')}_label",
                text=label_text,
                align="center",
                vertical_align="middle",
                font_size=5,
                offset=Offset(relative=(0, 1.3)),  # Position label above
                is_overlay=True,
            )
        return values


class ERN5pRecog(GeneticPart):
    part_type: str = "ERN_recog_site_5p"
    offset: Offset = Offset(relative=(0, -0.4), absolute=(0, 0.5))
    style: BoxStyle = Field(default_factory=lambda: BoxStyle(margin=(0, 2, 0, -4)))

    # Define anchors as AnchorComponent instances
    anchor_points: List[AnchorComponent] = Field(
        default_factory=lambda: [
            AnchorComponent(
                id="anchor_top",
                offset=Offset(relative=(0.5, -0.1)),
                direction=(0, -1),
                min_segment=30,
            ),  # Changed direction
            AnchorComponent(
                id="anchor_bottom",
                offset=Offset(relative=(0.5, 1.1)),
                direction=(0, 1),
                min_segment=30,
            ),  # Changed direction
        ]
    )

    # Removed label validator, assuming labels aren't typically needed here
