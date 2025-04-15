# File: jeanplot/genetic_elements.py
# -*- coding: utf-8 -*-
"""Components representing biological genetic elements like TUs, Promoters, ERNs."""

from typing import Optional, Union, Literal, List, Tuple
from pydantic import Field, model_validator, PrivateAttr
import numpy as np

from jeanplot.utils import load_file_if_exists
from jeanplot.svg import SVGElement, get_svg_data, make_svg_line, SVGContent
from jeanplot.container import Container
from jeanplot.style import jstyle
from jeanplot.component import Component, AnchorComponent
from jeanplot.text import Text
from jeanplot.models import Size, BoxStyle, LayoutConstraints, Offset
from jeanplot.network_utils import Interaction, PartInfo  # interaction type hint


DEFAULT_RESOURCE_PATH = "pkg:jeanplot:resources"
AGGREGATION_LOGO_PATH = DEFAULT_RESOURCE_PATH + "/parts/aggregation.svg"
PLASMID_LOGO_PATH = DEFAULT_RESOURCE_PATH + "/parts/l2.svg"


class TranscriptionUnit(Container):
    """
    represents a transcription unit, containing genetic parts.
    renders a central line visually connecting the parts.
    """

    name: Optional[str] = None
    label: Optional[Text] = None
    line_thickness: float = 1.0
    line_color: str = "#333333"
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row", align_items="center", justify_content="start", gap=0
        )
    )
    style: BoxStyle = Field(default_factory=lambda: BoxStyle(padding=(5, 0, 5, 0)))

    _tu_line: Optional[SVGElement] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def setup_label_and_line(self):
        # create label from name if needed
        if self.name and not self.label and not any(isinstance(c, Text) for c in self.children):
            self.label = Text(
                id=f"lbl_{self.id}",
                text=self.name,
                style_class=["tu-label"],
                is_overlay=True,
                color="#888888",
            )
            self.add_child(self.label)
        elif self.label and self.label not in self.children:
            if not any(c.id == self.label.id for c in self.children if c.id and self.label.id):
                self.add_child(self.label)

        # create the tu line element (initially zero width)
        if not self._tu_line:
            svg_line_content = make_svg_line(0, self.line_thickness, self.line_color)
            self._tu_line = SVGElement(
                id=f"line_{self.id}",
                svg_content=svg_line_content,
                is_overlay=True,
                parent=self,
                z_index=-1,  # draw line behind parts
                line_width_mode="data",
            )
            self.add_child(self._tu_line)
        return self

    def _layout_children(self, renderer=None):
        super()._layout_children(renderer)

        # update tu line width and position after children are laid out
        if self._tu_line and self.children:
            layout_children = self._layout_children_cache
            line_width = 0
            min_x = 0
            if layout_children:
                # filter out non-component Nones if any exist
                valid_children = [child for child in layout_children if child is not None]
                if valid_children:  # ensure there are children to calculate bounds from
                    child_origins_x = [
                        child._layout_origin_in_parent[0] for child in valid_children
                    ]
                    child_ends_x = [
                        child._layout_origin_in_parent[0] + child._dimensions.width
                        for child in valid_children
                    ]
                    if child_origins_x and child_ends_x:
                        min_x = min(child_origins_x)
                        max_x = max(child_ends_x)
                        line_width = max(0, max_x - min_x)

            svg_line_content = make_svg_line(line_width, self.line_thickness, self.line_color)
            self._tu_line.svg_content = svg_line_content
            self._tu_line._parse_and_validate_svg()  # re-parse updated svg

            # use calculated content box height for centering
            content_w, content_h = self.style.content_box(self._dimensions)  # pylint: disable=unused-variable
            line_y = self.style.padding[0] + (content_h - self.line_thickness) / 2.0
            self._tu_line.offset = Offset(absolute=(min_x, line_y))


class Source(Container):
    """represents a source of genetic material (e.g., plasmid, cotransfection mix)."""

    source_type: Optional[Literal["plasmid", "cotx"]] = "cotx"
    marker: Optional[str] = None
    tag_label: Optional[str] = None

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="column", align_items="stretch", justify_content="start", gap=2
        )
    )

    # Style is applied via theme using &source_base_style alias
    style: BoxStyle = Field(default_factory=BoxStyle)

    _tag_container: Optional[Container] = PrivateAttr(default=None)

    # note: this setup is primarily for the proxy used by SourceAnnotation
    @model_validator(mode="after")
    def setup_source_tag(self):
        # remove previous tag if exists
        if self._tag_container and self._tag_container in self.children:
            self.children.remove(self._tag_container)
            self._tag_container = None

        if self.source_type or self.tag_label:
            svg_path = None
            if self.source_type:
                svg_path = {"plasmid": PLASMID_LOGO_PATH, "cotx": AGGREGATION_LOGO_PATH}.get(
                    self.source_type
                )

            tag_elements = []
            if svg_path:
                tag_elements.append(SVGElement(svg_content=svg_path, id=f"{self.id}_tag_icon"))
            if self.tag_label is not None:
                label_lines = self.tag_label.split("\n")
                for i, line in enumerate(label_lines):
                    # use the correct text component id format if self.id is set
                    text_id = f"{self.id}_tag_label_{i}" if self.id else f"tag_label_{i}"
                    tag_elements.append(Text(text=line, id=text_id))

            if tag_elements:
                # use correct container id format if self.id is set
                tag_id = f"{self.id}_tag" if self.id else "source_tag_proxy"
                self._tag_container = Container(
                    id=tag_id,
                    style_class=["source_tag"],
                    is_overlay=True,  # tag itself is an overlay relative to its logical parent (Source/SourceAnnotation)
                    children=tag_elements,
                    parent=self,  # parent link established by add_child if needed
                )
                # important: apply styles to the proxy tag container here
                # so it gets its default layout etc., before it's potentially copied
                jstyle.apply(self._tag_container)
                # don't add to self.children here, SourceAnnotation will extract it
        return self


class GeneticPart(SVGElement, Container):
    """
    base class for genetic parts, combining svg rendering with container capabilities
    for labels and anchors. uses color_remap for theming svg colors.
    """

    part_type: str = "unknown_part"
    part_name: Optional[str] = None
    label: Optional[Text] = None

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(justify_content="center", align_items="start")
    )
    auto_resource_path: str = DEFAULT_RESOURCE_PATH

    def __init__(self, part_name: Optional[str] = None, **kwargs):
        # if 'children' not explicitly provided, default to empty list
        # prevents issues if label/anchors are added later via add_child
        if "children" not in kwargs:
            kwargs["children"] = []
        super().__init__(part_name=part_name, **kwargs)

    def model_post_init(self, *args, **kwargs):
        """loads svg and ensures label/anchors are children."""
        # call base Component init first (handles parent, style_class etc.)
        Component.model_post_init(self, *args, **kwargs)
        # call SVGElement validator to handle svg parsing and initial dimensions
        SVGElement._parse_and_validate_svg(self)
        # call Container validator (currently empty, but good practice)
        # Container.model_post_init(self, *args, **kwargs) # skip this?

        # if svg content is missing after init, try loading automatically
        if self._parsed_svg_content is None or not self._parsed_svg_content.paths:
            svg_to_load = None
            if self.part_name:
                svg_path_specific = (
                    f"{self.auto_resource_path}/parts/{self.part_type}.{self.part_name}.svg"
                )
                svg_to_load = load_file_if_exists(svg_path_specific)
            if not svg_to_load:
                svg_path_generic = f"{self.auto_resource_path}/parts/{self.part_type}.svg"
                svg_to_load = load_file_if_exists(svg_path_generic)

            if svg_to_load:
                self.svg_content = get_svg_data(svg_to_load)
                # re-run parse/validate after setting svg_content
                SVGElement._parse_and_validate_svg(self)
            elif self._parsed_svg_content is None:
                # if still nothing, set to empty content
                self._parsed_svg_content = SVGContent(paths=())
                # if dimensions haven't been set, use zero
                if not hasattr(self, "_dimensions") or self._dimensions is None:
                    self._dimensions = Size()

        # add label/anchors if they exist and aren't already children
        if self.label and not any(
            isinstance(c, Text) and c.id == self.label.id
            for c in self.children
            if c and self.label.id
        ):
            self.add_child(self.label)  # add_child sets parent link
        for anchor in self.anchor_points:
            if anchor and anchor not in self.children:
                self.add_child(anchor)  # add_child sets parent link

    # override render to handle both svg and container children (label/anchors)
    def render(self, renderer, context, matrix: np.ndarray):
        if not self.show:
            return
        # 1. render the svg part
        SVGElement.render(self, renderer, context, matrix)

        # 2. render container children (label, anchors) respecting their transforms/offsets
        # use the combined list of children prepared during layout
        all_renderable_children = [
            c
            for c in (
                getattr(self, "_layout_children_cache", [])
                + getattr(self, "_overlay_children_cache", [])
            )
            if c and c.show
        ]
        # sort children by z-index before rendering
        all_renderable_children.sort(key=lambda c: getattr(c, "z_index", 0))

        for child in all_renderable_children:
            # child's world matrix is calculated based on *this* component's matrix
            # compute_world_matrix handles offsets/attachments internally
            child_matrix = child.compute_world_matrix(parent_world_matrix=matrix)
            child.render(renderer, context, child_matrix)

        # render debug box last if enabled
        if self.debug:
            # use the potentially overridden render_debug from Component/Container
            Component.render(self, renderer, context, matrix)


# --- specific genetic part subclasses ---


class ERN(GeneticPart):
    part_type: str = "ERN"

    anchor_points: List[AnchorComponent] = Field(
        default_factory=lambda: [
            AnchorComponent(
                id="anchor_top",
                style_class=["ern-anchor", "ern-anchor-top"],
                direction=(0, 1),
            ),
            AnchorComponent(
                id="anchor_bottom",
                style_class=["ern-anchor", "ern-anchor-bottom"],
                direction=(0, -1),
            ),
        ]
    )

    @model_validator(mode="before")
    @classmethod
    def set_label_from_name(cls, values):
        if not values.get("label") and values.get("part_name"):
            part_id = values.get("id", f"ern_{values.get('part_name', 'unnamed')}")
            values["label"] = Text(
                id=f"{part_id}_label",
                text=values["part_name"],
                align="center",
                vertical_align="middle",
                is_overlay=True,
            )
        return values


class FluoMarker(GeneticPart):
    part_type: str = "fluo_marker"

    @model_validator(mode="before")
    @classmethod
    def set_label_from_name(cls, values):
        if not values.get("label") and values.get("part_name"):
            part_id = values.get("id", f"fluo_{values.get('part_name', 'unnamed')}")
            values["label"] = Text(
                id=f"{part_id}_label",
                text=values["part_name"],
                align="center",
                vertical_align="middle",
                is_overlay=True,
            )
        return values


class Promoter(GeneticPart):
    part_type: str = "Promoter"


class Terminator(GeneticPart):
    part_type: str = "Terminator"


class UorfGroup(GeneticPart):
    part_type: str = "uORF_group"

    @model_validator(mode="before")
    @classmethod
    def set_label_from_name(cls, values):
        if not values.get("label") and values.get("part_name"):
            part_name = values["part_name"]
            label_text = part_name.split("_")[0] if "_" in part_name else part_name
            part_id = values.get("id", f"uorf_{part_name}")
            values["label"] = Text(
                id=f"{part_id}_label",
                text=label_text,
                align="center",
                vertical_align="middle",
                font_size=5,
                offset=Offset(relative=(0.5, 1.0), absolute=(0, 1)),
                is_overlay=True,
            )
        return values


class ERN5pRecog(GeneticPart):
    part_type: str = "ERN_recog_site_5p"

    anchor_points: List[AnchorComponent] = Field(
        default_factory=lambda: [
            AnchorComponent(
                id="recog-anchor_top",
                style_class=["ern-recog-anchor", "ern-recog-anchor-top"],
                direction=(0, 1),
            ),
            AnchorComponent(
                id="recog-anchor_bottom",
                style_class=["ern-recog-anchor", "ern-recog-anchor-bottom"],
                direction=(0, -1),
            ),
        ]
    )
