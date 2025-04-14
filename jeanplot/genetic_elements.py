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
                line_width_mode="point",  # use point for simple line
                z_index=-1,  # draw line behind parts
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
                child_origins_x = [child._layout_origin_in_parent[0] for child in layout_children]
                child_ends_x = [
                    child._layout_origin_in_parent[0] + child._dimensions.width
                    for child in layout_children
                ]
                if child_origins_x and child_ends_x:
                    min_x = min(child_origins_x)
                    max_x = max(child_ends_x)
                    line_width = max(0, max_x - min_x)

            svg_line_content = make_svg_line(line_width, self.line_thickness, self.line_color)
            self._tu_line.svg_content = svg_line_content
            self._tu_line._parse_and_validate_svg()

            content_h = self.style.content_box(self._dimensions)[1]
            line_y = (
                self.style.padding[0] + (content_h - self.line_thickness) / 2.0
            )  # account for top padding
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

    _tag_container: Optional[Container] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def setup_source_tag(self):
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
                    tag_elements.append(Text(text=line, id=f"{self.id}_tag_label_{i}"))

            if tag_elements:
                self._tag_container = Container(
                    id=f"{self.id}_tag",
                    style_class=["source_tag"],
                    is_overlay=True,
                    children=tag_elements,
                    parent=self,
                )
                jstyle.apply(self._tag_container)
                self.add_child(self._tag_container)
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
        if "children" not in kwargs:
            kwargs["children"] = []
        super().__init__(part_name=part_name, **kwargs)

    def model_post_init(self, *args, **kwargs):
        """loads svg and ensures label/anchors are children."""
        super().model_post_init(*args, **kwargs)  # handle base class init

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
                self._parse_and_validate_svg()
            elif self._parsed_svg_content is None:
                self._parsed_svg_content = SVGContent(paths=())
                self._dimensions = Size()

        if self.label and not any(isinstance(c, Text) for c in self.children):
            self.add_child(self.label)
            if self.label:
                self.label.parent = self
        for anchor in self.anchor_points:
            if anchor not in self.children:
                self.add_child(anchor)
                if anchor:
                    anchor.parent = self

    def render(self, renderer, context, matrix: np.ndarray):
        if not self.show:
            return
        SVGElement.render(self, renderer, context, matrix)  # call svgelement render

        all_children = self._layout_children_cache + self._overlay_children_cache
        visible_children = [child for child in all_children if child and child.show]
        visible_children.sort(key=lambda c: getattr(c, "z_index", 0))

        if visible_children:
            for child in visible_children:
                child_matrix = child.compute_world_matrix(parent_world_matrix=matrix)
                child.render(renderer, context, child_matrix)


# --- specific genetic part subclasses ---


class ERN(GeneticPart):
    part_type: str = "ERN"
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
                id="anchor_top",
                offset=Offset(relative=(0.5, -0.1)),
                direction=(0, -1),
                min_segment=30,
            ),
            AnchorComponent(
                id="anchor_bottom",
                offset=Offset(relative=(0.5, 1.1)),
                direction=(0, 1),
                min_segment=30,
            ),
        ]
    )
