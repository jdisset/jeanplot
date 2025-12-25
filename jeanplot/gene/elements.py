# File: jeanplot/genetic_elements.py
# -*- coding: utf-8 -*-
"""Components representing biological genetic elements like TUs, Promoters, ERNs."""

from __future__ import annotations
from typing import Literal, ClassVar, TYPE_CHECKING
from pydantic import Field, model_validator, PrivateAttr

if TYPE_CHECKING:
    from jeanplot.gene.data import PartData

from pathlib import Path
from jeanplot.core.utils import load_file_if_exists
from jeanplot.core.svg import SVGElement, get_svg_data, make_svg_line, SVGContent
from jeanplot.core.container import Container
from jeanplot.core.style import jstyle
from jeanplot.core.component import AnchorComponent
from jeanplot.core.text import Text
from jeanplot.core.models import BoxStyle, LayoutConstraints, Offset
import logging

logger = logging.getLogger(__name__)


DEFAULT_RESOURCE_PATH = "pkg:jeanplot:resources"
AGGREGATION_LOGO_PATH = DEFAULT_RESOURCE_PATH + "/parts/aggregation.svg"
PLASMID_LOGO_PATH = DEFAULT_RESOURCE_PATH + "/parts/l2.svg"


def make_vertical_anchors(prefix: str = "", style_base: str = "ern") -> list[AnchorComponent]:
    return [
        AnchorComponent(
            id=f"{prefix}anchor_top",
            style_class=[f"{style_base}-anchor", f"{style_base}-anchor-top"],
            direction=(0, 1),
        ),
        AnchorComponent(
            id=f"{prefix}anchor_bottom",
            style_class=[f"{style_base}-anchor", f"{style_base}-anchor-bottom"],
            direction=(0, -1),
        ),
    ]


class AutoLabelMixin:
    """Mixin that auto-creates labels from part_name."""

    _auto_label: ClassVar[bool] = False
    _label_offset: ClassVar[Offset | None] = None
    _label_prefix: ClassVar[str] = ""
    _label_font_size: ClassVar[float | None] = None

    @model_validator(mode="before")
    @classmethod
    def _create_auto_label(cls, values):
        if not getattr(cls, "_auto_label", False):
            return values
        if values.get("label") or not values.get("part_name"):
            return values
        part_name = values["part_name"]
        part_id = values.get("id", f"{cls._label_prefix}{part_name}")
        label_text = (
            part_name.split("_")[0]
            if "_" in part_name and cls.__name__ == "UorfGroup"
            else part_name
        )
        label_kwargs = {
            "id": f"{part_id}_label",
            "text": label_text,
            "align": "center",
            "vertical_align": "middle",
            "is_overlay": True,
        }
        if getattr(cls, "_label_offset", None):
            label_kwargs["offset"] = cls._label_offset
        if getattr(cls, "_label_font_size", None):
            label_kwargs["font_size"] = cls._label_font_size
        values["label"] = Text(**label_kwargs)
        return values


class TranscriptionUnit(Container):
    """
    represents a transcription unit, containing genetic parts.
    renders a central line visually connecting the parts.
    """

    name: str | None = None
    label: Text | None = None
    line_thickness: float = 1.0
    line_color: str = "#333333"
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row", align_items="center", justify_content="start", gap=0
        )
    )
    style: BoxStyle = Field(default_factory=lambda: BoxStyle(padding=(5, 0, 5, 0)))

    _tu_line: SVGElement | None = PrivateAttr(default=None)

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
                font_size=3.5,
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
            content_w, content_h = self.style.content_box(self._dimensions)
            line_y = self.style.padding[0] + (content_h - self.line_thickness) / 2.0
            self._tu_line.offset = Offset(absolute=(min_x, line_y))


class Source(Container):
    """represents a source of genetic material (e.g., plasmid, cotransfection mix)."""

    source_type: Literal["plasmid", "cotx", "mix", "linear"] | None = "cotx"
    marker: str | None = None
    tag_label: str | None = None

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="column", align_items="stretch", justify_content="start", gap=2
        )
    )

    style: BoxStyle = Field(default_factory=BoxStyle)

    _tag_container: Container | None = PrivateAttr(default=None)

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
                    is_overlay=True,
                    children=tag_elements,
                    parent=self,
                )
                # important: apply styles to the proxy tag container here
                # so it gets its default layout etc., before it's potentially copied
                jstyle.apply(self._tag_container)
                # don't add to self.children here, SourceAnnotation will extract it
        return self


class GeneticPart(Container):
    """
    Base class for genetic parts using composition. It's a Container that holds
    an SVGElement for its shape, an optional Text label, and anchors.
    """

    part_type: str = "unknown_part"
    part_name: str | None = None
    label: Text | None = None
    auto_resource_path: str = DEFAULT_RESOURCE_PATH

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(justify_content="center", align_items="center")
    )

    style: BoxStyle = Field(default_factory=lambda: BoxStyle(padding=(0, 0, 0, 0)))

    _svg_shape: SVGElement | None = PrivateAttr(default=None)

    def model_post_init(self, *args, **kwargs):
        """
        Creates the internal SVGElement shape, adds it as a child,
        and adds the label (if any) and anchors as overlay children.
        """
        super().model_post_init(*args, **kwargs)

        svg_content_data = self._load_svg_content()
        svg_id = f"svg_{self.id}" if self.id else None
        self._svg_shape = SVGElement(
            id=svg_id,
            svg_content=svg_content_data,
            is_overlay=False,
            style=BoxStyle(padding=(0, 0, 0, 0), margin=(0, 0, 0, 0)),
        )
        self.add_child(self._svg_shape)

        if self.label:
            self.label.is_overlay = True
            if not any(c is self.label for c in self.children):
                self.add_child(self.label)

        for anchor in self.anchor_points:
            if anchor:
                anchor.is_overlay = True
                if not any(c is anchor for c in self.children):
                    self.add_child(anchor)

    def _load_svg_content(self) -> SVGContent | None:
        """Loads SVG data based on part_type and part_name."""
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
            return get_svg_data(svg_to_load)
        else:
            logger.warning(
                f"Could not find SVG resource for {self.part_type} (name: {self.part_name})"
            )
            return SVGContent(paths=())

    @property
    def svg_content(self) -> str | Path | bytes | SVGContent | None:
        return self._svg_shape.svg_content if self._svg_shape else None

    @svg_content.setter
    def svg_content(self, value: str | Path | bytes | SVGContent | None):
        if self._svg_shape:
            self._svg_shape.svg_content = value
            self._svg_shape._parse_and_validate_svg()
        else:
            logger.warning(
                f"Cannot set svg_content for {self.id}: internal _svg_shape not initialized."
            )

    @classmethod
    def from_data(cls, data: PartData) -> GeneticPart:
        """Create part from PartData."""
        role_map = {
            "promoter": Promoter,
            "terminator": Terminator,
            "regulator": ERN,
            "recognition_site": ERN5pRecog,
            "reporter": FluoMarker,
            "uorf": UorfGroup,
        }
        part_cls = role_map.get(data.role, cls)
        return part_cls(id=data.id, part_name=data.name)


class ERN(GeneticPart, AutoLabelMixin):
    part_type: str = "ERN"
    _auto_label: ClassVar[bool] = True
    _label_prefix: ClassVar[str] = "ern_"
    _label_font_size: ClassVar[float] = 5.0  # SVG height is 18
    anchor_points: list[AnchorComponent] = Field(
        default_factory=lambda: make_vertical_anchors("", "ern")
    )


class FluoMarker(GeneticPart, AutoLabelMixin):
    part_type: str = "fluo_marker"
    _auto_label: ClassVar[bool] = True
    _label_prefix: ClassVar[str] = "fluo_"
    _label_font_size: ClassVar[float] = 5.0  # SVG height is 19


class Promoter(GeneticPart):
    part_type: str = "Promoter"


class Terminator(GeneticPart):
    part_type: str = "Terminator"


class UorfGroup(GeneticPart, AutoLabelMixin):
    part_type: str = "uORF_group"
    _auto_label: ClassVar[bool] = True
    _label_prefix: ClassVar[str] = "uorf_"
    _label_font_size: ClassVar[float] = 5.0
    _label_offset: ClassVar[Offset] = Offset(relative=(0.5, 1.0), absolute=(0, 1))


class ERN5pRecog(GeneticPart):
    part_type: str = "ERN_recog_site_5p"
    anchor_points: list[AnchorComponent] = Field(
        default_factory=lambda: make_vertical_anchors("recog-", "ern-recog")
    )
