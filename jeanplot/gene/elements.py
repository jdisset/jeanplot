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
from jeanplot.core.text import Text, TextSegment
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
    _auto_label: ClassVar[bool] = False
    _label_prefix: ClassVar[str] = ""
    _label_aliases: ClassVar[dict[str, str]] = {}

    @classmethod
    def register_label_aliases(cls, mapping: dict[str, str]) -> None:
        """Add part_name → display-label overrides for this class."""
        # per-subclass copy: don't mutate an inherited shared dict
        if "_label_aliases" not in cls.__dict__:
            cls._label_aliases = dict(cls._label_aliases)
        cls._label_aliases.update(mapping)

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
        label_text = cls._label_aliases.get(label_text, label_text)
        values["label"] = Text(id=f"{part_id}_label", text=label_text, is_overlay=True)
        return values


class TranscriptionUnit(Container):
    """Transcription unit with a central line connecting parts."""

    name: str | None = None
    label: Text | None = None
    ratio_normalized: float | None = None  # smallest ratio normalized to 1
    disabled: bool = False
    line_thickness: float = 1.0
    line_color: str = "#333333"
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row", align_items="center", justify_content="start", gap=0
        )
    )
    style: BoxStyle = Field(default_factory=lambda: BoxStyle(padding=(5, 0, 5, 0)))
    style_class: list[str] = Field(default_factory=lambda: ["TranscriptionUnit"])

    _tu_line: SVGElement | None = PrivateAttr(default=None)
    _ratio_label: Text | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def setup_label_and_line(self):
        if self.disabled and "disabled" not in self.style_class:
            self.style_class.append("disabled")

        if self.name and not self.label and not any(isinstance(c, Text) for c in self.children):
            self.label = Text(
                id=f"lbl_{self.id}",
                text=self.name,
                style_class=["tu-label"],
                is_overlay=True,
            )
            self.add_child(self.label)
        elif self.label and self.label not in self.children:
            if not any(c.id == self.label.id for c in self.children if c.id and self.label.id):
                self.add_child(self.label)

        if self.ratio_normalized is not None and self._ratio_label is None:
            ratio_str = _format_ratio_multiplier(self.ratio_normalized)
            if self.label and self.label.segments is None:
                self.label.segments = [
                    TextSegment(text=self.label.text),
                    TextSegment(text=f" {ratio_str}", font_weight="bold"),
                ]
                self._ratio_label = self.label
            elif not self.label:
                self._ratio_label = Text(
                    id=f"ratio_{self.id}",
                    text=ratio_str,
                    style_class=["tu-ratio-label"],
                    is_overlay=True,
                )
                self.add_child(self._ratio_label)

        if not self._tu_line:
            svg_line_content = make_svg_line(0, self.line_thickness, self.line_color)
            self._tu_line = SVGElement(
                id=f"line_{self.id}",
                svg_content=svg_line_content,
                is_overlay=True,
                parent=self,
                z_index=-1,  # behind parts
                line_width_mode="data",
            )
            self.add_child(self._tu_line)
        return self

    def _layout_children(self, renderer=None):
        super()._layout_children(renderer)

        if self._tu_line and self.children:
            layout_children = self._layout_children_cache
            line_width = 0
            min_x = 0
            if layout_children:
                valid_children = [child for child in layout_children if child is not None]
                if valid_children:
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
            self._tu_line._parse_and_validate_svg()

            content_w, content_h = self.style.content_box(self._dimensions)
            line_y = self.style.padding[0] + (content_h - self.line_thickness) / 2.0
            self._tu_line.offset = Offset(absolute=(min_x, line_y))


def _format_ratio_multiplier(val: float) -> str:
    num = f"{int(round(val))}" if abs(val - round(val)) < 0.01 else f"{val:.1f}"
    return f"(\u00d7{num})"


class Source(Container):
    """Source of genetic material (plasmid, cotransfection mix, etc.)."""

    source_type: Literal["plasmid", "cotx", "mix", "linear"] | None = "cotx"
    marker: str | None = None
    marker_ratio: float | None = None  # marker source's normalized ratio
    ratios: list[float] | None = None  # per-TU ratios (normalized, smallest=1)
    tag_label: str | None = None

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="column", align_items="stretch", justify_content="start", gap=2
        )
    )

    style: BoxStyle = Field(default_factory=BoxStyle)

    _tag_container: Container | None = PrivateAttr(default=None)

    # primarily for the proxy used by SourceAnnotation
    @model_validator(mode="after")
    def setup_source_tag(self):
        if self._tag_container and self._tag_container in self.children:
            self.children.remove(self._tag_container)
            self._tag_container = None

        if self.source_type or self.tag_label or self.ratios:
            svg_path = None
            if self.source_type:
                svg_path = {
                    "plasmid": PLASMID_LOGO_PATH,
                    "cotx": AGGREGATION_LOGO_PATH,
                    "mix": AGGREGATION_LOGO_PATH,
                }.get(self.source_type)

            # "marker (×ratio)" with bold ratio
            effective_label = self.tag_label
            label_segments: list[TextSegment] | None = None
            if effective_label is None and self.marker:
                if self.marker_ratio is not None:
                    ratio_str = _format_ratio_multiplier(self.marker_ratio)
                    effective_label = f"{self.marker} {ratio_str}"
                    label_segments = [
                        TextSegment(text=self.marker),
                        TextSegment(text=f" {ratio_str}", font_weight="bold"),
                    ]
                else:
                    effective_label = self.marker

            tag_elements = []
            if svg_path:
                tag_elements.append(SVGElement(svg_content=svg_path, id=f"{self.id}_tag_icon"))
            if effective_label is not None:
                text_id = f"{self.id}_tag_label_0" if self.id else "tag_label_0"
                tag_text = Text(text=effective_label, id=text_id)
                if label_segments:
                    tag_text.segments = label_segments
                tag_elements.append(tag_text)

            if tag_elements:
                tag_id = f"{self.id}_tag" if self.id else "source_tag_proxy"
                self._tag_container = Container(
                    id=tag_id,
                    style_class=["source_tag"],
                    is_overlay=True,
                    children=tag_elements,
                    parent=self,
                )
                # apply styles now so defaults exist before SourceAnnotation extracts it
                jstyle.apply(self._tag_container)
        return self


class GeneticPart(Container):
    """Container holding an SVG shape, optional label, and anchors.
    Set `_label_fit_to_svg = True` to auto-shrink labels wider than the shape."""

    _label_fit_to_svg: ClassVar[bool] = False
    _label_fit_factor: ClassVar[float] = 0.9  # label width ≤ factor × svg width
    _label_fit_min_font_size: ClassVar[float] = 4.0

    part_type: str = "unknown_part"
    part_name: str | None = None
    label: Text | None = None
    auto_resource_path: str = DEFAULT_RESOURCE_PATH

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(justify_content="center", align_items="center")
    )

    style: BoxStyle = Field(default_factory=lambda: BoxStyle(padding=(0, 0, 0, 0)))

    _svg_shape: SVGElement | None = PrivateAttr(default=None)
    _label_fit_applied: bool = PrivateAttr(default=False)

    def model_post_init(self, *args, **kwargs):
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

    def measure_and_layout(self, renderer=None):
        result = super().measure_and_layout(renderer)
        # auto-shrink an oversized label to fit the SVG; idempotent via _label_fit_applied
        if (
            self._label_fit_to_svg
            and not self._label_fit_applied
            and self.label is not None
            and self._svg_shape is not None
            and renderer is not None
        ):
            svg_w = self._svg_shape._natural_dimensions.width
            lbl_w = self.label._natural_dimensions.width
            target = svg_w * self._label_fit_factor
            if svg_w > 0 and lbl_w > target:
                scale = target / lbl_w
                new_size = max(self.label.font_size * scale, self._label_fit_min_font_size)
                if new_size < self.label.font_size:
                    self.label.font_size = new_size
                    # mark user-set so the cascade won't reset it next layout pass
                    self.label._user_set_fields.add("font_size")
                    self.label._text_metrics_cache = None
                    self._label_fit_applied = True
                    result = super().measure_and_layout(renderer)
        return result

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
    anchor_points: list[AnchorComponent] = Field(
        default_factory=lambda: make_vertical_anchors("", "ern")
    )


class FluoMarker(GeneticPart, AutoLabelMixin):
    part_type: str = "fluo_marker"
    _auto_label: ClassVar[bool] = True
    _label_prefix: ClassVar[str] = "fluo_"
    _label_fit_to_svg: ClassVar[bool] = True  # long fluorophore names auto-shrink


class Promoter(GeneticPart):
    part_type: str = "promoter"


class Terminator(GeneticPart):
    part_type: str = "terminator"


class UorfGroup(GeneticPart, AutoLabelMixin):
    part_type: str = "uORF_group"
    _auto_label: ClassVar[bool] = True
    _label_prefix: ClassVar[str] = "uorf_"


class ERN5pRecog(GeneticPart):
    part_type: str = "ERN_recog_site_5p"
    anchor_points: list[AnchorComponent] = Field(
        default_factory=lambda: make_vertical_anchors("recog-", "ern-recog")
    )
