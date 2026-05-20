from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr, ConfigDict
import numpy as np
import re
from lxml import etree
from pathlib import Path
import logging

from jeanplot.core.utils import load_file
from jeanplot.core.path_utils import normalize_vector
from jeanplot.core.component import Component
from jeanplot.core.models import Size, LineWidthMode, normalize_color

logger = logging.getLogger(__name__)

LineStyle = Literal["solid", "dashed", "dotted", "custom"]


class SVGPathData(BaseModel):
    """single svg path with styling attributes."""

    model_config = ConfigDict(frozen=True)

    d: str
    fill: str | None = None
    stroke: str | None = None
    stroke_width: float = 1.0
    transform: str | None = None
    line_style: LineStyle = "solid"
    dash_array: tuple[float, ...] | None = None
    dash_offset: float = 0.0


class SVGContent(BaseModel):
    """parsed or generated svg data."""

    model_config = ConfigDict(frozen=True)

    width: float = 100
    height: float = 100
    viewBox: tuple[float, float, float, float] | None = None
    paths: tuple[SVGPathData, ...] = Field(default_factory=tuple)


class SVGTextContent(BaseModel):
    """svg content specialized for text: glyph paths + measured size."""

    glyph_paths: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    measured_width: float = 0
    measured_height: float = 0


class LineEndArrow(BaseModel):
    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    fill_color: str | None = None
    length: float = 8.0
    angle: float = 30.0
    closed: bool = False
    line_style: LineStyle = "solid"
    dash_array: tuple[float, ...] | None = None
    dash_offset: float = 0.0


class LineEndCircle(BaseModel):
    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    fill_color: str | None = None
    radius: float = 4.0
    line_style: LineStyle = "solid"
    dash_array: tuple[float, ...] | None = None
    dash_offset: float = 0.0


class LineEndFlat(BaseModel):
    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    length: float = 6.0
    line_style: LineStyle = "solid"
    dash_array: tuple[float, ...] | None = None
    dash_offset: float = 0.0


LineEndType = LineEndArrow | LineEndCircle | LineEndFlat


def make_svg_line(width: float, thickness: float, color: str | None) -> SVGContent:
    """svgcontent for a simple horizontal line."""
    norm_color = normalize_color(color)
    if width <= 0 or thickness <= 0 or norm_color is None:
        return SVGContent(width=0, height=0, paths=())
    path_d = f"M 0 {thickness / 2:.3f} L {width:.3f} {thickness / 2:.3f}"
    path_data = SVGPathData(
        d=path_d,
        stroke=norm_color,
        stroke_width=thickness,
        fill=None,
    )
    return SVGContent(width=width, height=thickness, paths=(path_data,))


def arc_to_bezier(
    center_x: float, center_y: float, radius: float, start_angle_deg: float, end_angle_deg: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """convert a 90-degree circular arc to a cubic bezier curve."""
    start_rad = np.radians(start_angle_deg)
    end_rad = np.radians(end_angle_deg)
    kappa = 0.5522847498
    dist = kappa * radius
    p0_x, p0_y = center_x + radius * np.cos(start_rad), center_y + radius * np.sin(start_rad)
    p3_x, p3_y = center_x + radius * np.cos(end_rad), center_y + radius * np.sin(end_rad)
    t0_x, t0_y = -np.sin(start_rad), np.cos(start_rad)
    t3_x, t3_y = -np.sin(end_rad), np.cos(end_rad)
    p1_x, p1_y = p0_x - dist * t0_x, p0_y - dist * t0_y
    p2_x, p2_y = p3_x + dist * t3_x, p3_y + dist * t3_y
    return (p0_x, p0_y), (p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y)


def create_arrow_cap(
    point: tuple[float, float], direction: tuple[float, float], arrow: LineEndArrow
) -> SVGPathData:
    """svgpathdata for an arrow cap."""
    dx, dy = normalize_vector(direction)
    perp_x, perp_y = -dy, dx
    angle_rad = np.radians(arrow.angle / 2.0)
    back_dist = arrow.length * np.cos(angle_rad)
    half_width = arrow.length * np.sin(angle_rad)
    left_x = point[0] - dx * back_dist + perp_x * half_width
    left_y = point[1] - dy * back_dist + perp_y * half_width
    right_x = point[0] - dx * back_dist - perp_x * half_width
    right_y = point[1] - dy * back_dist - perp_y * half_width
    path = (
        f"M {left_x:.3f} {left_y:.3f} L {point[0]:.3f} {point[1]:.3f} L {right_x:.3f} {right_y:.3f}"
    )
    if arrow.closed:
        path += " Z"
    return SVGPathData(
        d=path,
        fill=normalize_color(arrow.fill_color) if arrow.closed else None,
        stroke=normalize_color(arrow.stroke_color),
        stroke_width=arrow.stroke_width,
        line_style=arrow.line_style,
        dash_array=arrow.dash_array,
        dash_offset=arrow.dash_offset,
    )


def create_circle_cap(
    point: tuple[float, float], _: tuple[float, float], circle: LineEndCircle
) -> SVGPathData:
    """svgpathdata for a circle cap."""
    r = circle.radius
    path = f"M {point[0] - r:.3f} {point[1]:.3f} A {r:.3f} {r:.3f} 0 1 1 {point[0] + r:.3f} {point[1]:.3f} A {r:.3f} {r:.3f} 0 1 1 {point[0] - r:.3f} {point[1]:.3f} Z"
    return SVGPathData(
        d=path,
        fill=normalize_color(circle.fill_color),
        stroke=normalize_color(circle.stroke_color),
        stroke_width=circle.stroke_width,
        line_style=circle.line_style,
        dash_array=circle.dash_array,
        dash_offset=circle.dash_offset,
    )


def create_flat_cap(
    point: tuple[float, float], direction: tuple[float, float], flat: LineEndFlat
) -> SVGPathData:
    """svgpathdata for a flat ('t') cap."""
    dx, dy = normalize_vector(direction, (0, 1))
    perp_x, perp_y = -dy, dx
    half_len = flat.length / 2.0
    p1 = (point[0] + perp_x * half_len, point[1] + perp_y * half_len)
    p2 = (point[0] - perp_x * half_len, point[1] - perp_y * half_len)
    path = f"M {p1[0]:.3f} {p1[1]:.3f} L {p2[0]:.3f} {p2[1]:.3f}"
    return SVGPathData(
        d=path,
        fill=None,
        stroke=normalize_color(flat.stroke_color),
        stroke_width=flat.stroke_width,
        line_style=flat.line_style,
        dash_array=flat.dash_array,
        dash_offset=flat.dash_offset,
    )


def _parse_svg_dimension(dim_str: str, ppi: float) -> float:
    """parse svg dimension string (e.g., "100px", "50")."""
    if not isinstance(dim_str, str):
        return 0.0
    match = re.match(r"^\s*([\d\.]+)\s*([a-zA-Z%]*)", dim_str)
    if not match:
        return 0.0
    val_str, unit = match.groups()
    try:
        val = float(val_str)
    except ValueError:
        return 0.0
    unit = unit.lower()
    if unit == "mm":
        val *= ppi / 25.4
    elif unit == "cm":
        val *= ppi / 2.54
    elif unit == "in":
        val *= ppi
    elif unit == "pt":
        val *= ppi / 72.0
    return val


def get_svg_data(source: str | Path | bytes, ppi: float = 72.0) -> SVGContent:
    content_bytes: bytes | None = None
    if isinstance(source, bytes):
        content_bytes = source
    elif isinstance(source, str):
        if source.strip().startswith("<svg"):
            content_bytes = source.encode("utf-8")
        else:
            try:
                loaded_content = load_file(source)
                if isinstance(loaded_content, str):
                    content_bytes = loaded_content.encode("utf-8")
            except Exception as e_load:
                logger.error(f"error loading svg source path {source}: {e_load}")
                return SVGContent(paths=())
    elif isinstance(source, Path):
        try:
            loaded_content = load_file(source)
            if isinstance(loaded_content, str):
                content_bytes = loaded_content.encode("utf-8")
        except Exception as e_load:
            logger.error(f"error loading svg source path {source}: {e_load}")
            return SVGContent(paths=())
    else:
        logger.error(f"invalid svg source type: {type(source)}")
        return SVGContent(paths=())

    if not content_bytes:
        return SVGContent(paths=())

    try:
        parser = etree.XMLParser(remove_blank_text=True, recover=True)
        root = etree.fromstring(content_bytes, parser=parser)
        ns = {"svg": "http://www.w3.org/2000/svg"}

        width = _parse_svg_dimension(root.attrib.get("width", "100"), ppi)
        height = _parse_svg_dimension(root.attrib.get("height", "100"), ppi)
        viewBox = None
        if "viewBox" in root.attrib:
            try:
                vb_parts = [
                    float(x.strip())
                    for x in re.split(r"[,\s]+", root.attrib["viewBox"])
                    if x.strip()
                ]
                if len(vb_parts) == 4:
                    viewBox = tuple(vb_parts)
            except (ValueError, TypeError):
                pass

        paths_list: list[SVGPathData] = []
        for elem in root.xpath(".//*[local-name()='path']", namespaces=ns):
            try:
                path_d = elem.attrib.get("d", "")
                if not path_d:
                    continue

                style = elem.attrib.get("style", "")
                style_props = (
                    {
                        k.strip(): v.strip()
                        for k, v in (item.split(":") for item in style.split(";") if ":" in item)
                    }
                    if style
                    else {}
                )

                def get_style_attr(name, default):
                    val = elem.attrib.get(name)
                    return str(style_props.get(name, default) if val is None else val).strip()

                fill_color = normalize_color(get_style_attr("fill", "none"))
                stroke_color = normalize_color(get_style_attr("stroke", "none"))
                stroke_width = 1.0
                try:
                    stroke_width = float(get_style_attr("stroke-width", "1.0"))
                except ValueError:
                    pass
                transform = elem.attrib.get("transform")
                dash_array_str = get_style_attr("stroke-dasharray", "none")
                dash_offset_str = get_style_attr("stroke-dashoffset", "0")
                line_style: LineStyle = "solid"
                dash_array = None
                dash_offset = 0.0
                if dash_array_str != "none":
                    try:
                        dash_values = [
                            float(x.strip())
                            for x in re.split(r"[,\s]+", dash_array_str)
                            if x.strip()
                        ]
                        if dash_values:
                            dash_array = tuple(dash_values)
                        line_style = "custom" if dash_array else "solid"
                    except ValueError:
                        pass
                try:
                    dash_offset = float(dash_offset_str)
                except ValueError:
                    pass

                paths_list.append(
                    SVGPathData(
                        d=path_d,
                        fill=fill_color,
                        stroke=stroke_color,
                        stroke_width=stroke_width,
                        transform=transform,
                        line_style=line_style,
                        dash_array=dash_array,
                        dash_offset=dash_offset,
                    )
                )
            except Exception as e_path:
                logger.warning(f"failed parsing path: {e_path}")

        content = SVGContent(width=width, height=height, viewBox=viewBox, paths=tuple(paths_list))
        # mark colors as normalized during parsing
        object.__setattr__(content, "_colors_normalized", True)
        return content

    except Exception as e_gen:
        logger.error(f"failed processing svg: {e_gen}", exc_info=logger.isEnabledFor(logging.DEBUG))
        return SVGContent(paths=())


class SVGElement(Component):
    svg_content: str | Path | bytes | SVGContent | None = None
    color_remap: dict[str, str | None] = Field(default_factory=dict)
    line_width_mode: LineWidthMode = "data"

    _parsed_svg_content: SVGContent | None = PrivateAttr(default=None)

    @field_validator("color_remap")
    @classmethod
    def normalize_color_remap(cls, v: dict[str, str | None]) -> dict[str, str | None]:
        normalized_map = {}
        if not isinstance(v, dict):
            return {}
        for key_in, val_out in v.items():
            norm_key = normalize_color(key_in)
            norm_val = normalize_color(val_out)
            if norm_key:
                normalized_map[norm_key] = norm_val
        return normalized_map

    @model_validator(mode="after")
    def _parse_and_validate_svg(self):
        """load and parse svg data if not already done."""
        if isinstance(self.svg_content, SVGContent):
            if not getattr(self.svg_content, "_colors_normalized", False):
                paths = tuple(
                    p.model_copy(
                        update={
                            "fill": normalize_color(p.fill),
                            "stroke": normalize_color(p.stroke),
                        }
                    )
                    for p in self.svg_content.paths
                )
                self._parsed_svg_content = self.svg_content.model_copy(update={"paths": paths})
                object.__setattr__(self._parsed_svg_content, "_colors_normalized", True)
            else:
                self._parsed_svg_content = self.svg_content
        elif self.svg_content is not None:
            parsed = get_svg_data(self.svg_content)
            if parsed:
                self._parsed_svg_content = parsed
            else:
                logger.warning(f"failed to parse svg source for {self.id}")
                self._parsed_svg_content = SVGContent(paths=())
        elif self._parsed_svg_content is None:
            self._parsed_svg_content = SVGContent(paths=())

        if self._parsed_svg_content and (
            self._dimensions.width <= 0 or self._dimensions.height <= 0
        ):
            w = max(0.0, self._parsed_svg_content.width)
            h = max(0.0, self._parsed_svg_content.height)
            if self._dimensions.width <= 0:
                self._dimensions.width = w
            if self._dimensions.height <= 0:
                self._dimensions.height = h
        return self

    def _measure_natural(self, renderer=None) -> Size:
        """base svg dimensions adjusted by scale transform."""
        if not self._parsed_svg_content:
            self._parse_and_validate_svg()

        if not self._parsed_svg_content:
            return Size()

        base_width = self._parsed_svg_content.width
        base_height = self._parsed_svg_content.height
        scale_x, scale_y = self.transform.scale
        nat_w = base_width * abs(scale_x)
        nat_h = base_height * abs(scale_y)
        return Size(width=nat_w, height=nat_h)

    def render(self, renderer, context, matrix: np.ndarray):
        if not self.show:
            return

        if not self._parsed_svg_content:
            self._parse_and_validate_svg()

        if not self._parsed_svg_content or not self._parsed_svg_content.paths:
            if self.debug:
                renderer.render_debug(context, self, matrix)
            return

        self.add_renderer_option(renderer.RENDERER_NAME, "line_width_mode", self.line_width_mode)
        self.add_renderer_option(renderer.RENDERER_NAME, "color_remap", self.color_remap)
        renderer.render_svg(context, self, matrix)

        if self.debug:
            renderer.render_debug(context, self, matrix)
