# File: jeanplot/svg.py
# -*- coding: utf-8 -*-
"""SVG data models and utilities for parsing and creating SVG elements."""

from typing import Optional, Union, Any, List, Tuple, Literal, Sequence
from pydantic import BaseModel, Field, model_validator, PrivateAttr
import numpy as np
import re
from lxml import etree
from pathlib import Path
import logging

# use absolute imports
from jeanplot.utils import load_file, load_file_if_exists
from jeanplot.path_utils import normalize_vector
from jeanplot.component import Component
from jeanplot.models import Size, LineWidthMode
from jeanplot.debug import debug_print

logger = logging.getLogger(__name__)

# types
LineStyle = Literal["solid", "dashed", "dotted", "custom"]


class SVGPathData(BaseModel):
    """represents a single SVG path with styling attributes."""

    d: str  # path definition string
    fill: str = "none"
    stroke: str = "none"
    stroke_width: float = 1.0
    transform: Optional[str] = None  # raw svg transform string (applied before component transform)
    # flags for theme color application (e.g., for genetic parts)
    is_main_color: bool = False
    is_secondary_color: bool = False
    # line styling attributes
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0


class SVGContent(BaseModel):
    """structured representation of parsed or generated SVG data."""

    width: float = 100
    height: float = 100
    viewBox: Optional[Tuple[float, float, float, float]] = None
    paths: List[SVGPathData] = Field(default_factory=list)


class LineEndArrow(BaseModel):
    """arrow end cap definition."""

    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    fill_color: str = "none"  # default to no fill
    size: float = 8.0  # length along the line direction
    angle: float = 30.0  # angle of the arrowhead sides
    closed: bool = False  # whether to draw the closing line segment
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0


class LineEndCircle(BaseModel):
    """circle end cap definition."""

    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    fill_color: str = "none"  # default to no fill
    radius: float = 4.0
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0


class LineEndFlat(BaseModel):
    """flat ('T') end cap definition."""

    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    length: float = 6.0  # width of the flat cap perpendicular to line
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0


LineEndType = Union[LineEndArrow, LineEndCircle, LineEndFlat]


def arc_to_bezier(
    center_x: float, center_y: float, radius: float, start_angle_deg: float, end_angle_deg: float
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Convert a 90-degree circular arc to a cubic Bezier curve approximation.
    Angles are in degrees, counter-clockwise from positive x-axis.
    Uses the standard kappa approximation.
    """
    start_rad = np.radians(start_angle_deg)
    end_rad = np.radians(end_angle_deg)
    delta_angle = end_rad - start_rad

    # ensure it's approximately a 90-degree sweep (either direction)
    if not (
        np.isclose(abs(delta_angle), np.pi / 2, atol=1e-6)
        or np.isclose(abs(delta_angle), 3 * np.pi / 2, atol=1e-6)
    ):
        # Fallback for non-90 deg arcs? Could return straight line segment or raise error
        # For now, let's assume caller ensures 90-deg segments for rounded rects
        pass  # Or raise ValueError("arc_to_bezier approximation is best for 90-degree arcs.")

    kappa = 0.5522847498
    dist = kappa * radius

    # Calculate start and end points
    p0_x = center_x + radius * np.cos(start_rad)
    p0_y = center_y + radius * np.sin(start_rad)
    p3_x = center_x + radius * np.cos(end_rad)
    p3_y = center_y + radius * np.sin(end_rad)

    # calculate tangent vectors (normalized vector pointing counter-clockwise along the tangent)
    # tangent at start angle alpha is (-sin(alpha), cos(alpha))
    t0_x, t0_y = -np.sin(start_rad), np.cos(start_rad)
    t3_x, t3_y = -np.sin(end_rad), np.cos(end_rad)

    # P1 = P0 - dist * T0
    p1_x = p0_x - dist * t0_x
    p1_y = p0_y - dist * t0_y
    cp1 = (p1_x, p1_y)

    # P2 = P3 + dist * T3
    p2_x = p3_x + dist * t3_x
    p2_y = p3_y + dist * t3_y
    cp2 = (p2_x, p2_y)

    p_start = (p0_x, p0_y)
    cp1 = (p1_x, p1_y)
    cp2 = (p2_x, p2_y)
    p_end = (p3_x, p3_y)

    return p_start, cp1, cp2, p_end


def create_arrow_cap(
    point: Tuple[float, float], direction: Tuple[float, float], arrow: LineEndArrow
) -> SVGPathData:
    """create SVGPathData for an arrow cap."""
    # direction points outwards from the line end
    dx, dy = normalize_vector(direction)
    perp_x, perp_y = -dy, dx  # perpendicular vector

    angle_rad = np.radians(arrow.angle / 2.0)  # half angle
    back_dist = arrow.size * np.cos(angle_rad)
    half_width = arrow.size * np.sin(angle_rad)

    # calculate points relative to the tip (point)
    left_x = point[0] - dx * back_dist + perp_x * half_width
    left_y = point[1] - dy * back_dist + perp_y * half_width
    right_x = point[0] - dx * back_dist - perp_x * half_width
    right_y = point[1] - dy * back_dist - perp_y * half_width

    # create path string
    path = f"M {left_x} {left_y} L {point[0]} {point[1]} L {right_x} {right_y}"
    if arrow.closed:
        path += " Z"

    return SVGPathData(
        d=path,
        fill=arrow.fill_color if arrow.closed else "none",
        stroke=arrow.stroke_color,
        stroke_width=arrow.stroke_width,
        line_style=arrow.line_style,
        dash_array=arrow.dash_array,
        dash_offset=arrow.dash_offset,
    )


def create_circle_cap(point: Tuple[float, float], circle: LineEndCircle) -> SVGPathData:
    """create SVGPathData for a circle cap."""
    r = circle.radius
    # use two 180-degree arcs to draw the circle
    path = f"M {point[0] - r} {point[1]} A {r} {r} 0 1 1 {point[0] + r} {point[1]} A {r} {r} 0 1 1 {point[0] - r} {point[1]} Z"
    return SVGPathData(
        d=path,
        fill=circle.fill_color,
        stroke=circle.stroke_color,
        stroke_width=circle.stroke_width,
        line_style=circle.line_style,
        dash_array=circle.dash_array,
        dash_offset=circle.dash_offset,
    )


def create_flat_cap(
    point: Tuple[float, float], direction: Tuple[float, float], flat: LineEndFlat
) -> SVGPathData:
    """create SVGPathData for a flat ('T') cap."""
    dx, dy = normalize_vector(direction, (0, 1))  # direction pointing outwards
    perp_x, perp_y = -dy, dx
    half_len = flat.length / 2.0
    p1 = (point[0] + perp_x * half_len, point[1] + perp_y * half_len)
    p2 = (point[0] - perp_x * half_len, point[1] - perp_y * half_len)
    path = f"M {p1[0]} {p1[1]} L {p2[0]} {p2[1]}"
    return SVGPathData(
        d=path,
        fill="none",
        stroke=flat.stroke_color,
        stroke_width=flat.stroke_width,
        line_style=flat.line_style,
        dash_array=flat.dash_array,
        dash_offset=flat.dash_offset,
    )


def _parse_svg_dimension(dim_str: str, ppi: float) -> float:
    """parse svg dimension string (e.g., "100px", "50")"""
    match = re.match(r"[\d\.]+", dim_str)
    val = float(match.group()) if match else 0.0
    # rudimentary unit handling (assume px or unitless is points/ppi)
    if "mm" in dim_str:
        val *= ppi / 25.4
    elif "cm" in dim_str:
        val *= ppi / 2.54
    elif "in" in dim_str:
        val *= ppi
    # default assumption: px or unitless are scaled by ppi relative to points (e.g. ppi=1 for points)
    return val / ppi


def get_svg_data(
    source: Union[str, Path, bytes],
    ppi: float = 1.0,  # points per inch (for unit conversion)
    main_color: str = "#0000FF",  # color used to flag main geometry
    secondary_color: str = "#00FF00",  # color used to flag secondary geometry
) -> SVGContent:
    """extract SVG data from string, bytes, or file path."""
    content = None
    if isinstance(source, (str, Path)):
        try:
            content = load_file(source)
        except FileNotFoundError:
            logger.error(f"svg source file not found: {source}")
            return SVGContent()
    elif isinstance(source, bytes):
        content = source
    elif isinstance(source, str):  # treat as svg content string
        content = source.encode("utf-8")
    else:
        logger.error(f"invalid svg source type: {type(source)}")
        return SVGContent()

    if not content:
        return SVGContent()

    try:
        # use lxml for robust parsing
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(content, parser=parser)
        ns = {"svg": "http://www.w3.org/2000/svg"}  # namespace map

        # extract dimensions
        width = _parse_svg_dimension(root.attrib.get("width", "100"), ppi)
        height = _parse_svg_dimension(root.attrib.get("height", "100"), ppi)

        # extract viewBox
        viewBox = None
        if "viewBox" in root.attrib:
            try:
                vb_parts = [float(x.strip()) for x in root.attrib["viewBox"].split()]
                if len(vb_parts) == 4:
                    viewBox = tuple(vb_parts)
            except ValueError:
                pass  # ignore invalid viewBox

        # extract path data
        paths = []
        # find all path elements, handling potential groups and transforms
        for elem in root.xpath(".//*[local-name()='path']", namespaces=ns):
            try:
                path_data = {"d": elem.attrib.get("d", "")}
                if not path_data["d"]:
                    continue  # skip empty paths

                # basic style attribute extraction (more robust would parse 'style' attribute)
                style = elem.attrib.get("style", "")
                style_props = {}
                if style:
                    try:
                        style_props = dict(
                            item.split(":") for item in style.split(";") if ":" in item
                        )
                    except ValueError:
                        pass  # ignore malformed style string parts

                def get_style_attr(name, default):
                    return elem.attrib.get(name, style_props.get(name, default)).strip()

                path_data["fill"] = get_style_attr("fill", "none")
                path_data["stroke"] = get_style_attr("stroke", "none")
                path_data["stroke_width"] = float(get_style_attr("stroke-width", "1.0"))
                path_data["transform"] = elem.attrib.get("transform")  # keep raw transform string

                # parse dash array/offset
                dash_array_str = get_style_attr("stroke-dasharray", "none")
                dash_offset_str = get_style_attr("stroke-dashoffset", "0")
                line_style: LineStyle = "solid"
                dash_array = None
                dash_offset = 0.0
                if dash_array_str != "none":
                    try:
                        dash_array = tuple(float(x.strip()) for x in dash_array_str.split(","))
                        line_style = (
                            "custom" if dash_array else "solid"
                        )  # Treat empty dasharray as solid
                    except ValueError:
                        pass  # ignore invalid dasharray
                try:
                    dash_offset = float(dash_offset_str)
                except ValueError:
                    pass

                path_data["line_style"] = line_style
                path_data["dash_array"] = dash_array
                path_data["dash_offset"] = dash_offset

                # check for theme colors
                path_data["is_main_color"] = path_data["fill"] == main_color
                path_data["is_secondary_color"] = path_data["fill"] == secondary_color

                paths.append(SVGPathData(**path_data))
            except Exception as e_path:
                logger.warning(f"failed to parse path element: {e_path}")
                continue

        return SVGContent(width=width, height=height, viewBox=viewBox, paths=paths)

    except etree.XMLSyntaxError as e_xml:
        logger.error(f"failed to parse svg content (xml syntax error): {e_xml}")
        return SVGContent()
    except Exception as e_gen:
        logger.error(
            f"failed to process svg source: {e_gen}", exc_info=logger.isEnabledFor(logging.DEBUG)
        )
        return SVGContent()


class SVGTextContent(SVGContent):
    """svg content specialized for text, storing paths and measured size."""

    # paths list holds the glyph outlines as SVGPathData
    # width/height store the constrained allocated size
    measured_width: float = 0  # natural width from glyphs
    measured_height: float = 0  # natural height from glyphs
    text_paths: list[dict[str, Any]] = Field(default_factory=list)


class SVGElement(Component):
    """component that renders svg content loaded from a source."""

    main_color: str = "black"
    secondary_color: str = "gray"
    svg_content: Optional[Union[str, Path, bytes, SVGContent]] = None
    # controls if stroke width is in points or data units for this svg
    line_width_mode: LineWidthMode = "point"  # prefer point default

    _parsed_svg_content: Optional[SVGContent] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _parse_and_validate_svg(self):
        """load and parse svg data if not already done."""
        if isinstance(self.svg_content, SVGContent):
            self._parsed_svg_content = self.svg_content  # already parsed
        elif self.svg_content is not None:
            parsed = get_svg_data(
                self.svg_content, main_color=self.main_color, secondary_color=self.secondary_color
            )
            if parsed:
                self._parsed_svg_content = parsed
            else:
                logger.warning(f"failed to parse svg source for {self.id}")
                self._parsed_svg_content = SVGContent()  # fallback to empty
        else:
            self._parsed_svg_content = SVGContent()  # no source provided

        # set initial dimensions based on parsed svg, used by measure_natural
        if self._parsed_svg_content:
            self._dimensions = Size(
                width=self._parsed_svg_content.width, height=self._parsed_svg_content.height
            )
        else:
            self._dimensions = Size()
        return self

    def _measure_natural(self, renderer=None) -> Size:
        """return base svg dimensions, adjusted by scale transform."""
        if not self._parsed_svg_content:
            return Size()

        base_width = self._parsed_svg_content.width
        base_height = self._parsed_svg_content.height
        scale_x, scale_y = self.transform.scale
        # natural size is affected by scale, but not rotation/translation
        natural_size = Size(width=base_width * abs(scale_x), height=base_height * abs(scale_y))
        # self._log_debug(f"_measure_natural: base=({base_width:.1f},{base_height:.1f}), scale={self.transform.scale} -> {natural_size}")
        return natural_size

    def render(self, renderer, context, matrix: np.ndarray):
        """render svg using the provided renderer."""
        if not self.show or not self._parsed_svg_content or not self._parsed_svg_content.paths:
            return

        # pass line width mode to renderer via options
        self.add_renderer_option(renderer.RENDERER_NAME, "line_width_mode", self.line_width_mode)
        # self._log_debug("rendering svg content", self._parsed_svg_content)
        renderer.render_svg(context, self, matrix)

        if self.debug:
            renderer.render_debug(context, self, matrix)
