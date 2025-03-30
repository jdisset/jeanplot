from typing import Optional, Union, Any, List, Tuple, Literal, Sequence
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
import numpy as np
import re
from lxml import etree
from .component import Component
from .models import Size

# line style enum
LineStyle = Literal["solid", "dashed", "dotted", "custom"]


class SVGPathData(BaseModel):
    """represents a single SVG path with its attributes"""

    d: str
    fill: str = "none"
    stroke: str = "none"
    stroke_width: float = 1.0
    transform: Optional[str] = None
    is_main_color: bool = False
    is_secondary_color: bool = False
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0


class SVGContent(BaseModel):
    """structured representation of SVG data"""

    width: float = 100
    height: float = 100
    viewBox: Optional[Tuple[float, float, float, float]] = None
    paths: List[SVGPathData] = Field(default_factory=list)


def svg_line_path(
    length: float,
    thickness: float = 1.0,
    color: str = "#000000",
    line_style: LineStyle = "solid",
    dash_array: Optional[Tuple[float, ...]] = None,
    dash_offset: float = 0.0,
) -> SVGPathData:
    """create single path data for a line"""
    if line_style == "dashed" and not dash_array:
        dash_array = (thickness * 3, thickness * 2)
    elif line_style == "dotted" and not dash_array:
        dash_array = (thickness, thickness)

    return SVGPathData(
        d=f"M 0 {thickness / 2} L {length} {thickness / 2}",
        stroke=color,
        stroke_width=thickness,
        fill="none",
        line_style=line_style,
        dash_array=dash_array,
        dash_offset=dash_offset,
    )


def make_svg_line(
    length: float,
    thickness: float = 1.0,
    color: str = "#000000",
    line_style: LineStyle = "solid",
    dash_array: Optional[Tuple[float, ...]] = None,
    dash_offset: float = 0.0,
) -> SVGContent:
    """create SVG content for a line"""
    return SVGContent(
        width=length,
        height=thickness,
        viewBox=(0, 0, length, thickness),
        paths=[
            svg_line_path(length, thickness, color, line_style, dash_array, dash_offset),
        ],
    )


class LineEndArrow(BaseModel):
    """arrow end cap for lines"""

    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    fill_color: str = "#FFFFFF"
    size: float = 1.0
    angle: float = 30.0
    closed: bool = True
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0


class LineEndCircle(BaseModel):
    """circle end cap for lines"""

    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    fill_color: str = "#FFFFFF"
    radius: float = 1.0
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0


class LineEndFlat(BaseModel):
    """flat end cap for lines"""

    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    length: float = 1.0
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0


LineEndType = Union[LineEndArrow, LineEndCircle, LineEndFlat]


def create_bezier_path_string(
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    control_points: List[Tuple[float, float]],
) -> str:
    """create svg path string for a bezier curve"""
    path = f"M {start_point[0]} {start_point[1]}"

    if not control_points:
        path += f" L {end_point[0]} {end_point[1]}"
    elif len(control_points) == 1:
        ctrl = control_points[0]
        path += f" Q {ctrl[0]} {ctrl[1]}, {end_point[0]} {end_point[1]}"
    elif len(control_points) == 2:
        ctrl1, ctrl2 = control_points
        path += f" C {ctrl1[0]} {ctrl1[1]}, {ctrl2[0]} {ctrl2[1]}, {end_point[0]} {end_point[1]}"
    else:
        ctrl1, ctrl2 = control_points[:2]
        path += f" C {ctrl1[0]} {ctrl1[1]}, {ctrl2[0]} {ctrl2[1]}, {end_point[0]} {end_point[1]}"

    return path


def create_circle_cap(point: Tuple[float, float], circle: LineEndCircle) -> SVGPathData:
    """create circle end cap"""
    r = circle.radius
    path = (
        f"M {point[0] - r} {point[1]} "
        f"A {r} {r} 0 1 1 {point[0] + r} {point[1]} "
        f"A {r} {r} 0 1 1 {point[0] - r} {point[1]} Z"
    )

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
    point: Tuple[float, float], direction_vector: Tuple[float, float], flat: LineEndFlat
) -> SVGPathData:
    """create flat end cap"""
    # normalize direction
    length = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    if length > 0:
        dx, dy = direction_vector[0] / length, direction_vector[1] / length
    else:
        dx, dy = 0, 1

    # perpendicular vector
    perp_x, perp_y = -dy, dx

    # endpoints
    half_len = flat.length / 2
    p1_x = point[0] + perp_x * half_len
    p1_y = point[1] + perp_y * half_len
    p2_x = point[0] - perp_x * half_len
    p2_y = point[1] - perp_y * half_len

    path = f"M {p1_x} {p1_y} L {p2_x} {p2_y}"

    return SVGPathData(
        d=path,
        fill="none",
        stroke=flat.stroke_color,
        stroke_width=flat.stroke_width,
        line_style=flat.line_style,
        dash_array=flat.dash_array,
        dash_offset=flat.dash_offset,
    )


def make_svg_bezier(
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    control_points: List[Tuple[float, float]],
    width: float,
    height: float,
    color: str = "#000000",
    line_width: float = 1.0,
    line_style: LineStyle = "solid",
    dash_array: Optional[Tuple[float, ...]] = None,
    dash_offset: float = 0.0,
    start_cap: Optional[LineEndType] = None,
    end_cap: Optional[LineEndType] = None,
) -> SVGContent:
    """create SVG bezier curve with optional end caps"""
    paths = []

    # apply default dash arrays for common styles
    if line_style == "dashed" and not dash_array:
        dash_array = (line_width * 3, line_width * 2)
    elif line_style == "dotted" and not dash_array:
        dash_array = (line_width, line_width)

    # create main bezier path
    bezier_path = create_bezier_path_string(start_point, end_point, control_points)
    paths.append(
        SVGPathData(
            d=bezier_path,
            stroke=color,
            stroke_width=line_width,
            fill="none",
            line_style=line_style,
            dash_array=dash_array,
            dash_offset=dash_offset,
        )
    )

    # calculate direction vectors for caps
    if start_cap or end_cap:
        # calculate direction vectors based on control points or direct line
        if control_points:
            # direction at start
            start_dir = (
                control_points[0][0] - start_point[0],
                control_points[0][1] - start_point[1],
            )
            # use a small distance if vector is zero
            if start_dir[0] == 0 and start_dir[1] == 0:
                start_dir = (end_point[0] - start_point[0], end_point[1] - start_point[1])
                # still zero? use default
                if start_dir[0] == 0 and start_dir[1] == 0:
                    start_dir = (1, 0)

            # direction at end
            if len(control_points) > 1:
                end_dir = (
                    end_point[0] - control_points[-1][0],
                    end_point[1] - control_points[-1][1],
                )
            else:
                end_dir = (end_point[0] - control_points[0][0], end_point[1] - control_points[0][1])
            # use a small distance if vector is zero
            if end_dir[0] == 0 and end_dir[1] == 0:
                end_dir = (end_point[0] - start_point[0], end_point[1] - start_point[1])
                # still zero? use default
                if end_dir[0] == 0 and end_dir[1] == 0:
                    end_dir = (1, 0)
        else:
            # straight line
            line_dir = (end_point[0] - start_point[0], end_point[1] - start_point[1])
            # if points are identical, use horizontal direction
            if line_dir[0] == 0 and line_dir[1] == 0:
                line_dir = (1, 0)
            start_dir = line_dir
            end_dir = line_dir

        # add start cap
        if start_cap:
            if isinstance(start_cap, LineEndArrow):
                # reverse direction for start arrow
                rev_dir = (-start_dir[0], -start_dir[1])
                paths.append(create_arrow_cap(start_point, rev_dir, start_cap))
            elif isinstance(start_cap, LineEndCircle):
                paths.append(create_circle_cap(start_point, start_cap))
            elif isinstance(start_cap, LineEndFlat):
                rev_dir = (-start_dir[0], -start_dir[1])
                paths.append(create_flat_cap(start_point, rev_dir, start_cap))

        # add end cap
        if end_cap:
            if isinstance(end_cap, LineEndArrow):
                paths.append(create_arrow_cap(end_point, end_dir, end_cap))
            elif isinstance(end_cap, LineEndCircle):
                paths.append(create_circle_cap(end_point, end_cap))
            elif isinstance(end_cap, LineEndFlat):
                paths.append(create_flat_cap(end_point, end_dir, end_cap))

    # Create SVG content with viewBox matching the provided dimensions
    return SVGContent(width=width, height=height, viewBox=(0, 0, width, height), paths=paths)


def create_arrow_cap(
    point: Tuple[float, float], direction_vector: Tuple[float, float], arrow: LineEndArrow
) -> SVGPathData:
    """create arrow end cap"""
    # normalize direction vector
    length = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    if length > 0:
        dx, dy = direction_vector[0] / length, direction_vector[1] / length
    else:
        dx, dy = 1, 0  # default horizontal direction if vector is zero

    # perpendicular vector
    perp_x, perp_y = -dy, dx

    # arrow points
    size = arrow.size
    angle_rad = np.radians(arrow.angle)
    back_x = np.cos(angle_rad) * size
    back_y = np.sin(angle_rad) * size

    left_x = point[0] - dx * back_x + perp_x * back_y
    left_y = point[1] - dy * back_x + perp_y * back_y
    right_x = point[0] - dx * back_x - perp_x * back_y
    right_y = point[1] - dy * back_x - perp_y * back_y

    # create path
    if arrow.closed:
        path = f"M {point[0]} {point[1]} L {left_x} {left_y} L {right_x} {right_y} Z"
    else:
        path = f"M {left_x} {left_y} L {point[0]} {point[1]} L {right_x} {right_y}"

    return SVGPathData(
        d=path,
        fill=arrow.fill_color if arrow.closed else "none",
        stroke=arrow.stroke_color,
        stroke_width=arrow.stroke_width,
        line_style=arrow.line_style,
        dash_array=arrow.dash_array,
        dash_offset=arrow.dash_offset,
    )


def get_svg_data_from_string(
    svg_content: str,
    ppi: float = 1.0,
    main_color: str = "#0000FF",
    secondary_color: str = "#00FF00",
) -> SVGContent:
    """extract SVG data from string"""
    try:
        root = etree.fromstring(svg_content.encode("utf-8"))

        width_str = root.attrib.get("width", "100")
        height_str = root.attrib.get("height", "100")

        # extract dimensions
        width = float(re.match(r"[\d\.]+", width_str).group()) / ppi
        height = float(re.match(r"[\d\.]+", height_str).group()) / ppi

        # extract viewBox
        viewBox = None
        if "viewBox" in root.attrib:
            try:
                viewBox = tuple(float(x) for x in root.attrib["viewBox"].split())
            except ValueError:
                pass

        # extract paths
        paths = []
        for elem in root.findall(".//{http://www.w3.org/2000/svg}path"):
            try:
                fill = elem.attrib.get("fill", "none")
                stroke = elem.attrib.get("stroke", "none")
                stroke_width = float(elem.attrib.get("stroke-width", 1.0))
                transform = elem.attrib.get("transform")

                # extract dash array if present
                line_style = "solid"
                dash_array = None
                dash_offset = 0.0

                if "stroke-dasharray" in elem.attrib:
                    dash_str = elem.attrib["stroke-dasharray"]
                    if dash_str and dash_str != "none":
                        try:
                            dash_array = tuple(float(x) for x in dash_str.split(","))
                            line_style = "custom"
                        except ValueError:
                            pass

                if "stroke-dashoffset" in elem.attrib:
                    try:
                        dash_offset = float(elem.attrib["stroke-dashoffset"])
                    except ValueError:
                        pass

                # determine color flags
                is_main_color = fill == main_color
                is_secondary_color = fill == secondary_color

                path_data = SVGPathData(
                    d=elem.attrib["d"],
                    fill=fill,
                    stroke=stroke,
                    stroke_width=stroke_width,
                    transform=transform,
                    is_main_color=is_main_color,
                    is_secondary_color=is_secondary_color,
                    line_style=line_style,
                    dash_array=dash_array,
                    dash_offset=dash_offset,
                )
                paths.append(path_data)
            except Exception:
                continue

        return SVGContent(
            width=width,
            height=height,
            viewBox=viewBox,
            paths=paths,
        )
    except Exception as e:
        print(f"Failed to parse SVG string: {e}")
        return SVGContent()


def get_svg_data_from_file(
    file_path: Union[str, Path],
    ppi: float = 1.0,
    main_color: str = "#0000FF",
    secondary_color: str = "#00FF00",
) -> SVGContent:
    """extract SVG data from file"""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SVG file not found: {path.absolute()}")

        with open(path, "r") as f:
            svg_content = f.read()

        return get_svg_data_from_string(svg_content, ppi, main_color, secondary_color)
    except Exception as e:
        print(f"Failed to load SVG file: {e}")
        return SVGContent()


class SVGElement(Component):
    """svg element loaded from a file"""

    main_color: str = "black"
    secondary_color: str = "gray"
    svg_content: Optional[Union[str, Path, SVGContent]] = None

    @model_validator(mode="after")
    def load_svg_data(self):
        """load svg data from file or string"""
        if isinstance(self.svg_content, (str, Path)):
            try:
                self.svg_content = get_svg_data_from_file(self.svg_content)
            except Exception as e:
                print(f"Error loading SVG: {e}")
                self.svg_content = SVGContent()

        assert isinstance(self.svg_content, SVGContent), "Invalid SVG data"

        self._dimensions = Size(
            width=self.svg_content.width,
            height=self.svg_content.height,
        )
        self._transformed_aabb = self.compute_transformed_aabb()
        return self

    def measure(self, renderer=None) -> Size:
        """return svg dimensions"""
        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render svg using renderer"""
        renderer.render_svg(context, self, matrix)

        if self.debug:
            renderer.render_debug(context, self, matrix)
