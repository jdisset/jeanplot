from typing import Optional, Literal, Union, Dict, Any, List, Tuple
from pydantic import Field, PrivateAttr, BaseModel, model_validator
import numpy as np

from .component import Component
from .models import Size, Offset, Transform, LineWidthMode
from .svg import (
    SVGElement,
    SVGContent,
    SVGPathData,
    LineEndArrow,
    LineEndType,
    LineStyle,
    LineEndCircle,
    LineEndFlat,
    create_arrow_cap,
    create_circle_cap,
    create_flat_cap,
)


class CurveDefinition(BaseModel):
    """base class for different curve types"""

    pass


class StraightCurve(CurveDefinition):
    """straight line between points"""

    pass


class SimpleBezierCurve(CurveDefinition):
    """bezier curve with vectors from start and end points"""

    start_vec: tuple[float, float]  # vector from start point
    end_vec: tuple[float, float]  # vector from end point


class AdvancedBezierCurve(CurveDefinition):
    """bezier curve with explicit control points"""

    control_points: list[tuple[float, float]]


class OrthogonalCurve(CurveDefinition):
    """path with orthogonal segments (right angles)"""

    start_direction: Literal["up", "down", "left", "right"]
    start_length: float  # minimum length of the start segment
    end_direction: Literal["up", "down", "left", "right"]
    end_length: float  # minimum length of the end segment
    corner_radius: float = 10.0


class Connection(Component):
    """connects two components with a styled line/curve"""

    start_component: Component
    end_component: Component

    start_offset: Offset = Field(default_factory=Offset)  # offset from start component's origin
    end_offset: Offset = Field(default_factory=Offset)  # offset from end component's origin

    color: str = "#000000"
    width: float = 1.0  # aka thickness

    curve_type: CurveDefinition = Field(default_factory=StraightCurve)
    line_style: LineStyle = "solid"
    dash_array: Optional[tuple[float, ...]] = None
    dash_offset: float = 0.0

    start_cap: Optional[LineEndType] = None
    end_cap: Optional[LineEndType] = None

    is_overlay: bool = True

    _svg_element: Optional[SVGElement] = PrivateAttr(default=None)
    _local_start: Optional[tuple[float, float]] = PrivateAttr(default=None)
    _local_end: Optional[tuple[float, float]] = PrivateAttr(default=None)

    def measure(self, renderer=None) -> Size:
        """calculate dimensions based on connected components positions"""
        # need parent container to position properly
        if self.parent is None:
            self._dimensions = Size(width=0, height=0)
            self._transformed_aabb = Size(width=0, height=0)
            return self._dimensions

        # get world coordinates of connection points
        start_world, end_world = self._calculate_world_connection_points()

        # transform to parent's coordinate system
        parent_matrix = self.parent.compute_world_matrix()
        if np.linalg.det(parent_matrix) == 0:  # avoid singular matrix
            self._dimensions = Size(width=0, height=0)
            self._transformed_aabb = Size(width=0, height=0)
            return self._dimensions

        parent_inv = np.linalg.inv(parent_matrix)

        start_parent = parent_inv @ np.array([start_world[0], start_world[1], 1])
        end_parent = parent_inv @ np.array([end_world[0], end_world[1], 1])

        # calculate bounds with buffer
        min_x = min(start_parent[0], end_parent[0])
        max_x = max(start_parent[0], end_parent[0])
        min_y = min(start_parent[1], end_parent[1])
        max_y = max(start_parent[1], end_parent[1])

        # add buffer for line width and control points
        buffer = max(self.width * 3, 20)
        min_x -= buffer
        min_y -= buffer
        max_x += buffer
        max_y += buffer

        # set dimensions and position
        self._dimensions = Size(width=max_x - min_x, height=max_y - min_y)
        self.transform.translate = (min_x, min_y)

        # store local points for SVG
        self._local_start = (start_parent[0] - min_x, start_parent[1] - min_y)
        self._local_end = (end_parent[0] - min_x, end_parent[1] - min_y)

        # create SVG content
        self._create_svg_content()

        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def _calculate_world_connection_points(self):
        """calculate start and end points in world coordinates"""
        # get world matrices
        start_world_matrix = self.start_component.compute_world_matrix()
        end_world_matrix = self.end_component.compute_world_matrix()

        # calculate dimensions
        start_dims = self.start_component._dimensions
        end_dims = self.end_component._dimensions

        # apply offsets from origins
        start_ox, start_oy = self.start_offset.compute(start_dims)
        end_ox, end_oy = self.end_offset.compute(end_dims)

        # create points in local space with offset
        start_local = np.array([start_ox, start_oy, 1])
        end_local = np.array([end_ox, end_oy, 1])

        # transform to world space
        start_world = start_world_matrix @ start_local
        end_world = end_world_matrix @ end_local

        return (start_world[0], start_world[1]), (end_world[0], end_world[1])

    def _create_svg_content(self):
        """create SVG content based on connection points and curve type"""
        if not hasattr(self, "_local_start") or not hasattr(self, "_local_end"):
            return

        start = self._local_start
        assert start is not None
        end = self._local_end
        assert end is not None
        width, height = self._dimensions.width, self._dimensions.height

        # create path based on curve type
        path_str = ""
        control_points = []

        if isinstance(self.curve_type, StraightCurve):
            path_str = f"M {start[0]} {start[1]} L {end[0]} {end[1]}"

        elif isinstance(self.curve_type, SimpleBezierCurve):
            # calculate control points from vectors
            start_vec = self.curve_type.start_vec
            end_vec = self.curve_type.end_vec

            ctrl1 = (start[0] + start_vec[0], start[1] + start_vec[1])
            ctrl2 = (end[0] + end_vec[0], end[1] + end_vec[1])

            path_str = f"M {start[0]} {start[1]} C {ctrl1[0]} {ctrl1[1]}, {ctrl2[0]} {ctrl2[1]}, {end[0]} {end[1]}"
            control_points = [ctrl1, ctrl2]

        elif isinstance(self.curve_type, AdvancedBezierCurve):
            # use explicit control points
            cps = self.curve_type.control_points
            if len(cps) == 1:
                # quadratic bezier
                path_str = f"M {start[0]} {start[1]} Q {cps[0][0]} {cps[0][1]}, {end[0]} {end[1]}"
            else:
                # cubic bezier
                ctrl1 = cps[0] if len(cps) > 0 else start
                ctrl2 = cps[1] if len(cps) > 1 else end
                path_str = f"M {start[0]} {start[1]} C {ctrl1[0]} {ctrl1[1]}, {ctrl2[0]} {ctrl2[1]}, {end[0]} {end[1]}"
            control_points = cps

        elif isinstance(self.curve_type, OrthogonalCurve):
            # calculate orthogonal path
            points = self._calculate_orthogonal_path(start, end)

            if self.curve_type.corner_radius > 0:
                path_str = self._create_rounded_orthogonal_path(points)
            else:
                path_str = f"M {points[0][0]} {points[0][1]}"
                for p in points[1:]:
                    path_str += f" L {p[0]} {p[1]}"

        else:  # default to straight
            path_str = f"M {start[0]} {start[1]} L {end[0]} {end[1]}"

        # create path data
        path_data = SVGPathData(
            d=path_str,
            stroke=self.color,
            stroke_width=self.width,
            fill="none",
            line_style=self.line_style,
            dash_array=self.dash_array,
            dash_offset=self.dash_offset,
        )

        # add end caps
        paths = [path_data]
        self._add_end_caps(paths, start, end, control_points)

        svg_content = SVGContent(
            width=width, height=height, viewBox=(0, 0, width, height), paths=paths
        )

        self._svg_element = SVGElement(svg_content=svg_content)

    def _calculate_orthogonal_path(self, start, end):
        """calculate points for orthogonal path"""
        # get direction vectors
        direction_map = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

        start_dir = direction_map[self.curve_type.start_direction]
        end_dir = direction_map[self.curve_type.end_direction]

        first_segment_end = (
            start[0] + start_dir[0] * self.curve_type.start_length,
            start[1] + start_dir[1] * self.curve_type.start_length,
        )

        last_segment_start = (
            end[0] + end_dir[0] * self.curve_type.end_length,
            end[1] + end_dir[1] * self.curve_type.end_length,
        )

        # determine if we need a middle segment
        if (start_dir[0] == 0 and end_dir[0] == 0) or (start_dir[1] == 0 and end_dir[1] == 0):
            # parallel directions, need a middle segment
            mid_x = (first_segment_end[0] + last_segment_start[0]) / 2
            mid_y = (first_segment_end[1] + last_segment_start[1]) / 2

            if start_dir[0] == 0:  # vertical start
                return [
                    start,
                    first_segment_end,
                    (mid_x, first_segment_end[1]),
                    (mid_x, last_segment_start[1]),
                    last_segment_start,
                    end,
                ]
            else:  # horizontal start
                return [
                    start,
                    first_segment_end,
                    (first_segment_end[0], mid_y),
                    (last_segment_start[0], mid_y),
                    last_segment_start,
                    end,
                ]
        else:
            # perpendicular directions, connect with single corner
            if start_dir[0] == 0:  # vertical start
                corner = (first_segment_end[0], last_segment_start[1])
            else:  # horizontal start
                corner = (last_segment_start[0], first_segment_end[1])

            return [start, first_segment_end, corner, last_segment_start, end]

    def _create_rounded_orthogonal_path(self, points):
        """create path string for orthogonal path with rounded corners"""
        if len(points) < 3:
            return f"M {points[0][0]} {points[0][1]} L {points[-1][0]} {points[-1][1]}"

        radius = self.curve_type.corner_radius
        path = f"M {points[0][0]} {points[0][1]}"

        for i in range(1, len(points) - 1):
            prev = points[i - 1]
            curr = points[i]
            next_pt = points[i + 1]

            # check if this is a corner (90° turn)
            v1 = (curr[0] - prev[0], curr[1] - prev[1])
            v2 = (next_pt[0] - curr[0], next_pt[1] - curr[1])

            is_corner = (v1[0] == 0 and v2[0] != 0) or (v1[0] != 0 and v2[0] == 0)

            if is_corner:
                # calculate actual radius (can't exceed half of segment length)
                v1_len = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
                v2_len = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
                max_radius = min(v1_len, v2_len) / 2
                r = min(radius, max_radius)

                if r > 0:
                    # unit vectors
                    v1_norm = (v1[0] / v1_len, v1[1] / v1_len) if v1_len > 0 else (0, 0)
                    v2_norm = (v2[0] / v2_len, v2[1] / v2_len) if v2_len > 0 else (0, 0)

                    # arc points
                    arc_start = (curr[0] - v1_norm[0] * r, curr[1] - v1_norm[1] * r)
                    arc_end = (curr[0] + v2_norm[0] * r, curr[1] + v2_norm[1] * r)

                    # determine sweep flag (0 or 1) based on turn direction
                    cross_z = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
                    sweep = 0 if cross_z < 0 else 1

                    # add line to arc start then arc
                    path += f" L {arc_start[0]} {arc_start[1]}"
                    path += f" A {r} {r} 0 0 {sweep} {arc_end[0]} {arc_end[1]}"
                    continue

            # not a corner or no rounding, just add line
            path += f" L {curr[0]} {curr[1]}"

        # add final point
        path += f" L {points[-1][0]} {points[-1][1]}"
        return path

    def _add_end_caps(self, paths, start, end, control_points=None):
        """add end caps to the paths list"""
        if not (self.start_cap or self.end_cap):
            return

        # calculate direction vectors
        if isinstance(self.curve_type, StraightCurve):
            # straight line
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                direction = (dx / length, dy / length)
                start_dir = (-direction[0], -direction[1])
                end_dir = direction
            else:
                return  # can't determine direction

        elif isinstance(self.curve_type, (SimpleBezierCurve, AdvancedBezierCurve)):
            # bezier curve - use control points for direction
            if control_points:
                # start direction - from first control point
                dx1, dy1 = control_points[0][0] - start[0], control_points[0][1] - start[1]
                len1 = np.sqrt(dx1**2 + dy1**2)
                start_dir = (-dx1 / len1, -dy1 / len1) if len1 > 0 else (0, -1)

                # end direction - from last control point
                if len(control_points) > 1:
                    dx2 = end[0] - control_points[-1][0]
                    dy2 = end[1] - control_points[-1][1]
                else:
                    dx2 = end[0] - control_points[0][0]
                    dy2 = end[1] - control_points[0][1]

                len2 = np.sqrt(dx2**2 + dy2**2)
                end_dir = (dx2 / len2, dy2 / len2) if len2 > 0 else (0, 1)
            else:
                # fallback to straight line
                dx, dy = end[0] - start[0], end[1] - start[1]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    direction = (dx / length, dy / length)
                    start_dir = (-direction[0], -direction[1])
                    end_dir = direction
                else:
                    return

        elif isinstance(self.curve_type, OrthogonalCurve):
            # get from direction map
            direction_map = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
            start_dir = direction_map[self.curve_type.start_direction]
            end_dir = direction_map[self.curve_type.end_direction]
            # flip directions
            start_dir = (-start_dir[0], -start_dir[1])

        else:
            # default to straight line
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                direction = (dx / length, dy / length)
                start_dir = (-direction[0], -direction[1])
                end_dir = direction
            else:
                return

        if self.start_cap:
            if isinstance(self.start_cap, LineEndArrow):
                paths.append(create_arrow_cap(start, start_dir, self.start_cap))
            elif isinstance(self.start_cap, LineEndCircle):
                paths.append(create_circle_cap(start, self.start_cap))
            elif isinstance(self.start_cap, LineEndFlat):
                paths.append(create_flat_cap(start, start_dir, self.start_cap))

        if self.end_cap:
            if isinstance(self.end_cap, LineEndArrow):
                paths.append(create_arrow_cap(end, end_dir, self.end_cap))
            elif isinstance(self.end_cap, LineEndCircle):
                paths.append(create_circle_cap(end, self.end_cap))
            elif isinstance(self.end_cap, LineEndFlat):
                paths.append(create_flat_cap(end, end_dir, self.end_cap))

    def render(self, renderer, context, matrix: np.ndarray):
        """render connection as SVG element"""
        if not self._svg_element:
            self._create_svg_content()

        if self._svg_element:
            self._svg_element._dimensions = self._dimensions

            self._svg_element.render(renderer, context, matrix)

        if self.debug:
            renderer.render_debug(context, self, matrix)
