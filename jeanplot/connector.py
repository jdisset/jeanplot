from typing import Optional, Literal, Union, Dict, Any, List, Tuple
from pydantic import Field, PrivateAttr, BaseModel
import numpy as np

from .container import Overlay
from .component import Component
from .models import Size, Offset, Transform
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
from .debug import debug_print


class CurveDefinition(BaseModel):
    pass


class StraightCurve(CurveDefinition):
    pass


class SimpleBezierCurve(CurveDefinition):
    """bezier curve with vectors from start and end points"""

    start_vector: tuple[float, float]  # vector from start point
    end_vector: tuple[float, float]  # vector from end point


class AdvancedBezierCurve(CurveDefinition):
    """bezier curve with explicit control points"""

    control_points: list[tuple[float, float]]


class OrthogonalCurve(CurveDefinition):
    """path with orthogonal segments (right angles)"""

    start_direction: Literal["up", "down", "left", "right"] = "down"
    start_length: float = 5.0  # minimum length of start segment
    end_direction: Literal["up", "down", "left", "right"] = "down"
    end_length: float = 5.0  # minimum length of end segment
    corner_radius: float = 10.0
    checkpoints: List[Tuple[float, float]] = Field(default_factory=list)  # now in world coordinates
    auto_simplify: bool = True  # try to reduce turns when possible


class Connection(Overlay):
    """connects two components with a styled line/curve"""

    start_component: Component
    end_component: Component
    start_offset: Offset = Field(default_factory=lambda: Offset(relative=(0.5, 0.5)))
    end_offset: Offset = Field(default_factory=lambda: Offset(relative=(0.5, 0.5)))
    color: str = "#000000"
    line_width: float = 1.0
    curve_type: CurveDefinition = Field(default_factory=StraightCurve)
    line_style: LineStyle = "solid"
    dash_array: Optional[tuple[float, ...]] = None
    dash_offset: float = 0.0
    start_cap: Optional[LineEndType] = None
    end_cap: Optional[LineEndType] = None

    _svg_element: Optional[SVGElement] = PrivateAttr(default=None)
    _local_start: Optional[tuple[float, float]] = PrivateAttr(default=None)
    _local_end: Optional[tuple[float, float]] = PrivateAttr(default=None)
    _world_start: Optional[tuple[float, float]] = PrivateAttr(default=None)
    _world_end: Optional[tuple[float, float]] = PrivateAttr(default=None)
    _local_checkpoints: List[Tuple[float, float]] = PrivateAttr(default_factory=list)

    def _log_debug(self, message: str, data: Any = None):
        """log debug messages with connection id"""
        debug_print(self.id or "Connection", message, data)

    def measure(self, renderer=None) -> Size:
        """calculate dimensions based on connected components positions"""
        self._log_debug("Measuring connection")

        # need parent container to position properly
        if self.parent is None:
            self._log_debug("No parent for connection, skipping measurement")
            self._dimensions = Size()
            self._transformed_aabb = Size()
            return self._dimensions

        # Check if both components exist and have dimensions
        if not hasattr(self.start_component, "_dimensions") or not hasattr(
            self.end_component, "_dimensions"
        ):
            self._log_debug(
                "Components missing dimensions",
                {
                    "start_has_dims": hasattr(self.start_component, "_dimensions"),
                    "end_has_dims": hasattr(self.end_component, "_dimensions"),
                },
            )
            self._dimensions = Size()
            self._transformed_aabb = Size()
            return self._dimensions

        # get world coordinates of connection points
        self._world_start, self._world_end = self._calculate_world_connection_points()
        self._log_debug(
            "World connection points", {"start": self._world_start, "end": self._world_end}
        )

        # transform to parent's coordinate system
        parent_matrix = self.parent.compute_world_matrix()
        self._log_debug("Parent matrix", parent_matrix)

        # avoid singular matrix - use identity if needed
        try:
            parent_inv = np.linalg.inv(parent_matrix)
        except np.linalg.LinAlgError:
            self._log_debug("Singular parent matrix, using identity")
            parent_inv = np.eye(3)

        # transform world points to parent space
        start_parent_coords = np.dot(parent_inv, [self._world_start[0], self._world_start[1], 1])
        end_parent_coords = np.dot(parent_inv, [self._world_end[0], self._world_end[1], 1])

        start_parent = (start_parent_coords[0], start_parent_coords[1])
        end_parent = (end_parent_coords[0], end_parent_coords[1])

        self._log_debug("Parent space points", {"start": start_parent, "end": end_parent})

        # Transform checkpoints from world to parent space if using OrthogonalCurve with checkpoints
        self._local_checkpoints = []
        if isinstance(self.curve_type, OrthogonalCurve) and self.curve_type.checkpoints:
            world_checkpoints = self.curve_type.checkpoints
            self._log_debug("World checkpoints", {"checkpoints": world_checkpoints})

            # Transform each checkpoint to parent space
            for cp_world in world_checkpoints:
                cp_parent_coords = np.dot(parent_inv, [cp_world[0], cp_world[1], 1])
                self._local_checkpoints.append((cp_parent_coords[0], cp_parent_coords[1]))

            self._log_debug("Parent space checkpoints", {"checkpoints": self._local_checkpoints})

        # calculate bounds with buffer
        min_x = min(start_parent[0], end_parent[0])
        max_x = max(start_parent[0], end_parent[0])
        min_y = min(start_parent[1], end_parent[1])
        max_y = max(start_parent[1], end_parent[1])

        # Include checkpoints in bounds calculation
        for cp in self._local_checkpoints:
            min_x = min(min_x, cp[0])
            max_x = max(max_x, cp[0])
            min_y = min(min_y, cp[1])
            max_y = max(max_y, cp[1])

        # add buffer for line width and control points
        buffer = max(self.line_width * 3, 20)
        min_x -= buffer
        min_y -= buffer
        max_x += buffer
        max_y += buffer

        # ensure positive dimensions
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)

        self._log_debug(
            "Connection dimensions", {"width": width, "height": height, "position": (min_x, min_y)}
        )

        # set dimensions and position
        self._dimensions = Size(width=width, height=height)
        self.transform.translate = (min_x, min_y)

        # convert world points to local coordinates
        self._local_start = (start_parent[0] - min_x, start_parent[1] - min_y)
        self._local_end = (end_parent[0] - min_x, end_parent[1] - min_y)

        # also convert checkpoint positions to component-local coordinates
        self._local_checkpoints = [(cp[0] - min_x, cp[1] - min_y) for cp in self._local_checkpoints]

        self._log_debug(
            "Local points",
            {
                "start": self._local_start,
                "end": self._local_end,
                "checkpoints": self._local_checkpoints,
            },
        )

        # create SVG content
        self._create_svg_content()

        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def _get_component_world_position(self, component, offset):
        """get world position using component's world matrix"""
        if component is None:
            return (0, 0)

        # calculate offset point in local component space
        dims = getattr(component, "_dimensions", Size(width=1, height=1))
        ox, oy = offset.compute(dims)
        local_point = np.array([ox, oy, 1])

        # transform to world space
        world_matrix = component.compute_world_matrix()
        world_point = world_matrix @ local_point
        pos = (world_point[0], world_point[1])

        self._log_debug(
            f"World position for {component.id}", {"offset": (ox, oy), "world_point": pos}
        )
        return pos

    def _calculate_world_connection_points(self):
        """calculate connection points in world coordinates"""
        start_world = self._get_component_world_position(self.start_component, self.start_offset)
        end_world = self._get_component_world_position(self.end_component, self.end_offset)

        if start_world == end_world:
            self._log_debug("Warning: Identical start and end points", {"point": start_world})

        return start_world, end_world

    def _create_svg_content(self):
        """create SVG content based on connection points and curve type"""
        if not self._local_start or not self._local_end:
            self._log_debug("Missing local points for SVG creation")
            return

        start = self._local_start
        end = self._local_end
        width = max(self._dimensions.width, 1.0)
        height = max(self._dimensions.height, 1.0)

        # create path based on curve type
        path_str = ""
        control_points = []

        if isinstance(self.curve_type, StraightCurve):
            path_str = f"M {start[0]} {start[1]} L {end[0]} {end[1]}"
        elif isinstance(self.curve_type, SimpleBezierCurve):
            # calculate control points from vectors
            start_vec = self.curve_type.start_vector
            end_vec = self.curve_type.end_vector
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
            # calculate orthogonal path with minimal turns
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
            stroke_width=self.line_width,
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
        """Calculate optimal orthogonal path with minimal turns"""
        direction_map = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        start_dir = direction_map[self.curve_type.start_direction]
        end_dir = direction_map[self.curve_type.end_direction]
        end_dir_inv = (-end_dir[0], -end_dir[1])  # invert for calculation

        if self._local_checkpoints:
            return self._path_with_checkpoints(start, end, start_dir, end_dir_inv)

        # try a simple path first if auto-simplify is enabled
        if getattr(self.curve_type, "auto_simplify", True):
            simple_path = self._find_simple_path(start, end, start_dir, end_dir_inv)
            if simple_path:
                return simple_path

        start_len = self.curve_type.start_length
        end_len = self.curve_type.end_length

        first_segment_end = (
            start[0] + start_dir[0] * start_len,
            start[1] + start_dir[1] * start_len,
        )

        last_segment_start = (
            end[0] + end_dir_inv[0] * end_len,
            end[1] + end_dir_inv[1] * end_len,
        )

        # Connect with minimum possible turns
        return self._connect_segments(start, first_segment_end, last_segment_start, end)

    def _find_simple_path(self, start, end, start_dir, end_dir):
        """Try to find a path with minimal turns (0 or 1)"""
        # check if points are aligned for direct path
        if start[0] == end[0] or start[1] == end[1]:
            # points are aligned; check if directions allow direct connection
            if self._can_connect_directly(start, end, start_dir, end_dir):
                return [start, end]

        # try one-turn path
        corner = self._find_corner(start, end, start_dir, end_dir)
        if corner:
            return [start, corner, end]

        # no simple path possible
        return None

    def _can_connect_directly(self, p1, p2, dir1, dir2):
        """Check if two points can be connected with a straight line"""
        if p1[0] == p2[0]:  # vertical alignment
            if dir1[0] != 0 or dir2[0] != 0:  # directions must be vertical
                return False
            if (p2[1] - p1[1]) * dir1[1] <= 0:
                return False
            if (p1[1] - p2[1]) * dir2[1] <= 0:
                return False
            return True

        elif p1[1] == p2[1]:  # horizontal alignment
            if dir1[1] != 0 or dir2[1] != 0:  # directions must be horizontal
                return False
            if (p2[0] - p1[0]) * dir1[0] <= 0:
                return False
            if (p1[0] - p2[0]) * dir2[0] <= 0:
                return False
            return True

        return False  # points not aligned

    def _find_corner(self, start, end, start_dir, end_dir):
        """Find corner point for a one-turn path if possible"""
        # two possible corners:
        corners = [
            (
                start[0] + start_dir[0] * max(abs(end[0] - start[0]), self.curve_type.start_length),
                end[1] + end_dir[1] * max(abs(end[1] - start[1]), self.curve_type.end_length),
            ),
            (
                end[0] + end_dir[0] * max(abs(end[0] - start[0]), self.curve_type.end_length),
                start[1] + start_dir[1] * max(abs(end[1] - start[1]), self.curve_type.start_length),
            ),
        ]

        for corner in corners:
            # check if both components can reach corner in their respective directions
            can_reach = True

            if start_dir[0] != 0:  # horizontal start direction
                if (corner[0] - start[0]) * start_dir[0] <= 0:
                    can_reach = False
            else:  # vertical start direction
                if (corner[1] - start[1]) * start_dir[1] <= 0:
                    can_reach = False

            if end_dir[0] != 0:  # horizontal end direction
                if (end[0] - corner[0]) * end_dir[0] <= 0:
                    can_reach = False
            else:  # vertical end direction
                if (end[1] - corner[1]) * end_dir[1] <= 0:
                    can_reach = False

            if can_reach:
                # found valid corner
                return corner

        return None  # no valid corner

    def _connect_segments(self, start, first_end, last_start, end):
        """Connect points with minimal number of turns"""
        # check if middle points are aligned (can connect directly)
        if first_end[0] == last_start[0] or first_end[1] == last_start[1]:
            return [start, first_end, last_start, end]

        # try single turn between middle points
        corner1 = (first_end[0], last_start[1])
        corner2 = (last_start[0], first_end[1])

        # choose the first valid corner (both should be valid in this context)
        return [start, first_end, corner1, last_start, end]

    def _path_with_checkpoints(self, start, end, start_dir, end_dir_inv):
        """Create path from start to end through checkpoints"""
        # use local checkpoints that have been transformed from world space
        checkpoints = self._local_checkpoints

        start_len = self.curve_type.start_length
        end_len = self.curve_type.end_length

        first_segment_end = (
            start[0] + start_dir[0] * start_len,
            start[1] + start_dir[1] * start_len,
        )

        last_segment_start = (
            end[0] + end_dir_inv[0] * end_len,
            end[1] + end_dir_inv[1] * end_len,
        )

        path = [start, first_segment_end]

        # route through each checkpoint
        current = first_segment_end
        for cp in checkpoints:
            segments = self._connect_points(current, cp)
            path.extend(segments[1:])  # Skip first point (already in path)
            current = cp

        # connect last checkpoint to end segment
        segments = self._connect_points(current, last_segment_start)
        path.extend(segments[1:])

        path.append(end)

        return path

    def _connect_points(self, p1, p2):
        """Connect two points with minimal orthogonal path"""
        # if points are aligned, connect directly
        if p1[0] == p2[0] or p1[1] == p2[1]:
            return [p1, p2]

        corner = (p1[0], p2[1])

        return [p1, corner, p2]

    def _create_rounded_orthogonal_path(self, points):
        """Create path string with rounded corners"""
        if len(points) < 3:
            return f"M {points[0][0]} {points[0][1]} L {points[-1][0]} {points[-1][1]}"

        radius = self.curve_type.corner_radius
        path = f"M {points[0][0]} {points[0][1]}"

        for i in range(1, len(points) - 1):
            prev = points[i - 1]
            curr = points[i]
            next_pt = points[i + 1]

            # check if this is actually a corner (direction changes)
            v1 = (curr[0] - prev[0], curr[1] - prev[1])
            v2 = (next_pt[0] - curr[0], next_pt[1] - curr[1])

            # only consider actual corners (where direction changes)
            is_corner = (v1[0] == 0 and v2[0] != 0) or (v1[0] != 0 and v2[0] == 0)

            if is_corner:
                # calculate actual radius (can't exceed half of segment length)
                v1_len = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
                v2_len = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
                max_radius = min(v1_len, v2_len) / 2
                r = min(radius, max_radius)

                if r > 0:
                    v1_norm = (v1[0] / v1_len, v1[1] / v1_len) if v1_len > 0 else (0, 0)
                    v2_norm = (v2[0] / v2_len, v2[1] / v2_len) if v2_len > 0 else (0, 0)

                    arc_start = (curr[0] - v1_norm[0] * r, curr[1] - v1_norm[1] * r)
                    arc_end = (curr[0] + v2_norm[0] * r, curr[1] + v2_norm[1] * r)

                    # determine sweep flag based on turn direction
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
            # flip start direction
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

        # add start cap
        if self.start_cap:
            if isinstance(self.start_cap, LineEndArrow):
                paths.append(create_arrow_cap(start, start_dir, self.start_cap))
            elif isinstance(self.start_cap, LineEndCircle):
                paths.append(create_circle_cap(start, self.start_cap))
            elif isinstance(self.start_cap, LineEndFlat):
                paths.append(create_flat_cap(start, start_dir, self.start_cap))

        # add end cap
        if self.end_cap:
            if isinstance(self.end_cap, LineEndArrow):
                paths.append(create_arrow_cap(end, end_dir, self.end_cap))
            elif isinstance(self.end_cap, LineEndCircle):
                paths.append(create_circle_cap(end, self.end_cap))
            elif isinstance(self.end_cap, LineEndFlat):
                paths.append(create_flat_cap(end, end_dir, self.end_cap))

    def render(self, renderer, context, matrix: np.ndarray):
        """render connection as SVG element"""
        self._log_debug("Rendering connection")

        # recalculate world coordinates to get latest positions
        start_world, end_world = self._calculate_world_connection_points()

        positions_changed = start_world != getattr(
            self, "_last_start_world", None
        ) or end_world != getattr(self, "_last_end_world", None)

        if positions_changed:
            self._log_debug("World points changed, updating connection")
            self._last_start_world = start_world
            self._last_end_world = end_world

            # transform world points to scene space
            if self.parent:
                parent_matrix = self.parent.compute_world_matrix()
                try:
                    parent_inv = np.linalg.inv(parent_matrix)
                except np.linalg.LinAlgError:
                    parent_inv = np.eye(3)

                # transform world points to parent space
                start_parent_coords = np.dot(parent_inv, [start_world[0], start_world[1], 1])
                end_parent_coords = np.dot(parent_inv, [end_world[0], end_world[1], 1])
                start_parent = (start_parent_coords[0], start_parent_coords[1])
                end_parent = (end_parent_coords[0], end_parent_coords[1])

                # transform world checkpoints to parent space
                self._local_checkpoints = []
                if isinstance(self.curve_type, OrthogonalCurve) and self.curve_type.checkpoints:
                    for cp_world in self.curve_type.checkpoints:
                        cp_parent_coords = np.dot(parent_inv, [cp_world[0], cp_world[1], 1])
                        self._local_checkpoints.append((cp_parent_coords[0], cp_parent_coords[1]))

                self._update_connection_geometry(start_parent, end_parent)

        if self._svg_element:
            self._svg_element._dimensions = self._dimensions
            self._svg_element.render(renderer, context, matrix)
        else:
            self._log_debug("No SVG element to render")

        if self.debug:
            renderer.render_debug(context, self, matrix)
            # also render connection points when debugging
            if hasattr(self, "_local_start") and hasattr(self, "_local_end") and renderer:
                try:
                    # draw marker at connection points for debugging
                    import matplotlib.pyplot as plt
                    import matplotlib.transforms as mtransforms

                    transform = mtransforms.Affine2D(matrix=matrix) + context.transData

                    context.plot(
                        [self._local_start[0]],
                        [self._local_start[1]],
                        "x",
                        markersize=8,
                        transform=transform,
                        color="red",
                    )
                    context.text(
                        self._local_start[0] + 3,
                        self._local_start[1] + 3,
                        "start",
                        color="red",
                        fontsize=8,
                        transform=transform,
                    )

                    context.plot(
                        [self._local_end[0]],
                        [self._local_end[1]],
                        "x",
                        color="red",
                        markersize=8,
                        transform=transform,
                    )
                    context.text(
                        self._local_end[0] + 3,
                        self._local_end[1] + 3,
                        "end",
                        color="red",
                        fontsize=8,
                        transform=transform,
                    )

                    # also draw checkpoints when debugging
                    for i, cp in enumerate(self._local_checkpoints):
                        context.plot(
                            [cp[0]],
                            [cp[1]],
                            "go",
                            markersize=6,
                            transform=transform,
                            color="red",
                        )
                        context.text(
                            cp[0] + 3,
                            cp[1] + 3,
                            f"CP{i}",
                            color="red",
                            fontsize=8,
                            transform=transform,
                        )

                    context.plot(
                        [self._local_start[0], self._local_end[0]],
                        [self._local_start[1], self._local_end[1]],
                        "g--",
                        linewidth=1,
                        transform=transform,
                    )
                except Exception as e:
                    self._log_debug(f"Error rendering debug points: {e}")

    def _update_connection_geometry(self, start_parent, end_parent):
        """update connection geometry based on parent-space points"""
        min_x = min(start_parent[0], end_parent[0])
        max_x = max(start_parent[0], end_parent[0])
        min_y = min(start_parent[1], end_parent[1])
        max_y = max(start_parent[1], end_parent[1])

        # include checkpoints in bounds calculation
        for cp in self._local_checkpoints:
            min_x = min(min_x, cp[0])
            max_x = max(max_x, cp[0])
            min_y = min(min_y, cp[1])
            max_y = max(max_y, cp[1])

        # add buffer
        buffer = max(self.line_width * 3, 20)
        min_x -= buffer
        min_y -= buffer
        max_x += buffer
        max_y += buffer

        # ensure positive dimensions
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)

        # update dimensions and position
        self._dimensions = Size(width=width, height=height)
        self.transform.translate = (min_x, min_y)

        # update local points (relative to component origin)
        self._local_start = (start_parent[0] - min_x, start_parent[1] - min_y)
        self._local_end = (end_parent[0] - min_x, end_parent[1] - min_y)

        # update local checkpoints (relative to component origin)
        self._local_checkpoints = [(cp[0] - min_x, cp[1] - min_y) for cp in self._local_checkpoints]

        self._create_svg_content()
