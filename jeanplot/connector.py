from typing import Optional, Any, List, Tuple
from pydantic import Field, BeforeValidator, PrivateAttr
from typing_extensions import Annotated
import numpy as np
import math

from .path_utils import resolve_component_ref
from .curve import (
    CurveDefinition,
    OrthogonalCurve,
    SimpleBezierCurve,
    StraightCurve,
)
from .component import Component, Overlay, AnchorComponent, get_world_origin
from .models import Size, Offset
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

ValidatedComponentRef = Annotated[Component, BeforeValidator(resolve_component_ref)]


class Connection(Overlay):
    """connects two components with a styled line/curve"""

    start_component: ValidatedComponentRef
    end_component: ValidatedComponentRef

    start_offset: Offset = Field(default_factory=lambda: Offset(relative=(0.5, 0.5)))
    end_offset: Offset = Field(default_factory=lambda: Offset(relative=(0.5, 0.5)))
    color: str = "#000000"
    line_width: float = 1.0
    curve_type: CurveDefinition = Field(default_factory=StraightCurve)
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0
    start_cap: Optional[LineEndType] = None
    end_cap: Optional[LineEndType] = None
    auto_route: bool = True

    _svg_element: Optional[SVGElement] = PrivateAttr(default=None)
    _local_start: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _local_end: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _world_start: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _world_end: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _local_checkpoints: List[Tuple[float, float]] = PrivateAttr(default_factory=list)

    def _log_debug(self, message: str, data: Any = None, matrix: Optional[np.ndarray] = None):
        """concise debug logging with connection identity"""
        comp_id = (
            self.id
            or f"Conn({getattr(self.start_component, 'id', '?')}->{getattr(self.end_component, 'id', '?')})"
        )
        if matrix is not None:
            data_str = (
                f"{data}\nMatrix:\n{np.round(matrix, 3)}"
                if data
                else f"Matrix:\n{np.round(matrix, 3)}"
            )
            debug_print(comp_id, message, data_str)
        else:
            debug_print(comp_id, message, data)

    def _transform_world_to_parent(
        self, world_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """transform point from world to parent coordinates"""
        if not self.parent:
            return world_point

        parent_matrix = self.parent.compute_world_matrix()
        try:
            parent_inv = np.linalg.inv(parent_matrix)
        except np.linalg.LinAlgError:
            self._log_debug("warning: parent matrix not invertible, using identity")
            parent_inv = np.eye(3)

        parent_coords = np.dot(parent_inv, [world_point[0], world_point[1], 1])
        return (parent_coords[0], parent_coords[1])

    def measure(self, renderer=None) -> Size:
        """calculate dimensions based on connected components positions"""
        self._log_debug("measure: starting")

        # early exit checks
        if not self.parent or not self.start_component or not self.end_component:
            self._dimensions = Size()
            self._transformed_aabb = Size()
            return self._dimensions

        # ensure components are measured
        for comp in [self.start_component, self.end_component]:
            if (
                not hasattr(comp, "_dimensions")
                or comp._dimensions.width <= 0
                or comp._dimensions.height <= 0
            ):
                comp.measure_and_layout(renderer)
            if comp._dimensions.width <= 0 or comp._dimensions.height <= 0:
                self._log_debug(f"warning: component {comp.id} has zero dimensions after measure")
                self._dimensions = Size()
                self._transformed_aabb = Size()
                return self._dimensions

        # calculate connection points
        self._world_start, self._world_end = self._calculate_world_connection_points()
        if not self._world_start or not self._world_end:
            self._dimensions = Size()
            self._transformed_aabb = Size()
            return self._dimensions

        # transform points to parent coordinates
        start_parent = self._transform_world_to_parent(self._world_start)
        end_parent = self._transform_world_to_parent(self._world_end)
        if start_parent is None or end_parent is None:
            self._dimensions = Size()
            self._transformed_aabb = Size()
            return self._dimensions

        # transform checkpoints if present
        self._local_checkpoints = []
        world_checkpoints = []
        parent_checkpoints = []
        if isinstance(self.curve_type, OrthogonalCurve) and self.curve_type.checkpoints:
            world_checkpoints = self.curve_type.checkpoints
            parent_checkpoints = [self._transform_world_to_parent(cp) for cp in world_checkpoints]
            if any(p is None for p in parent_checkpoints):
                self._log_debug("warning: failed to transform checkpoints, ignoring them")
                parent_checkpoints = []

        # calculate bounds with buffer
        points = [start_parent, end_parent] + parent_checkpoints
        try:
            min_x = min(p[0] for p in points)
            max_x = max(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_y = max(p[1] for p in points)
        except Exception as e:
            self._log_debug(f"error calculating bounds: {e}")
            self._dimensions = Size()
            self._transformed_aabb = Size()
            return self._dimensions

        # add buffer for visual padding and end caps
        buffer = max(self.line_width * 3, 10)
        min_x -= buffer
        min_y -= buffer
        max_x += buffer
        max_y += buffer

        # update dimensions and offset
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)
        self._dimensions = Size(width=width, height=height)
        self.offset = Offset(absolute=(min_x, min_y))

        # calculate local coordinates relative to offset
        self._local_start = (start_parent[0] - min_x, start_parent[1] - min_y)
        self._local_end = (end_parent[0] - min_x, end_parent[1] - min_y)
        self._local_checkpoints = [(cp[0] - min_x, cp[1] - min_y) for cp in parent_checkpoints]

        # create svg content and calculate final aabb
        self._create_svg_content()
        self._transformed_aabb = self.compute_transformed_aabb()

        self._log_debug(f"measure: complete, dims={self._dimensions}, offset={self.offset}")
        return self._dimensions

    def _get_component_offset_world_position(
        self, component: Component, offset: Offset
    ) -> Optional[Tuple[float, float]]:
        """get world position of an offset point relative to a component"""
        if not hasattr(component, "_dimensions") or component._dimensions.width <= 0:
            component.measure_and_layout()

        # compute local position using offset
        dims = component._dimensions
        if dims.width <= 0 or dims.height <= 0:
            self._log_debug(f"warning: component {component.id} has zero dimensions")
            local_pos = (0, 0)
        else:
            local_pos = offset.compute(dims, dims)

        # transform to world space
        world_matrix = component.compute_world_matrix()
        world_point = world_matrix @ np.array([local_pos[0], local_pos[1], 1])
        return (world_point[0], world_point[1])

    def _find_best_target_anchor(
        self, component: Component, reference_point_world: Tuple[float, float]
    ) -> Optional[AnchorComponent]:
        """find best anchor on component based on distance to a reference point"""
        anchors = [
            a for a in getattr(component, "anchor_points", []) if isinstance(a, AnchorComponent)
        ]
        if not anchors:
            return None

        best_anchor = None
        min_dist_sq = float("inf")

        for anchor in anchors:
            # ensure anchor is measured
            if not hasattr(anchor, "_dimensions") or anchor._dimensions.width <= 0:
                anchor.measure_and_layout()

            try:
                anchor_pos = get_world_origin(anchor)
                dist_sq = math.dist(anchor_pos, reference_point_world) ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_anchor = anchor
            except Exception:
                continue  # skip anchor if error occurs

        return best_anchor

    def _calculate_world_connection_points(
        self,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """calculate endpoints in world coordinates with auto-routing"""
        start_comp = self.start_component
        end_comp = self.end_component
        if not start_comp or not end_comp:
            return None, None

        # ensure components are measured
        for comp in [start_comp, end_comp]:
            if not hasattr(comp, "_dimensions") or comp._dimensions.width <= 0:
                comp.measure_and_layout()

        # default targets are the components with specified offsets
        start_target, end_target = start_comp, end_comp
        start_offset, end_offset = self.start_offset, self.end_offset

        # get initial world positions for anchor finding
        start_world_initial = self._get_component_offset_world_position(start_target, start_offset)
        end_world_initial = self._get_component_offset_world_position(end_target, end_offset)
        if not start_world_initial or not end_world_initial:
            return None, None

        def update_curve_from_anchor(anchor: AnchorComponent, is_start: bool):
            """update curve parameters based on anchor properties"""
            prefix = "start" if is_start else "end"
            anchor_dir = getattr(anchor, "direction", (0, 1))
            anchor_len = getattr(anchor, "min_segment", 10.0)

            if isinstance(self.curve_type, OrthogonalCurve):
                setattr(
                    self.curve_type,
                    f"{prefix}_direction",
                    OrthogonalCurve.get_direction(anchor_dir),
                )
                setattr(self.curve_type, f"{prefix}_length", anchor_len)
            elif isinstance(self.curve_type, SimpleBezierCurve):
                setattr(
                    self.curve_type,
                    f"{prefix}_vector",
                    (anchor_dir[0] * anchor_len, anchor_dir[1] * anchor_len),
                )

        # find best anchors if auto-routing is enabled
        if self.auto_route:
            # find best anchor on end component
            best_end_anchor = self._find_best_target_anchor(end_comp, start_world_initial)
            if best_end_anchor:
                end_target = best_end_anchor
                end_offset = Offset()  # use anchor's origin
                update_curve_from_anchor(best_end_anchor, False)

            # find best anchor on start component
            ref_point = (
                self._get_component_offset_world_position(end_target, end_offset)
                or end_world_initial
            )
            best_start_anchor = self._find_best_target_anchor(start_comp, ref_point)
            if best_start_anchor:
                start_target = best_start_anchor
                start_offset = Offset()  # use anchor's origin
                update_curve_from_anchor(best_start_anchor, True)

        # calculate final world positions
        start_world = self._get_component_offset_world_position(start_target, start_offset)
        end_world = self._get_component_offset_world_position(end_target, end_offset)

        return start_world, end_world

    def render(self, renderer, context, matrix: np.ndarray):
        """render connection as SVG element"""
        if not self.show or not self.start_component or not self.end_component:
            return

        # calculate current target world points
        start_world_target, end_world_target = self._calculate_world_connection_points()
        if not start_world_target or not end_world_target:
            return

        # check if geometry update is needed
        tolerance = 1e-4
        needs_update = (
            not self._world_start
            or not self._world_end
            or math.dist(start_world_target, self._world_start) > tolerance
            or math.dist(end_world_target, self._world_end) > tolerance
            or not hasattr(self, "_dimensions")
            or self._dimensions.width <= 0
        )

        # update geometry if needed
        if needs_update:
            self._world_start, self._world_end = start_world_target, end_world_target
            if self.parent:
                start_parent = self._transform_world_to_parent(start_world_target)
                end_parent = self._transform_world_to_parent(end_world_target)
                if start_parent is not None and end_parent is not None:
                    self._update_connection_geometry(start_parent, end_parent)
                else:
                    return  # can't proceed without valid transforms
            else:
                return  # can't proceed without parent

        # render SVG element
        if self._svg_element:
            self._svg_element.render(renderer, context, matrix)

        # render debug visualization if enabled
        if self.debug:
            # verify point mapping in debug mode
            if self._local_start is not None and self._world_start is not None:
                self._debug_check_transform(matrix)

            # render debug box
            if hasattr(self, "_dimensions") and self._dimensions.width > 0:
                renderer.render_debug(context, self, matrix)

    def _debug_check_transform(self, matrix: np.ndarray):
        """verify world/local point transformations in debug mode"""
        local_start_h = np.array([self._local_start[0], self._local_start[1], 1])
        transformed_start = (matrix @ local_start_h)[:2]
        start_match = np.allclose(transformed_start, self._world_start, atol=1e-5)

        local_end_h = np.array([self._local_end[0], self._local_end[1], 1])
        transformed_end = (matrix @ local_end_h)[:2]
        end_match = np.allclose(transformed_end, self._world_end, atol=1e-5)

        print(f"--- Debug Connection {self.id or 'unnamed'} ---")
        parent_matrix = self.parent.compute_world_matrix() if self.parent else np.identity(3)
        print(f"Parent World Matrix:\n{np.round(parent_matrix, 3)}")
        print(f"Connection Offset: {self.offset}")
        print(f"Connection Dimensions: {self._dimensions}")
        try:
            local_mat = self.compute_local_matrix()
            print(f"Connection Local Matrix:\n{np.round(local_mat, 3)}")
        except Exception as e:
            print(f"Connection Local Matrix: Error - {e}")

        print(f"Connection World Matrix:\n{np.round(matrix, 3)}")
        print(f"World Start: {np.round(self._world_start, 3)}")
        print(f"World End: {np.round(self._world_end, 3)}")
        print(f"Local Start: {np.round(self._local_start, 3)}")
        print(f"Local End: {np.round(self._local_end, 3)}")
        print(f"Transformed Start: {np.round(transformed_start, 3)}")
        print(f"Transformed End: {np.round(transformed_end, 3)}")
        print(
            f"Start Match: {start_match} (Diff: {np.linalg.norm(transformed_start - self._world_start):.4e})"
        )
        print(
            f"End Match: {end_match} (Diff: {np.linalg.norm(transformed_end - self._world_end):.4e})"
        )

    def _update_connection_geometry(
        self, start_parent: Tuple[float, float], end_parent: Tuple[float, float]
    ):
        """update connection dimensions, offset and svg content"""
        # recalculate local checkpoints if present
        parent_checkpoints = []
        if isinstance(self.curve_type, OrthogonalCurve) and self.curve_type.checkpoints:
            world_checkpoints = self.curve_type.checkpoints
            parent_checkpoints_temp = [
                self._transform_world_to_parent(cp) for cp in world_checkpoints
            ]
            if not any(p is None for p in parent_checkpoints_temp):
                parent_checkpoints = parent_checkpoints_temp

        # calculate bounds with all points
        points = [start_parent, end_parent] + parent_checkpoints
        try:
            min_x = min(p[0] for p in points)
            max_x = max(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_y = max(p[1] for p in points)
        except Exception as e:
            self._log_debug(f"error calculating bounds: {e}")
            return

        # add buffer and calculate dimensions
        buffer = max(self.line_width * 3, 10)
        min_x -= buffer
        min_y -= buffer
        max_x += buffer
        max_y += buffer

        # update dimensions and offset
        self._dimensions = Size(width=max(max_x - min_x, 1.0), height=max(max_y - min_y, 1.0))
        self.offset = Offset(absolute=(min_x, min_y))

        # calculate local coordinates relative to offset
        self._local_start = (start_parent[0] - min_x, start_parent[1] - min_y)
        self._local_end = (end_parent[0] - min_x, end_parent[1] - min_y)
        self._local_checkpoints = [(cp[0] - min_x, cp[1] - min_y) for cp in parent_checkpoints]

        # create svg content with updated coordinates
        self._create_svg_content()

    def _create_svg_content(self):
        """create SVG content based on connection points and curve type"""
        if not self._local_start or not self._local_end or not hasattr(self, "_dimensions"):
            self._svg_element = None
            return

        width = max(self._dimensions.width, 1.0)
        height = max(self._dimensions.height, 1.0)

        # generate path from curve type
        try:
            path_str, control_points = self.curve_type.get_path(
                self._local_start, self._local_end, self._local_checkpoints
            )
        except Exception as e:
            self._log_debug(f"error getting path: {e}")
            self._svg_element = None
            return

        # create svg path
        path_data = SVGPathData(
            d=path_str,
            stroke=self.color,
            stroke_width=self.line_width,
            fill="none",
            line_style=self.line_style,
            dash_array=self.dash_array,
            dash_offset=self.dash_offset,
        )

        # add end caps if needed
        paths = [path_data]
        if self.start_cap or self.end_cap:
            self._add_end_caps(paths, self._local_start, self._local_end, control_points)

        # create svg content with viewBox
        svg_content = SVGContent(
            width=width,
            height=height,
            viewBox=(0, 0, width, height),
            paths=paths,
        )

        # create svg element and set dimensions
        self._svg_element = SVGElement(svg_content=svg_content, id=f"svg_{self.id or 'conn'}")
        self._svg_element._dimensions = Size(width=width, height=height)

    def _add_end_caps(
        self,
        paths: List[SVGPathData],
        start: Tuple[float, float],
        end: Tuple[float, float],
        control_points: Optional[List[Tuple[float, float]]] = None,
    ):
        """add end caps to the paths list"""
        if not (self.start_cap or self.end_cap):
            return

        # get directions from curve type
        try:
            start_dir, end_dir = self.curve_type.get_directions(start, end, control_points or [])
        except Exception:
            return  # skip caps if directions can't be determined

        # map cap types to creation functions
        cap_creators = {
            LineEndArrow: create_arrow_cap,
            LineEndCircle: create_circle_cap,
            LineEndFlat: create_flat_cap,
        }

        # add start cap if specified
        if self.start_cap:
            for cap_type, creator in cap_creators.items():
                if isinstance(self.start_cap, cap_type):
                    try:
                        paths.append(creator(start, start_dir, self.start_cap))
                    except Exception:
                        pass  # skip if creation fails
                    break

        # add end cap if specified
        if self.end_cap:
            for cap_type, creator in cap_creators.items():
                if isinstance(self.end_cap, cap_type):
                    try:
                        paths.append(creator(end, end_dir, self.end_cap))
                    except Exception:
                        pass  # skip if creation fails
                    break

    def measure_and_layout(self, renderer=None) -> Size:
        """measure and layout the connection"""
        # apply styles if parent exists
        if hasattr(self, "parent") and self.parent:
            from .style import jstyle

            jstyle.apply(self)
            self._resolve_attachment()

        # measure calculates dimensions, offset, and svg content
        measured_size = self.measure(renderer)

        # apply layout (no-op for Connection)
        self.apply_layout()

        return measured_size
