"""Component for drawing connections (lines/curves) between other components."""

from typing import Optional, Any, List, Tuple, Union
from pydantic import Field, PrivateAttr, model_validator
import numpy as np
import math
import copy
import logging

from jeanplot.path_utils import find_component_by_path
from jeanplot.curve import (
    CurveDefinition,
    OrthogonalCurve,
    SimpleBezierCurve,
    StraightCurve,
)
from jeanplot.component import Component, Overlay, AnchorComponent
from jeanplot.models import Size, Offset
from jeanplot.svg import LineEndType, LineStyle
from jeanplot.debug import debug_print
from jeanplot.style import jstyle

logger = logging.getLogger(__name__)


class Connection(Overlay):
    """connects two components with a styled line/curve."""

    start_component: Union[str, Component]
    end_component: Union[str, Component]
    # offset from the resolved start component's origin
    start_offset: Offset = Field(default_factory=lambda: Offset(reference_relative=(0.5, 0.5)))
    # offset from the resolved end component's origin
    end_offset: Offset = Field(default_factory=lambda: Offset(reference_relative=(0.5, 0.5)))

    color: str = "#000000"
    line_width: float = 1.0
    curve_type: CurveDefinition = Field(default_factory=StraightCurve)
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0
    start_cap: Optional[LineEndType] = None
    end_cap: Optional[LineEndType] = None
    # enable automatic selection of best anchor points if available
    auto_route: bool = True

    # --- Internal State ---
    # resolved component references
    _resolved_start_component: Optional[Component] = PrivateAttr(default=None)
    _resolved_end_component: Optional[Component] = PrivateAttr(default=None)
    # cached points from last calculation (relative to connection's own origin)
    _local_start: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _local_end: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _local_control_points: List[Tuple[float, float]] = PrivateAttr(default_factory=list)

    def _get_active_curve(self) -> CurveDefinition:
        """
        returns a copy of the currently styled curve instance to use for calculations.
        this ensures calculations start with the parameters set by jstyle.
        """
        # always return a fresh copy of the current curve_type state
        return copy.deepcopy(self.curve_type)

    def _resolve_references(self):
        """resolve string paths for start/end_component."""
        resolved_start = self._resolved_start_component
        resolved_end = self._resolved_end_component

        if isinstance(self.start_component, str) and resolved_start is None:
            resolved_start = self._resolve_single_ref(self.start_component, "start")
        elif isinstance(self.start_component, Component):
            resolved_start = self.start_component

        if isinstance(self.end_component, str) and resolved_end is None:
            resolved_end = self._resolve_single_ref(self.end_component, "end")
        elif isinstance(self.end_component, Component):
            resolved_end = self.end_component

        self._resolved_start_component = resolved_start
        self._resolved_end_component = resolved_end

    def _resolve_single_ref(self, path: str, which: str) -> Optional[Component]:
        """helper to resolve a single string reference."""
        if not self.parent:
            self._log_debug(f"cannot resolve {which} path '{path}' without parent.")
            return None
        try:
            root = self.parent
            while root.parent is not None:
                root = root.parent
            resolved = find_component_by_path(root, path)
            if resolved:
                self._log_debug(f"resolved {which} path '{path}' to {resolved.id}")
                return resolved
            else:  # find_component_by_path raises error if not found
                return None
        except ValueError as e:
            self._log_debug(f"error resolving {which} path '{path}': {e}")
            return None

    def _find_best_anchor(
        self, component: Component, reference_point_world: Tuple[float, float]
    ) -> Optional[AnchorComponent]:
        """find best anchor on component closest to a world reference point."""
        anchors = [
            a for a in getattr(component, "anchor_points", []) if isinstance(a, AnchorComponent)
        ]
        if not anchors:
            return None

        best_anchor, min_dist_sq = None, float("inf")
        for anchor in anchors:
            try:
                anchor_pos = anchor.get_world_origin()
                dist_sq = math.dist(anchor_pos, reference_point_world) ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_anchor = anchor
            except Exception as e:
                self._log_debug(f"error getting origin for anchor {anchor.id}: {e}")
        return best_anchor

    def _update_curve_params_from_anchor(
        self, anchor: Component, is_start: bool, curve_instance: CurveDefinition
    ):
        """
        updates the GIVEN curve instance's parameters based on anchor properties.
        modifies the curve_instance in place.
        """
        # check if it's an anchor and has direction/min_segment
        if not isinstance(anchor, AnchorComponent) or anchor.direction is None:
            return

        prefix = "start" if is_start else "end"
        anchor_dir = anchor.direction
        anchor_len = anchor.min_segment

        self._log_debug(
            f"updating curve from anchor '{anchor.id}': dir={anchor_dir}, len={anchor_len} for side '{prefix}'"
        )

        if isinstance(curve_instance, OrthogonalCurve):
            ortho_dir_name = OrthogonalCurve.get_direction_from_vector(anchor_dir)
            self._log_debug(
                f"  converted anchor vector {anchor_dir} to ortho direction '{ortho_dir_name}'"
            )
            if getattr(curve_instance, f"{prefix}_direction") != ortho_dir_name:
                setattr(curve_instance, f"{prefix}_direction", ortho_dir_name)
            # --------------------------------------------
            if getattr(curve_instance, f"{prefix}_length") != anchor_len:
                setattr(curve_instance, f"{prefix}_length", anchor_len)

        elif isinstance(curve_instance, SimpleBezierCurve):
            vector = (anchor_dir[0] * anchor_len, anchor_dir[1] * anchor_len)
            if getattr(curve_instance, f"{prefix}_mode") != "vector":
                setattr(curve_instance, f"{prefix}_mode", "vector")
            if getattr(curve_instance, f"{prefix}_vector") != vector:
                setattr(curve_instance, f"{prefix}_vector", vector)

    def _get_world_connection_points(
        self,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], CurveDefinition]]:
        """
        calculates connection world points, handling auto-routing and anchor updates.
        returns (world_start, world_end, active_curve_instance) or None on failure.
        the returned curve instance contains the final parameters used.
        """
        self._resolve_references()  # ensure components are resolved first
        start_comp_orig = self._resolved_start_component
        end_comp_orig = self._resolved_end_component

        if not start_comp_orig or not end_comp_orig:
            self._log_debug("cannot calculate points: missing resolved components.")
            return None

        # get a curve instance based on current style - this is the starting point
        active_curve = self._get_active_curve()

        # ensure target components have dimensions (needed for offset calculation)
        for comp in [start_comp_orig, end_comp_orig]:
            if comp._dimensions.width <= 0 or comp._dimensions.height <= 0:
                self._log_debug(
                    f"warning: target component {comp.id} has zero size for connection calc."
                )

        start_target, end_target = start_comp_orig, end_comp_orig
        start_offset, end_offset = self.start_offset, self.end_offset

        if self.auto_route:
            world_start_initial = self._get_offset_world_position(start_target, start_offset)
            world_end_initial = self._get_offset_world_position(end_target, end_offset)

            if world_start_initial and world_end_initial:
                best_end_anchor = self._find_best_anchor(end_comp_orig, world_start_initial)
                if best_end_anchor:
                    end_target = best_end_anchor
                    end_offset = Offset()

                ref_point_for_start = (
                    self._get_offset_world_position(end_target, end_offset) or world_end_initial
                )
                best_start_anchor = self._find_best_anchor(start_comp_orig, ref_point_for_start)
                if best_start_anchor:
                    start_target = best_start_anchor
                    start_offset = Offset()
            else:
                self._log_debug("skipping anchor finding due to missing initial positions.")

        # update the *active_curve* instance based on final targets (if anchors)
        # this modifies the curve object that will be used for path calculation
        self._update_curve_params_from_anchor(
            start_target, is_start=True, curve_instance=active_curve
        )
        self._update_curve_params_from_anchor(
            end_target, is_start=False, curve_instance=active_curve
        )

        # calculate final world positions using final targets/offsets
        world_start = self._get_offset_world_position(start_target, start_offset)
        world_end = self._get_offset_world_position(end_target, end_offset)

        if world_start and world_end:
            return world_start, world_end, active_curve
        else:
            self._log_debug(
                f"failed to calculate final world positions (start={world_start}, end={world_end})"
            )
            return None

    def _get_offset_world_position(
        self, component: Component, offset: Offset
    ) -> Optional[Tuple[float, float]]:
        """gets the world position corresponding to an offset on a component."""
        try:
            comp_dims = component._dimensions
            if comp_dims.width <= 0 or comp_dims.height <= 0:
                nat_dims = getattr(component, "_natural_dimensions", Size())
                comp_dims = nat_dims if nat_dims.width > 0 else Size()

            local_pos = offset.compute(self_dims=comp_dims, reference_dims=comp_dims)
            world_matrix = component.compute_world_matrix()
            world_point = world_matrix @ np.array([local_pos[0], local_pos[1], 1])
            return (world_point[0], world_point[1])
        except Exception as e:
            self._log_debug(f"error getting offset world pos for {component.id}: {e}")
            return None

    def _get_local_points_and_curve(
        self,
    ) -> Optional[
        Tuple[Tuple[float, float], Tuple[float, float], List[Tuple[float, float]], CurveDefinition]
    ]:
        """
        calculates world points, transforms them to connection's local space,
        and returns local start, end, control points, and the active curve instance.
        returns None if points cannot be calculated.
        """
        world_point_data = self._get_world_connection_points()
        if not world_point_data:
            return None
        world_start, world_end, active_curve = world_point_data  # get the final curve instance

        try:
            conn_world_matrix = self.compute_world_matrix()
            conn_inv_world = np.linalg.inv(conn_world_matrix)
        except np.linalg.LinAlgError:
            self._log_debug("warning: connection matrix not invertible, using identity.")
            conn_inv_world = np.identity(3)

        local_start_h = conn_inv_world @ np.array([world_start[0], world_start[1], 1])
        local_end_h = conn_inv_world @ np.array([world_end[0], world_end[1], 1])
        local_start = (local_start_h[0], local_start_h[1])
        local_end = (local_end_h[0], local_end_h[1])

        try:
            # use the potentially modified active_curve instance
            _, control_points = active_curve.get_path(local_start, local_end)
        except Exception as e:
            self._log_debug(f"error getting path/control points from curve: {e}")
            control_points = []

        # store for potential reuse (e.g., by renderer if measure/render are separate)
        self._local_start = local_start
        self._local_end = local_end
        self._local_control_points = control_points

        return local_start, local_end, control_points, active_curve

    def measure_and_layout(self, renderer=None):
        """
        measure connection bounds. connections are overlays, so layout is trivial.
        bounds are calculated based on the potential curve path.
        """
        self._apply_style()
        self._resolve_references()

        if not self._resolved_start_component or not self._resolved_end_component:
            self._log_debug("measure skipped: missing resolved components.")
            self._dimensions = Size()
            return self._dimensions

        # calculate world points and the curve instance used
        world_point_data = self._get_world_connection_points()
        if not world_point_data:
            self._dimensions = Size()
            return self._dimensions
        world_start, world_end, _ = world_point_data  # ignore curve instance here

        buffer = max(self.line_width * 3, 10)
        min_x = min(world_start[0], world_end[0]) - buffer
        min_y = min(world_start[1], world_end[1]) - buffer
        max_x = max(world_start[0], world_end[0]) + buffer
        max_y = max(world_start[1], world_end[1]) + buffer

        width = max(1.0, max_x - min_x)
        height = max(1.0, max_y - min_y)
        self._dimensions = Size(width=width, height=height)
        # set absolute offset *relative to root*, not just (min_x, min_y)
        self.offset = Offset(absolute=(min_x, min_y))

        self._log_debug(f"measured bounds: dims={self._dimensions}, offset={self.offset}")

        # calculate local points now that dimensions/offset are known
        self._get_local_points_and_curve()

        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render the connection curve and caps."""
        if not self.show or not self._resolved_start_component or not self._resolved_end_component:
            return

        # reuse cached local points if available, otherwise calculate
        # use the _current_curve_instance cached during measure
        local_point_data = self._get_local_points_and_curve()
        if not local_point_data:
            self._log_debug("render skipped: could not calculate local points.")
            return

        local_start, local_end, local_cp, active_curve = local_point_data

        try:
            path_str, _ = active_curve.get_path(local_start, local_end)
        except Exception as e:
            self._log_debug(f"render skipped: error getting path string from curve: {e}")
            return

        renderer.render_connection_curve(
            context, self, local_start, local_end, local_cp, path_str, matrix
        )

        # Render caps using the *same* active_curve instance for direction calculation
        if self.start_cap or self.end_cap:
            try:
                start_dir, end_dir = active_curve.get_directions(local_start, local_end, local_cp)

                from jeanplot.svg import (
                    LineEndArrow,
                    LineEndCircle,
                    LineEndFlat,
                    create_arrow_cap,
                    create_circle_cap,
                    create_flat_cap,
                )  # Local import

                cap_funcs = {
                    LineEndArrow: create_arrow_cap,
                    LineEndCircle: create_circle_cap,
                    LineEndFlat: create_flat_cap,
                }

                if self.start_cap:
                    cap_type = type(self.start_cap)
                    if cap_type in cap_funcs:
                        start_cap_path = cap_funcs[cap_type](local_start, start_dir, self.start_cap)
                        renderer.render_path(context, start_cap_path, matrix, "point")

                if self.end_cap:
                    cap_type = type(self.end_cap)
                    if cap_type in cap_funcs:
                        end_cap_path = cap_funcs[cap_type](local_end, end_dir, self.end_cap)
                        renderer.render_path(context, end_cap_path, matrix, "point")

            except Exception as e:
                self._log_debug(f"error rendering end caps for {self.id}: {e}")

        if self.debug:
            renderer.render_debug(context, self, matrix)

    def _log_debug(self, message: str, data: Any = None):
        """concise debug logging with connection identity"""
        start_id = (
            getattr(self._resolved_start_component, "id", "?")
            if self._resolved_start_component
            else "?"
        )
        end_id = (
            getattr(self._resolved_end_component, "id", "?")
            if self._resolved_end_component
            else "?"
        )
        comp_id = self.id or f"Conn({start_id}->{end_id})"
        debug_print(comp_id, message, data)
