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
    # store modified curve instance used for the last calculation
    _current_curve_instance: Optional[CurveDefinition] = PrivateAttr(default=None)
    # resolved component references
    _resolved_start_component: Optional[Component] = PrivateAttr(default=None)
    _resolved_end_component: Optional[Component] = PrivateAttr(default=None)
    # cached points from last calculation (relative to connection's own origin)
    _local_start: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _local_end: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _local_control_points: List[Tuple[float, float]] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _init_curve_instance(self):
        """ensure a copy of the curve definition is made."""
        self._current_curve_instance = copy.deepcopy(self.curve_type)
        return self

    def _get_active_curve(self) -> CurveDefinition:
        """returns the curve instance to use for calculations."""
        if self._current_curve_instance is None or type(self._current_curve_instance) != type(
            self.curve_type
        ):
            self._current_curve_instance = copy.deepcopy(self.curve_type)
        return self._current_curve_instance

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

    def _update_curve_params_from_anchor(self, anchor: AnchorComponent, is_start: bool):
        """updates the ACTIVE curve's parameters based on anchor properties."""
        if not isinstance(anchor, AnchorComponent) or anchor.direction is None:
            return

        active_curve = self._get_active_curve()
        prefix = "start" if is_start else "end"
        anchor_dir = anchor.direction
        anchor_len = anchor.min_segment
        updated = False

        self._log_debug(
            f"updating curve from anchor '{anchor.id}': dir={anchor_dir}, len={anchor_len} for side '{prefix}'"
        )

        if isinstance(active_curve, OrthogonalCurve):
            ortho_dir_name = OrthogonalCurve.get_direction(anchor_dir)
            if getattr(active_curve, f"{prefix}_direction") != ortho_dir_name:
                setattr(active_curve, f"{prefix}_direction", ortho_dir_name)
                updated = True
            if getattr(active_curve, f"{prefix}_length") != anchor_len:
                setattr(active_curve, f"{prefix}_length", anchor_len)
                updated = True
        elif isinstance(active_curve, SimpleBezierCurve):
            vector = (anchor_dir[0] * anchor_len, anchor_dir[1] * anchor_len)
            if getattr(active_curve, f"{prefix}_mode") != "vector":
                setattr(active_curve, f"{prefix}_mode", "vector")
                updated = True
            if getattr(active_curve, f"{prefix}_vector") != vector:
                setattr(active_curve, f"{prefix}_vector", vector)
                updated = True

    def _get_world_connection_points(
        self,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        calculates connection world points, handling auto-routing and anchor updates.
        returns (world_start, world_end) or None on failure.
        """
        self._resolve_references()  # ensure components are resolved first
        start_comp_orig = self._resolved_start_component
        end_comp_orig = self._resolved_end_component

        if not start_comp_orig or not end_comp_orig:
            self._log_debug("cannot calculate points: missing resolved components.")
            return None

        # reset active curve instance to base type before potentially modifying it
        self._current_curve_instance = copy.deepcopy(self.curve_type)

        # ensure target components have dimensions (needed for offset calculation)
        # ideally measure_and_layout has run, but check just in case
        for comp in [start_comp_orig, end_comp_orig]:
            if comp._dimensions.width <= 0 or comp._dimensions.height <= 0:
                self._log_debug(
                    f"warning: target component {comp.id} has zero size for connection calc."
                )
                # attempt measure? might cause loops if called during layout
                # comp.measure_and_layout() # careful with this

        start_target, end_target = start_comp_orig, end_comp_orig
        start_offset, end_offset = self.start_offset, self.end_offset

        if self.auto_route:
            # get initial positions using original components/offsets for anchor finding
            world_start_initial = self._get_offset_world_position(start_target, start_offset)
            world_end_initial = self._get_offset_world_position(end_target, end_offset)

            if world_start_initial and world_end_initial:
                best_end_anchor = self._find_best_anchor(end_comp_orig, world_start_initial)
                if best_end_anchor:
                    end_target = best_end_anchor
                    end_offset = Offset()  # connect to anchor origin

                # use potentially updated end target position to find start anchor
                ref_point_for_start = (
                    self._get_offset_world_position(end_target, end_offset) or world_end_initial
                )
                best_start_anchor = self._find_best_anchor(start_comp_orig, ref_point_for_start)
                if best_start_anchor:
                    start_target = best_start_anchor
                    start_offset = Offset()  # connect to anchor origin
            else:
                self._log_debug("skipping anchor finding due to missing initial positions.")

        # update curve parameters based on final targets (if anchors)
        self._update_curve_params_from_anchor(start_target, is_start=True)
        self._update_curve_params_from_anchor(end_target, is_start=False)

        # calculate final world positions using final targets/offsets
        world_start = self._get_offset_world_position(start_target, start_offset)
        world_end = self._get_offset_world_position(end_target, end_offset)

        if world_start and world_end:
            # self._log_debug(f"calculated world points: start={world_start}, end={world_end}")
            return world_start, world_end
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
                # try natural dims as fallback for offset calculation
                nat_dims = getattr(component, "_natural_dimensions", Size())
                comp_dims = nat_dims if nat_dims.width > 0 else Size()
                self._log_debug(
                    f"using {'natural' if comp_dims.width > 0 else 'zero'} dims for offset on {component.id}"
                )

            # connection offsets use the component itself as the reference frame
            local_pos = offset.compute(self_dims=comp_dims, reference_dims=comp_dims)

            world_matrix = component.compute_world_matrix()
            world_point = world_matrix @ np.array([local_pos[0], local_pos[1], 1])
            return (world_point[0], world_point[1])
        except Exception as e:
            self._log_debug(f"error getting offset world pos for {component.id}: {e}")
            return None  # return None on any error

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
        world_points = self._get_world_connection_points()
        if not world_points:
            return None
        world_start, world_end = world_points

        active_curve = self._get_active_curve()

        # get connection's own inverse world matrix to transform points to local
        try:
            # connection itself has identity local matrix unless explicitly transformed
            conn_world_matrix = self.compute_world_matrix()
            conn_inv_world = np.linalg.inv(conn_world_matrix)
        except np.linalg.LinAlgError:
            self._log_debug("warning: connection matrix not invertible, using identity.")
            conn_inv_world = np.identity(3)

        # transform world points to connection's local coordinate system
        local_start_h = conn_inv_world @ np.array([world_start[0], world_start[1], 1])
        local_end_h = conn_inv_world @ np.array([world_end[0], world_end[1], 1])
        local_start = (local_start_h[0], local_start_h[1])
        local_end = (local_end_h[0], local_end_h[1])

        # get path string and control points (in local coords) from curve instance
        # note: get_path uses local points directly
        try:
            # get_path currently assumes points are already local
            _, control_points = active_curve.get_path(local_start, local_end)
        except Exception as e:
            self._log_debug(f"error getting path/control points from curve: {e}")
            control_points = []  # fallback

        # self._log_debug(f"local points: start={local_start}, end={local_end}, cp={control_points}")
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

        # calculate world points to estimate bounds
        # this also updates the curve instance if auto-routing/anchors are used
        world_pts = self._get_world_connection_points()
        if not world_pts:
            self._dimensions = Size()
            return self._dimensions
        world_start, world_end = world_pts

        # use a rough bounding box based on world start/end for simplicity
        # a more accurate measure would trace the curve, but less performant
        # add buffer for line width and caps
        buffer = max(self.line_width * 3, 10)
        min_x = min(world_start[0], world_end[0]) - buffer
        min_y = min(world_start[1], world_end[1]) - buffer
        max_x = max(world_start[0], world_end[0]) + buffer
        max_y = max(world_start[1], world_end[1]) + buffer

        # set dimensions and offset based on world bounds
        # offset will position the connection container's origin at world (min_x, min_y)
        width = max(1.0, max_x - min_x)
        height = max(1.0, max_y - min_y)
        self._dimensions = Size(width=width, height=height)
        self.offset = Offset(absolute=(min_x, min_y))  # use absolute offset for positioning

        self._log_debug(f"measured bounds: dims={self._dimensions}, offset={self.offset}")

        # no internal layout needed for connection itself
        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render the connection curve and caps."""
        if not self.show or not self._resolved_start_component or not self._resolved_end_component:
            return

        # calculate local points relative to connection's origin just before rendering
        local_point_data = self._get_local_points_and_curve()
        if not local_point_data:
            self._log_debug("render skipped: could not calculate local points.")
            return
        local_start, local_end, local_cp, active_curve = local_point_data

        # get path string using local points
        try:
            path_str, _ = active_curve.get_path(local_start, local_end)
        except Exception as e:
            self._log_debug(f"render skipped: error getting path string from curve: {e}")
            return

        # delegate drawing the path to the renderer
        renderer.render_connection_curve(
            context, self, local_start, local_end, local_cp, path_str, matrix
        )

        if self.debug:
            renderer.render_debug(context, self, matrix)  # debug for connection container

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
