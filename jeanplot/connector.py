# File: jeanplot/connector.py
"""Component for drawing connections (lines/curves) between other components."""

from typing import Optional, Any, List, Tuple, Union
from pydantic import Field, PrivateAttr, model_validator
import numpy as np
import math
import copy
import logging

# NOTE: Import normalize_vector from path_utils here
from jeanplot.path_utils import find_component_by_path, normalize_vector
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
    # offset from the resolved start component's origin (used if auto_route=False or no anchor found)
    start_offset: Offset = Field(default_factory=lambda: Offset(reference_relative=(0.5, 0.5)))
    # offset from the resolved end component's origin (used if auto_route=False or no anchor found)
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
    # resolved component references (original targets before anchor selection)
    _resolved_start_component_orig: Optional[Component] = PrivateAttr(default=None)
    _resolved_end_component_orig: Optional[Component] = PrivateAttr(default=None)

    # state calculated during render phase
    _render_start_target: Optional[Component] = PrivateAttr(default=None)
    _render_end_target: Optional[Component] = PrivateAttr(default=None)
    _render_start_offset: Optional[Offset] = PrivateAttr(default=None)
    _render_end_offset: Optional[Offset] = PrivateAttr(default=None)
    _render_local_start: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _render_local_end: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _render_local_control_points: List[Tuple[float, float]] = PrivateAttr(default_factory=list)
    _render_active_curve: Optional[CurveDefinition] = PrivateAttr(default=None)

    # --- Methods ---

    def _get_active_curve(self) -> CurveDefinition:
        """
        returns a copy of the currently styled curve instance to use for calculations.
        this ensures calculations start with the parameters set by jstyle.
        """
        # always return a fresh copy of the current curve_type state
        return copy.deepcopy(self.curve_type)

    def _resolve_references_orig(self):
        """resolve string paths for original start/end_component."""
        resolved_start = self._resolved_start_component_orig
        resolved_end = self._resolved_end_component_orig

        if isinstance(self.start_component, str) and resolved_start is None:
            resolved_start = self._resolve_single_ref(self.start_component, "start")
        elif isinstance(self.start_component, Component):
            resolved_start = self.start_component

        if isinstance(self.end_component, str) and resolved_end is None:
            resolved_end = self._resolve_single_ref(self.end_component, "end")
        elif isinstance(self.end_component, Component):
            resolved_end = self.end_component

        self._resolved_start_component_orig = resolved_start
        self._resolved_end_component_orig = resolved_end

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
                # self._log_debug(f"resolved {which} path '{path}' to {resolved.id}")
                return resolved
            else:  # find_component_by_path raises error if not found
                return None
        except ValueError as e:
            self._log_debug(f"error resolving {which} path '{path}': {e}")
            return None

    def _get_component_center_world(self, component: Component) -> Optional[Tuple[float, float]]:
        """gets the world coordinates of the component's center."""
        if not component:
            return None
        # use a standard center offset for calculation
        center_offset = Offset(reference_relative=(0.5, 0.5))
        return self._get_offset_world_position(component, center_offset)

    def _get_anchors(self, component: Component) -> List[AnchorComponent]:
        """helper to get all valid anchor components associated with a component."""
        anchors = [a for a in getattr(component, "children", []) if isinstance(a, AnchorComponent)]
        anchors += [a for a in getattr(component, "anchor_points", []) if a not in anchors]
        valid_anchors = []
        for a in anchors:
            # ensure anchor has parent link
            if not a.parent:
                a.parent = component
            valid_anchors.append(a)
        return valid_anchors

    def _get_effective_anchor_pos(self, anchor: AnchorComponent) -> Optional[Tuple[float, float]]:
        """calculates the effective connection point considering min_segment."""
        anchor_pos = anchor.get_world_origin()
        if anchor_pos is None:
            return None

        direction = getattr(anchor, "direction", None)
        min_len = getattr(anchor, "min_segment", 0.0)

        if direction and min_len > 1e-6:
            norm_dir = normalize_vector(direction, default=(0, 0))
            if norm_dir != (0, 0):
                effective_pos = (
                    anchor_pos[0] + norm_dir[0] * min_len,
                    anchor_pos[1] + norm_dir[1] * min_len,
                )
                return effective_pos
        # return anchor origin if no direction/length
        return anchor_pos

    def _find_best_anchor_pair(
        self, start_comp: Component, end_comp: Component
    ) -> Optional[Tuple[AnchorComponent, AnchorComponent]]:
        """
        finds the pair of anchors (one from start, one from end) with min distance
        between their effective connection points (origin + direction*min_length).
        """
        start_anchors = self._get_anchors(start_comp)
        end_anchors = self._get_anchors(end_comp)

        if not start_anchors or not end_anchors:
            return None

        best_pair = None
        min_dist_sq = float("inf")

        start_anchor_details = []
        for s_anchor in start_anchors:
            s_eff_pos = self._get_effective_anchor_pos(s_anchor)
            if s_eff_pos:
                start_anchor_details.append({"anchor": s_anchor, "eff_pos": s_eff_pos})

        end_anchor_details = []
        for e_anchor in end_anchors:
            e_eff_pos = self._get_effective_anchor_pos(e_anchor)
            if e_eff_pos:
                end_anchor_details.append({"anchor": e_anchor, "eff_pos": e_eff_pos})

        if not start_anchor_details or not end_anchor_details:
            self._log_debug("could not get effective positions for all anchors.")
            return None  # could not calculate effective positions

        # self._log_debug(f"finding best anchor pair between {start_comp.id} ({len(start_anchor_details)} eff anchors) and {end_comp.id} ({len(end_anchor_details)} eff anchors)")
        # self._log_debug(f"start anchor details: {start_anchor_details}")
        # self._log_debug(f"end anchor details: {end_anchor_details}")

        for s_detail in start_anchor_details:
            s_anchor = s_detail["anchor"]
            s_eff_pos = s_detail["eff_pos"]
            for e_detail in end_anchor_details:
                e_anchor = e_detail["anchor"]
                e_eff_pos = e_detail["eff_pos"]

                dist_sq = (s_eff_pos[0] - e_eff_pos[0]) ** 2 + (s_eff_pos[1] - e_eff_pos[1]) ** 2
                # self._log_debug(f"  pair ({s_anchor.id}, {e_anchor.id}): eff_dist_sq = {dist_sq:.2f}")

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_pair = (s_anchor, e_anchor)

        # if best_pair:
        #     s_id = getattr(best_pair[0], 'id', '?')
        #     e_id = getattr(best_pair[1], 'id', '?')
        #     self._log_debug(f"best anchor pair found: ({s_id}, {e_id}), eff_dist_sq={min_dist_sq:.2f}")
        # else:
        #     self._log_debug("no suitable anchor pair found.")

        return best_pair

    def _update_curve_params_from_anchor(
        self, anchor: Component, is_start: bool, curve_instance: CurveDefinition
    ):
        """
        updates the GIVEN curve instance's parameters based on anchor properties.
        modifies the curve_instance in place.
        uses the anchor's outward direction.
        """
        if not isinstance(anchor, AnchorComponent) or anchor.direction is None:
            return

        prefix = "start" if is_start else "end"
        # anchor.direction is always the outward direction from the anchor
        anchor_dir_outward = anchor.direction
        anchor_len = anchor.min_segment

        # self._log_debug(
        #     f"updating curve from anchor '{anchor.id}': OUTWARD_dir={anchor_dir_outward}, len={anchor_len} for side '{prefix}'"
        # )

        if isinstance(curve_instance, OrthogonalCurve):
            # get the orthogonal direction name matching the anchor's outward vector
            ortho_dir_name = OrthogonalCurve.get_direction_from_vector(anchor_dir_outward)
            # self._log_debug(
            #     f"  converted anchor OUTWARD vector {anchor_dir_outward} to ortho direction '{ortho_dir_name}'"
            # )
            # Set the curve's direction for this end to match the anchor's outward direction
            if getattr(curve_instance, f"{prefix}_direction") != ortho_dir_name:
                setattr(curve_instance, f"{prefix}_direction", ortho_dir_name)
                # self._log_debug(f"  set curve {prefix}_direction = '{ortho_dir_name}'")

            if getattr(curve_instance, f"{prefix}_length") != anchor_len:
                setattr(curve_instance, f"{prefix}_length", anchor_len)
                # self._log_debug(f"  set curve {prefix}_length = {anchor_len}")

        elif isinstance(curve_instance, SimpleBezierCurve):
            # Bezier control vector should point OUTWARD from the anchor
            vector = (anchor_dir_outward[0] * anchor_len, anchor_dir_outward[1] * anchor_len)
            if getattr(curve_instance, f"{prefix}_mode") != "vector":
                setattr(curve_instance, f"{prefix}_mode", "vector")
                # self._log_debug(f"  set curve {prefix}_mode = 'vector'")
            if getattr(curve_instance, f"{prefix}_vector") != vector:
                setattr(curve_instance, f"{prefix}_vector", vector)
                # self._log_debug(f"  set curve {prefix}_vector = {vector}")

    def _get_offset_world_position(
        self, component: Component, offset: Offset
    ) -> Optional[Tuple[float, float]]:
        """gets the world position corresponding to an offset on a component."""
        if component is None:
            return None
        try:
            # Use current dimensions which should be final during render phase
            comp_dims = component._dimensions
            if comp_dims is None or comp_dims.width < 0 or comp_dims.height < 0:
                self._log_debug(f"warning: invalid dimensions {comp_dims} for {component.id}")
                return None

            local_pos = offset.compute(self_dims=comp_dims, reference_dims=comp_dims)
            world_matrix = component.compute_world_matrix()
            if world_matrix is None:
                self._log_debug(f"warning: could not compute world matrix for {component.id}")
                return None

            world_point = world_matrix @ np.array([local_pos[0], local_pos[1], 1])
            return (world_point[0], world_point[1])
        except Exception as e:
            self._log_debug(
                f"error getting offset world pos for {getattr(component, 'id', '?')}: {e}"
            )
            return None

    def _resolve_connection_endpoints_late(self) -> bool:
        """
        resolves final start/end components (considering anchors) and offsets.
        this MUST be called after the layout pass, e.g., during render.
        updates internal _render_* attributes.
        returns true if successful, false otherwise.
        """
        self._resolve_references_orig()  # ensure original components are resolved
        start_comp_orig = self._resolved_start_component_orig
        end_comp_orig = self._resolved_end_component_orig

        if not start_comp_orig or not end_comp_orig:
            self._log_debug("cannot resolve endpoints: missing original components.")
            return False

        # Default targets and offsets are the original components and configured offsets
        start_target = start_comp_orig
        end_target = end_comp_orig
        start_offset = self.start_offset
        end_offset = self.end_offset

        if self.auto_route:
            # Find the pair of anchors (one on start, one on end) with min distance
            # between their effective points (origin + dir*len)
            best_pair = self._find_best_anchor_pair(start_comp_orig, end_comp_orig)
            if best_pair:
                start_target, end_target = best_pair
                start_offset = Offset()  # anchor position is inherent
                end_offset = Offset()  # anchor position is inherent
                # self._log_debug(f"auto_route selected anchor pair: ({start_target.id}, {end_target.id})")
            # else:
            # self._log_debug("auto_route did not find a suitable anchor pair, using default offsets.")

        # store final targets and offsets used for rendering
        self._render_start_target = start_target
        self._render_end_target = end_target
        self._render_start_offset = start_offset
        self._render_end_offset = end_offset

        # calculate world points based on FINAL targets/offsets
        world_start = self._get_offset_world_position(start_target, start_offset)
        world_end = self._get_offset_world_position(end_target, end_offset)

        if world_start is None or world_end is None:
            self._log_debug(
                f"failed to calculate final world points (start={world_start}, end={world_end})"
            )
            return False

        # get curve instance and update based on chosen anchors (if they are anchors)
        active_curve = self._get_active_curve()
        self._update_curve_params_from_anchor(
            start_target, is_start=True, curve_instance=active_curve
        )
        self._update_curve_params_from_anchor(
            end_target, is_start=False, curve_instance=active_curve
        )
        self._render_active_curve = active_curve  # store the final curve instance

        # transform world points to connection's local space
        try:
            # Calculate connection's own matrix (it's usually an identity matrix at origin if offset is calculated correctly)
            conn_world_matrix = self.compute_world_matrix()
            conn_inv_world = np.linalg.inv(conn_world_matrix)
        except np.linalg.LinAlgError:
            self._log_debug("warning: connection matrix not invertible, using identity.")
            conn_inv_world = np.identity(3)
        except Exception as e_mat:
            self._log_debug(f"error calculating connection matrix: {e_mat}, using identity.")
            conn_inv_world = np.identity(3)

        local_start_h = conn_inv_world @ np.array([world_start[0], world_start[1], 1])
        local_end_h = conn_inv_world @ np.array([world_end[0], world_end[1], 1])
        self._render_local_start = (local_start_h[0], local_start_h[1])
        self._render_local_end = (local_end_h[0], local_end_h[1])

        # calculate control points using the final active curve and local points
        try:
            _, control_points = active_curve.get_path(
                self._render_local_start, self._render_local_end
            )
            self._render_local_control_points = control_points
        except Exception as e:
            self._log_debug(f"error getting path/control points from curve: {e}")
            self._render_local_control_points = []

        # self._log_debug(f"resolved endpoints: start_target={getattr(start_target,'id','?')}, end_target={getattr(end_target,'id','?')}")
        # self._log_debug(f"local points: start={self._render_local_start}, end={self._render_local_end}, cps={self._render_local_control_points}")

        return True

    def measure_and_layout(self, renderer=None):
        """
        connections are overlays. measure calculates rough bounds for debug/events.
        actual path calculation is deferred to render time.
        """
        self._apply_style()
        self._resolve_references_orig()

        # calculate very basic bounds based on original components for placeholder size
        start_comp = self._resolved_start_component_orig
        end_comp = self._resolved_end_component_orig

        if not start_comp or not end_comp:
            self._log_debug("measure skipped: missing resolved components.")
            self._dimensions = Size(1, 1)  # minimal size
            return self._dimensions

        # Get approx world bounds - positions might be slightly off pre-render
        # Use centers as reference points for bounds estimation
        start_pos = self._get_component_center_world(start_comp) or (0, 0)
        end_pos = self._get_component_center_world(end_comp) or (0, 0)

        buffer = max(self.line_width * 5, 10)  # increased buffer
        min_x = min(start_pos[0], end_pos[0]) - buffer
        min_y = min(start_pos[1], end_pos[1]) - buffer
        max_x = max(start_pos[0], end_pos[0]) + buffer
        max_y = max(start_pos[1], end_pos[1]) + buffer

        width = max(1.0, max_x - min_x)
        height = max(1.0, max_y - min_y)
        self._dimensions = Size(width=width, height=height)
        # set absolute offset *relative to root*
        self.offset = Offset(absolute=(min_x, min_y))

        # self._log_debug(f"measured rough bounds: dims={self._dimensions}, offset={self.offset}")
        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render the connection curve and caps."""
        if not self.show:
            return

        # --- Resolve endpoints and calculate path LATE ---
        if not self._resolve_connection_endpoints_late():
            self._log_debug("render skipped: could not resolve endpoints.")
            if self.debug:
                renderer.render_debug(context, self, matrix)  # still draw debug box
            return  # cannot render if endpoints failed

        local_start = self._render_local_start
        local_end = self._render_local_end
        local_cp = self._render_local_control_points
        active_curve = self._render_active_curve

        if local_start is None or local_end is None or active_curve is None:
            self._log_debug("render skipped: missing calculated local points or curve.")
            if self.debug:
                renderer.render_debug(context, self, matrix)
            return

        # --- Generate Path String ---
        try:
            path_str, _ = active_curve.get_path(local_start, local_end, local_cp)
        except Exception as e:
            self._log_debug(f"render skipped: error getting path string from curve: {e}")
            return

        # --- Render Curve ---
        renderer.render_connection_curve(
            context, self, local_start, local_end, local_cp, path_str, matrix
        )

        # --- Render Caps ---
        if self.start_cap or self.end_cap:
            try:
                # use the active curve instance from endpoint resolution for directions
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
                        # Pass outward start_dir
                        start_cap_path = cap_funcs[cap_type](local_start, start_dir, self.start_cap)
                        renderer.render_path(context, start_cap_path, matrix, "point")

                if self.end_cap:
                    cap_type = type(self.end_cap)
                    if cap_type in cap_funcs:
                        # Pass outward end_dir
                        end_cap_path = cap_funcs[cap_type](local_end, end_dir, self.end_cap)
                        renderer.render_path(context, end_cap_path, matrix, "point")

            except Exception as e:
                self._log_debug(f"error rendering end caps for {self.id}: {e}")

        if self.debug:
            # Use the rough bounds calculated during measure for the debug box
            renderer.render_debug(context, self, matrix)

    def _log_debug(self, message: str, data: Any = None):
        """concise debug logging with connection identity"""
        start_id = "?"
        if self._resolved_start_component_orig:
            start_id = getattr(self._resolved_start_component_orig, "id", "?")
        elif isinstance(self.start_component, Component):
            start_id = getattr(self.start_component, "id", "?")
        elif isinstance(self.start_component, str):
            start_id = self.start_component

        end_id = "?"
        if self._resolved_end_component_orig:
            end_id = getattr(self._resolved_end_component_orig, "id", "?")
        elif isinstance(self.end_component, Component):
            end_id = getattr(self.end_component, "id", "?")
        elif isinstance(self.end_component, str):
            end_id = self.end_component

        comp_id = self.id or f"Conn({start_id}->{end_id})"
        debug_print(comp_id, message, data)
