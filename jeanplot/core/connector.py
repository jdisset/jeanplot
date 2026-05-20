from pydantic import Field, PrivateAttr
import numpy as np
import copy
import logging

from jeanplot.core.path_utils import find_component_by_path
from jeanplot.core.connection_routing import find_best_anchor_pair
from jeanplot.core.curve import (
    CurveDefinition,
    OrthogonalCurve,
    SimpleBezierCurve,
    StraightCurve,
)
from jeanplot.core.component import Component, Overlay, AnchorComponent
from jeanplot.core.models import Size, Offset, NormalizedColor, LineWidthMode
from jeanplot.core.svg import LineEndType, LineStyle, SVGPathData

logger = logging.getLogger(__name__)


class Connection(Overlay):
    """connects two components with a styled line/curve."""

    start_component: str | Component
    end_component: str | Component
    start_offset: Offset = Field(default_factory=lambda: Offset(reference_relative=(0.5, 0.5)))
    end_offset: Offset = Field(default_factory=lambda: Offset(reference_relative=(0.5, 0.5)))

    color: NormalizedColor = "#000000"
    line_width: float = 1.0
    linewidth_mode: LineWidthMode = "data"
    curve_type: CurveDefinition = Field(default_factory=StraightCurve)
    line_style: LineStyle = "solid"
    dash_array: tuple[float, ...] | None = None
    dash_offset: float = 0.0
    start_cap: LineEndType | None = None
    end_cap: LineEndType | None = None
    auto_route: bool = True

    _resolved_start_component_orig: Component | None = PrivateAttr(default=None)
    _resolved_end_component_orig: Component | None = PrivateAttr(default=None)

    _render_start_target: Component | None = PrivateAttr(default=None)
    _render_end_target: Component | None = PrivateAttr(default=None)
    _render_start_offset: Offset | None = PrivateAttr(default=None)
    _render_end_offset: Offset | None = PrivateAttr(default=None)
    _render_local_start: tuple[float, float] | None = PrivateAttr(default=None)
    _render_local_end: tuple[float, float] | None = PrivateAttr(default=None)
    _render_local_control_points: list[tuple[float, float]] = PrivateAttr(default_factory=list)
    _render_active_curve: CurveDefinition | None = PrivateAttr(default=None)

    def _get_active_curve(self) -> CurveDefinition:
        # deepcopy so calculations don't mutate the styled curve_type
        return copy.deepcopy(self.curve_type)

    def _resolve_references_orig(self):
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

    def _resolve_single_ref(self, path: str, which: str) -> Component | None:
        if not self.parent:
            self._log_debug(f"cannot resolve {which} path '{path}' without parent.")
            return None
        try:
            root = self.parent
            while root.parent is not None:
                root = root.parent
            resolved = find_component_by_path(root, path)
            if resolved:
                return resolved
            else:
                return None
        except ValueError as e:
            self._log_debug(f"error resolving {which} path '{path}': {e}")
            return None

    def _get_component_center_world(self, component: Component) -> tuple[float, float] | None:
        if not component:
            return None
        center_offset = Offset(reference_relative=(0.5, 0.5))
        return self._get_offset_world_position(component, center_offset)

    def _update_curve_params_from_anchor(
        self, anchor: Component, is_start: bool, curve_instance: CurveDefinition
    ):
        """mutate curve_instance in place to match anchor's outward direction."""
        if not isinstance(anchor, AnchorComponent) or anchor.direction is None:
            return

        prefix = "start" if is_start else "end"
        anchor_dir_outward = anchor.direction
        anchor_len = anchor.min_segment

        if isinstance(curve_instance, OrthogonalCurve):
            ortho_dir_name = OrthogonalCurve.get_direction_from_vector(anchor_dir_outward)
            if getattr(curve_instance, f"{prefix}_direction") != ortho_dir_name:
                setattr(curve_instance, f"{prefix}_direction", ortho_dir_name)

            if getattr(curve_instance, f"{prefix}_length") != anchor_len:
                setattr(curve_instance, f"{prefix}_length", anchor_len)

        elif isinstance(curve_instance, SimpleBezierCurve):
            vector = (anchor_dir_outward[0] * anchor_len, anchor_dir_outward[1] * anchor_len)
            if getattr(curve_instance, f"{prefix}_mode") != "vector":
                setattr(curve_instance, f"{prefix}_mode", "vector")
            if getattr(curve_instance, f"{prefix}_vector") != vector:
                setattr(curve_instance, f"{prefix}_vector", vector)

    def _get_offset_world_position(
        self, component: Component, offset: Offset
    ) -> tuple[float, float] | None:
        if component is None:
            return None
        try:
            # dimensions should be final during render phase
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
        """resolve final start/end components/offsets after layout. updates _render_* attrs."""
        self._resolve_references_orig()
        start_comp_orig = self._resolved_start_component_orig
        end_comp_orig = self._resolved_end_component_orig

        if not start_comp_orig or not end_comp_orig:
            self._log_debug("cannot resolve endpoints: missing original components.")
            return False

        start_target = start_comp_orig
        end_target = end_comp_orig
        start_offset = self.start_offset
        end_offset = self.end_offset

        if self.auto_route:
            best_pair = find_best_anchor_pair(start_comp_orig, end_comp_orig)
            if best_pair:
                start_target, end_target = best_pair
                # anchor position is inherent; clear offsets
                start_offset = Offset()
                end_offset = Offset()

        self._render_start_target = start_target
        self._render_end_target = end_target
        self._render_start_offset = start_offset
        self._render_end_offset = end_offset

        world_start = self._get_offset_world_position(start_target, start_offset)
        world_end = self._get_offset_world_position(end_target, end_offset)

        if world_start is None or world_end is None:
            self._log_debug(
                f"failed to calculate final world points (start={world_start}, end={world_end})"
            )
            return False

        active_curve = self._get_active_curve()
        self._update_curve_params_from_anchor(
            start_target, is_start=True, curve_instance=active_curve
        )
        self._update_curve_params_from_anchor(
            end_target, is_start=False, curve_instance=active_curve
        )
        self._render_active_curve = active_curve

        try:
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

        try:
            _, control_points = active_curve.get_path(
                self._render_local_start, self._render_local_end
            )
            self._render_local_control_points = control_points
        except Exception as e:
            self._log_debug(f"error getting path/control points from curve: {e}")
            self._render_local_control_points = []

        return True

    def measure_and_layout(self, renderer=None):
        """rough placeholder bounds; actual path is calculated at render time."""
        self._apply_style()
        self._resolve_references_orig()

        start_comp = self._resolved_start_component_orig
        end_comp = self._resolved_end_component_orig

        if not start_comp or not end_comp:
            self._log_debug("measure skipped: missing resolved components.")
            self._dimensions = Size(1, 1)
            return self._dimensions

        # positions may be slightly off pre-render; use centers for bound estimate
        start_pos = self._get_component_center_world(start_comp) or (0, 0)
        end_pos = self._get_component_center_world(end_comp) or (0, 0)

        buffer = max(self.line_width * 5, 10)
        min_x = min(start_pos[0], end_pos[0]) - buffer
        min_y = min(start_pos[1], end_pos[1]) - buffer
        max_x = max(start_pos[0], end_pos[0]) + buffer
        max_y = max(start_pos[1], end_pos[1]) + buffer

        width = max(1.0, max_x - min_x)
        height = max(1.0, max_y - min_y)
        self._dimensions = Size(width=width, height=height)
        self.offset = Offset(absolute=(min_x, min_y))

        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        if not self.show:
            return

        if not self._resolve_connection_endpoints_late():
            self._log_debug("render skipped: could not resolve endpoints.")
            if self.debug:
                renderer.render_debug(context, self, matrix)
            return

        local_start = self._render_local_start
        local_end = self._render_local_end
        local_cp = self._render_local_control_points
        active_curve = self._render_active_curve

        if local_start is None or local_end is None or active_curve is None:
            self._log_debug("render skipped: missing calculated local points or curve.")
            if self.debug:
                renderer.render_debug(context, self, matrix)
            return

        path_str, _ = active_curve.get_path(local_start, local_end, local_cp)

        path_data = SVGPathData(
            d=path_str,
            stroke=self.color,
            stroke_width=self.line_width,
            line_style=self.line_style,
            dash_array=self.dash_array,
            dash_offset=self.dash_offset,
        )
        renderer.render_path(
            context,
            path_data,
            matrix,
            line_width_mode=self.linewidth_mode,
            component_id=f"{self.id}_main_curve",
            opacity=self.opacity,
        )

        if self.start_cap or self.end_cap:
            start_dir, end_dir = active_curve.get_directions(local_start, local_end, local_cp)

            from jeanplot.core.svg import (
                LineEndArrow,
                LineEndCircle,
                LineEndFlat,
                create_arrow_cap,
                create_circle_cap,
                create_flat_cap,
            )

            cap_funcs = {
                LineEndArrow: create_arrow_cap,
                LineEndCircle: create_circle_cap,
                LineEndFlat: create_flat_cap,
            }

            if self.start_cap:
                cap_type = type(self.start_cap)
                if cap_type in cap_funcs:
                    start_cap_path = cap_funcs[cap_type](local_start, start_dir, self.start_cap)
                    renderer.render_path(
                        context,
                        start_cap_path,
                        matrix,
                        line_width_mode=self.linewidth_mode,
                        component_id=f"{self.id}_start_cap",
                        opacity=self.opacity,
                    )

            if self.end_cap:
                cap_type = type(self.end_cap)
                if cap_type in cap_funcs:
                    end_cap_path = cap_funcs[cap_type](local_end, end_dir, self.end_cap)
                    renderer.render_path(
                        context,
                        end_cap_path,
                        matrix,
                        line_width_mode=self.linewidth_mode,
                        component_id=f"{self.id}_end_cap",
                        opacity=self.opacity,
                    )

        if self.debug:
            renderer.render_debug(context, self, matrix)

    def get_point_along(
        self, distance: float, *, relative: bool = True
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """(point, tangent) along rendered path in local coords.
        Returns None if connection hasn't been rendered yet."""
        if (
            self._render_active_curve is None
            or self._render_local_start is None
            or self._render_local_end is None
        ):
            return None
        return self._render_active_curve.evaluate_at_distance(
            distance,
            self._render_local_start,
            self._render_local_end,
            self._render_local_control_points,
            relative=relative,
        )

    def _get_world_connection_points(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], CurveDefinition] | None:
        """(start, end, active_curve) in world coords, or None if not calculated."""
        if not self._resolve_connection_endpoints_late():
            return None

        if self._render_start_target and self._render_end_target:
            world_start = self._get_offset_world_position(
                self._render_start_target, self._render_start_offset
            )
            world_end = self._get_offset_world_position(
                self._render_end_target, self._render_end_offset
            )

            if world_start and world_end and self._render_active_curve:
                return (world_start, world_end, self._render_active_curve)

        return None
