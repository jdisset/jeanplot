"""Definitions for different types of curves used by Connections."""

from typing import Optional, Literal, get_args, List, Tuple, Any
from pydantic import Field, BaseModel, PrivateAttr
import numpy as np
import math

# use absolute imports
from jeanplot.path_utils import (
    normalize_vector,
    path_with_checkpoints,
    find_simple_path,
    connect_segments,
    create_rounded_orthogonal_path,
)
from jeanplot.debug import debug_print

# types including 'auto'
OrthoDirection = Literal["up", "down", "left", "right", "auto"]
BezierDirectionMode = Literal["vector", "auto"]


class CurveDefinition(BaseModel):
    """base class for curve definitions."""

    # default length/strength for auto-calculated directions/vectors
    auto_direction_strength: float = 15.0

    def _log_debug(self, method: str, message: str, data: Any = None):
        debug_print(f"{self.__class__.__name__}.{method}", message, data)

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        """generate svg path data string and return control points (if any)."""
        raise NotImplementedError

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """get normalized direction vectors pointing *outwards* at start and end."""
        # fallback for straight lines or unknown curves
        dx, dy = end[0] - start[0], end[1] - start[1]
        dir_vec = normalize_vector((dx, dy))
        # start direction points away from end; end direction points away from start
        return (-dir_vec[0], -dir_vec[1]), dir_vec

    @staticmethod
    def _calculate_auto_direction(
        start: Tuple[float, float], end: Tuple[float, float], for_start: bool
    ) -> Tuple[float, float]:
        """calculate normalized direction vector based on start/end points."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        # direction points away from start if for_start=True, else away from end
        return normalize_vector((dx, dy)) if for_start else normalize_vector((-dx, -dy))


class StraightCurve(CurveDefinition):
    """straight line connection."""

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        # self._log_debug("get_path", f"{start} -> {end}")
        return f"M {start[0]} {start[1]} L {end[0]} {end[1]}", []

    # uses base get_directions


class SimpleBezierCurve(CurveDefinition):
    """bezier curve with vectors from start/end, supporting 'auto' direction."""

    start_mode: BezierDirectionMode = "auto"
    end_mode: BezierDirectionMode = "auto"
    # vector used if mode == 'vector' (relative offset from start/end point)
    start_vector: Optional[Tuple[float, float]] = None
    end_vector: Optional[Tuple[float, float]] = None

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        # self._log_debug("get_path", f"{start} -> {end}")

        # determine start control point
        if self.start_mode == "auto" or not self.start_vector:
            auto_dir = self._calculate_auto_direction(start, end, for_start=True)
            sx, sy = (
                auto_dir[0] * self.auto_direction_strength,
                auto_dir[1] * self.auto_direction_strength,
            )
        else:
            sx, sy = self.start_vector
        c1 = (start[0] + sx, start[1] + sy)

        # determine end control point
        if self.end_mode == "auto" or not self.end_vector:
            auto_dir = self._calculate_auto_direction(start, end, for_start=False)
            ex, ey = (
                auto_dir[0] * self.auto_direction_strength,
                auto_dir[1] * self.auto_direction_strength,
            )
        else:
            ex, ey = self.end_vector
        c2 = (end[0] + ex, end[1] + ey)  # end control vector points *away* from end point

        # self._log_debug("get_path", f"cp1={c1}, cp2={c2}")
        path = f"M {start[0]} {start[1]} C {c1[0]} {c1[1]}, {c2[0]} {c2[1]}, {end[0]} {end[1]}"
        control_points = [c1, c2]
        return path, control_points

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """get normalized direction vectors pointing outwards at start and end."""
        # self._log_debug("get_directions", f"{start} -> {end}", control_points)
        if len(control_points) < 2:
            return super().get_directions(start, end, control_points)

        c1, c2 = control_points
        # start direction (points from start towards first control point)
        start_dir_out = normalize_vector((c1[0] - start[0], c1[1] - start[1]))
        # end direction (points from second control point towards end)
        end_dir_out = normalize_vector((end[0] - c2[0], end[1] - c2[1]))
        # self._log_debug("get_directions", f"start_out={start_dir_out}, end_out={end_dir_out}")
        return start_dir_out, end_dir_out


class OrthogonalCurve(CurveDefinition):
    """path with orthogonal segments, supporting 'auto' direction."""

    start_direction: OrthoDirection = "auto"
    start_length: float = Field(default=5.0, description="Min length of the first segment")
    end_direction: OrthoDirection = "auto"
    end_length: float = Field(default=5.0, description="Min length of the last segment")
    corner_radius: float = 10.0

    # checkpoints: list[tuple[float, float]] = Field(default_factory=list) # handled by Connection
    auto_simplify: bool = True  # attempt 0 or 1 turn path first

    # internal cache for resolved directions from last path calculation
    _resolved_start_dir_vec: Optional[Tuple[float, float]] = PrivateAttr(default=None)
    _resolved_end_dir_vec: Optional[Tuple[float, float]] = PrivateAttr(default=None)

    @staticmethod
    def get_direction(vector: Tuple[float, float]) -> OrthoDirection:
        """get closest orthogonal direction name from a vector (assuming y-up)."""
        valid_directions = get_args(OrthoDirection)[:-1]  # up, down, left, right
        if not any(abs(d) > 1e-9 for d in vector):
            return "up"  # default fallback
        norm_vec = normalize_vector(vector)
        # vectors corresponding to up, down, left, right
        dir_vectors = np.array([(0, 1), (0, -1), (-1, 0), (1, 0)])
        dot_products = np.dot(dir_vectors, norm_vec)
        return valid_directions[np.argmax(dot_products)]

    def _resolve_direction(
        self,
        direction_mode: OrthoDirection,
        start: Tuple[float, float],
        end: Tuple[float, float],
        for_start: bool,
    ) -> OrthoDirection:
        """resolve 'auto' direction."""
        if direction_mode == "auto":
            auto_vec = self._calculate_auto_direction(start, end, for_start)
            resolved = self.get_direction(auto_vec)
            # self._log_debug("_resolve_direction", f"resolved 'auto' for {'start' if for_start else 'end'} -> '{resolved}'")
            return resolved
        return direction_mode

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        # self._log_debug("get_path", f"{start} -> {end}")

        dir_to_vector = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
        resolved_start_dir_name = self._resolve_direction(self.start_direction, start, end, True)
        resolved_end_dir_name = self._resolve_direction(self.end_direction, start, end, False)

        start_dir_vec = dir_to_vector.get(
            resolved_start_dir_name, (0, 1)
        )  # start dir points outwards
        end_dir_vec = dir_to_vector.get(resolved_end_dir_name, (0, -1))  # end dir points outwards
        end_dir_rev = (-end_dir_vec[0], -end_dir_vec[1])  # end dir points inwards from end

        points = []
        checkpoints = local_checkpoints or []

        if checkpoints:
            points = path_with_checkpoints(
                checkpoints,
                start,
                end,
                start_dir_vec,
                end_dir_rev,
                self.start_length,
                self.end_length,
            )
        elif self.auto_simplify:
            simple_path = find_simple_path(
                start, end, start_dir_vec, end_dir_vec, self.start_length, self.end_length
            )
            if simple_path:
                points = simple_path

        if not points:  # fallback
            first_seg_end = (
                start[0] + start_dir_vec[0] * self.start_length,
                start[1] + start_dir_vec[1] * self.start_length,
            )
            last_seg_start = (
                end[0] + end_dir_rev[0] * self.end_length,
                end[1] + end_dir_rev[1] * self.end_length,
            )
            points = connect_segments(start, first_seg_end, last_seg_start, end)

        # self._log_debug("get_path", f"points: {points}")
        if self.corner_radius > 0 and len(points) >= 3:
            path = create_rounded_orthogonal_path(points, self.corner_radius)
            # self._log_debug("get_path", f"rounded path string generated (len={len(path)})")
        else:
            path = f"M {points[0][0]} {points[0][1]}" + "".join(
                f" L {p[0]} {p[1]}" for p in points[1:]
            )
            # self._log_debug("get_path", f"straight path string generated (len={len(path)})")

        # cache resolved OUTWARD directions for get_directions call
        self._resolved_start_dir_vec = normalize_vector(start_dir_vec)
        self._resolved_end_dir_vec = normalize_vector(end_dir_vec)
        return path, []  # orthogonal paths don't have explicit control points

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """get normalized outward direction vectors using cached resolved directions."""
        # self._log_debug("get_directions", f"{start} -> {end}")
        if self._resolved_start_dir_vec and self._resolved_end_dir_vec:
            return self._resolved_start_dir_vec, self._resolved_end_dir_vec
        else:
            # fallback if called before get_path or resolution failed
            # self._log_debug("get_directions", "resolving directions dynamically (fallback)")
            resolved_start_dir_name = self._resolve_direction(
                self.start_direction, start, end, True
            )
            resolved_end_dir_name = self._resolve_direction(self.end_direction, start, end, False)
            dir_to_vector = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
            start_dir_vec = dir_to_vector.get(resolved_start_dir_name, (0, 1))
            end_dir_vec = dir_to_vector.get(resolved_end_dir_name, (0, -1))
            return normalize_vector(start_dir_vec), normalize_vector(end_dir_vec)
