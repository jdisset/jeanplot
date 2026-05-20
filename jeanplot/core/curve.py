from typing import Literal, get_args, Any, Sequence, ClassVar
from pydantic import Field, BaseModel, PrivateAttr
import numpy as np

from jeanplot.core.path_utils import (
    normalize_vector,
    create_orthogonal_path,
    create_rounded_orthogonal_path,
)
from jeanplot.core.debug import debug_print, DebugMixin


OrthoDirection = Literal["up", "down", "left", "right", "auto"]
ValidOrthoDirection = Literal["up", "down", "left", "right"]
BezierDirectionMode = Literal["vector", "auto"]


class CurveDefinition(DebugMixin, BaseModel):
    auto_direction_strength: float = 15.0

    def _log_debug(self, method: str, message: str, data: Any = None):
        comp_id = f"{self.__class__.__name__}.{method}"
        debug_print(comp_id, message, data)

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: list[tuple[float, float]] | None = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        raise NotImplementedError

    def evaluate_at(
        self,
        t: float,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """(point, tangent) at parameter t in [0,1]. Default: linear interpolation."""
        px = start[0] + t * (end[0] - start[0])
        py = start[1] + t * (end[1] - start[1])
        tangent = normalize_vector((end[0] - start[0], end[1] - start[1]))
        return (px, py), tangent

    def evaluate_at_distance(
        self,
        d: float,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
        *,
        relative: bool = True,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """(point, tangent) at arc-length distance d.
        relative=True: d in [0,1] fraction of total length. Negative means from end.
        relative=False: d in data units from start. Negative means from end."""
        if relative:
            t = d if d >= 0 else 1.0 + d
            t = max(0.0, min(1.0, t))
            return self.evaluate_at(t, start, end, control_points)
        cumulative = self._cumulative_arc_lengths(start, end, control_points)
        total = cumulative[-1]
        target = max(0.0, total + d) if d < 0 else d
        return self.evaluate_at(
            self._cumulative_to_t(cumulative, target),
            start,
            end,
            control_points,
        )

    def _cumulative_arc_lengths(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
        n_segments: int = 50,
    ) -> list[float]:
        """Cumulative arc lengths at N+1 evenly-spaced parameter values."""
        pts = [
            self.evaluate_at(i / n_segments, start, end, control_points)[0]
            for i in range(n_segments + 1)
        ]
        cumulative = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            cumulative.append(cumulative[-1] + (dx * dx + dy * dy) ** 0.5)
        return cumulative

    @staticmethod
    def _cumulative_to_t(cumulative: list[float], target: float) -> float:
        """Convert absolute arc-length distance to parameter t using a cumulative table."""
        total = cumulative[-1]
        if total < 1e-9:
            return 0.0
        n_segments = len(cumulative) - 1
        target = min(target, total)
        for i in range(1, len(cumulative)):
            if cumulative[i] >= target:
                seg_frac = (target - cumulative[i - 1]) / max(
                    cumulative[i] - cumulative[i - 1], 1e-12
                )
                return ((i - 1) + seg_frac) / n_segments
        return 1.0

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """normalized outward direction vectors at start and end."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        dir_vec = normalize_vector((dx, dy))
        return dir_vec, (-dir_vec[0], -dir_vec[1])

    @staticmethod
    def _calculate_auto_direction_vector(
        start: tuple[float, float], end: tuple[float, float], for_start: bool
    ) -> tuple[float, float]:
        """normalized outward direction vector along start->end (reversed for end)."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        vec = (dx, dy) if for_start else (-dx, -dy)
        return normalize_vector(vec, default=(0, 1))


class StraightCurve(CurveDefinition):
    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: list[tuple[float, float]] | None = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        return f"M {start[0]:.3f} {start[1]:.3f} L {end[0]:.3f} {end[1]:.3f}", []



class SimpleBezierCurve(CurveDefinition):
    start_mode: BezierDirectionMode = "auto"
    end_mode: BezierDirectionMode = "auto"
    start_vector: tuple[float, float] | None = None
    end_vector: tuple[float, float] | None = None

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: list[tuple[float, float]] | None = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        if self.start_mode == "auto" or not self.start_vector:
            auto_dir = self._calculate_auto_direction_vector(start, end, for_start=True)
            sx, sy = (
                auto_dir[0] * self.auto_direction_strength,
                auto_dir[1] * self.auto_direction_strength,
            )
        else:
            sx, sy = self.start_vector
        c1 = (start[0] + sx, start[1] + sy)

        if self.end_mode == "auto" or not self.end_vector:
            auto_dir = self._calculate_auto_direction_vector(start, end, for_start=False)
            ex, ey = (
                auto_dir[0] * self.auto_direction_strength,
                auto_dir[1] * self.auto_direction_strength,
            )
        else:
            ex, ey = self.end_vector
        c2 = (end[0] + ex, end[1] + ey)

        path = f"M {start[0]:.3f} {start[1]:.3f} C {c1[0]:.3f} {c1[1]:.3f}, {c2[0]:.3f} {c2[1]:.3f}, {end[0]:.3f} {end[1]:.3f}"
        control_points = [c1, c2]
        return path, control_points

    def evaluate_at(
        self,
        t: float,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Cubic Bezier B(t) and normalized B'(t)."""
        if len(control_points) < 2:
            return super().evaluate_at(t, start, end, control_points)
        p0, p3 = start, end
        p1, p2 = control_points[0], control_points[1]
        u = 1 - t
        px = u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0]
        py = u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]
        tx = 3 * u * u * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0])
        ty = 3 * u * u * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1])
        tangent = normalize_vector((tx, ty))
        return (px, py), tangent

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if len(control_points) < 2:
            return super().get_directions(start, end, control_points)

        c1, c2 = control_points
        start_dir_out = normalize_vector((c1[0] - start[0], c1[1] - start[1]))
        end_dir_out = normalize_vector((end[0] - c2[0], end[1] - c2[1]))
        return start_dir_out, end_dir_out


class OrthogonalCurve(CurveDefinition):
    start_direction: OrthoDirection = "auto"
    start_length: float = Field(default=10.0, description="min length of the first segment")
    end_direction: OrthoDirection = "auto"
    end_length: float = Field(default=10.0, description="min length of the last segment")
    corner_radius: float = 5.0
    auto_simplify: bool = True

    _resolved_start_dir_vec: tuple[float, float] | None = PrivateAttr(default=None)
    _resolved_end_dir_vec: tuple[float, float] | None = PrivateAttr(default=None)
    _cached_path_points: list[tuple[float, float]] = PrivateAttr(default_factory=list)

    _DIR_VECTORS: ClassVar[dict[ValidOrthoDirection, tuple[float, float]]] = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    _VALID_DIRECTIONS: ClassVar[Sequence[ValidOrthoDirection]] = get_args(ValidOrthoDirection)

    @staticmethod
    def get_direction_from_vector(vector: tuple[float, float]) -> ValidOrthoDirection:
        """closest orthogonal direction name to a vector."""
        if not any(abs(d) > 1e-9 for d in vector):
            return "up"
        norm_vec = normalize_vector(vector, default=(0, 1))
        dir_vectors = np.array(list(OrthogonalCurve._DIR_VECTORS.values()))
        dot_products = np.dot(dir_vectors, norm_vec)
        best_match_index = np.argmax(dot_products)
        return OrthogonalCurve._VALID_DIRECTIONS[best_match_index]

    def _resolve_direction_vector(
        self,
        direction_mode: OrthoDirection,
        start: tuple[float, float],
        end: tuple[float, float],
        for_start: bool,
    ) -> tuple[float, float]:
        """resolve direction mode to a normalized outward vector (away from anchor)."""
        if direction_mode == "auto":
            auto_vec = self._calculate_auto_direction_vector(start, end, for_start)
            resolved_name = self.get_direction_from_vector(auto_vec)
            return self._DIR_VECTORS[resolved_name]
        elif direction_mode in self._DIR_VECTORS:
            return self._DIR_VECTORS[direction_mode]
        else:
            return self._DIR_VECTORS["up"]

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: list[tuple[float, float]] | None = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        start_dir_out = self._resolve_direction_vector(self.start_direction, start, end, True)
        end_dir_out = self._resolve_direction_vector(self.end_direction, start, end, False)

        points = create_orthogonal_path(
            start,
            end,
            start_dir_out,
            end_dir_out,
            self.start_length,
            self.end_length,
            checkpoints=local_checkpoints or [],
            auto_simplify=self.auto_simplify,
        )

        self._cached_path_points = list(points)

        if self.corner_radius > 1e-3 and len(points) >= 3:
            path_str = create_rounded_orthogonal_path(points, self.corner_radius)
        else:
            path_str = f"M {points[0][0]:.3f} {points[0][1]:.3f}" + "".join(
                f" L {p[0]:.3f} {p[1]:.3f}" for p in points[1:]
            )

        self._resolved_start_dir_vec = start_dir_out
        self._resolved_end_dir_vec = end_dir_out
        return path_str, []

    def evaluate_at(
        self,
        t: float,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Piecewise linear evaluation along cached orthogonal path points."""
        pts = self._cached_path_points
        if len(pts) < 2:
            return super().evaluate_at(t, start, end, control_points)
        seg_lens = []
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            seg_lens.append((dx * dx + dy * dy) ** 0.5)
        total = sum(seg_lens)
        if total < 1e-9:
            return (pts[0], normalize_vector((end[0] - start[0], end[1] - start[1])))
        target = t * total
        cumulative = 0.0
        for i, sl in enumerate(seg_lens):
            if cumulative + sl >= target or i == len(seg_lens) - 1:
                frac = (target - cumulative) / max(sl, 1e-12)
                frac = max(0.0, min(1.0, frac))
                px = pts[i][0] + frac * (pts[i + 1][0] - pts[i][0])
                py = pts[i][1] + frac * (pts[i + 1][1] - pts[i][1])
                tangent = normalize_vector((pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
                return (px, py), tangent
            cumulative += sl
        return (pts[-1], normalize_vector((pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])))

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if self._resolved_start_dir_vec and self._resolved_end_dir_vec:
            return self._resolved_start_dir_vec, self._resolved_end_dir_vec
        start_dir_out = self._resolve_direction_vector(self.start_direction, start, end, True)
        end_dir_out = self._resolve_direction_vector(self.end_direction, start, end, False)
        return start_dir_out, end_dir_out
