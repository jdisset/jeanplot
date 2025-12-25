# File: jeanplot/curve.py
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

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """get normalized direction vectors pointing *outwards* at start and end."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        dir_vec = normalize_vector((dx, dy))
        # start direction points away from start; end direction points away from end
        return dir_vec, (-dir_vec[0], -dir_vec[1])

    @staticmethod
    def _calculate_auto_direction_vector(
        start: tuple[float, float], end: tuple[float, float], for_start: bool
    ) -> tuple[float, float]:
        """
        calculate normalized outward direction vector.
        for start = true, vector points from start towards end.
        for start = false, vector points from end towards start (away from end).
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        vec = (dx, dy) if for_start else (-dx, -dy)  # always outward
        return normalize_vector(vec, default=(0, 1))  # provide default


class StraightCurve(CurveDefinition):
    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: list[tuple[float, float]] | None = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        return f"M {start[0]:.3f} {start[1]:.3f} L {end[0]:.3f} {end[1]:.3f}", []

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        dx, dy = end[0] - start[0], end[1] - start[1]
        dir_vec = normalize_vector((dx, dy))
        return dir_vec, (-dir_vec[0], -dir_vec[1])


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
        # determine start control point vector (points away from start)
        if self.start_mode == "auto" or not self.start_vector:
            auto_dir = self._calculate_auto_direction_vector(start, end, for_start=True)
            sx, sy = (
                auto_dir[0] * self.auto_direction_strength,
                auto_dir[1] * self.auto_direction_strength,
            )
        else:
            sx, sy = self.start_vector
        c1 = (start[0] + sx, start[1] + sy)

        # determine end control point vector (points away from end)
        if self.end_mode == "auto" or not self.end_vector:
            # get outward vector *from end*
            auto_dir = self._calculate_auto_direction_vector(start, end, for_start=False)
            ex, ey = (
                auto_dir[0] * self.auto_direction_strength,
                auto_dir[1] * self.auto_direction_strength,
            )
        else:
            ex, ey = self.end_vector
        c2 = (end[0] + ex, end[1] + ey)  # add vector pointing *away* from end

        path = f"M {start[0]:.3f} {start[1]:.3f} C {c1[0]:.3f} {c1[1]:.3f}, {c2[0]:.3f} {c2[1]:.3f}, {end[0]:.3f} {end[1]:.3f}"
        control_points = [c1, c2]
        return path, control_points

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

    _DIR_VECTORS: ClassVar[dict[ValidOrthoDirection, tuple[float, float]]] = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    _VALID_DIRECTIONS: ClassVar[Sequence[ValidOrthoDirection]] = get_args(ValidOrthoDirection)

    @staticmethod
    def get_direction_from_vector(vector: tuple[float, float]) -> ValidOrthoDirection:
        """get closest orthogonal direction name from a vector."""
        if not any(abs(d) > 1e-9 for d in vector):
            return "up"  # default for zero vector
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
        """
        resolve direction mode ('up', 'auto', etc.) to a normalized outward vector.
        'outward' means pointing away from the anchor point (start or end).
        """
        if direction_mode == "auto":
            # 'auto' direction is derived from the vector between points, always pointing outward.
            auto_vec = self._calculate_auto_direction_vector(start, end, for_start)
            resolved_name = self.get_direction_from_vector(auto_vec)
            return self._DIR_VECTORS[resolved_name]
        elif direction_mode in self._DIR_VECTORS:
            # use the specified direction vector directly (it's already outward)
            return self._DIR_VECTORS[direction_mode]
        else:
            return self._DIR_VECTORS["up"]  # default to 'up'

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: list[tuple[float, float]] | None = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        # calculate the OUTWARD direction vectors for start and end based on curve properties
        start_dir_out = self._resolve_direction_vector(self.start_direction, start, end, True)
        end_dir_out = self._resolve_direction_vector(self.end_direction, start, end, False)


        # pass BOTH outward directions to the path generator
        points = create_orthogonal_path(
            start,
            end,
            start_dir_out,  # pass outward start direction
            end_dir_out,  # pass outward end direction
            self.start_length,
            self.end_length,
            checkpoints=local_checkpoints or [],
            auto_simplify=self.auto_simplify,
        )


        if self.corner_radius > 1e-3 and len(points) >= 3:
            # use the updated path generator for rounded corners
            path_str = create_rounded_orthogonal_path(points, self.corner_radius)
        else:
            path_str = f"M {points[0][0]:.3f} {points[0][1]:.3f}" + "".join(
                f" L {p[0]:.3f} {p[1]:.3f}" for p in points[1:]
            )

        # cache resolved OUTWARD directions for get_directions call
        self._resolved_start_dir_vec = start_dir_out
        self._resolved_end_dir_vec = end_dir_out
        return path_str, []  # orthogonal paths don't have explicit bezier control points

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """get normalized outward direction vectors using cached resolved directions."""
        # retrieve the outward directions calculated during get_path
        if self._resolved_start_dir_vec and self._resolved_end_dir_vec:
            return self._resolved_start_dir_vec, self._resolved_end_dir_vec
        else:
            # fallback calculation if cache wasn't populated
            # this recalculates the outward vectors based on current curve settings
            start_dir_out = self._resolve_direction_vector(self.start_direction, start, end, True)
            end_dir_out = self._resolve_direction_vector(self.end_direction, start, end, False)
            return start_dir_out, end_dir_out
