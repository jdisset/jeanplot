from typing import Optional, Literal, get_args
from pydantic import Field, BaseModel
from .path_utils import (
    path_with_checkpoints,
    find_simple_path,
    connect_segments,
    create_rounded_orthogonal_path,
)
import numpy as np


class CurveDefinition(BaseModel):
    """base class for curve definitions"""

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        """generate svg path data and return control points"""
        raise NotImplementedError

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """get direction vectors at start and end"""
        # fallback for straight lines
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dir_vec = (dx / length, dy / length)
            return (-dir_vec[0], -dir_vec[1]), dir_vec
        return (0, -1), (0, 1)


class StraightCurve(CurveDefinition):
    """straight line connection"""

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        return f"M {start[0]} {start[1]} L {end[0]} {end[1]}", []


class SimpleBezierCurve(CurveDefinition):
    """bezier curve with vectors from start and end points"""

    start_vector: tuple[float, float] = (0, 1)  # vector from start point
    end_vector: tuple[float, float] = (0, 1)  # vector from end point

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        c1 = (start[0] + self.start_vector[0], start[1] + self.start_vector[1])
        c2 = (end[0] + self.end_vector[0], end[1] + self.end_vector[1])
        path = f"M {start[0]} {start[1]} C {c1[0]} {c1[1]}, {c2[0]} {c2[1]}, {end[0]} {end[1]}"
        return path, [c1, c2]

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if len(control_points) < 2:
            return super().get_directions(start, end, control_points)

        c1, c2 = control_points

        # start direction
        dx1, dy1 = c1[0] - start[0], c1[1] - start[1]
        len1 = np.sqrt(dx1**2 + dy1**2)
        start_dir = (-dx1 / len1, -dy1 / len1) if len1 > 0 else (0, -1)

        # end direction
        dx2, dy2 = end[0] - c2[0], end[1] - c2[1]
        len2 = np.sqrt(dx2**2 + dy2**2)
        end_dir = (dx2 / len2, dy2 / len2) if len2 > 0 else (0, 1)

        return start_dir, end_dir


class AdvancedBezierCurve(CurveDefinition):
    """bezier curve with explicit control points"""

    control_points: list[tuple[float, float]] = Field(default_factory=list)

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        cps = self.control_points
        if not cps:
            return f"M {start[0]} {start[1]} L {end[0]} {end[1]}", []

        if len(cps) == 1:
            path = f"M {start[0]} {start[1]} Q {cps[0][0]} {cps[0][1]}, {end[0]} {end[1]}"
        else:
            c1, c2 = cps[0], cps[1] if len(cps) > 1 else end
            path = f"M {start[0]} {start[1]} C {c1[0]} {c1[1]}, {c2[0]} {c2[1]}, {end[0]} {end[1]}"

        return path, cps

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if not control_points:
            return super().get_directions(start, end, control_points)

        # start direction
        dx1, dy1 = control_points[0][0] - start[0], control_points[0][1] - start[1]
        len1 = np.sqrt(dx1**2 + dy1**2)
        start_dir = (-dx1 / len1, -dy1 / len1) if len1 > 0 else (0, -1)

        # end direction
        cp_idx = -1 if len(control_points) > 1 else 0
        dx2, dy2 = end[0] - control_points[cp_idx][0], end[1] - control_points[cp_idx][1]
        len2 = np.sqrt(dx2**2 + dy2**2)
        end_dir = (dx2 / len2, dy2 / len2) if len2 > 0 else (0, 1)

        return start_dir, end_dir


OrthoDirection = Literal["up", "down", "left", "right"]


class OrthogonalCurve(CurveDefinition):
    """path with orthogonal segments (right angles)"""

    start_direction: OrthoDirection = "down"
    start_length: float = 5.0  # minimum length of start segment
    end_direction: OrthoDirection = "up"
    end_length: float = 5.0  # minimum length of end segment
    corner_radius: float = 10.0
    checkpoints: list[tuple[float, float]] = Field(default_factory=list)
    auto_simplify: bool = True

    @staticmethod
    def get_direction(direction: tuple[float, float]) -> OrthoDirection:
        dir_vectors = np.array([(0, 1), (0, -1), (-1, 0), (1, 0)])  # up, down, left, right
        closest_idx = np.argmax(np.dot(dir_vectors, direction))
        return get_args(OrthoDirection)[closest_idx]

    def get_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        local_checkpoints: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[str, list[tuple[float, float]]]:
        dir_to_vector = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        start_dir = dir_to_vector[self.start_direction]
        end_dir = dir_to_vector[self.end_direction]
        end_dir_rev = (-end_dir[0], -end_dir[1])  # flip end direction

        checkpoints = local_checkpoints or self.checkpoints
        points = []

        if checkpoints:
            points = path_with_checkpoints(
                checkpoints, start, end, start_dir, end_dir_rev, self.start_length, self.end_length
            )
        elif self.auto_simplify:
            simple_path = find_simple_path(start, end, start_dir, end_dir_rev)
            if simple_path:
                points = simple_path

        if not points:
            # fallback to basic orthogonal path
            first_seg_end = (
                start[0] + start_dir[0] * self.start_length,
                start[1] + start_dir[1] * self.start_length,
            )
            last_seg_start = (
                end[0] + end_dir_rev[0] * self.end_length,
                end[1] + end_dir_rev[1] * self.end_length,
            )
            points = connect_segments(start, first_seg_end, last_seg_start, end)

        if self.corner_radius > 0:
            path = create_rounded_orthogonal_path(points, self.corner_radius)
        else:
            path = f"M {points[0][0]} {points[0][1]}" + "".join(
                f" L {p[0]} {p[1]}" for p in points[1:]
            )

        return path, []

    def get_directions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        control_points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        dir_to_vector = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        start_dir = dir_to_vector[self.start_direction]
        end_dir = dir_to_vector[self.end_direction]
        return (-start_dir[0], -start_dir[1]), end_dir
