# File: jeanplot/path_utils.py
# -*- coding: utf-8 -*-
"""utility functions for component path finding and geometric path calculations."""

from typing import Optional, Union, List, Tuple, Sequence
import numpy as np
import logging

# use absolute imports
from jeanplot.component import Component
from jeanplot.debug import debug_print

logger = logging.getLogger(__name__)

# constants
EPSILON = 1e-6  # tolerance for float comparisons

# --- vector helpers ---


def normalize_vector(v: Tuple[float, float], default=(1.0, 0.0)) -> Tuple[float, float]:
    """normalize a 2d vector, with fallback for zero vector."""
    norm = np.linalg.norm(v)
    return tuple(np.array(v) / norm) if norm > EPSILON else default


def is_collinear(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, tol: float = EPSILON) -> bool:
    """check if three points are collinear using cross product, robust to identical points."""
    if np.allclose(p1, p2, atol=tol) or np.allclose(p2, p3, atol=tol):
        return True
    return abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) < tol * tol


def is_on_segment(
    p_test: np.ndarray, p_start: np.ndarray, p_end: np.ndarray, tol: float = EPSILON
) -> bool:
    """check if p_test lies on the segment between p_start and p_end."""
    if not is_collinear(p_start, p_test, p_end, tol):
        return False
    # check bounding box
    return (
        min(p_start[0], p_end[0]) - tol <= p_test[0] <= max(p_start[0], p_end[0]) + tol
        and min(p_start[1], p_end[1]) - tol <= p_test[1] <= max(p_start[1], p_end[1]) + tol
    )


# --- component path finding ---


def find_component_by_path(root: Component, path: str) -> Optional[Component]:
    """find component by path relative to root (e.g., "parent/child/grandchild")."""
    if not path:
        return None
    parts = [p for p in path.split("/") if p]
    current = root
    for i, part_id in enumerate(parts):
        children = getattr(current, "children", []) + getattr(current, "anchor_points", [])
        found = next((c for c in children if getattr(c, "id", None) == part_id), None)
        if found is None:
            current_id = getattr(current, "id", "<no_id>")
            avail_ids = [getattr(c, "id", None) for c in children if c]
            raise ValueError(
                f"comp '{part_id}' not in children/anchors ({avail_ids}) of '{current_id}' at part {i+1} of '{path}'"
            )
        current = found
    return current


# --- orthogonal path creation ---


def _simplify_orthogonal_path(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """removes redundant points from an orthogonal path."""
    if len(points) < 3:
        return points
    simplified = [points[0]]
    for i in range(1, len(points) - 1):
        p1, p2, p3 = np.array(simplified[-1]), np.array(points[i]), np.array(points[i + 1])
        if np.allclose(p1, p2, atol=EPSILON) or np.allclose(p2, p3, atol=EPSILON):
            continue
        if not is_collinear(p1, p2, p3) or not is_on_segment(p2, p1, p3):
            simplified.append(points[i])
    if not simplified or not np.allclose(
        np.array(simplified[-1]), np.array(points[-1]), atol=EPSILON
    ):
        simplified.append(points[-1])
    return simplified if len(simplified) == len(points) else _simplify_orthogonal_path(simplified)


def _generate_basic_orthogonal_path(
    start_np: np.ndarray,
    end_np: np.ndarray,
    start_dir: np.ndarray,
    end_dir: np.ndarray,
    start_len: float,
    end_len: float,
) -> List[Tuple[float, float]]:
    """tries 0, 1, then 2 turn paths."""
    # try 0 turns (straight line)
    vec_se = end_np - start_np
    dist_se = np.linalg.norm(vec_se)
    if abs(vec_se[0]) < EPSILON or abs(vec_se[1]) < EPSILON:  # is orthogonal?
        if dist_se >= start_len - EPSILON and dist_se >= end_len - EPSILON:
            if dist_se < EPSILON:
                return [tuple(start_np), tuple(end_np)]  # coincident
            vec_se_norm = vec_se / dist_se
            if np.dot(vec_se_norm, start_dir) > 0.95 and np.dot(vec_se_norm, end_dir) > 0.95:
                debug_print("_generate_basic_orthogonal_path", "found 0-turn path")
                return [tuple(start_np), tuple(end_np)]

    # try 1 turn (check both corner types)
    corners = [np.array((end_np[0], start_np[1])), np.array((start_np[0], end_np[1]))]  # H-V, V-H
    for corner in corners:
        v_sc = corner - start_np
        v_ce = end_np - corner
        len_sc, len_ce = np.linalg.norm(v_sc), np.linalg.norm(v_ce)
        if len_sc < EPSILON or len_ce < EPSILON:
            continue  # zero length segment

        is_ortho_turn = (
            abs(v_sc[0]) < EPSILON
            and abs(v_ce[1]) < EPSILON  # V-H
            or abs(v_sc[1]) < EPSILON
            and abs(v_ce[0]) < EPSILON
        )  # H-V

        if is_ortho_turn and len_sc >= start_len - EPSILON and len_ce >= end_len - EPSILON:
            norm_sc, norm_ce = v_sc / len_sc, v_ce / len_ce
            if np.dot(norm_sc, start_dir) > 0.95 and np.dot(norm_ce, end_dir) > 0.95:
                debug_print(
                    "_generate_basic_orthogonal_path", f"found 1-turn path via {tuple(corner)}"
                )
                return [tuple(start_np), tuple(corner), tuple(end_np)]

    # generate 2 turn path (C-shape)
    p1 = start_np + start_dir * start_len
    p2 = end_np + end_dir * end_len  # points towards end
    corner = (
        np.array((p2[0], p1[1])) if abs(start_dir[1]) < 0.5 else np.array((p1[0], p2[1]))
    )  # H-V-H vs V-H-V corner
    points = [start_np, p1, corner, p2, end_np]
    tuple_points = [tuple(p) for p in points]
    simplified = _simplify_orthogonal_path(tuple_points)
    debug_print(
        "_generate_basic_orthogonal_path", f"generated {len(simplified)-2}-turn path: {simplified}"
    )
    return simplified


def create_orthogonal_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    start_dir_out: Tuple[float, float],
    end_dir_in: Tuple[float, float],
    start_length: float = 0.0,
    end_length: float = 0.0,
    checkpoints: Sequence[Tuple[float, float]] = [],
    auto_simplify: bool = True,
) -> List[Tuple[float, float]]:
    """generates orthogonal path points. tries 0, 1, then 2 turns unless checkpoints are given."""
    start_np, end_np = np.array(start), np.array(end)
    start_dir, end_dir = np.array(start_dir_out), np.array(end_dir_in)

    if np.allclose(start_np, end_np, atol=EPSILON):
        return [start, end]

    if checkpoints:
        # connect segments sequentially, simplifying at the end
        nodes = [start_np] + [np.array(cp) for cp in checkpoints] + [end_np]
        path = [start_np]
        curr_dir = start_dir
        for i in range(len(nodes) - 1):
            p_curr, p_next = np.array(path[-1]), nodes[i + 1]
            seg_start = p_curr + curr_dir * start_length if i == 0 else p_curr
            seg_end = p_next + end_dir * end_length if i == len(nodes) - 2 else p_next
            corner = (
                np.array((seg_end[0], seg_start[1]))
                if abs(curr_dir[1]) > 0.5
                else np.array((seg_start[0], seg_end[1]))
            )

            if not np.allclose(seg_start, p_curr, atol=EPSILON):
                path.append(seg_start)
            if not np.allclose(corner, seg_start, atol=EPSILON):
                path.append(corner)
            if not np.allclose(seg_end, corner, atol=EPSILON):
                path.append(seg_end)
            if i == len(nodes) - 2 and not np.allclose(end_np, seg_end, atol=EPSILON):
                path.append(end_np)

            # update direction for next segment
            vec_to_end = seg_end - corner
            if np.linalg.norm(vec_to_end) > EPSILON:
                curr_dir = normalize_vector(tuple(vec_to_end))
        tuple_path = [tuple(p) for p in path]
        return _simplify_orthogonal_path(tuple_path)
    else:
        # standard 0, 1, or 2 turn path without checkpoints
        return _generate_basic_orthogonal_path(
            start_np, end_np, start_dir, end_dir, start_length, end_length
        )


# --- rounded corners ---


def create_rounded_orthogonal_path(points: list[tuple[float, float]], radius: float) -> str:
    """create SVG path string with rounded corners for an orthogonal path."""
    if len(points) < 2 or radius <= EPSILON:  # straight line or sharp corners
        return f"M {points[0][0]:.3f} {points[0][1]:.3f}" + "".join(
            f" L {p[0]:.3f} {p[1]:.3f}" for p in points[1:]
        )

    path = f"M {points[0][0]:.3f} {points[0][1]:.3f}"
    for i in range(1, len(points) - 1):
        p_prev, p_curr, p_next = (
            np.array(points[i - 1]),
            np.array(points[i]),
            np.array(points[i + 1]),
        )
        v1, v2 = p_curr - p_prev, p_next - p_curr
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)

        if (
            l1 > EPSILON and l2 > EPSILON and abs(np.dot(v1, v2)) < EPSILON * l1 * l2
        ):  # is orthogonal corner?
            r = min(radius, l1 / 2.0, l2 / 2.0)  # effective radius
            if r > EPSILON:
                n1, n2 = v1 / l1, v2 / l2
                arc_start = tuple(p_curr - n1 * r)
                arc_end = tuple(p_curr + n2 * r)
                sweep = 0  # standard 90-degree outward corner uses sweep=0
                path += f" L {arc_start[0]:.3f} {arc_start[1]:.3f} A {r:.3f} {r:.3f} 0 0 {sweep} {arc_end[0]:.3f} {arc_end[1]:.3f}"
                continue  # skip adding the sharp corner line segment
        # add sharp corner or segment if not rounding
        path += f" L {p_curr[0]:.3f} {p_curr[1]:.3f}"

    path += f" L {points[-1][0]:.3f} {points[-1][1]:.3f}"
    return path
