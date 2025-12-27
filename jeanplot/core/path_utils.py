# File: jeanplot/path_utils.py
# -*- coding: utf-8 -*-
"""utility functions for component path finding and geometric path calculations."""

from typing import Sequence
import numpy as np
import logging

# use absolute imports
from jeanplot.core.component import Component
from jeanplot.core.debug import debug_print  # Corrected import if needed, or use logging

logger = logging.getLogger(__name__)

# constants
EPSILON = 1e-6  # tolerance for float comparisons


# --- vector helpers ---
# (normalize_vector, is_collinear, is_on_segment remain the same)
def normalize_vector(v: tuple[float, float], default=(1.0, 0.0)) -> tuple[float, float]:
    """normalize a 2d vector, with fallback for zero vector."""
    norm = np.linalg.norm(v)
    return tuple(np.array(v) / norm) if norm > EPSILON else default


def is_collinear(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, tol: float = EPSILON) -> bool:
    """check if three points are collinear using cross product, robust to identical points."""
    if np.allclose(p1, p2, atol=tol) or np.allclose(p2, p3, atol=tol):
        return True
    # use area of parallelogram which is twice the triangle area
    cross_product_mag_sq = (
        (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
    ) ** 2
    # compare squared magnitudes to avoid sqrt
    seg1_sq = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
    seg2_sq = (p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2
    base_sq = seg1_sq * seg2_sq  # use product of segment lengths squared

    # check if cross product is near zero relative to segment lengths
    # handles cases where points are very close but not identical
    if base_sq < tol * tol:  # avoid division by zero if segments are tiny
        return cross_product_mag_sq < tol * tol  # Check absolute tolerance if base is near zero
    # Check relative tolerance otherwise
    return cross_product_mag_sq / base_sq < tol * tol


def is_on_segment(
    p_test: np.ndarray, p_start: np.ndarray, p_end: np.ndarray, tol: float = EPSILON
) -> bool:
    """check if p_test lies on the segment between p_start and p_end."""
    if not is_collinear(p_start, p_test, p_end, tol):
        return False
    # check dot product to ensure it's between start and end
    dot_product = (p_test[0] - p_start[0]) * (p_end[0] - p_start[0]) + (p_test[1] - p_start[1]) * (
        p_end[1] - p_start[1]
    )
    squared_length = (p_end[0] - p_start[0]) ** 2 + (p_end[1] - p_start[1]) ** 2

    # Use tolerance when comparing dot product
    if dot_product < -tol * tol:
        return False  # outside start point (allow slight overshoot)
    if dot_product > squared_length + tol * tol:
        return False  # outside end point (allow slight overshoot)
    return True


# --- component path finding ---
def _find_recursive(root: Component, target_id: str) -> Component | None:
    """Recursively search for a component with the given id anywhere in the tree."""
    if getattr(root, "id", None) == target_id:
        return root
    children = []
    if hasattr(root, "children"):
        children.extend(root.children)
    if hasattr(root, "anchor_points"):
        children.extend([a for a in root.anchor_points if a not in children])
    for child in children:
        found = _find_recursive(child, target_id)
        if found:
            return found
    return None


def find_component_by_path(root: Component, path: str) -> Component | None:
    """Find component by path relative to root.

    Supports:
    - "parent/child/grandchild" - direct descent
    - "//component_id/child" - recursive search for first component, then descend
    """
    if not path:
        return None

    # Handle // prefix (recursive search for first component)
    if path.startswith("//"):
        path = path[2:]  # strip //
        parts = [p for p in path.split("/") if p]
        if not parts:
            return None
        # Recursively find the first component
        current = _find_recursive(root, parts[0])
        if not current:
            return None
        parts = parts[1:]  # remaining parts for direct descent
    else:
        parts = [p for p in path.split("/") if p]
        current = root

    # Direct descent for remaining parts
    for i, part_id in enumerate(parts):
        children = []
        if hasattr(current, "children"):
            children.extend(current.children)
        if hasattr(current, "anchor_points"):
            children.extend([a for a in current.anchor_points if a not in children])

        found = next((c for c in children if getattr(c, "id", None) == part_id), None)
        if found is None:
            current_id = getattr(current, "id", "<no_id>")
            avail_ids = [getattr(c, "id", None) for c in children if c]
            raise ValueError(
                f"comp '{part_id}' not in children/anchors ({avail_ids}) of '{current_id}' at part {i + 1} of '{path}'"
            )
        current = found
    return current


# --- orthogonal path creation ---


def _simplify_orthogonal_path(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """removes redundant points from an orthogonal path."""
    if len(points) < 3:
        return points
    simplified = [points[0]]
    for i in range(1, len(points) - 1):
        p1, p2, p3 = np.array(simplified[-1]), np.array(points[i]), np.array(points[i + 1])
        if np.allclose(p1, p2, atol=EPSILON) or np.allclose(p2, p3, atol=EPSILON):
            continue  # skip duplicate points
        if not is_collinear(p1, p2, p3) or not is_on_segment(p2, p1, p3):
            simplified.append(points[i])
    # always add the last point if it's different from the current last simplified point
    if not simplified or not np.allclose(
        np.array(simplified[-1]), np.array(points[-1]), atol=EPSILON
    ):
        simplified.append(points[-1])

    # run simplification again if points were removed, until stable
    return simplified if len(simplified) == len(points) else _simplify_orthogonal_path(simplified)


def _generate_basic_orthogonal_path(
    start_np: np.ndarray,
    end_np: np.ndarray,
    start_dir_out: np.ndarray,
    end_dir_out: np.ndarray,
    start_len: float,
    end_len: float,
) -> list[tuple[float, float]]:
    """
    Generates a simple orthogonal path (0, 1, or 2 turns typically).
    Prioritizes direct connection between extended points (p1, p2).
    Expects OUTWARD directions for both start and end.
    """
    func_name = "_generate_basic_orthogonal_path"  # for debug_print

    # --- Coincident Check ---
    if np.allclose(start_np, end_np, atol=EPSILON):
        return [tuple(start_np), tuple(end_np)]  # coincident

    # --- Try 0-Turn Path ---
    vec_se = end_np - start_np
    dist_se = np.linalg.norm(vec_se)
    if dist_se > EPSILON:  # Avoid division by zero if dist is tiny
        vec_se_norm = vec_se / dist_se
        is_ortho = abs(vec_se_norm[0]) < EPSILON or abs(vec_se_norm[1]) < EPSILON
        if is_ortho:
            # Check if segment matches *both* outward directions
            if dist_se >= start_len - EPSILON and dist_se >= end_len - EPSILON:
                if (
                    np.dot(vec_se_norm, start_dir_out) > 1.0 - EPSILON
                    and np.dot(-vec_se_norm, end_dir_out) > 1.0 - EPSILON
                ):
                    debug_print(func_name, "found 0-turn path")
                    return [tuple(start_np), tuple(end_np)]

    # --- Try 1-Turn Path ---
    corners = [np.array((end_np[0], start_np[1])), np.array((start_np[0], end_np[1]))]  # H-V, V-H
    for corner in corners:
        v_sc = corner - start_np
        v_ce = end_np - corner
        len_sc = np.linalg.norm(v_sc)
        len_ce = np.linalg.norm(v_ce)

        if len_sc < EPSILON or len_ce < EPSILON:
            continue  # zero length segment

        is_ortho_turn = (abs(v_sc[0]) < EPSILON and abs(v_ce[1]) < EPSILON) or (
            abs(v_sc[1]) < EPSILON and abs(v_ce[0]) < EPSILON
        )

        if is_ortho_turn and len_sc >= start_len - EPSILON and len_ce >= end_len - EPSILON:
            norm_sc = v_sc / len_sc
            norm_ce = v_ce / len_ce
            # Check directions
            if (
                np.dot(norm_sc, start_dir_out) > 1.0 - EPSILON
                and np.dot(norm_ce, -end_dir_out) > 1.0 - EPSILON
            ):
                debug_print(func_name, f"found 1-turn path via {tuple(corner)}")
                return [tuple(start_np), tuple(corner), tuple(end_np)]

    # --- Calculate Extended Points ---
    p1 = start_np + start_dir_out * start_len
    p2 = end_np + end_dir_out * end_len  # point away from end

    # --- Try Direct Connection between p1 and p2 (1 Turn relative to p1/p2) ---
    vec_p1_p2 = p2 - p1
    is_p1p2_ortho = abs(vec_p1_p2[0]) < EPSILON or abs(vec_p1_p2[1]) < EPSILON
    if is_p1p2_ortho and np.linalg.norm(vec_p1_p2) > EPSILON:
        # Check if segment p1->p2 aligns with the direction needed *to reach* p2 from p1
        # And check if segment p2->end aligns with the negative of end_dir_out
        vec_p2_end = end_np - p2
        if np.linalg.norm(vec_p2_end) > EPSILON:  # Check if p2 is not coincident with end
            norm_p2_end = normalize_vector(tuple(vec_p2_end))
            if np.dot(norm_p2_end, -end_dir_out) > 1.0 - EPSILON:
                debug_print(func_name, "found direct p1->p2 connection path (2 turns total)")
                points = [start_np, p1, p2, end_np]
                return _simplify_orthogonal_path([tuple(p) for p in points])
        # If p2 is coincident with end, path is simply start -> p1 -> end
        elif np.linalg.norm(p1 - end_np) > EPSILON:  # Check if p1 is not coincident with end
            debug_print(func_name, "found direct p1->end connection path (1 turn total)")
            points = [start_np, p1, end_np]
            return _simplify_orthogonal_path([tuple(p) for p in points])

    # --- Default 2-Turn C-Shape Path (Fallback) ---
    debug_print(func_name, "using default 2-turn C-shape path")
    is_vertical_start = abs(start_dir_out[1]) > 0.5
    corner = np.array((p2[0], p1[1])) if is_vertical_start else np.array((p1[0], p2[1]))
    points = [start_np, p1, corner, p2, end_np]
    return _simplify_orthogonal_path([tuple(p) for p in points])


def create_orthogonal_path(
    start: tuple[float, float],
    end: tuple[float, float],
    start_dir_out: tuple[float, float],
    end_dir_out: tuple[float, float],
    start_length: float = 0.0,
    end_length: float = 0.0,
    checkpoints: Sequence[tuple[float, float]] = [],
    auto_simplify: bool = True,
) -> list[tuple[float, float]]:
    """generates orthogonal path points. expects OUTWARD directions for both start and end."""
    start_np, end_np = np.array(start), np.array(end)
    start_dir_out_np, end_dir_out_np = np.array(start_dir_out), np.array(end_dir_out)

    if np.allclose(start_np, end_np, atol=EPSILON):
        return [start, end]

    if checkpoints:
        nodes = [start_np] + [np.array(cp) for cp in checkpoints] + [end_np]
        path_segments: list[list[tuple[float, float]]] = []
        current_start_dir = start_dir_out_np
        current_start_len = start_length

        for i in range(len(nodes) - 1):
            is_last_segment = i == len(nodes) - 2
            # Determine end direction for this segment
            # Only use the connection's end_dir_out for the very last segment
            if is_last_segment:
                current_end_dir = end_dir_out_np
                current_end_len = end_length
            else:
                # For intermediate segments, infer the outward direction from the
                # *next* segment's start point (nodes[i+1]) towards nodes[i+2]
                # If it's the second to last node, infer based on end_np
                next_node = nodes[i + 2] if (i + 2) < len(nodes) else end_np
                vec_to_next = next_node - nodes[i + 1]
                # Approximate outward direction by reversing the vector towards next
                # Or simply pass a zero vector if inference is complex/unreliable
                current_end_dir = (
                    np.array(normalize_vector(tuple(-vec_to_next), default=(0, 0)))
                    if np.linalg.norm(vec_to_next) > EPSILON
                    else np.array([0.0, 0.0])
                )
                current_end_len = 0.0  # No min length for intermediate ends

            segment_points_tuples = _generate_basic_orthogonal_path(
                nodes[i],
                nodes[i + 1],
                current_start_dir,
                current_end_dir,  # Pass the determined outward end dir
                current_start_len,
                current_end_len,
            )

            # Store points excluding the last one for all but the final segment
            path_segments.append(
                segment_points_tuples[:-1] if not is_last_segment else segment_points_tuples
            )

            # Determine start direction for the *next* segment from the end of this one
            if not is_last_segment and len(segment_points_tuples) >= 2:
                last_vec = np.array(segment_points_tuples[-1]) - np.array(segment_points_tuples[-2])
                # Check if last_vec is non-zero before normalizing
                if np.linalg.norm(last_vec) > EPSILON:
                    current_start_dir = np.array(normalize_vector(tuple(last_vec), default=(1, 0)))
                # else keep the previous current_start_dir if segment was zero length? Or default? Default is safer.
                else:
                    current_start_dir = np.array(
                        [1.0, 0.0]
                    )  # Default if last segment was zero length
            current_start_len = 0  # Only first segment has start_length constraint

        # combine segments
        combined_path = [p for segment in path_segments for p in segment]
        # Final simplify call on the fully combined path
        final_path = _simplify_orthogonal_path(combined_path) if auto_simplify else combined_path
        return final_path

    else:
        # standard path without checkpoints - already simplified by _generate_basic_orthogonal_path
        return _generate_basic_orthogonal_path(
            start_np, end_np, start_dir_out_np, end_dir_out_np, start_length, end_length
        )


# --- rounded corners ---
# (create_rounded_orthogonal_path remains the same)
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

        # check if vectors are non-zero and orthogonal
        is_orthogonal_corner = (
            l1 > EPSILON and l2 > EPSILON and abs(np.dot(v1, v2)) < EPSILON * l1 * l2
        )

        if is_orthogonal_corner:
            r = min(radius, l1 / 2.0, l2 / 2.0)  # effective radius
            if r > EPSILON:
                n1, n2 = v1 / l1, v2 / l2
                arc_start = tuple(p_curr - n1 * r)
                arc_end = tuple(p_curr + n2 * r)
                # calculate cross product z-component: (v1 x v2)_z = v1_x * v2_y - v1_y * v2_x
                cross_z = v1[0] * v2[1] - v1[1] * v2[0]
                # svg sweep flag: 1 for positive angle (ccw), 0 for negative angle (cw)
                sweep = 1 if cross_z > 0 else 0
                # add line to start of arc, then the arc command
                path += f" L {arc_start[0]:.3f} {arc_start[1]:.3f}"
                path += f" A {r:.3f} {r:.3f} 0 0 {sweep} {arc_end[0]:.3f} {arc_end[1]:.3f}"
                continue  # skip default line to p_curr as the arc replaces it

        # add sharp corner or segment if not rounding or not orthogonal
        path += f" L {p_curr[0]:.3f} {p_curr[1]:.3f}"

    # add final segment to the last point
    path += f" L {points[-1][0]:.3f} {points[-1][1]:.3f}"
    return path
