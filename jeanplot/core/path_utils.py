from typing import Sequence
import numpy as np

from jeanplot.core.debug import debug_print

EPSILON = 1e-6


def normalize_vector(v: tuple[float, float], default=(1.0, 0.0)) -> tuple[float, float]:
    """normalize a 2d vector, with fallback for zero vector."""
    norm = np.linalg.norm(v)
    return tuple(np.array(v) / norm) if norm > EPSILON else default


def is_collinear(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, tol: float = EPSILON) -> bool:
    """check three points for collinearity via cross product, robust to identical points."""
    if np.allclose(p1, p2, atol=tol) or np.allclose(p2, p3, atol=tol):
        return True
    cross_product_mag_sq = (
        (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
    ) ** 2
    seg1_sq = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
    seg2_sq = (p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2
    base_sq = seg1_sq * seg2_sq

    if base_sq < tol * tol:
        return cross_product_mag_sq < tol * tol
    return cross_product_mag_sq / base_sq < tol * tol


def is_on_segment(
    p_test: np.ndarray, p_start: np.ndarray, p_end: np.ndarray, tol: float = EPSILON
) -> bool:
    """check if p_test lies on the segment between p_start and p_end."""
    if not is_collinear(p_start, p_test, p_end, tol):
        return False
    dot_product = (p_test[0] - p_start[0]) * (p_end[0] - p_start[0]) + (p_test[1] - p_start[1]) * (
        p_end[1] - p_start[1]
    )
    squared_length = (p_end[0] - p_start[0]) ** 2 + (p_end[1] - p_start[1]) ** 2

    if dot_product < -tol * tol:
        return False
    if dot_product > squared_length + tol * tol:
        return False
    return True


def _simplify_orthogonal_path(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """remove redundant points from an orthogonal path."""
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

    # recurse until stable
    return simplified if len(simplified) == len(points) else _simplify_orthogonal_path(simplified)


def _generate_basic_orthogonal_path(
    start_np: np.ndarray,
    end_np: np.ndarray,
    start_dir_out: np.ndarray,
    end_dir_out: np.ndarray,
    start_len: float,
    end_len: float,
) -> list[tuple[float, float]]:
    """generate simple orthogonal path (0, 1, or 2 turns typically). expects OUTWARD directions."""
    func_name = "_generate_basic_orthogonal_path"

    if np.allclose(start_np, end_np, atol=EPSILON):
        return [tuple(start_np), tuple(end_np)]

    # try 0-turn path
    vec_se = end_np - start_np
    dist_se = np.linalg.norm(vec_se)
    if dist_se > EPSILON:
        vec_se_norm = vec_se / dist_se
        is_ortho = abs(vec_se_norm[0]) < EPSILON or abs(vec_se_norm[1]) < EPSILON
        if is_ortho:
            if dist_se >= start_len - EPSILON and dist_se >= end_len - EPSILON:
                if (
                    np.dot(vec_se_norm, start_dir_out) > 1.0 - EPSILON
                    and np.dot(-vec_se_norm, end_dir_out) > 1.0 - EPSILON
                ):
                    debug_print(func_name, "found 0-turn path")
                    return [tuple(start_np), tuple(end_np)]

    # try 1-turn path
    corners = [np.array((end_np[0], start_np[1])), np.array((start_np[0], end_np[1]))]
    for corner in corners:
        v_sc = corner - start_np
        v_ce = end_np - corner
        len_sc = np.linalg.norm(v_sc)
        len_ce = np.linalg.norm(v_ce)

        if len_sc < EPSILON or len_ce < EPSILON:
            continue

        is_ortho_turn = (abs(v_sc[0]) < EPSILON and abs(v_ce[1]) < EPSILON) or (
            abs(v_sc[1]) < EPSILON and abs(v_ce[0]) < EPSILON
        )

        if is_ortho_turn and len_sc >= start_len - EPSILON and len_ce >= end_len - EPSILON:
            norm_sc = v_sc / len_sc
            norm_ce = v_ce / len_ce
            if (
                np.dot(norm_sc, start_dir_out) > 1.0 - EPSILON
                and np.dot(norm_ce, -end_dir_out) > 1.0 - EPSILON
            ):
                debug_print(func_name, f"found 1-turn path via {tuple(corner)}")
                return [tuple(start_np), tuple(corner), tuple(end_np)]

    p1 = start_np + start_dir_out * start_len
    p2 = end_np + end_dir_out * end_len

    # try direct connection between p1 and p2 (1 turn relative to p1/p2)
    vec_p1_p2 = p2 - p1
    is_p1p2_ortho = abs(vec_p1_p2[0]) < EPSILON or abs(vec_p1_p2[1]) < EPSILON
    if is_p1p2_ortho and np.linalg.norm(vec_p1_p2) > EPSILON:
        vec_p2_end = end_np - p2
        if np.linalg.norm(vec_p2_end) > EPSILON:
            norm_p2_end = normalize_vector(tuple(vec_p2_end))
            if np.dot(norm_p2_end, -end_dir_out) > 1.0 - EPSILON:
                debug_print(func_name, "found direct p1->p2 connection path (2 turns total)")
                points = [start_np, p1, p2, end_np]
                return _simplify_orthogonal_path([tuple(p) for p in points])
        elif np.linalg.norm(p1 - end_np) > EPSILON:
            debug_print(func_name, "found direct p1->end connection path (1 turn total)")
            points = [start_np, p1, end_np]
            return _simplify_orthogonal_path([tuple(p) for p in points])

    # fallback: default 2-turn C-shape
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
    """generate orthogonal path points. expects OUTWARD directions for both start and end."""
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
            # only use the connection's end_dir_out for the very last segment;
            # for intermediate segments, infer outward direction from the next node
            if is_last_segment:
                current_end_dir = end_dir_out_np
                current_end_len = end_length
            else:
                next_node = nodes[i + 2] if (i + 2) < len(nodes) else end_np
                vec_to_next = next_node - nodes[i + 1]
                current_end_dir = (
                    np.array(normalize_vector(tuple(-vec_to_next), default=(0, 0)))
                    if np.linalg.norm(vec_to_next) > EPSILON
                    else np.array([0.0, 0.0])
                )
                current_end_len = 0.0

            segment_points_tuples = _generate_basic_orthogonal_path(
                nodes[i],
                nodes[i + 1],
                current_start_dir,
                current_end_dir,
                current_start_len,
                current_end_len,
            )

            path_segments.append(
                segment_points_tuples[:-1] if not is_last_segment else segment_points_tuples
            )

            if not is_last_segment and len(segment_points_tuples) >= 2:
                last_vec = np.array(segment_points_tuples[-1]) - np.array(segment_points_tuples[-2])
                if np.linalg.norm(last_vec) > EPSILON:
                    current_start_dir = np.array(normalize_vector(tuple(last_vec), default=(1, 0)))
                else:
                    current_start_dir = np.array([1.0, 0.0])
            current_start_len = 0  # only first segment has start_length constraint

        combined_path = [p for segment in path_segments for p in segment]
        final_path = _simplify_orthogonal_path(combined_path) if auto_simplify else combined_path
        return final_path

    else:
        return _generate_basic_orthogonal_path(
            start_np, end_np, start_dir_out_np, end_dir_out_np, start_length, end_length
        )


def create_rounded_orthogonal_path(points: list[tuple[float, float]], radius: float) -> str:
    """SVG path string with rounded corners for an orthogonal path."""
    if len(points) < 2 or radius <= EPSILON:
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

        is_orthogonal_corner = (
            l1 > EPSILON and l2 > EPSILON and abs(np.dot(v1, v2)) < EPSILON * l1 * l2
        )

        if is_orthogonal_corner:
            r = min(radius, l1 / 2.0, l2 / 2.0)
            if r > EPSILON:
                n1, n2 = v1 / l1, v2 / l2
                arc_start = tuple(p_curr - n1 * r)
                arc_end = tuple(p_curr + n2 * r)
                # cross product z-component: (v1 x v2)_z = v1_x * v2_y - v1_y * v2_x
                cross_z = v1[0] * v2[1] - v1[1] * v2[0]
                # svg sweep flag: 1 for ccw, 0 for cw
                sweep = 1 if cross_z > 0 else 0
                path += f" L {arc_start[0]:.3f} {arc_start[1]:.3f}"
                path += f" A {r:.3f} {r:.3f} 0 0 {sweep} {arc_end[0]:.3f} {arc_end[1]:.3f}"
                continue

        path += f" L {p_curr[0]:.3f} {p_curr[1]:.3f}"

    path += f" L {points[-1][0]:.3f} {points[-1][1]:.3f}"
    return path
