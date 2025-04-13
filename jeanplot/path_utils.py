"""Utility functions for component path finding and geometric path calculations."""

from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np
import logging

# use absolute imports
from jeanplot.component import Component  # direct import ok here
from jeanplot.debug import debug_print

logger = logging.getLogger(__name__)


def normalize_vector(v: Tuple[float, float], default=(1.0, 0.0)) -> Tuple[float, float]:
    """normalize a 2d vector, with fallback for zero vector."""
    norm = np.linalg.norm(v)
    return tuple(np.array(v) / norm) if norm > 1e-9 else default


def find_component_by_path(root: Component, path: str) -> Optional[Component]:
    """find component by path relative to root container (e.g., "parent/child/grandchild")."""
    if not path:
        return None
    parts = [p for p in path.split("/") if p]  # handle empty parts from //
    current = root

    for i, part_id in enumerate(parts):
        found = None
        # check children first
        if hasattr(current, "children") and current.children:
            found = next(
                (child for child in current.children if getattr(child, "id", None) == part_id), None
            )

        # if not in children, check anchor_points
        if found is None and hasattr(current, "anchor_points") and current.anchor_points:
            found = next(
                (
                    anchor
                    for anchor in current.anchor_points
                    if getattr(anchor, "id", None) == part_id
                ),
                None,
            )

        if found is None:
            current_id = getattr(current, "id", "<no_id>")
            available_children_ids = [
                getattr(c, "id", None) for c in getattr(current, "children", []) if c
            ]
            available_anchor_ids = [
                getattr(a, "id", None) for a in getattr(current, "anchor_points", []) if a
            ]
            raise ValueError(
                f"component '{part_id}' not found in children ({available_children_ids}) "
                f"or anchors ({available_anchor_ids}) of '{current_id}' at part {i+1} of path '{path}'"
            )
        current = found

    return current


# --- Orthogonal Path Geometry ---


def _can_connect_directly(p1, p2, dir1_out, dir2_out):
    """check if a straight line respects outward directions."""
    line_vec = (p2[0] - p1[0], p2[1] - p1[1])
    if np.linalg.norm(line_vec) < 1e-9:
        return True  # coincident points
    line_norm = normalize_vector(line_vec)
    # check if start direction allows moving towards end (dot product >= 0)
    dot1 = np.dot(line_norm, dir1_out)
    # check if end direction allows moving from start (dot product of line and *inward* end dir >= 0)
    dot2 = np.dot(line_norm, (-dir2_out[0], -dir2_out[1]))
    return dot1 >= -1e-6 and dot2 >= -1e-6  # use tolerance


def _find_single_corner(start, end, start_dir_out, end_dir_out, start_length, end_length):
    """find a valid single corner point, respecting directions/lengths."""
    corners_to_check = [
        (start[0], end[1]),  # vertical start segment, horizontal end segment
        (end[0], start[1]),  # horizontal start segment, vertical end segment
    ]

    for c in corners_to_check:
        v_start_c = (c[0] - start[0], c[1] - start[1])
        v_c_end = (end[0] - c[0], end[1] - c[1])
        len_start_c = np.linalg.norm(v_start_c)
        len_c_end = np.linalg.norm(v_c_end)

        # check alignment with directions and length constraints
        start_dir_match = (
            np.dot(normalize_vector(v_start_c, default=(0, 0)), start_dir_out) >= -1e-6
        )
        end_dir_match = (
            np.dot(normalize_vector(v_c_end, default=(0, 0)), (-end_dir_out[0], -end_dir_out[1]))
            >= -1e-6
        )  # compare with inward end dir

        start_len_match = len_start_c >= start_length - 1e-6
        end_len_match = len_c_end >= end_length - 1e-6

        # check if segments are orthogonal
        is_orthogonal = abs(v_start_c[0] * v_c_end[0] + v_start_c[1] * v_c_end[1]) < 1e-6

        # check if corner is distinct from start/end
        is_distinct = len_start_c > 1e-6 and len_c_end > 1e-6

        if (
            start_dir_match
            and end_dir_match
            and start_len_match
            and end_len_match
            and is_orthogonal
            and is_distinct
        ):
            return c

    return None


def find_simple_path(start, end, start_dir_out, end_dir_out, start_length=0, end_length=0):
    """try to find a path with 0 or 1 turn."""
    is_aligned = abs(start[0] - end[0]) < 1e-6 or abs(start[1] - end[1]) < 1e-6

    if is_aligned and _can_connect_directly(start, end, start_dir_out, end_dir_out):
        line_len = np.linalg.norm((end[0] - start[0], end[1] - start[1]))
        if line_len >= start_length - 1e-6 and line_len >= end_length - 1e-6:
            return [start, end]

    # try single corner
    corner = _find_single_corner(start, end, start_dir_out, end_dir_out, start_length, end_length)
    if corner:
        return [start, corner, end]

    return None


def connect_segments(start, first_seg_end, last_seg_start, end):
    """connect fixed start/end segments with minimal orthogonal turns in between."""
    # check if direct connection between middle points is possible
    if (
        abs(first_seg_end[0] - last_seg_start[0]) < 1e-6
        or abs(first_seg_end[1] - last_seg_start[1]) < 1e-6
    ):
        return [start, first_seg_end, last_seg_start, end]

    # introduce one corner: try horizontal first from first_seg_end
    corner1 = (last_seg_start[0], first_seg_end[1])
    if (
        np.linalg.norm(np.array(first_seg_end) - np.array(corner1)) > 1e-6
        and np.linalg.norm(np.array(last_seg_start) - np.array(corner1)) > 1e-6
    ):
        return [start, first_seg_end, corner1, last_seg_start, end]

    # try vertical first from first_seg_end
    corner2 = (first_seg_end[0], last_seg_start[1])
    if (
        np.linalg.norm(np.array(first_seg_end) - np.array(corner2)) > 1e-6
        and np.linalg.norm(np.array(last_seg_start) - np.array(corner2)) > 1e-6
    ):
        return [start, first_seg_end, corner2, last_seg_start, end]

    # fallback (should be rare): just connect middle points directly
    return [start, first_seg_end, last_seg_start, end]


def connect_points(p1, p2):
    """connect two points with minimal orthogonal path (prefers H-then-V)."""
    if abs(p1[0] - p2[0]) < 1e-6 or abs(p1[1] - p2[1]) < 1e-6:
        return [p1, p2]
    corner = (p2[0], p1[1])  # horizontal then vertical
    return [p1, corner, p2]


def path_with_checkpoints(
    checkpoints: list[tuple[float, float]],
    start: tuple[float, float],
    end: tuple[float, float],
    start_dir_out: tuple[float, float],
    end_dir_rev: tuple[float, float],
    start_length: float = 0,
    end_length: float = 0,
):
    """create orthogonal path from start to end through checkpoints."""
    first_segment_end = (
        start[0] + start_dir_out[0] * start_length,
        start[1] + start_dir_out[1] * start_length,
    )
    last_segment_start = (
        end[0] + end_dir_rev[0] * end_length,
        end[1] + end_dir_rev[1] * end_length,
    )

    path = [start, first_segment_end]
    current = first_segment_end
    for cp in checkpoints:
        segments = connect_points(current, cp)
        path.extend(segments[1:])  # add corner (if any) and checkpoint
        current = cp
    segments = connect_points(current, last_segment_start)
    path.extend(segments[1:])  # add corner (if any) and last segment start
    path.append(end)
    return path


def create_rounded_orthogonal_path(points: list[tuple[float, float]], radius: float):
    """create SVG path string with rounded corners for an orthogonal path."""
    if len(points) < 2:
        return ""
    if len(points) == 2 or radius <= 1e-3:  # straight line or sharp corners
        return f"M {points[0][0]} {points[0][1]}" + "".join(f" L {p[0]} {p[1]}" for p in points[1:])

    path = f"M {points[0][0]} {points[0][1]}"
    for i in range(1, len(points) - 1):
        prev, curr, next_pt = np.array(points[i - 1]), np.array(points[i]), np.array(points[i + 1])
        v1 = curr - prev
        v2 = next_pt - curr
        v1_len, v2_len = np.linalg.norm(v1), np.linalg.norm(v2)

        # check for orthogonal turn with non-zero length segments
        is_corner = abs(np.dot(v1, v2)) < 1e-6 and v1_len > 1e-6 and v2_len > 1e-6

        if is_corner:
            segment_max_radius = min(v1_len, v2_len) / 2.0
            r = min(radius, segment_max_radius)

            if r > 1e-3:  # significant radius
                v1_norm = v1 / v1_len
                v2_norm = v2 / v2_len
                arc_start = tuple(curr - v1_norm * r)
                arc_end = tuple(curr + v2_norm * r)
                # determine sweep flag based on cross product sign (z-component)
                cross_z = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
                sweep = 0 if cross_z < 0 else 1
                path += f" L {arc_start[0]} {arc_start[1]} A {r} {r} 0 0 {sweep} {arc_end[0]} {arc_end[1]}"
            else:  # radius too small, draw sharp corner
                path += f" L {curr[0]} {curr[1]}"
        else:  # not a corner, draw straight line
            path += f" L {curr[0]} {curr[1]}"

    # final segment to the last point
    path += f" L {points[-1][0]} {points[-1][1]}"
    return path
