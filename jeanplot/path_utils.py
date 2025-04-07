from typing import Optional, Union, Dict, Any, List, Tuple
from .component import Component
from collections import defaultdict
import numpy as np


def find_component_by_path(root: Component, path: str) -> Optional[Component]:
    """
    find component by path relative to root container
    path is a string with "/" separated ids
    """
    if not path:
        return None

    parts = path.split("/")
    current = root

    for part in parts:
        if part == "":
            continue

        if not hasattr(current, "children"):
            return None

        found = False
        for child in current.children:
            if getattr(child, "id", None) == part:
                current = child
                found = True
                break

        if not found:
            allids = [getattr(child, "id", None) for child in current.children]
            raise ValueError(f"Component {part} not found in {allids} at path {path}")

    return current


def resolve_component_ref(v: str | Component, info) -> Component:
    """validator for component references that accepts paths"""

    if isinstance(v, Component):
        return v

    if isinstance(v, str):
        # if we have parent context, try to resolve the path
        values = info.data
        parent = values.get("parent")
        print(f"parent: {parent}")
        if parent is not None:
            # find the root container
            root = parent
            while root.parent is not None:
                root = root.parent

            # try to resolve path
            component = find_component_by_path(root, v)
            if component is not None:
                return component

    raise ValueError(
        f"Invalid component reference: {v}. Must be a Component or a valid path string."
    )


# path utils related to svg paths (lol completely different path than above)


def can_connect_directly(p1, p2, dir1, dir2):
    """Check if two points can be connected with a straight line"""
    if p1[0] == p2[0]:  # vertical
        if dir1[0] != 0 or dir2[0] != 0:
            return False  # must be vertical dirs
        if (p2[1] - p1[1]) * dir1[1] <= 0:
            return False  # start points away
        if (p1[1] - p2[1]) * dir2[1] <= 0:
            return False  # end points away
        return True
    elif p1[1] == p2[1]:  # horizontal
        if dir1[1] != 0 or dir2[1] != 0:
            return False  # must be horizontal dirs
        if (p2[0] - p1[0]) * dir1[0] <= 0:
            return False  # start points away
        if (p1[0] - p2[0]) * dir2[0] <= 0:
            return False  # end points away
        return True
    return False


def find_corner(start, end, start_dir, end_dir, start_length=0, end_length=0):
    """Find corner point for a one-turn path if possible"""
    corners = [
        (
            start[0] + start_dir[0] * max(abs(end[0] - start[0]), start_length),
            end[1] + end_dir[1] * max(abs(end[1] - start[1]), end_length),
        ),
        (
            end[0] + end_dir[0] * max(abs(end[0] - start[0]), end_length),
            start[1] + start_dir[1] * max(abs(end[1] - start[1]), start_length),
        ),
    ]
    for corner in corners:
        can_reach = True
        if start_dir[0] != 0:  # horizontal start
            if (corner[0] - start[0]) * start_dir[0] <= 0:
                can_reach = False
        else:  # vertical start
            if (corner[1] - start[1]) * start_dir[1] <= 0:
                can_reach = False
        if end_dir[0] != 0:  # horizontal end
            if (end[0] - corner[0]) * end_dir[0] <= 0:
                can_reach = False
        else:  # vertical end
            if (end[1] - corner[1]) * end_dir[1] <= 0:
                can_reach = False
        if can_reach:
            return corner
    return None


def find_simple_path(start, end, start_dir, end_dir):
    """Try to find a path with minimal turns (0 or 1)"""
    if start[0] == end[0] or start[1] == end[1]:  # aligned
        if can_connect_directly(start, end, start_dir, end_dir):
            return [start, end]
    corner = find_corner(start, end, start_dir, end_dir)
    if corner:
        return [start, corner, end]
    return None


def connect_segments(start, first_end, last_start, end):
    """Connect points with minimal number of turns"""
    if first_end[0] == last_start[0] or first_end[1] == last_start[1]:
        return [start, first_end, last_start, end]
    corner1 = (first_end[0], last_start[1])
    return [start, first_end, corner1, last_start, end]


def connect_points(p1, p2):
    """Connect two points with minimal orthogonal path"""
    if p1[0] == p2[0] or p1[1] == p2[1]:
        return [p1, p2]
    corner = (p1[0], p2[1])  # simple corner strategy
    return [p1, corner, p2]


def path_with_checkpoints(
    checkpoints: list[tuple[float, float]],
    start: tuple[float, float],
    end: tuple[float, float],
    start_dir: tuple[float, float],
    end_dir: tuple[float, float],
    start_length: float = 0,
    end_length: float = 0,
):
    """Create path from start to end through checkpoints"""
    first_segment_end = (
        start[0] + start_dir[0] * start_length,
        start[1] + start_dir[1] * start_length,
    )
    last_segment_start = (
        end[0] - end_dir[0] * end_length,
        end[1] - end_dir[1] * end_length,
    )
    path = [start, first_segment_end]
    current = first_segment_end
    for cp in checkpoints:
        segments = connect_points(current, cp)
        path.extend(segments[1:])
        current = cp
    segments = connect_points(current, last_segment_start)
    path.extend(segments[1:])
    path.append(end)
    return path


def create_rounded_orthogonal_path(points, radius):
    """Create path string with rounded corners"""

    if len(points) < 3:
        return f"M {points[0][0]} {points[0][1]} L {points[-1][0]} {points[-1][1]}"
    path = f"M {points[0][0]} {points[0][1]}"
    for i in range(1, len(points) - 1):
        prev, curr, next_pt = points[i - 1], points[i], points[i + 1]
        v1 = (curr[0] - prev[0], curr[1] - prev[1])
        v2 = (next_pt[0] - curr[0], next_pt[1] - curr[1])
        is_corner = (v1[0] == 0 and v2[0] != 0) or (v1[0] != 0 and v2[0] == 0)
        if is_corner and radius > 0:
            v1_len = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
            v2_len = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
            max_radius = min(v1_len, v2_len) / 2
            r = min(radius, max_radius)
            if r > 1e-3:
                v1_norm = (v1[0] / v1_len, v1[1] / v1_len) if v1_len > 0 else (0, 0)
                v2_norm = (v2[0] / v2_len, v2[1] / v2_len) if v2_len > 0 else (0, 0)
                arc_start = (curr[0] - v1_norm[0] * r, curr[1] - v1_norm[1] * r)
                arc_end = (curr[0] + v2_norm[0] * r, curr[1] + v2_norm[1] * r)
                cross_z = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
                sweep = 0 if cross_z < 0 else 1
                path += f" L {arc_start[0]} {arc_start[1]} A {r} {r} 0 0 {sweep} {arc_end[0]} {arc_end[1]}"
                continue
        path += f" L {curr[0]} {curr[1]}"
    path += f" L {points[-1][0]} {points[-1][1]}"
    return path
