"""Shared renderer utilities used by all backends."""

from __future__ import annotations

from typing import Type

import numpy as np

from jeanplot.core.component import Component
from jeanplot.core.models import BoxStyle
from jeanplot.core.svg import SVGPathData

EPSILON = 1e-9


def get_matrix_avg_scale(matrix: np.ndarray) -> float:
    """Return the average affine scale factor for a 3x3 transform matrix."""
    scale_x = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    scale_y = np.sqrt(matrix[0, 1] ** 2 + matrix[1, 1] ** 2)
    avg_scale = (scale_x + scale_y) / 2.0
    return avg_scale if avg_scale > EPSILON else 0.0


def get_mpl_linestyle(path_data: SVGPathData) -> str | tuple[float, tuple[float, ...] | None]:
    """Map SVG path line styling to matplotlib line style values."""
    if path_data.dash_array and path_data.line_style == "custom":
        return (path_data.dash_offset, path_data.dash_array)
    return {"solid": "-", "dashed": "--", "dotted": ":"}.get(path_data.line_style, "-")


def get_svg_dasharray(path_data: SVGPathData) -> str | None:
    """Map SVG path styling to stroke-dasharray text for SVG output."""
    if path_data.dash_array and path_data.line_style == "custom":
        return ",".join(str(d) for d in path_data.dash_array)
    style_map = {"dashed": "5,5", "dotted": "2,2"}
    return style_map.get(path_data.line_style)


def get_mpl_linestyle_from_boxstyle(
    style: BoxStyle,
) -> str | tuple[float, tuple[float, ...] | None]:
    """Map box border style settings to matplotlib line style values."""
    if style.dash_sequence and style.border_style == "custom":
        return (style.dash_offset, style.dash_sequence)
    return {"solid": "-", "dashed": "--", "dotted": ":"}.get(style.border_style, "-")


def get_recursive_world_bounds(
    component: Component,
    *,
    current_bounds: tuple[float, float, float, float] | None = None,
    exclude_types: tuple[Type[object], ...] = (),
) -> tuple[float, float, float, float] | None:
    """Return recursive world bounds for visible scene nodes."""
    if not component or not component.show:
        return current_bounds

    overall = list(current_bounds) if current_bounds else [np.inf, np.inf, -np.inf, -np.inf]

    if not isinstance(component, exclude_types):
        comp_b = component.get_world_bounds()
        if comp_b:
            overall = [
                min(overall[0], comp_b[0]),
                min(overall[1], comp_b[1]),
                max(overall[2], comp_b[2]),
                max(overall[3], comp_b[3]),
            ]

    children_to_check = getattr(component, "children", []) + getattr(component, "anchor_points", [])
    for child in children_to_check:
        if not child or not child.show:
            continue
        child_bounds = get_recursive_world_bounds(
            child,
            current_bounds=None,
            exclude_types=exclude_types,
        )
        if child_bounds:
            overall = [
                min(overall[0], child_bounds[0]),
                min(overall[1], child_bounds[1]),
                max(overall[2], child_bounds[2]),
                max(overall[3], child_bounds[3]),
            ]

    return tuple(overall) if overall[0] != np.inf else None
