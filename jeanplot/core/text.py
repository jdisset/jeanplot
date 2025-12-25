# File: jeanplot/text.py
# -*- coding: utf-8 -*-
"""Text component definition."""

from typing import Literal, Any
import numpy as np
from pydantic import PrivateAttr, Field

# use absolute imports
from jeanplot.core.component import Component
from jeanplot.core.models import Size, TextMetrics
from jeanplot.core.debug import get_logger

logger = get_logger(__name__)


class Text(Component):
    """text component rendered using native matplotlib text."""

    text: str = ""
    font_name: str | None = None
    # font_size meaning depends on font_size_mode:
    # - "points": font_size is in typographic points (consistent visual size)
    # - "data": font_size is in local data units (scales with container transforms)
    font_size: float = 12.0
    # font_size_mode controls how font_size is interpreted:
    # - "points": font_size is directly in typographic points (72 points = 1 inch)
    #             This gives consistent visual size regardless of axis limits or
    #             container transforms. Use this when you want text to have a fixed
    #             visual appearance.
    # - "data": font_size is in the component's local data units. The rendered
    #           size scales with the component's world transform and axis limits.
    #           Use this when you want text to scale proportionally with its container.
    font_size_mode: Literal["points", "data"] = "data"
    font_weight: Literal["normal", "bold"] = "normal"
    font_style: Literal["normal", "italic"] = "normal"
    color: str = "black"
    align: Literal["left", "center", "right"] = "left"
    vertical_align: Literal["top", "middle", "bottom", "baseline"] = "middle"
    line_spacing: float = 0.2  # factor of base line height

    # renderers' native text rendering methods are used by default
    # we switch to path rendering if render_as_path is set to True
    # or if there are complex transforms (skew)
    render_as_path: Literal["auto"] | bool = "auto"

    _text_metrics_cache: TextMetrics | None = PrivateAttr(default=None)
    _render_path_cache: Any | None = PrivateAttr(default=None)

    # text components generally don't need a background/border by default
    # explicitly set style to None, jstyle can override if needed
    style: Any = Field(default=None, validate_default=False)

    def _measure_natural(self, renderer) -> Size:
        """
        calculates natural data-unit size based on point metrics
        and target data height (self.font_size).

        For font_size_mode="points": we need to convert from points to data units
        for layout purposes. The actual rendering will use the point size directly.

        For font_size_mode="data": font_size is already in data units.
        """
        if not self.text or not renderer or not hasattr(renderer, "measure_text"):
            self._text_metrics_cache = None
            return Size(0, 0)

        renderer.measure_text(self)

        if not self._text_metrics_cache or self._text_metrics_cache.height_points <= 1e-6:
            self._log_debug("warning: text metrics measurement failed or yielded zero height.")
            return Size(0, 0)

        nlines = len(self.text.split("\n"))
        point_height = self._text_metrics_cache.height_points
        point_width = self._text_metrics_cache.width_points

        if self.font_size_mode == "points":
            # font_size is in points - use it directly for point calculations
            # For layout, we need to convert to data units using renderer's context
            target_point_height = self.font_size * nlines * 1.2
            # Get points per data unit from renderer (if available)
            points_per_data_unit = getattr(renderer, '_points_per_data_unit', 1.0)
            if points_per_data_unit > 1e-6:
                target_data_height = target_point_height / points_per_data_unit
            else:
                target_data_height = target_point_height  # fallback
            scale_factor = target_point_height / point_height if point_height > 1e-6 else 0.0
            natural_data_width = (point_width * scale_factor) / points_per_data_unit if points_per_data_unit > 1e-6 else 0.0
        else:
            # font_size_mode == "data" - original behavior
            target_data_height = self.font_size * nlines * 1.2
            scale_factor = target_data_height / point_height if point_height > 1e-6 else 0.0
            natural_data_width = point_width * scale_factor

        natural_size = Size(width=natural_data_width, height=target_data_height)
        return natural_size

    def has_effective_skew(self, world_matrix: np.ndarray) -> bool:
        """
        detects skew by checking if the transformed axes are non-orthogonal.
        """
        EPSILON = 1e-6

        # extract the 2x2 transformation part (rotation, scale, skew)
        m2x2 = world_matrix[:2, :2]

        # get transformed basis vectors (columns of the 2x2 matrix)
        col1 = m2x2[:, 0]  # transformed x-axis
        col2 = m2x2[:, 1]  # transformed y-axis

        len1_sq = np.dot(col1, col1)
        len2_sq = np.dot(col2, col2)

        # if either axis is scaled to zero length, treat as no skew
        if len1_sq < EPSILON or len2_sq < EPSILON:
            return False

        dot_product = np.dot(col1, col2)

        # check if the normalized dot product (cosine of angle between axes) is significantly non-zero
        normalized_dot_sq = (dot_product * dot_product) / (len1_sq * len2_sq)
        skew_tolerance = 1e-3
        has_skew = normalized_dot_sq > skew_tolerance

        return has_skew

    def render(self, renderer, context: Any, matrix: np.ndarray):
        """render text using the appropriate method based on render_as_path."""
        if not self.show or not self.text:
            return

        if not self._text_metrics_cache:
            if not hasattr(renderer, "measure_text"):
                if self.debug:
                    renderer.render_debug(context, self, matrix)
                return
            renderer.measure_text(self)
            if not self._text_metrics_cache:
                if self.debug:
                    renderer.render_debug(context, self, matrix)
                return

        if self.render_as_path is True or (
            self.render_as_path == "auto" and self.has_effective_skew(matrix)
        ):
            if hasattr(renderer, "_render_text_as_paths"):
                renderer._render_text_as_paths(context, self, matrix)
            else:
                logger.warning(f"renderer does not support _render_text_as_paths for {self.id}")
                renderer.render_text(context, self, matrix)
        else:
            renderer.render_text(context, self, matrix)

        if self.debug:
            renderer.render_debug(context, self, matrix)
