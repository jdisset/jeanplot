# File: jeanplot/text.py
# -*- coding: utf-8 -*-
"""Text component definition."""

from typing import Optional, Literal, Any
import numpy as np
from pydantic import PrivateAttr, Field

# use absolute imports
from jeanplot.component import Component
from jeanplot.models import Size, TextMetrics
from jeanplot.debug import get_logger
from jeanplot.style import (
    jstyle,
)  # ensure jstyle is imported if needed for default style application

logger = get_logger(__name__)


class Text(Component):
    """text component rendered using native matplotlib text."""

    text: str = ""
    font_name: Optional[str] = None
    # font_size is desired height in data units
    font_size: float = 12.0
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

    _text_metrics_cache: Optional[TextMetrics] = PrivateAttr(default=None)
    _render_path_cache: Optional[Any] = PrivateAttr(default=None)

    # text components generally don't need a background/border by default
    # explicitly set style to None, jstyle can override if needed
    style: Any = Field(default=None, validate_default=False)

    def _measure_natural(self, renderer) -> Size:
        """
        calculates natural data-unit size based on point metrics
        and target data height (self.font_size).
        """
        if not self.text or not renderer or not hasattr(renderer, "measure_text"):
            self._text_metrics_cache = None
            return Size(0, 0)

        point_dims = renderer.measure_text(self)

        if not self._text_metrics_cache or self._text_metrics_cache.height_points <= 1e-6:
            self._log_debug("warning: text metrics measurement failed or yielded zero height.")
            return Size(0, 0)

        nlines = len(self.text.split("\n"))

        target_data_height = self.font_size * nlines * 1.2
        point_height = self._text_metrics_cache.height_points
        point_width = self._text_metrics_cache.width_points

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
