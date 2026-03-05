# File: jeanplot/text.py
# -*- coding: utf-8 -*-
"""Text component definition."""

from __future__ import annotations
from typing import Literal, Any
import numpy as np
from pydantic import BaseModel, PrivateAttr, Field

# use absolute imports
from jeanplot.core.component import Component
from jeanplot.core.models import Size, TextMetrics, TextHalo
from jeanplot.core.debug import get_logger

logger = get_logger(__name__)


class TextSegment(BaseModel):
    """A styled segment within a Text element, rendered as a tspan in SVG."""

    text: str
    font_weight: Literal["normal", "bold"] | None = None
    font_style: Literal["normal", "italic"] | None = None
    color: str | None = None
    font_size: float | None = None  # override, in same units as parent


class Text(Component):
    """text component rendered using native matplotlib text."""

    text: str = ""
    segments: list[TextSegment] | None = None  # styled segments (overrides text for rendering)
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
    halo: TextHalo | None = None

    # renderers' native text rendering methods are used by default
    # we switch to path rendering if render_as_path is set to True
    # or if there are complex transforms (skew)
    render_as_path: Literal["auto"] | bool = "auto"

    _text_metrics_cache: TextMetrics | None = PrivateAttr(default=None)
    _render_path_cache: Any | None = PrivateAttr(default=None)

    # text components generally don't need a background/border by default
    # explicitly set style to None, jstyle can override if needed
    style: Any = Field(default=None, validate_default=False)

    @property
    def effective_text(self) -> str:
        """Concatenated text from segments if present, otherwise self.text."""
        if self.segments:
            return "".join(s.text for s in self.segments)
        return self.text

    def _measure_natural(self, renderer) -> Size:
        """
        calculates natural data-unit size from renderer-provided point metrics.

        The renderer stores text metrics at a reference point size
        (TextMetrics.ref_font_size). We scale those metrics linearly to the
        requested font size and convert to data units when needed.
        """
        if not self.effective_text or not renderer or not hasattr(renderer, "measure_text"):
            self._text_metrics_cache = None
            return Size(0, 0)

        renderer.measure_text(self)

        if not self._text_metrics_cache or self._text_metrics_cache.height_points <= 1e-6:
            self._log_debug("warning: text metrics measurement failed or yielded zero height.")
            return Size(0, 0)

        point_height = self._text_metrics_cache.height_points
        point_width = self._text_metrics_cache.width_points
        ref_font_size = self._text_metrics_cache.ref_font_size

        if ref_font_size <= 1e-6:
            return Size(0, 0)

        scale_factor = self.font_size / ref_font_size
        target_point_width = point_width * scale_factor
        target_point_height = point_height * scale_factor

        if self.font_size_mode == "points":
            points_per_data_unit = getattr(renderer, "_points_per_data_unit", 1.0)
            if points_per_data_unit > 1e-6:
                natural_data_width = target_point_width / points_per_data_unit
                target_data_height = target_point_height / points_per_data_unit
            else:
                natural_data_width = 0.0
                target_data_height = 0.0
        else:
            natural_data_width = target_point_width
            target_data_height = target_point_height

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
        if not self.show or not self.effective_text:
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

        force_native_text = bool(getattr(renderer, "force_native_text", False))
        if not force_native_text and (
            self.render_as_path is True
            or (self.render_as_path == "auto" and self.has_effective_skew(matrix))
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
