from __future__ import annotations
from typing import Literal, Any
import numpy as np
from pydantic import BaseModel, PrivateAttr, Field, model_validator

from jeanplot.core.component import Component
from jeanplot.core.models import Size, TextMetrics, TextHalo
from jeanplot.core.debug import get_logger

logger = get_logger(__name__)


class TextSegment(BaseModel):
    """styled segment within a Text element, rendered as a tspan in svg."""

    text: str
    font_weight: Literal["normal", "bold"] | None = None
    font_style: Literal["normal", "italic"] | None = None
    color: str | None = None
    font_size: float | None = None


class Text(Component):
    """text component rendered using native matplotlib text."""

    @model_validator(mode="before")
    @classmethod
    def _bare_string_is_text(cls, v: Any) -> Any:
        # `!Text "hello"` -> `!Text { text: "hello" }`.
        return {"text": v} if isinstance(v, str) else v

    text: str = ""
    segments: list[TextSegment] | None = None  # overrides text when set
    font_name: str | None = None
    font_size: float = 12.0
    # "points": typographic points (consistent visual size).
    # "data": local data units; scales with container transforms.
    font_size_mode: Literal["points", "data"] = "data"
    font_weight: Literal["normal", "bold"] = "normal"
    font_style: Literal["normal", "italic"] = "normal"
    color: str = "black"
    align: Literal["left", "center", "right"] = "left"
    vertical_align: Literal["top", "middle", "bottom", "baseline"] = "middle"
    line_spacing: float = 0.2  # factor of base line height
    halo: TextHalo | None = None

    # native renderer text is used by default; switches to paths if forced or under skew
    render_as_path: Literal["auto"] | bool = "auto"

    _text_metrics_cache: TextMetrics | None = PrivateAttr(default=None)
    _render_path_cache: Any | None = PrivateAttr(default=None)

    # text doesn't get bg/border by default; jstyle can override
    style: Any = Field(default=None, validate_default=False)

    @property
    def effective_text(self) -> str:
        if self.segments:
            return "".join(s.text for s in self.segments)
        return self.text

    def _measure_natural(self, renderer) -> Size:
        """natural data-unit size from renderer-provided point metrics.

        The renderer stores metrics at TextMetrics.ref_font_size; we scale
        linearly to the requested font size and convert to data units when needed.
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
        """skew check: transformed basis axes non-orthogonal."""
        EPSILON = 1e-6
        m2x2 = world_matrix[:2, :2]
        col1 = m2x2[:, 0]
        col2 = m2x2[:, 1]

        len1_sq = np.dot(col1, col1)
        len2_sq = np.dot(col2, col2)

        if len1_sq < EPSILON or len2_sq < EPSILON:
            return False

        dot_product = np.dot(col1, col2)
        normalized_dot_sq = (dot_product * dot_product) / (len1_sq * len2_sq)
        skew_tolerance = 1e-3
        return normalized_dot_sq > skew_tolerance

    def render(self, renderer, context: Any, matrix: np.ndarray):
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
