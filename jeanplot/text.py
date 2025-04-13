from typing import Optional, Literal, Any
import numpy as np
from .component import Component
from .models import Size
from .style import jstyle
from .svg import SVGTextContent


class Text(Component):
    """text component rendered via svg path conversion"""

    text: str = ""  # default to empty string
    font_name: Optional[str] = None
    font_size: float = 12.0
    font_weight: Literal["normal", "bold"] = "normal"
    font_style: Literal["normal", "italic"] = "normal"
    color: str = "black"
    align: Literal["left", "center", "right"] = "left"
    vertical_align: Literal["top", "middle", "bottom"] = "top"
    line_spacing: float = 0.2  # factor of base line height

    _svg_cache: Optional[SVGTextContent] = None

    def _measure_natural(self, renderer) -> Size:
        """calculates the natural size of the text using the renderer."""
        if not self.text or not renderer or not hasattr(renderer, "measure_text"):
            self._svg_cache = None
            return Size(0, 0)

        # use renderer's measurement - this will also populate _svg_cache
        measured_size = renderer.measure_text(self)
        self._log_debug(f"_measure_natural: measured size = {measured_size}")
        return measured_size

    def render(self, renderer, context: Any, matrix: np.ndarray):
        """render text using the renderer."""

        if self.show and self.text:
            renderer.render_text(context, self, matrix)

        if self.debug:
            renderer.render_debug(context, self, matrix)
