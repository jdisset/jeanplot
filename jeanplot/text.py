from typing import Optional, Literal
import numpy as np
from .component import Component
from .models import Size


class Text(Component):
    """text component rendered via svg path conversion"""

    text: str
    font_name: Optional[str] = None
    font_size: float = 12.0
    font_weight: Literal["normal", "bold"] = "normal"
    font_style: Literal["normal", "italic"] = "normal"
    color: str = "black"
    align: Literal["left", "center", "right"] = "left"
    vertical_align: Literal["top", "middle", "bottom"] = "top"
    line_spacing: float = 0.2  # controls extra space between lines (0.0 = no extra space)

    def measure(self, renderer=None) -> Size:
        if not self.text:
            self._dimensions = Size()
            self._transformed_aabb = Size()
            return self._dimensions

        # use renderer's measurement if available
        if renderer and hasattr(renderer, "measure_text"):
            measured_size = renderer.measure_text(self)
            self._dimensions = Size(
                width=max(self.min_dimensions.width, measured_size.width),
                height=max(self.min_dimensions.height, measured_size.height),
            )
        else:
            self._dimensions = Size(
                width=self.min_dimensions.width, height=self.min_dimensions.height
            )

        # respect max dimensions
        self._dimensions = Size(
            width=min(self._dimensions.width, self.max_dimensions.width),
            height=min(self._dimensions.height, self.max_dimensions.height),
        )

        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        renderer.render_text(context, self, matrix)
        if self.debug:
            renderer.render_debug(context, self, matrix)
