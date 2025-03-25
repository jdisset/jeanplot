from typing import Optional, Dict, Any, Literal
from pydantic import PrivateAttr
import numpy as np
from .component import Component
from .models import Size


class Text(Component):
    """text component rendered via svg path conversion"""

    text: str
    font_name: Optional[str] = None  # use default font when None
    font_size: float = 12.0  # in data units
    font_weight: Literal["normal", "bold"] = "normal"
    font_style: Literal["normal", "italic"] = "normal"
    color: str = "black"
    align: Literal["left", "center", "right"] = "left"
    vertical_align: Literal["top", "middle", "bottom"] = "top"

    # cache for text path data
    _text_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def measure(self, renderer=None) -> Size:
        """measure text dimensions using renderer"""
        if not self.text:
            self._dimensions = Size(width=0, height=0)
            self._transformed_aabb = Size(width=0, height=0)
            return self._dimensions

        if renderer and hasattr(renderer, "measure_text"):
            # use renderer's measurement capability for exact dimensions
            measured_size = renderer.measure_text(self)
            self._dimensions = Size(
                width=max(self.min_dimensions.width, measured_size.width),
                height=max(self.min_dimensions.height, measured_size.height),
            )
        else:
            # if no renderer is available, use minimum dimensions
            self._dimensions = Size(
                width=self.min_dimensions.width, height=self.min_dimensions.height
            )

        # apply max constraints
        self._dimensions = Size(
            width=min(self._dimensions.width, self.max_dimensions.width),
            height=min(self._dimensions.height, self.max_dimensions.height),
        )

        self._transformed_aabb = self.compute_transformed_aabb()

        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render text component using renderer"""
        renderer.render_text(context, self, matrix)

        if self.debug:
            renderer.render_debug(context, self, matrix)
