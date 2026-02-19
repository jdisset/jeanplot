"""Base class definition for renderers."""

from typing import BinaryIO, TextIO, Callable, Any
import numpy as np

# use absolute imports
from jeanplot.core.component import Component
from jeanplot.core.models import Size, BoxStyle
from jeanplot.core.svg import SVGElement, SVGPathData


class BaseRenderer:
    """base renderer class defining the interface for backend implementations."""

    RENDERER_NAME = "base"

    def __init__(self):
        self.pre_render_callbacks: list[Callable] = []
        self.post_render_callbacks: list[Callable] = []

    def add_pre_render_callback(self, callback: Callable):
        """add callback to run before rendering."""
        self.pre_render_callbacks.append(callback)

    def add_post_render_callback(self, callback: Callable):
        """add callback to run after rendering."""
        self.post_render_callbacks.append(callback)

    def create_context(self, width: float, height: float, **kwargs) -> Any:
        """create a rendering context (e.g., matplotlib axes)."""
        raise NotImplementedError

    def render_component(self, context: Any, component: Component, adjust_lims: bool = True):
        """render a component and its children to the context."""
        raise NotImplementedError

    def render_to_output(
        self, context: Any, output: str | BinaryIO | TextIO | None = None, **kwargs
    ):
        """render the context to an output file or stream."""
        raise NotImplementedError

    # --- Primitive Drawing Methods ---

    def render_rectangle(
        self,
        context: Any,
        bounds: Size,
        style: BoxStyle,
        matrix: np.ndarray,
        component: Component | None = None,
    ):
        """render a rectangle with borders, background, and shadow."""
        raise NotImplementedError

    def render_svg(self, context: Any, svg_element: SVGElement, matrix: np.ndarray):
        """render an svg element (delegates to render_path usually)."""
        raise NotImplementedError

    def render_path(
        self,
        context: Any,
        path_data: SVGPathData,
        matrix: np.ndarray,
        line_width_mode: str = "point",
        main_color: str | None = None,
        secondary_color: str | None = None,
        opacity: float = 1.0,
    ):
        """render a single SVG path data object."""
        raise NotImplementedError

    def render_text(self, context: Any, text_component: Any, matrix: np.ndarray):
        """render text (specific component type handled by implementation)."""
        raise NotImplementedError

    def render_debug(self, context: Any, component: Component, matrix: np.ndarray):
        """render debug visuals for a component (bounding box, origin)."""
        raise NotImplementedError

    # --- Measurement Methods ---

    def measure_text(self, text_component: Any) -> Size:
        """measure the natural dimensions of a text component."""
        raise NotImplementedError
