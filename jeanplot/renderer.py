"""Base class definition for renderers."""

from typing import Optional, Union, BinaryIO, TextIO, Callable, Any, Tuple, List
import numpy as np

# use absolute imports
from jeanplot.component import Component
from jeanplot.models import Size, BoxStyle
from jeanplot.svg import SVGElement, SVGPathData, LineEndType
from jeanplot.connector import Connection  # needed for type hint


class BaseRenderer:
    """base renderer class defining the interface for backend implementations."""

    RENDERER_NAME = "base"

    def __init__(self):
        self.pre_render_callbacks: List[Callable] = []
        self.post_render_callbacks: List[Callable] = []

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
        self, context: Any, output: Optional[Union[str, BinaryIO, TextIO]] = None, **kwargs
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
        component: Optional[Component] = None,
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
        main_color: Optional[str] = None,
        secondary_color: Optional[str] = None,
    ):
        """render a single SVG path data object."""
        raise NotImplementedError

    def render_text(self, context: Any, text_component: Any, matrix: np.ndarray):
        """render text (specific component type handled by implementation)."""
        raise NotImplementedError

    def render_connection_curve(
        self,
        context: Any,
        connection: Connection,
        local_start: Tuple[float, float],
        local_end: Tuple[float, float],
        local_control_points: List[Tuple[float, float]],
        path_string: str,
        matrix: np.ndarray,
    ):
        """render the main curve part of a connection."""
        raise NotImplementedError

    def render_debug(self, context: Any, component: Component, matrix: np.ndarray):
        """render debug visuals for a component (bounding box, origin)."""
        raise NotImplementedError

    # --- Measurement Methods ---

    def measure_text(self, text_component: Any) -> Size:
        """measure the natural dimensions of a text component."""
        raise NotImplementedError
