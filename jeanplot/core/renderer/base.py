from typing import BinaryIO, TextIO, Callable, Any
import numpy as np

from jeanplot.core.component import Component
from jeanplot.core.models import Size, BoxStyle
from jeanplot.core.svg import SVGElement, SVGPathData


class BaseRenderer:
    """interface for backend renderer implementations."""

    RENDERER_NAME = "base"

    def __init__(self):
        self.pre_render_callbacks: list[Callable] = []
        self.post_render_callbacks: list[Callable] = []

    def add_pre_render_callback(self, callback: Callable):
        self.pre_render_callbacks.append(callback)

    def add_post_render_callback(self, callback: Callable):
        self.post_render_callbacks.append(callback)

    def create_context(self, width: float, height: float, **kwargs) -> Any:
        raise NotImplementedError

    def render_component(self, context: Any, component: Component, adjust_lims: bool = True):
        raise NotImplementedError

    def render_to_output(
        self, context: Any, output: str | BinaryIO | TextIO | None = None, **kwargs
    ):
        raise NotImplementedError

    def render_rectangle(
        self,
        context: Any,
        bounds: Size,
        style: BoxStyle,
        matrix: np.ndarray,
        component: Component | None = None,
    ):
        raise NotImplementedError

    def render_svg(self, context: Any, svg_element: SVGElement, matrix: np.ndarray):
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
        raise NotImplementedError

    def render_text(self, context: Any, text_component: Any, matrix: np.ndarray):
        raise NotImplementedError

    def render_debug(self, context: Any, component: Component, matrix: np.ndarray):
        raise NotImplementedError

    def measure_text(self, text_component: Any) -> Size:
        raise NotImplementedError
