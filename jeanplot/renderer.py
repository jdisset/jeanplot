from typing import Optional, Union, BinaryIO, TextIO, Callable
import numpy as np
from .component import Component


class BaseRenderer:
    """base renderer class defining a unified interface for all renderers"""

    RENDERER_NAME = "base"

    def __init__(self):
        self.pre_render_callbacks = []
        self.post_render_callbacks = []

    def add_pre_render_callback(self, callback: Callable):
        self.pre_render_callbacks.append(callback)

    def add_post_render_callback(self, callback: Callable):
        self.post_render_callbacks.append(callback)

    def create_context(self, width: float, height: float, **kwargs):
        """create a rendering context"""
        raise NotImplementedError("renderers must implement create_context")

    def render_component(
        self, context, component: Component, parent_matrix: Optional[np.ndarray] = None
    ):
        """render a component to the context"""
        matrix = component.compute_world_matrix(parent_matrix)
        component.render(self, context, matrix)

    def render_to_output(
        self, context, output: Optional[Union[str, BinaryIO, TextIO]] = None, **kwargs
    ):
        """render the context to output"""
        raise NotImplementedError("renderers must implement render_to_output")

    def render_rectangle(self, context, bounds, style, matrix, component=None):
        """render a rectangle"""
        raise NotImplementedError("renderers must implement render_rectangle")

    def render_svg(self, context, svg_element, matrix):
        """render an svg element"""
        raise NotImplementedError("renderers must implement render_svg")

    def render_text(self, context, text_component, matrix):
        """render text"""
        pass

    def render_debug(self, context, component, matrix):
        """render debug visuals"""
        raise NotImplementedError("renderers must implement render_debug")
