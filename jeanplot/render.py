from __future__ import annotations

from typing import Any, Literal

from jeanplot.core.component import Component
from jeanplot.core.renderer import MatplotlibRenderer, SVGRenderer

BackendName = Literal["matplotlib", "svg"]


def _build_renderer(backend: BackendName, **renderer_kwargs):
    if backend == "svg":
        return SVGRenderer(**renderer_kwargs)
    if backend == "matplotlib":
        return MatplotlibRenderer(**renderer_kwargs)
    raise ValueError(f"unsupported backend: {backend}")


def render(
    component: Component,
    *,
    backend: BackendName = "matplotlib",
    context: Any | None = None,
    width: float = 800,
    height: float = 600,
    adjust_lims: bool = True,
    output: str | None = None,
    renderer_kwargs: dict[str, Any] | None = None,
    context_kwargs: dict[str, Any] | None = None,
):
    """Render a component tree through a selected backend.

    Returns backend-native output:
    - `matplotlib`: Axes (or saved figure side-effect when `output` is set)
    - `svg`: SVG root element, or SVG string when `output` is None and no context provided
    """

    renderer = _build_renderer(backend, **(renderer_kwargs or {}))
    if context is None:
        context = renderer.create_context(width=width, height=height, **(context_kwargs or {}))

    renderer.render_component(context, component, adjust_lims=adjust_lims)

    if output is not None:
        renderer.render_to_output(context, output=output)
        return context

    if backend == "svg":
        return renderer.render_to_output(context)

    return context


def render_to_string(component: Component, *, width: float = 800, height: float = 600) -> str:
    renderer = SVGRenderer()
    root = renderer.create_context(width=width, height=height)
    renderer.render_component(root, component)
    return renderer.render_to_output(root)
