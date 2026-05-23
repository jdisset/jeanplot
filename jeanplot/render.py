from __future__ import annotations

from pathlib import Path
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
    output_path: str | None = None,
    overwrite: bool = True,
    renderer_kwargs: dict[str, Any] | None = None,
    context_kwargs: dict[str, Any] | None = None,
):
    from dracon.progress import step

    from jeanplot._figure_render import render_figure, save_figure
    from jeanplot._tui import current_tui
    from jeanplot.panels.figure import Figure

    if isinstance(component, Figure):
        if output_path is not None:
            p = Path(output_path)
            component.output_file = p.name
            component.output_dir = str(p.parent) if p.parent != Path("") else "./"
        if not overwrite and component.output_path is not None and component.output_path.exists():
            return None
        tui = current_tui()
        if tui is not None:
            tui.show_tree(component)
        with step("render"):
            mfig = render_figure(component)
            if tui is not None:
                tui.attach_mpl_figure(mfig)
            save_figure(component, mfig)
        if tui is not None:
            tui.receipt(component, component.output_path)
        return mfig

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
