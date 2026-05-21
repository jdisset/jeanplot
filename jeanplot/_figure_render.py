"""Figure-aware rendering: walks a Figure(Container), allocates one mpl Axes
per drawable leaf PlotPanel, then runs overlay draws.

This file owns the only branch in jeanplot.render that knows about `Figure`,
`PlotPanel`, and `Colorbar`. The scene-graph renderer is untouched.
"""

from pathlib import Path
from typing import Any, Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt

from jeanplot.core.component import Component
from jeanplot.core.style import jstyle
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.figure import Figure


def _iter_panels(root: Component) -> Iterator[PlotPanel]:
    if isinstance(root, PlotPanel):
        yield root
    for c in getattr(root, "children", None) or []:
        yield from _iter_panels(c)


def _panel_bbox(
    panel: PlotPanel, root_w: float, root_h: float
) -> tuple[float, float, float, float]:
    """Figure-relative (left, bottom, width, height). mpl is bottom-up; jeanplot top-down."""
    ox, oy = panel.get_world_origin()
    w = panel._dimensions.width
    h = panel._dimensions.height
    assert root_w > 0 and root_h > 0, f"figure has zero size: {root_w}x{root_h}"
    return (ox / root_w, 1.0 - (oy + h) / root_h, w / root_w, h / root_h)


def render_figure(fig: Figure) -> Any:
    """Build the mpl Figure, lay out the tree, allocate axes, draw panels and overlays."""
    with mpl.rc_context(rc=fig.rc_context):
        if fig.theme is not None:
            jstyle.update(fig.theme)
        jstyle.apply(fig)
        fig.measure_and_layout(None)

        figsize = (fig._dimensions.width, fig._dimensions.height)
        mfig = plt.figure(figsize=figsize, dpi=fig.dpi)
        fig._mpl_figure = mfig

        root_w, root_h = figsize
        panels = list(_iter_panels(fig))
        drawable = [p for p in panels if p.is_drawable and not p.is_overlay]
        overlays = [p for p in panels if p.is_overlay]

        for panel in drawable:
            bbox = _panel_bbox(panel, root_w, root_h)
            panel._axes = mfig.add_axes(bbox)

        for panel in drawable:
            assert panel._axes is not None, f"panel {panel.id} has no axes"
            res = panel.draw(panel._axes)
            if res is not None:
                if getattr(res, "mappable", None) is not None:
                    panel._mappable = res.mappable
                meta = getattr(res, "metadata", None)
                if meta:
                    fig.metadata.update(meta)
                panel._last_metadata = meta or {}

        for overlay in overlays:
            parent_ax = getattr(overlay.parent, "_axes", None) if overlay.parent else None
            if parent_ax is None:
                continue
            overlay.draw(parent_ax)

        if fig.subtitle:
            mfig.suptitle(fig.subtitle, **fig.subtitle_kwargs)

        return mfig


def _stringify_metadata(md: dict | None) -> dict | None:
    if not md:
        return None
    return {str(k): v if isinstance(v, str) else repr(v) for k, v in md.items()}


def save_figure(fig: Figure, mfig: Any) -> Path | None:
    """Write the figure to disk per `fig.output_path` and `fig.extra_output_paths`."""
    out = fig.output_path
    if out is None:
        return None
    md = _stringify_metadata(fig.metadata)
    out.parent.mkdir(parents=True, exist_ok=True)
    mfig.savefig(out, dpi=fig.dpi, metadata=md)
    for extra in fig.extra_output_paths:
        p = Path(extra)
        p.parent.mkdir(parents=True, exist_ok=True)
        mfig.savefig(p, dpi=fig.dpi, metadata=md)
    return out
