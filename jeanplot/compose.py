"""Tree-construction helpers for laying out PlotPanels.

Every helper is also registered as a dracon `!fn` template (via
`register_template`) so it doubles as a YAML tag. Python users call
`panel_row(...)`, YAML users write `!panel_row { ... }`.
"""

from typing import Any

from dracon import register_template

from jeanplot.core.container import Container
from jeanplot.core.component import Component
from jeanplot.core.models import LayoutConstraints
from jeanplot.panels.auto import auto_panel
from jeanplot.panels.base import PlotPanel


def panel_row(
    panels: list[Component],
    gap: float = 8.0,
    weights: list[float] | None = None,
) -> Container:
    """One row of panels (or any Components) with optional flex weights."""
    return Container(
        layout=LayoutConstraints(direction="row", gap=gap, main_axis_weights=weights),
        children=list(panels),
    )


def panel_grid(
    rows: list[list[Component]],
    gap: float = 8.0,
    col_weights: list[list[float]] | None = None,
    row_weights: list[float] | None = None,
) -> Container:
    """Multi-row grid. `col_weights[i]` is per-column weights for row i."""
    row_containers = [
        panel_row(row, gap=gap, weights=col_weights[i] if col_weights else None)
        for i, row in enumerate(rows)
    ]
    return Container(
        layout=LayoutConstraints(direction="column", gap=gap, main_axis_weights=row_weights),
        children=row_containers,
    )


def panels_from_datas(datas: list[Any], **kwargs) -> list[PlotPanel]:
    """Map a list of PlotData/LazyPlotData to a list of AutoPanels."""
    return [auto_panel(plot_data=d, **kwargs) for d in datas]


def default_output_name(
    plot_data: Any | None = None,
    fallback: str = "figure",
    prefix: str = "",
    suffix: str = ".png",
    override: str | None = None,
) -> str:
    """Pick a sensible output filename. `override` wins; else use plot_data's
    `metadata['network_name']` (if present) or `fallback`."""
    if override is not None:
        return override
    name = fallback
    md = getattr(plot_data, "metadata", None)
    if md:
        name = md.get("network_name", fallback)
    return f"{prefix}{name}{suffix}"


def build_figure_metadata(
    panels: list[PlotPanel] | None = None,
    extra: dict | None = None,
) -> dict:
    """Aggregate per-panel `_last_metadata` and `plot_data.metadata` into one dict.

    Keys later in `panels` win; `extra` wins last.
    """
    out: dict = {}
    for p in panels or []:
        pd = getattr(p, "plot_data", None)
        if pd is not None:
            md = getattr(pd, "metadata", None)
            if md:
                out.update(md)
        last = getattr(p, "_last_metadata", None)
        if last:
            out.update(last)
    if extra:
        out.update(extra)
    return out


COMPOSE_HELPERS: dict[str, Any] = {
    "panel_row": panel_row,
    "panel_grid": panel_grid,
    "panels_from_datas": panels_from_datas,
    "build_figure_metadata": build_figure_metadata,
    "default_output_name": default_output_name,
    "auto_panel": auto_panel,
}

register_template(panel_row)
register_template(panel_grid)
register_template(panels_from_datas, allow_extras=True)
register_template(build_figure_metadata)
register_template(default_output_name)
