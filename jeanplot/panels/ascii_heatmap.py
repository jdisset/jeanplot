"""AsciiHeatmapPanel — Unicode-art heatmap that only renders to text."""

from typing import Any, Literal

from jeanplot.panels.base import PlotPanel
from jeanplot.plots.ascii_heatmap import heatmap as ascii_heatmap_fn


class AsciiHeatmapPanel(PlotPanel):
    data: Any
    vmin: float | None = None
    vmax: float | None = None
    xres: int | None = None
    yres: int | None = None
    cmap: str | list[str] | None = None
    mode: Literal["single", "bigram"] = "single"
    levels: int = 5
    show_colorbar: bool = True
    border: bool = False
    color: bool = False
    resample: Literal["nearest", "mean"] = "mean"

    plot_data: None = None
    is_drawable: bool = False

    def render_txt(self) -> str | None:
        import numpy as np

        return ascii_heatmap_fn(
            np.asarray(self.data, dtype=float),
            vmin=self.vmin,
            vmax=self.vmax,
            xres=self.xres,
            yres=self.yres,
            cmap=self.cmap,
            mode=self.mode,
            levels=self.levels,
            show_colorbar=self.show_colorbar,
            border=self.border,
            color=self.color,
            resample=self.resample,
        )


AsciiHeatmapPanel.model_rebuild(force=True)
