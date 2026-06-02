"""ScatterPanel3D + GridHistogramPanel."""

from typing import Any

from pydantic import Field

from jeanplot.data import IdentityRescaler, PlotData, PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.plots.scatter import grid_histogram


class GridHistogramPanel(PlotPanel):
    plot_data: PlotData
    xlims: tuple[float | None, float | None] = (None, None)
    ylims: tuple[float | None, float | None] = (None, None)
    vlims: tuple[float | None, float | None] = (0, None)
    draw_xlabel: bool = True
    draw_ylabel: bool = True
    res: int = 300
    draw_colorbar: bool = True
    use_log_density: bool = True
    margins: float = 0.01
    noise_smooth: float = 0.25
    cmap: Any = None  # cmap name or Colormap; None -> DEFAULT_DENSITY_CMAP
    colorbar_params: dict = Field(default_factory=dict)

    def draw(self, ax) -> PlotFunctionResult | None:
        rescaler = self.rescaler if self.rescaler is not None else IdentityRescaler()
        im, _cbar = grid_histogram(
            cmap=self.cmap,
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            title=self.title,
            xtitle=self.xtitle,
            ytitle=self.ytitle,
            xlims=self.xlims,
            ylims=self.ylims,
            vlims=self.vlims,
            draw_xlabel=self.draw_xlabel,
            draw_ylabel=self.draw_ylabel,
            res=self.res,
            draw_colorbar=self.draw_colorbar,
            use_log_density=self.use_log_density,
            margins=self.margins,
            noise_smooth=self.noise_smooth,
            colorbar_params=self.colorbar_params,
        )
        self._mappable = im
        return PlotFunctionResult(rendering=im, metadata={}, mappable=im)


class ScatterPanel3D(PlotPanel):
    plot_data: PlotData
    size: float = 10.0
    lw: float = 0.1
    azim: float = 45.0
    elev: float = 30.0

    def draw(self, ax) -> PlotFunctionResult | None:
        x = self.plot_data.x
        y = self.plot_data.y.ravel() if self.plot_data.y.ndim > 1 else self.plot_data.y
        assert x.shape[1] >= 3, f"ScatterPanel3D requires 3 input dims; got {x.shape[1]}"
        ax.view_init(elev=self.elev, azim=self.azim)
        sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, s=self.size, lw=self.lw, edgecolor="k")
        names = self.plot_data.input_names
        if len(names) >= 3:
            ax.set_xlabel(names[0])
            ax.set_ylabel(names[1])
            ax.set_zlabel(names[2])
        if self.title:
            ax.set_title(self.title)
        return PlotFunctionResult(rendering=sc, metadata={}, mappable=sc)


GridHistogramPanel.model_rebuild(force=True)
ScatterPanel3D.model_rebuild(force=True)
