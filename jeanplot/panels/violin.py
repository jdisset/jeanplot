"""ViolinPanel — smooth voxel-conditioned violin plot."""

from typing import Literal

from pydantic import Field

from jeanplot.data import IdentityRescaler, PlotData, PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.plots.violin import smooth_voxel_conditioned_violin


class ViolinPanel(PlotPanel):
    plot_data: PlotData
    mode: Literal["single", "split"] = "single"
    xlims: tuple[float, float] = (0.0, 0.7)
    ylims: tuple[float, float] = (0.0, 0.7)
    draw_xlabel: bool = True
    draw_ylabel: bool = True
    grid_resolution: int = 64
    tick_count: int = 6
    tick_sigma: float = 0.05
    violin_width: float = 0.035
    violin_alpha: float = 0.2
    knn_stats_params: dict = Field(default_factory=dict)

    def draw(self, ax) -> PlotFunctionResult | None:
        rescaler = self.rescaler if self.rescaler is not None else IdentityRescaler()
        smooth_voxel_conditioned_violin(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            mode=self.mode,
            title=self.title,
            xtitle=self.xtitle,
            ytitle=self.ytitle,
            xlims=self.xlims,
            ylims=self.ylims,
            draw_xlabel=self.draw_xlabel,
            draw_ylabel=self.draw_ylabel,
            grid_resolution=self.grid_resolution,
            tick_count=self.tick_count,
            tick_sigma=self.tick_sigma,
            violin_width=self.violin_width,
            violin_alpha=self.violin_alpha,
            knn_stats_params=self.knn_stats_params,
        )
        return PlotFunctionResult(rendering=None, metadata={})


ViolinPanel.model_rebuild(force=True)
