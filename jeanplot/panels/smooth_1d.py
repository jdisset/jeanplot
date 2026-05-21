"""SmoothPanel1D — 1D smoothed line plot."""

from typing import Any

from pydantic import Field

from jeanplot.data import IdentityRescaler, PlotData, PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.plots.smooth_1d import smooth_1d


class SmoothPanel1D(PlotPanel):
    plot_data: PlotData
    slices: Any | None = None
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    vlims: tuple[float | None, float | None] = (0.0, None)
    draw_xlabel: bool = True
    draw_ylabel: bool = True
    res: int = 500
    show_std: bool = True
    show_legend: bool = True
    std_alpha: float = 0.15
    std_mode: str = "errorbar"
    n_errorbars: int = 5
    lineplot_props: list[dict] | dict | None = None
    errorbar_props: list[dict] | dict | None = None
    colors: list[Any] | None = None
    knn_stats_params: dict = Field(default_factory=dict)
    max_centroid_offset_frac: float = 0.0
    legend_kwargs: dict | None = None
    head_fit_frac: float = 0.0
    tail_fit_frac: float = 0.0

    def draw(self, ax) -> PlotFunctionResult | None:
        rescaler = self.rescaler if self.rescaler is not None else IdentityRescaler()
        return smooth_1d(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            slices=self.slices,
            title=self.title,
            xtitle=self.xtitle,
            ytitle=self.ytitle,
            xlims=self.xlims,
            vlims=self.vlims,
            draw_xlabel=self.draw_xlabel,
            draw_ylabel=self.draw_ylabel,
            res=self.res,
            show_std=self.show_std,
            show_legend=self.show_legend,
            std_alpha=self.std_alpha,
            std_mode=self.std_mode,
            n_errorbars=self.n_errorbars,
            lineplot_props=self.lineplot_props,
            errorbar_props=self.errorbar_props,
            colors=self.colors,
            knn_stats_params=self.knn_stats_params,
            max_centroid_offset_frac=self.max_centroid_offset_frac,
            legend_kwargs=self.legend_kwargs,
            head_fit_frac=self.head_fit_frac,
            tail_fit_frac=self.tail_fit_frac,
        )

    def render_txt(self) -> str | None:
        from jeanplot.plots.txt import smooth_1d_txt

        result = smooth_1d_txt(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            title=self.title,
            xtitle=self.xtitle,
            ytitle=self.ytitle,
            xlims=self.xlims,
        )
        return str(result)


SmoothPanel1D.model_rebuild(force=True)
