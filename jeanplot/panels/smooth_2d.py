"""SmoothPanel2D — 2D KNN-smoothed heatmap.

Plus gradient-magnitude and gradient-field variants.
"""

from typing import Any, Literal

from pydantic import Field

from jeanplot.data import IdentityRescaler, PlotData, PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.plots.smooth_2d import (
    gradient_field_2d,
    smooth_2d,
    smooth_grad_magnitude_2d,
)


class SmoothPanel2D(PlotPanel):
    plot_data: PlotData
    zslice: Any | None = None
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    ylims: tuple[float | None, float | None] = (None, None)
    vlims: tuple[float | None, float | None] = (None, None)
    vlim_quantiles: tuple[float | None, float | None] | None = (0.01, 0.99)
    vlim_min_floor: float | None = None
    vlim_min_range: float | None = None
    draw_xlabel: bool = True
    draw_ylabel: bool = True
    xaxis_labelpad: float | None = None
    yaxis_labelpad: float | None = None
    draw_colorbar: bool = True
    draw_colorbar_label: bool = True
    colorbar_params: dict = Field(default_factory=dict)
    knn_grid_params: dict = Field(default_factory=dict)
    heatmap_params: dict = Field(default_factory=dict)
    setup_transformed_axis_params: dict = Field(default_factory=dict)

    def draw(self, ax) -> PlotFunctionResult | None:
        rescaler = self.rescaler if self.rescaler is not None else IdentityRescaler()
        result = smooth_2d(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            zslice=self.zslice,
            title=self.title,
            title_kwargs=self.title_kwargs,
            xtitle=self.xtitle,
            ytitle=self.ytitle,
            vtitle=self.vtitle,
            xlims=self.xlims,
            ylims=self.ylims,
            vlims=self.vlims,
            vlim_quantiles=self.vlim_quantiles,
            vlim_min_floor=self.vlim_min_floor,
            vlim_min_range=self.vlim_min_range,
            draw_xlabel=self.draw_xlabel,
            draw_ylabel=self.draw_ylabel,
            xaxis_labelpad=self.xaxis_labelpad,
            yaxis_labelpad=self.yaxis_labelpad,
            draw_colorbar=self.draw_colorbar,
            draw_colorbar_label=self.draw_colorbar_label,
            colorbar_params=self.colorbar_params,
            knn_grid_params=self.knn_grid_params,
            heatmap_params=self.heatmap_params,
            setup_transformed_axis_params=self.setup_transformed_axis_params,
        )
        if result is not None and result.mappable is not None:
            self._mappable = result.mappable
        return result

    def render_txt(self) -> str | None:
        from jeanplot.plots.txt import smooth_2d_txt

        result = smooth_2d_txt(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            title=self.title,
            xtitle=self.xtitle,
            ytitle=self.ytitle,
            xlims=self.xlims,
            ylims=self.ylims,
        )
        return str(result)


class SmoothGradMagnitudePanel2D(PlotPanel):
    plot_data: PlotData
    space: Literal["raw", "latent"] = "latent"
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    ylims: tuple[float | None, float | None] = (None, None)
    vlims: tuple[float | None, float | None] = (None, None)
    knn_grid_params: dict = Field(default_factory=dict)
    heatmap_params: dict = Field(default_factory=dict)
    colorbar_params: dict = Field(default_factory=dict)

    def draw(self, ax) -> PlotFunctionResult | None:
        rescaler = self.rescaler if self.rescaler is not None else IdentityRescaler()
        return smooth_grad_magnitude_2d(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            space=self.space,
            title=self.title,
            title_kwargs=self.title_kwargs,
            xtitle=self.xtitle,
            ytitle=self.ytitle,
            vtitle=self.vtitle,
            xlims=self.xlims,
            ylims=self.ylims,
            vlims=self.vlims,
            knn_grid_params=self.knn_grid_params,
            heatmap_params=self.heatmap_params,
            colorbar_params=self.colorbar_params,
        )


class GradientFieldPanel2D(PlotPanel):
    plot_data: PlotData
    space: Literal["raw", "latent"] = "latent"
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    ylims: tuple[float | None, float | None] = (None, None)
    arrow_density: int = 25
    arrow_scale: float = 1.0
    arrow_color: str = "k"
    knn_grid_params: dict = Field(default_factory=dict)

    def draw(self, ax) -> PlotFunctionResult | None:
        rescaler = self.rescaler if self.rescaler is not None else IdentityRescaler()
        return gradient_field_2d(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            space=self.space,
            xlims=self.xlims,
            ylims=self.ylims,
            arrow_density=self.arrow_density,
            arrow_scale=self.arrow_scale,
            arrow_color=self.arrow_color,
            knn_grid_params=self.knn_grid_params,
        )


SmoothPanel2D.model_rebuild(force=True)
SmoothGradMagnitudePanel2D.model_rebuild(force=True)
GradientFieldPanel2D.model_rebuild(force=True)
