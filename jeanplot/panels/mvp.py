"""MVPPanel — measured-vs-predicted scatter with density and trendline."""

from typing import Any


from jeanplot.data import PlotFunctionResult
from jeanplot.panels.base import PlotPanel


class MVPPanel(PlotPanel):
    measured: Any
    predicted: Any
    plot_data: None = None

    show_density: bool = True
    density_res: int = 100
    density_cmap: str = "viridis"
    density_log: bool = True
    density_noise_smooth: float = 0.25

    show_trendline: bool = True
    trendline_color: str = "black"
    trendline_lw: float = 1.0
    trendline_eval_points: int = 200
    trendline_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    trendline_degree: int = 1

    knn_stats_params: dict | None = None
    density_cutoff_q: float = 0.05
    smooth_sigma: float = 3.0

    show_identity: bool = True
    identity_color: str = "grey"
    identity_ls: str = "--"
    identity_lw: float = 1.0

    show_stats: bool = True
    show_bias: bool = True
    bias_signed: bool = True
    show_calibration_rms: bool = True
    show_spread: bool = True
    show_crps: bool = True
    extra_metrics: dict | None = None

    noise_floor: float | None = None
    noise_local: bool = True
    noise_local_radius: float = 0.08
    noise_local_min_points: int = 30
    noise_color: str = "grey"
    noise_alpha: float = 0.12
    show_noise_floor: bool = True

    vlims: tuple = (0.0, 0.7)
    margins: float = 0.02
    xlabel: str = "Measured"
    ylabel: str = "Predicted"

    model_samples: Any | None = None
    model_bands: list[list[int]] | None = None
    model_band_color: str = "#d62728"
    model_band_lw: float = 1.5
    show_coverage: bool = True
    pit_values: Any | None = None

    def draw(self, ax) -> PlotFunctionResult | None:
        from jeanplot.plots.mvp import measured_vs_predicted

        return measured_vs_predicted(
            ax=ax,
            measured=self.measured,
            predicted=self.predicted,
            rescaler=self.rescaler,
            show_density=self.show_density,
            density_res=self.density_res,
            density_cmap=self.density_cmap,
            density_log=self.density_log,
            density_noise_smooth=self.density_noise_smooth,
            show_trendline=self.show_trendline,
            trendline_color=self.trendline_color,
            trendline_lw=self.trendline_lw,
            trendline_eval_points=self.trendline_eval_points,
            trendline_quantiles=self.trendline_quantiles,
            trendline_degree=self.trendline_degree,
            knn_stats_params=self.knn_stats_params,
            density_cutoff_q=self.density_cutoff_q,
            smooth_sigma=self.smooth_sigma,
            show_identity=self.show_identity,
            identity_color=self.identity_color,
            identity_ls=self.identity_ls,
            identity_lw=self.identity_lw,
            show_stats=self.show_stats,
            show_bias=self.show_bias,
            bias_signed=self.bias_signed,
            show_calibration_rms=self.show_calibration_rms,
            show_spread=self.show_spread,
            show_crps=self.show_crps,
            extra_metrics=self.extra_metrics,
            noise_floor=self.noise_floor,
            noise_local=self.noise_local,
            noise_local_radius=self.noise_local_radius,
            noise_local_min_points=self.noise_local_min_points,
            noise_color=self.noise_color,
            noise_alpha=self.noise_alpha,
            show_noise_floor=self.show_noise_floor,
            vlims=self.vlims,
            margins=self.margins,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            title=self.title,
            model_samples=self.model_samples,
            model_bands=self.model_bands,
            model_band_color=self.model_band_color,
            model_band_lw=self.model_band_lw,
            show_coverage=self.show_coverage,
            pit_values=self.pit_values,
        )


MVPPanel.model_rebuild(force=True)
