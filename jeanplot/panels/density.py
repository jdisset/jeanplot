"""DensityPanel1D — split-violin 1D density plot."""

from typing import Any

import numpy as np

from jeanplot.data import PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.plots.density import density_plot_1d


class DensityPanel1D(PlotPanel):
    values: Any
    values_right: Any | None = None
    color: str = "k"
    label: str | None = None
    ticks: list[float] | None = None
    minor_ticks: list[float] | None = None
    ticks_labels: list[str] | None = None
    bw_method: float = 0.01
    show_quantiles: tuple[float, float] | None = (0.01, 0.99)
    is_first: bool = False
    n_samples: int = 3000

    plot_data: None = None

    def draw(self, ax) -> PlotFunctionResult | None:
        values = np.asarray(self.values, dtype=float).ravel()
        finite = values[np.isfinite(values)]
        if finite.size < 3:
            return None
        sample_at = np.linspace(finite.min(), finite.max(), self.n_samples)
        density_plot_1d(
            x=finite,
            sample_at=sample_at,
            ax=ax,
            color=self.color,
            label=self.label,
            ticks=self.ticks,
            minor_ticks=self.minor_ticks,
            ticks_labels=self.ticks_labels,
            bw_method=self.bw_method,
            x2=np.asarray(self.values_right) if self.values_right is not None else None,
            show_quantiles=self.show_quantiles,
            is_first=self.is_first,
        )
        if self.title:
            ax.set_title(self.title, **self.title_kwargs)
        return PlotFunctionResult(rendering=None, metadata={})


DensityPanel1D.model_rebuild(force=True)
