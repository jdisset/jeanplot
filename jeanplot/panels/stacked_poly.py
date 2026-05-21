"""StackedPolyPanel — stacked-polynomial fit at quantile knots."""

from typing import Any

import numpy as np

from jeanplot.data import PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.plots.stacked_poly import evaluate_stacked_poly, fit_stacked_poly_at_quantiles


class StackedPolyPanel(PlotPanel):
    x: Any
    y: Any
    weights: Any | None = None
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    degree: int = 1
    n_eval: int = 200
    line_color: str = "k"
    line_lw: float = 1.0
    scatter: bool = True
    scatter_color: str = "#888"
    scatter_size: float = 5.0

    plot_data: None = None

    def draw(self, ax) -> PlotFunctionResult | None:
        x = np.asarray(self.x, dtype=float).ravel()
        y = np.asarray(self.y, dtype=float).ravel()
        w = (
            np.ones_like(x)
            if self.weights is None
            else np.asarray(self.weights, dtype=float).ravel()
        )
        params = fit_stacked_poly_at_quantiles(
            x, y, w, np.array(self.quantiles), degree=self.degree
        )
        eval_x = np.linspace(x.min(), x.max(), self.n_eval)
        eval_y = evaluate_stacked_poly(eval_x, params)

        if self.scatter:
            ax.scatter(x, y, s=self.scatter_size, c=self.scatter_color, alpha=0.5)
        ax.plot(eval_x, eval_y, color=self.line_color, lw=self.line_lw)
        if self.title:
            ax.set_title(self.title, **self.title_kwargs)
        return PlotFunctionResult(rendering=None, metadata={"params": params})


StackedPolyPanel.model_rebuild(force=True)
