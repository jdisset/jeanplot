"""ParticlePanel — particle/dot plot showing current values with history trails."""

from typing import Any

import numpy as np
from pydantic import Field

from jeanplot.data import PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.plots.particle import particle_plot


class ParticlePanel(PlotPanel):
    data: Any
    value_names: list[str]
    colors: Any | None = None
    ylims: tuple[float, float] | None = None
    max_line_extend: int = 40
    value_spacing: float = 25.0
    derivative: Any | None = None
    line_params: dict = Field(default_factory=dict)
    dot_params: dict = Field(default_factory=dict)
    arrow_params: dict = Field(default_factory=dict)
    vaxis_params: dict = Field(default_factory=dict)
    label_params: dict = Field(default_factory=dict)
    setup_yaxis_params: dict = Field(default_factory=dict)

    plot_data: None = None

    def draw(self, ax) -> PlotFunctionResult | None:
        particle_plot(
            ax=ax,
            data=np.asarray(self.data),
            value_names=self.value_names,
            colors=self.colors,
            rescaler=self.rescaler,
            ylims=self.ylims,
            max_line_extend=self.max_line_extend,
            value_spacing=self.value_spacing,
            derivative=np.asarray(self.derivative) if self.derivative is not None else None,
            line_params=self.line_params,
            dot_params=self.dot_params,
            arrow_params=self.arrow_params,
            vaxis_params=self.vaxis_params,
            label_params=self.label_params,
            setup_yaxis_params=self.setup_yaxis_params,
        )
        if self.title:
            ax.set_title(self.title, **self.title_kwargs)
        return PlotFunctionResult(rendering=None, metadata={})


ParticlePanel.model_rebuild(force=True)
