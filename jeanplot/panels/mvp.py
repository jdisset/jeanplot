"""MVPPanel — generated via `panel_from`."""

from jeanplot.panels.from_function import panel_from
from jeanplot.plots.mvp import measured_vs_predicted

MVPPanel = panel_from(
    measured_vs_predicted,
    name="MVPPanel",
    plot_data_keys=(),
)
