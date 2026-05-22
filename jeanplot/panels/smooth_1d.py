"""SmoothPanel1D — generated via `panel_from`."""

from jeanplot.panels.from_function import panel_from
from jeanplot.plots.smooth_1d import smooth_1d
from jeanplot.plots.txt import smooth_1d_txt

SmoothPanel1D = panel_from(smooth_1d, name="SmoothPanel1D", txt_fn=smooth_1d_txt)
