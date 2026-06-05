"""SmoothPanel1D — generated via `panel_from`. The `smooth_params` dict param is promoted
to a cascade-selectable `SmoothKernel` leaf (`.smooth` field), shared with the 2D grids."""

from jeanplot.panels.from_function import panel_from
from jeanplot.panels.smooth_spec import SmoothKernel
from jeanplot.plots.smooth_1d import smooth_1d
from jeanplot.plots.txt import smooth_1d_txt

SmoothPanel1D = panel_from(
    smooth_1d,
    name="SmoothPanel1D",
    cascade_leaf_params={"smooth_params": SmoothKernel},
    txt_fn=smooth_1d_txt,
)
