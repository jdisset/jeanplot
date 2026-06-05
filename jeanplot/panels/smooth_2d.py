"""SmoothPanel2D + gradient variants — generated via `panel_from`. The `smooth_grid_params`
dict param is promoted to a cascade-selectable `SmoothGrid` leaf (`.smooth_grid` field)."""

from jeanplot.panels.from_function import panel_from
from jeanplot.panels.smooth_spec import SmoothGrid
from jeanplot.plots.smooth_2d import (
    gradient_field_2d,
    smooth_2d,
    smooth_grad_magnitude_2d,
)
from jeanplot.plots.txt import smooth_2d_txt

_GRID_LEAF = {"smooth_grid_params": SmoothGrid}

SmoothPanel2D = panel_from(
    smooth_2d, name="SmoothPanel2D", cascade_leaf_params=_GRID_LEAF, txt_fn=smooth_2d_txt
)
SmoothGradMagnitudePanel2D = panel_from(
    smooth_grad_magnitude_2d, name="SmoothGradMagnitudePanel2D", cascade_leaf_params=_GRID_LEAF
)
GradientFieldPanel2D = panel_from(
    gradient_field_2d, name="GradientFieldPanel2D", cascade_leaf_params=_GRID_LEAF
)
