"""SmoothPanel2D + gradient variants — generated via `panel_from`."""

from jeanplot.panels.from_function import panel_from
from jeanplot.plots.smooth_2d import (
    gradient_field_2d,
    smooth_2d,
    smooth_grad_magnitude_2d,
)
from jeanplot.plots.txt import smooth_2d_txt

SmoothPanel2D = panel_from(smooth_2d, name="SmoothPanel2D", txt_fn=smooth_2d_txt)
SmoothGradMagnitudePanel2D = panel_from(smooth_grad_magnitude_2d, name="SmoothGradMagnitudePanel2D")
GradientFieldPanel2D = panel_from(gradient_field_2d, name="GradientFieldPanel2D")
