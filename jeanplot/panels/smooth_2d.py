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

# colorbar() signature defaults (axes-fraction) for the band when colorbar_params omits them.
_CBAR_POS, _CBAR_SIZE = (1.1, 0.4), (0.04, 0.52)


def _colorbar_right_overflow(self) -> float:
    """Inches the out-of-axes colorbar (drawn at axes-fraction position+size > 1) spills
    past the axes box: the measured band overflow scaled by axes width, plus a
    `label_reserve` allowance (ticks + rotated axis label). Reserved as right inset by
    `PlotPanel.effective_padding`, so the layout no longer clips/overlaps the colorbar.
    `draw_colorbar=False` (a real field on every heatmap panel) reserves nothing."""
    if not getattr(self, "draw_colorbar", True):
        return 0.0
    cp = self.colorbar_params or {}
    pos = cp.get("position", _CBAR_POS)
    size = cp.get("size", _CBAR_SIZE)
    band_overflow = max(0.0, (pos[0] + size[0]) - 1.0) * self.axes_size.width
    return band_overflow + cp.get("label_reserve", 0.5)


SmoothPanel2D = panel_from(
    smooth_2d, name="SmoothPanel2D", cascade_leaf_params=_GRID_LEAF, txt_fn=smooth_2d_txt
)
# `draw_colorbar` isn't a `smooth_grad_magnitude_2d` arg (it never draws one inline), but
# it IS the knob `_colorbar_right_overflow` reads to reserve the right gutter. Expose it as
# a panel field (default True keeps the historical gutter so it aligns with a sibling value
# heatmap) so a caller can set `draw_colorbar=False` to drop the reserve (e.g. DataBlock).
SmoothGradMagnitudePanel2D = panel_from(
    smooth_grad_magnitude_2d,
    name="SmoothGradMagnitudePanel2D",
    cascade_leaf_params=_GRID_LEAF,
    field_overrides={"draw_colorbar": (bool, True)},
)
GradientFieldPanel2D = panel_from(
    gradient_field_2d, name="GradientFieldPanel2D", cascade_leaf_params=_GRID_LEAF
)

# both heatmap panels draw an out-of-axes colorbar -> auto-reserve its width (replaces the
# old hand-tuned `style.padding.right` in the paper theme). The gradient *field* (quiver)
# has no colorbar, so it keeps the base 0.
SmoothPanel2D._right_overflow = _colorbar_right_overflow
SmoothGradMagnitudePanel2D._right_overflow = _colorbar_right_overflow
