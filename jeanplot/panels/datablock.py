"""DataBlockPanel — one SSOT data view: a fixed-aspect (2:1) pair of square sub-views,
dispatched by input dim. The SAME primitive renders experimental data OR a model prediction:

    1D   smooth curve (+std band)   |  2D density histogram of (x, y)
    2D   smooth value heatmap       |  gradient-magnitude heatmap + gradient field
    3D   data cube                  |  R×C grid of z-slices            (via SmoothPanel3D)

`is_drawable=False` (renderer hands each leaf its own axes). The block pins its OWN size
(min == max == aspect·height × height) so it occupies exactly one rigid 2:1 box and never
overflows its cell. All visual defaults come from the jstyle cascade keyed on the sub-panel
types under `DataBlockPanel`; this file wires structure + sizing only.
"""

from typing import Any

from pydantic import Field

from jeanplot.core.models import LayoutConstraints, Size
from jeanplot.data import PlotData
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.scatter import GridHistogramPanel
from jeanplot.panels.smooth_1d import SmoothPanel1D
from jeanplot.panels.smooth_2d import (
    GradientFieldPanel2D,
    SmoothGradMagnitudePanel2D,
    SmoothPanel2D,
)
from jeanplot.panels.smooth_3d import SmoothPanel3D

# Tiny per-view floor: flex (width) + stretch (height) grow each half to its real size, so
# this only has to be small enough that a half is never forced below its content insets.
_FLOOR = Size(width=0.4, height=0.4)


class DataBlockPanel(PlotPanel):
    plot_data: PlotData
    is_drawable: bool = False
    aspect: float = 2.0  # block width : height
    gap: float = 0.08
    show_colorbar: bool = False
    slice_grid: tuple[int, int] = (3, 3)
    cube_frac_w: float = 0.5
    # 3D only: clip the prediction cube at the ground-truth iso-level (shared contours).
    contour_reference: PlotData | None = Field(default=None)

    def model_post_init(self, ctx: Any) -> None:
        super().model_post_init(ctx)
        if self.children:
            return
        h = self.axes_size.height
        w = self.aspect * h
        dim = int(self.plot_data.x.shape[1])
        builder = {1: self._build_1d, 2: self._build_2d, 3: self._build_3d}.get(dim)
        if builder is None:
            raise ValueError(f"DataBlockPanel: unsupported input dim={dim}")
        children, weights = builder(w, h)
        self.layout = LayoutConstraints(
            direction="row", gap=self.gap, align_items="stretch", main_axis_weights=weights
        )
        self.add_children(children)
        # Rigid 2:1 block: a table column gets an exact width and the halves flex to fill it.
        object.__setattr__(self, "min_dimensions", Size(width=w, height=h))
        object.__setattr__(self, "max_dimensions", Size(width=w, height=h))

    def _common(self) -> dict:
        return {
            "plot_data": self.plot_data,
            "rescaler": self.rescaler,
            "axes_size": _FLOOR.model_copy(),
        }

    def _build_1d(self, w: float, h: float):
        left = SmoothPanel1D(show_std=True, **self._common())
        right = GridHistogramPanel(draw_colorbar=self.show_colorbar, **self._common())
        return [left, right], [1.0, 1.0]

    def _build_2d(self, w: float, h: float):
        left = SmoothPanel2D(draw_colorbar=self.show_colorbar, **self._common())
        right = SmoothGradMagnitudePanel2D(draw_colorbar=self.show_colorbar, **self._common())
        right.add_child(
            GradientFieldPanel2D(plot_data=self.plot_data, rescaler=self.rescaler, is_overlay=True)
        )
        return [left, right], [1.0, 1.0]

    def _build_3d(self, w: float, h: float):
        cube_and_grid = SmoothPanel3D(
            plot_data=self.plot_data,
            rescaler=self.rescaler,
            axes_size=Size(width=w, height=h),
            slice_grid=self.slice_grid,
            cube_frac_w=self.cube_frac_w,
            slice_show_colorbar=self.show_colorbar,
            cube_contour_reference_plot_data=self.contour_reference,
        )
        return [cube_and_grid], [1.0]


def data_block(plot_data: PlotData, **kwargs) -> DataBlockPanel:
    """Code-side twin of the `!DataBlockPanel` tag (mirrors `auto_panel`)."""
    return DataBlockPanel(plot_data=plot_data, **kwargs)


DataBlockPanel.model_rebuild(force=True)
