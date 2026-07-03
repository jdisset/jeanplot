"""DataBlockPanel — one SSOT data view, dispatched by input dim. The SAME primitive renders
experimental data OR a model prediction.

UNIFORM FACE SIZING: sub-plots' plotted axes boxes are pinned so tick labels + titles render at ONE
physical size across a table. The block is ALWAYS one row (h) tall: a single-row `[1,N]` z-slice
strip gives full-size slices (== a 2D value face); a grid `[R,C]` packs into that height (each slice
`~h/R` tall). The block WIDTH is DERIVED from the face count (`block_aspect` no longer drives it).
The pins are marked user-set (`_freeze_size`) so the render-time jstyle cascade can't regrow the cells
via `validate_assignment` — otherwise the block would draw wider than the width it reports to the
table's auto column, and spill into the next column.

  secondary_view=True (the paired block):
    1D   smooth curve (+std band)   |  2D density histogram of (x, y)
    2D   smooth value heatmap       |  gradient-magnitude heatmap + gradient field
    3D   data cube                  |  R×C grid of z-slices            (via SmoothPanel3D)
  secondary_view=False (the single value view — the analysis twin is dropped):
    1D   smooth curve   ·   2D   smooth value heatmap   ·   3D   cube + slice grid (always paired)

`is_drawable=False` (renderer hands each leaf its own axes). Face chrome (FACE_PAD / FACE_COLORBAR /
FACE_GAP) and the rigid pin live in smooth_3d.py, shared with the 3D slice grid so a slice and a 2D
face reserve the same margins. All visual defaults come from the jstyle cascade keyed on the
sub-panel types under `DataBlockPanel`; this file wires structure + sizing only.
"""

from typing import Any

from pydantic import Field

from jeanplot.core.models import BoxStyle, LayoutConstraints, Size
from jeanplot.data import PlotData
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.scatter import GridHistogramPanel
from jeanplot.panels.smooth_1d import SmoothPanel1D
from jeanplot.panels.smooth_2d import (
    GradientFieldPanel2D,
    SmoothGradMagnitudePanel2D,
    SmoothPanel2D,
)
from jeanplot.panels.smooth_3d import (
    FACE_COLORBAR,
    FACE_GAP,
    FACE_PAD,
    SmoothPanel3D,
    _freeze_size,
    face_cell_size,
    pin_axes_box,
)


class DataBlockPanel(PlotPanel):
    plot_data: PlotData
    is_drawable: bool = False
    single_aspect: float = 1.0  # face CELL width:height (1 = square). The canonical unit.
    aspect: float = 2.0  # VESTIGIAL: kept for API compat; block size is derived from face count.
    # Drop the 1D density / 2D gradient twin, leaving just the value plot (1D curve / 2D heatmap).
    # No-op for 3D (the cube + slice grid is the value view).
    secondary_view: bool = True
    gap: float = FACE_GAP
    show_colorbar: bool = False
    slice_grid: tuple[int, int] = (3, 3)  # 3D: rows×cols of z-slices in the grid view
    slice_zrange: tuple[float, float] = (0.05, 0.55)  # 3D: latent z range of the slice grid
    slice_colorbar: bool = True  # 3D: independent per-slice colorbar right of each slice
    slice_vlim_quantiles: tuple[float | None, float | None] = (0.02, 0.98)
    slice_axis_titles: bool = False  # 3D: draw x/y axis titles on the slices (cube carries them)
    slice_all_ticks: bool = True  # VESTIGIAL: uniform faces always draw per-cell ticks
    cube_frac_w: float = 0.5  # VESTIGIAL: superseded by cube_cell_units (below)
    cube_cell_units: float = 1.4  # 3D: cube width in face-widths (room for the projection)
    stack_n_slices: int = 4  # 3D: translucent slices drawn inside the cube view
    cube_grid_gap: float = 0.1  # 3D: spacing (inches) between the cube view and the grid
    # 3D only: clip the prediction cube at the ground-truth iso-level (shared contours).
    contour_reference: PlotData | None = Field(default=None)

    def model_post_init(self, ctx: Any) -> None:
        super().model_post_init(ctx)
        if self.children:
            return
        h = self.axes_size.height  # the block height, ALWAYS (a table row is one h tall)
        dim = int(self.plot_data.x.shape[1])
        builder = {1: self._build_1d, 2: self._build_2d, 3: self._build_3d}.get(dim)
        if builder is None:
            raise ValueError(f"DataBlockPanel: unsupported input dim={dim}")
        children, block_w, block_h = builder(h)
        self.layout = LayoutConstraints(direction="row", gap=self.gap, align_items="start")
        self.add_children(children)
        # Rigid block sized to the derived face-unit footprint. Set axes_size too so the base
        # `_compute_min_dimensions` validator (re-derives min from axes_size) can't shrink it.
        object.__setattr__(self, "axes_size", Size(width=block_w, height=block_h))
        object.__setattr__(self, "min_dimensions", Size(width=block_w, height=block_h))
        object.__setattr__(self, "max_dimensions", Size(width=block_w, height=block_h))
        _freeze_size(self)

    def _common(self) -> dict:
        return {"plot_data": self.plot_data, "rescaler": self.rescaler}

    def _face_box(self, h: float) -> tuple[float, float]:
        """The full-height face axes box: as tall as an h-cell allows once FACE_PAD (title strip +
        tick bands) is reserved, `single_aspect`·that wide (square by default)."""
        box_h = h - FACE_PAD.top - FACE_PAD.bottom
        return self.single_aspect * box_h, box_h

    def _row(self, panels, colorbar: bool, box: tuple[float, float]):
        """Block footprint of a row of face panels (each pinned to `box`): n cells + gaps × h."""
        cell_w, cell_h = face_cell_size(box[0], box[1], colorbar=colorbar)
        n = len(panels)
        return panels, n * cell_w + self.gap * (n - 1), cell_h

    def _build_1d(self, h: float):
        box = self._face_box(h)
        left = SmoothPanel1D(show_std=True, style=BoxStyle(padding=FACE_PAD), **self._common())
        pin_axes_box(left, *box)
        if not self.secondary_view:
            return self._row([left], colorbar=False, box=box)
        right = GridHistogramPanel(
            draw_colorbar=self.show_colorbar,
            colorbar_params=dict(FACE_COLORBAR),
            style=BoxStyle(padding=FACE_PAD),
            **self._common(),
        )
        pin_axes_box(right, *box)
        return self._row([left, right], colorbar=self.show_colorbar, box=box)

    def _build_2d(self, h: float):
        box = self._face_box(h)
        left = SmoothPanel2D(
            draw_colorbar=self.show_colorbar,
            colorbar_params=dict(FACE_COLORBAR),
            style=BoxStyle(padding=FACE_PAD),
            **self._common(),
        )
        pin_axes_box(left, *box)
        if not self.secondary_view:
            return self._row([left], colorbar=self.show_colorbar, box=box)
        right = SmoothGradMagnitudePanel2D(
            draw_colorbar=self.show_colorbar,
            colorbar_params=dict(FACE_COLORBAR),
            style=BoxStyle(padding=FACE_PAD),
            **self._common(),
        )
        right.add_child(
            GradientFieldPanel2D(plot_data=self.plot_data, rescaler=self.rescaler, is_overlay=True)
        )
        pin_axes_box(right, *box)
        return self._row([left, right], colorbar=self.show_colorbar, box=box)

    def _build_3d(self, h: float):
        # Cube + slice grid pack into the SAME height h; SmoothPanel3D self-pins to its derived
        # (width, h), so read that back for the block width (no duplicated width formula).
        s3d = SmoothPanel3D(
            plot_data=self.plot_data,
            rescaler=self.rescaler,
            uniform_height=h,
            uniform_single_aspect=self.single_aspect,
            cube_cell_units=self.cube_cell_units,
            slice_grid=self.slice_grid,
            slice_zrange=self.slice_zrange,
            stack_n_slices=self.stack_n_slices,
            cube_grid_gap=self.cube_grid_gap,
            slice_show_colorbar=self.slice_colorbar,
            slice_show_axis_titles=self.slice_axis_titles,
            slice_vlim_quantiles=self.slice_vlim_quantiles,
            cube_contour_reference_plot_data=self.contour_reference,
        )
        return [s3d], s3d.min_dimensions.width, h


def data_block(plot_data: PlotData, **kwargs) -> DataBlockPanel:
    """Code-side twin of the `!DataBlockPanel` tag (mirrors `auto_panel`)."""
    return DataBlockPanel(plot_data=plot_data, **kwargs)


DataBlockPanel.model_rebuild(force=True)
