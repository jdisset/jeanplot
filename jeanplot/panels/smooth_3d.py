"""SmoothPanel3D — a Container with cube view + R×C grid of 2D slices.

Per the refactor spec: this panel composes nested Containers and is not itself
drawable. The renderer hands each child its own matplotlib Axes from the laid-out
bbox.
"""

from typing import Any

import numpy as np
from pydantic import Field

from jeanplot.core.container import Container
from jeanplot.core.models import LayoutConstraints
from jeanplot.data import PlotData, PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.smooth_2d import SmoothPanel2D
from jeanplot.plots.cube import draw_cube_wireframe


class CubeView(PlotPanel):
    plot_data: PlotData
    xlims: tuple[float, float] = (0.0, 1.0)
    ylims: tuple[float, float] = (0.0, 1.0)
    zlims: tuple[float, float] = (0.0, 1.0)
    projection_angle: float = 45.0
    projection_diag_coef: float = 0.5
    edge_color: str = "#444444"
    edge_lw: float = 0.5

    def draw(self, ax) -> PlotFunctionResult | None:
        draw_cube_wireframe(
            ax,
            xlim=self.xlims,
            ylim=self.ylims,
            zlim=self.zlims,
            projection_angle=self.projection_angle,
            projection_diag_coef=self.projection_diag_coef,
            edge_color=self.edge_color,
            edge_lw=self.edge_lw,
            xtitle=self.xtitle
            or (self.plot_data.input_names[0] if self.plot_data.input_names else None),
            ytitle=self.ytitle
            or (self.plot_data.input_names[1] if len(self.plot_data.input_names) > 1 else None),
            ztitle=self.vtitle
            or (self.plot_data.input_names[2] if len(self.plot_data.input_names) > 2 else None),
        )
        if self.title:
            ax.set_title(self.title, **self.title_kwargs)
        return PlotFunctionResult(rendering=None, metadata={})


class SmoothPanel3D(PlotPanel):
    plot_data: PlotData
    zslices: list[float] = Field(default_factory=lambda: [0.05, 0.25, 0.4, 0.55])
    slice_grid: tuple[int, int] = (3, 3)
    cube_frac_w: float = 0.57
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    ylims: tuple[float | None, float | None] = (None, None)
    zlims: tuple[float | None, float | None] = (None, None)
    vlims: tuple[float | None, float | None] = (None, None)
    projection_angle: float = 45.0
    projection_diag_coef: float = 0.5

    is_drawable: bool = False

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.children:
            return

        rows, cols = self.slice_grid
        n_slices = rows * cols
        if len(self.zslices) >= 2:
            zs = np.linspace(self.zslices[0], self.zslices[-1], n_slices)
        else:
            zs = np.linspace(0.0, 1.0, n_slices)

        cube = CubeView(
            plot_data=self.plot_data,
            rescaler=self.rescaler,
            xlims=(self.xlims[0] or 0.0, self.xlims[1] or 1.0),
            ylims=(self.ylims[0] or 0.0, self.ylims[1] or 1.0),
            zlims=(self.zlims[0] or 0.0, self.zlims[1] or 1.0),
            projection_angle=self.projection_angle,
            projection_diag_coef=self.projection_diag_coef,
            title=self.title,
            title_kwargs=self.title_kwargs,
        )

        slice_panels = [
            SmoothPanel2D(
                plot_data=self.plot_data,
                rescaler=self.rescaler,
                zslice=[float(z)],
                xlims=self.xlims,
                ylims=self.ylims,
                vlims=self.vlims,
                draw_colorbar=False,
                title=f"z={z:.2f}",
            )
            for z in zs
        ]
        slice_rows = [
            Container(
                layout=LayoutConstraints(direction="row", gap=4),
                children=slice_panels[r * cols : (r + 1) * cols],
            )
            for r in range(rows)
        ]
        slice_grid_container = Container(
            layout=LayoutConstraints(direction="column", gap=4),
            children=slice_rows,
        )

        self.layout = LayoutConstraints(
            direction="row",
            gap=8,
            main_axis_weights=[self.cube_frac_w, 1.0 - self.cube_frac_w],
        )
        self.add_children([cube, slice_grid_container])

    def render_txt(self) -> str | None:
        from jeanplot.plots.txt import smooth_3d_txt

        zslices_arr = np.atleast_2d(np.asarray(self.zslices, dtype=float))
        result = smooth_3d_txt(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            zslices=zslices_arr,
            xlims=self.xlims,
            ylims=self.ylims,
            zlims=self.zlims,
            vlims=self.vlims,
            title=self.title,
        )
        return str(result)


CubeView.model_rebuild(force=True)
SmoothPanel3D.model_rebuild(force=True)
