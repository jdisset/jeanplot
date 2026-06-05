"""SmoothPanel3D — a Container with cube view + R×C grid of 2D slices.

Per the refactor spec: this panel composes nested Containers and is not itself
drawable. The renderer hands each child its own matplotlib Axes from the laid-out
bbox.
"""

from typing import Any

import numpy as np
from pydantic import Field

from jeanplot.core.container import Container
from jeanplot.core.models import BoxInset, BoxStyle, LayoutConstraints, Size
from jeanplot.data import PlotData, PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.smooth_2d import SmoothPanel2D
from jeanplot.plots.cube import draw_cube_wireframe


def _format_z_label(z_latent: float, rescaler=None, prefix: str = "z=") -> str:
    if rescaler is None:
        return f"{prefix}{float(z_latent):.2f}"
    from jeanplot.plots.ticks import format_powers

    return f"{prefix}{format_powers(float(rescaler.inv(float(z_latent))), n_decimals=0)}"


def _split_userset(parent: Any, spec: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split {child_kw: parent_field} into (user-set, default) kwargs by whether the
    parent field was explicitly set. User-set lims are passed at construction so they
    stay locked and beat the theme; the rest are applied via `with_defaults` so they're
    cascade-fillable (e.g. `SmoothPanel3D[id=prediction] CubeStackPanel: {vlims: ...}`
    reaches a cube that is built here, before the render-time cascade runs)."""
    user: dict[str, Any] = {}
    default: dict[str, Any] = {}
    for child_kw, pfield in spec.items():
        (user if pfield in parent.model_fields_set else default)[child_kw] = getattr(parent, pfield)
    return user, default


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
    slice_grid: tuple[int, int] = (3, 3)
    slice_zrange: tuple[float, float] = (0.05, 0.55)
    slice_zvalues: list[float] | None = None
    zslices: list[float] | None = None
    stack_zslices: list[float] | None = None
    stack_zrange: tuple[float, float] = (0.05, 0.55)
    stack_n_slices: int = 4
    cube_frac_w: float = 0.45
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    ylims: tuple[float | None, float | None] = (None, None)
    zlims: tuple[float | None, float | None] = (None, None)
    vlims: tuple[float | None, float | None] = (None, None)
    # Slice-grid color scale, independent of the cube's `vlims`. Defaults to
    # (None, None) so every slice auto-scales to its own value range, using
    # `slice_vlim_quantiles` for full per-slice contrast (no floor/range clamp).
    slice_vlims: tuple[float | None, float | None] = (None, None)
    slice_vlim_quantiles: tuple[float | None, float | None] = (0.02, 0.98)
    projection_angle: float = 45.0
    projection_diag_coef: float = 0.5
    slice_show_colorbar: bool = True
    slice_title_fontsize: int = 7
    slice_title_color: str = "#777777"
    slice_title_pad: float = 3.0
    cube_show_inner_spines: bool = False
    cube_show_slice_ticks: bool = False
    cube_show_front_face_ticks: bool = True
    # Resolve the cube's symbolic contour levels from this field instead of
    # `plot_data` (e.g. clip a prediction cube at the ground-truth iso-level).
    cube_contour_reference_plot_data: PlotData | None = None
    cube_smooth_2d_params: dict | None = Field(
        default_factory=lambda: {
            "draw_colorbar": False,
            "draw_xlabel": False,
            "draw_ylabel": False,
            "xtitle": "",
            "ytitle": "",
            "vtitle": "",
            "setup_transformed_axis_params": {
                "setup_xaxis_params": {"show_labels": False},
                "setup_yaxis_params": {"show_labels": False},
            },
        }
    )
    cube_colorbar_params: dict | None = None

    is_drawable: bool = False

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.children:
            return

        rows, cols = self.slice_grid
        n_slices = rows * cols

        # Uniform per-cell reservations (inches): a title strip on top, a colorbar on
        # the right, x/y tick-label room on the bottom/left. Kept IDENTICAL on every
        # cell so the equal flex weights below give every slice the same plot area at
        # any final panel size. (Size-dependent weights only equalize at one exact
        # size, which is why the bottom row drifted.) Which cells actually draw tick
        # labels is decided by show_labels below, not by changing their size.
        gap = 0.05
        # Per-cell padding is uniform (title strip on top, colorbar on the right, a
        # hairline elsewhere) so the equal flex weights below give every slice the
        # same plot area at any panel size. The bottom-row x labels and left-column y
        # labels live in the GRID container's margin (grid_pad), NOT per-cell padding,
        # so interior cells don't each waste a tick band -- that's what keeps the
        # slices large while staying equal.
        pad_title, pad_cbar, pad_edge = 0.16, 0.30, 0.03
        pad_xaxis, pad_yaxis = 0.40, 0.44
        cell_pad = BoxStyle(
            padding=BoxInset(top=pad_title, right=pad_cbar, bottom=pad_edge, left=pad_edge)
        )
        grid_pad = BoxStyle(padding=BoxInset(bottom=pad_xaxis, left=pad_yaxis))
        # Hug the heatmap with a thin, tall colorbar so it costs little width.
        slice_colorbar_params = {"position": (1.04, 0.08), "size": (0.05, 0.84)}

        # axes_size sizes the per-cell min_dimensions so the panel honors the size the
        # caller asked for (per_network_row's per-cell width x panel_scale). The equal
        # weights drive the actual cell sizes; this just sets a sensible floor.
        if "axes_size" in self.model_fields_set:
            w, h = self.axes_size.width, self.axes_size.height
            grid_w = w * (1.0 - self.cube_frac_w) - pad_yaxis
            grid_h = h - pad_xaxis
            cell_w = max(0.3, (grid_w - gap * (cols - 1)) / cols - (pad_edge + pad_cbar))
            cell_h = max(0.3, (grid_h - gap * (rows - 1)) / rows - (pad_title + pad_edge))
            cube_size = {"axes_size": Size(width=w * self.cube_frac_w, height=h)}
            cell_size = {"axes_size": Size(width=cell_w, height=cell_h)}
        else:
            cube_size = {}
            cell_size = {}

        if self.slice_zvalues is not None:
            zs = np.asarray(self.slice_zvalues, dtype=float)
            assert zs.size == n_slices, (
                f"slice_zvalues has {zs.size} entries, expected R*C={n_slices}"
            )
        else:
            zs = np.linspace(self.slice_zrange[0], self.slice_zrange[1], n_slices)

        cube_zs: list[float] = (
            list(self.stack_zslices)
            if self.stack_zslices is not None
            else list(self.zslices)
            if self.zslices is not None
            else list(np.linspace(self.stack_zrange[0], self.stack_zrange[1], self.stack_n_slices))
        )

        cube_user, cube_def = _split_userset(
            self, {"xlims": "xlims", "ylims": "ylims", "zlims": "zlims", "vlims": "vlims"}
        )
        cube = CubeStackPanel(
            plot_data=self.plot_data,
            rescaler=self.rescaler,
            zslices=[cube_zs],
            **cube_user,
            **cube_size,
            projection_angle=self.projection_angle,
            projection_diag_coef=self.projection_diag_coef,
            title=self.title,
            title_kwargs=self.title_kwargs,
            show_inner_spines=self.cube_show_inner_spines,
            show_slice_ticks=self.cube_show_slice_ticks,
            show_front_face_ticks=self.cube_show_front_face_ticks,
            contour_reference_plot_data=self.cube_contour_reference_plot_data,
            smooth_2d_params=self.cube_smooth_2d_params,
            colorbar_params=self.cube_colorbar_params,
            draw_colorbar=False,
        )
        cube.with_defaults(**cube_def)

        slice_user, slice_def = _split_userset(
            self, {"xlims": "xlims", "ylims": "ylims", "vlims": "slice_vlims"}
        )
        slice_panels: list[SmoothPanel2D] = []
        for i, z in enumerate(zs):
            r, c = i // cols, i % cols
            is_left = c == 0
            is_bottom = r == rows - 1
            title_label = _format_z_label(float(z), self.rescaler)
            sp = SmoothPanel2D(
                plot_data=self.plot_data,
                rescaler=self.rescaler,
                zslice=[float(z)],
                **slice_user,
                **cell_size,
                style=cell_pad,
                vlim_quantiles=self.slice_vlim_quantiles,
                vlim_min_floor=None,
                vlim_min_range=None,
                # Per-slice colorbar: each slice auto-scales to its own range, so every
                # cell needs its own scale. The label would just repeat the output name
                # in every cell, so it's left off.
                draw_colorbar=self.slice_show_colorbar,
                draw_colorbar_label=False,
                colorbar_params=slice_colorbar_params,
                draw_xlabel=is_bottom,
                draw_ylabel=is_left,
                # Shared-axes grid: only the bottom row / left column show tick labels;
                # interior cells suppress them (draw_x/ylabel toggle only the axis
                # title, not the ticks).
                setup_transformed_axis_params={
                    "setup_xaxis_params": {"show_labels": is_bottom},
                    "setup_yaxis_params": {"show_labels": is_left},
                },
                # Title sits in the reserved top strip, above the heatmap (not over it).
                title=title_label,
                title_inside=False,
                title_kwargs={
                    "color": self.slice_title_color,
                    "fontsize": self.slice_title_fontsize,
                    "pad": self.slice_title_pad,
                },
            )
            sp.with_defaults(**slice_def)
            slice_panels.append(sp)
        slice_rows = [
            Container(
                layout=LayoutConstraints(
                    direction="row",
                    gap=gap,
                    align_items="stretch",
                    main_axis_weights=[1.0] * cols,
                ),
                children=slice_panels[r * cols : (r + 1) * cols],
            )
            for r in range(rows)
        ]
        slice_grid_container = Container(
            style=grid_pad,
            layout=LayoutConstraints(
                direction="column",
                gap=gap,
                align_items="stretch",
                main_axis_weights=[1.0] * rows,
            ),
            children=slice_rows,
        )

        self.layout = LayoutConstraints(
            direction="row",
            gap=0.1,
            align_items="stretch",
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


class CubeStackPanel(PlotPanel):
    plot_data: PlotData
    zslices: list = Field(default_factory=lambda: [[0.05, 0.25, 0.4, 0.55]])
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    ylims: tuple[float | None, float | None] = (None, None)
    zlims: tuple[float | None, float | None] = (None, None)
    vlims: tuple[float | None, float | None] = (None, None)
    projection_angle: float = 45.0
    projection_diag_coef: float = 0.5
    draw_colorbar: bool | None = None
    show_inner_spines: bool = True
    show_slice_ticks: bool = True
    show_front_face_ticks: bool = False
    smooth_2d_params: dict | None = None
    colorbar_params: dict | None = None
    # When set, symbolic contour levels are resolved from this field instead of
    # `plot_data` (lets a prediction cube clip at the ground-truth iso-level).
    contour_reference_plot_data: PlotData | None = None
    cube_edge_props: dict | None = None
    xaxis_labelpad: int = 20
    yaxis_labelpad: int = 24
    zaxis_labelpad: int = 0
    xtitle: str | None = None
    ytitle: str | None = None
    ztitle: str | None = None
    vtitle: str | None = None

    def _face_smooth_grid_params(self) -> dict | None:
        """Cube faces are SmoothPanel2D: resolve one against the cascade so the shared
        smoothing rules (+ `CubeStackPanel SmoothGrid` specializations) reach its leaf."""
        from jeanplot.core.style import jstyle
        from jeanplot.panels.smooth_2d import SmoothPanel2D

        face = SmoothPanel2D(plot_data=self.plot_data, is_drawable=False)
        face.parent = self
        jstyle.apply_one(face)
        return face.smooth_grid.params

    def draw(self, ax) -> PlotFunctionResult | None:
        from jeanplot.plots.smooth_3d import smooth_3d

        ref = self.contour_reference_plot_data
        contour_reference = (ref.x, ref.y) if ref is not None else None
        smooth_2d_params = dict(self.smooth_2d_params or {})
        sgp = self._face_smooth_grid_params()
        if sgp is not None:
            smooth_2d_params["smooth_grid_params"] = sgp
        return smooth_3d(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            rescaler=self.rescaler,
            ax=[ax],
            zslices=self.zslices,
            xlims=self.xlims,
            ylims=self.ylims,
            zlims=self.zlims,
            vlims=self.vlims,
            contour_reference=contour_reference,
            draw_colorbar=self.draw_colorbar,
            cube_edge_props=self.cube_edge_props,
            projection_angle=self.projection_angle,
            projection_diag_coef=self.projection_diag_coef,
            colorbar_params=self.colorbar_params,
            show_inner_spines=self.show_inner_spines,
            show_slice_ticks=self.show_slice_ticks,
            show_front_face_ticks=self.show_front_face_ticks,
            smooth_2d_params=smooth_2d_params,
            xtitle=self.xtitle,
            ytitle=self.ytitle,
            ztitle=self.ztitle,
            title=self.title,
            xaxis_labelpad=self.xaxis_labelpad,
            yaxis_labelpad=self.yaxis_labelpad,
            zaxis_labelpad=self.zaxis_labelpad,
        )


CubeView.model_rebuild(force=True)
SmoothPanel3D.model_rebuild(force=True)
CubeStackPanel.model_rebuild(force=True)
