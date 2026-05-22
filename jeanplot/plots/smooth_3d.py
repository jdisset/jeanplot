"""Cabinet-projected 3D smooth-heatmap orchestrator.

Drives `plot_3d_stack` over `smooth_2d` slices: faces + projected 2D slices
into a single matplotlib axes. Symbolic-contour thresholds (e.g. "otsu:0.85")
are resolved globally across all slices so every face clips at the same level.
"""

from collections.abc import Sequence
from functools import partial

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray as NdArray

from jeanplot.data import GridData, PlotFunctionResult
from jeanplot.plots.colorbar import colorbar
from jeanplot.plots.cube import (
    DEFAULT_CUBE_EDGE_PROPS,
    PROJ_ALPHA,
    PROJ_D,
    back_face,
    cabinet_project,
    front_face_bl,
    front_face_tr,
    plot_3d_stack,
)
from jeanplot.plots.heatmap import _resolve_symbolic_level
from jeanplot.plots.smooth_2d import smooth_2d
from jeanplot.plots.smooth_kernel import knn_grid
from jeanplot.plots.ticks import format_powers, get_transformed_ticks_and_labels


def _resolve_global_contours(X, Y, xlims, ylims, zslices, smooth_2d_params):
    hp = smooth_2d_params.get("heatmap_params", {})
    contours = hp.get("contours")
    if not (isinstance(contours, list | tuple) and any(isinstance(c, str) for c in contours)):
        return smooth_2d_params

    knn_params = smooth_2d_params.get("knn_grid_params", {}) or {}
    finite_vals = []
    for zentry in zslices:
        for pos in np.atleast_1d(zentry):
            _, ov = knn_grid(X, Y, xlims, ylims, **{**knn_params, "zslice": np.atleast_1d(pos)})
            ov = np.asarray(ov)
            finite_vals.append(ov[np.isfinite(ov)])
    if not finite_vals:
        return smooth_2d_params
    cube_finite = np.concatenate(finite_vals) if len(finite_vals) > 1 else finite_vals[0]
    if cube_finite.size == 0:
        return smooth_2d_params
    resolved = [_resolve_symbolic_level(c, cube_finite) for c in contours]
    resolved = [c for c in resolved if not isinstance(c, str)]
    if not resolved:
        return smooth_2d_params
    return {**smooth_2d_params, "heatmap_params": {**hp, "contours": resolved}}


def smooth_3d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax: Sequence[Axes] | Axes,
    zslices,
    xlims=(0, 1),
    ylims=(None, None),
    zlims=(None, None),
    vlims=(None, None),
    draw_colorbar: bool | None = None,
    cube_edge_props: dict | None = None,
    projection_angle: float = PROJ_ALPHA,
    projection_diag_coef: float = PROJ_D,
    colorbar_position=(1.1, 0.4),
    colorbar_size=(0.04, 0.52),
    colorbar_params: dict | None = None,
    show_inner_spines: bool = True,
    show_slice_ticks: bool = True,
    smooth_2d_params: dict | None = None,
    show_front_face_ticks: bool = False,
    xtitle: str | None = None,
    ytitle: str | None = None,
    ztitle: str | None = None,
    title: str | None = None,
    xaxis_labelpad: int = 20,
    yaxis_labelpad: int = 24,
    zaxis_labelpad: int = 0,
    **_,
):
    smooth_2d_params = dict(smooth_2d_params or {})
    colorbar_params = dict(colorbar_params or {})
    cube_edge_props = cube_edge_props if cube_edge_props is not None else DEFAULT_CUBE_EDGE_PROPS

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        import logging
        n_invalid = int(
            np.sum(~np.all(np.isfinite(X), axis=1) | ~np.all(np.isfinite(Y), axis=1))
        )
        logging.getLogger(__name__).warning(
            f"smooth_3d: filtering {n_invalid}/{len(X)} non-finite rows"
        )

    smooth_2d_params = _resolve_global_contours(X, Y, xlims, ylims, zslices, smooth_2d_params)

    project = partial(cabinet_project, alpha=projection_angle, d=projection_diag_coef)

    if isinstance(ax, Axes):
        ax = [ax]
    assert len(ax) == len(zslices), (
        f"axes and zslices must have same length, got {len(ax)} vs {len(zslices)}"
    )

    ylims = xlims if ylims == (None, None) else ylims
    zlims = xlims if zlims == (None, None) else zlims
    cbar_loc = tuple(colorbar_position) + tuple(colorbar_size)
    all_grid_data: list[GridData] = []

    def _slice_drawer(zslice, cbar_ax, slice_index):
        def _draw(sl_ax: Axes):
            res = smooth_2d(
                X, Y, input_names, output_name,
                **{
                    **smooth_2d_params,
                    "ax": sl_ax,
                    "rescaler": rescaler,
                    "zslice": zslice,
                    "xlims": xlims,
                    "ylims": ylims,
                    "vlims": vlims,
                },
            )
            im = res.mappable if isinstance(res, PlotFunctionResult) else None
            if isinstance(res, PlotFunctionResult) and "grid_data" in (res.metadata or {}):
                z_val = float(zslice[0]) if hasattr(zslice, "__len__") and len(zslice) > 0 else float(zslice)
                for gd in res.metadata["grid_data"]:
                    all_grid_data.append(
                        GridData(
                            x_coords=gd.x_coords, y_coords=gd.y_coords, values=gd.values,
                            xlims=gd.xlims, ylims=gd.ylims, resolution=gd.resolution,
                            input_names=gd.input_names, output_name=gd.output_name,
                            z_value=z_val,
                        )
                    )
            sl_ax.set_xlabel("")
            sl_ax.set_ylabel("")
            sl_ax.set_facecolor("none")
            sl_ax.yaxis.label.set_zorder(2)
            sl_ax.xaxis.label.set_zorder(2)
            if slice_index is not None and len(zslice) > 0:
                z_val = float(zslice[0]) if hasattr(zslice, "__len__") else float(zslice)
                gid = f"jeanplot_3dslice_{slice_index}_z{z_val:.4f}"
                sl_ax.set_gid(gid)
                if im is not None:
                    im.set_gid(gid + "_image")
            if not show_inner_spines:
                for s in ("top", "right", "bottom", "left"):
                    sl_ax.spines[s].set_visible(False)
                sl_ax.set_xticks([])
                sl_ax.set_yticks([])
                sl_ax.set_xticks([], minor=True)
                sl_ax.set_yticks([], minor=True)
            for label in sl_ax.get_xticklabels() + sl_ax.get_yticklabels():
                label.set_bbox(dict(facecolor="white", edgecolor="None", alpha=1, pad=0.75, zorder=1.5))
            if cbar_ax is not None and im is not None:
                colorbar(sl_ax, im, rescaler, vlims, label=output_name, cax=cbar_ax, **colorbar_params)
        return _draw

    zticks, zlabels = get_transformed_ticks_and_labels(
        np.asarray(zlims) + np.array((0.1, 0)),
        rescaler=rescaler,
        skip_ticklabel_range=(-10, 2000),
    )
    major_zlabels = [(float(z), s, "major") for z, s in zlabels]
    zticks["major"] = np.asarray(zticks["major"])
    zticks["major"] = zticks["major"][zticks["major"] > 0.0]
    zticks["minor"] = np.asarray(zticks["minor"])
    zticks["minor"] = zticks["minor"][zticks["minor"] > 0.0]
    ztitle = ztitle if ztitle is not None else input_names[2]

    for i, s in enumerate(zslices):
        slice_ax = ax[i]
        positions = np.atleast_1d(s)
        s_zticks = dict(zticks)
        s_zlabels = list(major_zlabels)
        draw_cbar = draw_colorbar if draw_colorbar is not None else i == len(zslices) - 1
        cbar_ax = slice_ax.inset_axes(cbar_loc) if draw_cbar else None

        slice_funcs = []
        slice_ticks: list[float] = []
        slice_labels: list[tuple[float, str, str]] = []
        for j, pos in enumerate(positions):
            slice_funcs.append(_slice_drawer(np.atleast_1d(pos), cbar_ax, i * 10 + j))
            if show_slice_ticks:
                slice_ticks.append(float(pos))
                slice_labels.append(
                    (float(pos), f"$ \\approx $ {format_powers(rescaler.inv(pos), n_decimals=0)}", "slice")
                )
        s_zticks["slice"] = np.asarray(slice_ticks)
        s_zlabels += slice_labels

        plot_3d_stack(
            slice_ax,
            [
                partial(
                    front_face_bl,
                    labelpad=(xaxis_labelpad, yaxis_labelpad),
                    xlims=xlims, ylims=ylims,
                    input_names=input_names, rescaler=rescaler,
                    ticks=show_front_face_ticks,
                    xtitle=xtitle, ytitle=ytitle,
                ),
                partial(front_face_tr, xlims=xlims, ylims=ylims),
                *slice_funcs,
                partial(back_face, xlims=xlims, ylims=ylims),
            ],
            [zlims[0], zlims[0], *positions, zlims[1]],
            xlim=xlims, ylim=ylims, zlim=zlims,
            project=project, zticks=s_zticks, zlabels=s_zlabels,
            cube_edge_props=cube_edge_props,
            z_title=ztitle, zaxis_labelpad=zaxis_labelpad,
        )

    if title is not None:
        ax[0].set_title(title)

    if all_grid_data:
        return PlotFunctionResult(rendering=None, metadata={"grid_data": all_grid_data})
    return None
