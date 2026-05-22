"""2D smooth heatmap entry points + gradient field.

Adapted from biocomp `plotting_smooth_2d.py` (lines 437-785) with biocomp
imports replaced. The shared bits live in `jeanplot.plots.smooth_kernel`.
"""

from collections.abc import Sequence
from typing import Literal, NamedTuple, TypeAlias, TypeVar

import numpy as np

from jeanplot.data import PlotFunctionResult
from jeanplot.plots.smooth_kernel import (
    _finite_xy,
    _render_smooth_heatmap,
    _resolve_lims,
    knn_grid,
)

T = TypeVar("T")
ListOrSingle: TypeAlias = list[T] | T
NdArray = np.ndarray


class KnnGradientField(NamedTuple):
    input_coords: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    x1_lat: np.ndarray
    x2_lat: np.ndarray
    xlims: list
    ylims: list


def smooth_2d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    zslice: NdArray | None = None,
    title: str | None = None,
    title_kwargs: dict | None = None,
    xtitle: str | None = None,
    ytitle: str | None = None,
    vtitle: str | None = None,
    xlims=(0, 1),
    ylims=(None, None),
    vlims=(None, None),
    vlim_quantiles: tuple[float | None, float | None] | None = (0.01, 0.99),
    vlim_min_floor: float | None = None,
    vlim_min_range: float | None = None,
    draw_xlabel=True,
    draw_ylabel=True,
    xaxis_labelpad=None,
    yaxis_labelpad=None,
    draw_colorbar=True,
    draw_colorbar_label=True,
    colorbar_params: dict | None = None,
    knn_grid_params: dict | None = None,
    heatmap_params: dict | None = None,
    setup_transformed_axis_params: dict | None = None,
) -> PlotFunctionResult:
    if isinstance(ax, list | tuple):
        ax = ax[0]
    knn_grid_params = dict(knn_grid_params or {})
    xlims, ylims = _resolve_lims(X, xlims, ylims)
    X, Y = _finite_xy(X, Y)
    zslice = np.asarray(zslice) if zslice is not None else None
    resolution = knn_grid_params.get("grid_resolution", 200)
    input_coords, output_values = knn_grid(
        X,
        Y,
        xlims,
        ylims,
        **{**knn_grid_params, "zslice": zslice},
    )
    return _render_smooth_heatmap(
        ax,
        input_coords,
        output_values,
        input_names,
        output_name,
        rescaler,
        rescaler,
        xlims,
        ylims,
        resolution,
        title=title,
        title_kwargs=title_kwargs,
        xtitle=xtitle,
        ytitle=ytitle,
        vtitle=vtitle,
        vlims=vlims,
        vlim_quantiles=vlim_quantiles,
        vlim_min_floor=vlim_min_floor,
        vlim_min_range=vlim_min_range,
        draw_xlabel=draw_xlabel,
        draw_ylabel=draw_ylabel,
        xaxis_labelpad=xaxis_labelpad,
        yaxis_labelpad=yaxis_labelpad,
        draw_colorbar=draw_colorbar,
        draw_colorbar_label=draw_colorbar_label,
        colorbar_params=colorbar_params,
        heatmap_params=heatmap_params,
        setup_transformed_axis_params=setup_transformed_axis_params,
    )


def knn_gradient_grid(
    X: NdArray,
    Y: NdArray,
    xlims,
    ylims,
    knn_grid_params: dict | None = None,
    space: Literal["raw", "latent"] = "latent",
    rescaler=None,
) -> KnnGradientField:
    knn_grid_params = dict(knn_grid_params or {})
    resolution = knn_grid_params.get("grid_resolution", 200)
    input_coords, output_values = knn_grid(X, Y, xlims, ylims, **knn_grid_params)
    y_lat = np.asarray(output_values).reshape(resolution, resolution)
    x1_lat = np.linspace(xlims[0], xlims[1], resolution)
    x2_lat = np.linspace(ylims[0], ylims[1], resolution)

    if space == "raw":
        assert rescaler is not None, "rescaler required for space='raw'"
        x1_axis = np.asarray(rescaler.inv(x1_lat[:, None]).squeeze())
        x2_axis = np.asarray(rescaler.inv(x2_lat[:, None]).squeeze())
        y_field = np.asarray(rescaler.inv(y_lat[..., None]).squeeze())
    else:
        x1_axis, x2_axis, y_field = x1_lat, x2_lat, y_lat

    nan_mask = ~np.isfinite(y_field)
    if nan_mask.any() and not nan_mask.all():
        from scipy.ndimage import distance_transform_edt

        _, (ii, jj) = distance_transform_edt(nan_mask, return_indices=True)
        y_filled = y_field[ii, jj]
    else:
        y_filled = y_field

    gy, gx = np.gradient(y_filled, x2_axis, x1_axis)
    gy = np.where(nan_mask, np.nan, gy)
    gx = np.where(nan_mask, np.nan, gx)
    return KnnGradientField(input_coords, gx, gy, x1_lat, x2_lat, xlims, ylims)


def smooth_grad_magnitude_2d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    space: Literal["raw", "latent"] = "latent",
    title: str | None = None,
    title_kwargs: dict | None = None,
    xtitle: str | None = None,
    ytitle: str | None = None,
    vtitle: str | None = None,
    xlims=(0, 1),
    ylims=(None, None),
    vlims=(None, None),
    knn_grid_params: dict | None = None,
    heatmap_params: dict | None = None,
    colorbar_params: dict | None = None,
) -> PlotFunctionResult:
    knn_grid_params = dict(knn_grid_params or {})
    xlims, ylims = _resolve_lims(X, xlims, ylims)
    X, Y = _finite_xy(X, Y)
    field = knn_gradient_grid(X, Y, xlims, ylims, knn_grid_params, space, rescaler)
    magnitude = np.sqrt(field.gx**2 + field.gy**2)
    resolution = knn_grid_params.get("grid_resolution", 200)
    return _render_smooth_heatmap(
        ax,
        field.input_coords,
        magnitude.flatten(),
        input_names,
        output_name,
        rescaler,
        None,
        xlims,
        ylims,
        resolution,
        title=title,
        title_kwargs=title_kwargs,
        xtitle=xtitle,
        ytitle=ytitle,
        vtitle=vtitle,
        vlims=vlims,
        heatmap_params=heatmap_params,
        colorbar_params=colorbar_params,
        draw_colorbar=False,
    )


def gradient_field_2d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    xlims=(0, 1),
    ylims=(None, None),
    knn_grid_params: dict | None = None,
    space: Literal["raw", "latent"] = "latent",
    quiver_resolution: int = 22,
    normalize_arrows: bool = False,
    arrow_scale: float | None = None,
    arrow_width: float = 0.0025,
    quiver_props: dict | None = None,
    color_by: Literal["angle", "magnitude", "deviation_subtraction", "fixed"] = "angle",
    cmap: str = "twilight_shifted",
    fixed_color: str = "#222222",
    zero_dot_threshold: float = 0.05,
    zero_dot_size: float = 6.0,
    zero_dot_props: dict | None = None,
) -> PlotFunctionResult:
    import matplotlib as mpl

    if isinstance(ax, list | tuple):
        ax = ax[0]
    quiver_props = dict(quiver_props or {})
    xlims, ylims = _resolve_lims(X, xlims, ylims)
    X, Y = _finite_xy(X, Y)
    g = knn_gradient_grid(
        X,
        Y,
        xlims,
        ylims,
        knn_grid_params=knn_grid_params,
        space=space,
        rescaler=rescaler,
    )

    step_x = max(1, len(g.x1_lat) // quiver_resolution)
    step_y = max(1, len(g.x2_lat) // quiver_resolution)
    xs, ys = np.meshgrid(g.x1_lat[::step_x], g.x2_lat[::step_y])
    u, v = g.gx[::step_y, ::step_x], g.gy[::step_y, ::step_x]
    mag = np.hypot(u, v)
    safe = np.where(mag > 0, mag, 1.0)
    u_n, v_n = u / safe, v / safe
    u_plot, v_plot = (u_n, v_n) if normalize_arrows else (u, v)
    finite = np.isfinite(u_plot) & np.isfinite(v_plot) & (mag > 0)

    if color_by == "angle":
        c = np.arctan2(v_n, u_n)
    elif color_by == "deviation_subtraction":
        ref = np.array([-1.0, 1.0]) / np.sqrt(2.0)
        c = np.degrees(np.arccos(np.clip(u_n * ref[0] + v_n * ref[1], -1.0, 1.0)))
    elif color_by == "magnitude":
        c = mag
    else:
        c = None

    norm = None
    if c is not None and finite.any():
        norm = mpl.colors.Normalize(
            vmin=float(np.nanmin(c[finite])),
            vmax=float(np.nanmax(c[finite])),
        )

    if zero_dot_threshold > 0 and not normalize_arrows and finite.any():
        ref_mag = float(np.nanmax(mag[finite]))
        near_zero = (
            mag < zero_dot_threshold * ref_mag if ref_mag > 0 else np.zeros_like(mag, dtype=bool)
        )
        arrow_mask = finite & ~near_zero
        dot_mask = finite & near_zero
    else:
        arrow_mask = finite
        dot_mask = np.zeros_like(finite)

    q_kwargs = {
        **dict(
            angles="xy",
            scale_units="width",
            width=arrow_width,
            headwidth=4.5,
            headlength=4.5,
            headaxislength=4.0,
            pivot="middle",
            alpha=0.9,
        ),
        **quiver_props,
    }
    if arrow_scale is not None:
        q_kwargs["scale"] = arrow_scale
    elif normalize_arrows:
        q_kwargs.setdefault("scale", 28.0)

    if c is not None:
        q = ax.quiver(
            xs[arrow_mask],
            ys[arrow_mask],
            u_plot[arrow_mask],
            v_plot[arrow_mask],
            c[arrow_mask],
            cmap=cmap,
            norm=norm,
            **q_kwargs,
        )
    else:
        q = ax.quiver(
            xs[arrow_mask],
            ys[arrow_mask],
            u_plot[arrow_mask],
            v_plot[arrow_mask],
            color=fixed_color,
            **q_kwargs,
        )

    if dot_mask.any():
        dot_kwargs = {
            **dict(s=zero_dot_size, linewidths=0, alpha=0.9, marker="o", zorder=q.zorder),
            **(zero_dot_props or {}),
        }
        if c is not None and "color" not in dot_kwargs and "c" not in dot_kwargs:
            ax.scatter(
                xs[dot_mask],
                ys[dot_mask],
                c=c[dot_mask],
                cmap=cmap,
                norm=norm,
                **dot_kwargs,
            )
        else:
            dot_kwargs.setdefault("color", fixed_color)
            ax.scatter(xs[dot_mask], ys[dot_mask], **dot_kwargs)

    return PlotFunctionResult(
        rendering=q,
        metadata={
            "gx": g.gx,
            "gy": g.gy,
            "x1_lat": g.x1_lat,
            "x2_lat": g.x2_lat,
            "space": space,
        },
    )
