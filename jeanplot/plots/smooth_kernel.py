"""Smooth heatmap kernel: knn_grid + shared rendering of smooth heatmaps.

Adapted from biocomp `plotting_smooth_2d.py` and pieces of `plotting_core.py`
(build_tree, knn_stats). KNN primitives are now in `jeanplot.knn`.
"""

import threading
from typing import Literal

import numpy as np

from jeanplot.data import PlotFunctionResult
from jeanplot.knn import (
    get_gaussian_weighted_knn,
    get_knn_mean_and_variance,
    get_knn_mean_only,
    knn_density_chunked,
    make_tree,
)
from jeanplot.knn.gaussian import balance_weights_by_density, weighted_gather
from jeanplot.plots.colorbar import colorbar
from jeanplot.plots.heatmap import heatmap, make_xy_grid
from jeanplot.plots.ticks import setup_transformed_axis


_TREE_CACHE: dict = {}
_TREE_CACHE_MAX = 8
_TREE_CACHE_LOCK = threading.Lock()


def array_content_key(x):
    if not isinstance(x, np.ndarray):
        return None
    a = x if x.flags["C_CONTIGUOUS"] else np.ascontiguousarray(x)
    try:
        h = hash(bytes(memoryview(a).cast("B")))
    except Exception:
        return None
    return (h, x.shape, x.dtype.str)


def build_tree(x):
    key = array_content_key(x)
    if key is not None:
        with _TREE_CACHE_LOCK:
            cached = _TREE_CACHE.get(key)
        if cached is not None:
            return cached

    mask = np.all(np.isfinite(x), axis=1) if x.ndim > 1 else np.isfinite(x)
    x_clean = x if mask.all() else x[mask]
    if len(x_clean) == 0:
        raise ValueError("No finite data points available for building KD-tree")

    tree = make_tree(x_clean)
    if key is not None:
        with _TREE_CACHE_LOCK:
            if len(_TREE_CACHE) >= _TREE_CACHE_MAX:
                _TREE_CACHE.pop(next(iter(_TREE_CACHE)))
            _TREE_CACHE[key] = tree
    return tree


def knn_stats(
    xquery,
    y=None,
    tree=None,
    iw=None,
    k=500,
    min_points=20,
    stats: str | list[str] = "iw",
    weight_by_densities: bool = False,
    kdensity: int = 50,
    density_power: float = 0.0,
    density_floor_q: float | None = 0.01,
    density_cap_q: float | None = 0.99,
    rebalance_centroids: float = 0.0,
    rebalance_centroids_mode: Literal["smooth", "hard"] = "hard",
    rebalance_values: float = 0.0,
    rebalance_values_mode: Literal["smooth", "hard"] = "smooth",
    use_jax=None,
    **kw,
):
    # use_jax is accepted but ignored — jeanplot's KNN backends are configured
    # via env vars (BIOCOMP_KNN_BACKEND / JEANPLOT_KNN_BACKEND). Kept for
    # back-compat with biocomp callers that pass `use_jax=False` explicitly.
    _ = use_jax
    if isinstance(stats, str):
        stats = [stats]

    if tree is None and iw is None:
        tree = build_tree(xquery)

    densities = None
    if weight_by_densities or rebalance_centroids > 0.0 or rebalance_values > 0.0:
        from jeanplot.knn.density import per_point_knn_density

        X_ref = kw.get("X_ref", None)
        densities = per_point_knn_density(tree=tree, X_ref=X_ref, kdensity=kdensity)

    if weight_by_densities:
        if density_floor_q is not None:
            kw["density_floor"] = float(np.quantile(densities, density_floor_q))
        if density_cap_q is not None:
            kw["density_cap"] = float(np.quantile(densities, density_cap_q))
        kw["densities"] = densities
        kw["density_power"] = density_power

    do_rebalance_values = rebalance_values > 0.0 and densities is not None
    if not iw and stats == ["mean"] and not do_rebalance_values:
        return get_knn_mean_only(xquery, y, tree=tree, k=k, min_points=min_points, **kw)

    iw = iw or get_gaussian_weighted_knn(xquery, tree, k=k, min_points=min_points, **kw)
    indices, weights = iw

    def _rebalanced(strength: float, mode: Literal["smooth", "hard"]):
        cap = float(np.quantile(densities, 1.0 - strength))
        return balance_weights_by_density(indices, weights, densities, cap=cap, mode=mode)

    centroid_w = weights
    if rebalance_centroids > 0.0 and densities is not None:
        centroid_w = _rebalanced(rebalance_centroids, rebalance_centroids_mode)
    if do_rebalance_values:
        iw = (indices, _rebalanced(rebalance_values, rebalance_values_mode))

    need_var = {"variance", "std"} & set(stats)
    need_mv = {"mean", "variance", "std"} & set(stats)
    if need_mv and not need_var and y is not None and y.ndim == 2:
        mean, var = weighted_gather(iw[0], iw[1], y), None
    elif need_mv:
        mean, var = get_knn_mean_and_variance(
            xquery,
            y,
            iw=iw,
            k=k,
            min_points=min_points,
            compute_variance=bool(need_var),
            **kw,
        )
    else:
        mean, var = None, None

    def _centroid():
        pts = np.asarray(getattr(tree, "data", None))
        if pts is None:
            raise ValueError("centroid stats require a tree exposing `.data`")
        return weighted_gather(indices, centroid_w, pts)

    def calc(s):
        if s == "iw":
            return iw
        if s == "density":
            weights = iw[1]
            valid = np.isfinite(weights[:, 0])
            out = weights.sum(axis=1)
            if not valid.all():
                out = out.copy()
                out[~valid] = 0.0
            return out
        if s == "quantile":
            from jeanplot.knn.jax_kernel import get_knn_quantile

            return get_knn_quantile(xquery, y, iw=iw, k=k, min_points=min_points, **kw)
        if s == "mean":
            return mean
        if s == "variance":
            return var
        if s == "std":
            return np.sqrt(var)
        if s == "centroid":
            return _centroid()
        if s == "centroid_offset":
            return np.linalg.norm(_centroid() - np.asarray(xquery), axis=1)
        raise ValueError(f"Unknown stat: {s}")

    res = tuple([calc(s) for s in stats])
    return res[0] if len(res) == 1 else res


_KNN_GRID_CACHE: dict = {}
_KNN_GRID_CACHE_MAX = 8


def _knn_grid_cache_key(
    x,
    y,
    xlims,
    ylims,
    zslice,
    is_density_plot,
    grid_resolution,
    knn_stats_params,
    max_centroid_offset_frac=0.0,
    query_mode="grid",
    query_seed=0,
):
    kx = array_content_key(x)
    ky = array_content_key(y)
    if kx is None or ky is None:
        return None
    kz = array_content_key(np.asarray(zslice)) if zslice is not None else None
    return (
        kx,
        ky,
        tuple(xlims) if xlims is not None else None,
        tuple(ylims) if ylims is not None else None,
        kz,
        bool(is_density_plot),
        int(grid_resolution),
        tuple(sorted((knn_stats_params or {}).items())),
        float(max_centroid_offset_frac),
        str(query_mode),
        int(query_seed),
    )


def knn_grid(
    x,
    y,
    xlims,
    ylims,
    zslice=None,
    is_density_plot=False,
    grid_resolution=200,
    knn_stats_params=None,
    max_centroid_offset_frac: float = 0.0,
    query_mode: Literal["grid", "uniform"] = "grid",
    query_seed: int = 0,
):
    """KNN-smoothed (X, Y) at query points covering xlims × ylims."""
    if knn_stats_params is None:
        knn_stats_params = {}

    cache_key = _knn_grid_cache_key(
        x,
        y,
        xlims,
        ylims,
        zslice,
        is_density_plot,
        grid_resolution,
        knn_stats_params,
        max_centroid_offset_frac,
        query_mode,
        query_seed,
    )
    if cache_key is not None:
        cached = _KNN_GRID_CACHE.get(cache_key)
        if cached is not None:
            return cached

    mask = np.all(np.isfinite(x), axis=1) if x.ndim > 1 else np.isfinite(x)
    mask = mask & (np.all(np.isfinite(y), axis=1) if y.ndim > 1 else np.isfinite(y))

    if mask.all():
        x_clean, y_clean = x, y
    else:
        x_clean = x[mask]
        y_clean = y[mask]

    xmin, xmax = xlims
    ymin, ymax = ylims or xlims
    if query_mode == "uniform":
        rng = np.random.default_rng(int(query_seed))
        n_query = int(grid_resolution) ** 2
        xy = np.column_stack(
            [
                rng.uniform(xmin, xmax, size=n_query),
                rng.uniform(ymin, ymax, size=n_query),
            ]
        ).astype(np.float64)
    else:
        xy = make_xy_grid(
            xmin, xmax, xres=grid_resolution, ymin=ymin, ymax=ymax, yres=grid_resolution
        )

    if len(x_clean) == 0:
        return xy, np.full(xy.shape[0], np.nan)

    if x_clean.shape[1] > 2:
        assert zslice is not None
        if zslice.shape != (x_clean.shape[1] - 2,):
            raise ValueError(f"zslice.shape = {zslice.shape} != {x_clean.shape[1] - 2}")
        xquery = np.hstack([xy, [zslice] * xy.shape[0]])
    else:
        xquery = xy

    tree = build_tree(x_clean)
    primary = "density" if is_density_plot else "mean"
    requested = [primary, "centroid_offset"] if max_centroid_offset_frac > 0.0 else primary
    result = knn_stats(xquery, y_clean, tree=tree, stats=requested, **knn_stats_params)
    if max_centroid_offset_frac > 0.0:
        output_values, offset = result
        output_values = output_values.squeeze()
        radius = float(knn_stats_params.get("radius", 0.1))
        sigma_in_radius = float(knn_stats_params.get("sigma_in_radius", 3.0))
        boundary = np.asarray(offset) > max_centroid_offset_frac * (radius / sigma_in_radius)
        output_values = np.where(boundary, np.nan, output_values)
    else:
        output_values = result.squeeze()

    if output_values.shape != (xy.shape[0],):
        raise ValueError(f"output_values.shape = {output_values.shape} != {xy.shape[0]}")

    if cache_key is not None:
        if len(_KNN_GRID_CACHE) >= _KNN_GRID_CACHE_MAX:
            _KNN_GRID_CACHE.pop(next(iter(_KNN_GRID_CACHE)))
        _KNN_GRID_CACHE[cache_key] = (xy, output_values)
    return xy, output_values


def _resolve_lims(X, xlims, ylims):
    dx = [X[:, 0].min(), X[:, 0].max()]
    dy = [X[:, 1].min(), X[:, 1].max()]
    return (
        [dx[0] if xlims[0] is None else xlims[0], dx[1] if xlims[1] is None else xlims[1]],
        [dy[0] if ylims[0] is None else ylims[0], dy[1] if ylims[1] is None else ylims[1]],
    )


def _finite_xy(X, Y):
    m = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    return (X, Y) if m.all() else (X[m], Y[m])


def _resolve_vlims(values, vlims, vlim_quantiles, vlim_min_floor, vlim_min_range):
    vlims = list(vlims)
    if vlim_quantiles is not None:
        finite = np.asarray(values)
        finite = finite[np.isfinite(finite)]
        q_lo, q_hi = vlim_quantiles
        if vlims[0] is None and q_lo is not None and finite.size:
            vlims[0] = float(np.quantile(finite, q_lo))
        if vlims[1] is None and q_hi is not None and finite.size:
            vlims[1] = float(np.quantile(finite, q_hi))
    if vlim_min_floor is not None and vlims[0] is not None:
        vlims[0] = float(min(vlims[0], vlim_min_floor))
    if vlim_min_range is not None and vlims[0] is not None and vlims[1] is not None:
        if (vlims[1] - vlims[0]) < vlim_min_range:
            vlims[1] = float(vlims[0] + vlim_min_range)
    return tuple(vlims)


def _render_smooth_heatmap(
    ax,
    input_coords,
    output_values,
    input_names,
    output_name,
    axis_rescaler,
    value_rescaler,
    xlims,
    ylims,
    resolution,
    *,
    title=None,
    title_kwargs=None,
    xtitle=None,
    ytitle=None,
    vtitle=None,
    vlims=(None, None),
    vlim_quantiles=(0.01, 0.99),
    vlim_min_floor=None,
    vlim_min_range=None,
    draw_xlabel=True,
    draw_ylabel=True,
    xaxis_labelpad=None,
    yaxis_labelpad=None,
    draw_colorbar=True,
    draw_colorbar_label=True,
    colorbar_params=None,
    heatmap_params=None,
    setup_transformed_axis_params=None,
):
    heatmap_params = heatmap_params or {}
    colorbar_params = colorbar_params or {}
    setup_transformed_axis_params = setup_transformed_axis_params or {}

    vlims = _resolve_vlims(output_values, vlims, vlim_quantiles, vlim_min_floor, vlim_min_range)
    im, cntrs = heatmap(ax, input_coords, output_values, **{**heatmap_params, "vlims": vlims})

    xlabel = (input_names[0] if input_names else None) if xtitle is None else xtitle
    ylabel = (input_names[1] if len(input_names) > 1 else None) if ytitle is None else ytitle
    if draw_xlabel and xlabel:
        xkw = {"labelpad": xaxis_labelpad} if xaxis_labelpad is not None else {}
        ax.set_xlabel(xlabel, **xkw)
    if draw_ylabel and ylabel:
        ykw = {"labelpad": yaxis_labelpad} if yaxis_labelpad is not None else {}
        ax.set_ylabel(ylabel, **ykw)
    if title is not None:
        ax.set_title(title, **(title_kwargs or {}))

    if axis_rescaler is not None:
        setup_transformed_axis(
            ax,
            xaxis_lims=xlims,
            yaxis_lims=ylims,
            rescaler=axis_rescaler,
            **setup_transformed_axis_params,
        )
    else:
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

    if draw_colorbar and value_rescaler is not None:
        vlabel = (output_name if vtitle is None else vtitle) if draw_colorbar_label else None
        colorbar(ax, im, value_rescaler, vlims, **{**colorbar_params, "label": vlabel})

    return PlotFunctionResult(rendering=(im, cntrs), metadata={"vlims": vlims}, mappable=im)


def weighted_kde_1d(
    values,
    weights=None,
    *,
    kde_points: int = 80,
    pad_frac: float = 0.15,
    bw_method=None,
):
    """Return weighted 1D KDE as ``(grid, density)`` or ``None`` when ill-posed."""
    from scipy.stats import gaussian_kde

    v = np.asarray(values).ravel()
    if weights is None:
        w = np.ones_like(v, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != v.shape:
            raise ValueError(f"weights shape {w.shape} must match values shape {v.shape}")

    finite = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if finite.sum() < 3:
        return None
    v = v[finite]
    w = w[finite]
    if np.unique(v).size < 2:
        return None

    wsum = float(w.sum())
    if not np.isfinite(wsum) or wsum <= 0:
        return None
    w = w / wsum

    try:
        kde = gaussian_kde(v, weights=w, bw_method=bw_method)
    except (np.linalg.LinAlgError, ValueError):
        return None

    v_lo, v_hi = float(v.min()), float(v.max())
    span = max(v_hi - v_lo, 1e-9)
    pad = span * float(pad_frac)
    grid = np.linspace(v_lo - pad, v_hi + pad, int(kde_points))
    density = np.asarray(kde(grid), dtype=float)
    return grid, density


__all__ = [
    "build_tree",
    "knn_stats",
    "knn_density_chunked",
    "knn_grid",
    "weighted_kde_1d",
    "_resolve_lims",
    "_finite_xy",
    "_resolve_vlims",
    "_render_smooth_heatmap",
]
