"""Smooth heatmap kernel: splat-backed grid/stat smoothers + shared rendering.

`smooth_grid` / `smooth_grid_gradient` / `smooth_stats` delegate to
`jeanplot.splat`; `build_tree` survives only as a data holder (`tree.data`) for
the `smooth_stats(tree=...)` call convention.
"""

import threading
from typing import Literal

import numpy as np

from jeanplot.data import PlotFunctionResult
from jeanplot.knn import array_content_key, make_tree
from jeanplot.plots.colorbar import colorbar
from jeanplot.plots.heatmap import heatmap, make_xy_grid
from jeanplot.plots.ticks import setup_transformed_axis
from jeanplot.splat import SplatField


_TREE_CACHE: dict = {}
_TREE_CACHE_MAX = 32
_TREE_CACHE_LOCK = threading.Lock()


def clear_knn_caches():
    """Drop cached trees and smooth grids (release between batch figures)."""
    from jeanplot.splat.core import clear_density_cache, clear_fit_cache, clear_weights_cache

    with _TREE_CACHE_LOCK:
        _TREE_CACHE.clear()
    _SMOOTH_GRID_CACHE.clear()
    clear_density_cache()
    clear_weights_cache()
    clear_fit_cache()


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


_SMOOTH_GRID_CACHE: dict = {}
_SMOOTH_GRID_CACHE_MAX = 128


def _smooth_grid_cache_key(
    x,
    y,
    xlims,
    ylims,
    zslice,
    kind,
    grid_resolution,
    sp,
    max_centroid_offset_frac,
    query_mode,
    query_seed,
):
    kx = array_content_key(x)
    ky = array_content_key(y) if y is not None else None
    if kx is None or (y is not None and ky is None):
        return None
    kz = array_content_key(np.asarray(zslice)) if zslice is not None else None
    return (
        kx,
        ky,
        tuple(xlims) if xlims is not None else None,
        tuple(ylims) if ylims is not None else None,
        kz,
        kind,
        int(grid_resolution),
        tuple(sorted((sp or {}).items())),
        float(max_centroid_offset_frac),
        str(query_mode),
        int(query_seed),
    )


def _translate_smooth_params(sp):
    """Map the legacy `knn_stats_params` / `smooth_params` vocabulary onto
    `SplatField.fit` kwargs. `k`/`kdensity` (gather-only) are ignored."""
    sp = dict(sp or {})
    return dict(
        radius=float(sp.get("radius", 0.1)),
        sigma_in_radius=float(sp.get("sigma_in_radius", 3.0)),
        min_points=int(sp.get("min_points", 20)),
        rebalance_values=float(sp.get("rebalance_values", 0.0)),
        rebalance_values_mode=sp.get("rebalance_values_mode", "smooth"),
        rebalance_centroids=float(sp.get("rebalance_centroids", 0.0)),
        rebalance_centroids_mode=sp.get("rebalance_centroids_mode", "hard"),
    )


def _resolve_bounds(xlims, ylims):
    xmin, xmax = xlims
    ymin, ymax = ylims if ylims and ylims[0] is not None else xlims
    return (xmin, xmax), (ymin, ymax)


def _cache_store(out, key):
    if key is not None:
        if len(_SMOOTH_GRID_CACHE) >= _SMOOTH_GRID_CACHE_MAX:
            _SMOOTH_GRID_CACHE.pop(next(iter(_SMOOTH_GRID_CACHE)))
        _SMOOTH_GRID_CACHE[key] = out
    return out


def _smooth_query(field, stat, query_mode, xb, yb, grid_resolution, query_seed):
    if query_mode == "uniform":
        rng = np.random.default_rng(int(query_seed))
        n = int(grid_resolution) ** 2
        xy = np.column_stack([rng.uniform(xb[0], xb[1], n), rng.uniform(yb[0], yb[1], n)]).astype(
            np.float64
        )
        return xy, field.at(xy, stat)
    xy = make_xy_grid(
        xb[0], xb[1], xres=grid_resolution, ymin=yb[0], ymax=yb[1], yres=grid_resolution
    )
    return xy, field.flat_xy(stat)


def smooth_grid(
    x,
    y,
    xlims,
    ylims,
    zslice=None,
    is_density_plot=False,
    grid_resolution=200,
    smooth_params=None,
    knn_stats_params=None,
    max_centroid_offset_frac: float = 0.0,
    query_mode: Literal["grid", "uniform"] = "grid",
    query_seed: int = 0,
):
    """Splat-smoothed (X, Y) over xlims × ylims. SSOT grid smoother."""
    sp = smooth_params if smooth_params is not None else knn_stats_params
    primary = "density" if is_density_plot else "mean"
    key = _smooth_grid_cache_key(
        x,
        y,
        xlims,
        ylims,
        zslice,
        primary,
        grid_resolution,
        sp,
        max_centroid_offset_frac,
        query_mode,
        query_seed,
    )
    if key is not None and key in _SMOOTH_GRID_CACHE:
        return _SMOOTH_GRID_CACHE[key]

    xb, yb = _resolve_bounds(xlims, ylims)
    tp = _translate_smooth_params(sp)
    stats = [primary] + (["centroid_offset"] if max_centroid_offset_frac > 0.0 else [])
    field = SplatField.fit(
        np.asarray(x),
        None if is_density_plot else y,
        bounds=[xb, yb],
        resolution=grid_resolution,
        zslice=np.asarray(zslice) if zslice is not None else None,
        stats=stats,
        **tp,
    )
    xy, vals = _smooth_query(field, primary, query_mode, xb, yb, grid_resolution, query_seed)
    vals = np.asarray(vals)
    if vals.ndim == 2 and vals.shape[1] == 1:
        vals = vals[:, 0]
    if max_centroid_offset_frac > 0.0:
        _, offset = _smooth_query(
            field, "centroid_offset", query_mode, xb, yb, grid_resolution, query_seed
        )
        thresh = max_centroid_offset_frac * (tp["radius"] / tp["sigma_in_radius"])
        vals = np.where(np.asarray(offset) > thresh, np.nan, vals)
    return _cache_store((xy, vals), key)


def smooth_grid_gradient(
    x,
    y,
    xlims,
    ylims,
    zslice=None,
    grid_resolution=200,
    smooth_params=None,
    knn_stats_params=None,
    max_centroid_offset_frac: float = 0.0,
    query_mode: Literal["grid", "uniform"] = "grid",
    query_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted local-linear-regression gradient at grid points: (xy, grad[:, :2])."""
    sp = smooth_params if smooth_params is not None else knn_stats_params
    key = _smooth_grid_cache_key(
        x,
        y,
        xlims,
        ylims,
        zslice,
        "grad",
        grid_resolution,
        sp,
        max_centroid_offset_frac,
        query_mode,
        query_seed,
    )
    if key is not None and key in _SMOOTH_GRID_CACHE:
        return _SMOOTH_GRID_CACHE[key]

    xb, yb = _resolve_bounds(xlims, ylims)
    tp = _translate_smooth_params(sp)
    stats = ["grad"] + (["centroid_offset"] if max_centroid_offset_frac > 0.0 else [])
    field = SplatField.fit(
        np.asarray(x),
        y,
        bounds=[xb, yb],
        resolution=grid_resolution,
        zslice=np.asarray(zslice) if zslice is not None else None,
        stats=stats,
        **tp,
    )
    xy, grad = _smooth_query(field, "grad", query_mode, xb, yb, grid_resolution, query_seed)
    grad = np.asarray(grad)
    if max_centroid_offset_frac > 0.0:
        _, offset = _smooth_query(
            field, "centroid_offset", query_mode, xb, yb, grid_resolution, query_seed
        )
        thresh = max_centroid_offset_frac * (tp["radius"] / tp["sigma_in_radius"])
        grad = np.where((np.asarray(offset) > thresh)[:, None], np.nan, grad)
    return _cache_store((xy, grad), key)


_SPLAT_STATS = frozenset(
    {"mean", "variance", "std", "density", "centroid", "centroid_offset", "grad"}
)


def smooth_stats(
    xquery,
    y=None,
    tree=None,
    data=None,
    stats: str | list[str] = "mean",
    smooth_params=None,
    resolution=None,
    **params,
):
    """Splat stats at arbitrary query points. The query's varying dims become the
    lattice; constant (slice) dims are gaussian-banded. SSOT for the bare
    per-query `knn_stats` paths (mean/std/density/centroid/grad)."""
    single = isinstance(stats, str)
    stats_l = [stats] if single else list(stats)
    assert all(s in _SPLAT_STATS for s in stats_l), f"smooth_stats: non-splat stat {stats_l}"
    X = np.asarray(data if data is not None else tree.data, dtype=np.float64)
    q = np.asarray(xquery, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    d = X.shape[1]
    tp = _translate_smooth_params(smooth_params if smooth_params is not None else params)

    spans = q.max(0) - q.min(0)
    free = [i for i in range(d) if spans[i] > 1e-9] or [0]
    held = [i for i in range(d) if i not in free]
    order = free + held
    r = tp["radius"]
    bounds = []
    for i in free:  # pad a degenerate (single-point) free dim so the lattice is well-posed
        lo, hi = float(q[:, i].min()), float(q[:, i].max())
        bounds.append((lo - r, hi + r) if hi - lo < 1e-9 else (lo, hi))
    zslice = q[0, held] if held else None
    if resolution is None:
        span = max(hi - lo for lo, hi in bounds)
        cell = tp["radius"] / tp["sigma_in_radius"] / 2.0
        resolution = int(np.clip(np.ceil(span / max(cell, 1e-9)), 32, 256))

    field = SplatField.fit(
        X[:, order], y, bounds=bounds, resolution=resolution, zslice=zslice, stats=stats_l, **tp
    )
    out = tuple(field.at(q[:, free], s) for s in stats_l)
    return out[0] if single else out


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
    "smooth_stats",
    "smooth_grid",
    "smooth_grid_gradient",
    "weighted_kde_1d",
    "_resolve_lims",
    "_finite_xy",
    "_resolve_vlims",
    "_render_smooth_heatmap",
]
