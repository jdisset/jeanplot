"""Smooth heatmap kernel: knn_grid + shared rendering of smooth heatmaps.

Adapted from biocomp `plotting_smooth_2d.py` and pieces of `plotting_core.py`
(build_tree, knn_stats). KNN primitives are now in `jeanplot.knn`.
"""

import os
import threading
from typing import Literal

import numpy as np

from jeanplot.data import PlotFunctionResult
from jeanplot.knn import (
    array_content_key,
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
from jeanplot.splat import SplatField


_TREE_CACHE: dict = {}
_TREE_CACHE_MAX = 32
_TREE_CACHE_LOCK = threading.Lock()


# Shared, byte-budgeted LRU cache of the y-independent KNN prep. Entries are
# large (n_query x k) arrays, so the cap is in bytes, not count: one k=7000
# grid prep is ~10 GB. Sized to retain the working set across panels that share
# a query (e.g. ground-truth vs prediction over the same grid) when those panels
# render adjacently. Override via JEANPLOT_KNN_PREP_CACHE_GB.
_PREP_CACHE: dict = {}
_PREP_CACHE_BUDGET = int(float(os.environ.get("JEANPLOT_KNN_PREP_CACHE_GB", "16")) * 1e9)
_PREP_CACHE_BYTES = 0
_PREP_CACHE_LOCK = threading.Lock()


def _prep_nbytes(prep) -> int:
    return sum(a.nbytes for a in prep if hasattr(a, "nbytes"))


# quantile(densities, 0) is the global minimum, set by the single most isolated
# point; capping rebalancing there lets one sparse outlier dominate its whole
# neighbourhood. Floor the quantile prob so strength=1.0 caps at a low percentile
# instead of the absolute minimum.
_REBALANCE_Q_FLOOR = 0.02


def _rebalance_weights(indices, weights, densities, rc, rc_mode, rv, rv_mode):
    """Density-rebalanced (value_weights, centroid_weights) — SSOT for both the
    fresh-query and caller-supplied-iw paths."""

    def reb(strength, mode):
        cap = float(np.quantile(densities, max(1.0 - strength, _REBALANCE_Q_FLOOR)))
        return balance_weights_by_density(indices, weights, densities, cap=cap, mode=mode)

    has = densities is not None
    centroid_w = reb(rc, rc_mode) if has and rc > 0.0 else weights
    iw_weights = reb(rv, rv_mode) if has and rv > 0.0 else weights
    return iw_weights, centroid_w


def _cached_iw_prep(
    xquery,
    tree,
    k,
    min_points,
    kdensity,
    query_kw,
    densities,
    rebalance_centroids,
    rebalance_centroids_mode,
    rebalance_values,
    rebalance_values_mode,
):
    """(indices, value_weights, centroid_weights) for xquery.

    The neighbour query + density rebalancing depend only on (tree, query points,
    params), never on y or the requested stat — so a value heatmap, gradient map,
    and quiver over the same grid share one query instead of three.
    """
    kw = tuple(
        (kk, array_content_key(v) if isinstance(v, np.ndarray) else v)
        for kk, v in sorted(query_kw.items())
    )
    xk = array_content_key(xquery)
    key = (
        None
        if xk is None or any(isinstance(query_kw[kk], np.ndarray) and v is None for kk, v in kw)
        else (
            id(tree),
            xk,
            int(k),
            int(min_points),
            int(kdensity),
            kw,
            rebalance_centroids,
            rebalance_centroids_mode,
            rebalance_values,
            rebalance_values_mode,
        )
    )
    if key is not None:
        with _PREP_CACHE_LOCK:
            hit = _PREP_CACHE.pop(key, None)
            if hit is not None:
                _PREP_CACHE[key] = hit  # reinsert: most-recently-used
                return hit

    indices, weights = get_gaussian_weighted_knn(
        xquery, tree, k=k, min_points=min_points, **query_kw
    )
    iw_weights, centroid_w = _rebalance_weights(
        indices,
        weights,
        densities,
        rebalance_centroids,
        rebalance_centroids_mode,
        rebalance_values,
        rebalance_values_mode,
    )
    prep = (indices, iw_weights, centroid_w)

    if key is not None:
        global _PREP_CACHE_BYTES
        nbytes = _prep_nbytes(prep)
        with _PREP_CACHE_LOCK:
            while _PREP_CACHE and _PREP_CACHE_BYTES + nbytes > _PREP_CACHE_BUDGET:
                evicted = _PREP_CACHE.pop(next(iter(_PREP_CACHE)))
                _PREP_CACHE_BYTES -= _prep_nbytes(evicted)
            if nbytes <= _PREP_CACHE_BUDGET:
                _PREP_CACHE[key] = prep
                _PREP_CACHE_BYTES += nbytes
    return prep


def clear_knn_caches():
    """Drop cached trees, grids, and KNN prep arrays (release the large prep
    arrays between unrelated batch figures)."""
    global _PREP_CACHE_BYTES
    from jeanplot.knn.gaussian import clear_mean_ggw_cache

    with _TREE_CACHE_LOCK:
        _TREE_CACHE.clear()
    with _PREP_CACHE_LOCK:
        _PREP_CACHE.clear()
        _PREP_CACHE_BYTES = 0
    _SMOOTH_GRID_CACHE.clear()
    clear_mean_ggw_cache()


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

    if iw is None:
        indices, iw_weights, centroid_w = _cached_iw_prep(
            xquery,
            tree,
            k,
            min_points,
            kdensity,
            kw,
            densities,
            rebalance_centroids,
            rebalance_centroids_mode,
            rebalance_values,
            rebalance_values_mode,
        )
    else:
        indices, weights = iw
        iw_weights, centroid_w = _rebalance_weights(
            indices,
            weights,
            densities,
            rebalance_centroids,
            rebalance_centroids_mode,
            rebalance_values,
            rebalance_values_mode,
        )
    iw = (indices, iw_weights)

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
        if s == "grad":
            # weighted local-linear regression slope: Cov_w(x,x)^-1 · Cov_w(x,y).
            # Unbiased for the true gradient regardless of sample density (unlike
            # Cov_w(x,y)/σ², which assumes Var_w(x)=σ² and collapses in dense regions).
            pts = np.asarray(getattr(tree, "data", None))
            if pts is None:
                raise ValueError("grad stat requires a tree exposing `.data`")
            pts = pts[:, :2]
            yy = np.asarray(y)
            yy = yy if yy.ndim == 2 else yy[:, None]
            ix, w = iw
            mx = weighted_gather(ix, w, pts)
            cxy = weighted_gather(ix, w, pts * yy) - mx * weighted_gather(ix, w, yy)
            m2 = weighted_gather(
                ix, w, np.column_stack([pts[:, 0] ** 2, pts[:, 0] * pts[:, 1], pts[:, 1] ** 2])
            )
            cxx = np.empty((len(mx), 2, 2))
            cxx[:, 0, 0] = m2[:, 0] - mx[:, 0] ** 2
            cxx[:, 1, 1] = m2[:, 2] - mx[:, 1] ** 2
            cxx[:, 0, 1] = cxx[:, 1, 0] = m2[:, 1] - mx[:, 0] * mx[:, 1]
            ridge = 1e-9 + 1e-6 * (cxx[:, 0, 0] + cxx[:, 1, 1])
            cxx[:, 0, 0] += ridge
            cxx[:, 1, 1] += ridge
            grad = np.full_like(cxy, np.nan)
            ok = np.isfinite(cxy).all(1) & np.isfinite(cxx).all((1, 2))
            grad[ok] = np.linalg.solve(cxx[ok], cxy[ok][..., None])[..., 0]
            return grad
        raise ValueError(f"Unknown stat: {s}")

    res = tuple([calc(s) for s in stats])
    return res[0] if len(res) == 1 else res


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


def knn_grid(*args, **kwargs):
    """Deprecated alias for `smooth_grid` (knn_stats_params kwarg still accepted)."""
    return smooth_grid(*args, **kwargs)


def knn_grid_gradient(*args, **kwargs):
    """Deprecated alias for `smooth_grid_gradient`."""
    return smooth_grid_gradient(*args, **kwargs)


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
    "smooth_grid",
    "smooth_grid_gradient",
    "knn_grid",
    "knn_grid_gradient",
    "weighted_kde_1d",
    "_resolve_lims",
    "_finite_xy",
    "_resolve_vlims",
    "_render_smooth_heatmap",
]
