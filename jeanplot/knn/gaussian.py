import hashlib
import os
import threading
from typing import Literal

import numpy as np
from jeanplot.knn.tree import _query, KNN_WORKERS, KNN_MEAN_CHUNK_SIZE


# Byte-budgeted cache of the y-independent (indices, weights) from the chunked
# mean path. Lets ground-truth and prediction panels over the same grid share
# the neighbour query (they differ only in y). Mutation-safe: callers normalise
# weights in place, so hits hand back a fresh copy. Override with
# JEANPLOT_KNN_MEAN_CACHE_GB.
# The cached value pins the `tree` alongside its (indices, weights): the key
# uses id(tree), and ephemeral trees get GC'd and their id reused, so without
# the pin a recycled id collides with a live entry and hands back indices for a
# different, differently-sized training set (out-of-bounds gather -> garbage).
_MEAN_GGW_CACHE: dict = {}
_MEAN_GGW_BYTES = 0
_MEAN_GGW_BUDGET = int(float(os.environ.get("JEANPLOT_KNN_MEAN_CACHE_GB", "8")) * 1e9)
_MEAN_GGW_LOCK = threading.Lock()


def clear_mean_ggw_cache():
    global _MEAN_GGW_BYTES
    with _MEAN_GGW_LOCK:
        _MEAN_GGW_CACHE.clear()
        _MEAN_GGW_BYTES = 0


def _ggw_param_key(kw):
    def v_key(v):
        if isinstance(v, np.ndarray):
            return hashlib.blake2b(np.ascontiguousarray(v).view(np.uint8), digest_size=8).digest()
        return v

    return tuple((k, v_key(v)) for k, v in sorted(kw.items()))


def _ggw_unnormed_cached(x, tree, kw):
    """`get_gaussian_weighted_knn(..., normed_w=False)` with content caching.

    Returns a copy of the weights (mutated downstream by the mean normalise);
    indices are read-only downstream and shared."""
    xb = np.ascontiguousarray(x)
    key = (
        id(tree),
        hashlib.blake2b(xb.view(np.uint8), digest_size=16).digest(),
        _ggw_param_key(kw),
    )
    with _MEAN_GGW_LOCK:
        hit = _MEAN_GGW_CACHE.pop(key, None)
        if hit is not None:
            _MEAN_GGW_CACHE[key] = hit  # MRU
    if hit is not None:
        return hit[0], hit[1].copy()

    indices, weights = get_gaussian_weighted_knn(x, tree=tree, normed_w=False, **kw)
    nbytes = indices.nbytes + weights.nbytes
    if nbytes <= _MEAN_GGW_BUDGET:
        global _MEAN_GGW_BYTES
        with _MEAN_GGW_LOCK:
            while _MEAN_GGW_CACHE and _MEAN_GGW_BYTES + nbytes > _MEAN_GGW_BUDGET:
                ek = next(iter(_MEAN_GGW_CACHE))
                ei, ew, _ = _MEAN_GGW_CACHE.pop(ek)
                _MEAN_GGW_BYTES -= ei.nbytes + ew.nbytes
            _MEAN_GGW_CACHE[key] = (indices, weights, tree)
            _MEAN_GGW_BYTES += nbytes
        return indices, weights.copy()
    return indices, weights


try:
    import numba as _nb

    @_nb.njit(cache=True, parallel=False, fastmath=True)
    def _kernel_mean_numba(indices, weights, y, out):
        n_grid, k = indices.shape
        n_outs = y.shape[1]
        for i in range(n_grid):
            row_sum = 0.0
            valid = True
            for j in range(k):
                w = weights[i, j]
                if not np.isfinite(w):
                    valid = False
                    break
                row_sum += w
            if not valid or row_sum <= 0.0:
                for o in range(n_outs):
                    out[i, o] = np.nan
                continue
            inv = 1.0 / row_sum
            for o in range(n_outs):
                acc = 0.0
                for j in range(k):
                    acc += weights[i, j] * inv * y[indices[i, j], o]
                out[i, o] = acc

    @_nb.njit(cache=True, parallel=True, fastmath=True)
    def _kernel_weighted_gather_numba(indices, weights, source, out):
        """Weighted gather: out[i, d] = sum_j weights[i, j] * source[indices[i, j], d].

        Assumes weights are pre-normalized (rows sum to 1, or NaN for invalid
        rows). Skips j with zero weight to tolerate sentinel indices."""
        n_grid, k = indices.shape
        d_out = source.shape[1]
        for i in _nb.prange(n_grid):
            if not np.isfinite(weights[i, 0]):
                for dd in range(d_out):
                    out[i, dd] = np.nan
                continue
            for dd in range(d_out):
                acc = 0.0
                for j in range(k):
                    w = weights[i, j]
                    if w != 0.0:
                        acc += w * source[indices[i, j], dd]
                out[i, dd] = acc

    @_nb.njit(cache=True, parallel=True, fastmath=True)
    def _kernel_balance_weights_numba(indices, weights, densities, cap, hard, eps, out):
        """Fused density-cap + row-renormalise, no (n,k) temporaries; NaN rows preserved."""
        n, k = indices.shape
        for i in _nb.prange(n):
            s = 0.0
            for j in range(k):
                dn = densities[indices[i, j]]
                if hard:
                    inv = cap / (dn if dn > eps else eps)
                    sc = inv if inv < 1.0 else 1.0
                else:
                    sc = cap / (dn + cap + eps)
                r = weights[i, j] * sc
                out[i, j] = r
                s += r
            if s > 0.0:
                inv = 1.0 / s
                for j in range(k):
                    out[i, j] *= inv
            else:
                for j in range(k):
                    out[i, j] = np.nan

    @_nb.njit(cache=True, parallel=True, fastmath=True)
    def _kernel_gaussian_normed_numba(distances, indices, sigma, max_dist, min_points, out):
        """Row-normalized Gaussian weights in one fused pass. Entries beyond
        max_dist get zero weight (and sentinel index clamped to 0); rows with
        fewer than min_points valid neighbours get NaN. Uses an explicit
        distance comparison instead of np.isfinite so detection survives
        numba's fastmath optimizer (fastmath assumes no NaN/inf)."""
        n, k = distances.shape
        inv_sigma = 1.0 / sigma
        for i in _nb.prange(n):
            s = 0.0
            cnt = 0
            for j in range(k):
                d = distances[i, j]
                if d > max_dist:
                    out[i, j] = 0.0
                    indices[i, j] = 0
                else:
                    cnt += 1
                    z = np.exp(-0.5 * (d * inv_sigma) ** 2)
                    out[i, j] = z
                    s += z
            if cnt < min_points or s <= 0.0:
                for j in range(k):
                    out[i, j] = np.nan
            else:
                inv = 1.0 / s
                for j in range(k):
                    out[i, j] *= inv
except ImportError:
    _kernel_mean_numba = None
    _kernel_weighted_gather_numba = None
    _kernel_gaussian_normed_numba = None
    _kernel_balance_weights_numba = None


def _gaussian_normed_fast(distances, indices, sigma, max_dist, min_points):
    distances = np.ascontiguousarray(distances)
    indices = np.ascontiguousarray(indices)
    W = np.empty_like(distances)
    _kernel_gaussian_normed_numba(
        distances, indices, float(sigma), float(max_dist), int(min_points), W
    )
    return indices, W


def balance_weights_by_density(
    indices: np.ndarray,
    weights: np.ndarray,
    densities: np.ndarray,
    cap: float,
    mode: Literal["smooth", "hard"] = "smooth",
    eps: float = 1e-12,
) -> np.ndarray:
    """Cap each neighbor's weight by local density, re-normalise rows.
    hard: scale=min(1, cap/density); smooth: scale=cap/(density+cap). Lower cap =
    stronger rebalancing. NaN sentinel rows preserved."""
    if _kernel_balance_weights_numba is not None:
        ind = np.ascontiguousarray(indices)
        w = np.ascontiguousarray(weights, dtype=np.float64)
        dens = np.ascontiguousarray(densities, dtype=np.float64)
        out = np.empty_like(w)
        _kernel_balance_weights_numba(
            ind, w, dens, float(cap), mode == "hard", float(eps), out
        )
        return out
    dens_nei = densities[indices]
    if mode == "hard":
        scale = np.minimum(1.0, cap / np.maximum(dens_nei, eps))
    else:
        scale = cap / (dens_nei + cap + eps)
    raw = weights * scale
    row_sums = raw.sum(axis=1, keepdims=True)
    out = np.full_like(raw, np.nan)
    np.divide(raw, row_sums, out=out, where=row_sums > 0)
    return out


def weighted_gather(indices, weights, source):
    """Numba-accelerated weighted gather: out[i, d] = sum_j w[i,j] * source[idx[i,j], d].

    Mean (source=y) and centroid (source=pts) share this primitive."""
    if _kernel_weighted_gather_numba is None:
        nbr = source[indices]
        w = np.where(np.isfinite(weights), weights, 0.0)
        return (w[..., None] * nbr).sum(axis=1)
    ind = np.ascontiguousarray(indices)
    w = np.ascontiguousarray(weights)
    src = np.ascontiguousarray(source)
    out = np.empty((ind.shape[0], src.shape[1]), dtype=src.dtype)
    _kernel_weighted_gather_numba(ind, w, src, out)
    return out


def get_gaussian_weighted_knn(
    x,
    tree,
    k: int = 500,
    min_points: int = 20,
    radius: float = 0.1,
    sigma_in_radius: float = 3.0,
    adaptive_sigma: bool = False,
    max_radius: float | None = None,
    densities: np.ndarray | None = None,
    density_power: float = 0.0,
    density_floor: float | None = None,
    density_cap: float | None = None,
    normed_w: bool = True,
):
    """Gaussian-weighted KNN (Nadaraya-Watson smoothing).

    If `adaptive_sigma`:
        - Query the k nearest neighbours (no radius cut).
        - Set sigma per query as (dist to k-th neighbour) / sigma_in_radius.
        - If max_radius is set, zero weights for neighbours beyond it.
    Else:
        - Query up to k neighbours within `radius`; sigma = radius/sigma_in_radius.
        - Neighbours beyond radius get weight 0.

    If `densities` is provided and `density_power > 0`:
        multiply distance weights by densities[idx]**(-density_power),
        after applying optional floor/cap. Renormalise if normed_w=True.
    """
    eps = 1e-12

    if adaptive_sigma:
        distances, indices = _query(tree, x, k=k, workers=KNN_WORKERS)
        finite_mask = np.isfinite(distances)
        max_finite = np.where(finite_mask, distances, -np.inf).max(axis=1)
        sigma = (max_finite / sigma_in_radius).reshape(-1, 1) + eps
        if max_radius is not None:
            valid_mask = finite_mask & (distances <= max_radius)
        else:
            valid_mask = finite_mask
    else:
        distances, indices = _query(tree, x, k=k, distance_upper_bound=radius, workers=KNN_WORKERS)
        sigma = (radius / sigma_in_radius) + 0.0
        if normed_w and densities is None and _kernel_gaussian_normed_numba is not None:
            return _gaussian_normed_fast(distances, indices, sigma, radius, min_points)
        valid_mask = np.isfinite(distances)

    nb_points = valid_mask.sum(axis=1)
    too_few = nb_points < min_points
    enough = ~too_few
    invalid_mask = ~valid_mask
    if invalid_mask.any():
        indices = indices.copy()
        indices[invalid_mask] = 0

    if too_few.any() and not enough.any():
        return indices, np.full_like(distances, np.nan)

    if too_few.any():
        d_v = distances[enough]
        v_v = valid_mask[enough]
        inv_sigma_v = 1.0 / (sigma[enough] if not np.isscalar(sigma) else sigma)
        Z_v = d_v * inv_sigma_v
        Z_v *= Z_v
        Z_v *= -0.5
        np.exp(Z_v, out=Z_v)
        Z_v[~v_v] = 0.0

        if densities is not None and density_power > 0.0:
            dens_nei = densities[indices[enough]]
            if density_floor is not None:
                dens_nei = np.maximum(dens_nei, density_floor)
            if density_cap is not None:
                dens_nei = np.minimum(dens_nei, density_cap)
            Z_v *= np.power(dens_nei + eps, -density_power)

        if normed_w:
            row_sums = Z_v.sum(axis=1, keepdims=True)
            W_v = np.full_like(Z_v, np.nan)
            np.divide(Z_v, row_sums, out=W_v, where=row_sums > 0)
            W = np.full_like(distances, np.nan)
            W[enough] = W_v
            return indices, W
        Z = np.full_like(distances, np.nan)
        Z[enough] = Z_v
        return indices, Z

    inv_sigma = 1.0 / sigma
    Z = distances * inv_sigma
    Z *= Z
    Z *= -0.5
    np.exp(Z, out=Z)
    Z[invalid_mask] = 0.0

    if densities is not None and density_power > 0.0:
        dens_nei = densities[indices]
        if density_floor is not None:
            dens_nei = np.maximum(dens_nei, density_floor)
        if density_cap is not None:
            dens_nei = np.minimum(dens_nei, density_cap)
        Z *= np.power(dens_nei + eps, -density_power)

    if normed_w:
        row_sums = Z.sum(axis=1, keepdims=True)
        W = np.full_like(Z, np.nan)
        np.divide(Z, row_sums, out=W, where=row_sums > 0)
        return indices, W
    return indices, Z


def get_knn_mean_and_variance(x, y, tree=None, iw=None, compute_variance=True, **kw):
    indices, weights = iw if iw is not None else get_gaussian_weighted_knn(x, tree=tree, **kw)

    n_grid = indices.shape[0]
    valid_rows = np.isfinite(weights[:, 0])
    n_outs = y.shape[1] if y.ndim > 1 else 1
    all_valid = valid_rows.all()

    if all_valid:
        ind_v, w_v = indices, weights
    else:
        ind_v, w_v = indices[valid_rows], weights[valid_rows]

    y_neighbors = y[ind_v]
    w = w_v[..., None]
    wy = w * y_neighbors
    mean_v = wy.sum(axis=1)
    if compute_variance:
        second_moment = (wy * y_neighbors).sum(axis=1)
        w2sum = (w_v * w_v).sum(axis=1, keepdims=True)
        var_v = (second_moment - mean_v * mean_v) / np.maximum(1.0 - w2sum, 1e-12)
    else:
        var_v = None

    if all_valid:
        all_nan = np.all(np.isnan(weights), axis=1)
        if all_nan.any():
            mean_v[all_nan] = np.nan
            if var_v is not None:
                var_v[all_nan] = np.nan
        return mean_v, var_v

    weighted_mean = np.full((n_grid, n_outs), np.nan, dtype=mean_v.dtype)
    weighted_mean[valid_rows] = mean_v
    variance = None
    if var_v is not None:
        variance = np.full((n_grid, n_outs), np.nan, dtype=var_v.dtype)
        variance[valid_rows] = var_v
    return weighted_mean, variance


def _knn_mean_from_indices_weights(indices, weights, y):
    n_grid = indices.shape[0]
    n_outs = y.shape[1] if y.ndim > 1 else 1

    if _kernel_mean_numba is not None and y.ndim == 2:
        out = np.empty((n_grid, n_outs), dtype=y.dtype)
        _kernel_mean_numba(indices, weights, y, out)
        return out

    valid_rows = np.isfinite(weights[:, 0])
    if valid_rows.all():
        row_sums = weights.sum(axis=1, keepdims=True)
        np.divide(weights, row_sums, out=weights, where=row_sums > 0)
        y_neighbors = y[indices]
        if y.ndim == 1:
            y_neighbors *= weights
            return y_neighbors.sum(axis=1, keepdims=True)
        y_neighbors *= weights[..., None]
        return y_neighbors.sum(axis=1)

    if not valid_rows.any():
        return np.full((n_grid, n_outs), np.nan, dtype=y.dtype)

    ind_v = indices[valid_rows]
    w_v = weights[valid_rows]
    row_sums = w_v.sum(axis=1, keepdims=True)
    np.divide(w_v, row_sums, out=w_v, where=row_sums > 0)
    y_neighbors = y[ind_v]
    if y.ndim == 1:
        y_neighbors *= w_v
        mean_v = y_neighbors.sum(axis=1, keepdims=True)
    else:
        y_neighbors *= w_v[..., None]
        mean_v = y_neighbors.sum(axis=1)

    weighted_mean = np.full((n_grid, n_outs), np.nan, dtype=mean_v.dtype)
    weighted_mean[valid_rows] = mean_v
    return weighted_mean


def get_knn_mean_only(x, y, tree=None, iw=None, **kw):
    if iw is not None:
        return _knn_mean_from_indices_weights(iw[0], iw[1], y)

    chunk_size = KNN_MEAN_CHUNK_SIZE
    if chunk_size > 0 and x.shape[0] > chunk_size:
        chunks = []
        for start in range(0, x.shape[0], chunk_size):
            stop = min(start + chunk_size, x.shape[0])
            indices, weights = _ggw_unnormed_cached(x[start:stop], tree, kw)
            chunks.append(_knn_mean_from_indices_weights(indices, weights, y))
        return np.concatenate(chunks, axis=0)

    indices, weights = _ggw_unnormed_cached(x, tree, kw)
    return _knn_mean_from_indices_weights(indices, weights, y)
