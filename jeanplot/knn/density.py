import threading

import numpy as np
from jeanplot.knn.tree import array_content_key, make_tree, _query, KNN_WORKERS


def _ball_volume(d: int) -> float:
    from scipy.special import gamma

    return (np.pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)


# keyed by id(tree); the strong tree ref in the value pins id so it can't alias a freed tree
_PPD_CACHE: dict = {}
_PPD_CACHE_MAX = 16
_PPD_CACHE_LOCK = threading.Lock()


def per_point_knn_density(tree, X_ref=None, kdensity: int = 50):
    """KNN density per reference point; memoised by (tree, kdensity, X_ref)."""
    xref_key = array_content_key(X_ref) if X_ref is not None else None
    if X_ref is None:
        X_ref = getattr(tree, "data", None)
        if X_ref is None:
            raise ValueError("density needs X_ref or a tree exposing `.data`")

    key = (id(tree), int(kdensity), xref_key)
    with _PPD_CACHE_LOCK:
        ent = _PPD_CACHE.get(key)
        if ent is not None:
            return ent[1]

    dists, _ = _query(tree, X_ref, k=kdensity + 1, workers=KNN_WORKERS)
    d = X_ref.shape[1]
    result = kdensity / (_ball_volume(d) * np.maximum(dists[:, -1], 1e-12) ** d)

    with _PPD_CACHE_LOCK:
        if len(_PPD_CACHE) >= _PPD_CACHE_MAX:
            _PPD_CACHE.pop(next(iter(_PPD_CACHE)))
        _PPD_CACHE[key] = (tree, result)
    return result


def uniform_resampling(
    X, npoints: int = 1000, kdensity=50, density_floor_q=0.01, density_cap_q=0.99
):
    """Importance-weighted resampling that flattens KNN density."""
    tree = make_tree(X)
    densities = per_point_knn_density(tree=tree, X_ref=X, kdensity=kdensity)
    density_floor = float(np.quantile(densities, density_floor_q))
    density_cap = float(np.quantile(densities, density_cap_q))
    densities = np.clip(densities, density_floor, density_cap)
    weights = 1.0 / densities
    weights /= weights.sum()
    indices = np.random.choice(np.arange(X.shape[0]), size=npoints, replace=True, p=weights)
    return indices, weights[indices]


def knn_density(
    X: np.ndarray,
    k: int = 64,
    eps: float = 1e-12,
    tree=None,
) -> np.ndarray:
    """Density proxy via k-th nearest neighbour distance.

    Density ~ 1 / (d_k + eps)^D where D is dimensionality.
    Returns unnormalised density values suitable for importance sampling.
    """
    if tree is None:
        tree = make_tree(X)
    d, _ = _query(tree, X, k=k + 1)
    d_k = d[:, -1]
    dim = X.shape[1]
    return 1.0 / np.power(d_k + eps, dim)


def knn_density_chunked(
    X: np.ndarray,
    k: int = 64,
    eps: float = 1e-12,
    chunksize: int = 50000,
    tree=None,
) -> np.ndarray:
    """Chunked KNN density for large datasets. Builds tree once."""
    if tree is None:
        tree = make_tree(X)
    n = X.shape[0]
    dim = X.shape[1]
    result = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunksize):
        end = min(i + chunksize, n)
        d, _ = _query(tree, X[i:end], k=k + 1, workers=KNN_WORKERS)
        d_k = d[:, -1]
        result[i:end] = 1.0 / np.power(d_k + eps, dim)
    return result
