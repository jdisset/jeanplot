import numpy as np
from jeanplot.knn.tree import make_tree, _query, KNN_WORKERS


def _ball_volume(d: int) -> float:
    from scipy.special import gamma

    return (np.pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)


def per_point_knn_density(tree, X_ref=None, kdensity: int = 50):
    """KNN density (points per unit d-volume) for the points that define ``tree``.

    Works in any dimension ``d = X_ref.shape[1]``.
    """
    if X_ref is None:
        X_ref = getattr(tree, "data", None)
        if X_ref is None:
            raise ValueError(
                "Cannot infer reference coordinates for density. "
                "Pass X_ref or use a tree exposing `.data`."
            )
    dists, _ = _query(tree, X_ref, k=kdensity + 1)
    rk = dists[:, -1]
    d = X_ref.shape[1]
    Vd = _ball_volume(d)
    return kdensity / (Vd * np.maximum(rk, 1e-12) ** d)


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
