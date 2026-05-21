import numpy as np
from jeanplot.knn.tree import make_tree, _query, KNN_WORKERS


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
