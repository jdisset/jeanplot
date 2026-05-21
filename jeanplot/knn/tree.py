import os
import numpy as np
from scipy.spatial import cKDTree


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


KNN_BACKEND = os.environ.get("JEANPLOT_KNN_BACKEND", "usearch").lower()
KNN_WORKERS = _env_int("JEANPLOT_KNN_WORKERS", -1)
KNN_MEAN_CHUNK_SIZE = _env_int("JEANPLOT_KNN_MEAN_CHUNK_SIZE", 2500)
KNN_ANN_M = _env_int("JEANPLOT_KNN_ANN_M", 4)
KNN_ANN_EF_CONSTRUCTION = _env_int("JEANPLOT_KNN_ANN_EF_CONSTRUCTION", 30)
KNN_ANN_EF_SEARCH = _env_int("JEANPLOT_KNN_ANN_EF_SEARCH", 64)


try:
    from pykdtree.kdtree import KDTree as _PKDTree

    if KNN_WORKERS == 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
except ImportError:
    _PKDTree = None

try:
    from usearch.index import Index as _UIndex
except ImportError:
    _UIndex = None


def _resolve_backend(requested: str) -> str:
    if requested in ("usearch", "ann") and _UIndex is not None:
        return "usearch"
    if _PKDTree is not None:
        return "pykdtree"
    return "scipy"


_BACKEND = _resolve_backend(KNN_BACKEND)


def _resolve_threads() -> int:
    raw = _env_int("JEANPLOT_KNN_WORKERS", -1)
    return 0 if raw in (-1, 0) else max(1, raw)


class _UsearchTree:
    """Adapter so usearch HNSW Index quacks like scipy.cKDTree.query.

    Returns (distances, indices); invalid neighbors get inf distance and
    `n` (sentinel) index. Distances are true L2.
    """

    __slots__ = ("_index", "_n")

    def __init__(self, x: np.ndarray):
        n, d = x.shape
        idx = _UIndex(
            ndim=d,
            metric="l2sq",
            connectivity=KNN_ANN_M,
            expansion_add=KNN_ANN_EF_CONSTRUCTION,
            expansion_search=KNN_ANN_EF_SEARCH,
        )
        idx.add(
            np.arange(n, dtype=np.int64),
            np.ascontiguousarray(x, dtype=np.float32),
            threads=_resolve_threads(),
        )
        self._index = idx
        self._n = n

    def query(self, x, k, distance_upper_bound=None, **_):
        n = self._n
        x = np.atleast_2d(np.ascontiguousarray(x, dtype=np.float32))
        m = self._index.search(x, k, threads=_resolve_threads())
        labels = m.keys
        sq = m.distances
        counts = m.counts
        if (counts < k).any():
            col = np.arange(k)
            invalid = col[None, :] >= counts[:, None]
            sq[invalid] = 0.0
            distances = np.sqrt(sq, out=sq)
            distances[invalid] = np.inf
            labels[invalid] = n
        else:
            distances = np.sqrt(sq, out=sq)
        if distance_upper_bound is not None:
            mask = distances > distance_upper_bound
            distances[mask] = np.inf
            labels[mask] = n
        return distances, labels


def make_tree(x: np.ndarray):
    if _BACKEND == "usearch":
        return _UsearchTree(x)
    if _BACKEND == "pykdtree":
        return _PKDTree(np.ascontiguousarray(x, dtype=np.float64))
    return cKDTree(x, leafsize=32)


def _query(tree, x, **kw):
    if isinstance(tree, _UsearchTree):
        kw.pop("workers", None)
        return tree.query(x, **kw)
    if _PKDTree is not None and isinstance(tree, _PKDTree):
        kw.pop("workers", None)
    return tree.query(x, **kw)
