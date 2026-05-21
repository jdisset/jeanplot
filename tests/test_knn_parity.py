import numpy as np
import pytest

from jeanplot.knn import (
    make_tree,
    knn_density,
    get_gaussian_weighted_knn,
    get_knn_mean_only,
)


def _sample(seed: int = 0, n: int = 400, d: int = 2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, d)).astype(np.float32)
    y = (np.sin(2 * np.pi * x[:, :1]) + 0.05 * rng.standard_normal((n, 1))).astype(np.float32)
    return x, y


def test_tree_query_returns_distance_and_indices():
    x, _ = _sample()
    tree = make_tree(x)
    d, idx = get_gaussian_weighted_knn(x[:10], tree=tree, k=20, radius=0.5, min_points=2)
    assert d.shape == (10, 20)
    assert idx.shape == (10, 20)


def test_knn_density_dimensions():
    x, _ = _sample()
    dens = knn_density(x, k=16)
    assert dens.shape == (x.shape[0],)
    assert np.all(np.isfinite(dens))


def test_gaussian_weighted_knn_normed_sums_to_one():
    x, _ = _sample()
    tree = make_tree(x)
    _, w = get_gaussian_weighted_knn(x[:50], tree=tree, k=64, radius=0.4, min_points=5)
    finite_rows = np.isfinite(w[:, 0])
    if finite_rows.any():
        row_sums = np.nansum(w[finite_rows], axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_knn_mean_smooths_signal():
    x, y = _sample(seed=1, n=600)
    tree = make_tree(x)
    grid = np.linspace(0.0, 1.0, 30, dtype=np.float32)
    xq = np.stack([grid, np.full_like(grid, 0.5)], axis=1)
    mu = get_knn_mean_only(xq, y, tree=tree, k=64, radius=0.3, min_points=5)
    assert mu.shape == (30, 1)
    valid = np.isfinite(mu[:, 0])
    assert valid.sum() > 5


def test_parity_against_biocomp():
    bio_knn = pytest.importorskip("biocomp.plotting.knn_utils_np")
    x, _ = _sample(seed=42, n=300)
    tree_a = make_tree(x)
    tree_b = bio_knn.make_tree(x)
    kw = dict(k=64, radius=0.25, min_points=5)
    _, w_a = get_gaussian_weighted_knn(x[:50], tree=tree_a, **kw)
    _, w_b = bio_knn.get_gaussian_weighted_knn(x[:50], tree=tree_b, **kw)
    np.testing.assert_allclose(np.nan_to_num(w_a), np.nan_to_num(w_b), atol=1e-5)
