"""Density-saturated centroid / mean reweighting (jeanplot.knn.gaussian).

Covers:
- primitive math (hard + smooth) and row renorm
- true border still rejected
- imbalanced-but-covered region not rejected
- mean rebalances away from dense side
"""

import numpy as np

from jeanplot.knn import balance_weights_by_density, make_tree
from jeanplot.plots.smooth_kernel import knn_stats


def _two_clusters_x(n_dense=2000, n_sparse=100, seed=0):
    """Dense cluster at x=-1, sparse cluster at x=+1, both spread in y."""
    rng = np.random.default_rng(seed)
    dense = np.column_stack([rng.normal(-1.0, 0.15, n_dense), rng.normal(0.0, 0.5, n_dense)])
    sparse = np.column_stack([rng.normal(+1.0, 0.15, n_sparse), rng.normal(0.0, 0.5, n_sparse)])
    return np.vstack([dense, sparse]).astype(np.float64)


def test_balance_hard_caps_then_renormalises():
    indices = np.array([[0, 1]])
    weights = np.array([[0.5, 0.5]])
    densities = np.array([10.0, 1.0])
    out = balance_weights_by_density(indices, weights, densities, cap=1.0, mode="hard")
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    # neighbor 0 capped 10x, neighbor 1 untouched → ratio 1:10 after renorm
    np.testing.assert_allclose(out[0, 0] / out[0, 1], 1.0 / 10.0, rtol=1e-6)


def test_balance_smooth_no_cap_below_threshold():
    indices = np.array([[0, 1]])
    weights = np.array([[0.5, 0.5]])
    densities = np.array([1e-6, 1e-6])  # both far below cap
    out = balance_weights_by_density(indices, weights, densities, cap=1.0, mode="smooth")
    np.testing.assert_allclose(out, weights, atol=1e-6)


def test_balance_preserves_nan_rows():
    indices = np.array([[0, 1], [0, 1]])
    weights = np.array([[0.5, 0.5], [np.nan, np.nan]])
    densities = np.array([1.0, 1.0])
    out = balance_weights_by_density(indices, weights, densities, cap=0.5, mode="hard")
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


def test_true_border_still_rejected_with_saturation():
    """One-sided support → saturated centroid still shifted → offset large."""
    rng = np.random.default_rng(1)
    # all data at x<0; query at x=0 has empty right half
    x = np.column_stack([rng.uniform(-1.0, -0.05, 500), rng.uniform(-1, 1, 500)])
    y = np.zeros((x.shape[0], 1))
    xq = np.array([[0.0, 0.0]])

    tree = make_tree(x)
    base = dict(tree=tree, k=200, radius=0.5, min_points=5, stats=["centroid_offset"])
    off_plain = float(knn_stats(xq, y, **base)[0])
    off_sat = float(
        knn_stats(
            xq,
            y,
            rebalance_centroids=0.5,
            rebalance_centroids_mode="hard",
            **base,
        )[0]
    )
    # both should be a large fraction of the radius — border survives
    assert off_plain > 0.1
    assert off_sat > 0.1


def test_imbalanced_but_covered_recentered_by_saturation():
    """Data on both sides but dense vs sparse → saturated centroid moves back."""
    x = _two_clusters_x()
    y = np.zeros((x.shape[0], 1))
    xq = np.array([[0.0, 0.0]])  # midway between clusters

    tree = make_tree(x)
    base = dict(tree=tree, k=2000, radius=2.0, min_points=5, stats=["centroid_offset"])
    off_plain = float(knn_stats(xq, y, **base)[0])
    off_sat = float(
        knn_stats(
            xq,
            y,
            rebalance_centroids=0.95,
            rebalance_centroids_mode="hard",
            **base,
        )[0]
    )
    # plain centroid pulled hard to the dense cluster; saturation rebalances
    assert off_sat < off_plain * 0.5


def test_mean_rebalances_away_from_dense_side():
    """y encodes side: dense side y=-1, sparse side y=+1. Mean should move toward 0."""
    rng = np.random.default_rng(2)
    n_dense, n_sparse = 2000, 100
    x_dense = np.column_stack([rng.normal(-1.0, 0.15, n_dense), rng.normal(0.0, 0.5, n_dense)])
    x_sparse = np.column_stack([rng.normal(+1.0, 0.15, n_sparse), rng.normal(0.0, 0.5, n_sparse)])
    x = np.vstack([x_dense, x_sparse]).astype(np.float64)
    y = np.concatenate([-np.ones(n_dense), np.ones(n_sparse)])[:, None]
    xq = np.array([[0.0, 0.0]])

    tree = make_tree(x)
    base = dict(tree=tree, k=2000, radius=2.0, min_points=5, stats=["mean"])
    mu_plain = float(knn_stats(xq, y, **base).reshape(-1)[0])
    mu_sat = float(
        knn_stats(
            xq,
            y,
            rebalance_values=0.95,
            rebalance_values_mode="smooth",
            **base,
        ).reshape(-1)[0]
    )
    # plain mean dominated by dense (negative); saturation pulls toward 0
    assert mu_plain < -0.5
    assert mu_sat > mu_plain + 0.3
