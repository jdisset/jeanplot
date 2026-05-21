"""Stacked-polynomial fit at quantile-positioned knots.

NumPy port of the JAX-based biocomp `stacked_poly.py` (canonical source: calibrie).
The JAX original is kept on the biocomp side; jeanplot avoids the JAX dep.
"""

import numpy as np


def _gaussian_kernel(x, mean, std, scaler=1e12):
    return scaler * np.exp(-((x - mean) ** 2) / (2 * std**2))


def _fit_coeffs(x, y, w, mticks, stds, degree=1):
    n_ticks = len(mticks)
    coeffs = np.zeros((n_ticks, degree + 1), dtype=float)
    for i in range(n_ticks):
        weights = _gaussian_kernel(x, mticks[i], stds[i]) * w
        if weights.sum() <= 0:
            weights = np.ones_like(w) * 1e-12
        coeffs[i] = np.polyfit(x, y, deg=degree, w=weights)
    return coeffs


def evaluate_stacked_poly(x, params):
    coeffs, mticks, stds = params
    x = np.asarray(x, dtype=float)
    n_ticks = len(mticks)
    evals = np.stack([np.polyval(coeffs[i], x) for i in range(n_ticks)], axis=0)
    eval_weights = np.stack(
        [_gaussian_kernel(x, mticks[i], stds[i]) for i in range(n_ticks)], axis=0
    )
    EPS = 1e-9
    eval_weights[0] = np.where(x < mticks[0], np.clip(eval_weights[0], EPS, None), eval_weights[0])
    eval_weights[-1] = np.where(
        x > mticks[-1], np.clip(eval_weights[-1], EPS, None), eval_weights[-1]
    )
    return np.average(evals, weights=eval_weights, axis=0)


def fit_stacked_poly_at_quantiles(x, y, w, quantiles, degree=1):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    mticks = np.quantile(x, quantiles)
    diff = np.pad(np.diff(mticks), (1, 1), mode="edge")
    stds = (diff[:-1] + diff[1:]) / 2
    coeffs = _fit_coeffs(x, y, w, mticks, stds, degree=degree)
    return (coeffs, mticks, stds)
