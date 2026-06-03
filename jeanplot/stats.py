"""Elementary regression metrics — SSOT for the generic formulas.

These are the canonical, dependency-light (numpy only) kernels. `biocomp`
re-imports them rather than redefining. All ignore non-finite pairs and return
nan when there's nothing valid to compute. Domain-specific metrics (grid stats,
nRMSE, distributional objectives, pearson p-value) live in `biocomp.metric_utils`
on top of these.
"""

import numpy as np

EPS = 1e-9


def _masked(y, yhat):
    y = np.asarray(y).ravel()
    yhat = np.asarray(yhat).ravel()
    m = np.isfinite(y) & np.isfinite(yhat)
    return y[m], yhat[m]


def mse(y, yhat):
    y, yhat = _masked(y, yhat)
    return float(np.mean((y - yhat) ** 2)) if y.size else float("nan")


def rmse(y, yhat):
    return float(np.sqrt(mse(y, yhat)))


def mae(y, yhat):
    y, yhat = _masked(y, yhat)
    return float(np.mean(np.abs(y - yhat))) if y.size else float("nan")


def max_error(y, yhat):
    y, yhat = _masked(y, yhat)
    return float(np.max(np.abs(y - yhat))) if y.size else float("nan")


def r_squared(y, yhat):
    y, yhat = _masked(y, yhat)
    if not y.size:
        return float("nan")
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan")


def pearson_r(y, yhat):
    """Correlation coefficient only (nan if <2 valid pairs). For the p-value too,
    use biocomp.metric_utils.pearson_r."""
    y, yhat = _masked(y, yhat)
    return float(np.corrcoef(y, yhat)[0, 1]) if y.size >= 2 else float("nan")
