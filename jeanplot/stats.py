"""Small statistical helpers used by panel rendering code.

Mirrors `biocomp.metric_utils` for the bits that the drawing functions need.
Kept tiny on purpose; biocomp keeps its own canonical version.
"""

import numpy as np


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mse(y, yhat):
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def r_squared(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def pearson_r(y, yhat):
    y = np.asarray(y).ravel()
    yhat = np.asarray(yhat).ravel()
    if y.size < 2:
        return 0.0
    return float(np.corrcoef(y, yhat)[0, 1])
