# SPDX-License-Identifier: MIT
"""ConditionalSplat: local P(value | location) via a spatial x value histogram,
blurred along the spatial (conditioning) axes only. Replaces the gather `iw`
per-neighbour KDE pattern (mvp violins, voxel distributions)."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from jeanplot.splat.core import (
    _balance_factor,
    _ball_kernel,
    _cap,
    cic_corners,
    splat_point_density,
)


class ConditionalSplat:
    """Local value distribution conditioned on d-dim location (d in {1,2,3})."""

    def __init__(self, hist, axes, value_grid, n_eff):
        self._h = hist
        self._axes = axes
        self.value_grid = value_grid
        self._n_eff = n_eff
        self.ndim = len(axes)

    @classmethod
    def fit(
        cls,
        X,
        Y,
        *,
        bounds,
        resolution,
        radius,
        sigma_in_radius=3.0,
        min_points=1,
        value_range=None,
        n_value_bins=64,
        rebalance_values=0.0,
        rebalance_values_mode="smooth",
    ):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).ravel()
        d = len(bounds)
        res = int(resolution)
        sigma = radius / sigma_in_radius
        m = np.all(np.isfinite(X), axis=1) & np.isfinite(Y)
        X, Y = X[m, :d], Y[m]

        w = np.ones(X.shape[0])
        if X.shape[0] and rebalance_values > 0.0:
            dens = splat_point_density(X, radius=radius, sigma_in_radius=sigma_in_radius)
            w = _balance_factor(dens, _cap(dens, rebalance_values), rebalance_values_mode == "hard")

        vlo, vhi = (float(Y.min()), float(Y.max())) if value_range is None else value_range
        vhi = vhi if vhi > vlo else vlo + 1.0
        value_grid = np.linspace(vlo, vhi, n_value_bins)
        vcell = (vhi - vlo) / (n_value_bins - 1)

        cell = np.array([(hi - lo) / (res - 1) for lo, hi in bounds])
        margin = int(np.ceil(radius / cell.min())) + 1
        origin = np.array([lo for lo, _ in bounds]) - margin * cell
        pshape = tuple([res + 2 * margin] * d)
        ncells = int(np.prod(pshape))
        nb = n_value_bins

        H = np.zeros((ncells, nb))
        cnt = np.zeros(ncells)
        if X.shape[0]:
            vf = (Y - vlo) / vcell
            vb = np.clip(np.floor(vf).astype(np.int64), 0, nb - 1)
            vt = np.clip(vf - vb, 0.0, 1.0)
            for sflat, sw in cic_corners(X, origin, cell, pshape):
                cnt += np.bincount(sflat, weights=sw, minlength=ncells)
                for voff, vw in ((0, 1.0 - vt), (1, vt)):
                    vbin = np.clip(vb + voff, 0, nb - 1)
                    np.add.at(H, (sflat, vbin), sw * vw * w)

        H = H.reshape(*pshape, nb)
        sig = list(sigma / cell) + [0.0]
        H = gaussian_filter(H, sig, truncate=sigma_in_radius, mode="constant")
        crop = tuple(slice(margin, margin + res) for _ in range(d))
        H = H[crop]

        ball = _ball_kernel(cell, radius)
        n_eff = fftconvolve(cnt.reshape(*pshape), ball, mode="same")[crop]
        n_eff = np.where(n_eff >= max(min_points, 1), n_eff, np.nan)
        axes = tuple(np.linspace(lo, hi, res) for lo, hi in bounds)
        return cls(H, axes, value_grid, n_eff)

    def pdf_at(self, points):
        """(value_grid, pdf) for each query point; pdf rows sum to 1, NaN where
        unsupported."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        interp = RegularGridInterpolator(
            self._axes, self._h, method="linear", bounds_error=False, fill_value=0.0
        )
        sup = RegularGridInterpolator(
            self._axes,
            np.isfinite(self._n_eff).astype(float),
            method="nearest",
            bounds_error=False,
            fill_value=0.0,
        )
        h = np.asarray(interp(pts))
        row = h.sum(axis=1, keepdims=True)
        pdf = np.full_like(h, np.nan)
        ok = (row[:, 0] > 0) & (np.asarray(sup(pts)) > 0)
        pdf[ok] = h[ok] / row[ok]
        return self.value_grid, pdf

    def mean_at(self, points):
        _, pdf = self.pdf_at(points)
        return pdf @ self.value_grid

    def count_at(self, points):
        """Effective within-radius neighbour count at each query point."""
        interp = RegularGridInterpolator(
            self._axes,
            np.nan_to_num(self._n_eff),
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return np.asarray(interp(np.atleast_2d(np.asarray(points, dtype=np.float64))))

    def quantiles_at(self, points, qs):
        """(Nq, len(qs)) value-quantiles of the local distribution (NaN where
        unsupported)."""
        _, pdf = self.pdf_at(points)
        qs = np.atleast_1d(qs)
        out = np.full((pdf.shape[0], qs.shape[0]), np.nan)
        for i, row in enumerate(pdf):
            if not np.isfinite(row).all():
                continue
            cdf = np.cumsum(row)
            out[i] = np.interp(qs, cdf, self.value_grid)
        return out
