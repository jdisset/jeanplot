# SPDX-License-Identifier: MIT
"""Scatter/splat kernel smoother: deposit moment channels onto a lattice, blur
once, derive every stat as a ratio of blurred buffers. SSOT replacement for the
KNN-gather smoother (`jeanplot.knn`). Exact fixed-bandwidth Nadaraya-Watson."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

_Q_FLOOR = 0.02  # mirror smooth_kernel._REBALANCE_Q_FLOOR
_EPS = 1e-12

_VALUE_STATS = frozenset({"mean", "variance", "std", "density", "grad"})
_CENTROID_STATS = frozenset({"centroid", "centroid_offset"})


def _balance_factor(dens, cap, hard):
    if hard:
        return np.minimum(1.0, cap / np.maximum(dens, _EPS))
    return cap / (dens + cap + _EPS)


def _cap(dens, strength):
    return float(np.quantile(dens, max(1.0 - strength, _Q_FLOOR)))


def _ball_kernel(cell, radius):
    """Top-hat ball (radius in data units) discretised on the lattice."""
    halfs = np.ceil(radius / cell).astype(int)
    axes = np.meshgrid(*[np.arange(-h, h + 1) for h in halfs], indexing="ij")
    r2 = sum((g * c) ** 2 for g, c in zip(axes, cell))
    return (r2 <= radius**2).astype(np.float64)


def _deposit(coords, payloads, origin, cell, shape):
    """Bilinear (cloud-in-cell) scatter of every payload column into a padded
    lattice. Returns (prod(shape), n_channels) accumulated in C order."""
    n, d = coords.shape
    f = (coords - origin) / cell
    base = np.floor(f).astype(np.int64)
    frac = f - base
    ncells = int(np.prod(shape))
    out = np.zeros((ncells, payloads.shape[1]))
    for corner in range(1 << d):
        flat = np.zeros(n, dtype=np.int64)
        w = np.ones(n)
        valid = np.ones(n, dtype=bool)
        for ax in range(d):
            bit = (corner >> ax) & 1
            cidx = base[:, ax] + bit
            w = w * (frac[:, ax] if bit else 1.0 - frac[:, ax])
            valid &= (cidx >= 0) & (cidx < shape[ax])
            flat = flat * shape[ax] + np.clip(cidx, 0, shape[ax] - 1)
        w = w * valid
        for ch in range(payloads.shape[1]):
            out[:, ch] += np.bincount(flat, weights=w * payloads[:, ch], minlength=ncells)
    return out


def splat_point_density(X, *, radius, sigma_in_radius=3.0, res=64):
    """Tree-free KDE proxy at each point: deposit unit mass, blur, sample back.
    Scale-arbitrary (only relative density feeds the rebalance cap ratio)."""
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    sigma = radius / sigma_in_radius
    lo = X.min(axis=0) - radius
    hi = X.max(axis=0) + radius
    cell = np.where(hi > lo, (hi - lo) / (res - 1), 1.0)
    origin = lo
    shape = tuple([res] * d)
    buf = _deposit(X, np.ones((X.shape[0], 1)), origin, cell, shape).reshape(*shape)
    sig_cells = sigma / cell
    buf = gaussian_filter(buf, sig_cells, truncate=sigma_in_radius, mode="constant")
    axes = tuple(np.linspace(lo[k], hi[k], res) for k in range(d))
    interp = RegularGridInterpolator(axes, buf, method="linear", bounds_error=False, fill_value=0.0)
    return np.asarray(interp(X))


class SplatField:
    """Moment-buffer kernel smoother over a d-dim lattice (d in {1,2,3})."""

    def __init__(self, buffers, axes, n_eff, n_outs, sigma_in_radius):
        self._b = buffers
        self._axes = axes
        self._n_eff = n_eff
        self.ndim = len(axes)
        self.n_outs = n_outs
        self.sigma_in_radius = sigma_in_radius

    @classmethod
    def fit(
        cls,
        X,
        Y=None,
        *,
        bounds,
        resolution,
        radius,
        sigma_in_radius=3.0,
        min_points=0,
        rebalance_values=0.0,
        rebalance_values_mode="smooth",
        rebalance_centroids=0.0,
        rebalance_centroids_mode="hard",
        zslice=None,
        stats=("mean",),
    ):
        X = np.asarray(X, dtype=np.float64)
        d = len(bounds)
        res = int(resolution)
        sigma = radius / sigma_in_radius
        stats = set(stats)
        want_val = bool(_VALUE_STATS & stats)
        want_cen = bool(_CENTROID_STATS & stats)
        want_grad = "grad" in stats

        finite = np.all(np.isfinite(X), axis=1)
        if Y is not None:
            Y = np.asarray(Y, dtype=np.float64)
            Y = Y if Y.ndim == 2 else Y[:, None]
            finite &= np.all(np.isfinite(Y), axis=1)
        X = X[finite]
        if Y is not None:
            Y = Y[finite]
        n_outs = Y.shape[1] if Y is not None else 1

        # rebalance from tree-free density on the full (pre-band) cloud
        wv = np.ones(X.shape[0])
        wc = np.ones(X.shape[0])
        if X.shape[0] and (rebalance_values > 0.0 or rebalance_centroids > 0.0):
            dens = splat_point_density(X, radius=radius, sigma_in_radius=sigma_in_radius)
            if rebalance_values > 0.0:
                wv = _balance_factor(
                    dens, _cap(dens, rebalance_values), rebalance_values_mode == "hard"
                )
            if rebalance_centroids > 0.0:
                wc = _balance_factor(
                    dens, _cap(dens, rebalance_centroids), rebalance_centroids_mode == "hard"
                )

        sp = X[:, :d]
        base_w = np.ones(X.shape[0])
        if zslice is not None and X.shape[1] > d:
            z = X[:, d:]
            z0 = np.asarray(zslice, dtype=np.float64)
            dz = np.linalg.norm(z - z0, axis=1)
            band = dz <= radius
            base_w = np.exp(-0.5 * (dz / sigma) ** 2) * band
            sp, wv, wc, base_w = sp[band], wv[band], wc[band], base_w[band]
            if Y is not None:
                Y = Y[band]

        cell = np.array([(hi - lo) / (res - 1) for lo, hi in bounds])
        margin = int(np.ceil(radius / cell.min())) + 1
        origin = np.array([lo for lo, _ in bounds]) - margin * cell
        pshape = tuple([res + 2 * margin] * d)
        sig_cells = sigma / cell

        # `cnt` is a plain within-band indicator (ball-counted below for the
        # min_points mask); value/centroid channels carry the deposit weights.
        cols = []
        wv_eff = base_w * wv
        cols.append(("cnt", np.ones(sp.shape[0])))
        if want_val:
            cols.append(("C", wv_eff))
            if Y is not None:
                for o in range(n_outs):
                    cols.append((f"Sy{o}", wv_eff * Y[:, o]))
                if {"variance", "std"} & stats:
                    for o in range(n_outs):
                        cols.append((f"Syy{o}", wv_eff * Y[:, o] ** 2))
            if want_grad:
                assert Y is not None, "grad stat requires y"
                for k in range(d):
                    cols.append((f"Sx{k}", wv_eff * sp[:, k]))
                tri = [(a, b) for a in range(d) for b in range(a, d)]
                for a, b in tri:
                    cols.append((f"Sxx{a}{b}", wv_eff * sp[:, a] * sp[:, b]))
                for k in range(d):
                    cols.append((f"Sxy{k}", wv_eff * sp[:, k] * Y[:, 0]))
        if want_cen:
            wc_eff = base_w * wc
            cols.append(("Cc", wc_eff))
            for k in range(d):
                cols.append((f"Cx{k}", wc_eff * sp[:, k]))

        names = [c[0] for c in cols]
        if sp.shape[0] == 0:
            payload = np.zeros((0, len(cols)))
        else:
            payload = np.column_stack([c[1] for c in cols])
        dep = (
            _deposit(sp, payload, origin, cell, pshape)
            if sp.shape[0]
            else np.zeros((int(np.prod(pshape)), len(cols)))
        )
        dep = dep.reshape(*pshape, len(cols))

        crop = tuple(slice(margin, margin + res) for _ in range(d))
        buffers = {}
        cnt_idx = names.index("cnt")
        for j, nm in enumerate(names):
            if nm == "cnt":
                continue
            blurred = gaussian_filter(
                dep[..., j], sig_cells, truncate=sigma_in_radius, mode="constant"
            )
            buffers[nm] = blurred[crop]

        # min_points counts points within `radius` (hard) -> top-hat ball
        # convolution, NOT the gaussian soft-count (sigma=radius/3 undercounts).
        ball = _ball_kernel(cell, radius)
        n_eff = fftconvolve(dep[..., cnt_idx], ball, mode="same")[crop]
        n_eff = np.where(n_eff >= max(min_points, 1), n_eff, np.nan)
        axes = tuple(np.linspace(lo, hi, res) for lo, hi in bounds)
        return cls(buffers, axes, n_eff, n_outs, sigma_in_radius)

    def _support(self):
        return np.isfinite(self._n_eff)

    def lattice(self, stat):
        """Stat on the lattice: (res,)*ndim, plus a trailing axis for vector stats."""
        b = self._b
        sup = self._support()
        if stat == "density":
            return np.where(sup, 1.0, 0.0)
        if stat == "n_eff":
            return self._n_eff
        if stat in ("mean", "variance", "std"):
            C = b["C"]
            ok = sup & (C > 0)
            outs = []
            for o in range(self.n_outs):
                mean = np.where(ok, b[f"Sy{o}"] / np.where(C > 0, C, 1.0), np.nan)
                if stat == "mean":
                    outs.append(mean)
                    continue
                var = np.where(ok, b[f"Syy{o}"] / np.where(C > 0, C, 1.0) - mean**2, np.nan)
                var = np.maximum(var, 0.0)
                outs.append(var if stat == "variance" else np.sqrt(var))
            return np.stack(outs, axis=-1)
        if stat == "centroid":
            return self._centroid()
        if stat == "centroid_offset":
            cen = self._centroid()
            mesh = np.stack(np.meshgrid(*self._axes, indexing="ij"), axis=-1)
            return np.linalg.norm(cen - mesh, axis=-1)
        if stat == "grad":
            return self._grad()
        raise ValueError(f"unknown stat: {stat}")

    def _centroid(self):
        b = self._b
        Cc = b["Cc"]
        ok = self._support() & (Cc > 0)
        d = self.ndim
        cen = np.full((*Cc.shape, d), np.nan)
        for k in range(d):
            cen[..., k] = np.where(ok, b[f"Cx{k}"] / np.where(Cc > 0, Cc, 1.0), np.nan)
        return cen

    def _grad(self):
        b = self._b
        C = b["C"]
        d = self.ndim
        ok = self._support() & (C > 0)
        Cs = np.where(C > 0, C, 1.0)
        mx = np.stack([b[f"Sx{k}"] / Cs for k in range(d)], axis=-1)
        my = b["Sy0"] / Cs
        cxy = np.stack([b[f"Sxy{k}"] / Cs - mx[..., k] * my for k in range(d)], axis=-1)
        cxx = np.zeros((*C.shape, d, d))
        for a in range(d):
            for bdim in range(a, d):
                m2 = b[f"Sxx{a}{bdim}"] / Cs
                v = m2 - mx[..., a] * mx[..., bdim]
                cxx[..., a, bdim] = v
                cxx[..., bdim, a] = v
        tr = np.trace(cxx, axis1=-2, axis2=-1)
        ridge = 1e-9 + 1e-6 * tr
        for k in range(d):
            cxx[..., k, k] += ridge
        grad = np.full((*C.shape, d), np.nan)
        flat_ok = ok & np.isfinite(cxy).all(-1) & np.isfinite(cxx).all((-2, -1))
        idx = np.where(flat_ok)
        if idx[0].size:
            sol = np.linalg.solve(cxx[flat_ok], cxy[flat_ok][..., None])[..., 0]
            grad[flat_ok] = sol
        return grad

    def at(self, points, stat):
        """Interpolate a lattice stat to arbitrary query points (NaN outside)."""
        lat = self.lattice(stat)
        vec = lat.ndim > self.ndim
        arr = lat if vec else lat[..., None]
        interp = RegularGridInterpolator(
            self._axes, arr, method="linear", bounds_error=False, fill_value=np.nan
        )
        out = np.asarray(interp(np.asarray(points)))
        return out if vec else out[..., 0]

    def flat_xy(self, stat):
        """Lattice stat flattened in `make_xy_grid` order (ndim==2): out[i*res+j]
        = cell (x=axis0[j], y=axis1[i])."""
        assert self.ndim == 2
        lat = self.lattice(stat)
        if lat.ndim == 2:
            return lat.T.ravel()
        return lat.transpose(1, 0, 2).reshape(-1, lat.shape[-1])
