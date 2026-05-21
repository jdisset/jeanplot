"""Scatter / histogram-2D plotting.

Adapted from biocomp `plotting_scatter.py` with biocomp imports replaced.
The network-aware `scatter_3d` variant is intentionally dropped — it depends
on a biocomp `Network` object.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray as NdArray

from jeanplot.plots.colorbar import colorbar
from jeanplot.plots.ticks import setup_transformed_axis


def make_density_cmap(name=None, alpha_start=1.0, alpha_end=1.0, base_cmap="Spectral_r"):
    ncolors = 256
    color_array = plt.get_cmap(base_cmap)(range(ncolors))
    color_array[:, -1] = np.linspace(alpha_start, alpha_end, ncolors)
    cmap = LinearSegmentedColormap.from_list(name=name, colors=color_array)
    cmap.set_under("w", alpha=0)
    return cmap


DEFAULT_DENSITY_CMAP = make_density_cmap("density", alpha_start=1.0, alpha_end=1.0)


def _log_plus_one_fwd(x):
    return np.log1p(np.asarray(x, dtype=float))


def _log_plus_one_inv(x):
    return np.expm1(np.asarray(x, dtype=float))


class _IdentityScaler:
    def fwd(self, x):
        return np.asarray(x)

    def inv(self, x):
        return np.asarray(x)


class _LogPlusOneScaler:
    def fwd(self, x):
        return _log_plus_one_fwd(x)

    def inv(self, x):
        return _log_plus_one_inv(x)


def grid_histogram(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    title: str | None = None,
    xtitle: str | None = None,
    ytitle: str | None = None,
    xlims=(None, None),
    ylims=(None, None),
    vlims=(0, None),
    draw_xlabel=True,
    draw_ylabel=True,
    res: int = 300,
    draw_colorbar: bool = True,
    use_log_density: bool = True,
    cmap=None,
    margins: float = 0.01,
    noise_smooth: float = 0.25,
    colorbar_params: dict | None = None,
):
    if colorbar_params is None:
        colorbar_params = {}
    assert X.shape[1] == 1
    assert Y.shape[1] == 1

    mask = ~(np.isnan(X) | np.isnan(Y))
    X = X[mask]
    Y = Y[mask]

    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    xmin = xmin if xlims[0] is None else xlims[0]
    xmax = xmax if xlims[1] is None else xlims[1]
    ymin = ymin if ylims[0] is None else ylims[0]
    ymax = ymax if ylims[1] is None else ylims[1]
    xmargins = margins * (xmax - xmin)
    ymargins = margins * (ymax - ymin)
    xmin -= xmargins
    xmax += xmargins
    ymin -= ymargins
    ymax += ymargins

    nbins_x = max(1, int(res * (xmax - xmin)))
    nbins_y = max(1, int(res * (ymax - ymin)))

    if noise_smooth > 0:
        xres_val = (xmax - xmin) / nbins_x
        yres_val = (ymax - ymin) / nbins_y
        X = X + np.random.normal(size=X.shape) * noise_smooth * xres_val
        Y = Y + np.random.normal(size=Y.shape) * noise_smooth * yres_val

    h, xedges, yedges = np.histogram2d(
        X.ravel(),
        Y.ravel(),
        bins=[nbins_x, nbins_y],
        range=[[xmin, xmax], [ymin, ymax]],
        density=False,
    )
    h = np.ma.masked_where(h == 0, h)

    density_rescaler = _IdentityScaler() if not use_log_density else _LogPlusOneScaler()
    h = density_rescaler.fwd(h)

    if cmap is None:
        cmap = DEFAULT_DENSITY_CMAP

    if rescaler is not None:
        setup_transformed_axis(
            ax,
            xaxis_lims=[xmin, xmax],
            yaxis_lims=[ymin, ymax],
            rescaler=rescaler,
            margins=0.0,
        )

    im = ax.imshow(
        h.T,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vlims[0],
        vmax=vlims[1],
        interpolation="nearest",
    )
    ax.set_clip_path(ax.patch)

    if draw_xlabel:
        ax.set_xlabel(xtitle if xtitle is not None else (input_names[0] if input_names else ""))
    if draw_ylabel:
        ax.set_ylabel(ytitle if ytitle is not None else output_name)
    if title is not None:
        ax.set_title(title)

    ax.grid(color="k", alpha=0.25, linestyle="-", linewidth=0.2, which="major")
    ax.grid(color="k", alpha=0.1, linestyle="-", linewidth=0.1, which="minor")

    cbar = None
    if draw_colorbar:
        cbar = colorbar(ax, im, density_rescaler, vlims, **{**colorbar_params, "label": "Density"})
    return im, cbar
