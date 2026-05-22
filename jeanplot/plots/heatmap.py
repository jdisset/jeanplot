"""Heatmap rendering kernel.

Verbatim copy from biocomp `plotting_core.py` `heatmap()` (lines 812-1078)
with `@configurable` dropped and DEFAULT_CMAP_NAME inlined.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CMAP_NAME = "viridis"


def _smooth_otsu_threshold(values: np.ndarray, bias: float = 0.5) -> float:
    bias = float(np.clip(bias, 0.0, 1.0))
    vals = values[np.isfinite(values)]
    nbins = min(256, max(16, len(vals) // 100))
    counts, bin_edges = np.histogram(vals, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = counts.sum()
    if total == 0:
        return float(np.median(vals))
    w0 = np.cumsum(counts).astype(float)
    w1 = (total - w0).astype(float)
    mu0 = np.cumsum(counts * bin_centers)
    mu0[w0 <= 0] = 0.0
    np.divide(mu0, w0, out=mu0, where=w0 > 0)
    mu1 = np.cumsum((counts * bin_centers)[::-1])[::-1]
    mu1[w1 <= 0] = 0.0
    np.divide(mu1, w1, out=mu1, where=w1 > 0)

    if bias > 0:
        w0_term = w0 ** (2.0 * bias)
    else:
        w0_term = np.where(w0 > 0, 1.0, 0.0)
    if bias < 1:
        w1_term = w1 ** (2.0 * (1.0 - bias))
    else:
        w1_term = np.where(w1 > 0, 1.0, 0.0)

    objective = w0_term * w1_term * (mu0 - mu1) ** 2
    return float(bin_centers[np.argmax(objective)])


def _otsu_threshold(values: np.ndarray) -> float:
    """Vanilla Otsu threshold (alias for ``_smooth_otsu_threshold(values, bias=0.5)``)."""
    return _smooth_otsu_threshold(values, bias=0.5)


def _resolve_symbolic_level(level, finite_values: np.ndarray):
    if not isinstance(level, str) or len(finite_values) == 0:
        return level
    if level == "otsu":
        return _smooth_otsu_threshold(finite_values, bias=0.5)
    if level.startswith("otsu:"):
        try:
            bias = float(level.split(":", 1)[1])
        except ValueError:
            return level
        return _smooth_otsu_threshold(finite_values, bias=bias)
    if level.endswith("%"):
        try:
            pct = float(level.rstrip("%"))
        except ValueError:
            return level
        return float(np.percentile(finite_values, pct))
    return level


def heatmap(
    ax,
    xy_grid,
    output_values,
    vlims=(None, None),
    contours=3,
    contours_alpha=1,
    contours_color="k",
    contours_linewidth=0.5,
    contours_linestyle="solid",
    contours_print=False,
    opacities=None,
    show_image=True,
    axtransform=None,
    cmap=DEFAULT_CMAP_NAME,
    transparent_below=None,
    transparent_above=None,
    image_interpolation=None,
    opacity=1,
    bad_color="#EEEEEE00",
    clip_to_lowest_contour=False,
    flat_fill=None,
):
    if isinstance(ax, list):
        ax = ax[0]

    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color=bad_color)

    full_transform = ax.transData
    if axtransform is not None:
        full_transform = full_transform + axtransform

    xres = len(np.unique(xy_grid[:, 0]))
    yres = len(np.unique(xy_grid[:, 1]))

    xlims = np.array([xy_grid[:, 0].min(), xy_grid[:, 0].max()])
    ylims = np.array([xy_grid[:, 1].min(), xy_grid[:, 1].max()])
    vmin, vmax = vlims
    finite = np.isfinite(output_values)
    if not finite.any():
        vmin = 0.0 if vmin is None else vmin
        vmax = 1.0 if vmax is None else vmax
    else:
        vmin = vmin if vmin is not None else np.nanmin(output_values)
        vmax = vmax if vmax is not None else np.nanmax(output_values)

    Z = output_values.reshape((xres, yres)).T
    opacities = np.ones_like(Z) if opacities is None else opacities.reshape((xres, yres)).T
    opacities *= opacity

    if transparent_below is not None:
        opacities = np.where(Z < transparent_below, 0, opacities)
    if transparent_above is not None:
        opacities = np.where(Z > transparent_above, 0, opacities)

    if np.isnan(Z).all():
        Z = np.zeros_like(Z)

    cntrs = None
    clip_cntrs = None
    if contours is not None:
        Z_contour = Z.copy()
        Z_contour[:, 0] = 0
        Z_contour[:, -1] = 0
        Z_contour[0, :] = 0
        Z_contour[-1, :] = 0

        if isinstance(contours, list | tuple | np.ndarray):
            finite_vals = output_values[np.isfinite(output_values)]
            contours = [_resolve_symbolic_level(c, finite_vals) for c in contours]
            contours = [c for c in contours if not isinstance(c, str)]
            if not contours:
                contours = None

        cntrs = ax.contour(
            Z_contour.T,
            levels=contours if isinstance(contours, list | np.ndarray) else contours,
            linewidths=contours_linewidth,
            linestyles=contours_linestyle,
            extent=[*xlims, *ylims],
            alpha=contours_alpha,
            colors=contours_color,
        )

        if clip_to_lowest_contour:
            Z_contour = np.nan_to_num(Z_contour)
            if hasattr(cntrs, "levels") and len(cntrs.levels) > 0:
                lowest_level = cntrs.levels[0]
            else:
                lowest_level = cntrs.levels
            clip_cntrs = ax.contour(
                Z_contour.T,
                levels=[lowest_level],
                extent=[*xlims, *ylims],
                alpha=0,
                colors="none",
            )

            nan_mask = np.isnan(Z)
            if np.any(nan_mask):
                ax.contour(
                    Z_contour.T,
                    levels=cntrs.levels
                    if isinstance(cntrs.levels, list | np.ndarray)
                    else [cntrs.levels],
                    extent=[*xlims, *ylims],
                    alpha=0.4,
                    linewidths=contours_linewidth * 0.95,
                    linestyles=[(0, (1, 3))],
                    colors=contours_color,
                )

        if contours_print:
            ax.clabel(cntrs, inline=True, fontsize=8)

    im = None
    if show_image:
        if clip_to_lowest_contour and cntrs is not None:
            Z = np.nan_to_num(Z)

        if flat_fill is not None and clip_to_lowest_contour and clip_cntrs is not None:
            lowest_contour_path = clip_cntrs.get_paths()[0]
            if len(lowest_contour_path.vertices) > 0:
                im = mpl.patches.PathPatch(
                    lowest_contour_path,
                    transform=ax.transData,
                    facecolor=flat_fill,
                    edgecolor="none",
                    alpha=opacity,
                )
                ax.add_patch(im)
        else:
            im = ax.imshow(
                Z.T,
                origin="lower",
                aspect=1,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation=image_interpolation,
                alpha=opacities.T,
                extent=[*xlims, *ylims],
            )

            if clip_to_lowest_contour and clip_cntrs is not None:
                lowest_contour_path = clip_cntrs.get_paths()[0]
                clip_path = mpl.patches.PathPatch(lowest_contour_path, transform=ax.transData)
                im.set_clip_path(clip_path)
                if len(lowest_contour_path.vertices) == 0:
                    im.remove()

    return im, cntrs


def make_xy_grid(xmin, xmax, ymin=None, ymax=None, xres=100, yres=None):
    ymin = ymin if ymin is not None else xmin
    ymax = ymax if ymax is not None else xmax
    yres = yres if yres is not None else xres
    xx = np.linspace(xmin, xmax, xres)
    yy = np.linspace(ymin, ymax, yres)
    X, Y = np.meshgrid(xx, yy)
    return np.vstack([X.ravel(), Y.ravel()]).T
