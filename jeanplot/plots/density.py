"""1D density (violin-style) plot.

Copy from biocomp `plotting_density.py` `density_plot_1d` (lines 47-111) with
biocomp imports dropped.
"""

import numpy as np
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

_RANGE_BANDS = [
    (0.10, 0.90, 0.15, 1.5),
    (0.01, 0.99, 0.10, 1.0),
    (0.001, 0.999, 0.05, 0.5),
]

_RANGE_LABEL_KW = dict(fontsize=7, color="#999999", fontfamily="monospace", clip_on=True)


def _pct_label(q: float) -> str:
    pct = q * 100
    if pct == int(pct):
        return f"{int(pct)}%"
    return f"{pct:g}%"


def _draw_quantile_bands(ax: Axes, x, x2, color: str):
    for q_lo, q_hi, alpha, lw in _RANGE_BANDS:
        for data, xlims in [(x, (-0.5, 0)), (x2, (0, 0.5))]:
            lo_v, hi_v = np.quantile(data, q_lo), np.quantile(data, q_hi)
            x0, x1 = xlims
            ax.fill_betweenx([lo_v, hi_v], x0, x1, color=color, alpha=alpha, lw=0)
            ax.plot([x0, x1], [lo_v, lo_v], color=color, lw=lw, alpha=0.6)
            ax.plot([x0, x1], [hi_v, hi_v], color=color, lw=lw, alpha=0.6)

    tx = -0.48
    lo_vals_left = [
        (np.quantile(x, q_lo), np.quantile(x, q_hi), q_lo, q_hi)
        for q_lo, q_hi, _, _ in _RANGE_BANDS
    ]
    for lo_v, hi_v, q_lo, q_hi in lo_vals_left:
        ax.text(tx, hi_v, _pct_label(q_hi), va="center", ha="left", **_RANGE_LABEL_KW)
        ax.text(tx, lo_v, _pct_label(q_lo), va="center", ha="left", **_RANGE_LABEL_KW)


def density_plot_1d(
    x,
    sample_at,
    ax: Axes,
    color="k",
    label=None,
    ticks=None,
    minor_ticks=None,
    ticks_labels=None,
    bw_method=0.01,
    x2=None,
    show_quantiles=(0.01, 0.99),
    is_first=False,
    **_,
):
    left_kde = gaussian_kde(x.T, bw_method=bw_method)
    left_densities = left_kde(sample_at.T)
    if x2 is not None:
        right_kde = gaussian_kde(x2.T, bw_method=bw_method)
        right_densities = right_kde(sample_at.T)
    else:
        x2 = x
        right_densities = left_densities

    left_densities = (left_densities / left_densities.max()) * 0.4
    right_densities = (right_densities / right_densities.max()) * 0.4

    ax.plot(-left_densities, sample_at, color="k", alpha=1, lw=0.5)
    ax.plot(right_densities, sample_at, color="k", alpha=1, lw=0.5)

    if show_quantiles is not None:
        _draw_quantile_bands(ax, x, x2, color)

    ax.fill_betweenx(sample_at, -left_densities, 0, color=color, alpha=1, lw=0)
    ax.fill_betweenx(sample_at, 0, right_densities, color=color, alpha=1, lw=0)
    ax.axvline(0, color="k", alpha=0.5, lw=0.5, dashes=(10, 10), dash_capstyle="round")
    ax.set_xlim(-0.5, 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if label is not None:
        ax.set_xlabel(label, rotation=0, labelpad=20, fontsize=10)

    _hline_kw = dict(
        c="#777777",
        linewidth=0.2,
        zorder=0,
        clip_on=False,
        alpha=1,
        dash_capstyle="round",
    )
    if is_first and minor_ticks is not None:
        for t in minor_ticks:
            ax.axhline(t, xmin=0.3, xmax=0.7, dashes=(4, 12), **_hline_kw)
    if ticks is not None:
        for t in ticks:
            ax.axhline(t, xmin=-0.2, xmax=1, dashes=(10, 20), **_hline_kw)
        if ticks_labels is not None:
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks_labels)
            ax.tick_params(axis="y", which="both", length=0, pad=30)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(8)
                tick.label1.set_color("grey")
