"""Axis tick formatting and transformed-axis setup.

Verbatim copy from biocomp `plotting_core.py` (lines 189-340) with biocomp
imports replaced by jeanplot equivalents. `@configurable` markers dropped.
"""

import string
from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from jeanplot.data import Rescaler


def powers_of_ten(xmin, xmax, skip_ticklabel_range=None, resolution=1, **_):
    bounds = np.array([xmin, xmax])
    logbounds = np.sign(bounds) * np.floor(
        np.maximum(np.log10(np.maximum(np.abs(bounds), 0.1)), 0)
    ).astype(int)
    if logbounds[0] == logbounds[1]:
        logbounds[1] += 1

    try:
        powers = np.arange(logbounds[0], logbounds[1] + 1)
    except ValueError:
        powers = np.arange(1)

    if skip_ticklabel_range is not None:
        skip_power_low = np.floor(np.log10(max(skip_ticklabel_range[0], 0.1))).astype(int)
        skip_power_high = np.ceil(np.log10(skip_ticklabel_range[1])).astype(int)
        powers = np.delete(
            powers,
            np.where((np.abs(powers) >= skip_power_low) & (np.abs(powers) <= skip_power_high)),
        )

    base_powers = np.power(10, powers)

    if resolution > 1:
        increments = np.arange(2, resolution).reshape(-1, 1)
    else:
        increments = np.array([[1]])

    values = (base_powers * increments).flatten()
    values = values[(values >= xmin) & (values <= xmax)]
    return values


def format_powers(x, *_, n_decimals=1):
    x = float(x)
    abs_x = abs(x)
    if abs_x < 1000:
        if np.abs(x - int(x)) < 1e-3:
            return rf"${int(x)}$"
        return rf"${x:.1f}$"
    sign = "-" if x < 0 else ""
    E = int(np.floor(np.log10(abs_x)))
    mantissa = round(abs_x / 10**E, n_decimals)
    if mantissa >= 10:
        mantissa /= 10
        E += 1
    if abs(mantissa - round(mantissa)) < 10 ** (-n_decimals - 1):
        return rf"${sign}{mantissa:.0f}e{E}$"
    return r"${0}{1:.{3}f}e{2}$".format(sign, mantissa, E, n_decimals)


class PowerFormatter(ticker.Formatter):
    def __init__(self, values, skip_ticklabel_range=None, **_):
        self.values = values
        self.skip_ticklabel_range = skip_ticklabel_range

    def __call__(self, x, pos):
        v = self.values[pos]
        if (
            self.skip_ticklabel_range is not None
            and np.abs(v) < self.skip_ticklabel_range[1]
            and np.abs(v) > self.skip_ticklabel_range[0]
        ):
            return ""
        return format_powers(v, None)


def get_transformed_ticks_and_labels(axis_lims: Sequence[float], rescaler: Rescaler, **kw):
    lims_tr = np.asarray(axis_lims)
    lims_inv = rescaler.inv(np.asarray(lims_tr))
    assert isinstance(lims_inv, np.ndarray)
    assert lims_inv.shape == (2,)
    p10 = powers_of_ten(xmin=lims_inv[0], xmax=lims_inv[1])
    p10_minor = powers_of_ten(xmin=lims_inv[0], xmax=lims_inv[1], resolution=10)
    ticks = {"major": rescaler.fwd(p10), "minor": rescaler.fwd(p10_minor)}
    pf = PowerFormatter(p10, **kw)
    labels = [(rescaler.fwd(x), pf(x, i)) for i, x in enumerate(p10)]
    return ticks, labels


def _install_overlap_skip(ax, axis: str, min_gap_px: float = 2.0):
    """Hide tick labels that overlap with their next-higher-value neighbor."""
    state = {"done": False}

    def cb(event):
        if state["done"]:
            return
        labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
        labels = [tl for tl in labels if tl.get_visible() and tl.get_text().strip()]
        if len(labels) < 2:
            state["done"] = True
            return
        try:
            renderer = event.renderer
            keep_bb = labels[-1].get_window_extent(renderer)
            for label in reversed(labels[:-1]):
                bb = label.get_window_extent(renderer)
                if axis == "x":
                    overlaps = bb.x1 > keep_bb.x0 - min_gap_px
                else:
                    overlaps = bb.y1 > keep_bb.y0 - min_gap_px
                if overlaps:
                    label.set_visible(False)
                else:
                    keep_bb = bb
            state["done"] = True
        except Exception:
            pass

    ax.figure.canvas.mpl_connect("draw_event", cb)


def setup_transformed_axis_generic(
    ax,
    axis_lims,
    rescaler,
    axis="x",
    margins=0.0,
    show_minor_labels=False,
    major_tick_length=None,
    major_tick_width=None,
    minor_tick_length=None,
    minor_tick_width=None,
    label_fontsize=None,
    show_labels=True,
    spine_position=None,
    force_spine_only=False,
    auto_skip_overlap: bool = True,
    **kw,
):
    axis_obj = getattr(ax, f"{axis}axis")
    set_lim = getattr(ax, f"set_{axis}lim")
    set_ticks = getattr(ax, f"set_{axis}ticks")
    rc_prefix = f"{axis}tick"

    if spine_position is None:
        spine_position = "bottom" if axis == "x" else "left"

    if force_spine_only:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines[spine_position].set_visible(True)
        if axis == "x":
            ax.xaxis.set_ticks_position(spine_position)
            ax.xaxis.set_label_position(spine_position)
        else:
            ax.yaxis.set_ticks_position(spine_position)
            ax.yaxis.set_label_position(spine_position)

    lims_tr = np.asarray(axis_lims, dtype=float)
    if not np.all(np.isfinite(lims_tr)):
        return
    lims_inv = rescaler.inv(np.asarray(lims_tr))
    p10 = powers_of_ten(xmin=lims_inv[0], xmax=lims_inv[1])
    lims_margin = lims_tr + np.array([-1, 1]) * margins * np.diff(lims_tr)

    try:
        set_lim(lims_margin)
        set_ticks(rescaler.fwd(p10))
        axis_obj.set_major_formatter(PowerFormatter(p10, **kw))

        p10_minor = powers_of_ten(xmin=lims_inv[0], xmax=lims_inv[1], resolution=10)
        set_ticks(rescaler.fwd(p10_minor), minor=True)
        if show_minor_labels:
            axis_obj.set_minor_formatter(PowerFormatter(p10_minor, **kw))

        if force_spine_only:
            tick_params_dict = {
                spine_position: True,
                f"label{spine_position}": True,
                "which": "both",
            }
            other_positions = {"top", "bottom", "left", "right"} - {spine_position}
            for pos in other_positions:
                tick_params_dict[pos] = False
                tick_params_dict[f"label{pos}"] = False
                ax.spines[pos].set_visible(True)
            ax.tick_params(axis=axis, **tick_params_dict)
        else:
            spine_name = "bottom" if axis == "x" else "left"
            tick_params_dict = {
                spine_name: plt.rcParams[f"{rc_prefix}.{spine_name}"],
                f"label{spine_name}": plt.rcParams[f"{rc_prefix}.label{spine_name}"],
                "which": "both",
            }
            ax.tick_params(axis=axis, **tick_params_dict)

        if major_tick_length is not None or major_tick_width is not None:
            ax.tick_params(
                axis=axis,
                which="major",
                length=major_tick_length
                if major_tick_length is not None
                else plt.rcParams[f"{rc_prefix}.major.size"],
                width=major_tick_width
                if major_tick_width is not None
                else plt.rcParams[f"{rc_prefix}.major.width"],
            )

        if minor_tick_length is not None or minor_tick_width is not None:
            ax.tick_params(
                axis=axis,
                which="minor",
                length=minor_tick_length
                if minor_tick_length is not None
                else plt.rcParams[f"{rc_prefix}.minor.size"],
                width=minor_tick_width
                if minor_tick_width is not None
                else plt.rcParams[f"{rc_prefix}.minor.width"],
            )

        if label_fontsize is not None:
            ax.tick_params(axis=axis, labelsize=label_fontsize)

        if not show_labels:
            sides = ("bottom", "top") if axis == "x" else ("left", "right")
            ax.tick_params(
                axis=axis,
                which="both",
                **{f"label{s}": False for s in sides},
            )
            if axis == "x":
                ax.set_xticklabels([])
                ax.set_xticklabels([], minor=True)
            else:
                ax.set_yticklabels([])
                ax.set_yticklabels([], minor=True)
        elif auto_skip_overlap:
            _install_overlap_skip(ax, axis)

    except ValueError:
        pass

    return lims_inv


def setup_xaxis(ax, xaxis_lims, rescaler, **kw):
    return setup_transformed_axis_generic(ax, xaxis_lims, rescaler, axis="x", **kw)


def setup_yaxis(ax, yaxis_lims, rescaler, **kw):
    return setup_transformed_axis_generic(ax, yaxis_lims, rescaler, axis="y", **kw)


def setup_transformed_axis(
    ax,
    xaxis_lims=None,
    yaxis_lims=None,
    rescaler=None,
    x_rescaler=None,
    y_rescaler=None,
    setup_xaxis_params=None,
    setup_yaxis_params=None,
    **kw,
):
    if setup_yaxis_params is None:
        setup_yaxis_params = {}
    if setup_xaxis_params is None:
        setup_xaxis_params = {}
    xr = x_rescaler if x_rescaler is not None else rescaler
    yr = y_rescaler if y_rescaler is not None else rescaler
    if xaxis_lims is not None:
        xaxis_lims = setup_xaxis(ax, xaxis_lims, xr, **setup_xaxis_params, **kw)
    if yaxis_lims is not None:
        yaxis_lims = setup_yaxis(ax, yaxis_lims, yr, **setup_yaxis_params, **kw)
    return xaxis_lims, yaxis_lims


def setup_symlog_xaxis(ax, xaxis_lims, transform, margins=0.05, **kw):
    xlims_tr = transform(np.asarray(xaxis_lims))
    xp10 = powers_of_ten(*xaxis_lims)
    xlims_margin = xlims_tr + np.array([-1, 1]) * margins * np.diff(xlims_tr)
    ax.set_xlim(xlims_margin)
    ax.set_xticks(transform(xp10))
    ax.xaxis.set_major_formatter(PowerFormatter(xp10, **kw))


def setup_symlog_yaxis(ax, yaxis_lims, transform, margins=0.05, **kw):
    ylims_tr = transform(np.asarray(yaxis_lims))
    yp10 = powers_of_ten(*yaxis_lims)
    ylims_margin = ylims_tr + np.array([-1, 1]) * margins * np.diff(ylims_tr)
    ax.set_ylim(ylims_margin)
    ax.set_yticks(transform(yp10))
    ax.yaxis.set_major_formatter(PowerFormatter(yp10, **kw))


def setup_symlog_axis(
    ax,
    xaxis_lims=None,
    yaxis_lims=None,
    *,
    transform,
    inv_transform,
    margins=0.05,
    **kw,
):
    if xaxis_lims is not None:
        setup_symlog_xaxis(ax, xaxis_lims, transform, margins=margins, **kw)
    if yaxis_lims is not None:
        setup_symlog_yaxis(ax, yaxis_lims, transform, margins=margins, **kw)
    return transform, inv_transform, None, None


class ShortScientificFormatter(string.Formatter):
    def format_field(self, value, format_spec, precision=1):
        if format_spec == "m":
            if value < 1000:
                if value == int(value):
                    return super().format_field(int(value), "")
                return super().format_field(value, f".{precision}f")
            if value == int(value):
                return super().format_field(value, ".0e").replace("e+0", "e").replace("e+", "e")
            return super().format_field(value, ".1e").replace("e+0", "e").replace("e+", "e")
        return super().format_field(value, format_spec)


scformat = ShortScientificFormatter()
