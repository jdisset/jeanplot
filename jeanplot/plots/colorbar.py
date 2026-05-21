"""Colorbar drawing.

Verbatim copy from biocomp `plotting_smooth_2d.py` `colorbar()` (lines 267-429)
with biocomp imports replaced and `@configurable` dropped.
"""

from typing import Literal, TypeAlias, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from jeanplot.plots.ticks import setup_transformed_axis

T = TypeVar("T")
ListOrSingle: TypeAlias = list[T] | T


def colorbar(
    ax,
    im,
    rescaler,
    vlims=(None, None),
    yslice=None,
    label=None,
    position=(1.1, 0.4),
    size=(0.04, 0.52),
    orientation: Literal["horizontal", "vertical"] = "vertical",
    label_position: Literal["left", "right", "bottom", "top"] = "right",
    tick_position: Literal["left", "right", "bottom", "top"] | None = "right",
    label_props: dict | None = None,
    tick_props: ListOrSingle[dict] | None = None,
    border_width=0.7,
    setup_transformed_axis_params: dict | None = None,
    threshold_below=None,
    threshold_above=None,
    alpha_opacity=1.0,
    cax=None,
):
    if setup_transformed_axis_params is None:
        setup_transformed_axis_params = {}
    else:
        setup_transformed_axis_params = dict(setup_transformed_axis_params)
    _active_axis_key = "setup_yaxis_params" if orientation == "vertical" else "setup_xaxis_params"
    _sub = setup_transformed_axis_params.get(_active_axis_key) or {}
    if not isinstance(_sub, dict):
        _sub = {}
    else:
        _sub = dict(_sub)
    _sub["show_labels"] = True
    setup_transformed_axis_params[_active_axis_key] = _sub
    if label_props is None:
        label_props = {}
    imlims = im.get_clim()
    c_vmin = imlims[0] if vlims[0] is None else vlims[0]
    c_vmax = imlims[1] if vlims[1] is None else vlims[1]

    colorbar_ax = (
        cax if cax is not None else ax.inset_axes([position[0], position[1], size[0], size[1]])
    )

    if threshold_below is not None or threshold_above is not None:
        from matplotlib.colors import ListedColormap

        cmap = im.get_cmap()
        colors = np.array(cmap(np.linspace(0, 1, 256)))
        values = np.linspace(c_vmin, c_vmax, len(colors))

        alpha_mask = np.ones(len(colors)) * alpha_opacity
        if threshold_below is not None:
            alpha_mask = np.where(values < threshold_below, 0, alpha_mask)
        if threshold_above is not None:
            alpha_mask = np.where(values > threshold_above, 0, alpha_mask)

        colors = np.column_stack([colors[:, :3], alpha_mask])
        threshold_cmap = ListedColormap(colors)

        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=c_vmin, vmax=c_vmax), cmap=threshold_cmap
            ),
            cax=colorbar_ax,
            orientation=orientation,
            aspect=20,
        )
    else:
        cbar = plt.colorbar(im, cax=colorbar_ax, orientation=orientation, aspect=20)

    if tick_position is None:
        tick_position = label_position

    if orientation == "vertical":
        if tick_position == "right":
            colorbar_ax.yaxis.set_ticks_position("right")
            colorbar_ax.tick_params(left=False)
        else:
            colorbar_ax.yaxis.set_ticks_position("left")
            colorbar_ax.tick_params(right=False)
    else:
        if tick_position == "top":
            colorbar_ax.xaxis.set_ticks_position("top")
            colorbar_ax.tick_params(bottom=False)
        else:
            colorbar_ax.xaxis.set_ticks_position("bottom")
            colorbar_ax.tick_params(top=False)

    DEFAULT_TICK_PROPS = {
        "axis": "y" if orientation == "vertical" else "x",
        "which": "both",
        "direction": "out",
        "pad": 2,
        "labelsize": 8,
        "width": 0.7,
    }
    cbar.ax.tick_params(**DEFAULT_TICK_PROPS)

    if tick_props is not None:
        if not isinstance(tick_props, list):
            tick_props = [tick_props]
        for tick_prop in tick_props:
            cbar.ax.tick_params(**tick_prop)

    for spine in cbar.ax.spines.values():
        spine.set_linewidth(border_width)

    setup_transformed_axis_params_with_spine = {
        "spine_position": tick_position,
        "force_spine_only": True,
        **setup_transformed_axis_params,
    }

    if orientation == "vertical":
        setup_transformed_axis(
            cbar.ax,
            yaxis_lims=[c_vmin, c_vmax],
            xaxis_lims=None,
            rescaler=rescaler,
            **setup_transformed_axis_params_with_spine,
        )

        if label_position not in ["left", "right"]:
            raise ValueError("Vertical orientation: label_position must be left or right")
        if tick_position not in ["left", "right"]:
            raise ValueError("Vertical orientation: tick_position must be left or right")

        cbar.ax.yaxis.set_label_position(label_position)

        if label is not None:
            cbar.ax.set_ylabel(label, **label_props)

        cbar.ax.tick_params(axis="x", which="both", size=0)
        cbar.ax.set_xticks([])
    else:
        setup_transformed_axis(
            cbar.ax,
            xaxis_lims=[c_vmin, c_vmax],
            yaxis_lims=None,
            rescaler=rescaler,
            **setup_transformed_axis_params_with_spine,
        )

        if label_position not in ["bottom", "top"]:
            raise ValueError("Horizontal orientation: label_position must be bottom or top")
        if tick_position not in ["bottom", "top"]:
            raise ValueError("Horizontal orientation: tick_position must be bottom or top")

        cbar.ax.xaxis.set_label_position(label_position)

        if label is not None:
            cbar.ax.set_xlabel(label, **label_props)

        cbar.ax.tick_params(axis="y", which="both", size=0)
        cbar.ax.set_yticks([])

    return cbar
