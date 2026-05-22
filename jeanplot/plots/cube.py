"""Cabinet-projected 3D cube primitives.

Faces, z-axis lines, projected ticks/labels/title, slice-stack orchestrator.
The full machinery used by `jeanplot.plots.smooth_3d`.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

from jeanplot.plots.ticks import setup_transformed_axis

V3d = tuple[float, float, float] | np.ndarray
V2d = tuple[float, float] | np.ndarray
NdArray = np.ndarray
NumLike = int | float | np.number

PROJ_ALPHA = 45.0
PROJ_D = 0.5

_MAX_Z = np.iinfo(np.int32).max

CUBE_SPINE_PROPS = dict(linewidth=0.5, color="#000000", linestyle="-")
CUBE_SPINE_PROPS_HIDDEN = dict(CUBE_SPINE_PROPS, linestyle=":", alpha=0.5)

DEFAULT_SLICE_TICKS_PROPS = [
    {
        "length": 54,
        "direction": (1, 0, 0),
        "props": dict(color="k", linewidth=0.2, dashes=[5, 5], alpha=0.5),
    }
]
DEFAULT_MAJOR_TICKS_PROPS = [
    {"length": 8, "direction": (1, 0, 0), "props": dict(color="k", linewidth=0.4)},
]
DEFAULT_MINOR_TICKS_PROPS = [
    {"length": 2, "direction": (1, 0, 0), "props": dict(color="k", linewidth=0.2)},
]

DEFAULT_LABEL_PROPS = dict(
    ha="left",
    va="center",
    fontsize=8,
    bbox=dict(facecolor="white", alpha=1, edgecolor="none", pad=-0.25),
)
DEFAULT_SLICE_LABEL_PROPS = dict(ha="left", va="center", fontsize=7)
DEFAULT_TITLE_PROPS = dict(ha="center", va="center", fontsize=8, rotation=PROJ_ALPHA)

CUBE_EDGE_PROPS_VISIBLE = {
    "props": {**CUBE_SPINE_PROPS, "zorder": _MAX_Z - 10},
    "offset": (0.0, 0.0),
}
CUBE_EDGE_PROPS_HIDDEN = {
    "props": {**CUBE_SPINE_PROPS_HIDDEN, "zorder": -_MAX_Z + 10},
    "offset": (0.0, 0.0),
}

DEFAULT_CUBE_EDGE_PROPS = {
    "bottom_right": {
        **CUBE_EDGE_PROPS_VISIBLE,
        "ticks": {
            "major": DEFAULT_MAJOR_TICKS_PROPS,
            "minor": DEFAULT_MINOR_TICKS_PROPS,
            "slice": DEFAULT_SLICE_TICKS_PROPS,
        },
        "labels": {
            "major": {"offset": (10, 0), "props": DEFAULT_LABEL_PROPS},
            "slice": {"offset": (55, 0), "props": DEFAULT_SLICE_LABEL_PROPS},
        },
        "zaxis_title": {"offset": (40, 0), "props": DEFAULT_TITLE_PROPS},
    },
    "bottom_left": CUBE_EDGE_PROPS_HIDDEN,
    "top_left": CUBE_EDGE_PROPS_VISIBLE,
    "top_right": CUBE_EDGE_PROPS_VISIBLE,
}


def cabinet_project(pos: V3d, alpha: float = PROJ_ALPHA, d: float = PROJ_D) -> V2d:
    a = np.deg2rad(alpha)
    x, y, z = pos
    return np.array([x + d * z * np.cos(a), y + d * z * np.sin(a)])


def to_display_units(x_data: float, ax: Axes) -> float:
    ppd = 72.0 / ax.figure.dpi
    trans = ax.transData.transform
    return ((trans((1, x_data)) - trans((0, 0))) * ppd)[1]


def to_data_units(y_display: float, ax: Axes) -> float:
    ppd = 72.0 / ax.figure.dpi
    inv = ax.transData.inverted().transform
    origin = inv((0, 0))
    return inv((0, y_display / ppd))[1] - origin[1]


def to_ax_coords(pos: V2d, ax_lims, ax_size):
    x, y = pos
    return (x - ax_lims[0, 0]) / ax_size[0], (y - ax_lims[1, 0]) / ax_size[1]


def get_edge_pos(edge: str, xlim: V2d, ylim: V2d) -> tuple[float, float]:
    v, h = edge.split("_")
    return (xlim[0] if h == "left" else xlim[1], ylim[0] if v == "bottom" else ylim[1])


def main_ax_lims(ax: Axes, xlim, ylim, set_lims: bool = True):
    all_xlims = [ax.get_xlim(), xlim]
    all_ylims = [ax.get_ylim(), ylim]
    ax_lims = np.array(
        [[np.min(all_xlims), np.max(all_xlims)], [np.min(all_ylims), np.max(all_ylims)]]
    )
    ax_lims += np.array([[-1, 1], [-1, 1]]) * 0.05 * np.abs(ax_lims[:, 1] - ax_lims[:, 0])
    ax_size = np.abs(ax_lims[:, 1] - ax_lims[:, 0])
    if set_lims:
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])
    return ax_lims, ax_size


def get_axis_offsets(cube_edge_props: dict[str, Any], xlim: V2d, ylim: V2d):
    out = {}
    span = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0]])
    for edge, props in cube_edge_props.items():
        offset = props.pop("offset", None) if "offset" in props else None
        out[edge] = np.array((0.0, 0.0)) if offset is None else np.array(offset) * span
    return out


def cube_face(
    ax: Axes,
    xlims,
    ylims,
    facecolor: str = "none",
    visible_spines: Sequence[str] = ("bottom", "left"),
    hidden_spines: Sequence[str] = ("top", "right"),
):
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.patch.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for name in visible_spines or ():
        ax.spines[name].set_visible(True)
        ax.spines[name].set(**CUBE_SPINE_PROPS)
    for name in hidden_spines or ():
        ax.spines[name].set_visible(True)
        ax.spines[name].set(**CUBE_SPINE_PROPS_HIDDEN)


def front_face_bl(
    ax: Axes,
    xlims,
    ylims,
    input_names: Sequence[str],
    rescaler,
    labelpad: tuple[int, int] = (20, 24),
    ticks: bool = False,
    xtitle: str | None = None,
    ytitle: str | None = None,
):
    cube_face(ax, xlims, ylims, visible_spines=["bottom", "left"], hidden_spines=[])
    ax.set_xlabel(xtitle if xtitle is not None else input_names[0])
    ax.set_ylabel(ytitle if ytitle is not None else input_names[1])
    ax.set_zorder(-_MAX_Z)
    if ticks:
        setup_transformed_axis(ax, xaxis_lims=xlims, yaxis_lims=ylims, rescaler=rescaler, margins=0.0)
    ax.xaxis.labelpad = labelpad[0]
    ax.yaxis.labelpad = labelpad[1]


def front_face_tr(ax: Axes, xlims, ylims):
    cube_face(ax, xlims, ylims, visible_spines=["top", "right"], hidden_spines=[])
    ax.set_zorder(_MAX_Z - 10)


def back_face(ax: Axes, xlims, ylims):
    cube_face(ax, xlims, ylims, visible_spines=["top", "right"], hidden_spines=["bottom", "left"])


def draw_tick(
    ax: Axes,
    position: V3d,
    direction: V3d,
    length: float,
    props: dict[str, Any],
    project: Callable[[V3d], V2d],
):
    position = np.asarray(position)
    direction = np.asarray(direction)
    length_data = to_data_units(length, ax)
    p0 = project(position)
    p1 = project(position + direction * length_data)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], **props)


def draw_text(
    ax: Axes,
    position: V3d,
    label: str,
    props: dict[str, Any],
    project: Callable[[V3d], V2d],
    offset: V2d = (0, 0),
    offset_units: str = "axes",
):
    position = np.asarray(position)
    off = np.asarray(offset, dtype=float)
    if offset_units == "axes":
        off = np.array([to_data_units(off[0], ax), to_data_units(off[1], ax)])
    p = project(position)
    t = ax.text(p[0] + off[0], p[1] + off[1], label, **props)
    if "bbox" in props:
        t.set_bbox(props["bbox"])


def draw_z_axis(
    ax: Axes,
    xpos: float,
    ypos: float,
    zlim: V2d,
    axis_offset: V2d,
    project: Callable[[V3d], V2d],
    props=CUBE_SPINE_PROPS,
    **_,
):
    xo, yo = axis_offset
    coords_w = np.array([[xpos + xo, xpos + xo], [ypos + yo, ypos + yo], zlim])
    coords_p = np.array((project(coords_w[:, 0]), project(coords_w[:, 1])))
    ax.plot(coords_p[:, 0], coords_p[:, 1], **props)


def draw_z_ticks_along_axis(
    ax: Axes,
    xpos: float,
    ypos: float,
    project: Callable[[V3d], V2d],
    axis_offset: V2d | None = None,
    ticks: NdArray | None = None,
    tick_props: list | dict[str, Any] | None = None,
    **_,
):
    xpos, ypos = np.array([xpos, ypos]) + axis_offset
    if ticks is None or tick_props is None:
        return
    if isinstance(tick_props, dict):
        tick_props = [tick_props]
    for tick in ticks:
        for tp in tick_props:
            draw_tick(ax, (xpos, ypos, tick), project=project, **tp)


def draw_z_labels(
    ax: Axes,
    xpos: float,
    ypos: float,
    labels: list[tuple[NumLike, str, str]],
    axis_offset: V2d,
    project: Callable[[V3d], V2d],
    **props,
):
    xpos, ypos = np.array([xpos, ypos]) + axis_offset
    for z, label, ltype in labels:
        if props.get(ltype) is not None:
            draw_text(
                ax,
                position=np.array([xpos, ypos, float(z)]),
                project=project,
                label=label,
                **props[ltype],
            )


def draw_z_title(
    ax: Axes,
    xpos: float,
    ypos: float,
    zlim: V2d,
    project: Callable[[V3d], V2d],
    zaxis_labelpad: int = 0,
    axis_offset: V2d = (0, 0),
    z_title: str | None = None,
    **title_props,
):
    if z_title is None or not title_props:
        return
    labelpad = to_data_units(zaxis_labelpad, ax)
    xpos, ypos = np.array([xpos, ypos]) + np.asarray(axis_offset) + np.array([labelpad, 0])
    draw_text(
        ax,
        position=np.array([xpos, ypos, float(np.mean(zlim))]),
        label=z_title,
        project=project,
        **title_props,
    )


class InsetPositionLocator:
    def __init__(self, parent: Axes, rect: Sequence[float]):
        self.parent = parent
        self.rect = rect

    def __call__(self, ax, renderer):
        bb = self.parent.get_position(original=False)
        x, y, w, h = self.rect
        return Bbox.from_bounds(
            bb.x0 + bb.width * x, bb.y0 + bb.height * y, bb.width * w, bb.height * h
        )


def plot_3d_stack(
    ax: Axes,
    slice_functions: list,
    slice_zpositions: list,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    zticks: dict[str, NdArray],
    project: Callable[[V3d], V2d],
    zlabels: list[tuple[NumLike, str, str]] | None = None,
    cube_edge_props: dict[str, Any] | None = None,
    z_title: str | None = None,
    zaxis_labelpad: int = 0,
    **_,
):
    cube_edge_props = cube_edge_props if cube_edge_props is not None else DEFAULT_CUBE_EDGE_PROPS
    if zlabels is None and zticks is not None and "major" in zticks:
        zlabels = [(z, str(z), "major") for z in zticks["major"]]

    offsets = get_axis_offsets(cube_edge_props, xlim, ylim)

    for edge, props in cube_edge_props.items():
        draw_z_axis(
            ax,
            *get_edge_pos(edge, xlim, ylim),
            zlim,
            project=project,
            axis_offset=offsets[edge],
            **props,
        )

    ax_lims, ax_size = main_ax_lims(ax, xlim, ylim)

    for edge, props in cube_edge_props.items():
        pos = get_edge_pos(edge, xlim, ylim)
        if "ticks" in props:
            for tick_type, ticks in zticks.items():
                if tick_type in props["ticks"]:
                    draw_z_ticks_along_axis(
                        ax,
                        *pos,
                        project=project,
                        axis_offset=offsets[edge],
                        ticks=ticks,
                        tick_props=props["ticks"][tick_type],
                    )
        if "zaxis_title" in props:
            draw_z_title(
                ax,
                *pos,
                zlim,
                project=project,
                z_title=z_title,
                zaxis_labelpad=zaxis_labelpad,
                axis_offset=offsets[edge],
                **props["zaxis_title"],
            )
        if props.get("labels") is not None and zlabels is not None:
            draw_z_labels(
                ax,
                *pos,
                project=project,
                labels=zlabels,
                axis_offset=offsets[edge],
                **props["labels"],
            )

    for f, z in zip(slice_functions, slice_zpositions, strict=False):
        axin = ax.inset_axes([0, 0, 1, 1], zorder=-z)
        f(axin)
        inset_world = np.array([axin.get_xlim(), axin.get_ylim()])
        inset_size_world = np.abs(inset_world[:, 1] - inset_world[:, 0])
        inset_size_ax = inset_size_world / ax_size
        inset_proj = project((inset_world[0, 0], inset_world[1, 0], z))
        inset_ax = to_ax_coords(inset_proj, ax_lims, ax_size)
        axin.set_axes_locator(
            InsetPositionLocator(ax, [inset_ax[0], inset_ax[1], inset_size_ax[0], inset_size_ax[1]])
        )


def draw_cube_wireframe(
    ax: Axes,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    zlim=(0.0, 1.0),
    projection_angle: float = PROJ_ALPHA,
    projection_diag_coef: float = PROJ_D,
    edge_color: str = "#444444",
    edge_lw: float = 0.5,
    hidden_alpha: float = 0.4,
    hidden_dashes: tuple[float, float] = (3, 3),
    xtitle: str | None = None,
    ytitle: str | None = None,
    ztitle: str | None = None,
):
    project = partial(cabinet_project, alpha=projection_angle, d=projection_diag_coef)
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    hidden = {(0, 1), (1, 2), (0, 4)}
    for a, b in edges:
        pa, pb = project(corners[a]), project(corners[b])
        kwargs = {"color": edge_color, "linewidth": edge_lw}
        if (a, b) in hidden:
            kwargs["alpha"] = hidden_alpha
            kwargs["dashes"] = list(hidden_dashes)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], **kwargs)

    projected = np.array([project(c) for c in corners])
    pad_x = 0.05 * (projected[:, 0].max() - projected[:, 0].min())
    pad_y = 0.05 * (projected[:, 1].max() - projected[:, 1].min())
    ax.set_xlim(projected[:, 0].min() - pad_x, projected[:, 0].max() + pad_x)
    ax.set_ylim(projected[:, 1].min() - pad_y, projected[:, 1].max() + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    if xtitle:
        mid = project(((x0 + x1) / 2, y0, z0))
        ax.text(mid[0], mid[1] - 0.08, xtitle, ha="center", va="top", fontsize=8)
    if ytitle:
        mid = project((x0, (y0 + y1) / 2, z0))
        ax.text(mid[0] - 0.08, mid[1], ytitle, ha="right", va="center", fontsize=8, rotation=90)
    if ztitle:
        mid = project((x1, y0, (z0 + z1) / 2))
        ax.text(
            mid[0] + 0.05, mid[1], ztitle,
            ha="left", va="center", fontsize=8, rotation=projection_angle,
        )
