"""Cabinet-projection cube view for 3D smooth panels.

Minimal cube-wireframe drawer. The full feature-set (slice insets, z-axis ticks,
labels) is being progressively ported from biocomp `plotting_3d.py`. For now this
draws just the wireframe so `SmoothPanel3D` can compose cube + slice grid.
"""

import numpy as np

PROJ_ALPHA = 45.0
PROJ_D = 0.5


def cabinet_project(pos, alpha: float = PROJ_ALPHA, d: float = PROJ_D):
    a = np.deg2rad(alpha)
    x, y, z = pos
    return np.array([x + d * z * np.cos(a), y + d * z * np.sin(a)])


def draw_cube_wireframe(
    ax,
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
    def proj(p):
        return cabinet_project(p, projection_angle, projection_diag_coef)

    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim

    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    hidden = {(0, 1), (1, 2), (0, 4)}

    for a, b in edges:
        pa, pb = proj(corners[a]), proj(corners[b])
        kwargs = {"color": edge_color, "linewidth": edge_lw}
        if (a, b) in hidden:
            kwargs["alpha"] = hidden_alpha
            kwargs["dashes"] = list(hidden_dashes)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], **kwargs)

    projected = np.array([proj(c) for c in corners])
    pad_x = 0.05 * (projected[:, 0].max() - projected[:, 0].min())
    pad_y = 0.05 * (projected[:, 1].max() - projected[:, 1].min())
    ax.set_xlim(projected[:, 0].min() - pad_x, projected[:, 0].max() + pad_x)
    ax.set_ylim(projected[:, 1].min() - pad_y, projected[:, 1].max() + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    if xtitle:
        mid = proj(((x0 + x1) / 2, y0, z0))
        ax.text(mid[0], mid[1] - 0.08, xtitle, ha="center", va="top", fontsize=8)
    if ytitle:
        mid = proj((x0, (y0 + y1) / 2, z0))
        ax.text(mid[0] - 0.08, mid[1], ytitle, ha="right", va="center", fontsize=8, rotation=90)
    if ztitle:
        mid = proj((x1, y0, (z0 + z1) / 2))
        ax.text(
            mid[0] + 0.05,
            mid[1],
            ztitle,
            ha="left",
            va="center",
            fontsize=8,
            rotation=projection_angle,
        )
