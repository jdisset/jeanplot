"""SmoothPanel3D: cube view + R×C grid of 2D slices, composed as nested Containers.

Not itself drawable; the renderer hands each child its own Axes from the laid-out bbox.
"""

from typing import Any

import numpy as np
from pydantic import Field

from jeanplot.core.container import Container
from jeanplot.core.models import BoxInset, BoxStyle, LayoutConstraints, Size
from jeanplot.data import PlotData, PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.smooth_2d import SmoothPanel2D
from jeanplot.plots.cube import draw_cube_wireframe


def _format_z_label(z_latent: float, rescaler=None, prefix: str = "z=") -> str:
    if rescaler is None:
        return f"{prefix}{float(z_latent):.2f}"
    from jeanplot.plots.ticks import format_powers

    return f"{prefix}{format_powers(float(rescaler.inv(float(z_latent))), n_decimals=0)}"


def _split_userset(parent: Any, spec: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split {child_kw: parent_field} into (user-set, default) kwargs by whether the
    parent field was explicitly set. User-set lims are passed at construction so they
    stay locked and beat the theme; the rest are applied via `with_defaults` so they're
    cascade-fillable (e.g. `SmoothPanel3D[id=prediction] CubeStackPanel: {vlims: ...}`
    reaches a cube that is built here, before the render-time cascade runs)."""
    user: dict[str, Any] = {}
    default: dict[str, Any] = {}
    for child_kw, pfield in spec.items():
        (user if pfield in parent.model_fields_set else default)[child_kw] = getattr(parent, pfield)
    return user, default


# ── Uniform face sizing (DataBlock tables) ──────────────────────────────────────────────
# SSOT for the chrome reserved around EVERY sub-plot's AXES BOX (2D value face, 1D curve, each
# 3D z-slice). We pin the axes box (the plotted square) to one canonical size for every face, so
# tick labels + titles render at ONE physical size across the whole table; the CELL a face
# occupies then varies with its chrome (a colorbar face is wider than a bare 1D curve) but the
# plotted square is identical. top == TITLE_ROOM so a z-titled slice and an untitled 2D face
# reserve the same top strip. right == 0: a per-cell colorbar adds its own right gutter via
# `_right_overflow` (folded into `face_cell_size`). datablock.py imports all of these.
FACE_PAD = BoxInset(top=0.30, right=0.0, bottom=0.50, left=0.56)
FACE_COLORBAR = {"position": (1.04, 0.08), "size": (0.05, 0.84), "label_reserve": 0.22}
FACE_GAP = 0.1  # inter-face gap (inches), shared by the slice grid and the 1D/2D twin


def _freeze_size(panel: PlotPanel) -> None:
    """Mark axes_size / min / max as user-set so the render-time jstyle cascade (which re-runs the
    base `_compute_min_dimensions` validator via validate_assignment) can NEITHER recompute nor fill
    them. Without this the cascade silently regrows the pinned cells at render, so a block ends up
    drawing wider than the width it reported to the table's auto column -> cross-column overlap."""
    panel._user_set_fields.update(("axes_size", "min_dimensions", "max_dimensions"))


def face_cell_size(box_w: float, box_h: float, colorbar: bool) -> tuple[float, float]:
    """Footprint of a face whose plotted axes box is (box_w, box_h): the box plus FACE_PAD, plus
    the colorbar's right gutter (mirrors `_colorbar_right_overflow`: band overflow + label
    reserve) when present. SSOT so the block/grid width can be derived without a measure pass."""
    cp = FACE_COLORBAR
    overflow = (
        max(0.0, cp["position"][0] + cp["size"][0] - 1.0) * box_w + cp["label_reserve"]
        if colorbar
        else 0.0
    )
    return (
        box_w + FACE_PAD.left + FACE_PAD.right + overflow,
        box_h + FACE_PAD.top + FACE_PAD.bottom,
    )


def pin_axes_box(panel: PlotPanel, box_w: float, box_h: float) -> None:
    """Pin a face's plotted AXES BOX to exactly (box_w, box_h): set axes_size to the box, then
    min==max==box+effective_padding (the cell) so layout gives the panel that exact footprint and
    the axes fills the box. Uniform box across faces => identical tick-label size table-wide,
    whatever each face's chrome. No band-term circularity: effective_padding is read with axes_size
    already at box_w. Called from the PARENT after the child is built (survives re-validation, which
    lands on the same min)."""
    object.__setattr__(panel, "axes_size", Size(width=box_w, height=box_h))
    p = panel.effective_padding
    cell = Size(width=box_w + p.left + p.right, height=box_h + p.top + p.bottom)
    object.__setattr__(panel, "min_dimensions", cell)
    object.__setattr__(panel, "max_dimensions", cell)
    _freeze_size(panel)


def pin_to_cell(panel: PlotPanel, cell_w: float, cell_h: float) -> None:
    """Make `panel` a RIGID cell of exactly (cell_w, cell_h): axes box = cell minus the panel's
    effective padding, min==max==cell so layout can neither stretch nor shrink it. Used for the
    cube (fill an exact footprint) and for the whole block. Called from the PARENT after the
    child is built. Sets axes_size too so the min validator lands on the same cell."""
    object.__setattr__(panel, "axes_size", Size(width=cell_w, height=cell_h))  # seed cbar band term
    p = panel.effective_padding
    aw = max(0.2, cell_w - p.left - p.right)
    ah = max(0.2, cell_h - p.top - p.bottom)
    object.__setattr__(panel, "axes_size", Size(width=aw, height=ah))
    object.__setattr__(panel, "min_dimensions", Size(width=cell_w, height=cell_h))
    object.__setattr__(panel, "max_dimensions", Size(width=cell_w, height=cell_h))
    _freeze_size(panel)


class CubeView(PlotPanel):
    plot_data: PlotData
    xlims: tuple[float, float] = (0.0, 1.0)
    ylims: tuple[float, float] = (0.0, 1.0)
    zlims: tuple[float, float] = (0.0, 1.0)
    projection_angle: float = 45.0
    projection_diag_coef: float = 0.5
    edge_color: str = "#444444"
    edge_lw: float = 0.5

    def draw(self, ax) -> PlotFunctionResult | None:
        draw_cube_wireframe(
            ax,
            xlim=self.xlims,
            ylim=self.ylims,
            zlim=self.zlims,
            projection_angle=self.projection_angle,
            projection_diag_coef=self.projection_diag_coef,
            edge_color=self.edge_color,
            edge_lw=self.edge_lw,
            xtitle=self.xtitle
            or (self.plot_data.input_names[0] if self.plot_data.input_names else None),
            ytitle=self.ytitle
            or (self.plot_data.input_names[1] if len(self.plot_data.input_names) > 1 else None),
            ztitle=self.vtitle
            or (self.plot_data.input_names[2] if len(self.plot_data.input_names) > 2 else None),
        )
        if self.title:
            ax.set_title(self.title, **self.title_kwargs)
        return PlotFunctionResult(rendering=None, metadata={})


class SmoothPanel3D(PlotPanel):
    plot_data: PlotData
    slice_grid: tuple[int, int] = (3, 3)
    slice_zrange: tuple[float, float] = (0.05, 0.55)
    slice_zvalues: list[float] | None = None
    zslices: list[float] | None = None
    stack_zslices: list[float] | None = None
    stack_zrange: tuple[float, float] = (0.05, 0.55)
    stack_n_slices: int = 4
    cube_frac_w: float = 0.45
    cube_grid_gap: float = 0.1  # spacing (inches) between the cube view and the z-slice grid
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    ylims: tuple[float | None, float | None] = (None, None)
    zlims: tuple[float | None, float | None] = (None, None)
    vlims: tuple[float | None, float | None] = (None, None)
    # Slice-grid color scale, independent of the cube's `vlims`. Defaults to
    # (None, None) so every slice auto-scales to its own value range, using
    # `slice_vlim_quantiles` for full per-slice contrast (no floor/range clamp).
    slice_vlims: tuple[float | None, float | None] = (None, None)
    slice_vlim_quantiles: tuple[float | None, float | None] = (0.02, 0.98)
    projection_angle: float = 45.0
    projection_diag_coef: float = 0.5
    slice_show_colorbar: bool = True
    # Draw the x/y axis TITLES (input-protein names) on the bottom-row / left-column slices.
    # Off when the paired cube already carries them (DataBlockPanel), so they aren't repeated.
    slice_show_axis_titles: bool = True
    # Draw x/y tick LABELS on EVERY slice (each cell reserves its own tick band) instead of only
    # the bottom row / left column (shared-axes grid, the band living once in the grid margin).
    # Use for a wide single-row strip where each slice is read independently.
    slice_all_ticks: bool = False
    # UNIFORM FACE MODE (DataBlock tables). When set to the axes-box size (box_w, box_h), every
    # z-slices are pinned so tick labels render identically table-wide (see `pin_axes_box` /
    # `face_cell_size` / `FACE_PAD`). When `uniform_height` = the block height H, the R×C grid is
    # packed into ONE H (each slice = H/R tall, minus gaps): a single-row [1,N] strip gives
    # full-height slices == a 2D value face; a grid shrinks each row to keep the block one row tall.
    # The cube is full H tall, `cube_cell_units`·(a FULL-height face cell) wide. The panel pins
    # ITSELF to the derived (width, H). None = legacy flex layout (cube_frac_w split + stretch-to-
    # fill), kept for standalone `PaperSurface` / `AutoPanel` use.
    uniform_height: float | None = None
    uniform_single_aspect: float = 1.0  # slice/face axes-box width:height (1 = square)
    cube_cell_units: float = 1.4  # uniform mode: cube width in FULL-height face-cell-widths
    slice_title_fontsize: int = 7
    slice_title_color: str = "#777777"
    slice_title_pad: float = 3.0
    cube_show_inner_spines: bool = False
    cube_show_slice_ticks: bool = False
    cube_show_front_face_ticks: bool = True
    # Resolve the cube's symbolic contour levels from this field instead of
    # `plot_data` (e.g. clip a prediction cube at the ground-truth iso-level).
    cube_contour_reference_plot_data: PlotData | None = None
    cube_smooth_2d_params: dict | None = Field(
        default_factory=lambda: {
            "draw_colorbar": False,
            "draw_xlabel": False,
            "draw_ylabel": False,
            "xtitle": "",
            "ytitle": "",
            "vtitle": "",
            "setup_transformed_axis_params": {
                "setup_xaxis_params": {"show_labels": False},
                "setup_yaxis_params": {"show_labels": False},
            },
        }
    )
    cube_colorbar_params: dict | None = None

    is_drawable: bool = False

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.children:
            return

        if self.uniform_height is not None:
            self._build_uniform()
            return

        rows, cols = self.slice_grid
        n_slices = rows * cols

        # Uniform per-cell reservations (inches): title strip on top, colorbar on the
        # right, hairline elsewhere. Kept IDENTICAL on every cell so the equal flex
        # weights below give every slice the same plot area at any panel size.
        # (Size-dependent weights only equalize at one exact size, which is why the
        # bottom row drifted.) Bottom-row x labels / left-column y labels live in the GRID
        # container's margin (grid_pad), NOT per-cell padding, so interior cells don't each
        # waste a tick band. show_labels below decides which cells draw labels.
        gap = 0.02  # minimal gap between slices (they read as one tight grid)
        # The slice colorbar's TRUE right-gutter (band overflow + tick allowance) is what
        # `SmoothPanel2D._right_overflow` reserves at render; under-reserving it here was
        # what made the grid balloon past the panel. So `pad_cbar` MUST cover that gutter
        # when colorbars are on, and collapses to a hairline when they're off.
        pad_title, pad_edge = 0.13, 0.03
        pad_cbar = 0.34 if self.slice_show_colorbar else pad_edge
        pad_xaxis, pad_yaxis = 0.40, 0.44
        # slice_all_ticks: every cell draws its own x/y tick band → reserve it PER-CELL and drop
        # the shared grid margin. Otherwise only the bottom row / left column draw ticks and the
        # band lives once in the grid margin (interior cells stay tight).
        all_ticks = self.slice_all_ticks
        cell_left = pad_yaxis if all_ticks else pad_edge
        cell_bottom = pad_xaxis if all_ticks else pad_edge
        grid_left = pad_edge if all_ticks else pad_yaxis
        grid_bottom = pad_edge if all_ticks else pad_xaxis
        cell_pad = BoxStyle(
            padding=BoxInset(top=pad_title, right=pad_cbar, bottom=cell_bottom, left=cell_left)
        )
        grid_pad = BoxStyle(padding=BoxInset(bottom=grid_bottom, left=grid_left))
        # Hug the heatmap with a thin, tall colorbar so it costs little width; the slice
        # colorbars carry no rotated label, so a small `label_reserve` matches `pad_cbar`.
        slice_colorbar_params = {
            "position": (1.04, 0.08),
            "size": (0.05, 0.84),
            "label_reserve": 0.22,
        }

        # axes_size sizes the per-cell min_dimensions so the panel honors the size the
        # caller asked for (per_network_row's per-cell width x panel_scale). The equal
        # weights drive the actual cell sizes; this just sets a sensible floor.
        if "axes_size" in self.model_fields_set:
            w, h = self.axes_size.width, self.axes_size.height
            grid_w = w * (1.0 - self.cube_frac_w) - grid_left
            grid_h = h - grid_bottom
            cell_w = max(0.3, (grid_w - gap * (cols - 1)) / cols - (cell_left + pad_cbar))
            cell_h = max(0.3, (grid_h - gap * (rows - 1)) / rows - (pad_title + cell_bottom))
            cube_size = {"axes_size": Size(width=w * self.cube_frac_w, height=h)}
            cell_size = {"axes_size": Size(width=cell_w, height=cell_h)}
        else:
            cube_size = {}
            cell_size = {}

        if self.slice_zvalues is not None:
            zs = np.asarray(self.slice_zvalues, dtype=float)
            assert zs.size == n_slices, (
                f"slice_zvalues has {zs.size} entries, expected R*C={n_slices}"
            )
        else:
            zs = np.linspace(self.slice_zrange[0], self.slice_zrange[1], n_slices)

        cube_zs: list[float] = (
            list(self.stack_zslices)
            if self.stack_zslices is not None
            else list(self.zslices)
            if self.zslices is not None
            else list(np.linspace(self.stack_zrange[0], self.stack_zrange[1], self.stack_n_slices))
        )

        cube_user, cube_def = _split_userset(
            self, {"xlims": "xlims", "ylims": "ylims", "zlims": "zlims", "vlims": "vlims"}
        )
        cube = CubeStackPanel(
            plot_data=self.plot_data,
            rescaler=self.rescaler,
            zslices=[cube_zs],
            **cube_user,
            **cube_size,
            projection_angle=self.projection_angle,
            projection_diag_coef=self.projection_diag_coef,
            title=self.title,
            title_kwargs=self.title_kwargs,
            show_inner_spines=self.cube_show_inner_spines,
            show_slice_ticks=self.cube_show_slice_ticks,
            show_front_face_ticks=self.cube_show_front_face_ticks,
            contour_reference_plot_data=self.cube_contour_reference_plot_data,
            smooth_2d_params=self.cube_smooth_2d_params,
            colorbar_params=self.cube_colorbar_params,
            draw_colorbar=False,
        )
        cube.with_defaults(**cube_def)

        slice_user, slice_def = _split_userset(
            self, {"xlims": "xlims", "ylims": "ylims", "vlims": "slice_vlims"}
        )
        slice_panels: list[SmoothPanel2D] = []
        for i, z in enumerate(zs):
            r, c = i // cols, i % cols
            is_left = c == 0 or all_ticks
            is_bottom = r == rows - 1 or all_ticks
            title_label = _format_z_label(float(z), self.rescaler)
            sp = SmoothPanel2D(
                plot_data=self.plot_data,
                rescaler=self.rescaler,
                zslice=[float(z)],
                **slice_user,
                **cell_size,
                style=cell_pad,
                vlim_quantiles=self.slice_vlim_quantiles,
                vlim_min_floor=None,
                vlim_min_range=None,
                # Per-slice colorbar: each slice auto-scales to its own range, so every
                # cell needs its own scale. The label would just repeat the output name
                # in every cell, so it's left off.
                draw_colorbar=self.slice_show_colorbar,
                draw_colorbar_label=False,
                colorbar_params=slice_colorbar_params,
                draw_xlabel=is_bottom and self.slice_show_axis_titles,
                draw_ylabel=is_left and self.slice_show_axis_titles,
                # Shared-axes grid: only the bottom row / left column show tick labels;
                # interior cells suppress them (draw_x/ylabel toggle only the axis
                # title, not the ticks).
                setup_transformed_axis_params={
                    "setup_xaxis_params": {"show_labels": is_bottom},
                    "setup_yaxis_params": {"show_labels": is_left},
                },
                # Title sits in the reserved top strip, above the heatmap (not over it).
                title=title_label,
                title_inside=False,
                title_kwargs={
                    "color": self.slice_title_color,
                    "fontsize": self.slice_title_fontsize,
                    "pad": self.slice_title_pad,
                },
            )
            sp.with_defaults(**slice_def)
            slice_panels.append(sp)
        slice_rows = [
            Container(
                layout=LayoutConstraints(
                    direction="row",
                    gap=gap,
                    align_items="stretch",
                    main_axis_weights=[1.0] * cols,
                ),
                children=slice_panels[r * cols : (r + 1) * cols],
            )
            for r in range(rows)
        ]
        slice_grid_container = Container(
            style=grid_pad,
            layout=LayoutConstraints(
                direction="column",
                gap=gap,
                align_items="stretch",
                main_axis_weights=[1.0] * rows,
            ),
            children=slice_rows,
        )

        self.layout = LayoutConstraints(
            direction="row",
            gap=self.cube_grid_gap,
            align_items="stretch",
            main_axis_weights=[self.cube_frac_w, 1.0 - self.cube_frac_w],
        )
        self.add_children([cube, slice_grid_container])

    def _build_uniform(self) -> None:
        """Height-invariant uniform layout: cube (full H tall) + an R×C slice grid packed into the
        SAME height H, so a [1,N] strip gives full-size slices (== a 2D value face) and a grid
        shrinks each row to H/R. Slice axes boxes are pinned (FACE_PAD chrome) so tick labels stay
        consistent; the panel pins ITSELF to the derived (width, H). See `uniform_height`."""
        assert self.uniform_height is not None
        height = self.uniform_height
        sa = self.uniform_single_aspect
        cbar = self.slice_show_colorbar
        rows, cols = self.slice_grid
        n = rows * cols
        # Slice cell packs R rows into H; its plotted box = cell minus the FACE_PAD chrome.
        slice_cell_h = (height - FACE_GAP * (rows - 1)) / rows
        box_h = max(0.2, slice_cell_h - FACE_PAD.top - FACE_PAD.bottom)
        box_w = sa * box_h
        cell_w, _ = face_cell_size(box_w, box_h, colorbar=cbar)
        # Cube spans the full height; width = cube_cell_units × a FULL-height face cell (so the cube
        # reads at face scale regardless of how many grid rows shrank the slices).
        full_box_h = max(0.2, height - FACE_PAD.top - FACE_PAD.bottom)
        full_cell_w, _ = face_cell_size(sa * full_box_h, full_box_h, colorbar=cbar)
        cube_cell_w = self.cube_cell_units * full_cell_w

        if self.slice_zvalues is not None:
            zs = np.asarray(self.slice_zvalues, dtype=float)
            assert zs.size == n, f"slice_zvalues has {zs.size} entries, expected R*C={n}"
        else:
            zs = np.linspace(self.slice_zrange[0], self.slice_zrange[1], n)
        cube_zs: list[float] = (
            list(self.stack_zslices)
            if self.stack_zslices is not None
            else list(self.zslices)
            if self.zslices is not None
            else list(np.linspace(self.stack_zrange[0], self.stack_zrange[1], self.stack_n_slices))
        )

        cube_user, cube_def = _split_userset(
            self, {"xlims": "xlims", "ylims": "ylims", "zlims": "zlims", "vlims": "vlims"}
        )
        cube = CubeStackPanel(
            plot_data=self.plot_data,
            rescaler=self.rescaler,
            zslices=[cube_zs],
            **cube_user,
            projection_angle=self.projection_angle,
            projection_diag_coef=self.projection_diag_coef,
            title=self.title,
            title_kwargs=self.title_kwargs,
            show_inner_spines=self.cube_show_inner_spines,
            show_slice_ticks=self.cube_show_slice_ticks,
            show_front_face_ticks=self.cube_show_front_face_ticks,
            contour_reference_plot_data=self.cube_contour_reference_plot_data,
            smooth_2d_params=self.cube_smooth_2d_params,
            colorbar_params=self.cube_colorbar_params,
            draw_colorbar=False,
        )
        cube.with_defaults(**cube_def)
        pin_to_cell(cube, cube_cell_w, height)

        slice_user, slice_def = _split_userset(
            self, {"xlims": "xlims", "ylims": "ylims", "vlims": "slice_vlims"}
        )
        slice_panels: list[SmoothPanel2D] = []
        for z in zs:
            sp = SmoothPanel2D(
                plot_data=self.plot_data,
                rescaler=self.rescaler,
                zslice=[float(z)],
                **slice_user,
                style=BoxStyle(padding=FACE_PAD),
                vlim_quantiles=self.slice_vlim_quantiles,
                vlim_min_floor=None,
                vlim_min_range=None,
                draw_colorbar=self.slice_show_colorbar,
                draw_colorbar_label=False,
                colorbar_params=dict(FACE_COLORBAR),
                draw_xlabel=self.slice_show_axis_titles,
                draw_ylabel=self.slice_show_axis_titles,
                setup_transformed_axis_params={
                    "setup_xaxis_params": {"show_labels": True},
                    "setup_yaxis_params": {"show_labels": True},
                },
                title=_format_z_label(float(z), self.rescaler),
                title_inside=False,
                title_kwargs={
                    "color": self.slice_title_color,
                    "fontsize": self.slice_title_fontsize,
                    "pad": self.slice_title_pad,
                },
            )
            sp.with_defaults(**slice_def)
            pin_axes_box(sp, box_w, box_h)
            slice_panels.append(sp)

        slice_rows = [
            Container(
                layout=LayoutConstraints(direction="row", gap=FACE_GAP, align_items="start"),
                children=slice_panels[r * cols : (r + 1) * cols],
            )
            for r in range(rows)
        ]
        grid = Container(
            layout=LayoutConstraints(direction="column", gap=FACE_GAP, align_items="start"),
            children=slice_rows,
        )
        self.layout = LayoutConstraints(
            direction="row", gap=self.cube_grid_gap, align_items="start"
        )
        self.add_children([cube, grid])
        # Pin SELF to the derived footprint so the parent DataBlock (and the table column) reads the
        # true width. grid_w = cols slice cells + inter-cell gaps.
        grid_w = cols * cell_w + FACE_GAP * (cols - 1)
        total_w = cube_cell_w + self.cube_grid_gap + grid_w
        object.__setattr__(self, "axes_size", Size(width=total_w, height=height))
        object.__setattr__(self, "min_dimensions", Size(width=total_w, height=height))
        object.__setattr__(self, "max_dimensions", Size(width=total_w, height=height))
        _freeze_size(self)

    def render_txt(self) -> str | None:
        from jeanplot.plots.txt import smooth_3d_txt

        zslices_arr = np.atleast_2d(np.asarray(self.zslices, dtype=float))
        result = smooth_3d_txt(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            zslices=zslices_arr,
            xlims=self.xlims,
            ylims=self.ylims,
            zlims=self.zlims,
            vlims=self.vlims,
            title=self.title,
        )
        return str(result)


class CubeStackPanel(PlotPanel):
    plot_data: PlotData
    zslices: list = Field(default_factory=lambda: [[0.05, 0.25, 0.4, 0.55]])
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    ylims: tuple[float | None, float | None] = (None, None)
    zlims: tuple[float | None, float | None] = (None, None)
    vlims: tuple[float | None, float | None] = (None, None)
    projection_angle: float = 45.0
    projection_diag_coef: float = 0.5
    draw_colorbar: bool | None = None
    show_inner_spines: bool = True
    show_slice_ticks: bool = True
    show_front_face_ticks: bool = False
    smooth_2d_params: dict | None = None
    colorbar_params: dict | None = None
    # Opacity (0..1) of the translucent value-heatmap slices drawn inside the cube. 1.0
    # (default) is unchanged/opaque; lower lets back slices show through. Read at draw
    # time and folded into the face `heatmap_params`, so it is cascade-fillable even though
    # the cube is built procedurally (`CubeStackPanel: {slice_opacity: 0.7}`).
    slice_opacity: float = 1.0
    # When set, symbolic contour levels are resolved from this field instead of
    # `plot_data` (lets a prediction cube clip at the ground-truth iso-level).
    contour_reference_plot_data: PlotData | None = None
    cube_edge_props: dict | None = None
    xaxis_labelpad: int = 20
    yaxis_labelpad: int = 24
    zaxis_labelpad: int = 0
    xtitle: str | None = None
    ytitle: str | None = None
    ztitle: str | None = None
    vtitle: str | None = None

    def _face_smooth_grid_params(self) -> dict | None:
        """Cube faces are SmoothPanel2D: resolve one against the cascade so the shared
        smoothing rules (+ `CubeStackPanel SmoothGrid` specializations) reach its leaf."""
        from jeanplot.core.style import jstyle
        from jeanplot.panels.smooth_2d import SmoothPanel2D

        face = SmoothPanel2D(plot_data=self.plot_data, is_drawable=False)
        face.parent = self
        jstyle.apply_one(face)
        return face.smooth_grid.params

    def draw(self, ax) -> PlotFunctionResult | None:
        from jeanplot.plots.smooth_3d import smooth_3d

        ref = self.contour_reference_plot_data
        contour_reference = (ref.x, ref.y) if ref is not None else None
        smooth_2d_params = dict(self.smooth_2d_params or {})
        if self.slice_opacity != 1.0:
            hp = dict(smooth_2d_params.get("heatmap_params") or {})
            hp.setdefault("opacity", self.slice_opacity)
            smooth_2d_params["heatmap_params"] = hp
        sgp = self._face_smooth_grid_params()
        if sgp is not None:
            smooth_2d_params["smooth_grid_params"] = sgp
        return smooth_3d(
            X=self.plot_data.x,
            Y=self.plot_data.y,
            input_names=self.plot_data.input_names,
            output_name=self.plot_data.output_name,
            rescaler=self.rescaler,
            ax=[ax],
            zslices=self.zslices,
            xlims=self.xlims,
            ylims=self.ylims,
            zlims=self.zlims,
            vlims=self.vlims,
            contour_reference=contour_reference,
            draw_colorbar=self.draw_colorbar,
            cube_edge_props=self.cube_edge_props,
            projection_angle=self.projection_angle,
            projection_diag_coef=self.projection_diag_coef,
            colorbar_params=self.colorbar_params,
            show_inner_spines=self.show_inner_spines,
            show_slice_ticks=self.show_slice_ticks,
            show_front_face_ticks=self.show_front_face_ticks,
            smooth_2d_params=smooth_2d_params,
            xtitle=self.xtitle,
            ytitle=self.ytitle,
            ztitle=self.ztitle,
            title=self.title,
            xaxis_labelpad=self.xaxis_labelpad,
            yaxis_labelpad=self.yaxis_labelpad,
            zaxis_labelpad=self.zaxis_labelpad,
        )


CubeView.model_rebuild(force=True)
SmoothPanel3D.model_rebuild(force=True)
CubeStackPanel.model_rebuild(force=True)
