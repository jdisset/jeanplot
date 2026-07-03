"""Figure-aware rendering: walk a Figure, allocate one mpl Axes per drawable leaf, run overlays.

The only jeanplot.render branch that knows about `Figure`/`PlotPanel`/`Colorbar`; the
scene-graph renderer is untouched.
"""

from pathlib import Path
from typing import Any, Iterator

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt

from dracon.progress import step

from jeanplot.core.component import Component
from jeanplot.core.style import jstyle
from jeanplot.core.table import GridStyle, LineStyle, Table, TableCell, TableRow
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.figure import Figure


def _iter_panels(root: Component) -> Iterator[PlotPanel]:
    if isinstance(root, PlotPanel):
        yield root
    for c in getattr(root, "children", None) or []:
        yield from _iter_panels(c)


def _knn_group_key(p: PlotPanel) -> tuple:
    # Geometry-only signature (no data identity), so panels that issue the same
    # Y-independent KNN query over different Y (ground-truth vs prediction) sort
    # adjacent and share the neighbour-query cache. Stable sort keeps a
    # deterministic intra-group order.
    return (
        type(p).__name__,
        repr(getattr(p, "xlims", None)),
        repr(getattr(p, "ylims", None)),
        repr(getattr(p, "zlims", None)),
        repr(getattr(p, "zslice", getattr(p, "zslices", None))),
    )


def _iter_debug_components(root: Component) -> Iterator[Component]:
    if getattr(root, "debug", False):
        yield root
    for c in getattr(root, "children", None) or []:
        yield from _iter_debug_components(c)


def _panel_bbox(
    panel: PlotPanel, root_w: float, root_h: float
) -> tuple[float, float, float, float]:
    assert root_w > 0 and root_h > 0, f"figure has zero size: {root_w}x{root_h}"
    ox, oy = panel.get_world_origin()
    p = panel.effective_padding
    aw = panel._dimensions.width - p.left - p.right
    ah = panel._dimensions.height - p.top - p.bottom
    assert aw > 0 and ah > 0, f"panel {panel.id} has no room for axes after insets"
    return ((ox + p.left) / root_w, 1.0 - (oy + p.top + ah) / root_h, aw / root_w, ah / root_h)


def _component_bbox(
    c: Component, root_w: float, root_h: float
) -> tuple[float, float, float, float]:
    ox, oy = c.get_world_origin()
    w, h = c._dimensions.width, c._dimensions.height
    return (ox / root_w, 1.0 - (oy + h) / root_h, w / root_w, h / root_h)


def _iter_all_components(root: Component) -> Iterator[Component]:
    yield root
    for c in getattr(root, "children", None) or []:
        yield from _iter_all_components(c)


def _line_style(style: Any):
    """Map a BoxStyle's border_style/dash_sequence to a matplotlib linestyle."""
    seq = getattr(style, "dash_sequence", None)
    if seq:
        return (float(getattr(style, "dash_offset", 0.0) or 0.0), tuple(seq))
    return {"solid": "-", "dashed": (0, (4, 2)), "dotted": (0, (1, 1.6))}.get(
        getattr(style, "border_style", "solid"), "-"
    )


def _grid_linestyle(ls: LineStyle):
    if ls.dash_sequence:
        return (0.0, tuple(ls.dash_sequence))
    return {"solid": "-", "dashed": (0, (4, 2)), "dotted": (0, (1, 1.6))}.get(ls.style, "-")


# cubic-Bezier control factor for a quarter-circle (4/3 * tan(pi/8))
_KAPPA = 0.5522847498307936


def _rounded_rect_path(x: float, y: float, w: float, h: float, rx: float, ry: float) -> mpath.Path:
    """Rounded rectangle as an explicit Bezier path with INDEPENDENT x/y corner radii.
    Built in whatever coords the caller draws in (figure fraction here), with `rx`/`ry`
    chosen so the corner is circular in *display* — exact, and immune to FancyBboxPatch's
    mutation_aspect scaling. Counter-clockwise from the bottom edge."""
    rx, ry = min(rx, w / 2.0), min(ry, h / 2.0)
    cx, cy = _KAPPA * rx, _KAPPA * ry
    x1, y1 = x + w, y + h
    Path = mpath.Path
    verts = [
        (x + rx, y),
        (x1 - rx, y),  # bottom edge
        (x1 - rx + cx, y),
        (x1, y + ry - cy),
        (x1, y + ry),  # bottom-right corner
        (x1, y1 - ry),  # right edge
        (x1, y1 - ry + cy),
        (x1 - rx + cx, y1),
        (x1 - rx, y1),  # top-right corner
        (x + rx, y1),  # top edge
        (x + rx - cx, y1),
        (x, y1 - ry + cy),
        (x, y1 - ry),  # top-left corner
        (x, y + ry),  # left edge
        (x, y + ry - cy),
        (x + rx - cx, y),
        (x + rx, y),  # bottom-left corner
        (x + rx, y),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


def _table_grid_geometry(table: Table, root_w: float, root_h: float):
    """Collapsed-grid geometry in figure-fraction coords (y measured from the bottom).
    Returns (frame_xywh, v_segments, h_lines, header_y) or None.

    Everything is derived from the **cell** bboxes (never the rows'): the frame is exactly
    the cells' bounding box, vertical dividers are cell right-edges per row (so colspans
    contribute fewer), and a horizontal boundary is the shared edge between consecutive
    rows. Frame and lines share one source, so they cannot drift apart. `header_y` is the
    boundary just below the header rows (None if no header)."""
    rows = [
        r
        for r in table.children
        if isinstance(r, TableRow) and r._dimensions.width > 0 and r._dimensions.height > 0
    ]
    # per-row, sorted cell bboxes; drop empty rows
    row_cells = [
        sorted(
            (_component_bbox(c, root_w, root_h) for c in r.children if isinstance(c, TableCell)),
            key=lambda b: b[0],
        )
        for r in rows
    ]
    row_cells = [cb for cb in row_cells if cb]
    if not row_cells:
        return None
    row_cells.sort(key=lambda cb: max(b[1] + b[3] for b in cb), reverse=True)  # top -> bottom

    all_b = [b for cb in row_cells for b in cb]
    x_left = min(b[0] for b in all_b)
    x_right = max(b[0] + b[2] for b in all_b)
    y_top = max(b[1] + b[3] for b in all_b)
    y_bottom = min(b[1] for b in all_b)

    v_segments: list[tuple[float, float, float]] = []
    for cb in row_cells:
        ry0, ry1 = min(b[1] for b in cb), max(b[1] + b[3] for b in cb)
        for b in cb[:-1]:  # each cell's right edge except the row's last = a column divider
            v_segments.append((b[0] + b[2], ry0, ry1))

    h_lines: list[float] = []
    header_y: float | None = None
    for i in range(1, len(row_cells)):
        y = max(b[1] + b[3] for b in row_cells[i])  # top of row i == bottom of row i-1
        if i == table.header_rows:
            header_y = y
        else:
            h_lines.append(y)
    return (x_left, y_bottom, x_right - x_left, y_top - y_bottom), v_segments, h_lines, header_y


def _draw_table_grid(table: Table, mfig: Any, root_w: float, root_h: float) -> None:
    """Draw a Table's collapsed grid: cell/row + header-band backgrounds, interior
    separators, header line, and the rounded outer frame — each element once, styled by the
    `GridStyle` SSOT. Backgrounds are clipped to the frame path so a header/cell fill keeps
    the frame's rounded corners instead of poking square nubs past them."""
    geom = _table_grid_geometry(table, root_w, root_h)
    if geom is None:
        return
    (fx, fy, fw, fh), v_segments, h_lines, header_y = geom
    g: GridStyle = table.grid

    # The frame path (rounded; rx=ry=0 degenerates to a plain rectangle) is the single
    # source for both the stroked frame AND the background clip region. Per-axis radii make
    # a `cr`-inch circular corner in display regardless of the figure's aspect.
    cr = min(max(0.0, g.corner_radius), fw * root_w / 2.0, fh * root_h / 2.0)
    frame_path = _rounded_rect_path(fx, fy, fw, fh, cr / root_w, cr / root_h)
    clip = (frame_path, mfig.transFigure)

    def bg(x: float, y: float, w: float, h: float, color: str | None) -> None:
        if not color or w <= 0 or h <= 0:
            return
        rect = mpatches.Rectangle(
            (x, y), w, h, facecolor=color, edgecolor="none", transform=mfig.transFigure, zorder=0
        )
        rect.set_clip_path(*clip)
        mfig.add_artist(rect)

    def line(x0: float, y0: float, x1: float, y1: float, ls: LineStyle) -> None:
        if not ls.visible:
            return
        mfig.add_artist(
            mlines.Line2D(
                [x0, x1],
                [y0, y1],
                transform=mfig.transFigure,
                color=ls.color,
                lw=ls.width,
                linestyle=_grid_linestyle(ls),
                solid_capstyle="projecting",
                zorder=2,
            )
        )

    # backgrounds (under everything), all clipped to the frame: the table's own fill, then
    # each row/cell fill, then the header band.
    bg(fx, fy, fw, fh, getattr(table.style, "background_color", None))
    for c in _iter_all_components(table):
        if isinstance(c, (TableRow, TableCell)):
            style = getattr(c, "style", None)
            bg(*_component_bbox(c, root_w, root_h), getattr(style, "background_color", None))
    if header_y is not None:
        bg(fx, header_y, fw, (fy + fh) - header_y, g.header_fill)

    for x, y0, y1 in v_segments:
        line(x, y0, x, y1, g.inner)
    for y in h_lines:
        line(fx, y, fx + fw, y, g.inner)
    if header_y is not None:
        line(fx, header_y, fx + fw, header_y, g.header)

    if g.frame.visible:
        mfig.add_artist(
            mpatches.PathPatch(
                frame_path,
                facecolor="none",
                edgecolor=g.frame.color,
                linewidth=g.frame.width,
                linestyle=_grid_linestyle(g.frame),
                transform=mfig.transFigure,
                zorder=2,
            )
        )


def _draw_chrome(fig: Figure, mfig: Any, root_w: float, root_h: float) -> None:
    """Draw container/cell backgrounds + borders (the figure path otherwise draws only
    PlotPanel leaves, so a Table's grid / a container's border would never show).

    A ``Table`` is special: its whole chrome is drawn once by ``_draw_table_grid`` from its
    ``GridStyle``, so its rows/cells are skipped here entirely. For every other container,
    per-side borders (CellStyle.border_top/right/bottom/left, ``False`` hides a side) become
    grid lines; ``border_style``/``dash_sequence`` dash them; ``corner_radius`` rounds a full box."""
    bg_patches: list = []
    border_segs: list[tuple] = []
    rounded: list[tuple] = []

    for c in _iter_all_components(fig):
        if isinstance(c, PlotPanel) or not getattr(c, "show", True):
            continue
        # A Table draws ALL of its own chrome (frame, separators, and the cell/row/header
        # backgrounds, clipped to the frame) in one place; skip its rows/cells here.
        if isinstance(c, Table):
            _draw_table_grid(c, mfig, root_w, root_h)
            continue
        if isinstance(c, (TableRow, TableCell)):
            continue
        style = getattr(c, "style", None)
        if style is None or c._dimensions.width <= 0 or c._dimensions.height <= 0:
            continue
        x, y, w, h = _component_bbox(c, root_w, root_h)
        bg = getattr(style, "background_color", None)
        bc = getattr(style, "border_color", None)
        bw = float(getattr(style, "border_width", 0) or 0)
        # CellStyle exposes per-side toggles (None/True = draw, False = hide); plain
        # BoxStyle has none of these -> all four sides draw.
        top = getattr(style, "border_top", None) is not False
        right = getattr(style, "border_right", None) is not False
        bottom = getattr(style, "border_bottom", None) is not False
        left = getattr(style, "border_left", None) is not False
        cr = float(getattr(style, "corner_radius", 0) or 0)

        if cr > 0 and top and right and bottom and left:
            # rounded full box: one FancyBboxPatch in inch space (aspect-correct
            # rounding regardless of the figure's w/h ratio).
            rounded.append((x * root_w, y * root_h, w * root_w, h * root_h, bg, bc, bw, cr, style))
            continue

        if bg:
            bg_patches.append(mpatches.Rectangle((x, y), w, h, facecolor=bg, edgecolor="none"))
        if bc and bw > 0:
            ls = _line_style(style)
            x1, y1 = x + w, y + h
            for on, seg in (
                (top, ((x, y1), (x1, y1))),
                (bottom, ((x, y), (x1, y))),
                (left, ((x, y), (x, y1))),
                (right, ((x1, y), (x1, y1))),
            ):
                if on:
                    border_segs.append((seg, bc, bw, ls))
    # backgrounds first (under the axes), then borders (over the gaps between axes)
    for p in bg_patches:
        p.set_transform(mfig.transFigure)
        p.set_zorder(0)
        mfig.add_artist(p)
    for xi, yi, wi, hi, bg, bc, bw, cr, style in rounded:
        cr = min(cr, wi / 2.0, hi / 2.0)
        patch = mpatches.FancyBboxPatch(
            (xi + cr, yi + cr),
            wi - 2 * cr,
            hi - 2 * cr,
            boxstyle=mpatches.BoxStyle("round", pad=cr, rounding_size=cr),
            facecolor=bg or "none",
            edgecolor=bc if (bc and bw > 0) else "none",
            linewidth=bw,
            linestyle=_line_style(style),
            transform=mfig.dpi_scale_trans,
            mutation_aspect=1.0,
            zorder=0 if bg else 2,
        )
        mfig.add_artist(patch)
    for (p0, p1), bc, bw, ls in border_segs:
        line = mlines.Line2D(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            transform=mfig.transFigure,
            color=bc,
            lw=bw,
            linestyle=ls,
            solid_capstyle="projecting",
            zorder=2,
        )
        mfig.add_artist(line)


def _draw_debug_overlays(fig: Figure, mfig: Any, root_w: float, root_h: float) -> None:
    """outline every component with debug=True (figure path bypasses render_debug)."""

    def rect(x, y, w, h, ls, lw):
        mfig.add_artist(
            mpatches.Rectangle(
                (x, y),
                w,
                h,
                transform=mfig.transFigure,
                fill=False,
                ec="red",
                ls=ls,
                lw=lw,
                zorder=10000,
            )
        )

    for c in _iter_debug_components(fig):
        if c._dimensions.width <= 0 or c._dimensions.height <= 0:
            continue
        x, y, w, h = _component_bbox(c, root_w, root_h)
        rect(x, y, w, h, "--", 0.5)
        if isinstance(c, PlotPanel) and c.is_drawable and not c.is_overlay:
            rect(*_panel_bbox(c, root_w, root_h), ":", 0.4)
        mfig.text(
            x,
            y + h,
            c.id or type(c).__name__,
            transform=mfig.transFigure,
            fontsize=5,
            color="red",
            va="bottom",
            ha="left",
            zorder=10001,
        )


def render_figure(fig: Figure) -> Any:
    if fig.theme_overrides is not None:
        from jeanplot.core.style_engine import merge_jstyle_rules

        jstyle.update(
            merge_jstyle_rules(fig.theme if fig.theme is not None else {}, fig.theme_overrides)
        )
    elif fig.theme is not None:
        jstyle.update(fig.theme)
    jstyle.apply(fig)

    with mpl.rc_context(rc=fig.rc_context):
        with step("layout"):
            fig.measure_and_layout(None)

        figsize = (fig._dimensions.width, fig._dimensions.height)
        mfig = plt.figure(figsize=figsize, dpi=fig.dpi)
        fig._mpl_figure = mfig

        root_w, root_h = figsize
        panels = list(_iter_panels(fig))
        drawable = [p for p in panels if p.is_drawable and not p.is_overlay]
        overlays = [p for p in panels if p.is_overlay]

        for panel in drawable:
            panel._axes = mfig.add_axes(_panel_bbox(panel, root_w, root_h))

        # Draw order is invisible to the output (each panel owns its axes), but
        # pulling each panel that shares a KNN query signature next to the first
        # panel with that signature lets the shared neighbour-query cache hit
        # (e.g. prediction reuses ground-truth's weights over the same grid).
        # Group-leader stable sort: a duplicate only jumps past panels that
        # don't share its signature, so same-region compositing is preserved.
        leader: dict = {}
        for i, p in enumerate(drawable):
            leader.setdefault(_knn_group_key(p), i)
        draw_order = sorted(
            enumerate(drawable), key=lambda iv: (leader[_knn_group_key(iv[1])], iv[0])
        )

        with step("draw"):
            for _, panel in draw_order:
                assert panel._axes is not None, f"panel {panel.id} has no axes"
                with step(f"panel {type(panel).__name__}"):
                    res = panel.draw(panel._axes)
                if res is not None:
                    if getattr(res, "mappable", None) is not None:
                        panel._mappable = res.mappable
                    meta = getattr(res, "metadata", None)
                    if meta:
                        fig.metadata.update(meta)
                    panel._last_metadata = meta or {}

            for overlay in overlays:
                parent_ax = getattr(overlay.parent, "_axes", None) if overlay.parent else None
                if parent_ax is None:
                    continue
                overlay.draw(parent_ax)

        _draw_chrome(fig, mfig, root_w, root_h)
        _draw_debug_overlays(fig, mfig, root_w, root_h)

        if fig.subtitle:
            mfig.suptitle(fig.subtitle, **fig.subtitle_kwargs)

        return mfig


def _stringify_metadata(md: dict | None) -> dict | None:
    if not md:
        return None
    return {str(k): v if isinstance(v, str) else repr(v) for k, v in md.items()}


# Matplotlib metadata allowlists per backend. SVG uses Dublin-Core; PDF/PS use
# pdf-info dict; PNG accepts a different set. Unknown keys raise UserWarning at
# save time, so we whitelist by-format and fold extras into a Description/Subject
# free-text field where one exists.
_SVG_ALLOWED_KEYS = frozenset(
    {
        "Coverage",
        "Date",
        "Description",
        "Format",
        "Identifier",
        "Language",
        "Publisher",
        "Relation",
        "Rights",
        "Source",
        "Subject",
        "Title",
        "Type",
        "Creator",
        "Contributor",
        "Keywords",
    }
)

_PDF_ALLOWED_KEYS = frozenset(
    {
        "Title",
        "Author",
        "Subject",
        "Keywords",
        "Creator",
        "Producer",
        "CreationDate",
        "ModDate",
        "Trapped",
    }
)

_PNG_ALLOWED_KEYS = frozenset(
    {
        "Title",
        "Author",
        "Description",
        "Copyright",
        "Creation Time",
        "Software",
        "Disclaimer",
        "Warning",
        "Source",
        "Comment",
    }
)

_METADATA_ALLOWLISTS = {
    ".svg": (_SVG_ALLOWED_KEYS, "Description"),
    ".pdf": (_PDF_ALLOWED_KEYS, "Subject"),
    ".ps": (_PDF_ALLOWED_KEYS, "Subject"),
    ".eps": (_PDF_ALLOWED_KEYS, "Subject"),
    ".png": (_PNG_ALLOWED_KEYS, "Description"),
}


def _metadata_for(path: Path, md: dict | None) -> dict | None:
    if not md:
        return None
    entry = _METADATA_ALLOWLISTS.get(path.suffix.lower())
    if entry is None:
        return md
    allowed_keys, freetext_field = entry
    allowed = {k: v for k, v in md.items() if k in allowed_keys}
    extras = {k: v for k, v in md.items() if k not in allowed_keys}
    if extras:
        prior = allowed.get(freetext_field, "")
        joined = "; ".join(f"{k}={v}" for k, v in extras.items())
        allowed[freetext_field] = f"{prior}; {joined}" if prior else joined
    return allowed or None


def save_figure(fig: Figure, mfig: Any) -> Path | None:
    out = fig.output_path
    if out is None:
        return None
    md = _stringify_metadata(fig.metadata)
    # Colocated twins (same dir + stem, swapped suffix) derived from the FINAL path, plus any
    # explicit extra paths. Dedup so an ext that coincides with an explicit path isn't written twice.
    twins = [out.with_suffix("." + e.lstrip(".")) for e in fig.also_save_exts]
    extras = [Path(p) for p in fig.extra_output_paths]
    seen = {out}
    with step("save"), mpl.rc_context(rc=fig.rc_context):
        out.parent.mkdir(parents=True, exist_ok=True)
        mfig.savefig(out, dpi=fig.dpi, metadata=_metadata_for(out, md))
        for p in twins + extras:
            if p in seen:
                continue
            seen.add(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            mfig.savefig(p, dpi=fig.dpi, metadata=_metadata_for(p, md))
    return out
