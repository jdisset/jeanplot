"""Figure-aware rendering: walks a Figure(Container), allocates one mpl Axes
per drawable leaf PlotPanel, then runs overlay draws.

This file owns the only branch in jeanplot.render that knows about `Figure`,
`PlotPanel`, and `Colorbar`. The scene-graph renderer is untouched.
"""

from pathlib import Path
from typing import Any, Iterator

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from dracon.progress import step

from jeanplot.core.component import Component
from jeanplot.core.style import jstyle
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


def _draw_chrome(fig: Figure, mfig: Any, root_w: float, root_h: float) -> None:
    """Draw container/cell backgrounds + borders (the figure path otherwise only
    draws PlotPanel leaves, so a Table's cell borders / row backgrounds would never
    show). Per-side borders (CellStyle.border_top/right/bottom/left, where ``False``
    hides a side) become real table grid lines; ``border_style``/``dash_sequence``
    dash them; ``corner_radius`` rounds a full (four-sided) box's corners."""
    bg_patches: list = []
    border_segs: list[tuple] = []
    rounded: list[tuple] = []
    for c in _iter_all_components(fig):
        if isinstance(c, PlotPanel) or not getattr(c, "show", True):
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

        jstyle.update(merge_jstyle_rules(fig.theme if fig.theme is not None else {}, fig.theme_overrides))
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
    with step("save"), mpl.rc_context(rc=fig.rc_context):
        out.parent.mkdir(parents=True, exist_ok=True)
        mfig.savefig(out, dpi=fig.dpi, metadata=_metadata_for(out, md))
        for extra in fig.extra_output_paths:
            p = Path(extra)
            p.parent.mkdir(parents=True, exist_ok=True)
            mfig.savefig(p, dpi=fig.dpi, metadata=_metadata_for(p, md))
    return out
