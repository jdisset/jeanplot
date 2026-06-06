from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from svgpath2mpl import parse_path
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MplPath
from matplotlib.colors import to_rgba
from contextlib import contextmanager
import re
import math

from jeanplot.core.component import Component
from jeanplot.core.models import Size, BoxStyle
from jeanplot.core.renderer.base import BaseRenderer
from jeanplot.core.renderer.common import (
    EPSILON,
    get_matrix_avg_scale,
    get_mpl_linestyle,
    get_mpl_linestyle_from_boxstyle,
    get_recursive_world_bounds,
)
from jeanplot.core.svg import (
    SVGElement,
    SVGPathData,
    arc_to_bezier,
)
from jeanplot.core.debug import DebugMixin, get_logger
from jeanplot.core.text import Text
from jeanplot.core.text_metrics import (
    DEFAULT_TEXT_REF_FONT_SIZE,
    build_font_properties,
    measure_text_metrics,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jeanplot.core.connection_label import ConnectionLabel

logger = get_logger(__name__)


def _apply_opacity(color: str | tuple | None, opacity: float) -> str | tuple | None:
    if color == "transparent":
        return "none"
    if color is None or color == "none" or opacity >= 1.0:
        return color
    try:
        rgba = to_rgba(color)
        return (*rgba[:3], rgba[3] * opacity)
    except (ValueError, TypeError):
        return color


def _get_point_scale_factor(axis: Axes) -> float:
    """average points per data unit for the current view."""
    fig = axis.get_figure()
    if not fig:
        return 1.0
    try:
        p0_disp = axis.transData.transform([(0, 0)])[0]
        p1_disp_x = axis.transData.transform([(1, 0)])[0]
        p1_disp_y = axis.transData.transform([(0, 1)])[0]
        dx_disp = abs(p1_disp_x[0] - p0_disp[0])
        dy_disp = abs(p1_disp_y[1] - p0_disp[1])
        if dx_disp < EPSILON and dy_disp < EPSILON:
            return 0.0
        avg_pixels_per_data = (
            (dx_disp + dy_disp) / 2.0
            if (dx_disp > EPSILON and dy_disp > EPSILON)
            else max(dx_disp, dy_disp)
        )
        points_per_pixel = 72.0 / fig.dpi
        return avg_pixels_per_data * points_per_pixel
    except Exception:
        logger.warning("failed to get point scale factor", exc_info=True)
        return 1.0


def _linewidth_in_points(data_unit_value: float, axis: Axes) -> float:
    """convert a data-unit dimension to points using average axis scale."""
    if data_unit_value <= 0:
        return 0.0
    axis_scale_factor = _get_point_scale_factor(axis)
    return max(0.0, data_unit_value * axis_scale_factor) if axis_scale_factor > EPSILON else 0.0


def _calc_linewidth(
    linewidth_data: float, width_mode: str, matrix: np.ndarray, context: "Axes", has_edge: bool
) -> tuple[float, float, bool]:
    """returns (lw_points, scaled_width_for_tracking, is_data_width)."""
    if not has_edge or linewidth_data <= 0:
        return 0.0, 0.0, False
    if width_mode == "data":
        scaled = linewidth_data * get_matrix_avg_scale(matrix)
        return _linewidth_in_points(scaled, context), scaled, True
    return linewidth_data, 0.0, False


@contextmanager
def _no_autoscale(ax: Axes):
    autoscale_state = ax.get_autoscale_on()
    ax.set_autoscale_on(False)
    try:
        yield
    finally:
        ax.set_autoscale_on(autoscale_state)


def _get_points_per_unit_vector(
    ax: Axes, matrix: np.ndarray, vector: tuple[float, float] = (0, 1)
) -> float:
    """points per data unit along a vector at the component's transformed location."""
    fig = ax.get_figure()
    if not fig:
        return 1.0
    try:
        local_origin = np.array([0, 0, 1])
        local_endpoint = np.array([vector[0], vector[1], 1])
        world_origin = (matrix @ local_origin)[:2]
        world_endpoint = (matrix @ local_endpoint)[:2]
        display_origin = ax.transData.transform(world_origin)
        display_endpoint = ax.transData.transform(world_endpoint)
        dist_display = math.hypot(
            display_endpoint[0] - display_origin[0], display_endpoint[1] - display_origin[1]
        )
        points = dist_display * (72.0 / fig.dpi)
        vector_len = math.hypot(vector[0], vector[1])
        ppu = points / vector_len if vector_len > EPSILON else 0.0
        return ppu
    except Exception as e:
        logger.warning(f"error calculating points per unit vector: {e}")
        return 1.0


def _get_rotation_from_matrix(matrix: np.ndarray) -> float:
    return math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))


class MatplotlibRenderer(DebugMixin, BaseRenderer):
    RENDERER_NAME = "matplotlib"

    def __init__(self, debug: bool = False, force_native_text: bool = False):
        super().__init__()
        self._data_width_patches: list[tuple[mpatches.Patch, float]] = []
        # (text_artist, data_font_size, world_matrix) for font size refresh
        self._data_font_texts: list[tuple] = []
        self._context: Axes | None = None
        self._draw_event_cid: int | None = None
        self._points_per_data_unit: float = 1.0
        self.force_native_text: bool = force_native_text
        self._pending_connection_clips: list = []

    def _disconnect_draw_event(self):
        if self._draw_event_cid is not None and self._context is not None:
            fig = self._context.get_figure()
            if fig and fig.canvas:
                try:
                    fig.canvas.mpl_disconnect(self._draw_event_cid)
                except Exception:
                    pass
        self._draw_event_cid = None

    def create_context(
        self,
        width: float = 800,
        height: float = 600,
        dpi: int = 150,
        ax: Axes | None = None,
        **kwargs,
    ) -> Axes:
        self._disconnect_draw_event()
        self._data_width_patches = []
        self._data_font_texts = []
        if ax:
            self._context = ax
            fig = ax.get_figure()
        else:
            figsize = (width / dpi, height / dpi)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, **kwargs)
            self._context = ax

        if fig and fig.canvas:
            # update data-unit linewidths and font sizes on draw/zoom
            self._draw_event_cid = fig.canvas.mpl_connect("draw_event", self._refresh_data_units)
        else:
            self._log_debug("warning: could not connect draw event (no figure/canvas)")

        self._points_per_data_unit = _get_points_per_unit_vector(self._context, np.eye(3))
        return self._context

    def _refresh_data_units(self, event=None):
        self.refresh_linewidths(event)
        self.refresh_text_font_sizes(event)

    def refresh_linewidths(self, event=None):
        if not self._context or not self._data_width_patches:
            return

        valid_patches = []
        for patch, scaled_width_data in self._data_width_patches:
            if not (
                patch
                and hasattr(patch, "axes")
                and patch.axes is self._context
                and hasattr(patch, "figure")
                and patch.figure
            ):
                continue

            valid_patches.append((patch, scaled_width_data))
            if hasattr(patch, "set_linewidth"):
                try:
                    new_width_points = _linewidth_in_points(scaled_width_data, self._context)
                    if not np.isclose(patch.get_linewidth(), new_width_points, atol=0.05):
                        patch.set_linewidth(new_width_points)
                except Exception as e:
                    self._log_debug(f"error updating patch lw {id(patch)}: {e}")

        self._data_width_patches = valid_patches

    def refresh_text_font_sizes(self, event=None):
        """update text font sizes to maintain data-unit proportions after axis changes."""
        if not self._context or not self._data_font_texts:
            return

        valid_texts = []
        for text_artist, data_font_size, world_matrix in self._data_font_texts:
            if not (
                text_artist
                and hasattr(text_artist, "axes")
                and text_artist.axes is self._context
                and hasattr(text_artist, "figure")
                and text_artist.figure
            ):
                continue

            valid_texts.append((text_artist, data_font_size, world_matrix))
            try:
                ppu_y = _get_points_per_unit_vector(self._context, world_matrix, vector=(0, 1))
                new_font_size = data_font_size * ppu_y if ppu_y > EPSILON else data_font_size
                current_size = text_artist.get_fontsize()
                if not np.isclose(current_size, new_font_size, rtol=0.01):
                    text_artist.set_fontsize(new_font_size)
            except Exception as e:
                self._log_debug(f"error updating text font size {id(text_artist)}: {e}")

        self._data_font_texts = valid_texts

    def track_patch(self, patch: mpatches.Patch, scaled_width_data: float):
        self._data_width_patches.append((patch, scaled_width_data))

    def render_path(
        self,
        context: Axes,
        path_data: SVGPathData,
        matrix: np.ndarray,
        line_width_mode: str = "data",
        color_remap: dict[str, str | None] | None = None,
        component_id: str | None = None,
        opacity: float = 1.0,
    ):
        comp_id_str = component_id or "unknown"
        try:
            mpl_path = parse_path(path_data.d)
            final_matrix = matrix
            if path_data.transform:
                m = re.search(r"matrix\((.+)\)", path_data.transform)
                if m:
                    vals = [float(v.strip()) for v in m.group(1).split(",")]
                    if len(vals) == 6:
                        t_mat = np.array(
                            [[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]], [0, 0, 1]]
                        )
                        final_matrix = final_matrix @ t_mat

            transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData
            remap = color_remap or {}
            fill_color_out = remap.get(path_data.fill, path_data.fill) if path_data.fill else None
            stroke_color_out = (
                remap.get(path_data.stroke, path_data.stroke) if path_data.stroke else None
            )
            final_facecolor = _apply_opacity(fill_color_out, opacity) if fill_color_out else "none"
            final_edgecolor = (
                _apply_opacity(stroke_color_out, opacity) if stroke_color_out else "none"
            )

            initial_lw_points = 0.0
            scaled_width_data_for_tracking = 0.0
            is_data_width = (
                line_width_mode == "data"
                and final_edgecolor != "none"
                and path_data.stroke_width > 0
            )

            if is_data_width:
                matrix_scale = get_matrix_avg_scale(final_matrix)
                scaled_width_data = path_data.stroke_width * matrix_scale
                scaled_width_data_for_tracking = scaled_width_data
                initial_lw_points = _linewidth_in_points(scaled_width_data, context)
            elif final_edgecolor != "none" and path_data.stroke_width > 0:
                initial_lw_points = path_data.stroke_width

            linestyle = get_mpl_linestyle(path_data)
            with _no_autoscale(context):
                patch = mpatches.PathPatch(
                    mpl_path,
                    facecolor=final_facecolor,
                    edgecolor=final_edgecolor,
                    linewidth=initial_lw_points,
                    linestyle=linestyle,
                    transform=transform,
                    capstyle="round",
                    joinstyle="round",
                )
                context.add_patch(patch)
                if component_id:
                    patch.set_gid(component_id)
                if is_data_width:
                    self.track_patch(patch, scaled_width_data_for_tracking)

        except Exception as e:
            path_preview = path_data.d[:30] + "..." if path_data.d else "N/A"
            self._log_debug(f"error rendering path '{path_preview}' for '{comp_id_str}': {e}")

    def _create_rounded_rect_path(self, x, y, w, h, radius):
        if radius < EPSILON:
            verts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            codes = [
                MplPath.MOVETO,
                MplPath.LINETO,
                MplPath.LINETO,
                MplPath.LINETO,
                MplPath.CLOSEPOLY,
            ]
        else:
            r = min(radius, min(w, h) / 2.0)
            v, c = [], []
            v.append((x + r, y + h))
            c.append(MplPath.MOVETO)
            v.append((x + w - r, y + h))
            c.append(MplPath.LINETO)
            _, cp1, cp2, p = arc_to_bezier(x + w - r, y + h - r, r, 90, 0)
            v.extend([cp1, cp2, p])
            c.extend([MplPath.CURVE4] * 3)
            v.append((x + w, y + r))
            c.append(MplPath.LINETO)
            _, cp1, cp2, p = arc_to_bezier(x + w - r, y + r, r, 0, -90)
            v.extend([cp1, cp2, p])
            c.extend([MplPath.CURVE4] * 3)
            v.append((x + r, y))
            c.append(MplPath.LINETO)
            _, cp1, cp2, p = arc_to_bezier(x + r, y + r, r, -90, -180)
            v.extend([cp1, cp2, p])
            c.extend([MplPath.CURVE4] * 3)
            v.append((x, y + h - r))
            c.append(MplPath.LINETO)
            _, cp1, cp2, p = arc_to_bezier(x + r, y + h - r, r, -180, -270)
            v.extend([cp1, cp2, p])
            c.extend([MplPath.CURVE4] * 3)
            c.append(MplPath.CLOSEPOLY)
            v.append(v[0])
            verts, codes = v, c
        return MplPath(verts, codes)

    def render_rectangle(
        self, context: Axes, bounds: Size, style: BoxStyle, matrix: np.ndarray, component=None
    ):
        w, h = bounds.width, bounds.height
        if w <= 0 or h <= 0:
            return
        transform = mtransforms.Affine2D(matrix=matrix) + context.transData

        # multi-layer shadow for soft blur
        if style.shadow and style.shadow.blur_radius > 0:
            shadow = style.shadow
            scale_factor = _get_point_scale_factor(context)
            points_per_unit = max(scale_factor, 1.0)
            # heuristic layer count based on blur and resolution
            num_layers = min(
                100, max(4, int((shadow.blur_radius * points_per_unit**0.75) * shadow.resolution))
            )
            try:
                base_color_rgb, base_alpha = to_rgba(shadow.color)[:3], to_rgba(shadow.color)[3]
            except ValueError:
                base_color_rgb, base_alpha = (0, 0, 0), 0.5
            accumulated_alpha, min_render_alpha = 0.0, 1.0 / 256.0

            for i in range(num_layers - 1, -1, -1):  # outside in
                layer_frac = (i / (num_layers - 1)) if num_layers > 1 else 1.0
                alpha_frac = ((num_layers - 1 - i) / (num_layers - 1)) if num_layers > 1 else 0.0
                # distribute alpha across layers, more intense toward center
                target_intensity = (
                    base_alpha * (1 - alpha_frac**1.5) / num_layers
                    if num_layers > 0
                    else base_alpha
                )
                accumulated_alpha += target_intensity
                # quantize to avoid excessive layers for faint shadows
                patch_alpha = max(
                    0, min(1.0, round(accumulated_alpha / min_render_alpha) * min_render_alpha)
                )
                if patch_alpha <= min_render_alpha / 2.0:
                    continue
                accumulated_alpha = max(0, accumulated_alpha - patch_alpha)

                spread = shadow.spread + shadow.blur_radius * (1 - layer_frac)
                sw, sh = w + 2 * spread, h + 2 * spread
                sx, sy = -spread + shadow.offset_x, -spread + shadow.offset_y
                s_radius = (
                    min(style.corner_radius + spread, min(sw, sh) / 2.0) if min(sw, sh) > 0 else 0
                )
                if sw <= 0 or sh <= 0:
                    continue
                layer_path = self._create_rounded_rect_path(sx, sy, sw, sh, s_radius)
                with _no_autoscale(context):
                    context.add_patch(
                        mpatches.PathPatch(
                            layer_path,
                            facecolor=(*base_color_rgb, patch_alpha),
                            edgecolor="none",
                            lw=0,
                            transform=transform,
                            clip_on=False,
                        )
                    )

        opacity = getattr(component, "opacity", 1.0) if component else 1.0
        facecolor = _apply_opacity(style.background_color, opacity) or "none"
        edgecolor = _apply_opacity(style.border_color, opacity) or "none"
        has_edge = edgecolor != "none" and style.border_width > 0

        if facecolor != "none" or has_edge:
            main_path = self._create_rounded_rect_path(0, 0, w, h, style.corner_radius)
            lw_pts, scaled_w, is_data_w = _calc_linewidth(
                style.border_width, style.border_width_mode, matrix, context, has_edge
            )
            linestyle = get_mpl_linestyle_from_boxstyle(style)
            with _no_autoscale(context):
                main_patch = mpatches.PathPatch(
                    main_path,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=lw_pts,
                    linestyle=linestyle,
                    transform=transform,
                    capstyle="round",
                    joinstyle="round",
                )
                context.add_patch(main_patch)
                if is_data_w:
                    self.track_patch(main_patch, scaled_w)

    def render_polygon(
        self,
        context: Axes,
        path: list[tuple[float, float]],
        style: BoxStyle,
        matrix: np.ndarray,
        component=None,
    ):
        """render a polygon outline (border only, no fill)."""
        if len(path) < 3:
            return

        opacity = getattr(component, "opacity", 1.0) if component else 1.0
        edgecolor = _apply_opacity(style.border_color, opacity) or "none"
        if edgecolor == "none" or style.border_width <= 0:
            return

        vertices = list(path)
        if vertices[-1] != vertices[0]:
            vertices.append(vertices[0])
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]
        self._render_stroked_path(context, vertices, codes, style, matrix, edgecolor)

    def render_edges(
        self,
        context: Axes,
        edges: list[tuple[tuple[float, float], tuple[float, float]]],
        style: BoxStyle,
        matrix: np.ndarray,
        component=None,
    ):
        if not edges:
            return

        edgecolor = style.border_color or "none"
        if edgecolor == "none" or style.border_width <= 0:
            return

        vertices, codes = [], []
        for start, end in edges:
            vertices.extend([start, end])
            codes.extend([MplPath.MOVETO, MplPath.LINETO])
        self._render_stroked_path(context, vertices, codes, style, matrix, edgecolor)

    def _render_stroked_path(
        self,
        context: Axes,
        vertices: list[tuple[float, float]],
        codes: list[int],
        style: BoxStyle,
        matrix: np.ndarray,
        edgecolor: str,
    ):
        lw_pts, scaled_w, is_data_w = _calc_linewidth(
            style.border_width, style.border_width_mode, matrix, context, True
        )
        with _no_autoscale(context):
            patch = mpatches.PathPatch(
                MplPath(vertices, codes),
                facecolor="none",
                edgecolor=edgecolor,
                linewidth=lw_pts,
                linestyle=get_mpl_linestyle_from_boxstyle(style),
                transform=mtransforms.Affine2D(matrix=matrix) + context.transData,
                capstyle="round",
                joinstyle="round",
            )
            context.add_patch(patch)
            if is_data_w:
                self.track_patch(patch, scaled_w)

    def render_svg(self, context: Axes, svg_element: SVGElement, matrix: np.ndarray):
        comp_id = svg_element.id or "unknown_svg"
        if not svg_element._parsed_svg_content:
            svg_element._parse_and_validate_svg()
            if not svg_element._parsed_svg_content:
                self._log_debug(f"cannot render svg {comp_id}: parsing failed")
                if svg_element.debug:
                    self.render_debug(context, svg_element, matrix)
                return

        svg_data = svg_element._parsed_svg_content
        if not svg_data.paths:
            if svg_element.debug:
                self.render_debug(context, svg_element, matrix)
            return

        vb_x, vb_y, vb_w, vb_h = svg_data.viewBox or (0, 0, svg_data.width, svg_data.height)
        vb_w, vb_h = max(vb_w, EPSILON), max(vb_h, EPSILON)
        comp_w, comp_h = svg_element._dimensions.width, svg_element._dimensions.height
        if comp_w <= EPSILON or comp_h <= EPSILON:
            if svg_element.debug:
                self.render_debug(context, svg_element, matrix)
            return

        scale_x, scale_y = comp_w / vb_w, comp_h / vb_h
        # y-flip converts svg coords to matplotlib
        svg_internal_matrix = np.array(
            [[scale_x, 0, -scale_x * vb_x], [0, -scale_y, scale_y * vb_y + comp_h], [0, 0, 1]]
        )
        final_matrix = matrix @ svg_internal_matrix

        opacity = getattr(svg_element, "opacity", 1.0)
        for path_data in svg_data.paths:
            self.render_path(
                context,
                path_data,
                final_matrix,
                svg_element.line_width_mode,
                svg_element.color_remap,
                component_id=comp_id,
                opacity=opacity,
            )
        if svg_element.debug:
            self.render_debug(context, svg_element, matrix)

    def render_connection_label(
        self,
        context: Axes,
        label: "ConnectionLabel",
        wx: float,
        wy: float,
        angle_deg: float,
        world_matrix: np.ndarray,
    ) -> None:
        import matplotlib.patheffects as mpe

        data_fontsize = label.font_size
        ppu_y = _get_points_per_unit_vector(context, world_matrix, vector=(0, 1))
        fontsize_pts = data_fontsize * ppu_y if ppu_y > EPSILON else data_fontsize

        path_effects = []
        if label.halo and label.halo.width > 0:
            path_effects.append(
                mpe.withStroke(linewidth=label.halo.width, foreground=label.halo.color)
            )
        else:
            stroke_w = max(2.0, fontsize_pts * 0.4)
            path_effects.append(mpe.withStroke(linewidth=stroke_w, foreground="white"))

        opacity = getattr(label, "opacity", 1.0)
        text_color = _apply_opacity(label.color, opacity) if label.color else None

        props = build_font_properties(
            font_name=label.font_name,
            font_weight=label.font_weight,
            font_style=label.font_style,
        )

        with _no_autoscale(context):
            text_artist = context.text(
                wx,
                wy,
                label.effective_text,
                fontsize=fontsize_pts,
                color=text_color,
                ha="center",
                va="center",
                rotation=angle_deg,
                path_effects=path_effects,
                zorder=label.z_index,
                fontproperties=props,
                rotation_mode="anchor",
            )
        label._text_artist = text_artist

        self._data_font_texts.append((text_artist, data_fontsize, world_matrix.copy()))

        if label.clip_connection:
            self._pending_connection_clips.append(label)

    def render_component(
        self,
        context: Axes,
        component: Component,
        adjust_lims: bool = True,
        adjust_lims_padding: float = 0.1,
        adjust_lims_set_aspect: bool = True,
    ):
        if self._context != context:
            self._log_debug("context changed, updating and reconnecting event")
            self._disconnect_draw_event()
            self._context = context
            fig = context.get_figure()
            if fig and fig.canvas:
                self._draw_event_cid = fig.canvas.mpl_connect(
                    "draw_event", self._refresh_data_units
                )

        component.measure_and_layout(self)

        if adjust_lims and component.parent is None:
            self._adjust_limits(
                context, component,
                padding=adjust_lims_padding,
                set_aspect=adjust_lims_set_aspect,
            )

        # stabilize THIS axes' transform (aspect box adjustment) before rendering.
        # apply_aspect() does the same box fit as a full canvas.draw() but for one
        # axes only — a full figure draw here is O(n) in already-placed axes, so in
        # a multi-panel draw loop it becomes O(n^2). Data-unit widths/fonts are
        # refreshed below + again at savefig via the persistent draw_event.
        if isinstance(context, Axes):
            context.apply_aspect()

        for cb in self.pre_render_callbacks:
            cb(context)
        self._data_width_patches = []
        self._data_font_texts = []
        self._pending_connection_clips = []
        root_world_matrix = component.compute_world_matrix()
        component.render(self, context, root_world_matrix)
        if self._pending_connection_clips:
            from jeanplot.core.connection_label import ConnectionLabel

            ConnectionLabel.finalize_clips(self._pending_connection_clips, context)
            self._pending_connection_clips = []
        for cb in self.post_render_callbacks:
            cb(context)
        self._refresh_data_units()

    def _adjust_limits(
        self, context: Axes, root: Component,
        padding: float = 0.1, set_aspect: bool = True,
    ):
        from jeanplot.core.connector import Connection

        bounds = get_recursive_world_bounds(root, exclude_types=(Connection,))
        if bounds:
            min_x, min_y, max_x, max_y = bounds
            width = max(max_x - min_x, 1.0)
            height = max(max_y - min_y, 1.0)
            pad_x = width * padding * 1.5
            pad_y = height * padding * 1.5
            context.set_xlim(min_x - pad_x, max_x + pad_x)
            context.set_ylim(min_y - pad_y, max_y + pad_y)
            self._log_debug(
                f"adjusted limits: x=[{min_x - pad_x:.1f}, {max_x + pad_x:.1f}], y=[{min_y - pad_y:.1f}, {max_y + pad_y:.1f}]"
            )
        else:
            context.set_xlim(0, 100)
            context.set_ylim(0, 100)
            self._log_debug("no valid bounds found, setting default limits (0,100).")
        if set_aspect:
            context.set_aspect("equal", adjustable="box")

    def measure_text(self, text_component: Text) -> Size:
        metrics = measure_text_metrics(
            text=text_component.effective_text or "",
            font_name=text_component.font_name,
            font_weight=text_component.font_weight,
            font_style=text_component.font_style,
            line_spacing=text_component.line_spacing,
            ref_font_size=DEFAULT_TEXT_REF_FONT_SIZE,
        )
        text_component._text_metrics_cache = metrics
        return Size(width=metrics.width_points, height=metrics.height_points)

    def get_font_size_in_points(self, font_size) -> float:
        assert self._context is not None, "context is None"
        ppu_y = _get_points_per_unit_vector(self._context, np.eye(3), vector=(0, 1))
        if ppu_y <= EPSILON:
            self._log_debug(f"zero points per unit vector (ppu_y={ppu_y}), using default size")
            return font_size
        return font_size * ppu_y

    def render_text(self, context: Axes, text_component: Text, matrix: np.ndarray):
        comp_id = text_component.id or "unknown_text"
        if (
            not text_component.effective_text
            or not text_component.show
            or not text_component._text_metrics_cache
        ):
            self._log_debug(f"render_text skipped for {comp_id}: missing text/show/cache")
            if text_component.debug:
                self.render_debug(context, text_component, matrix)
            return

        target_data_height = text_component.font_size
        component_data_width = text_component._dimensions.width
        component_data_height = text_component._dimensions.height
        if component_data_height <= EPSILON:
            self._log_debug(f"skipping render text '{comp_id}': zero target height")
            return

        if text_component.font_size_mode == "points":
            required_point_size = text_component.font_size
        else:
            # "data" mode: scale font size with the world transform
            ppu_y = _get_points_per_unit_vector(context, matrix, vector=(0, 1))
            if ppu_y <= EPSILON:
                self._log_debug(f"zero points per unit vector for {comp_id}, using fallback")
                required_point_size = target_data_height
            else:
                required_point_size = target_data_height * ppu_y

        local_corners = np.array(
            [
                [0, 0, 1],
                [component_data_width, 0, 1],
                [0, component_data_height, 1],
                [component_data_width, component_data_height, 1],
                [component_data_width / 2, 0, 1],
                [component_data_width / 2, component_data_height, 1],
                [0, component_data_height / 2, 1],
                [component_data_width, component_data_height / 2, 1],
                [component_data_width / 2, component_data_height / 2, 1],
            ]
        ).T
        world_corners = (matrix @ local_corners).T
        # local coords have y increasing upward (y=0 is bottom, y=height is top)
        world_bottom_left = world_corners[0, :2]
        world_top_left = world_corners[2, :2]
        world_top_right = world_corners[3, :2]
        world_bottom_center = world_corners[4, :2]
        world_top_center = world_corners[5, :2]
        world_middle_left = world_corners[6, :2]
        world_middle_right = world_corners[7, :2]
        world_center_center = world_corners[8, :2]

        align = text_component.align
        va = text_component.vertical_align
        anchor_x, anchor_y = world_top_left[0], world_top_left[1]
        mpl_ha, mpl_va = "left", "top"

        if align == "left":
            mpl_ha = "left"
        elif align == "center":
            mpl_ha = "center"
        elif align == "right":
            mpl_ha = "right"

        if va == "top":
            mpl_va = "top"
        elif va == "middle":
            # mpl uses 'center' for vertical middle
            mpl_va = "center"
        elif va == "bottom":
            mpl_va = "bottom"
        elif va == "baseline":
            mpl_va = "baseline"

        if mpl_ha == "left":
            anchor_x = world_middle_left[0] if mpl_va == "center" else world_top_left[0]
        elif mpl_ha == "center":
            anchor_x = world_center_center[0] if mpl_va == "center" else world_top_center[0]
        elif mpl_ha == "right":
            anchor_x = world_middle_right[0] if mpl_va == "center" else world_top_right[0]

        if mpl_va == "top":
            anchor_y = world_top_center[1] if mpl_ha == "center" else world_top_left[1]
        elif mpl_va == "center":
            anchor_y = world_center_center[1] if mpl_ha == "center" else world_middle_left[1]
        elif mpl_va == "bottom":
            anchor_y = world_bottom_center[1] if mpl_ha == "center" else world_bottom_left[1]
        elif mpl_va == "baseline":
            anchor_y = world_bottom_center[1] if mpl_ha == "center" else world_bottom_left[1]

        rotation_deg = _get_rotation_from_matrix(matrix)
        props = build_font_properties(
            font_name=text_component.font_name,
            font_weight=text_component.font_weight,
            font_style=text_component.font_style,
        )

        opacity = getattr(text_component, "opacity", 1.0)
        text_color = _apply_opacity(text_component.color, opacity) if text_component.color else None
        with _no_autoscale(context):
            text_artist = context.text(
                anchor_x,
                anchor_y,
                text_component.effective_text,
                fontsize=required_point_size,
                color=text_color,
                ha=mpl_ha,
                va=mpl_va,
                rotation=rotation_deg,
                fontproperties=props,
                linespacing=1.0 + text_component.line_spacing,
                rotation_mode="anchor",
            )
            if text_component.halo and text_component.halo.width > 0:
                import matplotlib.patheffects as mpe

                text_artist.set_path_effects(
                    [
                        mpe.withStroke(
                            linewidth=text_component.halo.width,
                            foreground=text_component.halo.color,
                        )
                    ]
                )
            if text_component.font_size_mode == "data":
                self._data_font_texts.append((text_artist, target_data_height, matrix.copy()))

    def _render_text_as_paths(self, context: Axes, text_comp: Text, matrix: np.ndarray):
        comp_id = text_comp.id or "unknown_text_path"
        if not text_comp.effective_text or not text_comp.show or not text_comp._text_metrics_cache:
            self._log_debug(f"_render_text_as_paths skipped for {comp_id}: missing info")
            return

        if text_comp.font_size_mode == "points":
            required_point_size = max(0.1, text_comp.font_size)
            ppu_y = _get_points_per_unit_vector(context, matrix, vector=(0, 1))
        else:
            target_data_height = text_comp.font_size
            ppu_y = _get_points_per_unit_vector(context, matrix, vector=(0, 1))
            if ppu_y <= EPSILON:
                self._log_debug(
                    f"skipping render path '{comp_id}': zero points per vertical data unit"
                )
                return
            required_point_size = max(0.1, target_data_height * ppu_y)

        if (
            text_comp._render_path_cache is None
            or text_comp._render_path_cache[0] != required_point_size
        ):
            props = build_font_properties(
                font_name=text_comp.font_name,
                font_weight=text_comp.font_weight,
                font_style=text_comp.font_style,
            )
            mpl_render_path = TextPath(
                (0, 0), text_comp.effective_text, size=required_point_size, prop=props
            )
            text_comp._render_path_cache = (required_point_size, mpl_render_path)
        else:
            mpl_render_path = text_comp._render_path_cache[1]

        if mpl_render_path.vertices.shape[0] == 0:
            return

        path_xmin, path_ymin = np.min(mpl_render_path.vertices, axis=0)
        path_xmax, path_ymax = np.max(mpl_render_path.vertices, axis=0)
        path_w_pts = path_xmax - path_xmin
        path_h_pts = path_ymax - path_ymin

        ppu_x = _get_points_per_unit_vector(context, matrix, vector=(1, 0))
        point_scale_x = ppu_x if ppu_x > EPSILON else 1.0
        point_scale_y = ppu_y if ppu_y > EPSILON else 1.0

        comp_w = text_comp._dimensions.width
        comp_h = text_comp._dimensions.height
        align = text_comp.align
        va = text_comp.vertical_align

        path_w_data = path_w_pts / point_scale_x if point_scale_x > EPSILON else 0
        path_h_data = path_h_pts / point_scale_y if point_scale_y > EPSILON else 0

        local_dx, local_dy = 0.0, 0.0
        if align == "center":
            local_dx = (comp_w - path_w_data) / 2.0
        elif align == "right":
            local_dx = comp_w - path_w_data
        # TextPath starts near baseline-left (roughly ymin=0 in local data units)
        if va == "top":
            local_dy = comp_h - path_h_data
        elif va == "middle":
            local_dy = (comp_h - path_h_data) / 2.0
        elif va == "bottom":
            local_dy = 0.0

        offset_mat = np.array([[1, 0, local_dx], [0, 1, local_dy], [0, 0, 1]])
        path_world_matrix = matrix @ offset_mat
        path_transform = mtransforms.Affine2D(matrix=path_world_matrix) + context.transData

        with _no_autoscale(context):
            patch = mpatches.PathPatch(
                mpl_render_path,
                facecolor=text_comp.color,
                edgecolor="none",
                linewidth=0,
                transform=path_transform,
            )
            if text_comp.halo and text_comp.halo.width > 0:
                import matplotlib.patheffects as mpe

                patch.set_path_effects(
                    [
                        mpe.withStroke(
                            linewidth=text_comp.halo.width,
                            foreground=text_comp.halo.color,
                        )
                    ]
                )
            context.add_patch(patch)

    def render_debug(self, context: Axes, component: Component, matrix: np.ndarray):
        if (
            not hasattr(component, "_dimensions")
            or component._dimensions.width <= 0
            or component._dimensions.height <= 0
        ):
            return
        w, h = component._dimensions.width, component._dimensions.height
        transform = mtransforms.Affine2D(matrix=matrix) + context.transData
        lw = 0.5
        with _no_autoscale(context):
            context.add_patch(
                mpatches.Rectangle(
                    (0, 0), w, h, fill=False, ec="red", ls="--", lw=lw, transform=transform
                )
            )
            # origin crosshair drawn in display coords
            origin_world = (matrix @ [0, 0, 1])[:2]
            origin_disp = context.transData.transform(origin_world)
            sz = 3
            context.add_line(
                plt.Line2D(
                    [origin_disp[0] - sz, origin_disp[0] + sz],
                    [origin_disp[1], origin_disp[1]],
                    color="red",
                    lw=lw,
                    ls="-",
                    transform=None,
                    solid_capstyle="butt",
                )
            )
            context.add_line(
                plt.Line2D(
                    [origin_disp[0], origin_disp[0]],
                    [origin_disp[1] - sz, origin_disp[1] + sz],
                    color="red",
                    lw=lw,
                    ls="-",
                    transform=None,
                    solid_capstyle="butt",
                )
            )

    def render_to_output(self, context: Axes, output=None, **kwargs):
        # refresh so data-unit linewidths and font sizes are final before save
        self._refresh_data_units()
        if not hasattr(context, "figure"):
            raise ValueError("matplotlib context (Axes) must belong to a figure")
        opts = {"bbox_inches": "tight", "pad_inches": 0.1, **kwargs}
        if output:
            try:
                context.figure.savefig(output, **opts)
                self._log_debug(f"saved figure to {output}")
            except Exception as e:
                self._log_debug(f"error saving figure: {e}")
                raise
        return context.figure
