"""matplotlib rendering backend for jeanplot."""

from typing import Optional, Any, List, Dict, Tuple, Union, Literal, TextIO, BinaryIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MplPath  # avoid name clash with Pathlib
from matplotlib.colors import to_rgba
import logging
from contextlib import contextmanager
import re

from jeanplot.component import Component
from jeanplot.models import Size, BoxStyle, LineWidthMode
from jeanplot.renderer import BaseRenderer
from jeanplot.svg import (
    SVGElement,
    SVGTextContent,
    SVGPathData,
    arc_to_bezier,
    create_arrow_cap,
    create_circle_cap,
    create_flat_cap,
    LineEndFlat,
    LineEndArrow,
    LineEndCircle,
    _normalize_color,
)
from jeanplot.debug import debug_print
from jeanplot.connector import Connection
from jeanplot.text import Text

logger = logging.getLogger(__name__)


def _get_point_scale_factor(axis: Axes) -> float:
    fig = axis.get_figure()
    if not fig:
        return 1.0
    try:
        p0_disp = axis.transData.transform_point((0, 0))
        p1_disp_x = axis.transData.transform_point((1, 0))
        p1_disp_y = axis.transData.transform_point((0, 1))
        dx_disp = abs(p1_disp_x[0] - p0_disp[0])
        dy_disp = abs(p1_disp_y[1] - p0_disp[1])
        if dx_disp < 1e-6 and dy_disp < 1e-6:
            return 0.0
        avg_pixels_per_data_unit = (
            (dx_disp + dy_disp) / 2.0
            if (dx_disp > 1e-6 and dy_disp > 1e-6)
            else max(dx_disp, dy_disp)
        )
        points_per_pixel = 72.0 / fig.dpi
        points_per_data_unit = avg_pixels_per_data_unit * points_per_pixel
        return points_per_data_unit
    except Exception as e:
        return 1.0


def _linewidth_in_points(linewidth_data: float, axis: Axes) -> float:
    if linewidth_data <= 0:
        return 0.0
    scale_factor = _get_point_scale_factor(axis)
    points = linewidth_data * scale_factor
    final_points = max(0.0, points)
    return final_points


def _get_mpl_linestyle(
    path_data: SVGPathData,
) -> Union[str, Tuple[float, Optional[Tuple[float, ...]]]]:
    linestyle_map = {"solid": "-", "dashed": "--", "dotted": ":"}
    if path_data.dash_array and path_data.line_style == "custom":
        return (path_data.dash_offset, path_data.dash_array)
    return linestyle_map.get(path_data.line_style, "-")


@contextmanager
def _no_autoscale(ax):
    autoscale_state = ax.get_autoscale_on()
    ax.set_autoscale_on(False)
    try:
        yield
    finally:
        ax.set_autoscale_on(autoscale_state)


class MatplotlibRenderer(BaseRenderer):
    RENDERER_NAME = "matplotlib"

    def __init__(self, debug=False):
        super().__init__()
        self._data_width_patches: List[Tuple[mpatches.Patch, float]] = []
        self._context: Optional[Axes] = None
        self._draw_event_cid: Optional[int] = None

    def _log_debug(self, message: str, data: Any = None):
        debug_print(self.RENDERER_NAME, message, data)

    def _disconnect_draw_event(self):
        if self._draw_event_cid is not None and self._context is not None:
            fig = self._context.get_figure()
            if fig and fig.canvas:
                try:
                    fig.canvas.mpl_disconnect(self._draw_event_cid)
                except Exception as e:
                    pass
        self._draw_event_cid = None

    def create_context(
        self,
        width: float = 800,
        height: float = 600,
        dpi: int = 150,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Axes:
        self._disconnect_draw_event()
        self._data_width_patches = []
        if ax:
            self._context = ax
            fig = ax.get_figure()
        else:
            figsize = (width / dpi, height / dpi)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, **kwargs)
            self._context = ax
        if fig and fig.canvas:
            self._draw_event_cid = fig.canvas.mpl_connect("draw_event", self.refresh_linewidths)
        else:
            self._log_debug("warning: could not connect draw event (no figure/canvas)")
        return self._context

    def refresh_linewidths(self, event=None):
        if not self._context or not self._data_width_patches:
            return

        updated_count = 0
        patches_to_remove = []
        for i, (patch, width_data) in enumerate(self._data_width_patches):
            if not patch or not hasattr(patch, "figure") or patch.figure is None:
                patches_to_remove.append(i)
                continue
            if patch.axes is not self._context:
                patches_to_remove.append(i)
                continue
            if hasattr(patch, "set_linewidth"):
                try:
                    new_width_points = _linewidth_in_points(width_data, self._context)
                    current_lw = patch.get_linewidth()
                    if not np.isclose(current_lw, new_width_points, atol=0.05):
                        patch.set_linewidth(new_width_points)
                        updated_count += 1
                except Exception as e:
                    self._log_debug(f"  error updating patch {i}: {e}")
            else:
                patches_to_remove.append(i)

        if patches_to_remove:
            self._data_width_patches = [
                item
                for i, item in enumerate(self._data_width_patches)
                if i not in patches_to_remove
            ]

    def track_patch(self, patch: mpatches.Patch, width_data: float):
        self._log_debug(
            f"track_patch: tracking patch id={id(patch)} with data_width={width_data:.3f}"
        )
        self._data_width_patches.append((patch, width_data))

    def render_component(self, context: Axes, component: Component, adjust_lims: bool = True):
        if self._context != context:
            self._log_debug("context changed, updating and reconnecting event")
            self._disconnect_draw_event()
            self._context = context
            fig = context.get_figure()
            if fig and fig.canvas:
                self._draw_event_cid = fig.canvas.mpl_connect("draw_event", self.refresh_linewidths)

        component.measure_and_layout(self)

        if adjust_lims and component.parent is None:
            self._adjust_limits(context, component)

        for cb in self.pre_render_callbacks:
            cb(context)

        self._data_width_patches = []

        root_world_matrix = component.compute_world_matrix()

        component.render(self, context, root_world_matrix)

        for cb in self.post_render_callbacks:
            cb(context)

        self.refresh_linewidths()

    def _get_recursive_world_bounds(
        self, component: Component, current_bounds=None
    ) -> Optional[Tuple[float, float, float, float]]:
        if not component or not component.show:
            return current_bounds
        overall = (
            list(current_bounds) if current_bounds else [float("inf")] * 2 + [float("-inf")] * 2
        )
        comp_b = component.get_world_bounds()
        if comp_b:
            overall = [
                min(overall[0], comp_b[0]),
                min(overall[1], comp_b[1]),
                max(overall[2], comp_b[2]),
                max(overall[3], comp_b[3]),
            ]
        if hasattr(component, "children") and component.children:
            for child in component.children:
                updated = self._get_recursive_world_bounds(child, tuple(overall))
                if updated:
                    overall = list(updated)
        if hasattr(component, "anchor_points") and component.anchor_points:
            for anchor in component.anchor_points:
                anchor_b = anchor.get_world_bounds()
                if anchor_b:
                    overall = [
                        min(overall[0], anchor_b[0]),
                        min(overall[1], anchor_b[1]),
                        max(overall[2], anchor_b[2]),
                        max(overall[3], anchor_b[3]),
                    ]
        return tuple(overall) if overall[0] != float("inf") else None

    def _adjust_limits(self, context: Axes, root: Component, padding: float = 0.1):
        bounds = self._get_recursive_world_bounds(root) or (0, 0, 1, 1)
        min_x, min_y, max_x, max_y = bounds
        width, height = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)
        pad_x, pad_y = max(width * padding, 5), max(height * padding, 5)
        context.set_xlim(min_x - pad_x, max_x + pad_x)
        context.set_ylim(min_y - pad_y, max_y + pad_y)
        context.set_aspect("equal", adjustable="box")

    def render_path(
        self,
        context: Axes,
        path_data: SVGPathData,
        matrix: np.ndarray,
        line_width_mode: str = "data",
        color_remap: Optional[Dict[str, Optional[str]]] = None,
    ):
        try:
            from svgpath2mpl import parse_path
        except ImportError:
            logger.error("svgpath2mpl required")
            return

        try:
            mpl_path = parse_path(path_data.d)
            final_matrix = matrix
            if path_data.transform:
                m = re.search(r"matrix\((.+)\)", path_data.transform)
                if m:
                    try:
                        vals = [float(v.strip()) for v in m.group(1).split(",")]
                        if len(vals) == 6:
                            transform_mat = np.array(
                                [
                                    [vals[0], vals[2], vals[4]],
                                    [vals[1], vals[3], vals[5]],
                                    [0, 0, 1],
                                ]
                            )
                            final_matrix = final_matrix @ transform_mat
                    except ValueError:
                        pass

            transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData

            remap = color_remap or {}
            fill_color_in = path_data.fill
            stroke_color_in = path_data.stroke

            fill_color_out = remap.get(fill_color_in, fill_color_in) if fill_color_in else None
            stroke_color_out = (
                remap.get(stroke_color_in, stroke_color_in) if stroke_color_in else None
            )

            final_facecolor = "none" if fill_color_out is None else fill_color_out
            final_edgecolor = "none" if stroke_color_out is None else stroke_color_out

            current_path_lw_data = path_data.stroke_width
            is_data_width = (
                line_width_mode == "data" and final_edgecolor != "none" and current_path_lw_data > 0
            )
            initial_lw_points = 0.0 if is_data_width else current_path_lw_data
            linestyle = _get_mpl_linestyle(path_data)

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

                if is_data_width:
                    self.track_patch(patch, current_path_lw_data)
                    initial_points_estimate = _linewidth_in_points(current_path_lw_data, context)
                    patch.set_linewidth(initial_points_estimate)

        except Exception as e:
            self._log_debug(f"error rendering path '{path_data.d[:30]}...': {e}", e)

    def _create_rounded_rect_path(self, x, y, w, h, radius):
        if radius < 1e-3:
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

        if style.shadow and style.shadow.blur_radius > 0:
            shadow = style.shadow
            scale_factor = _get_point_scale_factor(context)
            points_per_unit = scale_factor
            num_layers = min(
                100, max(4, int((shadow.blur_radius * points_per_unit**0.75) * shadow.resolution))
            )
            try:
                base_color_rgb, base_alpha = to_rgba(shadow.color)[:3], to_rgba(shadow.color)[3]
            except ValueError:
                base_color_rgb, base_alpha = (0, 0, 0), 0.5
            accumulated_alpha = 0.0
            min_render_alpha = 1.0 / 256.0
            for i in range(num_layers - 1, -1, -1):
                layer_frac = (i / (num_layers - 1)) if num_layers > 1 else 1.0
                alpha_frac = ((num_layers - 1 - i) / (num_layers - 1)) if num_layers > 1 else 0.0
                target_intensity = (
                    base_alpha * (1 - alpha_frac**1.5) / num_layers
                    if num_layers > 0
                    else base_alpha
                )
                accumulated_alpha += target_intensity
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

        facecolor = style.background_color or "none"
        edgecolor = style.border_color or "none"
        linewidth_data = style.border_width
        width_mode = style.border_width_mode

        if facecolor != "none" or (edgecolor != "none" and linewidth_data > 0):
            main_path = self._create_rounded_rect_path(0, 0, w, h, style.corner_radius)
            initial_lw_points = 0.0
            is_data_width = width_mode == "data" and edgecolor != "none" and linewidth_data > 0
            if not is_data_width and edgecolor != "none":
                initial_lw_points = linewidth_data

            path_data = SVGPathData(
                d="",
                line_style=style.border_style,
                dash_array=style.dash_sequence,
                dash_offset=style.dash_offset,
            )
            linestyle = _get_mpl_linestyle(path_data)

            with _no_autoscale(context):
                main_patch = mpatches.PathPatch(
                    main_path,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=initial_lw_points,
                    linestyle=linestyle,
                    transform=transform,
                    capstyle="round",
                    joinstyle="round",
                )
                context.add_patch(main_patch)
                if is_data_width:
                    self.track_patch(main_patch, linewidth_data)
                    initial_points_estimate = _linewidth_in_points(linewidth_data, context)
                    main_patch.set_linewidth(initial_points_estimate)

    def render_svg(self, context: Axes, svg_element: SVGElement, matrix: np.ndarray):
        if not svg_element._parsed_svg_content or not svg_element._parsed_svg_content.paths:
            if svg_element.debug:
                # Render debug box using the component's world matrix 'matrix'
                self.render_debug(context, svg_element, matrix)
            return

        svg_data = svg_element._parsed_svg_content
        viewBox = svg_data.viewBox
        svg_dims = Size(svg_data.width, svg_data.height)
        comp_dims = svg_element._dimensions

        if comp_dims.width <= 1e-6 or comp_dims.height <= 1e-6:
            if svg_element.debug:
                self.render_debug(context, svg_element, matrix)
            return

        if viewBox:
            vb_x, vb_y, vb_w, vb_h = viewBox
        else:
            vb_x, vb_y = 0, 0
            vb_w, vb_h = svg_dims.width, svg_dims.height

        if vb_w <= 1e-6:
            vb_w = 1.0
        if vb_h <= 1e-6:
            vb_h = 1.0

        scale_x = comp_dims.width / vb_w
        scale_y = comp_dims.height / vb_h

        # combined transformation matrix: svg -> component local (y-up)
        svg_internal_matrix = np.array(
            [
                [scale_x, 0, -scale_x * vb_x],
                [0, -scale_y, scale_y * vb_y + comp_dims.height],
                [0, 0, 1],
            ]
        )

        # final transformation: world * internal
        final_matrix = matrix @ svg_internal_matrix

        default_lw_mode = svg_element.line_width_mode
        color_remap = svg_element.color_remap

        for path_data in svg_data.paths:
            # pass color_remap correctly
            self.render_path(context, path_data, final_matrix, default_lw_mode, color_remap)

        if svg_element.debug:
            self.render_debug(context, svg_element, matrix)

    def render_text(self, context: Axes, text_component: Text, matrix: np.ndarray):
        if not text_component.text:
            return
        svg_cache = getattr(text_component, "_svg_cache", None)
        if isinstance(svg_cache, SVGTextContent) and svg_cache.text_paths:
            self._render_text_paths(context, text_component, matrix)
        else:
            self._render_text_basic(context, text_component, matrix)

    def _render_text_paths(self, context: Axes, text_comp: Text, matrix: np.ndarray):
        svg_content = text_comp._svg_cache
        if not svg_content or not svg_content.text_paths:
            return
        alloc_w, alloc_h = text_comp._dimensions.width, text_comp._dimensions.height
        measured_w, measured_h = svg_content.measured_width, svg_content.measured_height
        dx = 0.0
        dy = 0.0
        if text_comp.align == "center":
            dx = (alloc_w - measured_w) / 2.0
        elif text_comp.align == "right":
            dx = alloc_w - measured_w

        if text_comp.vertical_align == "middle":
            dy = (alloc_h - measured_h) / 2.0
        elif text_comp.vertical_align == "bottom":
            dy = 0
        elif text_comp.vertical_align == "top":
            dy = alloc_h - measured_h

        align_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        final_matrix = matrix @ align_matrix
        transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData
        color = getattr(text_comp, "color", "black")
        with _no_autoscale(context):
            for path_info in svg_content.text_paths:
                context.add_patch(
                    mpatches.PathPatch(
                        path_info["path"], facecolor=color, edgecolor="none", transform=transform
                    )
                )

    def _render_text_basic(self, context: Axes, text_comp: Text, matrix: np.ndarray):
        props = {
            "fontsize": text_comp.font_size,
            "family": text_comp.font_name or "sans-serif",
            "weight": text_comp.font_weight,
            "style": text_comp.font_style,
            "color": text_comp.color,
            "ha": text_comp.align,
            "va": text_comp.vertical_align,
            "linespacing": 1.0 + text_comp.line_spacing,
            "transform": mtransforms.Affine2D(matrix=matrix) + context.transData,
            "clip_on": True,
        }
        x = 0.0
        y = text_comp._dimensions.height
        if props["ha"] == "center":
            x = text_comp._dimensions.width / 2.0
        elif props["ha"] == "right":
            x = text_comp._dimensions.width

        if props["va"] == "middle":
            y = text_comp._dimensions.height / 2.0
        elif props["va"] == "bottom":
            y = 0.0

        if props["va"] == "middle":
            props["va"] = "center_baseline"
        elif props["va"] == "top":
            props["va"] = "top"
        elif props["va"] == "bottom":
            props["va"] = "bottom"

        with _no_autoscale(context):
            context.text(x, y, text_comp.text, **props)

    def render_connection_curve(
        self,
        context: Axes,
        connection: Connection,
        local_start: Tuple[float, float],
        local_end: Tuple[float, float],
        local_control_points: List[Tuple[float, float]],
        path_string: str,
        matrix: np.ndarray,
    ):
        path_data = SVGPathData(
            d=path_string,
            stroke=connection.color,
            stroke_width=connection.line_width,
            fill="none",
            line_style=connection.line_style,
            dash_array=connection.dash_array,
            dash_offset=connection.dash_offset,
        )
        self.render_path(context, path_data, matrix, line_width_mode="data")

        if connection.start_cap or connection.end_cap:
            active_curve = connection._get_active_curve()
            try:
                start_dir, end_dir = active_curve.get_directions(
                    local_start, local_end, local_control_points
                )
                cap_funcs = {
                    LineEndArrow: create_arrow_cap,
                    LineEndCircle: create_circle_cap,
                    LineEndFlat: create_flat_cap,
                }
                if connection.start_cap and type(connection.start_cap) in cap_funcs:
                    start_cap_path = cap_funcs[type(connection.start_cap)](
                        local_start, start_dir, connection.start_cap
                    )
                    self.render_path(context, start_cap_path, matrix, line_width_mode="data")
                if connection.end_cap and type(connection.end_cap) in cap_funcs:
                    end_cap_path = cap_funcs[type(connection.end_cap)](
                        local_end, end_dir, connection.end_cap
                    )
                    self.render_path(context, end_cap_path, matrix, line_width_mode="data")
            except Exception as e:
                self._log_debug(f"error rendering end caps for {connection.id}: {e}")

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
            origin_world = (matrix @ [0, 0, 1])[:2]
            origin_disp = context.transData.transform(origin_world)
            sz = 4
            context.add_line(
                plt.Line2D(
                    [origin_disp[0] - sz, origin_disp[0] + sz],
                    [origin_disp[1], origin_disp[1]],
                    color="red",
                    lw=lw,
                    ls="-",
                    transform=None,
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
                )
            )

    def measure_text(self, text_comp: Text) -> Size:
        if not text_comp.text:
            text_comp._svg_cache = None
            return Size()
        props = fm.FontProperties(
            family=text_comp.font_name or "sans-serif",
            weight=text_comp.font_weight,
            style=text_comp.font_style,
        )
        lines = text_comp.text.split("\n")
        max_w = 0.0
        paths_info = []
        try:
            fig = plt.figure()
            renderer = fig.canvas.get_renderer()
        except Exception as e:
            return Size()

        total_h = 0.0
        min_y_overall, max_y_overall = float("inf"), float("-inf")
        y_cursor = 0.0

        for i, line in enumerate(lines):
            if not line.strip():
                p = TextPath((0, 0), " ", size=text_comp.font_size, prop=props)
                bbox = p.get_extents(renderer=renderer)
                line_h = bbox.height if bbox.height > 0 else text_comp.font_size
                line_min_y = bbox.y0
                line_max_y = bbox.y1
                paths_info.append(
                    {
                        "w": 0.0,
                        "h": line_h,
                        "min_y": line_min_y,
                        "max_y": line_max_y,
                        "p": p,
                        "t": line,
                        "y_pos": y_cursor,
                    }
                )
                max_w = max(max_w, 0.0)
            else:
                p = TextPath((0, 0), line, size=text_comp.font_size, prop=props)
                try:
                    bbox = p.get_extents(renderer=renderer)
                    line_w = bbox.width
                    line_h = bbox.height
                    line_min_y = bbox.y0
                    line_max_y = bbox.y1

                    paths_info.append(
                        {
                            "w": line_w,
                            "h": line_h,
                            "min_y": line_min_y,
                            "max_y": line_max_y,
                            "p": p,
                            "t": line,
                            "y_pos": y_cursor,
                        }
                    )
                    max_w = max(max_w, line_w)
                except Exception as e_bbox:
                    est_w = len(line) * text_comp.font_size * 0.6
                    est_h = text_comp.font_size
                    line_min_y = -est_h * 0.2
                    line_max_y = est_h * 0.8
                    paths_info.append(
                        {
                            "w": est_w,
                            "h": est_h,
                            "min_y": line_min_y,
                            "max_y": line_max_y,
                            "p": p,
                            "t": line,
                            "y_pos": y_cursor,
                        }
                    )
                    max_w = max(max_w, est_w)

            if paths_info:
                base_line_h = text_comp.font_size
                effective_line_height = base_line_h * (1.0 + text_comp.line_spacing)
                if i < len(lines) - 1:
                    y_cursor -= effective_line_height

        plt.close(fig)

        if not paths_info:
            return Size()

        for info in paths_info:
            min_y_overall = min(min_y_overall, info["y_pos"] + info["min_y"])
            max_y_overall = max(max_y_overall, info["y_pos"] + info["max_y"])
        measured_h = max(0, max_y_overall - min_y_overall)

        y_offset = -min_y_overall
        final_paths = []
        for info in paths_info:
            final_paths.append(
                {
                    "path": info["p"].transformed(
                        mtransforms.Affine2D().translate(0, info["y_pos"] + y_offset)
                    )
                }
            )

        text_comp._svg_cache = SVGTextContent(
            width=max_w,
            height=measured_h,
            viewBox=(0, 0, max_w, measured_h),
            paths=(),
            text_paths=tuple(final_paths),
            measured_width=max_w,
            measured_height=measured_h,
        )
        measured_size = Size(width=max_w, height=measured_h)
        return measured_size

    def render_to_output(self, context: Axes, output=None, **kwargs):
        self.refresh_linewidths()
        if not hasattr(context, "figure"):
            raise ValueError("context requires a figure")
        opts = {"bbox_inches": "tight", "pad_inches": 0.1, **kwargs}
        if output:
            try:
                context.figure.savefig(output, **opts)
            except Exception as e:
                self._log_debug(f"error saving figure: {e}", e)
                raise
        return context.figure
