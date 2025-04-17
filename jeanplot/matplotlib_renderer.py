# File: jeanplot/matplotlib_renderer.py
# -*- coding: utf-8 -*-
"""matplotlib rendering backend for jeanplot."""

from typing import Optional, Any, List, Dict, Tuple, Union, Literal, TextIO, BinaryIO
import numpy as np
import matplotlib.pyplot as plt
from svgpath2mpl import parse_path
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

# use absolute imports
from jeanplot.component import Component
from jeanplot.container import Container
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
    normalize_color,
)
from jeanplot.debug import debug_print, get_logger
from jeanplot.connector import Connection
from jeanplot.text import Text


logger = get_logger(__name__)
EPSILON = 1e-9  # small tolerance for float comparisons


# --- Helper Functions ---
def _get_point_scale_factor(axis: Axes) -> float:
    """calculates axis points per data unit based on current zoom"""
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
        elif dx_disp < EPSILON:
            avg_pixels_per_data_unit = dy_disp
        elif dy_disp < EPSILON:
            avg_pixels_per_data_unit = dx_disp
        else:
            avg_pixels_per_data_unit = (dx_disp + dy_disp) / 2.0

        points_per_pixel = 72.0 / fig.dpi
        points_per_data_unit = avg_pixels_per_data_unit * points_per_pixel
        # logger.debug(f"_get_point_scale_factor: ppd={points_per_data_unit:.3f}")
        return points_per_data_unit
    except Exception:
        # logger.warning(f"error calculating point scale factor: {e}") # too verbose
        return 1.0


def _linewidth_in_points(scaled_linewidth_data: float, axis: Axes) -> float:
    """converts a pre-scaled data-unit linewidth to points using axis zoom"""
    if scaled_linewidth_data <= 0:
        return 0.0
    axis_scale_factor = _get_point_scale_factor(axis)
    if axis_scale_factor < EPSILON:
        return 0.0  # avoid division by zero if axis scale is zero

    points = scaled_linewidth_data * axis_scale_factor
    final_points = max(0.0, points)
    # logger.debug(f"_linewidth_in_points: data={scaled_linewidth_data:.2f}, scale={axis_scale_factor:.2f} -> pts={final_points:.2f}")
    return final_points


def _get_matrix_avg_scale(matrix: np.ndarray) -> float:
    """extracts average scale factor from a 3x3 affine matrix"""
    a, c = matrix[0, 0], matrix[1, 0]  # How (1,0) transforms
    b, d = matrix[0, 1], matrix[1, 1]  # How (0,1) transforms
    scale_x = np.sqrt(a**2 + c**2)
    scale_y = np.sqrt(b**2 + d**2)
    avg_scale = (scale_x + scale_y) / 2.0
    return avg_scale if avg_scale > EPSILON else 0.0


def _get_mpl_linestyle(
    path_data: SVGPathData,
) -> Union[str, Tuple[float, Optional[Tuple[float, ...]]]]:
    """maps linestyle types/dash arrays to matplotlib format"""
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
        # store (patch, scaled_width_data) for dynamic updates
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
                    # self._log_debug("disconnected draw event")
                except Exception:
                    pass  # ignore errors
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
            # self._log_debug(f"connected draw event (cid={self._draw_event_cid})")
        else:
            self._log_debug("warning: could not connect draw event (no figure/canvas)")

        return self._context

    def refresh_linewidths(self, event=None):
        """callback to update linewidths based on current zoom/scale."""
        if not self._context or not self._data_width_patches:
            return

        # self._log_debug(f"refresh_linewidths triggered, {len(self._data_width_patches)} patches")
        patches_to_remove = []
        for i, (patch, scaled_width_data) in enumerate(self._data_width_patches):
            if (
                patch is None
                or not hasattr(patch, "figure")
                or patch.figure is None
                or patch.axes is not self._context
            ):
                patches_to_remove.append(i)
                continue

            if hasattr(patch, "set_linewidth"):
                try:
                    # convert the *already matrix-scaled* data width using current axis scale
                    new_width_points = _linewidth_in_points(scaled_width_data, self._context)
                    current_lw = patch.get_linewidth()
                    if not np.isclose(current_lw, new_width_points, atol=0.05):
                        patch.set_linewidth(new_width_points)
                        # self._log_debug(f"  patch {i}: updating lw to {new_width_points:.2f} (scaled_data={scaled_width_data:.2f})")
                except Exception as e:
                    # self._log_debug(f"  error updating patch {i}: {e}")
                    pass
            else:
                patches_to_remove.append(i)

        if patches_to_remove:
            patches_to_remove.sort(reverse=True)
            # self._log_debug(f"  removing {len(patches_to_remove)} patches")
            for index in patches_to_remove:
                self._data_width_patches.pop(index)

    def track_patch(self, patch: mpatches.Patch, scaled_width_data: float):
        """adds patch with its matrix-scaled data width for dynamic axis updates."""
        # self._log_debug(f"track_patch: id={id(patch)} scaled_data_width={scaled_width_data:.3f}")
        self._data_width_patches.append((patch, scaled_width_data))

    # --- Primitive Rendering Methods --- (render_rectangle, render_svg, render_text etc. assumed to be here)
    # ... (other methods like _get_recursive_world_bounds, _adjust_limits, render_rectangle etc.) ...

    def render_path(
        self,
        context: Axes,
        path_data: SVGPathData,
        matrix: np.ndarray,
        line_width_mode: str = "data",
        color_remap: Optional[Dict[str, Optional[str]]] = None,
        component_id: Optional[str] = None,  # for logging
    ):
        """renders a single svg path, handling linewidth scaling."""
        comp_id_str = component_id or "unknown"

        try:
            mpl_path = parse_path(path_data.d)
            # combine component matrix with potential path transform string
            final_matrix = matrix
            if path_data.transform:
                # simple matrix transform parsing (more robust parsing could be added)
                m = re.search(r"matrix\((.+)\)", path_data.transform)
                if m:
                    vals = [float(v.strip()) for v in m.group(1).split(",")]
                    if len(vals) == 6:
                        transform_mat = np.array(
                            [[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]], [0, 0, 1]]
                        )
                        final_matrix = final_matrix @ transform_mat

            transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData

            # apply color remapping
            remap = color_remap or {}
            fill_color_in = path_data.fill
            stroke_color_in = path_data.stroke
            fill_color_out = remap.get(fill_color_in, fill_color_in) if fill_color_in else None
            stroke_color_out = (
                remap.get(stroke_color_in, stroke_color_in) if stroke_color_in else None
            )

            final_facecolor = "none" if fill_color_out is None else fill_color_out
            final_edgecolor = "none" if stroke_color_out is None else stroke_color_out

            # --- Linewidth calculation ---
            initial_lw_points = 0.0
            scaled_width_data_for_tracking = 0.0  # value to track for axis zoom updates
            is_data_width = (
                line_width_mode == "data"
                and final_edgecolor != "none"
                and path_data.stroke_width > 0
            )

            if is_data_width:
                # 1. get scale factor from the transformation matrix
                matrix_scale = _get_matrix_avg_scale(final_matrix)
                self._log_debug(f"  matrix scale for path '{comp_id_str}': {matrix_scale:.3f}")

                # 2. apply matrix scale to the base data width
                base_width_data = path_data.stroke_width
                scaled_width_data = base_width_data * matrix_scale
                scaled_width_data_for_tracking = scaled_width_data  # track this value

                # 3. convert the matrix-scaled data width to points using current axis scale
                initial_lw_points = _linewidth_in_points(scaled_width_data, context)
                self._log_debug(
                    f"  data lw calc: base={base_width_data:.2f} * mat_scl={matrix_scale:.2f} -> scaled_data={scaled_width_data:.2f} -> initial_pts={initial_lw_points:.2f}"
                )
            elif final_edgecolor != "none" and path_data.stroke_width > 0:
                # point mode: use stroke_width directly
                initial_lw_points = path_data.stroke_width
                self._log_debug(f"  point lw: {initial_lw_points:.2f}")

            # --- End Linewidth Calculation ---

            linestyle = _get_mpl_linestyle(path_data)

            with _no_autoscale(context):
                patch = mpatches.PathPatch(
                    mpl_path,
                    facecolor=final_facecolor,
                    edgecolor=final_edgecolor,
                    linewidth=initial_lw_points,  # set initial value
                    linestyle=linestyle,
                    transform=transform,
                    capstyle="round",
                    joinstyle="round",
                )
                context.add_patch(patch)

                # if using data width, track the patch with the *matrix-scaled* data width
                if is_data_width:
                    self.track_patch(patch, scaled_width_data_for_tracking)
                    # no need to set lw again here, it's done initially

        except Exception as e:
            path_preview = path_data.d[:30] + "..." if path_data.d else "N/A"
            self._log_debug(f"error rendering path '{path_preview}' for '{comp_id_str}': {e}", e)

    def _create_rounded_rect_path(self, x, y, w, h, radius):
        """helper to create a matplotlib path for a rounded rectangle."""
        # handle zero radius (sharp corners)
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
            # clamp radius to half of smallest dimension
            r = min(radius, min(w, h) / 2.0)
            # build path segments (move, line, curve, line, curve...)
            v, c = [], []
            # start after top-left curve
            v.append((x + r, y + h))
            c.append(MplPath.MOVETO)
            # top edge
            v.append((x + w - r, y + h))
            c.append(MplPath.LINETO)
            # top-right curve (bezier approximation)
            _, cp1, cp2, p = arc_to_bezier(x + w - r, y + h - r, r, 90, 0)
            v.extend([cp1, cp2, p])
            c.extend([MplPath.CURVE4] * 3)
            # right edge
            v.append((x + w, y + r))
            c.append(MplPath.LINETO)
            # bottom-right curve
            _, cp1, cp2, p = arc_to_bezier(x + w - r, y + r, r, 0, -90)
            v.extend([cp1, cp2, p])
            c.extend([MplPath.CURVE4] * 3)
            # bottom edge
            v.append((x + r, y))
            c.append(MplPath.LINETO)
            # bottom-left curve
            _, cp1, cp2, p = arc_to_bezier(x + r, y + r, r, -90, -180)
            v.extend([cp1, cp2, p])
            c.extend([MplPath.CURVE4] * 3)
            # left edge
            v.append((x, y + h - r))
            c.append(MplPath.LINETO)
            # top-left curve
            _, cp1, cp2, p = arc_to_bezier(x + r, y + h - r, r, -180, -270)  # or 180 to 90
            v.extend([cp1, cp2, p])
            c.extend([MplPath.CURVE4] * 3)
            # close
            c.append(MplPath.CLOSEPOLY)
            v.append(v[0])  # explicitly close path for matplotlib
            verts, codes = v, c

        return MplPath(verts, codes)

    def render_rectangle(
        self, context: Axes, bounds: Size, style: BoxStyle, matrix: np.ndarray, component=None
    ):
        """renders a rectangle, possibly with rounded corners and shadow."""
        comp_id = getattr(component, "id", "unknown_rect")
        w, h = bounds.width, bounds.height
        if w <= 0 or h <= 0:
            # self._log_debug(f"skipping render_rectangle for {comp_id}: zero size")
            return

        transform = mtransforms.Affine2D(matrix=matrix) + context.transData

        # render shadow (if defined)
        if style.shadow and style.shadow.blur_radius > 0:
            shadow = style.shadow
            scale_factor = _get_point_scale_factor(context)
            points_per_unit = max(scale_factor, 1.0)
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

        # render main rectangle (background and border)
        facecolor = style.background_color or "none"
        edgecolor = style.border_color or "none"
        linewidth_data = style.border_width
        width_mode = style.border_width_mode

        if facecolor != "none" or (edgecolor != "none" and linewidth_data > 0):
            main_path = self._create_rounded_rect_path(0, 0, w, h, style.corner_radius)

            initial_lw_points = 0.0
            scaled_width_data_for_tracking = 0.0
            is_data_width = width_mode == "data" and edgecolor != "none" and linewidth_data > 0

            if is_data_width:
                matrix_scale = _get_matrix_avg_scale(matrix)
                scaled_width_data = linewidth_data * matrix_scale
                scaled_width_data_for_tracking = scaled_width_data
                initial_lw_points = _linewidth_in_points(scaled_width_data, context)
            elif edgecolor != "none" and linewidth_data > 0:
                initial_lw_points = linewidth_data  # point mode

            border_path_data = SVGPathData(
                d="",
                line_style=style.border_style,
                dash_array=style.dash_sequence,
                dash_offset=style.dash_offset,
            )
            linestyle = _get_mpl_linestyle(border_path_data)

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
                    self.track_patch(main_patch, scaled_width_data_for_tracking)

    def render_svg(self, context: Axes, svg_element: SVGElement, matrix: np.ndarray):
        """renders the paths within an svg element."""
        comp_id = svg_element.id or "unknown_svg"
        if not svg_element._parsed_svg_content:
            logger.warning(f"svg content not parsed for {comp_id}, attempting parse.")
            svg_element._parse_and_validate_svg()
            if not svg_element._parsed_svg_content:
                if svg_element.debug:
                    self.render_debug(context, svg_element, matrix)
                return

        if not svg_element._parsed_svg_content.paths:
            if svg_element.debug:
                self.render_debug(context, svg_element, matrix)
            return

        svg_data = svg_element._parsed_svg_content
        viewBox = svg_data.viewBox
        svg_dims = Size(svg_data.width, svg_data.height)
        comp_dims = svg_element._dimensions

        if comp_dims.width <= EPSILON or comp_dims.height <= EPSILON:
            if svg_element.debug:
                self.render_debug(context, svg_element, matrix)
            return

        vb_x, vb_y, vb_w, vb_h = viewBox or (0, 0, svg_dims.width, svg_dims.height)
        vb_w = max(vb_w, EPSILON)
        vb_h = max(vb_h, EPSILON)

        scale_x = comp_dims.width / vb_w
        scale_y = comp_dims.height / vb_h

        # transform: viewbox coords -> component local coords (y-up)
        svg_internal_matrix = np.array(
            [
                [scale_x, 0, -scale_x * vb_x],
                [0, -scale_y, scale_y * vb_y + comp_dims.height],  # flip y
                [0, 0, 1],
            ]
        )
        final_matrix = matrix @ svg_internal_matrix

        default_lw_mode = svg_element.line_width_mode
        color_remap = svg_element.color_remap

        for path_data in svg_data.paths:
            self.render_path(
                context, path_data, final_matrix, default_lw_mode, color_remap, component_id=comp_id
            )

        if svg_element.debug:
            self.render_debug(context, svg_element, matrix)

    # ... (render_text, measure_text, render_debug, render_to_output, _get_recursive_world_bounds, _adjust_limits assumed to be here and correct) ...
    # --- (Rest of the MatplotlibRenderer class methods...) ---

    def render_component(self, context: Axes, component: Component, adjust_lims: bool = True):
        """render a component tree into the matplotlib axes."""
        if self._context != context:
            self._log_debug("context changed, updating and reconnecting event")
            self._disconnect_draw_event()
            self._context = context
            fig = context.get_figure()
            if fig and fig.canvas:
                self._draw_event_cid = fig.canvas.mpl_connect("draw_event", self.refresh_linewidths)
                # self._log_debug(f"reconnected draw event (cid={self._draw_event_cid})")

        component.measure_and_layout(self)

        if adjust_lims and component.parent is None:
            self._adjust_limits(context, component)

        for cb in self.pre_render_callbacks:
            cb(context)

        self._data_width_patches = []  # clear before render

        root_world_matrix = component.compute_world_matrix()
        component.render(self, context, root_world_matrix)

        for cb in self.post_render_callbacks:
            cb(context)

        self.refresh_linewidths()  # initial update

    def _get_recursive_world_bounds(
        self, component: Component, current_bounds=None
    ) -> Optional[Tuple[float, float, float, float]]:
        """recursively find the min/max world coordinates occupied by visible components."""
        if not component or not component.show:
            return current_bounds

        overall = (
            list(current_bounds)
            if current_bounds
            else [float("inf"), float("inf"), float("-inf"), float("-inf")]
        )

        if not isinstance(component, Connection):
            comp_b = component.get_world_bounds()
            if comp_b:
                overall = [
                    min(overall[0], comp_b[0]),
                    min(overall[1], comp_b[1]),
                    max(overall[2], comp_b[2]),
                    max(overall[3], comp_b[3]),
                ]

        # recurse into children
        children_to_check = []
        if hasattr(component, "children"):
            children_to_check.extend(component.children)
        if hasattr(component, "anchor_points"):
            children_to_check.extend(component.anchor_points)

        for child in children_to_check:
            if not child or not child.show:
                continue

            child_bounds = None
            if isinstance(child, Connection):
                # connection bounds estimation (pre-render)
                conn_points_est = []
                child._resolve_references_orig()
                start_comp_orig = child._resolved_start_component_orig
                end_comp_orig = child._resolved_end_component_orig
                if start_comp_orig and end_comp_orig:
                    est_start = child._get_offset_world_position(
                        start_comp_orig, child.start_offset
                    )
                    if est_start:
                        conn_points_est.append(est_start)
                    est_end = child._get_offset_world_position(end_comp_orig, child.end_offset)
                    if est_end:
                        conn_points_est.append(est_end)

                if len(conn_points_est) >= 2:
                    buffer = max(child.line_width * 3, 10)
                    min_x = min(p[0] for p in conn_points_est) - buffer
                    min_y = min(p[1] for p in conn_points_est) - buffer
                    max_x = max(p[0] for p in conn_points_est) + buffer
                    max_y = max(p[1] for p in conn_points_est) + buffer
                    child_bounds = (min_x, min_y, max_x, max_y)
                else:
                    comp_b = child.get_world_bounds()  # fallback rough bounds
                    if comp_b:
                        child_bounds = comp_b
            elif hasattr(child, "children") or hasattr(
                child, "anchor_points"
            ):  # recurse into containers/anchors
                child_bounds = self._get_recursive_world_bounds(child, None)
            else:  # basic component bounds
                child_bounds = child.get_world_bounds()

            if child_bounds:
                overall = [
                    min(overall[0], child_bounds[0]),
                    min(overall[1], child_bounds[1]),
                    max(overall[2], child_bounds[2]),
                    max(overall[3], child_bounds[3]),
                ]

        return tuple(overall) if overall[0] != float("inf") else None

    def _adjust_limits(self, context: Axes, root: Component, padding: float = 0.1):
        """set axis limits to encompass the rendered content."""
        bounds = self._get_recursive_world_bounds(root)
        if bounds:
            min_x, min_y, max_x, max_y = bounds
            width = max(max_x - min_x, 1.0)
            height = max(max_y - min_y, 1.0)
            pad_x_base = max(width * padding, 5.0)
            pad_y_base = max(height * padding, 5.0)

            is_schematic_type = (
                "Schematic" in root.__class__.__name__ or "Diagram" in root.__class__.__name__
            )
            children_to_check = getattr(root, "children", []) + getattr(root, "anchor_points", [])
            has_connections = any(isinstance(c, Connection) for c in children_to_check)

            if is_schematic_type or has_connections:
                pad_x = max(pad_x_base, width * 0.15, 20.0)
                pad_y = max(pad_y_base, height * 0.15, 20.0)
            else:
                pad_x = pad_x_base
                pad_y = pad_y_base

            context.set_xlim(min_x - pad_x, max_x + pad_x)
            context.set_ylim(min_y - pad_y, max_y + pad_y)
            # self._log_debug(f"adjusted limits: x=({min_x-pad_x:.1f}, {max_x+pad_x:.1f}), y=({min_y-pad_y:.1f}, {max_y+pad_y:.1f})")
        else:
            context.set_xlim(0, 100)
            context.set_ylim(0, 100)
            # self._log_debug("no valid bounds found, setting default limits (0,100).")

        context.set_aspect("equal", adjustable="box")

    def render_text(self, context: Axes, text_component: Text, matrix: np.ndarray):
        """renders text using cached svg paths."""
        comp_id = text_component.id or "unknown_text"
        if not text_component.text or not text_component.show:
            return

        if text_component._svg_cache is None:
            self._log_debug(f"text '{comp_id}' svg_cache is None, measuring now.")
            self.measure_text(text_component)

        svg_cache = text_component._svg_cache
        if isinstance(svg_cache, SVGTextContent) and svg_cache.glyph_paths:
            self._render_text_paths(context, text_component, matrix)
        else:
            self._log_debug(f"no valid glyph paths found for text '{comp_id}', skipping render.")
            if text_component.debug:
                self.render_debug(context, text_component, matrix)

    def _render_text_paths(self, context: Axes, text_comp: Text, matrix: np.ndarray):
        """renders text using the pre-calculated matplotlib paths."""
        comp_id = text_comp.id or "unknown_text_path"
        svg_content = text_comp._svg_cache
        if not svg_content or not svg_content.glyph_paths:
            return

        alloc_w, alloc_h = text_comp._dimensions.width, text_comp._dimensions.height
        measured_w, measured_h = svg_content.measured_width, svg_content.measured_height

        dx, dy = 0.0, 0.0
        if text_comp.align == "center":
            dx = (alloc_w - measured_w) / 2.0
        elif text_comp.align == "right":
            dx = alloc_w - measured_w
        if text_comp.vertical_align == "middle":
            dy = (alloc_h - measured_h) / 2.0
        elif text_comp.vertical_align == "top":
            dy = alloc_h - measured_h

        align_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        final_matrix = matrix @ align_matrix
        transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData

        color = getattr(text_comp, "color", "black") or "black"

        with _no_autoscale(context):
            for path_info in svg_content.glyph_paths:
                mpl_path = path_info.get("path")
                if mpl_path:
                    context.add_patch(
                        mpatches.PathPatch(
                            mpl_path,
                            facecolor=color,
                            edgecolor="none",
                            linewidth=0,
                            transform=transform,
                        )
                    )

    def measure_text(self, text_comp: Text) -> Size:
        """measures text by generating matplotlib TextPaths."""
        if not text_comp.text:
            text_comp._svg_cache = None
            return Size()

        props = fm.FontProperties(
            family=text_comp.font_name or "sans-serif",
            weight=text_comp.font_weight,
            style=text_comp.font_style,
        )
        lines = text_comp.text.split("\n")
        all_glyph_paths = []
        y_cursor = 0.0
        min_x_overall, max_x_overall = float("inf"), float("-inf")
        min_y_overall, max_y_overall = float("inf"), float("-inf")
        base_line_h = text_comp.font_size
        effective_line_height = base_line_h * (1.0 + text_comp.line_spacing)

        for i, line in enumerate(lines):
            if not line.strip():
                if i < len(lines) - 1:
                    y_cursor -= effective_line_height
                continue

            mpl_path = TextPath((0, 0), line, size=text_comp.font_size, prop=props)
            all_glyph_paths.append({"path": mpl_path, "y_offset": y_cursor})

            if mpl_path.vertices.shape[0] > 0:
                line_vertices = mpl_path.vertices + [0, y_cursor]
                min_x_overall = min(min_x_overall, np.min(line_vertices[:, 0]))
                max_x_overall = max(max_x_overall, np.max(line_vertices[:, 0]))
                min_y_overall = min(min_y_overall, np.min(line_vertices[:, 1]))
                max_y_overall = max(max_y_overall, np.max(line_vertices[:, 1]))
            else:
                max_x_overall = max(max_x_overall, 0)
                min_x_overall = min(min_x_overall, 0)

            if i < len(lines) - 1:
                y_cursor -= effective_line_height

        if not all_glyph_paths:
            text_comp._svg_cache = None
            return Size()

        if min_x_overall == float("inf"):
            measured_width, measured_height, final_paths = 0.0, 0.0, []
        else:
            measured_width = max(0, max_x_overall - min_x_overall)
            measured_height = max(0, max_y_overall - min_y_overall)
            final_paths = []
            y_shift = -min_y_overall
            x_shift = -min_x_overall
            for p_info in all_glyph_paths:
                final_transform = mtransforms.Affine2D().translate(
                    x_shift, p_info["y_offset"] + y_shift
                )
                final_paths.append({"path": p_info["path"].transformed(final_transform)})

        text_comp._svg_cache = SVGTextContent(
            glyph_paths=tuple(final_paths),
            measured_width=measured_width,
            measured_height=measured_height,
        )
        measured_size = Size(width=measured_width, height=measured_height)
        # self._log_debug(f"measured text '{text_comp.id}', nat_size={measured_size}")
        return measured_size

    def render_debug(self, context: Axes, component: Component, matrix: np.ndarray):
        """renders debug bounding box and origin."""
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
        """saves the rendered figure to a file or stream."""
        self.refresh_linewidths()
        if not hasattr(context, "figure"):
            raise ValueError("matplotlib context (Axes) must belong to a figure")
        opts = {"bbox_inches": "tight", "pad_inches": 0.1, **kwargs}
        if output:
            try:
                # self._log_debug(f"saving figure to {output} with options {opts}")
                context.figure.savefig(output, **opts)
            except Exception as e:
                self._log_debug(f"error saving figure: {e}", e)
                raise
        return context.figure
