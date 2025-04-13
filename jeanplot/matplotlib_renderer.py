"""Matplotlib rendering backend for jeanplot."""

from typing import Optional, Any, List, Dict, Tuple, Union, Literal, TextIO, BinaryIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MplPath  # avoid name clash with Pathlib
from pathlib import Path
from matplotlib.colors import to_rgba
import logging
from contextlib import contextmanager

# use absolute imports
from jeanplot.component import Component
from jeanplot.models import Size, BoxStyle, LineWidthMode
from jeanplot.renderer import BaseRenderer
from jeanplot.svg import (
    SVGElement,
    SVGContent,
    SVGTextContent,
    SVGPathData,
    arc_to_bezier,
    create_arrow_cap,
    create_circle_cap,
    create_flat_cap,
    LineEndType,
)
from jeanplot.debug import debug_print
from jeanplot.connector import Connection  # needed for type hint
from jeanplot.text import Text  # needed for type hint

logger = logging.getLogger(__name__)

# --- Matplotlib Helpers ---


def _linewidth_from_data_units(
    linewidth: float, axis: Axes, reference: Literal["x", "y", "avg"] = "avg"
) -> float:
    """convert linewidth in data units to points."""
    fig = axis.get_figure()
    if not fig:
        return linewidth  # fallback if no figure

    try:
        trans_data_to_disp = axis.transData.transform
        # transform 0 and linewidth along the reference axis
        p0_disp = trans_data_to_disp((0, 0))
        if reference == "x":
            p1_disp = trans_data_to_disp((linewidth, 0))
        elif reference == "y":
            p1_disp = trans_data_to_disp((0, linewidth))
        else:  # avg
            p1_disp_x = trans_data_to_disp((linewidth, 0))
            p1_disp_y = trans_data_to_disp((0, linewidth))
            avg_dx = p1_disp_x[0] - p0_disp[0]
            avg_dy = p1_disp_y[1] - p0_disp[1]
            dist_disp = np.sqrt(avg_dx**2 + avg_dy**2) / np.sqrt(2)  # geometric mean approx
            # need to convert display dist to points (72 points per inch)
            points = dist_disp * (72.0 / fig.dpi)
            return max(0.1, points)  # ensure minimum width

        # calculate distance in display coords
        dist_disp = np.linalg.norm(p1_disp - p0_disp)
        # convert display dist to points (72 points per inch)
        points = dist_disp * (72.0 / fig.dpi)
        # debug_print("linewidth_from_data_units", f"in={linewidth}, ref={reference}, out={points:.2f}")
        return max(0.1, points)  # ensure minimum width
    except Exception as e:
        debug_print("linewidth_from_data_units", f"error calculating: {e}")
        return max(0.1, linewidth)  # fallback


def _get_mpl_linestyle(
    path_data: SVGPathData,
) -> Union[str, Tuple[float, Optional[Tuple[float, ...]]]]:
    """get matplotlib linestyle argument from SVGPathData."""
    linestyle_map = {"solid": "-", "dashed": "--", "dotted": ":"}
    if path_data.dash_array and path_data.line_style == "custom":
        return (path_data.dash_offset, path_data.dash_array)
    else:
        return linestyle_map.get(path_data.line_style, "-")


# context manager for temporarily disabling autoscale
@contextmanager
def _no_autoscale(ax):
    autoscale_state = ax.get_autoscale_on()
    ax.set_autoscale_on(False)
    try:
        yield
    finally:
        ax.set_autoscale_on(autoscale_state)


# --- Renderer Implementation ---


class MatplotlibRenderer(BaseRenderer):
    RENDERER_NAME = "matplotlib"

    def _log_debug(self, message: str, data: Any = None):
        debug_print(self.RENDERER_NAME, message, data)

    def create_context(
        self,
        width: float = 800,
        height: float = 600,
        dpi: int = 150,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Axes:
        """create matplotlib figure and axes."""
        if ax:
            self._log_debug("using existing Axes")
            return ax

        figsize = (width / dpi, height / dpi)
        self._log_debug(f"creating new Figure/Axes, size={figsize}, dpi={dpi}")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, **kwargs)
        return ax

    def render_component(self, context: Axes, component: Component, adjust_lims: bool = True):
        """render component to matplotlib axes."""
        comp_id = component.id or type(component).__name__
        self._log_debug(f"\n--- render_component starting for '{comp_id}' ---")

        # 1. ensure component is measured and laid out
        component.measure_and_layout(self)

        # 2. optionally adjust plot limits based on root component bounds
        if adjust_lims and component.parent is None:  # only adjust for root
            self._adjust_limits(context, component)

        # 3. execute pre-render callbacks
        for cb in self.pre_render_callbacks:
            cb(context)

        # 4. render the component recursively
        # component's render method calculates world matrix and calls renderer primitives
        # start with identity matrix for root component
        root_matrix = np.identity(3)
        component.render(self, context, root_matrix)

        # 5. execute post-render callbacks
        for cb in self.post_render_callbacks:
            cb(context)

        self._log_debug(f"--- render_component finished for '{comp_id}' ---")

    def _get_recursive_world_bounds(
        self,
        component: Component,
        current_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Recursively calculate the union of world bounds for a component and its descendants."""
        if not component or not component.show:
            return current_bounds

        if current_bounds is None:
            overall_bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        else:
            overall_bounds = list(current_bounds)

        comp_bounds = component.get_world_bounds()

        if comp_bounds:
            overall_bounds[0] = min(overall_bounds[0], comp_bounds[0])
            overall_bounds[1] = min(overall_bounds[1], comp_bounds[1])
            overall_bounds[2] = max(overall_bounds[2], comp_bounds[2])
            overall_bounds[3] = max(overall_bounds[3], comp_bounds[3])

        if hasattr(component, "children") and component.children:
            for child in component.children:
                # Pass the updated overall_bounds down
                updated_bounds_tuple = self._get_recursive_world_bounds(
                    child, tuple(overall_bounds)
                )
                if updated_bounds_tuple:
                    overall_bounds = list(updated_bounds_tuple)

        if overall_bounds[0] == float("inf"):
            return None  # No valid bounds found

        return tuple(overall_bounds)

    def _adjust_limits(self, context: Axes, root: Component, padding: float = 0.1):
        """adjust plot limits to fit the rendered component tree."""
        root_id = root.id or type(root).__name__
        self._log_debug(f"adjust_limits: calculating recursive world bounds for root '{root_id}'")

        bounds = self._get_recursive_world_bounds(root)

        if not bounds:
            self._log_debug("adjust_limits: no valid bounds found, using default [0,1]")
            bounds = (0, 0, 1, 1)

        min_x, min_y, max_x, max_y = bounds
        self._log_debug(
            f"adjust_limits: recursive world bounds x=[{min_x:.1f},{max_x:.1f}], y=[{min_y:.1f},{max_y:.1f}]"
        )

        width = max(max_x - min_x, 1.0)  # ensure non-zero size for padding
        height = max(max_y - min_y, 1.0)
        pad_x = max(width * padding, 5)  # minimum padding
        pad_y = max(height * padding, 5)

        final_xlim = (min_x - pad_x, max_x + pad_x)
        final_ylim = (min_y - pad_y, max_y + pad_y)

        self._log_debug(f"adjust_limits: setting final limits xlim={final_xlim}, ylim={final_ylim}")
        context.set_xlim(final_xlim)
        context.set_ylim(final_ylim)
        context.set_aspect("equal", adjustable="box")
        self._log_debug("adjust_limits: set aspect ratio to 'equal'")

    def render_path(
        self,
        context: Axes,
        path_data: SVGPathData,
        matrix: np.ndarray,
        line_width_mode: str = "point",
        main_color: Optional[str] = None,
        secondary_color: Optional[str] = None,
    ):
        """renders a single path described by SVGPathData."""
        # self._log_debug(f"render_path: d='{path_data.d[:30]}...', stroke={path_data.stroke}, fill={path_data.fill}")
        try:
            from svgpath2mpl import parse_path
        except ImportError:
            logger.error("svgpath2mpl is required for path rendering.")
            return

        try:
            mpl_path = parse_path(path_data.d)
            # combine component's world matrix with any raw SVG transform
            final_matrix = matrix
            if path_data.transform:
                # rudimentary transform parsing - only handles matrix() for now
                m = re.search(r"matrix\((.+)\)", path_data.transform)
                if m:
                    vals = [float(v.strip()) for v in m.group(1).split(",")]
                    if len(vals) == 6:
                        svg_tf_mat = np.array(
                            [[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]], [0, 0, 1]]
                        )
                        final_matrix = (
                            final_matrix @ svg_tf_mat
                        )  # apply svg transform *before* component world transform

            transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData

            # determine colors, potentially overriding with theme colors
            fill = path_data.fill
            stroke = path_data.stroke
            if path_data.is_main_color and main_color:
                fill = main_color
            if path_data.is_secondary_color and secondary_color:
                fill = secondary_color
            # TODO: theme color override for stroke?

            fill_color = fill if fill != "none" else "none"
            edge_color = stroke if stroke != "none" else "none"
            lw = path_data.stroke_width

            # calculate linewidth in points if needed
            if line_width_mode == "data" and edge_color != "none" and lw > 0:
                lw_points = _linewidth_from_data_units(lw, context)
            else:
                lw_points = lw if lw > 0 else 0  # use points directly or 0 if no stroke

            linestyle = _get_mpl_linestyle(path_data)

            with _no_autoscale(context):
                patch = mpatches.PathPatch(
                    mpl_path,
                    facecolor=fill_color,
                    edgecolor=edge_color,
                    linewidth=lw_points,
                    linestyle=linestyle,
                    transform=transform,
                    capstyle="round",
                    joinstyle="round",
                )
                context.add_patch(patch)

        except Exception as e:
            self._log_debug(f"error rendering path '{path_data.d[:30]}...': {e}", e)

    def _create_rounded_rect_path(self, x, y, w, h, radius):
        """helper to create matplotlib Path for a rounded rectangle."""
        if radius < 1e-3:  # simple rectangle
            verts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            codes = [
                MplPath.MOVETO,
                MplPath.LINETO,
                MplPath.LINETO,
                MplPath.LINETO,
                MplPath.CLOSEPOLY,
            ]
        else:
            r = min(radius, min(w, h) / 2.0)  # effective radius
            verts = []
            codes = []
            # start after top-left corner curve
            verts.append((x + r, y + h))
            codes.append(MplPath.MOVETO)
            # top edge
            verts.append((x + w - r, y + h))
            codes.append(MplPath.LINETO)
            # top-right corner (arc approx)
            _, cp1, cp2, p_end = arc_to_bezier(x + w - r, y + h - r, r, 90, 0)
            verts.extend([cp1, cp2, p_end])
            codes.extend([MplPath.CURVE4] * 3)
            # right edge
            verts.append((x + w, y + r))
            codes.append(MplPath.LINETO)
            # bottom-right corner
            _, cp1, cp2, p_end = arc_to_bezier(x + w - r, y + r, r, 0, -90)
            verts.extend([cp1, cp2, p_end])
            codes.extend([MplPath.CURVE4] * 3)
            # bottom edge
            verts.append((x + r, y))
            codes.append(MplPath.LINETO)
            # bottom-left corner
            _, cp1, cp2, p_end = arc_to_bezier(x + r, y + r, r, -90, -180)
            verts.extend([cp1, cp2, p_end])
            codes.extend([MplPath.CURVE4] * 3)
            # left edge
            verts.append((x, y + h - r))
            codes.append(MplPath.LINETO)
            # top-left corner
            _, cp1, cp2, p_end = arc_to_bezier(x + r, y + h - r, r, -180, -270)
            verts.extend([cp1, cp2, p_end])
            codes.extend([MplPath.CURVE4] * 3)
            # close
            codes.append(MplPath.CLOSEPOLY)
            verts.append(verts[0])  # close back to start

        return MplPath(verts, codes)

    def render_rectangle(
        self,
        context: Axes,
        bounds: Size,
        style: BoxStyle,
        matrix: np.ndarray,
        component: Optional[Component] = None,
    ):
        # self._log_debug(f"render_rectangle for {getattr(component, 'id', 'N/A')}")
        w, h = bounds.width, bounds.height
        if w <= 0 or h <= 0:
            return  # skip zero-size

        transform = mtransforms.Affine2D(matrix=matrix) + context.transData

        # --- render shadow (multi-layer approximation) ---
        if style.shadow and style.shadow.blur_radius > 0:
            shadow = style.shadow
            # heuristic layer count based on blur, dpi, and resolution factor
            points_per_unit = _linewidth_from_data_units(1.0, context)
            num_layers = max(
                4, int((shadow.blur_radius * points_per_unit**0.75) * shadow.resolution)
            )
            num_layers = min(num_layers, 100)  # performance cap

            try:
                base_color_rgb = to_rgba(shadow.color)[:3]
                base_alpha = to_rgba(shadow.color)[3]
            except ValueError:
                base_color_rgb, base_alpha = (0, 0, 0), 0.5  # fallback color

            accumulated_alpha_intensity = 0.0
            min_render_alpha = 1.0 / 256.0  # smallest distinguishable alpha step

            for i in range(num_layers - 1, -1, -1):  # draw from outside in
                layer_fraction = (i / (num_layers - 1)) if num_layers > 1 else 1.0

                # approximate gaussian falloff with intensity ramp
                alpha_frac = ((num_layers - 1 - i) / (num_layers - 1)) if num_layers > 1 else 0.0
                target_layer_intensity = base_alpha * (1 - alpha_frac**1.5) / num_layers
                accumulated_alpha_intensity += target_layer_intensity

                # quantize alpha for rendering
                alpha_to_draw_int = int(accumulated_alpha_intensity / min_render_alpha)
                if alpha_to_draw_int <= 0:
                    continue

                alpha_to_draw_int = min(alpha_to_draw_int, 255)
                patch_alpha = alpha_to_draw_int * min_render_alpha
                accumulated_alpha_intensity -= patch_alpha  # subtract what's actually drawn
                accumulated_alpha_intensity = max(0, accumulated_alpha_intensity)

                # calculate geometry for this layer
                additional_spread = shadow.blur_radius * (1 - layer_fraction)
                current_spread = shadow.spread + additional_spread
                current_offset_x = shadow.offset_x
                current_offset_y = shadow.offset_y

                sw, sh = w + 2 * current_spread, h + 2 * current_spread
                # path origin relative to component origin (0,0)
                sx, sy = -current_spread + current_offset_x, -current_spread + current_offset_y
                s_radius = (
                    min(style.corner_radius + current_spread, min(sw, sh) / 2.0)
                    if min(sw, sh) > 0
                    else 0
                )

                if sw <= 0 or sh <= 0:
                    continue

                layer_path = self._create_rounded_rect_path(sx, sy, sw, sh, s_radius)
                with _no_autoscale(context):
                    layer_patch = mpatches.PathPatch(
                        layer_path,
                        facecolor=(*base_color_rgb, patch_alpha),
                        edgecolor="none",
                        linewidth=0,
                        transform=transform,
                        clip_on=False,  # allow shadow outside axes limits
                    )
                    context.add_patch(layer_patch)

        # --- render main rectangle ---
        facecolor = style.background_color or "none"
        edgecolor = style.border_color or "none"
        linewidth = style.border_width

        if facecolor != "none" or (edgecolor != "none" and linewidth > 0):
            main_path = self._create_rounded_rect_path(0, 0, w, h, style.corner_radius)

            lw_points = 0.0
            if edgecolor != "none" and linewidth > 0:
                lw_points = (
                    _linewidth_from_data_units(linewidth, context)
                    if style.border_width_mode == "data"
                    else linewidth
                )
                lw_points = max(0.1, lw_points)  # ensure minimum visible linewidth if > 0

            # get linestyle from boxstyle properties
            path_data = SVGPathData(  # dummy object to reuse linestyle helper
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
                    linewidth=lw_points,
                    linestyle=linestyle,
                    transform=transform,
                    capstyle="round",  # consistent cap/join styles
                    joinstyle="round",
                )
                context.add_patch(main_patch)

    def render_svg(self, context: Axes, svg_element: SVGElement, matrix: np.ndarray):
        """render an svg element by rendering its paths."""
        # comp_id = svg_element.id or type(svg_element).__name__
        # self._log_debug(f"render_svg: starting for '{comp_id}'")

        if not svg_element._parsed_svg_content or not svg_element._parsed_svg_content.paths:
            # self._log_debug(f"no paths found for '{comp_id}'")
            return

        svg_data = svg_element._parsed_svg_content
        viewBox = svg_data.viewBox
        svg_dims = Size(width=svg_data.width, height=svg_data.height)  # use size from parsed data

        # calculate transform to map viewBox to component's allocated _dimensions
        comp_dims = svg_element._dimensions  # final size after constraints
        scale_x, scale_y, trans_x, trans_y = 1.0, 1.0, 0.0, 0.0

        if viewBox and comp_dims.width > 1e-6 and comp_dims.height > 1e-6:
            vb_x, vb_y, vb_w, vb_h = viewBox
            if vb_w > 1e-6:
                scale_x = comp_dims.width / vb_w
            if vb_h > 1e-6:
                scale_y = comp_dims.height / vb_h
            # translate viewBox origin (vb_x, vb_y) to component origin (0,0) *after* scaling
            trans_x = -vb_x * scale_x
            trans_y = -vb_y * scale_y
        elif svg_dims.width > 1e-6 and svg_dims.height > 1e-6:
            # simple scale if no viewbox but svg has dimensions
            scale_x = comp_dims.width / svg_dims.width
            scale_y = comp_dims.height / svg_dims.height

        svg_internal_matrix = np.array([[scale_x, 0, trans_x], [0, scale_y, trans_y], [0, 0, 1]])

        final_matrix = matrix @ svg_internal_matrix  # apply internal scaling/translation first

        width_mode = svg_element.get_renderer_options(self.RENDERER_NAME).get(
            "line_width_mode", "point"
        )
        main_color = getattr(svg_element, "main_color", None)
        secondary_color = getattr(svg_element, "secondary_color", None)

        # self._log_debug(f"rendering {len(svg_data.paths)} paths for '{comp_id}'")
        for path_data in svg_data.paths:
            self.render_path(
                context, path_data, final_matrix, width_mode, main_color, secondary_color
            )

    def render_text(self, context: Axes, text_component: Text, matrix: np.ndarray):
        """render text using matplotlib's Text or TextPath."""
        # comp_id = text_component.id or type(text_component).__name__
        # self._log_debug(f"render_text: starting for '{comp_id}'")

        if not text_component.text:
            return

        # use cached text paths if available (from measure_text) for precision
        svg_cache = getattr(text_component, "_svg_cache", None)
        if isinstance(svg_cache, SVGTextContent) and svg_cache.text_paths:
            self._render_text_paths(context, text_component, matrix)
        else:
            # fallback to basic matplotlib text rendering if no cache (less precise alignment)
            self._render_text_basic(context, text_component, matrix)

    def _render_text_paths(self, context: Axes, text_comp: Text, matrix: np.ndarray):
        """render text using pre-calculated glyph paths from SVGTextContent cache."""
        svg_content = text_comp._svg_cache
        if not svg_content or not svg_content.text_paths:
            return

        alloc_w, alloc_h = text_comp._dimensions.width, text_comp._dimensions.height
        measured_w, measured_h = svg_content.measured_width, svg_content.measured_height

        # horizontal alignment offset (relative to component origin 0,0)
        dx = 0.0
        if text_comp.align == "center":
            dx = (alloc_w - measured_w) / 2.0
        elif text_comp.align == "right":
            dx = alloc_w - measured_w

        # vertical alignment offset (relative to component origin 0,0)
        # text path origin (0,0) is bottom-left of measured bounds
        dy = 0.0  # default 'bottom'
        if text_comp.vertical_align == "middle":
            dy = (alloc_h - measured_h) / 2.0
        elif text_comp.vertical_align == "top":
            dy = alloc_h - measured_h

        align_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        final_matrix = matrix @ align_matrix
        transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData
        color = getattr(text_comp, "color", "black")

        # self._log_debug(f"rendering {len(svg_content.text_paths)} cached text paths for {text_comp.id}")
        with _no_autoscale(context):
            for path_info in svg_content.text_paths:
                mpl_path = path_info[
                    "path"
                ]  # path is already positioned relative to measured bottom-left
                patch = mpatches.PathPatch(
                    mpl_path, facecolor=color, edgecolor="none", transform=transform
                )
                context.add_patch(patch)

    def _render_text_basic(self, context: Axes, text_comp: Text, matrix: np.ndarray):
        """fallback text rendering using ax.text (less precise alignment)."""
        # self._log_debug(f"rendering text using basic ax.text for {text_comp.id}")
        text_props = {
            "fontsize": text_comp.font_size,
            "fontfamily": text_comp.font_name or "sans-serif",
            "fontweight": text_comp.font_weight,
            "fontstyle": text_comp.font_style,
            "color": text_comp.color,
            "ha": text_comp.align,  # horizontal alignment
            "va": text_comp.vertical_align,  # vertical alignment needs mapping
            "linespacing": 1.0 + text_comp.line_spacing,
            "transform": mtransforms.Affine2D(matrix=matrix) + context.transData,
            "clip_on": True,  # clip text to axes bounds
        }

        # determine anchor point (x,y) in component's local coords based on alignment
        x = 0.0
        if text_comp.align == "center":
            x = text_comp._dimensions.width / 2.0
        elif text_comp.align == "right":
            x = text_comp._dimensions.width
        y = text_comp._dimensions.height
        if text_comp.vertical_align == "middle":
            y = text_comp._dimensions.height / 2.0
        elif text_comp.vertical_align == "bottom":
            y = 0.0

        # matplotlib's 'va' differs slightly, map common cases
        if text_props["va"] == "top":
            text_props["va"] = "top"  # usually correct
        elif text_props["va"] == "middle":
            text_props["va"] = "center_baseline"  # often better than 'center'
        elif text_props["va"] == "bottom":
            text_props["va"] = "bottom"  # usually correct

        with _no_autoscale(context):
            context.text(x, y, text_comp.text, **text_props)

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
        """render the main curve and end caps for a connection."""
        # self._log_debug(f"rendering connection curve for {connection.id}")

        # 1. render the main path
        path_data = SVGPathData(
            d=path_string,
            stroke=connection.color,
            stroke_width=connection.line_width,
            fill="none",
            line_style=connection.line_style,
            dash_array=connection.dash_array,
            dash_offset=connection.dash_offset,
        )
        self.render_path(
            context, path_data, matrix, "point"
        )  # connection lines usually use point width

        # 2. render end caps (using local points and calculated directions)
        if connection.start_cap or connection.end_cap:
            active_curve = connection._get_active_curve()
            try:
                # get directions pointing outwards from the curve ends
                start_dir, end_dir = active_curve.get_directions(
                    local_start, local_end, local_control_points
                )

                cap_funcs = {
                    LineEndArrow: create_arrow_cap,
                    LineEndCircle: create_circle_cap,
                    LineEndFlat: create_flat_cap,
                }

                if connection.start_cap:
                    cap_type = type(connection.start_cap)
                    if cap_type in cap_funcs:
                        start_cap_path = cap_funcs[cap_type](
                            local_start, start_dir, connection.start_cap
                        )
                        self.render_path(
                            context, start_cap_path, matrix, "point"
                        )  # caps use point width

                if connection.end_cap:
                    cap_type = type(connection.end_cap)
                    if cap_type in cap_funcs:
                        end_cap_path = cap_funcs[cap_type](local_end, end_dir, connection.end_cap)
                        self.render_path(
                            context, end_cap_path, matrix, "point"
                        )  # caps use point width

            except Exception as e:
                self._log_debug(f"error rendering end caps for {connection.id}: {e}")

    def render_debug(self, context: Axes, component: Component, matrix: np.ndarray):
        """render debug visuals (bounding box, origin) for a component."""
        comp_id = component.id or type(component).__name__
        # self._log_debug(f"render_debug: drawing for '{comp_id}'")
        if (
            not hasattr(component, "_dimensions")
            or component._dimensions.width <= 0
            or component._dimensions.height <= 0
        ):
            return

        w, h = component._dimensions.width, component._dimensions.height
        transform = mtransforms.Affine2D(matrix=matrix) + context.transData
        debug_lw = 0.5  # points

        with _no_autoscale(context):
            # draw bounding box
            rect = mpatches.Rectangle(
                (0, 0),
                w,
                h,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=debug_lw,
                transform=transform,
            )
            context.add_patch(rect)

            # draw origin marker (X) - use display coordinates for fixed size marker
            origin_world = (matrix @ np.array([0, 0, 1]))[:2]
            origin_disp = context.transData.transform(origin_world)
            marker_size_disp = 4  # pixels
            line1 = plt.Line2D(
                [origin_disp[0] - marker_size_disp, origin_disp[0] + marker_size_disp],
                [origin_disp[1] - marker_size_disp, origin_disp[1] + marker_size_disp],
                color="red",
                linewidth=debug_lw,
                linestyle="-",
                transform=None,  # display coords
            )
            line2 = plt.Line2D(
                [origin_disp[0] - marker_size_disp, origin_disp[0] + marker_size_disp],
                [origin_disp[1] + marker_size_disp, origin_disp[1] - marker_size_disp],
                color="red",
                linewidth=debug_lw,
                linestyle="-",
                transform=None,  # display coords
            )
            context.add_line(line1)
            context.add_line(line2)

    def measure_text(self, text_comp: Text) -> Size:
        """measure text using matplotlib's TextPath for accurate glyph bounds."""
        comp_id = text_comp.id or type(text_comp).__name__
        # self._log_debug(f"measure_text starting for '{comp_id}'")

        if not text_comp.text:
            text_comp._svg_cache = None
            return Size(0, 0)

        font_props = fm.FontProperties(
            family=text_comp.font_name or "sans-serif",
            weight=text_comp.font_weight,
            style=text_comp.font_style,
        )
        lines = text_comp.text.split("\n")
        max_width = 0.0
        paths_info = []

        # use a dummy figure/renderer for measurement
        # TODO: investigate if a renderer instance can be used without creating a figure
        try:
            fig = plt.figure()  # needed to get a renderer instance
            renderer = fig.canvas.get_renderer()
        except Exception as e:
            logger.error(f"failed to create dummy figure/renderer for text measurement: {e}")
            # fallback: estimate size based on font size and text length
            est_h = text_comp.font_size * (1 + text_comp.line_spacing) * len(lines)
            est_w = max(len(line) for line in lines) * text_comp.font_size * 0.6  # rough estimate
            return Size(width=est_w, height=est_h)

        # measure each line
        for line in lines:
            if not line.strip():  # handle empty lines
                path = TextPath(
                    (0, 0), " ", size=text_comp.font_size, prop=font_props, usetex=False
                )
                bbox = path.get_extents(renderer=renderer)
                line_height = (
                    bbox.height if bbox.height > 0 else text_comp.font_size
                )  # estimate height
                paths_info.append(
                    {
                        "width": 0.0,
                        "height": line_height,
                        "min_y": bbox.y0,
                        "max_y": bbox.y1,
                        "path_at_origin": path,
                        "is_empty": True,
                    }
                )
                continue

            path = TextPath((0, 0), line, size=text_comp.font_size, prop=font_props, usetex=False)
            try:
                bbox = path.get_extents(renderer=renderer)  # use renderer for accurate bounds
                paths_info.append(
                    {
                        "width": bbox.width,
                        "height": bbox.height,
                        "min_y": bbox.y0,
                        "max_y": bbox.y1,
                        "path_at_origin": path,
                        "is_empty": False,
                    }
                )
                max_width = max(max_width, bbox.width)
            except Exception as e:
                self._log_debug(f"warning: failed to measure text path for '{line}': {e}")
                # estimate size as fallback for this line
                est_w = len(line) * text_comp.font_size * 0.6
                est_h = text_comp.font_size
                paths_info.append(
                    {
                        "width": est_w,
                        "height": est_h,
                        "min_y": -est_h * 0.2,
                        "max_y": est_h * 0.8,
                        "path_at_origin": path,
                        "is_empty": False,
                    }
                )
                max_width = max(max_width, est_w)

        plt.close(fig)  # close dummy figure

        if not paths_info:
            return Size(0, 0)

        # determine effective line height for layout spacing
        first_line_metrics = next((p for p in paths_info if not p["is_empty"]), paths_info[0])
        # use max_y - min_y as robust height measure
        base_line_height = first_line_metrics["max_y"] - first_line_metrics["min_y"]
        line_spacing_pixels = base_line_height * text_comp.line_spacing
        effective_line_height = base_line_height + line_spacing_pixels
        if effective_line_height <= 0:
            effective_line_height = text_comp.font_size * (1 + text_comp.line_spacing)  # fallback

        # position paths vertically and find overall bounds
        y_cursor, actual_min_y, actual_max_y = 0.0, float("inf"), float("-inf")
        positioned_paths_info = []
        for i, initial_info in enumerate(paths_info):
            target_y_baseline = y_cursor
            line_min_y = target_y_baseline + initial_info["min_y"]
            line_max_y = target_y_baseline + initial_info["max_y"]
            actual_min_y = min(actual_min_y, line_min_y)
            actual_max_y = max(actual_max_y, line_max_y)
            # store path at origin and its target baseline for final positioning
            positioned_paths_info.append(
                {
                    "path_at_origin": initial_info["path_at_origin"],
                    "target_y_baseline": target_y_baseline,
                }
            )
            y_cursor -= effective_line_height  # move down for next line

        # calculate final measured size and y-offset for cache
        measured_height = max(0, actual_max_y - actual_min_y)
        y_offset_to_final_origin = -actual_min_y  # shift so lowest point is at y=0

        # create final paths relative to measured bottom-left origin
        final_paths_for_cache = []
        for info in positioned_paths_info:
            final_y_baseline = info["target_y_baseline"] + y_offset_to_final_origin
            # translate the path originally at (0,0) to (0, final_y_baseline)
            translated_path = info["path_at_origin"].transformed(
                mtransforms.Affine2D().translate(0, final_y_baseline)
            )
            final_paths_for_cache.append({"path": translated_path})  # store only the final path

        # store results in cache object
        text_comp._svg_cache = SVGTextContent(
            width=max_width,
            height=measured_height,  # natural size
            viewBox=(0, 0, max_width, measured_height),
            text_paths=final_paths_for_cache,
            measured_width=max_width,
            measured_height=measured_height,
        )
        # self._log_debug("stored measured text paths in _svg_cache", text_comp._svg_cache)

        return Size(width=max_width, height=measured_height)

    def render_to_output(
        self, context: Axes, output: Optional[Union[str, Path, TextIO, BinaryIO]] = None, **kwargs
    ):
        """saves the figure context to a file or stream."""
        self._log_debug(f"render_to_output: saving to '{output}' with kwargs={kwargs}")
        if not hasattr(context, "figure"):
            raise ValueError("context must be a matplotlib Axes object with a figure attribute.")

        # commonly used options for saving figures
        default_opts = {"bbox_inches": "tight", "pad_inches": 0.1}
        save_opts = {**default_opts, **kwargs}

        if output:
            try:
                context.figure.savefig(output, **save_opts)
                self._log_debug(f"successfully saved figure to '{output}'")
            except Exception as e:
                self._log_debug(f"error saving figure: {e}", e)
                raise
        else:
            self._log_debug("no output path provided, returning figure object.")

        return context.figure
