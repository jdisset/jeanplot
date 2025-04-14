"""matplotlib rendering backend for jeanplot."""

# ... (imports and other helpers remain the same) ...
from typing import Optional, Any, List, Dict, Tuple, Union, Literal, TextIO, BinaryIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure  # Import Figure
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MplPath  # avoid name clash with Pathlib
from pathlib import Path
from matplotlib.colors import to_rgba
import logging
from contextlib import contextmanager
import re  # needed for svg transform parsing

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
    LineEndFlat,
    LineEndArrow,
    LineEndCircle,
)
from jeanplot.debug import debug_print
from jeanplot.connector import Connection  # needed for type hint
from jeanplot.text import Text  # needed for type hint

logger = logging.getLogger(__name__)

# --- Matplotlib Helpers ---
# _get_point_scale_factor, _linewidth_in_points, _get_mpl_linestyle, _no_autoscale
# remain the same as the previous version...


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
    final_points = max(0.0, points)  # allow zero width
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


# --- Renderer Implementation ---


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
                    pass  # ignore errors disconnecting
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

        # self._log_debug(f"refresh_linewidths: updating {len(self._data_width_patches)} patches...")
        updated_count = 0
        patches_to_remove = []
        for i, (patch, width_data) in enumerate(self._data_width_patches):
            if not patch or not hasattr(patch, "figure") or patch.figure is None:
                patches_to_remove.append(i)
                continue
            if hasattr(patch, "set_linewidth"):
                try:
                    new_width_points = _linewidth_in_points(width_data, self._context)
                    current_lw = patch.get_linewidth()
                    # Optimization: only update if visually significant change
                    if not np.isclose(current_lw, new_width_points, atol=0.05):
                        patch.set_linewidth(new_width_points)
                        updated_count += 1
                        # log only significant updates to reduce noise
                        # is_cap_like = isinstance(patch, mpatches.PathPatch) and len(patch.get_path().vertices) < 5
                        # log_prefix = f"  Patch {i}{' (CAP?)' if is_cap_like else ''}:"
                        # self._log_debug(f"{log_prefix} data={width_data:.2f} -> pts={new_width_points:.3f} (current was {current_lw:.3f})")
                except Exception as e:
                    self._log_debug(f"  error updating patch {i}: {e}")
            else:
                patches_to_remove.append(i)
        if patches_to_remove:
            for index in sorted(patches_to_remove, reverse=True):
                del self._data_width_patches[index]
        # if updated_count > 0: self._log_debug(f"refresh_linewidths: updated {updated_count} patches.")

    def track_patch(self, patch: mpatches.Patch, width_data: float):
        # log the tracking action with the specific width
        self._log_debug(
            f"track_patch: Tracking patch id={id(patch)} with data_width={width_data:.3f}"
        )
        self._data_width_patches.append((patch, width_data))

    def render_component(self, context: Axes, component: Component, adjust_lims: bool = True):
        """
        Measures, lays out, and renders the component tree, ensuring correct
        world matrices are passed down.
        """
        if self._context != context:
            self._log_debug("context changed, updating and reconnecting event")
            self._disconnect_draw_event()
            self._context = context
            fig = context.get_figure()
            if fig and fig.canvas:
                self._draw_event_cid = fig.canvas.mpl_connect("draw_event", self.refresh_linewidths)

        # step 1: measure and layout the entire tree. this calculates all
        #         internal sizes, layout positions, and resolves attachments.
        component.measure_and_layout(self)

        # step 2: adjust limits if requested (typically for the root)
        if adjust_lims and component.parent is None:
            self._adjust_limits(context, component)

        # step 3: run pre-render callbacks
        for cb in self.pre_render_callbacks:
            cb(context)

        # step 4: prepare for rendering pass
        self._data_width_patches = []  # clear patch tracker

        # step 5: calculate the world matrix for the root component *after* layout
        #         this matrix incorporates the root's own offset and transform.
        root_world_matrix = component.compute_world_matrix()
        self._log_debug(
            f"render_component: rendering root '{component.id or type(component).__name__}' with world matrix",
            root_world_matrix,
        )

        # step 6: initiate the recursive rendering process, passing the correct world matrix
        component.render(self, context, root_world_matrix)

        # step 7: run post-render callbacks
        for cb in self.post_render_callbacks:
            cb(context)

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
        main_color: Optional[str] = None,
        secondary_color: Optional[str] = None,
    ):
        """renders a single path described by SVGPathData, tracking data width."""
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
                    vals = [float(v.strip()) for v in m.group(1).split(",")]
                    if len(vals) == 6:
                        final_matrix = final_matrix @ np.array(
                            [[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]], [0, 0, 1]]
                        )
            transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData

            fill = main_color if path_data.is_main_color and main_color else path_data.fill
            fill = secondary_color if path_data.is_secondary_color and secondary_color else fill
            fill_color = fill if fill != "none" else "none"
            edge_color = path_data.stroke if path_data.stroke != "none" else "none"

            # --- critical part: get the width from path_data ---
            current_path_lw_data = path_data.stroke_width
            # ---------------------------------------------------

            # determine if this path should use data units
            is_data_width = (
                line_width_mode == "data" and edge_color != "none" and current_path_lw_data > 0
            )
            # set initial linewidth for patch creation (0 if data, actual if point)
            initial_lw_points = 0.0 if is_data_width else current_path_lw_data

            linestyle = _get_mpl_linestyle(path_data)

            with _no_autoscale(context):
                patch = mpatches.PathPatch(
                    mpl_path,
                    facecolor=fill_color,
                    edgecolor=edge_color,
                    linewidth=initial_lw_points,  # start with 0 or point size
                    linestyle=linestyle,
                    transform=transform,
                    capstyle="round",
                    joinstyle="round",
                )
                context.add_patch(patch)

                # track if data width, otherwise set point width immediately
                if is_data_width:
                    # --- Make sure to track with the correct width ---
                    self.track_patch(patch, current_path_lw_data)
                    # -----------------------------------------------
                    # set initial point size based on current zoom
                    initial_points_estimate = _linewidth_in_points(current_path_lw_data, context)
                    patch.set_linewidth(initial_points_estimate)
                # No 'elif' needed here, initial_lw_points already handled point width case

        except Exception as e:
            self._log_debug(f"error rendering path '{path_data.d[:30]}...': {e}", e)

    def _create_rounded_rect_path(self, x, y, w, h, radius):
        if radius < 1e-3:
            verts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            codes = [1, 2, 2, 2, 79]  # mpl path codes
        else:
            r = min(radius, min(w, h) / 2.0)
            v, c = [], []
            v.append((x + r, y + h))
            c.append(1)
            v.append((x + w - r, y + h))
            c.append(2)
            _, cp1, cp2, p = arc_to_bezier(x + w - r, y + h - r, r, 90, 0)
            v.extend([cp1, cp2, p])
            c.extend([4] * 3)
            v.append((x + w, y + r))
            c.append(2)
            _, cp1, cp2, p = arc_to_bezier(x + w - r, y + r, r, 0, -90)
            v.extend([cp1, cp2, p])
            c.extend([4] * 3)
            v.append((x + r, y))
            c.append(2)
            _, cp1, cp2, p = arc_to_bezier(x + r, y + r, r, -90, -180)
            v.extend([cp1, cp2, p])
            c.extend([4] * 3)
            v.append((x, y + h - r))
            c.append(2)
            _, cp1, cp2, p = arc_to_bezier(x + r, y + h - r, r, -180, -270)
            v.extend([cp1, cp2, p])
            c.extend([4] * 3)
            c.append(79)
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

        if style.shadow and style.shadow.blur_radius > 0:  # render shadow if present
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
                target_intensity = base_alpha * (1 - alpha_frac**1.5) / num_layers
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
                initial_lw_points = linewidth_data  # point mode

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
            return
        svg_data = svg_element._parsed_svg_content
        viewBox = svg_data.viewBox
        svg_dims = Size(svg_data.width, svg_data.height)
        comp_dims = svg_element._dimensions
        scale_x, scale_y, trans_x, trans_y = 1.0, 1.0, 0.0, 0.0
        if viewBox and comp_dims.width > 1e-6 and comp_dims.height > 1e-6:
            vb_x, vb_y, vb_w, vb_h = viewBox
            if vb_w > 1e-6:
                scale_x = comp_dims.width / vb_w
            if vb_h > 1e-6:
                scale_y = comp_dims.height / vb_h
            trans_x = -vb_x * scale_x
            trans_y = -vb_y * scale_y
        elif svg_dims.width > 1e-6 and svg_dims.height > 1e-6:
            scale_x = comp_dims.width / svg_dims.width
            scale_y = comp_dims.height / svg_dims.height
        svg_internal_matrix = np.array([[scale_x, 0, trans_x], [0, scale_y, trans_y], [0, 0, 1]])
        final_matrix = matrix @ svg_internal_matrix
        default_lw_mode = svg_element.line_width_mode
        main_color = getattr(svg_element, "main_color", None)
        secondary_color = getattr(svg_element, "secondary_color", None)
        for path_data in svg_data.paths:
            self.render_path(
                context, path_data, final_matrix, default_lw_mode, main_color, secondary_color
            )

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
        """render connection curve and caps, ensuring correct line width tracking."""
        path_data = SVGPathData(
            d=path_string,
            stroke=connection.color,
            stroke_width=connection.line_width,
            fill="none",
            line_style=connection.line_style,
            dash_array=connection.dash_array,
            dash_offset=connection.dash_offset,
        )
        # use 'data' mode explicitly for main connection line
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
                    # use 'data' mode explicitly for caps
                    self.render_path(context, start_cap_path, matrix, line_width_mode="data")
                if connection.end_cap and type(connection.end_cap) in cap_funcs:
                    end_cap_path = cap_funcs[type(connection.end_cap)](
                        local_end, end_dir, connection.end_cap
                    )
                    # use 'data' mode explicitly for caps
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
            origin_disp = context.transData.transform((matrix @ [0, 0, 1])[:2])
            sz = 4
            context.add_line(
                plt.Line2D(
                    [origin_disp[0] - sz, origin_disp[0] + sz],
                    [origin_disp[1] - sz, origin_disp[1] + sz],
                    color="red",
                    lw=lw,
                    ls="-",
                    transform=None,
                )
            )
            context.add_line(
                plt.Line2D(
                    [origin_disp[0] - sz, origin_disp[0] + sz],
                    [origin_disp[1] + sz, origin_disp[1] - sz],
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
        except Exception:
            return Size()  # cannot measure without dummy renderer
        for line in lines:
            if not line.strip():
                p = TextPath((0, 0), " ", size=text_comp.font_size, prop=props)
                bbox = p.get_extents()
                h = bbox.height or text_comp.font_size
                paths_info.append(
                    {"w": 0.0, "h": h, "min_y": bbox.y0, "max_y": bbox.y1, "p": p, "t": line}
                )
                continue
            p = TextPath((0, 0), line, size=text_comp.font_size, prop=props)
            try:
                bbox = p.get_extents()
                paths_info.append(
                    {
                        "w": bbox.width,
                        "h": bbox.height,
                        "min_y": bbox.y0,
                        "max_y": bbox.y1,
                        "p": p,
                        "t": line,
                    }
                )
                max_w = max(max_w, bbox.width)
            except Exception:
                est_w = len(line) * text_comp.font_size * 0.6
                est_h = text_comp.font_size
                paths_info.append(
                    {
                        "w": est_w,
                        "h": est_h,
                        "min_y": -est_h * 0.2,
                        "max_y": est_h * 0.8,
                        "p": p,
                        "t": line,
                    }
                )
                max_w = max(max_w, est_w)
        plt.close(fig)
        if not paths_info:
            return Size()
        first = next((p for p in paths_info if p["w"] > 0), paths_info[0])
        base_h = first["max_y"] - first["min_y"]
        eff_lh = max(
            base_h * (1 + text_comp.line_spacing),
            text_comp.font_size * (1 + text_comp.line_spacing),
        )
        y_cur, min_y, max_y = 0.0, float("inf"), float("-inf")
        final_paths = []
        for info in paths_info:
            y_base = y_cur
            min_y = min(min_y, y_base + info["min_y"])
            max_y = max(max_y, y_base + info["max_y"])
            final_y = y_base - (min_y if min_y < 0 else 0)
            final_paths.append(
                {"path": TextPath((0, final_y), info["t"], size=text_comp.font_size, prop=props)}
            )
            y_cur -= eff_lh
        measured_h = max(0, max_y - min_y) if max_y > min_y else text_comp.font_size
        y_offset = -min_y
        for p_info in final_paths:
            p_info["path"] = p_info["path"].transformed(
                mtransforms.Affine2D().translate(0, y_offset)
            )
        text_comp._svg_cache = SVGTextContent(
            width=max_w,
            height=measured_h,
            viewBox=(0, 0, max_w, measured_h),
            paths=[],
            text_paths=final_paths,
            measured_width=max_w,
            measured_height=measured_h,
        )
        measured_size = Size(width=max_w, height=measured_h)
        return Size(
            width=min(
                max(text_comp.min_dimensions.width, measured_size.width),
                text_comp.max_dimensions.width,
            ),
            height=min(
                max(text_comp.min_dimensions.height, measured_size.height),
                text_comp.max_dimensions.height,
            ),
        )

    def render_to_output(self, context: Axes, output=None, **kwargs):
        self.refresh_linewidths()  # ensure final widths before save
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
