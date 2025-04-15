# File: jeanplot/matplotlib_renderer.py
# -*- coding: utf-8 -*-
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
    _normalize_color,
)
from jeanplot.debug import debug_print, get_logger  # Import get_logger
from jeanplot.connector import Connection
from jeanplot.text import Text

# use get_logger instead of logging.getLogger
logger = get_logger(__name__)


# --- Helper Functions ---
# ( _get_point_scale_factor, _linewidth_in_points, _get_mpl_linestyle, _no_autoscale remain the same)
def _get_point_scale_factor(axis: Axes) -> float:
    fig = axis.get_figure()
    if not fig:
        return 1.0
    try:
        # transform two points slightly separated in data coords to display coords
        p0_disp = axis.transData.transform([(0, 0)])[0]
        p1_disp_x = axis.transData.transform([(1, 0)])[0]
        p1_disp_y = axis.transData.transform([(0, 1)])[0]

        # calculate distance in display coords (pixels)
        dx_disp = abs(p1_disp_x[0] - p0_disp[0])
        dy_disp = abs(p1_disp_y[1] - p0_disp[1])

        # handle cases where one dimension might be zero scale (e.g., flat line)
        if dx_disp < 1e-6 and dy_disp < 1e-6:
            return 0.0  # avoid division by zero if scale is zero
        elif dx_disp < 1e-6:
            avg_pixels_per_data_unit = dy_disp
        elif dy_disp < 1e-6:
            avg_pixels_per_data_unit = dx_disp
        else:
            avg_pixels_per_data_unit = (dx_disp + dy_disp) / 2.0

        points_per_pixel = 72.0 / fig.dpi
        points_per_data_unit = avg_pixels_per_data_unit * points_per_pixel
        # logger.debug(f"_get_point_scale_factor: ppd={points_per_data_unit:.3f}")
        return points_per_data_unit

    except Exception as e:
        # logger.warning(f"error calculating point scale factor: {e}") # too verbose
        return 1.0  # fallback


def _linewidth_in_points(linewidth_data: float, axis: Axes) -> float:
    if linewidth_data <= 0:
        return 0.0
    scale_factor = _get_point_scale_factor(axis)
    # if scale factor is near zero, maybe return a minimum line width?
    # for now, let it be zero if scale is zero
    if scale_factor < 1e-6:
        return 0.0

    points = linewidth_data * scale_factor
    # ensure non-negative, maybe add a max clamp if needed?
    final_points = max(0.0, points)
    # logger.debug(f"_linewidth_in_points: data={linewidth_data:.2f}, scale={scale_factor:.2f} -> pts={final_points:.2f}")
    return final_points


def _get_mpl_linestyle(
    path_data: SVGPathData,
) -> Union[str, Tuple[float, Optional[Tuple[float, ...]]]]:
    linestyle_map = {"solid": "-", "dashed": "--", "dotted": ":"}
    if path_data.dash_array and path_data.line_style == "custom":
        # matplotlib dash format: (offset, onoffseq)
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


# --- MatplotlibRenderer Class ---


class MatplotlibRenderer(BaseRenderer):
    RENDERER_NAME = "matplotlib"

    def __init__(self, debug=False):
        super().__init__()
        self._data_width_patches: List[Tuple[mpatches.Patch, float]] = []
        self._context: Optional[Axes] = None
        self._draw_event_cid: Optional[int] = None

    def _log_debug(self, message: str, data: Any = None):
        # use the centralized debug print
        debug_print(self.RENDERER_NAME, message, data)

    def _disconnect_draw_event(self):
        if self._draw_event_cid is not None and self._context is not None:
            fig = self._context.get_figure()
            if fig and fig.canvas:
                try:
                    fig.canvas.mpl_disconnect(self._draw_event_cid)
                    # self._log_debug("disconnected draw event")
                except Exception as e:
                    # self._log_debug(f"error disconnecting draw event: {e}")
                    pass  # ignore if already disconnected etc.
        self._draw_event_cid = None

    def create_context(
        self,
        width: float = 800,
        height: float = 600,
        dpi: int = 150,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Axes:
        self._disconnect_draw_event()  # ensure old connection is removed
        self._data_width_patches = []  # clear patches from previous context
        if ax:
            self._context = ax
            fig = ax.get_figure()
        else:
            figsize = (width / dpi, height / dpi)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, **kwargs)
            self._context = ax

        # connect draw event for linewidth updates
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

        # self._log_debug(f"refresh_linewidths triggered (event={event}), {len(self._data_width_patches)} patches")
        updated_count = 0
        patches_to_remove = []  # indices to remove

        # use enumerate to allow safe removal by index later
        for i, (patch, width_data) in enumerate(self._data_width_patches):
            # --- sanity checks ---
            if patch is None or not hasattr(patch, "figure") or patch.figure is None:
                # self._log_debug(f"  patch {i} is invalid or removed, marking for removal.")
                patches_to_remove.append(i)
                continue
            if patch.axes is not self._context:
                # self._log_debug(f"  patch {i} belongs to different axes, marking for removal.")
                patches_to_remove.append(i)
                continue
            # --------------------

            if hasattr(patch, "set_linewidth"):
                try:
                    new_width_points = _linewidth_in_points(width_data, self._context)
                    current_lw = patch.get_linewidth()
                    # only update if significantly different to avoid unnecessary redraws
                    if not np.isclose(current_lw, new_width_points, atol=0.05):
                        # self._log_debug(f"  patch {i}: updating lw from {current_lw:.2f} to {new_width_points:.2f} (data={width_data:.2f})")
                        patch.set_linewidth(new_width_points)
                        updated_count += 1
                    # else:
                    # self._log_debug(f"  patch {i}: lw {current_lw:.2f} vs {new_width_points:.2f} (data={width_data:.2f}) - no update needed")
                except Exception as e:
                    # self._log_debug(f"  error updating patch {i}: {e}")
                    pass  # ignore errors for specific patches
            else:
                # self._log_debug(f"  patch {i} has no set_linewidth, marking for removal.")
                patches_to_remove.append(i)  # remove if it doesn't support linewidth

        # remove invalid/processed patches efficiently by index
        if patches_to_remove:
            # sort indices in descending order to avoid messing up indices during removal
            patches_to_remove.sort(reverse=True)
            # self._log_debug(f"  removing {len(patches_to_remove)} patches")
            for index in patches_to_remove:
                self._data_width_patches.pop(index)

    def track_patch(self, patch: mpatches.Patch, width_data: float):
        """adds a patch to the list for dynamic linewidth updates."""
        # self._log_debug(f"track_patch: tracking patch id={id(patch)} with data_width={width_data:.3f}")
        self._data_width_patches.append((patch, width_data))

    def render_component(self, context: Axes, component: Component, adjust_lims: bool = True):
        """render a component tree into the matplotlib axes."""
        # ensure internal context matches provided context
        if self._context != context:
            self._log_debug("context changed, updating and reconnecting event")
            self._disconnect_draw_event()  # disconnect from old context
            self._context = context
            fig = context.get_figure()
            if fig and fig.canvas:
                self._draw_event_cid = fig.canvas.mpl_connect("draw_event", self.refresh_linewidths)
                # self._log_debug(f"reconnected draw event (cid={self._draw_event_cid})")

        # perform layout calculation using this renderer for measurements
        component.measure_and_layout(self)

        # adjust axis limits if requested and this is the root component
        # *** important: this happens BEFORE rendering starts ***
        if adjust_lims and component.parent is None:
            self._adjust_limits(context, component)

        # run pre-render callbacks
        for cb in self.pre_render_callbacks:
            cb(context)

        # clear patches list before rendering starts
        self._data_width_patches = []
        # self._log_debug("cleared data width patches list.")

        # get root transform and start recursive rendering
        root_world_matrix = component.compute_world_matrix()
        # self._log_debug(f"rendering component tree starting with {component.id or component.__class__.__name__}")
        component.render(
            self, context, root_world_matrix
        )  # render call eventually calls connection.render()

        # run post-render callbacks
        for cb in self.post_render_callbacks:
            cb(context)

        # perform initial linewidth update after all patches are added
        # self._log_debug("performing initial linewidth refresh after render.")
        self.refresh_linewidths()

    # (_get_recursive_world_bounds, _adjust_limits methods remain the same)
    def _get_recursive_world_bounds(
        self, component: Component, current_bounds=None
    ) -> Optional[Tuple[float, float, float, float]]:
        """recursively find the min/max world coordinates occupied by visible components."""
        if not component or not component.show:
            return current_bounds

        # initialize bounds if first call
        overall = (
            list(current_bounds)
            if current_bounds
            else [float("inf"), float("inf"), float("-inf"), float("-inf")]
        )

        # update bounds with current component's world bounds (if not a connection)
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
        if hasattr(component, "children") and component.children:
            for child in component.children:
                if not child or not child.show:
                    continue

                child_bounds = None
                if isinstance(child, Connection):
                    # --- Connection Bounds Estimation (Pre-Render) ---
                    conn_points_est = []
                    # Resolve original components first
                    child._resolve_references_orig()
                    start_comp_orig = child._resolved_start_component_orig
                    end_comp_orig = child._resolved_end_component_orig

                    if start_comp_orig and end_comp_orig:
                        # Estimate start point using connection's start_offset on original component
                        est_start = child._get_offset_world_position(
                            start_comp_orig, child.start_offset
                        )
                        if est_start:
                            conn_points_est.append(est_start)

                        # Estimate end point using connection's end_offset on original component
                        est_end = child._get_offset_world_position(end_comp_orig, child.end_offset)
                        if est_end:
                            conn_points_est.append(est_end)

                    if len(conn_points_est) == 2:
                        # simple buffer based on estimated line width
                        buffer = max(child.line_width * 3, 10)  # use a reasonable buffer
                        min_x = min(p[0] for p in conn_points_est) - buffer
                        min_y = min(p[1] for p in conn_points_est) - buffer
                        max_x = max(p[0] for p in conn_points_est) + buffer
                        max_y = max(p[1] for p in conn_points_est) + buffer
                        child_bounds = (min_x, min_y, max_x, max_y)
                        # self._log_debug(f"estimated connection bounds for {child.id}: {child_bounds}")
                    else:
                        # fallback if we couldn't estimate points
                        comp_b = child.get_world_bounds()  # use component's rough overlay bounds
                        if comp_b:
                            child_bounds = comp_b
                        # self._log_debug(f"using rough overlay bounds for connection {child.id}: {child_bounds}")

                    # --- End Connection Bounds Estimation ---
                else:
                    # default recursive call for other components
                    child_bounds = self._get_recursive_world_bounds(child, None)

                # update overall bounds if child had valid bounds
                if child_bounds:
                    overall = [
                        min(overall[0], child_bounds[0]),
                        min(overall[1], child_bounds[1]),
                        max(overall[2], child_bounds[2]),
                        max(overall[3], child_bounds[3]),
                    ]

        # include anchor points in bounds calculation as well
        if hasattr(component, "anchor_points") and component.anchor_points:
            for anchor in component.anchor_points:
                # only include if visible or specifically needed for bounds
                if anchor and (anchor.show or anchor.debug):
                    anchor_b = anchor.get_world_bounds()
                    if anchor_b:
                        overall = [
                            min(overall[0], anchor_b[0]),
                            min(overall[1], anchor_b[1]),
                            max(overall[2], anchor_b[2]),
                            max(overall[3], anchor_b[3]),
                        ]

        # return tuple if bounds are valid, else None
        return tuple(overall) if overall[0] != float("inf") else None

    def _adjust_limits(self, context: Axes, root: Component, padding: float = 0.1):
        """set axis limits to encompass the rendered content."""
        # Use the potentially enhanced recursive bounds calculation
        bounds = self._get_recursive_world_bounds(root)

        if bounds:
            min_x, min_y, max_x, max_y = bounds
            # ensure non-zero width/height for padding calculation
            width = max(max_x - min_x, 1.0)
            height = max(max_y - min_y, 1.0)

            # Determine base padding amounts
            pad_x_base = max(width * padding, 5.0)
            pad_y_base = max(height * padding, 5.0)

            # Apply potentially larger padding for schematics or diagrams
            # Increased padding for schematics based on root component type or content
            # Avoid direct import of Network* types here
            is_schematic_type = (
                "Schematic" in root.__class__.__name__ or "Diagram" in root.__class__.__name__
            )

            if is_schematic_type or (
                isinstance(root, Container)
                and any(isinstance(c, Connection) for c in root.children)
            ):
                pad_x = max(pad_x_base, width * 0.15, 20.0)  # increased factor and minimum
                pad_y = max(pad_y_base, height * 0.15, 20.0)  # increased factor and minimum
            else:
                pad_x = pad_x_base
                pad_y = pad_y_base

            context.set_xlim(min_x - pad_x, max_x + pad_x)
            context.set_ylim(min_y - pad_y, max_y + pad_y)
            # self._log_debug(f"adjusted limits: x=({min_x-pad_x:.1f}, {max_x+pad_x:.1f}), y=({min_y-pad_y:.1f}, {max_y+pad_y:.1f})")
        else:
            # default limits if no bounds found
            context.set_xlim(0, 100)
            context.set_ylim(0, 100)
            # self._log_debug("no valid bounds found, setting default limits (0,100).")

        # ensure equal aspect ratio after setting limits
        context.set_aspect("equal", adjustable="box")

    # --- Primitive Rendering Methods ---
    # (render_path, _create_rounded_rect_path, render_rectangle, render_svg,
    #  render_text, _render_text_paths, measure_text remain the same)
    def render_path(
        self,
        context: Axes,
        path_data: SVGPathData,
        matrix: np.ndarray,
        line_width_mode: str = "data",  # default changed from 'point'
        color_remap: Optional[Dict[str, Optional[str]]] = None,
        component_id: Optional[str] = None,  # added for logging context
    ):
        """renders a single svg path."""
        comp_id_str = component_id or "unknown"
        # self._log_debug(f"render_path called for component '{comp_id_str}'", path_data.d[:50])
        try:
            # dynamically import to avoid hard dependency if not used
            from svgpath2mpl import parse_path
        except ImportError:
            logger.error("svgpath2mpl is required for rendering SVG paths. please install it.")
            return

        try:
            mpl_path = parse_path(path_data.d)
            # self._log_debug(f"  parsed path: {mpl_path.vertices.shape[0]} vertices")

            # combine component matrix with potential path transform
            final_matrix = matrix
            if path_data.transform:
                # basic matrix transform parsing (could be more robust)
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
                            # self._log_debug(f"  applied path transform: {vals}")
                    except ValueError:
                        # self._log_debug(f"  could not parse path transform matrix: {path_data.transform}")
                        pass  # ignore invalid matrix format

            # create matplotlib transform
            transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData

            # apply color remapping
            remap = color_remap or {}
            fill_color_in = path_data.fill
            stroke_color_in = path_data.stroke

            # use normalized colors from path_data, then remap
            fill_color_out = remap.get(fill_color_in, fill_color_in) if fill_color_in else None
            stroke_color_out = (
                remap.get(stroke_color_in, stroke_color_in) if stroke_color_in else None
            )

            # convert to matplotlib color format ('none' or actual color)
            final_facecolor = "none" if fill_color_out is None else fill_color_out
            final_edgecolor = "none" if stroke_color_out is None else stroke_color_out

            current_path_lw_data = path_data.stroke_width

            # determine if linewidth should be dynamic
            is_data_width = (
                line_width_mode == "data" and final_edgecolor != "none" and current_path_lw_data > 0
            )

            # set initial linewidth: 0 for data width (will be updated), fixed value otherwise
            initial_lw_points = 0.0 if is_data_width else current_path_lw_data

            # get linestyle
            linestyle = _get_mpl_linestyle(path_data)

            # create and add the patch
            with _no_autoscale(context):
                patch = mpatches.PathPatch(
                    mpl_path,
                    facecolor=final_facecolor,
                    edgecolor=final_edgecolor,
                    linewidth=initial_lw_points,
                    linestyle=linestyle,
                    transform=transform,
                    capstyle="round",  # consider making configurable?
                    joinstyle="round",  # consider making configurable?
                )
                context.add_patch(patch)

                # if using data width, track the patch and set initial estimate
                if is_data_width:
                    self.track_patch(patch, current_path_lw_data)
                    # set initial width based on current scale factor
                    initial_points_estimate = _linewidth_in_points(current_path_lw_data, context)
                    patch.set_linewidth(initial_points_estimate)
                    # self._log_debug(f"  added path patch (data width initial={initial_points_estimate:.2f})")
                # else:
                # self._log_debug(f"  added path patch (point width={initial_lw_points:.2f})")

        except Exception as e:
            self._log_debug(
                f"error rendering path '{path_data.d[:30]}...' for '{comp_id_str}': {e}", e
            )

    def _create_rounded_rect_path(self, x, y, w, h, radius):
        """helper to create a matplotlib path for a rounded rectangle."""
        # handle zero radius (sharp corners)
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
            return  # cannot render zero-size rectangle

        # base transform for the rectangle's local coordinates
        transform = mtransforms.Affine2D(matrix=matrix) + context.transData

        # --- render shadow (if defined) ---
        if style.shadow and style.shadow.blur_radius > 0:
            shadow = style.shadow
            # estimate scale factor for shadow resolution
            scale_factor = _get_point_scale_factor(context)
            points_per_unit = max(scale_factor, 1.0)  # avoid zero scale
            # number of layers based on blur radius and resolution parameter
            num_layers = min(
                100, max(4, int((shadow.blur_radius * points_per_unit**0.75) * shadow.resolution))
            )

            try:
                # get base color and alpha for shadow
                base_color_rgb, base_alpha = to_rgba(shadow.color)[:3], to_rgba(shadow.color)[3]
            except ValueError:
                base_color_rgb, base_alpha = (0, 0, 0), 0.5  # default shadow

            accumulated_alpha = 0.0  # track alpha to avoid excessive opacity
            min_render_alpha = 1.0 / 256.0  # minimum alpha step

            # render shadow layers from back to front (most blurred to least blurred)
            for i in range(num_layers - 1, -1, -1):
                layer_frac = (
                    (i / (num_layers - 1)) if num_layers > 1 else 1.0
                )  # progress from 1 to 0
                alpha_frac = (
                    ((num_layers - 1 - i) / (num_layers - 1)) if num_layers > 1 else 0.0
                )  # progress from 0 to 1
                # target intensity for this layer (distributes total alpha across layers)
                target_intensity = (
                    base_alpha * (1 - alpha_frac**1.5) / num_layers  # non-linear falloff
                    if num_layers > 0
                    else base_alpha
                )
                accumulated_alpha += target_intensity
                # quantize alpha to avoid excessive unique colors
                patch_alpha = max(
                    0, min(1.0, round(accumulated_alpha / min_render_alpha) * min_render_alpha)
                )

                # skip rendering if alpha is negligible
                if patch_alpha <= min_render_alpha / 2.0:
                    continue

                # adjust accumulated alpha by the amount actually rendered
                accumulated_alpha = max(0, accumulated_alpha - patch_alpha)

                # calculate spread for this layer (blur + base spread)
                spread = shadow.spread + shadow.blur_radius * (1 - layer_frac)
                # dimensions and position of the shadow layer rectangle
                sw, sh = w + 2 * spread, h + 2 * spread
                sx, sy = -spread + shadow.offset_x, -spread + shadow.offset_y
                # radius of the shadow layer (increases with spread)
                s_radius = (
                    min(style.corner_radius + spread, min(sw, sh) / 2.0) if min(sw, sh) > 0 else 0
                )

                # skip if shadow layer has no size
                if sw <= 0 or sh <= 0:
                    continue

                # create path and add patch for this shadow layer
                layer_path = self._create_rounded_rect_path(sx, sy, sw, sh, s_radius)
                with _no_autoscale(context):
                    context.add_patch(
                        mpatches.PathPatch(
                            layer_path,
                            facecolor=(*base_color_rgb, patch_alpha),
                            edgecolor="none",
                            lw=0,
                            transform=transform,  # use the main rectangle's transform
                            clip_on=False,  # allow shadow to extend beyond axes limits if needed
                        )
                    )

        # --- render main rectangle (background and border) ---
        facecolor = style.background_color or "none"
        edgecolor = style.border_color or "none"
        linewidth_data = style.border_width
        width_mode = style.border_width_mode

        # only render if there's a fill or a visible border
        if facecolor != "none" or (edgecolor != "none" and linewidth_data > 0):
            main_path = self._create_rounded_rect_path(0, 0, w, h, style.corner_radius)

            # determine if linewidth is dynamic
            is_data_width = width_mode == "data" and edgecolor != "none" and linewidth_data > 0
            # set initial points (0 for data width, fixed value otherwise)
            initial_lw_points = 0.0 if is_data_width else linewidth_data

            # create dummy SVGPathData just to use _get_mpl_linestyle helper
            border_path_data = SVGPathData(
                d="",  # path 'd' is not used here
                line_style=style.border_style,
                dash_array=style.dash_sequence,
                dash_offset=style.dash_offset,
            )
            linestyle = _get_mpl_linestyle(border_path_data)

            # add the main rectangle patch
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

                # track patch if linewidth is dynamic
                if is_data_width:
                    self.track_patch(main_patch, linewidth_data)
                    # set initial width estimate
                    initial_points_estimate = _linewidth_in_points(linewidth_data, context)
                    main_patch.set_linewidth(initial_points_estimate)
                    # self._log_debug(f"added main rectangle patch (data width initial={initial_points_estimate:.2f})")
                # else:
                # self._log_debug(f"added main rectangle patch (point width={initial_lw_points:.2f})")

    def render_svg(self, context: Axes, svg_element: SVGElement, matrix: np.ndarray):
        """renders the paths within an svg element."""
        comp_id = svg_element.id or "unknown_svg"
        # ensure the svg data is parsed
        if not svg_element._parsed_svg_content:
            # this should ideally be handled by the component's validator or pre-render logic
            logger.warning(f"svg content not parsed for {comp_id}, attempting parse now.")
            svg_element._parse_and_validate_svg()
            if not svg_element._parsed_svg_content:
                if svg_element.debug:
                    self.render_debug(context, svg_element, matrix)
                return  # exit if still not parsed

        # check if there are paths to render
        if not svg_element._parsed_svg_content.paths:
            # self._log_debug(f"no paths found in parsed svg for {comp_id}")
            if svg_element.debug:
                self.render_debug(context, svg_element, matrix)
            return

        svg_data = svg_element._parsed_svg_content
        viewBox = svg_data.viewBox
        # use svg dimensions from parsed content
        svg_dims = Size(svg_data.width, svg_data.height)
        # use component's current dimensions (set by layout)
        comp_dims = svg_element._dimensions

        # handle case where component dimensions might be zero (e.g., if hidden)
        if comp_dims.width <= 1e-6 or comp_dims.height <= 1e-6:
            # self._log_debug(f"skipping svg render for {comp_id} due to zero component dimensions.")
            if svg_element.debug:
                self.render_debug(context, svg_element, matrix)
            return

        # determine the effective viewbox
        if viewBox:
            vb_x, vb_y, vb_w, vb_h = viewBox
        else:
            # default viewbox to svg dimensions if not specified
            vb_x, vb_y = 0, 0
            vb_w, vb_h = svg_dims.width, svg_dims.height

        # handle zero-size viewbox to prevent division by zero
        if vb_w <= 1e-6:
            vb_w = 1.0
        if vb_h <= 1e-6:
            vb_h = 1.0

        # calculate scaling to fit viewbox content into component dimensions
        scale_x = comp_dims.width / vb_w
        scale_y = comp_dims.height / vb_h

        # transformation matrix: viewbox coords -> component local coords (y-up)
        # 1. scale viewbox to component size
        # 2. translate by negative viewbox origin (scaled)
        # 3. flip y-axis and translate to component height
        svg_internal_matrix = np.array(
            [
                [scale_x, 0, -scale_x * vb_x],
                [0, -scale_y, scale_y * vb_y + comp_dims.height],  # flip y and translate
                [0, 0, 1],
            ]
        )

        # final transformation: world * component_local * svg_internal
        final_matrix = matrix @ svg_internal_matrix

        # retrieve renderer options set on the component
        default_lw_mode = svg_element.line_width_mode
        color_remap = svg_element.color_remap

        # render each path within the svg
        for path_data in svg_data.paths:
            # pass the final matrix and options to render_path
            self.render_path(
                context, path_data, final_matrix, default_lw_mode, color_remap, component_id=comp_id
            )

        # render debug box if enabled
        if svg_element.debug:
            self.render_debug(context, svg_element, matrix)

    def render_text(self, context: Axes, text_component: Text, matrix: np.ndarray):
        """renders text using cached svg paths."""
        comp_id = text_component.id or "unknown_text"
        if not text_component.text or not text_component.show:
            # self._log_debug(f"skipping render_text for {comp_id}: no text or not shown")
            return

        # measure text if cache is missing (should have happened during layout)
        if text_component._svg_cache is None:
            self._log_debug(f"text '{comp_id}' svg_cache is None, measuring now.")
            self.measure_text(text_component)

        # check cache again after attempting measurement
        svg_cache = text_component._svg_cache
        if isinstance(svg_cache, SVGTextContent) and svg_cache.glyph_paths:
            self._render_text_paths(context, text_component, matrix)
        else:
            # fallback or if text is empty after measurement
            self._log_debug(f"no valid glyph paths found for text '{comp_id}', skipping render.")
            if text_component.debug:  # still render debug box if needed
                self.render_debug(context, text_component, matrix)

    def _render_text_paths(self, context: Axes, text_comp: Text, matrix: np.ndarray):
        """renders text using the pre-calculated matplotlib paths."""
        comp_id = text_comp.id or "unknown_text_path"
        svg_content = text_comp._svg_cache
        if not svg_content or not svg_content.glyph_paths:
            self._log_debug(f"render_text_paths: no glyph paths in cache for {comp_id}")
            return  # cannot render without paths

        # allocated size (from layout)
        alloc_w, alloc_h = text_comp._dimensions.width, text_comp._dimensions.height
        # natural measured size (from measurement phase)
        measured_w, measured_h = svg_content.measured_width, svg_content.measured_height

        # calculate alignment offset to position the measured box within the allocated box
        dx = 0.0
        dy = 0.0
        # horizontal alignment
        if text_comp.align == "center":
            dx = (alloc_w - measured_w) / 2.0
        elif text_comp.align == "right":
            dx = alloc_w - measured_w
        # vertical alignment (relative to top-left origin of allocated box)
        if text_comp.vertical_align == "middle":
            # text paths origin is bottom-left of text block's bounding box
            # align center of measured height with center of allocated height
            dy = (alloc_h - measured_h) / 2.0
        elif text_comp.vertical_align == "bottom":
            dy = 0  # paths origin is already at bottom
        elif text_comp.vertical_align == "top":
            dy = alloc_h - measured_h  # shift up by difference

        # create alignment transform matrix
        align_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

        # combine component's world matrix with alignment matrix
        final_matrix = matrix @ align_matrix
        # create the final matplotlib transform
        transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData

        # get text color
        color = getattr(text_comp, "color", "black")
        if not color or color.lower() == "none":
            color = "black"  # default if color is invalid

        # self._log_debug(f"rendering {len(svg_content.glyph_paths)} text paths for {comp_id}", {
        #     "alloc": (alloc_w, alloc_h), "measured": (measured_w, measured_h),
        #     "align_offset": (dx, dy), "color": color
        # })

        # render each glyph path
        with _no_autoscale(context):
            for path_info in svg_content.glyph_paths:
                # path_info should contain a 'path' key with the MplPath object
                mpl_path = path_info.get("path")
                if mpl_path:
                    context.add_patch(
                        mpatches.PathPatch(
                            mpl_path,
                            facecolor=color,
                            edgecolor="none",  # text paths usually have no stroke
                            linewidth=0,
                            transform=transform,
                        )
                    )
                else:
                    self._log_debug(f"missing 'path' in glyph_paths cache for {comp_id}")

        # self._log_debug(f"rendered text '{comp_id}' using {len(svg_content.glyph_paths)} paths.")

    def measure_text(self, text_comp: Text) -> Size:
        """
        measures text by generating matplotlib TextPaths and calculating their
        bounding box. The dimensions are in 'data' units relative to font_size.
        """
        if not text_comp.text:
            text_comp._svg_cache = None
            return Size()

        # --- create matplotlib paths for text ---
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
        base_line_h = text_comp.font_size  # use font_size as base height unit
        effective_line_height = base_line_h * (1.0 + text_comp.line_spacing)

        # generate path for each line
        for i, line in enumerate(lines):
            if not line.strip():  # handle empty lines for spacing
                # advance cursor but don't add paths
                if i < len(lines) - 1:
                    y_cursor -= effective_line_height
                continue

            # generate path for the line at (0, 0) origin
            # size=font_size means the path vertices will be scaled by font_size
            mpl_path = TextPath((0, 0), line, size=text_comp.font_size, prop=props)
            # store the path along with its baseline y-offset
            all_glyph_paths.append({"path": mpl_path, "y_offset": y_cursor})

            # update overall bounds based on this path's vertices *at its baseline*
            if mpl_path.vertices.shape[0] > 0:
                line_vertices = mpl_path.vertices + [0, y_cursor]  # translate to baseline
                min_x_overall = min(min_x_overall, np.min(line_vertices[:, 0]))
                max_x_overall = max(max_x_overall, np.max(line_vertices[:, 0]))
                min_y_overall = min(min_y_overall, np.min(line_vertices[:, 1]))
                max_y_overall = max(max_y_overall, np.max(line_vertices[:, 1]))
            else:  # handle space or empty path case if TextPath generates it
                max_x_overall = max(max_x_overall, 0)  # ensure width is at least 0
                min_x_overall = min(min_x_overall, 0)
                # y bounds won't change for empty path

            # advance cursor for next line
            if i < len(lines) - 1:
                y_cursor -= effective_line_height

        # --- calculate final dimensions and adjust path origins ---
        if not all_glyph_paths:  # if text was only whitespace/empty lines
            text_comp._svg_cache = None
            return Size()

        # if bounds were never updated (e.g., text path failed?), return zero size
        if min_x_overall == float("inf"):
            measured_width = 0.0
            measured_height = 0.0
            final_paths = []
        else:
            measured_width = max(0, max_x_overall - min_x_overall)
            measured_height = max(0, max_y_overall - min_y_overall)

            # translate all paths so the entire text block's bottom-left corner is at (0,0)
            final_paths = []
            y_shift = -min_y_overall  # shift text up so min_y becomes 0
            x_shift = -min_x_overall  # shift text right so min_x becomes 0
            for p_info in all_glyph_paths:
                # apply both baseline offset and overall shift
                final_transform = mtransforms.Affine2D().translate(
                    x_shift, p_info["y_offset"] + y_shift
                )
                final_paths.append({"path": p_info["path"].transformed(final_transform)})

        # --- store results in cache ---
        text_comp._svg_cache = SVGTextContent(
            glyph_paths=tuple(final_paths),
            measured_width=measured_width,
            measured_height=measured_height,
        )

        measured_size = Size(width=measured_width, height=measured_height)
        # self._log_debug(f"measured text '{text_comp.id}', nat_size={measured_size}")
        return measured_size

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
        """renders the main curve of a connection. caps are handled by connection.render."""
        comp_id = connection.id or "unknown_connection"
        # self._log_debug(f"render_connection_curve for {comp_id}")
        # self._log_debug(f"  path_string: {path_string[:100]}...")
        # self._log_debug(f"  matrix:", matrix)

        # create svg path data from connection properties for the main curve
        path_data = SVGPathData(
            d=path_string,
            stroke=_normalize_color(connection.color),  # use normalized color
            stroke_width=connection.line_width,
            fill="none",  # connections are typically not filled
            line_style=connection.line_style,
            dash_array=connection.dash_array,
            dash_offset=connection.dash_offset,
        )
        # self._log_debug(f"  path_data for render_path:", path_data)

        # render using the generic path method, enforcing data linewidth mode
        self.render_path(context, path_data, matrix, line_width_mode="data", component_id=comp_id)

    def render_debug(self, context: Axes, component: Component, matrix: np.ndarray):
        """renders debug bounding box and origin."""
        # check if component has dimensions and they are valid
        if (
            not hasattr(component, "_dimensions")
            or component._dimensions.width <= 0
            or component._dimensions.height <= 0
        ):
            # self._log_debug(f"skipping debug render for {component.id}: invalid dimensions {getattr(component, '_dimensions', 'N/A')}")
            return

        w, h = component._dimensions.width, component._dimensions.height
        # transform for the bounding box corners
        transform = mtransforms.Affine2D(matrix=matrix) + context.transData
        lw = 0.5  # fixed point size for debug lines

        with _no_autoscale(context):
            # bounding box
            context.add_patch(
                mpatches.Rectangle(
                    (0, 0), w, h, fill=False, ec="red", ls="--", lw=lw, transform=transform
                )
            )

            # origin marker (crosshair)
            # transform the local origin (0,0) to world, then to display
            origin_world = (matrix @ [0, 0, 1])[:2]
            origin_disp = context.transData.transform(origin_world)

            # draw crosshair in display coordinates (so it's fixed size)
            sz = 4  # size of crosshair arms in pixels
            context.add_line(
                plt.Line2D(
                    [origin_disp[0] - sz, origin_disp[0] + sz],
                    [origin_disp[1], origin_disp[1]],
                    color="red",
                    lw=lw,
                    ls="-",
                    transform=None,  # use display coords directly
                    solid_capstyle="butt",  # sharp ends for lines
                )
            )
            context.add_line(
                plt.Line2D(
                    [origin_disp[0], origin_disp[0]],
                    [origin_disp[1] - sz, origin_disp[1] + sz],
                    color="red",
                    lw=lw,
                    ls="-",
                    transform=None,  # use display coords directly
                    solid_capstyle="butt",
                )
            )

    def render_to_output(self, context: Axes, output=None, **kwargs):
        """saves the rendered figure to a file or stream."""
        # ensure linewidths are updated before saving
        self.refresh_linewidths()

        if not hasattr(context, "figure"):
            raise ValueError("matplotlib context (Axes) must belong to a figure")

        # default save options (tight bounding box)
        opts = {"bbox_inches": "tight", "pad_inches": 0.1, **kwargs}

        if output:
            try:
                # self._log_debug(f"saving figure to {output} with options {opts}")
                context.figure.savefig(output, **opts)
            except Exception as e:
                self._log_debug(f"error saving figure: {e}", e)
                raise  # re-raise the exception
        # return the figure object (useful for notebooks or further manipulation)
        return context.figure
