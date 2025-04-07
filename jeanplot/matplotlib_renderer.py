# jeanplot/matplotlib_renderer.py
from typing import Optional, Any, List, Dict, Tuple, Union, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from pydantic import Field, BaseModel
from matplotlib.path import Path
from .component import Component
from .models import Size, BoxStyle, LineWidthMode
from .renderer import BaseRenderer
from .svg import SVGElement, SVGContent, SVGTextContent, SVGPathData, arc_to_bezier
from matplotlib.colors import to_rgba


def linewidth_from_data_units(linewidth, axis, reference: Literal["x", "y", "avg", "max"] = "max"):
    """convert linewidth in data units to points"""
    fig = axis.get_figure()

    if reference == "x":
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == "y":
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    else:
        # avg of x and y scales
        length_x = fig.bbox_inches.width * axis.get_position().width
        value_range_x = np.diff(axis.get_xlim())
        length_y = fig.bbox_inches.height * axis.get_position().height
        value_range_y = np.diff(axis.get_ylim())

        # handle potential zero range
        if (
            value_range_x is None
            or value_range_y is None
            or np.any(value_range_x <= 0)
            or np.any(value_range_y <= 0)
        ):
            return 1.0  # fallback point size

        if reference == "avg":
            value_range = (value_range_x + value_range_y) / 2
            length = (length_x + length_y) / 2
        elif reference == "max":
            value_range = np.maximum(value_range_x, value_range_y)
            length = (length_x + length_y) / 2

    # handle potential zero range
    if value_range is None or np.any(value_range <= 0):
        return 1.0  # fallback point size

    length *= 72
    # index 0 assuming single range diff
    return linewidth * (length / value_range[0]) if value_range[0] > 0 else 1.0


class MatplotlibRenderer(BaseRenderer):
    RENDERER_NAME = "matplotlib"

    def __init__(self, debug=False):
        super().__init__()
        self.data_width_patches = []
        self.debug = debug

    def debug_print(self, msg):
        if self.debug:
            print(f"[MatplotlibRenderer] {msg}")

    def refresh_linewidths(self, context):
        """update linewidths for data-unit elements"""
        for patch, width in self.data_width_patches:
            if hasattr(patch, "set_linewidth"):
                new_width = linewidth_from_data_units(width, context)
                patch.set_linewidth(new_width)

    def track_patch(self, patch, width, context):
        """track a patch with data-unit width"""
        self.data_width_patches.append((patch, width))
        return patch

    def render_path(self, context, path_data: SVGPathData, matrix: np.ndarray):
        """renders a single path described by SVGPathData"""
        try:
            from svgpath2mpl import parse_path
        except ImportError:
            print("svgpath2mpl is required for path rendering")
            return

        try:
            path = parse_path(path_data.d)
            transform = mtransforms.Affine2D(matrix=matrix) + context.transData

            fill = path_data.fill if path_data.fill != "none" else "none"
            stroke = path_data.stroke if path_data.stroke != "none" else "none"
            linewidth = path_data.stroke_width
            use_data_width = False  # assume point width for perimeter path

            linestyle_map = {"solid": "-", "dashed": "--", "dotted": ":", "custom": "-"}
            linestyle = linestyle_map.get(path_data.line_style, "-")

            patch = mpatches.PathPatch(
                path,
                facecolor=fill,
                edgecolor=stroke,
                linewidth=1.0 if use_data_width else linewidth,
                linestyle=linestyle,
                transform=transform,
                capstyle="round",  # often looks better for paths
                joinstyle="round",  # often looks better for paths
            )

            # apply custom dash array if specified
            if path_data.dash_array and hasattr(patch, "set_dashes"):
                # ensure dash_array is a tuple, not numpy array
                dashes = tuple(path_data.dash_array)
                patch.set_linestyle("-")  # custom dashes require solid base style
                patch.set_dashes(dashes)
                if path_data.dash_offset != 0.0 and hasattr(patch, "set_dash_offset"):
                    patch.set_dash_offset(path_data.dash_offset)

            if use_data_width:
                self.track_patch(patch, linewidth, context)

            context.add_patch(patch)
        except Exception as e:
            print(f"Error rendering SVG path data: {e} (Path: {path_data.d})")

    def create_context(self, width=None, height=None, dpi=100, ax=None, **kwargs):
        """create matplotlib figure and axes"""
        if ax:
            return ax
        if not (width and height):
            raise ValueError("width and height required for new figure")

        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi, **kwargs)
        self.data_width_patches = []
        return ax

    def render_component(self, context, component, parent_matrix=None, adjust_lims=True):
        """render component to context"""
        component.measure_and_layout(self)
        if adjust_lims:
            self.adjust_limits(context, component)

        for cb in self.pre_render_callbacks:
            cb(context)

        self.data_width_patches = []

        # calculate final matrix including potential y-flip for matplotlib
        world_matrix = component.compute_world_matrix(parent_matrix)

        # apply y-flip if rendering directly to matplotlib axes (common case)
        # assumption: context is a matplotlib axes object
        if isinstance(context, Axes):
            _, view_h = context.get_figure().get_size_inches() * context.get_figure().get_dpi()
            ax_pos = context.get_position()
            ax_h_pixels = view_h * ax_pos.height
            # flip matrix for the whole render pass, since matplotlib uses "inverted" y-axis
            # flip needs to account for the component's final position in world coords
            # find world origin of component
            origin_world = world_matrix @ np.array([0, 0, 1])
            flip_y_coord = (
                origin_world[1] + component._dimensions.height
            )  # top edge in world coords
            # This flip seems complex to get right universally with transforms...
            # alternative: apply flip in plotting calls? maybe simpler.
            # let's try applying flip locally within render methods instead.
            final_matrix = world_matrix  # don't flip globally
        else:
            final_matrix = world_matrix  # non-matplotlib context

        component.render(self, context, final_matrix)
        self.refresh_linewidths(context)  # refresh after all patches added

        for cb in self.post_render_callbacks:
            cb(context)

    def adjust_limits(self, context, root, padding=0.1):  # increased padding
        """adjust plot limits to fit components"""
        min_x, min_y, max_x, max_y = self._calculate_world_bounds(root)

        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)

        pad_x, pad_y = width * padding, height * padding

        # ensure non-zero padding even for zero-sized components
        pad_x = max(pad_x, 5)
        pad_y = max(pad_y, 5)

        context.set_xlim(min_x - pad_x, max_x + pad_x)
        # Matplotlib Y-axis is typically inverted in plots, but our coords are standard
        # Set ylim normally, let aspect ratio handle visual scaling
        context.set_ylim(min_y - pad_y, max_y + pad_y)
        # Ensure equal aspect ratio AFTER setting limits
        context.set_aspect("equal", adjustable="box")

    def _calculate_world_bounds(self, comp, parent_matrix=None):
        """calculate world bounds recursively"""
        if not hasattr(comp, "_dimensions"):  # skip if component hasn't been measured
            return float("inf"), float("inf"), float("-inf"), float("-inf")

        world = comp.compute_world_matrix(parent_matrix)
        w, h = comp._dimensions.width, comp._dimensions.height

        # if dimensions are zero, treat as a point at the origin
        if w == 0 and h == 0:
            world_origin = (world @ np.array([0, 0, 1])).T
            min_x, min_y = world_origin[0], world_origin[1]
            max_x, max_y = min_x, min_y
        else:
            corners = np.array(
                [
                    [0, 0, 1],
                    [w, 0, 1],
                    [0, h, 1],
                    [w, h, 1],
                ]
            )
            world_corners = (world @ corners.T).T
            if world_corners.shape[0] > 0:
                min_x = np.min(world_corners[:, 0])
                min_y = np.min(world_corners[:, 1])
                max_x = np.max(world_corners[:, 0])
                max_y = np.max(world_corners[:, 1])
            else:  # should not happen if w,h > 0
                min_x = min_y = max_x = max_y = 0

        # include children bounds
        if hasattr(comp, "children") and comp.children:
            for child in comp.children:
                cx, cy, cX, cY = self._calculate_world_bounds(child, world)
                min_x, min_y = min(min_x, cx), min(min_y, cy)
                max_x, max_y = max(max_x, cX), max(max_y, cY)

        return min_x, min_y, max_x, max_y

    def _create_rounded_rect_path(self, x, y, w, h, radius):
        """helper to create the Path object for a rounded rectangle."""
        if radius < 1e-3:  # simple rectangle
            verts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        else:
            verts = []
            codes = []
            # start point on top edge, after top-left corner
            start_point = (x + radius, y + h)
            verts.append(start_point)
            codes.append(Path.MOVETO)
            # line to top-right corner start
            verts.append((x + w - radius, y + h))
            codes.append(Path.LINETO)
            # top-right arc (90 to 0 degrees)
            _, cp1, cp2, p_end = arc_to_bezier(x + w - radius, y + h - radius, radius, 90, 0)
            verts.extend([cp1, cp2, p_end])
            codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
            # line down right edge
            verts.append((x + w, y + radius))
            codes.append(Path.LINETO)
            # bottom-right arc (0 to -90 degrees / 270)
            _, cp1, cp2, p_end = arc_to_bezier(x + w - radius, y + radius, radius, 0, -90)
            verts.extend([cp1, cp2, p_end])
            codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
            # line across bottom edge
            verts.append((x + radius, y))
            codes.append(Path.LINETO)
            # bottom-left arc (-90 to -180 degrees / 180)
            _, cp1, cp2, p_end = arc_to_bezier(x + radius, y + radius, radius, -90, -180)
            verts.extend([cp1, cp2, p_end])
            codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
            # line up left edge
            verts.append((x, y + h - radius))
            codes.append(Path.LINETO)
            # top-left arc (-180 to -270 degrees / 90)
            _, cp1, cp2, p_end = arc_to_bezier(x + radius, y + h - radius, radius, -180, -270)
            verts.extend([cp1, cp2, p_end])
            codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
            # closepoly code and corresponding (dummy) vertex
            codes.append(Path.CLOSEPOLY)
            verts.append(start_point)

        return Path(verts, codes)

    def render_rectangle(
        self, context, bounds: Size, style: BoxStyle, matrix: np.ndarray, component=None
    ):
        """render a rectangle with optional shadow (alpha accumulation)"""
        comp_id = getattr(component, "id", "N/A")
        comp_cls = component.__class__.__name__ if component else "Unknown"
        bg_color = style.background_color if style else "NoStyleObject"
        print(f"Renderer: Drawing rect for {comp_cls}(id={comp_id}) with background: {bg_color}")

        w, h = bounds.width, bounds.height
        facecolor_main = style.background_color or "none"
        edgecolor_main = style.border_color or "none"
        linewidth_main = style.border_width
        width_mode = style.border_width_mode
        max_radius_main = min(w, h) / 2
        radius_main = min(style.corner_radius, max_radius_main) if max_radius_main > 0 else 0

        if style.shadow and style.shadow.blur_radius > 0:
            shadow = style.shadow
            points_per_data_unit = linewidth_from_data_units(1.0, context)
            # Use the user's refined formula for num_layers
            num_layers = max(
                4,
                int((shadow.blur_radius**1) * (points_per_data_unit**0.75) * shadow.resolution),
            )
            num_layers = min(num_layers, 100)  # cap layers for sanity

            try:
                rgba = to_rgba(shadow.color)
                base_color_rgb = rgba[:3]
                base_alpha = rgba[3]
            except ValueError:
                base_color_rgb = (0, 0, 0)
                base_alpha = 0.5

            accumulated_alpha_intensity = 0.0  # tracks theoretical alpha sum
            min_render_alpha = 1.0 / 256.0  # minimum alpha step we can render

            # draw layers from bottom (most blurred/spread) up
            for i in range(num_layers - 1, -1, -1):
                layer_fraction = (i / (num_layers - 1)) if num_layers > 1 else 1.0
                alpha_frac = ((num_layers - 1 - i) / (num_layers - 1)) if num_layers > 1 else 0.0

                # calculate the ideal intensity this layer contributes
                target_layer_intensity = (
                    base_alpha * (1 - alpha_frac**1.5) / num_layers
                )  # distribute total intensity

                # add this layer's ideal intensity to the accumulator
                accumulated_alpha_intensity += target_layer_intensity

                # determine if accumulated intensity is enough to draw a visible layer
                # convert accumulated intensity to 8-bit integer alpha
                alpha_to_draw_int = int(accumulated_alpha_intensity / min_render_alpha)

                if alpha_to_draw_int > 0:
                    alpha_to_draw_int = min(alpha_to_draw_int, 255)
                    patch_alpha = alpha_to_draw_int * min_render_alpha

                    accumulated_alpha_intensity -= patch_alpha
                    accumulated_alpha_intensity = max(0, accumulated_alpha_intensity)

                    # calculate geometry for this visible layer
                    additional_spread = shadow.blur_radius * (1 - layer_fraction)
                    current_spread = shadow.spread + additional_spread
                    current_offset_x = shadow.offset_x * (1 - layer_fraction)
                    current_offset_y = shadow.offset_y * (1 - layer_fraction)

                    sw, sh = w + 2 * current_spread, h + 2 * current_spread
                    sx, sy = -current_spread + current_offset_x, -current_spread + current_offset_y
                    s_radius = (
                        min(radius_main + current_spread, min(sw, sh) / 2) if min(sw, sh) > 0 else 0
                    )

                    if sw <= 0 or sh <= 0:
                        continue

                    layer_path = self._create_rounded_rect_path(sx, sy, sw, sh, s_radius)
                    rgba_tuple = base_color_rgb + (patch_alpha,)  # use calculated patch_alpha
                    layer_patch = mpatches.PathPatch(
                        layer_path,
                        facecolor=rgba_tuple,
                        edgecolor="none",
                        linewidth=0,
                        transform=mtransforms.Affine2D(matrix=matrix) + context.transData,
                    )
                    context.add_patch(layer_patch)
                # else: accumulated alpha is still < 1/256, skip drawing this layer

        if facecolor_main != "none" or (edgecolor_main != "none" and linewidth_main > 0):
            main_path = self._create_rounded_rect_path(0, 0, w, h, radius_main)

            linestyle_map = {"solid": "-", "dashed": "--", "dotted": ":", "custom": "-"}
            linestyle_main = linestyle_map.get(style.border_style, "-")

            if style.dash_sequence:
                linestyle_main = tuple(style.dash_sequence)
                linestyle_main = (style.dash_offset, linestyle_main)

            main_patch = mpatches.PathPatch(
                main_path,
                facecolor=facecolor_main,
                edgecolor=edgecolor_main,
                linewidth=1.0 if width_mode == "data" else linewidth_main,
                linestyle=linestyle_main,
                transform=mtransforms.Affine2D(matrix=matrix) + context.transData,
                capstyle="round",
                joinstyle="round",
            )
            if width_mode == "data":
                self.track_patch(main_patch, linewidth_main, context)
            elif (
                width_mode == "point"
                and linewidth_main > 0
                and hasattr(main_patch, "set_linewidth")
            ):
                main_patch.set_linewidth(linewidth_main)

            context.add_patch(main_patch)

    def render_svg(self, context, svg_element, matrix):
        """render an svg element"""
        if hasattr(svg_element, "svg_content") and isinstance(
            svg_element.svg_content, SVGTextContent
        ):
            self._render_text_paths(context, svg_element, matrix)
            return

        try:
            from svgpath2mpl import parse_path
        except ImportError:
            print("svgpath2mpl is required for SVG rendering")
            return

        if not hasattr(svg_element, "svg_content") or not isinstance(
            svg_element.svg_content, SVGContent
        ):
            print(f"Warning: SVGElement {svg_element.id} has no valid svg_content.")
            return

        paths = svg_element.svg_content.paths
        if not paths:
            return

        svg_dims = svg_element._dimensions
        viewBox = svg_element.svg_content.viewBox

        if svg_dims.width == 0 or svg_dims.height == 0:
            return  # don't render zero-size svgs

        if viewBox and (svg_dims.width > 1e-6 and svg_dims.height > 1e-6):
            vb_x, vb_y, vb_w, vb_h = viewBox
            vb_w = max(vb_w, 1e-6)  # avoid div by zero
            vb_h = max(vb_h, 1e-6)
            scale_x = svg_dims.width / vb_w
            scale_y = svg_dims.height / vb_h
            # Translate to handle viewBox origin, then scale
            translate_vb_origin = np.array([[1, 0, -vb_x], [0, 1, -vb_y], [0, 0, 1]])
            scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
            # Final SVG matrix: Scale first, then translate the scaled origin
            svg_matrix = scale_matrix @ translate_vb_origin
            # print(f"DEBUG SVG {svg_element.id}: viewBox={viewBox}, dims={svg_dims}, svg_matrix=\n{np.round(svg_matrix, 3)}") # Add debug
        else:
            svg_matrix = np.identity(3)  # Default to identity if no viewBox or zero dims
            # print(f"DEBUG SVG {svg_element.id}: No viewBox or zero dims, using identity svg_matrix") # Add debug

        # matrix received is conn_matrix (world matrix of the Connection component)
        final_matrix = matrix @ svg_matrix
        # Ensure transform uses this final matrix correctly
        transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData

        width_mode = svg_element.get_renderer_options(self.RENDERER_NAME).get("line_width_mode", "")

        for path_data in paths:
            try:
                path = parse_path(path_data.d)

                fill = path_data.fill
                if path_data.is_main_color:
                    fill = getattr(svg_element, "main_color", "blue")
                elif path_data.is_secondary_color:
                    fill = getattr(svg_element, "secondary_color", "green")

                stroke = path_data.stroke if path_data.stroke != "none" else "none"
                linewidth = path_data.stroke_width
                use_data_width = width_mode == "data" and stroke != "none" and linewidth > 0

                linestyle_map = {"solid": "-", "dashed": "--", "dotted": ":", "custom": "-"}
                linestyle = linestyle_map.get(path_data.line_style, "-")

                patch = mpatches.PathPatch(
                    path,
                    facecolor=fill if fill != "none" else "none",
                    edgecolor=stroke if stroke != "none" else "none",
                    linewidth=1.0 if use_data_width else linewidth,
                    linestyle=linestyle,
                    transform=transform,
                    capstyle="round",
                    joinstyle="round",
                )

                # apply custom dash array if specified
                if path_data.dash_array and hasattr(patch, "set_dashes"):
                    # ensure dash_array is a tuple
                    dashes = tuple(path_data.dash_array)
                    patch.set_linestyle("-")  # requires solid base style
                    patch.set_dashes(dashes)
                    if path_data.dash_offset != 0.0 and hasattr(patch, "set_dash_offset"):
                        patch.set_dash_offset(path_data.dash_offset)

                if use_data_width:
                    self.track_patch(patch, linewidth, context)

                context.add_patch(patch)
            except Exception as e:
                print(f"Error rendering SVG path: {path_data.d} - {e}")

    def render_debug(self, context, component, matrix):
        """render debug box around component"""
        if not hasattr(component, "_dimensions"):
            return  # skip if no dims

        debug_width = 0.5  # points
        w, h = component._dimensions.width, component._dimensions.height
        if w <= 0 or h <= 0:
            return  # skip zero-size debug box

        transform = mtransforms.Affine2D(matrix=matrix) + context.transData

        # draw rectangle boundary
        rect = mpatches.Rectangle(
            (0, 0),
            w,
            h,
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=debug_width,  # use point width for debug lines
            transform=transform,
        )
        # self.track_patch(rect, debug_width, context) # track debug box? maybe not needed
        context.add_patch(rect)

        # draw origin marker (X)
        marker_size = 4
        origin_marker = plt.Line2D(
            [0 - marker_size / 2, 0 + marker_size / 2],
            [0 - marker_size / 2, 0 + marker_size / 2],
            marker=None,
            color="red",
            markersize=0,
            linewidth=debug_width,
            linestyle="-",
            transform=transform,
        )
        origin_marker2 = plt.Line2D(
            [0 - marker_size / 2, 0 + marker_size / 2],
            [0 + marker_size / 2, 0 - marker_size / 2],
            marker=None,
            color="red",
            markersize=0,
            linewidth=debug_width,
            linestyle="-",
            transform=transform,
        )

        context.add_line(origin_marker)
        context.add_line(origin_marker2)

        # add label (use data coords relative to transformed origin)
        label_text = f"{component.id or 'unnamed'} ({w:.1f}x{h:.1f})"
        # place text slightly outside the top-left corner in data coords
        # need inverse transform to find appropriate data coords for text placement?
        # simpler: use annotation with offset in points
        context.annotate(
            label_text,
            xy=(0, h),
            xycoords=transform,  # top-left in component space
            xytext=(-2, 2),
            textcoords="offset points",  # offset points
            color="red",
            fontsize=5,
            ha="right",
            va="bottom",
        )

    def _render_text_paths(self, context, text_element, matrix):
        """render cached text paths"""
        svg_content = None
        if hasattr(text_element, "_svg_cache"):
            svg_content = text_element._svg_cache
        elif hasattr(text_element, "svg_content") and isinstance(
            text_element.svg_content, SVGTextContent
        ):
            svg_content = text_element.svg_content

        if not svg_content or not hasattr(svg_content, "text_paths") or not svg_content.text_paths:
            return

        # apply alignment offset before final transform
        text_w = text_element._dimensions.width
        text_h = text_element._dimensions.height
        measured_w = svg_content.measured_width
        measured_h = svg_content.measured_height

        # horizontal alignment
        dx = 0
        if text_element.align == "center":
            dx = (text_w - measured_w) / 2
        elif text_element.align == "right":
            dx = text_w - measured_w

        # vertical alignment
        dy = 0
        if text_element.vertical_align == "middle":
            dy = (text_h - measured_h) / 2
        elif text_element.vertical_align == "bottom":
            dy = text_h - measured_h

        align_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

        # transform setup
        # final_matrix includes parent matrix, local offset/transform, AND alignment offset
        final_matrix = matrix @ align_matrix
        transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData

        for path_info in svg_content.text_paths:
            color = getattr(text_element, "color", "black")
            path = path_info["path"]

            # create patch for the text path
            patch = mpatches.PathPatch(
                path,
                facecolor=color,
                edgecolor="none",
                transform=transform,
            )
            context.add_patch(patch)

    def render_to_output(self, context, output=None, **kwargs):
        """saves the figure context"""
        # self.refresh_linewidths(context) # refresh needed if widths changed
        if not hasattr(context, "figure"):
            raise ValueError("Context must be a matplotlib Axes object with a figure attribute.")

        if output:
            # remove non-standard args for savefig
            kwargs.pop("adjust_lims", None)
            context.figure.savefig(output, **kwargs)
        return context.figure

    def measure_text(self, text_comp) -> Size:
        """measure text with tight bounds based on actual glyph extents"""
        if not text_comp.text:
            text_comp._svg_cache = None
            return Size(0, 0)

        font_props = fm.FontProperties(
            family=text_comp.font_name or "sans-serif",
            weight=text_comp.font_weight,
            style=text_comp.font_style,
        )

        lines = text_comp.text.split("\n")
        max_width = 0
        paths_info = []  # store initial paths and metrics

        # use a dummy figure to get text path extents reliably
        fig = plt.figure()
        renderer = fig.canvas.get_renderer()

        for line in lines:
            if not line.strip():
                # get height of a space character for empty lines
                path = TextPath(
                    (0, 0), " ", size=text_comp.font_size, prop=font_props, usetex=False
                )
                bbox = path.get_extents()
                paths_info.append(
                    {
                        "text": line,
                        "path_at_origin": path,
                        "width": 0,
                        "height": bbox.height if bbox.height > 0 else text_comp.font_size,
                        "min_x": 0,
                        "min_y": bbox.y0,
                        "max_x": 0,
                        "max_y": bbox.y1,
                        "is_empty": True,
                    }
                )
                continue

            path = TextPath((0, 0), line, size=text_comp.font_size, prop=font_props, usetex=False)
            try:
                bbox = path.get_extents()
                width = bbox.width
                height = bbox.height
                min_x, min_y = bbox.x0, bbox.y0
                max_x, max_y = bbox.x1, bbox.y1

                paths_info.append(
                    {
                        "text": line,
                        "path_at_origin": path,
                        "width": width,
                        "height": height,
                        "min_x": min_x,
                        "min_y": min_y,
                        "max_x": max_x,
                        "max_y": max_y,
                        "is_empty": False,
                    }
                )

                if width > max_width:
                    max_width = width

            except Exception as e:
                print(f"Warning: Failed to measure text path for '{line}': {e}. Using estimation.")
                est_w = len(line) * text_comp.font_size * 0.6
                est_h = text_comp.font_size
                paths_info.append(
                    {
                        "text": line,
                        "path_at_origin": path,
                        "width": est_w,
                        "height": est_h,
                        "min_x": 0,
                        "min_y": -est_h * 0.2,
                        "max_x": est_w,
                        "max_y": est_h * 0.8,
                        "is_empty": False,
                    }
                )
                if est_w > max_width:
                    max_width = est_w

        plt.close(fig)

        if not paths_info:
            return Size(0, 0)

        first_line_metrics = next((p for p in paths_info if not p["is_empty"]), paths_info[0])
        base_line_height = first_line_metrics["height"] * (1 + text_comp.line_spacing)
        line_height = (
            base_line_height
            if base_line_height > 0
            else text_comp.font_size * (1 + text_comp.line_spacing)
        )

        y_cursor = 0
        actual_min_y = float("inf")
        actual_max_y = float("-inf")
        positioned_paths_info = []

        for i, line in enumerate(lines):
            initial_path_info = paths_info[i]
            target_y_baseline = y_cursor

            line_min_y = target_y_baseline + initial_path_info["min_y"]
            line_max_y = target_y_baseline + initial_path_info["max_y"]

            actual_min_y = min(actual_min_y, line_min_y)
            actual_max_y = max(actual_max_y, line_max_y)

            positioned_paths_info.append(
                {
                    "text": line,
                    "target_y_baseline": target_y_baseline,
                }
            )

            y_cursor -= line_height

        total_height = (
            actual_max_y - actual_min_y if actual_max_y > actual_min_y else text_comp.font_size
        )
        y_offset_to_final_origin = -actual_min_y

        final_paths_for_cache = []
        for i, pos_info in enumerate(positioned_paths_info):
            line_text = pos_info["text"]
            final_y_baseline = pos_info["target_y_baseline"] + y_offset_to_final_origin

            final_path = TextPath(
                (0, final_y_baseline),
                line_text,
                size=text_comp.font_size,
                prop=font_props,
                usetex=False,
            )
            final_paths_for_cache.append({"path": final_path})  # store the newly created path

        svg_content = SVGTextContent(
            width=max_width,
            height=total_height,
            viewBox=(0, 0, max_width, total_height),  # origin is bottom-left of measured bounds
            paths=[],
            text_paths=final_paths_for_cache,
            measured_width=max_width,
            measured_height=total_height,
        )
        text_comp._svg_cache = svg_content

        measured = Size(width=max_width, height=total_height)
        size = Size(
            width=min(
                max(text_comp.min_dimensions.width, measured.width), text_comp.max_dimensions.width
            ),
            height=min(
                max(text_comp.min_dimensions.height, measured.height),
                text_comp.max_dimensions.height,
            ),
        )
        return size

    def render_text(self, context, text_component, matrix):
        """render text using cached svg paths"""
        if not hasattr(text_component, "_svg_cache") or not text_component._svg_cache:
            # measure text if not already cached (e.g., if rendered directly without measure_and_layout)
            self.measure_text(text_component)
            if not text_component._svg_cache:
                return  # cannot render if measurement failed

        self._render_text_paths(context, text_component, matrix)

        if text_component.debug:
            self.render_debug(context, text_component, matrix)
