from typing import Optional, Any, List, Dict, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from pydantic import Field, BaseModel

from .component import Component
from .models import Size, VisualStyle
from .renderer import BaseRenderer
from .svg import SVGElement, SVGContent


class SVGTextContent(SVGContent):
    """svg content for text"""

    text_paths: List[Dict[str, Any]] = Field(default_factory=list)
    measured_width: float = 0
    measured_height: float = 0


def linewidth_from_data_units(linewidth, axis, reference="y"):
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

        if value_range_x <= 0 or value_range_y <= 0:
            return 1.0

        scale_x = length_x / value_range_x
        scale_y = length_y / value_range_y
        return linewidth * ((scale_x + scale_y) / 2) * 72

    length *= 72
    return linewidth * (length / value_range) if value_range > 0 else 1.0


class MatplotlibRenderer(BaseRenderer):
    RENDERER_NAME = "matplotlib"

    def __init__(self, debug=False):
        super().__init__()
        self.data_width_patches = []
        self.debug = debug

    def debug_print(self, msg):
        if self.debug:
            print(f"[MatplotlibRenderer] {msg}")

    def refresh_linewidths(self, ctx):
        """update linewidths for data-unit elements"""
        for patch, width in self.data_width_patches:
            if hasattr(patch, "set_linewidth"):
                new_width = linewidth_from_data_units(width, ctx, "avg")
                patch.set_linewidth(new_width)

    def track_patch(self, patch, width, ctx):
        """track a patch with data-unit width"""
        self.data_width_patches.append((patch, width))
        return patch

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

        # flip matrix for the whole render pass, since matplotlib uses "inverted" y-axis
        flip = np.array(
            [
                [1, 0, 0],
                [0, -1, component._dimensions.height],
                [0, 0, 1],
            ]
        )
        world = component.compute_world_matrix(parent_matrix)
        final = flip @ world

        component.render(self, context, final)
        self.refresh_linewidths(context)

        for cb in self.post_render_callbacks:
            cb(context)

    def adjust_limits(self, ctx, root, padding=0.05):
        """adjust plot limits to fit components"""
        min_x, min_y, max_x, max_y = self._calculate_world_bounds(root)

        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)

        pad_x, pad_y = width * padding, height * padding

        ctx.set_xlim(min_x - pad_x, max_x + pad_x)
        ctx.set_ylim(min_y - pad_y, max_y + pad_y)

    def _calculate_world_bounds(self, comp, parent_matrix=None):
        """calculate world bounds recursively"""
        world = comp.compute_world_matrix(parent_matrix)
        w, h = comp._dimensions.width, comp._dimensions.height

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
        else:
            min_x = min_y = max_x = max_y = 0

        if hasattr(comp, "children") and comp.children:
            for child in comp.children:
                cx, cy, cX, cY = self._calculate_world_bounds(child, world)
                min_x, min_y = min(min_x, cx), min(min_y, cy)
                max_x, max_y = max(max_x, cX), max(max_y, cY)

        return min_x, min_y, max_x, max_y

    def render_rectangle(self, ctx, bounds, style, matrix, component=None):
        """render a rectangle"""
        facecolor = style.background_color or "none"
        edgecolor = style.border_color or "none"
        width_val = style.border_width
        width_mode = style.border_width_mode

        linestyle_map = {"solid": "-", "dashed": "--", "dotted": ":", "custom": "-"}
        linestyle = linestyle_map.get(style.border_style, "-")

        transform = mtransforms.Affine2D(matrix=matrix) + ctx.transData

        patch = mpatches.FancyBboxPatch(
            (0, 0),
            bounds.width,
            bounds.height,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=style.corner_radius),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0 if width_mode == "data" else width_val,
            linestyle=linestyle,
            transform=transform,
        )

        if width_mode == "data":
            self.track_patch(patch, width_val, ctx)

        if style.border_style == "custom" and style.dash_sequence and hasattr(patch, "set_dashes"):
            dash = style.dash_sequence
            if width_mode == "data":
                avg_lw = linewidth_from_data_units(1.0, ctx, "avg")
                dash = tuple(d * avg_lw for d in dash) if avg_lw > 0 else dash
            patch.set_dashes(dash)

        ctx.add_patch(patch)

    def render_svg(self, context, svg_elem, matrix):
        """render an svg element"""
        # handle text paths
        if hasattr(svg_elem, "svg_content") and isinstance(svg_elem.svg_content, SVGTextContent):
            self._render_text_paths(context, svg_elem, matrix)
            return

        try:
            from svgpath2mpl import parse_path
        except ImportError:
            print("svgpath2mpl is required for SVG rendering")
            return

        paths = svg_elem.svg_content.paths
        if not paths:
            return

        svg_dims = svg_elem._dimensions
        viewBox = svg_elem.svg_content.viewBox

        # calculate scaling matrix
        if viewBox:
            vb_w, vb_h = max(viewBox[2], 1.0), max(viewBox[3], 1.0)
            scale_x, scale_y = svg_dims.width / vb_w, svg_dims.height / vb_h
            svg_matrix = np.array(
                [
                    [scale_x, 0, -viewBox[0] * scale_x],
                    [0, scale_y, -viewBox[1] * scale_y],
                    [0, 0, 1],
                ]
            )
        else:
            content_w = max(svg_elem.svg_content.width, 1.0)
            content_h = max(svg_elem.svg_content.height, 1.0)
            scale_x, scale_y = svg_dims.width / content_w, svg_dims.height / content_h
            svg_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

        final_matrix = matrix @ svg_matrix
        transform = mtransforms.Affine2D(matrix=final_matrix) + context.transData

        width_mode = svg_elem.get_renderer_options(self.RENDERER_NAME).get("line_width_mode", "")

        for path_data in paths:
            try:
                path = parse_path(path_data.d)

                fill = path_data.fill
                if path_data.is_main_color:
                    fill = svg_elem.main_color
                elif path_data.is_secondary_color:
                    fill = svg_elem.secondary_color

                stroke = path_data.stroke if path_data.stroke != "none" else "none"
                linewidth = path_data.stroke_width
                use_data_width = width_mode == "data" and stroke != "none" and linewidth > 0

                patch = mpatches.PathPatch(
                    path,
                    facecolor=fill,
                    edgecolor=stroke,
                    linewidth=1.0 if use_data_width else linewidth,
                    transform=transform,
                )

                if use_data_width:
                    self.track_patch(patch, linewidth, context)

                context.add_patch(patch)
            except Exception as e:
                print(f"Error rendering SVG path: {e}")

    def render_debug(self, context, comp, matrix):
        """renders a red box around the component + origin marker and name"""
        debug_width = 0.5
        w, h = comp._dimensions.width, comp._dimensions.height
        comp_id = comp.id or "unnamed"

        transform = mtransforms.Affine2D(matrix=matrix) + context.transData

        # draw rectangle
        rect = mpatches.Rectangle(
            (0, 0),
            w,
            h,
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.0,
            transform=transform,
        )
        self.track_patch(rect, debug_width, context)
        context.add_patch(rect)

        # draw origin marker
        marker = plt.Line2D(
            [0], [0], marker="+", color="red", markersize=6, linestyle="", transform=transform
        )
        context.add_line(marker)

        # add label
        world_coords = matrix @ np.array([0, 0, 1])
        context.text(
            world_coords[0],
            world_coords[1] + 2,
            f"{comp_id} ({w:.1f}x{h:.1f})",
            color="red",
            fontsize=6,
            ha="left",
            va="bottom",
        )

    def _render_text_paths(self, ctx, text_elem, matrix):
        """render cached text paths"""
        if hasattr(text_elem, "_svg_cache"):
            svg_content = text_elem._svg_cache
        elif hasattr(text_elem, "svg_content"):
            svg_content = text_elem.svg_content
        else:
            return

        if not svg_content or not hasattr(svg_content, "text_paths") or not svg_content.text_paths:
            return

        text_h = text_elem._dimensions.height
        unflip_y = np.array([[1, 0, 0], [0, -1, text_h], [0, 0, 1]])
        final_matrix = matrix @ unflip_y
        transform = mtransforms.Affine2D(matrix=final_matrix) + ctx.transData

        for path_info in svg_content.text_paths:
            patch = mpatches.PathPatch(
                path_info["path"],
                facecolor=text_elem.main_color
                if hasattr(text_elem, "main_color")
                else text_elem.color,
                edgecolor="none",
                transform=transform,
            )
            ctx.add_patch(patch)

    def render_to_output(self, context, output=None, **kwargs):
        self.refresh_linewidths(context)
        if output:
            context.figure.savefig(output, **kwargs)
        return context.figure

    def measure_text(self, text_comp) -> Size:
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
        paths_info = []

        for line in lines:
            if not line.strip():
                continue

            path = TextPath((0, 0), line, size=text_comp.font_size, prop=font_props)
            try:
                bbox = path.get_extents()
                path_info = {
                    "text": line,
                    "path": path,
                    "width": bbox.width,
                    "height": bbox.height,
                    "x0": bbox.x0,
                    "y0": bbox.y0,
                    "x1": bbox.x1,
                    "y1": bbox.y1,
                }
                max_width = max(max_width, bbox.width)
            except Exception:
                # fallback estimation
                est_w = len(line) * text_comp.font_size * 0.6
                est_h = text_comp.font_size
                path_info = {
                    "text": line,
                    "path": path,
                    "width": est_w,
                    "height": est_h,
                    "x0": 0,
                    "y0": -est_h * 0.2,
                    "x1": est_w,
                    "y1": est_h * 0.8,
                }
                max_width = max(max_width, est_w)

            paths_info.append(path_info)

        line_height = text_comp.font_size * 1.2
        if paths_info:
            line_height = sum(p["height"] for p in paths_info) / len(paths_info) * 1.2

        # position paths from top to bottom
        y_pos = 0
        final_paths = []

        for i, line in enumerate(lines):
            if not line.strip():
                y_pos += line_height
                continue

            path_info = next((p for p in paths_info if p["text"] == line), None)
            if not path_info:
                y_pos += line_height
                continue

            new_path = TextPath((0, y_pos), line, size=text_comp.font_size, prop=font_props)
            final_paths.append(
                {
                    "path": new_path,
                    "width": path_info["width"],
                    "height": path_info["height"],
                    "y_pos": y_pos,
                    "line_index": i,
                }
            )
            y_pos += line_height

        total_height = y_pos

        # store paths in svg cache
        svg_content = SVGTextContent(
            width=max_width,
            height=total_height,
            viewBox=(0, 0, max_width, total_height),
            paths=[],
            text_paths=final_paths,
            measured_width=max_width,
            measured_height=total_height,
        )
        text_comp._svg_cache = svg_content

        measured = Size(width=max_width, height=total_height)
        size = Size(
            width=max(text_comp.min_dimensions.width, measured.width),
            height=max(text_comp.min_dimensions.height, measured.height),
        )
        return Size(
            width=min(size.width, text_comp.max_dimensions.width),
            height=min(size.height, text_comp.max_dimensions.height),
        )

    def render_text(self, context, text_component, matrix):
        """render text using cached svg paths"""
        if not hasattr(text_component, "_svg_cache") or not text_component._svg_cache:
            self.measure_text(text_component)
            if not text_component._svg_cache:
                return

        self._render_text_paths(context, text_component, matrix)

        if text_component.debug:
            self.render_debug(context, text_component, matrix)
