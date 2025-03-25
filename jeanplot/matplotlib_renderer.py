from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from .component import Component
from .models import Size
from .renderer import BaseRenderer


def linewidth_from_data_units(linewidth, axis, reference="y"):
    """convert linewidth in data units to points"""
    fig = axis.get_figure()
    if reference == "x":
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == "y":
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    length *= 72
    # scale linewidth to value range
    assert value_range > 0, "value_range must be positive"
    result = linewidth * (length / value_range)
    return result


class DataWidthPatch:
    """wrapper to track patches that use data units for line width"""

    def __init__(self, patch, data_width, axis):
        self.patch = patch
        self.data_width = data_width
        self.axis = axis
        # update initially
        self.update_linewidth()

    def update_linewidth(self):
        """update line width based on current axis state"""
        new_width = linewidth_from_data_units(self.data_width, self.axis)
        if hasattr(self.patch, "set_linewidth"):
            self.patch.set_linewidth(new_width)


class MatplotlibRenderer(BaseRenderer):
    """renderer that uses matplotlib for visualization"""

    RENDERER_NAME = "matplotlib"

    def __init__(self, debug=False):
        super().__init__()
        self.data_width_patches = []
        self.debug = debug

    def debug_print(self, message):
        """print debug information if debug is enabled"""
        if self.debug:
            print(f"[MatplotlibRenderer] {message}")

    def refresh_linewidths(self, context):
        """update all line widths for data-unit elements"""
        self.debug_print(f"Refreshing {len(self.data_width_patches)} tracked patches")
        for patch_wrapper in self.data_width_patches:
            patch_wrapper.update_linewidth()

    def track_data_width_patch(self, patch, data_width, context):
        """track a patch that uses data units for linewidth"""
        wrapper = DataWidthPatch(patch, data_width, context)
        self.data_width_patches.append(wrapper)
        self.debug_print(f"Now tracking {len(self.data_width_patches)} patches")
        return patch

    def create_context(self, width=None, height=None, dpi: int = 100, ax=None, **kwargs):
        """create a matplotlib figure and axes context"""
        if ax is not None:
            return ax

        if width is None or height is None:
            raise ValueError("width and height must be provided when creating a new figure")

        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi, **kwargs)
        ax.set_aspect("equal")

        # clear tracking when creating a new context
        self.data_width_patches = []
        self.debug_print("Created new context and cleared tracking")

        return ax

    def render_component(
        self,
        context,
        component: Component,
        parent_matrix: Optional[np.ndarray] = None,
        adjust_lims: bool = True,
    ):
        """render a component to the context, optionally adjusting limits"""
        # ensure measurement and layout are current before rendering
        component.measure_and_layout(self)

        if adjust_lims:
            self.adjust_limits(context, component)

        # run pre-render callbacks after limits are set but before rendering
        for callback in self.pre_render_callbacks:
            callback(context)

        # clear tracking before rendering to avoid duplicates
        self.data_width_patches = []

        # render the component
        matrix = component.compute_world_matrix(parent_matrix)
        component.render(self, context, matrix)

        # refresh line widths after component is rendered
        self.debug_print("Refreshing line widths after component render")
        self.refresh_linewidths(context)

        # run post-render callbacks
        for callback in self.post_render_callbacks:
            callback(context)

    def adjust_limits(self, context: Axes, root_component: Component, padding: float = 0.05):
        """adjust plot limits to fit all components with padding"""
        # calculate the world bounds of all components
        min_x, min_y, max_x, max_y = self._calculate_world_bounds(root_component)

        # add padding
        width = max_x - min_x
        height = max_y - min_y
        pad_x = width * padding if width > 0 else 1.0
        pad_y = height * padding if height > 0 else 1.0

        # set the plot limits
        context.set_xlim(min_x - pad_x, max_x + pad_x)
        context.set_ylim(min_y - pad_y, max_y + pad_y)

        self.debug_print(
            f"Set limits: x=[{min_x-pad_x:.1f}, {max_x+pad_x:.1f}], y=[{min_y-pad_y:.1f}, {max_y+pad_y:.1f}]"
        )

    def _calculate_world_bounds(
        self, component: Component, parent_matrix: Optional[np.ndarray] = None
    ):
        """recursively calculate the world bounds of all components"""
        # compute the world matrix for this component
        world_matrix = component.compute_world_matrix(parent_matrix)

        # get the component's dimensions
        width = component._dimensions.width
        height = component._dimensions.height

        # define the corners of the component in local space
        corners = np.array(
            [
                [0, 0, 1],  # bottom left
                [width, 0, 1],  # bottom right
                [0, height, 1],  # top left
                [width, height, 1],  # top right
            ]
        )

        # transform the corners to world space
        world_corners = (world_matrix @ corners.T).T

        # initialize min/max coordinates
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        # update min/max coordinates based on corners
        for corner in world_corners:
            min_x = min(min_x, corner[0])
            min_y = min(min_y, corner[1])
            max_x = max(max_x, corner[0])
            max_y = max(max_y, corner[1])

        # if this is a container, recursively process its children
        if hasattr(component, "children") and component.children:
            for child in component.children:
                child_min_x, child_min_y, child_max_x, child_max_y = self._calculate_world_bounds(
                    child, world_matrix
                )
                min_x = min(min_x, child_min_x)
                min_y = min(min_y, child_min_y)
                max_x = max(max_x, child_max_x)
                max_y = max(max_y, child_max_y)

        return min_x, min_y, max_x, max_y

    def render_rectangle(
        self,
        context: Axes,
        bounds: Size,
        style: dict[str, Any],
        matrix: np.ndarray,
        component=None,
    ):
        """render a rectangle to the matplotlib axes"""
        facecolor = style.get("background_color", "none")
        edgecolor = style.get("border_color", "none")
        corner_radius = style.get("corner_radius", 0.0)

        # map border style to matplotlib line style
        border_style = style.get("border_style", "solid")
        linestyle_map = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "custom": "-",  # we'll handle custom dash patterns separately
        }
        linestyle = linestyle_map.get(border_style, "-")

        # get the line width and mode
        width_value = style.get("width", 1.0)
        width_mode = style.get("border_width_mode", "point")

        comp_id = component.id if component and component.id else "unknown"

        if width_mode == "data":
            self.debug_print(
                f"Creating data-width rectangle for {comp_id} with width={width_value}"
            )

            # create with dummy linewidth - we'll update it later
            patch = mpatches.FancyBboxPatch(
                (0, 0),
                bounds.width,
                bounds.height,
                boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=corner_radius),
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.0,  # temporary value
                linestyle=linestyle,
                transform=mtransforms.Affine2D(matrix=matrix) + context.transData,
            )

            # track for updating
            self.track_data_width_patch(patch, width_value, context)
        else:
            # regular point-based line width
            self.debug_print(
                f"Creating point-width rectangle for {comp_id} with width={width_value}"
            )

            patch = mpatches.FancyBboxPatch(
                (0, 0),
                bounds.width,
                bounds.height,
                boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=corner_radius),
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=width_value,
                linestyle=linestyle,
                transform=mtransforms.Affine2D(matrix=matrix) + context.transData,
            )

        # apply custom dash pattern if needed
        if border_style == "custom" and style.get("dash_sequence") and hasattr(patch, "set_dashes"):
            dash_sequence = style.get("dash_sequence")
            dash_offset = style.get("dash_offset", 0)
            patch.set_dashes(dash_offset, dash_sequence)

        context.add_patch(patch)

    def render_svg(self, context: Axes, svg_element, matrix: np.ndarray):
        """render an svg element to the matplotlib axes"""
        try:
            from svgpath2mpl import parse_path
        except ImportError:
            print("svgpath2mpl is required for SVG rendering")
            return

        paths_data = svg_element.svg_content.paths
        viewBox = svg_element.svg_content.viewBox

        if not paths_data:
            return  # nothing to render

        scale_x = svg_element._dimensions.width
        scale_y = svg_element._dimensions.height

        if viewBox:
            scale_x = scale_x / viewBox[2]
            scale_y = scale_y / viewBox[3]

            # create a combined transform matrix
            svg_matrix = np.array(
                [
                    [scale_x, 0, -viewBox[0] * scale_x],
                    [0, scale_y, -viewBox[1] * scale_y],
                    [0, 0, 1],
                ]
            )
        else:
            # if no viewBox, just scale to bounds
            svg_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

        # combine with component transform
        combined_matrix = matrix @ svg_matrix

        svg_width_mode = svg_element.get_renderer_options(self.RENDERER_NAME).get(
            "line_width_mode", "point"
        )

        self.debug_print(
            f"Rendering SVG '{svg_element.id}' with {len(paths_data)} paths in {svg_width_mode} mode"
        )

        # render each path individually to avoid array dimension issues
        for i, path_data in enumerate(paths_data):
            try:
                path = parse_path(path_data.d)

                # determine colors
                fill = path_data.fill
                if path_data.is_main_color:
                    fill = svg_element.main_color
                elif path_data.is_secondary_color:
                    fill = svg_element.secondary_color

                stroke = path_data.stroke

                # get the raw line width
                raw_linewidth = path_data.stroke_width

                if svg_width_mode == "data":
                    self.debug_print(f"  Path {i}: data width = {raw_linewidth}")

                    # create patch with dummy width
                    patch = mpatches.PathPatch(
                        path,
                        facecolor=fill if fill != "none" else "none",
                        edgecolor=stroke if stroke != "none" else "none",
                        linewidth=1.0,  # temporary
                        transform=mtransforms.Affine2D(matrix=combined_matrix) + context.transData,
                    )

                    # track for updating
                    self.track_data_width_patch(patch, raw_linewidth, context)
                else:
                    # point mode - use raw value
                    self.debug_print(f"  Path {i}: point width = {raw_linewidth}")

                    patch = mpatches.PathPatch(
                        path,
                        facecolor=fill if fill != "none" else "none",
                        edgecolor=stroke if stroke != "none" else "none",
                        linewidth=raw_linewidth,
                        transform=mtransforms.Affine2D(matrix=combined_matrix) + context.transData,
                    )

                context.add_patch(patch)

            except Exception as e:
                print(f"Error rendering SVG path: {e}")

    def render_debug(self, context: Axes, component: Component, matrix: np.ndarray):
        """render debug visuals for a component, showing origin and bounds"""
        # debug line width in data units (0.5)
        debug_width = 0.5

        # draw bounds rectangle with initial thin width
        rect = mpatches.Rectangle(
            (0, 0),
            component._dimensions.width,
            component._dimensions.height,
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.0,  # temporary
            transform=mtransforms.Affine2D(matrix=matrix) + context.transData,
        )

        # track for updating
        self.track_data_width_patch(rect, debug_width, context)
        context.add_patch(rect)

        # calculate origin point based on relative offset
        origin_x = component.offset.relative[0] * component._dimensions.width
        origin_y = component.offset.relative[1] * component._dimensions.height

        # draw origin marker
        component_id = component.id or "unnamed"
        origin_marker = plt.Line2D(
            [origin_x],
            [origin_y],
            marker="+",
            color="red",
            markersize=6,
            linestyle="",
            transform=mtransforms.Affine2D(matrix=matrix) + context.transData,
        )
        context.add_line(origin_marker)

        # add id text above the component
        world_coords = matrix @ np.array([0, component._dimensions.height, 1])
        context.text(
            world_coords[0],
            world_coords[1] + 2,  # slight offset
            f"{component_id} ({component._dimensions.width:.1f}x{component._dimensions.height:.1f})",
            color="red",
            fontsize=6,
            ha="left",
            va="bottom",
        )

    def render_to_output(self, context, output=None, **kwargs):
        """render the matplotlib figure to output"""
        # refresh line widths before output
        self.debug_print("Refreshing line widths before output")
        self.refresh_linewidths(context)

        if output is None:
            context.figure.show()
        else:
            context.figure.savefig(output, **kwargs)
        return context.figure

    def measure_text(self, text_component) -> Size:
        """measure text dimensions and create SVG paths for rendering"""
        from matplotlib.textpath import TextPath
        import matplotlib.font_manager as fm
        import numpy as np

        # create font properties object
        font_props = fm.FontProperties(
            family=text_component.font_name if text_component.font_name else "sans-serif",
            weight=text_component.font_weight,
            style=text_component.font_style,
        )

        # split text into lines
        lines = text_component.text.split("\n")

        # estimated line height based on font size
        line_height = text_component.font_size * 1.2

        # we'll store all rendered paths and their positions
        paths = []
        line_widths = []

        # process each line
        for i, line in enumerate(lines):
            if not line:  # skip empty lines
                continue

            # create path for the current line
            path = TextPath((0, 0), line, size=text_component.font_size, prop=font_props)

            # get path bounds
            path_vertices = path.vertices
            if len(path_vertices) > 0:
                x_min = np.min(path_vertices[:, 0])
                x_max = np.max(path_vertices[:, 0])
                line_width = x_max - x_min
            else:
                line_width = 0

            line_widths.append(line_width)

            # store path for later use
            paths.append({"path": path, "width": line_width, "index": i})

        # calculate max width and total height
        max_width = max(line_widths) if line_widths else 0
        total_height = len(lines) * line_height

        # store the paths and measurements for rendering
        text_component._text_cache = {
            "paths": paths,
            "line_widths": line_widths,
            "max_width": max_width,
            "total_height": total_height,
            "line_height": line_height,
            "line_count": len(lines),
        }

        return Size(width=max_width, height=total_height)

    def render_text(self, context, text_component, matrix):
        """render pre-created text paths with proper alignment"""
        import matplotlib.transforms as mtransforms
        import matplotlib.patches as mpatches

        # measure if not already measured
        if not hasattr(text_component, "_text_cache") or not text_component._text_cache:
            self.measure_text(text_component)

        cache = text_component._text_cache
        paths = cache["paths"]
        line_widths = cache["line_widths"]
        max_width = cache["max_width"]
        total_height = cache["total_height"]
        line_height = cache["line_height"]
        line_count = cache["line_count"]

        # calculate vertical starting position based on alignment
        if text_component.vertical_align == "top":
            # align to top - first line starting at top of bounding box
            y_start = text_component._dimensions.height
        elif text_component.vertical_align == "middle":
            # center vertically - center of text block at center of bounding box
            y_start = text_component._dimensions.height / 2 + total_height / 2
        else:  # bottom
            # align to bottom - last line ending at bottom of bounding box
            y_start = total_height

        # render each line
        for path_info in paths:
            path = path_info["path"]
            line_width = path_info["width"]
            line_index = path_info["index"]

            # calculate horizontal offset for this specific line
            if text_component.align == "left":
                x_offset = 0  # aligned to left
            elif text_component.align == "center":
                x_offset = (text_component._dimensions.width - line_width) / 2  # centered
            else:  # right
                x_offset = text_component._dimensions.width - line_width  # aligned to right

            # vertical position - each line is positioned below the previous
            y_position = y_start - (line_index + 1) * line_height

            # create position transform
            position_transform = mtransforms.Affine2D().translate(x_offset, y_position)

            # combine with component matrix and data transform
            combined_transform = (
                position_transform + mtransforms.Affine2D(matrix=matrix) + context.transData
            )

            # create and add path patch
            patch = mpatches.PathPatch(
                path, facecolor=text_component.color, edgecolor="none", transform=combined_transform
            )

            context.add_patch(patch)
