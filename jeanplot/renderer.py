from typing import Optional, Dict, Any, Union, BinaryIO, TextIO, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
from matplotlib.path import Path
from .components import Component
from .models import Size


def generate_rounded_rect_points(
    width: float, height: float, radius: float, inset: float = 0, segments: int = 10
) -> np.ndarray:
    """
    generate points for drawing a rounded rectangle with optional inset

    Args:
        width: width of the rectangle
        height: height of the rectangle
        radius: corner radius
        inset: amount to inset the rectangle (for creating inner border)
        segments: number of segments to use for each corner arc
    """
    # apply inset to dimensions
    if inset > 0:
        width = max(0, width - 2 * inset)
        height = max(0, height - 2 * inset)

    if width <= 0 or height <= 0:
        return np.array([[0, 0]])

    # clamp radius to half the min dimension to avoid invalid shapes
    radius = min(radius, min(width, height) / 2)

    # if radius is zero or negative, return a simple rectangle
    if radius <= 0:
        return np.array(
            [
                [inset, inset],  # top-left
                [width + inset, inset],  # top-right
                [width + inset, height + inset],  # bottom-right
                [inset, height + inset],  # bottom-left
                [inset, inset],  # close the path
            ]
        )

    points = []

    # top-left corner
    for i in range(segments + 1):
        angle = np.pi + (i / segments) * (np.pi / 2)
        points.append(
            [inset + radius + radius * np.cos(angle), inset + radius + radius * np.sin(angle)]
        )

    # top-right corner
    for i in range(segments + 1):
        angle = 3 * np.pi / 2 + (i / segments) * (np.pi / 2)
        points.append(
            [
                inset + width - radius + radius * np.cos(angle),
                inset + radius + radius * np.sin(angle),
            ]
        )

    # bottom-right corner
    for i in range(segments + 1):
        angle = 0 + (i / segments) * (np.pi / 2)
        points.append(
            [
                inset + width - radius + radius * np.cos(angle),
                inset + height - radius + radius * np.sin(angle),
            ]
        )

    # bottom-left corner
    for i in range(segments + 1):
        angle = np.pi / 2 + (i / segments) * (np.pi / 2)
        points.append(
            [
                inset + radius + radius * np.cos(angle),
                inset + height - radius + radius * np.sin(angle),
            ]
        )

    # close the path
    points.append(points[0])

    return np.array(points)


def generate_dashed_points(points: np.ndarray, dash_pattern: Tuple[float, float]) -> np.ndarray:
    """generate points for a dashed line following the given points"""
    if len(points) < 2:
        return points

    dash_length, gap_length = dash_pattern
    result_points = []

    # compute the cumulative distance along the path
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        segment = points[i] - points[i - 1]
        distances[i] = distances[i - 1] + np.sqrt(np.sum(segment**2))

    total_length = distances[-1]

    # generate dashes
    dash_start = 0
    while dash_start < total_length:
        dash_end = min(dash_start + dash_length, total_length)

        # find points for dash start
        start_idx = np.searchsorted(distances, dash_start) - 1
        if start_idx < 0:
            start_idx = 0
        end_idx = np.searchsorted(distances, dash_end)

        # interpolate start point if needed
        if dash_start > distances[start_idx]:
            t = (dash_start - distances[start_idx]) / (
                distances[start_idx + 1] - distances[start_idx]
            )
            start_point = points[start_idx] + t * (points[start_idx + 1] - points[start_idx])
            result_points.append(start_point)
        else:
            result_points.append(points[start_idx])

        # add intermediate points
        for i in range(start_idx + 1, end_idx):
            result_points.append(points[i])

        # interpolate end point if needed
        if end_idx < len(points) and dash_end < distances[end_idx]:
            t = (dash_end - distances[end_idx - 1]) / (distances[end_idx] - distances[end_idx - 1])
            end_point = points[end_idx - 1] + t * (points[end_idx] - points[end_idx - 1])
            result_points.append(end_point)
        elif end_idx < len(points):
            result_points.append(points[end_idx])

        # move to next dash
        dash_start = dash_end + gap_length
        result_points.append(None)  # None creates a break in the line

    return np.array(result_points, dtype=object)


class BaseRenderer:
    """base renderer class defining a unified interface for all renderers"""

    RENDERER_NAME = "base"

    def create_context(self, width: float, height: float, **kwargs):
        """create a rendering context with the specified dimensions"""
        raise NotImplementedError("renderers must implement create_context")

    def render_component(
        self, context, component: Component, parent_matrix: Optional[np.ndarray] = None
    ):
        """render a component to the context"""
        matrix = component.compute_world_matrix(parent_matrix)
        component.render(self, context, matrix)

    def render_to_output(
        self, context, output: Optional[Union[str, BinaryIO, TextIO]] = None, **kwargs
    ):
        """render the context to the specified output"""
        raise NotImplementedError("renderers must implement render_to_output")

    def render_rectangle(self, context, bounds, style, matrix, component=None):
        """render a rectangle in the context"""
        raise NotImplementedError("renderers must implement render_rectangle")

    def render_svg(self, context, svg_element, matrix):
        """render an svg element in the context"""
        raise NotImplementedError("renderers must implement render_svg")

    def render_text(self, context, text_component, matrix):
        """render text in the context"""
        pass

    def render_debug(self, context, component, matrix):
        """render debug visuals for a component"""
        raise NotImplementedError("renderers must implement render_debug")


class MatplotlibRenderer(BaseRenderer):
    """renderer that uses matplotlib for visualization"""

    RENDERER_NAME = "matplotlib"

    def create_context(self, width=None, height=None, dpi: int = 100, ax=None, **kwargs):
        """
        create a matplotlib figure and axes context

        args:
            width: width of the figure
            height: height of the figure
            dpi: dpi of the figure
            ax: optional existing axes to use
            **kwargs: additional args for plt.subplots
        """
        if ax is not None:
            return ax

        if width is None or height is None:
            raise ValueError("width and height must be provided when creating a new figure")

        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi, **kwargs)
        ax.set_aspect("equal")
        # ax.axis("off")

        return ax

    def render_component(
        self,
        context,
        component: Component,
        parent_matrix: Optional[np.ndarray] = None,
        adjust_lims: bool = True,
    ):
        """render a component to the context, optionally adjusting limits"""
        component.measure_and_layout(self)

        if adjust_lims:
            self.adjust_limits(context, component)

        matrix = component.compute_world_matrix(parent_matrix)
        component.render(self, context, matrix)

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

    def _calculate_world_bounds(
        self, component: Component, parent_matrix: Optional[np.ndarray] = None
    ):
        """recursively calculate the world bounds of all components"""
        world_matrix = component.compute_world_matrix(parent_matrix)

        width = component._dimensions.width
        height = component._dimensions.height

        # corners of the component in local space
        corners = np.array(
            [
                [0, 0, 1],  # bottom left
                [width, 0, 1],  # bottom right
                [0, height, 1],  # top left
                [width, height, 1],  # top right
            ]
        )

        world_corners = (world_matrix @ corners.T).T

        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

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
        style: Dict[str, Any],
        matrix: np.ndarray,
        component=None,
    ):
        """render a rectangle using the double-polygon approach for consistent borders"""
        background_color = style.get("background_color", "none")
        border_color = style.get("border_color", "none")
        border_width = style.get("width", 1.0)
        corner_radius = style.get("corner_radius", 0.0)
        border_style = style.get("border_style", "solid")
        dash_sequence = style.get("dash_sequence")
        dash_offset = style.get("dash_offset", 0.0)

        # early exit if no fill or border
        if (not background_color or background_color == "none") and (
            not border_color or border_color == "none"
        ):
            return

        # handle border with double polygon approach
        if border_color and border_color != "none" and border_width > 0:
            outer_points = generate_rounded_rect_points(bounds.width, bounds.height, corner_radius)

            homogeneous_outer = np.hstack([outer_points, np.ones((len(outer_points), 1))])
            transformed_outer = (matrix @ homogeneous_outer.T).T[:, :2]

            # for dashed/dotted border style
            if border_style in ["dashed", "dotted", "custom"]:
                if border_style == "dashed":
                    dash_pattern = (4 * border_width, 2 * border_width)
                elif border_style == "dotted":
                    dash_pattern = (border_width, border_width)
                elif border_style == "custom" and dash_sequence:
                    dash_pattern = dash_sequence
                else:
                    dash_pattern = (4 * border_width, 2 * border_width)

                dashed_points = generate_dashed_points(transformed_outer, dash_pattern)

                # create line segments for dashed border
                for i in range(0, len(dashed_points) - 1):
                    if dashed_points[i] is None or dashed_points[i + 1] is None:
                        continue

                    dash_segment = np.array([dashed_points[i], dashed_points[i + 1]])

                    # calculate perpendicular vector for line thickness
                    dx = dash_segment[1][0] - dash_segment[0][0]
                    dy = dash_segment[1][1] - dash_segment[0][1]
                    length = np.sqrt(dx * dx + dy * dy)

                    if length > 0:
                        # normalize and rotate 90 degrees
                        nx = -dy / length * border_width / 2
                        ny = dx / length * border_width / 2

                        poly_points = np.array(
                            [
                                [dash_segment[0][0] + nx, dash_segment[0][1] + ny],
                                [dash_segment[1][0] + nx, dash_segment[1][1] + ny],
                                [dash_segment[1][0] - nx, dash_segment[1][1] - ny],
                                [dash_segment[0][0] - nx, dash_segment[0][1] - ny],
                            ]
                        )

                        dash_poly = plt.Polygon(
                            poly_points,
                            closed=True,
                            fill=True,
                            facecolor=border_color,
                            edgecolor="none",
                        )
                        context.add_patch(dash_poly)
            else:
                # for solid border, create inner points inset by border width
                inner_points = generate_rounded_rect_points(
                    bounds.width,
                    bounds.height,
                    max(0, corner_radius - border_width),
                    inset=border_width,
                )

                homogeneous_inner = np.hstack([inner_points, np.ones((len(inner_points), 1))])
                transformed_inner = (matrix @ homogeneous_inner.T).T[:, :2]

                # create a polygon for the full rectangle (border + fill)
                outer_poly = plt.Polygon(
                    transformed_outer,
                    closed=True,
                    fill=True,
                    facecolor=border_color,
                    edgecolor="none",
                )
                context.add_patch(outer_poly)

                # if there's a background fill, create an inner polygon to "cut out" the center
                if background_color and background_color != "none":
                    inner_poly = plt.Polygon(
                        transformed_inner,
                        closed=True,
                        fill=True,
                        facecolor=background_color,
                        edgecolor="none",
                    )
                    context.add_patch(inner_poly)
                else:  # if no background color, still need to cut out center to create a border
                    # use figure background color if available, otherwise white
                    bg_color = context.figure.get_facecolor()
                    if bg_color == (0, 0, 0, 0):  # transparent
                        bg_color = "white"

                    inner_poly = plt.Polygon(
                        transformed_inner,
                        closed=True,
                        fill=True,
                        facecolor=bg_color,
                        edgecolor="none",
                    )
                    context.add_patch(inner_poly)
        elif background_color and background_color != "none":
            points = generate_rounded_rect_points(bounds.width, bounds.height, corner_radius)

            homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
            transformed_points = (matrix @ homogeneous_points.T).T[:, :2]

            poly = plt.Polygon(
                transformed_points,
                closed=True,
                fill=True,
                facecolor=background_color,
                edgecolor="none",
            )
            context.add_patch(poly)

    def render_svg(self, context: Axes, svg_element, matrix: np.ndarray):
        """render an svg element to the matplotlib axes"""
        try:
            from svgpath2mpl import parse_path
        except ImportError:
            print("svgpath2mpl is required for SVG rendering")
            return

        paths_data = svg_element.svg_data.get("paths", [])
        viewBox = svg_element.svg_data.get("viewBox")

        if not paths_data:
            return  # nothing to render

        scale_x = svg_element._dimensions.width
        scale_y = svg_element._dimensions.height

        if viewBox:
            scale_x = scale_x / viewBox[2]
            scale_y = scale_y / viewBox[3]

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

        mpl_paths = []
        facecolors = []
        edgecolors = []
        linewidths = []

        for path_data in paths_data:
            try:
                path = parse_path(path_data["d"])

                mpl_paths.append(path)

                # determine colors, apply customization
                fill = path_data["fill"]
                if path_data.get("is_main_color"):
                    fill = svg_element.main_color
                elif path_data.get("is_secondary_color"):
                    fill = svg_element.secondary_color

                stroke = path_data["stroke"]

                facecolors.append(fill if fill != "none" else "none")
                edgecolors.append(stroke if stroke != "none" else "none")
                linewidths.append(path_data["stroke_width"])
            except Exception as e:
                print(f"Error parsing SVG path: {e}")

        if mpl_paths:
            collection = mcollections.PathCollection(
                mpl_paths,
                facecolors=facecolors,
                edgecolors=edgecolors,
                linewidths=linewidths,
                transform=mtransforms.Affine2D(matrix=combined_matrix) + context.transData,
            )
            context.add_collection(collection)

    def render_debug(self, context: Axes, component: Component, matrix: np.ndarray):
        """render debug visuals for a component, showing bounds and origin"""
        # generate points for a rectangle outline with fixed width
        outer_points = generate_rounded_rect_points(
            component._dimensions.width, component._dimensions.height, 0
        )
        inner_points = generate_rounded_rect_points(
            component._dimensions.width,
            component._dimensions.height,
            0,
            inset=0.3,  # fixed debug border width
        )

        homogeneous_outer = np.hstack([outer_points, np.ones((len(outer_points), 1))])
        transformed_outer = (matrix @ homogeneous_outer.T).T[:, :2]

        homogeneous_inner = np.hstack([inner_points, np.ones((len(inner_points), 1))])
        transformed_inner = (matrix @ homogeneous_inner.T).T[:, :2]

        # create dashed border effect - we'll just use 8 segments for simplicity
        segments = 8
        for i in range(segments):
            start_idx = int(i * len(transformed_outer) / segments)
            end_idx = int((i + 0.5) * len(transformed_outer) / segments)

            if start_idx >= len(transformed_outer) or end_idx >= len(transformed_outer):
                continue

            dash_points = transformed_outer[start_idx : end_idx + 1]

            if len(dash_points) > 1:
                # Draw this dash segment
                dash_poly = plt.Polygon(
                    dash_points,
                    closed=False,
                    fill=False,
                    edgecolor="red",
                    linewidth=0.5,
                )
                context.add_patch(dash_poly)

        # calculate origin point in local coordinates based on relative offset
        origin_x = component.offset.relative[0] * component._dimensions.width
        origin_y = component.offset.relative[1] * component._dimensions.height

        origin = matrix @ np.array([origin_x, origin_y, 1])

        # create a plus marker at the origin
        marker_size = 1  # size in data units
        horizontal_line = plt.Line2D(
            [origin[0] - marker_size, origin[0] + marker_size],
            [origin[1], origin[1]],
            color="red",
            linewidth=0.5,
        )
        vertical_line = plt.Line2D(
            [origin[0], origin[0]],
            [origin[1] - marker_size, origin[1] + marker_size],
            color="red",
            linewidth=0.5,
        )
        context.add_line(horizontal_line)
        context.add_line(vertical_line)

        # add id text above the component
        world_coords = matrix @ np.array([0, component._dimensions.height, 1])
        context.text(
            world_coords[0],
            world_coords[1] + 2,  # slight offset
            f"{component.id if component.id else ''} ({component._dimensions.width:.1f}x{component._dimensions.height:.1f})",
            color="red",
            fontsize=6,
            ha="left",
            va="bottom",
        )

    def render_to_output(self, context, output=None, **kwargs):
        """render the matplotlib figure to output"""
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

        font_props = fm.FontProperties(
            family=text_component.font_name if text_component.font_name else "sans-serif",
            weight=text_component.font_weight,
            style=text_component.font_style,
        )

        lines = text_component.text.split("\n")
        line_height = text_component.font_size * 1.2

        paths = []
        line_widths = []

        for i, line in enumerate(lines):
            if not line:
                continue

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

            paths.append({"path": path, "width": line_width, "index": i})

        max_width = max(line_widths) if line_widths else 0
        total_height = len(lines) * line_height

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
            y_start = text_component._dimensions.height
        elif text_component.vertical_align == "middle":
            y_start = text_component._dimensions.height / 2 + total_height / 2
        else:  # bottom
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

            y_position = y_start - (line_index + 1) * line_height

            position_transform = mtransforms.Affine2D().translate(x_offset, y_position)

            # combine with component matrix and data transform
            combined_transform = (
                position_transform + mtransforms.Affine2D(matrix=matrix) + context.transData
            )

            patch = mpatches.PathPatch(
                path, facecolor=text_component.color, edgecolor="none", transform=combined_transform
            )

            context.add_patch(patch)
