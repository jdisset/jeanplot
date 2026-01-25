"""SVG rendering backend for jeanplot."""

from __future__ import annotations

from typing import BinaryIO, TextIO
import numpy as np
from lxml import etree

from jeanplot.core.component import Component
from jeanplot.core.models import Size, BoxStyle, TextMetrics
from jeanplot.core.renderer.base import BaseRenderer
from jeanplot.core.svg import SVGElement, SVGPathData
from jeanplot.core.connector import Connection
from jeanplot.core.text import Text
from jeanplot.core.debug import DebugMixin, get_logger

logger = get_logger(__name__)
EPSILON = 1e-9
SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {None: SVG_NS}


def _matrix_to_svg_transform(matrix: np.ndarray) -> str:
    return f"matrix({matrix[0, 0]:.6f},{matrix[1, 0]:.6f},{matrix[0, 1]:.6f},{matrix[1, 1]:.6f},{matrix[0, 2]:.6f},{matrix[1, 2]:.6f})"


def _get_mpl_linestyle_svg(path_data: SVGPathData) -> str | None:
    if path_data.dash_array and path_data.line_style == "custom":
        return ",".join(str(d) for d in path_data.dash_array)
    style_map = {"dashed": "5,5", "dotted": "2,2"}
    return style_map.get(path_data.line_style)


def _get_matrix_avg_scale(matrix: np.ndarray) -> float:
    scale_x = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    scale_y = np.sqrt(matrix[0, 1] ** 2 + matrix[1, 1] ** 2)
    return (scale_x + scale_y) / 2.0 if (scale_x + scale_y) / 2.0 > EPSILON else 0.0


class SVGRenderer(DebugMixin, BaseRenderer):
    RENDERER_NAME = "svg"

    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug
        self._root: etree._Element | None = None
        self._defs: etree._Element | None = None
        self._current_group: etree._Element | None = None
        self._width: float = 0
        self._height: float = 0

    def create_context(self, width: float = 800, height: float = 600, **kwargs) -> etree._Element:
        self._width = width
        self._height = height
        self._root = etree.Element("svg", nsmap=NSMAP)
        self._root.set("width", str(width))
        self._root.set("height", str(height))
        self._root.set("viewBox", f"0 0 {width} {height}")
        self._defs = etree.SubElement(self._root, "defs")
        self._current_group = self._root
        return self._root

    def render_component(
        self, context: etree._Element, component: Component, adjust_lims: bool = True
    ):
        if self._root is None:
            self._root = context

        component.measure_and_layout(self)

        if adjust_lims and component.parent is None:
            bounds = self._get_recursive_world_bounds(component)
            if bounds:
                min_x, min_y, max_x, max_y = bounds
                width = max(max_x - min_x, 1.0)
                height = max(max_y - min_y, 1.0)
                pad = max(width, height) * 0.1
                vb_x, vb_y = min_x - pad, min_y - pad
                vb_w, vb_h = width + 2 * pad, height + 2 * pad
                self._root.set("viewBox", f"{vb_x:.2f} {vb_y:.2f} {vb_w:.2f} {vb_h:.2f}")
                self._root.set("width", str(int(vb_w)))
                self._root.set("height", str(int(vb_h)))

        for cb in self.pre_render_callbacks:
            cb(context)

        root_matrix = component.compute_world_matrix()
        y_flip = np.array([[1, 0, 0], [0, -1, float(self._height)], [0, 0, 1]])
        component.render(self, context, y_flip @ root_matrix)

        for cb in self.post_render_callbacks:
            cb(context)

    def _get_recursive_world_bounds(
        self, component: Component, current_bounds=None
    ) -> tuple[float, float, float, float] | None:
        if not component or not component.show:
            return current_bounds
        overall = list(current_bounds) if current_bounds else [np.inf, np.inf, -np.inf, -np.inf]

        if not isinstance(component, Connection):
            comp_b = component.get_world_bounds()
            if comp_b:
                overall = [
                    min(overall[0], comp_b[0]),
                    min(overall[1], comp_b[1]),
                    max(overall[2], comp_b[2]),
                    max(overall[3], comp_b[3]),
                ]

        children_to_check = getattr(component, "children", []) + getattr(
            component, "anchor_points", []
        )
        for child in children_to_check:
            if not child or not child.show:
                continue
            child_bounds = self._get_recursive_world_bounds(child, None)
            if child_bounds:
                overall = [
                    min(overall[0], child_bounds[0]),
                    min(overall[1], child_bounds[1]),
                    max(overall[2], child_bounds[2]),
                    max(overall[3], child_bounds[3]),
                ]

        return tuple(overall) if overall[0] != np.inf else None

    def render_to_string(self, component: Component) -> str:
        root = self.create_context(800, 600)
        self.render_component(root, component)
        return etree.tostring(self._root, encoding="unicode", pretty_print=True)

    def render_to_output(
        self, context: etree._Element, output: str | BinaryIO | TextIO | None = None, **kwargs
    ):
        if self._root is None:
            raise ValueError("no svg root created")

        svg_str = etree.tostring(self._root, encoding="unicode", pretty_print=True)
        if output is None:
            return svg_str

        if isinstance(output, str):
            with open(output, "w") as f:
                f.write(svg_str)
        else:
            if hasattr(output, "mode") and "b" in output.mode:
                output.write(svg_str.encode("utf-8"))
            else:
                output.write(svg_str)
        return svg_str

    def render_rectangle(
        self,
        context: etree._Element,
        bounds: Size,
        style: BoxStyle,
        matrix: np.ndarray,
        component: Component | None = None,
    ):
        w, h = bounds.width, bounds.height
        if w <= 0 or h <= 0:
            return

        parent = self._current_group if self._current_group is not None else context
        g = etree.SubElement(parent, "g")
        if component and component.id:
            g.set("id", component.id)
        g.set("transform", _matrix_to_svg_transform(matrix))

        if style.shadow and style.shadow.blur_radius > 0:
            shadow = style.shadow
            shadow_g = etree.SubElement(
                g, "g", opacity=str(shadow.color[-2:] if len(shadow.color or "") == 9 else "0.5")
            )
            spread = shadow.spread + shadow.blur_radius
            sw, sh = w + 2 * spread, h + 2 * spread
            sx, sy = -spread + shadow.offset_x, -spread + shadow.offset_y
            rect = etree.SubElement(shadow_g, "rect")
            rect.set("x", f"{sx:.2f}")
            rect.set("y", f"{sy:.2f}")
            rect.set("width", f"{sw:.2f}")
            rect.set("height", f"{sh:.2f}")
            rect.set("rx", f"{style.corner_radius + spread:.2f}")
            rect.set("fill", shadow.color[:7] if shadow.color else "#000000")
            rect.set("filter", f"url(#blur-{id(shadow)})")

            if self._defs is not None:
                filt = etree.SubElement(self._defs, "filter")
                filt.set("id", f"blur-{id(shadow)}")
                blur = etree.SubElement(filt, "feGaussianBlur")
                blur.set("stdDeviation", str(shadow.blur_radius / 2))

        rect = etree.SubElement(g, "rect")
        rect.set("x", "0")
        rect.set("y", "0")
        rect.set("width", f"{w:.2f}")
        rect.set("height", f"{h:.2f}")
        if style.corner_radius > 0:
            rect.set("rx", f"{style.corner_radius:.2f}")

        if style.background_color and style.background_color != "none":
            rect.set("fill", style.background_color)
        else:
            rect.set("fill", "none")

        if style.border_color and style.border_color != "none" and style.border_width > 0:
            rect.set("stroke", style.border_color)
            rect.set("stroke-width", f"{style.border_width:.2f}")
            if style.dash_sequence:
                rect.set("stroke-dasharray", ",".join(str(d) for d in style.dash_sequence))
        else:
            rect.set("stroke", "none")

    def render_path(
        self,
        context: etree._Element,
        path_data: SVGPathData,
        matrix: np.ndarray,
        line_width_mode: str = "data",
        color_remap: dict[str, str | None] | None = None,
        component_id: str | None = None,
        opacity: float | None = None,
    ):
        parent = self._current_group if self._current_group is not None else context
        path = etree.SubElement(parent, "path")
        if component_id:
            path.set("id", f"{component_id}_path")

        path.set("d", path_data.d)
        path.set("transform", _matrix_to_svg_transform(matrix))

        if opacity is not None and opacity < 1.0:
            path.set("opacity", f"{opacity:.2f}")

        remap = color_remap or {}
        fill = remap.get(path_data.fill, path_data.fill) if path_data.fill else None
        stroke = remap.get(path_data.stroke, path_data.stroke) if path_data.stroke else None

        path.set("fill", fill if fill else "none")

        if stroke and stroke != "none" and path_data.stroke_width > 0:
            path.set("stroke", stroke)
            sw = path_data.stroke_width
            if line_width_mode == "data":
                sw *= _get_matrix_avg_scale(matrix)
            path.set("stroke-width", f"{sw:.2f}")

            dash = _get_mpl_linestyle_svg(path_data)
            if dash:
                path.set("stroke-dasharray", dash)
        else:
            path.set("stroke", "none")

    def render_svg(self, context: etree._Element, svg_element: SVGElement, matrix: np.ndarray):
        if not svg_element._parsed_svg_content or not svg_element._parsed_svg_content.paths:
            if svg_element.debug:
                self.render_debug(context, svg_element, matrix)
            return

        svg_data = svg_element._parsed_svg_content
        vb_x, vb_y, vb_w, vb_h = svg_data.viewBox or (0, 0, svg_data.width, svg_data.height)
        vb_w, vb_h = max(vb_w, EPSILON), max(vb_h, EPSILON)
        comp_w, comp_h = svg_element._dimensions.width, svg_element._dimensions.height
        if comp_w <= EPSILON or comp_h <= EPSILON:
            return

        scale_x, scale_y = comp_w / vb_w, comp_h / vb_h
        svg_internal_matrix = np.array(
            [[scale_x, 0, -scale_x * vb_x], [0, scale_y, -scale_y * vb_y], [0, 0, 1]]
        )
        final_matrix = matrix @ svg_internal_matrix

        parent = self._current_group if self._current_group is not None else context
        g = etree.SubElement(parent, "g")
        if svg_element.id:
            g.set("id", svg_element.id)

        old_group = self._current_group
        self._current_group = g

        for path_data in svg_data.paths:
            self.render_path(
                context,
                path_data,
                final_matrix,
                svg_element.line_width_mode,
                svg_element.color_remap,
                component_id=svg_element.id,
            )

        self._current_group = old_group

        if svg_element.debug:
            self.render_debug(context, svg_element, matrix)

    def render_text(self, context: etree._Element, text_component: Text, matrix: np.ndarray):
        if not text_component.text or not text_component.show:
            return

        parent = self._current_group if self._current_group is not None else context
        g = etree.SubElement(parent, "g")
        if text_component.id:
            g.set("id", text_component.id)
        g.set("transform", _matrix_to_svg_transform(matrix))

        text_elem = etree.SubElement(g, "text")
        text_elem.set("x", "0")
        text_elem.set("y", str(text_component.font_size))
        text_elem.set("font-size", str(text_component.font_size))
        text_elem.set("fill", text_component.color or "#000000")

        if text_component.font_name:
            text_elem.set("font-family", text_component.font_name)
        if text_component.font_weight:
            text_elem.set("font-weight", text_component.font_weight)
        if text_component.font_style:
            text_elem.set("font-style", text_component.font_style)

        anchor_map = {"left": "start", "center": "middle", "right": "end"}
        text_elem.set("text-anchor", anchor_map.get(text_component.align, "start"))

        text_elem.text = text_component.text

        if text_component.debug:
            self.render_debug(context, text_component, matrix)

    def render_connection_curve(
        self,
        context: etree._Element,
        connection: Connection,
        local_start: tuple[float, float],
        local_end: tuple[float, float],
        local_control_points: list[tuple[float, float]],
        path_string: str,
        matrix: np.ndarray,
    ):
        parent = self._current_group if self._current_group is not None else context
        g = etree.SubElement(parent, "g")
        if connection.id:
            g.set("id", connection.id)
        g.set("transform", _matrix_to_svg_transform(matrix))

        path = etree.SubElement(g, "path")
        path.set("d", path_string)
        path.set("fill", "none")

        style = connection.safe_style
        if style.border_color and style.border_color != "none":
            path.set("stroke", style.border_color)
            path.set("stroke-width", f"{style.border_width:.2f}")
            if style.dash_sequence:
                path.set("stroke-dasharray", ",".join(str(d) for d in style.dash_sequence))
        else:
            path.set("stroke", "#000000")
            path.set("stroke-width", "1")

    def render_debug(self, context: etree._Element, component: Component, matrix: np.ndarray):
        if (
            not hasattr(component, "_dimensions")
            or component._dimensions.width <= 0
            or component._dimensions.height <= 0
        ):
            return

        w, h = component._dimensions.width, component._dimensions.height
        parent = self._current_group if self._current_group is not None else context
        g = etree.SubElement(parent, "g")
        g.set("class", "debug")
        g.set("transform", _matrix_to_svg_transform(matrix))

        rect = etree.SubElement(g, "rect")
        rect.set("x", "0")
        rect.set("y", "0")
        rect.set("width", f"{w:.2f}")
        rect.set("height", f"{h:.2f}")
        rect.set("fill", "none")
        rect.set("stroke", "red")
        rect.set("stroke-width", "0.5")
        rect.set("stroke-dasharray", "2,2")

        line1 = etree.SubElement(g, "line")
        line1.set("x1", "-3")
        line1.set("y1", "0")
        line1.set("x2", "3")
        line1.set("y2", "0")
        line1.set("stroke", "red")
        line1.set("stroke-width", "0.5")

        line2 = etree.SubElement(g, "line")
        line2.set("x1", "0")
        line2.set("y1", "-3")
        line2.set("x2", "0")
        line2.set("y2", "3")
        line2.set("stroke", "red")
        line2.set("stroke-width", "0.5")

    def measure_text(self, text_component: Text) -> Size:
        text = text_component.text or ""
        font_size = text_component.font_size
        lines = text.split("\n")
        max_width = max(len(line) for line in lines) * font_size * 0.6 if lines else 0
        total_height = len(lines) * font_size * 1.2
        text_component._text_metrics_cache = TextMetrics(
            ref_font_size=font_size, width_points=max_width, height_points=total_height
        )
        return Size(width=max_width, height=total_height)
