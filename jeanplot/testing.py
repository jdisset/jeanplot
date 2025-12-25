"""Testing utilities for jeanplot."""

from __future__ import annotations

import re
import hashlib
from lxml import etree
import numpy as np

from jeanplot.core.component import Component
from jeanplot.core.models import Size, TextMetrics
from jeanplot.core.renderer.base import BaseRenderer
from jeanplot.core.text import Text


class MockRenderer(BaseRenderer):
    """Renderer that returns cached metrics without matplotlib dependency."""

    RENDERER_NAME = "mock"

    _text_metrics: dict[str, tuple[float, float]] = {}

    def __init__(self):
        super().__init__()

    def create_context(self, width: float = 800, height: float = 600, **kwargs):
        return {"width": width, "height": height}

    def render_component(self, context, component: Component, adjust_lims: bool = True):
        component.measure_and_layout(self)

    def render_to_output(self, context, output=None, **kwargs):
        pass

    def render_rectangle(self, context, bounds, style, matrix, component=None):
        pass

    def render_svg(self, context, svg_element, matrix):
        pass

    def render_path(self, context, path_data, matrix, line_width_mode="point", **kwargs):
        pass

    def render_text(self, context, text_component, matrix):
        pass

    def render_connection_curve(
        self, context, connection, local_start, local_end, local_control_points, path_string, matrix
    ):
        pass

    def render_debug(self, context, component, matrix):
        pass

    def measure_text(self, text_component: Text) -> Size:
        text = text_component.text or ""
        font_size = text_component.font_size
        key = f"{text}_{font_size}"
        if key not in self._text_metrics:
            lines = text.split("\n")
            width = max(len(line) for line in lines) * font_size * 0.6 if lines else 0
            height = len(lines) * font_size * 1.2
            self._text_metrics[key] = (width, height)

        w, h = self._text_metrics[key]
        text_component._text_metrics_cache = TextMetrics(
            ref_font_size=font_size, width_points=w, height_points=h
        )
        return Size(width=w, height=h)


def render_to_svg(component: Component) -> str:
    """Render component to SVG string for testing."""
    from jeanplot.core.renderer.svg import SVGRenderer

    renderer = SVGRenderer()
    return renderer.render_to_string(component)


def parse_svg(svg_string: str) -> etree._Element:
    """Parse SVG string to element tree."""
    return etree.fromstring(svg_string.encode())


def get_element_by_id(svg: etree._Element, element_id: str) -> etree._Element | None:
    """Find element by ID in SVG tree."""
    results = svg.xpath(f"//*[@id='{element_id}']")
    return results[0] if results else None


def get_element_transform(element: etree._Element) -> np.ndarray | None:
    """Extract transform matrix from element."""
    transform_str = element.get("transform")
    if not transform_str:
        return np.eye(3)

    matrix_match = re.search(r"matrix\(([^)]+)\)", transform_str)
    if matrix_match:
        vals = [float(v.strip()) for v in matrix_match.group(1).split(",")]
        if len(vals) == 6:
            return np.array([[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]], [0, 0, 1]])

    translate_match = re.search(r"translate\(([^)]+)\)", transform_str)
    if translate_match:
        vals = [float(v.strip()) for v in translate_match.group(1).split(",")]
        if len(vals) >= 2:
            return np.array([[1, 0, vals[0]], [0, 1, vals[1]], [0, 0, 1]])

    return np.eye(3)


def get_element_bounds(svg: etree._Element, element_id: str) -> tuple[float, float, float, float]:
    """Extract bounding box of element by ID."""
    elem = get_element_by_id(svg, element_id)
    if elem is None:
        raise ValueError(f"element '{element_id}' not found")

    matrix = get_element_transform(elem)

    rect = elem.find(".//{http://www.w3.org/2000/svg}rect")
    if rect is None:
        rect = elem if elem.tag.endswith("rect") else None

    if rect is not None:
        x = float(rect.get("x", 0))
        y = float(rect.get("y", 0))
        w = float(rect.get("width", 0))
        h = float(rect.get("height", 0))

        corners = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]]).T
        transformed = matrix @ corners

        min_x = np.min(transformed[0, :])
        min_y = np.min(transformed[1, :])
        max_x = np.max(transformed[0, :])
        max_y = np.max(transformed[1, :])
        return (min_x, min_y, max_x, max_y)

    origin = matrix @ np.array([0, 0, 1])
    return (origin[0], origin[1], origin[0], origin[1])


def get_element_position(svg: etree._Element, element_id: str) -> tuple[float, float]:
    """Get top-left position of element."""
    bounds = get_element_bounds(svg, element_id)
    return (bounds[0], bounds[1])


def assert_element_position(svg: str, element_id: str, x: float, y: float, tol: float = 0.1):
    """Assert element is at expected position."""
    root = parse_svg(svg)
    bounds = get_element_bounds(root, element_id)
    assert abs(bounds[0] - x) < tol, f"{element_id} x={bounds[0]:.2f}, expected {x:.2f}"
    assert abs(bounds[1] - y) < tol, f"{element_id} y={bounds[1]:.2f}, expected {y:.2f}"


def assert_element_size(svg: str, element_id: str, width: float, height: float, tol: float = 0.1):
    """Assert element has expected dimensions."""
    root = parse_svg(svg)
    bounds = get_element_bounds(root, element_id)
    actual_w = bounds[2] - bounds[0]
    actual_h = bounds[3] - bounds[1]
    assert abs(actual_w - width) < tol, f"{element_id} width={actual_w:.2f}, expected {width:.2f}"
    assert abs(actual_h - height) < tol, (
        f"{element_id} height={actual_h:.2f}, expected {height:.2f}"
    )


def assert_elements_connected(svg: str, start_id: str, end_id: str) -> bool:
    """Check if a path exists connecting two elements (by ID pattern)."""
    root = parse_svg(svg)
    paths = root.xpath("//*[local-name()='path']")

    start_bounds = get_element_bounds(root, start_id)
    end_bounds = get_element_bounds(root, end_id)

    start_center = (
        (start_bounds[0] + start_bounds[2]) / 2,
        (start_bounds[1] + start_bounds[3]) / 2,
    )
    end_center = ((end_bounds[0] + end_bounds[2]) / 2, (end_bounds[1] + end_bounds[3]) / 2)

    for path in paths:
        d = path.get("d", "")
        move_match = re.search(r"M\s*([-\d.]+)[,\s]+([-\d.]+)", d)
        if not move_match:
            continue

        path_start = (float(move_match.group(1)), float(move_match.group(2)))

        end_matches = list(re.finditer(r"[LCQ]\s*(?:[-\d.]+[,\s]+)*?([-\d.]+)[,\s]+([-\d.]+)", d))
        if not end_matches:
            continue

        path_end = (float(end_matches[-1].group(1)), float(end_matches[-1].group(2)))

        tol = 20.0
        start_near = (
            abs(path_start[0] - start_center[0]) < tol
            and abs(path_start[1] - start_center[1]) < tol
        ) or (abs(path_start[0] - end_center[0]) < tol and abs(path_start[1] - end_center[1]) < tol)
        end_near = (
            abs(path_end[0] - start_center[0]) < tol and abs(path_end[1] - start_center[1]) < tol
        ) or (abs(path_end[0] - end_center[0]) < tol and abs(path_end[1] - end_center[1]) < tol)

        if start_near and end_near:
            return True

    return False


def normalize_svg(svg: str) -> str:
    """Normalize SVG for comparison by removing timestamps and IDs."""
    svg = re.sub(r'id="[^"]*"', "", svg)
    svg = re.sub(r"\s+", " ", svg)
    svg = re.sub(r">\s+<", "><", svg)
    return svg.strip()


def svg_hash(svg: str) -> str:
    """Generate hash of normalized SVG for visual regression testing."""
    normalized = normalize_svg(svg)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]
