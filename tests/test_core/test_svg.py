"""Tests for SVG data models and utilities."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from jeanplot.core.svg import (
    SVGPathData,
    SVGContent,
    SVGTextContent,
    LineEndArrow,
    LineEndCircle,
    LineEndFlat,
    make_svg_line,
    arc_to_bezier,
    create_arrow_cap,
    create_circle_cap,
    create_flat_cap,
    get_svg_data,
    _parse_svg_dimension,
)


class TestSVGPathData:
    """SVGPathData model."""

    def test_minimal_path(self):
        """Creates path with just d attribute."""
        path = SVGPathData(d="M 0 0 L 10 10")
        assert path.d == "M 0 0 L 10 10"
        assert path.fill is None
        assert path.stroke is None

    def test_styled_path(self):
        """Path with styling attributes."""
        path = SVGPathData(
            d="M 0 0 L 10 10",
            fill="#ff0000",
            stroke="#000000",
            stroke_width=2.0,
        )
        assert path.fill == "#ff0000"
        assert path.stroke == "#000000"
        assert path.stroke_width == 2.0

    def test_dash_array(self):
        """Path with dash array."""
        path = SVGPathData(
            d="M 0 0",
            dash_array=(5.0, 3.0),
            line_style="custom",
        )
        assert path.dash_array == (5.0, 3.0)
        assert path.line_style == "custom"

    def test_frozen(self):
        """Path is immutable."""
        path = SVGPathData(d="M 0 0")
        with pytest.raises(Exception):  # Pydantic ValidationError
            path.d = "M 1 1"


class TestSVGContent:
    """SVGContent container model."""

    def test_defaults(self):
        """Default dimensions are 100x100."""
        content = SVGContent()
        assert content.width == 100
        assert content.height == 100
        assert content.paths == ()

    def test_with_paths(self):
        """Content with path list."""
        paths = (SVGPathData(d="M 0 0"),)
        content = SVGContent(width=200, height=150, paths=paths)
        assert len(content.paths) == 1
        assert content.width == 200

    def test_viewbox(self):
        """Content with viewBox."""
        content = SVGContent(viewBox=(0, 0, 100, 100))
        assert content.viewBox == (0, 0, 100, 100)


class TestSVGTextContent:
    """SVGTextContent for text rendering."""

    def test_defaults(self):
        """Default size is zero."""
        text = SVGTextContent()
        assert text.measured_width == 0
        assert text.measured_height == 0
        assert text.glyph_paths == ()

    def test_with_measurements(self):
        """Text with measured size."""
        text = SVGTextContent(measured_width=50, measured_height=12)
        assert text.measured_width == 50
        assert text.measured_height == 12


class TestLineEndTypes:
    """Line end cap definitions."""

    def test_arrow_defaults(self):
        """Arrow has default styling."""
        arrow = LineEndArrow()
        assert arrow.stroke_color == "#000000"
        assert arrow.length == 8.0
        assert arrow.angle == 30.0
        assert not arrow.closed

    def test_circle_defaults(self):
        """Circle has default styling."""
        circle = LineEndCircle()
        assert circle.radius == 4.0
        assert circle.stroke_color == "#000000"

    def test_flat_defaults(self):
        """Flat cap has default styling."""
        flat = LineEndFlat()
        assert flat.length == 6.0


class TestMakeSvgLine:
    """make_svg_line helper."""

    def test_creates_horizontal_line(self):
        """Creates line SVGContent."""
        content = make_svg_line(100, 2, "#ff0000")
        assert content.width == 100
        assert content.height == 2
        assert len(content.paths) == 1

    def test_path_is_horizontal(self):
        """Path is horizontal line."""
        content = make_svg_line(50, 4, "#000000")
        path = content.paths[0]
        assert "M 0" in path.d
        assert "L 50.000" in path.d

    def test_zero_width_empty(self):
        """Zero width returns empty."""
        content = make_svg_line(0, 2, "#000000")
        assert content.paths == ()

    def test_zero_thickness_empty(self):
        """Zero thickness returns empty."""
        content = make_svg_line(100, 0, "#000000")
        assert content.paths == ()

    def test_none_color_empty(self):
        """None color returns empty."""
        content = make_svg_line(100, 2, None)
        assert content.paths == ()


class TestArcToBezier:
    """arc_to_bezier conversion."""

    def test_quarter_arc_returns_four_points(self):
        """Quarter arc has start, control1, control2, end."""
        p0, p1, p2, p3 = arc_to_bezier(0, 0, 10, 0, 90)
        assert len(p0) == 2
        assert len(p1) == 2
        assert len(p2) == 2
        assert len(p3) == 2

    def test_start_end_on_circle(self):
        """Start and end are on circle."""
        p0, _, _, p3 = arc_to_bezier(0, 0, 10, 0, 90)
        assert_allclose(np.linalg.norm(p0), 10, rtol=0.01)
        assert_allclose(np.linalg.norm(p3), 10, rtol=0.01)


class TestCreateArrowCap:
    """create_arrow_cap SVG path generation."""

    def test_creates_path(self):
        """Arrow cap creates path."""
        arrow = LineEndArrow(length=10, angle=60)
        path = create_arrow_cap((100, 50), (1, 0), arrow)
        assert "M" in path.d
        assert "L" in path.d

    def test_closed_arrow_has_z(self):
        """Closed arrow ends with Z."""
        arrow = LineEndArrow(closed=True)
        path = create_arrow_cap((0, 0), (1, 0), arrow)
        assert path.d.endswith("Z")

    def test_open_arrow_no_z(self):
        """Open arrow has no Z."""
        arrow = LineEndArrow(closed=False)
        path = create_arrow_cap((0, 0), (1, 0), arrow)
        assert "Z" not in path.d


class TestCreateCircleCap:
    """create_circle_cap SVG path generation."""

    def test_creates_circle_path(self):
        """Circle cap creates arc path."""
        circle = LineEndCircle(radius=5)
        path = create_circle_cap((50, 50), (1, 0), circle)
        assert "A" in path.d  # arc command
        assert path.d.endswith("Z")


class TestCreateFlatCap:
    """create_flat_cap SVG path generation."""

    def test_creates_line_path(self):
        """Flat cap creates short line."""
        flat = LineEndFlat(length=10)
        path = create_flat_cap((50, 50), (1, 0), flat)
        assert "M" in path.d
        assert "L" in path.d

    def test_perpendicular_to_direction(self):
        """Flat cap is perpendicular to direction."""
        flat = LineEndFlat(length=10)
        path = create_flat_cap((50, 50), (1, 0), flat)
        # direction is (1,0), so cap should be vertical
        assert "50.000" in path.d  # y stays at 50


class TestParseSvgDimension:
    """_parse_svg_dimension unit conversion."""

    def test_pixels(self):
        """Pixel value unchanged."""
        result = _parse_svg_dimension("100px", 72)
        assert result == 100

    def test_no_unit(self):
        """No unit treated as user units."""
        result = _parse_svg_dimension("50", 72)
        assert result == 50

    def test_millimeters(self):
        """Millimeters converted."""
        result = _parse_svg_dimension("25.4mm", 72)
        assert_allclose(result, 72, rtol=0.01)

    def test_centimeters(self):
        """Centimeters converted."""
        result = _parse_svg_dimension("2.54cm", 72)
        assert_allclose(result, 72, rtol=0.01)

    def test_inches(self):
        """Inches converted."""
        result = _parse_svg_dimension("1in", 72)
        assert result == 72

    def test_points(self):
        """Points converted."""
        result = _parse_svg_dimension("72pt", 72)
        assert_allclose(result, 72, rtol=0.01)

    def test_invalid_returns_zero(self):
        """Invalid string returns 0."""
        result = _parse_svg_dimension("abc", 72)
        assert result == 0

    def test_non_string_returns_zero(self):
        """Non-string returns 0."""
        result = _parse_svg_dimension(123, 72)
        assert result == 0


class TestGetSvgData:
    """get_svg_data parsing."""

    def test_parses_svg_string(self):
        """Parses SVG string content."""
        svg = '<svg width="100" height="50"><path d="M 0 0 L 10 10"/></svg>'
        content = get_svg_data(svg)
        assert content.width == 100
        assert content.height == 50
        assert len(content.paths) == 1

    def test_parses_viewbox(self):
        """Parses viewBox attribute."""
        svg = '<svg viewBox="0 0 200 100"><path d="M 0 0"/></svg>'
        content = get_svg_data(svg)
        assert content.viewBox == (0, 0, 200, 100)

    def test_parses_path_styling(self):
        """Parses path fill and stroke."""
        svg = '<svg><path d="M 0 0" fill="#ff0000" stroke="#0000ff" stroke-width="2"/></svg>'
        content = get_svg_data(svg)
        path = content.paths[0]
        assert path.fill.startswith("#ff0000")
        assert path.stroke.startswith("#0000ff")
        assert path.stroke_width == 2.0

    def test_bytes_input(self):
        """Accepts bytes input."""
        svg = b'<svg width="50" height="50"><path d="M 0 0"/></svg>'
        content = get_svg_data(svg)
        assert content.width == 50

    def test_empty_on_invalid(self):
        """Returns empty on invalid content."""
        content = get_svg_data("not valid svg")
        assert content.paths == ()

    def test_style_attribute_parsing(self):
        """Parses style attribute."""
        svg = '<svg><path d="M 0 0" style="fill:#00ff00;stroke:#ff0000"/></svg>'
        content = get_svg_data(svg)
        path = content.paths[0]
        assert path.fill.startswith("#00ff00")
        assert path.stroke.startswith("#ff0000")

    def test_dash_array_parsing(self):
        """Parses dash array."""
        svg = '<svg><path d="M 0 0" stroke-dasharray="5,3"/></svg>'
        content = get_svg_data(svg)
        path = content.paths[0]
        assert path.dash_array == (5.0, 3.0)
        assert path.line_style == "custom"
