"""Tests for Connection and curve types."""

import pytest
from jeanplot import (
    Container,
    Connection,
    StraightCurve,
    SimpleBezierCurve,
    OrthogonalCurve,
    AnchorComponent,
    Size,
    BoxStyle,
    render_to_svg,
    parse_svg,
)


@pytest.fixture
def two_boxes(mock_renderer):
    """Two positioned boxes for connection tests."""
    parent = Container(id="parent", min_dimensions=Size(400, 200))
    box1 = Container(
        id="box1",
        min_dimensions=Size(50, 50),
        style=BoxStyle(background_color="#ff0000"),
    )
    box2 = Container(
        id="box2",
        min_dimensions=Size(50, 50),
        offset=dict(absolute=(200, 100)),
        style=BoxStyle(background_color="#00ff00"),
    )
    parent.add_children([box1, box2])
    parent.measure_and_layout(mock_renderer)
    return parent, box1, box2


class TestConnectionBasics:
    """Connection creation and endpoint references."""

    def test_connection_stores_ids(self, two_boxes):
        """Connection stores component ID strings."""
        parent, box1, box2 = two_boxes
        conn = Connection(
            id="conn",
            start_component="box1",
            end_component="box2",
        )
        parent.add_child(conn)

        assert conn.start_component == "box1"
        assert conn.end_component == "box2"

    def test_connection_is_overlay(self, two_boxes):
        """Connection defaults to overlay."""
        conn = Connection(
            id="conn",
            start_component="box1",
            end_component="box2",
        )
        assert conn.is_overlay is True


class TestCurveTypes:
    """Different curve type behaviors."""

    def test_straight_curve_path(self):
        """StraightCurve produces direct line path."""
        curve = StraightCurve()
        start, end = (0, 0), (100, 50)
        path, _ = curve.get_path(start, end)

        assert "M 0" in path
        assert "L 100" in path

    def test_bezier_produces_curve(self):
        """SimpleBezierCurve produces curved path."""
        curve = SimpleBezierCurve()
        start, end = (0, 0), (100, 50)
        path, _ = curve.get_path(start, end)

        # Bezier curves use C or Q commands
        assert "C" in path or "Q" in path

    def test_orthogonal_segments(self):
        """OrthogonalCurve produces right-angle segments."""
        curve = OrthogonalCurve()
        start, end = (0, 0), (100, 50)
        path, _ = curve.get_path(start, end)

        # Orthogonal uses L commands (lines)
        assert "L" in path


class TestConnectionRendering:
    """Connection visual output."""

    def test_connection_renders_path(self, two_boxes, mock_renderer):
        """Connection renders as path in SVG."""
        parent, box1, box2 = two_boxes
        conn = Connection(
            id="conn",
            start_component="box1",
            end_component="box2",
        )
        parent.add_child(conn)

        svg = render_to_svg(parent)
        root = parse_svg(svg)

        # Connection should produce a path element
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        # At least one path (from connection)
        assert len(paths) >= 1

    def test_connection_styled(self, two_boxes, mock_renderer):
        """Connection style attributes applied."""
        parent, box1, box2 = two_boxes
        conn = Connection(
            id="conn",
            start_component="box1",
            end_component="box2",
            color="#ff0000",
            line_width=3.0,
        )
        parent.add_child(conn)

        svg = render_to_svg(parent)
        assert "#ff0000" in svg or "ff0000" in svg


class TestAnchors:
    """Anchor-based connection endpoints."""

    def test_anchors_on_component(self, mock_renderer):
        """Component can have multiple anchors."""
        box = Container(id="box", min_dimensions=Size(50, 50))
        anchor1 = AnchorComponent(id="right", direction=(1, 0))
        anchor2 = AnchorComponent(id="left", direction=(-1, 0))
        box.add_children([anchor1, anchor2])
        box.measure_and_layout(mock_renderer)

        assert anchor1 in box.children
        assert anchor2 in box.children

    def test_anchor_path_reference_works(self, mock_renderer):
        """Connection can target anchor by path."""
        parent = Container(id="parent", min_dimensions=Size(400, 200))
        box1 = Container(id="box1", min_dimensions=Size(50, 50))
        anchor = AnchorComponent(id="right", direction=(1, 0))
        box1.add_child(anchor)

        box2 = Container(
            id="box2",
            min_dimensions=Size(50, 50),
            offset=dict(absolute=(200, 0)),
        )
        parent.add_children([box1, box2])

        conn = Connection(
            id="conn",
            start_component="right",
            end_component="box2",
        )
        parent.add_child(conn)

        # Should render without error
        svg = render_to_svg(parent)
        assert "path" in svg.lower()
