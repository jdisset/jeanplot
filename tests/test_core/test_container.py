"""Tests for Container layout engine."""
import pytest
from jeanplot import Container, Size, BoxStyle, LayoutConstraints, Offset


class TestRowLayout:
    """Row layout positioning."""

    def test_children_horizontal(self, mock_renderer):
        """Children flow left to right."""
        parent = Container(
            id="row",
            min_dimensions=Size(300, 100),
            layout=LayoutConstraints(direction="row", gap=0),
        )
        c1 = Container(id="c1", min_dimensions=Size(50, 50))
        c2 = Container(id="c2", min_dimensions=Size(50, 50))
        parent.add_children([c1, c2])
        parent.measure_and_layout(mock_renderer)

        assert c1._layout_origin_in_parent[0] == 0
        assert c2._layout_origin_in_parent[0] == 50

    def test_gap_adds_spacing(self, mock_renderer):
        """Gap increases spacing between children."""
        parent = Container(
            id="row",
            min_dimensions=Size(300, 100),
            layout=LayoutConstraints(direction="row", gap=20),
        )
        c1 = Container(id="c1", min_dimensions=Size(50, 50))
        c2 = Container(id="c2", min_dimensions=Size(50, 50))
        parent.add_children([c1, c2])
        parent.measure_and_layout(mock_renderer)

        assert c2._layout_origin_in_parent[0] == 70  # 50 + 20

    def test_padding_offsets_children(self, mock_renderer):
        """Padding moves children inward."""
        parent = Container(
            id="row",
            min_dimensions=Size(300, 100),
            layout=LayoutConstraints(direction="row"),
            style=BoxStyle(padding=(10, 10, 10, 10)),
        )
        c1 = Container(id="c1", min_dimensions=Size(50, 50))
        parent.add_child(c1)
        parent.measure_and_layout(mock_renderer)

        assert c1._layout_origin_in_parent == (10, 10)


class TestColumnLayout:
    """Column layout positioning."""

    def test_children_vertical(self, mock_renderer):
        """Children flow top to bottom."""
        parent = Container(
            id="col",
            min_dimensions=Size(100, 300),
            layout=LayoutConstraints(direction="column", gap=0),
        )
        c1 = Container(id="c1", min_dimensions=Size(50, 50))
        c2 = Container(id="c2", min_dimensions=Size(50, 50))
        parent.add_children([c1, c2])
        parent.measure_and_layout(mock_renderer)

        assert c1._layout_origin_in_parent[1] == 0
        assert c2._layout_origin_in_parent[1] == 50


class TestAlignItems:
    """Cross-axis alignment."""

    def test_start_alignment(self, mock_renderer):
        """align_items=start places at cross-axis start."""
        parent = Container(
            id="row",
            min_dimensions=Size(200, 100),
            layout=LayoutConstraints(direction="row", align_items="start"),
        )
        c1 = Container(id="c1", min_dimensions=Size(50, 30))
        parent.add_child(c1)
        parent.measure_and_layout(mock_renderer)

        assert c1._layout_origin_in_parent[1] == 0

    def test_center_alignment(self, mock_renderer):
        """align_items=center places at cross-axis middle."""
        parent = Container(
            id="row",
            min_dimensions=Size(200, 100),
            layout=LayoutConstraints(direction="row", align_items="center"),
        )
        c1 = Container(id="c1", min_dimensions=Size(50, 30))
        parent.add_child(c1)
        parent.measure_and_layout(mock_renderer)

        assert c1._layout_origin_in_parent[1] == 35  # (100-30)/2

    def test_end_alignment(self, mock_renderer):
        """align_items=end places at cross-axis end."""
        parent = Container(
            id="row",
            min_dimensions=Size(200, 100),
            layout=LayoutConstraints(direction="row", align_items="end"),
        )
        c1 = Container(id="c1", min_dimensions=Size(50, 30))
        parent.add_child(c1)
        parent.measure_and_layout(mock_renderer)

        assert c1._layout_origin_in_parent[1] == 70  # 100-30


class TestJustifyContent:
    """Main-axis distribution."""

    @pytest.mark.parametrize("justify,expected_x1", [
        ("start", 0),
        ("end", 100),  # 200 - 2*50
    ])
    def test_justify_positions(self, mock_renderer, justify, expected_x1):
        """Justify content positions first child correctly."""
        parent = Container(
            id="row",
            min_dimensions=Size(200, 100),
            layout=LayoutConstraints(direction="row", justify_content=justify, gap=0),
        )
        c1 = Container(id="c1", min_dimensions=Size(50, 50))
        c2 = Container(id="c2", min_dimensions=Size(50, 50))
        parent.add_children([c1, c2])
        parent.measure_and_layout(mock_renderer)

        assert c1._layout_origin_in_parent[0] == expected_x1


class TestOverlay:
    """Overlay component behavior."""

    def test_overlay_not_in_layout(self, mock_renderer):
        """Overlay children don't participate in flow layout."""
        parent = Container(
            id="row",
            min_dimensions=Size(200, 100),
            layout=LayoutConstraints(direction="row"),
        )
        overlay = Container(id="over", min_dimensions=Size(50, 50), is_overlay=True)
        c1 = Container(id="c1", min_dimensions=Size(50, 50))
        parent.add_children([overlay, c1])
        parent.measure_and_layout(mock_renderer)

        # c1 at origin, not pushed by overlay
        assert c1._layout_origin_in_parent[0] == 0

    def test_overlay_at_origin(self, mock_renderer):
        """Overlay positioned at parent origin by default."""
        parent = Container(id="row", min_dimensions=Size(200, 100))
        overlay = Container(id="over", min_dimensions=Size(50, 50), is_overlay=True)
        parent.add_child(overlay)
        parent.measure_and_layout(mock_renderer)

        assert overlay._layout_origin_in_parent == (0, 0)


class TestOffset:
    """Component offset behavior."""

    def test_absolute_offset(self, mock_renderer):
        """Absolute offset shifts component position."""
        from numpy.testing import assert_allclose

        parent = Container(id="row", min_dimensions=Size(200, 100))
        c1 = Container(
            id="c1",
            min_dimensions=Size(50, 50),
            offset=Offset(absolute=(10, 20)),
        )
        parent.add_child(c1)
        parent.measure_and_layout(mock_renderer)

        origin = c1.get_world_origin()
        assert_allclose(origin, (10, 20))

    def test_relative_offset(self, mock_renderer):
        """Relative offset based on self dimensions."""
        from numpy.testing import assert_allclose

        parent = Container(id="row", min_dimensions=Size(200, 100))
        c1 = Container(
            id="c1",
            min_dimensions=Size(100, 50),
            offset=Offset(relative=(0.5, 0)),  # half width
        )
        parent.add_child(c1)
        parent.measure_and_layout(mock_renderer)

        origin = c1.get_world_origin()
        assert_allclose(origin, (50, 0))


class TestNaturalSize:
    """Container natural sizing."""

    def test_shrink_to_content(self, mock_renderer):
        """Container with no min dims sizes to content."""
        parent = Container(id="parent")
        c1 = Container(id="c1", min_dimensions=Size(80, 60))
        parent.add_child(c1)
        parent.measure_and_layout(mock_renderer)

        assert parent._dimensions.width == 80
        assert parent._dimensions.height == 60

    def test_padding_increases_natural_size(self, mock_renderer):
        """Padding adds to natural size."""
        parent = Container(
            id="parent",
            style=BoxStyle(padding=(10, 10, 10, 10)),
        )
        c1 = Container(id="c1", min_dimensions=Size(80, 60))
        parent.add_child(c1)
        parent.measure_and_layout(mock_renderer)

        assert parent._dimensions.width == 100
        assert parent._dimensions.height == 80
