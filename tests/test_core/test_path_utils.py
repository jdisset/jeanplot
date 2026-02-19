"""Tests for path utilities and vector helpers."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from jeanplot import Container
from jeanplot.core.path_utils import (
    normalize_vector,
    is_collinear,
    is_on_segment,
    find_component_by_path,
    create_orthogonal_path,
    create_rounded_orthogonal_path,
    _simplify_orthogonal_path,
)


class TestNormalizeVector:
    """normalize_vector helper."""

    def test_unit_vector_unchanged(self):
        """Unit vector stays normalized."""
        result = normalize_vector((1.0, 0.0))
        assert_allclose(result, (1.0, 0.0))

    def test_scales_to_unit_length(self):
        """Non-unit vector normalized to length 1."""
        result = normalize_vector((3.0, 4.0))
        assert_allclose(result, (0.6, 0.8))

    def test_zero_vector_returns_default(self):
        """Zero vector returns default."""
        result = normalize_vector((0.0, 0.0))
        assert_allclose(result, (1.0, 0.0))

    def test_custom_default(self):
        """Custom default for zero vector."""
        result = normalize_vector((0.0, 0.0), default=(0.0, 1.0))
        assert_allclose(result, (0.0, 1.0))

    def test_tiny_vector_returns_default(self):
        """Near-zero vector returns default."""
        result = normalize_vector((1e-10, 1e-10))
        assert_allclose(result, (1.0, 0.0))


class TestIsCollinear:
    """is_collinear point check."""

    def test_collinear_horizontal(self):
        """Horizontal points are collinear."""
        p1, p2, p3 = np.array([0, 0]), np.array([5, 0]), np.array([10, 0])
        assert is_collinear(p1, p2, p3)

    def test_collinear_diagonal(self):
        """Diagonal points are collinear."""
        p1, p2, p3 = np.array([0, 0]), np.array([1, 1]), np.array([2, 2])
        assert is_collinear(p1, p2, p3)

    def test_not_collinear(self):
        """Triangle points not collinear."""
        p1, p2, p3 = np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
        assert not is_collinear(p1, p2, p3)

    def test_identical_points(self):
        """Identical points are collinear."""
        p = np.array([5, 5])
        assert is_collinear(p, p, p)

    def test_two_identical(self):
        """Two identical points are collinear with third."""
        p1 = np.array([0, 0])
        p2 = np.array([0, 0])
        p3 = np.array([10, 10])
        assert is_collinear(p1, p2, p3)


class TestIsOnSegment:
    """is_on_segment point check."""

    def test_midpoint_on_segment(self):
        """Midpoint is on segment."""
        start, end = np.array([0, 0]), np.array([10, 0])
        mid = np.array([5, 0])
        assert is_on_segment(mid, start, end)

    def test_endpoint_on_segment(self):
        """Endpoints are on segment."""
        start, end = np.array([0, 0]), np.array([10, 0])
        assert is_on_segment(start, start, end)
        assert is_on_segment(end, start, end)

    def test_point_beyond_segment(self):
        """Point beyond segment is not on it."""
        start, end = np.array([0, 0]), np.array([10, 0])
        beyond = np.array([15, 0])
        assert not is_on_segment(beyond, start, end)

    def test_point_off_line(self):
        """Point off line is not on segment."""
        start, end = np.array([0, 0]), np.array([10, 0])
        off = np.array([5, 5])
        assert not is_on_segment(off, start, end)


class TestFindComponentByPath:
    """find_component_by_path navigation."""

    def test_find_direct_child(self):
        """Finds direct child by ID."""
        parent = Container(id="parent")
        child = Container(id="child")
        parent.add_child(child)
        result = find_component_by_path(parent, "child")
        assert result is child

    def test_find_nested_child(self):
        """Finds nested child via path."""
        root = Container(id="root")
        mid = Container(id="mid")
        leaf = Container(id="leaf")
        root.add_child(mid)
        mid.add_child(leaf)
        result = find_component_by_path(root, "mid/leaf")
        assert result is leaf

    def test_empty_path_returns_none(self):
        """Empty path returns None."""
        root = Container(id="root")
        result = find_component_by_path(root, "")
        assert result is None

    def test_missing_child_raises(self):
        """Missing child raises ValueError."""
        root = Container(id="root")
        with pytest.raises(ValueError, match="not in children"):
            find_component_by_path(root, "nonexistent")

    def test_missing_nested_raises(self):
        """Missing nested child raises ValueError."""
        root = Container(id="root")
        child = Container(id="child")
        root.add_child(child)
        with pytest.raises(ValueError, match="not in children"):
            find_component_by_path(root, "child/missing")


class TestCreateOrthogonalPath:
    """create_orthogonal_path generation."""

    def test_horizontal_direct(self):
        """Direct horizontal path."""
        path = create_orthogonal_path(
            start=(0, 0), end=(100, 0),
            start_dir_out=(1, 0), end_dir_out=(-1, 0),
        )
        assert (0, 0) in path
        assert (100, 0) in path

    def test_vertical_direct(self):
        """Direct vertical path."""
        path = create_orthogonal_path(
            start=(0, 0), end=(0, 100),
            start_dir_out=(0, 1), end_dir_out=(0, -1),
        )
        assert (0, 0) in path
        assert (0, 100) in path

    def test_coincident_points(self):
        """Coincident start/end returns both."""
        path = create_orthogonal_path(
            start=(50, 50), end=(50, 50),
            start_dir_out=(1, 0), end_dir_out=(-1, 0),
        )
        assert len(path) == 2
        assert_allclose(path[0], (50, 50))
        assert_allclose(path[1], (50, 50))

    def test_one_turn_path(self):
        """L-shaped path has corner."""
        path = create_orthogonal_path(
            start=(0, 0), end=(50, 50),
            start_dir_out=(1, 0), end_dir_out=(0, -1),
        )
        assert len(path) >= 3

    def test_with_checkpoints(self):
        """Path respects checkpoints."""
        path = create_orthogonal_path(
            start=(0, 0), end=(100, 100),
            start_dir_out=(1, 0), end_dir_out=(-1, 0),
            checkpoints=[(50, 50)],
        )
        # should pass through or near checkpoint
        assert any(abs(p[0] - 50) < 1 and abs(p[1] - 50) < 1 for p in path)


class TestSimplifyOrthogonalPath:
    """_simplify_orthogonal_path reduces redundant points."""

    def test_no_change_minimal(self):
        """Two points unchanged."""
        path = [(0, 0), (10, 0)]
        result = _simplify_orthogonal_path(path)
        assert result == path

    def test_removes_collinear_middle(self):
        """Removes collinear middle point."""
        path = [(0, 0), (5, 0), (10, 0)]
        result = _simplify_orthogonal_path(path)
        assert len(result) == 2
        assert result[0] == (0, 0)
        assert result[1] == (10, 0)

    def test_preserves_corners(self):
        """Preserves L-corner."""
        path = [(0, 0), (10, 0), (10, 10)]
        result = _simplify_orthogonal_path(path)
        assert len(result) == 3

    def test_removes_duplicates(self):
        """Removes duplicate adjacent points."""
        path = [(0, 0), (0, 0), (10, 0)]
        result = _simplify_orthogonal_path(path)
        assert len(result) == 2


class TestCreateRoundedOrthogonalPath:
    """create_rounded_orthogonal_path SVG generation."""

    def test_straight_line(self):
        """Two points creates line path."""
        points = [(0, 0), (100, 0)]
        path = create_rounded_orthogonal_path(points, radius=5)
        assert path.startswith("M 0.000 0.000")
        assert "L 100.000 0.000" in path

    def test_zero_radius_sharp(self):
        """Zero radius creates sharp corners."""
        points = [(0, 0), (50, 0), (50, 50)]
        path = create_rounded_orthogonal_path(points, radius=0)
        assert "A" not in path  # no arc

    def test_rounded_corner_has_arc(self):
        """Positive radius creates arc."""
        points = [(0, 0), (50, 0), (50, 50)]
        path = create_rounded_orthogonal_path(points, radius=10)
        assert "A" in path  # has arc command

    def test_radius_clipped_to_segment(self):
        """Radius clipped to half segment length."""
        points = [(0, 0), (10, 0), (10, 10)]
        path = create_rounded_orthogonal_path(points, radius=100)
        # should still produce valid path with clipped radius
        assert "A" in path

    def test_single_point(self):
        """Single point creates move."""
        points = [(50, 50)]
        path = create_rounded_orthogonal_path(points, radius=5)
        assert "M 50.000 50.000" in path
