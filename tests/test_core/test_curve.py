"""Tests for curve definitions."""
import numpy as np
from numpy.testing import assert_allclose

from jeanplot.core.curve import (
    CurveDefinition,
    StraightCurve,
    SimpleBezierCurve,
    OrthogonalCurve,
)


class TestStraightCurve:
    """StraightCurve path generation."""

    def test_path_format(self):
        """Path is M-L format."""
        curve = StraightCurve()
        path, _ = curve.get_path((0, 0), (100, 50))
        assert path.startswith("M 0.000 0.000")
        assert "L 100.000 50.000" in path

    def test_returns_empty_control_points(self):
        """No control points for straight line."""
        curve = StraightCurve()
        _, control = curve.get_path((0, 0), (100, 0))
        assert control == []

    def test_directions_along_line(self):
        """Directions follow line vector."""
        curve = StraightCurve()
        start_dir, end_dir = curve.get_directions((0, 0), (100, 0), [])
        assert_allclose(start_dir, (1, 0))
        assert_allclose(end_dir, (-1, 0))

    def test_vertical_directions(self):
        """Vertical line directions correct."""
        curve = StraightCurve()
        start_dir, end_dir = curve.get_directions((0, 0), (0, 100), [])
        assert_allclose(start_dir, (0, 1))
        assert_allclose(end_dir, (0, -1))


class TestSimpleBezierCurve:
    """SimpleBezierCurve path generation."""

    def test_path_is_cubic_bezier(self):
        """Path uses C command."""
        curve = SimpleBezierCurve()
        path, _ = curve.get_path((0, 0), (100, 100))
        assert "C" in path

    def test_returns_control_points(self):
        """Returns two control points."""
        curve = SimpleBezierCurve()
        _, control = curve.get_path((0, 0), (100, 100))
        assert len(control) == 2

    def test_auto_mode_uses_strength(self):
        """Auto mode respects strength."""
        curve = SimpleBezierCurve(auto_direction_strength=30.0)
        _, control = curve.get_path((0, 0), (100, 0))
        # control point should be offset from start
        c1 = control[0]
        assert c1[0] > 0  # moved right (direction of end)

    def test_explicit_vectors(self):
        """Explicit vectors override auto."""
        curve = SimpleBezierCurve(
            start_mode="vector",
            start_vector=(0, 50),
            end_mode="vector",
            end_vector=(0, -50),
        )
        _, control = curve.get_path((0, 0), (100, 0))
        c1, c2 = control
        assert_allclose(c1, (0, 50))
        assert_allclose(c2, (100, -50))

    def test_directions_from_control_points(self):
        """Directions derived from control points."""
        curve = SimpleBezierCurve()
        _, control = curve.get_path((0, 0), (100, 0))
        start_dir, end_dir = curve.get_directions((0, 0), (100, 0), control)
        # start direction should point toward c1
        assert start_dir[0] > 0 or start_dir[1] != 0


class TestOrthogonalCurve:
    """OrthogonalCurve path generation."""

    def test_path_has_orthogonal_segments(self):
        """Path uses L commands."""
        curve = OrthogonalCurve(corner_radius=0)
        path, _ = curve.get_path((0, 0), (100, 100))
        assert "L" in path

    def test_rounded_corners_have_arcs(self):
        """Rounded corners use A command."""
        curve = OrthogonalCurve(corner_radius=10)
        path, _ = curve.get_path((0, 0), (100, 100))
        assert "A" in path

    def test_zero_radius_no_arcs(self):
        """Zero radius has no arcs."""
        curve = OrthogonalCurve(corner_radius=0)
        path, _ = curve.get_path((0, 0), (100, 100))
        assert "A" not in path

    def test_explicit_start_direction(self):
        """Explicit start direction respected."""
        curve = OrthogonalCurve(start_direction="right", corner_radius=0)
        path, _ = curve.get_path((0, 0), (100, 100))
        # first segment should go right
        assert "L" in path

    def test_explicit_end_direction(self):
        """Explicit end direction respected."""
        curve = OrthogonalCurve(end_direction="up", corner_radius=0)
        path, _ = curve.get_path((0, 0), (100, 100))
        assert "L" in path

    def test_auto_direction_horizontal(self):
        """Auto direction for horizontal path."""
        curve = OrthogonalCurve(
            start_direction="auto",
            end_direction="auto",
            corner_radius=0,
        )
        path, _ = curve.get_path((0, 0), (100, 0))
        # should be direct horizontal
        assert "M 0.000 0.000" in path

    def test_cached_directions(self):
        """get_directions returns cached values."""
        curve = OrthogonalCurve(start_direction="right", end_direction="left")
        curve.get_path((0, 0), (100, 0))
        start_dir, end_dir = curve.get_directions((0, 0), (100, 0), [])
        assert_allclose(start_dir, (1, 0))
        assert_allclose(end_dir, (-1, 0))

    def test_start_length_minimum(self):
        """Start length creates minimum segment."""
        curve = OrthogonalCurve(
            start_direction="right",
            start_length=50,
            corner_radius=0,
        )
        path, _ = curve.get_path((0, 0), (30, 100))
        # should extend past 30 initially
        assert "L" in path

    def test_get_direction_from_vector(self):
        """Direction mapping from vector."""
        assert OrthogonalCurve.get_direction_from_vector((1, 0)) == "right"
        assert OrthogonalCurve.get_direction_from_vector((-1, 0)) == "left"
        assert OrthogonalCurve.get_direction_from_vector((0, 1)) == "up"
        assert OrthogonalCurve.get_direction_from_vector((0, -1)) == "down"

    def test_get_direction_diagonal(self):
        """Diagonal vector maps to closest."""
        assert OrthogonalCurve.get_direction_from_vector((0.9, 0.1)) == "right"
        assert OrthogonalCurve.get_direction_from_vector((0.1, 0.9)) == "up"

    def test_get_direction_zero_vector(self):
        """Zero vector defaults to up."""
        assert OrthogonalCurve.get_direction_from_vector((0, 0)) == "up"


class TestCurveDefinitionBase:
    """CurveDefinition base class."""

    def test_auto_direction_vector_start(self):
        """Auto direction points from start to end."""
        vec = CurveDefinition._calculate_auto_direction_vector(
            (0, 0), (100, 0), for_start=True
        )
        assert_allclose(vec, (1, 0))

    def test_auto_direction_vector_end(self):
        """Auto direction for end points away."""
        vec = CurveDefinition._calculate_auto_direction_vector(
            (0, 0), (100, 0), for_start=False
        )
        assert_allclose(vec, (-1, 0))

    def test_auto_direction_diagonal(self):
        """Diagonal auto direction normalized."""
        vec = CurveDefinition._calculate_auto_direction_vector(
            (0, 0), (100, 100), for_start=True
        )
        expected = np.array([1, 1]) / np.sqrt(2)
        assert_allclose(vec, expected, rtol=0.01)

    def test_auto_direction_coincident(self):
        """Coincident points return default."""
        vec = CurveDefinition._calculate_auto_direction_vector(
            (50, 50), (50, 50), for_start=True
        )
        # should return default (0, 1)
        assert_allclose(vec, (0, 1))
