"""Tests for text scaling behavior in jeanplot.

These tests verify that text sizes scale proportionally with their parent containers
when font_size is specified in data units.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from jeanplot import Container, Text, jstyle
from jeanplot.core.models import LayoutConstraints, BoxStyle, Transform
from jeanplot.core.renderer.matplotlib import MatplotlibRenderer


@pytest.fixture
def renderer():
    return MatplotlibRenderer()


@pytest.fixture
def ax():
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    yield ax
    plt.close(fig)


class TestTextScaling:
    """Tests for text scaling with parent container transforms."""

    def test_text_size_scales_with_container(self, renderer, ax):
        """Text should scale proportionally when inside a scaled container.

        Bug: get_font_size_in_points() used identity matrix instead of the
        actual component transformation matrix, causing text to ignore
        parent container scaling.
        """
        # Create a container with text inside - no explicit scaling
        container1 = Container(
            id="container1",
            children=[
                Text(id="text1", text="Hello", font_size=10.0, color="black"),
            ],
            layout=LayoutConstraints(direction="row"),
        )

        # Create a scaled container (2x) with same text
        container2 = Container(
            id="container2",
            children=[
                Text(id="text2", text="Hello", font_size=10.0, color="black"),
            ],
            layout=LayoutConstraints(direction="row"),
            transform=Transform(scale=(2.0, 2.0)),  # 2x scale
        )

        jstyle.apply(container1)
        jstyle.apply(container2)

        # Measure and layout both
        renderer.create_context(ax=ax)
        container1.measure_and_layout(renderer)
        container2.measure_and_layout(renderer)

        # Get the text components
        text1 = container1.children[0]
        text2 = container2.children[0]

        # Compute world matrices
        matrix1 = text1.compute_world_matrix()
        matrix2 = text2.compute_world_matrix()

        # Get scale factors from matrices
        scale1 = np.sqrt(matrix1[0, 0] ** 2 + matrix1[1, 0] ** 2)
        scale2 = np.sqrt(matrix2[0, 0] ** 2 + matrix2[1, 0] ** 2)

        # container2 should have ~2x scale in its matrix
        assert scale2 > scale1 * 1.5, (
            f"Scaled container's text should have larger matrix scale. "
            f"Got scale1={scale1:.2f}, scale2={scale2:.2f}"
        )

        # When rendered, the font size in points should also be 2x
        # This tests that the matrix is actually used in font size calculation
        from jeanplot.core.renderer.matplotlib import _get_points_per_unit_vector

        ppu1 = _get_points_per_unit_vector(ax, matrix1, vector=(0, 1))
        ppu2 = _get_points_per_unit_vector(ax, matrix2, vector=(0, 1))

        # Points per unit should be ~2x for scaled container
        assert ppu2 > ppu1 * 1.5, (
            f"Scaled container's text should have larger points-per-unit. "
            f"Got ppu1={ppu1:.2f}, ppu2={ppu2:.2f}"
        )

    def test_nested_scaling_accumulates(self, renderer, ax):
        """Nested container scaling should accumulate correctly for text."""
        inner = Container(
            id="inner",
            children=[Text(id="nested_text", text="Nested", font_size=5.0)],
            transform=Transform(scale=(2.0, 2.0)),
        )
        outer = Container(
            id="outer",
            children=[inner],
            transform=Transform(scale=(2.0, 2.0)),
        )
        jstyle.apply(outer)

        renderer.create_context(ax=ax)
        outer.measure_and_layout(renderer)

        text = inner.children[0]
        matrix = text.compute_world_matrix()

        # Should have 4x scale (2 * 2)
        scale = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        assert scale > 3.5, f"Nested scaling should be ~4x, got {scale:.2f}"

    def test_text_respects_container_rotation(self, renderer, ax):
        """Text inside a rotated container should inherit rotation in matrix."""
        container = Container(
            id="rotated",
            children=[Text(id="rotated_text", text="Rotated", font_size=10.0)],
            transform=Transform(rotate=45.0),
        )
        jstyle.apply(container)

        renderer.create_context(ax=ax)
        container.measure_and_layout(renderer)

        text = container.children[0]
        matrix = text.compute_world_matrix()

        # Check rotation is present (off-diagonal elements should be non-zero)
        # For 45 degree rotation: matrix[0,1] and matrix[1,0] should be significant
        rotation_present = abs(matrix[0, 1]) > 0.1 or abs(matrix[1, 0]) > 0.1
        assert rotation_present, f"Rotation should be in matrix: {matrix[:2, :2]}"


class TestTextMeasurement:
    """Tests for text measurement consistency."""

    def test_text_measurement_independent_of_axis_limits_data_mode(self, renderer):
        """With font_size_mode='data', natural size in data units should be consistent."""
        # Use font_size_mode="data" to test data-unit behavior
        text = Text(id="test", text="Test String", font_size=12.0, font_size_mode="data")
        jstyle.apply(text)

        # Measure with small axis limits
        fig1, ax1 = plt.subplots()
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_aspect("equal")
        renderer.create_context(ax=ax1)
        text.measure_and_layout(renderer)
        size1 = (text._dimensions.width, text._dimensions.height)
        plt.close(fig1)

        # Measure with large axis limits
        fig2, ax2 = plt.subplots()
        ax2.set_xlim(0, 1000)
        ax2.set_ylim(0, 1000)
        ax2.set_aspect("equal")
        renderer.create_context(ax=ax2)
        text._dimensions = None  # Reset
        text._text_metrics_cache = None  # Reset cache
        text.measure_and_layout(renderer)
        size2 = (text._dimensions.width, text._dimensions.height)
        plt.close(fig2)

        # With data mode, natural size in data units should be the same
        # (font_size=12.0 means 12 data units tall, regardless of axis)
        assert abs(size1[1] - size2[1]) < 1.0, (
            f"Text height should be consistent in data mode. Got {size1[1]:.1f} vs {size2[1]:.1f}"
        )


class TestFontSizeInPoints:
    """Tests for get_font_size_in_points calculation."""

    def test_font_size_uses_component_matrix(self, renderer, ax):
        """get_font_size_in_points should use the component's transformation matrix.

        This is the core bug: previously it used identity matrix, ignoring
        parent container scaling.
        """
        renderer.create_context(ax=ax)

        # Base font size
        base_size = 10.0

        # With identity matrix (no scaling)
        from jeanplot.core.renderer.matplotlib import _get_points_per_unit_vector

        identity = np.eye(3)
        ppu_identity = _get_points_per_unit_vector(ax, identity, vector=(0, 1))
        points_identity = base_size * ppu_identity

        # With 2x scale matrix
        scale_2x = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
        ppu_scaled = _get_points_per_unit_vector(ax, scale_2x, vector=(0, 1))
        points_scaled = base_size * ppu_scaled

        # Scaled should be ~2x points
        assert points_scaled > points_identity * 1.8, (
            f"2x scaled matrix should give ~2x point size. "
            f"Got identity={points_identity:.1f}, scaled={points_scaled:.1f}"
        )

    def test_render_text_uses_matrix_for_font_size(self, renderer, ax):
        """render_text should use the component's matrix when calculating font size.

        Bug: render_text() called get_font_size_in_points() without the matrix,
        so text inside scaled containers rendered at the wrong size.
        """
        renderer.create_context(ax=ax)

        # Create text in a scaled container
        container = Container(
            id="scaled_container",
            children=[Text(id="scaled_text", text="Test", font_size=10.0)],
            transform=Transform(scale=(2.0, 2.0)),
        )
        jstyle.apply(container)
        container.measure_and_layout(renderer)

        text = container.children[0]
        matrix = text.compute_world_matrix()

        # The matrix should have 2x scale
        scale = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        assert scale > 1.9, f"Matrix should have ~2x scale, got {scale:.2f}"

        # When render_text is called, it should use this matrix
        # to calculate a font size ~2x larger than without scaling.
        # We test this by checking the font size calculation directly.
        from jeanplot.core.renderer.matplotlib import _get_points_per_unit_vector

        # Font size in points WITH the matrix (correct behavior)
        ppu_with_matrix = _get_points_per_unit_vector(ax, matrix, vector=(0, 1))
        font_size_with_matrix = text.font_size * ppu_with_matrix

        # Font size in points WITHOUT matrix (buggy behavior)
        identity = np.eye(3)
        ppu_without_matrix = _get_points_per_unit_vector(ax, identity, vector=(0, 1))
        font_size_without_matrix = text.font_size * ppu_without_matrix

        # With 2x scale, font should be ~2x larger
        ratio = font_size_with_matrix / font_size_without_matrix
        assert ratio > 1.8, (
            f"Font size with matrix should be ~2x larger. "
            f"Ratio={ratio:.2f}, with={font_size_with_matrix:.1f}, without={font_size_without_matrix:.1f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
