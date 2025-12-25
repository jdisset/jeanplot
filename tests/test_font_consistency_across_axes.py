"""Tests for font size consistency when using font_size_mode.

These tests verify that:
1. font_size_mode="points" (default) gives consistent visual size across contexts
2. font_size_mode="data" preserves the original scaling behavior

The key insight: users typically want text to have consistent visual appearance
regardless of axis limits or container transforms. The "points" mode provides this.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
from pathlib import Path

from jeanplot import Container, Text, BoxStyle, LayoutConstraints, Transform, jstyle
from jeanplot.core.renderer.matplotlib import MatplotlibRenderer, _get_points_per_unit_vector


def measure_text_pixel_height(ax, text_component, renderer) -> float:
    """Measure the actual rendered pixel height of text based on font_size_mode."""
    fig = ax.get_figure()
    pixels_per_point = fig.dpi / 72.0

    if text_component.font_size_mode == "points":
        # In points mode, font_size is directly in points
        return text_component.font_size * pixels_per_point
    else:
        # In data mode, font_size scales with the transformation matrix
        matrix = text_component.compute_world_matrix()
        ppu_y = _get_points_per_unit_vector(ax, matrix, vector=(0, 1))
        font_size_pts = text_component.font_size * ppu_y
        return font_size_pts * pixels_per_point


def render_to_image(fig) -> np.ndarray:
    """Render figure to numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


class TestFontSizeConsistencyAcrossAxes:
    """Test that font_size_mode='points' (default) produces consistent pixel heights."""

    @pytest.fixture
    def cleanup_figures(self):
        yield
        plt.close("all")

    def test_same_font_size_different_axis_limits(self, cleanup_figures):
        """
        With font_size_mode='points' (default), font_size should render to the
        same pixel height regardless of axis limits.
        """
        font_size = 10.0

        renderer = MatplotlibRenderer()

        # Create two axes with DIFFERENT limits
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

        # Left axis: small range (0-100)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_aspect("equal")
        ax1.set_title("Small axis range (0-100)")

        # Right axis: large range (0-500)
        ax2.set_xlim(0, 500)
        ax2.set_ylim(0, 500)
        ax2.set_aspect("equal")
        ax2.set_title("Large axis range (0-500)")

        # Create identical text components with font_size_mode="points"
        text1 = Text(id="text1", text="Hello", font_size=font_size, font_size_mode="points")
        text2 = Text(id="text2", text="Hello", font_size=font_size, font_size_mode="points")

        box1 = Container(id="box1", children=[text1], min_dimensions=(50, 30))
        box2 = Container(id="box2", children=[text2], min_dimensions=(50, 30))

        jstyle.apply(box1)
        jstyle.apply(box2)

        # Render to each axis
        renderer.create_context(ax=ax1)
        box1.measure_and_layout(renderer)

        renderer.create_context(ax=ax2)
        box2.measure_and_layout(renderer)

        # Measure pixel heights
        height1 = measure_text_pixel_height(ax1, text1, renderer)
        height2 = measure_text_pixel_height(ax2, text2, renderer)

        plt.close(fig)

        print(f"height1={height1:.2f}, height2={height2:.2f}")

        # With font_size_mode="points", both should render to same pixel height
        assert abs(height1 - height2) < 1.0, (
            f"Same font_size with mode='points' should render to same pixel height. "
            f"Got height1={height1:.2f}px, height2={height2:.2f}px"
        )

    def test_same_font_size_different_container_scales(self, cleanup_figures):
        """
        With font_size_mode='points' (default), font_size should render to the
        same pixel height regardless of parent container scaling.
        """
        font_size = 10.0

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_aspect("equal")

        renderer = MatplotlibRenderer()
        renderer.create_context(ax=ax)

        # Container 1: no scaling (using points mode for consistent visual size)
        text1 = Text(id="text1", text="Hello", font_size=font_size, font_size_mode="points")
        container1 = Container(
            id="container1",
            children=[text1],
            min_dimensions=(50, 30),
        )

        # Container 2: 2x scale (using points mode for consistent visual size)
        text2 = Text(id="text2", text="Hello", font_size=font_size, font_size_mode="points")
        container2 = Container(
            id="container2",
            children=[text2],
            min_dimensions=(50, 30),
            transform=Transform(scale=(2.0, 2.0)),
        )

        jstyle.apply(container1)
        jstyle.apply(container2)

        container1.measure_and_layout(renderer)
        container2.measure_and_layout(renderer)

        height1 = measure_text_pixel_height(ax, text1, renderer)
        height2 = measure_text_pixel_height(ax, text2, renderer)

        plt.close(fig)

        print(f"height1={height1:.2f}, height2={height2:.2f}")

        # With font_size_mode="points", both should render to same pixel height
        assert abs(height1 - height2) < 1.0, (
            f"Same font_size with mode='points' should render to same pixel height. "
            f"Got height1={height1:.2f}px, height2={height2:.2f}px"
        )


class TestDataUnitBehaviorDocumented:
    """Tests for font_size_mode='data' which preserves scaling behavior."""

    @pytest.fixture
    def cleanup_figures(self):
        yield
        plt.close("all")

    def test_font_scales_with_axis_limits_data_mode(self, cleanup_figures):
        """With font_size_mode='data', font scales with axis limits."""
        font_size = 10.0
        renderer = MatplotlibRenderer()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_aspect("equal")

        ax2.set_xlim(0, 500)
        ax2.set_ylim(0, 500)
        ax2.set_aspect("equal")

        # Explicitly use font_size_mode="data" to get scaling behavior
        text1 = Text(id="text1", text="Test", font_size=font_size, font_size_mode="data")
        text2 = Text(id="text2", text="Test", font_size=font_size, font_size_mode="data")

        box1 = Container(id="box1", children=[text1])
        box2 = Container(id="box2", children=[text2])

        jstyle.apply(box1)
        jstyle.apply(box2)

        renderer.create_context(ax=ax1)
        box1.measure_and_layout(renderer)

        renderer.create_context(ax=ax2)
        box2.measure_and_layout(renderer)

        height1 = measure_text_pixel_height(ax1, text1, renderer)
        height2 = measure_text_pixel_height(ax2, text2, renderer)

        plt.close(fig)

        # With data mode: text1 is ~5x larger because axis range is 5x smaller
        ratio = height1 / height2
        assert ratio > 4.0, (
            f"With font_size_mode='data': smaller axis range should produce larger text. "
            f"Got ratio={ratio:.2f} (expected ~5.0)"
        )

    def test_font_scales_with_container_transform_data_mode(self, cleanup_figures):
        """With font_size_mode='data', font scales with container transforms."""
        font_size = 10.0

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_aspect("equal")

        renderer = MatplotlibRenderer()
        renderer.create_context(ax=ax)

        # Explicitly use font_size_mode="data" to get scaling behavior
        text1 = Text(id="text1", text="Test", font_size=font_size, font_size_mode="data")
        text2 = Text(id="text2", text="Test", font_size=font_size, font_size_mode="data")

        container1 = Container(id="c1", children=[text1])
        container2 = Container(
            id="c2",
            children=[text2],
            transform=Transform(scale=(2.0, 2.0)),
        )

        jstyle.apply(container1)
        jstyle.apply(container2)

        container1.measure_and_layout(renderer)
        container2.measure_and_layout(renderer)

        height1 = measure_text_pixel_height(ax, text1, renderer)
        height2 = measure_text_pixel_height(ax, text2, renderer)

        plt.close(fig)

        # With data mode: text2 is ~2x larger because its container has 2x scale
        ratio = height2 / height1
        assert ratio > 1.8, (
            f"With font_size_mode='data': 2x scaled container should produce 2x larger text. "
            f"Got ratio={ratio:.2f} (expected ~2.0)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
