"""Tests to diagnose text font size refresh bug.

The issue: linewidths have a refresh callback but text font sizes don't.
After zoom/resize, linewidths update but text stays at original point size,
breaking the data-unit proportions.
"""

import pytest
import matplotlib.pyplot as plt

from jeanplot import Container, Text, jstyle
from jeanplot.core.models import BoxStyle, Size
from jeanplot.core.renderer.matplotlib import MatplotlibRenderer, _get_points_per_unit_vector


class TestTextRefreshBug:
    """Diagnose whether text font sizes are refreshed on axis changes."""

    @pytest.fixture
    def cleanup(self):
        yield
        plt.close("all")

    def test_text_vs_linewidth_after_axis_change(self, cleanup):
        """Compare text and linewidth behavior after axis limits change."""
        renderer = MatplotlibRenderer()

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")

        # Create a box with text and a connection with linewidth
        # Both should scale proportionally with axis changes
        text = Text(id="test_text", text="Test", font_size=10.0, font_size_mode="data")
        box = Container(
            id="box",
            children=[text],
            min_dimensions=Size(50, 30),
            style=BoxStyle(border_width=2.0, border_color="#000000"),  # 2 data units border
        )
        jstyle.apply(box)

        renderer.create_context(ax=ax)
        box.measure_and_layout(renderer)
        renderer.render_component(ax, box, adjust_lims=False)

        # Get initial font size in points and border width
        text_artists = [c for c in ax.get_children() if hasattr(c, 'get_fontsize')]
        initial_font_size = text_artists[0].get_fontsize() if text_artists else None

        # Get initial linewidth from tracked patches
        initial_linewidths = [p.get_linewidth() for p, _ in renderer._data_width_patches]

        # Now change axis limits (simulate zoom out)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)

        # Trigger a redraw
        fig.canvas.draw()

        # Check linewidths after refresh
        after_linewidths = [p.get_linewidth() for p, _ in renderer._data_width_patches]

        # Check text font sizes after redraw
        after_font_size = text_artists[0].get_fontsize() if text_artists else None

        assert initial_linewidths and after_linewidths
        assert initial_font_size is not None and after_font_size is not None

        linewidth_ratio = after_linewidths[0] / initial_linewidths[0]
        font_ratio = after_font_size / initial_font_size

        # With 2x axis range, data-unit elements should appear at half the pixel size.
        assert 0.35 < linewidth_ratio < 0.65, f"linewidth ratio out of range: {linewidth_ratio:.3f}"
        assert 0.35 < font_ratio < 0.65, f"font ratio out of range: {font_ratio:.3f}"

    def test_text_point_size_consistency_at_render_time(self, cleanup):
        """Verify text is rendered with correct point size at initial render."""
        renderer = MatplotlibRenderer()

        # Create two scenarios with different axis limits but same data-unit font size
        results = []

        for axis_range in [100, 200]:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            ax.set_xlim(0, axis_range)
            ax.set_ylim(0, axis_range)
            ax.set_aspect("equal")

            text = Text(id="test_text", text="Test", font_size=10.0, font_size_mode="data")
            box = Container(id="box", children=[text], min_dimensions=Size(50, 30))
            jstyle.apply(box)

            renderer.create_context(ax=ax)
            box.measure_and_layout(renderer)
            renderer.render_component(ax, box, adjust_lims=False)

            text_artists = [c for c in ax.get_children() if hasattr(c, 'get_fontsize')]
            font_size_pts = text_artists[0].get_fontsize() if text_artists else 0

            # Calculate expected: font_size (data units) * points_per_data_unit
            matrix = text.compute_world_matrix()
            ppu = _get_points_per_unit_vector(ax, matrix, vector=(0, 1))
            expected_pts = 10.0 * ppu

            results.append({
                'axis_range': axis_range,
                'font_size_pts': font_size_pts,
                'expected_pts': expected_pts,
                'ppu': ppu,
            })

            plt.close(fig)

        # Font sizes should be 2x different for 2x different axis ranges
        ratio = results[0]['font_size_pts'] / results[1]['font_size_pts']

        # At render time, font sizes should be correct
        assert 1.8 < ratio < 2.2, f"Initial render font sizes incorrect: ratio={ratio:.3f}"
