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
            style=BoxStyle(border_width=2.0),  # 2 data units border
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

        print("\nInitial axis limits: (0, 100)")
        print("After axis limits: (0, 200)")
        print(f"Initial font size: {initial_font_size:.2f} points")
        print(f"After font size: {after_font_size:.2f} points")
        print(f"Initial linewidths: {initial_linewidths}")
        print(f"After linewidths: {after_linewidths}")

        # With 2x axis range, data-unit elements should appear at half the pixel size
        # Linewidths should be ~half (refresh_linewidths updates them)
        # But text font sizes likely stayed the same (BUG!)

        if initial_linewidths and after_linewidths:
            linewidth_ratio = after_linewidths[0] / initial_linewidths[0]
            print(f"Linewidth ratio (should be ~0.5): {linewidth_ratio:.3f}")

        if initial_font_size and after_font_size:
            font_ratio = after_font_size / initial_font_size
            print(f"Font size ratio (should be ~0.5 if refreshed): {font_ratio:.3f}")

            # This is the bug: font size ratio is 1.0 (not refreshed)
            # while linewidth ratio is ~0.5 (correctly refreshed)
            if abs(font_ratio - 1.0) < 0.1 and linewidth_ratio < 0.6:
                pytest.fail(
                    f"BUG CONFIRMED: Text font size not refreshed on axis change! "
                    f"Font ratio={font_ratio:.3f} (expected ~0.5), "
                    f"Linewidth ratio={linewidth_ratio:.3f}"
                )

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

        print("\nAt-render-time font sizes:")
        for r in results:
            print(f"  axis_range={r['axis_range']}: font={r['font_size_pts']:.2f} pts, "
                  f"expected={r['expected_pts']:.2f} pts, ppu={r['ppu']:.2f}")

        # Font sizes should be 2x different for 2x different axis ranges
        ratio = results[0]['font_size_pts'] / results[1]['font_size_pts']
        print(f"Ratio (should be ~2.0): {ratio:.3f}")

        # At render time, font sizes should be correct
        assert 1.8 < ratio < 2.2, f"Initial render font sizes incorrect: ratio={ratio:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
