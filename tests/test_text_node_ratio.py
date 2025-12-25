"""Test that text-to-node ratios are consistent across diagram complexities.

The core invariant: if text has font_size=7 and node has size=18 (both in data units),
then the rendered pixel ratio should be 7/18 ≈ 0.389 regardless of diagram complexity.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from jeanplot import Container, Text, jstyle
from jeanplot.core.models import LayoutConstraints, BoxStyle, Size
from jeanplot.core.renderer.matplotlib import MatplotlibRenderer, _get_points_per_unit_vector


class TestTextNodeRatio:
    """Test text-to-node proportions across diagram complexities."""

    @pytest.fixture
    def cleanup(self):
        yield
        plt.close("all")

    def create_diagram(self, num_nodes: int, gap: float):
        """Create a diagram with configurable complexity."""
        node_size = 18  # data units
        font_size = 7   # data units

        nodes = []
        for i in range(num_nodes):
            text = Text(
                id=f"label_{i}",
                text="Tx",
                font_size=font_size,
                font_size_mode="data",
                align="center",
                vertical_align="middle",
            )
            node = Container(
                id=f"node_{i}",
                children=[text],
                min_dimensions=Size(node_size, node_size),
                style=BoxStyle(background_color="#555", corner_radius=1e6),
                layout=LayoutConstraints(align_items="center", justify_content="center"),
            )
            nodes.append(node)

        root = Container(
            id="root",
            children=nodes,
            layout=LayoutConstraints(direction="row", gap=gap),
            style=BoxStyle(padding=(20, 20, 20, 20)),
        )
        return root, node_size, font_size

    def get_text_and_node_sizes(self, ax, renderer, root):
        """Extract actual rendered text point sizes and node pixel sizes."""
        # Get text artists we created (filter out axis labels etc)
        text_artists = []
        for child in ax.get_children():
            if hasattr(child, 'get_fontsize') and hasattr(child, 'get_text'):
                text = child.get_text()
                if text == "Tx":  # Our text
                    text_artists.append(child)

        text_point_sizes = [t.get_fontsize() for t in text_artists]

        # Calculate node pixel size from data units
        # Nodes are 18 data units, need to convert to pixels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bbox = ax.get_window_extent()

        # Pixels per data unit
        data_height = ylim[1] - ylim[0]
        ppu = bbox.height / data_height if data_height > 0 else 0

        return text_point_sizes, ppu

    def test_ratio_consistency_across_complexities(self, cleanup):
        """Text/node ratio should be constant regardless of diagram size."""
        renderer = MatplotlibRenderer()

        # Different diagram complexities
        configs = [
            ("simple", 1, 0),      # 1 node, no gap
            ("medium", 3, 50),     # 3 nodes, medium gap
            ("complex", 5, 100),   # 5 nodes, large gap
            ("very_complex", 8, 150),  # 8 nodes, very large gap
        ]

        node_size = 18
        font_size = 7
        expected_ratio = font_size / node_size  # 0.389

        results = []

        for name, num_nodes, gap in configs:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            ax.set_aspect("equal")
            ax.axis("off")

            root, _, _ = self.create_diagram(num_nodes, gap)
            jstyle.apply(root)

            renderer.create_context(ax=ax)
            renderer.render_component(ax, root, adjust_lims=True)

            # Force draw
            fig.canvas.draw()

            text_sizes, ppu = self.get_text_and_node_sizes(ax, renderer, root)

            # Text font size in data units should give us font_size * ppu_y points
            # But we need the world matrix to calculate correctly
            first_node = root.children[0]
            first_text = first_node.children[0]
            matrix = first_text.compute_world_matrix()
            ppu_y = _get_points_per_unit_vector(ax, matrix, vector=(0, 1))

            # Node size in pixels
            node_pixels = node_size * ppu

            # Text size in pixels (convert from points)
            # points -> pixels: multiply by dpi/72
            if text_sizes:
                text_points = text_sizes[0]
                text_pixels = text_points * (fig.dpi / 72.0)

                # Actual ratio
                actual_ratio = text_pixels / node_pixels if node_pixels > 0 else 0

                results.append({
                    'name': name,
                    'num_nodes': num_nodes,
                    'gap': gap,
                    'data_extent': (ax.get_xlim()[1] - ax.get_xlim()[0],
                                   ax.get_ylim()[1] - ax.get_ylim()[0]),
                    'ppu': ppu,
                    'ppu_y': ppu_y,
                    'text_points': text_points,
                    'text_pixels': text_pixels,
                    'node_pixels': node_pixels,
                    'expected_ratio': expected_ratio,
                    'actual_ratio': actual_ratio,
                })

            plt.close(fig)

        # Print results
        print("\n" + "="*80)
        print("TEXT-TO-NODE RATIO TEST RESULTS")
        print("="*80)
        print(f"Expected ratio: {expected_ratio:.4f} (font_size={font_size} / node_size={node_size})")
        print()

        for r in results:
            deviation = abs(r['actual_ratio'] - expected_ratio) / expected_ratio * 100
            status = "OK" if deviation < 10 else "FAIL"
            print(f"{r['name']:15} | nodes={r['num_nodes']} gap={r['gap']:3} | "
                  f"extent={r['data_extent'][0]:6.1f}x{r['data_extent'][1]:5.1f} | "
                  f"ppu={r['ppu']:5.2f} | "
                  f"text={r['text_pixels']:6.1f}px node={r['node_pixels']:6.1f}px | "
                  f"ratio={r['actual_ratio']:.4f} ({deviation:5.1f}% off) [{status}]")

        print("="*80)

        # Check all ratios are within 10% of expected
        ratios = [r['actual_ratio'] for r in results]
        deviations = [abs(r - expected_ratio) / expected_ratio for r in ratios]
        max_deviation = max(deviations)

        if max_deviation > 0.10:
            pytest.fail(
                f"Text/node ratios are inconsistent! Max deviation: {max_deviation*100:.1f}%\n"
                f"Expected: {expected_ratio:.4f}, Got: {ratios}"
            )

    def test_text_size_scales_with_ppu(self, cleanup):
        """Text point size should scale correctly with points-per-data-unit."""
        renderer = MatplotlibRenderer()

        font_size = 7  # data units

        configs = [
            (100, 100),   # Small axis range
            (500, 500),   # Medium range
            (1000, 1000), # Large range
        ]

        results = []

        for xlim_max, ylim_max in configs:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            ax.set_xlim(0, xlim_max)
            ax.set_ylim(0, ylim_max)
            ax.set_aspect("equal")

            text = Text(id="test", text="Tx", font_size=font_size, font_size_mode="data",
                       align="center", vertical_align="middle")
            container = Container(
                id="container",
                children=[text],
                min_dimensions=Size(50, 30),
            )
            jstyle.apply(container)

            renderer.create_context(ax=ax)
            renderer.render_component(ax, container, adjust_lims=False)

            fig.canvas.draw()

            # Get actual text size
            matrix = text.compute_world_matrix()
            ppu_y = _get_points_per_unit_vector(ax, matrix, vector=(0, 1))
            expected_points = font_size * ppu_y

            text_artists = [c for c in ax.get_children()
                          if hasattr(c, 'get_text') and c.get_text() == "Tx"]
            actual_points = text_artists[0].get_fontsize() if text_artists else 0

            results.append({
                'xlim': xlim_max,
                'ppu_y': ppu_y,
                'expected': expected_points,
                'actual': actual_points,
            })

            plt.close(fig)

        print("\n" + "="*60)
        print("TEXT SIZE VS PPU TEST")
        print("="*60)
        for r in results:
            diff = abs(r['actual'] - r['expected'])
            print(f"xlim={r['xlim']:4} | ppu={r['ppu_y']:.3f} | "
                  f"expected={r['expected']:.2f} actual={r['actual']:.2f} | diff={diff:.4f}")

        # All text sizes should match expected
        for r in results:
            assert abs(r['actual'] - r['expected']) < 0.1, (
                f"Text size mismatch: expected {r['expected']:.2f}, got {r['actual']:.2f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
