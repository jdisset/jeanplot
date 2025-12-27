"""Test text sizes in actual NetworkDiagram components.

The user reports:
- simple_single_reporter: Tx/Tl labels are about the same height as node diameter (BAD)
- multi_cotx_aggregation: Tx/Tl labels are tiny and hard to read (BAD)

This test investigates the actual text sizes in network diagrams.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from jeanplot import jstyle
from jeanplot.core.renderer.matplotlib import MatplotlibRenderer, _get_points_per_unit_vector

pytest.importorskip("biocomptools", reason="biocomptools required for this test")


def get_test_network(name: str):
    """Load a test network by name."""
    from dracon import load
    import biocomptools.toollib.common as cm

    recipes_path = Path(__file__).parent.parent.parent / "biocompiler" / "biocomp-jobs" / "examples" / "declarative_recipes.yaml"
    if not recipes_path.exists():
        recipes_path = Path("/Users/jeandisset/Code/Weiss/biocompiler/biocomp-jobs/examples/declarative_recipes.yaml")

    recipes = load(recipes_path)['recipes']

    for recipe in recipes:
        if recipe.name == name:
            from biocomp.network import recipe_to_networks
            from biocomp.library import LibraryContext, PartsLibrary

            if LibraryContext.get_library() is None:
                LibraryContext.set_library(PartsLibrary.from_file(cm.config.paths.parts_library))

            networks = recipe_to_networks(recipe, invert=True)
            return networks[0] if networks else None

    return None


class TestNetworkDiagramText:
    """Test text rendering in actual network diagrams."""

    @pytest.fixture
    def cleanup(self):
        yield
        plt.close("all")

    def measure_diagram_text(self, network, name: str):
        """Measure text sizes in a network diagram."""
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram, TranscriptionNode
        from jeanplot import Container, LayoutConstraints

        from jeanplot import load_default_theme
        load_default_theme(force=True)

        diagram = NetworkDiagram(network=network, simplified=True)
        root = Container(
            children=[diagram],
            layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
        )
        jstyle.apply(root)

        # Find actual TranscriptionNode to get real node bounds
        tx_node = None
        def find_tx(comp):
            nonlocal tx_node
            if isinstance(comp, TranscriptionNode):
                tx_node = comp
                return
            if hasattr(comp, 'children'):
                for c in comp.children:
                    find_tx(c)
        find_tx(root)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.set_aspect("equal")
        ax.axis("off")

        renderer = MatplotlibRenderer()
        renderer.create_context(ax=ax)
        renderer.render_component(ax, root, adjust_lims=True)

        fig.canvas.draw()

        # Get axis info
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        data_width = xlim[1] - xlim[0]
        data_height = ylim[1] - ylim[0]
        bbox = ax.get_window_extent()
        ppu = bbox.height / data_height if data_height > 0 else 0

        # Get actual node pixel height from world bounds
        node_size_data = 18
        if tx_node:
            bounds = tx_node.get_world_bounds()
            if bounds:
                node_world_height = bounds[3] - bounds[1]
                node_size_pixels = node_world_height * ppu
            else:
                node_size_pixels = node_size_data * ppu
        else:
            node_size_pixels = node_size_data * ppu

        # Find text artists with "Tx" or "Tl" labels
        tx_tl_artists = []
        for child in ax.get_children():
            if hasattr(child, 'get_text') and hasattr(child, 'get_fontsize'):
                text = child.get_text()
                if text in ["Tx", "Tl"]:
                    tx_tl_artists.append({
                        'text': text,
                        'fontsize': child.get_fontsize(),
                    })

        font_size_data = 7  # from theme

        results = {
            'name': name,
            'data_extent': (data_width, data_height),
            'ppu': ppu,
            'node_size_data': node_size_data,
            'node_size_pixels': node_size_pixels,
            'font_size_data': font_size_data,
            'text_artists': tx_tl_artists,
        }

        if tx_tl_artists:
            # Calculate actual text pixels from font points
            avg_fontsize = np.mean([t['fontsize'] for t in tx_tl_artists])
            text_pixels = avg_fontsize * (fig.dpi / 72.0)
            results['avg_text_points'] = avg_fontsize
            results['avg_text_pixels'] = text_pixels
            results['text_to_node_ratio'] = text_pixels / node_size_pixels if node_size_pixels > 0 else 0
            results['expected_ratio'] = font_size_data / node_size_data  # 7/18 = 0.389

        plt.close(fig)
        return results

    def test_simple_single_reporter(self, cleanup):
        """Test text sizes in simple_single_reporter diagram."""
        network = get_test_network("simple_single_reporter")
        if network is None:
            pytest.skip("Could not load simple_single_reporter network")

        results = self.measure_diagram_text(network, "simple_single_reporter")

        print("\n" + "="*70)
        print(f"DIAGRAM: {results['name']}")
        print("="*70)
        print(f"Data extent: {results['data_extent'][0]:.1f} x {results['data_extent'][1]:.1f}")
        print(f"Pixels per data unit: {results['ppu']:.2f}")
        print(f"Node size: {results['node_size_data']} data units = {results['node_size_pixels']:.1f} pixels")
        print(f"Text artists found: {len(results['text_artists'])}")
        for t in results['text_artists']:
            print(f"  - '{t['text']}': {t['fontsize']:.2f} points")

        if 'avg_text_points' in results:
            print(f"Average text size: {results['avg_text_points']:.2f} points = {results['avg_text_pixels']:.1f} pixels")
            print(f"Text/Node ratio: {results['text_to_node_ratio']:.4f} (expected: {results['expected_ratio']:.4f})")

            deviation = abs(results['text_to_node_ratio'] - results['expected_ratio']) / results['expected_ratio'] * 100
            print(f"Deviation: {deviation:.1f}%")

            if deviation > 20:
                pytest.fail(f"Text/node ratio off by {deviation:.1f}% (expected ~{results['expected_ratio']:.3f}, got {results['text_to_node_ratio']:.3f})")

    def test_multi_networks(self, cleanup):
        """Test text sizes across multiple network diagrams."""
        test_names = [
            "simple_single_reporter",
            "simple_two_reporters",
            "simple_single_ern",
            "two_reporters_with_ern",
            "multi_cotx_aggregation",
        ]

        results = []
        for name in test_names:
            network = get_test_network(name)
            if network is None:
                print(f"Skipping {name} - could not load")
                continue

            result = self.measure_diagram_text(network, name)
            results.append(result)

        print("\n" + "="*80)
        print("MULTI-NETWORK TEXT SIZE COMPARISON")
        print("="*80)
        print(f"{'Network':<30} | {'Extent':>15} | {'PPU':>6} | {'Text px':>8} | {'Node px':>8} | {'Ratio':>7} | Status")
        print("-"*80)

        expected_ratio = 7 / 18  # 0.389

        for r in results:
            if 'text_to_node_ratio' in r:
                deviation = abs(r['text_to_node_ratio'] - expected_ratio) / expected_ratio * 100
                status = "OK" if deviation < 20 else f"FAIL ({deviation:.0f}% off)"
                print(f"{r['name']:<30} | {r['data_extent'][0]:>6.0f}x{r['data_extent'][1]:<6.0f} | "
                      f"{r['ppu']:>6.2f} | {r['avg_text_pixels']:>8.1f} | {r['node_size_pixels']:>8.1f} | "
                      f"{r['text_to_node_ratio']:>7.4f} | {status}")

        print("="*80)

        # Check for consistency
        ratios = [r['text_to_node_ratio'] for r in results if 'text_to_node_ratio' in r]
        if ratios:
            variance = np.var(ratios)
            mean = np.mean(ratios)
            cv = np.sqrt(variance) / mean if mean > 0 else 0
            print(f"\nCoefficient of variation: {cv:.3f}")

            if cv > 0.10:
                pytest.fail(f"Text/node ratios vary too much across diagrams! CV={cv:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
