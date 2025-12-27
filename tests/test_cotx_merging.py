"""Test case for cotransfection merging issue"""
import pytest
import matplotlib.pyplot as plt

pytest.importorskip("biocomptools", reason="biocomptools required for this test")

from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit
from biocomp.network import recipe_to_networks
from biocomp.library import load_lib, LibraryContext
from biocomptools.toollib.figuremakers.geneticcircuit import render_circuit_to_ax

# Test data
P = "hEF1a"
T = "L0.T_4560"
ERNS = ['CasE', 'Csy4', 'PgU']

COLORS = {
    'x1': 'mKO2',
    'x2': 'eBFP2',
    'x3': 'iRFP720',
    'b': 'mMaroon1',
    'b1': 'mMaroon1',
    'b2': 'eYFP',
    'y': 'mNeonGreen',
}


def test_cotx_merging():
    """Test that all 4 cotransfections are rendered correctly"""
    erns = ERNS
    recs = [f"{ern}_rec" for ern in erns]

    lib = load_lib()

    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_cotx",
            content=[
                CoTransfection(
                    name="TNFa",
                    units=[
                        TranscriptionUnit(slots=[P, COLORS['x1'], T], name="x1_marker"),
                        TranscriptionUnit(slots=[P, recs[0], COLORS['y'], T], name="x1_a+"),
                    ],
                ),
                CoTransfection(
                    name="TGF-β",
                    units=[
                        TranscriptionUnit(slots=[P, COLORS['x2'], T], name="x2_marker"),
                        TranscriptionUnit(slots=[P, recs[0], COLORS['y'], T], name="x2_a+"),
                    ],
                ),
                CoTransfection(
                    name="IFNab",
                    units=[
                        TranscriptionUnit(slots=[P, COLORS['x3'], T], name="x3_marker"),
                        TranscriptionUnit(slots=[P, erns[0], T], name="x3_a-"),
                    ],
                ),
                CoTransfection(
                    name="Bb",
                    units=[
                        TranscriptionUnit(slots=[P, COLORS['b'], T], name="ba_marker"),
                        TranscriptionUnit(slots=[P, erns[0], T], name="b_a-"),
                    ],
                ),
            ],
        )

        # Build networks from recipe
        networks = recipe_to_networks(recipe, invert=True)

        # Debug: Print networks
        print(f"Number of networks generated: {len(networks)}")
        for i, net in enumerate(networks):
            print(f"  Network {i+1}: {net.name}")

        # Use the first network for testing
        network = networks[0]

        # Create figure and render
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

        render_circuit_to_ax(network, ax)

        # Check TU information
        from biocomptools.toollib.figuremakers.network_adapter import get_tu_informations
        tu_infos = get_tu_informations(network)
        print("\nTU Information (checking cotx_name):")
        for tu_id, tu_info in sorted(tu_infos.items()):
            print(f"  TU {tu_id}:")
            print(f"    cotx_name: {tu_info.cotx_name}")
            print(f"    is_marker: {tu_info.is_marker}")
            print(f"    cotx_marker: {tu_info.cotx_marker}")

        # Save the figure for inspection
        fig.savefig('test_cotx_merging_output.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Check the recipe content
        print(f"\nRecipe cotx count: {len(recipe.content)}")
        for i, cotx in enumerate(recipe.content):
            print(f"  CoTx {i+1}: {cotx.name}")
            print(f"    Units: {len(cotx.units)}")

        # Assert we have 4 cotransfections
        assert len(recipe.content) == 4, f"Expected 4 cotransfections, got {len(recipe.content)}"

        # Check names are preserved
        expected_names = ["TNFa", "TGF-β", "IFNab", "Bb"]
        actual_names = [cotx.name for cotx in recipe.content]
        assert set(actual_names) == set(expected_names), f"Expected names {expected_names}, got {actual_names}"


if __name__ == "__main__":
    test_cotx_merging()
    print("Test completed - check test_cotx_merging_output.png")
