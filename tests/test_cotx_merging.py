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


def test_cotx_merging(tmp_path):
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

        assert len(networks) == 1

        # Use the first network for testing
        network = networks[0]

        # Create figure and render
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

        render_circuit_to_ax(network, ax)

        # Check TU information and cotx grouping from generated network
        from biocomptools.toollib.figuremakers.network_adapter import get_tu_informations
        tu_infos = get_tu_informations(network)
        expected_tu_ids = {
            "x1_marker_TNFa",
            "x1_a+_TNFa",
            "x2_marker_TGF-β",
            "x2_a+_TGF-β",
            "x3_marker_IFNab",
            "x3_a-_IFNab",
            "ba_marker_Bb",
            "b_a-_Bb",
        }
        assert set(tu_infos.keys()) == expected_tu_ids
        assert sum(1 for info in tu_infos.values() if info.is_marker) == 4

        # Save artifact in tmp_path for debugging if needed
        fig.savefig(tmp_path / "test_cotx_merging_output.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Assert we have 4 cotransfections
        assert len(recipe.content) == 4, f"Expected 4 cotransfections, got {len(recipe.content)}"

        # Check names are preserved
        expected_names = {"TNFa", "TGF-β", "IFNab", "Bb"}
        actual_names = [cotx.name for cotx in recipe.content]
        assert set(actual_names) == expected_names, f"Expected names {sorted(expected_names)}, got {actual_names}"
