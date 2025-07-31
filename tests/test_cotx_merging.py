"""Test case for cotransfection merging issue"""
import matplotlib.pyplot as plt
from biocomp.network import Network, CoTransfection, Unit
from biocomp.utils import load_lib
from jeanplot.biocomp_diagrams import render_circuit_schematic
import biocomptools.toollib.models as md

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

    n2 = Network(
        cotx=[
            CoTransfection(
                name="TNFa",
                units=[
                    Unit(slots=[P, COLORS['x1'], T], name="x1_marker"),
                    Unit(slots=[P, recs[0], COLORS['y'], T], name="x1_a+"),
                ],
            ),
            CoTransfection(
                name="TGF-β",
                units=[
                    Unit(slots=[P, COLORS['x2'], T], name="x2_marker"),
                    Unit(slots=[P, recs[0], COLORS['y'], T], name="x2_a+"),
                ],
            ),
            CoTransfection(
                name="IFNab",
                units=[
                    Unit(slots=[P, COLORS['x3'], T], name="x3_marker"),
                    Unit(slots=[P, erns[0], T], name="x3_a-"),
                ],
            ),
            CoTransfection(
                name="Bb",
                units=[
                    Unit(slots=[P, COLORS['b'], T], name="ba_marker"),
                    Unit(slots=[P, erns[0], T], name="b_a-"),
                ],
            ),
        ],
        invert_on_build=True,
    )
    
    # Debug: Print original cotransfections
    print(f"Original number of cotransfections: {len(n2.cotx)}")
    for i, cotx in enumerate(n2.cotx):
        print(f"  CoTx {i+1}: {cotx.name}")
        for unit in cotx.units:
            print(f"    Unit: {unit.name} - Slots: {unit.slots}")
    
    # Convert to biocomp_diagrams Network
    mnet = md.Network.from_network(n2)
    
    # Debug: Print converted network
    print(f"\nConverted network type: {type(mnet)}")
    print(f"Available attributes: {[attr for attr in dir(mnet) if not attr.startswith('_')]}")
    
    # Create library first
    lib = load_lib()  # Use biocomp library loader
    
    # Try to build the network to see what happens
    mnet.build(lib)
    
    # Create figure and render
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    # Enable debug mode to see what's happening
    from jeanplot import set_debug
    set_debug(False)  # Turn off debug for cleaner output
    
    render_circuit_schematic(mnet, lib, ax)
    
    # Check if cotx names were propagated to TUs
    from jeanplot.network_utils import get_tu_informations, get_tu_grid_layout
    tu_infos = get_tu_informations(mnet._network)
    print(f"\nTU Information (checking cotx_name):")
    for tu_id, tu_info in sorted(tu_infos.items()):
        print(f"  TU {tu_id}:")
        print(f"    cotx_name: {tu_info.cotx_name}")
        print(f"    is_marker: {tu_info.is_marker}")
        print(f"    cotx_marker: {tu_info.cotx_marker}")
        print(f"    aggregation_node_id: {tu_info.aggregation_node_id}")
    
    # Check grid layout with translation nodes
    grid_layout = get_tu_grid_layout(mnet._network, node_type="translation")
    print(f"\nGrid Layout (translation nodes):")
    for i, layer in enumerate(grid_layout):
        print(f"  Layer {i}: {layer}")
        
    # Check grid layout with transcription nodes
    grid_layout_tx = get_tu_grid_layout(mnet._network, node_type="transcription")
    print(f"\nGrid Layout (transcription nodes):")
    for i, layer in enumerate(grid_layout_tx):
        print(f"  Layer {i}: {layer}")
    
    # Check compute graph translation nodes
    cg = mnet._network.compute_graph
    translation_nodes = cg[cg["type"] == "translation"]
    print(f"\nTranslation nodes:")
    for idx, row in translation_nodes.iterrows():
        cdg_inputs = row.get("cdg_input", [])
        print(f"  Node {idx}: cdg_input={cdg_inputs}")
        # Check which TUs these map to
        for cdg_id in cdg_inputs:
            if cdg_id in mnet._network.central_dogma_graph.index:
                tu_id = mnet._network.central_dogma_graph.loc[cdg_id].get("tu_id", [None])[0]
                print(f"    -> TU: {tu_id}")
    
    # Check all TUs in central dogma graph
    cdg = mnet._network.central_dogma_graph
    print(f"\nAll TUs in central dogma graph:")
    tu_ids_in_cdg = set()
    for idx, row in cdg.iterrows():
        tu_id_list = row.get("tu_id", [])
        if tu_id_list and tu_id_list[0]:
            tu_id = tu_id_list[0]
            tu_ids_in_cdg.add(tu_id)
            print(f"  CDG {idx}: TU={tu_id}, type={row.get('type', 'unknown')}")
    
    print(f"\nMissing TUs (not connected to translation nodes):")
    all_tu_ids = set(tu_infos.keys())
    tus_in_grid = set()
    for layer in grid_layout:
        tus_in_grid.update(layer)
    missing_tus = all_tu_ids - tus_in_grid
    for tu_id in missing_tus:
        print(f"  {tu_id}")
    
    # Save the figure for inspection
    fig.savefig('test_cotx_merging_output.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Debug: Check the built network
    print(f"\nBuilt network type: {type(mnet._network)}")
    print(f"Built network cotx count: {len(mnet._network.cotx)}")
    for i, cotx in enumerate(mnet._network.cotx):
        print(f"  CoTx {i+1}: {cotx.name}")
        print(f"    Units: {len(cotx.units)}")
        for unit in cotx.units:
            print(f"      Unit: {unit.name} - Slots: {[str(s) for s in unit.slots]}")
    
    # Check network info
    network_info = mnet._network.generate_network_info()
    print(f"\nNetwork info cotx_str:")
    print(network_info.get("cotx_str"))
    
    # Check compute graph sources
    sources = mnet._network.compute_graph[mnet._network.compute_graph["type"] == "source"]
    print(f"\nCompute graph sources count: {len(sources)}")
    for idx, (source_id, source) in enumerate(sources.iterrows()):
        print(f"  Source {idx+1}: {source_id}")
        print(f"    source_id: {source.get('source_id', 'NO_SOURCE_ID')}")
        print(f"    cotx_name: {source.get('cotx_name', 'NO_COTX_NAME')}")
        
    # Check aggregation nodes
    agg_nodes = mnet._network.compute_graph[mnet._network.compute_graph["type"] == "aggregation"]
    print(f"\nAggregation nodes count: {len(agg_nodes)}")
    for idx, (agg_id, agg) in enumerate(agg_nodes.iterrows()):
        print(f"  Aggregation {idx+1}: {agg_id}")
        extra = agg.get('extra', {})
        if extra and isinstance(extra, dict):
            print(f"    name from extra: {extra.get('name', 'NO_NAME')}")
        else:
            print(f"    extra field: {extra}")
    
    # Assert we have 4 cotransfections
    assert len(mnet._network.cotx) == 4, f"Expected 4 cotransfections, got {len(mnet._network.cotx)}"
    
    # Check names are preserved
    expected_names = ["TNFa", "TGF-β", "IFNab", "Bb"]
    actual_names = [cotx.name for cotx in mnet._network.cotx]
    assert set(actual_names) == set(expected_names), f"Expected names {expected_names}, got {actual_names}"

if __name__ == "__main__":
    test_cotx_merging()
    print("Test completed - check test_cotx_merging_output.png")