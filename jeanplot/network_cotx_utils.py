"""Utilities for handling CoTransfection names in network diagrams"""

from typing import Dict, Optional, Any
import pandas as pd


def inject_cotx_names_into_compute_graph(network: Any) -> None:
    """
    Inject CoTransfection names from the Network.cotx field into the compute graph.
    This modifies the compute graph in-place by adding 'name' to aggregation nodes' extra data.
    
    Args:
        network: A biocomp Network object that has been built
    """
    if not hasattr(network, 'cotx') or network.cotx is None:
        return
    
    if not hasattr(network, 'compute_graph') or network.compute_graph is None:
        return
    
    # Create a mapping of source names to cotransfection names
    source_to_cotx_name: Dict[str, str] = {}
    
    for cotx in network.cotx:
        if cotx.name:
            for unit in cotx.units:
                if unit.source:
                    source_to_cotx_name[unit.source] = cotx.name
    
    # Find aggregation nodes and map them to cotransfection names
    compute_graph = network.compute_graph
    agg_nodes = compute_graph[compute_graph["type"] == "aggregation"]
    
    for idx in agg_nodes.index:
        # Get the source nodes that feed into this aggregation
        output_to = compute_graph.at[idx, "output_to"]
        
        # Find source nodes connected to this aggregation
        found_cotx_name = None
        for target_id, _ in output_to:
            if target_id in compute_graph.index:
                target_row = compute_graph.loc[target_id]
                if target_row["type"] == "source" and "source_id" in target_row:
                    source_id = target_row["source_id"]
                    # The source_id is already the base source name (e.g., "plsmd_1")
                    base_source = source_id
                    if base_source in source_to_cotx_name:
                        found_cotx_name = source_to_cotx_name[base_source]
                        break
        
        # Add the name to the aggregation node's extra data
        if found_cotx_name:
            if pd.isna(compute_graph.at[idx, "extra"]) or compute_graph.at[idx, "extra"] is None:
                compute_graph.at[idx, "extra"] = {}
            compute_graph.at[idx, "extra"]["name"] = found_cotx_name


def get_cotx_name_from_markers(network: Any, markers: set) -> Optional[str]:
    """
    Try to find a cotransfection name based on marker combinations.
    
    Args:
        network: A biocomp Network object
        markers: Set of marker names found in the aggregation
        
    Returns:
        The cotransfection name if found, None otherwise
    """
    if not hasattr(network, 'cotx') or network.cotx is None:
        return None
    
    for cotx in network.cotx:
        if not cotx.name:
            continue
            
        # Get all markers from this cotransfection
        cotx_markers = set()
        for unit in cotx.units:
            for slot in unit.slots:
                if hasattr(slot, 'part') and slot.part:
                    # Check if this part is a marker
                    if isinstance(slot.part, str) and network.lib and slot.part in network.lib.pc.index:
                        category = network.lib.pc.loc[slot.part, "category"]
                        if category == "fluor":
                            cotx_markers.add(slot.part)
        
        # Check if the markers match
        if cotx_markers == markers:
            return cotx.name
    
    return None