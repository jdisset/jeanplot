from typing import List, Dict, Optional, Any, Set
from pydantic import BaseModel
from collections import defaultdict


class PartInfo(BaseModel):
    name: str
    category: str


class TUInfo(BaseModel):
    tu_id: str
    tu_name: str
    cotx_marker: Optional[str] = None
    is_marker: bool = False
    plasmid_name: str
    in_l2: bool = False
    position_in_plasmid: int = 0
    number_of_tu_in_plasmid: int = 1
    aggregation_ratio: Optional[float] = None
    aggregation_node_id: Optional[int] = None
    in_aggregation: bool = False
    aggregation_ratio_norm: float = 1.0
    marker_ratio: Optional[float] = None
    aggregation_ratio_label: str = ""
    marker_in_l2: bool = False
    parts: List[PartInfo] = []


class Interaction(BaseModel):
    src_tu_id: str
    src_part_name: str
    tgt_tu_id: str
    tgt_part_name: str
    type: str = "ERN"


def get_tu_informations(network: Any) -> Dict[str, TUInfo]:
    network_info = network.generate_network_info()
    markers = set(network_info.get("markers", []))
    sources = network.compute_graph[network.compute_graph["type"] == "source"]
    tus = {}
    aggr_to_tus = defaultdict(list)

    for _, src in sources.iterrows():
        plasmid_name = "_".join(src["source_id"].split("_")[:-1])
        tu_cdgs = network.central_dogma_graph.loc[src["cdg_output"]]
        is_in_l2 = len(tu_cdgs) > 1

        for pos, (_, tu_row) in enumerate(tu_cdgs.iterrows()):
            # use .get() with default to handle potential missing tu_id gracefully
            tu_id = tu_row.get("tu_id", [None])[0]
            if not tu_id:
                continue  # skip if no tu_id found
            tu_name = "_".join(tu_id.split("_")[:-1])
            content = tu_row.get("content", [])
            is_marker = any(item in markers for item in content)
            marker = next((item for item in content if item in markers), None)

            parts_dict = network_info.get("all_parts", {}).get(tu_id, {})
            parts = [
                PartInfo(name=name, category=category) for name, category in parts_dict.items()
            ]

            tus[tu_id] = TUInfo(
                tu_id=tu_id,
                tu_name=tu_name,
                cotx_marker=marker,
                is_marker=is_marker,
                plasmid_name=plasmid_name,
                in_l2=is_in_l2,
                position_in_plasmid=pos,
                number_of_tu_in_plasmid=len(tu_cdgs),
                parts=parts,
            )

            input_from = src.get("input_from")  # use .get()
            if input_from and len(input_from) > 0:
                upstream_id = input_from[0][0]
                try:
                    if network.compute_graph.at[upstream_id, "type"] == "aggregation":
                        ratios = network.compute_graph.at[upstream_id, "extra"]["ratios"]
                        # ensure ratios list has enough elements for the position
                        if pos < len(ratios):
                            tus[tu_id].aggregation_node_id = upstream_id
                            tus[tu_id].aggregation_ratio = ratios[pos]
                            tus[tu_id].in_aggregation = True
                            aggr_to_tus[upstream_id].append(tu_id)
                        else:
                            print(
                                f"Warning: Ratio missing for TU {tu_id} (pos {pos}) in aggregation {upstream_id}"
                            )

                except (KeyError, IndexError, TypeError):  # added TypeError for safety
                    # print(f"Warning: Could not process aggregation info for TU {tu_id}. Error: {e}")
                    pass  # ignore errors during aggregation processing

    # normalize ratios
    for aggr_id, tu_ids in aggr_to_tus.items():
        valid_ratios = [
            tus[tu_id].aggregation_ratio
            for tu_id in tu_ids
            if tus[tu_id].aggregation_ratio is not None
        ]
        if not valid_ratios:
            continue
        min_ratio = min(valid_ratios)
        if min_ratio > 0:
            for tu_id in tu_ids:
                if tus[tu_id].aggregation_ratio is not None:
                    tus[tu_id].aggregation_ratio_norm = tus[tu_id].aggregation_ratio / min_ratio

    # propagate marker info within aggregation groups
    for aggr_id, tu_ids in aggr_to_tus.items():
        marker_tu = next((tus[tu_id] for tu_id in tu_ids if tus[tu_id].is_marker), None)
        if marker_tu:
            for tu_id in tu_ids:
                tus[tu_id].cotx_marker = marker_tu.cotx_marker
                tus[tu_id].marker_ratio = marker_tu.aggregation_ratio_norm

    # create ratio labels
    for aggr_id, tu_ids in aggr_to_tus.items():
        # sort by original position within source for consistent labels
        sorted_tu_ids = sorted(tu_ids, key=lambda tid: tus[tid].position_in_plasmid)
        ratios = [tus[tu_id].aggregation_ratio_norm for tu_id in sorted_tu_ids]
        label = ":".join([f"{r:.0f}" if r is not None else "?" for r in ratios])
        for tu_id in tu_ids:
            tus[tu_id].aggregation_ratio_label = label

    return tus


def get_tu_grid_layout(
    network: Any,
    node_type="translation",
    tu_id_allow_set: Optional[Set[str]] = None,
) -> List[List[str]]:
    """
    create topological grid layout layers, considering only allowed TUs.
    if tu_id_allow_set is None, all TUs are considered.
    """
    cnodes = network.compute_graph[network.compute_graph["type"] == node_type]
    if cnodes.empty:
        return []

    node_to_tus = {}
    for node_id, inputs in cnodes["cdg_input"].items():
        # Extract TUs, handling potential errors
        tus_in_node = []
        for input_id in inputs:
            if input_id in network.central_dogma_graph.index:
                tu_id = network.central_dogma_graph.loc[input_id].get("tu_id", [None])[0]
                if tu_id and (tu_id_allow_set is None or tu_id in tu_id_allow_set):
                    tus_in_node.append(tu_id)
        if tus_in_node:  # only store nodes that have allowed TUs
            node_to_tus[node_id] = tus_in_node

    # filter nodes for topological sort based on whether they map to allowed TUs
    nodes_to_sort = [node_id for node_id in cnodes.index if node_id in node_to_tus]
    if not nodes_to_sort:
        return []

    topo_order = network.topological_order(nodes_to_sort)
    if not topo_order or not topo_order[0]:
        return []

    # build layers using only the filtered node_to_tus map
    layers = []
    processed_nodes = set()
    for layer_nodes in topo_order:
        current_layer = []
        # only consider nodes that we kept earlier
        valid_layer_nodes = [node for node in layer_nodes if node in node_to_tus]
        for node in valid_layer_nodes:
            if node not in processed_nodes:
                # extend with TUs we know are allowed
                current_layer.extend(node_to_tus[node])
                processed_nodes.add(node)
        if current_layer:
            layers.append(current_layer)

    return layers


def _get_source_id(tu_id: str, tu_infos: Dict[str, TUInfo]) -> str:
    """determines the unique source identifier for a TU"""
    if tu_id not in tu_infos:
        return "unknown_source"
    info = tu_infos[tu_id]
    if info.in_l2:
        return f"plasmid_{info.plasmid_name}"
    else:
        # include marker, node_id, and ratio label for uniqueness in cotx
        return (
            f"source_{info.cotx_marker}_{info.aggregation_node_id}_{info.aggregation_ratio_label}"
        )


def optimize_grid_for_source_adjacency(
    grid_layers: List[List[str]], tu_infos: Dict[str, TUInfo]
) -> List[List[str]]:
    """
    sorts TUs *within* each layer of the grid to group by source.
    preserves the layer structure.
    """
    optimized_layers = []
    for layer in grid_layers:
        if not layer:
            optimized_layers.append([])
            continue

        # create tuples of (tu_id, source_id, original_index) for sorting
        sortable_tus = []
        for i, tu_id in enumerate(layer):
            if tu_id and tu_id in tu_infos:  # handle potential None/missing TUs
                source_id = _get_source_id(tu_id, tu_infos)
                sortable_tus.append((tu_id, source_id, i))
            # else: implicitly skip None or invalid tu_ids

        # sort primarily by source_id, secondarily by original index (for stability)
        sortable_tus.sort(key=lambda x: (x[1], x[2]))

        # extract sorted tu_ids
        optimized_layer = [tu_id for tu_id, _, _ in sortable_tus]
        optimized_layers.append(optimized_layer)

    return optimized_layers


def get_interactions(network: Any) -> List[Interaction]:
    """extract interactions between genetic elements"""
    # handle case where sequestron_ERN nodes might not exist
    if "sequestron_ERN" not in network.compute_graph["type"].values:
        return []
    ern_nodes = network.compute_graph[network.compute_graph["type"] == "sequestron_ERN"]
    interactions = []

    for _, ern in ern_nodes.iterrows():
        cdg_inputs_idx = ern.get("cdg_input")
        if cdg_inputs_idx is None or len(cdg_inputs_idx) != 2:
            continue

        # safely access cdg graph data
        try:
            cdg_inputs = network.central_dogma_graph.loc[cdg_inputs_idx]
            ern_row = cdg_inputs.iloc[0]
            rec_row = cdg_inputs.iloc[1]

            ern_tu_ids = ern_row.get("tu_id", [])
            ern_part_name = ern_row.get("content", [None])[0]
            rec_tu_ids = rec_row.get("tu_id", [])
            rec_parts = rec_row.get("content", [])

            if not ern_tu_ids or not rec_tu_ids or not ern_part_name or not rec_parts:
                continue

            rec_part_name = next((p for p in rec_parts if ern_part_name in p), None)
            if not rec_part_name:
                continue

            for src_tu_id in ern_tu_ids:
                for tgt_tu_id in rec_tu_ids:
                    interactions.append(
                        Interaction(
                            src_tu_id=src_tu_id,
                            src_part_name=ern_part_name,
                            tgt_tu_id=tgt_tu_id,
                            tgt_part_name=rec_part_name,
                            type="ERN",
                        )
                    )
        except (KeyError, IndexError):
            # handle cases where cdg_input indices might be invalid
            continue

    return interactions
