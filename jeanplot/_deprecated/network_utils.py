from typing import Any
from pydantic import BaseModel
from collections import defaultdict
import numpy as np


class PartInfo(BaseModel):
    name: str
    category: str


class TUInfo(BaseModel):
    tu_id: str
    tu_name: str
    cotx_marker: str | None = None
    is_marker: bool = False
    plasmid_name: str
    in_l2: bool = False
    position_in_plasmid: int = 0
    number_of_tu_in_plasmid: int = 1
    aggregation_ratio: float | None = None
    aggregation_node_id: int | None = None
    in_aggregation: bool = False
    aggregation_ratio_norm: float = 1.0
    marker_ratio: float | None = None
    aggregation_ratio_label: str = ""
    marker_in_l2: bool = False
    parts: list[PartInfo] = []
    cotx_name: str | None = None


class Interaction(BaseModel):
    src_tu_id: str
    src_part_name: str
    tgt_tu_id: str
    tgt_part_name: str
    type: str = "ERN"


def get_tu_informations(network: Any) -> dict[str, TUInfo]:
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
            tu_id = tu_row.get("tu_id", [None])[0]
            if not tu_id:
                continue
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

            input_from = src.get("input_from")
            if input_from and len(input_from) > 0:
                upstream_id = input_from[0][0]
                if (
                    upstream_id in network.compute_graph.index
                    and network.compute_graph.at[upstream_id, "type"] == "aggregation"
                ):
                    tus[tu_id].aggregation_node_id = upstream_id
                    tus[tu_id].in_aggregation = True
                    aggr_to_tus[upstream_id].append(tu_id)

    for agg_id, tu_ids_in_agg in aggr_to_tus.items():
        try:
            agg_node_series = network.compute_graph.loc[agg_id]
            extra_data = agg_node_series.get("extra", {})
            raw_ratios = extra_data.get("ratios") if isinstance(extra_data, dict) else None
            cotx_name = extra_data.get("name") if isinstance(extra_data, dict) else None
            output_tu_ids = agg_node_series.get("cdg_output", [])

            # First, assign cotx_name to all TUs in this aggregation
            if cotx_name:
                for tu_id in tu_ids_in_agg:
                    if tu_id in tus:
                        tus[tu_id].cotx_name = cotx_name

            if raw_ratios is None or not isinstance(raw_ratios, list):
                continue
            if not output_tu_ids or not isinstance(output_tu_ids, list):
                continue
            if len(raw_ratios) != len(output_tu_ids):
                continue
            if not all(isinstance(r, (int, float)) for r in raw_ratios):
                continue

            tu_id_to_raw_ratio_map = {
                tu_id: ratio for tu_id, ratio in zip(output_tu_ids, raw_ratios)
            }

            group_data = []
            for tu_id in tu_ids_in_agg:
                if tu_id in tus and tu_id in tu_id_to_raw_ratio_map:
                    group_data.append((tus[tu_id], tu_id_to_raw_ratio_map[tu_id]))

            if not group_data:
                continue

            group_data.sort(key=lambda item: (item[0].plasmid_name, item[0].position_in_plasmid))
            sorted_raw_ratios = [item[1] for item in group_data]

            ratios_np = np.array(sorted_raw_ratios)
            min_ratio = np.maximum(ratios_np.min(), 1e-6)
            normed_ratios_float = np.round(ratios_np / min_ratio, 2)

            def is_round(x):
                return np.isclose(x, np.round(x), atol=1e-9)

            formatted_ratio_strings = [
                str(int(round(r))) if is_round(r) else str(r) for r in normed_ratios_float
            ]

            label = ":".join(formatted_ratio_strings)

            for tu_id in tu_ids_in_agg:
                if tu_id in tus:  # Ensure TU exists
                    tus[tu_id].aggregation_ratio_label = label
                    if tu_id in tu_id_to_raw_ratio_map:
                        tus[tu_id].aggregation_ratio = tu_id_to_raw_ratio_map[tu_id]

        except Exception:
            label = "Error"
            for tu_id in tu_ids_in_agg:
                if tu_id in tus:
                    tus[tu_id].aggregation_ratio_label = label

    for aggr_id, tu_ids in aggr_to_tus.items():
        marker_tu = next(
            (tus[tu_id] for tu_id in tu_ids if tu_id in tus and tus[tu_id].is_marker), None
        )
        if marker_tu:
            for tu_id in tu_ids:
                if tu_id in tus:
                    tus[tu_id].cotx_marker = marker_tu.cotx_marker
                    if hasattr(marker_tu, "aggregation_ratio_norm"):
                        tus[tu_id].marker_ratio = marker_tu.aggregation_ratio_norm

    return tus


def get_tu_grid_layout(
    network: Any,
    node_type="translation",
    tu_id_allow_set: set[str] | None = None,
) -> list[list[str]]:
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


def _get_source_id(tu_id: str, tu_infos: dict[str, TUInfo]) -> str:
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
    grid_layers: list[list[str]], tu_infos: dict[str, TUInfo]
) -> list[list[str]]:
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


def get_interactions(network: Any) -> list[Interaction]:
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
