from typing import List, Dict, Optional
from pydantic import BaseModel
from collections import defaultdict


class PartInfo(BaseModel):
    """information about a genetic part"""

    name: str
    category: str


class TUInfo(BaseModel):
    """transcription unit info"""

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
    """interaction between genetic elements"""

    src_tu_id: str
    src_part_name: str
    tgt_tu_id: str
    tgt_part_name: str
    type: str = "ERN"


def get_tu_informations(network) -> Dict[str, TUInfo]:
    """extract TU info from network, returns dict of TUInfo keyed by tu_id"""
    network_info = network.generate_network_info()
    markers = set(network_info.get("markers", []))

    # get source and aggregation data
    sources = network.compute_graph[network.compute_graph["type"] == "source"]

    # first pass: collect basic TU data
    tus = {}
    aggr_to_tus = defaultdict(list)  # map aggregation node -> tu_ids

    for _, src in sources.iterrows():
        plasmid_name = "_".join(src["source_id"].split("_")[:-1])
        tu_cdgs = network.central_dogma_graph.loc[src["cdg_output"]]
        is_in_l2 = len(tu_cdgs) > 1  # l2 = plasmid with multiple TUs

        # process each TU in this plasmid
        for pos, (_, tu_row) in enumerate(tu_cdgs.iterrows()):
            tu_id = tu_row["tu_id"][0]  # assuming one tu_id per row
            tu_name = "_".join(tu_id.split("_")[:-1])

            # find if this is a marker TU
            content = tu_row["content"]
            is_marker = any(item in markers for item in content)
            marker = next((item for item in content if item in markers), None)

            # get part information from network_info
            parts = []
            if network_info and "all_parts" in network_info:
                parts_dict = network_info.get("all_parts", {}).get(tu_id, {})
                parts = [
                    PartInfo(name=name, category=category) for name, category in parts_dict.items()
                ]

            # create TU info
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

            # check for aggregation
            if src["input_from"] and len(src["input_from"]) > 0:
                upstream_id = src["input_from"][0][0]
                try:
                    upstream_type = network.compute_graph.at[upstream_id, "type"]

                    if upstream_type == "aggregation":
                        ratios = network.compute_graph.at[upstream_id, "extra"]["ratios"]

                        tus[tu_id].aggregation_node_id = upstream_id
                        tus[tu_id].aggregation_ratio = ratios[pos]
                        tus[tu_id].in_aggregation = True
                        aggr_to_tus[upstream_id].append(tu_id)
                except (KeyError, IndexError):
                    pass

    # second pass: normalize ratios within aggregation groups
    for aggr_id, tu_ids in aggr_to_tus.items():
        # find min ratio in this aggregation group
        min_ratio = min(tus[tu_id].aggregation_ratio or 1.0 for tu_id in tu_ids)
        if min_ratio > 0:
            # normalize ratios
            for tu_id in tu_ids:
                if tus[tu_id].aggregation_ratio:
                    tus[tu_id].aggregation_ratio_norm = tus[tu_id].aggregation_ratio / min_ratio

    # third pass: ensure all TUs in same aggregation have the same marker
    for aggr_id, tu_ids in aggr_to_tus.items():
        # find marker TU
        marker_tu_id = next((tu_id for tu_id in tu_ids if tus[tu_id].is_marker), None)

        # if found, propagate marker to all TUs in this aggregation
        if marker_tu_id:
            marker = tus[marker_tu_id].cotx_marker
            marker_ratio = tus[marker_tu_id].aggregation_ratio_norm

            for tu_id in tu_ids:
                tus[tu_id].cotx_marker = marker
                tus[tu_id].marker_ratio = marker_ratio

    # final pass: create ratio labels
    for aggr_id, tu_ids in aggr_to_tus.items():
        # collect all normalized ratios
        ratios = sorted([tus[tu_id].aggregation_ratio_norm for tu_id in tu_ids])

        # create a combined label
        label = ":".join([f"{r:.0f}" for r in ratios])

        # apply to all TUs in this aggregation
        for tu_id in tu_ids:
            tus[tu_id].aggregation_ratio_label = label

    return tus


def get_tu_grid_layout(network, node_type="translation") -> List[List[str]]:
    """create grid layout of TUs based on network topology"""
    # get nodes of the specified type
    cnodes = network.compute_graph[network.compute_graph["type"] == node_type]

    # map nodes to their TU ids
    node_to_tus = {}
    for node_id, inputs in cnodes["cdg_input"].items():
        node_to_tus[node_id] = [
            network.central_dogma_graph.loc[input_id]["tu_id"][0] for input_id in inputs
        ]

    # get topological ordering of nodes
    topo_order = network.topological_order(cnodes.index.tolist())

    # organize into columns
    columns = [[] for _ in range(len(topo_order))]
    columns[-1] = topo_order[-1]  # last column is easy

    # optimize earlier columns based on upstream relationships
    for col_idx in range(len(topo_order) - 2, -1, -1):
        next_col = topo_order[col_idx + 1]
        columns[col_idx] = []

        # first add nodes that are upstream of next column
        for next_node in next_col:
            for node in topo_order[col_idx]:
                if (
                    network.compute_node_is_upstream_of(node, next_node)
                    and node not in columns[col_idx]
                ):
                    columns[col_idx].append(node)

        # then add any remaining nodes
        for node in topo_order[col_idx]:
            if node not in columns[col_idx]:
                columns[col_idx].append(node)

    # convert node IDs to TU IDs and flatten
    return [[tu_id for node_id in col for tu_id in node_to_tus.get(node_id, [])] for col in columns]


def get_interactions(network) -> List[Interaction]:
    """extract interactions between genetic elements"""
    ern_nodes = network.compute_graph[network.compute_graph["type"] == "sequestron_ERN"]
    interactions = []

    for _, ern in ern_nodes.iterrows():
        cdg_inputs = network.central_dogma_graph.loc[ern["cdg_input"]]

        if len(cdg_inputs) != 2:
            continue

        # first input is ERN part
        ern_row = cdg_inputs.iloc[0]
        ern_tu_ids = ern_row["tu_id"]
        ern_part_name = ern_row["content"][0]

        # second input is recognition site
        rec_row = cdg_inputs.iloc[1]
        rec_tu_ids = rec_row["tu_id"]
        rec_parts = rec_row["content"]

        # find matching recognition part (has ERN name in it)
        rec_part_name = next((p for p in rec_parts if ern_part_name in p), None)
        if not rec_part_name:
            continue

        # create interaction for each source-target pair
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

    return interactions
