# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from itertools import combinations, permutations
from statistics import median


def min_crossing_permutation(
    slot_ys: list[float],
    group_target_ys: list[list[float]],
    max_brute: int = 8,
) -> list[int]:
    """Assign groups to fixed vertical slots so wires cross as little as possible.

    `slot_ys[i]` is the y of the slot at list position i; `group_target_ys[g]`
    the y's that group g wires to. Returns `perm` with `perm[i]` = group placed
    at slot i. Exhaustive for n <= max_brute, else barycenter.
    """
    n = len(slot_ys)
    assert n == len(group_target_ys)
    if n < 2:
        return list(range(n))

    def cost(perm: tuple[int, ...]) -> tuple[int, float]:
        edges = [(slot_ys[i], t) for i, g in enumerate(perm) for t in group_target_ys[g]]
        cross = sum((a[0] - b[0]) * (a[1] - b[1]) < 0 for a, b in combinations(edges, 2))
        wire = sum(abs(s - t) for s, t in edges)
        return cross, wire

    if n <= max_brute:
        return list(min(permutations(range(n)), key=cost))

    def mean(ys: list[float]) -> float:
        return sum(ys) / len(ys) if ys else 0.0

    by_target = sorted(range(n), key=lambda g: mean(group_target_ys[g]))
    perm = [0] * n
    for slot_i, g in zip(sorted(range(n), key=lambda i: slot_ys[i]), by_target):
        perm[slot_i] = g
    return perm


def _separate(members: list[int], y: list[float], movable: list[bool], gap: float) -> None:
    order = sorted(members, key=lambda i: y[i])
    for k in range(1, len(order)):
        i, prev = order[k], order[k - 1]
        if movable[i] and y[i] - y[prev] < gap:
            y[i] = y[prev] + gap
    for k in range(len(order) - 2, -1, -1):
        i, nxt = order[k], order[k + 1]
        if movable[i] and y[nxt] - y[i] < gap:
            y[i] = y[nxt] - gap


def relax_y(
    pos: list[float],
    neighbors: list[list[int]],
    movable: list[bool],
    columns: list[int],
    min_gap: float = 0.0,
    sweeps: int = 4,
) -> list[float]:
    """Median-relax node y's to straighten edges (Sugiyama coordinate phase).

    Each sweep pulls every movable node to the median of its neighbours' y, then
    a per-column min-gap pass separates overlaps with fixed nodes as anchors.
    Returns new y per node. O(edges + nodes log nodes) per sweep.
    """
    y = list(pos)
    cols: dict[int, list[int]] = {}
    for i, c in enumerate(columns):
        cols.setdefault(c, []).append(i)
    for _ in range(sweeps):
        for i, nb in enumerate(neighbors):
            if movable[i] and nb:
                y[i] = median(y[j] for j in nb)
        if min_gap > 0:
            for members in cols.values():
                _separate(members, y, movable, min_gap)
    return y
