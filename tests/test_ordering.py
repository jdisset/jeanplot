from jeanplot.core.ordering import min_crossing_permutation, relax_y


def _crossings(slot_ys, group_target_ys, perm):
    from itertools import combinations

    edges = [(slot_ys[i], t) for i, g in enumerate(perm) for t in group_target_ys[g]]
    return sum((a[0] - b[0]) * (a[1] - b[1]) < 0 for a, b in combinations(edges, 2))


def test_trivial_sizes():
    assert min_crossing_permutation([], []) == []
    assert min_crossing_permutation([0.0], [[1.0]]) == [0]


def test_swaps_to_uncross():
    # slots top->bottom = [10, 0]; group 0 wires low, group 1 wires high -> swap.
    slot_ys = [10.0, 0.0]
    targets = [[0.0], [10.0]]
    perm = min_crossing_permutation(slot_ys, targets)
    assert perm == [1, 0]
    assert _crossings(slot_ys, targets, perm) == 0


def test_already_optimal_is_identity():
    slot_ys = [10.0, 0.0]
    targets = [[10.0], [0.0]]
    assert min_crossing_permutation(slot_ys, targets) == [0, 1]


def test_four_groups_finds_zero_crossing_layout():
    slot_ys = [3.0, 2.0, 1.0, 0.0]
    targets = [[0.0], [1.0], [2.0], [3.0]]  # fully reversed
    perm = min_crossing_permutation(slot_ys, targets)
    assert _crossings(slot_ys, targets, perm) == 0
    assert perm == [3, 2, 1, 0]


def test_tiebreak_prefers_straighter_wiring():
    # both orders cross zero times; pick the one with shorter total wire.
    slot_ys = [1.0, 0.0]
    targets = [[1.0], [0.0]]
    assert min_crossing_permutation(slot_ys, targets) == [0, 1]


def test_barycenter_fallback_matches_brute_on_separable():
    slot_ys = [float(i) for i in range(9)]
    targets = [[float(8 - i)] for i in range(9)]
    perm = min_crossing_permutation(slot_ys, targets, max_brute=8)
    assert _crossings(slot_ys, targets, perm) == 0


def test_relax_straightens_to_neighbor_median():
    # node 1 (free, col 1) bridges fixed 0 @0 and fixed 2 @10 -> pulled to 5.
    pos = [0.0, 99.0, 10.0]
    neighbors = [[1], [0, 2], [1]]
    movable = [False, True, False]
    columns = [0, 1, 2]
    out = relax_y(pos, neighbors, movable, columns)
    assert out[0] == 0.0 and out[2] == 10.0
    assert abs(out[1] - 5.0) < 1e-9


def test_relax_leaves_fixed_and_orphan_nodes():
    pos = [3.0, 7.0]
    out = relax_y(pos, [[], []], [True, False], [0, 0], min_gap=0.0)
    assert out == [3.0, 7.0]  # node 0 movable but has no neighbors -> unchanged


def test_relax_min_gap_separates_within_column():
    # two free nodes both want y=5 in the same column; gap keeps them apart.
    pos = [5.0, 5.0, 5.0]
    neighbors = [[2], [2], []]
    movable = [True, True, False]
    columns = [0, 0, 1]
    out = relax_y(pos, neighbors, movable, columns, min_gap=4.0)
    assert abs(out[0] - out[1]) >= 4.0 - 1e-9


def test_relax_does_not_move_across_columns():
    # free nodes in different columns can share y without separation.
    pos = [5.0, 5.0]
    out = relax_y(pos, [[], []], [True, True], [0, 1], min_gap=4.0)
    assert out == [5.0, 5.0]
