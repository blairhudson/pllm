import itertools

import pytest

from scripts import probe_column_residue_placement as p


def stage(weights):
    widths = [p.width(sum(map(abs, row))) for row in weights]
    return {"stage": "toy", "shape": [len(weights), len(weights[0])],
            "stage_bits": max(widths),
            "row_residue_width_histogram": {str(h): widths.count(h) for h in set(widths)}}


def test_oracle_all_subsets_and_corners():
    assert p.checks()["independent_integer_outputs"] == 3000
    with pytest.raises(ValueError, match="activation domain"):
        p.split_product([[1]], [-128], [], [0], [0])
    with pytest.raises(ValueError, match="columns"):
        p.split_product([[1]], [127], [0, 0], [0], [0])


def test_centered_bound_thresholds():
    for s in range(40000):
        h = p.width(s)
        assert p.minimum_sum(h) <= s
        assert -(1 << (h - 1)) <= -127 * s
        assert 127 * s < 1 << (h - 1)
        assert p.width(p.minimum_sum(h)) == h
    assert p.width(0) == 1
    for h in range(10, 27):
        t = ((1 << (h - 1)) - 1) // 127
        assert p.width(t) == h
        assert p.width(t + 1) == h + 1


def test_histogram_relaxation_dominates_every_actual_subset():
    weights = [[127, -128, 1, 0], [20, 3, -17, 2], [0, 0, 0, 0]]
    s = stage(weights)
    for k in range(1, 4):
        bound = p.option(s, k, 150)
        for columns in itertools.combinations(range(4), k):
            widths = [p.width(sum(abs(v) for i, v in enumerate(row) if i not in columns))
                      for row in weights]
            assert bound["remote_output_bits_sum_lower"] <= sum(widths)
            assert bound["remote_request_bits_lower"] <= max(widths)
    assert p.option(s, 4, 150)["net_upper"] == 150 * p.baseline(s) - (12 + p.HEADER)


def test_whole_group_knapsack_matches_independent_enumeration():
    stages = [stage([[i + 1] * (i + 2)] * (i + 1)) for i in range(5)]
    for i, s in enumerate(stages):
        s["stage"] = str(i)
    for budget in (0, 6, 20, 50):
        actual = p.group_optimum(stages, 157, budget)
        expected = 0
        for bits in itertools.product((False, True), repeat=5):
            chosen = [s for s, yes in zip(stages, bits, strict=True) if yes]
            if sum(a["shape"][0] * a["shape"][1] for a in chosen) <= budget:
                expected = max(expected, sum(p.option(a, a["shape"][1], 157)["net_upper"] for a in chosen))
        assert actual["net_saving"] == expected


def test_certificate_upper_bounds_actual_column_optimizer():
    w = [[127, 1, -128, 0], [1, 127, 0, -128]]
    for rows in (1, 157, 1000):
        for budget in (0, 2, 4, 6, 8):
            exact = p.exact_toy_optimum(w, rows, budget)
            upper = p.certificate([stage(w)], rows, budget)["column_net_saving_upper"]
            assert exact["net_saving"] <= upper


def test_cost_cache_and_frame_ledger():
    s = stage([[127] * 4] * 3)
    cold = p.option(s, 1, 157)
    warm = p.option(s, 1, 157, False)
    assert warm["net_upper"] - cold["net_upper"] == cold["delivery_bytes"] == 71
    assert p.option(s, 0, 157)["net_upper"] == 0
    assert p.option(s, 1, 1)["selected_native_weight_bytes"] == 3
    with pytest.raises(ValueError):
        p.option(s, 5, 1)


def test_source_lock_and_geometry():
    import json

    assert p.sha(p.SOURCE) == p.SOURCE_SHA
    source = json.loads(p.SOURCE.read_text())["checkpoint_geometry"]
    assert len(source["stages"]) == 96
    assert sum(s["shape"][0] for s in source["stages"]) == 304128
    assert sum(s["shape"][0] * s["shape"][1] for s in source["stages"]) == 357826560
