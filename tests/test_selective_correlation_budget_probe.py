"""Isolated bounded cost and data-contract tests; no checkpoint/model execution."""
import copy
from fractions import Fraction
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_selective_correlation_budget.py"
SPEC = importlib.util.spec_from_file_location("selective_budget", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture(scope="module")
def sources():
    return probe.load_sources()


def test_frozen_geometry_and_semantic_demand(sources):
    geometry = probe.validate_geometry(sources)
    roles, headers = probe.compile_headers(sources, geometry)
    assert len(roles) == 96
    assert sum(s["role"] == "mlp_gate_up" for s in roles.values()) == 24
    assert sum(headers["150/both"]["phase_rows"]["prefill"].values()) == 13953
    assert sum(headers["288/both-placement"]["phase_rows"]["prefill"].values()) == 15843
    for header in headers.values():
        assert all(Fraction(f) >= Fraction(9, 10) for f in header["remote_mac_fractions"].values())


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "ring", "width", "histogram", "shape", "weight", "rows", "payload", "lock", "body"])
def test_geometry_rejects(sources, mutation):
    changed = copy.deepcopy(sources)
    narrow = changed["research-narrow-ingress.json"]["checkpoint_geometry"]
    stage = narrow["stages"][0]
    bootstrap = changed["research-bootstrap-ingress.json"]["cost"]["stages"][0]
    if mutation == "duplicate":
        narrow["stages"].append(copy.deepcopy(stage))
    elif mutation == "missing":
        narrow["stages"].pop()
    elif mutation == "ring":
        bootstrap["original_ring_bits"] = 16
    elif mutation == "width":
        stage["stage_bits"] -= 1
    elif mutation == "histogram":
        stage["row_residue_width_histogram"]["20"] += 1
    elif mutation == "shape":
        stage["shape"][0] += 1
    elif mutation == "weight":
        stage["weight_sha256"] = "0" * 64
    elif mutation == "rows":
        bootstrap["rows"] += 1
    elif mutation == "payload":
        bootstrap["output_payload"] += 1
    elif mutation == "lock":
        narrow["lock"]["source_lock_digest"] = "0" * 64
    else:
        narrow["lock"]["body_fingerprint"] = "0" * 64
    with pytest.raises(ValueError):
        probe.validate_geometry(changed)


def test_hash_drift_rejects(tmp_path):
    folder = tmp_path / "docs/evidence"
    folder.mkdir(parents=True)
    (folder / next(iter(probe.LOCKS))).write_text("{}")
    with pytest.raises(ValueError, match="source hash drift"):
        probe.load_sources(tmp_path)


def test_duplicate_json_key():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        probe.unique_object([("stage", 1), ("stage", 2)])


def test_horizon_independent_oracle():
    for horizon in (1, 2, 7, 64):
        for reuse in (False, True):
            total = 1000 * horizon - (horizon - 1) * 200 * reuse
            result = probe.break_even(1000, 200, 300, 100, horizon, Fraction(1, 5), reuse, 50)
            assert result == Fraction(300 * horizon - Fraction(1, 5) * total - 50, 100 * horizon)


def test_renewal_boundary_burns_and_unused():
    expected = [(3, "10", 1, 1), (4, "10", 1, 0), (5, "40", 2, 3), (8, "40", 2, 0), (9, "70", 3, 3)]
    for rows, cost, pools, unused in expected:
        assert probe.additional_cost(rows, rows * 10, 4, setup=10, renewal=30) == {
            "bytes": cost, "pools": pools, "unused_rows": unused}
    assert probe.additional_cost(4, 40, 4, burned=1, setup=10, renewal=30)["bytes"] == "40"


@pytest.mark.parametrize("unknown", ["capacity", "setup", "renewal", "public_creation", "public_distribution", "lane_cost", "control", "burned"])
def test_unknown_not_zero(unknown):
    kwargs = {unknown: None}
    if unknown != "capacity":
        kwargs["capacity"] = 4
    assert probe.additional_cost(3, 30, **kwargs)["bytes"] is None
    assert probe.break_even(1000, 200, 300, 100, 1, Fraction(1, 5), True, None) is None


def test_costs_monotone():
    baseline = Fraction(probe.additional_cost(9, 90, 4, setup=10, renewal=30)["bytes"])
    for name in ("setup", "renewal", "public_creation", "public_distribution", "control", "lane_cost", "burned"):
        kwargs = {"setup": 10, "renewal": 30, name: 100}
        assert Fraction(probe.additional_cost(9, 90, 4, **kwargs)["bytes"]) > baseline
    assert probe.break_even(1000, 200, 300, 100, 4, Fraction(1, 5), True, 20) < probe.break_even(
        1000, 200, 300, 100, 4, Fraction(1, 5), True, 10)


def test_threshold_optimizer_exhaustive():
    items = [{"lanes": 10, "payload_bytes": 20}, {"lanes": 5, "payload_bytes": 15},
             {"lanes": 8, "payload_bytes": 9}]
    for cost in (Fraction(), Fraction(1), Fraction(2), Fraction(5, 2), Fraction(3)):
        chosen = probe.choose_subset(items, "cost_threshold", cost)
        optimum = max(sum(items[i]["payload_bytes"] - cost * items[i]["lanes"] for i in range(3) if mask >> i & 1)
                      for mask in range(8))
        assert sum(i["payload_bytes"] - cost * i["lanes"] for i in chosen) == optimum
    assert probe.choose_subset(items, "cost_threshold", None) is None


def test_finite_matrix_columns():
    items = [{"stage": "opaque", "rows": 3}]
    geometry = {"opaque": {"shape": [3, 5], "ring_bits": 24}}
    first, second = probe.matrix_columns(items, geometry, block=4, horizons=(1, 2))
    assert first == {"pairs": 1, "triples": 2, "opening_bytes": 192,
                     "unused_block_columns": 2, "extra_client_dense_macs": 45}
    assert second["opening_bytes"] == 384
    assert second["unused_block_columns"] == 4


def test_renewal_can_reduce_horizon_saving():
    before = probe.additional_cost(4, 40, 4, setup=0, renewal=1000)
    after = probe.additional_cost(5, 50, 4, setup=0, renewal=1000)
    b1 = probe.break_even(1000, 200, 300, 10, 4, Fraction(1, 5), True, Fraction(before["bytes"]))
    b2 = probe.break_even(1000, 200, 300, 10, 5, Fraction(1, 5), True, Fraction(after["bytes"]))
    assert b2 < b1


def test_target_monotonicity_and_subset_phase():
    budgets = [probe.break_even(1000, 200, 300, 10, 4, t, True) for t in probe.TARGETS]
    assert budgets[0] > budgets[1] > budgets[2]
    items = [{"role": "mlp_gate_up", "phase": "prefill", "rows": rows} for rows in (1, 150)]
    items.append({"role": "mlp_gate_up", "phase": "decode", "rows": 14})
    assert probe.choose_subset(items, "gate_up_bulk_prefill") == [items[1]]


def test_publication_summary_replays():
    import sys
    sys.path.insert(0, str(PATH.parent))
    report = probe.build_report()
    summary = probe.publication_summary(report)
    assert len(summary["cases"]) == 4
    for case in summary["cases"]:
        assert len(case["samples"]) == 3
        assert case["bulk_gate_semantic_layers"] == list(range(23))
        for s in case["subsets"].values():
            assert s[0] <= s[1]
    cold = summary["cases"][0]["subsets"]["gate_up_prefill"]
    assert Fraction(cold[4]) < Fraction(1, 5)  # generous upper bound kills cold gate
    assert Fraction(cold[7][0]) > 0  # does not veto warm deployment
    for case in summary["cases"]:
        sample = next(s for s in case["samples"] if s["replicate"] == case["representative_replicate"])
        residual = sample["correction_residual"]
        for subset in case["subsets"].values():
            lanes = subset[2]
            for mandatory, optimistic in zip(subset[6], subset[8], strict=True):
                assert Fraction(optimistic) - Fraction(mandatory) == Fraction(residual, lanes)
            mandatory_headroom, optimistic_headroom = map(Fraction, subset[9])
            assert optimistic_headroom - mandatory_headroom == residual
            assert mandatory_headroom == Fraction(subset[6][0]) * lanes
            assert optimistic_headroom == Fraction(subset[8][0]) * lanes
    # Rejection remains valid only after testing the generous residual credit.
    assert Fraction(cold[8][0]) < 0


def test_negative_mandatory_allowance_need_not_reject():
    mandatory = probe.break_even(1000, 200, 190, 100, 1, Fraction(1, 5), True)
    optimistic = probe.break_even(1000, 200, 220, 100, 1, Fraction(1, 5), True)
    assert mandatory < 0 < optimistic
    assert optimistic - mandatory == Fraction(30, 100)


@pytest.mark.parametrize("bad", [-1, True, 0.5])
def test_invalid_integer(bad):
    with pytest.raises(ValueError):
        probe.additional_cost(bad, 10, 4)
