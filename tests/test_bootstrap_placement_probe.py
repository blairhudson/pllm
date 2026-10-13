"""Independent split-row, resource-budget, and warm amortization checks."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "bootstrap_probe", Path(__file__).resolve().parents[1] / "scripts/probe_bootstrap_placement.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("selected", [(), (0,), (1, 3), (3, 1, 0, 2)])
def test_integer_split_against_independent_python_oracle(selected):
    x = np.array([[-128, 127, 0], [127, -128, 1]], dtype=np.int8)
    w = np.array(list(itertools.islice(itertools.product((-128, 127, 1), repeat=3), 8)), dtype=np.int8)
    expected = [[sum(int(a) * int(b) for a, b in zip(row, weight, strict=True)) for weight in w] for row in x]
    np.testing.assert_array_equal(probe.split_integer(x, w, selected), expected)


@pytest.mark.parametrize("selected", [(1, 1), (-1,), (4,), (0.5,), (True,)])
def test_invalid_or_unpaired_channel_selector_rejects(selected):
    with pytest.raises(ValueError):
        probe.paired_rows(4, selected)


def toy_items():
    return [{"id": str(i), "role": "mlp_gate_up", "rows": 6, "columns": 2,
             "prefill_rows": p, "prefill_macs": p * 12, "decode_macs": 12,
             "weight_bytes": 12, "scale_bytes": 24, "output_correction_bytes": gain,
             "ingress_bytes": 7}
            for i, (p, gain) in enumerate(((3, 60), (3, 90), (1, 30)))]


@pytest.mark.parametrize("minimum", [0.9, 0.8, 0.7, 0.5])
@pytest.mark.parametrize("horizon", [1, 4])
def test_pair_selector_matches_exhaustive_linear_objective(minimum, horizon):
    items = toy_items()
    dp = (1 - minimum + 1e-12) * sum(i["decode_macs"] for i in items)
    pp = (1 - minimum + 1e-12) * sum(i["prefill_macs"] for i in items)

    def value(counts):
        return sum(n * (horizon * i["output_correction_bytes"] / 3 - 12) for i, n in zip(items, counts, strict=True))

    candidates = [counts for counts in itertools.product(range(4), repeat=3)
                  if sum(4 * n for n in counts) <= dp
                  and sum(4 * n * i["prefill_rows"] for i, n in zip(items, counts, strict=True)) <= pp]
    chosen = probe.choose_pairs(items, minimum, horizon)
    assert value([chosen.get(i["id"], 0) for i in items]) == pytest.approx(max(map(value, candidates)))


@pytest.mark.parametrize("minimum", [0.9, 0.7, 0.5, 0.0])
def test_whole_group_selector_matches_independent_subset_oracle(minimum):
    items = toy_items()
    actual = probe.whole_groups(items, minimum, 4)
    expected = max(sum(4 * (i["output_correction_bytes"] + i["ingress_bytes"]) - i["weight_bytes"] - i["scale_bytes"] for i in subset)
                   for mask in itertools.product((False, True), repeat=3)
                   for subset in [[i for i, yes in zip(items, mask, strict=True) if yes]]
                   if sum(i["prefill_macs"] for i in subset) <= (1 - minimum + 1e-12) * sum(i["prefill_macs"] for i in items)
                   and sum(i["decode_macs"] for i in subset) <= (1 - minimum + 1e-12) * sum(i["decode_macs"] for i in items))
    assert actual["net_bytes_at_horizon"] == expected


def test_partial_output_ownership_never_halves_ingress():
    items = toy_items()
    partial = probe.price_selection(items, {"0": 1}, loss=0, frame_per_row=0)
    assert partial["retained_ingress_bytes_per_pair"] == 21
    assert partial["removed_ingress_bytes_per_pair"] == 0
    assert partial["gross_correction_reply_bytes_per_pair"] == 20
    assert partial["weight_bytes"] == 4
    assert partial["scale_bytes"] == 8
    assert partial["map_bytes"] == 4
    assert probe.price_selection(items, {"0": 3})["removed_ingress_bytes_per_pair"] == 7


def test_amortization_warms_control_and_charges_bootstrap_and_overhead():
    result = probe.amortization(100, 400, 200, 120, 10, 0.2)
    assert result == 8
    for r in (result - 1, result):
        control = 100 + r * 400
        candidate = 100 + 200 + r * (400 - 120 + 10)
        assert ((control - candidate) / control >= 0.2) == (r == result)
    assert probe.amortization(100, 400, 200, 90, 10, 0.2) is None


def test_synthetic_post_stitch_identity():
    assert probe.tiny_algebra()["integer_dequant_silu_multiply_bit_identity"]


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_fresh_mask_remote_ring_algebra_with_independent_signed_oracle(bits):
    rng = np.random.default_rng(77)
    x = np.array([[-128, 127], [127, -128]], dtype=np.int8)
    w = np.array([[-128, 127], [127, -128], [1, 0]], dtype=np.int8)
    modulus = 1 << bits
    expected = np.array([[sum(int(a) * int(b) for a, b in zip(row, weight, strict=True)) % modulus
                          for weight in w] for row in x])
    expected = np.where(expected >= modulus // 2, expected - modulus, expected)
    for _ in range(3):
        r = rng.integers(0, modulus, size=x.shape, dtype=np.int64)
        s = rng.integers(0, modulus, size=(2, 3), dtype=np.int64)
        np.testing.assert_array_equal(probe.masked_remote_integer(x, w, r, s, bits), expected)


def test_frozen_evidence_phase_budgets_and_pair_scope():
    path = probe.ROOT / "docs/evidence/research-bootstrap-placement.json"
    if not path.exists():
        pytest.skip("evidence not generated yet")
    document = json.loads(path.read_text())
    assert document["complete"]
    for row in document["remote_budget_frontier"]:
        assert min(row["prefill_remote_fraction"], row["decode_remote_fraction"]) >= row["remote_minimum"] - 1e-12
        comparator = row["whole_group_comparator"]
        assert min(comparator["prefill_remote_fraction"], comparator["decode_remote_fraction"]) >= row["remote_minimum"] - 1e-12
        for scenario in row["amortization_sensitivities"]:
            assert scenario["C0"] + scenario["T0"] == document["control"]["pair_bytes"]
    assert document["memory"]["qualified"] is False
    assert document["algebra"]["full_logits_KV_parity"] is False


def test_compile_rejects_different_body_before_native_geometry():
    prior = json.loads(probe.PRIOR.read_text())
    geometry = copy.deepcopy(json.loads(probe.GEOMETRY.read_text()))
    geometry["source"]["body_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        probe.compile_geometry({}, None, prior["control"], geometry)
