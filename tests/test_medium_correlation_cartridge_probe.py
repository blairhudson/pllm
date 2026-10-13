"""Independent algebra/accounting claims; never imports the native SDK."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import random

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("medium_probe", ROOT / "scripts/probe_medium_correlation_cartridge.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


@pytest.mark.parametrize("bits", [3, 16, 24, 32])
def test_direct_independent_signed_oracle(bits):
    w = [[-128, 127], [32, -16]]
    x = [-128, 127]
    expected = [-128 * -128 + 127 * 127, 32 * -128 - 16 * 127]
    for r in ([0, 0], [(1 << bits) - 1, 17]):
        got = p.direct_control(w, x, r, [7, 13], bits, [bits, max(1, bits - 2)])
        assert got == [expected[0] % (1 << bits), expected[1] % (1 << max(1, bits - 2))]


def test_exhaustive_direct_not_using_modular_oracle():
    for x, r, q in itertools.product(range(8), repeat=3):
        assert p.direct_control([[3]], [x], [r], [q], 3, [3]) == [3 * x % 8]


@pytest.mark.parametrize("bits", [3, 16, 24, 32])
def test_sequential_columns_single_triple(bits):
    w = [[-128, 127, 1], [3, 0, -8]]
    xs = [[-128, 127, -2], [127, -128, 4], [3, 2, 1]]
    got = p.bounded_column_control(w, xs, bits, 3, random.Random(9))
    expected = [[sum(row[j] * x[j] for j in range(3)) % (1 << bits) for row in w] for x in xs]
    assert got == expected
    with pytest.raises(ValueError, match="column capacity"):
        p.bounded_column_control(w, xs, bits, 2, random.Random(9))


@pytest.mark.parametrize("bits", [0, 33, True])
def test_strict_ring(bits):
    with pytest.raises(ValueError):
        p.mv([[1]], [1], bits)


@pytest.mark.parametrize("widths", [[], [0], [33], [True]])
def test_strict_outputs(widths):
    with pytest.raises(ValueError):
        p.direct_control([[1]], [1], [2], [3], 32, widths)


def test_malformed_algebra_dimensions():
    with pytest.raises(ValueError):
        p.mv([[1, 2]], [1], 24)
    with pytest.raises(ValueError):
        p.direct_control([[1]], [1, 2], [2], [3], 24, [24])


@pytest.fixture(scope="module")
def geometry():
    return p.geometry()


def test_saved_totals_independent(geometry):
    _, stages = geometry
    t = p.totals(stages)
    assert t["weight_i8_bytes"] == 357826560
    assert t["prefill_body_macs"] + t["decode_body_macs"] == 56615845888
    assert t["encoded_ingress_bytes"] == 90104112
    assert t["residue_reply_bytes"] == 138864059
    assert t["prefill_body_macs"] == sum(a["shape"][0] * a["shape"][1] * (a["rows"] - 14) for a in stages)
    assert sum(s["rows"] for s in stages) == 15297


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "zero_dim", "bool_dim", "rows", "histogram", "ring", "macs"])
def test_strict_saved_geometry(geometry, mutation):
    _, original = geometry
    stages = copy.deepcopy(original)
    if mutation == "duplicate":
        stages[-1] = copy.deepcopy(stages[0])
    elif mutation == "missing":
        stages.pop()
    elif mutation == "zero_dim":
        stages[0]["shape"][0] = 0
    elif mutation == "bool_dim":
        stages[0]["shape"][0] = True
    elif mutation == "rows":
        stages[0]["rows"] = 14
    elif mutation == "histogram":
        stages[0]["row_residue_width_histogram"]["24"] = True
    elif mutation == "ring":
        stages[0]["original_ring_bits"] = 31
    else:
        stages[0]["body_macs"] += 1
    with pytest.raises(ValueError):
        p.validate_stages(stages)


def test_source_lock_rejects_before_import(tmp_path):
    record = p.load(ROOT / "docs/evidence/research-matrix-triple-prefill.json")
    evidence = tmp_path / "docs/evidence"
    evidence.mkdir(parents=True)
    (evidence / "research-matrix-triple-prefill.json").write_text(json.dumps(record))
    (evidence / "research-all-link-frontier.json").write_text("{}")
    with pytest.raises(ValueError, match="historical source hash mismatch"):
        p.geometry(tmp_path)


def test_duplicate_json_keys():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads('{"x": 1, "x": 2}', object_pairs_hook=p.unique_object)


@pytest.mark.parametrize("h", [1, 4, 16, 1024])
def test_all_link_ledger_independent(h, geometry):
    control, _ = geometry
    result = p.ledger(control, h, 139060689, 123, 456, 789)
    baseline = 124905091 + h * 368670156
    expected = 124905091 + h * (368670156 - 139060689 + 789) + 123 + 456
    assert result["control_bytes"] == baseline
    assert result["candidate_conditional_bytes"] == expected
    assert result["remaining_20pct_budget_times_five"] == 4 * baseline - 5 * expected
    assert result["conditional_20pct_survives"] == (expected * 5 <= baseline * 4)


def test_direct_renewal_delivery_unused_and_work(geometry):
    control, stages = geometry
    result = p.direct_projection(control, stages, 2, capacity=165)
    # Ordinary stage uses 328 => 2 epochs, terminal 30 => 1 epoch.
    assert result["stage_epochs"] == 93 * 2 + 3
    assert result["both_role_issuance_bytes"] == (93 * 2 + 3) * (64 + 32)
    assert result["public_manifest_deliveries_all_stage_epochs_bytes"] == (93 * 2 + 3) * 192
    assert result["unused_indices_at_final_retirement"] == 93 * 2 + 3 * 135
    assert result["extra_client_Wr_macs"] == 2 * 56615845888
    assert result["extra_client_Wr_macs_prefill"] + result["extra_client_Wr_macs_decode"] == result["extra_client_Wr_macs"]
    assert result["remote_original_body_fraction_prefill"] == result["remote_original_body_fraction_decode"] == 1


def test_burned_attempts_trigger_renewal(geometry):
    control, stages = geometry
    before = p.direct_projection(control, stages, 1, capacity=164)
    after = p.direct_projection(control, stages, 1, capacity=164, burned_per_stage=1)
    assert after["stage_epochs"] - before["stage_epochs"] == 93
    assert after["both_role_issuance_bytes"] - before["both_role_issuance_bytes"] == 93 * 96
    assert after["burned_attempts"] == 96
    assert after["failed_attempt_body_and_expansion_costs"] is None


def test_storage_program_and_PRG_counts_separate(geometry):
    _, stages = geometry
    result = p.logical_resources(stages)
    assert result["persistent_public_weight_bytes"] == 357826560
    assert result["private_program_root_bytes_all_stages"] == 6144
    assert result["client_PRG_full_ring_output_bytes_per_pair"] == 252493440
    assert result["inference_PRG_full_ring_output_bytes_per_pair"] == 147874688
    assert result["whole_client_peak_RSS_bytes"] is None


def test_locally_existing_weights_do_not_make_work_free(geometry):
    control, stages = geometry
    delivered = p.direct_projection(control, stages, 6)
    local = p.direct_projection(control, stages, 6, weights_preexisting_client=True)
    assert delivered["candidate_conditional_bytes"] - local["candidate_conditional_bytes"] == 357826560
    assert local["extra_client_Wr_macs"] == delivered["extra_client_Wr_macs"]
    assert local["public_weight_delivery_client_bytes"] == 0
    assert local["earlier_checkpoint_delivery_if_preexisting_not_in_control_bytes"] is None


def test_direct_widths_change_20pct_gate(geometry):
    control, stages = geometry
    narrow = p.direct_projection(control, stages, 6, quotient=True)
    full = p.direct_projection(control, stages, 6, quotient=False)
    assert narrow["conditional_20pct_survives"]
    assert not full["conditional_20pct_survives"]
    assert full["candidate_conditional_bytes"] > narrow["candidate_conditional_bytes"]
    assert p.direct_projection(control, stages, 10, quotient=False)["conditional_20pct_survives"]
    assert full["new_recurring_bytes_per_pair"] == 23525269
    assert not any(p.direct_projection(control, stages, h)["conditional_20pct_survives"] for h in range(1, 6))
    assert not any(p.direct_projection(control, stages, h, quotient=False)["conditional_20pct_survives"] for h in range(1, 10))


def test_bounded_128_openings_and_underfill(geometry):
    control, stages = geometry
    result = p.bounded_projection(control, stages, 1, columns=128)
    # Ordinary 164 columns => 2 triples; terminal 15 => 1. Sequential decode uses leftovers.
    assert result["triples_or_rectangular_batches"] == 93 * 2 + 3
    assert result["unused_columns_retired"] == 93 * (256 - 164) + 3 * (128 - 15)
    expected = sum(2 * ((s["rows"] + 127) // 128) * ((s["shape"][1] * sum(int(k) * v for k, v in s["row_residue_width_histogram"].items()) + 7) // 8) for s in stages)
    assert result["D_both_role_deliveries_bytes"] == expected
    assert result["extra_client_dense_adaptation_macs"] == 56615845888
    assert not result["conditional_20pct_survives"]


def test_column_renewal_not_static_A(geometry):
    control, stages = geometry
    a = p.bounded_projection(control, stages, 1, columns=256)
    b = p.bounded_projection(control, stages, 1024, columns=256)
    assert b["D_both_role_deliveries_bytes"] > 500 * a["D_both_role_deliveries_bytes"]
    assert not b["conditional_20pct_survives"]


def test_square_block_specific_ingress_and_both_role_generation(geometry):
    control, stages = geometry
    result = p.bounded_projection(control, stages, 1, 128, True)
    block_rows = sum(((s["shape"][0] + 127) // 128) * ((s["shape"][1] + 127) // 128) * s["rows"] for s in stages)
    assert result["ingress_bytes"] == result["reply_bytes"] == block_rows * 128 * 8
    count = result["triples_or_rectangular_batches"]
    assert result["both_role_generated_matrix_bytes"] == 6 * count * 128**2 * 8
    assert result["padded_each_role_adaptation_macs"] == count * 128**3
    assert result["estimated_dealer_both_role_seed_delivery_bytes"] == 2 * result["PCG_epochs_4096_triples"] * 4096 * 5453
    assert result["candidate_with_estimated_dealer_seed_delivery_bytes"] == result["candidate_conditional_bytes"] + result["estimated_dealer_both_role_seed_delivery_bytes"]
    assert result["natural_per_party_workspace_bytes"] == 2 * 2**30
    assert not result["qualifies_Medium"]


def test_algebra_claim_counts():
    result = p.algebra()
    assert result["bounded_column_outputs"] == result["direct_quotient_outputs"] == 84
    assert result["exhaustive_inference_views"] == 512
    assert not result["secure_pcg_implemented"]


@pytest.mark.parametrize("horizon", [1, 16, 1024])
def test_256_square_does_not_borrow_published_128_costs(geometry, horizon):
    control, stages = geometry
    result = p.bounded_projection(control, stages, horizon, 256, True)
    assert not result["published_setup_workspace_parameter_point"]
    for name in (
        "PCG_epochs_4096_triples", "unused_PCG_triples_at_retirement",
        "estimated_dealer_both_role_seed_delivery_bytes",
        "estimated_distributed_setup_MiB_combined", "natural_per_party_workspace_bytes",
        "candidate_with_estimated_dealer_seed_delivery_bytes",
        "candidate_with_estimated_distributed_setup_bytes",
        "dealer_delivery_client_estimate_bytes", "dealer_delivery_inference_estimate_bytes",
    ):
        assert result[name] is None, name
    count = sum(((s["shape"][0] + 255) // 256) * ((s["shape"][1] + 255) // 256)
                * ((horizon * s["rows"] + 255) // 256) for s in stages)
    assert result["triples_or_rectangular_batches"] == count
    assert result["D_both_role_deliveries_bytes"] == 2 * count * 256**2 * 8
    assert result["both_role_generated_matrix_bytes"] == 6 * count * 256**2 * 8
    assert not result["conditional_20pct_survives"]
