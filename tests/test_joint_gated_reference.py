"""Audit fresh numeric identity, exact arithmetic and pre-issuance cost vetoes."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def report():
    return json.loads((ROOT / "docs/evidence/research-joint-gated-reference-qwen25.json").read_text())


def test_frozen_profiles_sources_and_fresh_prompt_cohort(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.joint_gated_reference import CONFIGURATION, PUBLIC_HELDOUT
    from benchmarks.research.nonlinear_block_reference import PUBLIC_PROMPTS
    from benchmarks.research.piecewise_gated_reference import PUBLIC_HELDOUT as OLD_HELDOUT

    r = report()
    assert r["configuration"] == CONFIGURATION
    assert len(PUBLIC_HELDOUT) == len(set(PUBLIC_HELDOUT)) == 12
    assert not set(PUBLIC_HELDOUT) & (set(PUBLIC_PROMPTS) | set(OLD_HELDOUT))
    assert r["numeric_gate"]["public_heldout_sha256"] == hashlib.sha256(json.dumps(PUBLIC_HELDOUT).encode()).hexdigest()
    for category in ("source_sha256", "control_sha256"):
        for path, digest in r[category].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert set(r["profiles"]) == {f"q{f}-{p}" for f, p in itertools.product((12, 16), (512, 2048))}
    for profile in r["profiles"].values():
        h = hashlib.sha256(b"pllm.numeric.joint_silu_gated.reference.v1")
        h.update(bytes((profile["fraction"], 28)))
        h.update(profile["pieces"].to_bytes(2, "little"))
        for value in (48, 128, 16):
            h.update(value.to_bytes(4, "little", signed=True))
        assert len(profile["intervals"]) == profile["pieces"] + 2
        assert 2**63 < int(profile["maximum_numerator_absolute"]) < 2**80
        for start, coefficients in profile["intervals"]:
            h.update(start.to_bytes(4, "little", signed=True))
            for coefficient in coefficients:
                h.update(coefficient.to_bytes(8, "little", signed=True))
        assert h.hexdigest() == profile["digest"]


def test_segmented_wide_oracle_against_unbounded_integers(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.joint_gated_reference import integer_oracle, scalar_oracle

    for profile in report()["profiles"].values():
        scale = 1 << profile["fraction"]
        # Boundaries, signed endpoints and deterministic unrelated interior values.
        pairs = [(start + delta, u) for start, _ in profile["intervals"]
                 for delta in (-1, 0, 1) if abs(start + delta) <= 48 * scale
                 for u in (-128 * scale, -3, 1, 128 * scale)]
        pairs += [((i * 7919) % (96 * scale + 1) - 48 * scale,
                   (i * 104729) % (256 * scale + 1) - 128 * scale) for i in range(1000)]
        g, u = np.asarray(pairs, dtype=np.int64).T
        expected = np.array([scalar_oracle(profile, int(a), int(b)) for a, b in pairs])
        np.testing.assert_array_equal(integer_oracle(profile, g, u), expected)
        with pytest.raises(ValueError, match="profile domain"):
            integer_oracle(profile, np.array([np.iinfo(np.int64).min]), np.array([0]))
        boundary = report()["boundary_checks"][f'q{profile["fraction"]}-{profile["pieces"]}']
        assert boundary["checked_outputs"] > profile["pieces"] * 15
        assert boundary["non_affinity_xor"] != 0
        assert boundary["non_affinity_values"][:3] == [0, 0, 0]


def test_complete_trajectory_decisions_retain_logits_and_kv_differences():
    r = report()
    gate = r["numeric_gate"]
    for name, summary in gate["summary"].items():
        rows = [row for row in gate["trials"] if row["candidate"] == name]
        positions = [p for row in rows for p in row["positions"]]
        assert len(rows) == summary["attempted_prompts"] == 12
        assert summary["completed_positions"] == len(positions)
        assert summary["screen_pass"] == (len(positions) == 48 and all(p["selection_match"] for p in positions))
        assert summary["prefill_matches"] == sum(p["selection_match"] for p in positions if p["position"] == 0)
        assert summary["decode_matches"] == sum(p["selection_match"] for p in positions if p["position"] > 0)
        assert summary["exact_kv_positions"] == sum(p["kv_bit_exact"] for p in positions)
        assert summary["bit_exact_positions"] == sum(p["all_logits_bit_exact"] for p in positions)
        if name != "control":
            assert summary["native_oracle_checked_elements"] == sum(row["domain"]["elements"] for row in rows)
            assert r["candidate_decisions"][name]["numeric_screen_pass"] == summary["screen_pass"]
    control = gate["summary"]["control"]
    assert (control["prefill_matches"], control["decode_matches"], control["exact_kv_positions"]) == (12, 36, 48)
    assert not gate["protected_checkpoint_execution"]


def test_protocol_cost_gate_prices_both_parties_and_preserves_unknowns(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.joint_gated_reference import protocol_costs

    r = report()
    for row in r["geometry_projections"]:
        count = row["gated_elements"]
        assert count == 24 * 4864 * (39 + row["output_tokens"] - 1)
        baseline = row["historical_prepared_covered_bytes"]
        assert row["total_bytes_per_lane_budget"] == baseline / count
        for name, costs in row["profiles"].items():
            p = r["profiles"][name]
            assert costs == protocol_costs(p, count, baseline)
            assert costs["table_entries_per_lane"] == 2 ** (2 * p["fraction"] + 16)
            mask_bytes = 6 if p["fraction"] == 12 else 7
            dense = costs["dense_function_shares"]
            assert dense["both_party_table_and_mask_payload_bytes"] == (8 * costs["table_entries_per_lane"] + 2 * mask_bytes) * count
            assert dense["peer_opening_payload_bytes"] == 2 * mask_bytes * count
            assert dense["input_output_sharing_payload_bytes_if_not_resident"] == (2 * mask_bytes + 8) * count
            assert not dense["allocated"] and not dense["cost_gate_pass"]
            circuit = costs["uncompressed_half_gates"]
            assert circuit["ciphertext_body_floor_bytes"] == 32 * count > baseline
            assert circuit["cost_gate_ruled_out_by_floor"]
            assert circuit["complete_material_and_peer_bytes"] is None
            hypothetical = costs["hypothetical_compressed_function_shares"]
            assert hypothetical["remaining_material_budget_bytes"] + hypothetical["peer_opening_payload_bytes"] == baseline
            assert hypothetical["cost_gate_pass"] is None
            assert not hypothetical["construction_available"]
    assert r["decision"] == "no-go"
    for decision in r["candidate_decisions"].values():
        assert not decision["further_integration_justified"]
        assert not decision["dense_cost_screen_pass"]
        assert decision["half_gates_ruled_out_by_floor"]
    for key in ("protected_prototype_issued", "executable_sdk", "privacy_reviewed", "whole_generation_quality"):
        assert not r[key]


def test_algebra_witnesses_reject_missing_carry_and_public_coefficients():
    audit = report()["algebra_audit"]
    assert audit["checked_masked_pairs"] == 4096
    assert 0 < audit["missing_wrap_wrong_outputs"] < 4096
    assert audit["sharewise_rounding"] == [1, 0]
    assert audit["recovered_gate_up"] == [3, 5]
    assert not audit["protected_execution"]
