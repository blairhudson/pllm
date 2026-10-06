"""Preserve complete costs, fresh held-out numerics and the promotion decision."""
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def report():
    return json.loads((ROOT / "docs/evidence/research-piecewise-gated-reference-qwen25.json").read_text())


def test_sources_profiles_and_fresh_heldout_are_bound(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.nonlinear_block_reference import PUBLIC_PROMPTS
    from benchmarks.research.piecewise_gated_reference import CONFIGURATION, PUBLIC_HELDOUT

    evidence = report()
    assert evidence["configuration"] == CONFIGURATION
    assert len(PUBLIC_HELDOUT) == len(set(PUBLIC_HELDOUT)) == 12
    assert not set(PUBLIC_HELDOUT) & set(PUBLIC_PROMPTS)
    assert evidence["numeric_gate"]["public_heldout_sha256"] == hashlib.sha256(json.dumps(PUBLIC_HELDOUT).encode()).hexdigest()
    for category in ("source_sha256", "control_sha256"):
        for path, digest in evidence[category].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for profile in evidence["profiles"].values():
        digest = hashlib.sha256(b"pllm.numeric.silu_affine_gated.reference.v1")
        digest.update(bytes((16, 14, profile["fraction"])))
        digest.update(profile["pieces"].to_bytes(2, "little"))
        for bound in (48, 128, 8):
            digest.update(bound.to_bytes(4, "little", signed=True))
        assert len(profile["intervals"]) == profile["pieces"] + 2
        assert profile["intervals"][0][0] == 0
        for start, coefficients in profile["intervals"]:
            for value in (start, *coefficients):
                digest.update(value.to_bytes(4, "little"))
        assert digest.hexdigest() == profile["digest"]


def test_native_layouts_match_independent_integers_and_price_every_round(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.piecewise_gated_reference import CONFIGURATION, oracle_digest

    evidence = report()
    indexed = {(c["fraction"], c["pieces"], c["layout"], c["lanes"]): c for c in evidence["cases"]}
    assert set(indexed) == set(itertools.product(*(CONFIGURATION[k] for k in ("fractions", "pieces", "layouts", "lanes"))))
    assert sum(c["checked_outputs"] for c in indexed.values()) == 1560
    for (fraction, pieces, layout, lanes), case in indexed.items():
        profile = evidence["profiles"][f"q{fraction}-{pieces}"]
        assert case["profile_digest"] == profile["digest"]
        assert case["output_sha256"] == oracle_digest(profile, lanes)
        assert case["samples"] == 3 and case["ring_bits"] == 32
        assert case["rounds"] == 6
        # Four input/activation/output rescalings, two Beaver openings, lookup.
        assert case["peer_frame_bytes"] == (72 if layout == "vector" else 80) * lanes + 6 * 156
        assert case["triple_key_payload_bytes"] == 24 * lanes
        assert case["party_key_payload_bytes"] == sum(case[k] for k in (
            "lookup_key_payload_bytes", "rescale_key_payload_bytes", "triple_key_payload_bytes",
        ))
        assert case["input_sharing_payload_bytes"] + case["output_sharing_payload_bytes"] == 24 * lanes
        assert 0 < case["total_allocation_estimate_bytes"] <= 64 << 20
        assert not case["executable_sdk"]
        other = indexed[fraction, pieces, "separate" if layout == "vector" else "vector", lanes]
        assert case["output_sha256"] == other["output_sha256"]
        assert case["rescale_key_payload_bytes"] == other["rescale_key_payload_bytes"]
        if layout == "vector":
            assert case["lookup_key_payload_bytes"] < other["lookup_key_payload_bytes"]


def test_quality_requires_every_position_and_preserves_full_kv_differences():
    evidence = report()
    numeric = evidence["numeric_gate"]
    assert not numeric["protected_checkpoint_execution"]
    for name, summary in numeric["summary"].items():
        rows = [row for row in numeric["trials"] if row["candidate"] == name]
        positions = [position for row in rows for position in row["positions"]]
        assert len(rows) == summary["attempted_prompts"] == 12
        assert summary["completed_positions"] == len(positions)
        assert summary["screen_pass"] == (len(positions) == 48 and all(p["selection_match"] for p in positions))
        assert summary["exact_kv_positions"] == sum(p["kv_bit_exact"] for p in positions)
        assert summary["bit_exact_positions"] == sum(p["all_logits_bit_exact"] for p in positions)
        if name != "control":
            assert summary["native_oracle_checked_elements"] > 0
            decision = evidence["candidate_decisions"][name]
            assert decision["numeric_screen_pass"] == summary["screen_pass"]
    control = numeric["summary"]["control"]
    assert (control["prefill_matches"], control["decode_matches"], control["exact_kv_positions"]) == (12, 36, 48)


def test_geometry_conserves_material_all_peer_frames_and_go_no_go():
    evidence = report()
    indexed = {(c["fraction"], c["pieces"], c["layout"], c["lanes"]): c for c in evidence["cases"]}
    for projection in evidence["geometry_projections"]:
        count = projection["gated_elements"]
        assert count == 24 * 4864 * (39 + projection["output_tokens"] - 1)
        assert not projection["executable_decoder"]
        for case in projection["cases"]:
            measured = indexed[case["fraction"], case["pieces"], case["layout"], case["batch_lanes"]]
            assert case["both_party_material_payload_bytes"] == 2 * measured["party_key_payload_bytes"] * count // measured["lanes"]
            assert case["peer_frame_bytes"] == measured["peer_frame_bytes"] * count // measured["lanes"]
            assert case["input_output_sharing_payload_bytes_if_not_resident"] == 24 * count
            assert case["material_plus_peer_bytes"] == case["both_party_material_payload_bytes"] + case["peer_frame_bytes"]
            assert case["cost_gate_pass"] == (case["material_plus_peer_bytes"] < projection["historical_prepared_covered_bytes"])
    for name, decision in evidence["candidate_decisions"].items():
        profile = evidence["profiles"][name]
        expected = all(any(case["cost_gate_pass"] and (case["fraction"], case["pieces"]) == (profile["fraction"], profile["pieces"])
                           for case in projection["cases"]) for projection in evidence["geometry_projections"])
        assert decision["cost_screen_pass"] == expected
        assert decision["further_integration_justified"] == (expected and decision["numeric_screen_pass"])
    assert evidence["decision"] == "no-go"
    assert not any(d["further_integration_justified"] for d in evidence["candidate_decisions"].values())
    for key in ("executable_sdk", "privacy_reviewed", "whole_generation_quality"):
        assert not evidence[key]
    assert evidence["full_wire_bytes"] is None
