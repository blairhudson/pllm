"""Paper-derived protocol evidence keeps numerics, material and scope distinct."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def report():
    return json.loads((ROOT/"docs/evidence/research-masked-wire-reference-qwen25.json").read_text())


def test_source_locked_fresh_numeric_cohort(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.masked_wire_reference import CONFIGURATION, PUBLIC_HELDOUT, paper_sources
    from benchmarks.research.joint_gated_reference import PUBLIC_HELDOUT as joint
    from benchmarks.research.piecewise_gated_reference import PUBLIC_HELDOUT as piecewise
    from benchmarks.research.nonlinear_block_reference import PUBLIC_PROMPTS as initial

    r = report()
    assert r["configuration"] == CONFIGURATION
    assert r["papers"] == paper_sources()
    assert len(set(PUBLIC_HELDOUT)) == 12
    assert not set(PUBLIC_HELDOUT) & (set(joint) | set(piecewise) | set(initial))
    assert r["numeric_gate"]["public_heldout_sha256"] == hashlib.sha256(json.dumps(PUBLIC_HELDOUT).encode()).hexdigest()
    for category in ("source_sha256", "control_sha256"):
        for path, digest in r[category].items():
            if path == "crates/pllm-garble/src/masked_wire_reference.rs":
                # Historical pre-fusion source identity stays immutable. The new
                # fused/unfused cohort and native tests exercise today's source.
                assert digest == "29b9daea35c6676e57d888e7aa8804b25331ded6488cd1f57e096b82e66b1f7d"
                continue
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest


def test_public_residual_contract_and_floor_asymmetry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.masked_wire_reference import integer_oracle

    p = report()["profile"]
    assert p["table"] == [math.floor(4096*(i/64)/(1+math.exp(i/64))) for i in range(1024)]
    assert p["table"][0] == p["table"][-1] == 0
    assert max(p["table"]) <= 1141
    h = hashlib.sha256(b"pllm.residual_silu_reference.sigma_gpu_order.q12.v1\0")
    h.update(bytes([12, 40, 19]))
    for v in (48, 128, *p["table"]):
        h.update(v.to_bytes(8, "little", signed=True))
    assert h.hexdigest() == p["digest"]
    # The source moves truncation before abs; floor does not commute with it.
    assert integer_oracle(p, np.array([-1]), np.array([2**24]))[0] < 0
    assert integer_oracle(p, np.array([1]), np.array([2**24]))[0] == 0
    for bad in (-(48 << 24)-1, (48 << 24)+1, np.iinfo(np.int64).min):
        with pytest.raises(ValueError, match="profile domain"):
            integer_oracle(p, np.array([bad]), np.array([0]))


def test_fresh_protected_cases_match_independent_integer_oracle(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.masked_wire_reference import integer_oracle, public_values

    r = report()
    assert len(r["native_cases"]) == 12
    assert sum(c["checked_outputs"] for c in r["native_cases"]) == 5136
    for case in r["native_cases"]:
        lanes = case["lanes"]
        expected = integer_oracle(r["profile"], public_values(lanes, 48), public_values(lanes, 128)[::-1])
        assert hashlib.sha256(expected.astype("<i8").tobytes()).hexdigest() == case["output_sha256"]
        assert case["profile_digest"] == r["profile"]["digest"]
        assert case["combined_allocation_estimate_bytes"] <= 64*2**20
        assert case["checked_outputs"] == lanes*4
        for key in ("issuance_ns", "online_ns", "clear_ns"):
            # A single clear scalar can finish below the clock's resolution.
            assert len(case[key]) == 4 and min(case[key]) >= 0
            if key != "clear_ns":
                assert min(case[key]) > 0


def test_complete_raw_key_and_phase_accounting():
    r = report()
    # Paper construction with independent 128-bit seeds and separate flags.
    def dpf(n):
        return 2**n if n <= 7 else 256+130*(n-7)
    assert (dpf(7)+3*32)//8 == 28
    bits_per_party = (3*(dpf(12)+3*40) + dpf(6)+2*13 + dpf(12)+2
                      + 8*10 + 4*40 + dpf(10)+2*40 + 3*40)
    assert bits_per_party == 5162
    phases = (2, 80, 1, 13, 2, 10, 40, 40, 40, 1, 40)
    for case in r["native_cases"]:
        n = case["lanes"]
        assert case["rounds"] == len(phases)
        assert case["party_key_bits"] == n*bits_per_party
        assert case["party_key_payload_bytes"] == math.ceil(n*bits_per_party/8)
        assert case["peer_payload_bits"] == 2*n*sum(phases)
        assert case["peer_frame_bytes"] == 2*sum(78+math.ceil(n*p/8) for p in phases)
    for row in r["geometry_projections"]:
        n = row["gated_elements"]
        assert n == 24*4864*(39+row["output_tokens"]-1)
        assert row["both_party_raw_key_payload_bytes"] == 2*math.ceil(n*bits_per_party/8)
        assert row["per_party_raw_key_payload_bytes"]*2 == row["both_party_raw_key_payload_bytes"]
        assert row["peer_frame_bytes"] == 2*n*sum(phases)//8+2*11*78*row["bounded_256_lane_batches"]
        assert row["client_entry_exit_payload_bytes_if_not_resident"] == 25*n
        assert row["upstream_protected_float_conversion_bytes"] is None
        assert row["whole_decoder_bytes"] is None
        assert row["cost_gate_pass"] == (row["both_party_raw_key_payload_bytes"]+row["peer_frame_bytes"] < row["historical_prepared_covered_bytes"])
        assert not row["executable_decoder"]


def test_quality_and_claims_preserve_every_failed_position():
    r = report()
    gate = r["numeric_gate"]
    for name, summary in gate["summary"].items():
        rows = [row for row in gate["trials"] if row["candidate"] == name]
        positions = [p for row in rows for p in row["positions"]]
        assert len(rows) == 12
        assert summary["completed_positions"] == len(positions)
        assert summary["prefill_matches"] == sum(p["selection_match"] for p in positions if p["position"] == 0)
        assert summary["decode_matches"] == sum(p["selection_match"] for p in positions if p["position"] > 0)
        assert summary["screen_pass"] == (len(positions) == 48 and all(p["selection_match"] for p in positions))
        assert summary["exact_kv_positions"] == sum(p["kv_bit_exact"] for p in positions)
        if name != "control":
            assert summary["native_oracle_checked_elements"] == sum(row["domain"]["elements"] for row in rows)
    assert gate["summary"]["control"]["bit_exact_positions"] == 48
    assert gate["summary"]["control"]["exact_kv_positions"] == 48
    assert not gate["protected_checkpoint_execution"]
    assert all(not r[k] for k in ("executable_sdk", "privacy_reviewed", "protected_checkpoint_execution", "distributed_dealer"))
    assert r["full_wire_bytes"] is None
