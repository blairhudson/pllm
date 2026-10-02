"""Exact E3 integer-domain and real SEAL packing regressions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_exact_linear_he", ROOT / "scripts/probe_exact_linear_he.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_full_public_down_domain_and_insufficient_old_plaintext():
    weight = np.full((896, 4864), 127, dtype=np.int8)
    bounds = probe.public_bounds(weight)
    assert bounds["per_output_abs_bound"] == [78_451_456] * 896
    with pytest.raises(ValueError, match="entire signed domain"):
        probe.prove_lift(bounds["actual_max_abs_bound"], 1_032_193)
    lift = probe.prove_lift(bounds["actual_max_abs_bound"])
    assert lift["plain_modulus_prime_checked"]
    assert lift["batching_congruence"] == 0
    with pytest.raises(ValueError, match="entire signed domain"):
        probe.prove_lift(1 << 31)


@pytest.mark.parametrize("bound", [0, 18_993_866, 78_451_456, probe.PLAIN // 2])
def test_centered_conversion_both_boundaries(bound):
    probe.prove_lift(bound)
    for signed in (-bound, bound):
        bfv = signed % probe.PLAIN
        centered = bfv - probe.PLAIN if bfv > probe.PLAIN // 2 else bfv
        assert centered == signed
        ring = centered % (1 << 32)
        restored = ring - (1 << 32) if ring >= (1 << 31) else ring
        assert restored == signed


def test_fail_closed_domains_and_prime():
    with pytest.raises(ValueError, match="symmetric W8"):
        probe.public_bounds(np.full((1, 4864), -128, np.int8))
    with pytest.raises(ValueError, match="integer public matrix"):
        probe.public_bounds(np.ones((3, 4), np.float32))
    with pytest.raises(ValueError, match="batching"):
        probe.prove_lift(10, 1_000_003)
    with pytest.raises(ValueError, match="prime"):
        probe.prove_lift(10, 49_153)  # 13*19*199, still 1 mod 16384.
    with pytest.raises(ValueError, match="entire signed domain"):
        probe.prove_lift(probe.PLAIN // 2 + 1)


@pytest.mark.he
@pytest.mark.parametrize("sign", [1, -1])
def test_fresh_secret_free_full_width_boundary_and_nested_packing(sign):
    ts = pytest.importorskip("tenseal")
    client, provider, context = probe.new_contexts()
    assert not provider.has_secret_key()
    assert not provider.has_relin_keys()
    assert context["coeff_modulus_total_bits"] == 218
    # 64 full-width outputs force two pack groups. Mixed weights catch output
    # ordering; partial 768 tail catches non-power-of-two BFV dot packing bug.
    weight = np.full((64, 4864), 127, np.int8)
    weight[1::2] *= -1
    weight[2::4, ::2] *= -1
    weight[3::8] = 0
    row = np.full(4864, sign * 127, np.int8)
    requests, _ = probe.encrypt_row(client, row)
    assert len(requests) == 2
    with pytest.raises(ValueError):
        ts.bfv_vector_from(provider, requests[0]).decrypt()
    response, costs = probe.evaluate(provider, requests, weight)
    parity = probe.decrypt_and_check(client, response, weight, row)
    assert parity["verified_outputs"] == 64
    assert parity["client_measured_noise_budget_bits"][0] > 0
    assert costs["output_ciphertexts"] == 1
    assert costs["public_plaintext_dots"] == 128
    wrong_request = [ts.bfv_vector(client, [127] * 768).serialize()] * 2
    with pytest.raises(ValueError, match="padded to exactly"):
        probe.evaluate(provider, wrong_request, weight)
    with pytest.raises(ValueError, match="refuses a secret key"):
        probe.evaluate(client, requests, weight)


def test_unknown_full_shape_and_compute_cannot_be_selected():
    result = probe.comparisons({})
    assert result["region_25_percent_gate_pass"] is False
    assert result["complete_cost_bytes"] is None


def test_prepared_full_width_u32_fresh_masks_and_native_offsets():
    weight = np.full((4, 4864), 127, np.int8)
    weight[1] *= -1
    row = np.full(4864, 127, np.int8)
    lock = {
        "body_fingerprint": "a" * 64,
        "weight_i8_sha256": probe.sha(weight.tobytes()),
        "client_output_scale_bytes": 16,
    }
    control = probe.prepared_control(weight, row, lock, 78_451_456)
    assert control["exact_parity"]
    assert control["ring"]["wire_bits"] == 32
    assert control["online_input_body_bytes"] > 4864 * 4
    assert control["offline_correction_body_bytes"] > 4 * 4
    assert control["client_dense_weight_bytes"] == 0
    assert control["two_offset_complete_protocol_cpu_seconds"] is None


def test_archived_real_shape_resource_gate_and_evidence_hashes():
    screen_path = ROOT / "docs/evidence/exact-linear-he-screen-2026-10-01.json"
    screen = json.loads(screen_path.read_text())
    review = json.loads((ROOT / "docs/evidence/exact-linear-he-review-2026-10-01.json").read_text())
    assert review["screen_sha256"] == probe.file_sha(screen_path)
    assert screen["script_source_sha256"] == probe.file_sha(
        ROOT / "scripts/probe_exact_linear_he.py"
    )
    projection = screen["full_projection"]
    assert projection["actual_full_shape_execution"]
    assert projection["shape_input_by_output"] == [4864, 896]
    assert projection["verified_outputs"] == len(screen["bounds"]["per_output_abs_bound"]) == 896
    assert projection["all_outputs_exact"]
    assert projection["output_signed_min_max"][1] == screen["bounds"]["actual_max_abs_bound"]
    assert not screen["context"]["provider_has_secret_key"]
    assert screen["supervisor"]["peak_sampled_parent_plus_worker_rss_bytes"] < probe.MAX_RSS
    assert screen["supervisor"]["observed_worker_cpu_seconds"] < probe.MAX_CPU
    assert (
        projection["online_ciphertext_bytes"]
        > screen["prepared_control"]["covered_measured_stage_body_bytes"]
    )
    assert not screen["selection_admitted"]
    assert all(value is None for value in screen["unknown_costs"].values())
    assert not screen["comparison"]["region_25_percent_gate_pass"]
