"""Arithmetic/one-use regressions and honest five-idea feasibility evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.interaction_reduction_reference import (
    InteractionReferenceError,
    fresh_shares,
    issue_broadcast,
    issue_lookup,
    issue_sum_squares,
)

from test_shared_resources import CONFIG

_ROOT = Path(__file__).parents[1]
_EVIDENCE = _ROOT / "docs/evidence/interaction-reduction-screen-qwen25-2026-09-30.json"
_SPEC = importlib.util.spec_from_file_location(
    "interaction_probe", _ROOT / "scripts/probe_interaction_reductions.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)


@pytest.mark.parametrize("width", [1, 17, 896])
def test_specialized_square_and_scalar_product_match_independent_ring_arithmetic(
    width: int,
) -> None:
    value = (np.arange(width, dtype=np.int64) % 255 - 127).astype(np.uint32)
    shares = fresh_shares(value)
    materials = issue_sum_squares("session", "statistic", width)
    frames = [material.open(share) for material, share in zip(materials, shares, strict=True)]
    outputs = [materials[0].finish(frames[1]), materials[1].finish(frames[0])]
    assert sum(outputs) % (1 << 32) == sum(int(x) ** 2 for x in value) % (1 << 32)
    with pytest.raises(InteractionReferenceError, match="consumed"):
        materials[0].finish(frames[1])
    with pytest.raises(InteractionReferenceError, match="consumed"):
        materials[1].open(shares[1])

    scalar = np.asarray([0xFFFFFFFF], dtype=np.uint32)
    scalars = fresh_shares(scalar)
    triples = issue_broadcast("session", "normalize", width)
    opened = [
        triple.open(share, factor)
        for triple, share, factor in zip(triples, shares, scalars, strict=True)
    ]
    products = [triples[0].finish(opened[1]), triples[1].finish(opened[0])]
    assert np.array_equal(np.add(*products, dtype=np.uint32), value * scalar)
    assert sum(len(frame.body) for frame in opened) == 2 * (width + 1) * 4
    assert sum(triple.body_bytes for triple in triples) == (4 * width + 2) * 4
    with pytest.raises(InteractionReferenceError, match="consumed"):
        triples[0].finish(opened[1])


def test_reduction_burns_on_bad_input_cross_session_bad_frame_and_cancel() -> None:
    materials = issue_sum_squares("a", "square", 2)
    with pytest.raises(InteractionReferenceError, match="uint32"):
        materials[0].open(np.zeros(2, dtype=np.int32))
    with pytest.raises(InteractionReferenceError, match="consumed"):
        materials[0].open(np.zeros(2, dtype=np.uint32))
    frame = materials[1].open(np.zeros(2, dtype=np.uint32))
    with pytest.raises(InteractionReferenceError, match="binding"):
        materials[1].finish(replace(frame, party=0, session="b"))
    with pytest.raises(InteractionReferenceError, match="consumed"):
        materials[1].finish(replace(frame, party=0))

    triples = issue_broadcast("a", "product", 2)
    frame = triples[0].open(np.zeros(2, dtype=np.uint32), np.zeros(1, dtype=np.uint32))
    with pytest.raises(InteractionReferenceError, match="size"):
        triples[0].finish(replace(frame, party=1, body=frame.body[:-1]))
    with pytest.raises(InteractionReferenceError, match="consumed"):
        triples[0].finish(replace(frame, party=1))
    triples[1].cancel()
    with pytest.raises(InteractionReferenceError, match="consumed"):
        triples[1].open(np.zeros(2, dtype=np.uint32), np.zeros(1, dtype=np.uint32))
    with pytest.raises(InteractionReferenceError, match="capacity"):
        issue_sum_squares("a", "b", 4097)


@pytest.mark.parametrize("vocabulary", [2, 7, 16, 33])
def test_lookup_exhaustive_small_domains_and_padding_match_signed_table(vocabulary: int) -> None:
    table = np.random.default_rng(42).integers(-128, 128, size=(vocabulary, 5), dtype=np.int8)
    fingerprint = hashlib.sha256(memoryview(table)).digest()
    for token in range(vocabulary):
        keys = issue_lookup("a", f"q{token}", token, vocabulary, fingerprint)
        outputs = [
            key.evaluate(table, session="a", query=f"q{token}", fingerprint=fingerprint)
            for key in keys
        ]
        assert np.array_equal(np.add(*outputs, dtype=np.uint32), table[token].astype(np.uint32))
        with pytest.raises(InteractionReferenceError, match="consumed"):
            keys[0].evaluate(table, session="a", query=f"q{token}", fingerprint=fingerprint)


def test_lookup_burns_on_wrong_table_query_and_cancel_before_expansion() -> None:
    table = np.ones((7, 3), dtype=np.int8)
    fingerprint = hashlib.sha256(memoryview(table)).digest()
    first, second = issue_lookup("a", "q", 6, 7, fingerprint)
    changed = table.copy()
    changed[0, 0] = 2
    with pytest.raises(InteractionReferenceError, match="fingerprint"):
        first.evaluate(changed, session="a", query="q", fingerprint=fingerprint)
    with pytest.raises(InteractionReferenceError, match="consumed"):
        first.serialize()
    with pytest.raises(InteractionReferenceError, match="binding"):
        second.evaluate(table, session="a", query="wrong", fingerprint=fingerprint)
    with pytest.raises(InteractionReferenceError, match="consumed"):
        second.evaluate(table, session="a", query="q", fingerprint=fingerprint)
    pair = issue_lookup("a", "q2", 0, 7, fingerprint)
    pair[0].cancel()
    with pytest.raises(InteractionReferenceError, match="consumed"):
        pair[0].serialize()
    for token, size in ((7, 7), (-1, 7), (0, 1 << 19)):
        with pytest.raises(InteractionReferenceError, match="invalid"):
            issue_lookup("a", "bad", token, size, fingerprint)


def test_lookup_crosses_conversion_chunks_and_selects_final_non_power_of_two_row() -> None:
    table = np.random.default_rng(42).integers(-128, 128, size=(1031, 7), dtype=np.int8)
    fingerprint = hashlib.sha256(memoryview(table)).digest()
    keys = issue_lookup("a", "chunked", 1030, 1031, fingerprint)
    outputs = [
        key.evaluate(table, session="a", query="chunked", fingerprint=fingerprint) for key in keys
    ]
    assert np.array_equal(np.add(*outputs, dtype=np.uint32), table[-1].astype(np.uint32))


def test_conservative_fusion_audit_keeps_norm_attention_gate_and_rounding_barriers() -> None:
    plan = lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=32)
    composition = MaskedLinearCpu(
        Model("org/test"), quantization=SymmetricPerRow(weight_bits=8, activation_bits=8)
    )
    result = _PROBE.fusion_audit(plan, composition)
    for phase in result["phases"].values():
        assert phase["serial_linear_candidate_edges"] == []
        assert phase["new_exact_w8a8_fusions"] == 0
        assert phase["additional_remote_messages_eliminated"] == 0
        assert phase["existing_parallel_linear_groups"] == 48
        assert phase["existing_duplicate_input_sends_avoided"] == 72
        assert phase["source_operator_barriers"]["multiply"] == 24
        assert phase["source_operator_barriers"]["attention_values"] == 24


def test_secure_reduction_report_does_not_turn_modular_statistic_into_complete_rmsnorm() -> None:
    result = _PROBE.probe_reduction(896)
    assert result["sum_of_squares_body_bytes_online_both_directions"] == 7168
    assert result["sum_of_squares_dealer_body_bytes_both_parties"] == 7176
    assert result["broadcast_multiply_online_bytes"] == 7176
    assert result["broadcast_multiply_dealer_bytes"] == 14344
    assert not result["whole_rmsnorm_executable"]
    assert not result["checkpoint_numeric_fidelity"]
    numeric = result["numeric_order_counterexample"]
    values = numeric["input"]
    assert sum(round(x * x / 2) for x in values) == 0
    assert round(sum(x * x for x in values) / 2) == 1


@pytest.mark.he
def test_same_client_simd_and_seeded_return_failure_are_actual_backend_measurements() -> None:
    pytest.importorskip("tenseal")
    result = _PROBE.probe_he()
    assert not result["provider_has_secret_key"]
    samples = result["batching"]["samples"]
    assert [sample["ciphertexts_each_direction"] for sample in samples] == [1, 1, 4]
    assert [sample["independent_same_client_sequences"] for sample in samples] == [1, 4, 16]
    for sample in samples:
        assert sample["max_absolute_error"] < 1e-3
        assert max(sample["values_per_ciphertext"]) <= 4096
    assert (
        3.8
        < samples[0]["both_direction_body_bytes_per_sequence"]
        / samples[1]["both_direction_body_bytes_per_sequence"]
        < 4.2
    )
    assert (
        abs(
            samples[1]["both_direction_body_bytes_per_sequence"]
            / samples[2]["both_direction_body_bytes_per_sequence"]
            - 1
        )
        < 0.05
    )
    serial = result["seeded_serialization"]
    assert not serial["seeded_ciphertext_transport_supported_by_tested_python_api"]
    assert "Serializable" in serial["seeded_low_level_return_failure"]["message"]
    assert serial["seeded_low_level_return_failure"]["type"] == "TypeError"
    mean_sizes = [
        np.mean([sample["request_bytes"] for sample in row["samples"]]) for row in serial["samples"]
    ]
    assert 0.95 < mean_sizes[1] / mean_sizes[0] < 1.05


def test_locked_evidence_keeps_all_five_results_and_missing_work_distinct() -> None:
    evidence = json.loads(_EVIDENCE.read_text())
    assert evidence["schema"] == "pllm.interaction_reduction_screen.v1"
    assert not evidence["whole_decoder_executable"]
    assert not evidence["aggregate_compute_cap_validated"]
    assert not evidence["checkpoint_quality_validated"]
    assert not evidence["full_wire_measured"]
    lookup = evidence["private_lookup"]
    assert lookup["query_body_bytes_both_workers"] == 872
    assert lookup["modular_table_macs_per_worker_per_token"] == 151936 * 896
    for output_tokens in (8, 32):
        cohort = evidence["cohorts"][str(output_tokens)]
        assert cohort["executed_rows"] == 39 + output_tokens - 1
        assert (
            cohort["rmsnorm_projection"]["known_online_floor_bytes"]
            > cohort["online_tenfold_budget_bytes"]
        )
        for phase in cohort["fusion_audit"]["phases"].values():
            assert phase["new_exact_w8a8_fusions"] == 0
        projection = cohort["lookup_projection"]
        assert projection["saved_boundary_upload_body_bytes"] == cohort["executed_rows"] * (
            7168 - 872
        )


@pytest.mark.skipif(not os.getenv("PLLM_RUN_REAL_QWEN25"), reason="requires cached pinned config")
def test_pinned_config_reproduces_fusion_and_norm_resource_gates() -> None:
    report = _PROBE.run(sample_he=False, sample_lookup=False)
    locked = json.loads(_EVIDENCE.read_text())
    assert report["source"] == locked["source"]
    for tokens in ("8", "32"):
        assert (
            report["cohorts"][tokens]["fusion_audit"] == locked["cohorts"][tokens]["fusion_audit"]
        )
        assert (
            report["cohorts"][tokens]["rmsnorm_projection"]
            == locked["cohorts"][tokens]["rmsnorm_projection"]
        )
