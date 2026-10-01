from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts/probe_pretrained_bottleneck.py"
_PREPARED_SCRIPT = _ROOT / "scripts/probe_prepared_candidate_bytes.py"
_EVIDENCE = _ROOT / "docs/evidence/pretrained-bottleneck-screen-2026-09-29.json"
_COLD_EVIDENCE = _ROOT / "docs/evidence/prepared-cold-compare-qwen-smol-2026-09-29.json"
_STAGE_EVIDENCE = _ROOT / "docs/evidence/prepared-stage-attribution-qwen25-2026-09-29.json"
_SOCKET_VETO = _ROOT / "docs/evidence/nettop-loopback-counter-veto-2026-09-29.json"
_SHARED_HUB = os.environ.get("HF_HUB_CACHE") or str(
    Path(os.environ["HF_HOME"]) / "hub"
    if "HF_HOME" in os.environ
    else Path.home() / ".cache/huggingface/hub"
)


def _probe(*args: str, script: Path = _SCRIPT) -> dict[str, Any]:
    environment = {
        **os.environ,
        "HF_HUB_CACHE": _SHARED_HUB,
        "OPENBLAS_NUM_THREADS": "4",
        "VECLIB_MAXIMUM_THREADS": "4",
    }
    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_six_source_semantic_bottleneck_screen_matches_locked_evidence() -> None:
    result = _probe("--structure-only")
    evidence = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    compatibility = json.loads(
        (_ROOT / "docs/data/model-compatibility.json").read_text(encoding="utf-8")
    )
    smol_status = next(
        row
        for row in compatibility["adapters"]
        if row.get("pinned_real_checkpoint_functionality") == "SmolLM2-135M"
    )
    assert smol_status["adapter"] == "pllm.dense_gated_decoder.v1"
    assert smol_status["fixture"] == (
        "crates/pllm-models/tests/fixtures/SmolLM2-135M-Instruct-12fd25f-config.json"
    )
    assert smol_status["evidence"] == {
        "checkpoint": "real-local",
        "provider": "real-prepared-local",
        "quality": "narrow-smol-prefill",
    }
    sources = result["candidate_sources"]
    assert set(sources) == set(evidence["semantic_candidates"])
    assert (
        result["qwen_prepared_control_covered_body_bytes"]
        == evidence["comparison"]["pinned_qwen25_w8a8_covered_all_link_control_body_bytes"]
    )
    for model, locked in evidence["semantic_candidates"].items():
        candidate = sources[model]
        assert candidate["config_sha256"] == locked["config_sha256"]
        assert candidate["hidden_width"] == locked["hidden_width"]
        assert candidate["layers"] == locked["layers"]
        assert candidate["remote_stages"] == locked["remote_stages"]
        assert (
            candidate["optimistic_u16_prepared_stage_bodies_39_plus_32_bytes"]
            == locked["optimistic_u16_stage_body_bytes"]
        )
        assert (
            candidate["optimistic_two_hidden_source_online_openings_bytes"]
            == locked["optimistic_two_source_opening_bytes"]
        )
        assert (
            candidate["public_i8_remote_weight_snapshot_bytes_excluding_scales"]
            == locked["i8_remote_weight_bytes_excluding_scales"]
        )
        assert candidate["remote_body_linear_macs"] == locked["tracked_remote_linear_macs"]
        assert candidate["local_head_linear_macs"] == locked["tracked_local_head_macs"]
        assert candidate["remote_fraction_of_tracked_linear_macs"] > 0.8
        assert len(candidate["schedule_digest"]) == 64
    assert (
        sources["HuggingFaceTB/SmolLM2-135M-Instruct"]["semantic_plan_digest"]
        == evidence["semantic_candidates"]["HuggingFaceTB/SmolLM2-135M-Instruct"][
            "semantic_plan_digest"
        ]
    )
    assert (
        sources["HuggingFaceTB/SmolLM2-135M-Instruct"][
            "optimistic_two_hidden_source_online_openings_bytes"
        ]
        > result["qwen_hundredfold_budget_bytes"]
    )


def test_cold_matched_cohorts_preserve_scope_and_same_body_comparator() -> None:
    cold = json.loads(_COLD_EVIDENCE.read_text(encoding="utf-8"))
    stage = json.loads(_STAGE_EVIDENCE.read_text(encoding="utf-8"))
    small = json.loads(_EVIDENCE.read_text(encoding="utf-8"))["smollm_prepared_candidate_control"]
    qwen = cold["qwen25_prepared"]
    offset = cold["qwen25_two_online_offset"]
    smol = cold["smollm2_prepared"]
    assert qwen["body_fingerprint"] == offset["body_fingerprint"]
    assert qwen["body_fingerprint"] != smol["body_fingerprint"]
    assert qwen["covered_online_body_bytes"] == stage["cohorts"]["32"]["covered_online_body_bytes"]
    assert qwen["covered_cold_all_link_body_bytes"] == (
        stage["cohorts"]["32"]["covered_all_link_body_bytes"]
        + qwen["cold_client_bundle_delivery_body_bytes"]
    )
    assert smol["covered_cold_all_link_body_bytes"] == (
        small["covered_all_link_body_bytes"] + smol["cold_client_bundle_delivery_body_bytes"]
    )
    for cohort in (qwen, offset, smol):
        delta = cohort["aggregate_cold_cpu_seconds"] - sum(
            cohort["cold_cpu_seconds_by_role"].values()
        )
        assert abs(delta) < 0.00001
    repeats = cold["repeat_aggregate_cold_cpu_seconds"]
    for key, cohort in (
        ("qwen25_prepared", qwen),
        ("qwen25_two_online_offset", offset),
        ("smollm2_prepared", smol),
    ):
        repeat_cpu = cold["repeat_cold_cpu_seconds_by_role"][key]
        assert abs(repeats[key] - sum(repeat_cpu.values())) < 0.00001
        assert (
            cold["two_sample_median_aggregate_cold_cpu_seconds"][key]
            == (cohort["aggregate_cold_cpu_seconds"] + repeats[key]) / 2
        )
    assert (
        cold["two_sample_median_aggregate_cold_cpu_seconds"]["qwen25_prepared"]
        > cold["two_sample_median_aggregate_cold_cpu_seconds"]["qwen25_two_online_offset"]
    )
    assert offset["covered_cold_all_link_body_bytes"] is None
    assert cold["method"]["covered_body_bytes_not_full_wire"] is True
    veto = json.loads(_SOCKET_VETO.read_text(encoding="utf-8"))
    assert veto["body_fingerprint"] == qwen["body_fingerprint"]
    assert veto["status"].startswith("rejected")
    assert (
        veto["nettop_process_loopback_counter_deltas"]["summed_role_sent_bytes"]
        < veto["reconciled_protocol_body_counters"]["covered_online_all_link_bytes"]
    )


@pytest.mark.quality
@pytest.mark.skipif(
    os.environ.get("PLLM_RUN_REAL_SMOLLM") != "1",
    reason="opt in to downloading and executing the pinned public SmolLM checkpoint",
)
def test_disjoint_real_checkpoint_bottleneck_oracle_cannot_admit_decoder() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    result = _probe()
    evidence = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    locked = evidence["smollm_frozen_output_rank_oracle"]
    oracle = result["heldout_numeric_oracle"]
    assert (
        oracle["source_weight_sha256"]
        == evidence["semantic_candidates"]["HuggingFaceTB/SmolLM2-135M-Instruct"]["weight_sha256"]
    )
    assert oracle["public_calibration_sha256"] == locked["public_calibration_and_tuning_sha256"]
    assert oracle["disjoint_confirmation_sha256"] == locked["disjoint_confirmation_sha256"]
    assert oracle["cohort_lengths"] == {
        "calibration": 20,
        "tuning": 6,
        "heldout": 12,
        "confirmation": 12,
    }
    for name, row in locked["observations"].items():
        if name.startswith("all_30_layers_rank"):
            profile = f"rank{name.removeprefix('all_30_layers_rank')}_every1"
        else:
            profile = f"rank{name.removeprefix('every_tenth_layer_rank')}_every10"
        for cohort, prefix in (("heldout", "heldout"), ("confirmation", "confirmation")):
            actual = oracle["float32_optimistic_output_rank_cut"][cohort][profile]
            assert actual["prefill_matches"] == row[f"{prefix}_prefill"]
            assert actual["same_token_decode_matches"] == row[f"{prefix}_same_token_decode"]
            assert (
                actual["optimistic_two_worker_cut_body_projection_bytes"]
                == row["projected_cut_bodies_39_plus_32_bytes"]
            )
    assert all(
        score["same_token_decode_matches"] < 12
        for score in oracle["float32_optimistic_output_rank_cut"]["confirmation"].values()
    )


@pytest.mark.quality
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("PLLM_RUN_REAL_SMOLLM") != "1",
    reason="opt in to the pinned real SmolLM prepared roles and FP32 reference",
)
def test_real_prepared_candidate_stages_and_reference_quality_are_locked() -> None:
    result = _probe("--reference-quality", script=_PREPARED_SCRIPT)
    evidence = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    locked = evidence["smollm_prepared_candidate_control"]
    assert result["body_fingerprint"] == locked["body_fingerprint"]
    assert (result["input_tokens"], result["output_tokens"]) == (39, 32)
    assert result["remote_stages"] == 120
    assert result["stage_body_bytes_reconciled"] is True
    assert result["attributed_stage_body_bytes"] == locked["compiler_stage_body_bytes_reconciled"]
    assert result["covered_all_link_body_bytes"] == locked["covered_all_link_body_bytes"]
    assert result["covered_online_body_bytes"] == locked["covered_online_body_bytes"]
    quality = result["reference_quality"]
    expected = locked["reference_quality"]
    assert quality["source_lock_digest"] == expected["source_lock_digest"]
    assert quality["dataset_digest"] == expected["dataset_digest"]
    assert quality["token_cohort_digest"] == expected["token_cohort_digest"]
    assert quality["sample_count"] == expected["prompts"] == 12
    assert quality["top1_agreement"] == expected["top1_matches"] / 12
    assert quality["top5_recall"] == expected["top5_recall"]
    assert abs(quality["max_abs_logit_error"] - expected["worst_absolute_logit_error"]) < 0.001
    assert result["total_wire_bytes"] is None
    assert result["representative_generation_quality_established"] is False
