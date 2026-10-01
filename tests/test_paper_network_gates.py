"""Real-source checks for bounded HE and long-context KV research probes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_SOURCE_AVAILABLE = bool(os.environ.get("PLLM_RUN_REAL_QWEN25"))
_SHARED_HUB_CACHE = os.environ.get("HF_HUB_CACHE") or str(
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
)


def _probe(script: str, *args: str) -> dict:
    process = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
        env={
            **os.environ,
            "HF_HUB_CACHE": _SHARED_HUB_CACHE,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        },
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


@pytest.mark.he
@pytest.mark.skipif(not _SOURCE_AVAILABLE, reason="requires pinned cached Qwen source")
def test_ckks_gate_measures_numeric_oracle_and_vetoes_naive_boundary() -> None:
    pytest.importorskip("tenseal")
    evidence = json.loads(
        (_ROOT / "docs/evidence/ckks-boundary-cost-qwen25-2026-09-29.json").read_text(
            encoding="utf-8"
        )
    )
    report = _probe("probe_ckks_boundary_cost.py")
    assert not report["ckks"]["provider_has_secret_key"]
    assert report["ckks"]["slots_per_ciphertext"] == 4096
    assert set(report["ckks"]["samples"]) == {"768", "1280", "4096"}
    assert all(
        200_000 < sample["outbound_ciphertext_bytes"] < 260_000
        and sample["affine_max_absolute_error"] < 1e-5
        for sample in report["ckks"]["samples"].values()
    )
    sampled = report["optimistic_ciphertext_bytes_per_gate_chunk"]
    for expected in evidence["cohorts"]:
        row = report["cohorts"][str(expected["output_tokens"])]
        assert row["plan_digest"] == expected["plan_digest"]
        assert row["schedule_digest"] == expected["schedule_digest"]
        assert row["composition_digest"] == evidence["source"]["composition_digest"]
        assert row["prefill_gate_ciphertexts"] == expected["prefill_gate_ciphertexts"]
        assert row["decode_gate_ciphertexts"] == expected["decode_gate_ciphertexts"]
        assert (
            row["one_way_gate_body_bytes"]
            == (row["prefill_gate_ciphertexts"] + row["decode_gate_ciphertexts"]) * sampled
        )
        # Fresh HE randomness changes exact bytes slightly; the veto survives.
        assert (
            abs(row["one_way_gate_body_bytes"] - expected["optimistic_one_way_gate_body_bytes"])
            < 1 << 20
        )
        assert row["gate_only_exceeds_prepared_covered"]
        assert row["gate_only_exceeds_tenfold_budget"]
        assert (
            row["measured_prepared_covered_body_bytes"]
            == expected["measured_prepared_covered_all_link_body_bytes"]
        )
    assert not report["decoder_executable"]
    assert not report["measured_full_wire"]


@pytest.mark.slow
@pytest.mark.skipif(not _SOURCE_AVAILABLE, reason="requires pinned cached Qwen source")
def test_kv_oracle_quality_changes_on_disjoint_public_prompts_without_stage_byte_savings() -> None:
    evidence = json.loads(
        (_ROOT / "docs/evidence/kv-selection-qwen25-2026-09-29.json").read_text(encoding="utf-8")
    )
    primary = _probe("probe_kv_selection.py")
    confirmation = _probe("probe_kv_selection.py", "--confirmation")
    for key, report in (("primary", primary), ("confirmation", confirmation)):
        recorded = evidence[key]
        assert report["plan_digest"] == recorded["plan_digest"]
        assert report["schedule_digest"] == recorded["schedule_digest"]
        assert report["transformed_plan_digest"] == recorded["transformed_plan_digest"]
        assert report["source"]["input_token_counts"] == evidence["source"][f"{key}_input_tokens"]
        assert report["stage_linear_shapes_unchanged"]
        assert not report["transformed_schedule_executable"]
        assert not report["runtime_kv_evicted"]
        assert report["prepared_stage_body_reduction_estimate_bytes"] == 0
        for candidate, score in recorded["candidate_scores"].items():
            actual = report["candidates"][candidate]
            assert actual["selections_checked"] == score["checked"] == 24
            assert actual["attention_calls"] == 24 * 7 * 3
            assert actual["same_token_top1_matches"] == score["same_token_top1_matches"]
            assert actual["free_token_matches"] == score["free_generation_token_matches"]
            assert actual["original_key_positions"] > actual["retained_key_positions"]
    assert primary["decode_attention_scores_width_transformed"] == 16
    assert (
        primary["candidates"]["recent8_spaced8"]["free_token_matches"]
        > confirmation["candidates"]["recent8_spaced8"]["free_token_matches"]
    )
