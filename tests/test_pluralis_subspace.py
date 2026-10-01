"""Pinned source and independent held-out confirmation of subspace cost gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("PLLM_RUN_REAL_QWEN25"), reason="requires pinned cached Qwen source"
)
def test_pinned_qwen_selective_subspace_confirmation_gate() -> None:
    root = Path(__file__).parents[1]
    recorded = json.loads(
        (root / "docs/evidence/pluralis-subspace-sensitivity-qwen25-2026-09-29.json").read_text(
            encoding="utf-8"
        )
    )
    invocation = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/probe_selective_subspace.py"),
            "--confirmation",
            str(root / "examples/benchmarks/selective_subspace_confirmation.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4"},
    )
    assert invocation.returncode == 0, invocation.stderr
    measured = json.loads(invocation.stdout)
    for field in ("config_sha256", "weights_sha256", "revision"):
        assert measured["source"][field] == recorded["source"][field]
    assert (
        measured["source"]["prompts_sha256"]
        == recorded["source"]["calibration_and_tuning_prompts_sha256"]
    )
    assert (
        measured["source"]["confirmation_prompts_sha256"]
        == recorded["source"]["confirmation_prompts_sha256"]
    )
    assert (
        measured["source"]["token_cohort_digest"]
        == recorded["source"]["confirmation_token_cohort_digest"]
    )
    assert (
        abs(
            measured["shared128_calibration_centered_output_energy_fraction"]
            - recorded["shared_space_gate"]["pooled_calibration_output_energy_at_rank128"]
        )
        < 0.00001
    )
    for name, expected in recorded["candidate_profiles"].items():
        actual = measured["results"][name]
        assert actual["prefill"]["top1_matches"] == expected["confirmation_prefill"]
        assert (
            actual["decode_same_token"]["top1_matches"]
            == expected["confirmation_same_token_decode"]
        )
        assert actual["free_second_token_matches"] == expected["confirmation_free_second_token"]
        assert (
            actual["projected_cost"]["projected_two_worker_cut_body_bytes"]
            == expected["projected_cut_body_bytes"]
        )
        assert (
            actual["total_heldout_prompts"] == recorded["comparison"]["heldout_prompts_per_cohort"]
        )
