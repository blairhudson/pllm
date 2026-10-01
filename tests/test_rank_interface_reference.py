from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pllm.runtime.rank_interface_reference import (
    certify_top1_from_error_bound,
    fit_affine_residual_basis,
    fit_output_rank_basis,
)


def test_fitted_affine_subspace_recovers_independent_low_rank_rows() -> None:
    x = np.column_stack((
        np.arange(1, 9, dtype=np.float32),
        np.asarray([1, 3, 2, 5, 7, 4, 8, 6], dtype=np.float32),
        np.zeros(8, dtype=np.float32),
    ))
    x[:, 2] = x[:, 0] - 3 * x[:, 1] + 7
    basis = fit_output_rank_basis(x, rank=2)
    np.testing.assert_allclose(basis.project(x), x, rtol=0, atol=0.00003)
    with pytest.raises(ValueError):
        basis.mean[0] = 0
    with pytest.raises(ValueError):
        basis.directions[0, 0] = 0


@pytest.mark.parametrize("invalid", [0, -1, True, 8, 129])
def test_rank_fit_rejects_unsupported_rank(invalid: int) -> None:
    with pytest.raises(ValueError):
        fit_output_rank_basis(np.ones((8, 3), dtype=np.float32), rank=invalid)


def test_rank_fit_and_projection_fail_closed_on_invalid_data() -> None:
    rows = np.ones((8, 3), dtype=np.float32)
    for bad in (rows.astype(np.float64), np.full((1, 3), 1, dtype=np.float32)):
        with pytest.raises(ValueError):
            fit_output_rank_basis(bad, rank=1)
    rows[0, 1] = np.nan
    with pytest.raises(ValueError):
        fit_output_rank_basis(rows, rank=1)
    basis = fit_output_rank_basis(np.ones((8, 3), dtype=np.float32), rank=1)
    with pytest.raises(ValueError):
        basis.project(np.asarray([[1.0, 2.0, float("inf")]], dtype=np.float32))


def test_public_affine_bypass_recovers_low_rank_nonlinear_residual() -> None:
    rng = np.random.default_rng(419)
    calibration = rng.normal(size=(120, 4)).astype(np.float32)
    heldout = rng.normal(size=(30, 4)).astype(np.float32)
    public = rng.normal(size=(4, 4)).astype(np.float32)
    direction = rng.normal(size=(4,)).astype(np.float32)

    def target(x: np.ndarray) -> np.ndarray:
        return np.asarray(x @ public + np.tanh(x[:, :1]) * direction, dtype=np.float32)

    basis = fit_affine_residual_basis(calibration, target(calibration), 1, ridge_ratio=0.001)
    np.testing.assert_allclose(basis.project(heldout, target(heldout)), target(heldout), atol=0.01)
    with pytest.raises(ValueError):
        basis.public_map[0, 0] = 0
    with pytest.raises(ValueError):
        basis.project(heldout, target(heldout).astype(np.float64))


def test_margin_certificate_needs_an_actual_uniform_error_bound() -> None:
    scores = np.asarray([2.5, -4.0, 0.0], dtype=np.float32)
    assert certify_top1_from_error_bound(scores, 1.0)
    assert not certify_top1_from_error_bound(scores, 1.25)
    for invalid in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            certify_top1_from_error_bound(scores, invalid)


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("PLLM_RUN_REAL_QWEN25"), reason="requires pinned cached Qwen source")
def test_pinned_qwen_output_rank_gate_matches_retained_evidence() -> None:
    root = Path(__file__).parents[1]
    recorded = json.loads(
        (root / "docs/evidence/rank-interface-output-oracle-qwen25-2026-09-29.json")
        .read_text(encoding="utf-8")
    )
    invocation = subprocess.run(
        [sys.executable, str(root / "scripts/probe_rank_interface.py")],
        capture_output=True, text=True, check=False, timeout=600,
    )
    assert invocation.returncode == 0, invocation.stderr
    actual = json.loads(invocation.stdout)
    for key in ("config_sha256", "weights_sha256", "prompts_sha256", "token_cohort_digest"):
        assert actual[key] == recorded[key]
    assert actual["evaluation_input_token_counts"] == recorded["evaluation_input_token_counts"]
    for name, rows in recorded["results"].items():
        measured = actual["results"][name]
        assert measured["free_second_token_matches"] == rows["free_second_token_matches"]
        for phase in ("prefill", "decode_same_token"):
            assert measured[phase]["top1_matches"] == rows[phase]["top1_matches"]
            assert abs(measured[phase]["worst_logit_error"] - rows[phase]["worst_logit_error"]) < 0.05


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("PLLM_RUN_REAL_QWEN25"), reason="requires pinned cached Qwen source")
def test_pinned_qwen_affine_residual_does_not_pass_quality_gate() -> None:
    root = Path(__file__).parents[1]
    evidence = json.loads(
        (root / "docs/evidence/affine-residual-oracle-qwen25-2026-09-29.json")
        .read_text(encoding="utf-8")
    )
    invocation = subprocess.run(
        [sys.executable, str(root / "scripts/probe_rank_interface.py"),
         "--mode", "affine_residual", "--ranks", "16", "32", "--cut-strides", "1", "2", "4"],
        capture_output=True, text=True, check=False, timeout=600,
    )
    assert invocation.returncode == 0, invocation.stderr
    actual = json.loads(invocation.stdout)
    for key in ("weights_sha256", "prompts_sha256", "token_cohort_digest"):
        assert actual[key] == evidence["source"][key]
    for name, retained in evidence["results"].items():
        measured = actual["results"][name]
        assert measured["prefill"]["top1_matches"] == retained["prefill_top1"]
        assert measured["decode_same_token"]["top1_matches"] == retained["decode_same_token_top1"]
        assert measured["free_second_token_matches"] == retained["free_second_token_matches"]
        assert measured["prefill"]["retrospective_margin_witnesses"] == 0
        assert measured["decode_same_token"]["retrospective_margin_witnesses"] == 0
        assert abs(
            measured["maximum_layer_eval_trajectory_relative_residual"]
            - retained["maximum_reference_trajectory_relative_residual"]
        ) < 0.001
