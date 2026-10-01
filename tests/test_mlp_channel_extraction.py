from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pllm.runtime.mlp_channel_extraction import (
    ChannelExtractionError,
    calibrate_gated_mlp_channels,
    extract_gated_mlp_channels,
    projected_channel_cut_bodies,
)


def _toy_model() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random = np.random.default_rng(912)
    gate = (random.normal(size=(7, 4)) * 0.3).astype(np.float32)
    up = (random.normal(size=(7, 4)) * 0.3).astype(np.float32)
    down = (random.normal(size=(4, 7)) * 0.3).astype(np.float32)
    calibration = random.normal(size=(60, 4)).astype(np.float32)
    heldout = random.normal(size=(12, 4)).astype(np.float32)
    return gate, up, down, calibration, heldout


def test_all_selected_original_channels_reproduce_an_independent_gated_mlp() -> None:
    gate, up, down, calibration, heldout = _toy_model()
    profile = extract_gated_mlp_channels(
        gate_weight=gate, up_weight=up, down_weight=down,
        calibration_inputs=calibration, selected_width=gate.shape[0],
    )
    gate_values = heldout @ gate.T
    expected = (gate_values / (1 + np.exp(-gate_values)) * (heldout @ up.T)) @ down.T
    np.testing.assert_allclose(profile.evaluate(heldout), expected, rtol=0, atol=0.00001)
    assert np.array_equal(profile.indices, np.arange(gate.shape[0]))
    with pytest.raises(ValueError):
        profile.affine[0, 0] = 0
    gate[0, 0] = 50
    np.testing.assert_allclose(profile.evaluate(heldout), expected, rtol=0, atol=0.00001)


def test_channel_selection_binds_source_and_calibration_without_training() -> None:
    gate, up, down, calibration, heldout = _toy_model()
    public = calibrate_gated_mlp_channels(
        gate_weight=gate, up_weight=up, down_weight=down, calibration_inputs=calibration,
    )
    def extract(width: int, public_rows: np.ndarray):
        return extract_gated_mlp_channels(
            gate_weight=gate, up_weight=up, down_weight=down,
            calibration_inputs=public_rows, selected_width=width,
        )
    empty, narrow = extract(0, calibration), extract(2, calibration)
    assert narrow.digest == public.select(2).digest
    assert public.select(7).digest == extract(7, calibration).digest
    assert empty.width == 0 and narrow.width == 2
    assert not np.allclose(empty.evaluate(heldout), narrow.evaluate(heldout))
    assert narrow.digest == extract(2, calibration).digest
    altered = calibration.copy()
    altered[0, 0] += 1
    assert narrow.digest != extract(2, altered).digest
    assert narrow.calibration_digest != extract(2, altered).calibration_digest
    gate[0, 0] += 1
    assert narrow.source_digest != extract(2, calibration).source_digest


def test_closed_form_public_affine_fit_reduces_calibration_error_without_weight_changes() -> None:
    gate, up, down, calibration, heldout = _toy_model()
    args = dict(gate_weight=gate, up_weight=up, down_weight=down, calibration_inputs=calibration)
    taylor = calibrate_gated_mlp_channels(**args)
    fitted = calibrate_gated_mlp_channels(**args, affine_method="least_squares")
    assert taylor.source_digest == fitted.source_digest
    assert taylor.calibration_digest == fitted.calibration_digest
    assert taylor.select(2).digest != fitted.select(2).digest
    assert fitted.select(2).digest == calibrate_gated_mlp_channels(
        **args, affine_method="least_squares",
    ).select(2).digest
    actual_gate = calibration @ gate.T
    reference = (actual_gate / (1 + np.exp(-actual_gate)) * (calibration @ up.T)) @ down.T
    error_taylor = np.linalg.norm(taylor.select(0).evaluate(calibration) - reference)
    error_fit = np.linalg.norm(fitted.select(0).evaluate(calibration) - reference)
    assert error_fit < error_taylor
    heldout_gate = heldout @ gate.T
    heldout_full = (heldout_gate / (1 + np.exp(-heldout_gate)) * (heldout @ up.T)) @ down.T
    np.testing.assert_allclose(fitted.select(gate.shape[0]).evaluate(heldout), heldout_full, atol=0.00001)
    with pytest.raises(ChannelExtractionError):
        calibrate_gated_mlp_channels(**args, affine_method="private_optimized")


def test_invalid_source_or_private_tensor_fails_before_extraction() -> None:
    gate, up, down, calibration, heldout = _toy_model()
    args = dict(gate_weight=gate, up_weight=up, down_weight=down, calibration_inputs=calibration)
    for invalid in (-1, True, 8):
        with pytest.raises(ChannelExtractionError):
            extract_gated_mlp_channels(**args, selected_width=invalid)
    with pytest.raises(ChannelExtractionError):
        extract_gated_mlp_channels(**(args | {"up_weight": up[:, :-1]}), selected_width=1)
    with pytest.raises(ChannelExtractionError):
        extract_gated_mlp_channels(**(args | {"gate_weight": gate.astype(np.float64)}), selected_width=1)
    bad = calibration.copy()
    bad[0, 1] = np.nan
    with pytest.raises(ChannelExtractionError):
        extract_gated_mlp_channels(**(args | {"calibration_inputs": bad}), selected_width=1)
    selected = extract_gated_mlp_channels(**args, selected_width=1)
    with pytest.raises(ChannelExtractionError):
        selected.evaluate(heldout.astype(np.float64))
    bad_row = heldout.copy()
    bad_row[0, 0] = float("inf")
    with pytest.raises(ChannelExtractionError):
        selected.evaluate(bad_row)
    full = extract_gated_mlp_channels(**args, selected_width=gate.shape[0])
    with pytest.raises(ChannelExtractionError):
        full.evaluate(np.full((1, heldout.shape[1]), 1e38, dtype=np.float32))


def test_10x_channel_cut_cost_requires_a_narrow_public_width() -> None:
    assert projected_channel_cut_bodies(layers=24, rows=70, channels_per_layer=0, word_bytes=8)[
        "all_link_body_bytes"
    ] == 0
    narrow = projected_channel_cut_bodies(layers=24, rows=70, channels_per_layer=128, word_bytes=8)
    wide = projected_channel_cut_bodies(layers=24, rows=70, channels_per_layer=256, word_bytes=8)
    assert narrow["all_link_body_bytes"] == 10_657_920
    assert narrow["all_link_body_bytes"] < 11_354_502
    assert wide["all_link_body_bytes"] > 17_897_055
    with pytest.raises(ChannelExtractionError):
        projected_channel_cut_bodies(layers=24, rows=70, channels_per_layer=True, word_bytes=8)


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("PLLM_RUN_REAL_QWEN25"), reason="requires pinned cached Qwen source")
@pytest.mark.parametrize("layers,cohort", [
    ([23], "last_layer_only"),
    ([0, 12, 23], "first_middle_last_layers_together"),
])
def test_pinned_qwen_channel_extraction_is_not_a_10x_quality_result(
    layers: list[int], cohort: str,
) -> None:
    root = Path(__file__).parents[1]
    evidence = json.loads(
        (root / "docs/evidence/extracted-mlp-channels-qwen25-2026-09-29.json")
        .read_text(encoding="utf-8")
    )
    invocation = subprocess.run(
        [sys.executable, str(root / "scripts/probe_mlp_channel_extraction.py"),
         "--layers", *(str(index) for index in layers), "--widths",
         "0", "64", "128", "256", "512", "1024", "2048", "4864"],
        capture_output=True, text=True, check=False, timeout=600,
    )
    assert invocation.returncode == 0, invocation.stderr
    measured = json.loads(invocation.stdout)
    assert measured["evaluated_layers"] == layers
    assert measured["evaluation_token_counts"] == evidence["evaluation_input_token_counts"]
    for key in ("weights_sha256", "config_sha256", "token_cohort_digest"):
        assert measured["source"][key] == evidence["source"][key]
    for index in layers:
        assert measured["full_channel_reconstruction_max_error"][str(index)] < 0.001
    for width, retained in evidence[cohort]["narrow_results"].items():
        actual = measured["results"][width]
        assert actual["prefill"]["top1_matches"] == retained["prefill_top1"]
        assert actual["decode_same_token"]["top1_matches"] == retained["same_token_decode_top1"]
        assert actual["free_second_token_matches"] == retained["free_second_token_matches"]
    if cohort == "last_layer_only":
        assert measured["results"]["128"]["public_extracted_profile_digests"]["23"] == (
            evidence[cohort]["rank128_extracted_profile_digest"]
        )
    else:
        assert measured["results"]["128"]["public_extracted_profile_digests"] == (
            evidence[cohort]["rank128_extracted_profile_digests"]
        )
    assert measured["results"]["128"]["cost_if_every_layer_used_this_width"]["all_link_body_bytes"] == (
        evidence["body_budget_projection"]["rank128_if_all_24_layers_cut_body_bytes"]
    )


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("PLLM_RUN_REAL_QWEN25"), reason="requires pinned cached Qwen source")
@pytest.mark.parametrize("layers,phase,key", [
    ([0, 12, 23], "all_prefill", "least_squares_all_prefill"),
    ([0, 12, 23], "decision_and_decode", "least_squares_decision_and_decode"),
    ([0, 12, 23], "all_prefill_and_decode", "least_squares_all_prefill_and_decode"),
    ([23], "all_prefill", "last_layer_all_prefill"),
])
def test_public_affine_calibration_retains_locked_quality_boundary(
    layers: list[int], phase: str, key: str,
) -> None:
    root = Path(__file__).parents[1]
    evidence = json.loads(
        (root / "docs/evidence/extracted-mlp-affine-quality-qwen25-2026-09-29.json")
        .read_text(encoding="utf-8")
    )
    invocation = subprocess.run(
        [sys.executable, str(root / "scripts/probe_mlp_channel_extraction.py"),
         "--layers", *(str(index) for index in layers), "--widths", "0", "128", "2048", "4864",
         "--affine-method", "least_squares", "--calibration-phase", phase],
        capture_output=True, text=True, check=False, timeout=600,
    )
    assert invocation.returncode == 0, invocation.stderr
    measured = json.loads(invocation.stdout)
    assert measured["affine_method"] == "least_squares" and measured["calibration_phase"] == phase
    assert measured["source"]["weights_sha256"] == evidence["source"]["weights_sha256"]
    assert measured["source"]["token_cohort_digest"] == evidence["source"]["token_cohort_digest"]
    retained = evidence["three_layer_cohort"]["results"].get(key)
    if retained is not None:
        assert measured["calibration_rows"] == retained["public_rows_per_layer"]
        for width in (0, 128, 2048):
            actual = measured["results"][str(width)]
            assert actual["prefill"]["top1_matches"] == retained[f"width_{width}"]["prefill_top1"]
            assert actual["decode_same_token"]["top1_matches"] == retained[f"width_{width}"]["decode_top1"]
        if "width_128_profile_digests" in retained:
            assert measured["results"]["128"]["public_extracted_profile_digests"] == (
                retained["width_128_profile_digests"]
            )
    else:
        assert measured["calibration_rows"] == 322
        for width in (0, 128):
            actual = measured["results"][str(width)]
            assert actual["prefill"]["top1_matches"] == evidence[key][
                f"least_squares_width_{width}"
            ]["prefill_top1"]
            assert actual["decode_same_token"]["top1_matches"] == evidence[key][
                f"least_squares_width_{width}"
            ]["decode_top1"]
    full = measured["results"]["4864"]
    assert full["prefill"]["top1_matches"] == 12
    assert full["decode_same_token"]["top1_matches"] == 12
