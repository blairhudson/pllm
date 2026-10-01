from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row

ROOT = Path(__file__).resolve().parents[1]
# Capture before conftest's autouse XDG cache isolation runs.
HUB_CACHE = os.environ.get("HF_HUB_CACHE") or str(
    Path(os.environ["HF_HOME"]) / "hub"
    if "HF_HOME" in os.environ
    else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface/hub"
)
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "temporal_delta_probe", ROOT / "scripts/probe_temporal_deltas.py"
)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_exact_sparse_update_matches_independent_dense_signed_and_modular_products():
    rng = np.random.default_rng(4187)
    weight = rng.integers(-127, 128, (13, 19), dtype=np.int16).astype(np.int8)
    previous = rng.integers(-127, 128, 19, dtype=np.int16).astype(np.int8)
    for changed in (0, 1, 7, 19):
        current = previous.copy()
        current[:changed] = rng.integers(-127, 128, changed, dtype=np.int16)
        prior = weight.astype(np.int64) @ previous.astype(np.int64)
        delta = probe.temporal_delta(previous, current)
        updated = probe.exact_sparse_update(weight, prior, delta)
        # Scalar integer oracle, independent of NumPy's sparse matmul.
        expected = np.array([sum(int(w) * int(x) for w, x in zip(row, current)) for row in weight])
        np.testing.assert_array_equal(updated, expected)
        for bits in (16, 24, 32):
            np.testing.assert_array_equal(updated % (1 << bits), expected % (1 << bits))


def test_extreme_temporal_alphabet_needs_nine_bits_and_two_i8_limbs():
    previous = np.array([-127, 127, -127, 0], dtype=np.int8)
    current = np.array([127, -127, 127, 0], dtype=np.int8)
    delta = probe.temporal_delta(previous, current)
    np.testing.assert_array_equal(delta, [254, -254, 254, 0])
    first = np.clip(delta, -127, 127).astype(np.int8)
    second = (delta - first.astype(np.int16)).astype(np.int8)
    np.testing.assert_array_equal(first.astype(np.int16) + second, delta)
    weight = np.array([[127, -127, 127, 1]], dtype=np.int8)
    prior = np.array([sum(int(a) * int(b) for a, b in zip(weight[0], previous))])
    np.testing.assert_array_equal(probe.exact_sparse_update(weight, prior, delta), [48387])
    assert int((weight.astype(np.int64) @ delta.astype(np.int64))[0]) > (1 << 16)


def test_scale_churn_with_identical_quantized_row_still_changes_float_output():
    previous = quantize_activation_per_row(np.array([[1.0, -0.5, 0.25]], np.float32), bits=8)
    current = quantize_activation_per_row(np.array([[2.0, -1.0, 0.5]], np.float32), bits=8)
    np.testing.assert_array_equal(previous.values, current.values)
    assert previous.scales[0] != current.scales[0]
    weight = np.array([[3, -7, 2]], dtype=np.int8)
    weight_scale = np.array([0.125], np.float32)
    prior = previous.values.astype(np.int64) @ weight.astype(np.int64).T
    updated = probe.exact_sparse_update(
        weight, prior[0], probe.temporal_delta(previous.values[0], current.values[0])
    )
    dense = current.values.astype(np.int64) @ weight.astype(np.int64).T
    np.testing.assert_array_equal(updated, dense[0])
    correct = dequantize_matmul(updated[None], current.scales, weight_scale)
    np.testing.assert_array_equal(correct, dequantize_matmul(dense, current.scales, weight_scale))
    assert not np.array_equal(correct, dequantize_matmul(prior, previous.scales, weight_scale))


def test_combinatorial_oracle_encoding_counts_alphabet_support_and_dense_outputs():
    empty = probe.encoding_estimates(8, 0, 3, 24)
    one = probe.encoding_estimates(8, 1, 3, 24)
    full = probe.encoding_estimates(8, 8, 3, 24)
    assert empty["known_support_9bit_plaintext_bytes"] == 0
    assert one["known_support_9bit_plaintext_bytes"] == 2  # nine bits, not one byte
    assert one["enumerative_support_9bit_plaintext_bytes"] == 2  # 3 + 9 bits
    assert one["enumerative_support_masked_ring_oracle_input_bytes"] == 4  # 3 + 24 bits
    assert full["enumerative_support_masked_ring_oracle_input_bytes"] == 24
    assert full["dense_exact_ring_input_bytes"] == 24
    assert empty["masked_ring_oracle_online_bytes"] == 9
    assert empty["masked_ring_oracle_all_link_bytes"] == 18
    assert full["public_full_width_padded_index_ring_input_bytes"] == 27
    # Even perfect input sparsity leaves two dense output bodies all-link.
    assert empty["masked_ring_oracle_all_link_bytes"] / empty["dense_all_link_bytes"] > 0.1


@pytest.mark.parametrize(
    "args",
    [(0, 0, 1, 24), (8, 9, 1, 24), (8, -1, 1, 24), (8, 0, 0, 24), (8, 0, 1, 9), (8, True, 1, 24)],
)
def test_encoding_rejects_invalid_domains(args):
    with pytest.raises(ValueError):
        probe.encoding_estimates(*args)


def test_delta_rejects_i8_overflow_domains():
    with pytest.raises(ValueError, match="excludes -128"):
        probe.temporal_delta(np.array([-128], np.int8), np.array([127], np.int8))
    with pytest.raises(ValueError, match="int8"):
        probe.temporal_delta(np.array([0], np.int16), np.array([127], np.int8))


def test_locked_temporal_screen_is_not_protocol_admission():
    path = ROOT / "docs/evidence/temporal-delta-screen-2026-10-01.json"
    evidence = json.loads(path.read_text())
    assert evidence["source"]["revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert evidence["native_backend"] == "rust" and evidence["model_threads"] == 1
    assert not evidence["protocol_admission"]
    assert len(evidence["workload"]) == 6
    assert evidence["verification"]["exact_native_signed_integer_decode_checks"] == 6 * 4 * 96
    assert evidence["verification"]["independent_sparse_integer_checks"] == 2 * 96
    assert evidence["verification"]["exact_float32_logit_rows"] == 30
    assert (
        evidence["controls_39_plus_32_context_only"]["sdk_attention_placement_warm_all_link_bytes"]
        == 148297986
    )
    assert not evidence["controls_39_plus_32_context_only"]["matched_to_this_short_cohort"]
    assert sum(len(row["stage_ids"]) for row in evidence["retained_state_by_role"].values()) == 96
    for key, row in evidence["by_role_phase"].items():
        assert row["unchanged_coordinate_fraction"] < 0.5, key
        assert row["scale_change_fraction"] > 0.9, key
        assert row["delta_coordinates_outside_signed_i8"] > 0, key
        assert row["masked_ring_oracle_all_link_bytes"] > row["dense_all_link_bytes"] // 10


def test_sdk_attention_candidate_scope_excludes_both_attention_roles_and_counts_resources():
    # Deliberately unrelated stage IDs: projection must use semantic roles,
    # never stage-name matching. Exercise 24 layers and all four baseline roles.
    widths = {
        "qkv_projection": (896, 1152, 24),
        "attention_output": (896, 896, 24),
        "mlp_gate_up": (896, 9728, 24),
        "mlp_down": (4864, 896, 32),
    }
    metadata = {}
    for layer in range(24):
        for index, (role, (inp, out, bits)) in enumerate(widths.items()):
            metadata[f"opaque-{layer}-{index}"] = {
                "role": role,
                "input_width": inp,
                "output_width": out,
                "current_exact_ring_bits": bits,
                "retained_client_previous_q_i8_bytes": inp,
                "retained_client_exact_product_i32_bytes": 4 * out,
                "retained_client_activation_scale_f32_bytes": 4,
            }
    scope = probe.sdk_derived_scope(metadata)
    assert scope["remote_roles"] == ["mlp_down", "mlp_gate_up"]
    assert scope["client_attention_roles"] == ["attention_output", "qkv_projection"]
    assert scope["execution_composition"] == "baseline MaskedLinearCpu"
    assert not scope["distributed_candidate_placement_executed"]
    assert scope["remote_stage_count"] == 48
    assert scope["remote_stage_count_by_role"] == {"mlp_gate_up": 24, "mlp_down": 24}
    assert set(scope["remote_stage_ids"]) == {
        key for key, row in metadata.items() if row["role"] in {"mlp_gate_up", "mlp_down"}
    }
    resources = scope["resources_per_live_sequence"]
    assert resources["retained_client_previous_q_i8_bytes"] == 138240
    assert resources["retained_client_exact_product_i32_bytes"] == 1019904
    assert resources["retained_client_activation_scale_f32_bytes"] == 192
    assert resources["retained_client_temporal_state_bytes"] == 1158336
    assert resources["dense_all_link_bytes_per_initialization_sequence"] == 2104320
    assert resources["dense_online_bytes_per_initialization_sequence"] == 1317888
    assert resources["dense_exact_ring_input_bytes_per_initialization_sequence"] == 531456
    assert resources["dense_initialization_remote_macs_per_sequence"] == 313786368
    groups = {}
    for role in widths:
        measurement = probe.Measurements()
        measurement.counts.update({"dense_all_link_bytes": 7, "dense_remote_macs": 13})
        groups[("test", role, "decode")] = measurement
    # Wrong cohort/phase and both attention roles must contribute nothing.
    extra = probe.Measurements()
    extra.counts["dense_all_link_bytes"] = 1000000
    groups[("other", "mlp_down", "decode")] = extra
    groups[("test", "mlp_gate_up", "prefill")] = extra
    assert probe.sdk_phase_arithmetic(groups, "test", "decode") == {
        "dense_all_link_bytes": 14,
        "dense_remote_macs": 26,
    }


def test_locked_sdk_arithmetic_sums_only_mlp_and_preserves_all_baseline_roles():
    evidence = json.loads(
        (ROOT / "docs/evidence/temporal-delta-screen-2026-10-01.json").read_text()
    )
    scope = evidence["sdk_derived_scope"]
    assert scope["remote_stage_count"] == 48
    assert scope["resources_per_live_sequence"]["retained_client_temporal_state_bytes"] == 1158336
    for role, row in evidence["retained_state_by_role"].items():
        assert row["sdk_remote_under_attention_candidate_client_placement"] == (
            role in {"mlp_gate_up", "mlp_down"}
        )
        for cohort in ("screen", "confirmation"):
            for phase in ("prefill_adjacent_positions", "same_token_decode"):
                assert f"{cohort}/{role}/{phase}" in evidence["by_role_phase"]
    for key, aggregate in evidence["sdk_remote_transition_arithmetic_only"].items():
        cohort, phase = key.split("/")
        for metric in (
            "dense_all_link_bytes",
            "dense_online_bytes",
            "dense_remote_macs",
            "masked_ring_oracle_all_link_bytes",
            "masked_ring_oracle_online_bytes",
            "plaintext_support_oracle_remote_macs",
        ):
            assert aggregate[metric] == sum(
                evidence["by_role_phase"][f"{cohort}/{role}/{phase}"][metric]
                for role in ("mlp_gate_up", "mlp_down")
            ), (key, metric)


@pytest.mark.quality
@pytest.mark.skipif(
    os.environ.get("PLLM_RUN_TEMPORAL_DELTA") != "1",
    reason="opt in to one-thread cached checkpoint replay",
)
def test_real_temporal_replay_reproduces_every_locked_measurement():
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/probe_temporal_deltas.py"), "--summary"],
        cwd=ROOT,
        env={
            **os.environ,
            "HF_HUB_CACHE": HUB_CACHE,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "PYTHONPATH": "python",
        },
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    actual = json.loads(process.stdout)
    evidence = json.loads(
        (ROOT / "docs/evidence/temporal-delta-screen-2026-10-01.json").read_text()
    )

    def check_subset(locked, found, path=""):
        if isinstance(locked, dict):
            for key, value in locked.items():
                assert key in found, f"{path}/{key}"
                check_subset(value, found[key], f"{path}/{key}")
        else:
            assert locked == found, path

    # Evidence deliberately locks a compact subset; full probe retains richer
    # per-stage histograms. Every stored integer, quantile and digest must match.
    check_subset(evidence, actual)
