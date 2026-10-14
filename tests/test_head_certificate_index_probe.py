"""Tiny independent correctness screens, never timing or real checkpoint cohorts."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_head_certificate_index.py"
SPEC = importlib.util.spec_from_file_location("head_certificate_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def check(weights, x, scales=None, bias=None, activation_scale=1, **kwargs):
    w = np.asarray(weights, np.int8)
    index = probe.HeadIndex(w, np.ones(len(w), np.float32) if scales is None else scales,
                            bias, leaf_rows=1)
    result = index.query(np.asarray(x, np.int8), activation_scale, **kwargs)
    probe.verify_oracle(index, np.asarray(x, np.int8), activation_scale, result)
    return index, result


def test_equal_ties_and_first_index():
    _, result = check([[7], [8], [1]], [1], [8, 7, 1])
    assert result["winner"] == 0
    assert result["skipped"]
    _, result = check([[0], [2], [2]], [1])
    assert result["winner"] == 1
    _, result = check([[-128], [127], [0]], [0])
    assert result["winner"] == 0


def test_negative_dot_scale_corner_counterexample():
    # Using only the largest positive scale at a negative upper endpoint is unsafe:
    # -1*10=-10 falsely prunes a row scoring -1 against incumbent -5.
    index = probe.HeadIndex(np.array([[-1], [-1]], np.int8), [1, 10], leaf_rows=2)
    assert index.upper(index.root, np.array([1], np.int8), 1) == -1
    _, result = check([[-5], [-1], [-1]], [1], [1, 1, 10])
    assert result["winner"] == 1


def test_rounding_tie_counterexample_and_signed_zero():
    # Exact dot difference disappears after f32 cast. Float64 bounds/selection
    # would incorrectly choose row 1 instead of first-index float32 tie row 0.
    w = np.full((2, 2048), 127, np.int8)
    w[0, -1] = 125
    w[1, -1] = 126
    x = np.full(2048, 127, np.int8)
    x[-1] = 1
    _, result = check(w, x)
    assert result["winner"] == 0
    index = probe.HeadIndex(np.array([[-1]], np.int8), [1])
    result = index.query(np.array([1], np.int8), np.nextafter(np.float32(0), np.float32(1)))
    probe.verify_oracle(index, np.array([1], np.int8), np.nextafter(np.float32(0), np.float32(1)), result)
    _, result = check([[-1], [0], [1]], [1], [np.nextafter(np.float32(0), np.float32(1))] * 3,
                      activation_scale=np.nextafter(np.float32(0), np.float32(1)))
    assert result["winner"] == 0


def test_max_magnitudes_bias_and_scale_boundaries():
    w = np.full((3, 16384), -128, np.int8)
    w[1] = 127
    w[2, ::2] = 127
    check(w, np.full(16384, -128, np.int8), [1, 1e-5, 1e5], [1, -2, 3], 0.125)
    _, result = check([[10], [11], [-128]], [1], [1, 1, 1], [0, -1, 200])
    assert result["winner"] == 2
    for scale in [np.nextafter(np.float32(1), np.float32(0)), np.float32(1),
                  np.nextafter(np.float32(1), np.float32(2))]:
        check([[127, -128], [-128, 127], [127, 127]], [127, -128],
              [scale, 2 * scale, scale], [0.5, -0.5, 1], scale)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0, -1])
def test_nonfinite_or_nonpositive_scales_rejected(bad):
    with pytest.raises(ValueError):
        probe.HeadIndex(np.array([[1]], np.int8), [bad])
    index = probe.HeadIndex(np.array([[1]], np.int8), [1])
    with pytest.raises(ValueError):
        index.query(np.array([1], np.int8), bad)


def test_nonfinite_bias_overflow_and_wrong_dtype_rejected():
    with pytest.raises(ValueError):
        probe.HeadIndex(np.array([[1]], np.int8), [1], [np.nan])
    with pytest.raises(ValueError):
        probe.HeadIndex(np.array([[128]], np.int64), [1])
    index = probe.HeadIndex(np.array([[127], [-128]], np.int8), [1e30, 1e30])
    with pytest.raises(ValueError):
        index.query(np.array([127], np.int8), 1e30)
    with pytest.raises(ValueError):
        index.query(np.array([1], np.int64))
    with pytest.raises(ValueError):
        index.weights[0, 0] = 0


def test_budget_failure_full_exact_recompute_charged():
    _, result = check([[1], [5], [-1]], [1], max_bound_nodes=0)
    assert result["fallback"] and result["winner"] == 1
    assert result["exact_row_evaluations"] == 4
    assert result["skipped"] == []


def test_installed_float32_dequant_contract_no_bias():
    from pllm.runtime.quantization import dequantize_matmul
    dots = np.array([-(1 << 28), (1 << 24) + 1, (1 << 24) + 3, 0], np.int64)
    scales = np.array([0.25, np.nextafter(np.float32(1), np.float32(2)), 1e-5, 1], np.float32)
    expected = dequantize_matmul(dots, np.float32(0.125), scales)
    actual = probe.dequant(dots, np.float32(0.125), scales, None)
    assert actual.tobytes() == expected.tobytes()


def test_random_each_skipped_row_independent_oracle():
    rng = np.random.default_rng(51)
    for _ in range(20):
        w = rng.integers(-128, 128, (17, 31), dtype=np.int8)
        scales = np.exp2(rng.integers(-10, 10, 17)).astype(np.float32)
        bias = rng.uniform(-100, 100, 17).astype(np.float32)
        x = rng.integers(-128, 128, 31, dtype=np.int8)
        for leaves in (1, 4, 17):
            index = probe.HeadIndex(w, scales, bias, leaf_rows=leaves)
            result = index.query(x, 0.0078125)
            probe.verify_oracle(index, x, 0.0078125, result)


def test_tiny_screen_report_redaction_and_costs():
    report = probe.screen({})
    assert len(report["cases"]) == 4
    for case in report["cases"]:
        assert "winner" not in case and "skipped" not in case
        assert case["source_admission_hash_bytes"] == case["immutable_source_bytes"]
        assert case["oracle_macs"] == case["full_head_macs"]
    index = probe.HeadIndex(np.array([[1], [2]], np.int8), [1, 2])
    assert len(index.serialize()) == index.index_payload_bytes


def test_source_locked_raw_prefix_tiny_only(tmp_path):
    weights = tmp_path / "public.i8"
    scales = tmp_path / "scales.f32"
    weights.write_bytes(np.array([[1, 2], [3, 4], [-1, -2]], np.int8).tobytes())
    scales.write_bytes(np.array([1, 2, 3], dtype="<f4").tobytes())
    payload = dict(weights=str(weights), scales=str(scales), rows=3, columns=2,
                   scan_rows=2, weights_sha256=probe.file_digest(weights),
                   scales_sha256=probe.file_digest(scales))
    report = probe.screen(payload)
    assert report["source"]["full_artifact_delivery_bytes"] == 18
    assert report["source"]["admitted_rows"] == 2
    assert all(case["rows"] == 2 for case in report["cases"])
    with pytest.raises(ValueError, match="source lock"):
        probe.screen(payload | {"weights_sha256": "0" * 64})
    with pytest.raises(ValueError, match="prefix scan"):
        probe.screen(payload | {"scan_rows": 0})
