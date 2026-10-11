"""Tiny synthetic integer correctness only: no checkpoint, SVD or timing cohort."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("residual_probe", Path(__file__).parents[1] / "scripts/probe_residual_delegation.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_bounded_integer_search_keeps_every_residual_entry():
    rng = np.random.default_rng(5)
    weight = rng.integers(-127, 128, (32, 32), dtype=np.int8)
    inputs = rng.integers(-127, 128, (4, 32), dtype=np.int8)
    for a, b, e in probe.residual_search(weight, rank=1):
        result, admitted = probe.residual_exact(weight, a, b, e, inputs, native=True)
        assert b.shape[0] * sum(weight.shape) * 10 <= weight.size
        assert result[:, [0, 7, 31]].tolist() == probe.residual_oracle(weight, inputs, [0, 7, 31])
        assert admitted == (-128 <= e.min() and e.max() <= 127)


def test_low_rank_public_structure_reduces_widths_without_approximation():
    a = np.full((32, 1), 2, np.int8)
    b = np.full((1, 32), 16, np.int8)
    weight = (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int8)
    factors = list(probe.residual_search(weight, rank=1))
    left, right, e = factors[-1]
    assert not e.any()
    assert probe.residual_body_bytes(e)["total"] < probe.residual_body_bytes(weight)["total"]
    result, admitted = probe.residual_exact(weight, left, right, e, np.full((2, 32), -127, np.int8))
    assert admitted and np.all(result == -127 * 32 * 32)


def test_dense_int16_residual_is_exact_but_native_gate_rejects():
    weight = np.full((32, 32), 127, np.int8)
    a = np.full((32, 1), -2, np.int8)
    b = np.full((1, 32), 16, np.int8)
    e = weight.astype(np.int16) - a.astype(np.int16) @ b.astype(np.int16)
    result, admitted = probe.residual_exact(weight, a, b, e, np.full((1, 32), 127, np.int8))
    assert not admitted and e.min() == 159
    assert np.all(result == 127 * 127 * 32)
    from pllm.runtime.native import MaskedGEMM, NativeKernelError
    with pytest.raises(NativeKernelError, match="values must be"):
        MaskedGEMM(threads=1, build=False).compile(e)


def test_dropping_residual_rejects():
    weight = np.ones((32, 32), np.int8)
    with pytest.raises(ValueError, match="E must remain exact"):
        probe.residual_exact(weight, np.zeros((32, 0), np.int8), np.zeros((0, 32), np.int8),
                             np.zeros((32, 32), np.int16), np.ones((1, 32), np.int8))


@pytest.mark.parametrize("rank,seeds", [(4, 1), (1, 3), (5, 1)])
def test_mac_and_search_limits_reject(rank, seeds):
    with pytest.raises(ValueError):
        list(probe.residual_search(np.ones((32, 32), np.int8), rank=rank, seeds=seeds))


def test_ring_thresholds_are_strict_and_exact():
    # 127 * (257 + 1) = 32766 remains u16; next integer L1 crosses threshold.
    assert probe.residual_bounds(np.array([[258]], np.int16))[2] == 16
    assert probe.residual_bounds(np.array([[259]], np.int16))[2] == 24
    assert probe.residual_bounds(np.zeros((2, 2), np.int8))[1].tolist() == [1, 1]


def test_tiny_native_residue_correction_and_original_stage_parity():
    weight = np.full((32, 32), 32, np.int8)
    a, b, e = list(probe.residual_search(weight, rank=1))[-1]
    # Keep nonzero E to exercise signed reconstruction rather than zero only.
    weight[0, 0] = 31
    e = weight.astype(np.int16) - a.astype(np.int16) @ b.astype(np.int16)
    inputs = np.full((2, 32), -127, np.int8)
    inputs[1].fill(127)
    result, admitted = probe.residual_exact(weight, a, b, e, inputs, native=True)
    assert admitted and result[0, 0] == -127 * (31 + 31 * 32)
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime.quantization import dequantize_matmul
    original = MaskedGEMM(threads=1, build=False).compile(weight).clear(inputs)
    activation_scales = np.array([0.1, 0.25], np.float32)
    weight_scales = np.linspace(0.01, 0.5, 32, dtype=np.float32)
    np.testing.assert_array_equal(
        dequantize_matmul(result.astype(np.int32), activation_scales, weight_scales).view(np.uint32),
        dequantize_matmul(original, activation_scales, weight_scales).view(np.uint32))
