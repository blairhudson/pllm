"""The same reference vectors exercise both paths. CI requires a real extension."""

from __future__ import annotations

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pllm.runtime.native import MaskedGEMM, NativeKernelError
from pllm.runtime.compact import pack_unsigned, unpack_unsigned
from pllm.runtime.quantization import quantize_activation_per_row
from pllm.runtime.stage_protocol import StageCorrelation, encode_signed_with_mask, decode_unmasked

RUST_PRESENT = importlib.util.find_spec("pllm._native") is not None


def test_native_required_job_cannot_skip_or_fallback():
    if os.environ.get("PLLM_REQUIRE_RUST") == "1":
        assert RUST_PRESENT, "Build the native wheel before running this job"
        assert MaskedGEMM().backend == "rust"


def test_native_executor_uses_available_cores_by_default(monkeypatch):
    monkeypatch.delenv("PLLM_NATIVE_THREADS", raising=False)
    assert MaskedGEMM().threads == min(32, os.cpu_count() or 1)


@pytest.mark.parametrize("width", [1, 7, 8, 15, 16, 17, 31, 32, 33, 511, 4097])
@pytest.mark.parametrize("modulus", [65537, 33554467, 2147483647])
def test_integer_simd_extrema_and_tails(width, modulus):
    rng = np.random.default_rng(width)
    w = rng.integers(-128, 128, (5, width), dtype=np.int8)
    x = rng.integers(0, modulus, (3, width), dtype=np.uint32)
    x[0] = modulus - 1
    expected = x.astype(np.int64) @ w.astype(np.int64).T % modulus
    for simd in (False, True):
        stage = MaskedGEMM(threads=2, simd=simd).compile(w)
        np.testing.assert_array_equal(stage.modular(x, modulus), expected)


def test_old_cpp_product_overflow_regression():
    stage = MaskedGEMM(threads=1).compile(np.full((1, 8), 7, dtype=np.int8))
    result = stage.modular(np.full((1, 8), 2147483646, dtype=np.uint32), 2147483647)
    assert result.tolist() == [[2147483591]]


@pytest.mark.parametrize("batch", [0, 1, 7, 8, 9, 65])
def test_batches_zero_rows_and_immutability(batch):
    w = np.ones((2, 37), np.int8)
    w[0] = 0
    compiled = MaskedGEMM(threads=2).compile(w)
    w[:] = 7  # Compilation is a snapshot, not an id-based alias cache.
    x = np.ones((batch, 37), np.uint32)
    expected = np.tile([0, 37], (batch, 1))
    np.testing.assert_array_equal(compiled.modular(x, 65537), expected)


def test_wrap32_and_clear_extremes():
    w = np.full((3, 8193), -128, np.int8)
    s = MaskedGEMM(threads=2).compile(w)
    x = np.full((2, 8193), np.iinfo(np.uint32).max, np.uint32)
    np.testing.assert_array_equal(
        s.wrap32(x), (x.astype(np.int64) @ w.astype(np.int64).T).astype(np.uint32)
    )
    np.testing.assert_array_equal(
        s.clear(np.full((2, 8193), -128, np.int8)), np.full((2, 3), 8193 * 16384)
    )


def test_wrap64_extremes_and_leading_dimensions():
    weights = np.array([[127, -128, 0], [-1, 1, 7]], dtype=np.int8)
    inputs = np.array(
        [
            [[0, np.iinfo(np.uint64).max, 1], [2**63, 7, 11]],
            [[13, 17, 19], [23, 29, 31]],
        ],
        dtype=np.uint64,
    )
    expected = np.asarray(
        inputs.astype(object) @ weights.astype(object).T % (1 << 64), dtype=np.uint64
    )
    stage = MaskedGEMM(threads=2).compile(weights)
    np.testing.assert_array_equal(stage.wrap64(inputs), expected)
    np.testing.assert_array_equal(stage.wrap64(inputs[0, 0]), expected[0, 0])
    np.testing.assert_array_equal(MaskedGEMM(threads=2).wrap64(weights, inputs), expected)
    np.testing.assert_array_equal(stage.wrap64(inputs.astype(">u8")), expected)


@pytest.mark.parametrize("q", [65537, 786433, (1 << 53) - 111])
def test_ciphertext_coefficients_against_unbounded_integers(q):
    w = np.array([[127, -128, 7], [-7, 0, -128]], np.int8)
    a = np.array([[q - 1, 0, 1], [q - 2, q - 1, 0], [q - 3, 17, q - 1]], np.uint64)
    expected = np.array((w.astype(object) @ a.astype(object)) % q, dtype=np.uint64)
    np.testing.assert_array_equal(MaskedGEMM(threads=2).compile(w).coefficients(a, q), expected)


def test_shared_executor_concurrency():
    rng = np.random.default_rng(4)
    w = rng.integers(-7, 8, (97, 257), dtype=np.int8)
    x = rng.integers(0, 786433, (4, 257), dtype=np.uint32)
    stage = MaskedGEMM(threads=2).compile(w)
    expected = x.astype(np.int64) @ w.astype(np.int64).T % 786433
    with ThreadPoolExecutor(4) as pool:
        for result in pool.map(lambda _: stage.modular(x, 786433), range(12)):
            np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_codecs_empty_and_large(width):
    for size in (0, 1, 33, 65539):
        x = np.random.default_rng(size).integers(0, 1 << (8 * width), size, dtype=np.uint32)
        payload = pack_unsigned(x, width=width)
        assert len(payload) == size * width
        np.testing.assert_array_equal(unpack_unsigned(payload, width=width, count=size), x)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_quantization_matches_numpy_float32_sequence(bits):
    x = np.random.default_rng(5).normal(size=(5, 3, 137)).astype(np.float32)
    x[0] = 0
    qmax = (1 << (bits - 1)) - 1
    flat = x.reshape(-1, 137)
    maxima = np.abs(flat).max(axis=1)
    scales = np.where(maxima > 0, maxima / qmax, 1).astype(np.float32)
    expected = np.clip(np.rint(flat / scales[:, None]), -qmax, qmax).astype(np.int8)
    actual = quantize_activation_per_row(x, bits=bits)
    np.testing.assert_array_equal(actual.values, expected)
    np.testing.assert_array_equal(actual.scales, scales)


def test_quantization_ties():
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5], np.float32)
    actual = quantize_activation_per_row(x, scales=1.0)
    assert actual.values.tolist() == [[0, 2, 2, 0, -2, -2]]


@pytest.mark.parametrize(
    "p,ring", [(65536, "u16"), (65537, "prime"), (33554467, "prime"), (1 << 32, "u32")]
)
def test_mask_unmask(p, ring):
    from pllm.runtime.secure_random import uniform_residues

    x = np.array([-128, -7, 0, 7, 127], np.int8)
    r = uniform_residues(p, x.shape)
    c = StageCorrelation("mask-1", "layer", r, r.copy(), p, ring)
    masked = encode_signed_with_mask(x, c)
    np.testing.assert_array_equal(decode_unmasked(masked, c), x.astype(np.int64))


def test_invalid_inputs_fail_before_narrowing():
    kernel = MaskedGEMM()
    with pytest.raises(NativeKernelError):
        kernel.compile(np.array([[128]], np.int64))
    stage = kernel.compile(np.ones((1, 1), np.int8))
    for value in [-1, 65537, 2**40]:
        with pytest.raises(NativeKernelError):
            stage.modular(np.array([[value]], np.int64), 65537)
    with pytest.raises(NativeKernelError):
        stage.modular(np.array([[1.1]]), 65537)
    with pytest.raises(NativeKernelError):
        stage.modular(np.array([[1, 2]], np.uint32), 65537)
    with pytest.raises(NativeKernelError):
        MaskedGEMM(threads=1000)


@pytest.mark.rust
@pytest.mark.skipif(not RUST_PRESENT, reason="compiled Rust extension not present")
def test_raw_binding_truncated_and_malicious_lengths():
    from pllm import _native

    executor = _native.Executor(1)
    matrix = _native.Matrix(b"\x01", 1, 1)
    with pytest.raises(ValueError):
        executor.modular(matrix, b"\x00", 1, 65537)
    with pytest.raises(ValueError):
        executor.wrap64(matrix, b"\x00", 1)
    with pytest.raises(ValueError):
        _native.Matrix(b"\x00", 2**63, 2**63)
    with pytest.raises(ValueError):
        _native.unpack_unsigned(b"\x00", 3)
    with pytest.raises(ValueError):
        _native.quantize(np.array([float("nan")], "<f4").tobytes(), 1, 1)


@pytest.mark.rust
@pytest.mark.skipif(not RUST_PRESENT, reason="compiled Rust extension not present")
def test_native_backend_is_really_loaded():
    from pllm.native import capabilities

    assert capabilities()["implementation"] == "rust"
    assert capabilities()["compiled"]


@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), 0.0, -1.0])
def test_quantization_rejects_invalid_scales(bad_scale):
    from pllm.runtime.quantization import QuantizationError

    with pytest.raises(QuantizationError):
        quantize_activation_per_row(np.ones((1, 3), np.float32), scales=bad_scale)
