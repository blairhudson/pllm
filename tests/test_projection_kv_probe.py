"""Independent boundary checks for the research-only projection-origin codec."""
import dataclasses
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("projection_kv_probe", Path(__file__).parents[1] / "scripts/probe_projection_kv.py")
probe = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def profile():
    weights = np.array([[0, 0], [1, -2], [-127, 127]], np.int8)
    return probe.Profile.from_weights(weights, 8, np.array([1, .1, .01], np.float32))


def seal(page, **changes):
    page = dataclasses.replace(page, **changes)
    return dataclasses.replace(page, commitment=hashlib.sha256(page.payload + page.scales + page.positions).hexdigest())


def test_public_l1_widths_and_independent_lsb_stream_endpoints():
    p = profile()
    assert p.bounds == (0, 381, 32258)
    assert list(p.widths) == [1, 10, 16]
    integers = np.array([[0, -381, 32258], [0, 381, -32258], [0, 0, 0]], np.int64)
    page = probe.Page.capture(integers, np.ones(3, np.float32), np.arange(3, dtype=np.int64), p, verified=True)
    bits = []
    for row in integers:
        for value, width in zip(row, p.widths):
            bits.extend((int(value) >> bit) & 1 for bit in range(width))
    independent = bytes(sum(bits[i+j] << j for j in range(min(8, len(bits)-i))) for i in range(0, len(bits), 8))
    assert page.payload == independent
    np.testing.assert_array_equal(page.integers(p), integers)
    with pytest.raises(ValueError):
        seal(page, payload=page.payload[:-1] + bytes([page.payload[-1] | 0x80])).integers(p)
    with pytest.raises(ValueError):
        seal(page, payload=page.payload[:-1]).integers(p)


@pytest.mark.parametrize("fault", ["range", "verify", "float", "scale", "position", "rows"])
def test_fail_closed_capture(fault):
    p = profile()
    integer = np.zeros((1, 3), np.int64)
    scales, positions, verified = np.ones(1, np.float32), np.zeros(1, np.int64), True
    if fault == "range": integer[0, 1] = 382
    if fault == "verify": verified = False
    if fault == "float": integer = integer.astype(np.float32)
    if fault == "scale": scales[0] = np.nan
    if fault == "position": positions[0] = -1
    if fault == "rows":
        integer = np.zeros((65, 3), np.int64)
        scales, positions = np.ones(65, np.float32), np.arange(65, dtype=np.int64)
    with pytest.raises(ValueError):
        probe.Page.capture(integer, scales, positions, p, verified=verified)


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_invalid_scale_metadata(scale):
    with pytest.raises(ValueError):
        probe.Profile.from_weights(np.ones((2, 3), np.int8), 8, np.full(2, scale, np.float32))


def test_restored_range_and_metadata_checked_even_with_new_unkeyed_commitment():
    from pllm import _native
    p = profile()
    page = probe.Page.capture(np.zeros((1, 3), np.int64), np.ones(1, np.float32), np.zeros(1, np.int64), p, verified=True)
    raw = _native.offset_pack_rows(np.array([[0, 382, 0]], "<u4").tobytes(), p.widths, 1)
    with pytest.raises(ValueError, match="bound"):
        seal(page, payload=raw).integers(p)
    with pytest.raises(ValueError, match="positive"):
        seal(page, scales=np.zeros(1, "<f4").tobytes()).integers(p)
    with pytest.raises(ValueError, match="commitment"):
        dataclasses.replace(page, payload=bytes([1]) + page.payload[1:]).integers(p)


def test_native_original_fused_decoder_matches_reconstructed_f32_order():
    from pllm import _native
    p = probe.Profile.from_weights(np.full((2, 2048), 127, np.int8), 8,
                                  np.array([.003, 1e-5], np.float32), np.array([-.125, .003], np.float32))
    # Above 2**24 tests integer-to-f32 rounding, not only exactly representable integers.
    integer = np.array([[16777217, -16777219], [33032192, -33032192]], np.int64)
    scales = np.array([.00073, .0031], np.float32)
    page = probe.Page.capture(integer, scales, np.array([0, 1], np.int64), p, verified=True)
    raw = _native.prepared_dequantize_output(page.payload, bytes(4 * integer.size), p.widths, 2,
                                            page.scales, p.weight_scales, p.bias)
    np.testing.assert_array_equal(page.projection(p).view(np.uint32), np.frombuffer(raw, "<f4").reshape(2, 2).view(np.uint32))


def test_original_attention_outputs_qwen2_qwen3_llama_and_odd_slabs():
    result = probe.synthetic_probe()
    assert [r["family_fixture"] for r in result["fixtures"]] == ["qwen2", "qwen3", "llama"]
    assert all(r["kv_and_attention_bits_equal"] for r in result["fixtures"])


def test_source_lock_rejects_different_numeric_identity():
    identity = probe.source_identity()
    probe.require_identity(identity["digest"])
    with pytest.raises(ValueError, match="identity"):
        probe.require_identity("0" * 64)


def test_coefficients_bind_original_domain():
    pos = np.array([0], np.int64)
    c = probe.CapturedCoefficients(pos, np.ones((1, 1, 1, 2), np.float32), np.zeros((1, 1, 1, 2), np.float32), 2, 10000.)
    with pytest.raises(ValueError): c.get(pos + 1, 2, 10000.)
    with pytest.raises(ValueError): c.get(pos, 2, 5000.)
