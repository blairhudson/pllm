"""Small independent exact-ring tests; no checkpoint scan or timing cohort."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("exception_probe", Path(__file__).resolve().parents[1] / "scripts/probe_exception_corrections.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("base_bits", [4, 5, 6, 7])
@pytest.mark.parametrize("ring", [16, 24, 32])
def test_extremes_wraparound_partial_blocks(base_bits, ring):
    weights = np.resize(np.asarray([-128, 127, -1, 0, 1, -8, 7, 64], np.int8), (4, 31))
    inputs = np.resize(np.asarray([0, (1 << ring) - 1, 128, 255, 32768, (1 << (ring - 1)) - 1], np.uint32), (3, 31))
    actual = probe.project(*probe.split(weights, base_bits), inputs, ring, base_bits, block=7)
    np.testing.assert_array_equal(actual, probe.integer_oracle(weights, inputs, ring))


@pytest.mark.parametrize("all_exceptions", [False, True])
def test_empty_and_full_exception_sets(all_exceptions):
    weights = np.resize(np.asarray([-128, 127] if all_exceptions else [-8, 7], np.int8), (3, 9))
    arrays = probe.split(weights, 4)
    assert len(arrays[2]) == (weights.size if all_exceptions else 0)
    inputs = np.full((2, 9), 0xffffffff, dtype=np.uint32)
    np.testing.assert_array_equal(probe.project(*arrays, inputs, 32, 4), probe.integer_oracle(weights, inputs, 32))


@pytest.mark.parametrize("bits", [4, 5, 6, 7])
def test_packed_base_roundtrip_and_padding(bits):
    h = 1 << (bits - 1)
    base = np.asarray([[-h, h - 1, -1, 0, 1]], np.int8)
    packed = probe.pack_base(base, bits)
    np.testing.assert_array_equal(probe.unpack_base(packed, base.shape, bits), base)
    with pytest.raises(ValueError):
        probe.unpack_base(packed + b"\0", base.shape, bits)
    if base.size * bits % 8:
        with pytest.raises(ValueError):
            probe.unpack_base(packed[:-1] + bytes([packed[-1] | 128]), base.shape, bits)


@pytest.mark.parametrize("fault", ["past_end", "negative_wrap", "duplicate", "unsorted", "wrong_pointer", "zero", "dtype"])
def test_invalid_sparse_indices_reject(fault):
    base, ptr, ix, coef = probe.split(np.asarray([[127, 127, 127]], np.int8), 4)
    if fault == "past_end": ix[0] = 3
    if fault == "negative_wrap": ix[0] = 0xffffffff
    if fault == "duplicate": ix[1] = ix[0]
    if fault == "unsorted": ix[:] = ix[::-1]
    if fault == "wrong_pointer": ptr[-1] = 2
    if fault == "zero": coef[0] = 0
    if fault == "dtype": ix = ix.astype(np.int64)
    with pytest.raises(ValueError):
        probe.project(base, ptr, ix, coef, np.zeros((1, 3), np.uint32), 32, 4)


def test_signed_asymmetric_bound_and_strict_limit():
    for bits in range(4, 9):
        p = probe.bounds(bits)
        h = 1 << (bits - 1)
        assert p["product_max"] == 128 * h
        assert p["product_min"] == min(-128 * (h - 1), -127 * h)
        assert p["partial_abs_max"] < 1 << 24
        assert (p["block"] + 1) * p["product_max"] >= 1 << 24
    assert [probe.bounds(b)["block"] for b in range(4, 9)] == [16383, 8191, 4095, 2047, 1023]


def test_float32_counterexample_beyond_bound():
    # At 16384 maximal b4 products the sum is 2^24. One additional unit is lost.
    exact = 16384 * (-128) * (-8) + (-1) * (-1)
    assert exact == (1 << 24) + 1
    assert int(np.float32(1 << 24) + np.float32(1)) != exact


def test_long_real_partial_block():
    weights = np.full((1, 16384), -8, np.int8)
    inputs = np.full((1, 16384), 128, np.uint32)
    np.testing.assert_array_equal(probe.project(*probe.split(weights, 4), inputs, 16, 4), probe.integer_oracle(weights, inputs, 16))


def test_invalid_ring_base_and_block():
    arrays = probe.split(np.ones((1, 1), np.int8), 4)
    with pytest.raises(ValueError): probe.project(*arrays, np.asarray([[65536]], np.uint32), 16, 4)
    with pytest.raises(ValueError): probe.project(*arrays, np.zeros((1, 1), np.uint32), 17, 4)
    with pytest.raises(ValueError): probe.project(*arrays, np.zeros((1, 1), np.uint32), 32, 4, block=16384)
    with pytest.raises(ValueError): probe.split(np.ones((1, 1), np.int16), 4)


def test_calls_and_storage_counterexamples():
    gate = probe.stage_cost(9728, 896, 150, 32, 4, 0)
    assert gate["native_digit8_sgemm_calls"] == gate["proposed_sgemm_calls"] == 40
    down = probe.stage_cost(896, 4864, 150, 32, 4, 0)
    assert down["native_digit8_sgemm_calls"] == 20
    assert down["proposed_sgemm_calls"] == 4
    dense = probe.stage_cost(896, 4864, 150, 32, 4, 896 * 4864)
    assert not dense["packed_storage_better_than_original"]
    assert dense["whole_topology_admitted"] is None
    assert dense["workspace_bytes"] <= 64 << 20


def test_wider_block_can_force_more_tiles():
    cost = probe.stage_cost(1024, 16384, 150, 32, 4, 0)
    assert cost["output_tile"] < 1024
    assert cost["mask_digit_conversion_elements"] > cost["native_digit8_mask_digit_conversion_elements"]
    assert cost["float_weight_conversion_elements"] == cost["native_digit8_float_weight_conversion_elements"]


def test_uniform_public_histogram_tail_counterexample():
    hist = np.ones(256, np.int64)
    for bits in range(4, 8):
        exceptions = 256 - (1 << bits)
        result = probe.histogram_summary(hist, 1, 256, bits, [exceptions])
        assert result["density"] == 1 - (1 << bits) / 256
        assert not result["cost_150_ring32"]["packed_storage_better_than_original"]


def test_split_coefficients_reject_non_endpoint_and_excess():
    base, ptr, ix, coeff = probe.split(np.asarray([[127]], np.int8), 4)
    base[0, 0] = 0
    with pytest.raises(ValueError): probe.project(base, ptr, ix, coeff, np.zeros((1, 1), np.uint32), 32, 4)
    base[0, 0] = 7
    coeff[0] = 127
    with pytest.raises(ValueError): probe.project(base, ptr, ix, coeff, np.zeros((1, 1), np.uint32), 32, 4)


def test_reproducibility():
    assert probe.screening() == probe.screening()


def test_masked_correction_identity_without_mask_reuse():
    # Algebra only. Production mask generator/domains/lifecycle are not replaced.
    weights = np.asarray([[-128, 127, 9], [7, -8, 0]], np.int8)
    rng = np.random.default_rng(23)
    for ring in (16, 24, 32):
        mask = (1 << ring) - 1
        x = rng.integers(0, 1 << ring, (1, 3), dtype=np.uint32)
        r = rng.integers(0, 1 << ring, (1, 3), dtype=np.uint32)
        s = rng.integers(0, 1 << ring, (1, 2), dtype=np.uint32)
        wr = probe.project(*probe.split(weights, 4), r, ring, 4)
        correction = (wr.astype(np.int64) - s.astype(np.int64)) & mask
        masked = ((x.astype(np.int64) - r.astype(np.int64)) & mask).astype(np.uint32)
        reply = (probe.integer_oracle(weights, masked, ring).astype(np.int64) + correction + s) & mask
        np.testing.assert_array_equal(reply, probe.integer_oracle(weights, x, ring))


def test_source_locked_row_chunk_scanner(tmp_path):
    """Exercise header admission and a partial final native quantization chunk."""
    from safetensors.numpy import save_file
    from pllm.runtime.quantization import quantize_weight_per_row

    weights = np.random.default_rng(309).normal(size=(37, 41)).astype(np.float32)
    key = "fixture.down.weight"
    (tmp_path / "config.json").write_text("{}")
    save_file({key: weights}, tmp_path / "model.safetensors")
    lock = {"revision": "generated-test-only", "sha256": {
        name: probe.digest_file(tmp_path / name)
        for name in ("config.json", "model.safetensors")}}
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    payload = {"checkpoint": str(tmp_path), "source_lock": str(lock_path), "keys": [key]}
    result = probe.scan(payload)
    stage = result["stages"][0]
    expected = quantize_weight_per_row(weights, bits=8).values
    assert stage["signed_i8_histogram"] == np.bincount(
        expected.astype(np.int16).ravel() + 128, minlength=256).tolist()
    assert result["estimated_scan_peak_bytes"] == (
        (tmp_path / "model.safetensors").stat().st_size + 32 * 32 * 41 + (128 << 20))
    assert stage["shape"] == [37, 41]
    for candidate in stage["candidates"]:
        half = 1 << (candidate["base_bits"] - 1)
        assert candidate["exceptions"] == np.count_nonzero((expected < -half) | (expected >= half))
    (tmp_path / "config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="checkpoint source hash mismatch"):
        probe.scan(payload)
