"""Fused client boundaries preserve encoded inputs and every float32 output bit."""
import numpy as np
import pytest

from pllm import _native
from pllm.runtime.stage_protocol import PreparedStageBatchRequest


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("bits", [1, 7, 11, 16, 23, 24, 31, 32])
def test_fused_input_matches_independent_numpy_quantization_and_bit_packing(precision, bits):
    rng = np.random.default_rng(795)
    values = rng.normal(size=(17, 33)).astype("<f4")
    maximum = (1 << (precision - 1)) - 1
    values[0] = 0
    values[1, :9] = [-maximum, maximum, -.5, .5, -1.5, 1.5, -2.5, 2.5, -0.0]
    masks = rng.integers(0, 1 << 32, values.shape, dtype="<u4")
    maxima = np.abs(values).max(axis=1)
    scales = np.where(maxima > 0, maxima / maximum, 1).astype("<f4")
    codes = np.clip(np.rint(values / scales[:, None]), -maximum, maximum).astype(np.int64)
    masked = (codes - masks.astype(np.int64)) & ((1 << bits) - 1)
    # A Python integer bit stream is independent of both native packers.
    word = sum(int(v) << (bits * i) for i, v in enumerate(masked.flat))
    expected = word.to_bytes((masked.size * bits + 7) // 8, "little")
    packed, scale_bytes = _native.prepared_quantize_pack(
        values.tobytes(), masks.tobytes(), *values.shape, precision, bits)
    assert packed == expected and scale_bytes == scales.tobytes()
    wire = next(w for w in (16, 24, 32) if w >= bits)
    ids = tuple(f"{i:032x}" for i in range(values.shape[0]))
    ordinary = PreparedStageBatchRequest("ab" * 16, ids, masked.astype(np.uint32), wire, bits).pack()
    assert PreparedStageBatchRequest.pack_encoded("ab" * 16, ids, packed, *values.shape, wire, bits) == ordinary


@pytest.mark.parametrize("with_bias", [False, True])
def test_fused_reply_preserves_all_float_bits_over_mixed_rings(with_bias):
    rng = np.random.default_rng(879)
    widths = bytes(range(1, 33))
    rows, cols = 17, len(widths)
    masks = rng.integers(0, 1 << 32, (rows, cols), dtype="<u4")
    signed = np.empty((rows, cols), np.int64)
    for i, width in enumerate(widths):
        signed[:, i] = rng.integers(-(1 << (width - 1)), 1 << (width - 1), rows)
        signed[:3, i] = [-(1 << (width - 1)), (1 << (width - 1)) - 1, 0]
    bits, word = 0, 0
    for value, mask, width in zip(signed.flat, masks.flat, list(widths) * rows, strict=True):
        word |= ((int(value) - int(mask)) & ((1 << width) - 1)) << bits
        bits += width
    packed = word.to_bytes((bits + 7) // 8, "little")
    a = np.exp(rng.uniform(-12, 4, rows)).astype("<f4")
    w = np.exp(rng.uniform(-12, 4, cols)).astype("<f4")
    bias = rng.normal(size=cols).astype("<f4") if with_bias else None
    expected = signed.astype(np.float32) * a[:, None] * w[None, :]
    if bias is not None:
        expected = expected + bias
    result = _native.prepared_dequantize_output(packed, masks.tobytes(), widths, rows,
        a.tobytes(), w.tobytes(), None if bias is None else bias.tobytes())
    assert result == expected.tobytes()


def test_fusion_rejects_noncanonical_and_unbounded_inputs():
    for rows, cols, precision, width in [(4097, 1, 8, 16), (1, 4_000_001, 8, 16),
            (0, 1, 8, 16), (1, 1, 1, 16), (1, 1, 8, 33)]:
        with pytest.raises(ValueError):
            _native.prepared_quantize_pack(b"", b"", rows, cols, precision, width)
    for x in [np.nan, np.inf, -np.inf, np.nextafter(np.float32(0), np.float32(1))]:
        with pytest.raises(ValueError):
            _native.prepared_quantize_pack(np.array([x], "<f4").tobytes(), b"\0" * 4, 1, 1, 8, 16)
    one = np.array([1], "<f4").tobytes()
    valid = [b"\0", b"\0" * 4, b"\x01", 1, one, one]
    for index, bad in [(0, b"\x80"), (0, b"\0\0"), (1, b""), (2, b"\x21"),
                       (3, 4097), (4, b""), (5, np.array([np.nan], "<f4").tobytes())]:
        fields = valid.copy()
        fields[index] = bad
        with pytest.raises(ValueError):
            _native.prepared_dequantize_output(*fields)
