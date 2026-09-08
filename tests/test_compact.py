from __future__ import annotations

import numpy as np
import pytest

from pllm.runtime.compact import (
    CompactCodecError,
    PackedTensor,
    U24Codec,
    pack_unsigned,
    required_modulus_bits,
    storage_bytes,
    unpack_unsigned,
)


def test_required_w4a4_storage_classes():
    assert required_modulus_bits(128) <= 16
    assert 16 < required_modulus_bits(1536) <= 24
    assert storage_bytes(required_modulus_bits(128)) == 2
    assert storage_bytes(required_modulus_bits(1536)) == 3


@pytest.mark.parametrize("width,maximum", [(2, 0xFFFF), (3, 0xFFFFFF), (4, 0xFFFFFFFF)])
def test_unsigned_codec_round_trip(width: int, maximum: int):
    rng = np.random.default_rng(17)
    values = rng.integers(0, maximum + 1, size=(17, 29), dtype=np.uint32)
    payload = pack_unsigned(values, width=width)
    decoded = unpack_unsigned(payload, width=width, count=values.size).reshape(values.shape)
    assert np.array_equal(values, decoded)
    assert len(payload) == values.size * width


def test_centered_tensor_round_trip():
    values = np.array([[-32768, -1, 0, 1, 32767]], dtype=np.int64)
    packed = PackedTensor.from_centered(values, modulus=65537, width=3)
    assert np.array_equal(values, packed.to_centered(modulus=65537))


def test_u24_rejects_out_of_range_and_truncation():
    codec = U24Codec()
    with pytest.raises(CompactCodecError):
        codec.pack(np.array([0x1000000], dtype=np.uint32))
    with pytest.raises(CompactCodecError):
        codec.unpack(b"\x00\x01")
