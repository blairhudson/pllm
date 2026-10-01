from __future__ import annotations

import struct
import zlib

import pytest

from pllm.runtime.bundle_compression import (
    CHUNK_BYTES,
    BundleFrameError,
    decode_bundle_frames,
    encode_bundle_frames,
)


@pytest.mark.parametrize("size", [1, CHUNK_BYTES, CHUNK_BYTES + 7, 2 * CHUNK_BYTES + 3])
def test_bounded_frames_survive_arbitrary_transport_splits(size: int):
    payload = (bytes(range(251)) * ((size // 251) + 1))[:size]
    framed = b"".join(encode_bundle_frames(payload))
    sizes: list[int] = []
    restored = decode_bundle_frames(
        (framed[i : i + 65_537] for i in range(0, len(framed), 65_537)),
        expected_size=size,
        on_wire_bytes=sizes.append,
    )
    assert restored == payload
    assert sum(sizes) == len(framed)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda frame: frame[:-1],
        lambda frame: frame + b"x",
        lambda frame: struct.pack(">II", CHUNK_BYTES + 1, len(frame) - 8) + frame[8:],
        lambda frame: struct.pack(">II", 5, CHUNK_BYTES + 65_537) + frame[8:],
        lambda frame: frame[:8] + b"\x00" + frame[9:],
        lambda frame: struct.pack(">II", 5, len(frame) - 8 + len(zlib.compress(b"x")))
        + frame[8:]
        + zlib.compress(b"x"),
        lambda frame: struct.pack(">II", 1, len(frame) - 8) + frame[8:],
    ],
)
def test_frames_reject_truncation_trailing_bytes_bombs_and_bad_sizes(corrupt):
    good = b"".join(encode_bundle_frames(b"hello"))
    with pytest.raises(BundleFrameError):
        decode_bundle_frames([corrupt(good)], expected_size=5, on_wire_bytes=lambda _: None)


def test_declared_size_and_transport_chunk_are_bounded():
    with pytest.raises(BundleFrameError, match="size"):
        decode_bundle_frames([], expected_size=0, on_wire_bytes=lambda _: None)
    with pytest.raises(BundleFrameError, match="transport chunk"):
        decode_bundle_frames(
            [b"x" * (CHUNK_BYTES + 65_545)], expected_size=1, on_wire_bytes=lambda _: None
        )
