"""Bounded, independently compressed public client-bundle frames."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable, Iterable, Iterator


ENCODING = "zlib-chunks-v1"
CHUNK_BYTES = 1024 * 1024
_MAX_FRAME_OVERHEAD = 64 * 1024
_HEADER = struct.Struct(">II")


class BundleFrameError(ValueError):
    pass


def encode_bundle_frames(payload: bytes) -> Iterator[bytes]:
    for offset in range(0, len(payload), CHUNK_BYTES):
        block = memoryview(payload)[offset : offset + CHUNK_BYTES]
        compressed = zlib.compress(block, level=1)
        yield _HEADER.pack(len(block), len(compressed)) + compressed


def decode_bundle_frames(
    chunks: Iterable[bytes], *, expected_size: int, on_wire_bytes: Callable[[int], None]
) -> bytearray:
    if not 0 < expected_size <= 8 * 1024 * 1024 * 1024:
        raise BundleFrameError("compressed bundle size is outside policy")
    output = bytearray()
    pending = bytearray()
    for chunk in chunks:
        on_wire_bytes(len(chunk))
        if not chunk or len(chunk) > CHUNK_BYTES + _MAX_FRAME_OVERHEAD + _HEADER.size:
            raise BundleFrameError("compressed bundle transport chunk is invalid")
        pending.extend(chunk)
        while len(pending) >= _HEADER.size:
            if len(output) == expected_size:
                raise BundleFrameError("compressed bundle has trailing data")
            raw_size, encoded_size = _HEADER.unpack_from(pending)
            if raw_size != min(CHUNK_BYTES, expected_size - len(output)) or not (
                0 < encoded_size <= raw_size + _MAX_FRAME_OVERHEAD
            ):
                raise BundleFrameError("compressed bundle frame exceeds declared bounds")
            frame_size = _HEADER.size + encoded_size
            if len(pending) < frame_size:
                break
            compressed = bytes(pending[_HEADER.size : frame_size])
            del pending[:frame_size]
            try:
                decoder = zlib.decompressobj()
                block = decoder.decompress(compressed, raw_size + 1)
            except zlib.error as exc:
                raise BundleFrameError("compressed bundle frame is invalid") from exc
            if (
                len(block) != raw_size
                or not decoder.eof
                or decoder.unused_data
                or decoder.unconsumed_tail
            ):
                raise BundleFrameError("compressed bundle frame has invalid size or trailing data")
            output.extend(block)
        if len(pending) > CHUNK_BYTES + _MAX_FRAME_OVERHEAD + _HEADER.size:
            raise BundleFrameError("compressed bundle frame buffer exceeds policy")
    if pending or len(output) != expected_size:
        raise BundleFrameError("compressed bundle is incomplete")
    return output
