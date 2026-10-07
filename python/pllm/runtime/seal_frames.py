"""Bounded SEAL file serialization shared by local encrypted references."""
from __future__ import annotations

from pathlib import Path
import struct


def save_frame(obj, path: Path, capacity: int) -> bytes:
    """Serialize with public padding; the binding exposes file APIs only."""
    try:
        obj.save(str(path))
        length = path.stat().st_size
        if not 0 < length <= capacity - 8:
            raise ValueError("SEAL serialization exceeds its public frame bound")
        raw = path.read_bytes()
        if len(raw) != length:
            raise ValueError("SEAL serialization changed during receipt")
        return struct.pack("<Q", length) + raw + bytes(capacity - 8 - length)
    finally:
        path.unlink(missing_ok=True)


def load_frame(obj, context, frame: bytes, path: Path, capacity: int):
    if len(frame) != capacity:
        raise ValueError("SEAL frame length differs from its public geometry")
    length = struct.unpack_from("<Q", frame)[0]
    if not 0 < length <= capacity - 8 or any(frame[8 + length:]):
        raise ValueError("invalid bounded SEAL frame")
    try:
        path.write_bytes(frame[8:8 + length])
        if context is None:
            obj.load(str(path))
        else:
            obj.load(context, str(path))
    finally:
        path.unlink(missing_ok=True)
    return obj
