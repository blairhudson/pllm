from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


class CompactCodecError(ValueError):
    pass


def required_modulus_bits(in_features: int, *, qmax: int = 7) -> int:
    """Bits required for an exact signed W4A4 dot-product residue.

    A length-``in_features`` dot product is bounded by
    ``[-in_features*qmax**2, +in_features*qmax**2]``.  A modulus larger than
    twice the bound preserves the signed result without wrap ambiguity.
    """

    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if qmax <= 0:
        raise ValueError("qmax must be positive")
    return math.ceil(math.log2(2 * in_features * qmax * qmax + 1))


def storage_bytes(required_bits: int) -> int:
    if required_bits <= 0:
        raise ValueError("required_bits must be positive")
    if required_bits <= 16:
        return 2
    if required_bits <= 24:
        return 3
    if required_bits <= 32:
        return 4
    raise CompactCodecError("values wider than 32 bits are unsupported")


def _as_u32(values: np.ndarray | Iterable[int]) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in {"i", "u"}:
        raise CompactCodecError("compact tensors require integer values")
    if array.size and (int(array.min()) < 0 or int(array.max()) > 0xFFFFFFFF):
        raise CompactCodecError("value outside uint32 range")
    return np.ascontiguousarray(array, dtype=np.uint32).reshape(-1)


class U24Codec:
    """Exact codec using the installed Rust extension or the explicit reference."""
    def __init__(self, library=None):
        if library is not None:
            raise CompactCodecError("External codec libraries are retired; use the Maturin extension")

    @classmethod
    def from_environment(cls):
        return cls()

    @property
    def native(self):
        from ._native_support import extension
        return extension() is not None

    def pack(self, values):
        return pack_unsigned(values, width=3)

    def unpack(self, payload, *, count=None):
        return unpack_unsigned(payload, width=3, count=count)


def pack_unsigned(values, *, width: int) -> bytes:
    from ._native_support import extension
    array = _as_u32(values)
    if width not in {2,3,4}:
        raise CompactCodecError("width must be 2, 3, or 4 bytes")
    if array.size and int(array.max()) >= (1 << (8*width)):
        raise CompactCodecError("value exceeds the wire width")
    native = extension()
    if native is not None:
        return native.pack_unsigned(array.astype("<u4", copy=False).tobytes(), width)
    if width == 2:
        return array.astype("<u2").tobytes()
    if width == 4:
        return array.astype("<u4", copy=False).tobytes()
    out = np.empty(array.size*3, dtype=np.uint8)
    out[0::3] = array & 255
    out[1::3] = (array >> 8) & 255
    out[2::3] = (array >> 16) & 255
    return out.tobytes()


def unpack_unsigned(payload: bytes, *, width: int, count: int | None = None) -> np.ndarray:
    from ._native_support import extension
    if width not in {2,3,4} or len(payload) % width:
        raise CompactCodecError("invalid width or truncated payload")
    if count is not None and len(payload)//width != count:
        raise CompactCodecError("element count mismatch")
    native = extension()
    if native is not None:
        return np.frombuffer(native.unpack_unsigned(payload, width), dtype="<u4")
    if width in {2,4}:
        return np.frombuffer(payload,dtype=f"<u{width}").astype(np.uint32, copy=True)
    s = np.frombuffer(payload, dtype=np.uint8)
    return s[0::3].astype(np.uint32) | (s[1::3].astype(np.uint32)<<8) | (s[2::3].astype(np.uint32)<<16)


@dataclass(frozen=True, slots=True)
class PackedTensor:
    shape: tuple[int, ...]
    width: int
    payload: bytes

    @classmethod
    def from_unsigned(cls, values: np.ndarray, *, width: int) -> "PackedTensor":
        array = np.asarray(values)
        return cls(tuple(int(v) for v in array.shape), width, pack_unsigned(array, width=width))

    def to_unsigned(self) -> np.ndarray:
        count = math.prod(self.shape)
        return unpack_unsigned(self.payload, width=self.width, count=count).reshape(self.shape)

    @classmethod
    def from_centered(cls, values: np.ndarray, *, modulus: int, width: int | None = None) -> "PackedTensor":
        if modulus <= 1 or modulus > 0xFFFFFFFF:
            raise CompactCodecError("modulus must fit in uint32")
        residues = np.asarray(values, dtype=np.int64) % modulus
        selected = width or storage_bytes(math.ceil(math.log2(modulus)))
        return cls.from_unsigned(residues.astype(np.uint32), width=selected)

    def to_centered(self, *, modulus: int) -> np.ndarray:
        raw = self.to_unsigned().astype(np.int64)
        if raw.size and int(raw.max()) >= modulus:
            raise CompactCodecError("residue is outside the declared modulus")
        return np.where(raw > modulus // 2, raw - modulus, raw)
