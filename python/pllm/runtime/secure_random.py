"""Unbiased field sampling from the operating system, never NumPy's PRNG."""
from __future__ import annotations
import math
import secrets
import numpy as np


def uniform_residues(modulus: int, shape: tuple[int, ...]) -> np.ndarray:
    if not 2 <= modulus <= 2**32:
        raise ValueError('modulus must be in [2, 2**32]')
    if any(not isinstance(x, int) or x < 0 for x in shape):
        raise ValueError('dimensions must be nonnegative integers')
    count = math.prod(shape)
    from ._native_support import extension
    native = extension()
    if native is not None:
        return np.frombuffer(native.uniform_residues(modulus, count), dtype="<u4").reshape(shape)
    out = np.empty(count, dtype=np.uint32)
    limit = (2**32 // modulus) * modulus
    offset = 0
    while offset < count:
        # Rejection, not biased reduction. Only the OS supplies entropy.
        raw = np.frombuffer(secrets.token_bytes(4 * (count - offset)), dtype='<u4')
        good = raw if limit == 2**32 else raw[raw < limit]
        size = min(good.size, count - offset)
        out[offset:offset+size] = good[:size].astype(np.uint64) % modulus
        offset += size
    return out.reshape(shape)


class FieldRandom:
    """Small integers-only compatibility surface for cryptographic callers."""
    def integers(self, low, high=None, size=None, dtype=np.int64):
        if high is None:
            low, high = 0, low
        if low != 0:
            raise ValueError("cryptographic field masks must start at zero")
        shape = () if size is None else ((size,) if isinstance(size,int) else tuple(size))
        return uniform_residues(int(high),shape).astype(dtype)
