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
