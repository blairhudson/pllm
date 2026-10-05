"""Bounded public-weight screens; none issues live protocol material.

The integer oracles deliberately use NumPy/Python independently of the native
transport. Synthetic masks here are deterministic test values, never secrets.
"""
from __future__ import annotations

import hashlib

import numpy as np


def row_widths(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights)
    if w.dtype != np.int8 or w.ndim != 2 or not w.size:
        raise ValueError("expected a nonempty signed-i8 matrix")
    bounds = np.abs(w.astype(np.int64)).sum(axis=1) * 127
    widths = np.array([max(1, int(x).bit_length() + 1) for x in bounds], np.uint8)
    if int(widths.max()) > 32:
        raise ValueError("output bound exceeds wrap32")
    return widths


def annihilator_widths(weights: np.ndarray) -> np.ndarray:
    """Minimal coordinate rings killing every omitted high-bit contribution.

    k_i = max_j(max(0, b_j - v2(W_ji))).  The codec retains one bit even for an
    all-zero column; this avoids introducing a zero-width transport contract.
    """
    widths = row_widths(weights)
    valuations = np.array([32 if x == 0 else (x & -x).bit_length() - 1
                           for x in range(256)], np.int16)
    result = np.ones(weights.shape[1], np.uint8)
    for start in range(0, len(weights), 128):
        block = weights[start:start + 128]
        need = widths[start:start + len(block)].astype(np.int16)[:, None] - valuations[block.view(np.uint8)]
        result = np.maximum(result, np.maximum(need.max(axis=0), 1).astype(np.uint8))
    return result


def packed_size(widths: np.ndarray, rows: int) -> int:
    return (int(np.asarray(widths, dtype=np.int64).sum()) * rows + 7) // 8


def radix_size(count: int, modulus: int) -> int:
    if type(count) is not int or not 1 <= count <= 4_000_000:
        raise ValueError("invalid radix count")
    if type(modulus) is not int or not 2 <= modulus <= 2**32:
        raise ValueError("invalid radix modulus")
    complete, tail = divmod(count, 16)
    return complete * (((modulus**16 - 1).bit_length() + 7) // 8) + (
        ((modulus**tail - 1).bit_length() + 7) // 8 if tail else 0)


def radix_pack(values: np.ndarray, modulus: int) -> bytes:
    values = np.asarray(values)
    radix_size(values.size, modulus)
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= modulus):
        raise ValueError("noncanonical radix residue")
    output = bytearray()
    flat = values.reshape(-1)
    for start in range(0, len(flat), 16):
        block = flat[start:start + 16]
        word = 0
        for value in reversed(block):
            word = word * modulus + int(value)
        output.extend(word.to_bytes(((modulus**len(block) - 1).bit_length() + 7) // 8, "little"))
    return bytes(output)


def radix_unpack(payload: bytes, count: int, modulus: int) -> np.ndarray:
    if len(payload) != radix_size(count, modulus):
        raise ValueError("radix length differs")
    output, offset = np.empty(count, np.uint32), 0
    for start in range(0, count, 16):
        length = min(16, count - start)
        size = ((modulus**length - 1).bit_length() + 7) // 8
        word = int.from_bytes(payload[offset:offset + size], "little")
        if word >= modulus**length:
            raise ValueError("noncanonical radix padding")
        for index in range(start, start + length):
            word, residue = divmod(word, modulus)
            output[index] = residue
        offset += size
    return output


def moment_bypass(weights: np.ndarray, groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Cheap block sums, rather than a dense client-side low-rank anchor.

    Each group's public integer median minimizes its row L1 residual; clipping
    preserves signed-i8 weights in BOTH retained matrices. No floats enter the
    bypass addition. A new stage/bundle/verification contract would be needed.
    """
    row_widths(weights)
    if type(groups) is not int or not 1 <= groups <= 64 or weights.shape[1] % groups:
        raise ValueError("group count must divide the input width")
    block = weights.astype(np.int16).reshape(len(weights), groups, -1)
    median = np.floor(np.median(block, axis=2)).astype(np.int16)
    low, high = block.max(axis=2) - 127, block.min(axis=2) + 128
    center = np.clip(median, np.maximum(low, -128), np.minimum(high, 127)).astype(np.int8)
    residual = block - center.astype(np.int16)[:, :, None]
    assert residual.min() >= -128 and residual.max() <= 127
    return residual.reshape(weights.shape).astype(np.int8), center


def signed_row_dictionary(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact public equality only; no approximate clustering or private indices."""
    row_widths(weights)
    seen: dict[bytes, int] = {}
    selected, mapping, signs = [], [], []
    for index, row in enumerate(weights):
        nz = np.flatnonzero(row)
        if not len(nz):
            mapping.append(-1)
            signs.append(0)
            continue
        sign = 1 if int(row[nz[0]]) > 0 else -1
        canonical = (row.astype(np.int16) * sign).astype("<i2").tobytes()
        key = hashlib.sha256(canonical).digest()
        if key not in seen:
            seen[key] = len(selected)
            selected.append(index)
        else:
            # A hash hit is never authority for algebraic equivalence.
            previous = weights[selected[seen[key]]].astype(np.int16)
            previous *= 1 if int(previous[np.flatnonzero(previous)[0]]) > 0 else -1
            if previous.astype("<i2").tobytes() != canonical:
                raise ValueError("row dictionary hash collision")
        representative = weights[selected[seen[key]]]
        relative = sign * (1 if int(representative[np.flatnonzero(representative)[0]]) > 0 else -1)
        mapping.append(seen[key])
        signs.append(relative)
    return np.array(selected, np.int32), np.array(mapping, np.int32), np.array(signs, np.int8)


def gf2_rank(matrix: np.ndarray) -> int:
    """Exact mod-2 minor rank is a lower bound on integer displacement rank."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or max(matrix.shape) > 1024:
        raise ValueError("bounded rank matrix required")
    basis: dict[int, int] = {}
    for row in np.packbits((matrix.astype(np.int64) & 1).astype(np.uint8), axis=1, bitorder="little"):
        word = int.from_bytes(row.tobytes(), "little")
        while word:
            pivot = word.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = word
                break
            word ^= basis[pivot]
    return len(basis)


def displacement_rank_bound(weights: np.ndarray, limit: int = 256) -> tuple[int, int]:
    # Interior minor of Z_m W - W Z_n for the public unit subdiagonal shifts.
    size = min(limit, weights.shape[0] - 1, weights.shape[1] - 1)
    if size < 1:
        raise ValueError("matrix too small for displacement minor")
    delta = (weights[:size, :size].astype(np.int16)
             - weights[1:size + 1, 1:size + 1].astype(np.int16))
    return gf2_rank(delta), size


def recurrence_leakage_witness() -> dict:
    """One fresh coordinate does not refresh a cyclically shifted full mask."""
    rng, modulus, width = np.random.default_rng(714), 65536, 32
    x, y = rng.integers(-127, 128, size=(2, width), dtype=np.int64)
    recovered = []
    for _ in range(4):
        r = rng.integers(0, modulus, width, dtype=np.int64)
        next_mask = np.roll(r, 1)
        next_mask[0] = rng.integers(0, modulus)
        a, b = (x - r) % modulus, (y - next_mask) % modulus
        witness = (b - np.roll(a, 1)) % modulus
        assert np.array_equal(witness[1:], (y - np.roll(x, 1))[1:] % modulus)
        recovered.append(witness[1:])
    assert all(np.array_equal(recovered[0], item) for item in recovered)
    return {"width": width, "fresh_coordinates": 1, "exposed_private_relations": width - 1,
            "independent_mask_trials": len(recovered), "privacy_gate": "rejected"}
