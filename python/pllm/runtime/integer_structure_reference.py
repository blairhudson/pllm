"""Exact algebra diagnostics; not a selectable inference protocol.

Matrices use output-by-input orientation. GF(2) witnesses certify only a lower
bound on factor width over Z/(2**w); a deficient GF(2) rank is not a factorization.
Small factor/reference products use Python integers to avoid machine overflow.
"""

from __future__ import annotations

import operator

import numpy as np


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or 0 in value.shape or value.dtype.kind not in "iuO":
        raise ValueError(f"{name} must be a nonempty integer matrix")
    if value.dtype.kind == "O":
        try:
            for item in value.flat:
                operator.index(item)
        except TypeError as exc:
            raise ValueError(f"{name} must contain integers") from exc
    return value


def _modulus(bits: int) -> int:
    if type(bits) is not int or not 1 <= bits <= 64:
        raise ValueError("ring bits must be an integer in [1, 64]")
    return 1 << bits


def odd_minor_witness(weight: np.ndarray) -> dict[str, list[int] | int]:
    """Find independent original rows and pivot columns with an odd minor.

    Python bitset row elimination. Stops at min(m,n), never claims full column
    rank for wide matrices. Selected original rows, not transformed row indices,
    make the witness independently checkable against the checkpoint.
    """
    weight = _matrix(weight, "weight")
    parity = np.asarray(weight % 2, dtype=np.uint8)
    packed = np.packbits(parity, axis=1, bitorder="little")
    basis: dict[int, int] = {}
    rows, columns = [], []
    limit = min(weight.shape)
    for index, row in enumerate(packed):
        vector = int.from_bytes(row.tobytes(), "little")
        while vector:
            pivot = (vector & -vector).bit_length() - 1
            if pivot not in basis:
                basis[pivot] = vector
                rows.append(index)
                columns.append(pivot)
                break
            vector ^= basis[pivot]
        if len(rows) == limit:
            break
    return {"size": len(rows), "rows": rows, "columns": columns}


def verify_odd_minor(weight: np.ndarray, witness: dict) -> bool:
    """Independent Boolean column elimination verifies determinant parity.

    No determinant in float, no reliance on the witness generator's bitsets.
    Invalid or empty witnesses cannot certify rank.
    """
    weight = _matrix(weight, "weight")
    try:
        rows, columns, size = witness["rows"], witness["columns"], witness["size"]
        if (
            type(size) is not int
            or not 0 < size <= min(weight.shape)
            or len(rows) != size
            or len(columns) != size
            or any(type(i) is not int for i in (*rows, *columns))
            or len(set(rows)) != size
            or len(set(columns)) != size
            or any(not 0 <= i < weight.shape[0] for i in rows)
            or any(not 0 <= j < weight.shape[1] for j in columns)
        ):
            return False
    except (KeyError, TypeError):
        return False
    minor = np.asarray(weight[np.ix_(rows, columns)] % 2, dtype=np.bool_)
    # Transpose: eliminate columns of the original selected minor.
    reduced = minor.T.copy()
    for column in range(size):
        candidates = np.flatnonzero(reduced[column:, column])
        if candidates.size == 0:
            return False
        pivot = column + int(candidates[0])
        reduced[[column, pivot]] = reduced[[pivot, column]]
        targets = column + 1 + np.flatnonzero(reduced[column + 1 :, column])
        reduced[targets] ^= reduced[column]
    return True


def verify_factorization(
    weight: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    bits: int,
    residual: np.ndarray | None = None,
) -> bool:
    """Check every coefficient of W = A B + R modulo 2**bits, overflow-free."""
    modulus = _modulus(bits)
    weight, a, b = (_matrix(v, n) for v, n in ((weight, "weight"), (a, "A"), (b, "B")))
    if a.shape[1] != b.shape[0] or (a.shape[0], b.shape[1]) != weight.shape:
        raise ValueError("factor shapes do not match weight")
    product = a.astype(object) @ b.astype(object)
    if residual is not None:
        residual = _matrix(residual, "residual")
        if residual.shape != weight.shape:
            raise ValueError("residual shape does not match weight")
        product += residual.astype(object)
    return bool(np.all((product - weight.astype(object)) % modulus == 0))


def factor_apply(
    a: np.ndarray,
    b: np.ndarray,
    inputs: np.ndarray,
    *,
    bits: int,
    residual: np.ndarray | None = None,
) -> np.ndarray:
    """Client Bx then remote Az + Rx in the same ring, without requantization."""
    modulus = _modulus(bits)
    a, b, inputs = (_matrix(v, n) for v, n in ((a, "A"), (b, "B"), (inputs, "inputs")))
    if a.shape[1] != b.shape[0] or inputs.shape[1] != b.shape[1]:
        raise ValueError("factor/input shapes differ")
    projected = (inputs.astype(object) @ b.astype(object).T) % modulus
    outputs = projected @ a.astype(object).T
    if residual is not None:
        residual = _matrix(residual, "residual")
        if residual.shape != (a.shape[0], b.shape[1]):
            raise ValueError("residual shape does not match factors")
        outputs += inputs.astype(object) @ residual.astype(object).T
    return np.asarray(outputs % modulus, dtype=np.uint64)


def centered_lift(residues: np.ndarray, *, bits: int, output_bound: int) -> np.ndarray:
    """Lift only with a public strict bound; small ring equality is insufficient."""
    modulus = _modulus(bits)
    if type(output_bound) is not int or not 0 <= output_bound < modulus // 2:
        raise ValueError("output bound must be strictly below half the modulus")
    values = _matrix(residues, "residues").astype(object) % modulus
    values = np.where(values >= modulus // 2, values - modulus, values)
    if np.any(np.abs(values) > output_bound):
        raise ValueError("output violates certified bound")
    return np.asarray(values, dtype=np.int64)


def minimum_signed_bits(bound: int) -> int:
    if type(bound) is not int or bound < 0:
        raise ValueError("bound must be a nonnegative integer")
    return max(2, bound.bit_length() + 1)


def factor_cost_gate(
    *,
    inputs: int,
    outputs: int,
    rank: int,
    ring_bits: int,
    rows: int = 70,
    coefficient_bits: int = 8,
    residual_nnz: int = 0,
    residual_input_columns: int = 0,
) -> dict[str, int | float]:
    """Dense-factor prices, optimistic one-ingress/one-egress/fresh-correction bodies.

    R uses a separate full-ring input path. This is an explicit route estimate,
    not a lower bound on all protocols. Residual coefficients, framing, scale,
    bias, entropy refresh, and cryptographic setup are additional costs.
    """
    for name, value in (
        ("inputs", inputs),
        ("outputs", outputs),
        ("rank", rank),
        ("rows", rows),
        ("coefficient_bits", coefficient_bits),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _modulus(ring_bits)
    if coefficient_bits > 64:
        raise ValueError("coefficient bits must not exceed 64")
    if (
        type(residual_nnz) is not int
        or not 0 <= residual_nnz <= inputs * outputs
        or type(residual_input_columns) is not int
        or not 0 <= residual_input_columns <= inputs
        or bool(residual_nnz) != bool(residual_input_columns)
        or residual_nnz < residual_input_columns
        or residual_nnz > outputs * residual_input_columns
    ):
        raise ValueError("residual support is inconsistent")
    packed = lambda count: (rows * count * ring_bits + 7) // 8
    ingress = packed(rank + residual_input_columns)
    egress = packed(outputs)
    projected_bound = inputs * 127 * (1 << (coefficient_bits - 1))
    expanded_bound = rank * (1 << (coefficient_bits - 1)) * (1 << (ring_bits - 1))
    return {
        "client_dense_projection_macs": rows * inputs * rank,
        "client_dense_projection_mac_fraction": rank / outputs,
        "client_dense_factor_weight_bytes": (inputs * rank * coefficient_bits + 7) // 8,
        "unreduced_projection_worst_abs_w8_input_centered_coefficients": projected_bound,
        "unreduced_projection_minimum_signed_bits": minimum_signed_bits(projected_bound),
        "masked_projection_ring_bits_no_requantization": ring_bits,
        "unreduced_expansion_minimum_signed_bits_after_centered_ring_projection": minimum_signed_bits(
            expanded_bound
        ),
        "remote_dense_expansion_macs": rows * outputs * rank,
        "remote_sparse_residual_macs": rows * residual_nnz,
        "fresh_mask_input_elements_per_row": rank + residual_input_columns,
        "fresh_mask_output_elements_per_row": outputs,
        "optimistic_online_bytes": ingress + egress,
        "optimistic_all_link_bytes": ingress + 2 * egress,
    }
