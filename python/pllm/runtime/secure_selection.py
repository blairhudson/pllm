from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .authenticated_mpc import (
    AuthenticatedMPC,
    AuthenticatedValue,
    AuthenticationError,
    BitDecompositionMask,
)


@dataclass(frozen=True, slots=True)
class SecureSelectionResult:
    indices: np.ndarray
    conversion_rounds: int
    comparison_rounds: int
    opened_values: int


def _triple(mpc: AuthenticatedMPC, shape: Sequence[int]):
    return mpc.preprocessor.multiplication_triple(shape)


def _xor(
    mpc: AuthenticatedMPC,
    left: AuthenticatedValue,
    right: AuthenticatedValue,
) -> AuthenticatedValue:
    product = mpc.multiply(left, right, _triple(mpc, left.shape))
    return mpc.sub(mpc.add(left, right), mpc.mul_public(product, 2))


def _prefix_scan(
    mpc: AuthenticatedMPC,
    generate: AuthenticatedValue,
    propagate: AuthenticatedValue,
) -> tuple[AuthenticatedValue, AuthenticatedValue]:
    """Compute all carry prefixes with a logarithmic number of rounds."""
    width = generate.shape[-1]
    distance = 1
    g = generate
    p = propagate
    while distance < width:
        g_high = mpc.take(g, (..., slice(distance, None)))
        p_high = mpc.take(p, (..., slice(distance, None)))
        g_low = mpc.take(g, (..., slice(None, -distance)))
        p_low = mpc.take(p, (..., slice(None, -distance)))
        products = mpc.multiply_many(
            [p_high, p_high],
            [g_low, p_low],
            [_triple(mpc, p_high.shape), _triple(mpc, p_high.shape)],
        )
        g_tail = mpc.add(g_high, products[0])
        p_tail = products[1]
        g = mpc.concatenate([mpc.take(g, (..., slice(None, distance))), g_tail], axis=-1)
        p = mpc.concatenate([mpc.take(p, (..., slice(None, distance))), p_tail], axis=-1)
        distance *= 2
    return g, p


def to_bits(
    mpc: AuthenticatedMPC,
    value: AuthenticatedValue,
    mask: BitDecompositionMask,
) -> AuthenticatedValue:
    """Convert a field value to authenticated little endian bits.

    The runtime opens ``value + random_value``. The opened value is uniform in
    the field. A parallel prefix subtractor then removes the secret random
    value, followed by a conditional addition of the field modulus.
    """
    if value.shape != mask.random_value.shape:
        raise ValueError("bit mask shape mismatch")
    if mask.random_bits.shape != value.shape + (mask.bit_width,):
        raise ValueError("bit mask bit shape mismatch")
    width = mask.bit_width
    p = mpc.modulus
    if 1 << width <= p:
        raise ValueError("bit width is too small for the field")

    opened = mpc.open(mpc.add(value, mask.random_value), to_client=True)
    positions = np.arange(width, dtype=np.int64)
    public_bits = ((opened[..., None] >> positions) & 1).astype(np.int64)
    random_bits = mask.random_bits

    # Borrow generation and propagation for public_bits - random_bits.
    generate = mpc.mul_public(random_bits, 1 - public_bits)
    propagate = mpc.add_public(
        mpc.mul_public(random_bits, 2 * public_bits - 1),
        1 - public_bits,
    )
    prefix_generate, _ = _prefix_scan(mpc, generate, propagate)
    zero = mpc.public_value(0, value.shape + (1,))
    borrow_in = mpc.concatenate(
        [zero, mpc.take(prefix_generate, (..., slice(None, -1)))],
        axis=-1,
    )

    random_times_borrow = mpc.multiply(
        random_bits,
        borrow_in,
        _triple(mpc, random_bits.shape),
    )
    random_xor_borrow = mpc.sub(
        mpc.add(random_bits, borrow_in),
        mpc.mul_public(random_times_borrow, 2),
    )
    difference_bits = mpc.add_public(
        mpc.mul_public(random_xor_borrow, 1 - 2 * public_bits),
        public_bits,
    )

    # When the public value is smaller than the random value, add the prime.
    final_borrow = mpc.take(prefix_generate, (..., -1))
    # Add an explicit bit axis without exposing the share.
    final_borrow_bits = mpc.reshape(final_borrow, value.shape + (1,))
    final_borrow_bits = mpc.broadcast_to(final_borrow_bits, value.shape + (width,))
    modulus_bits = ((p >> positions) & 1).astype(np.int64)
    addend_bits = mpc.mul_public(final_borrow_bits, modulus_bits)

    product = mpc.multiply(
        difference_bits,
        addend_bits,
        _triple(mpc, difference_bits.shape),
    )
    add_propagate = mpc.sub(
        mpc.add(difference_bits, addend_bits),
        mpc.mul_public(product, 2),
    )
    add_generate = product
    add_prefix_generate, _ = _prefix_scan(mpc, add_generate, add_propagate)
    carry_in = mpc.concatenate(
        [zero, mpc.take(add_prefix_generate, (..., slice(None, -1)))],
        axis=-1,
    )
    carry_product = mpc.multiply(
        add_propagate,
        carry_in,
        _triple(mpc, add_propagate.shape),
    )
    return mpc.sub(
        mpc.add(add_propagate, carry_in),
        mpc.mul_public(carry_product, 2),
    )


def greater_and_equal(
    mpc: AuthenticatedMPC,
    left_bits: AuthenticatedValue,
    right_bits: AuthenticatedValue,
) -> tuple[AuthenticatedValue, AuthenticatedValue]:
    """Return secret greater and equal bits using a tree comparison."""
    if left_bits.shape != right_bits.shape:
        raise ValueError("comparison bit shapes must match")
    product = mpc.multiply(
        left_bits,
        right_bits,
        _triple(mpc, left_bits.shape),
    )
    greater = mpc.sub(left_bits, product)
    equal = mpc.add_public(
        mpc.add(mpc.mul_public(product, 2), mpc.mul_public(mpc.add(left_bits, right_bits), -1)),
        1,
    )

    # Work from the most significant bit to the least significant bit.
    g = mpc.take(greater, (..., slice(None, None, -1)))
    e = mpc.take(equal, (..., slice(None, None, -1)))
    while g.shape[-1] > 1:
        count = g.shape[-1]
        pair_count = count // 2
        even_width = pair_count * 2
        g_even = mpc.take(g, (..., slice(None, even_width)))
        e_even = mpc.take(e, (..., slice(None, even_width)))
        g_high = mpc.take(g_even, (..., slice(0, None, 2)))
        g_low = mpc.take(g_even, (..., slice(1, None, 2)))
        e_high = mpc.take(e_even, (..., slice(0, None, 2)))
        e_low = mpc.take(e_even, (..., slice(1, None, 2)))
        products = mpc.multiply_many(
            [e_high, e_high],
            [g_low, e_low],
            [_triple(mpc, e_high.shape), _triple(mpc, e_high.shape)],
        )
        combined_g = mpc.add(g_high, products[0])
        combined_e = products[1]
        if count % 2:
            combined_g = mpc.concatenate([combined_g, mpc.take(g, (..., slice(-1, None)))], axis=-1)
            combined_e = mpc.concatenate([combined_e, mpc.take(e, (..., slice(-1, None)))], axis=-1)
        g, e = combined_g, combined_e
    return mpc.take(g, (..., 0)), mpc.take(e, (..., 0))


def select(
    mpc: AuthenticatedMPC,
    condition: AuthenticatedValue,
    when_true: AuthenticatedValue,
    when_false: AuthenticatedValue,
) -> AuthenticatedValue:
    if when_true.shape != when_false.shape:
        raise ValueError("selection shapes must match")
    cond = condition
    while len(cond.shape) < len(when_true.shape):
        cond = mpc.reshape(cond, cond.shape + (1,))
    cond = mpc.broadcast_to(cond, when_true.shape)
    difference = mpc.sub(when_true, when_false)
    product = mpc.multiply(cond, difference, _triple(mpc, difference.shape))
    return mpc.add(when_false, product)


def secure_argmax(
    mpc: AuthenticatedMPC,
    values: AuthenticatedValue,
    *,
    bit_mask: BitDecompositionMask,
) -> SecureSelectionResult:
    """Reveal only the index of the largest value in each row."""
    if len(values.shape) != 2:
        raise ValueError("argmax expects shape (batch, choices)")
    batch, choices = values.shape
    if choices < 1:
        raise ValueError("argmax needs at least one choice")
    start_rounds = mpc.stats.multiplication_rounds
    bits = to_bits(mpc, values, bit_mask)
    conversion_rounds = mpc.stats.multiplication_rounds - start_rounds

    index_width = max(1, (choices - 1).bit_length())
    indices = np.arange(choices, dtype=np.int64)
    index_bits_clear = ((indices[:, None] >> np.arange(index_width)) & 1).astype(np.int64)
    index_bits_clear = np.broadcast_to(index_bits_clear[None, :, :], (batch, choices, index_width))
    index_bits = mpc.public_value(index_bits_clear)

    comparison_start = mpc.stats.multiplication_rounds
    candidate_values = bits
    candidate_indices = index_bits
    count = choices
    while count > 1:
        pair_count = count // 2
        even_width = pair_count * 2
        left_values = mpc.take(candidate_values, (..., slice(0, even_width, 2), slice(None)))
        right_values = mpc.take(candidate_values, (..., slice(1, even_width, 2), slice(None)))
        left_indices = mpc.take(candidate_indices, (..., slice(0, even_width, 2), slice(None)))
        right_indices = mpc.take(candidate_indices, (..., slice(1, even_width, 2), slice(None)))
        greater, equal = greater_and_equal(mpc, left_values, right_values)
        choose_left = mpc.add(greater, equal)

        # Select value bits and index bits in one communication round.
        value_condition = mpc.broadcast_to(
            mpc.reshape(choose_left, choose_left.shape + (1,)),
            left_values.shape,
        )
        index_condition = mpc.broadcast_to(
            mpc.reshape(choose_left, choose_left.shape + (1,)),
            left_indices.shape,
        )
        value_difference = mpc.sub(left_values, right_values)
        index_difference = mpc.sub(left_indices, right_indices)
        selected_products = mpc.multiply_many(
            [value_condition, index_condition],
            [value_difference, index_difference],
            [_triple(mpc, value_difference.shape), _triple(mpc, index_difference.shape)],
        )
        selected_values = mpc.add(right_values, selected_products[0])
        selected_indices = mpc.add(right_indices, selected_products[1])
        if count % 2:
            selected_values = mpc.concatenate(
                [selected_values, mpc.take(candidate_values, (..., slice(-1, None), slice(None)))],
                axis=-2,
            )
            selected_indices = mpc.concatenate(
                [selected_indices, mpc.take(candidate_indices, (..., slice(-1, None), slice(None)))],
                axis=-2,
            )
        candidate_values = selected_values
        candidate_indices = selected_indices
        count = candidate_values.shape[-2]

    final_index_bits = mpc.take(candidate_indices, (..., 0, slice(None)))
    opened_bits = mpc.open(final_index_bits, to_client=True)
    powers = 1 << np.arange(index_width, dtype=np.int64)
    opened_indices = np.sum(opened_bits * powers, axis=-1, dtype=np.int64)
    comparison_rounds = mpc.stats.multiplication_rounds - comparison_start
    return SecureSelectionResult(
        indices=opened_indices,
        conversion_rounds=conversion_rounds,
        comparison_rounds=comparison_rounds,
        opened_values=mpc.stats.opened_values,
    )


def verify_one_hot(mpc: AuthenticatedMPC, value: AuthenticatedValue) -> None:
    """Check a secret one hot vector without revealing its selected position."""
    if len(value.shape) != 2:
        raise ValueError("one hot input must have shape (batch, vocabulary)")
    value_minus_one = mpc.add_public(value, -1)
    bit_constraints = mpc.multiply(
        value,
        value_minus_one,
        _triple(mpc, value.shape),
    )
    sum_constraint = mpc.add_public(mpc.sum(value, axis=-1), -1)
    opened_bits, opened_sum = mpc.open_many([bit_constraints, sum_constraint], to_client=False)
    if np.any(opened_bits) or np.any(opened_sum):
        raise AuthenticationError("private token input is not one hot")
