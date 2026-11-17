"""Explicit weight-two RAA witness, not a protocol security certificate.

The calculation concerns independent uniform fixed-weight supports avoiding a
fixed witness support. It does not establish generator provenance, production
admission weakness, or an exact minimum distance. Import has no side effects.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
import json
from math import comb


FIELD = 2_013_265_921
DIMENSION = 64
REPEAT = 8
LENGTH = 512


def _integer(name: str, value: int, minimum: int, maximum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (not bool)")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")


def primary_witness() -> tuple[
    list[int], list[int], list[int], list[int], list[int]
]:
    """Return message, input-gather permutations, and destination scales."""
    message = [0] * DIMENSION
    message[-1] = 1
    pi1 = list(range(LENGTH))
    first_support = [504, 506, 508, 510]
    pi2 = [index for index in range(LENGTH) if index not in first_support]
    pi2.extend(first_support)
    scales1 = [1] * LENGTH
    for index in (505, 507, 509, 511):
        scales1[index] = FIELD - 1
    scales2 = [1] * LENGTH
    for index in (509, 511):
        scales2[index] = FIELD - 1
    return message, pi1, pi2, scales1, scales2


def prefix_sums(values: Iterable[int], modulus: int) -> list[int]:
    """Literal modular prefix sums of canonical integer residues."""
    _integer("modulus", modulus, 2)
    result = []
    total = 0
    for value in values:
        _integer("value", value, 0, modulus - 1)
        total = (total + value) % modulus
        result.append(total)
    return result


def encode_literal(
    message: Sequence[int],
    repeat: int,
    modulus: int,
    pi1: Sequence[int],
    pi2: Sequence[int],
    scales1: Sequence[int],
    scales2: Sequence[int],
) -> list[int]:
    """Materialize contiguous repetition, input gathers, and both accumulators."""
    _integer("repeat", repeat, 1)
    _integer("modulus", modulus, 2)
    if not message:
        raise ValueError("message must be nonempty")
    for value in message:
        _integer("message value", value, 0, modulus - 1)
    length = len(message) * repeat
    for permutation in (pi1, pi2):
        if len(permutation) != length:
            raise ValueError("permutation length mismatch")
        for index in permutation:
            _integer("permutation index", index, 0, length - 1)
        if sorted(permutation) != list(range(length)):
            raise ValueError("indices must form a permutation")
    for scales in (scales1, scales2):
        if len(scales) != length:
            raise ValueError("scale length mismatch")
        for scale in scales:
            _integer("scale", scale, 1, modulus - 1)
    repeated = [value for value in message for _ in range(repeat)]
    first_input = [repeated[pi1[j]] * scales1[j] % modulus for j in range(length)]
    first = prefix_sums(first_input, modulus)
    second_input = [first[pi2[j]] * scales2[j] % modulus for j in range(length)]
    return prefix_sums(second_input, modulus)


def miss_bound(length: int, weight: int, check_weight: int, repetitions: int) -> Fraction:
    """Exact avoidance probability for independent uniform check supports.

    Avoidance is sufficient, not necessary, for zero field inner products; this
    is only a conditional ideal-check lower bound on missing this fixed word.
    """
    _integer("length", length, 0)
    _integer("weight", weight, 0, length)
    _integer("check_weight", check_weight, 0, length)
    _integer("repetitions", repetitions, 1)
    if check_weight > length - weight:
        return Fraction(0)
    return Fraction(
        comb(length - weight, check_weight), comb(length, check_weight)
    ) ** repetitions


def main() -> int:
    message, pi1, pi2, scales1, scales2 = primary_witness()
    encoded = encode_literal(message, REPEAT, FIELD, pi1, pi2, scales1, scales2)
    support = [index for index, value in enumerate(encoded) if value != 0]
    expected = [0] * LENGTH
    expected[508] = expected[510] = 1
    assert LENGTH == DIMENSION * REPEAT
    assert encoded == expected
    bound = miss_bound(LENGTH, 2, 31, 2)
    assert bound == Fraction(481 * 480, 512 * 511) ** 2
    assert bound > Fraction(3, 4)
    print(json.dumps({
        "scope": "conditional_independent_uniform_support_avoidance_only",
        "geometry": {"field": FIELD, "dimension": DIMENSION, "repeat": REPEAT, "length": LENGTH},
        "message_support": [63],
        "witness_support": support,
        "witness_values": [encoded[index] for index in support],
        "distance_conclusion": "at_most_two_not_exact_minimum_distance",
        "check_weight": 31,
        "repetitions": 2,
        "miss_bound": {"numerator": bound.numerator, "denominator": bound.denominator},
        "greater_than_three_quarters": bound > Fraction(3, 4),
        "limitations": (
            "No protocol or production security certification; no claim that the actual "
            "generator selects this code or that production admission accepts it. "
            "Production OwnerBridgeMissing and other gates remain unchanged."
        ),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
