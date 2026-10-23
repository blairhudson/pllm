"""Conditional format-class proof, not a source importer or model implementation.

Only integers/Fractions and the existing research rounding oracle are used.
No payload, byte decoder, native arithmetic or checkpoint is consumed. Results
cover mathematical format classes under D32 and the existing 0<S<=2**45 premise;
source and whole-graph status remain unresolved. Fractions do not retain signed
zero: its IEEE sign rule is checked separately, not claimed as machine bit parity.

Run from repository root: python -B -m benchmarks.research.maverick_weight_product_bounds
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from benchmarks.research.maverick_totality_bounds import (
    I64_MAX,
    I64_MIN,
    nearest_integer,
    power,
    round_bound,
)


@dataclass(frozen=True)
class DyadicClass:
    """Exact positive magnitudes m*2**exponent, lo<=m<=hi; not serialized input."""

    lo: int
    hi: int
    exponent: int


# The e=15,m=15 E4M3FN NaN is deliberately absent; zero is handled separately.
WEIGHT_CLASSES = (
    DyadicClass(1, 7, -9),
    *(DyadicClass(8, 15, e - 10) for e in range(1, 15)),
    DyadicClass(8, 14, 5),
)
# BF16 subnormals, complete normal exponents through 2**44, then exactly 2**45.
SCALE_CLASSES = (
    DyadicClass(1, 127, -133),
    *(DyadicClass(128, 255, a - 134) for a in range(1, 172)),
    DyadicClass(128, 128, 38),
)
SCALE_MAX = power(45)
MIN_NONZERO_PRODUCT = power(-142)
MAX_PRODUCT = 448 * SCALE_MAX
MAX_Q8_ABS = 7 * (1 << 59)


@dataclass(frozen=True)
class ProductClassProof:
    """A conditional mathematical conclusion bound to the two stated classes."""

    weight: DyadicClass
    scale: DyadicClass
    coefficient_bits: int
    quantum_exponent: int
    minimum: Fraction
    maximum: Fraction
    q8_abs_bound: int

    @property
    def source_status(self) -> str:
        return "unresolved"

    @property
    def whole_graph_status(self) -> str:
        return "unresolved"


def verify_class(weight: DyadicClass, scale: DyadicClass) -> ProductClassProof:
    """Check sufficient exact-F32 conditions for every coefficient in one class.

    This is not an endpoint-only proof: coefficient precision, minimum grid and
    maximum magnitude jointly establish representability for the entire class.
    Endpoint oracle calls independently regress the cast/nearest-even identities.
    """
    classes = (weight, scale)
    if (
        any(type(c) is not DyadicClass for c in classes)
        or any(type(v) is not int for c in classes for v in (c.lo, c.hi, c.exponent))
        or weight not in WEIGHT_CLASSES
        or scale not in SCALE_CLASSES
    ):
        raise ValueError("outside the selected finite format classes and scale premise")
    coefficient = weight.hi * scale.hi
    exponent = weight.exponent + scale.exponent
    low = weight.lo * scale.lo * power(exponent)
    high = coefficient * power(exponent)
    assert coefficient.bit_length() <= 12 < 24
    assert exponent >= -142 > -149
    assert MIN_NONZERO_PRODUCT <= low <= high <= MAX_PRODUCT < power(54)
    # A subnormal result is an integer multiple of the F32 quantum; a normal
    # result needs <=12 significant bits and has exponent below overflow.
    assert (power(exponent) / power(-149)).denominator == 1
    for value in (low, high):
        for sign in (1, -1):
            assert round_bound(sign * value, 24) == sign * value
    bound = nearest_integer(256 * high)
    assert 0 <= bound <= MAX_Q8_ABS < I64_MAX
    assert I64_MIN < -MAX_Q8_ABS
    return ProductClassProof(
        weight, scale, coefficient.bit_length(), exponent, low, high, bound,
    )


def verify_special_cases() -> None:
    """IEEE zero-sign truth table and scalar exclusions, not hardware tests."""
    for weight_sign in (0, 1):
        # Positive finite S has sign zero. IEEE multiplication uses sign XOR.
        product_sign = weight_sign ^ 0
        assert product_sign == weight_sign
        assert product_sign << 31 == weight_sign << 31
    assert nearest_integer(Fraction(0)) == 0  # integer Q8 has no signed zero
    assert WEIGHT_CLASSES[-1] == DyadicClass(8, 14, 5)  # m=15 would be NaN
    assert all(c.lo > 0 and c.exponent + 134 < 255 for c in SCALE_CLASSES)

    # Finite BF16 alone is insufficient when the existing upper bound is removed.
    assert round_bound(power(120), 8) == power(120)
    assert round_bound(448 * power(120), 24) is None
    assert round_bound(448 * power(47), 24) == 448 * power(47)
    assert nearest_integer(256 * 448 * power(47)) > I64_MAX

    # Arbitrary F32 scales are NOT D32's exactly promoted BF16 scales.
    arbitrary_f32 = 1 + power(-23)
    assert round_bound(arbitrary_f32, 24) == arbitrary_f32
    assert round_bound(arbitrary_f32, 8) != arbitrary_f32
    product = Fraction(15, 8) * arbitrary_f32
    assert round_bound(product, 24) != product
    assert round_bound(power(-149), 24) == power(-149)
    assert round_bound(power(-9) * power(-149), 24) == 0


def main() -> None:
    if not __debug__:
        raise SystemExit("Run without -O: mathematical regressions use assertions.")
    for weight in WEIGHT_CLASSES:
        for scale in SCALE_CLASSES:
            verify_class(weight, scale)
    verify_special_cases()
    print(f"Passed {len(WEIGHT_CLASSES) * len(SCALE_CLASSES)} factored positive format classes; signs follow by symmetry.")
    print("R32(v*S)=v*S exactly; nonzero magnitudes lie in [2**-142, 448*2**45].")
    print("Q8 nearest-even remains; |Wq|<=7*2**59. Zero signs/exclusions checked symbolically.")
    print("Conditional mathematics only: source, whole graph and honest no-wrap remain unproved.")


if __name__ == "__main__":
    main()
