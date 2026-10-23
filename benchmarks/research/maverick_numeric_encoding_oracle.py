"""Independent D31/D32 scalar encoding oracle: stdlib integers/Fractions only.

Selected equations: docs/research/maverick-numeric-source-recovery.md, section 3.2.
This checks format/conversion conformance, not tensor provenance, honest no-wrap,
Qwen3 execution, quality, privacy, or an adopted runtime profile. No PUMA/FSS
numeric semantics are imported. Float rounding searches the exact finite encoding
lattice; it never calls host float arithmetic, libm, or a native implementation.

Run small diagnostic vectors: python -B -m benchmarks.research.maverick_numeric_encoding_oracle
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json

BABYBEAR = 2_013_265_921
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
SCALE_MAX = 1 << 45


@dataclass(frozen=True)
class ExactFloat:
    """Finite exact value plus the zero sign which Fraction cannot represent."""

    value: Fraction
    negative_zero: bool = False

    def __post_init__(self) -> None:
        if type(self.value) is not Fraction or type(self.negative_zero) is not bool:
            raise TypeError("expected Fraction and Boolean zero sign")
        if self.negative_zero and self.value:
            raise ValueError("a nonzero value cannot have a negative-zero tag")


def _word(value: int, bits: int) -> int:
    if type(value) is not int:
        raise TypeError("serialized words must be integers, not floats or Booleans")
    if not 0 <= value < (1 << bits):
        raise ValueError(f"not a canonical {bits}-bit word")
    return value


def _fraction(value: Fraction | int) -> Fraction:
    if type(value) not in (Fraction, int):
        raise TypeError("oracle accepts only exact Fraction/int values")
    return Fraction(value)


def _power(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def decode_e4m3fn(byte: int) -> ExactFloat:
    """Decode all 254 finite encodings; both e=15,f=7 NaNs reject."""
    byte = _word(byte, 8)
    sign, exponent, fraction = byte >> 7, (byte >> 3) & 15, byte & 7
    if exponent == 15 and fraction == 7:
        raise ValueError("E4M3FN NaN")
    magnitude = (
        Fraction(fraction, 512)
        if exponent == 0
        else (8 + fraction) * _power(exponent - 10)
    )
    return ExactFloat(-magnitude if sign else magnitude, bool(sign and not magnitude))


def _positive_ieee(bits: int, fraction_bits: int) -> Fraction:
    exponent, fraction = bits >> fraction_bits, bits & ((1 << fraction_bits) - 1)
    if exponent == 0:
        return fraction * _power(-126 - fraction_bits)
    return ((1 << fraction_bits) + fraction) * _power(exponent - 127 - fraction_bits)


def _decode_ieee(word: int, fraction_bits: int) -> ExactFloat:
    word = _word(word, fraction_bits + 9)
    sign = word >> (fraction_bits + 8)
    magnitude_bits = word & ((1 << (fraction_bits + 8)) - 1)
    if magnitude_bits >> fraction_bits == 255:
        raise ValueError("nonfinite IEEE encoding")
    magnitude = _positive_ieee(magnitude_bits, fraction_bits)
    return ExactFloat(-magnitude if sign else magnitude, bool(sign and not magnitude))


def decode_bf16(word: int) -> ExactFloat:
    return _decode_ieee(word, 7)


def decode_f32(word: int) -> ExactFloat:
    return _decode_ieee(word, 23)


def promote_bf16_to_f32_bits(word: int) -> int:
    """Exact finite promotion, including subnormals and signed zero; not a scale gate."""
    decode_bf16(word)
    return word << 16


def validate_scale_bf16(word: int) -> ExactFloat:
    """Check the separate conditional premise 0 < stored BF16 scale <= 2**45.

    This is not source authentication or evidence that checkpoint payloads satisfy
    the premise. Exact promotion is the sole D32 storage conversion.
    """
    scale = decode_f32(promote_bf16_to_f32_bits(word))
    if not 0 < scale.value <= SCALE_MAX:
        raise ValueError("scale must be positive, finite, and at most 2**45")
    return scale


def _nearest_ieee_bits(
    value: Fraction | int, fraction_bits: int, negative_zero: bool,
) -> int:
    value = _fraction(value)
    if type(negative_zero) is not bool:
        raise TypeError("zero sign must be Boolean")
    if negative_zero and value:
        raise ValueError("negative_zero is only meaningful for exact zero")
    sign = int(value < 0 or negative_zero) << (fraction_bits + 8)
    magnitude = abs(value)
    maximum = (255 << fraction_bits) - 1
    # Infinity is the even endpoint at the overflow midpoint. Nonfinite
    # non-mask intermediates reject; values below that midpoint can round finite.
    overflow_midpoint = _positive_ieee(maximum, fraction_bits) + _power(126 - fraction_bits)
    if magnitude >= overflow_midpoint:
        raise OverflowError("round-to-nearest result is nonfinite")
    lower, upper = 0, maximum
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if _positive_ieee(middle, fraction_bits) <= magnitude:
            lower = middle
        else:
            upper = middle - 1
    low_value = _positive_ieee(lower, fraction_bits)
    if magnitude == low_value or lower == maximum:
        return sign | lower
    high_value = _positive_ieee(lower + 1, fraction_bits)
    low_distance, high_distance = magnitude - low_value, high_value - magnitude
    choose_lower = low_distance < high_distance or (low_distance == high_distance and lower % 2 == 0)
    return sign | (lower if choose_lower else lower + 1)


def round_bf16_bits(value: Fraction | int, *, negative_zero: bool = False) -> int:
    """Direct exact rational -> BF16, including gradual underflow and ties-even."""
    return _nearest_ieee_bits(value, 7, negative_zero)


def round_f32_bits(value: Fraction | int, *, negative_zero: bool = False) -> int:
    return _nearest_ieee_bits(value, 23, negative_zero)


def q8_i64(value: Fraction | int) -> int:
    """I(256 * exact_value), ties to even; no float intermediate or clipping."""
    scaled = 256 * _fraction(value)
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    rounded = quotient + int(
        twice > scaled.denominator or (twice == scaled.denominator and quotient % 2 != 0)
    )
    if not I64_MIN <= rounded <= I64_MAX:
        raise OverflowError("nearest-even Q8 integer is outside signed i64")
    return rounded


def projection_f32_bits(byte: int, scale_bf16: int) -> int:
    """R32(E4M3FN(byte) * exact-promoted stored BF16 scale), not inverse scale."""
    weight, scale = decode_e4m3fn(byte), validate_scale_bf16(scale_bf16)
    return round_f32_bits(weight.value * scale.value, negative_zero=weight.negative_zero)


def projection_q8(byte: int, scale_bf16: int) -> int:
    return q8_i64(decode_f32(projection_f32_bits(byte, scale_bf16)).value)


def centered_reply(residue: int) -> int:
    if type(residue) is not int:
        raise TypeError("field residues must be integers")
    if not 0 <= residue < BABYBEAR:
        raise ValueError("noncanonical BabyBear reply")
    return residue if residue <= (BABYBEAR - 1) // 2 else residue - BABYBEAR


def centered_reply_to_bf16_bits(residue: int) -> int:
    """R16(exact centered residue / 65536); intentionally NO R32 intermediate."""
    return round_bf16_bits(Fraction(centered_reply(residue), 65536))


def diagnostic_vectors() -> list[dict[str, int]]:
    """Small reusable direct-vs-double-rounding vectors, not a model experiment."""
    n = (1 << 24) + (1 << 16) + 1
    vectors = []
    for centered in (n - 2, n - 1, n, n + 1, -n + 2, -n + 1, -n, -n - 1):
        residue = centered % BABYBEAR
        f32_bits = round_f32_bits(Fraction(centered, 65536))
        vectors.append({
            "centered": centered,
            "residue": residue,
            "direct_bf16_bits": centered_reply_to_bf16_bits(residue),
            "f32_bits": f32_bits,
            "via_f32_bf16_bits": round_bf16_bits(decode_f32(f32_bits).value),
        })
    return vectors


if __name__ == "__main__":
    print(json.dumps(diagnostic_vectors(), indent=2))
