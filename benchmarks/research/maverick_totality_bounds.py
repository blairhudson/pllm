"""Exact scalar regressions for maverick-canonical-reply-totality.md, section 5.

Run with ordinary Python, without -O. Only integers and Fractions are used.
These are certificate-endpoint calculations, not a model/Experiment runtime or
a transcendental implementation. No checkpoint conditions, honest no-wrap,
security, admission, timing or quality are established. Fraction has no signed
zero: these magnitude checks do not establish floating-point bit parity.
"""
from __future__ import annotations

from fractions import Fraction


P = 2_013_265_921
H = (P - 1) // 2
L = 15_360
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
ENVELOPES = tuple((h, g) for h in range(22, 45) for g in range(45 - h))


def power(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def nearest_integer(value: Fraction) -> int:
    """Exact integer nearest-even, without a floating intermediate."""
    value = Fraction(value)
    quotient, remainder = divmod(abs(value.numerator), value.denominator)
    twice = 2 * remainder
    if twice > value.denominator or (twice == value.denominator and quotient % 2):
        quotient += 1
    return -quotient if value < 0 else quotient


def round_bound(value: Fraction, precision: int) -> Fraction | None:
    """Round one scalar bound to BF16/F32; None denotes nonfinite overflow.

    This research-only arithmetic helper includes gradual underflow and ties.
    None is an observation about the mathematical cast, not an online failure or
    fallback policy. No machine floating-point operations are performed.
    """
    if precision not in (8, 24):
        raise ValueError("only the selected BF16/F32 precisions are checked")
    value = Fraction(value)
    if not value:
        return Fraction(0)
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < power(exponent):
        exponent -= 1
    quantum = power(max(exponent, -126) - precision + 1)
    rounded = nearest_integer(value / quantum) * quantum
    return None if abs(rounded) >= power(128) else rounded


def finite_round(value: Fraction, precision: int) -> Fraction:
    rounded = round_bound(value, precision)
    assert rounded is not None, "a proof endpoint became nonfinite"
    return rounded


def residual_endpoints(h: int) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """The two scalar bounding traces, including all 72 separate residual adds."""
    if not 22 <= h <= 44:
        raise ValueError("h is outside the sufficient proof envelope")
    traces = []
    for sign in (1, -1):
        values = [sign * power(h)]
        for _ in range(72):
            values.append(finite_round(finite_round(values[-1] + sign * L, 24), 8))
        traces.append(tuple(values))
    return traces[0], traces[1]


def verify_envelope(h: int, g: int) -> None:
    """Check section 5's endpoint identities, not private or source tensor values."""
    if h < 22 or g < 0 or h + g > 44:
        raise ValueError("outside the sufficient h/g certificate family")
    residual, gamma = power(h), power(g)
    midpoint = residual + power(h - 8)
    assert L < power(h - 8)
    assert finite_round(midpoint, 24) == midpoint
    assert finite_round(midpoint, 8) == residual
    positive, negative = residual_endpoints(h)
    assert len(positive) == len(negative) == 73
    assert all(value == residual for value in positive)
    assert all(value == -residual for value in negative)

    eps = finite_round(Fraction(1, 1_000_000), 24)
    assert power(-20) <= eps <= power(-19)
    for s, dimension in ((h, 2560), (14, 128)):
        square = finite_round(power(s) * power(s), 24)
        total = dimension * square
        assert dimension.bit_length() <= 12
        assert finite_round(total, 24) == total < power(2 * s + 12) <= power(100)
        mean = finite_round(total / dimension, 24)
        assert mean == power(2 * s)
        assert finite_round(eps, 24) == eps > 0  # all-zero variance endpoint
        assert finite_round(mean + eps, 24) <= power(2 * s + 1)

        # Squared rational inequalities prove the rsqrt interval without sqrt.
        lower_b, upper_b = power(-s - 1), power(10)
        assert lower_b * lower_b * power(2 * s + 1) == Fraction(1, 2)
        assert upper_b * upper_b * eps >= 1
        assert finite_round(lower_b, 24) == lower_b
        assert finite_round(upper_b, 24) == upper_b
        before_gamma = finite_round(finite_round(power(s) * upper_b, 24), 8)
        norm = finite_round(finite_round(gamma * before_gamma, 24), 8)
        assert norm == power(s + g + 10)
        if s == h:
            xq = nearest_integer(256 * norm)
            assert xq == power(h + g + 18) <= power(62)
            assert I64_MIN < -xq <= xq < I64_MAX

    qk_norm = power(g + 24)
    rope = finite_round(finite_round(qk_norm + qk_norm, 24), 8)
    assert rope == power(g + 25)
    product = finite_round(rope * rope, 24)
    score = finite_round(128 * product, 24)
    assert score == power(2 * g + 57) <= power(101)
    assert finite_round(score, 8) == score  # required cast BEFORE alpha
    assert finite_round(finite_round(score, 24), 8) == score  # alpha upper bound 1
    assert finite_round(-2 * score, 24) == -2 * score >= -power(102)

    # Unit exp/probability endpoints, not an implementation of exp or softmax.
    for n in range(1, 9):
        assert finite_round(Fraction(n), 24) == n
        assert finite_round(finite_round(Fraction(1, n), 24), 8) > 0
        context = finite_round(finite_round(Fraction(n * L), 24), 8)
        assert context == n * L <= 8 * L
    assert nearest_integer(256 * Fraction(8 * L)) == 31_457_280

    # Source-derived Weff and unchanged BF16-head sufficient bounds.
    assert finite_round(448 * power(45), 24) <= power(54)
    assert nearest_integer(256 * power(54)) == power(62) < I64_MAX
    assert nearest_integer(256 * residual) == power(h + 8) < I64_MAX


def verify_fixed_endpoints() -> None:
    """Symbolic SiLU squeeze, Q8/mod-p identities, and excluded source examples."""
    assert H == L * 65_536
    # exp(L)>L^2/2 implies the exact SiLU value lies in this rounding interval.
    assert Fraction(2, L) < power(-11)
    assert finite_round(L - Fraction(2, L), 24) == finite_round(Fraction(L), 24) == L
    ff = finite_round(finite_round(Fraction(L * L), 24), 8)
    assert ff == 235_929_600
    xq = nearest_integer(256 * ff)
    assert xq == 60_397_977_600 < I64_MAX
    assert divmod(xq, P) == (29, P - 30)
    assert (xq % P) - P == -30
    assert (P - 1) ** 2 + P - 1 == 4_053_239_666_646_712_320 < (1 << 63)

    # Excluded source: finite BF16 embedding, nonfinite F32 square.
    assert round_bound(power(64), 8) == power(64)
    assert round_bound(power(128), 24) is None

    # Excluded source: all-one RMS input, finite gamma, out-of-i64 Q8.
    eps = finite_round(Fraction(1, 1_000_000), 24)
    a = finite_round(1 + eps, 24)
    lower_b = 1 - power(-19)
    assert lower_b * lower_b * a < 1  # b >= lower_b, with b <= 1
    assert finite_round(lower_b, 24) == lower_b
    assert finite_round(lower_b, 8) == finite_round(Fraction(1), 8) == 1
    norm = finite_round(finite_round(power(55), 24), 8)
    assert norm == power(55)
    assert nearest_integer(256 * norm) == 1 << 63
    assert nearest_integer(256 * norm) > I64_MAX


def main() -> None:
    if not __debug__:
        raise SystemExit("Run without -O: mathematical regressions use assertions.")
    for h, g in ENVELOPES:
        verify_envelope(h, g)
    verify_fixed_endpoints()
    print(f"Passed {len(ENVELOPES)} conditional h/g envelopes; 72 residual updates per sign/pair.")
    print("RMS/Q8/field endpoints and excluded nonfinite/i64 counterexamples checked exactly.")
    print("Mathematics only: checkpoint, honest no-wrap, security and admission remain unproved.")


if __name__ == "__main__":
    main()
