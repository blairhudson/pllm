"""Finite locked-implementation reference, not a general mathematical pow.

Source: registry libm 0.2.16, src/math/powf.rs:20-43,217-342;
checksum b6d2cec3eae94f9f509c767b45932f1ada8350c4bdb85af2fcab4a3c14807981.
Assume RN-even binary32 after EACH operator, no fusion/reassociation. Only
base 0x49742400 and exact j/32, j=0..31, are admitted. No observed words enter
the algorithm. General ideal-math reference and failing Core ideal contract
remain independent and unchanged. No target/native parity is asserted here.

sqrt.rs:16-28 selects arch sqrt for enabled AArch64+NEON, SSE2, or wasm32
intrinsics, otherwise generic sqrt. This module proves only the exact positive
square 1000**2; it does not emulate a general sqrt or qualify target FP controls.

The specialized arithmetic below derives from Sun's powf implementation:
Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunPro, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely granted,
provided that this notice is preserved.
"""

from fractions import Fraction

import pytest

from rotary_reference import f32_dyadic, round_f32


# Decimal source spellings, not Python floats. Hex annotations independently
# check the literal decoding, not outputs of the implemented algorithm.
_SOURCE_CONSTANTS = (
    ("L1", "6.0000002384e-01", 0x3F19999A),
    ("L2", "4.2857143283e-01", 0x3EDB6DB7),
    ("L3", "3.3333334327e-01", 0x3EAAAAAB),
    ("L4", "2.7272811532e-01", 0x3E8BA305),
    ("L5", "2.3066075146e-01", 0x3E6C3255),
    ("L6", "2.0697501302e-01", 0x3E53F142),
    ("P1", "1.6666667163e-01", 0x3E2AAAAB),
    ("P2", "-2.7777778450e-03", 0xBB360B61),
    ("P3", "6.6137559770e-05", 0x388AB355),
    ("P4", "-1.6533901999e-06", 0xB5DDEA0E),
    ("P5", "4.1381369442e-08", 0x3331BB4C),
    ("LG2", "6.9314718246e-01", 0x3F317218),
    ("LG2_H", "6.93145752e-01", 0x3F317200),
    ("LG2_L", "1.42860654e-06", 0x35BFBE8C),
    ("CP", "9.6179670095e-01", 0x3F76384F),
    ("CP_H", "9.6191406250e-01", 0x3F764000),
    ("CP_L", "-1.1736857402e-04", 0xB8F623C6),
)


def _require(condition, message):
    if not condition:
        raise ArithmeticError(f"unproved locked-power branch: {message}")


class _F:
    """Binary32 word; exact rational operators with explicit signed-zero rules."""

    def __init__(self, word):
        if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
            raise ValueError("exact binary32 word required")
        _require((word & 0x7F800000) != 0x7F800000, "nonfinite word")
        self.word = word

    @property
    def value(self):
        return f32_dyadic(self.word)

    @classmethod
    def rational(cls, value, negative_zero=False):
        _require(type(value) in (int, Fraction), "nonrational operand")
        _require(abs(value.numerator).bit_length() <= 512
                 and value.denominator.bit_length() <= 512, "arithmetic bound")
        # Finite source-profile intermediates are far below overflow. Refuse
        # it rather than implement unrelated exceptional powf branches.
        _require(abs(value) <= f32_dyadic(0x7F7FFFFF), "overflow")
        return cls(round_f32(value, negative_zero=negative_zero))

    def __neg__(self):
        return _F(self.word ^ 0x80000000)

    def __add__(self, other):
        value = self.value + other.value
        # RN exact cancellation is +0, except -0 + -0.
        negative_zero = self.word == other.word == 0x80000000
        return _F.rational(value, negative_zero)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return _F.rational(self.value * other.value,
                           bool((self.word ^ other.word) & 0x80000000))

    def __truediv__(self, other):
        _require(other.value != 0, "division by zero")
        return _F.rational(self.value / other.value,
                           bool((self.word ^ other.word) & 0x80000000))

    def high(self, mask):
        _require(mask in (0xFFFFF000, 0xFFFF8000), "unsupported high mask")
        return _F(self.word & mask)


def _constants():
    return {name: _F.rational(Fraction(decimal))
            for name, decimal, _ in _SOURCE_CONSTANTS}


def _common_log(c):
    """Specialize 217-277 to the proven final base interval; no seeded pair."""
    one, zero, three = _F.rational(1), _F.rational(0), _F.rational(3)
    ix = 0x49742400
    _require(f32_dyadic(ix) == 1000000, "fixed positive normal base")
    _require(0x00800000 <= ix < 0x7F800000, "subnormal/special base")
    n = (ix >> 23) - 0x7F
    fraction = ix & 0x007FFFFF
    _require(fraction >= 0x5DB3D7, "other log interval")
    ix = (fraction | 0x3F800000) - 0x00800000
    n += 1
    # This branch has k=0, BP=1 and both DP terms exactly +0.
    ax = _F(ix)
    u = ax - one
    v = one / (ax + one)
    s = u * v
    s_h = s.high(0xFFFFF000)
    denominator_word = ((ix >> 1) & 0xFFFFF000) | 0x20000000
    t_h = _F(denominator_word + 0x00400000)
    t_l = ax - (t_h - one)
    s_l = v * ((u - s_h * t_h) - s_h * t_l)
    s2 = s * s
    r = s2 * s2 * (c["L1"] + s2 * (c["L2"] + s2 * (
        c["L3"] + s2 * (c["L4"] + s2 * (c["L5"] + s2 * c["L6"])))))
    r = r + s_l * (s_h + s)
    s2 = s_h * s_h
    t_h = ((three + s2) + r).high(0xFFFFF000)
    t_l = r - ((t_h - three) - s2)
    u = s_h * t_h
    v = s_l * t_h + t_l * s
    p_h = (u + v).high(0xFFFFF000)
    p_l = v - (p_h - u)
    z_h = c["CP_H"] * p_h
    z_l = (c["CP_L"] * p_h + p_l * c["CP"]) + zero
    t = _F.rational(n)
    t1 = (((z_h + z_l) + zero) + t).high(0xFFFFF000)
    t2 = z_l - (((t1 - t) - zero) - z_h)
    return t1, t2, {"base_n": n, "ax": ax.word, "s": s.word,
                    "s_h": s_h.word, "s_l": s_l.word,
                    "log_high": t1.word, "log_low": t2.word}


def _evaluate(j):
    # Eager, before constant parsing or any arithmetic. Not runtime admission.
    if type(j) is not int or not 0 <= j <= 31:
        raise ValueError("exact non-bool integer j in [0, 31] required")
    y = _F.rational(Fraction(j, 32))
    trace = {"j": j, "x": 0x49742400, "y": y.word}
    if j == 0:
        trace["branch"] = "zero exponent"
        return 0x3F800000, trace
    if j == 16:
        # Positive exact square, no general sqrt calculation or host FP.
        root = 1000
        _require(root > 0 and root * root == f32_dyadic(0x49742400),
                 "positive perfect square")
        trace["branch"] = "exact positive sqrt"
        return _F.rational(root).word, trace
    _require(0 < y.word < 0x3F800000 and y.word != 0x3F000000,
             "special/huge/negative exponent")
    c = _constants()
    t1, t2, log_trace = _common_log(c)
    trace.update(log_trace)
    trace["branch"] = "finite generic"
    one, two = _F.rational(1), _F.rational(2)
    # 280-285. Keep both low-product terms, even though this y split is exact.
    y1 = y.high(0xFFFFF000)
    p_l = (y - y1) * t1 + y * t2
    p_h = y1 * t1
    z = p_l + p_h
    trace.update(y_high=y1.word, product_high=p_h.word,
                 product_low=p_l.word, product_sum=z.word)
    word = z.word
    # Strictly positive z<128 excludes EVERY overflow/underflow test 287-304,
    # including equality branches; nothing silently substitutes HUGE/TINY.
    _require(0 < word < 0x43000000, "exceptional exponent range")
    k = (word >> 23) - 0x7F
    n = 0
    if word > 0x3F000000:
        _require(-1 <= k <= 6, "reduction shift range")
        rounded_word = word + (0x00800000 >> (k + 1))
        _require(rounded_word < 0x80000000, "signed i32 addition")
        k = (rounded_word >> 23) - 0x7F
        _require(0 <= k <= 7, "integer extraction shift range")
        t = _F(rounded_word & (~(0x007FFFFF >> k) & 0xFFFFFFFF))
        n = ((rounded_word & 0x007FFFFF) | 0x00800000) >> (23 - k)
        _require(t.value == n, "exact positive integer reduction")
        p_h = p_h - t
    else:
        _require(z.value <= Fraction(1, 2), "unreduced exponent")
    trace.update(reduction_n=n, reduced_high=p_h.word)
    # 323-329: second, distinct high mask, separate low correction.
    t = (p_l + p_h).high(0xFFFF8000)
    u = t * c["LG2_H"]
    v = (p_l - (t - p_h)) * c["LG2"] + t * c["LG2_L"]
    z = u + v
    w = v - (z - u)
    trace.update(residual_high=t.word, exp_argument=z.word, exp_low=w.word)
    t = z * z
    t1 = z - t * (c["P1"] + t * (c["P2"] + t * (
        c["P3"] + t * (c["P4"] + t * c["P5"]))))
    r = (z * t1) / (t1 - two) - (w + z * w)
    difference = r - z
    trace.update(exp_polynomial=t1.word, exp_correction=r.word,
                 difference_exact=r.value - z.value,
                 difference_rounded=difference.word,
                 reconstruction_exact=one.value - difference.value)
    z = one - difference
    trace["reconstruction"] = z.word
    _require(0x00800000 <= z.word < 0x7F800000, "reconstruction sign/normality")
    scaled_word = z.word + (n << 23)
    _require(0x00800000 <= scaled_word < 0x7F800000,
             "normal word rescale (no scalbnf/overflow)")
    # Signed i32 reinterpretations coincide with positive ints at these sites.
    result = one * _F(scaled_word)  # sn=+1, source's final operation retained.
    return result.word, trace


def _locked_power(j):
    """Predict one locked-profile word; only caller input is exact integer j."""
    return _evaluate(j)[0]


@pytest.mark.parametrize("name,decimal,word", _SOURCE_CONSTANTS)
def test_locked_source_decimal_constants(name, decimal, word):
    assert _constants()[name].word == word
    assert round_f32(Fraction(decimal)) == word


def test_locked_complete_ordered_profile():
    # Compute entire profile using one algorithm before any native comparison.
    profile = tuple((j, _evaluate(j)) for j in range(32))
    assert len(profile) == 32
    assert tuple(j for j, _ in profile) == tuple(range(32))
    inputs = tuple(result[1]["y"] for _, result in profile)
    assert len(set(inputs)) == 32 and inputs == tuple(sorted(inputs))
    assert all(type(result[0]) is int and 0x3F800000 <= result[0] < 0x49742400
               for _, result in profile)
    assert sum(result[1]["branch"] == "finite generic" for _, result in profile) == 30
    assert all(result[1]["log_high"] == 0x419F7000
               and result[1]["log_low"] == 0x3AF68E36
               for _, result in profile if result[1]["branch"] == "finite generic")


@pytest.mark.parametrize("j", [True, False, -1, 32, 1 << 100, Fraction(9), "9", None])
def test_locked_eager_exact_integer_validation(j):
    with pytest.raises(ValueError, match="exact non-bool integer"):
        _locked_power(j)


def test_locked_endpoint_and_positive_sqrt_proof():
    assert f32_dyadic(0x49742400) == 1000000
    assert 1000 * 1000 == 1000000
    assert f32_dyadic(0x447A0000) == 1000
    assert _locked_power(0) == 0x3F800000
    assert _locked_power(16) == 0x447A0000


def test_locked_common_log_independent_witness():
    high, low, trace = _common_log(_constants())
    assert trace == {"base_n": 20, "ax": 0x3F742400,
                     "s": 0xBCC23FD6, "s_h": 0xBCC23000, "s_l": 0xB6FD68A2,
                     "log_high": 0x419F7000, "log_low": 0x3AF68E36}
    assert (high.word, low.word) == (0x419F7000, 0x3AF68E36)


def test_locked_j9_stages_and_decisive_midpoint():
    word, trace = _evaluate(9)
    assert trace["y"] == 0x3E900000
    assert (trace["product_high"], trace["product_low"], trace["product_sum"]) == (
        0x40B35E00, 0x3A0AAFFE, 0x40B36255)
    assert trace["reduction_n"] == 6
    assert (trace["exp_argument"], trace["exp_low"]) == (0xBE8BEA24, 0xB25B0000)
    assert (trace["exp_polynomial"], trace["exp_correction"]) == (0xBE924775, 0xBD0BE8B9)
    assert trace["difference_exact"] == Fraction(64186471, 1 << 28)
    assert trace["difference_rounded"] == 0x3E74DA1A
    assert f32_dyadic(trace["difference_rounded"]) == Fraction(64186472, 1 << 28)
    m = 0xC2C979
    midpoint = Fraction(2 * m + 1, 1 << 25)
    assert trace["reconstruction_exact"] == Fraction(204248984, 1 << 28) == midpoint
    assert midpoint == (Fraction(m, 1 << 24) + Fraction(m + 1, 1 << 24)) / 2
    assert m & 1 and not (m + 1) & 1
    assert trace["reconstruction"] == 0x3F42C97A
    assert word == 0x4242C97A


def test_ideal_j9_independent_integer_certificate():
    # Exact real power is 10**(27/16). Raising its two binary32 midpoints
    # to degree 16 gives strict inequalities, independently of either helper.
    m = 0xC2C979
    a = 10 ** 27 * 2 ** 304
    lower_margin = a - (2 * m - 1) ** 16
    upper_margin = (2 * m + 1) ** 16 - a
    assert lower_margin > 0 and upper_margin > 0
    assert lower_margin >> 360 == 17249
    assert upper_margin >> 360 == 144
    assert 0x4242C979 == (132 << 23) | (m - (1 << 23))
    # Locked prediction differs; it must never replace the ideal expectation.
    assert _locked_power(9) == 0x4242C97A != 0x4242C979


def test_locked_signed_zero_masks_and_refusals():
    positive, negative = _F(0), _F(0x80000000)
    one, minus_one = _F.rational(1), _F.rational(-1)
    assert (negative + negative).word == 0x80000000
    assert (negative + positive).word == 0
    assert (one - one).word == 0
    assert (negative - positive).word == 0x80000000
    assert (positive * minus_one).word == 0x80000000
    assert (negative / minus_one).word == 0
    assert _F(0xBCC23FD6).high(0xFFFFF000).word == 0xBCC23000
    assert _F(0xBCC23FD6).high(0xFFFF8000).word == 0xBCC20000
    with pytest.raises(ArithmeticError, match="division by zero"):
        one / positive
    with pytest.raises(ArithmeticError, match="nonfinite"):
        _F(0x7F800000)
    with pytest.raises(ArithmeticError, match="arithmetic bound"):
        _F.rational(1 << 512)
    with pytest.raises(ArithmeticError, match="overflow"):
        _F.rational(1 << 128)
    with pytest.raises(ArithmeticError, match="unsupported high mask"):
        one.high(0)
