"""Finite, source-staged trig fixture; not a general trig oracle or parity claim.

Provenance: registry libm 0.2.16, src/math/{sinf,cosf,k_sinf,k_cosf}.rs;
crate checksum b6d2cec3eae94f9f509c767b45932f1ada8350c4bdb85af2fcab4a3c14807981.
The equations used here descend from FreeBSD /usr/src/lib/msun/src/
{s_sinf,s_cosf,k_sinf,k_cosf}.c. Conversion to float by Ian Lance Taylor,
Cygnus Support, ian@cygnus.com; optimized (k_cosf debugged and optimized)
by Bruce D. Evans.

Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunPro, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely granted,
provided that this notice is preserved.

Only theta=1000000, width=64, p=0..4, j=0..31. Assume individual RN-even
binary64 operations, then a kernel RN-even binary32 cast; no fusion or
reassociation. Tiny paths predict return values, not floating-point flags.
No native results or mathematical trig enter locked prediction.
"""

from collections import Counter
from fractions import Fraction
from hashlib import sha256
from typing import NamedTuple

import pytest

from rotary_reference import f32_dyadic, reference_pairs, round_f32
from test_rotary_locked_power import _locked_power


_SIGN64 = 1 << 63
_SIGN32 = 1 << 31
_MIN_BIN, _MAX_BIN = -256, 8
_MAX_BITS = 512


def _pow2(shift):
    return Fraction(1 << shift) if shift >= 0 else Fraction(1, 1 << -shift)


class _F64:
    """Signed zero or a binary64 normal in the fixture's bounded binades."""

    def __init__(self, word):
        if type(word) is not int or not 0 <= word < 1 << 64:
            raise ValueError("exact binary64 word required")
        exponent = ((word >> 52) & 0x7FF) - 1023
        if word & (_SIGN64 - 1) and not _MIN_BIN <= exponent <= _MAX_BIN:
            raise ArithmeticError("binary64 binade outside fixture bound")
        self.word = word

    @property
    def value(self):
        mag = self.word & (_SIGN64 - 1)
        if not mag:
            return Fraction(0)
        value = ((1 << 52) | (mag & ((1 << 52) - 1))) * _pow2((mag >> 52) - 1075)
        return -value if self.word & _SIGN64 else value

    @classmethod
    def rational(cls, value, *, negative_zero=False):
        if type(value) not in (int, Fraction) or type(negative_zero) is not bool:
            raise ValueError("exact rational and bool zero sign required")
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > _MAX_BITS:
            raise ArithmeticError("rational outside fixture bit bound")
        if not value:
            return cls(_SIGN64 if negative_zero else 0)
        sign = _SIGN64 if value < 0 else 0
        magnitude = abs(value)
        exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
        if magnitude < _pow2(exponent):
            exponent -= 1
        if not _MIN_BIN <= exponent <= _MAX_BIN:
            raise ArithmeticError("binary64 binade outside fixture bound")
        # Round the exact significand on this binade's 53-bit lattice.
        scaled = magnitude / _pow2(exponent - 52)
        q, r = divmod(scaled.numerator, scaled.denominator)
        if 2 * r > scaled.denominator or (2 * r == scaled.denominator and q & 1):
            q += 1
        if q == 1 << 53:
            q >>= 1
            exponent += 1
        return cls(sign | ((exponent + 1023) << 52) | (q - (1 << 52)))

    def __neg__(self):
        return _F64(self.word ^ _SIGN64)

    def __add__(self, other):
        return _F64.rational(self.value + other.value,
                             negative_zero=self.word == other.word == _SIGN64)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return _F64.rational(self.value * other.value,
                             negative_zero=bool((self.word ^ other.word) & _SIGN64))

    def to_f32(self):
        return round_f32(self.value, negative_zero=bool(self.word & _SIGN64))


# Decimal spellings are evaluator inputs; integer*2**shift annotations are
# separate checks, transcribed from the respective kernel source comments.
_KERNEL_LITERALS = (
    ("S1", "-0.166666666416265235595", -0x15555554CBAC77, -55),
    ("S2", "0.0083333293858894631756", 0x111110896EFBB2, -59),
    ("S3", "-0.000198393348360966317347", -0x1A00F9E2CAE774, -65),
    ("S4", "0.0000027183114939898219064", 0x16CD878C3B46A7, -71),
    ("C0", "-0.499999997251031003120", -0x1FFFFFFD0C5E81, -54),
    ("C1", "0.0416666233237390631894", 0x155553E1053A42, -57),
    ("C2", "-0.00138867637746099294692", -0x16C087E80F1E27, -62),
    ("C3", "0.0000243904487962774090654", 0x199342E0EE5069, -68),
)
# core::f64::consts::FRAC_PI_2, core/src/num/f64.rs. Multiples staged as
# in libm; sinf.rs/cosf.rs annotate each rounded word independently.
_PI_OVER_TWO = "1.57079632679489661923132169163975144"


def _constants():
    kernels = {name: _F64.rational(Fraction(decimal))
               for name, decimal, _, _ in _KERNEL_LITERALS}
    half_pi = _F64.rational(Fraction(_PI_OVER_TWO))
    reductions = tuple(_F64.rational(n) * half_pi for n in (1, 2, 3))
    return kernels, reductions


def _sine_polynomial(x, c):
    z = x * x
    w = z * z
    r = c["S3"] + z * c["S4"]
    s = z * x
    return (x + s * (c["S1"] + z * c["S2"])) + (s * w) * r


def _cosine_polynomial(x, c):
    z = x * x
    w = z * z
    r = c["C2"] + z * c["C3"]
    return ((_F64.rational(1) + z * c["C0"]) + w * c["C1"]) + (w * z) * r


class _Trace(NamedTuple):
    branch: str
    inputs64: tuple  # actual kernel arguments, in output (cos, sin) order
    polynomial64: tuple  # before the kernel cast and any outer sign


def _trig(angle, constants):
    # Positive-word order equals magnitude order. Check before widening and
    # dispatch: no negative/-0 input, outside-domain, or nonfinite branches.
    if type(angle) is not int or not 0 <= angle <= 0x40800000:
        raise ValueError("positive-sign binary32 angle word in [+0, 4] required")
    if angle < 0x39800000:
        return (0x3F800000, angle), _Trace("tiny", (), ())

    c, (d1, d2, d3) = constants
    exact = f32_dyadic(angle)
    x = _F64.rational(exact)
    if x.value != exact:
        raise ArithmeticError("inexact binary32 widening")

    flip_cos = flip_sin = False
    if angle <= 0x3F490FDA:
        branch = "direct"
        cx = sx = x
        cp, sp = _cosine_polynomial(cx, c), _sine_polynomial(sx, c)
    elif angle <= 0x4016CBE3:
        branch = "D1"
        cx, sx = d1 - x, x - d1
        cp, sp = _sine_polynomial(cx, c), _cosine_polynomial(sx, c)
    elif angle <= 0x407B53D1:
        branch = "D2"
        cx = x - d2
        sx = -cx
        cp, sp = _cosine_polynomial(cx, c), _sine_polynomial(sx, c)
        flip_cos = True
    else:
        branch = "D3"
        cx = sx = x - d3
        cp, sp = _sine_polynomial(cx, c), _cosine_polynomial(sx, c)
        flip_sin = True

    cosine, sine = cp.to_f32(), sp.to_f32()
    if flip_cos:
        cosine ^= _SIGN32
    if flip_sin:
        sine ^= _SIGN32
    return (cosine, sine), _Trace(branch, (cx.word, sx.word), (cp.word, sp.word))


class LockedTrigRow(NamedTuple):
    p: int
    j: int
    power: int
    inverse: int
    angle: int
    cosine: int
    sine: int
    cos_q18: int
    sin_q18: int


def _inverse(power):
    exact = f32_dyadic(power)
    if not 1 <= exact <= 1000000:
        raise ArithmeticError("locked power outside fixture bound")
    return round_f32(1 / exact)


def _q18(word):
    exact = f32_dyadic(word)
    if not -1 <= exact <= 1:
        raise ArithmeticError("locked coefficient outside [-1, 1]")
    return int(exact * (1 << 18))  # signed truncation toward zero


def _row(p, j, power, inverse, constants):
    angle = round_f32(p * f32_dyadic(inverse))
    (cosine, sine), trace = _trig(angle, constants)
    return LockedTrigRow(p, j, power, inverse, angle, cosine, sine,
                         _q18(cosine), _q18(sine)), trace


def locked_trig_pair(p, j):
    """Predict one row; reject other coordinates before doing arithmetic."""
    if type(p) is not int or type(j) is not int or not (0 <= p <= 4 and 0 <= j < 32):
        raise ValueError("exact non-bool integer p in [0, 4] and j in [0, 31] required")
    power = _locked_power(j)
    return _row(p, j, power, _inverse(power), _constants())[0]


def _grid():
    powers = tuple(_locked_power(j) for j in range(32))
    inverses = tuple(_inverse(power) for power in powers)
    constants = _constants()
    for p in range(5):
        for j in range(32):
            yield _row(p, j, powers[j], inverses[j], constants)


def locked_trig_pairs():
    """Yield all Rust-literal rows, p-major/j-minor, cosine before sine."""
    for row, _ in _grid():
        yield row


def _word_digest(rows):
    words = (word for row in rows for word in (row.cosine, row.sine))
    return sha256(b"".join(word.to_bytes(4, "big") for word in words)).hexdigest()


def _ideal_differences(rows):
    """Compare to the separate mathematical grid, not trig on locked angles.

    Difference tuple: (p, j, function, locked word, ideal word,
    locked signed Q18, ideal signed Q18). Order matches fixture serialization.
    """
    words, integers = [], []
    for row, ideal in zip(rows, reference_pairs(1000000, 64, 4), strict=True):
        assert row[:2] == ideal[:2]
        for name, word, ideal_word, q18, ideal_q18 in (
            ("cos", row.cosine, ideal[6], row.cos_q18, ideal[8]),
            ("sin", row.sine, ideal[7], row.sin_q18, ideal[9]),
        ):
            witness = (*row[:2], name, word, ideal_word, q18, ideal_q18)
            if word != ideal_word:
                words.append(witness)
            if q18 != ideal_q18:
                integers.append(witness)
    return words, integers


def test_locked_trig_source_literals():
    c, d = _constants()
    for name, decimal, significand, shift in _KERNEL_LITERALS:
        dyadic = significand * _pow2(shift)
        assert c[name].value == dyadic, name
        # Independently check decimal lies strictly between neighboring
        # midpoints; these annotations have neither binade-edge significand.
        assert (1 << 52) < abs(significand) < (1 << 53) - 1
        assert abs(Fraction(decimal) - dyadic) < _pow2(shift - 1), name
    assert tuple(x.word for x in d) == (
        0x3FF921FB54442D18, 0x400921FB54442D18, 0x4012D97C7F3321D2,
    )
    assert d[0].value == Fraction(0x1921FB54442D18, 1 << 52)
    assert abs(Fraction(_PI_OVER_TWO) - d[0].value) < Fraction(1, 1 << 53)
    assert d[1].value == 2 * d[0].value and d[2].value == 3 * d[0].value


def test_locked_trig_binary64_rounding_and_cast():
    one, zero, minus_zero = _F64.rational(1), _F64(0), _F64(_SIGN64)
    # Halfway above 1: even lower, then even upper; repeat with negatives.
    unit = Fraction(1, 1 << 53)
    for exact, word in ((1 + unit, 0x3FF0000000000000),
                        (1 + 3 * unit, 0x3FF0000000000002),
                        (2 - unit, 0x4000000000000000),  # carry into next binade
                        (2 - 3 * unit, 0x3FFFFFFFFFFFFFFE)):
        assert _F64.rational(exact).word == word
        assert _F64.rational(-exact).word == word | _SIGN64
    assert (one + _F64.rational(unit)).word == one.word
    assert (one - one).word == 0
    assert (minus_zero + minus_zero).word == _SIGN64
    assert (minus_zero + zero).word == 0
    assert (minus_zero - zero).word == _SIGN64
    assert (zero * -one).word == _SIGN64
    assert (minus_zero * -one).word == 0
    assert minus_zero.to_f32() == 0x80000000
    assert _F64.rational(-1 - Fraction(1, 1 << 24)).to_f32() == 0xBF800000
    assert _F64.rational(1 + Fraction(3, 1 << 24)).to_f32() == 0x3F800002
    assert _F64.rational(Fraction(1, 1 << 256)).word == 0x2FF0000000000000


def test_locked_trig_finite_refusals():
    for p, j in ((True, 0), (0, False), (-1, 0), (5, 0), (0, -1), (0, 32),
                 (Fraction(1), 0), (0, "9"), (None, 0), (1 << 100, 0)):
        with pytest.raises(ValueError, match="exact non-bool integer"):
            locked_trig_pair(p, j)
    for angle in (-1, True, 0x40800001, 0x7F800000, 0x7FC00000, 0x80000000):
        # Invalid input rejected before even consulting the constants.
        with pytest.raises(ValueError, match="positive-sign binary32"):
            _trig(angle, None)
    for word in (1, 0x7FF0000000000000, 0x7FF8000000000000):
        with pytest.raises(ArithmeticError, match="binade"):
            _F64(word)
    for exact in (Fraction(1, 1 << 257), 512, 512 - Fraction(1, 1 << 45)):
        with pytest.raises(ArithmeticError, match="binade"):
            _F64.rational(exact)
    with pytest.raises(ArithmeticError, match="bit bound"):
        _F64.rational(Fraction((1 << 512) + 1, 1 << 512))
    with pytest.raises(ValueError, match="exact rational"):
        _F64.rational("1")


def test_locked_trig_literal_stage_anchors():
    grid = dict(((row.p, row.j), (row, trace)) for row, trace in _grid())

    # Literal f64 stage witnesses checked by independent midpoint bisection
    # of exact dyadics, not observed bits or a mathematical trig calculation.
    # j=0 starts at an exact power/inverse of 1. j=16 starts at exact 1000.
    anchors = (
        ((1, 16), 0x3A83126F, "direct",
         (0x3F50624DE0000000, 0x3F50624DE0000000),
         (0x3FEFFFFEF39085D9, 0x3F50624DB22FDFFA),
         (0x3F7FFFF8, 0x3A83126E, 262143, 262)),
        ((1, 0), 0x3F800000, "D1",
         (0x3FE243F6A8885A30, 0xBFE243F6A8885A30),
         (0x3FE14A280FB52CBE, 0x3FEAED548F0BEC20),
         (0x3F0A5140, 0x3F576AA4, 141637, 220586)),
        ((2, 0), 0x40000000, "D1",
         (0xBFDB7812AEEF4BA0, 0x3FDB7812AEEF4BA0),
         (0xBFDAA2265752E594, 0x3FED18F6EACAB02F),
         (0xBED51133, 0x3F68C7B7, -109090, 238366)),
        ((3, 0), 0x40400000, "D2",
         (0xBFC21FB54442D180, 0x3FC21FB54442D180),
         (0x3FEFAE04BE8B4DEA, 0x3FC210386DB71CC0),
         (0xBF7D7026, 0x3E1081C3, -259520, 36993)),
        ((4, 0), 0x40800000, "D3",
         (0xBFE6CBE3F9990E90, 0xBFE6CBE3F9990E90),
         (0xBFE4EAA606DAF23A, 0x3FE837B9DDD913BA),
         (0xBF275530, 0xBF41BDCF, -171348, -198391)),
    )
    for key, angle, branch, inputs, polynomial, outputs in anchors:
        row, trace = grid[key]
        assert row.angle == angle, key
        assert trace == (branch, inputs, polynomial), key
        assert row[5:] == outputs, key
        assert locked_trig_pair(*key) == row
    assert grid[1, 16][0][2:4] == (0x447A0000, 0x3A83126F)
    assert grid[1, 9][0][2:] == (
        0x4242C97A, 0x3CA8398A, 0x3CA8398A, 0x3F7FF22F, 0x3CA83683, 262088, 5382,
    )
    tiny, trace = grid[1, 20]
    assert tiny[4:] == (0x393A7753, 0x3F800000, 0x393A7753, 262144, 46)
    assert trace == ("tiny", (), ())
    # D3 subtraction is exactly representable: subtract integer numerators
    # on D3's annotated 2**-50 lattice before decoding the witness word.
    assert _F64(0xBFE6CBE3F9990E90).value == Fraction(
        (4 << 50) - 0x12D97C7F3321D2, 1 << 50,
    )
    assert f32_dyadic(0x3F0A5140) == Fraction(141637, 1 << 18)
    row, _ = grid[3, 3]
    assert row.angle == 0x3F524F86
    assert round_f32(3 / f32_dyadic(row.power)) == 0x3F524F85


def test_locked_trig_complete_frozen_grid():
    rows = tuple(locked_trig_pairs())
    assert len(rows) == 160
    assert tuple(row[:2] for row in rows) == tuple((p, j) for p in range(5) for j in range(32))
    assert all(row[4:] == (0, 0x3F800000, 0, 262144, 0) for row in rows[:32])
    assert sum(len(row[5:7]) for row in rows) == 320
    assert Counter(trace.branch for _, trace in _grid()) == {
        "tiny": 74, "direct": 74, "D1": 9, "D2": 2, "D3": 1,
    }
    # Frozen source predictions: p-major/j-minor, big-endian u32 cos then sin.
    assert _word_digest(rows) == "03385e151098598cfa2914609194cb1c0c959939f15d524ec8fc7175864f53b8"


def test_locked_trig_independent_ideal_comparison():
    words, integers = _ideal_differences(locked_trig_pairs())
    # Distinct questions: a word mismatch is not masked by Q18 agreement.
    assert (len(words), words[0] if words else None) == (
        11, (1, 9, "sin", 0x3CA83683, 0x3CA83684, 5382, 5382),
    )
    assert Counter(witness[2] for witness in words) == {"sin": 11}
    assert (len(integers), integers[0] if integers else None) == (0, None)


def test_locked_trig_ideal_comparison_reports_q18_with_matching_words():
    rows = list(locked_trig_pairs())
    assert (len(rows), rows[0].p, rows[0].j, rows[0].sin_q18) == (160, 0, 0, 0)
    rows[0] = rows[0]._replace(sin_q18=1)
    words, integers = _ideal_differences(rows)
    assert len(words) == 11
    assert integers == [(0, 0, "sin", 0, 0, 1, 0)]
