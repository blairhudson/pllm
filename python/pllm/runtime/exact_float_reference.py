"""Bounded exact scalar building blocks for a clear numerical correctness oracle.

No host float, libm, Decimal convergence, model/native import or source I/O.
Returned words are IEEE RNE results, established by exact comparisons or proved
enclosures wholly inside one rounding cell, NEVER stabilized approximants.
Work limits can produce Unresolved; they do not change the numerical function.

This is not the eight-ID semantic decoder, an owner-admitted protected executor,
an online private-value failure policy, or a source/no-wrap/security certificate.
The original semantic/source owners and public-matrix rules remain independent.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F


class Unresolved(ArithmeticError):
    """Finite implementation-work budget did not prove a unique rounded word."""


class NonFinite(ArithmeticError):
    """Non-mask input/result is nonfinite; no saturation/fallback is permitted."""


class Domain(ValueError):
    """Malformed word or undefined real scalar input; not online admission."""


@dataclass(frozen=True)
class Limits:
    terms: int = 64
    dyadic_bits: int = 192

    def __post_init__(self):
        if type(self.terms) is not int or not 1 <= self.terms <= 128:
            raise Domain("terms must be an integer in [1,128]")
        if type(self.dyadic_bits) is not int or not 16 <= self.dyadic_bits <= 256:
            raise Domain("dyadic_bits must be an integer in [16,256]")


def _limits(limits):
    if type(limits) is not Limits:
        raise Domain("explicit scalar work limits required")
    limits.__post_init__()


def _word(word, width):
    if type(word) is not int or not 0 <= word < 1 << width:
        raise Domain("noncanonical IEEE word")


def _power(exponent):
    return F(2)**exponent


def _decode(word, fraction_bits):
    _word(word, fraction_bits+9)
    exponent = (word >> fraction_bits) & 255
    if exponent == 255:
        raise NonFinite("nonfinite input")
    fraction = word & ((1 << fraction_bits)-1)
    coefficient = fraction if exponent == 0 else (1 << fraction_bits)+fraction
    value = coefficient*_power((1 if exponent == 0 else exponent)-127-fraction_bits)
    return -value if word >> (fraction_bits+8) else value


def decode_f32(word):
    return _decode(word, 23)


def _nearest_integer(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + (2*remainder > value.denominator or
                       (2*remainder == value.denominator and quotient % 2 != 0))


def round_f32(value, *, negative_zero=False):
    """Exact rational -> F32 RNE, gradual underflow. Zero sign is explicit."""
    if type(value) not in (int, F) or type(negative_zero) is not bool:
        raise Domain("exact integer/Fraction and Boolean zero sign required")
    value = F(value)
    if value and negative_zero:
        raise Domain("zero sign supplied for nonzero input")
    sign = int(value < 0 or negative_zero) << 31
    magnitude = abs(value)
    if not magnitude:
        return sign
    exponent = magnitude.numerator.bit_length()-magnitude.denominator.bit_length()
    if magnitude < _power(exponent):
        exponent -= 1
    quantum = max(exponent-23, -149)
    coefficient = _nearest_integer(magnitude/_power(quantum))
    if coefficient == 0:
        return sign
    if exponent < -126:
        return sign | coefficient  # includes carry into smallest normal
    if coefficient == 1 << 24:
        coefficient >>= 1
        exponent += 1
    if exponent > 127:
        raise NonFinite("rounded result overflows F32")
    return sign | ((exponent+127) << 23) | (coefficient-(1 << 23))


def _same_cell(lower, upper, *, negative=False):
    """Inclusion proof: both CLOSED interval endpoints round to identical bits."""
    if lower > upper:
        raise AssertionError("reversed rigorous enclosure")
    overflow = _power(128)-_power(103)
    if lower >= overflow or upper <= -overflow:
        raise NonFinite("entire real enclosure rounds nonfinite")
    try:
        lo = round_f32(lower, negative_zero=negative and lower == 0)
        hi = round_f32(upper, negative_zero=negative and upper == 0)
    except NonFinite:
        return None
    return lo if lo == hi else None


def _root(target, degree):
    """RNE of positive target**(1/degree): exact midpoint-power comparisons."""
    low, high = 0, 0x7f800000
    while low+1 < high:
        middle = (low+high)//2
        if decode_f32(middle)**degree <= target:
            low = middle
        else:
            high = middle
    if high == 0x7f800000:
        midpoint = _power(128)-_power(103)
    else:
        midpoint = (decode_f32(low)+decode_f32(high))/2
    comparison = midpoint**degree
    selected = high if target > comparison or (target == comparison and low % 2) else low
    if selected == 0x7f800000:
        raise NonFinite("root rounds nonfinite")
    return selected


def sqrt_f32(word):
    value = decode_f32(word)
    if value < 0:
        raise Domain("negative square root")
    return word if value == 0 else _root(value, 2)


def rsqrt_f32(word):
    value = decode_f32(word)
    if value <= 0:
        raise Domain("reciprocal square root requires strictly positive input")
    # sqrt(1/x), NOT R32(1/R32(sqrt(x))). Exact comparison avoids double rounding.
    return _root(1/value, 2)


def pow_ratio_f32(base, numerator, denominator):
    """Exact real positive-integer base**(n/d), then ONE F32 rounding.

    Resource envelope (not a numeric-map change): <=32-bit base and n,d<=64.
    It includes every fixed-position rotary exponent i/64 without using log/exp.
    No statement about general non-rational-exponent pow is made.
    """
    if any(type(x) is not int for x in (base, numerator, denominator)):
        raise Domain("integer algebraic-power parameters required")
    if base <= 0 or numerator < 0 or denominator <= 0:
        raise Domain("undefined positive-base algebraic-power input")
    if base.bit_length() > 32 or numerator > 64 or denominator > 64:
        raise Unresolved("algebraic-power work envelope exceeded")
    return _root(F(base**numerator), denominator)


def _outward(lower, upper, bits):
    """Rational enclosure -> enclosing fixed dyadic grid, NOT a float cast."""
    unit = 1 << bits
    lo = lower.numerator*unit//lower.denominator
    hi = -((-upper.numerator*unit)//upper.denominator)
    return F(lo, unit), F(hi, unit)


def _exp_intervals(value, limits):
    """Enclose exp(value), |value|<=160, using Taylor plus geometric remainder.

    Reduce |x| to u<=1/2. For S_n=sum_{k=0}^n u^k/k!, the positive
    remainder is <= next_term/(1-u/(n+2)); every later term ratio is smaller.
    Repeated squaring uses outward dyadic rounding at EVERY step. For negative
    x, invert positive endpoints in reverse order. No float approximation occurs.
    """
    if abs(value) > 160:
        raise Unresolved("exponential enclosure work envelope exceeded")
    u, squares = abs(value), 0
    while u > F(1, 2):
        u /= 2
        squares += 1
    total, term = F(1), F(1)
    for n in range(1, limits.terms+1):
        term *= u/n
        total += term
        following = term*u/(n+1)
        lower, upper = _outward(total, total+following/(1-u/(n+2)), limits.dyadic_bits)
        for _ in range(squares):
            lower, upper = _outward(lower*lower, upper*upper, limits.dyadic_bits)
        if value < 0:
            lower, upper = 1/upper, 1/lower
        yield lower, upper


def exp_f32(word, limits=Limits()):
    _limits(limits)
    value = decode_f32(word)
    if value <= -150:
        # e>2, hence exp(-150)<2^-150: strictly below half min F32 subnormal.
        return 0
    if value >= 128:
        # e>2, exp(128)>2^128: above the F32 overflow midpoint.
        raise NonFinite("proved exponential overflow")
    for lower, upper in _exp_intervals(value, limits):
        result = _same_cell(lower, upper)
        if result is not None:
            return result
    raise Unresolved("exponential enclosure straddles a rounding boundary")


def silu_f32(bf16_word, limits=Limits()):
    """ONE R32(exact_real_silu(BF16)); no rounded intermediate exponential.

    Caller applies the selected BF16 cast AFTER this F32 result. Do not replace
    this with direct exact-real-to-BF16 rounding or a stock overflowing exp(-x).
    """
    _limits(limits)
    value = _decode(bf16_word, 7)
    if not value:
        return bf16_word << 16
    negative, a = value < 0, abs(value)
    if a >= 160:
        # a*exp(-a) decreases for a>=1; 160*exp(-160)<2^-152.
        # This bounds BOTH the positive correction and negative result magnitude.
        lower, upper = (-_power(-152), F(0)) if negative else (a-_power(-152), a)
        result = _same_cell(lower, upper, negative=negative)
        if result is not None:
            return result
        raise Unresolved("SiLU tail enclosure straddles a rounding boundary")
    for lo, hi in _exp_intervals(-a, limits):
        if negative:
            lower, upper = -a*hi/(1+hi), -a*lo/(1+lo)
        else:
            lower, upper = a/(1+hi), a/(1+lo)
        result = _same_cell(lower, upper, negative=negative)
        if result is not None:
            return result
    raise Unresolved("SiLU enclosure straddles a rounding boundary")


def sincos_f32(word, limits=Limits()):
    """Separate correctly rounded sin/cos of the SAME exact F32 angle.

    |angle|<=8 is an implementation-work bound covering positions0..7, not a new
    numeric domain. Larger arguments refuse rather than invent range reduction.
    Taylor-Lagrange bound uses |d^k sin|,|d^k cos|<=1. Including the next zero
    coefficient gives remainder <=abs(next nonzero term), for ALL such x; no
    assumption of initially decreasing alternating terms is needed.
    """
    _limits(limits)
    value = decode_f32(word)
    if abs(value) > 8:
        raise Unresolved("trigonometric work envelope exceeded")
    if not value:
        return word, 0x3f800000
    sine, cosine = value, F(1)
    sin_term, cos_term = value, F(1)
    sine_word = cosine_word = None
    for n in range(limits.terms):
        next_sin = -sin_term*value*value/((2*n+2)*(2*n+3))
        next_cos = -cos_term*value*value/((2*n+1)*(2*n+2))
        sine_word = _same_cell(sine-abs(next_sin), sine+abs(next_sin))
        cosine_word = _same_cell(cosine-abs(next_cos), cosine+abs(next_cos))
        if sine_word is not None and cosine_word is not None:
            return sine_word, cosine_word
        sin_term, cos_term = next_sin, next_cos
        sine += sin_term
        cosine += cos_term
    raise Unresolved("trigonometric enclosure straddles a rounding boundary")
