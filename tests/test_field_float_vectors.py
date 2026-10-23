"""Independent Fraction checks of shared Rust scalar fixtures, not Rust execution.

No project numerics, research helpers, models or native module are imported.
Cargo tests must separately establish implementation/fixture conformance.
"""
from fractions import Fraction as F
import json
from pathlib import Path

import pytest


VECTORS = json.loads((Path(__file__).resolve().parents[1] /
                     "crates/pllm-core/src/field_float/vectors.json").read_text())
P = 2013265921


def binary(bits, precision, exponent_bits, bias):
    fraction_bits = precision-1
    sign = -1 if bits >> (fraction_bits+exponent_bits) else 1
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits)-1)
    fraction = bits & ((1 << fraction_bits)-1)
    if exponent == (1 << exponent_bits)-1:
        return None
    coefficient = fraction if exponent == 0 else (1 << fraction_bits)+fraction
    power = (1-bias if exponent == 0 else exponent-bias)-fraction_bits
    return sign*F(coefficient)*(F(2)**power)


def nearest(value):
    lo = value.numerator // value.denominator
    distance = value-lo
    return lo + (distance > F(1, 2) or (distance == F(1, 2) and lo % 2 != 0))


def bf16_rational(value, negative_zero=False):
    # Independent ordered-lattice search, not implementation's bit-shift cast.
    negative = value < 0 or (value == 0 and negative_zero)
    value = abs(value)
    low, high = 0, 0x7f80
    while low+1 < high:
        mid = (low+high)//2
        if binary(mid, 8, 8, 127) <= value:
            low = mid
        else:
            high = mid
    lower = binary(low, 8, 8, 127)
    upper = F(2)**128 if high == 0x7f80 else binary(high, 8, 8, 127)
    choose_high = value-lower > upper-value or (value-lower == upper-value and low % 2)
    selected = high if choose_high else low
    return None if selected == 0x7f80 else (0x8000 if negative else 0) | selected


@pytest.mark.parametrize("byte,expected", VECTORS["fp8"])
def test_fp8_fixture_independent_rationals(byte, expected):
    bits = int(byte, 16)
    exponent, fraction = (bits >> 3) & 15, bits & 7
    if exponent == 15 and fraction == 7:
        assert expected is None
        return
    value = F(fraction, 512) if exponent == 0 else (1+F(fraction, 8))*F(2)**(exponent-7)
    if bits >> 7:
        value = -value
    assert binary(int(expected, 16), 24, 8, 127) == value
    if not value:
        assert int(expected, 16) >> 31 == bits >> 7


@pytest.mark.parametrize("byte,scale,expected", VECTORS["dequant"])
def test_product_fixture_independent_rationals(byte, scale, expected):
    byte, scale = int(byte, 16), int(scale, 16)
    e, f = (byte >> 3) & 15, byte & 7
    s = binary(scale, 8, 8, 127)
    if s is None or s <= 0 or (e == 15 and f == 7):
        assert expected is None
        return
    v = F(f, 512) if e == 0 else (1+F(f, 8))*F(2)**(e-7)
    product = (-v if byte >> 7 else v)*s
    if abs(product) >= F(2)**128:
        assert expected is None
    else:
        assert binary(int(expected, 16), 24, 8, 127) == product
        if not product:
            assert int(expected, 16) >> 31 == byte >> 7


@pytest.mark.parametrize("bits,expected", VECTORS["q8"])
def test_i64_fixture_independent_rational_rounding(bits, expected):
    value = binary(int(bits, 16), 24, 8, 127)
    result = None if value is None else nearest(256*value)
    if result is not None and not -(1 << 63) <= result < (1 << 63):
        result = None
    assert result == expected


@pytest.mark.parametrize("bits,expected", VECTORS["bf16"])
def test_bf16_fixture_against_ordered_lattice(bits, expected):
    bits = int(bits, 16)
    value = binary(bits, 24, 8, 127)
    result = None if value is None else bf16_rational(value, bool(bits >> 31))
    assert result == (None if expected is None else int(expected, 16))


@pytest.mark.parametrize("residue,expected", VECTORS["reply"])
def test_direct_reply_fixture_against_ordered_lattice(residue, expected):
    if residue >= P:
        assert expected is None
        return
    centered = residue if residue <= (P-1)//2 else residue-P
    assert bf16_rational(F(centered, 65536)) == int(expected, 16)
