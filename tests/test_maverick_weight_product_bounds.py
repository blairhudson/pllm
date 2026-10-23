"""Format mathematics only; no source payloads or model/native imports."""
from __future__ import annotations

from fractions import Fraction as F
import importlib
import importlib.abc
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def bounds():
    class NoRuntimeImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in {
                "pllm", "numpy", "torch", "transformers", "safetensors", "tokenizers",
            }:
                raise AssertionError(f"mathematical check imported runtime: {fullname}")
            return None

    guard = NoRuntimeImports()
    root = str(Path(__file__).resolve().parents[1])
    sys.meta_path.insert(0, guard)
    sys.path.insert(0, root)
    try:
        module = importlib.import_module("benchmarks.research.maverick_weight_product_bounds")
        assert Path(module.__file__).resolve() == Path(root) / "benchmarks/research/maverick_weight_product_bounds.py"
        yield module
    finally:
        sys.path.remove(root)
        sys.meta_path.remove(guard)


def test_format_partition_includes_every_finite_nonzero_positive_fp8(bounds):
    cases = bounds.WEIGHT_CLASSES
    assert len(cases) == 16
    assert cases[0] == bounds.DyadicClass(1, 7, -9)
    assert cases[1:-1] == tuple(bounds.DyadicClass(8, 15, e - 10) for e in range(1, 15))
    assert cases[-1] == bounds.DyadicClass(8, 14, 5)
    values = [m * bounds.power(c.exponent) for c in cases for m in range(c.lo, c.hi + 1)]
    assert len(values) == len(set(values)) == 126
    assert min(values) == bounds.power(-9)
    assert max(values) == 448
    assert 2 * len(values) + 2 == 254  # plus both zero signs; two NaNs excluded


def test_scale_partition_is_exactly_positive_bf16_through_existing_bound(bounds):
    cases = bounds.SCALE_CLASSES
    assert len(cases) == 173
    assert cases[0] == bounds.DyadicClass(1, 127, -133)
    assert cases[1:-1] == tuple(bounds.DyadicClass(128, 255, a - 134) for a in range(1, 172))
    assert cases[-1] == bounds.DyadicClass(128, 128, 38)
    assert sum(c.hi - c.lo + 1 for c in cases) == 22_016
    assert cases[0].lo * bounds.power(cases[0].exponent) == bounds.power(-133)
    assert cases[-1].hi * bounds.power(cases[-1].exponent) == bounds.SCALE_MAX
    assert all(c.lo > 0 for c in cases)  # no zero/negative/nonfinite scale premise


@pytest.mark.parametrize("weight_index", range(16))
def test_all_scale_exponent_classes_have_exact_f32_products(bounds, weight_index):
    weight = bounds.WEIGHT_CLASSES[weight_index]
    for scale in bounds.SCALE_CLASSES:
        proof = bounds.verify_class(weight, scale)
        assert (proof.weight, proof.scale) == (weight, scale)
        assert proof.coefficient_bits <= 12
        assert proof.quantum_exponent >= -142
        assert bounds.MIN_NONZERO_PRODUCT <= proof.minimum <= proof.maximum <= bounds.MAX_PRODUCT
        # Independent integer conditions, including subnormal F32 lattice.
        for coefficient in (weight.lo * scale.lo, weight.hi * scale.hi):
            assert coefficient.bit_length() <= 24
            units = coefficient << (proof.quantum_exponent + 149)
            value = coefficient * bounds.power(proof.quantum_exponent)
            assert units * bounds.power(-149) == value
            if value < bounds.power(-126):
                assert 0 < units < 1 << 23
        assert proof.source_status == proof.whole_graph_status == "unresolved"


def test_smallest_subnormal_product_is_exact_not_flushed(bounds):
    proof = bounds.verify_class(bounds.WEIGHT_CLASSES[0], bounds.SCALE_CLASSES[0])
    assert proof.minimum == bounds.power(-142)
    assert proof.minimum / bounds.power(-149) == 128
    assert bounds.round_bound(proof.minimum, 24) == proof.minimum != 0
    assert bounds.round_bound(-proof.minimum, 24) == -proof.minimum
    assert bounds.nearest_integer(256 * proof.minimum) == 0  # Q8 rounding still exists


def test_largest_product_q8_and_i64_bounds_are_attained(bounds):
    proof = bounds.verify_class(bounds.WEIGHT_CLASSES[-1], bounds.SCALE_CLASSES[-1])
    assert proof.maximum == 448 * bounds.power(45) == 7 * bounds.power(51)
    assert proof.q8_abs_bound == bounds.MAX_Q8_ABS == 7 * (1 << 59)
    assert bounds.round_bound(proof.maximum, 24) == proof.maximum
    assert bounds.I64_MIN < -proof.q8_abs_bound <= proof.q8_abs_bound < bounds.I64_MAX


@pytest.mark.parametrize("sign", [-1, 1])
def test_q8_ties_remain_even_despite_exact_product(bounds, sign):
    # FP8 magnitudes 1 and 3, positive BF16 scale 2**-9.
    half = sign * bounds.power(-9)
    three_halves = sign * 3 * bounds.power(-9)
    assert bounds.round_bound(half, 24) == half
    assert bounds.round_bound(three_halves, 24) == three_halves
    assert bounds.nearest_integer(256 * half) == 0
    assert bounds.nearest_integer(256 * three_halves) == sign * 2


@pytest.mark.parametrize("sign_bit", [0, 1])
def test_ieee_zero_sign_rule_is_separate_from_rational_zero(bounds, sign_bit):
    # This is the selected IEEE sign-XOR axiom, not a host-float experiment.
    positive_finite_scale_sign = 0
    product_sign = sign_bit ^ positive_finite_scale_sign
    assert product_sign << 31 == sign_bit << 31
    assert bounds.nearest_integer(F(0)) == 0


@pytest.mark.parametrize("bad_weight,bad_scale", [
    ((8, 15, 5), (128, 128, 38)),  # FP8 e=15 includes its forbidden NaN coefficient
    ((0, 7, -9), (128, 128, 38)),  # zeros belong to separate sign proof
    ((8, 14, 5), (128, 129, 38)),  # exceeds existing S bound
    ((8, 14, 5), (0, 127, -133)),  # zero scale
    ((8, 14, 5), (-127, -1, -133)),  # negative scale
    ((8, 14, 5), (128, 255, 121)),  # BF16 exponent field 255 is nonfinite
])
def test_unproved_classes_cannot_receive_a_conditional_certificate(bounds, bad_weight, bad_scale):
    with pytest.raises(ValueError, match="outside the selected"):
        bounds.verify_class(bounds.DyadicClass(*bad_weight), bounds.DyadicClass(*bad_scale))


def test_excluded_finite_sources_can_overflow_or_fail_i64(bounds):
    bounds.verify_special_cases()
    assert bounds.round_bound(bounds.power(120), 8) == bounds.power(120)
    for sign in (-1, 1):
        assert bounds.round_bound(sign * 448 * bounds.power(120), 24) is None
    too_large_q8 = bounds.nearest_integer(256 * 448 * bounds.power(47))
    assert bounds.round_bound(448 * bounds.power(47), 24) is not None
    assert too_large_q8 > bounds.I64_MAX
    assert -too_large_q8 < bounds.I64_MIN


def test_generic_f32_scale_does_not_satisfy_bf16_origin_premise(bounds):
    scale = 1 + bounds.power(-23)
    product = F(15, 8) * scale
    assert bounds.round_bound(scale, 24) == scale
    assert bounds.round_bound(scale, 8) != scale
    assert bounds.round_bound(product, 24) != product
    tiny_scale = bounds.power(-149)
    assert bounds.round_bound(tiny_scale, 24) == tiny_scale
    assert bounds.round_bound(tiny_scale, 8) == 0
    assert bounds.round_bound(bounds.power(-9) * tiny_scale, 24) == 0


@pytest.mark.parametrize("bad_weight", [(1.0, 7, -9), (True, 7, -9), (F(1), 7, -9)])
def test_class_parameters_require_exact_integers_not_equal_float_guesses(bounds, bad_weight):
    with pytest.raises(ValueError, match="outside the selected"):
        bounds.verify_class(bounds.DyadicClass(*bad_weight), bounds.SCALE_CLASSES[0])


def test_conditional_row_error_retains_q8_ties_not_product_rounding(bounds):
    # Synthetic mathematical row only, not source/checkpoint values.
    products = [
        bounds.power(-142),
        bounds.power(-9), 3 * bounds.power(-9), bounds.MAX_PRODUCT,
        -bounds.power(-9), -3 * bounds.power(-9), -bounds.MAX_PRODUCT,
    ]
    errors = []
    for value in products:
        assert bounds.round_bound(value, 24) == value
        error = bounds.nearest_integer(256 * value) - 256 * value
        assert abs(error) <= F(1, 2)
        errors.append(error)
    assert sum(error * error for error in errors) <= F(len(products), 4)
    assert F(1, 2) in errors and -F(1, 2) in errors  # no zero-error shortcut
