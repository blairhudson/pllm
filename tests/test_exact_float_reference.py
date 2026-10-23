"""Pure exact tests; no package/native/model initialization or source payloads.

Root assertions use independent midpoint inequalities, exp/SiLU/trig assertions
use independent alternating-series bounds. Neither assumes precision convergence.
"""
from fractions import Fraction as F
import builtins
import importlib.util
from math import factorial
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).resolve().parents[1]/"python/pllm/runtime/exact_float_reference.py"
SPEC = importlib.util.spec_from_file_location("_tested_exact_float_reference", PATH)
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def value(word):
    """Independent F32 dyadic decoder for checking rounding CELL boundaries."""
    exponent, fraction = (word >> 23) & 255, word & 0x7fffff
    assert exponent != 255
    significand = fraction if exponent == 0 else fraction+(1 << 23)
    result = F(significand)*F(2)**(-149 if exponent == 0 else exponent-150)
    return -result if word >> 31 else result


def cell(word):
    if word & 0x7fffffff == 0:
        return (-F(2)**-150, F(0)) if word >> 31 else (F(0), F(2)**-150)
    previous, following = (word+1, word-1) if word >> 31 else (word-1, word+1)
    return (value(previous)+value(word))/2, (value(word)+value(following))/2


def contains(word, lower, upper):
    lo, hi = cell(word)
    assert lo <= lower <= upper <= hi
    if word & 1:
        assert lo < lower and upper < hi


def exp_minus_one():
    # Alternating series, not backend's positive Taylor/geometric/squaring proof.
    low = sum((F((-1)**n, factorial(n)) for n in range(32)), F(0))
    high = low+F(1, factorial(32))
    return low, high


@pytest.mark.parametrize("word", [1, 2, 0x007fffff, 0x00800000, 0x3f000000,
                                   0x3f800000, 0x40000000, 0x43000000, 0x7f7ffffe])
def test_roots_by_independent_exact_midpoint_powers(word):
    x = value(word)
    for result, target in [(oracle.sqrt_f32(word), x), (oracle.rsqrt_f32(word), 1/x)]:
        lo, hi = cell(result)
        assert lo*lo <= target <= hi*hi
        if result & 1:
            assert lo*lo < target < hi*hi


@pytest.mark.parametrize("word,expected", [(0, 0), (0x80000000, 0x80000000),
                                          (0x40800000, 0x40000000),
                                          (0x3f800000, 0x3f800000)])
def test_sqrt_exact_and_signed_zero(word, expected):
    assert oracle.sqrt_f32(word) == expected


@pytest.mark.parametrize("base,n,d,expected", [(1000000, 0, 64, 0x3f800000),
                                               (1000000, 32, 64, 0x447a0000),
                                               (1000000, 64, 64, 0x49742400),
                                               (16, 1, 2, 0x40800000),
                                               (16, 1, 4, 0x40000000)])
def test_algebraic_power_exact_anchors(base, n, d, expected):
    assert oracle.pow_ratio_f32(base, n, d) == expected


@pytest.mark.parametrize("index", [1, 7, 16, 31, 63])
def test_rotary_rational_exponents_proved_against_integer_midpoints(index):
    result = oracle.pow_ratio_f32(1000000, index, 64)
    lo, hi = cell(result)
    target = 1000000**index
    assert lo**64 <= target <= hi**64
    if result & 1:
        assert lo**64 < target < hi**64


def test_alpha_is_one_rsqrt_rounding_not_reciprocal_of_rounded_sqrt():
    assert oracle.rsqrt_f32(0x43000000) == 0x3db504f3
    # A concrete double-rounding counterexample, discovered by fixed small inputs.
    word = 0x3f000002
    direct = oracle.rsqrt_f32(word)
    bridge = oracle.round_f32(1/value(oracle.sqrt_f32(word)))
    assert direct != bridge
    assert (direct, bridge) == (0x3fb504f2, 0x3fb504f1)
    lo, hi = cell(direct)
    assert lo*lo < 1/value(word) < hi*hi


def test_exp_constants_by_independent_alternating_bounds():
    lo, hi = exp_minus_one()
    negative = oracle.exp_f32(0xbf800000)
    positive = oracle.exp_f32(0x3f800000)
    assert negative == 0x3ebc5ab2 and positive == 0x402df854
    contains(negative, lo, hi)
    contains(positive, 1/hi, 1/lo)
    assert oracle.exp_f32(0) == oracle.exp_f32(0x80000000) == 0x3f800000


def test_exp_underflow_proof_and_overflow_not_clipping():
    assert oracle.exp_f32(0xc3160000) == 0  # -150
    assert oracle.exp_f32(0xff7fffff) == 0
    with pytest.raises(oracle.NonFinite):
        oracle.exp_f32(0x43000000)  # 128
    with pytest.raises(oracle.NonFinite):
        oracle.exp_f32(0x42c80000)  # 100; enclosure proves actual overflow


def test_exp_and_silu_gradual_underflow_from_independent_e_inverse_bounds():
    lo, hi = exp_minus_one()
    assert oracle.exp_f32(0xc2ce0000) == 1  # -103 -> min positive subnormal
    assert oracle.exp_f32(0xc2d00000) == 0  # -104 -> below half min subnormal
    contains(1, lo**103, hi**103)
    contains(0, lo**104, hi**104)
    silu = oracle.silu_f32(0xc2d0)  # BF16 -104 must not flush to zero
    assert silu & 0x80000000 and silu & 0x7fffffff
    contains(silu, -104*hi**104/(1+hi**104), -104*lo**104/(1+lo**104))


@pytest.mark.parametrize("bf16,negative", [(0x3f80, False), (0xbf80, True)])
def test_silu_exact_function_not_rounded_exp(bf16, negative):
    lo, hi = exp_minus_one()
    result = oracle.silu_f32(bf16)
    if negative:
        contains(result, -hi/(1+hi), -lo/(1+lo))
    else:
        contains(result, 1/(1+hi), 1/(1+lo))


@pytest.mark.parametrize("bf16,expected", [(0, 0), (0x8000, 0x80000000),
                                          (0x4320, 0x43200000),
                                          (0xc320, 0x80000000),
                                          (0x7f7f, 0x7f7f0000),
                                          (0xff7f, 0x80000000)])
def test_silu_stable_tails_and_zero_sign(bf16, expected):
    assert oracle.silu_f32(bf16) == expected


@pytest.mark.parametrize("angle,word", [(1, 0x3f800000), (-1, 0xbf800000),
                                       (2, 0x40000000), (7, 0x40e00000),
                                       (-7, 0xc0e00000)])
def test_sincos_independent_alternating_tail_enclosures(angle, word):
    sine, cosine = oracle.sincos_f32(word)
    for result, offset in [(sine, 1), (cosine, 0)]:
        first = sum((F((-1)**n*angle**(2*n+offset), factorial(2*n+offset))
                     for n in range(37)), F(0))
        following = first+F((-1)**37*angle**(74+offset), factorial(74+offset))
        # Tail ratios after this index are <1; alternating remainder is between
        # consecutive partial sums, independent of backend's symmetric Lagrange.
        assert F(angle*angle, (75+offset)*(76+offset)) < 1
        contains(result, min(first, following), max(first, following))
    if angle == 1:
        assert (sine, cosine) == (0x3f576aa4, 0x3f0a5140)


def test_sincos_signed_zero_is_exact():
    assert oracle.sincos_f32(0) == (0, 0x3f800000)
    assert oracle.sincos_f32(0x80000000) == (0x80000000, 0x3f800000)


@pytest.mark.parametrize("index", [0, 1, 31, 63])
@pytest.mark.parametrize("position", [0, 1, 7])
def test_selected_rotary_cast_sites_have_proved_scalar_results(index, position):
    # Scalar constants only, not a request/model runner or source admission.
    # i/64 is exactly F32: no unselected logarithm/exponential implementation.
    power = oracle.pow_ratio_f32(1000000, index, 64)
    frequency = oracle.round_f32(1/value(power))
    angle = oracle.round_f32(position*value(frequency))
    x = value(angle)
    assert 0 <= x <= position <= 7
    sine, cosine = oracle.sincos_f32(angle)
    if position == 0:
        assert (sine, cosine) == (0, 0x3f800000)
        return
    for result, offset in [(sine, 1), (cosine, 0)]:
        a = sum(((-1)**n*x**(2*n+offset)/factorial(2*n+offset)
                 for n in range(37)), F(0))
        b = a+(-1)**37*x**(74+offset)/factorial(74+offset)
        contains(result, min(a, b), max(a, b))


def test_work_exhaustion_is_typed_unresolved_never_a_stable_approximation():
    tiny = oracle.Limits(terms=1, dyadic_bits=16)
    for operation in [lambda: oracle.exp_f32(0x3f800000, tiny),
                      lambda: oracle.silu_f32(0x3f80, tiny),
                      lambda: oracle.sincos_f32(0x40e00000, tiny),
                      lambda: oracle.sincos_f32(0x41100000),
                      lambda: oracle.pow_ratio_f32(1000000, 1, 65)]:
        with pytest.raises(oracle.Unresolved):
            operation()


@pytest.mark.parametrize("operation", [lambda: oracle.sqrt_f32(0xbf800000),
                                      lambda: oracle.rsqrt_f32(0),
                                      lambda: oracle.rsqrt_f32(0x80000000),
                                      lambda: oracle.exp_f32(True),
                                      lambda: oracle.pow_ratio_f32(0, 1, 2),
                                      lambda: oracle.exp_f32(0, object()),
                                      lambda: oracle.Limits(terms=True)])
def test_invalid_scalar_or_work_inputs(operation):
    with pytest.raises(oracle.Domain):
        operation()


@pytest.mark.parametrize("operation", [lambda: oracle.sqrt_f32(0x7f800000),
                                      lambda: oracle.exp_f32(0x7fc00001),
                                      lambda: oracle.silu_f32(0x7fc1),
                                      lambda: oracle.sincos_f32(0xff800000)])
def test_nonfinite_rejects_without_host_math(operation):
    with pytest.raises(oracle.NonFinite):
        operation()


def test_r32_rational_midpoints_gradual_underflow_and_overflow():
    assert oracle.round_f32(F(1, 1 << 150)) == 0
    assert oracle.round_f32(-F(1, 1 << 150)) == 0x80000000
    assert oracle.round_f32(F(3, 1 << 150)) == 2
    assert oracle.round_f32(F(1)+F(1, 1 << 24)) == 0x3f800000
    assert oracle.round_f32(F(1)+F(3, 1 << 24)) == 0x3f800002
    assert oracle.round_f32(F(2)**128-F(2)**103-1) == 0x7f7fffff
    with pytest.raises(oracle.NonFinite):
        oracle.round_f32(F(2)**128-F(2)**103)


def test_standalone_backend_never_imports_runtime_or_model_packages(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"pllm", "torch", "transformers", "numpy", "mpmath"}:
            raise AssertionError("scalar reference attempted runtime/model import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("_isolated_exact_float_reference", PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.exp_f32(0x3f800000) == 0x402df854
    assert module.sincos_f32(0x3f800000) == (0x3f576aa4, 0x3f0a5140)
