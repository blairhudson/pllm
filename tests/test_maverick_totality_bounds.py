"""Exact proof-endpoint regressions; no model/runtime or source-payload imports."""
from __future__ import annotations

from fractions import Fraction as F
import importlib.abc
import importlib.util
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
                raise AssertionError(f"mathematical checks imported runtime dependency: {fullname}")
            return None

    guard = NoRuntimeImports()
    sys.meta_path.insert(0, guard)
    try:
        path = Path(__file__).resolve().parents[1] / "benchmarks/research/maverick_totality_bounds.py"
        spec = importlib.util.spec_from_file_location("maverick_totality_bounds", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.meta_path.remove(guard)


@pytest.mark.parametrize("h,g", [(h, g) for h in range(22, 45) for g in range(45 - h)])
def test_all_conditional_source_envelopes(bounds, h, g):
    bounds.verify_envelope(h, g)


def test_pair_set_is_complete_and_has_no_duplicates(bounds):
    expected = {(h, g) for h in range(22, 45) for g in range(23) if h + g <= 44}
    assert len(bounds.ENVELOPES) == len(set(bounds.ENVELOPES)) == 276
    assert set(bounds.ENVELOPES) == expected


@pytest.mark.parametrize("h,g", [(21, 0), (22, -1), (44, 1), (45, 0), (22, 23)])
def test_outside_sufficient_family_is_not_a_proven_envelope(bounds, h, g):
    with pytest.raises(ValueError, match="certificate family"):
        bounds.verify_envelope(h, g)


@pytest.mark.parametrize("h", range(22, 45))
def test_all_72_residual_updates_include_both_casts_and_signs(bounds, h):
    positive, negative = bounds.residual_endpoints(h)
    R = F(1 << h)
    assert len(positive) == len(negative) == 1 + 2 * 36
    assert positive == (R,) * 73
    assert negative == (-R,) * 73
    midpoint = R + F(1 << (h - 8))
    assert bounds.round_bound(midpoint, 24) == midpoint
    assert bounds.round_bound(midpoint, 8) == R
    assert bounds.round_bound(-midpoint, 8) == -R


@pytest.mark.parametrize("s,dimension", [(0, 2560), (14, 128), (22, 2560), (44, 2560)])
def test_rms_ordered_reduction_endpoints_are_really_exact(bounds, s, dimension):
    term = F(1 << (2 * s))
    running = F(0)
    for k in range(1, dimension + 1):
        running = bounds.round_bound(running + term, 24)
        assert running == k * term
    assert bounds.round_bound(running / dimension, 24) == term


@pytest.mark.parametrize("precision", [8, 24])
def test_nearest_even_normal_ties_and_gradual_underflow(bounds, precision):
    half_ulp = F(1, 1 << precision)
    assert bounds.round_bound(1 + half_ulp, precision) == 1
    assert bounds.round_bound(1 + 3 * half_ulp, precision) == 1 + 4 * half_ulp
    assert bounds.round_bound(-1 - half_ulp, precision) == -1
    tiny = F(1, 1 << (125 + precision))
    assert bounds.round_bound(tiny, precision) == tiny
    assert bounds.round_bound(tiny / 2, precision) == 0
    assert bounds.round_bound(3 * tiny / 2, precision) == 2 * tiny
    assert bounds.round_bound(-3 * tiny / 2, precision) == -2 * tiny


@pytest.mark.parametrize("precision", [8, 24])
def test_finite_maximum_and_overflow_are_not_conflated(bounds, precision):
    ulp = F(1 << (128 - precision))
    largest = F(1 << 128) - ulp
    assert bounds.round_bound(largest, precision) == largest
    assert bounds.round_bound(largest + ulp / 4, precision) == largest
    assert bounds.round_bound(largest + ulp / 2, precision) is None
    assert bounds.round_bound(F(1 << 128), precision) is None


@pytest.mark.parametrize("value,expected", [(F(1, 2), 0), (F(3, 2), 2), (F(-1, 2), 0),
                                           (F(-3, 2), -2), (F(5, 2), 2), (F(7, 2), 4)])
def test_q8_integer_rounding_is_exact_nearest_even(bounds, value, expected):
    assert bounds.nearest_integer(value) == expected


def test_position_and_trigonometric_bounding_endpoints_without_functions(bounds):
    eps = bounds.round_bound(F(1, 1_000_000), 24)
    assert 0 < eps <= 1
    for i in range(64):
        assert bounds.round_bound(F(2 * i, 128), 24) == F(i, 64) < 1
    for t in range(8):
        assert bounds.round_bound(F(t), 24) == t <= 7
    for endpoint in (-1, 1):
        assert bounds.round_bound(F(endpoint), 24) == endpoint
        assert bounds.round_bound(F(endpoint), 8) == endpoint


def test_mlp_and_modular_endpoints_do_not_claim_honest_no_wrap(bounds):
    bounds.verify_fixed_endpoints()
    xq = bounds.nearest_integer(256 * F(235_929_600))
    assert xq > bounds.H
    assert bounds.I64_MIN < -xq < xq < bounds.I64_MAX
    assert xq % bounds.P == bounds.P - 30
    assert F(xq, 65_536) == 921_600  # hypothetical Wq=1; not a checkpoint row
    assert F(-30, 65_536) != F(xq, 65_536)


def test_excluded_finite_sources_really_expose_different_failures(bounds):
    assert bounds.round_bound(F(1 << 64), 8) == 1 << 64
    assert bounds.round_bound(F(1 << 128), 24) is None
    finite_norm = bounds.round_bound(F(1 << 55), 8)
    assert finite_norm == 1 << 55
    xq = bounds.nearest_integer(256 * finite_norm)
    assert xq == 1 << 63 and xq > bounds.I64_MAX
    assert -xq == bounds.I64_MIN  # positive endpoint, not negative endpoint, fails


def test_ordinary_output_keeps_scope_explicit(bounds, capsys):
    bounds.main()
    text = capsys.readouterr().out
    assert "Passed 276 conditional h/g envelopes" in text
    assert "checkpoint, honest no-wrap, security and admission remain unproved" in text
