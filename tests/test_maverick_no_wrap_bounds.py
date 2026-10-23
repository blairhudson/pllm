"""Small exact math fixtures, never trained/checkpoint evidence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
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
                "pllm",
                "numpy",
                "torch",
                "transformers",
                "safetensors",
            }:
                raise AssertionError(
                    f"mathematical checker imported runtime dependency: {fullname}"
                )
            return None

    guard = NoRuntimeImports()
    root = str(Path(__file__).resolve().parents[1])
    sys.meta_path.insert(0, guard)
    sys.path.insert(0, root)
    try:
        module = importlib.import_module("benchmarks.research.maverick_no_wrap_bounds")
        assert Path(module.__file__).resolve().parents[2] == Path(root)
        yield module
    finally:
        sys.path.remove(root)
        sys.meta_path.remove(guard)


def facts(check):
    return dict(check.facts)


@pytest.mark.parametrize("value,expected", [(F(1, 512), 0), (F(3, 512), 2), (F(-3, 512), -2)])
def test_q8_exact_ties_and_no_extra_cast(bounds, value, expected):
    result = bounds.check_input_q8(value)
    assert result.math_status is bounds.Status.CERTIFIED
    assert facts(result)["quantized"] == expected
    assert result.parameters.value == value


@pytest.mark.parametrize("value", [0.1, True, "1/2"])
def test_inexact_or_coerced_parameters_are_rejected(bounds, value):
    with pytest.raises(TypeError):
        bounds.check_input_q8(value)
    with pytest.raises(TypeError):
        bounds.check_row_l2([1], value)


def test_non_bf16_input_is_not_silently_rounded(bounds):
    result = bounds.check_input_q8(1 + F(1, 256))
    assert result.math_status is bounds.Status.UNRESOLVED
    assert result.parameters.value == 1 + F(1, 256)
    assert facts(result)["bf16_cast"] == 1


def test_source_overflow_counterexamples_and_signed_i64_edge(bounds):
    assert bounds.check_cast(2**64, 8).math_status is bounds.Status.CERTIFIED
    assert bounds.check_cast(2**128, 24).math_status is bounds.Status.UNRESOLVED
    positive = bounds.check_input_q8(2**55)
    negative = bounds.check_input_q8(-(2**55))
    assert facts(positive)["quantized"] == 2**63
    assert positive.math_status is bounds.Status.UNRESOLVED
    assert facts(negative)["quantized"] == -(2**63)
    assert negative.math_status is bounds.Status.CERTIFIED
    assert bounds.check_rms_bound(2560, 2**55, 51).math_status is bounds.Status.UNRESOLVED


@pytest.mark.parametrize("dimension,t", [(128, 12), (2560, 51)])
def test_rms_bound_matches_independently_written_cast_margins(bounds, dimension, t):
    result = bounds.check_rms_bound(dimension, F(3, 2), 100)
    u, eta, v, xi = F(1, 2**24), F(1, 2**150), F(1, 2**8), F(1, 2**134)
    k, e = (1 + u) * (1 + v), (1 + v) * eta + xi
    expected = k * k * F(3, 2) * ((1 + u) * t + eta * 100) + (k * F(3, 2) + 1) * e * t
    assert result.math_status is bounds.Status.CERTIFIED
    assert facts(result)["norm_l2_bound"] == expected
    assert facts(result)["quantized_l2_bound"] == 256 * expected + F(t, 2)
    assert facts(result)["c"] * t * t >= dimension
    assert facts(result)["epsilon_margin"] > 0
    assert result.source_status is result.whole_graph_status is bounds.Status.UNRESOLVED


@pytest.mark.parametrize(
    "row,q,passes",
    [
        ([3, 4], 5, True),
        ([1], 1_006_632_960, True),
        ([1], 1_006_632_961, False),
        ([2**63], 0, False),
    ],
)
def test_l2_row_uses_exact_coefficients_and_centered_integer_boundary(bounds, row, q, passes):
    result = bounds.check_row_l2(row, q)
    assert (result.math_status is bounds.Status.CERTIFIED) is passes
    assert result.parameters.row == tuple(row)
    assert result.parameters.quantized_input_l2 == q
    if "row_l2_squared" in facts(result):
        assert facts(result)["row_l2_squared"] == sum(x * x for x in row)
        assert facts(result)["dot_bound_squared"] == sum(x * x for x in row) * q * q


def test_joint_bound_checks_gate_up_first_not_just_small_down_row(bounds):
    result = bounds.check_joint_gate_up([0], [[bounds.H + 1]], [[1]], 1, [1])
    assert facts(result)["dot_bound"] == 0
    assert result.math_status is bounds.Status.UNRESOLVED
    assert "gate_up_no_wrap_not_established" in result.unresolved_reasons


def test_joint_bound_matches_independent_young_trace_and_all_rounding_terms(bounds):
    result = bounds.check_joint_gate_up(
        [3, -2], [[1, 0], [0, 2]], [[0, 2], [1, 0]], 5, [F(1, 2), 2]
    )
    # Explicit K matrix is diagonal: [5/4, 20]. Its trace is 85/4.
    # Recompute independently instead of trusting the helper's source quantities.
    g, u, d, tau = (
        result.parameters.gate,
        result.parameters.up,
        result.parameters.down,
        result.parameters.tau,
    )
    matrix = [
        [
            sum(
                F(abs(d[j]), 2) * (tau[j] * g[j][a] * g[j][b] + u[j][a] * u[j][b] / tau[j])
                for j in range(2)
            )
            for b in range(2)
        ]
        for a in range(2)
    ]
    trace = sum(matrix[i][i] for i in range(2))
    row_sum = max(sum(abs(x) for x in row) for row in matrix)
    kappa = min(trace, row_sum)
    assert facts(result)["kappa_trace"] == trace
    assert facts(result)["kappa_row_sum"] == row_sum
    assert facts(result)["kappa"] == kappa
    unit32, unit16 = F(1, 2**24), F(1, 2**8)
    e = (1 + unit16) * F(1, 2**150) + F(1, 2**134)
    k = (1 + unit32) * (1 + unit16)
    expected = F(256, 65536**2) * k * (1 + unit16) ** 2 * kappa * 25 + (256 * e + F(1, 2)) * 5
    assert facts(result)["dot_bound"] == expected
    assert result.math_status is bounds.Status.CERTIFIED
    for x in range(-3, 4):
        for y in range(-3, 4):
            if x * x + y * y > 25:
                continue
            product = sum(
                abs(d[j]) * abs(g[j][0] * x + g[j][1] * y) * abs(u[j][0] * x + u[j][1] * y)
                for j in range(2)
            )
            assert product <= kappa * (x * x + y * y)


def test_joint_bound_includes_off_diagonal_young_terms(bounds):
    result = bounds.check_joint_gate_up([1], [[1, 1]], [[1, 1]], 2, [1])
    assert facts(result)["kappa"] == 2
    assert 4 <= facts(result)["kappa"] * 2  # z=(1,1), not just coordinate-axis witnesses


def test_row_boundary_keeps_fractional_margin_and_direct_bf16_rounding(bounds):
    at_boundary = bounds.check_row_l2([3, 4], F(bounds.H, 5))
    above_boundary = bounds.check_row_l2([3, 4], F(bounds.H, 5) + F(1, 2**100))
    assert at_boundary.math_status is bounds.Status.CERTIFIED
    assert above_boundary.math_status is bounds.Status.UNRESOLVED
    assert facts(above_boundary)["margin_squared"] < 0
    # Selected spec's double-rounding counterexample, not a checkpoint value.
    r = 16_842_753
    row_result = bounds.check_row_l2([1], r)
    assert facts(row_result)["integer_bound"] == r
    assert facts(row_result)["direct_bf16_bound"] == 258
    first_cast = bounds.check_cast(r, 24)
    extra_cast = bounds.check_cast(facts(first_cast)["rounded"] / 65536, 8)
    assert facts(extra_cast)["rounded"] == 256


def test_rational_root_is_an_upward_proof_bound_not_float_sqrt(bounds):
    result = bounds.check_row_l2([3, 4], F(1, 2))
    assert facts(result)["dot_bound_squared"] == F(25, 4)
    assert facts(result)["integer_bound"] == 3
    assert facts(result)["integer_bound"] ** 2 >= facts(result)["dot_bound_squared"]
    assert bounds.check_row_l2([1], F(bounds.P, 2)).math_status is bounds.Status.UNRESOLVED


def test_parameter_binding_copies_mutable_caller_containers(bounds):
    down, gate, up, tau = [1], [[2]], [[3]], [1]
    result = bounds.check_joint_gate_up(down, gate, up, 1, tau)
    down[0], gate[0][0], up[0][0], tau[0] = 9, 10, 11, 12
    assert result.parameters.down == (1,)
    assert result.parameters.gate == ((2,),)
    assert result.parameters.up == ((3,),)
    assert result.parameters.tau == (F(1),)


def test_real_input_row_bound_includes_integer_rounding_budget(bounds):
    z = F(2 * bounds.H - 1, 512)
    result = bounds.check_row_real_l2([1], z)
    assert result.math_status is bounds.Status.CERTIFIED
    assert facts(result)["remaining_budget"] == bounds.H - F(1, 2)
    assert facts(result)["scaled_bound_squared"] == (bounds.H - F(1, 2)) ** 2
    larger = bounds.check_row_real_l2([1], z + F(1, 2**100))
    assert larger.math_status is bounds.Status.UNRESOLVED


def test_negative_rounding_budget_cannot_be_squared_into_a_pass(bounds):
    result = bounds.check_row_real_l2([2 * bounds.H + 1], 0)
    assert result.math_status is bounds.Status.UNRESOLVED
    assert facts(result)["remaining_budget"] == -F(1, 2)
    # The actual zero witness is fine; failure of R2 is not a source counterexample.
    assert bounds.check_integer_dot([2 * bounds.H + 1], [0]).math_status is bounds.Status.CERTIFIED


def test_real_input_unknown_bound_cannot_be_certified(bounds):
    assert bounds.check_row_real_l2([1], None).math_status is bounds.Status.UNRESOLVED


@pytest.mark.parametrize("row", [[1.0], [True], [F(1)]])
def test_integer_coefficients_are_not_silently_coerced(bounds, row):
    with pytest.raises(TypeError):
        bounds.check_row_l2(row, 1)


def test_joint_failure_is_unresolved_and_bound_to_unchanged_supplied_numbers(bounds):
    result = bounds.check_joint_gate_up([bounds.H], [[1]], [[1]], 0, [1])
    assert result.math_status is bounds.Status.CERTIFIED  # only conservative integer rounding term
    larger = bounds.check_joint_gate_up([2 * bounds.H], [[1]], [[1]], 0, [1])
    assert larger.math_status is bounds.Status.UNRESOLVED  # +E margin cannot be discarded
    assert larger.parameters.down == (2 * bounds.H,)
    assert result.parameters != larger.parameters
    with pytest.raises(FrozenInstanceError):
        result.parameters.down = (0,)


@pytest.mark.parametrize("tau", [[0], [-1], []])
def test_bad_joint_proof_weights_are_not_repaired(bounds, tau):
    result = bounds.check_joint_gate_up([1], [[1]], [[1]], 1, tau)
    assert result.math_status is bounds.Status.UNRESOLVED


@pytest.mark.parametrize(
    "total,passes,center",
    [
        (1_006_632_960, True, 1_006_632_960),
        (1_006_632_961, False, -1_006_632_960),
        (-1_006_632_960, True, -1_006_632_960),
        (-1_006_632_961, False, 1_006_632_960),
    ],
)
def test_p_over_two_is_strict_integer_boundary_not_rounded_tie(bounds, total, passes, center):
    result = bounds.check_integer_dot([1], [total])
    assert (result.math_status is bounds.Status.CERTIFIED) is passes
    assert (2 * abs(total) < bounds.P) is passes
    assert facts(result)["center"] == center
    with pytest.raises(TypeError):
        bounds.centered_integer(F(bounds.P, 2))


def test_final_dot_not_partial_sum_controls_wrap_not_host_safety(bounds):
    result = bounds.check_integer_dot([2**62, -(2**62)], [4, 4])
    assert result.math_status is bounds.Status.CERTIFIED
    assert facts(result)["partials"] == (0, 2**64, 0)
    assert facts(result)["center"] == facts(result)["integer_dot"] == 0
    assert result.source_status is bounds.Status.UNRESOLVED
    wrapped = bounds.check_integer_dot([bounds.H, 1], [1, 1])
    assert wrapped.math_status is bounds.Status.UNRESOLVED
    assert facts(wrapped)["partials"] == (0, bounds.H, bounds.H + 1)
    zero_residue = bounds.check_integer_dot([bounds.P], [1])
    assert facts(zero_residue)["center"] == 0
    assert zero_residue.math_status is bounds.Status.UNRESOLVED


def test_caller_parameters_never_become_checkpoint_or_whole_graph_proof(bounds):
    results = [
        bounds.check_row_l2([1], 1),
        bounds.check_joint_gate_up([1], [[1]], [[1]], 1, [1]),
        bounds.check_rms_bound(2560, 1, 51),
        bounds.check_input_q8(1),
    ]
    for result in results:
        assert result.math_status is bounds.Status.CERTIFIED
        assert result.source_status is bounds.Status.UNRESOLVED
        assert result.whole_graph_status is bounds.Status.UNRESOLVED
        assert asdict(result)["source_status"] is bounds.Status.UNRESOLVED
        assert asdict(result)["whole_graph_status"] is bounds.Status.UNRESOLVED
        with pytest.raises((AttributeError, FrozenInstanceError)):
            result.source_status = bounds.Status.CERTIFIED
        with pytest.raises(TypeError):
            bounds.ConditionalCertificate(
                result.math_status, result.parameters, source_status=bounds.Status.CERTIFIED
            )


def test_unknown_source_bounds_stay_typed_unresolved(bounds):
    results = [
        bounds.check_row_l2([1], None),
        bounds.check_joint_gate_up([1], [[1]], [[1]], None, [1]),
        bounds.check_rms_bound(2560, None, 51),
        bounds.check_rms_bound(2560, 1, None),
    ]
    for result in results:
        assert result.math_status is bounds.Status.UNRESOLVED
        assert result.source_status is result.whole_graph_status is bounds.Status.UNRESOLVED
        assert result.unresolved_reasons
