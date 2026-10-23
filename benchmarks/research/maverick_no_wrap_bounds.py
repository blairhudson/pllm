"""Conditional exact inequalities from maverick-honest-no-wrap-conditions.md.

Research mathematics only. No files, checkpoint values, SDK, model, transport or
machine floating point are used. A CERTIFIED math_status applies to supplied
parameters under the recorded assumptions, never to their source provenance or
coverage of the real graph. source_status and whole_graph_status stay UNRESOLVED.
Failure of a sufficient bound is not a counterexample for an uninspected source.
Fractions forget signed zero: these are value/bound checks, not bit-parity proofs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from math import isqrt

from benchmarks.research.maverick_totality_bounds import (
    H,
    I64_MAX,
    I64_MIN,
    P,
    nearest_integer,
    power,
    round_bound,
)


D = 256 * 256
U32, ETA32, U16, ETA16 = power(-24), power(-150), power(-8), power(-134)
B, R = 1 + U32, 1 + U16
K, E = B * R, R * ETA32 + ETA16


class Status(str, Enum):
    CERTIFIED = "certified"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class CastParameters:
    value: Fraction
    precision: int


@dataclass(frozen=True)
class Q8Parameters:
    value: Fraction


@dataclass(frozen=True)
class RMSParameters:
    dimension: int
    gamma_max: Fraction | None
    input_l2: Fraction | None


@dataclass(frozen=True)
class RowParameters:
    row: tuple[int, ...]
    quantized_input_l2: Fraction | None


@dataclass(frozen=True)
class RealRowParameters:
    row: tuple[int, ...]
    input_l2: Fraction | None


@dataclass(frozen=True)
class JointParameters:
    down: tuple[int, ...]
    gate: tuple[tuple[int, ...], ...]
    up: tuple[tuple[int, ...], ...]
    quantized_input_l2: Fraction | None
    tau: tuple[Fraction, ...]


@dataclass(frozen=True)
class DotParameters:
    row: tuple[int, ...]
    quantized_input: tuple[int, ...]


Parameters = (
    CastParameters
    | Q8Parameters
    | RMSParameters
    | RowParameters
    | RealRowParameters
    | JointParameters
    | DotParameters
)
Fact = Fraction | int | None | tuple[int, ...]


@dataclass(frozen=True)
class ConditionalCertificate:
    math_status: Status
    parameters: Parameters
    facts: tuple[tuple[str, Fact], ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_reasons: tuple[str, ...] = ()
    source_status: Status = field(default=Status.UNRESOLVED, init=False)
    whole_graph_status: Status = field(default=Status.UNRESOLVED, init=False)


def _exact(value: int | Fraction) -> Fraction:
    if type(value) not in (int, Fraction):
        raise TypeError("only exact int/Fraction parameters are supported")
    return Fraction(value)


def _integer(value: int) -> int:
    if type(value) is not int:
        raise TypeError("integer parameters must be int, not rounded/coerced values")
    return value


def _bound(value: int | Fraction | None) -> Fraction | None:
    return None if value is None else _exact(value)


def _row(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(_integer(value) for value in values)


def _fits(values: Iterable[int]) -> bool:
    return all(I64_MIN <= value <= I64_MAX for value in values)


def _result(
    parameters: Parameters,
    facts: Iterable[tuple[str, Fact]] = (),
    assumptions: Iterable[str] = (),
    reasons: Iterable[str] = (),
) -> ConditionalCertificate:
    reasons = tuple(reasons)
    return ConditionalCertificate(
        Status.UNRESOLVED if reasons else Status.CERTIFIED,
        parameters,
        tuple(facts),
        tuple(assumptions),
        tuple(reasons),
    )


def check_cast(value: int | Fraction, precision: int) -> ConditionalCertificate:
    """One scalar rounding obligation, not a tensor import or numeric fallback."""
    params = CastParameters(_exact(value), _integer(precision))
    if precision not in (8, 24):
        return _result(params, reasons=("precision_not_in_selected_map",))
    rounded = round_bound(params.value, precision)
    return _result(
        params,
        (("rounded", rounded),),
        reasons=() if rounded is not None else ("nonfinite_rounding",),
    )


def check_input_q8(value: int | Fraction) -> ConditionalCertificate:
    """Require an existing BF16 value; exact 256*x then I, with NO F32 cast."""
    params = Q8Parameters(_exact(value))
    bf16 = round_bound(params.value, 8)
    if bf16 != params.value:
        return _result(params, (("bf16_cast", bf16),), reasons=("input_not_finite_bf16",))
    quantized = nearest_integer(256 * params.value)
    return _result(
        params,
        (("quantized", quantized),),
        reasons=() if _fits((quantized,)) else ("input_q8_outside_i64",),
    )


def check_rms_bound(
    dimension: int, gamma_max: int | Fraction | None, input_l2: int | Fraction | None
) -> ConditionalCertificate:
    """Note (N,Q): rounded RMS bound, conditional on F and supplied norm bounds.

    A Q bound too large to prove every coordinate i64-safe is UNRESOLVED;
    it is not evidence that the actual graph overflows.
    """
    params = RMSParameters(_integer(dimension), _bound(gamma_max), _bound(input_l2))
    assumptions = (
        "F: finite local selected arithmetic",
        "input_l2 bounds actual norm input",
        "gamma_max bounds every actual learned coefficient",
    )
    if params.gamma_max is None or params.input_l2 is None:
        return _result(params, assumptions=assumptions, reasons=("supplied_norm_bound_unknown",))
    if dimension not in (128, 2560) or min(params.gamma_max, params.input_l2) < 0:
        return _result(params, assumptions=assumptions, reasons=("unsupported_norm_bounds",))
    t = Fraction(12 if dimension == 128 else 51)
    c = 1 - (dimension + 3) * U32
    eps = round_bound(Fraction(1, 1_000_000), 24)
    if eps is None or c <= 0 or t * t * c < dimension:
        return _result(params, assumptions=assumptions, reasons=("rms_lemma_not_established",))
    eps0 = (1 - U32) * eps - 4 * ETA32
    if eps0 <= 0:
        return _result(params, assumptions=assumptions, reasons=("epsilon_margin_not_positive",))
    norm = (
        K * K * params.gamma_max * (B * t + ETA32 * params.input_l2)
        + (K * params.gamma_max + 1) * E * t
    )
    q = 256 * norm + t / 2
    return _result(
        params,
        (("c", c), ("epsilon_margin", eps0), ("norm_l2_bound", norm), ("quantized_l2_bound", q)),
        assumptions,
        () if q <= I64_MAX else ("bound_does_not_prove_i64",),
    )


def check_row_l2(
    row: Iterable[int], quantized_input_l2: int | Fraction | None
) -> ConditionalCertificate:
    """(R1), using actual supplied integer coefficients, never a float row norm."""
    params = RowParameters(_row(row), _bound(quantized_input_l2))
    assumptions = (
        "input is an exact integer Q8 vector with norm at most supplied Q",
        "F: actual input encoding and selected arithmetic are valid",
    )
    if params.quantized_input_l2 is None:
        return _result(params, assumptions=assumptions, reasons=("supplied_input_bound_unknown",))
    if not params.row or params.quantized_input_l2 < 0 or not _fits(params.row):
        return _result(params, assumptions=assumptions, reasons=("invalid_row_bounds",))
    row_sq = sum(value * value for value in params.row)
    lhs = row_sq * params.quantized_input_l2**2
    integer_bound = output_bound = None
    if lhs <= H * H:
        # Exact rational ceiling sqrt, satisfying K_i^2 >= lhs in the note.
        root = isqrt(lhs.numerator // lhs.denominator)
        integer_bound = root + (root * root < lhs)
        output_bound = round_bound(Fraction(integer_bound, D), 8)
    return _result(
        params,
        (
            ("row_l2_squared", row_sq),
            ("dot_bound_squared", lhs),
            ("margin_squared", H * H - lhs),
            ("integer_bound", integer_bound),
            ("direct_bf16_bound", output_bound),
        ),
        assumptions,
        () if lhs <= H * H else ("sufficient_row_bound_not_met",),
    )


def check_row_real_l2(
    row: Iterable[int], input_l2: int | Fraction | None
) -> ConditionalCertificate:
    """(R2): independently retain the per-coordinate 1/2 Q8 error budget."""
    params = RealRowParameters(_row(row), _bound(input_l2))
    assumptions = (
        "actual BF16 input norm is bounded by supplied Z",
        "F: selected exact integer Q8 encoding is valid",
    )
    if params.input_l2 is None:
        return _result(params, assumptions=assumptions, reasons=("supplied_input_bound_unknown",))
    if not params.row or params.input_l2 < 0 or not _fits(params.row):
        return _result(params, assumptions=assumptions, reasons=("invalid_row_bounds",))
    row_sq = sum(value * value for value in params.row)
    l1 = sum(abs(value) for value in params.row)
    budget = H - Fraction(l1, 2)
    lhs = 256 * 256 * row_sq * params.input_l2**2
    reasons = []
    if budget < 0:
        reasons.append("quantization_error_budget_exceeds_H")
    elif lhs > budget * budget:
        reasons.append("sufficient_real_input_bound_not_met")
    return _result(
        params,
        (
            ("row_l2_squared", row_sq),
            ("row_l1", l1),
            ("remaining_budget", budget),
            ("scaled_bound_squared", lhs),
        ),
        assumptions,
        reasons,
    )


def check_joint_gate_up(
    down: Iterable[int],
    gate: Iterable[Iterable[int]],
    up: Iterable[Iterable[int]],
    quantized_input_l2: int | Fraction | None,
    tau: Iterable[int | Fraction],
) -> ConditionalCertificate:
    """Check (DWN). Derive kappa from rows, without trusting a spectral estimate.

    Trace and a triangle/Gershgorin upper bound certify the Young matrix's
    spectral bound. Computation is linear in the supplied coefficient count;
    no Gram matrix, eigensolver, runtime activation or source loading is used.
    """
    params = JointParameters(
        _row(down),
        tuple(_row(row) for row in gate),
        tuple(_row(row) for row in up),
        _bound(quantized_input_l2),
        tuple(_exact(value) for value in tau),
    )
    assumptions = (
        "gate/up use the SAME honest exact Q8 vector bounded by supplied Q",
        "F: finite selected local arithmetic and i64 encodings",
        "supplied gate/up matrices are the actual honest selected projection roles",
    )
    if params.quantized_input_l2 is None:
        return _result(params, assumptions=assumptions, reasons=("supplied_input_bound_unknown",))
    m = len(params.down)
    if (
        not m
        or len(params.gate) != m
        or len(params.up) != m
        or len(params.tau) != m
        or params.quantized_input_l2 < 0
        or any(value <= 0 for value in params.tau)
    ):
        return _result(params, assumptions=assumptions, reasons=("invalid_joint_bounds",))
    n = len(params.gate[0])
    if (
        not n
        or any(len(row) != n for row in (*params.gate, *params.up))
        or not _fits(params.down)
        or any(not _fits(row) for row in (*params.gate, *params.up))
    ):
        return _result(params, assumptions=assumptions, reasons=("invalid_joint_rows",))
    # Discharge the asserted gate/up no-wrap premise using the SAME Q, not booleans.
    q_sq = params.quantized_input_l2**2
    gate_up_margin = min(
        H * H - sum(x * x for x in row) * q_sq for row in (*params.gate, *params.up)
    )
    trace = Fraction(0)
    row_sums = [Fraction(0) for _ in range(n)]
    for weight, g, u, t in zip(params.down, params.gate, params.up, params.tau):
        for vector, factor in ((g, abs(weight) * t / 2), (u, abs(weight) / (2 * t))):
            trace += factor * sum(x * x for x in vector)
            l1 = sum(abs(x) for x in vector)
            for j, x in enumerate(vector):
                row_sums[j] += factor * abs(x) * l1
    row_sum_bound = max(row_sums)
    kappa = min(trace, row_sum_bound)
    l1_down = sum(abs(value) for value in params.down)
    # r^2: direct projection casts; K,E: selected product's two roundings;
    # 1/2: final EXACT integer Q8 rounding. D is not an F32 cast.
    lhs = Fraction(256, D * D) * K * R * R * kappa * q_sq + (256 * E + Fraction(1, 2)) * l1_down
    reasons = []
    if gate_up_margin < 0:
        reasons.append("gate_up_no_wrap_not_established")
    if lhs > H:
        reasons.append("sufficient_down_bound_not_met")
    return _result(
        params,
        (
            ("kappa_trace", trace),
            ("gate_up_min_margin_squared", gate_up_margin),
            ("kappa_row_sum", row_sum_bound),
            ("kappa", kappa),
            ("down_l1", l1_down),
            ("dot_bound", lhs),
            ("margin", H - lhs),
        ),
        assumptions,
        reasons,
    )


def centered_integer(value: int) -> int:
    """Selected exact modular map; p/2 is NOT rounded or compared as a float."""
    residue = _integer(value) % P
    return residue if residue <= H else residue - P


def check_integer_dot(row: Iterable[int], quantized_input: Iterable[int]) -> ConditionalCertificate:
    """A supplied integer witness: final no-wrap, not host partial-sum safety.

    This is not a linear runtime/model oracle. Arbitrary-precision partials only
    show why the selected reduced field evaluator must not use an unreduced i64
    accumulator. A witness says nothing about all actual source graph inputs.
    """
    params = DotParameters(_row(row), _row(quantized_input))
    if (
        not params.row
        or len(params.row) != len(params.quantized_input)
        or not _fits(params.row)
        or not _fits(params.quantized_input)
    ):
        return _result(params, reasons=("invalid_integer_witness",))
    partials = [0]
    for weight, value in zip(params.row, params.quantized_input):
        partials.append(partials[-1] + weight * value)
    total = partials[-1]
    no_wrap = -H <= total <= H
    return _result(
        params,
        (
            ("partials", tuple(partials)),
            ("integer_dot", total),
            ("residue", total % P),
            ("center", centered_integer(total)),
        ),
        ("supplied witness only, not a universal graph/input certificate",),
        () if no_wrap else ("supplied_dot_wraps",),
    )
