"""Bounded NEXUS GELU research experiment; no HE, model, or SDK execution.

Run with optional mpmath==1.3.0 installed, e.g.:
    python examples/benchmarks/nexus_gelu_numerics.py --output target/nexus.json
    python examples/benchmarks/nexus_gelu_numerics.py --follow-up --output target/nexus-gelu-follow-up.json

Printed decimals are exact rationals. MP evaluations are estimates; IV evaluations
use directed rounding, including coefficient conversion and transcendental bounds.
No fitted coefficients, clipping, runtime profile or acceptance tolerance is adopted.
Follow-up repetition counts are separately approved research candidates only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import heapq
import json
from pathlib import Path
import signal
import time


P = tuple(
    map(
        Fraction,
        (
            "-0.5054031199708174",
            "-0.42226581151983866",
            "-0.11807612951181953",
            "-0.011034134030615728",
        ),
    )
)
Q = tuple(
    map(
        Fraction,
        (
            "0.008526321541038084",
            "0.5",
            "0.3603292692789629",
            "0",
            "-0.037688200365904236",
            "0",
            "0.0018067462606141187",
        ),
    )
)
SHORT_P = tuple(
    map(
        Fraction,
        (
            "-0.5054031199708174",
            "-0.4222658115198386",
            "-0.1180761295118195",
            "-0.0110341340306157",
        ),
    )
)
SHORT_Q = tuple(
    map(
        Fraction,
        (
            "0.0085263215410380",
            "0.5",
            "0.3603292692789629",
            "0",
            "-0.037688200365904",
            "0",
            "0.0018067462606141",
        ),
    )
)
F4 = tuple(Fraction(c, 128) for c in (0, 315, 0, -420, 0, 378, 0, -180, 0, 35))
G4 = tuple(Fraction(c, 1024) for c in (0, 5850, 0, -34974, 0, 97015, 0, -113492, 0, 46623))
DOMAIN = (Fraction(-8), Fraction(8))
THRESHOLDS = (Fraction(-4), Fraction("-1.95"), Fraction(3))
BAND = Fraction(1, 2**20)
BASELINE = (2, 2)
FOLLOW_UP_SCHEDULES = (BASELINE, (6, 2), (10, 2), (10, 3))
SEPARATIONS = tuple(Fraction(1, 2**k) for k in (4, 10, 20))


def difference_scales(domain=DOMAIN, thresholds=THRESHOLDS):
    """Public difference bounds, not input-magnitude or actual-layer claims."""
    lo, hi = map(Fraction, domain)
    if lo >= hi:
        raise ValueError("domain must have positive width")
    scales = tuple(max(abs(lo - t), abs(hi - t)) for t in thresholds)
    if any(d <= 0 for d in scales):
        raise ValueError("normalizers must be positive")
    return scales


SCALES = difference_scales()


def polynomial(x, coefficients, convert=lambda x: x):
    """Ordinary-power Horner evaluation, usable with Fraction, MP, or IV."""
    if not coefficients:
        return convert(Fraction(0))
    result = convert(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        result = result * x + convert(coefficient)
    return result


def sign_stages(z, convert=lambda x: x, schedule=BASELINE):
    if schedule not in FOLLOW_UP_SCHEDULES:
        raise ValueError("only the four approved research schedules are allowed")
    stages = []
    r, s = schedule
    for coefficients in (G4,) * r + (F4,) * s:
        z = polynomial(z, coefficients, convert)
        stages.append(z)
    return tuple(stages)


def piecewise(x, convert=lambda x: x, p=P, q=Q):
    """NEXUS endpoint oracle, deliberately distinct from comparator blending."""
    if x <= convert(THRESHOLDS[0]):
        return convert(Fraction(0))
    if x <= convert(THRESHOLDS[1]):
        return polynomial(x, p, convert)
    if x <= convert(THRESHOLDS[2]):
        return polynomial(x, q, convert)
    return x


def blend(x, convert=lambda x: x, p=P, q=Q, schedule=BASELINE):
    stages = tuple(
        sign_stages((x - convert(t)) / convert(d), convert, schedule)
        for t, d in zip(THRESHOLDS, SCALES)
    )
    h = tuple((s[-1] + 1) / 2 for s in stages)
    b = (1 - h[0], h[0] - h[1], h[1] - h[2], h[2])
    px, qx = polynomial(x, p, convert), polynomial(x, q, convert)
    return {
        "stages": stages,
        "h": h,
        "b": b,
        "p": px,
        "q": qx,
        "y": b[1] * px + b[2] * qx + b[3] * x,
    }


def sample_points(grid=257):
    if not 2 <= grid <= 4097:
        raise ValueError("grid must be in [2,4097]")
    lo, hi = DOMAIN
    points = {lo + (hi - lo) * Fraction(i, grid - 1) for i in range(grid)}
    points.update(THRESHOLDS)
    for t, d in zip(THRESHOLDS, SCALES):
        for k in range(25):
            for direction in (-1, 1):
                x = t + direction * d / 2**k
                if lo <= x <= hi:
                    points.add(x)
    return sorted(points)


class Arithmetic:
    """Private contexts: no mutation of process-global mpmath precision."""

    def __init__(self, dps=80):
        if not 40 <= dps <= 200:
            raise ValueError("dps must be in [40,200]")
        try:
            import mpmath
            from mpmath.ctx_mp import MPContext
            from mpmath.ctx_iv import MPIntervalContext
        except ImportError as exc:
            raise RuntimeError("optional experiment dependency mpmath==1.3.0 is required") from exc
        if mpmath.__version__ != "1.3.0":
            raise RuntimeError("this experiment requires optional mpmath==1.3.0")
        self.mp, self.iv = MPContext(), MPIntervalContext()
        self.mp.dps = self.iv.dps = dps
        self.dps = dps
        self._cdf_cache = {}

    def point(self, value):
        value = Fraction(value)
        return self.mp.mpf(value.numerator) / value.denominator

    def exact(self, value):
        value = Fraction(value)
        return self.iv.mpf(value.numerator) / value.denominator

    def interval(self, lo, hi):
        return self.hull(self.exact(lo), self.exact(hi))

    def hull(self, *values):
        return self.iv.make_mpf(
            (
                min((v._mpi_[0] for v in values), key=self.mp.make_mpf),
                max((v._mpi_[1] for v in values), key=self.mp.make_mpf),
            )
        )

    def ends(self, value):
        return tuple(self.mp.make_mpf(x) for x in value._mpi_)

    def cdf_endpoint(self, x):
        """Enclose Phi at rational x via integrated exp Taylor plus remainder.

        |x|<=8, u=x²/2<=32. Terms decrease after n>=32. At 192 terms,
        alternating-series remainder is bounded by magnitude of next term.
        All finite sums, pi, sqrt, and remainder arithmetic round outward.
        """
        x = Fraction(x)
        if abs(x) > 8:
            raise ValueError("CDF enclosure supports research domain [-8,8] only")
        if x in self._cdf_cache:
            return self._cdf_cache[x]
        z = self.exact(abs(x))
        u = z * z / 2
        term = self.exact(1)
        total = term
        for n in range(1, 192):
            term = -term * u / n
            total += term / (2 * n + 1)
        remainder = abs(term * u / 192 / 385)
        radius = self.iv.make_mpf(((-remainder)._mpi_[0], remainder._mpi_[1]))
        half_erf = z / self.iv.sqrt(2 * self.iv.pi) * (total + radius)
        value = self.exact(Fraction(1, 2)) + (half_erf if x >= 0 else -half_erf)
        self._cdf_cache[x] = value
        return value

    def targets(self, x, lo=None, hi=None):
        if lo is None:
            m = self.mp
            return (
                x * (1 + m.tanh(m.sqrt(2 / m.pi) * (x + m.mpf("0.044715") * x**3))) / 2,
                x * (1 + m.erf(x / m.sqrt(2))) / 2,
            )
        iv = self.iv
        u = iv.sqrt(2 / iv.pi) * (x + self.exact(Fraction("0.044715")) * x**3)
        # tanh through exp: MPIntervalContext.tanh is not available in 1.3.0.
        e = iv.exp(2 * u)
        tanh = (e - 1) / (e + 1)
        cdf = self.hull(self.cdf_endpoint(lo), self.cdf_endpoint(hi))
        return x * (1 + tanh) / 2, x * cdf

    def encode(self, value):
        """Exact binary endpoints; decimal display is explicitly approximate."""
        if not all(self.mp.isfinite(v) for v in self.ends(value)):
            raise ArithmeticError("nonfinite interval; enclosure unresolved")

        def binary(raw):
            sign, mantissa, exponent, _ = raw
            return {"mantissa": str(-mantissa if sign else mantissa), "exponent2": exponent}

        return {
            "lower": binary(value._mpi_[0]),
            "upper": binary(value._mpi_[1]),
            "display_estimate": [self.mp.nstr(v, 18) for v in self.ends(value)],
        }


def _ideal_interval(a, lo, hi, threshold):
    if hi <= threshold:
        return a.exact(0)
    if lo > threshold:
        return a.exact(1)
    return a.interval(0, 1)


def cell_metrics(a, lo, hi, schedule=BASELINE):
    """Rigorous enclosures on closed rational cell, including discontinuous oracle."""
    x = a.interval(lo, hi)
    data = blend(x, a.exact, schedule=schedule)
    ideal = [_ideal_interval(a, lo, hi, t) for t in THRESHOLDS]
    oracle_parts = []
    if lo <= THRESHOLDS[0]:
        oracle_parts.append(a.exact(0))
    for left, right, coefficients in (
        (THRESHOLDS[0], THRESHOLDS[1], P),
        (THRESHOLDS[1], THRESHOLDS[2], Q),
    ):
        if hi > left and lo <= right:
            oracle_parts.append(
                polynomial(a.interval(max(lo, left), min(hi, right)), coefficients, a.exact)
            )
    if hi > THRESHOLDS[2]:
        oracle_parts.append(a.interval(max(lo, THRESHOLDS[2]), hi))
    oracle = a.hull(*oracle_parts)
    tanh, phi = a.targets(x, lo, hi)
    result = {
        "output_error_piecewise": abs(data["y"] - oracle),
        "output_error_tanh_gelu": abs(data["y"] - tanh),
        "output_error_xPhi": abs(data["y"] - phi),
        "piecewise_error_tanh_gelu": abs(oracle - tanh),
        "piecewise_error_xPhi": abs(oracle - phi),
        "tanh_gelu_error_xPhi": abs(tanh - phi),
        "output": data["y"],
    }
    ideal_b = (1 - ideal[0], ideal[0] - ideal[1], ideal[1] - ideal[2], ideal[2])
    for i, (b, exact_b) in enumerate(zip(data["b"], ideal_b)):
        result[f"selector_{i}"] = b
        result[f"selector_leakage_{i}"] = abs(b - exact_b)
        overshoot = a.hull(a.exact(0), -b, b - 1)
        result[f"selector_overshoot_{i}"] = a.iv.make_mpf((a.exact(0)._mpi_[0], overshoot._mpi_[1]))
    factors = (data["p"], data["q"] - data["p"], x - data["q"])
    amplification = a.exact(0)
    for i, (h, exact_h, factor, t, d, stages) in enumerate(
        zip(data["h"], ideal, factors, THRESHOLDS, SCALES, data["stages"])
    ):
        term = abs((h - exact_h) * factor)
        result[f"inactive_branch_amplification_{i}"] = term
        amplification += term
        if hi <= t - d * BAND:
            result[f"sign_error_outside_band_{i}"] = abs(stages[-1] + 1)
        elif lo >= t + d * BAND:
            result[f"sign_error_outside_band_{i}"] = abs(stages[-1] - 1)
        for j, stage in enumerate(stages):
            result[f"stage_range_{i}_{j}"] = stage
    result["amplification_bound"] = amplification
    # Evaluate coefficient differences directly: avoid cancellation of two Y intervals.
    dp = polynomial(x, tuple(s - p for s, p in zip(SHORT_P, P)), a.exact)
    dq = polynomial(x, tuple(s - q for s, q in zip(SHORT_Q, Q)), a.exact)
    result["printed_precision_sensitivity"] = abs(data["b"][1] * dp + data["b"][2] * dq)
    return result


def adaptive_partition(a, intervals, evaluate, priority, max_cells, max_depth, max_work,
                       spend=lambda: None):
    """Shared bounded interval refinement; never uses sampled estimates."""
    if len(intervals) > min(max_cells, max_work):
        raise ValueError("initial partition exceeds cell/work cap")
    heap = []
    work = 0

    def add(lo, hi, depth):
        nonlocal work
        spend()  # Count attempted evaluations too, including interrupted work.
        metrics = evaluate(lo, hi)
        work += 1
        low, high = a.ends(metrics[priority])
        heapq.heappush(heap, (-(high - low), work, lo, hi, depth, metrics))

    for lo, hi in intervals:
        add(lo, hi, 0)
    frozen = []
    while heap and len(heap) + len(frozen) < max_cells and work + 2 <= max_work:
        cell = heapq.heappop(heap)
        _, _, lo, hi, depth, _ = cell
        if depth >= max_depth:
            frozen.append(cell)
            continue
        mid = (lo + hi) / 2
        add(lo, mid, depth + 1)
        add(mid, hi, depth + 1)
    return heap + frozen, work


def merge_metrics(a, leaves):
    merged = {}
    unresolved = 0
    for _, _, lo, hi, depth, metrics in leaves:
        stage_ok = all(
            a.ends(v)[0] >= -1 and a.ends(v)[1] <= 1
            for k, v in metrics.items()
            if k.startswith("stage_range_")
        )
        unresolved += not stage_ok
        for key, value in metrics.items():
            merged[key] = a.hull(merged[key], value) if key in merged else value
    return merged, unresolved


def enclosure_report(a, leaves, work, max_cells, max_depth, max_work):
    merged, unresolved = merge_metrics(a, leaves)
    return {
        "arithmetic": "outward interval enclosures; exact binary endpoint encoding",
        "coverage": "entire [-8,8]; threshold ties included conservatively",
        "leaf_cells": len(leaves),
        "cell_evaluations": work,
        "maximum_leaf_depth": max(c[4] for c in leaves),
        "caps": {"cells": max_cells, "depth": max_depth, "work": max_work},
        "stopped_at": {
            "cell_cap": len(leaves) >= max_cells,
            "work_cap": work + 2 > max_work,
            "depth_limited_cells": sum(c[4] >= max_depth for c in leaves),
        },
        "stage_domain_check": "enclosed_in_[-1,1]" if not unresolved else "unresolved",
        "unresolved_stage_cells": unresolved,
        "error_acceptance": "unresolved: no error tolerance approved; bounds may be loose",
        "bounds": {k: a.encode(v) for k, v in sorted(merged.items())},
    }


def enclose_domain(a, max_cells=128, max_depth=12, max_work=256):
    """Whole-domain outward bounds; unresolved caps are not acceptance."""
    validate_resources(a.dps, 2, max_cells, max_depth, max_work)
    cuts = {DOMAIN[0], DOMAIN[1], *THRESHOLDS}
    for t, d in zip(THRESHOLDS, SCALES):
        cuts.update((t - d * BAND, t + d * BAND))
    cuts = sorted(cuts)
    leaves, work = adaptive_partition(
        a, list(zip(cuts, cuts[1:])), lambda lo, hi: cell_metrics(a, lo, hi),
        "output_error_piecewise", max_cells, max_depth, max_work,
    )
    return enclosure_report(a, leaves, work, max_cells, max_depth, max_work)


def sampled_report(a, grid=257, schedule=BASELINE, separations=()):
    maxima = {}
    endpoints = []
    near_band = []
    stage_ranges = {}
    groups = {
        f"{kind}_{epsilon}": {"count": 0, "maxima": {}}
        for epsilon in separations for kind in ("separated", "within_band")
    }
    active = []

    def observe(name, value, x):
        for destination in (maxima, *active):
            if name not in destination or value > destination[name][0]:
                destination[name] = (value, x)

    points = sample_points(grid)
    for rational_x in points:
        active = []
        separation = min(abs((rational_x - t) / d) for t, d in zip(THRESHOLDS, SCALES))
        for epsilon in separations:
            kind = "separated" if separation >= epsilon else "within_band"
            group = groups[f"{kind}_{epsilon}"]
            group["count"] += 1
            active.append(group["maxima"])
        x = a.point(rational_x)
        data = blend(x, a.point, schedule=schedule)
        for i, stages in enumerate(data["stages"]):
            for j, value in enumerate(stages):
                key = f"stage_range_{i}_{j}"
                low, high = stage_ranges.get(key, (value, value))
                stage_ranges[key] = min(low, value), max(high, value)
        y = data["y"]
        oracle = piecewise(x, a.point)
        tanh, phi = a.targets(x)
        for name, target in (("piecewise", oracle), ("tanh_gelu", tanh), ("xPhi", phi)):
            observe(f"output_error_{name}", abs(y - target), rational_x)
        observe("piecewise_error_tanh_gelu", abs(oracle - tanh), rational_x)
        observe("piecewise_error_xPhi", abs(oracle - phi), rational_x)
        observe("tanh_gelu_error_xPhi", abs(tanh - phi), rational_x)
        dp = polynomial(x, tuple(s - p for s, p in zip(SHORT_P, P)), a.point)
        dq = polynomial(x, tuple(s - q for s, q in zip(SHORT_Q, Q)), a.point)
        observe("printed_precision_sensitivity", abs(data["b"][1] * dp + data["b"][2] * dq), rational_x)
        ideal = tuple(int(rational_x > t) for t in THRESHOLDS)
        exact_b = (1 - ideal[0], ideal[0] - ideal[1], ideal[1] - ideal[2], ideal[2])
        for i, (b, eb) in enumerate(zip(data["b"], exact_b)):
            observe(f"selector_leakage_{i}", abs(b - eb), rational_x)
            observe(f"selector_overshoot_{i}", max(0, -b, b - 1), rational_x)
            observe(f"negative_selector_{i}", max(0, -b), rational_x)
        amplification = 0
        for i, (h, eh, factor, t, d) in enumerate(
            zip(
                data["h"],
                ideal,
                (data["p"], data["q"] - data["p"], x - data["q"]),
                THRESHOLDS,
                SCALES,
            )
        ):
            term = abs((h - eh) * factor)
            amplification += term
            observe(f"inactive_branch_amplification_{i}", term, rational_x)
            z = (rational_x - t) / d
            if abs(z) >= BAND:
                observe(
                    f"sign_error_outside_band_{i}",
                    abs(2 * h - 1 - (1 if z > 0 else -1)),
                    rational_x,
                )
            if abs(z) == BAND:
                near_band.append(
                    {
                        "threshold": str(t),
                        "x_exact": str(rational_x),
                        "z_exact": str(z),
                        "sign_estimate": a.mp.nstr(2 * h - 1, 30),
                        "output_error_piecewise_estimate": a.mp.nstr(abs(y - oracle), 30),
                    }
                )
        observe("amplification_bound", amplification, rational_x)
        if rational_x in THRESHOLDS or rational_x in DOMAIN:
            endpoints.append(
                {
                    "x_exact": str(rational_x),
                    "piecewise_estimate": a.mp.nstr(oracle, 30),
                    "blend_estimate": a.mp.nstr(y, 30),
                    "endpoint_blending_error_estimate": a.mp.nstr(abs(y - oracle), 30),
                    "comparators_estimate": [a.mp.nstr(h, 30) for h in data["h"]],
                }
            )
    def encode_maxima(values):
        return {
            k: {"estimate": a.mp.nstr(v, 30), "x_exact": str(x)}
            for k, (v, x) in sorted(values.items())
        }

    return {
        "status": "high_precision_estimates_only_not_uniform_bounds",
        "count": len(points),
        "diagnostic_grid": grid,
        "normalized_offsets": "+/-2^-k, k=0..24, in domain",
        "maxima": encode_maxima(maxima),
        "stage_ranges_estimate": {
            k: [a.mp.nstr(v, 30) for v in bounds] for k, bounds in stage_ranges.items()
        },
        "separated_and_within_band": {
            k: {"count": group["count"], "maxima": encode_maxima(group["maxima"])}
            for k, group in groups.items()
        },
        "endpoints_and_thresholds": endpoints,
        "normalized_2^-20_cases": near_band,
    }


def oracle_jumps():
    """Exact rational one-sided jumps; independent of comparison schedule."""
    jumps = []
    for t, left, right in (
        (THRESHOLDS[0], Fraction(0), polynomial(THRESHOLDS[0], P)),
        (THRESHOLDS[1], polynomial(THRESHOLDS[1], P), polynomial(THRESHOLDS[1], Q)),
        (THRESHOLDS[2], polynomial(THRESHOLDS[2], Q), THRESHOLDS[2]),
    ):
        jumps.append(
            {
                "threshold_exact": str(t),
                "oracle_value_and_left_limit_exact": str(left),
                "oracle_right_limit_exact": str(right),
                "jump_exact": str(right - left),
                "continuous_approximation_uniform_error_floor_exact": str(abs(right - left) / 2),
            }
        )
    return jumps


def numeric_blockers(a):
    """Point enclosures and exact jump floors, not whole-domain upper bounds.

    Strictly outside-band witnesses avoid relying on whether the uncertainty band
    includes its endpoints. A continuous polynomial blend cannot uniformly match
    a discontinuous oracle more closely than half its largest jump: both one-sided
    errors share the same limiting blend value, so their sum bounds the jump.
    """
    sign_witnesses = []
    for multiple in (1, 2):
        for direction in (-1, 1):
            z = direction * multiple * BAND
            sign = sign_stages(a.exact(z), a.exact)[-1]
            error = abs(sign - direction)
            sign_witnesses.append(
                {
                    "z_exact": str(z),
                    "strictly_outside_normalized_band": abs(z) > BAND,
                    "x_by_threshold_exact": [
                        str(t + d * z) for t, d in zip(THRESHOLDS, SCALES)
                    ],
                    "sign_enclosure": a.encode(sign),
                    "sign_error_enclosure": a.encode(error),
                    "comparator_error_enclosure": a.encode(error / 2),
                }
            )
    jumps = oracle_jumps()
    # Adjacent differences can be negative even when each H lies in [0,1].
    # These fixed rational stress points are diagnostics, not new breakpoints.
    selector_witnesses = []
    for x, index in ((Fraction(-85, 64), 1), (Fraction(83, 16), 2)):
        data = blend(a.exact(x), a.exact)
        selector_witnesses.append(
            {
                "x_exact": str(x),
                "selector_index": index,
                "selector_enclosure": a.encode(data["b"][index]),
            }
        )
    return {
        "status": "numeric blockers only; no acceptance tolerance or profile adopted",
        "sign_point_witnesses": sign_witnesses,
        "oracle_jumps": jumps,
        "continuous_oracle_uniform_error_floor_exact": str(
            max(Fraction(j["continuous_approximation_uniform_error_floor_exact"]) for j in jumps)
        ),
        "negative_selector_point_witnesses": selector_witnesses,
        "interpretation": (
            "Point error lower-bounds any uniform sign-error budget covering that point; "
            "jump floors concern the piecewise oracle, not tanh GELU or x*Phi(x). "
            "Whole-domain upper-error acceptance remains unresolved."
        ),
    }


class StudyLimit(RuntimeError):
    """Hard wall-time or shared work limit; never an acceptance result."""


@contextmanager
def wall_limit(seconds):
    """Main-thread POSIX wall alarm, preserving an enclosing deadline/handler."""
    if seconds <= 0:
        raise StudyLimit("wall-time cap exceeded")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(_signum, _frame):
        raise StudyLimit("wall-time cap exceeded")

    signal.signal(signal.SIGALRM, expired)
    limit = min(seconds, previous_timer[0]) if previous_timer[0] else seconds
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            remaining = max(0.000001, previous_timer[0] - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def normalized_sample_points():
    points = {Fraction(-1) + Fraction(i, 128) for i in range(257)}
    points.update(direction * Fraction(1, 2**k) for k in range(25) for direction in (-1, 1))
    return sorted(points)


def arithmetic_cost(schedule):
    """Literal Horner DAG (no zero-coefficient folding), not an HE cost ledger."""
    stages = sum(schedule)
    return {
        "dag": "degree-nine ordinary Horner per stage; seed highest coefficient; no zero folding",
        "composed_stages": stages,
        "sign_formal_degree": 9**stages,
        "blend_degree_upper_bound": 9**stages + 6,
        "sign": {
            "constant_products": stages,
            "variable_products": 8 * stages,
            "additions": 9 * stages,
            "multiplication_dependency_depth": 9 * stages,
        },
        "one_blend": {
            "constant_products": 3 * stages + 2,
            "variable_products": 24 * stages + 10,
            "additions_and_subtractions": 27 * stages + 20,
            "divisions_by_public_constants": 6,
            "multiplication_dependency_depth": 9 * stages + 1,
        },
        "excludes": "diagnostic targets, sensitivity controls, conversions and interval refinement",
        "qualification": "clear scalar arithmetic only; not CKKS depth, latency, memory or security",
    }


def sign_sampled_report(a, schedule):
    points = normalized_sample_points()
    groups = {epsilon: [] for epsilon in SEPARATIONS}
    ranges = {}
    witnesses = []
    witness_points = {direction * epsilon for epsilon in (*SEPARATIONS, 2 * BAND)
                      for direction in (-1, 1)}
    for z in points:
        stages = sign_stages(a.point(z), a.point, schedule)
        for index, value in enumerate(stages):
            lo, hi = ranges.get(index, (value, value))
            ranges[index] = min(lo, value), max(hi, value)
        if z:
            error = abs(stages[-1] - (1 if z > 0 else -1))
            for epsilon in SEPARATIONS:
                if abs(z) >= epsilon:
                    groups[epsilon].append((error, z))
        if z in witness_points:
            enclosed_stages = sign_stages(a.exact(z), a.exact, schedule)
            error = abs(enclosed_stages[-1] - (1 if z > 0 else -1))
            witnesses.append({
                "z_exact": str(z),
                "strictly_outside_nominal_band": abs(z) > BAND,
                "sign": a.encode(enclosed_stages[-1]),
                "sign_error": a.encode(error),
                "comparator_error": a.encode(error / 2),
            })
    return {
        "status": "sampled maxima are not uniform bounds; point enclosures separately labeled",
        "count": len(points),
        "separated_domains": {
            str(epsilon): {
                "count": len(values),
                "max_sign_error_estimate": a.mp.nstr(max(values)[0], 30),
                "max_comparator_error_estimate": a.mp.nstr(max(values)[0] / 2, 30),
                "maximizer_z_exact": str(max(values)[1]),
            } for epsilon, values in groups.items()
        },
        "stage_ranges_estimate": {
            str(index): [a.mp.nstr(v, 30) for v in bounds] for index, bounds in ranges.items()
        },
        "outward_point_witnesses": witnesses,
    }


def separated_cell(lo, hi, epsilon):
    """Closed cell wholly in E_epsilon, with exact original-unit band widths."""
    return all(hi <= t - d * epsilon or lo >= t + d * epsilon
               for t, d in zip(THRESHOLDS, SCALES))


def follow_up_enclosure(a, schedule, epsilon, spend):
    if epsilon is not None:
        intervals = [(Fraction(-1), -epsilon), (epsilon, Fraction(1))]

        def evaluate(lo, hi):
            stages = sign_stages(a.interval(lo, hi), a.exact, schedule)
            error = abs(stages[-1] - (1 if lo > 0 else -1))
            return {"sign_error": error, "comparator_error": error / 2,
                    **{f"stage_range_{i}": value for i, value in enumerate(stages)}}

        priority = "sign_error"
    else:
        cuts = {DOMAIN[0], DOMAIN[1], *THRESHOLDS}
        for t, d in zip(THRESHOLDS, SCALES):
            for separation in SEPARATIONS:
                cuts.update((t - d * separation, t + d * separation))
        cuts = sorted(cuts)
        intervals = list(zip(cuts, cuts[1:]))

        def evaluate(lo, hi):
            return cell_metrics(a, lo, hi, schedule)

        priority = "output_error_piecewise"
    leaves, work = adaptive_partition(a, intervals, evaluate, priority, 64, 12, 128, spend)
    report = enclosure_report(a, leaves, work, 64, 12, 128)
    report["coverage"] = (f"[-1,-{epsilon}] union [{epsilon},1]" if epsilon is not None
                          else "entire [-8,8]; all thresholds and separation boundaries cut")
    if epsilon is None:
        report["separated_subsets"] = {}
        for separation in SEPARATIONS:
            selected = [cell for cell in leaves if separated_cell(cell[2], cell[3], separation)]
            merged, unresolved = merge_metrics(a, selected)
            report["separated_subsets"][str(separation)] = {
                "coverage": "E_epsilon; initial cuts include every band boundary",
                "selected_cells": len(selected),
                "unresolved_stage_cells": unresolved,
                "error_acceptance": "unresolved; no tolerance approved",
                "bounds": {k: a.encode(v) for k, v in sorted(merged.items())},
            }
    return report


def run_follow_up(result=None):
    """Approved bounded clear study only. Limits are fixed, not CLI-tunable."""
    result = {} if result is None else result
    started = time.monotonic()
    work = 0

    def spend():
        nonlocal work
        if work >= 2048:
            raise StudyLimit("total interval cell-evaluation cap exceeded")
        work += 1

    result.update({
        "experiment": "nexus_gelu_follow_up",
        "scope": "approved research study only; no numerical contract or CKKS adoption",
        "source": "Cheon-Kim-Kim ASIACRYPT 2020, DOI 10.1007/978-3-030-64834-3_8; pinned printed f4/g_(4,1/4)",
        "coefficients_exact": {name: list(map(str, values)) for name, values in
                               (("P", P), ("Q", Q), ("short_P", SHORT_P), ("short_Q", SHORT_Q),
                                ("f4", F4), ("g4", G4))},
        "domain_exact": list(map(str, DOMAIN)),
        "thresholds_exact": list(map(str, THRESHOLDS)),
        "endpoint_rules": "zero at -4; P at -1.95; Q at 3; identity only above 3",
        "difference_scales_exact": list(map(str, SCALES)),
        "separations_exact": list(map(str, SEPARATIONS)),
        "original_unit_separation_widths_exact": {
            str(epsilon): [str(d * epsilon) for d in SCALES] for epsilon in SEPARATIONS
        },
        "decimal_precision": 80,
        "mpmath_version": "1.3.0",
        "oracle_jumps": oracle_jumps(),
        "continuous_oracle_uniform_error_floor_exact": str(abs(polynomial(Fraction(3), Q) - 3) / 2),
        "limits": {"candidates": 4, "normalized_samples_per_candidate": 307,
                   "original_unit_samples_per_candidate": 381, "total_cell_evaluations": 2048,
                   "cells_per_pass": 64, "evaluations_per_pass": 128, "depth": 12,
                   "seconds_per_candidate": 120, "seconds_total": 480},
        "candidates": [],
    })
    try:
        with wall_limit(480):
            a = Arithmetic(80)
            for schedule in FOLLOW_UP_SCHEDULES:
                candidate = {"schedule": list(schedule), "status": "incomplete",
                             "arithmetic_cost": arithmetic_cost(schedule)}
                result["candidates"].append(candidate)
                before = work
                candidate_started = time.monotonic()
                try:
                    with wall_limit(120):
                        candidate["sign_sampled"] = sign_sampled_report(a, schedule)
                        candidate["gelu_sampled"] = sampled_report(a, 257, schedule, SEPARATIONS)
                        candidate["sign_enclosed"] = {}
                        for epsilon in SEPARATIONS:
                            candidate["sign_enclosed"][str(epsilon)] = follow_up_enclosure(
                                a, schedule, epsilon, spend)
                        candidate["gelu_enclosed"] = follow_up_enclosure(a, schedule, None, spend)
                        candidate["status"] = "completed_bounded_work_not_certified"
                except (StudyLimit, ArithmeticError, ValueError) as exc:
                    candidate["failure"] = str(exc)
                finally:
                    candidate["elapsed_seconds"] = time.monotonic() - candidate_started
                    candidate["attempted_cell_evaluations"] = work - before
    except StudyLimit as exc:
        result["failure"] = str(exc)
    finally:
        result["attempted_cell_evaluations"] = work
        result["elapsed_seconds"] = time.monotonic() - started
    for schedule in FOLLOW_UP_SCHEDULES[len(result["candidates"]):]:
        result["candidates"].append({"schedule": list(schedule), "status": "not_started"})
    result["status"] = ("completed_bounded_study_not_certified" if
                        len(result["candidates"]) == 4 and all(
                            c["status"] == "completed_bounded_work_not_certified"
                            for c in result["candidates"]) else "incomplete")
    result["adoption"] = "unresolved: target, error/statistic, boundary/negative-weight and range policies not approved"
    return result


def validate_resources(dps, grid, max_cells, max_depth, max_work):
    for name, value, low, high in (
        ("dps", dps, 40, 200),
        ("grid", grid, 2, 4097),
        ("max_cells", max_cells, 10, 4096),
        ("max_depth", max_depth, 0, 24),
        ("max_work", max_work, 10, 8192),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low},{high}]")


def run(*, dps=80, grid=257, max_cells=128, max_depth=12, max_work=256):
    validate_resources(dps, grid, max_cells, max_depth, max_work)
    a = Arithmetic(dps)
    return {
        "experiment": "nexus_gelu_numerics",
        "scope": "research only; no actual-layer range claim",
        "policy": "no refit, clipping, extra repetitions, HE/profile adoption, or acceptance tolerance",
        "domain_exact": list(map(str, DOMAIN)),
        "thresholds_exact": list(map(str, THRESHOLDS)),
        "difference_scales_exact": list(map(str, SCALES)),
        "normalized_band_exact": str(BAND),
        "original_unit_bands_exact": [str(d * BAND) for d in SCALES],
        "composition": "f4(f4(g_(4,1/4)(g_(4,1/4)(z))))",
        "sign_derivative_at_zero_exact": str(Fraction(315, 128) ** 2 * Fraction(5850, 1024) ** 2),
        "coefficient_interpretation": "PUMA printed exact decimal rationals; BumbleBee sensitivity only",
        "coefficients_exact": {
            "P": list(map(str, P)),
            "Q": list(map(str, Q)),
            "short_P": list(map(str, SHORT_P)),
            "short_Q": list(map(str, SHORT_Q)),
            "f4": list(map(str, F4)),
            "g4": list(map(str, G4)),
        },
        "mpmath_version": "1.3.0",
        "decimal_precision": dps,
        "numeric_blockers": numeric_blockers(a),
        "sampled": sampled_report(a, grid),
        "enclosed": enclose_domain(a, max_cells, max_depth, max_work),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="caller-selected JSON path")
    parser.add_argument("--follow-up", action="store_true", help="approved four-schedule bounded study")
    defaults = (
        ("dps", 80),
        ("grid", 257),
        ("max-cells", 128),
        ("max-depth", 12),
        ("max-work", 256),
    )
    for name, default in defaults:
        parser.add_argument("--" + name, type=int, default=default)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    follow_up = args.pop("follow_up")
    if follow_up:
        if any(args[name.replace("-", "_")] != default for name, default in defaults):
            parser.error("follow-up resources are fixed by approval; omit baseline resource flags")
        result = {}
        try:
            # This outer alarm includes setup, report encoding and ordinary output.
            # A timeout stops study work; only a small incomplete report is saved afterward.
            with wall_limit(480):
                run_follow_up(result)
                output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        except StudyLimit as exc:
            # Do not serialize potentially large partial enclosures after the hard
            # total deadline. Save failure/status only; no further study operations.
            result = {
                "experiment": "nexus_gelu_follow_up", "status": "incomplete",
                "failure": str(exc),
                "attempted_cell_evaluations": result.get("attempted_cell_evaluations"),
                "candidate_statuses": [
                    {"schedule": c["schedule"], "status": c["status"]}
                    for c in result.get("candidates", [])
                ],
            }
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if result["status"] == "incomplete":
            raise SystemExit(1)
        return
    try:
        result = run(**args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
