"""Bounded NEXUS GELU research experiment; no HE, model, or SDK execution.

Run with optional mpmath==1.3.0 installed, e.g.:
    python examples/benchmarks/nexus_gelu_numerics.py --output target/nexus.json

Printed decimals are exact rationals. MP evaluations are estimates; IV evaluations
use directed rounding, including coefficient conversion and transcendental bounds.
No tolerance, fitted coefficients, clipping, or extra sign repetitions are adopted.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import heapq
import json
from pathlib import Path


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
    result = convert(Fraction(0))
    for coefficient in reversed(coefficients):
        result = result * x + convert(coefficient)
    return result


def sign_stages(z, convert=lambda x: x):
    stages = []
    for coefficients in (G4, G4, F4, F4):
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


def blend(x, convert=lambda x: x, p=P, q=Q):
    stages = tuple(
        sign_stages((x - convert(t)) / convert(d), convert) for t, d in zip(THRESHOLDS, SCALES)
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


def cell_metrics(a, lo, hi):
    """Rigorous enclosures on closed rational cell, including discontinuous oracle."""
    x = a.interval(lo, hi)
    data = blend(x, a.exact)
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


def enclose_domain(a, max_cells=128, max_depth=12, max_work=256):
    """Adaptive finite partition. No sampled value is used to form any bound.

    Every accepted cell remains covered, even if its enclosure is very loose.
    Refinement chooses largest piecewise-error width, not an acceptance tolerance.
    max_work counts complete cell evaluations, each having fixed polynomial work.
    """
    validate_resources(a.dps, 2, max_cells, max_depth, max_work)
    cuts = {DOMAIN[0], DOMAIN[1], *THRESHOLDS}
    for t, d in zip(THRESHOLDS, SCALES):
        cuts.update((t - d * BAND, t + d * BAND))
    cuts = sorted(cuts)
    heap = []
    work = 0

    def add(lo, hi, depth):
        nonlocal work
        metrics = cell_metrics(a, lo, hi)
        work += 1
        low, high = a.ends(metrics["output_error_piecewise"])
        heapq.heappush(heap, (-(high - low), work, lo, hi, depth, metrics))

    for lo, hi in zip(cuts, cuts[1:]):
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
    leaves = heap + frozen
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


def sampled_report(a, grid=257):
    maxima = {}
    endpoints = []
    near_band = []

    def observe(name, value, x):
        if name not in maxima or value > maxima[name][0]:
            maxima[name] = (value, x)

    points = sample_points(grid)
    for rational_x in points:
        x = a.point(rational_x)
        data = blend(x, a.point)
        y = data["y"]
        oracle = piecewise(x, a.point)
        tanh, phi = a.targets(x)
        for name, target in (("piecewise", oracle), ("tanh_gelu", tanh), ("xPhi", phi)):
            observe(f"output_error_{name}", abs(y - target), rational_x)
        observe("piecewise_error_tanh_gelu", abs(oracle - tanh), rational_x)
        observe("piecewise_error_xPhi", abs(oracle - phi), rational_x)
        observe("tanh_gelu_error_xPhi", abs(tanh - phi), rational_x)
        short_y = blend(x, a.point, SHORT_P, SHORT_Q)["y"]
        observe("printed_precision_sensitivity", abs(short_y - y), rational_x)
        ideal = tuple(int(rational_x > t) for t in THRESHOLDS)
        exact_b = (1 - ideal[0], ideal[0] - ideal[1], ideal[1] - ideal[2], ideal[2])
        for i, (b, eb) in enumerate(zip(data["b"], exact_b)):
            observe(f"selector_leakage_{i}", abs(b - eb), rational_x)
            observe(f"selector_overshoot_{i}", max(0, -b, b - 1), rational_x)
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
    return {
        "status": "high_precision_estimates_only_not_uniform_bounds",
        "count": len(points),
        "diagnostic_grid": grid,
        "normalized_offsets": "+/-2^-k, k=0..24, in domain",
        "maxima": {
            k: {"estimate": a.mp.nstr(v, 30), "x_exact": str(x)}
            for k, (v, x) in sorted(maxima.items())
        },
        "endpoints_and_thresholds": endpoints,
        "normalized_2^-20_cases": near_band,
    }


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
        "sampled": sampled_report(a, grid),
        "enclosed": enclose_domain(a, max_cells, max_depth, max_work),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="caller-selected JSON path")
    for name, default in (
        ("dps", 80),
        ("grid", 257),
        ("max-cells", 128),
        ("max-depth", 12),
        ("max-work", 256),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    try:
        result = run(**args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
