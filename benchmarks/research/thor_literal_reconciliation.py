"""D30's bounded clear THOR literal study; Python standard library only.

Run: python3 benchmarks/research/thor_literal_reconciliation.py > target/thor-literals.json

P is the literal increasing-order Appendix C composition, not a recovered fit
domain. A uses every pinned artifact decimal, reversed lists and complete caller
scaling. Its binary64 diagnostic preserves the source's Stockmeyer operation
grouping (including GELU's coefficient halving before evaluation). Bootstrap is
identity-on-payload algebra here, NOT simulated or qualified CKKS.

Transcription sources: papers/thor.pdf, Appendix C p15, SHA256 below; pinned
crypto-starlab/THOR nonlinear/{gelu,tanh,polynomial}.py, as cited in
docs/research/thor-multiblock-indexing-audit.md. No upstream import/runtime needed.
"""
from decimal import (
    Decimal, DivisionByZero, InvalidOperation, Overflow, ROUND_HALF_EVEN, localcontext,
)
import json
import math
import resource
import signal
import sys
import time


PAPER_SHA256 = "ddb57004b56ee8e00beffc4f563fd0921242753fda6c704065d4900b792439cd"
ARTIFACT_COMMIT = "b513c06aaabccc04b560721e35e9a98008428c25"
MAX_EVALUATIONS = 5000
MAX_SECONDS = 30
MAX_MEMORY_BYTES = 64 << 20
MAX_OUTPUT_BYTES = 1 << 20
MODES = ("decimal80", "decimal160", "binary64")

# Strings retain all printed digits; neither source is generated from the other.
P = {
    "gelu": (
        """-1.0624e-5 1.6445e-4 -5.8353e-4 -3.8091e-4
        2.2443e-3 8.9230e-3 -1.0528e-2 -1.9183e-2
        -2.0463e-1 4.5401e-1 -5.4076e-1 5.6775e0
        -1.3643e1 1.8257e1 -8.4885e1 1.2869e2
        3.6672e2 -1.0140e3 -1.2628e2 2.2173e3
        -9.9542e2 -2.3106e3 1.7358e3 1.2738e3
        -1.2784e3 -3.6678e2 4.7966e2 4.9461e1
        -9.0675e1 -2.3652e0 8.7431e0 1.6284e-2""".split(),
        """-1.7027e2 6.8108e1 1.7920e3 -6.8162e2
        -8.4926e3 3.0563e3 2.3958e4 -8.1044e3
        -4.4815e4 1.4130e4 5.8620e4 -1.7037e4
        -5.5133e4 1.4553e4 3.7787e4 -8.8767e3
        -1.8951e4 3.8497e3 6.9417e3 -1.1690e3
        -1.8466e3 2.4169e2 3.5445e2 -3.2450e1
        -4.9192e1 2.5812e0 5.7839e0 -9.4517e-2""".split(),
    ),
    "tanh": (
        """-7.1453e3 -7.7652e1 2.7428e4 2.4515e2
        -4.2579e4 -3.0195e2 3.4219e4 1.8299e2
        -1.5116e4 -5.6410e1 3.5876e3 8.1760e0
        -4.1334e2 -4.2902e-1 1.9506e1 2.0620e-3""".split(),
        """-9.0257e-3 -1.1232e-4 1.0876e-1 7.9679e-4
        -5.4133e-1 -1.4287e-3 1.4648e0 -2.2242e-3
        -2.4326e0 1.1738e-2 2.7497e-1 -1.7763e-2
        -2.3893e0 1.3019e-2 2.0287e0 -4.0844e-3""".split(),
    ),
}
A = {
    "gelu": (
        """-1.06240033e-05 1.64454894e-04 -5.83533517e-04 -3.80912692e-04
        2.24431193e-03 8.92295204e-03 -1.05277477e-02 -1.91827040e-02
        -2.04634786e-01 4.54014410e-01 -5.40759203e-01 5.67745523e+00
        -1.36433727e+01 1.82574621e+01 -8.48849601e+01 1.28686741e+02
        3.66720281e+02 -1.01400159e+03 -1.26278856e+02 2.21728878e+03
        -9.95421415e+02 -2.31059465e+03 1.73583957e+03 1.27394360e+03
        -1.27836230e+03 -3.66781716e+02 4.79663919e+02 4.94610178e+01
        -9.06754761e+01 -2.36515790e+00 8.74311855e+00 1.62838703e-02""".split(),
        """-1.70270667e+02 6.81076279e+01 1.79197364e+03 -6.81621043e+02
        -8.49256169e+03 3.05629446e+03 2.39579397e+04 -8.10435126e+03
        -4.48145152e+04 1.41297616e+04 5.86197512e+04 -1.70371505e+04
        -5.51326382e+04 1.45532495e+04 3.77866438e+04 -8.87673890e+03
        -1.89514802e+04 3.84972853e+03 6.94169727e+03 -1.16901058e+03
        -1.84658407e+03 2.41693754e+02 3.54452276e+02 -3.24499570e+01
        -4.91918227e+01 2.58122977e+00 5.78392852e+00 -9.45171527e-02""".split(),
    ),
    "tanh": (
        """-7.14529052e+03 -7.76519925e+01 2.74279201e+04 2.45150249e+02
        -4.25793697e+04 -3.01953016e+02 3.42189880e+04 1.82989351e+02
        -1.51158283e+04 -5.64098990e+01 3.58757327e+03 8.17596753e+00
        -4.13341496e+02 -4.29024545e-01 1.95056729e+01 2.06201784e-03""".split(),
        """-9.02573450e-03 -1.12320034e-04 1.08762008e-01 7.96793166e-04
        -5.41327356e-01 -1.42873183e-03 1.46476749e+00 -2.22416152e-03
        -2.43259032e+00 1.17381072e-02 2.74974898e+00 -1.77631073e-02
        -2.38934873e+00 1.30194294e-02 2.02874846e+00 -4.08442578e-03""".split(),
    ),
}


def points(function):
    if function not in ("gelu", "tanh"):
        raise ValueError("unknown function")
    scale = 1 if function == "gelu" else 5
    with localcontext() as context:
        context.prec = 80  # Exact dyadics; never inherit a caller's short context.
        return tuple(sorted([Decimal(scale * k) / 16 for k in range(-128, 129)] + [
            Decimal(sign) / (2 ** power) for power in (10, 20) for sign in (-1, 1)]))


def horner(coefficients, x):
    """Increasing-order polynomial; no input/caller rescaling."""
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * x + coefficient
    return value


def stockmeyer(coefficients, x):
    """Independent scalar implementation of pinned polynomial.py:12-64 grouping."""
    if len(coefficients) not in (16, 28, 32):
        raise ValueError("only the source study's complete tuple lengths are supported")
    x2 = x * x
    x4 = x2 * x2
    x8 = x4 * x4
    babies = []
    for offset in range(0, len(coefficients), 4):
        c0, c1, c2, c3 = coefficients[offset:offset + 4]
        value = c1 * x
        value = value + c2 * x2
        value = value + x2 * (c3 * x)  # Not c3*(x*x2): preserve source grouping.
        babies.append(value + c0)

    def pair(values, power):
        return [values[i] + values[i + 1] * power if i + 1 < len(values) else values[i]
                for i in range(0, len(values), 2)]

    first = pair(babies, x4)
    if len(coefficients) == 16:
        return first[0] + first[1] * x8
    second = pair(first, x8)
    return second[0] + second[1] * (x8 * x8)


def _evaluate(function, candidate, x, convert):
    first, second = (P if candidate == "P" else A)[function]
    first, second = tuple(map(convert, first)), tuple(map(convert, second))
    if candidate == "P":
        return horner(second, horner(first, x))
    first, second = first[::-1], second[::-1]
    if function == "tanh":
        return stockmeyer(second, stockmeyer(first, x / 40))
    u = x / 64
    inner = stockmeyer(first, u)
    # Source halves all outer coefficients first, then forms (64*u)*(1/2+helper).
    # In real algebra this is Z/2 * [1 + F2(F1(Z/64))], not direct composition.
    half = convert("0.5")
    helper = stockmeyer(tuple(c * half for c in second), inner)
    return (64 * u) * (half + helper)


def evaluate(function, candidate, x, mode):
    if function not in P or candidate not in ("P", "A") or mode not in MODES:
        raise ValueError("only the two approved candidates/functions and three arithmetic modes")
    if mode == "binary64":
        return _evaluate(function, candidate, float(x), float)
    with localcontext() as context:
        context.prec = 80 if mode == "decimal80" else 160
        context.rounding = ROUND_HALF_EVEN
        context.Emax, context.Emin = 999999, -999999
        for condition in (InvalidOperation, DivisionByZero, Overflow):
            context.traps[condition] = False  # Preserve/report NaN and infinities.
        return _evaluate(function, candidate, Decimal(x), Decimal)


def baseline(function, x):
    x = float(x)
    return .5 * x * (1 + math.erf(x / math.sqrt(2))) if function == "gelu" else math.tanh(x)


class StudyLimit(RuntimeError):
    pass


def peak_rss_bytes():
    scale = {"darwin": 1, "linux": 1024}.get(sys.platform)
    if scale is None:
        raise StudyLimit("working-RSS units unavailable on this platform")
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale


class Budget:
    """Bounded scalar work; no worker process, sampling thread or output archive."""

    def __init__(self, *, evaluations=MAX_EVALUATIONS):
        if type(evaluations) is not int or not 1 <= evaluations <= MAX_EVALUATIONS:
            raise ValueError("cannot raise the approved evaluation limit")
        self.limit = evaluations
        self.evaluations = 0
        self.started = time.monotonic()

    def check(self):
        if time.monotonic() - self.started >= MAX_SECONDS:
            raise StudyLimit("30-second wall limit")
        if peak_rss_bytes() > MAX_MEMORY_BYTES:
            raise StudyLimit("64-MiB process working-RSS limit")

    def count(self):
        self.check()
        if self.evaluations >= self.limit:
            raise StudyLimit("complete candidate-point evaluation limit")
        self.evaluations += 1


def finite(value):
    return value.is_finite() if isinstance(value, Decimal) else math.isfinite(value)


def classification(value):
    if finite(value):
        return "finite"
    if value.is_nan() if isinstance(value, Decimal) else math.isnan(value):
        return "nan"
    return "positive_infinity" if value > 0 else "negative_infinity"


def _record_max(summary, name, value, x):
    previous = summary.get(name)
    if previous is None or value > Decimal(previous["value"]):
        summary[name] = {"value": str(value), "input": str(x)}


def _difference(summary, observed, reference, x):
    if not finite(observed) or not finite(reference):
        summary["nonfinite_pairs"] = summary.get("nonfinite_pairs", 0) + 1
        return
    summary["finite_pairs"] = summary.get("finite_pairs", 0) + 1
    difference = abs(observed - reference)
    _record_max(summary, "max_absolute", difference, x)
    if reference:
        _record_max(summary, "max_relative", difference / abs(reference), x)
    else:
        summary["zero_reference_pairs"] = summary.get("zero_reference_pairs", 0) + 1


def study(budget):
    rows, summaries = {}, {}
    for function in ("gelu", "tanh"):
        rows[function], summaries[function] = [], {}
        for candidate in ("P", "A"):
            summaries[function][candidate] = {
                "classifications": {mode: {key: 0 for key in (
                    "finite", "nan", "positive_infinity", "negative_infinity")} for mode in MODES},
                "decimal80_vs_160": {"rounded160_to80_matches": 0},
                "binary64_vs_decimal160": {}, "baseline_binary64_errors": {mode: {} for mode in MODES},
            }
        for x in points(function):
            reference = baseline(function, x)
            row = {"input": str(x), "baseline_binary64": repr(reference)}
            for candidate in ("P", "A"):
                values = {}
                for mode in MODES:
                    budget.count()
                    values[mode] = evaluate(function, candidate, x, mode)
                    budget.check()
                row[candidate] = {mode: str(value) for mode, value in values.items()}
                summary = summaries[function][candidate]
                for mode, value in values.items():
                    summary["classifications"][mode][classification(value)] += 1
                with localcontext() as context:
                    context.prec = 160
                    _difference(summary["decimal80_vs_160"], values["decimal80"], values["decimal160"], x)
                    converted = Decimal.from_float(values["binary64"])
                    _difference(summary["binary64_vs_decimal160"], converted, values["decimal160"], x)
                    for mode in MODES:
                        value = converted if mode == "binary64" else values[mode]
                        _difference(summary["baseline_binary64_errors"][mode], value,
                                    Decimal.from_float(reference), x)
                with localcontext() as context:
                    context.prec = 80
                    if finite(values["decimal160"]) and values["decimal80"] == +values["decimal160"]:
                        summary["decimal80_vs_160"]["rounded160_to80_matches"] += 1
            rows[function].append(row)
    budget.check()
    return {
        "status": "completed",
        "study": "D30 THOR literal P/A reconciliation; diagnostic clear algebra only",
        "limits": {"candidate_point_evaluations": MAX_EVALUATIONS, "wall_seconds": MAX_SECONDS,
                   "process_working_rss_bytes": MAX_MEMORY_BYTES, "output_bytes": MAX_OUTPUT_BYTES},
        "inputs": {"paper_sha256": PAPER_SHA256, "artifact_commit": ARTIFACT_COMMIT,
                   "points_per_function": 261, "gelu_grid": "k/16, -128<=k<=128; plus +/-2^-10,+/-2^-20",
                   "tanh_grid": "5*k/16, -128<=k<=128; plus +/-2^-10,+/-2^-20"},
        "formulas": {"P_gelu": "f2(f1(Z)), increasing printed tuples",
                     "P_tanh": "g2(g1(A)), increasing printed tuples",
                     "A_gelu": "Z/2*[1+F2(F1(Z/64))], descending pinned literals",
                     "A_tanh": "G2(G1(A/40)), descending pinned literals"},
        "literal_conflicts": {
            "tanh_position_10": {"printed_term": "0.27497*g1(A)^10",
                                 "artifact_term": "2.74974898*G1(A/40)^5"},
            "gelu_position_23": {"printed": P["gelu"][0][23], "artifact": A["gelu"][0][23]},
        },
        "limitations": [
            "No hybrid, refit, clipping, coefficient replacement or profile/backend adoption.",
            "GELU [-8,8] is a diagnostic window, not a recovered fit/activation domain.",
            "Tanh [-40,40] is a helper parameter interval, not checkpoint coverage.",
            "erf-based GELU and math.tanh are binary64 diagnostics, not certified oracles.",
            "Decimal precision agreement is sensitivity evidence, not certified approximation error.",
            "A assumes ideal real-lane mapping and identity-on-payload bootstrap, not CKKS execution.",
            "No intended-author identity, checkpoint/model quality, security or encrypted-cost inference.",
        ],
        "resources": {"candidate_point_evaluations": budget.evaluations,
                      "evaluation_wall_seconds": time.monotonic() - budget.started,
                      "process_peak_rss_bytes_through_evaluation": peak_rss_bytes(), "output_bytes": 0},
        "summary": summaries, "points": rows,
    }


def encode_output(result):
    # Ordinary JSON only; exact decimal strings and round-trip binary64 strings.
    for _ in range(4):
        encoded = (json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
        if len(encoded) > MAX_OUTPUT_BYTES:
            raise StudyLimit("1-MiB output limit")
        resources = result.get("resources")
        if resources is None or resources["output_bytes"] == len(encoded):
            return encoded
        resources["output_bytes"] = len(encoded)
    raise StudyLimit("output-size accounting did not settle")


def main():
    budget = Budget()
    previous = signal.getsignal(signal.SIGALRM)

    def deadline(_signal, _frame):
        raise StudyLimit("30-second wall limit")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        result = study(budget)
        encoded = encode_output(result)
        budget.check()
        result["resources"]["wall_seconds_before_output"] = time.monotonic() - budget.started
        result["resources"]["process_peak_rss_bytes_before_output"] = peak_rss_bytes()
        encoded = encode_output(result)
        budget.check()  # Also enforce memory/time after final serialization.
        sys.stdout.buffer.write(encoded)
        return 0
    except StudyLimit as error:
        # No partial candidate result is promoted to a completed study.
        sys.stdout.buffer.write(encode_output({"status": "stopped", "reason": str(error),
                                             "candidate_point_evaluations": budget.evaluations}))
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
