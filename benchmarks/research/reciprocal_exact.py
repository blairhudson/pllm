"""D38 N-Q only. Run: python3 -m benchmarks.research.reciprocal_exact

Fixed synthetic exact-real cells, not calibration, CKKS or state qualification.
JSON stdout is published only after the entire campaign and serialization pass.
"""
from fractions import Fraction
import json
import signal
import sys
import time

from benchmarks.research.rational_reciprocal import (
    InvalidReciprocal, MAX_OUTPUT_BYTES, RationalBudget, StudyLimit,
    peak_rss_bytes, reciprocal,
)


def records():
    for denominator in (1, 2, 8, 256):
        a, b = Fraction(1, denominator), Fraction(1)
        for m in (0, 1, 2, 4, 8):
            for position, x in enumerate((a, (a + b) / 2, b)):
                yield f"d{denominator}-m{m}-p{position}", a, b, x, m, False
    for name, a, b, m in (
        ("a_zero", 0, 1, 1), ("a_negative", -1, 1, 1),
        ("b_zero", 1, 0, 1), ("b_negative", 1, -1, 1),
        ("a_above_b", 2, 1, 1), ("m_negative", 1, 1, -1),
    ):
        yield name, Fraction(a), Fraction(b), Fraction(1), m, True


def study(budget):
    rows = []
    report = {"status": "failed", "scope": "D38 N-Q synthetic exact-real only",
              "completed_valid": 0, "rejected_invalid": 0, "attempted": 0,
              "ckks_qualified": False, "state_cells": 0}
    try:
        for identity, a, b, x, m, invalid in records():
            budget.count()
            report["attempted"] += 1
            inputs = {"a": str(a), "b": str(b), "x": str(x), "m": m}
            try:
                y = reciprocal(a, b, x, m, budget)
            except InvalidReciprocal:
                if not invalid:
                    raise
                report["rejected_invalid"] += 1
                rows.append({"id": identity, "inputs": inputs, "status": "invalid_domain"})
                continue
            if invalid:
                raise AssertionError("invalid domain accepted")
            # Independent closed-form/power oracle, not the accumulator loop.
            residual = (1 - x / b) ** (1 << m)
            oracle = (1 - residual) / x
            product = Fraction(1, 1) / b
            for k in range(m):
                product *= 1 + (1 - x / b) ** (1 << k)
            relative = abs(1 - x * y)
            absolute = abs(1 / x - y)
            bound = (1 - a / b) ** (1 << m)
            if not (y == oracle == product and relative == residual <= bound
                    and absolute == relative / x):
                raise AssertionError("independent rational identity mismatch")
            rows.append({"id": identity, "inputs": inputs, "status": "passed", "relative_error": str(relative),
                         "absolute_error": str(absolute), "relative_bound": str(bound),
                         "reciprocal": str(y)})
            report["completed_valid"] += 1
            budget.check()
        if (report["completed_valid"], report["rejected_invalid"]) != (60, 6):
            raise AssertionError("incomplete campaign")
        report.update(status="completed", records=rows)
        # JSON sizing was reserved before each cell; final bytes checked before stdout.
        payload = json.dumps(report, sort_keys=True)
        if len(payload.encode()) > MAX_OUTPUT_BYTES - 4096:
            budget.fail("serialization limit")
        budget.check()
    except (StudyLimit, MemoryError, OverflowError, AssertionError, InvalidReciprocal) as error:
        budget.poisoned = True
        report.pop("records", None)
        report.update(status="failed", error=type(error).__name__, reason=str(error))
    return report


def encode_report(report, budget):
    """No completed JSON escapes a late guard/serialization failure."""
    try:
        if report["status"] == "completed":
            budget.check()
        output = json.dumps(report, sort_keys=True) + "\n"
        if len(output.encode()) > MAX_OUTPUT_BYTES:
            budget.fail("final serialization limit")
        if report["status"] == "completed":
            budget.check()
        return output
    except (StudyLimit, MemoryError, OverflowError) as error:
        budget.poisoned = True
        report.pop("records", None)
        report.update(status="failed", error=type(error).__name__, reason=str(error))
        return json.dumps(report, sort_keys=True) + "\n"


def main():
    budget = RationalBudget()
    def timeout(_signum, _frame):
        budget.fail("30-second wall limit")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 30)
    report = study(budget)
    report["observations"] = {"process_peak_rss_bytes": peak_rss_bytes(),
                              "elapsed_seconds": time.monotonic() - budget.started,
                              "charged_bit_work": budget.work, "reserved_json_bytes": budget.output,
                              "single_process": True, "single_thread": True}
    output = encode_report(report, budget)
    sys.stdout.write(output)
    signal.setitimer(signal.ITIMER_REAL, 0)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
