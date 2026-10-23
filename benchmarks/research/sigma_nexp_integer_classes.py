"""Finite D29 integer-map checks; no native/model/HE work or exp evaluation.

The selected literal floors were independently certified separately. This helper
loads those exact pinned integers; it does NOT reissue that proof, authenticate a
source/runtime, grant a profile, or make a privacy/full-model-quality claim.

The optional error enclosure is conditional algebra: assuming the selected table
entries are floors of the declared real exponential factors, bound error against
the *clipped* ideal scalar map. No libm/Decimal baseline or replacement map.
"""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "paper/reproduction-contracts/fusefss/gate-set/fixtures.json"
SCALE = 4096
INPUT_LIMIT = 1 << 38
CLASS_COUNT = 1 << 16
PINS = (
    "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b",
    "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9",
)


def load_tables(path=FIXTURE):
    """Literal identity/shape checks, NOT a substitute for floor certification."""
    source = json.loads(Path(path).read_text())
    tables = tuple(tuple(source["tables"][key]) for key in ("exp0", "exp1"))
    for values, expected in zip(tables, PINS):
        if len(values) != 256 or any(type(v) is not int or not 0 <= v <= SCALE for v in values):
            raise ValueError("selected literal shape/domain mismatch")
        if sha256(struct.pack("<256Q", *values)).hexdigest() != expected:
            raise ValueError("selected literal bytes mismatch")
    return tables


def lane(x, tables):
    """Actual approved clear graph on raw Q12 integer codes; masks not modeled."""
    if type(x) is not int or not 0 <= x < INPUT_LIMIT:
        raise ValueError("approved clear root requires 0 <= X < 2**38")
    x40 = x % (1 << 40)
    a = x40 - CLASS_COUNT
    h = int(a >= 0)
    b = h ^ 1
    w = (x40 - (CLASS_COUNT - 1)) % CLASS_COUNT
    v = b * w % CLASS_COUNT
    c = (v + CLASS_COUNT - 1) % CLASS_COUNT
    lo, hi = c % 256, c // 256  # unsigned TR16,8, not signed ARS.
    u0, u1 = tables[0][lo], tables[1][hi]
    p = u0 * u1
    return {"x": x, "a40_signed": a, "h": h, "b": b, "w16": w, "v16": v,
            "c16": c, "j0": lo, "j1": hi, "u0": u0, "u1": u1, "p": p,
            "e": p // SCALE, "remainder": p % SCALE}


def factor_floor_enclosure(c, tables):
    """Conditional exact rational enclosure for ideal scaled clipped exp.

    A=S*exp(-j0/S), B=S*exp(-j1/16). Certified floors give
    t0 <= A < t0+1, t1 <= B < t1+1. Coordinate0 is exactly S,
    so use that singleton rather than needlessly add1. The ideal code is A*B/S.
    Return lower, upper, upper_open; upper is strict except both arguments0.
    """
    if type(c) is not int or not 0 <= c < CLASS_COUNT:
        raise ValueError("effective coordinate outside unsigned16")
    lo, hi = c % 256, c // 256
    a, b = tables[0][lo], tables[1][hi]
    return (Fraction(a * b, SCALE),
            Fraction((a + int(lo != 0)) * (b + int(hi != 0)), SCALE),
            c != 0)


def rational(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def sweep(tables):
    """Exactly65536 effective classes, bounded integer work and small summary."""
    increases = []
    decreases = 0
    flats = 0
    within_row_increases = 0
    histogram = {}
    outputs = []
    zero_runs = []
    ties = []
    tie_count = 0
    max_remainder = -1
    max_error_numerator = -1
    error_cases = []
    previous = None
    for c in range(CLASS_COUNT):
        lo, hi = c % 256, c // 256
        a, b = tables[0][lo], tables[1][hi]
        p = a * b
        e, remainder = divmod(p, SCALE)
        if not 0 <= p <= 1 << 24 or not 0 <= e <= SCALE:
            raise ArithmeticError("selected map violates approved product/output range")
        outputs.append(e)
        if previous is not None:
            delta = e - previous
            histogram[str(delta)] = histogram.get(str(delta), 0) + 1
            if delta > 0:
                increases.append({"before": c - 1, "after": c,
                                  "before_e": previous, "after_e": e, "delta": delta})
                within_row_increases += int(lo != 0)
            elif delta < 0:
                decreases += 1
            else:
                flats += 1
        previous = e
        if e == 0:
            if zero_runs and zero_runs[-1][1] == c:
                zero_runs[-1][1] = c + 1
            else:
                zero_runs.append([c, c + 1])  # Half-open intervals.
        if remainder == SCALE // 2:
            tie_count += 1
            if len(ties) < 8:
                ties.append({"c": c, "p": p, "floor": e})
        max_remainder = max(max_remainder, remainder)
        # No Fraction allocation per point. Common denominator S, exact ints.
        upper_numerator = (a + int(lo != 0)) * (b + int(hi != 0)) - e * SCALE
        if upper_numerator > max_error_numerator:
            max_error_numerator, error_cases = upper_numerator, [c]
        elif upper_numerator == max_error_numerator and len(error_cases) < 8:
            error_cases.append(c)

    high_zero = next(i for i, value in enumerate(tables[1]) if value == 0)
    last_nonzero = max(c for c, value in enumerate(outputs) if value != 0)
    cases = {0, 1, CLASS_COUNT - 2, CLASS_COUNT - 1, CLASS_COUNT, CLASS_COUNT + 1, INPUT_LIMIT - 1}
    for high in range(1, 256):
        cases.update((high * 256 - 1, high * 256, high * 256 + 1))
    for x in sorted(cases):
        row = lane(x, tables)
        if row["c16"] != min(x, CLASS_COUNT - 1) or row["e"] != outputs[row["c16"]]:
            raise ArithmeticError("clip/TR boundary graph differs from literal class equation")

    bound = Fraction(max_error_numerator, SCALE)
    return {
        "scope": "Approved D29 clear literal nExp only; no native/HE/privacy/quality/adoption evidence.",
        "table_pins_u64le": list(PINS),
        "floor_status": "All512 separately certified; this run checks literal identity, not exponential floors.",
        "effective_classes": CLASS_COUNT,
        "root_domain_codes": [0, INPUT_LIMIT], "root_domain_half_open": True,
        "boundary_graph_cases": len(cases),
        "product_range_inclusive": [0, max(tables[0]) * max(tables[1])],
        "output_range_inclusive": [min(outputs), max(outputs)],
        "table_nonincreasing": [all(x >= y for x, y in zip(row, row[1:])) for row in tables],
        "global_nonincreasing": not increases,
        "within_byte_block_nonincreasing": within_row_increases == 0,
        "increase_count": len(increases), "increase_examples": increases[:8],
        "max_increase": max((row["delta"] for row in increases), default=0),
        "decrease_count": decreases, "flat_count": flats, "delta_histogram": histogram,
        "zero_count": sum(hi - lo for lo, hi in zero_runs), "zero_runs_half_open": zero_runs,
        "first_zero": zero_runs[0][0], "last_nonzero": last_nonzero,
        "terminal_zero_start": last_nonzero + 1,
        "first_zero_high_table_index": high_zero,
        "high_table_underflow_from_c": high_zero * 256,
        "zero_to_positive_transitions": sum(row["before_e"] == 0 for row in increases),
        "half_code_tie_examples": ties,
        "half_code_tie_count": tie_count,
        "max_product_floor_gap_codes": rational(Fraction(max_remainder, SCALE)),
        "conditional_clipped_ideal_error": {
            "assumption": "Selected literals are exact declared exponential floors; no ideal exp evaluated.",
            "inequality": "0 <= 4096*exp(-min(X,65535)/4096)-E < upper_codes",
            "upper_codes": rational(bound), "upper_real": rational(bound / SCALE),
            "max_enclosure_classes": error_cases,
            "qualification": "Conditional source-specific enclosure, not observed error or a quality target.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = sweep(load_tables())
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}: {result['effective_classes']} integer classes; "
              f"{result['increase_count']} upward transitions; "
              f"{result['zero_count']} zero classes")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
