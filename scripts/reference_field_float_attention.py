#!/usr/bin/env python3
"""Independent exact attention-contraction oracle; no native/project imports.

Only integer/Fraction arithmetic. Each RN32 and R16 boundary is a separate
nearest-neighbor search on the destination IEEE code lattice, with a virtual
2**128 successor defining the overflow midpoint. Never host float arithmetic.

Fixtures are independently specified, manually derived cases, not Rust outputs.
--check recomputes and checks those expectations. Without --check, emit the same
cases with recomputed expectations to stdout; never overwrite the fixture.
Both modes also assert exact NONFUSED/FMA equations for two underflow witnesses.
Source authoring is not evidence of execution or a model-profile approval.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path


class NonFinite(ValueError):
    """Nonfinite operand or rounded overflow, matching the numeric contract."""


def power_two(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def magnitude(code, fraction_bits):
    """Positive code lattice, including virtual overflow successor, exactly."""
    end = 255 << fraction_bits
    if code == end:
        return power_two(128)
    if not 0 <= code < end:
        raise NonFinite()
    exponent, fraction = divmod(code, 1 << fraction_bits)
    if exponent == 0:
        return fraction * power_two(-126 - fraction_bits)
    return ((1 << fraction_bits) + fraction) * power_two(exponent - 127 - fraction_bits)


def decode(bits, fraction_bits):
    sign_bit = 1 << (fraction_bits + 8)
    if not 0 <= bits < sign_bit * 2:
        raise ValueError("bit word outside format")
    if (bits & (sign_bit - 1)) >> fraction_bits == 255:
        raise NonFinite()
    value = magnitude(bits & (sign_bit - 1), fraction_bits)
    return -value if bits & sign_bit else value


def rne(value, fraction_bits, negative_zero=False):
    """Round exact rational by lattice neighbors, preserving specified zero sign."""
    negative = value < 0 or (value == 0 and negative_zero)
    target = abs(value)
    end = 255 << fraction_bits
    low, high = 0, end
    while low < high:
        middle = (low + high + 1) // 2
        if magnitude(middle, fraction_bits) <= target:
            low = middle
        else:
            high = middle - 1
    if low == end:
        raise NonFinite()
    down = target - magnitude(low, fraction_bits)
    up = magnitude(low + 1, fraction_bits) - target
    if down > up or (down == up and low % 2 == 1):
        low += 1
    if low == end:
        raise NonFinite()
    return low | (int(negative) << (fraction_bits + 8))


def add32(a, b):
    # Exact cancellation is +0 except -0 + -0. Nonzero same-sign sums cannot
    # become exact zero; sign for underflow follows exact rational sign.
    return rne(decode(a, 23) + decode(b, 23), 23, bool(a & b & (1 << 31)))


def mul32(a, b):
    return rne(decode(a, 23) * decode(b, 23), 23, bool((a ^ b) & (1 << 31)))


def cast16(bits):
    return rne(decode(bits, 23), 7, bool(bits & (1 << 31)))


def dot32(left, right):
    if len(left) != len(right):
        raise ValueError("dot length mismatch")
    total = 0
    for a, b in zip(left, right):
        # BF16 promotion is exact, not a rounding boundary.
        total = add32(total, mul32(a << 16, b << 16))
    return total


def score(query, keys, alpha):
    if not query or not keys or len(keys) % len(query):
        raise ValueError("positive rectangular score geometry required")
    width = len(query)
    return [
        cast16(mul32(cast16(dot32(query, keys[start:start + width])) << 16, alpha))
        for start in range(0, len(keys), width)
    ]


def context(probabilities, values):
    if not probabilities or not values or len(values) % len(probabilities):
        raise ValueError("positive rectangular context geometry required")
    width = len(values) // len(probabilities)
    # No renormalization, including when the supplied rounded sum differs from 1.
    return [cast16(dot32(probabilities, values[k::width])) for k in range(width)]


def check_context_rounding_witnesses(cases):
    """Compare two exact Fraction equations, not candidate/host FMA results.

    NONFUSED: s' = RN32(s + RN32(p*v)).
    FMA mutation: f' = RN32(f + exact(p*v)).
    Actual fixture operands feed both paths; every rounding uses the independent
    lattice oracle. Explicit intermediate bits certify the disputed last step.
    """
    indexed = {case["name"]: case for case in cases}
    for name, products, separate_steps, fused_steps, outputs in [
        (
            "context_underflow_nondistinguishing_fma_control",
            [power_two(-134), power_two(-150)],
            [0x8000, 0x8000], [0x8000, 0x8000], (0x0000, 0x0000),
        ),
        (
            "context_nonfused_vs_fma_three_term_witness",
            [power_two(-134), power_two(-149), -power_two(-150)],
            [0x8000, 0x8001, 0x8001], [0x8000, 0x8001, 0x8000], (0x0001, 0x0000),
        ),
    ]:
        case = indexed[name]
        probabilities = [decode(int(word, 16), 7) for word in case["probabilities"]]
        values = [decode(int(word, 16), 7) for word in case["values"]]
        assert len(probabilities) == len(values) == len(products)
        actual_products = [p * v for p, v in zip(probabilities, values)]
        assert actual_products == products, name
        separate, fused = 0, 0  # Positive binary32 zero bit words.
        for product, separate_step, fused_step in zip(actual_products, separate_steps, fused_steps):
            rounded_product = rne(product, 23)
            separate = rne(decode(separate, 23) + decode(rounded_product, 23), 23)
            fused = rne(decode(fused, 23) + product, 23)
            assert separate == separate_step, name
            assert fused == fused_step, name
        assert (cast16(separate), cast16(fused)) == outputs, name
        assert case["output"] == [f"{outputs[0]:04x}"], name
    assert rne(-power_two(-150), 23) == 0x80000000  # Separate product is -0.


def expected(case):
    words = lambda name: [int(word, 16) for word in case[name]]
    try:
        if case["kind"] == "score":
            result = score(words("query"), words("keys"), int(case["alpha"], 16))
        elif case["kind"] == "context":
            result = context(words("probabilities"), words("values"))
        else:
            raise ValueError("unknown contraction kind")
        return {"output": [f"{word:04x}" for word in result]}
    except NonFinite:
        return {"error": "NonFinite"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fixture", type=Path, default=(
        Path(__file__).resolve().parents[1] / "tests/fixtures/field_float_attention.json"
    ))
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    assert fixture["case_count"] == len(fixture["cases"])
    check_context_rounding_witnesses(fixture["cases"])
    for case in fixture["cases"]:
        actual = expected(case)
        specified = {key: case[key] for key in ("output", "error") if key in case}
        if args.check and specified != actual:
            raise AssertionError(f"{case['name']}: specified={specified}, oracle={actual}")
        case.pop("output", None)
        case.pop("error", None)
        case.update(actual)
    if args.check:
        print(f"checked {len(fixture['cases'])} exact contraction cases")
    else:
        print(json.dumps(fixture, indent=2))


if __name__ == "__main__":
    main()
