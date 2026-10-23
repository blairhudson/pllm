"""Independent D29 floor/literal oracles; no float, upstream or native calls.

Definitions for normal tests once the parent permits execution. The production
verifier is bounded u128 Rust; unbounded Fraction is used ONLY in these independent
tests through the existing project rational nonlinear-table proof machinery.
Decimal's correctly-rounded exp, widened one neighbor in each direction, is an
additional interval regression oracle, not a substitute for rational evidence.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALUES = ROOT / "crates/pllm-garble/src/fixed_exp_table_values.rs"
RECIPES = (
    ("EXP0", 4096, "exp0", "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b"),
    ("EXP1", 16, "exp1", "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9"),
)


def rational_reference():
    path = ROOT / "paper/reproduction-contracts/fusefss/gate-set/reference.py"
    spec = importlib.util.spec_from_file_location("sigma_exp_independent_rational_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selected_literals(name):
    match = re.search(rf"static {name}: \[u64; 256\] = \[(.*?)\];", VALUES.read_text(), re.S)
    if match is None:
        raise ValueError("missing complete selected literal array")
    return tuple(int(value.strip()) for value in match[1].split(",") if value.strip())


def rational_scaled_interval(reference, index, divisor):
    # Reuse exact positive-exp/Taylor/tail proof, then reverse reciprocal bounds.
    lo, hi = reference.exp_bounds(Fraction(index, divisor))
    return 4096 / hi, 4096 / lo


def decimal_scaled_interval(index, divisor, precision):
    if index == 0:
        return Fraction(4096), Fraction(4096)  # Exact integer boundary, no widening.
    with localcontext() as context:
        context.prec = precision
        # All selected arguments terminate exactly in Decimal before exp.
        argument = -Decimal(index) / divisor
        value = argument.exp()  # Decimal documents correctly-rounded half-even exp.
        return Fraction(context.next_minus(value)) * 4096, Fraction(context.next_plus(value)) * 4096


class SigmaExpFloorOracleTests(unittest.TestCase):
    def test_every_selected_literal_has_existing_exact_rational_floor_certificate(self):
        reference = rational_reference()
        for name, divisor, _, _ in RECIPES:
            values = selected_literals(name)
            self.assertEqual(len(values), 256)
            for index, selected in enumerate(values):
                lower, upper = rational_scaled_interval(reference, index, divisor)
                with self.subTest(table=name, index=index):
                    self.assertEqual(reference.certified_floor(lower, upper), selected)
                    self.assertGreaterEqual(lower, selected)
                    self.assertLess(upper, selected + 1)

    def test_decimal_outward_intervals_independently_resolve_all_same_literals(self):
        reference = rational_reference()
        for name, divisor, _, _ in RECIPES:
            for index, selected in enumerate(selected_literals(name)):
                rational_lo, rational_hi = rational_scaled_interval(reference, index, divisor)
                for precision in (80, 160):
                    lower, upper = decimal_scaled_interval(index, divisor, precision)
                    with self.subTest(table=name, index=index, precision=precision):
                        self.assertEqual(lower // 1, selected)
                        self.assertEqual(upper // 1, selected)
                        self.assertLessEqual(rational_lo, upper)
                        self.assertLessEqual(lower, rational_hi)

    def test_selected_bytes_match_original_fixture_and_exact_declared_hashes(self):
        # This test is runnable alone: finish both families' exact certificates
        # before computing even the first table digest.
        reference = rational_reference()
        for name, divisor, _, _ in RECIPES:
            for index, value in enumerate(selected_literals(name)):
                lo, hi = rational_scaled_interval(reference, index, divisor)
                self.assertEqual(reference.certified_floor(lo, hi), value)
        fixture = json.loads((ROOT / "paper/reproduction-contracts/fusefss/gate-set/fixtures.json").read_text())
        for name, divisor, key, expected in RECIPES:
            self.assertIn(divisor, (16, 4096))
            actual = selected_literals(name)
            self.assertEqual(actual, tuple(fixture["tables"][key]))
            self.assertEqual(sha256(b"".join(struct.pack("<Q", value) for value in actual)).hexdigest(), expected)
            self.assertEqual(fixture["sha256_u64le"][key], expected)

    def test_literal_boundaries_and_exact_zero_are_not_libm_substitutions(self):
        reference = rational_reference()
        for divisor, expected in ((4096, {0: 4096, 1: 4095, 255: 3848}),
                                  (16, {0: 4096, 1: 3847, 133: 1, 134: 0, 255: 0})):
            for index, literal in expected.items():
                lo, hi = rational_scaled_interval(reference, index, divisor)
                self.assertEqual(reference.certified_floor(lo, hi), literal)
                if index == 0:
                    self.assertEqual((lo, hi), (Fraction(4096), Fraction(4096)))
        with self.assertRaises(ArithmeticError):
            reference.certified_floor(Fraction(4095, 4096), Fraction(1))


if __name__ == "__main__":
    unittest.main()
