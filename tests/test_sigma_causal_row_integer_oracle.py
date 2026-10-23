"""Small clear source-paper fixtures only; no native/model execution or approval."""
from __future__ import annotations

from fractions import Fraction
import importlib
import importlib.abc
from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


class NoRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {
            "pllm", "torch", "numpy", "transformers", "tokenizers", "safetensors",
        }:
            raise AssertionError(f"integer fixture imported runtime: {fullname}")
        return None


class SigmaCausalRowIntegerOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = NoRuntimeImports()
        sys.meta_path.insert(0, cls.guard)
        try:
            cls.o = importlib.import_module("benchmarks.research.sigma_causal_row_integer_oracle")
        except BaseException:
            sys.meta_path.remove(cls.guard)
            raise
        expected = Path(__file__).resolve().parents[1] / "benchmarks/research/sigma_causal_row_integer_oracle.py"
        if Path(cls.o.__file__).resolve() != expected:
            sys.meta_path.remove(cls.guard)
            raise AssertionError("fixture resolved outside selected tree")

    @classmethod
    def tearDownClass(cls):
        sys.meta_path.remove(cls.guard)

    def test_reuses_committed_d29_function_and_actual_literal_pins(self):
        from benchmarks.research import sigma_nexp_integer_classes as d29

        self.assertIs(self.o.nexp_lane, d29.lane)
        self.assertEqual(self.o.NEXP_PINS, (
            "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b",
            "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9",
        ))
        self.assertEqual(self.o._TABLES, d29.load_tables())

    def test_complete_q8_inverse_has_zero_sentinel_and_exact_integer_floor(self):
        self.assertEqual(self.o.inverse_q8(0), 0)
        self.assertEqual(
            self.o.inverse_table_sha256(),
            "b03af89917afce4cad1e1923bc09064e7d0b19e7f5b906f7694c3a421c3e8f3d",
        )
        for z in range(1, 256):
            value = self.o.inverse_q8(z)
            self.assertLessEqual(value * z, 1 << 18)
            self.assertLess(1 << 18, (value + 1) * z)
        for bad in (-1, 256, True, Fraction(64)):
            with self.assertRaises(ValueError):
                self.o.inverse_q8(bad)

    def test_bounds_are_all_clear_value_implications_not_masked_value_ranges(self):
        for k, expected_inverse in ((1, 4096), (2, 2048), (3, 1365)):
            bounds = self.o.arithmetic_bounds(k)
            self.assertEqual(bounds["sum"], [4096, 4096 * k])
            self.assertEqual(bounds["denominator"], [64, 64 * k])
            self.assertEqual(bounds["reciprocal"], [expected_inverse, 4096])
            self.assertEqual(bounds["final_product"], [0, 1 << 24])
            self.assertLess(bounds["final_product"][1], 1 << 48)
        for bad in (0, 4, True):
            with self.assertRaises(ValueError):
                self.o.arithmetic_bounds(bad)

    def test_first_row_is_public_4096_independent_of_score_and_inactive_coordinates(self):
        for score in (-(1 << 37), -1, 0, (1 << 37) - 1):
            row = self.o.clear_row((score, (1 << 37) - 1, -(1 << 37)), 1)
            self.assertEqual(row["exp"], [4096, 0, 0])
            self.assertEqual(row["output"], [4096, 0, 0])
            self.assertEqual((row["sum"], row["denominator"], row["reciprocal"]), (4096, 64, 4096))

    def test_equal_three_not_ideal_normalization_and_requires_unsigned_denominator(self):
        for k, expected in ((1, 4096), (2, 2048), (3, 1365)):
            row = self.o.clear_row((5,) * k)
            self.assertEqual(row["output"], [expected] * k)
            self.assertEqual(row["sum"], 4096 * k)
            self.assertEqual(row["denominator"], 64 * k)
        row = self.o.clear_row((0, 0, 0))
        self.assertEqual(sum(row["output"]), 4095)
        self.assertGreaterEqual(row["sum"], 1 << 13)  # not nonnegative signed14
        self.assertGreaterEqual(row["denominator"], 1 << 7)  # not nonnegative signed8

    def test_source_flooring_can_produce_sum_above_4096(self):
        row = self.o.clear_row((0, -4096))
        self.assertEqual(row["delta"], [0, 4096])
        self.assertEqual(row["exp"], [4096, 1506])
        self.assertEqual((row["sum"], row["denominator"], row["reciprocal"]), (5602, 87, 3013))
        self.assertEqual(row["output"], [3013, 1107])
        self.assertEqual(sum(row["output"]), 4120)

    def test_sum_precedes_tr6_and_floor_is_not_a_nearest_integer_round(self):
        row = self.o.clear_row((0, -4096, -4096))
        self.assertEqual(row["exp"], [4096, 1506, 1506])
        self.assertEqual((row["sum"], row["denominator"], row["reciprocal"]), (7108, 111, 2361))
        self.assertEqual(sum(e // 64 for e in row["exp"]), 110)
        self.assertEqual(row["output"], [2361, 868, 868])
        self.assertGreater(Fraction(1 << 18, 111) - 2361, Fraction(1, 2))
        self.assertEqual(self.o.inverse_q8(192), 1365)
        self.assertNotEqual(self.o.inverse_q8(192), 1366)

    def test_d29_byte_carry_increase_survives_final_row_floor(self):
        before, after = (self.o.clear_row((0, -x)) for x in (1279, 1280))
        self.assertEqual(before["coordinate"], [0, 1279])
        self.assertEqual(after["coordinate"], [0, 1280])
        self.assertEqual(before["exp"], [4096, 2995])
        self.assertEqual(after["exp"], [4096, 2996])
        self.assertEqual(before["output"], [2383, 1742])
        self.assertEqual(after["output"], [2383, 1743])

    def test_first_d29_zero_and_zero_reset_are_preserved_not_repaired(self):
        for delta, expected in ((31233, 0), (31487, 0), (31488, 1)):
            row = self.o.clear_row((0, -delta))
            self.assertEqual(row["exp"], [4096, expected])
            self.assertEqual(row["output"], [4096, expected])
        self.assertEqual(sum(self.o.clear_row((0, -31488))["output"]), 4097)

    def test_clipping_and_signed38_endpoints_require_signed39_subtraction(self):
        maximum, minimum = (1 << 37) - 1, -(1 << 37)
        row = self.o.clear_row((minimum, maximum))
        self.assertEqual(row["delta"], [(1 << 38) - 1, 0])
        self.assertEqual(row["coordinate"], [65535, 0])
        self.assertEqual(row["output"], [0, 4096])
        self.assertGreater(row["delta"][0], (1 << 37) - 1)
        self.assertLess(row["delta"][0], 1 << 38)
        for delta in (65535, 65536, (1 << 37) - 1):
            clipped = self.o.clear_row((0, -delta))
            self.assertEqual(clipped["coordinate"], [0, 65535])
            self.assertEqual(clipped["exp"], [4096, 0])

    def test_inactive_lanes_never_enter_max_exp_or_denominator(self):
        actual_lane = self.o.nexp_lane
        with patch.object(self.o, "nexp_lane", wraps=actual_lane) as called:
            row = self.o.clear_row((0, -4096, (1 << 37) - 1), 2)
        self.assertEqual(called.call_count, 2)
        self.assertEqual(row["maximum"], 0)
        self.assertEqual(row["delta"], [0, 4096, None])
        self.assertEqual(row["coordinate"], [0, 4096, None])
        self.assertEqual(row["output"], [3013, 1107, 0])
        self.assertEqual(row["sum"], 5602)

    def test_finite_minimum_score_is_not_an_inactive_negative_infinity(self):
        minimum = -(1 << 37)
        excluded = self.o.clear_row((minimum, minimum), 1)
        substituted = self.o.clear_row((minimum, minimum), 2)
        self.assertEqual(excluded["output"], [4096, 0])
        self.assertEqual(substituted["output"], [2048, 2048])
        self.assertNotEqual(excluded["sum"], substituted["sum"])

    def test_tiny_258_row_grid_checks_bound_obligations_not_global_model_claims(self):
        scores = (-(1 << 37), -31488, -1280, 0, 1279, (1 << 37) - 1)
        rows = calls = 0
        for k in (1, 2, 3):
            for values in product(scores, repeat=k):
                row = self.o.clear_row(values)
                rows += 1
                calls += k
                self.assertTrue(all(0 <= d < 1 << 38 for d in row["delta"]))
                self.assertEqual(max(row["exp"]), 4096)
                self.assertTrue(4096 <= row["sum"] <= 4096 * k)
                self.assertTrue(64 <= row["denominator"] <= 64 * k)
                self.assertTrue((1 << 18) // (64 * k) <= row["reciprocal"] <= 4096)
                self.assertTrue(all(0 <= p <= 1 << 24 for p in row["product"]))
                self.assertEqual(row["output"], [p // 4096 for p in row["product"]])
                # Small Fraction check of the separate per-lane final floor.
                for e, p in zip(row["exp"], row["output"]):
                    value = Fraction(e * row["reciprocal"], 4096)
                    self.assertLessEqual(p, value)
                    self.assertLess(value, p + 1)
        self.assertEqual((rows, calls), (258, 726))
        self.assertLess(calls, 4096)

    def test_invalid_domain_and_empty_prefix_reject_before_any_nexp_call(self):
        with patch.object(self.o, "nexp_lane", side_effect=AssertionError("must reject first")):
            for scores, k in (
                ((), None), ((0, 1, 2, 3), None), ((True,), None),
                ((1 << 37,), None), ((-(1 << 37) - 1,), None), (("-inf",), None),
                ((0,), 0), ((0,), 2), ((0,), True),
            ):
                with self.subTest(scores=scores, k=k), self.assertRaises(ValueError):
                    self.o.clear_row(scores, k)


if __name__ == "__main__":
    unittest.main()
