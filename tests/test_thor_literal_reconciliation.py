"""D30 clear scalar correctness tests; stdlib unittest, no model/native imports."""
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import unittest
from unittest.mock import patch

from benchmarks.research import thor_literal_reconciliation as study


def direct_polynomial(coefficients, x):
    # Independent exact rational sum, not either scalar evaluator's grouping.
    return sum((Fraction(c) * x ** i for i, c in enumerate(coefficients)), Fraction())


def direct_candidate(function, candidate, x):
    first, second = (study.P if candidate == "P" else study.A)[function]
    if candidate == "P":
        return direct_polynomial(second, direct_polynomial(first, x))
    first, second = first[::-1], second[::-1]
    if function == "tanh":
        return direct_polynomial(second, direct_polynomial(first, x / 40))
    return x / 2 * (1 + direct_polynomial(second, direct_polynomial(first, x / 64)))


class ThorLiteralTests(unittest.TestCase):
    def test_complete_tuples_and_discrepancies_are_not_repaired(self):
        for source in (study.P, study.A):
            self.assertEqual(tuple(map(len, source["gelu"])), (32, 28))
            self.assertEqual(tuple(map(len, source["tanh"])), (16, 16))
        self.assertEqual(Decimal(study.P["tanh"][1][10]), Decimal("0.27497"))
        self.assertEqual(Decimal(study.A["tanh"][1][10]), Decimal("2.74974898"))
        self.assertEqual(Decimal(study.P["gelu"][0][23]), Decimal("1273.8"))
        self.assertEqual(Decimal(study.A["gelu"][0][23]), Decimal("1273.94360"))
        self.assertEqual(study.A["gelu"][0][0], "-1.06240033e-05")
        self.assertEqual(study.A["gelu"][1][-1], "-9.45171527e-02")

    def test_declared_grids_are_exact_not_rescaled_extra_points(self):
        for function, endpoint, step in (("gelu", 8, Fraction(1, 16)), ("tanh", 40, Fraction(5, 16))):
            actual = tuple(map(Fraction, study.points(function)))
            expected = {step * k for k in range(-128, 129)} | {
                Fraction(sign, 2 ** power) for power in (10, 20) for sign in (-1, 1)}
            self.assertEqual(len(actual), 261)
            self.assertEqual(set(actual), expected)
            self.assertEqual((actual[0], actual[-1]), (-endpoint, endpoint))
        self.assertEqual(2 * 2 * 261 * len(study.MODES), 3132)
        self.assertLess(3132, study.MAX_EVALUATIONS)
        with localcontext() as context:
            context.prec = 2
            self.assertEqual(set(map(Fraction, study.points("tanh"))), expected)

    def test_complete_compositions_against_independent_exact_rational_sum(self):
        for function in ("gelu", "tanh"):
            for candidate in ("P", "A"):
                step = Fraction(1, 8) if function == "gelu" else Fraction(5, 16)
                for x in (Fraction(0), step, -step):
                    exact = direct_candidate(function, candidate, x)
                    with localcontext() as context:
                        context.prec = 180
                        reference = Decimal(exact.numerator) / Decimal(exact.denominator)
                        argument = Decimal(x.numerator) / Decimal(x.denominator)
                        # Loose enough for evaluated polynomial cancellation,
                        # still checks >140 decimal digits against exact algebra.
                        for mode, tolerance in (("decimal80", "1e-60"), ("decimal160", "1e-140")):
                            observed = study.evaluate(function, candidate, argument, mode)
                            self.assertLess(abs(observed - reference), max(abs(reference), Decimal(1)) * Decimal(tolerance))
        self.assertNotEqual(study.evaluate("gelu", "P", Decimal(0), "decimal160"), 0)
        self.assertEqual(study.evaluate("gelu", "A", Decimal(0), "decimal160"), 0)

    def test_stockmeyer_source_binary64_grouping_for_all_three_tuple_lengths(self):
        x = .13
        x2, x4 = x * x, (x * x) * (x * x)
        x8 = x4 * x4
        for literals in (study.A["gelu"][0], study.A["gelu"][1], study.A["tanh"][0]):
            coefficients = tuple(map(float, literals[::-1]))
            babies = []
            for i in range(0, len(coefficients), 4):
                a, b, c, d = coefficients[i:i + 4]
                babies.append(((b * x + c * x2) + x2 * (d * x)) + a)
            first = babies[0] + babies[1] * x4
            first = first + (babies[2] + babies[3] * x4) * x8
            if len(babies) == 4:
                expected = first
            else:
                last = babies[6] if len(babies) == 7 else babies[6] + babies[7] * x4
                last = (babies[4] + babies[5] * x4) + last * x8
                expected = first + last * (x8 * x8)
            self.assertEqual(study.stockmeyer(coefficients, x).hex(), expected.hex())

    def test_nonfinite_values_are_classified_not_clipped(self):
        observed = study.evaluate("gelu", "P", Decimal(8), "binary64")
        self.assertFalse(study.finite(observed))
        self.assertIn(study.classification(observed), ("positive_infinity", "negative_infinity", "nan"))
        self.assertTrue(study.finite(study.evaluate("gelu", "P", Decimal(8), "decimal160")))
        self.assertEqual(study.classification(float("nan")), "nan")
        self.assertEqual(study.classification(Decimal("-Infinity")), "negative_infinity")

    def test_nonfinite_and_zero_reference_error_pairs_remain_explicit(self):
        summary = {}
        study._difference(summary, Decimal("Infinity"), Decimal(1), Decimal(8))
        self.assertEqual(summary, {"nonfinite_pairs": 1})
        study._difference(summary, Decimal(1), Decimal(0), Decimal(0))
        self.assertEqual(summary["finite_pairs"], 1)
        self.assertEqual(summary["zero_reference_pairs"], 1)
        self.assertEqual(summary["max_absolute"]["value"], "1")
        self.assertNotIn("max_relative", summary)  # No invented denominator floor.

    def test_approved_evaluation_budget_cannot_be_raised(self):
        with self.assertRaises(ValueError):
            study.Budget(evaluations=5001)
        budget = study.Budget(evaluations=1)
        budget.count()
        with self.assertRaises(study.StudyLimit):
            budget.count()
        self.assertEqual(budget.evaluations, 1)

    def test_wall_memory_and_output_limits_stop_not_expand(self):
        budget = study.Budget()
        with patch.object(study.time, "monotonic", return_value=budget.started + 31):
            with self.assertRaises(study.StudyLimit):
                budget.check()
        with patch.object(study, "peak_rss_bytes", return_value=(64 << 20) + 1):
            with self.assertRaises(study.StudyLimit):
                budget.check()
        with self.assertRaises(study.StudyLimit):
            study.encode_output({"oversize": "x" * (1 << 20)})

    def test_output_keeps_decimal_strings_and_accounts_for_actual_encoded_bytes(self):
        payload = {"values": ["Infinity", "NaN", "1.234567890123456789e+700"], "resources": {"output_bytes": 0}}
        encoded = study.encode_output(payload)
        observed = json.loads(encoded)
        self.assertEqual(observed["values"], payload["values"])
        self.assertEqual(observed["resources"]["output_bytes"], len(encoded))


if __name__ == "__main__":
    unittest.main()
