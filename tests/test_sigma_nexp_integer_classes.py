"""Exact finite D29 clear-map regressions, no native/model/exp evaluation."""

from fractions import Fraction
import json
import unittest
from unittest.mock import patch

from benchmarks.research import sigma_nexp_integer_classes as study


class SigmaNexpIntegerClassesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = study.load_tables()
        cls.report = study.sweep(cls.tables)
        cls.direct = tuple(cls.tables[0][c & 255] * cls.tables[1][c >> 8] // 4096
                           for c in range(65536))

    def test_exact_selected_literals_not_an_exp_substitute(self):
        self.assertEqual(tuple(map(len, self.tables)), (256, 256))
        self.assertEqual(tuple((min(row), max(row)) for row in self.tables), ((3848, 4096), (0, 4096)))
        self.assertEqual(self.tables[1][133:135], (1, 0))
        self.assertEqual(self.report["first_zero_high_table_index"], 134)
        wrong = json.loads(study.FIXTURE.read_text())
        wrong["tables"]["exp0"][1] -= 1
        with patch.object(study.Path, "read_text", return_value=json.dumps(wrong)):
            with self.assertRaises(ValueError):
                study.load_tables()

    def test_all65536_graph_values_match_independent_fraction_floor(self):
        for c, expected in enumerate(self.direct):
            result = study.lane(c, self.tables)
            rational_product = Fraction(self.tables[0][c & 255]) * self.tables[1][c >> 8] / 4096
            self.assertEqual(result["e"], rational_product.numerator // rational_product.denominator)
            self.assertEqual(result["e"], expected)
            self.assertEqual((result["c16"], result["j0"], result["j1"]), (c, c & 255, c >> 8))
            self.assertTrue(0 <= result["p"] <= 1 << 24)
            self.assertTrue(0 <= expected <= 4096)
            self.assertLess(result["p"], 1 << 48)  # GapARS50 quarter premise.
        self.assertEqual(self.report["product_range_inclusive"], [0, 16777216])
        self.assertEqual(self.report["output_range_inclusive"], [0, 4096])

    def test_byte_carry_increases_are_source_semantics_not_repaired(self):
        increases = [i for i in range(1, 65536) if self.direct[i] > self.direct[i - 1]]
        self.assertEqual(len(increases), 70)
        self.assertTrue(all(i % 256 == 0 for i in increases))
        self.assertTrue(all(self.direct[i] - self.direct[i - 1] == 1 for i in increases))
        self.assertEqual((self.direct[1279], self.direct[1280]), (2995, 2996))
        self.assertFalse(self.report["global_nonincreasing"])
        self.assertTrue(self.report["within_byte_block_nonincreasing"])
        self.assertEqual(self.report["delta_histogram"], {"-1": 4166, "0": 61299, "1": 70})
        self.assertEqual(sum(self.report["delta_histogram"].values()), 65535)

    def test_zero_islands_and_terminal_zero_have_distinct_boundaries(self):
        self.assertEqual(sum(v == 0 for v in self.direct), 34292)
        self.assertEqual(self.report["first_zero"], 31233)
        self.assertEqual((self.direct[31487], self.direct[31488]), (0, 1))
        resets = sum(a == 0 and b > 0 for a, b in zip(self.direct, self.direct[1:]))
        self.assertEqual(resets, 11)
        self.assertEqual(self.report["zero_to_positive_transitions"], 11)
        self.assertEqual(self.report["last_nonzero"], 34048)
        self.assertEqual(self.report["terminal_zero_start"], 34049)
        self.assertEqual(self.direct[34048], 1)
        self.assertTrue(all(v == 0 for v in self.direct[34049:]))
        reconstructed = {c for lo, hi in self.report["zero_runs_half_open"] for c in range(lo, hi)}
        self.assertEqual(reconstructed, {c for c, v in enumerate(self.direct) if v == 0})

    def test_final_gap_floor_ties_and_exact_fraction_remainders(self):
        ties = []
        max_gap = Fraction(0)
        for c in range(65536):
            p = self.tables[0][c & 255] * self.tables[1][c >> 8]
            gap = Fraction(p, 4096) - self.direct[c]
            self.assertTrue(0 <= gap < 1)
            max_gap = max(max_gap, gap)
            if gap == Fraction(1, 2):
                ties.append(c)
        self.assertEqual(ties, [2690, 10370, 15424, 15556, 18306, 22658])
        self.assertEqual(self.report["half_code_tie_count"], 6)
        self.assertEqual(max_gap, Fraction(4095, 4096))
        self.assertEqual(self.report["max_product_floor_gap_codes"], study.rational(max_gap))
        first = study.lane(2690, self.tables)
        self.assertEqual(Fraction(first["p"], 4096), Fraction(4247, 2))
        self.assertEqual(first["e"], 2123)  # NOT nearest-even2124.

    def test_conditional_enclosure_is_exact_algebra_not_observed_exp_error(self):
        maximum = Fraction(0)
        maximizers = []
        for c in range(65536):
            lo, hi = c & 255, c >> 8
            a, b = self.tables[0][lo], self.tables[1][hi]
            lower, upper, upper_open = study.factor_floor_enclosure(c, self.tables)
            # Independent positive interval corner multiplication. Exact exp0
            # factors are4096; all other factors have certified unit floor cells.
            a_upper = Fraction(4096) if lo == 0 else Fraction(a + 1)
            b_upper = Fraction(4096) if hi == 0 else Fraction(b + 1)
            self.assertEqual(lower, Fraction(a) * b / 4096)
            self.assertEqual(upper, a_upper * b_upper / 4096)
            self.assertEqual(upper_open, c != 0)
            self.assertTrue(self.direct[c] <= lower <= upper)
            bound = upper - self.direct[c]
            if bound > maximum:
                maximum, maximizers = bound, [c]
            elif bound == maximum:
                maximizers.append(c)
        self.assertEqual(maximum, Fraction(1489, 512))
        self.assertEqual(maximizers, [272])
        enclosure = self.report["conditional_clipped_ideal_error"]
        self.assertEqual(enclosure["upper_codes"], study.rational(maximum))
        self.assertEqual(enclosure["upper_real"], study.rational(maximum / 4096))
        self.assertEqual(study.factor_floor_enclosure(0, self.tables), (Fraction(4096), Fraction(4096), False))

    def test_clipping_drelu_and_unsigned_tr_boundary_cases(self):
        points = {0, 1, 65534, 65535, 65536, 65537, (1 << 38) - 1}
        for high in range(1, 256):
            points.update((high * 256 - 1, high * 256, high * 256 + 1))
        self.assertEqual(len(points), 772)
        self.assertEqual(self.report["boundary_graph_cases"], len(points))
        for x in points:
            result = study.lane(x, self.tables)
            c = min(x, 65535)
            self.assertEqual(result["c16"], c)
            self.assertEqual(result["e"], self.direct[c])
            self.assertEqual(result["j1"], c // 256)
            self.assertTrue(-65536 <= result["a40_signed"] < (1 << 38) - 65536)
        self.assertEqual(study.lane(65536, self.tables)["h"], 1)
        self.assertEqual(study.lane(32768, self.tables)["j1"], 128)
        for x in (-1, 1 << 38, True, 0.0):
            with self.assertRaises(ValueError):
                study.lane(x, self.tables)


if __name__ == "__main__":
    unittest.main()
