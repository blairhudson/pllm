"""Small D29 enclosure witnesses; no repeated table/full-class/Decimal sweep."""

from fractions import Fraction
import importlib.util
import sys
import unittest

from benchmarks.research import sigma_nexp_integer_classes as prior


S = 4096
CODE_BOUND = Fraction(1489, 512)
REAL_BOUND = Fraction(1489, 2097152)


def rational_reference():
    # Existing project-authored Taylor/tail math only, not an upstream import.
    path = prior.ROOT / "paper/reproduction-contracts/fusefss/gate-set/reference.py"
    name = "sigma_nexp_small_enclosure_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SigmaNexpErrorEnclosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = prior.load_tables()  # Identity only; prior512 floor proofs reused.

    def test_independent_fraction_cells_include_product_floor_residual(self):
        # Sparse witnesses only. Global maximum was checked by the existing
        # completed65536-class tests, not established anew by this small subset.
        for c in (0, 1, 255, 256, 272, 1279, 1280, 2690, 31233, 31488, 65535):
            lo, hi = c % 256, c // 256
            a, b = self.tables[0][lo], self.tables[1][hi]
            e, residual = divmod(a * b, S)
            lower, upper, upper_open = prior.factor_floor_enclosure(c, self.tables)
            exact_lower = Fraction(a) * Fraction(b) / S
            exact_upper = Fraction(S if lo == 0 else a + 1) * Fraction(S if hi == 0 else b + 1) / S
            self.assertEqual((lower, upper), (exact_lower, exact_upper))
            self.assertEqual(lower - e, Fraction(residual, S))
            self.assertEqual(upper_open, c != 0)
            self.assertLessEqual(upper - e, CODE_BOUND)
        row = prior.lane(272, self.tables)
        self.assertEqual((row["u0"], row["u1"], row["p"], row["e"], row["remainder"]),
                         (4080, 3847, 15695760, 3831, 3984))
        lower, upper, _ = prior.factor_floor_enclosure(272, self.tables)
        self.assertEqual(lower - row["e"], Fraction(249, 256))
        self.assertEqual(upper - row["e"], CODE_BOUND)
        self.assertEqual(CODE_BOUND / S, REAL_BOUND)
        self.assertLess(CODE_BOUND, 3)

    def test_four_small_rational_exp_witnesses_not_a_baseline_sweep(self):
        reference = rational_reference()
        for c in (0, 272, 31233, 65535):
            # Exact rational exp(+x) Taylor/tail, reciprocal bounds for exp(-x).
            exp_lower, exp_upper = reference.exp_bounds(Fraction(c, S))
            ideal_lower, ideal_upper = S / exp_upper, S / exp_lower
            row = prior.lane(c, self.tables)
            cell_lower, cell_upper, upper_open = prior.factor_floor_enclosure(c, self.tables)
            self.assertLessEqual(cell_lower, ideal_lower)
            self.assertLessEqual(ideal_upper, cell_upper)
            self.assertGreaterEqual(ideal_lower - row["e"], 0)
            self.assertLess(ideal_upper - row["e"], CODE_BOUND)
            if upper_open:
                self.assertLess(ideal_upper, cell_upper)
            else:
                self.assertEqual((ideal_lower, ideal_upper, row["e"]), (S, S, S))

    def test_zero_tail_discharges_saturation_extension_not_generic_clamp_rule(self):
        self.assertEqual((self.tables[0][255], self.tables[1][255]), (3848, 0))
        tail_lower, tail_upper, upper_open = prior.factor_floor_enclosure(65535, self.tables)
        self.assertEqual((tail_lower, tail_upper, upper_open), (Fraction(0), Fraction(3849, S), True))
        self.assertLess(tail_upper, CODE_BOUND)
        for x in (65535, 65536, 65537, (1 << 38) - 1):
            row = prior.lane(x, self.tables)
            self.assertEqual((row["c16"], row["p"], row["e"]), (65535, 0, 0))
            # Ideal exp decreases in its argument. No enormous-argument exp
            # evaluation is needed or substituted at the root-domain endpoint.
            self.assertGreaterEqual(Fraction(x, S) - Fraction(row["c16"], S), 0)

    def test_carry_and_zero_reset_are_not_monotonicity_or_relative_quality_proof(self):
        self.assertEqual(tuple(prior.lane(x, self.tables)["e"] for x in (1279, 1280)), (2995, 2996))
        self.assertEqual(tuple(prior.lane(x, self.tables)["e"] for x in (31487, 31488)), (0, 1))
        self.assertEqual(prior.lane(31233, self.tables)["e"], 0)
        # For any positive ideal value at a zero output, relative error is1.
        for positive in (Fraction(1, 10**12), Fraction(1, S), Fraction(3, 2*S)):
            self.assertEqual((positive - 0) / positive, 1)

    def test_true_value_scale_and_only_gap50_quarter_premise(self):
        maximum_product = S * S
        self.assertEqual(maximum_product, 1 << 24)
        self.assertLess(maximum_product, 1 << 48)
        self.assertLessEqual(maximum_product + 1, 1 << 48)  # Complete half-open interval.
        self.assertEqual(prior.lane(0, self.tables)["p"], maximum_product)
        for x in (0, 65535, 65536, (1 << 38) - 1):
            row = prior.lane(x, self.tables)
            self.assertTrue(-(1 << 39) <= row["a40_signed"] < 1 << 39)
            self.assertTrue(0 <= row["p"] <= maximum_product)
            self.assertEqual(row["e"], row["p"] // (1 << 12))
            self.assertEqual(row["j1"], row["c16"] // (1 << 8))
        self.assertEqual(prior.lane(32768, self.tables)["j1"], 128)
        # A common masked ring50 word may exceed the entire small secret range.
        self.assertGreater((1 << 50) - 1, maximum_product)


if __name__ == "__main__":
    unittest.main()
