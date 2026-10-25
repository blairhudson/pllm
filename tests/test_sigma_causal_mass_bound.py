"""Independent arithmetic controls; no native/framework/model imports."""

from collections import Counter
import unittest

from benchmarks.research.sigma_causal_mass_bound import (
    _floor_sum,
    _loss_count,
    intrinsic_bounds,
    mass_classes,
    verify_contract,
)


class CausalMassBoundTests(unittest.TestCase):
    def test_euclidean_floor_sum_against_literal_small_sums(self):
        for n in range(12):
            for modulus in range(1, 14):
                for a in range(15):
                    for b in (0, 1, 17, 63):
                        self.assertEqual(
                            _floor_sum(n, modulus, a, b),
                            sum((a * i + b) // modulus for i in range(n)),
                        )

    def test_loss_counts_against_literal_residues_and_short_intervals(self):
        for t in (0, 1, 63, 64, 4095, 4096, 4097, 8128, 8191, 8192):
            for r in (1365, 1379, 1885, 2048, 4096):
                lower, upper = max(0, t - 4096), min(4096, t)
                expected = sum(x * r % 4096 > t * r % 4096 for x in range(lower, upper + 1))
                self.assertEqual(_loss_count(t, r), (upper - lower + 1, expected))

    def test_all_8193_denominators_and_all_ordered_partitions_counted(self):
        report = verify_contract()
        self.assertEqual(report["denominators_checked"], 8193)
        self.assertEqual(report["ordered_partitions_covered"], {1: 1, 2: 8193, 3: 50343937})
        self.assertEqual(report["mass_bounds"], {1: (4096, 4096), 2: (4094, 4159), 3: (4092, 4159)})

    def test_selected_full_partition_classes_against_independent_inclusion_exclusion(self):
        for d in (4096, 4159, 8191, 8192, 8193, 8896, 12160, 12287, 12288):
            r = 262144 // (d // 64)
            remaining = d - 4096
            # Enumerate only two-dimensional fixed-max boundary, not 4097^3 rows.
            classes = Counter()
            for x in range(max(0, remaining - 4096), min(4096, remaining) + 1):
                row = (4096, x, remaining - x)
                mass = sum(e * r // 4096 for e in row)
                classes[mass] += 3
            if d >= 8192:
                mass = sum(e * r // 4096 for e in (4096, 4096, d - 8192))
                classes[mass] -= 3
            if d == 12288:
                classes[3 * r] += 1
            self.assertEqual(mass_classes(3, d), dict(+classes))

    def test_sharp_numeric_witnesses_are_not_nexp_attainability_claims(self):
        report = verify_contract()
        for k in (2, 3):
            for row, wanted in zip(report["witnesses_under_numeric_premises_only"][str(k)], intrinsic_bounds(k)):
                self.assertEqual(len(row), k)
                self.assertEqual(max(row), 4096)
                self.assertTrue(all(0 <= e <= 4096 for e in row))
                d = sum(row)
                r = 262144 // (d // 64)
                self.assertEqual(sum(e * r // 4096 for e in row), wanted)
        self.assertEqual(sum((4096, 63)), 4159)

    def test_coordinate_carry_is_after_sum_and_zero_sentinel_unreachable(self):
        e = (4096, 1506, 1506)
        self.assertEqual(sum(e) // 64, 111)
        self.assertEqual(sum(x // 64 for x in e), 110)
        for d in range(4096, 12289):
            self.assertNotEqual(d // 64, 0)
        self.assertEqual(mass_classes(1, 4096), {4096: 1})

    def test_invalid_premises_rejected_without_any_source_import(self):
        for k in (0, 4, True, 2.0):
            with self.assertRaises(ValueError):
                intrinsic_bounds(k)
        for d in (0, 4095, 12289, True, 4096.0):
            with self.assertRaises(ValueError):
                mass_classes(3, d)
        for args in ((-1, 1, 0, 0), (1, 0, 0, 0), (1, 1, -1, 0)):
            with self.assertRaises(ValueError):
                _floor_sum(*args)


if __name__ == "__main__":
    unittest.main()
