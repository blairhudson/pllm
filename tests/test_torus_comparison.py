"""Stdlib-only checks; direct-file imports avoid project/native package imports.

Run: python3 -B -m unittest discover -s tests -p test_torus_comparison.py -v
"""

from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest


_path = Path(__file__).resolve().parents[1] / "benchmarks/research/torus_comparison.py"
_spec = importlib.util.spec_from_file_location("torus_comparison_offline", _path)
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


class TorusComparisonTests(unittest.TestCase):
    def test_streaming_functional_sweep(self):
        # This asserts genuine sign/low-digit/GE agreement, NOT merely existence
        # of mismatches. Any recurrence counterexample raises immediately.
        report = oracle.run()
        counts = report["counts"]
        self.assertEqual(counts["ge_cases"], 4096)
        self.assertGreaterEqual(counts["signed_cases"], 4092)
        self.assertLessEqual(counts["cases"], 20_000)
        self.assertEqual(counts["max_pbs"], 13)
        self.assertEqual(counts["max_peels"], 6)
        self.assertEqual(report["recurrence_counterexamples"], 0)

    def test_dispatch_boundaries_and_direct_arithmetic_sign(self):
        for width, bits, peels in ((32, 5, 0), (32, 6, 1), (32, 9, 1),
                                  (32, 10, 2), (64, 5, 0), (64, 9, 1),
                                  (64, 10, 2), (64, 33, 6)):
            for d in (-(1 << (bits - 1)), -1, 0, 1, (1 << (bits - 1)) - 1):
                with self.subTest(width=width, bits=bits, d=d):
                    out, work = oracle.extract(oracle.lattice(d, width, bits), width, bits,
                                               arithmetic=True)
                    self.assertEqual(out, (1 << 31) if d < 0 else 0)
                    self.assertEqual(work.peels, peels)
                    self.assertEqual(work.pbs, 2 * peels + 1)
                    self.assertEqual(work.conversions, int(width == 64))

    def test_invalid_domains_and_no_wrap_admission(self):
        for w, b in ((31, 5), (32, 0), (32, 11), (64, 34)):
            with self.assertRaises(ValueError):
                oracle.lattice(0, w, b)
        with self.assertRaises(TypeError):
            oracle.lattice(True, 32, 5)
        for d in (-65, 64):
            with self.assertRaises(ValueError):
                oracle.lattice(d, 32, 7)
        with self.assertRaises(ValueError):
            oracle.logic_ge(64, -1, 32, 7)
        with self.assertRaises(ValueError):
            oracle.extract(1, 32, 7)
        with self.assertRaises(ValueError):
            oracle.exact_scale32(1, error=0)
        with self.assertRaises(ValueError):
            oracle.exact_scale32(1 << 32, error=1)
        self.assertEqual(oracle.exact_scale32(7 << 32, error=0), 7)

    def test_negacyclic_signs_and_identity_coefficients(self):
        for width, n, t in ((32, 1024, 4), (64, 2048, 5)):
            q = 1 << width
            self.assertEqual(oracle.table_value(0, width, constant=-q // 8), 7 * q // 8)
            self.assertEqual(oracle.table_value(n, width, constant=-q // 8), q // 8)
            for i in (0, 63, 64, n - 1):
                expected = (1 << 12) * (i // 64)
                self.assertEqual(oracle.table_value(i, width, scale_bits=12), expected)
                self.assertEqual(oracle.table_value(i + n, width, scale_bits=12), -expected % q)
            self.assertEqual(n >> t, 64)

    def test_literal_converter_is_not_claimed_arithmetic_contract(self):
        q = 1 << 32
        self.assertEqual(oracle.literal_log_to_ari(q // 8), 3 * q // 8)
        self.assertEqual(oracle.literal_log_to_ari(7 * q // 8), 7 * q // 8)
        # Direct arithmetic MSB is a different path, independently checked.
        self.assertEqual(oracle.direct_msb(0, 32, arithmetic=True), 0)
        self.assertEqual(oracle.direct_msb(q // 2, 32, arithmetic=True), q // 2)
        with self.assertRaises(ValueError):
            oracle.literal_log_to_ari(0)

    def test_component_rounding_independent_control_and_word_wrap(self):
        for n in (1024, 2048):
            q = 1 << 32  # BOTH actual lookups consume lvl0 uint32.
            h = q // (2 * n)
            index, error = oracle.blind_rotation(h // 2, ((h // 2, 1),),
                                                  domain_bits=32, n=n)
            self.assertEqual(index, 2 * n - 1)
            self.assertEqual(error, -1)
            self.assertEqual(((h // 2 - h // 2) % q) // h, 0)
            # Maximal mask rounds through the unsigned word boundary to zero.
            index, error = oracle.blind_rotation(0, ((q - 1, 1),), domain_bits=32, n=n)
            self.assertEqual(index, 0)
            self.assertEqual(error, -Fraction(1, h))
            for a in (0, h // 2 - 1, h // 2, h - 1, q - 1):
                body = h // 3
                index, error = oracle.blind_rotation(body, ((a, 1),), domain_bits=32, n=n)
                exact_phase = Fraction(body - a, h)
                self.assertEqual((exact_phase + error) % (2 * n), index)
                mask_error = Fraction((a + h // 2) // h) - Fraction(a, h)
                self.assertGreater(mask_error, -Fraction(1, 2))
                self.assertLessEqual(mask_error, Fraction(1, 2))

    def test_cell_bound_validates_domain_even_for_zero_error(self):
        for error in (0, 1, -1):
            for c, width in ((1 << 32, 32), (-1, 32), (0, 31), (1 << 64, 64)):
                with self.subTest(c=c, width=width, error=error):
                    with self.assertRaises(ValueError):
                        oracle.same_cell_bound(c, error, width)
        for width in (32, 64):
            for c in (0, (1 << width) - 1):
                self.assertTrue(oracle.same_cell_bound(c, 0, width))
        for error in (True, 0.0, Fraction(0)):
            with self.assertRaises(TypeError):
                oracle.same_cell_bound(0, error, 32)

    def test_conditional_cell_bounds_and_error_recurrence(self):
        for width in (32, 64):
            q = 1 << width
            n, _, divisor = oracle.profile(width)
            h, offset = q // (2 * n), q // divisor
            c = (h // 2 - offset) % q
            self.assertEqual(oracle.circular_margin(c, width, offset=offset), h // 2)
            self.assertTrue(oracle.same_cell_bound(c, h // 2 - 1, width, offset=offset))
            self.assertFalse(oracle.same_cell_bound(c, h // 2, width, offset=offset))
            self.assertNotEqual(oracle.phase_index(c, width, offset=offset),
                                oracle.phase_index((c + h // 2) % q, width, offset=offset))
            boundary = (-offset) % q
            self.assertFalse(oracle.same_cell_bound(boundary, -1, width, offset=offset))
            self.assertTrue(oracle.same_cell_bound(boundary, 0, width, offset=offset))
            self.assertEqual(oracle.error_recurrence(3, 5, 2, 4, width), (48, 43, 1))
            self.assertEqual(oracle.error_recurrence(q // 2, 0, 0, 1, width),
                             (0, 0, -q // 2))
            # Zero final residual error does not establish intermediate cells.
            shifted, before_identity, final = oracle.error_recurrence(h, 0, h, 1, width)
            self.assertEqual(final, 0)
            self.assertFalse(oracle.same_cell_bound(c, shifted, width, offset=offset))
            self.assertFalse(oracle.same_cell_bound(c, before_identity, width, offset=offset))


if __name__ == "__main__":
    unittest.main()
