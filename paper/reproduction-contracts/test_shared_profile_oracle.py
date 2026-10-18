"""Conformance and counterexamples for the source-resolution companion.

Run: python3 -B -m unittest discover -s paper/reproduction-contracts
     -p 'test_shared_profile_oracle.py' -v
"""

from bisect import bisect_right
from collections import Counter
from fractions import Fraction
from itertools import repeat
import unittest

import shared_profile_oracle as o


class PumaTruncation(unittest.TestCase):
    def test_exhaustive_mask_error_law(self):
        # Compare reconstructed high-bit protocol equation to an independent
        # low-residue carry identity, including every negative/gap endpoint.
        for bits in range(4, 9):
            n = 1 << bits
            for drop in range(1, bits - 1):
                unit = 1 << drop
                for x in range(-n // 4, n // 4):
                    counts = Counter()
                    for mask in range(n):
                        actual = o.puma_trunc(x, bits, drop, mask)
                        expected = x // unit + int(x % unit + mask % unit >= unit)
                        self.assertEqual(actual, expected)
                        counts[actual - x // unit] += 1
                    self.assertEqual(counts[1], (x % unit) * n // unit)

    def test_gap_rejection_and_exact_multiples(self):
        for x in (-65, 64):
            with self.assertRaises(ValueError):
                o.puma_trunc(x, 8, 3, 0)
        self.assertEqual({o.puma_trunc(-16, 8, 3, r) for r in range(256)}, {-2})
        self.assertEqual(Counter(o.puma_trunc(-9, 8, 3, r) for r in range(256)), {-2: 32, -1: 224})

    def test_pinned_hal_zero_mask_tapes(self):
        inputs = [65536, 131072, 262144, 524288, 1048576]
        self.assertEqual([o.puma_reciprocal(x, repeat(0)) for x in inputs],
                         [1048570, 524285, 262142, 131070, 65535])
        self.assertEqual([o.puma_rsqrt(x, repeat(0)) for x in inputs],
                         [524249, 370660, 262124, 185330, 131062])
        for fn in (o.puma_reciprocal, o.puma_rsqrt):
            with self.assertRaises(ValueError):
                fn(0, repeat(0))

    def test_approved_rmsnorm_epsilon_zero_rows_and_no_centering(self):
        one = 1 << 18
        self.assertEqual(o.encode_f32("1e-6", 18), 0)
        self.assertEqual(o.encode_f32(Fraction(1, one), 18), 1)
        for width in (1, 3, 128):
            for mask in (0, one - 1):
                self.assertEqual(o.puma_rmsnorm([0] * width, [one] * width, repeat(mask)), [0] * width)
        # Centered LayerNorm would produce zero. Unit gamma preserves the
        # separately normalized result; negative gamma negates it exactly.
        positive = o.puma_rmsnorm([2 * one] * 4, [one] * 4, repeat(0))
        self.assertTrue(all(one * 99 // 100 < v < one * 101 // 100 for v in positive))
        self.assertEqual(o.puma_rmsnorm([2 * one] * 4, [-one] * 4, repeat(0)), [-v for v in positive])
        with self.assertRaises(ValueError):
            o.puma_rmsnorm([], [], repeat(0))
        with self.assertRaises(ValueError):
            o.puma_rmsnorm([one], [], repeat(0))


class BumbleBeeShares(unittest.TestCase):
    def test_truncation_conditioned_distribution(self):
        bits, drop = 8, 3
        n, unit = 1 << bits, 1 << drop
        for x in range(-n // 4, n // 4):
            counts = Counter()
            for share0 in range(n):
                carry = int(share0 % unit > x % unit)
                parity = ((share0 + n // 2) % n >> drop) & 1
                expected = x // unit - carry + parity
                for wrap_share in (0, 1, 37, 255):
                    self.assertEqual(o.bumblebee_trunc(x, share0, wrap_share, bits, drop), expected)
                counts[expected - x // unit] += 1
            self.assertEqual(counts[0], n // 2)
            self.assertEqual(counts[-1], (unit - 1 - x % unit) * n // (2 * unit))
            self.assertEqual(counts[1], (x % unit + 1) * n // (2 * unit))

    def test_comparison_pivot_depends_on_shares(self):
        # f=18, retain four fractional bits => drop14. Smaller equivalent ring
        # demonstrates the exact same missing low carry at equality.
        counts = Counter(o.bumblebee_less(0, 0, a, 8, 4) for a in range(256))
        self.assertEqual(counts, {0: 16, 1: 240})
        for x in range(-64, 64):
            for a in range(256):
                expected = x // 16 - int(a % 16 > x % 16) < 0
                self.assertEqual(o.bumblebee_less(x, 0, a, 8, 4), expected)

    def test_positive_offset_extension_all_shares(self):
        for x in range(-64, 64):
            for a in range(256):
                self.assertEqual(o.positive_extension(x, a, 8, 16), x)
        for x in (-65, 64):
            with self.assertRaises(ValueError):
                o.positive_extension(x, 0, 8, 16)

    def test_lift_tie_and_bole_error_support(self):
        self.assertEqual(o.lift(8, 4, 257), 129)  # 128.5, NOT ties-even 128
        errors = set()
        for x in range(16):
            for y in range(16):
                for r in range(257):
                    reconstructed = o.down(o.lift(x, 4, 257) * y + r, 4, 257) - o.down(r, 4, 257)
                    error = o.signed(reconstructed - x * y, 4)
                    self.assertIn(error, (-1, 0, 1))
                    errors.add(error)
        self.assertEqual(errors, {-1, 0, 1})

    def test_source_parameter_factory_numbers(self):
        olt = [576460752303210497, 36028797017571329, 562949952847873, 36028797018652673]
        bole = [1152921504606830593, 576460752303210497, 1099511480321]
        plains = [140737488273409, 140737488125953, 140737488044033]
        self.assertEqual(o.coefficient_moduli([59, 55, 49, 55]), olt)
        self.assertEqual(o.coefficient_moduli([60, 59, 40]), bole)
        self.assertEqual(sorted(o.coefficient_moduli([47, 47, 47]), reverse=True), plains)
        for p in set(olt + bole + plains):
            self.assertEqual(p % 16384, 1)
            self.assertTrue(o.prime_u64(p))
        for composite in (1, 4, 341, 2047, 3215031751, 3825123056546413051):
            self.assertFalse(o.prime_u64(composite))


class FuseCompilerAlgebra(unittest.TestCase):
    def test_every_small_partition_mask_and_input(self):
        # All 128 full-domain partitions on ring8, including singleton/full
        # partitions, boundary hits, wrap splits and deterministic padding.
        n = 8
        for subset in range(1 << (n - 1)):
            boundaries = [0] + [i for i in range(1, n) if subset >> (i - 1) & 1] + [n]
            m = len(boundaries) - 1
            for r in range(n):
                b, order = o.fuse_partition(boundaries, r)
                self.assertEqual((len(b), len(order)), (min(m + 1, n) + 1, min(m + 1, n)))
                self.assertTrue(all(a < c for a, c in zip(b, b[1:])))
                for x in range(n):
                    self.assertEqual(order[bisect_right(b, (x + r) % n) - 1],
                                     bisect_right(boundaries, x) - 1)

    def test_all_predicates_masks_and_low_views(self):
        for bits in range(1, 6):
            n = 1 << bits
            for x in range(n):
                for r in range(n):
                    for beta in range(n + 1):
                        self.assertEqual(o.fuse_predicate((x + r) % n, r, beta, bits), x < beta)
                    for low_bits in range(1, bits):
                        for gamma in range((1 << low_bits) + 1):
                            self.assertEqual(o.fuse_predicate((x + r) % n, r, gamma, low_bits),
                                             x % (1 << low_bits) < gamma)


class SigmaPaperProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = o.sigma_tables()

    def test_table_floors_stable_under_extra_precision(self):
        self.assertEqual(self.tables, o.sigma_tables(96))
        expected = {
            "exp0": "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b",
            "exp1": "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9",
            "gelu": "b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b",
            "silu": "ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f",
        }
        for name, table in self.tables.items():
            self.assertEqual(o.sha256(b"".join(o.struct.pack("<Q", x) for x in table)).hexdigest(), expected[name])

    def test_early_truncation_and_softmax_counterexamples(self):
        self.assertEqual(o.sigma_activation(-1, self.tables["gelu"]), -31)
        self.assertEqual(o.sigma_activation(63, self.tables["gelu"]), 63)
        self.assertEqual(o.sigma_activation(64, self.tables["gelu"]), 33)
        self.assertEqual(o.sigma_softmax([0, -4096], self.tables), [3013, 1107])
        self.assertEqual(o.sigma_softmax([-81920, -81920], self.tables), [2048, 2048])
        self.assertEqual(o.sigma_softmax([0, -65536], self.tables), [4096, 0])

    def test_norm_width_and_rational_table(self):
        self.assertEqual(o.sigma_norm([-4096, -4096, 4096, 4096]), [-4096, -4096, 4096, 4096])
        self.assertEqual(o.sigma_norm([4096] * 4), [0] * 4)
        # Figure 12 floors 4096/768 to 5; replacing with exact mean is a change.
        self.assertEqual(o.sigma_norm([4096] * 768), [4096] * 768)
        for k in (4, 768):
            for index in range(8192):
                r = o.sigma_rsqrt_table_entry(index, k)
                m, e = divmod(index, 64)
                a, b = (1 << 55) * k, (128 + m) << e
                self.assertLessEqual(r * r * b, a)
                self.assertGreater((r + 1) ** 2 * b, a)


class SourceConflicts(unittest.TestCase):
    def test_encoding_and_epsilon_are_not_interchangeable(self):
        x = Fraction(-3, 2)
        self.assertEqual([o.encode(x, 0, p) for p in ("floor", "toward_zero", "nearest_even", "nearest_up")], [-2, -1, -2, -1])
        self.assertEqual(o.encode("0.000001", 18, "toward_zero"), 0)
        self.assertEqual(o.encode("0.00001", 18, "floor"), 2)
        self.assertEqual(o.encode("0.00001", 16, "toward_zero"), 0)
        # Binary32 c2 at Q18 differs from exact decimal truncation by one.
        self.assertEqual(o.encode("38.4714796", 18, "toward_zero"), 10085067)
        self.assertEqual(o.encode_f32("38.4714796", 18), 10085068)

    def test_puma_pseudocode_conflicts(self):
        # Equal logits -20: literal unshifted cutoff yields zero outputs;
        # shifted-before-sum intent has two ones before normalization.
        row = [-20, -20]
        self.assertEqual([int(x >= -14) for x in row], [0, 0])
        self.assertEqual([int(x - max(row) > -14) for x in row], [1, 1])
        # [-1,-1,1,1]: sum of squares 4 versus mean square 1.
        self.assertEqual(sum(x * x for x in (-1, -1, 1, 1)), 4)
        self.assertEqual(Fraction(1, 2), Fraction(1, o.isqrt(4)))


if __name__ == "__main__":
    unittest.main()
