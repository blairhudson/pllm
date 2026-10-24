"""D43 width-four expected arithmetic, not protected/native conformance.

Reuse the historical independent oracle with explicit signed-12 affine values.
Its wider width/default-gamma domain is NOT D43 admission. Malformed native
inputs, actual resource quotes, shares, transcripts and custody need separate
native tests. No source, security or SDK qualification follows from these checks.

Run: python3 -I -B paper/reproduction-contracts/test_sigma_layernorm4_d43.py -q
"""

import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "_d43_shared_profile_oracle", Path(__file__).with_name("shared_profile_oracle.py")
)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)

ROWS = (
    (0, 0, 0, 0),
    (4095, 4095, 4095, 4095),
    (-4096, -4096, -4096, -4096),
    (-1, -1, 0, 0),  # S=3
    (-1, 0, 0, 0),  # S=4
    (-2, 0, 0, 0),  # S=5
    (-4, -4, 4, 3),  # S=60
    (-4, -4, 4, 4),  # S=65
    (-4096, 4095, -1, 0),
    (-3, 0, 1, 3),
)
AFFINE_PAIRS = (
    ((-4096, -1, 1, 4095), (4095, -4096, 1, -1)),
    ((4095, 2048, -2048, -4096), (-1, 1, -4096, 4095)),
)


class LayerNorm4D43(unittest.TestCase):
    def test_twenty_explicit_affine_rows_and_signed_floor_certificates(self):
        count = 0
        for row in ROWS:
            for gamma, beta in AFFINE_PAIRS:
                with self.subTest(row=row, gamma=gamma, beta=beta):
                    self.assertTrue(all(-4096 <= x <= 4095 for x in row+gamma+beta))
                    expected = ORACLE.sigma_norm(row, gamma, beta)
                    self.assertEqual(len(expected), 4)
                    mean = sum(row)//4
                    self.assertLessEqual(mean*4, sum(row))
                    self.assertLess(sum(row), (mean+1)*4)
                    centered = tuple(x-mean for x in row)
                    s = sum(x*x for x in centered)+1
                    e = s.bit_length()-1
                    m = (128*s//(1 << e))-128
                    inverse = ORACLE.sigma_rsqrt_table_entry(64*m+e, 4)
                    for c, g, b, y in zip(centered, gamma, beta, expected):
                        first = c*inverse//4096
                        self.assertLessEqual(first*4096, c*inverse)
                        self.assertLess(c*inverse, (first+1)*4096)
                        second = y-b
                        self.assertLessEqual(second*4096, first*g)
                        self.assertLess(first*g, (second+1)*4096)
                    if len(set(row)) == 1:
                        self.assertEqual(expected, list(beta))
                    count += 1
        self.assertEqual(count, 20)

    def test_two_literal_analytic_controls(self):
        # The first deliberately duplicates one of the twenty affine cases.
        # Both have S=4 and inverse=2^24, so first=4096*centered and
        # second=centered*gamma exactly. These literals require no table
        # reimplementation or host sqrt/division.
        gamma, beta = AFFINE_PAIRS[0]
        controls = (
            ((-1, 0, 0, 0), (4095, -4097, 2, 4094)),
            ((-1, 1, 1, 0), (8191, -4097, 2, -1)),
        )
        for row, literal in controls:
            with self.subTest(row=row):
                self.assertEqual(ORACLE.sigma_norm(row, gamma, beta), list(literal))

    def test_structural_ring_coordinate_including_constant_rows(self):
        for row, known_s in zip(ROWS, (1, 1, 1, 3, 4, 5, 60, 65, None, None)):
            with self.subTest(row=row):
                mean = sum(row)//4
                centered = tuple(x-mean for x in row)
                s = sum(c*c for c in centered)+1
                if known_s is not None:
                    self.assertEqual(s, known_s)
                e = s.bit_length()-1
                self.assertLessEqual(1 << e, s)
                self.assertLess(s, 1 << (e+1))
                self.assertLessEqual(e, 49)
                m = 128*s//(1 << e)-128
                u = 1 << (49-e)
                t = (s*u) % (1 << 50)
                self.assertEqual((t >> 42) % 128, m)
                self.assertTrue(0 <= m < 128)
                self.assertEqual(64*m+e, (m << 6) | e)
                if len(set(row)) == 1:
                    self.assertEqual((s, e, u, t, t >> 42, m),
                                     (1, 0, 1 << 49, 1 << 49, 128, 0))

    def test_all_8192_existing_table_entries_by_exact_square_inequalities(self):
        # Includes coordinates unreachable by these rows. None is replaced by
        # zero; the old table-entry function remains the only table evaluator.
        for index in range(8192):
            with self.subTest(index=index):
                m, e = divmod(index, 64)
                d = (128+m) << e
                a = ORACLE.sigma_rsqrt_table_entry(index, 4)
                self.assertIs(type(a), int)
                self.assertGreaterEqual(a, 0)
                self.assertLessEqual(a*a*d, 1 << 57)
                self.assertLess(1 << 57, (a+1)*(a+1)*d)


if __name__ == "__main__":
    unittest.main()
