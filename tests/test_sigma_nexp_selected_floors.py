"""Exact public floor regression using existing rational proof machinery.

Mathematical fixture evidence, not a native Verified token, source-authentication
grant, profile registration or material-issuance path.
"""

from fractions import Fraction
import importlib.util
import json
import sys
import unittest

from benchmarks.research import sigma_nexp_integer_classes as study


class SigmaNexpSelectedFloorsTests(unittest.TestCase):
    def test_selected_fixture_literals_have_existing_exact_rational_floor_cells(self):
        path = study.ROOT / "paper/reproduction-contracts/fusefss/gate-set/reference.py"
        name = "sigma_nexp_research_selected_floor_reference"
        spec = importlib.util.spec_from_file_location(name, path)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[name] = reference
        spec.loader.exec_module(reference)
        fixture = json.loads(study.FIXTURE.read_text())
        tables = tuple(tuple(fixture["tables"][key]) for key in ("exp0", "exp1"))
        for values, divisor in zip(tables, (4096, 16)):
            self.assertEqual(len(values), 256)
            for index, selected in enumerate(values):
                lo, hi = reference.exp_bounds(Fraction(index, divisor))
                # Positive-exp enclosure, reciprocal order reversed exactly.
                lower, upper = 4096 / hi, 4096 / lo
                with self.subTest(divisor=divisor, index=index):
                    self.assertEqual(reference.certified_floor(lower, upper), selected)
                    self.assertGreaterEqual(lower, selected)
                    self.assertLess(upper, selected + 1)
        # Complete both mathematical families before computing identity digests;
        # these public-byte hashes still issue no native/material authority.
        self.assertEqual(tables, study.load_tables())


if __name__ == "__main__":
    unittest.main()
