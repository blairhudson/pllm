"""Independent hand oracles plus explicitly NumPy-required source comparison.

Source-dependent tests deliberately fail (not skip) if NumPy is unavailable.
No NumPy import occurs in this module's helpers or at collection time.
Future command from repository root (not pytest; bypasses tests/conftest.py):
  .venv/bin/python -m unittest discover -s tests -p test_thor_scalar_certificate_experiment.py
"""

from fractions import Fraction
import builtins
import math
import unittest
from unittest.mock import patch
import warnings

from scripts import thor_scalar_certificate_experiment as study


EXPECTED_PROFILES = ((0.15, 10, 1), (0.2, 150, 2), (0.75, 2500, 2))


def step(p=1, q=1, h=3, pre=Fraction(1, 2), post=1):
    """Exact synthetic trace, distinct from the independent source-root oracle."""
    values = {name: Fraction(value).as_integer_ratio() for name, value in (
        ("p", p), ("q", q), ("h", h), ("h_again", h), ("pre", pre),
        ("post", post), ("k", 1), ("stop", Fraction(999, 1000)))}
    return study.Step(1, ((2, 1), (1, 1)), 1, values)


def synthetic_trace(D=(Fraction(1, 2), Fraction(2)), record=None):
    # b=1/768, g=1, d=0 makes D(v)=v without float construction.
    values = {"a": D[0], "M": D[1], "g": 1, "b": Fraction(1, 768),
              "d": 0, "en0": Fraction(1, 2), "stop": Fraction(999, 1000)}
    return study.Trace((0, 0, 1), "synthetic-exact",
                       {name: Fraction(value).as_integer_ratio()
                        for name, value in values.items()},
                       [record or step()], completed=True)


class ExactOracleTests(unittest.TestCase):
    def test_fixed_profile_values_literal_types_and_dimension(self):
        self.assertIs(type(study.PROFILES), tuple)
        self.assertEqual(len(study.PROFILES), 3)
        self.assertEqual(study.PROFILES, EXPECTED_PROFILES)
        for actual, expected in zip(study.PROFILES, EXPECTED_PROFILES):
            self.assertIs(type(actual), tuple)
            self.assertEqual(len(actual), 3)
            self.assertEqual(actual, expected)
            self.assertEqual(tuple(type(value) for value in actual), (float, int, int))
        self.assertIs(type(study.N), int)
        self.assertEqual(study.N, 768)

    def test_hand_cubic_ranges_include_both_stationary_points(self):
        for bounds, expected in (
            ((0, Fraction(1, 2)), (0, Fraction(25, 8))),
            ((Fraction(1, 2), 2), (2, 4)),
            ((2, 4), (0, 4)),
            ((0, 4), (0, 4)),
        ):
            with self.subTest(bounds=bounds):
                self.assertEqual(study.cubic_range(1, 3, *bounds), expected)

    def test_sqrt_hand_oracles_and_exact_upper(self):
        self.assertEqual(study.sqrt_bracket(0), (0, 0))
        self.assertEqual(study.sqrt_bracket(Fraction(9, 16)),
                         (Fraction(3, 4), Fraction(3, 4)))
        self.assertEqual(study.sqrt_bracket(2, bits=2),
                         (Fraction(5, 4), Fraction(3, 2)))
        self.assertEqual(study.sqrt_bracket(1), (1, 1))
        with self.assertRaises(study.Refused):
            study.sqrt_bracket(-1)

    def test_q_not_p_squared_identity_and_simultaneous_old_A(self):
        A, R, P = study.exact_step(1, 1, 1, p=1, q=2, h=3)
        self.assertEqual((A, R, P), (8, 2, 2))
        self.assertEqual(A/P, R**2)  # D=1, rho=2, not sqrt(8).
        result = study.direct_enclosure((1, 1), [(1, 2, 3)])
        self.assertEqual((result["A"], result["R"], result["P"]),
                         ((8, 8), (2, 2), 2))
        self.assertEqual(result["rho"], (2, 2))
        self.assertEqual(result["error"], 1)

    def test_failed_maximum_uses_direct_route_not_conditional_pass(self):
        trace = synthetic_trace()
        report = study.check_sufficient(trace)
        self.assertFalse(report["success"])
        self.assertIn("step_1.cubic_maximum", report["failed"])
        result = study.analyze(trace)
        self.assertEqual(result["route"], "direct-square-identity")
        self.assertEqual(result["A"], (2, 4))
        lower, upper = result["rho"]
        self.assertLessEqual(lower**2, 2)
        self.assertGreater((lower+Fraction(1, 2**192))**2, 2)
        self.assertEqual(upper, 2)
        self.assertEqual(result["error"], 1)
        self.assertEqual(result["status"], "CERTIFIED_BOUND")

    def test_all_premises_required_for_conditional_theorem(self):
        e0 = Fraction(99, 100)
        trace = synthetic_trace(
            D=(Fraction(95, 100)*e0, Fraction(1)),
            record=step(p=Fraction(1, 2), q=Fraction(1, 4), h=3,
                        pre=e0, post=Fraction(999, 1000)))
        trace.ratios["en0"] = e0.as_integer_ratio()
        report = study.check_sufficient(trace)
        self.assertTrue(report["success"], report["failed"])
        result = study.analyze(trace)
        self.assertEqual(result["route"], "conditional-theorem")
        self.assertEqual(result["rho"], (Fraction(974, 1000), Fraction(1001, 1000)))
        self.assertEqual(result["error"], Fraction(26, 1000))
        self.assertTrue(result["rho_strict"])
        self.assertTrue(result["error_strict"])
        # Independent exact lower/upper implications, no floating square roots.
        eps = Fraction(1, 2**40)
        self.assertGreater(Fraction(95, 100)*(1-eps)**32
                           * (Fraction(999, 1000)-eps)/(1+32*eps),
                           Fraction(974, 1000)**2)
        self.assertLess((1+eps)/(1-32*eps), Fraction(1001, 1000)**2)
        trace.steps[0].ratios["h_again"] = (4, 1)
        report = study.check_sufficient(trace)
        self.assertIn("step_1.repeated_h_equal", report["failed"])
        self.assertFalse(report["success"])
        self.assertNotEqual(study.analyze(trace)["route"], "conditional-theorem")

    def test_reversed_empty_and_singleton_intervals(self):
        with self.assertRaises(study.Refused):
            study.cubic_range(1, 3, 2, 1)
        with self.assertRaises(study.Refused):
            study.direct_enclosure((2, 1), [])
        self.assertEqual(study.cubic_range(1, 3, 3, 3), (0, 0))

    def test_nonfinite_and_complex_rejected_not_convergence(self):
        for value in (float("nan"), float("inf"), -float("inf"), complex(1, 0)):
            with self.subTest(value=value), self.assertRaises(study.Refused):
                study.float_ratio(value)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(study.Refused):
                study.sqrt_bracket(value)
            with self.assertRaises(study.Refused):
                study.cubic_range(1, 3, 0, value)

    def test_zero_negative_coefficients_and_nonpositive_domain(self):
        for coefficients in ((0, 1, 3), (-1, 1, 3), (1, 0, 3),
                             (1, -1, 3), (1, 1, 0), (1, 1, -3)):
            with self.subTest(coefficients=coefficients):
                with self.assertRaises(study.Refused):
                    study.direct_enclosure((1, 2), [coefficients])
        for bounds in ((0, 1), (-1, 1)):
            with self.assertRaises(study.Refused):
                study.direct_enclosure(bounds, [])

    def test_sign_changing_factor_uses_signed_R_fallback(self):
        result = study.direct_enclosure((2, 4), [(1, 1, 3)])
        self.assertFalse(result["nonnegative_factors"])
        self.assertFalse(result["strictly_positive_factors"])
        self.assertEqual(result["route"], "direct-signed-R")
        self.assertEqual(result["R"], (-1, 1))
        self.assertEqual(result["rho"], (-2, 2))
        self.assertEqual(result["error"], 3)
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_zero_factor_allows_nonnegative_not_strict_positive(self):
        result = study.direct_enclosure((3, 3), [(1, 1, 3)])
        self.assertTrue(result["nonnegative_factors"])
        self.assertFalse(result["strictly_positive_factors"])
        self.assertEqual(result["rho"], (0, 0))
        self.assertEqual(result["error"], 1)

    def test_outward_rounding_handles_negative_and_exact_grid(self):
        self.assertEqual(study.outward((Fraction(-1, 3), Fraction(1, 3)), bits=2),
                         (Fraction(-1, 2), Fraction(1, 2)))
        self.assertEqual(study.outward((Fraction(-1, 2), Fraction(3, 4)), bits=2),
                         (Fraction(-1, 2), Fraction(3, 4)))
        self.assertEqual(study.decimal_outward(Fraction(-1, 3), places=2), "-0.34")
        self.assertEqual(study.decimal_outward(Fraction(-1, 3), places=2, upper=True),
                         "-0.33")
        self.assertEqual(study.decimal_outward(Fraction(1, 3), places=2, upper=True),
                         "0.34")

    def test_stop_overshoot_not_clipped_or_hidden(self):
        trace = synthetic_trace(record=step(post=Fraction(11, 10)))
        self.assertEqual(trace.terminal, Fraction(11, 10))
        report = study.check_sufficient(trace)
        self.assertTrue(report["premises"]["terminating_trace"])
        self.assertFalse(report["premises"]["step_1.cubic_maximum"])
        self.assertNotEqual(study.analyze(trace)["route"], "conditional-theorem")
        trace.completed = False
        with self.assertRaises(study.Refused):
            study.analyze(trace)

    def test_missing_numpy_explicit_refusal_no_replacement(self):
        original = builtins.__import__

        def missing(name, *args, **kwargs):
            if name == "numpy":
                raise ModuleNotFoundError("deliberately missing NumPy")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaisesRegex(study.Refused, "NumPy required"):
                study.source_trace(EXPECTED_PROFILES[0])

    def test_safety_limit_validation_precedes_numpy_import(self):
        for limit in (0, -1, 257, True, 1.5):
            with self.assertRaisesRegex(study.Refused, "safety_limit"):
                study.source_trace(EXPECTED_PROFILES[0], limit)


class SourceNumpyRequiredTests(unittest.TestCase):
    """Future execution requires NumPy; absence is an explicit test failure."""

    def test_unsorted_actual_returned_root_index_and_complete_source_trace(self):
        try:
            import numpy as np
        except ImportError:
            self.fail("NumPy REQUIRED for pinned-source tests; not skipped")
        for profile in EXPECTED_PROFILES:
            with self.subTest(profile=profile):
                trace = study.source_trace(profile)
                a, M, g = profile
                # Independent literal source transcription: np.roots result used
                # directly, never expected roots derived from candidate trace.
                en = a/M
                C = (M*1.05+10**(-5))*768**2
                raw_b = 1/C**(1/2)
                # Fixed positive normal binary64 values halve exactly, without
                # underflow. Independent exact oracle allocates no mask/vector.
                expected_b = Fraction(*raw_b.as_integer_ratio()) / g
                self.assertEqual(trace.Q("b"), expected_b)
                self.assertEqual(trace.ratios["d"], (10**(-5)/C).as_integer_ratio())
                stop = 1-.001
                self.assertEqual(trace.ratios["stop"], stop.as_integer_ratio())
                count = 0
                while en < stop:
                    self.assertLess(count, 256, "independent source oracle safety refusal")
                    roots = np.roots([1-en**3,6*en**2-6,9-9*en])
                    self.assertFalse(np.iscomplexobj(roots))
                    self.assertTrue(all(math.isfinite(x) and x > 0 for x in roots))
                    k = roots[1]
                    observed = trace.steps[count]
                    self.assertEqual(observed.roots,
                                     tuple(x.as_integer_ratio() for x in roots))
                    self.assertEqual(observed.root_index, 1)
                    self.assertEqual(observed.ratios["k"], roots[1].as_integer_ratio())
                    self.assertEqual(observed.ratios["pre"], en.as_integer_ratio())
                    self.assertEqual(observed.ratios["p"], (k**(3/2)/2).as_integer_ratio())
                    self.assertEqual(observed.ratios["q"], (k**3/4).as_integer_ratio())
                    self.assertEqual(observed.ratios["h"], (3/k).as_integer_ratio())
                    self.assertEqual(observed.ratios["h_again"], (3/k).as_integer_ratio())
                    en = k*en*(3-k*en)**2/4
                    self.assertTrue(math.isfinite(en) and en > 0)
                    self.assertEqual(observed.ratios["post"], en.as_integer_ratio())
                    self.assertEqual(observed.counter, count+1)
                    count += 1
                self.assertTrue(trace.completed)
                self.assertEqual(len(trace.steps), count)
                self.assertEqual(trace.terminal, Fraction(*en.as_integer_ratio()))
                lower, upper = study.domain(trace)
                d = Fraction(*(10**(-5)/C).as_integer_ratio())
                scale = 768**2*expected_b*expected_b*g*g
                self.assertEqual(lower, scale*Fraction(*a.as_integer_ratio())+d)
                self.assertEqual(upper, scale*M+d)
                self.assertLess(lower, trace.Q("en0"))  # Never erase initial mismatch.
                report = study.check_sufficient(trace)
                self.assertEqual(report["D_minus_en0"], lower-trace.Q("en0"))
                self.assertEqual(report["success"], all(report["premises"].values()))
                result = study.analyze(trace)
                self.assertEqual(result["route"] == "conditional-theorem", report["success"])

    def test_unfinished_source_safety_is_refusal_with_partial_state(self):
        try:
            import numpy  # noqa: F401 -- explicit source-test requirement
        except ImportError:
            self.fail("NumPy REQUIRED for pinned-source tests; not skipped")
        with self.assertRaisesRegex(study.Refused, "unfinished") as caught:
            study.source_trace(EXPECTED_PROFILES[2], safety_limit=1)
        self.assertEqual(len(caught.exception.trace.steps), 1)
        self.assertFalse(caught.exception.trace.completed)

    def test_source_complex_nonfinite_and_nonpositive_roots_refused(self):
        try:
            import numpy as np
        except ImportError:
            self.fail("NumPy REQUIRED for pinned-source tests; not skipped")
        for roots in (np.array([1+0j, 2+0j]), np.array([1., float("nan")]),
                      np.array([1., float("inf")]), np.array([1., 0.]),
                      np.array([1., -1.]), np.array([1.])):
            with self.subTest(roots=roots), patch.object(np, "roots", return_value=roots):
                with self.assertRaises(study.Refused):
                    study.source_trace(EXPECTED_PROFILES[0])

    def test_source_nonfinite_arithmetic_rejected_before_false_stop(self):
        try:
            import numpy as np
        except ImportError:
            self.fail("NumPy REQUIRED for pinned-source tests; not skipped")
        # Positive finite roots, but evaluating q overflows; no NaN false exit.
        with patch.object(np, "roots", return_value=np.array([1., 1.e103])):
            with self.assertRaises(study.Refused):
                study.source_trace(EXPECTED_PROFILES[0])

    def test_promoted_runtime_warning_refused_with_incomplete_trace(self):
        try:
            import numpy as np
        except ImportError:
            self.fail("NumPy REQUIRED for pinned-source tests; not skipped")
        # Local test policy only; source equations and NumPy error policy stay.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with patch.object(np, "roots", return_value=np.array([1., 1.e103])):
                with self.assertRaisesRegex(study.Refused, "source arithmetic refused") as caught:
                    study.source_trace(EXPECTED_PROFILES[0])
        self.assertIsNotNone(caught.exception.trace)
        self.assertFalse(caught.exception.trace.completed)
        self.assertEqual(caught.exception.trace.steps, [])

    def test_nonfinite_post_state_is_not_false_convergence(self):
        try:
            import numpy as np
        except ImportError:
            self.fail("NumPy REQUIRED for pinned-source tests; not skipped")

        class PoisonState(float):
            def __mul__(self, other):
                # All coefficients remain finite; poison only k*en state update.
                return float("nan")

        with patch.object(np, "roots", return_value=[2., PoisonState(1.)]):
            with self.assertRaisesRegex(study.Refused, "nonfinite") as caught:
                study.source_trace(EXPECTED_PROFILES[0])
        self.assertFalse(caught.exception.trace.completed)
        self.assertEqual(caught.exception.trace.steps, [])


if __name__ == "__main__":
    unittest.main()
