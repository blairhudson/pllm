"""Run: python3 -B -m unittest discover -s paper/reproduction-contracts/maverick -v"""

import hashlib
import itertools
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
import unittest

from fixtures import all_vectors, code, small_case
from oracle import (BABYBEAR, PROPOSED_Q31_PLUS, FrozenBatch, Raa,
                    canonical, centered,
                    distance_dimension_two, dot, draw_below, exact_sparse_pass,
                    is_prime32, lemma_bound, matvec, partial_shuffle_support,
                    privacy_weight, protocol3, sparse_vector, transpose,
                    validate_publication)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class FieldAndSamplingTests(unittest.TestCase):
    def test_field_threshold_primality_and_machine_bounds(self):
        self.assertLess(BABYBEAR, 2**31)
        self.assertGreater(PROPOSED_Q31_PLUS, 2**31)
        for q in (BABYBEAR, PROPOSED_Q31_PLUS):
            self.assertTrue(is_prime32(q))
            self.assertLess((q - 1)**2 + q - 1, 2**63)
            self.assertEqual((q - 1)**2 % q, 1)
            self.assertEqual((2 * (q - 1)) % q, q - 2)
            wide_dot = 64 * (q - 1)**2
            self.assertGreater(wide_dot, 2**64)
            self.assertEqual(wide_dot % q, 64)
            self.assertEqual(matvec([[q - 1] * 64], [q - 1] * 64, q), [64])
            self.assertNotEqual((wide_dot % 2**64) % q, 64)
            self.assertEqual(centered(q - 1, q), -1)
            self.assertEqual(centered(q // 2, q), q // 2)
            self.assertEqual(centered(q // 2 + 1, q), -q // 2 + 1)
        self.assertFalse(any(is_prime32(n) for n in range(2**31 + 1, PROPOSED_Q31_PLUS)))
        self.assertEqual(Fraction(BABYBEAR // 2, 65536), 15360)

    def test_modular_correctness_is_not_integer_no_wrap(self):
        q = BABYBEAR
        integer = (q // 2) * 2
        self.assertEqual(integer % q, q - 1)
        self.assertEqual(centered(integer % q, q), -1)
        self.assertNotEqual(centered(integer % q, q), integer)
        with self.assertRaises(ValueError):
            canonical([q], q)

    def test_rejection_sampler_is_unbiased_and_burns_exhausted_stream(self):
        # Exhaust every 8-bit word under the identical rejection rule.
        for upper in (3, 5, 129, 251):
            counts = Counter()
            rejected = 0
            for word in range(256):
                try:
                    counts[draw_below(upper, iter([word]), bits=8)] += 1
                except ValueError:
                    rejected += 1
            self.assertEqual(set(counts.values()), {256 // upper})
            self.assertEqual(rejected, 256 % upper)
        self.assertEqual(draw_below(5, iter([255, 254]), bits=8), 4)
        with self.assertRaises(ValueError):
            draw_below(5, iter([255]), bits=8)

    def test_exact_weight_distribution_by_exhaustive_draw_paths(self):
        # Each unordered support appears t! times; each nonzero assignment once
        # per support ordering. This tests the distribution, not only cardinality.
        counts = Counter()
        for offsets in itertools.product(range(4), range(3)):
            support = partial_shuffle_support(4, 2, offsets)
            for values in itertools.product(range(1, 5), repeat=2):
                counts[tuple(sorted(zip(support, values)))] += 1
        self.assertEqual(len(counts), 6 * 4**2)
        self.assertEqual(set(counts.values()), {2})
        permutations = [partial_shuffle_support(4, 4, offsets)
                        for offsets in itertools.product(range(4), range(3), range(2), range(1))]
        self.assertEqual(len(set(permutations)), 24)

    def test_sparse_distribution_rejects_invalid_encodings(self):
        for entries in ([], [(0, 0)], [(0, 5)], [(4, 1)], [(0, 1), (0, 2)]):
            with self.assertRaises(ValueError):
                sparse_vector(4, entries, 5)
        with self.assertRaises(ValueError):
            sparse_vector(4, [(0, 1)], 5, weight=2)


class CodeAndProbabilityTests(unittest.TestCase):
    def test_literal_matrix_product_matches_both_linear_time_directions(self):
        for q in (BABYBEAR, PROPOSED_Q31_PLUS):
            c = code(q, 3, 1)
            g = c.dense()
            # Basis coverage proves equality for every input of these fixed codes.
            for i in range(c.n):
                w = [int(i == j) for j in range(c.n)]
                self.assertEqual(c.encode(w), matvec(transpose(g), w, q))
            for j in range(c.N):
                e = [int(i == j) for i in range(c.N)]
                self.assertEqual(c.syndrome(e), matvec(g, e, q))
            w = [q - 1, 257, q // 2]
            e = sparse_vector(c.N, [(0, q - 1), (3, 65536), (23, 2)], q)
            self.assertEqual(dot(c.encode(w), e, q), dot(w, c.syndrome(e), q))

    def test_code_description_and_dense_resource_guards(self):
        c = code(BABYBEAR, 3, 1).record()
        for name in ("pi1", "pi2"):
            bad = c | {name: (0,) * 24}
            with self.assertRaises(ValueError):
                Raa(**bad)
        with self.assertRaises(ValueError):
            Raa(**(c | {"v1": (0,) * 24}))
        with self.assertRaises(ValueError):
            code(BABYBEAR, 64, 2).dense()

    def test_exact_dimension_two_certificate_matches_all_messages(self):
        q = 5
        c = Raa(q, 2, 4, tuple(range(8)), (1, 4, 7, 2, 5, 0, 3, 6),
                (1, 2, 3, 4, 1, 3, 2, 4), (4, 3, 2, 1, 2, 1, 4, 3))
        g = c.dense()
        exact = min(sum(v != 0 for v in c.encode(x))
                    for x in itertools.product(range(q), repeat=2) if any(x))
        self.assertEqual(distance_dimension_two(g, q)["minimum_distance"], exact)
        self.assertEqual(distance_dimension_two([[1, 0, 0, 0], [0, 1, 0, 0]], q)
                         ["minimum_distance"], 1)
        for field in (BABYBEAR, PROPOSED_Q31_PLUS):
            cert = distance_dimension_two(code(field, 2, 3).dense(), field)
            self.assertGreaterEqual(cert["minimum_distance"], cert["N"] // 2)

    def test_soundness_probability_against_exhaustive_nonzero_coefficients(self):
        q, N, t = 5, 6, 2
        for weight in range(N + 1):
            z = [1 + i % 4 for i in range(weight)] + [0] * (N - weight)
            accepted, total = 0, 0
            for support in itertools.combinations(range(N), t):
                for values in itertools.product(range(1, q), repeat=t):
                    total += 1
                    accepted += sum(z[i] * a for i, a in zip(support, values)) % q == 0
            exact = exact_sparse_pass(q, N, weight, t)
            self.assertEqual(Fraction(accepted, total), exact)
            self.assertLessEqual(exact, lemma_bound(q, Fraction(weight, N), t, 1))
        # An intersection of exactly one position cannot cancel; two can.
        self.assertEqual(exact_sparse_pass(5, 2, 2, 1), 0)
        self.assertEqual(exact_sparse_pass(5, 2, 2, 2), Fraction(1, 4))

    def test_source_parameters_and_whole_window_bound(self):
        self.assertEqual([privacy_weight(n) for n in (1024, 2048, 4096, 8192, 16384)],
                         [160, 159, 157, 156, 154])
        self.assertEqual(privacy_weight(64), 165)
        profile = json.loads((HERE / "profile.json").read_text())
        p3, p2 = profile["protocol3_capsule"], profile["protocol2_capsule"]
        self.assertEqual(p3["tx"], privacy_weight(p3["n"]))
        self.assertEqual(p2["Nprime"], 8 * p2["B"] * p2["n"])
        self.assertEqual(p2["Qaux_bytes"], 4 * p2["N"] * p2["Nprime"])
        self.assertFalse(profile["security_admission"]["certified_private_verifiable_method"])
        for q in (BABYBEAR, PROPOSED_Q31_PLUS):
            bound = lemma_bound(q)
            self.assertLess(bound, Fraction(1, 2**40))
            self.assertLess(2 * bound * 65536, Fraction(1, 2**40))
            # Distinct windows, unlike claims sharing fixed challenges, do add.
            self.assertGreater(2 * lemma_bound(BABYBEAR) * 2**19, Fraction(1, 2**40))
            self.assertLessEqual(exact_sparse_pass(q, 32, 16, 31),
                                 lemma_bound(q, t=31, ell=1))


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = BABYBEAR
        cls.gx, cls.gy, cls.gp, cls.ms, cls.xs, cls.exs, cls.es, cls.eps = small_case(cls.q)
        cls.claims = [protocol3(m, x, cls.gx, cls.gy, ex, cls.es)
                      for m, x, ex in zip(cls.ms, cls.xs, cls.exs)]
        cls.inputs = [c["masked_x"] for c in cls.claims]
        cls.outputs = [c["masked_y"] for c in cls.claims]

    def fresh_batch(self, outputs=None):
        batch = FrozenBatch(self.ms, self.gy, self.gp)
        batch.fix_originals(self.inputs, self.outputs if outputs is None else outputs)
        es = batch.reveal_primary(self.es)
        responses = [matvec(batch.stack, e, self.q) for e in es]
        return batch, responses

    def test_protocol3_corrects_mask_and_validates_p_q(self):
        for m, x, ex, claim in zip(self.ms, self.xs, self.exs, self.claims):
            self.assertEqual(claim["y"], matvec(m, x, self.q))
            self.assertEqual(claim["correction"], matvec(m, claim["mask"], self.q))
            self.assertTrue(all(a == b for a, b in claim["checks"]))
            bad_p = [list(row) for row in claim["P"]]
            bad_p[0][0] = (bad_p[0][0] + 1) % self.q
            self.assertTrue(validate_publication(m, self.gx, self.gy, claim["P"], claim["Q"]))
            self.assertFalse(validate_publication(m, self.gx, self.gy, bad_p, claim["Q"]))
            other_q = self.claims[1]["Q"] if m == self.ms[0] else self.claims[0]["Q"]
            self.assertFalse(validate_publication(m, self.gx, self.gy, claim["P"], other_q))
            bad_q = [list(row) for row in claim["Q"]]
            bad_q[0][0] = self.q
            self.assertFalse(validate_publication(m, self.gx, self.gy, claim["P"], bad_q))

    def test_batch_accepts_only_after_two_fixation_barriers(self):
        batch, responses = self.fresh_batch()
        self.assertIsNone(batch.accepted_state)
        batch.fix_auxiliary(responses)
        self.assertIsNone(batch.accepted_state)
        accepted, checks = batch.finish(self.eps)
        self.assertTrue(accepted)
        self.assertEqual(batch.phase, "accepted")
        self.assertEqual(len(checks["secondary"]), 4)
        self.assertEqual(len(checks["original"]), 4)
        with self.assertRaises(ValueError):
            batch.finish(self.eps)

    def test_every_single_original_coordinate_corruption_rejects(self):
        for b in range(2):
            for i in range(4):
                outputs = [list(y) for y in self.outputs]
                outputs[b][i] = (outputs[b][i] + 1) % self.q
                batch, responses = self.fresh_batch(outputs)
                batch.fix_auxiliary(responses)
                accepted, checks = batch.finish(self.eps)
                self.assertFalse(accepted)
                self.assertEqual(batch.phase, "retired")
                self.assertIsNone(batch.accepted_state)
                self.assertTrue(all(a == b for a, b in checks["secondary"]))

    def test_auxiliary_forgery_can_pass_original_checks_but_secondary_rejects(self):
        batch, responses = self.fresh_batch()
        # Add a nonzero vector orthogonal to x_0, so the original scalar stays
        # unchanged. A verifier omitting the auxiliary checks accepts this lie.
        x = self.inputs[0]
        responses[0][0] = (responses[0][0] + x[1]) % self.q
        responses[0][1] = (responses[0][1] - x[0]) % self.q
        batch.fix_auxiliary(responses)
        accepted, checks = batch.finish(self.eps)
        self.assertFalse(accepted)
        self.assertTrue(all(a == b for a, b in checks["original"]))
        self.assertTrue(any(a != b for a, b in checks["secondary"]))

    def test_fixation_copies_and_forbids_adaptive_replacement(self):
        outputs = [list(y) for y in self.outputs]
        batch = FrozenBatch(self.ms, self.gy, self.gp)
        batch.fix_originals(self.inputs, outputs)
        outputs[0][0] = (outputs[0][0] + 1) % self.q
        self.assertEqual(batch.outputs[0], tuple(self.outputs[0]))
        with self.assertRaises(ValueError):
            batch.fix_originals(self.inputs, outputs)
        batch, responses = self.fresh_batch()
        batch.fix_auxiliary(responses)
        responses[0][0] = (responses[0][0] + 1) % self.q
        self.assertNotEqual(batch.responses[0], tuple(responses[0]))
        with self.assertRaises(ValueError):
            batch.fix_auxiliary(responses)

    def test_omissions_noncanonical_responses_wrong_order_and_cancel_retire(self):
        batch = FrozenBatch(self.ms, self.gy, self.gp)
        with self.assertRaises(ValueError):
            batch.reveal_primary(self.es)
        batch = FrozenBatch(self.ms, self.gy, self.gp)
        with self.assertRaises(ValueError):
            batch.fix_originals(self.inputs[:1], self.outputs[:1])
        batch, responses = self.fresh_batch()
        responses[0][0] = self.q
        with self.assertRaises(ValueError):
            batch.fix_auxiliary(responses)
        self.assertEqual(batch.phase, "retired")
        batch, responses = self.fresh_batch()
        batch.retire()
        with self.assertRaises(ValueError):
            batch.fix_auxiliary(responses)
        self.assertIsNone(batch.accepted_state)

    def test_wrong_auxiliary_publication_is_detected(self):
        batch, responses = self.fresh_batch()
        # Change an entry actually read by the first e and first e'.
        i, _ = self.es[0][0]
        j, _ = self.eps[0][0]
        batch.aux[i][j] = (batch.aux[i][j] + 1) % self.q
        batch.fix_auxiliary(responses)
        self.assertFalse(batch.finish(self.eps)[0])


class FrozenEvidenceTests(unittest.TestCase):
    def test_frozen_vectors(self):
        expected = json.loads((HERE / "vectors.json").read_text())
        actual = all_vectors()
        for left, right in zip(actual["small_cases"], expected["small_cases"]):
            # libm rounding is diagnostic, never the exact soundness predicate.
            self.assertAlmostEqual(left["lemma_31_two_checks"].pop("minus_log2_diagnostic"),
                                   right["lemma_31_two_checks"].pop("minus_log2_diagnostic"), places=9)
        self.assertEqual(actual, expected)

    def test_pinned_local_specification_sources(self):
        lock = json.loads((HERE / "source-lock.json").read_text())
        for entry in lock["local_snapshots"]:
            if entry["verify_current_bytes"]:
                self.assertEqual(hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest(),
                                 entry["sha256"], entry["path"])


if __name__ == "__main__":
    unittest.main()
