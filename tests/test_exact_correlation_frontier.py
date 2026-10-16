"""Independent integer, leakage and directed-accounting challenges; stdlib only."""
import importlib.util
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_exact_correlation_frontier.py"
SPEC = importlib.util.spec_from_file_location("exact_frontier", PATH)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def integer_oracle(w, x, bits):
    # No calls to implementation products, compilation or reconstruction helpers.
    result = []
    for row in w:
        accumulator = 0
        for i in range(len(x)):
            accumulator += int(row[i]) * int(x[i])
        result.append(accumulator & ((1 << bits) - 1))
    return tuple(result)


class AlgebraTests(unittest.TestCase):
    def test_arbitrary_signed_matrices_and_full_ring_inputs(self):
        rng = random.Random(91037)
        for bits, m, n in itertools.product((1, 3, 24, 32), (1, 3, 6), (1, 2, 5)):
            with self.subTest(bits=bits, shape=(m, n)):
                w = tuple(tuple(rng.randrange(-128, 128) for _ in range(n)) for _ in range(m))
                a = tuple(tuple(rng.randrange(1 << bits) for _ in range(2)) for _ in range(n))
                u = (rng.randrange(1 << bits), rng.randrange(1 << bits))
                noise = tuple((i, rng.randrange(1 << bits)) for i in range(0, n, 2))
                r, image = p.sparse_image(w, a, u, noise, bits)
                self.assertEqual(image, integer_oracle(w, r, bits))
                for x in ((0,) * n, ((1 << bits) - 1,) * n,
                          tuple(rng.randrange(-(1 << bits), 1 << bits) for _ in range(n))):
                    q = tuple(rng.randrange(1 << bits) for _ in range(m))
                    self.assertEqual(p.reconstruct(w, x, r, image, q, bits),
                                     integer_oracle(w, x, bits))

    def test_exact_weight_circuit_not_low_rank_mask(self):
        u, v = ((1,), (2,), (3,)), ((1, 2, 3),)
        # Rank-one part plus identity: full rational rank, masks entirely unrestricted.
        w = ((2, 2, 3), (2, 5, 6), (3, 6, 10))
        e = p.residual(w, u, v, 32)
        self.assertEqual(e, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        for r in itertools.product(range(4), repeat=3):
            self.assertEqual(p.circuit_image(u, v, e, r, 32), integer_oracle(w, r, 32))
        # Dropping even one residual is a lossy operator, not a safe optimization.
        self.assertNotEqual(p.circuit_image(u, v, ((0, 0, 0),) * 3, (1, 0, 0), 32),
                            integer_oracle(w, (1, 0, 0), 32))

    def test_residual_fallback_for_nonstructured_w(self):
        w = ((-128, 127, 7), (19, -3, 11))
        u, v = ((1,), (1,)), ((1, 1, 1),)
        e = p.residual(w, u, v, 24)
        self.assertEqual(p.circuit_image(u, v, e, (2**24 - 1, 0, 3), 24),
                         integer_oracle(w, (2**24 - 1, 0, 3), 24))
        self.assertEqual(sum(x != 0 for row in e for x in row), 6)

    def test_malformed_sparse_indices_rejected(self):
        for noise in (((0, 1), (0, 2)), ((-1, 1),), ((2, 1),)):
            with self.assertRaises(ValueError):
                p.sparse_image(((1, 2),), ((1,), (1,)), (3,), noise, 32)

    def test_wrong_image_or_wrong_ring_fails(self):
        w, x, r = ((127, -128),), (1, 2), (2**24, 0)
        good = integer_oracle(w, r, 32)
        self.assertNotEqual(p.reconstruct(w, x, r, (good[0] + 1,), (7,), 32),
                            integer_oracle(w, x, 32))
        self.assertNotEqual(p.reconstruct(w, x, r, integer_oracle(w, r, 24), (7,), 32),
                            integer_oracle(w, x, 32))


class PrivacyNegativeControls(unittest.TestCase):
    def test_gf2_witness_for_every_small_binary_matrix(self):
        for n, d in ((2, 1), (3, 1), (3, 2), (4, 2)):
            for flat in itertools.product((0, 1), repeat=n * d):
                rows = tuple(flat[i * d:(i + 1) * d] for i in range(n))
                support = p.dual_witness(rows)
                self.assertTrue(0 < len(support) <= d + 1)
                self.assertEqual(len(set(support)), len(support))
                self.assertTrue(all(sum(rows[i][j] for i in support) % 2 == 0 for j in range(d)))

    def test_public_seeded_a_witness_not_just_rank_deficient_toy(self):
        raw = hashlib.shake_256(b"independent public A fixture for GF2 test").digest(129 * 128 * 4)
        rows = tuple(tuple(int.from_bytes(raw[4 * (128 * i + j):4 * (128 * i + j + 1)], "little")
                           for j in range(128)) for i in range(129))
        support = p.dual_witness(rows)
        self.assertTrue(0 < len(support) <= 129)
        # Independent coordinatewise parity, without integer-bitset elimination.
        self.assertEqual([sum(rows[i][j] for i in support) & 1 for j in range(128)], [0] * 128)
        self.assertGreater(p.parity_tv(896, len(support), 27), Fraction(1, 100))

    def test_full_column_rank_positive_witness_control(self):
        for d in (1, 3, 128):
            # Identity rows are independent, so the final all-ones row forces
            # the sole dependency to use all d+1 rows, attaining the weight bound.
            rows = tuple(tuple(int(i == j) for j in range(d)) for i in range(d)) + ((1,) * d,)
            self.assertEqual(p.dual_witness(rows), tuple(range(d + 1)))

    def test_parity_tv_from_independent_exhaustive_full_views(self):
        matrices = (((0,), (1,), (1,)), ((1,), (1,), (1,)),
                    ((1, 0), (0, 1), (1, 1), (1, 0)))
        for a, bits in itertools.product(matrices, (1, 2)):
            n, d, modulus = len(a), len(a[0]), 1 << bits
            support = p.dual_witness(a)
            x1 = [0] * n
            x1[support[0]] = 1
            for t in range(n + 1):
                views = [Counter(), Counter()]
                for u in itertools.product(range(modulus), repeat=d):
                    for positions in itertools.combinations(range(n), t):
                        for errors in itertools.product(range(modulus), repeat=t):
                            r = [sum(a[i][j] * u[j] for j in range(d)) for i in range(n)]
                            for i, error in zip(positions, errors, strict=True):
                                r[i] += error
                            for choice, x in enumerate(((0,) * n, x1)):
                                z = tuple((x[i] - r[i]) % modulus for i in range(n))
                                views[choice][z] += 1
                projected = []
                for view in views:
                    parity = Counter()
                    for z, count in view.items():
                        parity[sum(z[i] for i in support) % 2] += count
                    projected.append(parity)
                denominator = views[0].total()
                parity_distance = Fraction(sum(abs(projected[0][b] - projected[1][b]) for b in (0, 1)),
                                           2 * denominator)
                full_distance = Fraction(sum(abs(views[0][z] - views[1][z])
                                             for z in views[0].keys() | views[1].keys()), 2 * denominator)
                # Independent combinatorial event enumeration, not comb() formula.
                all_sets = list(itertools.combinations(range(n), t))
                miss = Fraction(sum(not (set(s) & set(support)) for s in all_sets), len(all_sets))
                self.assertEqual(parity_distance, miss)
                self.assertEqual(p.parity_tv(n, len(support), t), miss)
                self.assertGreaterEqual(full_distance, parity_distance)

    def test_bound_numbers_and_strict_necessary_thresholds(self):
        for n, minimum, decimal in ((896, 413, 0.014057816285166622),
                                    (4864, 2387, 0.4830101776756551)):
            self.assertAlmostEqual(float(p.parity_tv(n, 129, 27)), decimal, places=15)
            self.assertEqual(p.necessary_noise(n, 128), minimum)
            self.assertGreaterEqual(p.parity_tv(n, 129, minimum - 1), Fraction(1, 2**128))
            self.assertLess(p.parity_tv(n, 129, minimum), Fraction(1, 2**128))

    def test_full_independent_noise_positive_privacy_control(self):
        # Exhaustive full ring errors make each opening exactly uniform, even A=0.
        for x in ((0, 0), (1, 3)):
            views = Counter(tuple((x[i] - e[i]) % 4 for i in range(2))
                            for e in itertools.product(range(4), repeat=2))
            self.assertEqual(set(views.values()), {1})
            self.assertEqual(len(views), 16)
        self.assertEqual(p.parity_tv(2, 1, 2), 0)

    def test_sparse_toy_support_is_publicly_distinguishable(self):
        # Independent event: x=0 always gives two equal opening coordinates.
        views = []
        for x in ((0, 0, 0), (0, 1, 2)):
            equal = 0
            for u, i, e in itertools.product(range(4), range(3), range(4)):
                r = [u] * 3
                r[i] = (r[i] + e) % 4
                z = [(x[j] - r[j]) % 4 for j in range(3)]
                equal += len(set(z)) < 3
            views.append(equal)
        self.assertEqual(views[0], 48)
        self.assertLess(views[1], 48)
        self.assertGreaterEqual(p.sparse_toy_tv(), (views[0] - views[1]) / 48)

    def test_key_linear_cross_query_relation_for_every_key(self):
        for secret in itertools.product(range(8), repeat=2):
            for x in ((0, 0, 0), (1, 0, 0), (3, 7, 6)):
                self.assertEqual(p.homomorphic_leak(secret, x, 3),
                                 (x[0] - 2 * x[1] + x[2]) % 8)

    def test_fresh_labels_do_not_fix_key_linearity(self):
        self.assertNotEqual(p.homomorphic_leak((7, 13), (3, 8, 21), 32), 0)

    def test_mask_reuse_leaks_delta_even_with_new_q(self):
        modulus = 1 << 32
        x0, x1, r = (5, -7), (17, 2), (193, 713)
        z0 = tuple((x - a) % modulus for x, a in zip(x0, r))
        z1 = tuple((x - a) % modulus for x, a in zip(x1, r))
        self.assertEqual(tuple((b - a) % modulus for a, b in zip(z0, z1)), (12, 9))


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.stages = p.geometry()

    def test_locked_geometry_and_actual_workload_rows(self):
        self.assertEqual(sum(s["m"] * s["n"] for s in self.stages), 357826560)
        for outputs, count in ((8, 14625), (64, 20001)):
            self.assertEqual(p.ledger(self.stages, outputs)["known_counts"]["successful_stage_rows"], count)

    def test_directed_totals_and_cold_delivery_not_free(self):
        for outputs in (8, 64):
            result = p.ledger(self.stages, outputs)
            c = result["known_counts"]
            online = c["client_to_inference"] + c["inference_to_client"]
            self.assertEqual(result["baseline_known_all_link_bytes"], online +
                             c["baseline_preparation_to_client"] + c["baseline_preparation_to_inference"])
            self.assertEqual(result["candidate_known_all_link_bytes"], online +
                             c["candidate_preparation_to_client"] + c["candidate_preparation_to_inference"])
            self.assertGreaterEqual(c["new_public_delivery"], 357826560)
            self.assertGreater(result["known_byte_delta"], 0)
            self.assertTrue(all(v is None for v in result["unknown"].values()))

    def test_renewal_and_burn_charge_whole_epoch(self):
        s = [{"m": 3, "n": 5, "bits": 24, "input_bits": 23,
              "output_bits": 65, "terminal": False}]
        a = p.ledger(s, 8, dimension=2, noise=1, capacity=157, request_sized=False)
        b = p.ledger(s, 8, dimension=2, noise=1, capacity=157, burned=1, request_sized=False)
        ca, cb = a["known_counts"], b["known_counts"]
        self.assertEqual(cb["baseline_preparation_to_inference"] - ca["baseline_preparation_to_inference"],
                         64 + 157 * 9)
        self.assertEqual(cb["candidate_preparation_to_inference"] - ca["candidate_preparation_to_inference"], 96)
        self.assertEqual(cb["new_public_delivery"], ca["new_public_delivery"])
        self.assertIsNone(b["unknown"]["failed_online_bytes"])

    def test_manual_one_stage_ledger(self):
        s = [{"m": 3, "n": 5, "bits": 24, "input_bits": 23,
              "output_bits": 65, "terminal": True}]
        r = p.ledger(s, 8, dimension=2, noise=1, capacity=4, request_sized=False)
        c = r["known_counts"]
        self.assertEqual(c["successful_stage_rows"], 8)
        self.assertEqual(c["issued_stage_rows"], 8)
        self.assertEqual(c["new_public_delivery"], 15 + 18 + 64)
        self.assertEqual(c["candidate_preparation_to_client"], 256 + 15 + 18 + 32)
        self.assertEqual(c["candidate_preparation_to_inference"], 192 + 32)
        self.assertEqual(c["client_to_inference"], 8 * (40 + 15))
        self.assertEqual(c["client_streamed_wa_read_bytes"], 8 * 18)
        self.assertEqual(c["client_a_u_ring_macs"], 8 * 5 * 2)
        self.assertEqual(c["client_sparse_w_e_ring_macs"], 8 * 3)

    def test_hash_drift_rejected_before_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = next(iter(p.LOCKS))
            (root / first).parent.mkdir(parents=True)
            (root / first).write_text(json.dumps({"stages": []}))
            with self.assertRaisesRegex(ValueError, "source hash drift"):
                p.geometry(root)

    def test_invalid_counts_rejected(self):
        for args in ({"capacity": 0}, {"burned": -1}, {"noise": 50000}, {"requests": True}):
            with self.assertRaises(ValueError):
                p.ledger(self.stages, 8, **args)

    def test_transparent_key_fallback_never_hides_client_tape(self):
        for outputs in (8, 64):
            alternative = p.alternative_ledgers(self.stages, outputs)
            key = alternative["transparent_exact_key_fallback"]
            self.assertGreater(key["known_all_link_bytes"],
                               p.ledger(self.stages, outputs)["baseline_known_all_link_bytes"])
            self.assertGreater(key["client_streamed_io_bytes"], 0)
            self.assertIsNone(alternative["exact_weight_circuit"]["known_all_link_bytes"])

    def test_request_sized_control_does_not_credit_unused_corrections(self):
        for outputs in (8, 64):
            c = p.ledger(self.stages, outputs, request_sized=True)["known_counts"]
            self.assertEqual(c["baseline_correction_tape_writes"], c["baseline_correction_tape_reads"])
            self.assertEqual(c["issued_stage_rows"], c["successful_stage_rows"])

    def test_frozen_evidence_replays_with_source_and_code_hashes(self):
        frozen = p.ROOT / "docs/evidence/exact-correlation-frontier-bounded-v2.json"
        if not frozen.exists():
            self.skipTest("generate frozen evidence before replay check")
        self.assertEqual(json.loads(json.dumps(p.report())), json.loads(frozen.read_text()))

    def test_v1_bytes_and_archived_sources_unchanged(self):
        archive = p.ROOT / "docs/evidence/exact-correlation-frontier-v1-source"
        manifest = json.loads((archive / "manifest.json").read_text())
        for original, expected in manifest["original_paths_sha256"].items():
            self.assertEqual(hashlib.sha256((archive / Path(original).name).read_bytes()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256((p.ROOT / manifest["preserved_evidence"]).read_bytes()).hexdigest(),
                         manifest["preserved_evidence_sha256"])

    def test_necessary_noise_work_counts_are_not_security_claim(self):
        attack = p.parity_attack_report(self.stages)
        for row, expected in zip(attack["necessary_noise_work"], (25411997184, 34823007744), strict=True):
            self.assertEqual(row["necessary_bound_only_w_e_ring_macs"], expected)
            self.assertFalse(row["security_established"])
            self.assertIsNone(row["complete_client_peak_bytes"])

    def test_request_sized_headlines_independently(self):
        for outputs, expected in ((8, 352774558), (64, 483271358)):
            result = p.ledger(self.stages, outputs)
            self.assertEqual(result["baseline_known_all_link_bytes"], expected)
            self.assertEqual(result["capacity_policy"], "per-stage one-request rows")
            c = result["known_counts"]
            self.assertEqual(c["baseline_correction_tape_writes"], c["baseline_correction_tape_reads"])
            # Linear horizon rule with request-sized renewal, independently of report search.
            # Candidate one-time public artifacts are reusable, all other payload recurs.
            for target, minimum in ((Fraction(0), 4 if outputs == 8 else 3),
                                    (Fraction(1, 5), 8 if outputs == 8 else 6)):
                extra_public = c["new_public_delivery"]
                recurring = result["candidate_known_all_link_bytes"] - extra_public
                self.assertGreater(extra_public + (minimum - 1) * recurring,
                                   (1 - target) * (minimum - 1) * expected)
                self.assertLessEqual(extra_public + minimum * recurring,
                                     (1 - target) * minimum * expected)


if __name__ == "__main__":
    unittest.main()
