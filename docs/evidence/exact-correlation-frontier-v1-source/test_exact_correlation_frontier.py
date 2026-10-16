"""Independent integer, leakage and directed-accounting challenges; stdlib only."""
import importlib.util
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
        a = p.ledger(s, 8, dimension=2, noise=1, capacity=157)
        b = p.ledger(s, 8, dimension=2, noise=1, capacity=157, burned=1)
        ca, cb = a["known_counts"], b["known_counts"]
        self.assertEqual(cb["baseline_preparation_to_inference"] - ca["baseline_preparation_to_inference"],
                         64 + 157 * 9)
        self.assertEqual(cb["candidate_preparation_to_inference"] - ca["candidate_preparation_to_inference"], 96)
        self.assertEqual(cb["new_public_delivery"], ca["new_public_delivery"])
        self.assertIsNone(b["unknown"]["failed_online_bytes"])

    def test_manual_one_stage_ledger(self):
        s = [{"m": 3, "n": 5, "bits": 24, "input_bits": 23,
              "output_bits": 65, "terminal": True}]
        r = p.ledger(s, 8, dimension=2, noise=1, capacity=4)
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
        frozen = p.ROOT / "docs/evidence/exact-correlation-frontier-bounded-v1.json"
        if not frozen.exists():
            self.skipTest("generate frozen evidence before replay check")
        self.assertEqual(json.loads(json.dumps(p.report())), json.loads(frozen.read_text()))


if __name__ == "__main__":
    unittest.main()
