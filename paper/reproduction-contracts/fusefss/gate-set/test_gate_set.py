import copy
from fractions import Fraction
from hashlib import sha256
import json
from math import isqrt
import unittest

import reference as ref
import check_outputs as checker


class GateSet(unittest.TestCase):
    def test_source_lock(self):
        repo = ref.HERE.parents[3]
        lock = json.loads((ref.HERE / "source-lock.json").read_bytes())
        for path, expected in lock["local_sha256"].items():
            self.assertEqual(sha256((repo / path).read_bytes()).hexdigest(), expected)

    def test_frozen_fixture(self):
        data, obj = checker.verify_fixture()
        self.assertEqual(obj["sha256_u64le"]["silu"], "ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f")
        self.assertEqual(obj["sha256_u64le"]["exp0"], "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b")
        self.assertEqual(obj["sha256_u64le"]["exp1"], "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9")
        self.assertLess(len(data), 2 << 20)

    def test_rational_exp_certificates_all_entries(self):
        for key, values in ref.certify_tables().items():
            self.assertEqual(tuple(values), ref.tables()[key])

    def test_silu_every_floor_boundary_and_neighbors(self):
        table = ref.tables()["silu"]
        for boundary in range(-1025 * 64, 1026 * 64, 64):
            for x in (boundary - 1, boundary, boundary + 1):
                result = ref.activation(x)
                self.assertEqual(result["output"], ref.math_ref.sigma_activation(x, table) % ref.N)
        self.assertEqual(ref.activation(-1)["output"], ref.N - 31)
        self.assertNotEqual(ref.activation(-1)["index"], abs(-1) // 64)
        self.assertEqual(ref.scalar.gelu_descriptor().digest(), "5f1e888e7bdabe3cbad4b6bbf1abd6749c8dff0918c6f243916db071398e79d3")
        self.assertEqual(ref.scalar.gelu_descriptor().phi, "identity")

    def test_nexp_exhaustive_bounded_domain(self):
        tables = ref.tables()
        for c in range(65536):
            expected = tables["exp0"][c & 255] * tables["exp1"][c >> 8] // 4096
            self.assertEqual(ref.nexp(c), expected)
            self.assertLessEqual(ref.nexp(c), 4096)
        self.assertEqual(ref.nexp(0), 4096)

    def test_reciprocal_all_admitted_denominators_and_total_tables(self):
        for k in (1, 2, 3, 768, 1024):
            self.assertEqual(ref.reciprocal_entry(0, k), 0)
            self.assertGreater(1 << ref.reciprocal_q(k), 64 * k)
            for z in range(64, 64 * k + 1):
                r = ref.reciprocal(z, k)
                self.assertLessEqual(r * z, 1 << 18)
                self.assertGreater((r + 1) * z, 1 << 18)
        self.assertEqual(ref.reciprocal_q(1024), 17)

    def test_softmax_quantization_mask_and_maximum(self):
        out = ref.softmax([0, -4096])["output"]
        self.assertEqual(out, [3013, 1107])
        self.assertEqual(sum(out), 4120)
        self.assertEqual(ref.softmax([0, 0, 0])["output"], [1365] * 3)
        self.assertEqual(ref.softmax([0, 99999, -4096], [1, 0, 1])["output"], [3013, 0, 1107])
        self.assertEqual(ref.softmax([123])["output"], [4096])
        self.assertEqual(ref.softmax([0, -65535]), ref.softmax([0, -65536]))

    def test_norm_integer_tables_all_slots(self):
        for k in (1, 4, 768, 4096):
            for index in range(8192):
                m, e = divmod(index, 64)
                numerator, denominator = (1 << 55) * k, (128 + m) << e
                t = ref.rsqrt_entry(index, k)
                self.assertEqual(t, isqrt(numerator // denominator))
                self.assertLessEqual(t * t * denominator, numerator)
                self.assertGreater((t + 1) ** 2 * denominator, numerator)
                self.assertLessEqual(t, 1 << 30)

    def test_norm_index_all_mantissas_and_exponents(self):
        for e in range(50):
            for m in range(128):
                # First integer in this bin; low exponents have empty bins.
                lo = ((128 + m) * (1 << e) + 127) // 128
                hi = ((129 + m) * (1 << e) + 127) // 128
                if lo < hi:
                    for s in (lo, hi - 1):
                        got = ref.norm_index(s)
                        self.assertEqual(got, {"exponent": e, "mantissa": m, "index": 64 * m + e})

    def test_norm_source_mean_guard_and_separate_affine_shifts(self):
        result = ref.norm([4096] * 768)
        self.assertEqual((result["mean"], result["sum_squares_plus_one"], result["inverse"]), (3840, 50331649, 65536))
        self.assertEqual(result["output"], [4096] * 768)
        self.assertEqual(ref.norm([4096] * 4)["output"], [0] * 4)
        zero = ref.norm([0] * 4, kind="rms")
        self.assertEqual((zero["sum_squares_plus_one"], zero["exponent"], zero["index"]), (1, 0, 0))
        self.assertEqual(ref.norm([4096] * 4, kind="rms")["output"], [4096] * 4)
        for row in ([-65, 64], [-4096, -4096, 4096, 4096]):
            self.assertEqual(ref.norm(row)["output"], ref.math_ref.sigma_norm(row))
        # Floor-before-affine is observably not one combined shift.
        x, t, g = -65, 8001, -4096
        self.assertNotEqual((x * t // 4096) * g // 4096, x * t * g // (4096 ** 2))

    def test_signed_TR_ARS_exhaustive_small(self):
        for bits in range(2, 9):
            for word in range(1 << bits):
                x = word if word < 1 << (bits - 1) else word - (1 << bits)
                for drop in range(bits):
                    tr = ref.shift(word, bits, drop, "TR")
                    ars = ref.shift(word, bits, drop, "ARS")
                    self.assertEqual(tr["word"], word >> drop)
                    self.assertEqual(ars["word"], (x // (1 << drop)) % (1 << bits))
                    self.assertEqual(tr["bits"], bits - drop)
        self.assertNotEqual(ref.shift(255, 8, 6, "TR"), ref.shift(255, 8, 6, "ARS"))

    def test_B2A_Phi_and_external_rounding(self):
        for a in (0, 1):
            for b in (0, 1):
                self.assertEqual(ref.b2a_xor_shares(ref.B(a), ref.B(b)).word, a ^ b)
        self.assertEqual(ref.b2a_xor_shares(ref.B(1), ref.B(1)).word, 0)
        self.assertEqual(ref.external(Fraction(-1, 8192)), 0)
        self.assertEqual(ref.external(Fraction(-3, 8192)), ref.N - 2)
        self.assertEqual(ref.phi_select(ref.B(1), ref.A(7), ref.A(9)).word, 7)

    def test_comparison_domains_exhaustive_small(self):
        for word in range(256):
            for threshold in range(257):
                self.assertEqual(ref.compare(word, 8, threshold).value, int(word < threshold))
            for threshold in range(-128, 129):
                signed = word if word < 128 else word - 256
                self.assertEqual(ref.compare(word, 8, threshold, "signed").value, int(signed < threshold))
            for width in range(1, 9):
                for threshold in (0, 1, 1 << width):
                    self.assertEqual(ref.compare(word, 8, threshold, "low", width).value,
                                     int(word % (1 << width) < threshold))

    def test_malformed_and_range_obligations(self):
        bad = [lambda: ref.activation(True), lambda: ref.activation(ref.HALF),
               lambda: ref.nexp(-1), lambda: ref.nexp(65536), lambda: ref.nexp(True),
               lambda: ref.reciprocal(0, 1), lambda: ref.reciprocal(65, 1),
               lambda: ref.reciprocal(64, 0), lambda: ref.reciprocal(64, 1025),
               lambda: ref.norm_index(0), lambda: ref.norm_index(ref.N),
               lambda: ref.rsqrt_entry(8192, 4), lambda: ref.rsqrt_entry(0, 0),
               lambda: ref.norm([]), lambda: ref.norm([ref.HALF - 1], kind="rms"),
               lambda: ref.norm([1], gamma=[]), lambda: ref.norm([0], gamma=[True]),
               lambda: ref.norm([1], kind="qwen"), lambda: ref.norm([0] * 4097),
               lambda: ref.softmax([]), lambda: ref.softmax([0], [0]),
               lambda: ref.softmax([0], [True]), lambda: ref.softmax([0], []),
               lambda: ref.softmax([ref.HALF - 1, -ref.HALF]),
               lambda: ref.shift(256, 8, 1, "TR"), lambda: ref.shift(0, 8, 8, "TR"),
               lambda: ref.shift(0, 8, 1, "gap"), lambda: ref.B(True), lambda: ref.B(2),
               lambda: ref.A(-1), lambda: ref.b2a(ref.A(1)),
               lambda: ref.b2a_xor_shares(ref.B(1), 0),
               lambda: ref.phi_select(ref.B(0), ref.A(0), ref.A(0, 50, 12)),
               lambda: ref.external(0.5), lambda: ref.external(Fraction(1 << 40))]
        bad += [lambda: ref.compare(0, 8, 257), lambda: ref.compare(0, 8, -129, "signed"),
                lambda: ref.compare(0, 8, 1, "low", 0), lambda: ref.compare(0, 8, 1, "unknown"),
                lambda: ref.compare(0, 8, 1, "unsigned", 2),
                lambda: ref.b2a_xor_shares(ref.B(0), ref.B(0), 51),
                lambda: ref.exp_bounds(Fraction(17)), lambda: ref.exp_bounds(1),
                lambda: ref.exp_bounds(Fraction(1), 79)]
        for call in bad:
            with self.assertRaises(ValueError):
                call()


class OutputChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frozen, fixture = checker.verify_fixture()
        cls.records = [{"schema": "pllm.fusefss.gate-set.v1/native-output",
                        "fixture_sha256": sha256(frozen).hexdigest(),
                        "execution": "protected-local-reconstruction", "implementation": "checker-self-test-not-native"}]
        cls.records += [{"id": c["id"], "gate": c["gate"], "output": c["expected"]} for c in fixture["cases"]]

    def encoded(self, records):
        return b"".join(ref.canonical(x) for x in records)

    def test_synthetic_acceptance_only(self):
        self.assertEqual(checker.check(self.encoded(self.records))["status"], "pass")

    def test_mutation_controls(self):
        mutations = []
        for key, value in (("fixture_sha256", "0" * 64), ("execution", "clear"), ("implementation", "")):
            r = copy.deepcopy(self.records)
            r[0][key] = value
            mutations.append(self.encoded(r))
        r = copy.deepcopy(self.records)
        r[1]["output"]["index"] += 1
        mutations.append(self.encoded(r))
        r = copy.deepcopy(self.records)
        r[-1]["output"] = True
        mutations.append(self.encoded(r))
        mutations += [self.encoded(self.records[:-1]), self.encoded(self.records + [self.records[-1]]),
                      self.encoded([self.records[0], self.records[2], self.records[1], *self.records[3:]]),
                      self.encoded(self.records)[:-1], self.encoded(self.records) + b"\n",
                      self.encoded(self.records).replace(b'"output":', b'"output":0,"output":', 1),
                      self.encoded(self.records).replace(b'"index":1023', b'"index":1023.0', 1)]
        for data in mutations:
            with self.assertRaises(ValueError):
                checker.check(data)


if __name__ == "__main__":
    unittest.main()
