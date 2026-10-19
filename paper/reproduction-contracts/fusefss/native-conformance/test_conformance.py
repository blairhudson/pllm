"""Checker tests use hand-solved transcripts, NOT native/compiler-generated truth."""

import copy
import hashlib
import io
import json
import math
from pathlib import Path
import unittest

from independent import (CONTEXT, GELU_DIGEST, canonical, descriptor, digest, gelu,
                         gelu_descriptor, load_table, nearest_even, predicate, unfused)
from check_native import Mismatch, check_stream, load_fixture, validate_compilation


def hand_case():
    expressions = [["low_lt", 2, 3], ["slt", 0], ["lt", 8], ["msb", 1]]
    d = descriptor(3, [0, 3, 8], [
        {"coefficients": [1, 3], "booleans": expressions},
        {"coefficients": [6, 2], "booleans": expressions}])
    ident = digest(d)
    shape = {"source_intervals": 2, "padded_intervals": 3, "boolean_outputs": 4,
             "queries": 28, "degree": 1, "coefficients_per_interval": 2, "horner_products": 1}
    outputs = [[1, [1, 0, 1, 0]], [4, [1, 0, 1, 0]], [7, [1, 0, 1, 0]], [4, [0, 0, 1, 1]],
               [6, [1, 1, 1, 1]], [0, [1, 1, 1, 1]], [2, [1, 1, 1, 1]], [4, [0, 1, 1, 0]]]
    # Hand-listed original atom order; lt(0), lt(8) are public sentinels.
    atoms = [(3, 0, 3), (2, 0, 3), (3, 0, 3), (2, 0, 3),
             (3, 0, 3), (3, 4, 4), (3, 0, 3), (3, 4, 4),
             (3, 0, 3), (3, 0, 3),
             (3, 0, 3), (3, 1, 4), (3, 0, 3), (3, 1, 4)]
    runs, records = [], [{"schema": CONTEXT + "/native-output", "fixture_sha256": "a" * 64,
                          "execution": "compiler-clear", "suite": "all",
                          "implementation": "hand-solved-checker-test-not-native"}]
    for mask, boundaries, ordinals in [(0, [0, 1, 3, 8], [0, 0, 1]),
                                       (2, [0, 2, 5, 8], [1, 0, 1]),
                                       (5, [0, 1, 5, 8], [1, 1, 0])]:
        run = {"id": str(mask), "descriptor_id": ident, "mask": mask,
               "context": "hand-context/" + str(mask), "words": list(range(8)),
               "masked_words": [(x + mask) % 8 for x in range(8)], "expected": outputs}
        runs.append(run)
        queries = []
        for width, shift, beta in atoms:
            low = mask % (1 << width)
            queries.extend([[width, shift, (low + beta) % (1 << width)], [width, shift, low]])
        records.append({**{k: run[k] for k in ("id", "descriptor_id", "mask", "context")},
                        "shape": shape, "native_bytes": 777,
                        "compiled": {"boundaries": boundaries, "ordinals": ordinals,
                                     "payloads": [d["pieces"][i]["coefficients"] for i in ordinals], "queries": queries},
                        "outputs": outputs})
    records.append({"external_encoding": [0, -2, 2]})
    fixture = {"descriptors": [{"id": ident, "descriptor": d, "shape": shape}], "runs": runs,
               "external_encoding": [{"signed_q12": v} for v in (0, -2, 2)],
               "counts": {"evaluations": 24}}
    return fixture, copy.deepcopy(records)


def check_records(fixture, records):
    return check_stream(io.BytesIO(b"".join(canonical(r) + b"\n" for r in records)), fixture, "a" * 64)


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.fixture, self.records = hand_case()

    def reject(self, records):
        with self.assertRaises(Mismatch):
            check_records(self.fixture, records)

    def test_hand_solved_valid_stream(self):
        self.assertEqual(check_records(self.fixture, self.records)["status"], "pass")
        d = self.fixture["descriptors"][0]["descriptor"]
        for x, result in enumerate(self.records[1]["outputs"]):
            self.assertEqual(unfused(d, x, []), result)

    def test_missing_reordered_duplicate_records(self):
        self.reject(self.records[:-1])
        self.reject([self.records[0], self.records[2], self.records[1], *self.records[3:]])
        self.reject([*self.records[:2], self.records[1], *self.records[2:]])
        self.reject([*self.records, self.records[-1]])

    def test_missing_reordered_duplicate_outputs(self):
        for operation in ("missing", "reordered", "duplicate"):
            rows = copy.deepcopy(self.records)
            values = rows[1]["outputs"]
            if operation == "missing":
                values.pop()
            elif operation == "reordered":
                values[0], values[1] = values[1], values[0]
            else:
                values.insert(1, values[0])
            self.reject(rows)

    def test_wrong_arithmetic_boolean_and_types(self):
        for value in (0, True, 1.0, -1, 8):
            rows = copy.deepcopy(self.records)
            rows[1]["outputs"][0][0] = value
            self.reject(rows)
        rows = copy.deepcopy(self.records)
        rows[1]["outputs"][0][1][0] = 0
        self.reject(rows)

    def test_wrong_mask_descriptor_context_and_fixture(self):
        for key, value in (("mask", 1), ("descriptor_id", "b" * 64), ("context", "replayed"), ("id", "2")):
            rows = copy.deepcopy(self.records)
            rows[1][key] = value
            self.reject(rows)
        rows = copy.deepcopy(self.records)
        rows[0]["fixture_sha256"] = "0" * 64
        self.reject(rows)

    def test_wrong_coefficient_and_order(self):
        for coefficients in ([0, 3], [3, 1], [1, True], [1, 3, 0]):
            rows = copy.deepcopy(self.records)
            rows[1]["compiled"]["payloads"][0] = coefficients
            self.reject(rows)

    def test_padding_sentinel_and_inverse_image(self):
        for boundaries in ([0, 2, 3, 8], [0, 1, 3, 0], [0, 1, 2, 8], [0, 1, 1, 8]):
            rows = copy.deepcopy(self.records)
            rows[1]["compiled"]["boundaries"] = boundaries
            self.reject(rows)
        rows = copy.deepcopy(self.records)
        rows[2]["compiled"]["ordinals"][0] = 0
        self.reject(rows)

    def test_fixed_shape_and_zero_threshold_slot(self):
        rows = copy.deepcopy(self.records)
        rows[2]["native_bytes"] += 1
        self.reject(rows)
        rows = copy.deepcopy(self.records)
        rows[2]["shape"]["queries"] -= 1
        self.reject(rows)
        rows = copy.deepcopy(self.records)
        # mask=5 and beta=3 creates zero threshold, still mandatory query slot.
        self.assertIn([3, 0, 0], rows[3]["compiled"]["queries"])
        rows[3]["compiled"]["queries"].remove([3, 0, 0])
        self.reject(rows)

    def test_wrong_low_mask_shift_query_order(self):
        for query in ([2, 0, 5], [2, 1, 0], [3, 0, 0]):
            rows = copy.deepcopy(self.records)
            rows[3]["compiled"]["queries"][2] = query
            self.reject(rows)

    def test_nearest_even_mutation(self):
        rows = copy.deepcopy(self.records)
        rows[-1]["external_encoding"][0] = -1
        self.reject(rows)

    def test_suite_binding(self):
        rows = copy.deepcopy(self.records)
        rows[0]["suite"] = "gelu"
        self.reject(rows)

    def test_gelu_suite_filters_without_changing_order(self):
        fixture = copy.deepcopy(self.fixture)
        fixture["receipt"] = {"canonical_gelu_descriptor_sha256": fixture["descriptors"][0]["id"]}
        rows = copy.deepcopy(self.records)
        rows[0]["suite"] = "gelu"
        raw = b"".join(canonical(r) + b"\n" for r in rows)
        result = check_stream(io.BytesIO(raw), fixture, "a" * 64, "gelu")
        self.assertEqual(result["checked"]["evaluations"], 24)

    def test_json_duplicates_trailing_unterminated_and_budget(self):
        raw = b"".join(canonical(r) + b"\n" for r in self.records)
        for broken in (raw + b"\n", raw[:-1], raw.replace(b'"mask":0', b'"mask":0,"mask":0', 1),
                       b" " * ((2 << 20) + 1)):
            with self.assertRaises(Mismatch):
                check_stream(io.BytesIO(broken), self.fixture, "a" * 64)

    def test_gelu_floor_and_extra_rescale_mutations_rejected(self):
        # Hand-specified mask-zero Protocol-3 padding; numeric answers are
        # mathematical counterexamples, not any compiler's output.
        d = gelu_descriptor(load_table())
        ident = digest(d)
        n = 1 << 50
        shape = {"source_intervals": 511, "padded_intervals": 512, "boolean_outputs": 0,
                 "queries": 0, "degree": 1, "coefficients_per_interval": 2, "horner_products": 1}
        run = {"id": "negative-near-zero-and-q0", "descriptor_id": ident, "mask": 0,
               "context": "hand-gelu", "words": [n - 1, 64], "expected": [[n - 31, []], [33, []]]}
        ordinals = [0] + list(range(511))
        record = {**{k: run[k] for k in ("id", "descriptor_id", "mask", "context")},
                  "shape": shape, "native_bytes": 11846, "outputs": run["expected"],
                  "compiled": {"boundaries": [0, 1] + d["boundaries"][1:], "ordinals": ordinals,
                               "payloads": [d["pieces"][i]["coefficients"] for i in ordinals], "queries": []}}
        fixture = {"descriptors": [{"id": ident, "descriptor": d, "shape": shape}], "runs": [run],
                   "external_encoding": [], "counts": {"evaluations": 2}}
        rows = [self.records[0], record, {"external_encoding": []}]
        self.assertEqual(check_records(fixture, rows)["status"], "pass")
        for i in (0, 1):
            wrong = copy.deepcopy(rows)
            wrong[1]["outputs"][i][0] = 0  # abs-first/round-to-zero or extra ARS12
            with self.assertRaises(Mismatch):
                check_records(fixture, wrong)


class IndependentMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = load_table()

    def test_exact_formula_hash_and_host_formula_crosscheck(self):
        self.assertEqual(digest(gelu_descriptor(self.table)), GELU_DIGEST)
        for j, entry in enumerate(self.table):
            # Supplement only: exact certified bytes, NOT libm, define floors.
            self.assertEqual(entry, math.floor(2048 * (j / 64) * math.erfc((j / 64) / math.sqrt(2))))

    def test_floor_abs_q0_and_original_x_counterexamples(self):
        n = 1 << 50
        self.assertEqual(gelu(n - 1, self.table), n - 31)
        self.assertNotEqual(gelu(n - 1, self.table), -self.table[abs(-1) // 64] % n)
        self.assertEqual(gelu(64, self.table), 33)
        self.assertNotEqual(gelu(64, self.table), 33 // 4096)
        self.assertNotEqual((1 + 3 * 2) % 8, (1 + 3 * ((2 + 4) % 8)) % 8)

    def test_source_range_vs_total_ring(self):
        n = 1 << 50
        for x in (-(1 << 49), -(1 << 37) - 1, (1 << 37), (1 << 37) + 64, (1 << 49) - 1):
            result = gelu(x % n, self.table)
            self.assertEqual(result, max(0, x) % n)  # clipped far-tail table is zero
        self.assertEqual(gelu((1 << 37) + 64, self.table), (1 << 37) + 64)
        self.assertEqual((((1 << 37) + 64) % (1 << 38)) - (1 << 38), -(1 << 37) + 64)

    def test_external_ties_independently_solved(self):
        for a, expected in [(-7, -4), (-5, -2), (-3, -2), (-1, 0), (1, 0), (3, 2), (5, 2), (7, 4)]:
            self.assertEqual(nearest_even(a, 8192), expected)

    def test_all_small_predicates_truth_tables(self):
        for bits in range(1, 5):
            n = 1 << bits
            for x in range(n):
                for b in range(-n // 2, n // 2 + 1):
                    expected = int((x if x < n // 2 else x - n) < b)
                    self.assertEqual(predicate(["slt", b], x, bits), expected)
                for k in range(1, bits + 1):
                    for b in range((1 << k) + 1):
                        self.assertEqual(predicate(["low_lt", k, b], x, bits), int(x % (1 << k) < b))

    def test_frozen_fixture_reproducible_and_bounded(self):
        fixture, checksum = load_fixture()
        self.assertEqual(len(checksum), 64)
        self.assertEqual(fixture["counts"]["small_partition_evaluations"], 8328)
        self.assertLessEqual(fixture["counts"]["evaluations"], 100000)
        for item in fixture["descriptors"]:
            self.assertEqual(digest(item["descriptor"]), item["id"])

    def test_rounding_descriptor_mutation_changes_identity(self):
        d = gelu_descriptor(self.table)
        for key, value in (("internal_rounding", "nearest-even"), ("coefficient_fractional", [12, 12]),
                           ("selector_semantics", "abs-before-floor")):
            mutated = copy.deepcopy(d)
            mutated[key] = value
            self.assertNotEqual(digest(mutated), GELU_DIGEST)

    def test_receipt_matches_frozen_fixture(self):
        fixture, checksum = load_fixture()
        receipt = json.loads((Path(__file__).resolve().parent / "receipt.json").read_bytes())
        self.assertEqual(receipt["fixture_sha256"], checksum)
        self.assertEqual(receipt["independent_gate_evaluations"], fixture["counts"]["evaluations"])
        self.assertEqual(receipt["runs"], fixture["counts"]["runs"])


if __name__ == "__main__":
    unittest.main()
