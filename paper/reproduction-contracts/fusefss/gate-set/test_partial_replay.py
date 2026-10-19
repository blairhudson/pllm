"""Synthetic checker controls only. No native outputs produced by these tests."""

import copy
from hashlib import sha256
import json
import unittest
import check_outputs as gates
import check_tables as tables
import reference as ref


def encode(records):
    return b"".join(ref.canonical(record) for record in records)


class PartialGateReplay(unittest.TestCase):
    labels = "shift,b2a,compare,phi-select,external"

    @classmethod
    def setUpClass(cls):
        frozen, fixture = gates.verify_fixture()
        known = list(dict.fromkeys(case["gate"] for case in fixture["cases"]))
        selected = gates.selection(cls.labels, known)
        cases = [case for case in fixture["cases"] if case["gate"] in selected]
        cls.records = [{"schema": "pllm.fusefss.gate-set.v2/selected-native-output",
                        "fixture_sha256": sha256(frozen).hexdigest(),
                        "source_lock_sha256": fixture["source_lock_sha256"],
                        "execution": "protected-local-reconstruction", "implementation": "synthetic-checker-control",
                        "coverage": "selected-gates-only", "selected_gates": selected, "selected_case_count": len(cases)}]
        cls.records += [{"id": case["id"], "gate": case["gate"], "output": case["expected"]} for case in cases]

    def test_partial_exact_selection(self):
        result = gates.check(encode(self.records), self.labels)
        self.assertEqual(result["cases"], 66)
        self.assertEqual(result["coverage"], "selected-gates-only")
        self.assertEqual(gates.check(encode(self.records), ",".join(reversed(self.labels.split(","))))["cases"], 66)

    def test_no_implicit_partial_or_full_claim(self):
        with self.assertRaises(ValueError):
            gates.check(encode(self.records))
        for key, value in (("coverage", "all-frozen-cases"), ("selected_case_count", 275),
                           ("selected_case_count", True), ("selected_gates", self.labels),
                           ("selected_gates", list(reversed(self.records[0]["selected_gates"]))),
                           ("source_lock_sha256", "0" * 64),
                           ("schema", "pllm.fusefss.gate-set.v1/native-output")):
            records = copy.deepcopy(self.records)
            records[0][key] = value
            with self.assertRaises(ValueError):
                gates.check(encode(records), self.labels)

    def test_labels_strict(self):
        for labels in ("", ",", "shift,", ",shift", "shift,shift", "shift,unknown", " shift", "Shift"):
            with self.assertRaises(ValueError):
                gates.check(encode(self.records), labels)

    def test_case_shape_order_and_numeric_types(self):
        controls = [self.records[:-1], self.records + [self.records[-1]],
                    [self.records[0], self.records[2], self.records[1], *self.records[3:]]]
        for value in (True, 0.0, "0", None, {"bits": 8, "word": True}):
            records = copy.deepcopy(self.records)
            records[1]["output"] = value
            controls.append(records)
        for records in controls:
            with self.assertRaises(ValueError):
                gates.check(encode(records), self.labels)


class TableReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frozen = (ref.HERE / "table-fixtures.json").read_bytes()
        fixture = json.loads(frozen)
        _, base = gates.verify_fixture()
        cls.records = [{"schema": "pllm.fusefss.table-replay.v1/native-output",
                        "fixture_sha256": sha256(frozen).hexdigest(),
                        "gate_fixture_sha256": fixture["gate_fixture_sha256"],
                        "source_lock_sha256": fixture["source_lock_sha256"],
                        "execution": "protected-local-reconstruction", "implementation": "synthetic-checker-control",
                        "coverage": "public-table-lookup-only", "table_count": fixture["table_count"],
                        "case_count": fixture["case_count"]}]
        cls.records += [{"table": name, "sha256_u64le": base["sha256_u64le"][name],
                         "outputs": base["tables"][name]} for name in tables.TABLES]

    def test_fixture_and_synthetic_acceptance(self):
        self.assertEqual((ref.HERE / "table-fixtures.json").read_bytes(), ref.canonical(tables.fixture()))
        result = tables.check(encode(self.records))
        self.assertEqual((result["tables"], result["cases"]), (8, 18560))
        self.assertEqual(result["coverage"], "public-table-lookup-only")

    def test_wrong_header_binding_full_claim(self):
        for key, value in (("fixture_sha256", "0" * 64), ("gate_fixture_sha256", "0" * 64),
                           ("source_lock_sha256", "0" * 64), ("case_count", 8), ("table_count", True),
                           ("coverage", "complete-gates"), ("execution", "clear"),
                           ("schema", "pllm.fusefss.gate-set.v1/native-output")):
            records = copy.deepcopy(self.records)
            records[0][key] = value
            with self.assertRaises(ValueError):
                tables.check(encode(records))

    def test_missing_reorder_duplicate_output_type_and_mutations(self):
        controls = [self.records[:-1], self.records + [self.records[-1]],
                    [self.records[0], self.records[2], self.records[1], *self.records[3:]]]
        for key, value in (("table", "unknown"), ("sha256_u64le", "0" * 64),
                           ("outputs", [0]), ("outputs", self.records[1]["outputs"][:-1])):
            records = copy.deepcopy(self.records)
            records[1][key] = value
            controls.append(records)
        for value in (True, 0.0, -1, 1 << 50, "0"):
            records = copy.deepcopy(self.records)
            records[1]["outputs"][0] = value
            controls.append(records)
        for records in controls:
            with self.assertRaises(ValueError):
                tables.check(encode(records))

    def test_serialization_and_bounds(self):
        data = encode(self.records)
        for mutated in (data[:-1], data + b"\n", data.replace(b'"outputs":', b'"outputs":[],"outputs":', 1),
                        b"x" * (tables.MAX_FILE + 1)):
            with self.assertRaises(ValueError):
                tables.check(mutated)


if __name__ == "__main__":
    unittest.main()
