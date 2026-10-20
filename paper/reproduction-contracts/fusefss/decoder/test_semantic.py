"""Independent strict semantic contract mutation tests; no native execution."""
import copy
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from check_semantic import check
from freeze_semantic import original_unchanged
from oracle import HERE, canonical, read_json, strict_equal
from semantic_reference import expected, run_semantic, validate_graph


class SemanticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = expected()
        cls.inputs = read_json(HERE / "inputs.json")["cases"]
        cls.graphs = read_json(HERE / "semantic-graphs.json")["cases"]

    def rejected(self, mutate):
        result = copy.deepcopy(self.data)
        mutate(result)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_bytes(canonical(result))
            with self.assertRaises(ValueError):
                check(path)

    def test_positive_is_clear_reference_only_not_native_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clear.json"
            path.write_bytes(canonical(self.data))
            self.assertEqual(check(path), 5)

    def test_original_payload_arithmetic_and_5125_tensors_unchanged(self):
        original_unchanged()
        self.assertEqual(sum(len(r["tensors"]) for r in read_json(HERE / "expected.json")["results"]), 5125)

    def test_all_nodes_all_selections_including_padding(self):
        self.assertEqual([len(r["nodes"]) for r in self.data["results"]], [174, 174, 186, 198, 186])
        self.assertEqual(sum(len(r["nodes"]) for r in self.data["results"]), 918)
        for result in self.data["results"]:
            heads = [n for n in result["nodes"] if n["id"] == "output_head"]
            self.assertEqual(len(heads), 3)
            self.assertTrue(all(n["shape"] == [1, 7] for n in heads))
            first = result["nodes"][0]
            self.assertEqual(first["shape"][1], 4)
            self.assertEqual(len(first["values"]), 4 * first["shape"][-1])

    def test_active_checkpoints_and_full_vocab_match_original_oracle(self):
        originals = read_json(HERE / "expected.json")["results"]
        for original, result in zip(originals, self.data["results"]):
            tensors = {t["name"]: t for t in original["tensors"]}
            for node in result["nodes"]:
                step = node["selection"]
                prefix = len(self.inputs[0]["prefix"]) + step
                if node["id"] == "final_norm":
                    width = node["shape"][-1]
                    strict_equal(tensors[f"step.{step}.final_norm.output"]["values"], node["values"][:prefix * width])
                if node["id"] == "output_head":
                    old = tensors[f"step.{step}.head.output"]["values"]
                    vocab = node["shape"][-1]
                    strict_equal(old[(prefix - 1) * vocab:prefix * vocab], node["values"])
            strict_equal(original["final"], result["final"])

    def test_missing_case(self):
        self.rejected(lambda r: r["results"].pop())

    def test_extra_case(self):
        self.rejected(lambda r: r["results"].append(copy.deepcopy(r["results"][0])))

    def test_reordered_cases(self):
        self.rejected(lambda r: r["results"].reverse())

    def test_missing_node(self):
        self.rejected(lambda r: r["results"][0]["nodes"].pop(12))

    def test_extra_node(self):
        self.rejected(lambda r: r["results"][0]["nodes"].append(copy.deepcopy(r["results"][0]["nodes"][0])))

    def test_reordered_node(self):
        self.rejected(lambda r: r["results"][0]["nodes"].reverse())

    def test_changed_active_value(self):
        self.rejected(lambda r: r["results"][2]["nodes"][0]["values"].__setitem__(0, 99))

    def test_changed_padding_value(self):
        self.rejected(lambda r: r["results"][0]["nodes"][0]["values"].__setitem__(-1, 99))

    def test_wrong_shape(self):
        self.rejected(lambda r: r["results"][0]["nodes"][0]["shape"].__setitem__(1, 2))

    def test_qkv_selection_and_feedback_values_are_required(self):
        for id_ in ("layer.0.q_heads", "layer.0.k_heads", "layer.0.v_heads", "token_selection", "token_feedback"):
            def mutate(result, id_=id_):
                node = next(n for n in result["results"][0]["nodes"] if n["id"] == id_)
                node["values"][0] += 1
            self.rejected(mutate)

    def test_profile_hashes_and_fractional_metadata_locked(self):
        for field in ("profile", "input_sha256", "semantic_graph_sha256"):
            self.rejected(lambda r, field=field: r["results"][0].__setitem__(field, "changed"))
        self.rejected(lambda r: r["results"][0]["nodes"][0].__setitem__("fractional", 0))

    def test_empty_batch_and_reordered_feedback_states_reject(self):
        self.rejected(lambda r: r["results"].clear())
        self.rejected(lambda r: r["results"][0]["states"].reverse())

    def test_source_encoding_changed(self):
        self.rejected(lambda r: r["results"][0]["source_encodings"][0]["q12_values"].__setitem__(0, 99))

    def test_source_ieee_word_changed(self):
        self.rejected(lambda r: r["results"][0]["source_encodings"][0]["ieee_words"].__setitem__(0, "00000000"))

    def test_source_tensor_missing_or_reordered(self):
        self.rejected(lambda r: r["results"][0]["source_encodings"].pop())
        self.rejected(lambda r: r["results"][0]["source_encodings"].reverse())

    def test_partial_vocabulary(self):
        self.rejected(lambda r: next(n for n in r["results"][0]["nodes"] if n["id"] == "output_head")["values"].pop())

    def test_changed_selection_feedback_final_and_state(self):
        for field in ("selected_id_private", "active_prefix_ids_private", "input_token_buffer_private"):
            self.rejected(lambda r, field=field: r["results"][0]["states"][1].__setitem__(field, 99))
        self.rejected(lambda r: r["results"][0]["final"]["generated_ids"].__setitem__(0, 0))

    def test_numeric_type_and_extra_primitive_trace_rejected(self):
        for value in (True, 0.0):
            self.rejected(lambda r, value=value: r["results"][0]["nodes"][0]["values"].__setitem__(0, value))
        self.rejected(lambda r: r["results"][0].__setitem__("primitive_registers", []))

    def test_graph_id_dependency_order_and_family_locked(self):
        for mutate in (lambda g: g["operators"].pop(), lambda g: g["operators"].reverse(),
                       lambda g: g["operators"][1]["inputs"].append("unbound.native.input")):
            graph = copy.deepcopy(self.graphs[0])
            mutate(graph)
            with self.assertRaises(ValueError):
                validate_graph(self.inputs[0], graph)

    def test_original_schema_is_not_semantic_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.json"
            path.write_bytes((HERE / "expected.json").read_bytes())
            with self.assertRaises(ValueError):
                check(path)

    def test_duplicate_keys_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema":"x","schema":"y","results":[]}')
            with self.assertRaises(ValueError):
                check(path)


if __name__ == "__main__":
    unittest.main()
