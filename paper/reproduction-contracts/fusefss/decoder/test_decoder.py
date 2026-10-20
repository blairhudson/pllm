"""Pure Python arithmetic, generated feedback and adversarial exchange controls."""

from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

from generate import fixture, fixtures
from oracle import (GRAPH, HALF, HERE, Trace, activation, canonical, checked, dot, graph_profile,
                    external_tensor, ieee_q12, linear, load_tables, norm, read_json,
                    rotary, run, softmax, strict_equal, validate_contract)


def scalar_reference():
    spec = importlib.util.spec_from_file_location("decoder_test_gate_ref", HERE.parent / "gate-set" / "reference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Encoding(unittest.TestCase):
    def test_ieee_signed_zero_subnormal_and_nonfinite(self):
        for dtype, positive, negative, subnormal, infinity, nan in (
            ("f32", "00000000", "80000000", "00000001", "7f800000", "7fc00000"),
            ("f16", "0000", "8000", "0001", "7c00", "7e00"),
            ("bf16", "0000", "8000", "0001", "7f80", "7fc0"),
        ):
            for bits in (positive, negative, subnormal):
                self.assertEqual(ieee_q12(dtype, bits), 0)
            for bits in (infinity, nan):
                with self.assertRaises(ValueError):
                    ieee_q12(dtype, bits)

    def test_exact_ties_and_negative_nearest_even(self):
        for n in (-9, -7, -5, -3, -1, 1, 3, 5, 7, 9):
            for dtype, fmt in (("f32", ">f"), ("f16", ">e"), ("bf16", ">f")):
                bits = struct.pack(fmt, n / 8192).hex()
                if dtype == "bf16":
                    bits = bits[:4]
                self.assertEqual(ieee_q12(dtype, bits), round(Fraction(n, 2)))

    def test_all_f16_patterns_against_exact_python_fraction(self):
        # Binary64 represents every finite F16 exactly; independent exhaustive
        # decoder/rounding control, including every subnormal and negative tie.
        for bits in range(65536):
            if (bits >> 10) & 31 == 31:
                continue
            value = struct.unpack(">e", bits.to_bytes(2, "big"))[0]
            self.assertEqual(ieee_q12("f16", f"{bits:04x}"), round(Fraction(value) * 4096))

    def test_all_admitted_bf16_patterns_against_fraction(self):
        admitted = rejected = 0
        for bits in range(65536):
            if (bits >> 7) & 255 == 255:
                continue
            value = struct.unpack(">f", (bits << 16).to_bytes(4, "big"))[0]
            expected = round(Fraction(value) * 4096)
            if -HALF <= expected < HALF:
                self.assertEqual(ieee_q12("bf16", f"{bits:04x}"), expected)
                admitted += 1
            else:
                with self.assertRaises(ValueError):
                    ieee_q12("bf16", f"{bits:04x}")
                rejected += 1
        self.assertGreater(admitted, 0)
        self.assertGreater(rejected, 0)

    def test_malformed_encodings_and_tensor_shapes(self):
        for dtype, bits in (("f64", "0000"), ("f32", "0"), ("f16", "FFFF"), ("f16", 0), ("f32", "7f000000")):
            with self.assertRaises(ValueError):
                ieee_q12(dtype, bits)
        with self.assertRaises(ValueError):
            external_tensor({"shape": [True], "dtype": "f32", "words": ["00000000"]}, [1])

    def test_f32_boundaries_and_deterministic_bit_sweep(self):
        self.assertEqual(ieee_q12("f32", "d2000000"), -HALF)
        with self.assertRaises(ValueError):
            ieee_q12("f32", "52000000")
        # LCG bit words, independent IEEE decode via struct/Fraction.
        bits = 7
        for _ in range(4096):
            bits = (1664525 * bits + 1013904223) % (1 << 32)
            if (bits >> 23) & 255 == 255:
                continue
            value = struct.unpack(">f", bits.to_bytes(4, "big"))[0]
            expected = round(Fraction(value) * 4096)
            if -HALF <= expected < HALF:
                self.assertEqual(ieee_q12("f32", f"{bits:08x}"), expected)
            else:
                with self.assertRaises(ValueError):
                    ieee_q12("f32", f"{bits:08x}")


class Arithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_tables()
        cls.scalar = scalar_reference()

    def test_negative_floor_and_accumulate_then_shift(self):
        tr = Trace()
        self.assertEqual(linear([[-1, -1]], [[1, 1]], [0], tr, "linear"), [[-1]])
        self.assertEqual(linear([[1, 1]], [[2048, 2048]], [3], Trace(), "linear"), [[4]])
        self.assertEqual(tr.values[0]["values"], [-2])
        self.assertNotEqual((-1 // 4096) * 2, -2 // 4096)

    def test_scalar_gate_differential(self):
        samples = [-16385, -4097, -4096, -65, -64, -63, -1, 0, 1, 63, 64, 65, 4096, 65473]
        for gate in ("gelu", "silu"):
            result = activation([samples], self.tables[gate], Trace(), gate)[0]
            for x, y in zip(samples, result):
                scalar_word = self.scalar.activation(x, gate)["output"]
                expected = scalar_word if scalar_word < HALF else scalar_word - 2 * HALF
                self.assertEqual(y, expected)
        for kind in ("layer", "rms"):
            for row in ([4096] * 4, [-65, 64], [-4096, -4096, 4096, 4096], [0] * 8, [4096] * 768):
                output = norm([row], [4096] * len(row), [0] * len(row), kind, Trace(), "norm")[0]
                self.assertEqual(output, self.scalar.norm(row, kind=kind)["output"])
        for row in ([0], [0, 0], [0, 0, 0], [0, -4096], [0, 99999, -4096]):
            for p in range(len(row)):
                expected = self.scalar.softmax(list(row), [int(t <= p) for t in range(len(row))])["output"]
                self.assertEqual(softmax(row, p, self.tables, Trace(), "softmax"), expected)

    def test_no_softmax_renormalization_and_mask(self):
        self.assertEqual(softmax([0, -4096], 1, self.tables, Trace(), "s"), [3013, 1107])
        self.assertEqual(softmax([0, 99999], 0, self.tables, Trace(), "s"), [4096, 0])

    def test_norm_guard_and_non_power_of_two_mean(self):
        self.assertEqual(norm([[0] * 4], [4096] * 4, [0] * 4, "rms", Trace(), "n"), [[0] * 4])
        tr = Trace()
        self.assertEqual(norm([[4096] * 768], [4096] * 768, [0] * 768, "layer", tr, "n"), [[4096] * 768])
        self.assertEqual(next(t["values"] for t in tr.values if t["name"] == "n.mean"), [3840])

    def test_rotary_rescale_order_distinguishable(self):
        # Per-term floor gives -2, combined-product floor gives -1.
        x = [[[-1, 1]]]
        result = rotary(x, [[1, 1]], [[1, 1]], Trace(), "r")
        self.assertEqual(result, [[[-2, -1]]])
        self.assertNotEqual(result[0][0][0], (-1 - 1) // 4096)

    def test_overflow_checked_before_cancellation_or_shift(self):
        for op in (lambda: checked(HALF), lambda: checked(True),
                   lambda: dot([HALF - 1, -(HALF - 1)], [2, 2]),
                   lambda: dot([HALF - 1, 1], [1, 1]),
                   lambda: norm([[HALF // 2, 0]], [4096, 4096], [0, 0], "rms", Trace(), "n")):
            with self.assertRaises(ValueError):
                op()


class Decoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fixtures()
        cls.outputs = [run(c) for c in cls.inputs]

    def test_multiple_full_prefix_feedback_and_no_release(self):
        for c, result in zip(self.inputs, self.outputs):
            self.assertEqual(result["graph_status"], GRAPH)
            self.assertEqual(result["profile"], graph_profile(c["family"]))
            self.assertEqual(len(result["states"]), 3)
            self.assertEqual([s["prefix_length_in"] for s in result["states"]], [2, 3, 4])
            for step, state in enumerate(result["states"]):
                self.assertEqual(state["positions_recomputed"], list(range(2 + step)))
                self.assertFalse(state["persistent_kv"])
                self.assertFalse(state["released"])
                next_id = result["final"]["generated_ids"][step]
                self.assertEqual(state["prefix_ids_private"], c["prefix"] + result["final"]["generated_ids"][:step + 1])
                self.assertEqual(state["selected_id_private"], next_id)
            self.assertEqual(result["final"]["recipient"], "client-only")
            self.assertEqual(result["final"]["selection_count"], 3)

    def test_ties_lowest_index_and_eos_not_stop(self):
        result = self.outputs[1]
        self.assertEqual(result["final"]["generated_ids"], [0, 0, 0])
        self.assertEqual(result["final"]["token_ids"], [2, 4, 0, 0, 0])
        equal = next(t for t in result["tensors"] if t["name"] == "step.0.argmax.equal_maximum")
        self.assertEqual(equal["values"], [1] * 7)

    def test_physical_gqa_axes_and_group_mapping(self):
        for c, result in zip(self.inputs, self.outputs):
            ts = {t["name"]: t for t in result["tensors"]}
            k = ts["step.0.layer.0.k.physical"]
            self.assertEqual(k["shape"], [1, c["dimensions"]["kv_heads"], 2, 4])
            self.assertEqual(k["axes"], ["batch", "heads", "sequence", "feature"])
            expected = {"gpt2": [0, 1], "qwen2": [0, 0, 1, 1], "qwen3": [0, 0], "llama": [0, 0]}[c["family"]]
            self.assertEqual(ts["step.0.layer.0.query_to_kv_head"]["values"], expected)
            self.assertEqual(ts["step.0.layer.0.head.1.softmax.0.mask"]["values"], [1, 0])
            if c["family"] == "qwen2":
                self.assertNotEqual(expected, [head % 2 for head in range(4)])
                physical = ts["step.0.layer.0.k.physical"]["values"]
                self.assertNotEqual(physical[:8], physical[8:])
                q = ts["step.0.layer.0.q_rope.output"]["values"][16:20]
                k = ts["step.0.layer.0.k_rope.output"]["values"]
                expected_dot = sum(a * b for a, b in zip(q, k[8:12]))
                wrong_modulo_dot = sum(a * b for a, b in zip(q, k[:4]))
                self.assertEqual(ts["step.0.layer.0.head.2.qk_accumulator"]["values"][0], expected_dot)
                self.assertNotEqual(expected_dot, wrong_modulo_dot)

    def test_family_structures_and_qk_norm_before_rope(self):
        names = [[t["name"] for t in out["tensors"]] for out in self.outputs]
        self.assertTrue(any("positions" == name.split(".")[-1] for name in names[0]))
        self.assertTrue(any(".gelu." in name for name in names[0]))
        self.assertFalse(any("_rope." in name for name in names[0]))
        for family_index in (2, 3, 4):
            self.assertTrue(any(".silu." in name for name in names[family_index]))
            self.assertTrue(any(".q_rope." in name for name in names[family_index]))
        q3 = names[3]
        self.assertLess(q3.index("step.0.layer.0.q_norm.head.0.output"), q3.index("step.0.layer.0.q_rope.output"))
        self.assertFalse(any("q_norm.head" in name for name in names[2]))

    def test_each_stage_and_feedback_mutations_detected(self):
        expected = self.outputs[0]
        representative_names = ["public.embedding", "step.0.embedding_equality", "step.0.embedding_output",
                                "step.0.layer.0.norm1.output", "step.0.layer.0.q.output",
                                "step.0.layer.0.head.0.scores", "step.0.layer.0.head.0.softmax.1.output",
                                "step.0.layer.0.attention.physical", "step.0.layer.0.attention_residual",
                                "step.0.layer.0.gelu.output", "step.0.layer.0.mlp_residual",
                                "step.0.final_norm.output", "step.0.head.output", "step.0.argmax.one_hot",
                                "step.0.appended_ids", "step.1.embedding_output", "step.2.selected_id"]
        for name in representative_names:
            mutated = deepcopy(expected)
            tensor = next(t for t in mutated["tensors"] if t["name"] == name)
            tensor["values"][0] += 1
            with self.assertRaisesRegex(ValueError, "value mismatch"):
                strict_equal(expected, mutated)
        for mutation in (lambda r: r["states"][0].update(released=True),
                         lambda r: r["states"][1].update(persistent_kv=True),
                         lambda r: r["tensors"][0].update(bits=True),
                         lambda r: r["final"].update(recipient="provider"),
                         lambda r: r.update(extra=0), lambda r: r["tensors"].pop()):
            mutated = deepcopy(expected)
            mutation(mutated)
            with self.assertRaises(ValueError):
                strict_equal(expected, mutated)

    def test_public_contract_mutations_rejected(self):
        mutations = [lambda c: c.update(graph_status="candidate-graph-not-approved"),
                     lambda c: c.update(profile="fusefss.dense-rope-ring50-q12.v1"),
                     lambda c: c.update(norm_contract="epsilon-1e-5"),
                     lambda c: c.update(rope_contract="cache"), lambda c: c.update(prefix=[True]),
                     lambda c: c.update(generation_count=6), lambda c: c["dimensions"].update(kv_heads=1),
                     lambda c: c["tensors"].update(unexpected={}), lambda c: c["tensors"].pop("positions"),
                     lambda c: c["tensors"]["embedding"].update(shape=[8, 7]),
                     lambda c: c["tensors"]["embedding"]["words"].__setitem__(0, "7f800000"),
                     lambda c: c.update(attention_scale={"shape": [1], "dtype": "f32", "words": ["00000000"]})]
        for mutation in mutations:
            c = deepcopy(self.inputs[0])
            mutation(c)
            with self.assertRaises(ValueError):
                validate_contract(c)

    def test_public_changes_change_computation_and_hash(self):
        c = deepcopy(self.inputs[0])
        c["tensors"]["positions"]["words"][0] = "3f80"  # generated position tensor is BF16
        actual = run(c)
        self.assertNotEqual(actual["input_sha256"], self.outputs[0]["input_sha256"])
        self.assertNotEqual(actual["tensors"], self.outputs[0]["tensors"])

    def test_d13_coefficient_contract_rejects_changed_scale_and_non_f32_rope(self):
        c = deepcopy(self.inputs[0])
        c["attention_scale"]["words"] = ["3f400000"]
        with self.assertRaises(ValueError):
            validate_contract(c)
        c = deepcopy(self.inputs[2])
        c["tensors"]["rope.cos"]["dtype"] = "bf16"
        c["tensors"]["rope.cos"]["words"] = [word[:4] for word in c["tensors"]["rope.cos"]["words"]]
        with self.assertRaises(ValueError):
            validate_contract(c)

    def test_gpt2_metadata_actual_bytes_and_source_only_scope(self):
        from fetch_gpt2_source import OUT, verify
        manifest = verify()
        self.assertFalse(manifest["weights_downloaded"])
        config = json.loads((OUT / "config.json").read_bytes())
        self.assertEqual(config["model_type"], "gpt2")
        self.assertEqual([config[k] for k in ("n_embd", "n_head", "n_layer", "n_positions", "vocab_size")],
                         [768, 12, 12, 1024, 50257])
        self.assertEqual(config["activation_function"], "gelu_new")
        self.assertEqual(config["layer_norm_epsilon"], 1e-5)
        absent = {r["path"] for r in manifest["files"] if r["http_status"] == 404}
        self.assertEqual(absent, {"special_tokens_map.json", "model.safetensors.index.json"})

    def test_strict_json_rejects_duplicate_nonfinite_and_float_int(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "bad.json"
            for data in (b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}'):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_json(path)
        with self.assertRaises(ValueError):
            strict_equal({"value": 1}, {"value": 1.0})

    def test_frozen_inputs_outputs_and_source_hashes(self):
        from freeze import expected, inputs, verify_lock
        verify_lock()
        strict_equal(inputs(), read_json(HERE / "inputs.json"))
        strict_equal(expected(inputs()), read_json(HERE / "expected.json"))

    def test_checker_positive_copy_and_negative_mutation_not_native_replay(self):
        from check_outputs import check
        # Feeding Python expected back into checker validates checker only.
        self.assertEqual(check(HERE / "expected.json"), 5)
        data = read_json(HERE / "expected.json")
        data["results"][2]["states"][1]["persistent_kv"] = True
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "mutated.json"
            path.write_bytes(canonical(data))
            with self.assertRaises(ValueError):
                check(path)

    def test_json_object_key_order_not_tensor_trace_order(self):
        c = self.inputs[0]
        reordered = json.loads(canonical(c))
        strict_equal(run(c), run(reordered))


if __name__ == "__main__":
    unittest.main()
