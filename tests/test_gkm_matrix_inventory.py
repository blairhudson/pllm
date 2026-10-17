"""Source-owner reconciliation and conditional publication geometry, not execution."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from scripts import probe_gkm_checkpoint_bounds as probe


def fixture():
    from pllm.modeling import ModelPlan

    operations = [
        {"id": "lookup", "operator": "token_lookup", "inputs": ["input.tokens"], "layer": None,
         "output_shape": [1, 1, 2], "attributes": {"weight": "opaque.shared"}},
        {"id": "a", "operator": "linear", "inputs": ["lookup"], "layer": 0,
         "output_shape": [1, 1, 2], "attributes": {"weight": "opaque.a", "bias": "opaque.bias"}},
        {"id": "b", "operator": "linear", "inputs": ["lookup"], "layer": 0,
         "output_shape": [1, 1, 1], "attributes": {"weight": "opaque.b"}},
        {"id": "last", "operator": "last_token", "inputs": ["a"], "layer": None,
         "output_shape": [1, 2], "attributes": {"axis": 1}},
        {"id": "head", "operator": "output_head", "inputs": ["last"], "layer": None,
         "output_shape": [1, 3], "attributes": {"weight": "opaque.shared"}},
    ]
    phases = {phase: {"mode": phase, "batch": 1, "query_sequence": 1,
                      "maximum_key_sequence": 1 if phase == "prefill" else 2,
                      "operations": operations} for phase in ("prefill", "decode")}
    plan = ModelPlan(probe.canonical({"schema_version": "pllm.decoder_plan.v1",
                                    "config_digest": "a" * 64, **phases}))
    tensors = {"opaque.shared": {"shape": [3, 2], "elements": 6, "dtype": "BF16"},
               "opaque.a": {"shape": [2, 2], "elements": 4, "dtype": "BF16"},
               "opaque.b": {"shape": [1, 2], "elements": 2, "dtype": "F16"},
               "opaque.bias": {"shape": [2], "elements": 2, "dtype": "F32"}}

    def member(index, offset):
        op = operations[index]
        bias = op["attributes"].get("bias")
        return {"operation_id": op["id"], "weight_id": op["attributes"]["weight"],
                "bias_id": bias, "source_shape": tensors[op["attributes"]["weight"]]["shape"],
                "bias_shape": None if bias is None else tensors[bias]["shape"],
                "output_shape": op["output_shape"], "output_offset": offset,
                "output_width": op["output_shape"][-1]}

    groups = [
        {"order": 0, "operator": "token_lookup", "layer": None,
         "input_id": "input.tokens", "input_shape": [1, 1],
         "artifact_kind": "token_table", "orientation": "vocabulary_feature",
         "logical_shape": [3, 2], "members": [member(0, 0)]},
        {"order": 1, "operator": "linear", "layer": 0,
         "input_id": "lookup", "input_shape": [1, 1, 2],
         "artifact_kind": "matrix", "orientation": "output_input",
         "logical_shape": [3, 2], "members": [member(1, 0), member(2, 2)]},
        {"order": 2, "operator": "output_head", "layer": None,
         "input_id": "last", "input_shape": [1, 2],
         "artifact_kind": "matrix", "orientation": "output_input",
         "logical_shape": [3, 2], "members": [member(4, 0)]},
    ]
    native = {"schema_version": "pllm.decoder_matrix_inventory.v1",
              "model_plan_digest": plan.digest, "model_config_digest": "a" * 64,
              **{phase: {key: value for key, value in row.items() if key != "operations"}
                 | {"groups": deepcopy(groups)} for phase, row in phases.items()}}
    return plan, tensors, native


def api(plan, document, *, corrupt_digest=False):
    """Deliberately injected binding double: these checks do not qualify native code."""
    def lower(encoded):
        assert encoded == plan.canonical_bytes()
        raw = probe.canonical(document)
        digest = hashlib.sha256(b"pllm.decoder_matrix_inventory.v1\0" + raw).hexdigest()
        return raw, "0" * 64 if corrupt_digest else digest

    return SimpleNamespace(decoder_matrix_inventory=lower)


def test_grouped_geometry_keeps_table_head_encodings_and_bias_owner_separate():
    plan, tensors, native = fixture()
    inventory = probe.semantic_inventory(plan, tensors, matrix_api=api(plan, native))
    geometry = probe.geometry(inventory, tensors, 1000)
    assert inventory["fused_groups"]["inventory"] == native
    assert inventory["fused_groups"]["final_gkm_publication_layout"] == "not_admitted"
    assert geometry["body_i16_bytes_per_worker"] == 12
    assert geometry["head_i16_bytes_per_worker"] == 12
    assert geometry["raw_float32_table_bytes_per_worker"] == 24
    assert geometry["table_plus_linear_payload_bytes_per_worker"] == 48
    assert geometry["client_bias_float32_bytes"] == 8
    grouped = geometry["conditional_compiler_grouped_publication"]
    assert grouped["status"] == "geometry_only_not_published"
    assert grouped["matrix_count"] == 2
    assert grouped["body_group_count"] == grouped["head_group_count"] == grouped["table_group_count"] == 1
    assert grouped["global_R_elements"] == 3
    assert grouped["S_elements"] == 4
    assert grouped["compressed_points_33_bytes_each"] == 231
    assert grouped["client_accumulator_scalar_payload_32_bytes_each"] == 320
    assert geometry["conditional_unfused_publication"]["compressed_points_33_bytes_each"] == 297
    assert geometry["conditional_unfused_publication"]["client_accumulator_scalar_payload_32_bytes_each"] == 384


@pytest.mark.parametrize("mutation,reason", [
    (lambda value: value.update(model_plan_digest="b" * 64), "binding mismatch"),
    (lambda value: value.update(model_config_digest="b" * 64), "binding mismatch"),
    (lambda value: value.update(schema_version="unsupported"), "binding mismatch"),
    (lambda value: value["decode"].update(query_sequence=2), "phase binding"),
    (lambda value: value["prefill"]["groups"].pop(), "operation coverage"),
    (lambda value: value["prefill"]["groups"][1]["members"].pop(), "grouped geometry"),
    (lambda value: value["decode"]["groups"][1]["members"].append(
        deepcopy(value["decode"]["groups"][1]["members"][0])), "operation coverage"),
    (lambda value: value["decode"]["groups"][1]["members"][0].update(source_shape=[4, 2]), "source/header"),
    (lambda value: value["decode"]["groups"][1]["members"][0].update(bias_id=None), "source/header"),
    (lambda value: value["decode"]["groups"][-1].update(artifact_kind="token_table"), "source/header"),
    (lambda value: value["prefill"]["groups"][1].update(logical_shape=[3000, 2]), "grouped geometry"),
    (lambda value: value["prefill"]["groups"][0].update(logical_shape=[2, 3]), "grouped geometry"),
    (lambda value: value["decode"]["groups"][1].update(input_shape=[1, 1, 3]), "grouped geometry"),
    (lambda value: value["decode"]["groups"][1].update(input_shape=[True, 1, 2]), "grouped geometry"),
    (lambda value: value["decode"]["groups"][1]["members"][1].update(output_offset=1), "output-slice geometry"),
    (lambda value: value["decode"]["groups"][1]["members"][1].update(output_width=100), "output-slice geometry"),
    (lambda value: value["decode"]["groups"][1]["members"][1].update(output_width=True), "output-slice geometry"),
    (lambda value: value["decode"]["groups"].__setitem__(1, []), "group must be an object"),
    (lambda value: value["decode"]["groups"][1]["members"].__setitem__(1, []), "member must be an object"),
])
def test_native_metadata_cannot_supply_other_plan_source_or_incomplete_coverage(mutation, reason):
    plan, tensors, native = fixture()
    mutation(native)
    with pytest.raises(probe.Rejected, match=reason):
        probe.semantic_inventory(plan, tensors, matrix_api=api(plan, native))


def test_explicit_native_inventory_request_cannot_fall_back_or_accept_bad_digest():
    plan, tensors, native = fixture()
    with pytest.raises(probe.Rejected, match="rebuild"):
        probe.semantic_inventory(plan, tensors, matrix_api=SimpleNamespace())
    with pytest.raises(probe.Rejected, match="digest mismatch"):
        probe.semantic_inventory(plan, tensors, matrix_api=api(plan, native, corrupt_digest=True))


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_actual_native_inventory_reconciles_generated_family_source_headers(family):
    from pllm import _native, lower_model
    from pllm.runtime.semantic_tensors import required_checkpoint_tensors

    if not hasattr(_native, "decoder_matrix_inventory"):
        pytest.skip("native extension must be rebuilt with decoder_matrix_inventory")
    config = {"model_type": family, "hidden_size": 8, "intermediate_size": 16,
              "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
              "vocab_size": 32, "max_position_embeddings": 32, "hidden_act": "silu",
              "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True}
    if family == "qwen3":
        config.update(head_dim=4, attention_bias=False, attention_dropout=0.0,
                      use_sliding_window=False, use_cache=True, max_window_layers=2)
    plan = lower_model(config, batch=1, max_input_tokens=1, max_new_tokens=1)
    tensors = {key: {"shape": list(shape), "elements": probe.math.prod(shape), "dtype": "BF16"}
               for key, shape in required_checkpoint_tensors(plan).items()}
    inventory = probe.semantic_inventory(plan, tensors, matrix_api=_native)
    grouped = probe.geometry(inventory, tensors, 0)["conditional_compiler_grouped_publication"]
    assert grouped["body_group_count"] == 8
    assert grouped["head_group_count"] == grouped["table_group_count"] == 1
    assert inventory["fused_groups"]["inventory"]["model_plan_digest"] == plan.digest
