"""Ordered token boundary stages follow semantic artifacts, not model names."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.models import StageSpec
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.semantic_executor import SemanticLocalOperators
from pllm.runtime.semantic_stages import (
    _linear_stage,
    _token_lookup_stage,
    scheduled_stage_specs,
    semantic_stage_role,
)
from pllm.runtime.semantic_tensors import SemanticTensorError, preflight_semantic_checkpoint
from pllm.runtime.transformer_client import TransformerClientError


def test_native_frozen_contracts_materialize_real_io_checkpoint_stages(tmp_path):
    import asyncio
    from absolute_position_decoder_oracle import tiny_checkpoint, write_checkpoint
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    config, weights = tiny_checkpoint()
    root = write_checkpoint(tmp_path, config, weights)
    manifest = load_hf_directory(root, model_id="frozen-io-contract")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    assert any(stage.transpose_weight for stage in manifest.stages)
    asyncio.run(engine.load(manifest))
    stages = engine.models[manifest.id].stages
    assert set(stages) == {stage.id for stage in manifest.stages}
    for loaded in stages.values():
        result = loaded.compiled_weight.clear(np.zeros((1, loaded.spec.in_features), dtype=np.int8))
        assert result.shape == (1, loaded.spec.out_features)


def _pinned_token_group() -> tuple[dict, dict, int]:
    source = Path("crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json")
    config = json.loads(source.read_text())
    plan = lower_model(config, batch=1, max_input_tokens=2, max_new_tokens=2)
    operations = {operation["id"]: operation for operation in plan.to_dict()["prefill"]["operations"]}
    first = operations["main_embedding"]
    second = operations["ple_token_embedding"]
    first_width = first["output_shape"][-1]
    step = {
        "operation_ids": [first["id"], second["id"]],
        "operators": ["token_lookup", "token_lookup"],
        "input_ids": ["input.tokens"],
        "layer": None,
        "weight_ids": [first["attributes"]["weight"], second["attributes"]["weight"]],
        "outputs": [
            {"operation_id": first["id"], "stage_offset": 0, "stage_width": first_width},
            {
                "operation_id": second["id"],
                "stage_offset": first_width,
                "stage_width": second["output_shape"][-1],
            },
        ],
    }
    return step, operations, config["text_config"]["vocab_size"]


def test_pinned_token_tables_form_one_ordered_boundary_stage() -> None:
    step, operations, vocabulary = _pinned_token_group()
    stage = _token_lookup_stage(step, operations, vocabulary)
    assert stage.id == "token_lookup"
    assert stage.in_features == 262_144
    assert stage.out_features == 1536 + 8960
    assert stage.metadata == {"ple_width": 8960}
    assert stage.weight_keys == tuple(step["weight_ids"])

    forged = copy.deepcopy(step)
    forged["outputs"][1]["stage_offset"] -= 1
    with pytest.raises(ValueError, match="offsets"):
        _token_lookup_stage(forged, operations, vocabulary)
    forged = copy.deepcopy(step)
    forged["weight_ids"].reverse()
    with pytest.raises(ValueError, match="weights"):
        _token_lookup_stage(forged, operations, vocabulary)


def test_pinned_projection_roles_follow_nearest_semantic_producers() -> None:
    _, operations, _ = _pinned_token_group()

    def grouped(op_ids: list[str], order: int) -> tuple[str, StageSpec]:
        source = operations[op_ids[0]]
        step = {
            "operation_ids": op_ids,
            "operators": ["linear"] * len(op_ids),
            "input_ids": list(source["inputs"]),
            "layer": source["layer"],
            "order": order,
            "weight_ids": [operations[op_id]["attributes"]["weight"] for op_id in op_ids],
        }
        role = semantic_stage_role(step, operations)
        return role, _linear_stage(step, operations, role)

    role, boundary = grouped(["ple_context_projection"], 2)
    assert role == "semantic_linear"
    assert boundary.id == "semantic.boundary.linear.2"
    assert boundary.out_features == operations["ple_context_projection"]["output_shape"][-1]
    role, gate = grouped(["layer.0.ple_gate"], 40)
    assert role == "semantic_linear"
    assert gate.id == "semantic.layer.0.linear.40"
    role, projection = grouped(["layer.0.ple_projection"], 41)
    assert role == "semantic_linear"
    assert projection.id == "semantic.layer.0.linear.41"
    role, down = grouped(["layer.0.down_proj"], 42)
    assert role == "mlp_down"
    assert down.id == "layers.0.mlp.down_proj"
    role, gate_up = grouped(["layer.0.gate_proj", "layer.0.up_proj"], 43)
    assert role == "mlp_gate_up"
    assert gate_up.id == "layers.0.mlp.gate_up_proj"
    role, qkv = grouped(["layer.0.q_linear", "layer.0.k_linear", "layer.0.v_linear"], 44)
    assert role == "qkv_projection"
    assert qkv.id == "layers.0.self_attn.qkv_proj"


def test_pinned_gemma_stage_inventory_is_completely_semantic() -> None:
    config = json.loads(Path("crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json").read_text())
    plan = lower_model(config, batch=1, max_input_tokens=2, max_new_tokens=2)
    composition = MaskedLinearCpu(Model("org/model"))
    schedule = plan.runtime_schedule(composition)
    assert schedule.complete
    stages = scheduled_stage_specs(plan, composition)
    assert stages[0].role == "token_lookup"
    assert stages[-1].role == "lm_head"
    assert stages[0].metadata["ple_width"] == 8960
    assert len({stage.id for stage in stages}) == len(stages)
    expected_weights = {
        weight
        for operation in plan.prefill["operations"]
        if operation["operator"] in {"token_lookup", "linear", "output_head"}
        for weight in (operation["attributes"]["weight"],)
    }
    assert {weight for stage in stages for weight in stage.weight_keys} == expected_weights


def test_pinned_token_stage_matches_real_checkpoint_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    path = os.environ.get("PLLM_GEMMA4_E2B_PATH")
    if path is None:
        pytest.skip("set PLLM_GEMMA4_E2B_PATH for the pinned real-checkpoint check")
    step, operations, vocabulary = _pinned_token_group()
    stage = _token_lookup_stage(step, operations, vocabulary)
    store = SafeTensorStore(path)
    widths = (stage.out_features - stage.metadata["ple_width"], stage.metadata["ple_width"])
    for key, width in zip(stage.weight_keys, widths, strict=True):
        assert store.tensor_shape(key) == (vocabulary, width)
        assert store.tensor_dtype(key) == "BF16"
    source = Path("crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json")
    plan = lower_model(json.loads(source.read_text()), batch=1, max_input_tokens=2, max_new_tokens=2)
    assert preflight_semantic_checkpoint(plan, store) == 540
    original = store.tensor_shape
    monkeypatch.setattr(
        store, "tensor_shape", lambda key: (vocabulary, widths[1] + 1) if key == stage.weight_keys[1] else original(key)
    )
    with pytest.raises(SemanticTensorError, match="invalid shape"):
        preflight_semantic_checkpoint(plan, store)


def _tiny_token_output_group(dtypes):
    operations = {}
    outputs = []
    for name, width, offset, dtype in (
        ("main_embedding", 2, 0, dtypes[0]),
        ("ple_token_embedding", 3, 2, dtypes[1]),
    ):
        operations[name] = {
            "id": name, "operator": "token_lookup", "inputs": ["input.tokens"],
            "output_shape": [1, 2, width], "attributes": {
                "index_domain": "tokens", "weight": name + ".weight",
                "weight_shape": [8, width], "output_dtype": dtype,
            },
        }
        outputs.append({"operation_id": name, "stage_offset": offset, "stage_width": width})
    step = {
        "operation_ids": list(operations), "operators": ["token_lookup", "token_lookup"],
        "input_ids": ["input.tokens"], "layer": None,
        "weight_ids": [op["attributes"]["weight"] for op in operations.values()],
        "outputs": outputs,
    }
    return step, operations


@pytest.mark.parametrize("dtypes", [("bfloat16", "float32"), ("float32", "bfloat16")])
def test_fused_token_lookup_assigns_each_declared_numeric_output(dtypes, monkeypatch) -> None:
    # Stateless placement only: no executable graph or fabricated runtime owner.
    step, operations = _tiny_token_output_group(dtypes)
    stage = _token_lookup_stage(step, operations, 8)
    primary = np.asarray([[1.001953125, -1.005859375],
                          [1.005859375, -1.001953125]], dtype=np.float32)
    auxiliary = np.asarray([[2.00390625, 2.01171875, -2.00390625],
                            [-2.01171875, 0.5009765625, 0.5029296875]], dtype=np.float32)
    rounded_primary = np.asarray([[1.0, -1.0078125], [1.0078125, -1.0]], dtype=np.float32)
    rounded_auxiliary = np.asarray([[2.0, 2.015625, -2.0],
                                    [-2.015625, 0.5, 0.50390625]], dtype=np.float32)
    calls = []
    numeric_output = SemanticLocalOperators._numeric_output

    def count_conversion(operation, value):
        calls.append(operation["id"])
        return numeric_output(operation, value)

    monkeypatch.setattr(SemanticLocalOperators, "_numeric_output", staticmethod(count_conversion))
    actual = SemanticLocalOperators.materialize_token_lookup_outputs(
        step, operations, stage, primary, auxiliary)
    assert stage.in_features == 8 and stage.out_features == 5
    assert list(actual) == calls == ["main_embedding", "ple_token_embedding"]
    expected = (rounded_primary if dtypes[0] == "bfloat16" else primary,
                rounded_auxiliary if dtypes[1] == "bfloat16" else auxiliary)
    for name, width, value in zip(step["operation_ids"], (2, 3), expected, strict=True):
        assert actual[name].shape == (2, width) and actual[name].dtype == np.float32
        np.testing.assert_array_equal(actual[name], value)
    assert not np.array_equal(primary, rounded_primary)
    assert not np.array_equal(auxiliary, rounded_auxiliary)

    with pytest.raises(TransformerClientError, match="semantic token lookup has an invalid width"):
        SemanticLocalOperators.materialize_token_lookup_outputs(
            step, operations, stage, primary, auxiliary[:, :1])
    assert calls == ["main_embedding", "ple_token_embedding"]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_single_token_lookup_materializes_without_auxiliary(dtype):
    step, operations = _tiny_token_output_group((dtype, "float32"))
    step["operation_ids"] = step["operation_ids"][:1]
    step["operators"] = step["operators"][:1]
    step["weight_ids"] = step["weight_ids"][:1]
    step["outputs"] = step["outputs"][:1]
    operations = {"main_embedding": operations["main_embedding"]}
    stage = _token_lookup_stage(step, operations, 8)
    primary = np.asarray([[1.001953125, -1.005859375]], dtype=np.float32)
    actual = SemanticLocalOperators.materialize_token_lookup_outputs(step, operations, stage, primary, None)
    expected = np.asarray([[1.0, -1.0078125]], dtype=np.float32) if dtype == "bfloat16" else primary
    assert list(actual) == ["main_embedding"]
    assert actual["main_embedding"].shape == (1, 2) and actual["main_embedding"].dtype == np.float32
    np.testing.assert_array_equal(actual["main_embedding"], expected)


@pytest.mark.skipif(not os.environ.get("PLLM_GEMMA4_E2B_PATH"), reason="set PLLM_GEMMA4_E2B_PATH")
def test_pinned_real_projection_import_has_bounded_w8_error(tmp_path: Path) -> None:
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    root = Path(os.environ["PLLM_GEMMA4_E2B_PATH"])
    manifest = load_hf_directory(root, model_id="google/gemma-4-E2B-it@3e22461f")
    store = SafeTensorStore(root)
    stage = min(
        (
            candidate
            for candidate in manifest.stages
            if candidate.op == "linear"
            and len(candidate.weight_keys) == 1
            and not candidate.bias_keys
            and not candidate.transpose_weight
        ),
        key=lambda candidate: candidate.in_features * candidate.out_features,
    )
    manifest.metadata["stage_origin"] = "semantic_schedule_v1"
    engine = MaskedTransformerEngine(
        weight_bits=8, activation_bits=8, threads=1, compiled_cache_dir=tmp_path
    )
    loaded = engine._load_stage(store, stage, manifest)
    activation = np.zeros((1, stage.in_features), dtype=np.float32)
    activation[0, 0] = 1.0
    quantized = quantize_activation_per_row(activation, bits=stage.activation_bits)
    integer = loaded.compiled_weight.clear(quantized.values)
    result = dequantize_matmul(
        integer,
        quantized.scales,
        loaded.weight.scales,
        output_shape=(1, stage.out_features),
    )
    original = store.get(stage.weight_keys[0], dtype=np.float32)[:, 0]
    np.testing.assert_array_less(
        np.abs(result[0] - original), loaded.weight.scales * 0.51 + 0.000001
    )
    assert manifest.id == "google/gemma-4-E2B-it@3e22461f"
