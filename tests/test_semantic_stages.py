"""Ordered token boundary stages follow semantic artifacts, not model names."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.models import StageSpec
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.semantic_stages import (
    _linear_stage,
    _token_lookup_stage,
    scheduled_stage_specs,
    semantic_stage_role,
)
from pllm.runtime.semantic_tensors import SemanticTensorError, preflight_semantic_checkpoint
from pllm.runtime.transformer_client import TransformerClientError


def _pinned_token_group() -> tuple[dict, dict, int]:
    source = Path("crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json")
    config = json.loads(source.read_text())
    plan = lower_model(config, batch=1, max_input_tokens=2, max_new_tokens=2)
    operations = {operation["id"]: operation for operation in plan.prefill["operations"]}
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


@pytest.mark.parametrize("selected", ["main_embedding", "ple_token_embedding"])
def test_fused_token_lookup_assigns_each_declared_numeric_output(selected: str) -> None:
    step, operations, vocabulary = _pinned_token_group()
    stage = _token_lookup_stage(step, operations, vocabulary)
    decoder = object.__new__(SemanticDecoderRuntime)
    decoder.position = 0
    decoder.caches = []
    decoder._text_only_tokens = frozenset()
    decoder._hybrid_states = {}
    decoder._hybrid_contracts = {}
    decoder.bundle = SimpleNamespace(stages={"token_lookup": stage})
    primary_width = stage.out_features - stage.metadata["ple_width"]
    decoder._token_lookup = lambda _ids: (
        np.full((1, primary_width), 1.0, dtype=np.float32),
        np.full((1, stage.metadata["ple_width"]), 2.0, dtype=np.float32),
    )
    graph = {
        "query_sequence": 1,
        "maximum_key_sequence": 2,
        "state_inputs": [],
        "state_outputs": [],
        "operations": [
            operations["main_embedding"],
            operations["ple_token_embedding"],
            {"id": "selection", "operator": "greedy_token_selection", "inputs": [selected], "attributes": {}},
        ],
    }
    schedule = {
        "steps": [
            {"executor": "remote_stage", **step},
            {"executor": "client_local", "operation_ids": ["selection"], "input_ids": [selected]},
        ]
    }
    decoder._graphs = {"prefill": graph, "decode": graph}
    decoder._schedule = {"prefill": schedule, "decode": schedule}
    decoder._stages = {
        f"{phase}:{op_id}": "token_lookup"
        for phase in ("prefill", "decode")
        for op_id in step["operation_ids"]
    }
    actual = decoder._forward(np.asarray([3], dtype=np.int64))
    expected = 1.0 if selected == "main_embedding" else 2.0
    width = primary_width if selected == "main_embedding" else stage.metadata["ple_width"]
    np.testing.assert_array_equal(actual, np.full((1, width), expected, dtype=np.float32))
    assert decoder.position == 1

    decoder.position = 0
    decoder._token_lookup = lambda _ids: (
        np.ones((1, primary_width), dtype=np.float32), np.ones((1, 1), dtype=np.float32)
    )
    with pytest.raises(TransformerClientError, match="invalid width"):
        decoder._forward(np.asarray([3], dtype=np.int64))
    assert decoder.position == 0


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
