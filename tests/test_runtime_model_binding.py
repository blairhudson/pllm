from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import pllm
from pllm.modeling import ModelPlan
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import (
    RuntimeBindingError,
    compile_runtime_model,
)
from pllm.runtime.models import transformer_stage_plan
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, MaskedTransformerClientRuntime
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def _bundle(path: Path, *, model_id: str = "tiny-binding", **checkpoint):
    root = create_tiny_llama_checkpoint(path / "model", **checkpoint)
    manifest = load_hf_directory(root, model_id=model_id)
    engine = MaskedTransformerEngine(threads=1)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    config = json.loads((root / "config.json").read_text())
    return engine, bundle, config


def _plan(config: dict, **workload) -> ModelPlan:
    workload = {"batch": 1, "max_input_tokens": 8, "max_new_tokens": 4, **workload}
    return pllm.lower_model(config, **workload)


def _remote(engine: MaskedTransformerEngine, model_id: str, bundle: ClientBundle):
    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = bundle.stages[stage_id]
        runtime = engine.models[model_id].stages[stage_id]
        result = np.asarray(activation, dtype=np.float32) @ runtime.weight.dequantize().T
        if stage.bias is not None:
            result = result + stage.bias
        return np.ascontiguousarray(result, dtype=np.float32)

    return remote


def test_stage_plan_roles_and_layers():
    fused = transformer_stage_plan(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64,
        include_embedding=True,
    )
    assert (fused[0].role, fused[0].layer_index) == ("token_lookup", None)
    by_id = {stage.id: stage for stage in fused}
    for layer in range(2):
        expected = {
            f"layers.{layer}.self_attn.qkv_proj": "qkv_projection",
            f"layers.{layer}.self_attn.o_proj": "attention_output",
            f"layers.{layer}.mlp.gate_up_proj": "mlp_gate_up",
            f"layers.{layer}.mlp.down_proj": "mlp_down",
        }
        for stage_id, role in expected.items():
            assert by_id[stage_id].role == role
            assert by_id[stage_id].layer_index == layer
    assert (by_id["lm_head"].role, by_id["lm_head"].layer_index) == ("lm_head", None)

    unfused = transformer_stage_plan(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64,
        fuse_qkv=False, fuse_gate_up=False, include_lm_head=False,
    )
    assert [stage.role for stage in unfused] == [
        "query_projection", "key_projection", "value_projection",
        "attention_output", "mlp_gate", "mlp_up", "mlp_down",
    ]
    assert all(stage.layer_index == 0 for stage in unfused)


def test_compiled_binding_is_canonical(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    compiled = compile_runtime_model(plan, bundle)

    assert compiled.complete is True
    assert compiled.model_plan_digest == plan.digest
    assert len(compiled.digest) == 64
    assert len(compiled.bundle_fingerprint) == 64
    assert len(compiled.stage_bindings) == 10
    assert {
        (binding.role, binding.layer_index) for binding in compiled.stage_bindings
    } == {
        ("token_lookup", None),
        ("lm_head", None),
        *{
            (role, layer)
            for layer in range(2)
            for role in ("qkv_projection", "attention_output", "mlp_gate_up", "mlp_down")
        },
    }

    bindings = {binding.stage_id: binding for binding in compiled.stage_bindings}
    qkv = bindings["layers.0.self_attn.qkv_proj"]
    assert set(qkv.semantic_operations) == {
        f"{phase}:layer.0.{name}_linear"
        for phase in ("prefill", "decode")
        for name in ("q", "k", "v")
    }
    gate_up = bindings["layers.1.mlp.gate_up_proj"]
    assert set(gate_up.semantic_operations) == {
        f"{phase}:layer.1.{name}_proj"
        for phase in ("prefill", "decode")
        for name in ("gate", "up")
    }

    covered = set(compiled.local_operations)
    for binding in compiled.stage_bindings:
        assert not covered & set(binding.semantic_operations)
        covered |= set(binding.semantic_operations)
    document = plan.to_dict()
    assert covered == {
        f"{phase}:{operation['id']}"
        for phase in ("prefill", "decode")
        for operation in document[phase]["operations"]
    }
    assert compiled.to_spec()["operation_count"] == len(covered)

    spec = compiled.to_spec()
    assert spec["schema"] == "pllm.runtime_model_binding.v1"
    assert len(spec["local_tensors"]) == 5
    assert [row["weight_id"] for row in spec["local_tensors"]] == sorted(
        row["weight_id"] for row in spec["local_tensors"]
    )

    def assert_jsonable(value):
        if isinstance(value, dict):
            assert "source" not in value
            assert "created_at" not in value
            assert "weight_scales" not in value
            for item in value.values():
                assert_jsonable(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                assert_jsonable(item)
        else:
            assert not isinstance(value, (bytes, bytearray, np.ndarray))

    assert_jsonable(spec)
    json.dumps(spec)

    repeated = compile_runtime_model(plan, bundle)
    assert repeated.digest == compiled.digest
    assert repeated.canonical_bytes() == compiled.canonical_bytes()
    assert repeated.bundle_fingerprint == compiled.bundle_fingerprint


def test_runtime_shares_bundle_and_executes(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    compiled = compile_runtime_model(plan, bundle)
    remote = _remote(engine, bundle.model_id, bundle)
    runtime = compiled.runtime(remote)
    assert isinstance(runtime, MaskedTransformerClientRuntime)
    assert runtime.bundle is bundle

    logits = runtime.forward_ids([0, 2])
    assert logits.shape == (2, int(bundle.cfg["vocab_size"]))
    assert np.all(np.isfinite(logits))

    direct = MaskedTransformerClientRuntime(bundle, remote)
    np.testing.assert_array_equal(logits, direct.forward_ids([0, 2]))


def test_repeated_load_yields_identical_binding(tmp_path: Path):
    root = create_tiny_llama_checkpoint(tmp_path / "model")
    config = json.loads((root / "config.json").read_text())
    plan = _plan(config)
    bundles = []
    for _ in range(2):
        manifest = load_hf_directory(root, model_id="tiny-repeat")
        engine = MaskedTransformerEngine(threads=1)
        asyncio.run(engine.load(manifest))
        bundles.append(ClientBundle.unpack(engine.client_bundle("tiny-repeat")))
    first = compile_runtime_model(plan, bundles[0])
    second = compile_runtime_model(plan, bundles[1])
    assert bundles[0].manifest["created_at"] != bundles[1].manifest["created_at"]
    assert first.bundle_fingerprint == second.bundle_fingerprint
    assert first.digest == second.digest
    assert first.canonical_bytes() == second.canonical_bytes()


def test_plan_mismatch_rejected(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)

    mismatched = [
        {**config, "hidden_size": 48},
        {**config, "num_hidden_layers": 3},
        {**config, "vocab_size": 512},
    ]
    for value in mismatched:
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(_plan(value), bundle)

    bounded = dataclasses.replace(
        bundle, cfg={**bundle.cfg, "max_position_embeddings": 4}
    )
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(_plan(config), bounded)

    qwen3 = {
        **config,
        "model_type": "qwen3",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "use_cache": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "rope_scaling": None,
        "max_window_layers": config["num_hidden_layers"],
        "layer_types": ["full_attention"] * config["num_hidden_layers"],
    }
    wrong_family = _plan(qwen3)
    assert wrong_family.to_dict()["model_family"] == "qwen3"
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(wrong_family, bundle)

    transformed = _plan(config).to_dict()
    transformed["transformations"] = [{"operator": "unknown"}]
    transformed_plan = ModelPlan(
        json.dumps(transformed, sort_keys=True, separators=(",", ":")).encode()
    )
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(transformed_plan, bundle)

    wrong_adapter = _plan(config).to_dict()
    wrong_adapter["adapter"] = "pllm.other.v1"
    wrong_adapter_plan = ModelPlan(
        json.dumps(wrong_adapter, sort_keys=True, separators=(",", ":")).encode()
    )
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(wrong_adapter_plan, bundle)

    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan=_plan(config), bundle=object())
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan=object(), bundle=bundle)


def _replace_stage(bundle: ClientBundle, stage_id: str, **changes) -> ClientBundle:
    stage = bundle.stages[stage_id]
    stages = {**bundle.stages, stage_id: dataclasses.replace(stage, **changes)}
    return dataclasses.replace(bundle, stages=stages)


def test_bundle_tamper_rejected(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    stage_id = "layers.0.self_attn.qkv_proj"
    stage = bundle.stages[stage_id]

    cases = [
        _replace_stage(bundle, stage_id, role="other"),
        _replace_stage(bundle, stage_id, layer_index=7),
        _replace_stage(bundle, stage_id, out_features=stage.out_features + 1),
        _replace_stage(bundle, stage_id, weight_bits=8),
        _replace_stage(bundle, stage_id, ring="u32"),
        _replace_stage(bundle, stage_id, weight_scales=np.zeros_like(stage.weight_scales)),
        _replace_stage(bundle, stage_id, bias=stage.bias + 1.0),
        _replace_stage(bundle, stage_id, weight_digest="0" * 64),
        _replace_stage(bundle, stage_id, wire_bits=32),
        _replace_stage(bundle, "layers.0.mlp.down_proj", weight_digest="f" * 64),
        dataclasses.replace(bundle, privacy={**bundle.privacy, "mode": "proprietary"}),
        dataclasses.replace(bundle, privacy={**bundle.privacy, "protocol": "other"}),
        dataclasses.replace(bundle, privacy={**bundle.privacy, "preprocessed": False}),
        dataclasses.replace(
            bundle, privacy={**bundle.privacy, "client_intermediate_activations": False}
        ),
        dataclasses.replace(bundle, privacy={**bundle.privacy, "body_fingerprint": ""}),
        dataclasses.replace(bundle, privacy={**bundle.privacy, "stage_commitment": "0" * 64}),
        dataclasses.replace(bundle, privacy={**bundle.privacy, "weight_bits": 8}),
        dataclasses.replace(bundle, model_id="other-model"),
        dataclasses.replace(
            bundle,
            arrays={k: v for k, v in bundle.arrays.items() if k != "model.norm.weight"},
        ),
    ]

    ambiguous_arrays = {
        key: value for key, value in bundle.arrays.items()
        if key != "model.layers.0.input_layernorm.weight"
    }
    ambiguous_arrays["other.model.layers.0.input_layernorm.weight"] = np.ones(32, np.float32)
    ambiguous_arrays["extra.model.layers.0.input_layernorm.weight"] = np.ones(32, np.float32)
    cases.append(dataclasses.replace(bundle, arrays=ambiguous_arrays))

    suffix_arrays = {
        key: value for key, value in bundle.arrays.items()
        if key != "model.layers.0.input_layernorm.weight"
    }
    suffix_arrays["other.model.layers.0.input_layernorm.weight"] = np.ones(32, np.float32)
    resolved = dataclasses.replace(bundle, arrays=suffix_arrays)
    resolved_spec = compile_runtime_model(plan, resolved).to_spec()
    assert any(
        row["key"] == "other.model.layers.0.input_layernorm.weight"
        for row in resolved_spec["local_tensors"]
    )

    bad_norm = dict(bundle.arrays)
    bad_norm["model.layers.0.input_layernorm.weight"] = np.ones(16, np.float32)
    cases.append(dataclasses.replace(bundle, arrays=bad_norm))

    nan_norm = dict(bundle.arrays)
    nan_norm["model.layers.0.input_layernorm.weight"] = np.full(32, np.nan, np.float32)
    cases.append(dataclasses.replace(bundle, arrays=nan_norm))

    for index, tampered in enumerate(cases):
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, tampered)


def test_runtime_rejects_post_bind_mutation(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    remote = _remote(engine, bundle.model_id, bundle)

    compiled = compile_runtime_model(plan, bundle)
    stage_id = "layers.0.self_attn.qkv_proj"
    original = bundle.stages[stage_id]
    bundle.stages[stage_id] = dataclasses.replace(
        original, weight_scales=original.weight_scales * 2
    )
    with pytest.raises(RuntimeBindingError):
        compiled.runtime(remote)
    bundle.stages[stage_id] = original

    compiled = compile_runtime_model(plan, bundle)
    key = "model.layers.0.input_layernorm.weight"
    array = bundle.arrays[key]
    saved = array.copy()
    array[0] += 1.0
    with pytest.raises(RuntimeBindingError):
        compiled.runtime(remote)
    bundle.arrays[key] = saved

    compiled = compile_runtime_model(plan, bundle)
    bundle.privacy["mode"] = "proprietary"
    with pytest.raises(RuntimeBindingError):
        compiled.runtime(remote)
    bundle.privacy["mode"] = "public"

    runtime = compile_runtime_model(plan, bundle).runtime(remote)
    assert runtime.bundle is bundle


def test_embed_tokens_alias_excluded_and_cannot_duplicate(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    assert "embed_tokens" in bundle.stages
    assert bundle.stages["embed_tokens"].id == "token_lookup"

    compiled = compile_runtime_model(plan, bundle)
    assert all(
        binding.stage_id != "embed_tokens" for binding in compiled.stage_bindings
    )

    duplicate = dataclasses.replace(
        bundle.stages["token_lookup"], id="embed_tokens"
    )
    stages = {**bundle.stages, "embed_tokens": duplicate}
    tampered = dataclasses.replace(bundle, stages=stages)
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, tampered)
