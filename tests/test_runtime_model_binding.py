from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

import pllm
from pllm.modeling import ModelPlan
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import (
    CompiledRuntimeModel,
    RuntimeBindingError,
    compile_runtime_model,
)
from pllm.runtime.models import ModelManifest, transformer_stage_plan
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
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
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
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        fuse_qkv=False,
        fuse_gate_up=False,
        include_lm_head=False,
    )
    assert [stage.role for stage in unfused] == [
        "query_projection",
        "key_projection",
        "value_projection",
        "attention_output",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
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
    assert {(binding.role, binding.layer_index) for binding in compiled.stage_bindings} == {
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
        f"{phase}:layer.1.{name}_proj" for phase in ("prefill", "decode") for name in ("gate", "up")
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
    binding_schema = json.loads(
        Path("schemas/runtime-model-binding.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(binding_schema).validate(spec)
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


def test_compiled_binding_uses_same_path_for_qwen3(tmp_path: Path):
    engine, bundle, config = _bundle(
        tmp_path,
        model_id="tiny-qwen3-binding",
        model_type="qwen3",
        qk_norm=True,
        with_qkv_bias=False,
    )
    plan = _plan(config)

    assert plan.to_dict()["adapter"] == "pllm.qwen3.v1"
    assert plan.coverage("baseline.masked_linear_cpu").complete is True
    compiled = compile_runtime_model(plan, bundle)
    assert compiled.complete is True
    assert compiled.to_spec()["model_family"] == "qwen3"
    binding_schema = json.loads(
        Path("schemas/runtime-model-binding.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(binding_schema).validate(compiled.to_spec())

    remote = _remote(engine, bundle.model_id, bundle)
    logits = compiled.runtime(remote).forward_ids([0, 2])
    direct = MaskedTransformerClientRuntime(bundle, remote).forward_ids([0, 2])
    np.testing.assert_array_equal(logits, direct)


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

    bounded = dataclasses.replace(bundle, cfg={**bundle.cfg, "max_position_embeddings": 4})
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
        key: value
        for key, value in bundle.arrays.items()
        if key != "model.layers.0.input_layernorm.weight"
    }
    ambiguous_arrays["other.model.layers.0.input_layernorm.weight"] = np.ones(32, np.float32)
    ambiguous_arrays["extra.model.layers.0.input_layernorm.weight"] = np.ones(32, np.float32)
    cases.append(dataclasses.replace(bundle, arrays=ambiguous_arrays))

    suffix_arrays = {
        key: value
        for key, value in bundle.arrays.items()
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
    assert all(binding.stage_id != "embed_tokens" for binding in compiled.stage_bindings)

    duplicate = dataclasses.replace(bundle.stages["token_lookup"], id="embed_tokens")
    stages = {**bundle.stages, "embed_tokens": duplicate}
    tampered = dataclasses.replace(bundle, stages=stages)
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, tampered)


def _edited_plan(config: dict, edit) -> ModelPlan:
    document = _plan(config).to_dict()
    edit(document)
    return ModelPlan(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())


def _operation(document: dict, phase: str, operation_id: str) -> dict:
    return next(
        operation for operation in document[phase]["operations"] if operation["id"] == operation_id
    )


def test_completeness_scope_and_private_constructor(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    compiled = compile_runtime_model(plan, bundle)

    assert compiled.complete is True
    assert compiled.completeness_scope == "runtime_binding"
    assert compiled.to_spec()["completeness_scope"] == "runtime_binding"
    assert plan.coverage().complete is False
    assert plan.coverage("baseline.masked_linear_cpu").complete is True

    message = "CompiledRuntimeModel must be created by compile_runtime_model"
    with pytest.raises(RuntimeBindingError, match=message):
        CompiledRuntimeModel()
    with pytest.raises(TypeError):
        CompiledRuntimeModel(
            plan=plan,
            bundle=bundle,
            canonical=b"{}",
            digest="0" * 64,
            fingerprint="0" * 64,
            stages=(),
            local_operations=(),
            runtime_config_digest="0" * 64,
            tokenizer_digest="0" * 64,
        )


def test_portable_manifest_identity_across_paths(tmp_path: Path):
    bundles = []
    for name in ("first", "second"):
        root = create_tiny_llama_checkpoint(tmp_path / name / "model")
        manifest = load_hf_directory(root, model_id="tiny-portable")
        engine = MaskedTransformerEngine(threads=1)
        asyncio.run(engine.load(manifest))
        bundles.append(ClientBundle.unpack(engine.client_bundle("tiny-portable")))
    config = json.loads((tmp_path / "first" / "model" / "config.json").read_text())
    plan = _plan(config)

    assert bundles[0].manifest["source"] != bundles[1].manifest["source"]
    assert bundles[0].manifest["fingerprint"] != bundles[1].manifest["fingerprint"]

    first = compile_runtime_model(plan, bundles[0])
    second = compile_runtime_model(plan, bundles[1])
    assert first.bundle_fingerprint == second.bundle_fingerprint
    assert first.digest == second.digest
    assert first.canonical_bytes() == second.canonical_bytes()


def test_stale_manifest_fingerprint_rejected(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    compile_runtime_model(plan, bundle)

    moved = dataclasses.replace(bundle, manifest={**bundle.manifest, "source": "/elsewhere/model"})
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, moved)

    edited = dataclasses.replace(
        bundle,
        manifest={**bundle.manifest, "architecture": "tampered-architecture"},
    )
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, edited)

    bad_type = dataclasses.replace(bundle, manifest={**bundle.manifest, "fingerprint": 12})
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, bad_type)


def test_cfg_and_manifest_dimension_mismatch_rejected(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)

    for key, wrong in (
        ("hidden_size", 48),
        ("intermediate_size", 96),
        ("num_hidden_layers", 3),
        ("num_attention_heads", 8),
        ("num_key_value_heads", 4),
        ("head_dim", 4),
        ("vocab_size", 512),
        ("model_type", "qwen3"),
        ("max_position_embeddings", 16),
    ):
        tampered = dataclasses.replace(bundle, cfg={**bundle.cfg, key: wrong})
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, tampered)

    for key, wrong in (
        ("hidden_size", 48),
        ("num_hidden_layers", 3),
        ("context_length", 16),
    ):
        tampered = dataclasses.replace(bundle, manifest={**bundle.manifest, key: wrong})
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, tampered)


def test_stage_spec_drift_rejected(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    rows = bundle.manifest["stages"]
    by_id = {row["id"]: row for row in rows}
    qkv_row = by_id["layers.0.self_attn.qkv_proj"]

    def with_row(stage_id: str, **changes):
        replaced = [{**row, **changes} if row["id"] == stage_id else dict(row) for row in rows]
        return dataclasses.replace(bundle, manifest={**bundle.manifest, "stages": replaced})

    tampered = [
        with_row("layers.0.self_attn.qkv_proj", fused_from=["q_proj", "k_proj"]),
        with_row(
            "layers.0.self_attn.qkv_proj",
            fused_from=["q_proj", "k_proj", "v_proj"],
            role="attention_output",
        ),
        with_row("layers.0.self_attn.qkv_proj", layer_index=1),
        with_row("layers.0.self_attn.qkv_proj", op="embedding"),
        with_row("layers.0.self_attn.qkv_proj", weight_keys=list(qkv_row["weight_keys"][:2])),
        with_row("layers.0.mlp.gate_up_proj", fused_from=["gate_proj", "down_proj"]),
        with_row(
            "layers.0.mlp.down_proj",
            weight_keys=["extra.weight", *by_id["layers.0.mlp.down_proj"]["weight_keys"]],
        ),
        with_row("token_lookup", op="linear"),
        with_row("token_lookup", fused_from=["embed_tokens"]),
        with_row("token_lookup", layer_index=0),
        with_row("token_lookup", weight_keys=["model.other.weight"]),
        with_row("lm_head", op="linear"),
        with_row("lm_head", weight_keys=["model.other.weight"]),
    ]
    tampered.append(
        dataclasses.replace(
            bundle,
            manifest={**bundle.manifest, "stages": [dict(row) for row in rows[:-1]]},
        )
    )
    extra = [dict(row) for row in rows]
    extra.append({**dict(rows[-1]), "id": "extra_stage"})
    tampered.append(dataclasses.replace(bundle, manifest={**bundle.manifest, "stages": extra}))

    for candidate in tampered:
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, candidate)


def test_cross_layer_weight_labels_fail_with_fresh_manifest_fingerprint(tmp_path: Path):
    _, bundle, config = _bundle(tmp_path, num_hidden_layers=2)
    plan = _plan(config)
    rows = [dict(row) for row in bundle.manifest["stages"]]
    first = next(row for row in rows if row["id"] == "layers.0.self_attn.qkv_proj")
    second = next(row for row in rows if row["id"] == "layers.1.self_attn.qkv_proj")
    first["weight_keys"], second["weight_keys"] = (
        second["weight_keys"],
        first["weight_keys"],
    )
    document = {**bundle.manifest, "stages": rows}
    document.pop("fingerprint")
    refreshed = ModelManifest.from_dict(document).to_dict()

    with pytest.raises(RuntimeBindingError, match="semantic operation"):
        compile_runtime_model(plan, dataclasses.replace(bundle, manifest=refreshed))


def test_fused_weight_order_is_bound_to_semantic_outputs(tmp_path: Path):
    _, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    rows = [dict(row) for row in bundle.manifest["stages"]]
    qkv = next(row for row in rows if row["id"] == "layers.0.self_attn.qkv_proj")
    qkv["weight_keys"] = [
        qkv["weight_keys"][0],
        qkv["weight_keys"][2],
        qkv["weight_keys"][1],
    ]
    document = {**bundle.manifest, "stages": rows}
    document.pop("fingerprint")
    refreshed = ModelManifest.from_dict(document).to_dict()

    with pytest.raises(RuntimeBindingError, match="order"):
        compile_runtime_model(plan, dataclasses.replace(bundle, manifest=refreshed))

    rows = [dict(row) for row in bundle.manifest["stages"]]
    qkv = next(row for row in rows if row["id"] == "layers.0.self_attn.qkv_proj")
    qkv["fused_from"] = list(reversed(qkv["fused_from"]))
    document = {**bundle.manifest, "stages": rows}
    document.pop("fingerprint")
    refreshed = ModelManifest.from_dict(document).to_dict()
    with pytest.raises(RuntimeBindingError, match="fused order"):
        compile_runtime_model(plan, dataclasses.replace(bundle, manifest=refreshed))


def test_qwen3_runtime_knobs_are_bound_to_semantic_topology(tmp_path: Path):
    _, bundle, config = _bundle(
        tmp_path,
        model_id="tiny-qwen3-semantics",
        model_type="qwen3",
        qk_norm=True,
        with_qkv_bias=False,
    )
    plan = _plan(config)
    tampered = {**bundle.cfg, "qk_norm": False}
    config_digest = hashlib.sha256(
        json.dumps(
            tampered,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    metadata = {**bundle.manifest["metadata"], "runtime_config_digest": config_digest}
    document = {**bundle.manifest, "metadata": metadata}
    document.pop("fingerprint")
    refreshed = ModelManifest.from_dict(document).to_dict()

    with pytest.raises(RuntimeBindingError, match="normalization diverges"):
        compile_runtime_model(
            plan,
            dataclasses.replace(bundle, cfg=tampered, manifest=refreshed),
        )


def test_verified_profile_fails_closed_without_verifier_bound_executor(tmp_path: Path):
    root = create_tiny_llama_checkpoint(tmp_path / "verified-model")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = load_hf_directory(root, model_id="verified-model")
    engine = MaskedTransformerEngine(
        threads=1,
        verification_component="pllm/freivalds-verify/v1",
        verification_target_failure_bits=40,
    )
    asyncio.run(engine.load(manifest))
    verified = ClientBundle.unpack(engine.client_bundle("verified-model"))
    plan = _plan(config)

    with pytest.raises(RuntimeBindingError, match="verifier-bound remote executor"):
        compile_runtime_model(
            plan,
            verified,
            runtime_profile="research.verified_masked_linear_cpu",
        )

    _, baseline, _ = _bundle(tmp_path, model_id="baseline-model")
    forged = dataclasses.replace(
        baseline,
        privacy={
            **baseline.privacy,
            "runtime_profile": "research.verified_masked_linear_cpu",
            "verification_component": "pllm/freivalds-verify/v1",
            "verification_target_failure_bits": 40,
        },
    )
    with pytest.raises(RuntimeBindingError, match="verifier-bound remote executor"):
        compile_runtime_model(
            plan,
            forged,
            runtime_profile="research.verified_masked_linear_cpu",
        )


def test_topology_geometry_and_weight_ownership(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    compiled = compile_runtime_model(plan, bundle)
    assert compiled.complete is True

    def rename_weight(document):
        for phase in ("prefill", "decode"):
            _operation(document, phase, "layer.0.q_linear")["attributes"]["weight"] = (
                "renamed.projection.weight"
            )

    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(_edited_plan(config, rename_weight), bundle)

    def skip_rotary(document):
        _operation(document, "prefill", "layer.0.attention_scores")["inputs"][0] = "layer.0.q_heads"

    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(_edited_plan(config, skip_rotary), bundle)

    def wrong_key_state(document):
        _operation(document, "prefill", "layer.0.key_append")["state_kind"] = "value"

    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(_edited_plan(config, wrong_key_state), bundle)


def test_phase_accounting_and_parity_rejected(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)

    def decode_output_drift(document):
        operation = _operation(document, "decode", "layer.0.q_linear")
        shape = list(operation["output_shape"])
        shape[-1] += 1
        operation["output_shape"] = shape

    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(_edited_plan(config, decode_output_drift), bundle)

    def decode_input_drift(document):
        _operation(document, "decode", "layer.0.q_linear")["inputs"][0] = "layer.0.post_norm"

    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(_edited_plan(config, decode_input_drift), bundle)

    def decode_weight_swap(document):
        _operation(document, "decode", "layer.0.q_linear")["attributes"]["weight"] = (
            "model.layers.0.self_attn.o_proj.weight"
        )

    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(_edited_plan(config, decode_weight_swap), bundle)


def test_modulus_policy_rejected(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    metadata = dict(bundle.manifest["metadata"])
    stage = bundle.stages["layers.0.self_attn.qkv_proj"]

    def with_metadata(**changes):
        return dataclasses.replace(
            bundle,
            manifest={
                **bundle.manifest,
                "metadata": {**metadata, **changes},
            },
        )

    dropped = {key: value for key, value in metadata.items() if key != "stage_specific_moduli"}
    cases = [
        dataclasses.replace(bundle, manifest={**bundle.manifest, "metadata": dropped}),
        with_metadata(stage_specific_moduli="yes"),
        with_metadata(stage_specific_moduli=None),
        with_metadata(stage_specific_moduli=False),
        with_metadata(plain_moduli="65537"),
        with_metadata(plain_moduli=[65537.0]),
        with_metadata(plain_moduli=[]),
        _replace_stage(bundle, "layers.0.self_attn.qkv_proj", modulus=65539),
        _replace_stage(bundle, "layers.0.self_attn.qkv_proj", modulus=131071),
    ]
    for candidate in cases:
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, candidate)


def test_boundary_client_weight_tamper_rejected(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    token = bundle.stages["token_lookup"]
    head = bundle.stages["lm_head"]

    cases = [
        _replace_stage(bundle, "token_lookup", client_weight=None),
        _replace_stage(bundle, "token_lookup", client_weight_scales=None),
        _replace_stage(
            bundle,
            "token_lookup",
            client_weight=np.zeros_like(token.client_weight),
        ),
        _replace_stage(
            bundle,
            "token_lookup",
            client_weight_scales=np.zeros_like(token.client_weight_scales),
        ),
        _replace_stage(
            bundle,
            "token_lookup",
            client_weight_scales=token.client_weight_scales * 2,
        ),
        _replace_stage(bundle, "token_lookup", client_weight_layout="linear"),
        _replace_stage(bundle, "token_lookup", client_weight_layout="transposed_embedding"),
        _replace_stage(
            bundle,
            "token_lookup",
            client_aux_weight=np.zeros((4, token.in_features), np.int8),
        ),
        _replace_stage(
            bundle,
            "token_lookup",
            client_aux_weight=np.zeros((4, token.in_features), np.int8),
            client_aux_scales=np.full(4, np.nan, np.float32),
        ),
        _replace_stage(bundle, "lm_head", client_weight=None),
        _replace_stage(bundle, "lm_head", client_weight_scales=None),
        _replace_stage(
            bundle,
            "lm_head",
            client_weight=np.zeros_like(head.client_weight),
        ),
        _replace_stage(bundle, "lm_head", client_weight_layout="embedding"),
        _replace_stage(
            bundle,
            "layers.0.self_attn.qkv_proj",
            client_weight=np.zeros((4, 4), np.int8),
            client_weight_scales=np.ones(4, np.float32),
        ),
    ]
    for candidate in cases:
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, candidate)


def test_boundary_client_mutation_rejected_at_runtime(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    remote = _remote(engine, bundle.model_id, bundle)
    token = bundle.stages["token_lookup"]

    compiled = compile_runtime_model(plan, bundle)
    saved = token.client_weight[0, 0]
    token.client_weight[0, 0] = np.int8(int(saved) + 1)
    with pytest.raises(RuntimeBindingError):
        compiled.runtime(remote)
    token.client_weight[0, 0] = saved

    compiled = compile_runtime_model(plan, bundle)
    saved_scale = token.client_weight_scales[0]
    token.client_weight_scales[0] = np.float32(saved_scale * 2)
    with pytest.raises(RuntimeBindingError):
        compiled.runtime(remote)
    token.client_weight_scales[0] = saved_scale

    compiled = compile_runtime_model(plan, bundle)
    original_stage = bundle.stages["token_lookup"]
    bundle.stages["token_lookup"] = dataclasses.replace(
        original_stage,
        client_aux_weight=np.zeros((2, original_stage.in_features), np.int8),
        client_aux_scales=np.ones(2, np.float32),
    )
    with pytest.raises(RuntimeBindingError):
        compiled.runtime(remote)
    bundle.stages["token_lookup"] = original_stage

    assert compile_runtime_model(plan, bundle).runtime(remote).bundle is bundle


def test_config_reconstruction_binds_native_fields(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    compiled = compile_runtime_model(plan, bundle)
    assert compiled.complete is True
    without_explicit_head_dim = dataclasses.replace(
        bundle,
        cfg={key: value for key, value in bundle.cfg.items() if key != "head_dim"},
    )
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, without_explicit_head_dim)

    for key, wrong in (
        ("rms_norm_eps", 1e-4),
        ("rope_theta", 500000.0),
        ("hidden_act", "gelu"),
        ("tie_word_embeddings", False),
    ):
        tampered = dataclasses.replace(bundle, cfg={**bundle.cfg, key: wrong})
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, tampered)

    for key, wrong in (
        ("block_style", "gemma4"),
        ("model_family", "gemma"),
        ("norm_offset", 1.0),
        ("embedding_multiplier", 2.0),
        ("qk_norm", True),
        ("v_norm", True),
        ("layer_types", ["full_attention", "sliding_attention"]),
        ("sliding_window", 2),
        ("num_kv_shared_layers", 1),
        ("attention_k_eq_v", True),
        ("hidden_size_per_layer_input", 16),
        ("output_multiplier", 2.0),
        ("final_logit_softcapping", 30.0),
        ("attention_scaling", 0.5),
        ("rope_parameters", {"rope_theta": 500000.0}),
        ("per_layer_config", {"0": {}}),
        ("hidden_activation", "gelu"),
        ("token_lookup_batch", "many"),
    ):
        tampered = dataclasses.replace(bundle, cfg={**bundle.cfg, key: wrong})
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, tampered)


def test_runtime_config_and_tokenizer_digests(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    compiled = compile_runtime_model(plan, bundle)
    spec = compiled.to_spec()
    assert len(compiled.runtime_config_digest) == 64
    assert len(compiled.runtime_schedule_digest) == 64
    assert len(compiled.tokenizer_digest) == 64
    assert spec["runtime_config_digest"] == compiled.runtime_config_digest
    assert spec["runtime_schedule_digest"] == compiled.runtime_schedule_digest
    assert compiled.runtime_schedule_digest == plan.runtime_schedule().digest
    assert spec["tokenizer_digest"] == compiled.tokenizer_digest
    serialized = json.dumps(spec)
    assert "chat_template" not in serialized
    assert "tokenizer" not in serialized.replace("tokenizer_digest", "")

    for key, wrong in (
        ("vocab_size", 512),
        ("bos_token_id", 9),
        ("eos_token_id", 9),
        ("type", "alphabet"),
    ):
        descriptor = {**bundle.tokenizer_descriptor, key: wrong}
        tampered = dataclasses.replace(bundle, tokenizer_descriptor=descriptor)
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, tampered)

    remote = _remote(engine, bundle.model_id, bundle)
    compiled = compile_runtime_model(plan, bundle)
    bundle.cfg["rope_theta"] = 500000.0
    with pytest.raises(RuntimeBindingError):
        compiled.runtime(remote)
    bundle.cfg["rope_theta"] = 10000.0

    compiled = compile_runtime_model(plan, bundle)
    saved = bundle.tokenizer_descriptor["chat_template"]
    bundle.tokenizer_descriptor["chat_template"] = "tampered {{"
    with pytest.raises(RuntimeBindingError):
        compiled.runtime(remote)
    bundle.tokenizer_descriptor["chat_template"] = saved

    assert compile_runtime_model(plan, bundle).runtime(remote).bundle is bundle


def test_lm_head_weight_keys_and_aux_rejected(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    rows = bundle.manifest["stages"]

    def with_row(stage_id: str, **changes):
        replaced = [{**row, **changes} if row["id"] == stage_id else dict(row) for row in rows]
        return dataclasses.replace(bundle, manifest={**bundle.manifest, "stages": replaced})

    head_row = next(row for row in rows if row["id"] == "lm_head")
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(
            plan,
            with_row("lm_head", weight_keys=[*head_row["weight_keys"], "model.extra.weight"]),
        )
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, with_row("lm_head", weight_keys=[]))
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, with_row("lm_head", weight_keys=["model.other.weight"]))

    head = bundle.stages["lm_head"]
    auxed = _replace_stage(
        bundle,
        "lm_head",
        client_aux_weight=np.zeros((2, head.in_features), np.int8),
        client_aux_scales=np.ones(2, np.float32),
    )
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, auxed)


def test_rope_scaling_and_semantic_bias_contract(tmp_path: Path):
    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    compiled = compile_runtime_model(plan, bundle)

    for key, wrong in (
        ("rope_scaling", {"type": "linear", "factor": 2.0}),
        ("rope_scaling", {"rope_type": "yarn"}),
        ("rope_scaling", {"type": "default", "factor": 2.0}),
        ("rope_scaling", {"rope_type": "default", "factor": 2.0}),
        ("rope_scaling", {"type": "default", "rope_type": "linear"}),
        ("rope_scaling", "linear"),
        ("use_sliding_window", True),
        ("attention_bias", False),
    ):
        tampered = dataclasses.replace(bundle, cfg={**bundle.cfg, key: wrong})
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, tampered)

    for accepted in ({}, {"type": "default"}, {"rope_type": "default"}):
        accepted_bundle = dataclasses.replace(bundle, cfg={**bundle.cfg, "rope_scaling": accepted})
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, accepted_bundle)
    assert (
        compile_runtime_model(
            plan,
            dataclasses.replace(bundle, cfg={**bundle.cfg, "rope_scaling": None}),
        ).digest
        == compiled.digest
    )
    disabled_window = dataclasses.replace(
        bundle,
        cfg={**bundle.cfg, "sliding_window": 32768, "use_sliding_window": False},
    )
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, disabled_window)

    _, unbiased_bundle, _ = _bundle(tmp_path / "unbiased", with_qkv_bias=False)
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, unbiased_bundle)

    qkv_id = "layers.0.self_attn.qkv_proj"
    o_id = "layers.0.self_attn.o_proj"
    down_id = "layers.0.mlp.down_proj"
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, _replace_stage(bundle, qkv_id, bias=None))
    for stage_id in (o_id, down_id, "lm_head"):
        stage = bundle.stages[stage_id]
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(
                plan,
                _replace_stage(
                    bundle,
                    stage_id,
                    bias=np.zeros(stage.out_features, np.float32),
                ),
            )

    def drop_q_bias(document):
        for phase in ("prefill", "decode"):
            _operation(document, phase, "layer.0.q_linear")["attributes"]["bias"] = None

    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(_edited_plan(config, drop_q_bias), bundle)

    def add_o_bias(document):
        for phase in ("prefill", "decode"):
            _operation(document, phase, "layer.0.o_proj")["attributes"]["bias"] = (
                "model.layers.0.self_attn.o_proj.bias"
            )

    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(_edited_plan(config, add_o_bias), bundle)

    remote = _remote(engine, bundle.model_id, bundle)
    compiled = compile_runtime_model(plan, bundle)
    qkv_stage = bundle.stages[qkv_id]
    qkv_stage.bias[0] = np.float32(qkv_stage.bias[0] + 1.0)
    with pytest.raises(RuntimeBindingError):
        compiled.runtime(remote)
