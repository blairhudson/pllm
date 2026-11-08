"""Exact semantic sources must survive the provider/client bundle boundary."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from pllm import lower_model
from pllm.runtime.model_binding import RuntimeBindingError, _runtime_config, _validate_runtime_semantics
from pllm.runtime.semantic_source import SemanticSourceError, semantic_source_config


_FIXTURE = Path("crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json")


def _flattened() -> tuple[dict, dict]:
    source = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return {**source["text_config"], "semantic_source_config": source}, source


def test_nested_semantic_source_reconstructs_exact_adapter_plan() -> None:
    flattened, source = _flattened()
    assert semantic_source_config(flattened) is source
    plan = lower_model(semantic_source_config(flattened), batch=1, max_input_tokens=2, max_new_tokens=2)
    assert plan.to_dict()["adapter"] == "pllm.gemma4_e2b_text.v1"
    runtime = _runtime_config(flattened, nested_source=True)
    assert runtime["sliding_window"] == 512
    assert "sliding_attention" in runtime["layer_types"]
    graph = plan.to_dict()
    phases = {
        phase: (graph[phase], {op["id"]: op for op in graph[phase]["operations"]})
        for phase in ("prefill", "decode")
    }
    _validate_runtime_semantics(runtime, phases, nested_source=True)


@pytest.mark.parametrize("key", ["hidden_size", "head_dim", "layer_types", "rope_parameters"])
def test_flattened_semantic_config_rejects_drift(key: str) -> None:
    flattened, _ = _flattened()
    flattened[key] = "out-of-band override"
    with pytest.raises(SemanticSourceError, match=key):
        semantic_source_config(flattened)


def test_semantic_source_rejects_missing_outer_or_unbound_rotary_width() -> None:
    flattened, _ = _flattened()
    flattened["semantic_source_config"] = {"text_config": {"model_type": "gemma4_text"}}
    with pytest.raises(SemanticSourceError, match="malformed"):
        semantic_source_config(flattened)

    flattened, _ = _flattened()
    del flattened["layer_types"]
    with pytest.raises(SemanticSourceError, match="layer_types"):
        semantic_source_config(flattened)

    flattened, source = _flattened()
    plan = lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=2).to_dict()
    phases = {
        phase: (plan[phase], {op["id"]: op for op in plan[phase]["operations"]})
        for phase in ("prefill", "decode")
    }
    rotary = next(
        op for op in phases["prefill"][1].values() if op["operator"] == "rotary_embedding"
    )
    rotary["attributes"]["head_dim"] += 1
    with pytest.raises(RuntimeBindingError, match="rotary width"):
        _validate_runtime_semantics(_runtime_config(flattened, nested_source=True), phases, nested_source=True)


def test_flat_semantic_source_preserves_existing_decoder_identity() -> None:
    config = {"model_type": "qwen2", "hidden_size": 32}
    assert semantic_source_config(config) is config


def _flat_source() -> dict:
    return {
        "model_type": "qwen2", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "vocab_size": 258, "max_position_embeddings": 256, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": False,
        "bos_token_id": 0, "eos_token_id": 1,
    }


@pytest.mark.parametrize("nested", [False, True])
def test_semantic_transport_roundtrip_keeps_native_source_digests(nested: bool) -> None:
    source = _flattened()[1] if nested else _flat_source()
    view = source["text_config"] if nested else source
    transport = copy.deepcopy(view)
    transport.update({
        "semantic_source_config": copy.deepcopy(source),
        "attention_scaling": None, "prime_modulus": 65537,
    })
    transported = json.loads(json.dumps(transport))
    restored = semantic_source_config(transported)
    assert restored == source
    assert restored is not transported
    assert "attention_scaling" not in restored
    assert "prime_modulus" not in restored
    original = lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=2)
    reconstructed = lower_model(restored, batch=1, max_input_tokens=2, max_new_tokens=2)
    assert reconstructed.digest == original.digest
    assert reconstructed.to_dict()["config_digest"] == original.to_dict()["config_digest"]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_source_transport_requires_every_original_field(nested: bool, missing: bool) -> None:
    source = _flattened()[1] if nested else _flat_source()
    view = source["text_config"] if nested else source
    transport = {**copy.deepcopy(view), "semantic_source_config": source}
    if missing:
        del transport["hidden_size"]
    else:
        transport["hidden_size"] += 1
    with pytest.raises(SemanticSourceError, match="hidden_size"):
        semantic_source_config(transport)


@pytest.mark.parametrize("source", [None, [], {}, {"model_type": "qwen2", "text_config": None}])
def test_present_source_envelope_must_be_well_formed(source) -> None:
    with pytest.raises(SemanticSourceError, match="malformed"):
        semantic_source_config({"semantic_source_config": source})


@pytest.mark.parametrize("field,value", [
    ("attention_scaling", 0.5),
    ("unsupported_source_extension", {"enabled": True}),
])
def test_preserved_source_does_not_filter_unsupported_extensions(field: str, value) -> None:
    from absolute_position_decoder_oracle import tiny_checkpoint

    source, _ = tiny_checkpoint()
    # Establish native admission before testing strict source-field refusal.
    lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=2)
    source[field] = value
    transport = {**source, "semantic_source_config": copy.deepcopy(source)}
    restored = semantic_source_config(transport)
    assert restored == source
    assert restored[field] == value
    for native_source in (source, restored):
        with pytest.raises(ValueError, match=rf"unsupported source field {field}\b"):
            lower_model(native_source, batch=1, max_input_tokens=2, max_new_tokens=2)


@pytest.fixture
def flat_bundle(tmp_path: Path):
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    root = create_tiny_llama_checkpoint(tmp_path / "model")
    source = json.loads((root / "config.json").read_text(encoding="utf-8"))
    # Omit these declarations so runtime-only mutations below reach permission
    # checks, rather than failing the earlier source-agreement check.
    del source["use_sliding_window"]
    del source["rope_scaling"]
    (root / "config.json").write_text(json.dumps(source), encoding="utf-8")
    model_id = "semantic-source-roundtrip"
    engine = MaskedTransformerEngine(threads=1)
    asyncio.run(engine.load(load_hf_directory(root, model_id=model_id)))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    plan = lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=2)
    return engine, bundle, source, plan


def test_real_flat_bundle_preserves_independent_source_and_compiles(flat_bundle) -> None:
    from pllm.runtime.model_binding import compile_runtime_model

    engine, bundle, source, plan = flat_bundle
    assert bundle.cfg["semantic_source_config"] == source
    assert "attention_scaling" in bundle.cfg
    assert "attention_scaling" not in bundle.cfg["semantic_source_config"]
    reconstructed = lower_model(
        semantic_source_config(bundle.cfg), batch=1, max_input_tokens=2, max_new_tokens=2,
    )
    assert reconstructed.digest == plan.digest
    assert reconstructed.to_dict()["config_digest"] == plan.to_dict()["config_digest"]
    assert compile_runtime_model(plan, bundle).complete
    loaded = engine.models[bundle.model_id]
    loaded.config["layer_types"][0] = "runtime-only mutation"
    assert loaded.config["semantic_source_config"] == source


@pytest.mark.parametrize("field,value,message", [
    ("use_sliding_window", True, "sliding windows"),
    ("rope_scaling", {"type": "unapproved"}, "rope scaling"),
])
def test_flat_envelope_does_not_grant_nested_runtime_permissions(
    flat_bundle, field: str, value, message: str,
) -> None:
    from pllm.runtime.model_binding import compile_runtime_model

    _, bundle, _, plan = flat_bundle
    bundle.cfg[field] = value
    with pytest.raises(RuntimeBindingError, match=message):
        compile_runtime_model(plan, bundle)


@pytest.mark.parametrize("mutation,message", [
    ("changed", "config (cannot reproduce|does not reproduce)"),
    ("missing", "config (cannot reproduce|does not reproduce)"),
    ("missing-envelope", "^client runtime config does not match its manifest commitment$"),
    ("changed-source", "config (cannot reproduce|does not reproduce)"),
])
def test_real_flat_bundle_rejects_source_reconstruction_drift(
    flat_bundle, mutation: str, message: str,
) -> None:
    from pllm.runtime.model_binding import compile_runtime_model

    _, bundle, _, plan = flat_bundle
    if mutation == "changed":
        bundle.cfg["hidden_size"] += 1
    elif mutation == "missing":
        del bundle.cfg["hidden_size"]
    elif mutation == "missing-envelope":
        del bundle.cfg["semantic_source_config"]
        # Native reconstruction still succeeds; manifest identity owns refusal.
        reconstructed = lower_model(
            semantic_source_config(bundle.cfg), batch=1, max_input_tokens=2, max_new_tokens=2,
        )
        assert reconstructed.digest == plan.digest
        assert reconstructed.to_dict()["config_digest"] == plan.to_dict()["config_digest"]
    else:
        # Matching runtime/source edits must still fail the native plan digest.
        bundle.cfg["rope_theta"] *= 2
        bundle.cfg["semantic_source_config"]["rope_theta"] *= 2
    with pytest.raises(RuntimeBindingError, match=message):
        compile_runtime_model(plan, bundle)


def _nonrotary_contract() -> tuple[dict, dict]:
    from absolute_position_decoder_oracle import tiny_checkpoint
    from pllm.runtime.loaders import resolve_hf_source_geometry

    source, _ = tiny_checkpoint()
    source.update({"bos_token_id": 9, "eos_token_id": 10})
    plan = lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=2).to_dict()
    runtime = _runtime_config({
        **source, **resolve_hf_source_geometry(source), "block_style": "llama",
        "embedding_multiplier": 1.0, "qk_norm": False, "v_norm": False,
    })
    phases = {
        phase: (plan[phase], {op["id"]: op for op in plan[phase]["operations"]})
        for phase in ("prefill", "decode")
    }
    return runtime, phases


def test_nonrotary_native_graph_accepts_ordinary_runtime_controls() -> None:
    runtime, phases = _nonrotary_contract()
    assert runtime["rope_theta"] == 10000.0
    assert runtime["hidden_activation"] == "silu"
    # Native graph, not these inert transport fields, owns LN epsilon and GELU.
    _validate_runtime_semantics(runtime, phases)


@pytest.mark.parametrize("key", ["num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size"])
def test_nonrotary_runtime_geometry_must_match_native_graph(key: str) -> None:
    runtime, phases = _nonrotary_contract()
    runtime[key] += 1
    with pytest.raises(RuntimeBindingError, match="nonrotary"):
        _validate_runtime_semantics(runtime, phases)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("damage", [
    "query-rank", "key-dimension", "boolean-dimension", "score-output", "score-group",
    "scale", "scale-input", "value-shape", "value-group", "reshape-layout", "reshape-output",
])
def test_nonrotary_structural_validator_rejects_bad_metadata(phase: str, damage: str) -> None:
    runtime, phases = _nonrotary_contract()
    operations = phases[phase][1]
    score = next(op for op in operations.values() if op["operator"] == "attention_scores")
    scale = next(op for op in operations.values() if op["operator"] == "attention_scale")
    values = next(op for op in operations.values() if op["operator"] == "attention_values")
    reshape = next(op for op in operations.values()
                   if op["operator"] == "reshape" and op["inputs"] == [values["id"]])
    if damage == "query-rank":
        operations[score["inputs"][0]]["output_shape"].pop()
    elif damage == "key-dimension":
        operations[score["inputs"][1]]["output_shape"][-1] = 0
    elif damage == "boolean-dimension":
        operations[score["inputs"][0]]["output_shape"][0] = True
    elif damage == "score-output":
        score["output_shape"][-1] += 1
    elif damage == "score-group":
        score["attributes"]["group_size"] = True
    elif damage == "scale":
        scale["attributes"]["head_dim"] += 1
    elif damage == "scale-input":
        scale["inputs"] = [score["inputs"][0]]
    elif damage == "value-shape":
        operations[values["inputs"][1]]["output_shape"][-2] += 1
    elif damage == "value-group":
        values["attributes"]["group_size"] += 1
    elif damage == "reshape-layout":
        reshape["attributes"]["input_layout"] = "batch_sequence_heads_feature"
    else:
        reshape["output_shape"][-1] += 1
    # Deliberately mutated metadata tests only this structural check. No forged
    # ModelPlan or fake binding is granted execution authority.
    with pytest.raises(RuntimeBindingError, match="nonrotary"):
        _validate_runtime_semantics(runtime, phases)


@pytest.mark.parametrize("key,value", [
    ("rope_scaling", {"type": "default"}), ("qk_norm", True), ("v_norm", True),
    ("rope_parameters", {"rope_theta": 10000}), ("attention_scaling", 0.5),
    ("layer_types", ["sliding_attention", "full_attention"]), ("sliding_window", 4),
])
def test_nonrotary_does_not_skip_existing_runtime_control_rejections(key: str, value) -> None:
    runtime, phases = _nonrotary_contract()
    runtime[key] = value
    with pytest.raises(RuntimeBindingError):
        _validate_runtime_semantics(runtime, phases)


def test_flat_rotary_head_width_and_qk_checks_remain() -> None:
    source = _flat_source()
    document = lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=2).to_dict()
    phases = {phase: (document[phase], {op["id"]: op for op in document[phase]["operations"]})
              for phase in ("prefill", "decode")}
    runtime = _runtime_config({**source, "block_style": "llama", "embedding_multiplier": 1.0,
                               "qk_norm": False, "v_norm": False})
    _validate_runtime_semantics(runtime, phases)
    with pytest.raises(RuntimeBindingError, match="head dimension"):
        _validate_runtime_semantics({**runtime, "head_dim": runtime["head_dim"] + 1}, phases)
    with pytest.raises(RuntimeBindingError, match="Q/K normalization"):
        _validate_runtime_semantics({**runtime, "qk_norm": True}, phases)


@pytest.fixture
def alias_bundle(tmp_path: Path):
    from absolute_position_decoder_oracle import tiny_checkpoint, write_checkpoint
    from pllm.profiles import ClientOnlyCpu
    from pllm.configuration import Model
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    source, weights = tiny_checkpoint()
    write_checkpoint(tmp_path, source, weights)
    identity = "source-alias-geometry"
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1)
    asyncio.run(engine.load(load_hf_directory(tmp_path, model_id=identity)))
    bundle = ClientBundle.unpack(engine.client_bundle(identity, placement="client"))
    plan = lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=2)
    composition = ClientOnlyCpu(Model(identity), quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
    return source, bundle, plan, composition


def test_source_aliases_bind_with_canonical_runtime_geometry(alias_bundle) -> None:
    from pllm.runtime.loaders import resolve_hf_source_geometry
    from pllm.runtime.model_binding import compile_runtime_model

    source, bundle, plan, composition = alias_bundle
    assert bundle.cfg["semantic_source_config"] == source
    assert "hidden_size" not in source and "max_position_embeddings" not in source
    for key, value in resolve_hf_source_geometry(source).items():
        assert bundle.cfg[key] == value
    assert bundle.cfg["max_position_embeddings"] > plan.prefill["query_sequence"]
    assert compile_runtime_model(plan, bundle, composition=composition).complete


@pytest.mark.parametrize("key", [
    "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings",
])
@pytest.mark.parametrize("owner", ["runtime", "manifest", "both"])
def test_alias_geometry_tampering_cannot_gain_manifest_authority(alias_bundle, key: str, owner: str) -> None:
    from pllm.runtime.model_binding import compile_runtime_model

    _, bundle, plan, composition = alias_bundle
    manifest_key = "context_length" if key == "max_position_embeddings" else key
    if owner in {"runtime", "both"}:
        bundle.cfg[key] += 1
    if owner in {"manifest", "both"}:
        bundle.manifest[manifest_key] += 1
        # Keep ordinary manifest integrity valid: source geometry must reject
        # even a freshly fingerprinted self-consistent transport declaration.
        payload = {k: v for k, v in bundle.manifest.items() if k not in {"created_at", "fingerprint"}}
        bundle.manifest["fingerprint"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest()
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, bundle, composition=composition)


def test_nested_source_geometry_remains_source_owned() -> None:
    from pllm.runtime.loaders import resolve_hf_source_geometry
    from pllm.runtime.model_binding import _validate_source_geometry

    cfg, source = _flattened()
    # Native admission remains independent of this metadata-only check.
    lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=2)
    geometry = resolve_hf_source_geometry(source)
    manifest = {("context_length" if key == "max_position_embeddings" else key): value
                for key, value in geometry.items()}
    runtime = _runtime_config(cfg, nested_source=True)
    _validate_source_geometry(source, cfg, runtime, manifest)
    manifest["context_length"] = 2
    with pytest.raises(RuntimeBindingError, match="source geometry"):
        _validate_source_geometry(source, cfg, runtime, manifest)
