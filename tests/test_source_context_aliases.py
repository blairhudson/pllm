"""Source geometry aliases share validation, independent of decoder family."""

import asyncio
import copy
import itertools
import json

import pytest

import pllm
from pllm.runtime.loaders import ModelLoadError, load_hf_directory, resolve_hf_source_geometry

from absolute_position_decoder_oracle import tiny_checkpoint, write_checkpoint


ALIASES = ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions")


def _source(tmp_path, fields, *, nested=False):
    # Unsupported family deliberately uses the ordinary descriptive import path;
    # these assertions test shared source metadata, not family-specific lowering.
    config = {
        "model_type": "context_alias_metadata_test", "hidden_size": 4,
        "intermediate_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "vocab_size": 11, **fields,
    }
    raw = {"text_config": config, "max_position_embeddings": 999} if nested else config
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    return load_hf_directory(tmp_path)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("alias", ALIASES)
def test_single_context_alias(tmp_path, alias, nested):
    assert _source(tmp_path, {alias: 17}, nested=nested).context_length == 17


@pytest.mark.parametrize("nested", [False, True])
def test_all_matching_context_aliases(tmp_path, nested):
    assert _source(tmp_path, dict.fromkeys(ALIASES, 17), nested=nested).context_length == 17


@pytest.mark.parametrize("higher,lower", list(itertools.combinations(ALIASES, 2)))
def test_conflicting_context_aliases(tmp_path, higher, lower):
    with pytest.raises(ModelLoadError, match="conflicting.*aliases"):
        _source(tmp_path, {higher: 17, lower: 18})


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("bad", [None, True, False, "17", 17.0, 17.5, 0, -1, [], {}])
def test_malformed_context_alias_is_not_coerced(tmp_path, alias, bad):
    with pytest.raises(ModelLoadError, match="positive integer"):
        _source(tmp_path, {alias: bad})


@pytest.mark.parametrize("higher,lower", list(itertools.combinations(ALIASES, 2)))
@pytest.mark.parametrize("bad", [None, True, "17", 17.0, 0, -1])
@pytest.mark.parametrize("bad_first", [False, True])
def test_malformed_alias_cannot_hide_behind_valid_alias(tmp_path, higher, lower, bad, bad_first):
    fields = {higher: bad if bad_first else 17, lower: 17 if bad_first else bad}
    with pytest.raises(ModelLoadError, match="positive integer"):
        _source(tmp_path, fields)


@pytest.mark.parametrize("nested", [False, True])
def test_context_default_only_when_all_aliases_absent(tmp_path, nested):
    assert _source(tmp_path, {}, nested=nested).context_length == 2048


def test_normal_source_context_matches_native_position_table(tmp_path):
    config, _ = tiny_checkpoint()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    manifest = load_hf_directory(tmp_path)
    plan = pllm.lower_model(config, batch=1, max_input_tokens=6, max_new_tokens=3)
    assert manifest.context_length == config["n_positions"] == 8
    for phase in ("prefill", "decode"):
        position = next(op for op in plan.to_dict()[phase]["operations"]
                        if op["id"] == "position_lookup")
        assert position["attributes"]["weight_shape"] == [manifest.context_length, 4]


GEOMETRY_ALIASES = {
    "hidden_size": ("hidden_size", "d_model", "n_embd"),
    "intermediate_size": ("intermediate_size", "ffn_dim", "n_inner"),
    "num_hidden_layers": ("num_hidden_layers", "n_layer", "num_layers"),
    "num_attention_heads": ("num_attention_heads", "n_head", "num_heads"),
    "num_key_value_heads": ("num_key_value_heads", "n_kv_head"),
    "head_dim": ("head_dim",),
    "vocab_size": ("vocab_size",),
}
CANONICAL_GEOMETRY = {
    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
    "vocab_size": 258, "max_position_embeddings": 256,
}


@pytest.mark.parametrize("nested", [False, True])
def test_canonical_qwen_source_geometry_is_pure(nested):
    config = {"model_type": "qwen2", **CANONICAL_GEOMETRY}
    raw = {"text_config": config, "hidden_size": 999} if nested else config
    before = copy.deepcopy(raw)
    result = resolve_hf_source_geometry(raw)
    assert result == CANONICAL_GEOMETRY
    assert raw == before
    result["hidden_size"] = 123
    assert raw == before


@pytest.mark.parametrize("nested", [False, True])
def test_absolute_source_alias_geometry_matches_canonical_and_manifest(tmp_path, nested):
    config, _ = tiny_checkpoint()
    expected = {
        "hidden_size": 4, "intermediate_size": 7, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 2,
        "vocab_size": 11, "max_position_embeddings": 8,
    }
    raw = {"text_config": config, "hidden_size": 999} if nested else config
    assert resolve_hf_source_geometry(raw) == resolve_hf_source_geometry(expected) == expected
    # Keep lowering independent of nested adapter support; geometry uses the
    # same selected text view, while this manifest remains descriptive metadata.
    raw = copy.deepcopy(raw)
    view = raw["text_config"] if nested else raw
    view["model_type"] = "context_alias_metadata_test"
    view.pop("architectures")
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    manifest = load_hf_directory(tmp_path)
    for key, value in expected.items():
        assert getattr(manifest, "context_length" if key == "max_position_embeddings" else key) == value
    assert manifest.source == str(tmp_path.resolve())
    assert manifest.metadata["text_config"] == view
    assert manifest.metadata["model_type"] == view["model_type"]


@pytest.mark.parametrize("canonical,alias", [
    (canonical, alias) for canonical, aliases in GEOMETRY_ALIASES.items() for alias in aliases
])
@pytest.mark.parametrize("nested", [False, True])
def test_each_geometry_alias_and_matching_canonical(canonical, alias, nested):
    config = dict(CANONICAL_GEOMETRY)
    value = config.pop(canonical)
    config[alias] = value
    raw = {"text_config": config} if nested else config
    assert resolve_hf_source_geometry(raw) == CANONICAL_GEOMETRY
    config[canonical] = value
    assert resolve_hf_source_geometry(raw) == CANONICAL_GEOMETRY


@pytest.mark.parametrize("canonical,higher,lower", [
    (canonical, higher, lower)
    for canonical, aliases in GEOMETRY_ALIASES.items()
    for higher, lower in itertools.combinations(aliases, 2)
])
def test_geometry_aliases_never_first_win(canonical, higher, lower):
    config = dict(CANONICAL_GEOMETRY)
    config.pop(canonical)
    config.update({higher: 32, lower: 33})
    with pytest.raises(ModelLoadError, match="conflicting.*aliases"):
        resolve_hf_source_geometry(config)


@pytest.mark.parametrize("canonical,alias", [
    (canonical, alias) for canonical, aliases in GEOMETRY_ALIASES.items() for alias in aliases
])
@pytest.mark.parametrize("bad", [None, True, False, "8", 8.0, 0, -1, [], {}])
def test_every_geometry_alias_requires_exact_positive_integer(canonical, alias, bad):
    # Retain valid canonical aliases too: malformed lower-priority fields must
    # not hide behind them, and present invalid optional fields must not default.
    config = {**CANONICAL_GEOMETRY, alias: bad}
    before = copy.deepcopy(config)
    with pytest.raises(ModelLoadError, match="positive integer"):
        resolve_hf_source_geometry(config)
    assert config == before


@pytest.mark.parametrize("required", [
    "hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size",
])
@pytest.mark.parametrize("nested", [False, True])
def test_source_geometry_missing_required_does_not_use_outer_config(required, nested):
    config = dict(CANONICAL_GEOMETRY)
    del config[required]
    raw = {**CANONICAL_GEOMETRY, "text_config": config} if nested else config
    with pytest.raises(ModelLoadError, match="missing required model field"):
        resolve_hf_source_geometry(raw)


@pytest.mark.parametrize("nested", [False, True])
def test_source_geometry_only_documented_omission_defaults(nested):
    config = {"hidden_size": 30, "num_attention_heads": 4,
              "num_hidden_layers": 2, "vocab_size": 11}
    raw = {"text_config": config} if nested else config
    assert resolve_hf_source_geometry(raw) == {
        **config, "intermediate_size": 120, "num_key_value_heads": 4,
        "head_dim": 7, "max_position_embeddings": 2048,
    }


@pytest.mark.parametrize("nested", [False, True])
def test_explicit_head_dim_and_gqa_do_not_impose_universal_layout(nested):
    config = {**CANONICAL_GEOMETRY, "hidden_size": 30,
              "num_attention_heads": 5, "num_key_value_heads": 2, "head_dim": 9}
    raw = {"text_config": config} if nested else config
    assert resolve_hf_source_geometry(raw) == config


def test_omitted_head_dim_must_still_resolve_positive():
    config = {"hidden_size": 1, "num_attention_heads": 2,
              "num_hidden_layers": 1, "vocab_size": 11}
    with pytest.raises(ModelLoadError, match="head_dim.*positive integer"):
        resolve_hf_source_geometry(config)


def test_absolute_semantic_export_fills_runtime_geometry_without_source_drift(tmp_path):
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError

    source, weights = tiny_checkpoint()
    root = write_checkpoint(tmp_path / "absolute", source, weights)
    original_json = (root / "config.json").read_text(encoding="utf-8")
    original = json.loads(original_json)
    expected = resolve_hf_source_geometry(original)
    plan = pllm.lower_model(original, batch=1, max_input_tokens=6, max_new_tokens=3)
    manifest = load_hf_directory(root, model_id="absolute-geometry-export")
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(manifest.id, placement="client"))
    assert {key: bundle.cfg[key] for key in expected} == expected
    snapshot = bundle.cfg["semantic_source_config"]
    assert snapshot == original == source
    assert (root / "config.json").read_text(encoding="utf-8") == original_json
    for alias in ("n_embd", "n_head", "n_layer", "n_inner", "n_positions"):
        assert snapshot[alias] == original[alias]
    for key in expected.keys() - original.keys():
        assert key not in snapshot
    reconstructed = pllm.lower_model(snapshot, batch=1, max_input_tokens=6, max_new_tokens=3)
    assert reconstructed.digest == plan.digest
    assert reconstructed.to_dict()["config_digest"] == plan.to_dict()["config_digest"]
    engine.models[manifest.id].config["hidden_size"] = None
    with pytest.raises(TransformerEngineError, match="runtime geometry.*hidden_size.*source"):
        engine.client_bundle(manifest.id, placement="client")
    assert engine.models[manifest.id].config["semantic_source_config"] == original


@pytest.mark.parametrize("key", list(CANONICAL_GEOMETRY))
@pytest.mark.parametrize("bad", [None, True, False, "999", 999.0, 0, -1, 999])
def test_semantic_runtime_geometry_refuses_invalid_or_conflicting_canonical_before_allocation(
    tmp_path, key, bad,
):
    from pllm.runtime.transformer_engine import (
        TransformerEngineError, _populate_semantic_runtime_geometry,
    )

    raw, _ = tiny_checkpoint()
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    manifest = load_hf_directory(tmp_path)
    runtime = {**raw, key: bad}
    before = copy.deepcopy(runtime)
    original = copy.deepcopy(raw)
    with pytest.raises(TransformerEngineError, match=f"runtime geometry.*{key}.*source"):
        _populate_semantic_runtime_geometry(runtime, raw, manifest)
    assert runtime == before
    assert raw == original


@pytest.mark.parametrize("key", list(CANONICAL_GEOMETRY))
def test_semantic_runtime_geometry_checks_manifest_against_original_source(tmp_path, key):
    from pllm.runtime.transformer_engine import (
        TransformerEngineError, _populate_semantic_runtime_geometry,
    )

    raw, _ = tiny_checkpoint()
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    manifest = load_hf_directory(tmp_path)
    field = "context_length" if key == "max_position_embeddings" else key
    setattr(manifest, field, getattr(manifest, field) + 1)
    runtime = dict(raw)
    with pytest.raises(TransformerEngineError, match=f"manifest geometry.*{field}.*source"):
        _populate_semantic_runtime_geometry(runtime, raw, manifest)
    assert runtime == raw


def test_semantic_runtime_geometry_refuses_conflicting_source_aliases(tmp_path):
    from pllm.runtime.transformer_engine import (
        TransformerEngineError, _populate_semantic_runtime_geometry,
    )

    raw, _ = tiny_checkpoint()
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    manifest = load_hf_directory(tmp_path)
    raw["hidden_size"] = raw["n_embd"] + 1
    before = copy.deepcopy(raw)
    runtime = dict(raw)
    with pytest.raises(TransformerEngineError, match="invalid semantic source geometry.*conflicting"):
        _populate_semantic_runtime_geometry(runtime, raw, manifest)
    assert runtime == raw == before
