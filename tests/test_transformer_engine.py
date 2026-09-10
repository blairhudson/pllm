import asyncio
import json
from pathlib import Path

import numpy as np
import msgpack
import pytest

from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.preparation_protocol import seeded_ring_profile
from pllm.runtime.stage_protocol import MaskedStageRequest, MaskedStageResponse, StageCorrelation, unmask_stage_output
from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint
from pllm.runtime.transformer_client import ClientBundle, LayerCache, TransformerClientError
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def run(value):
    return asyncio.run(value)


def test_layer_cache_grows_geometrically_and_snapshots_only_active_rows():
    cache = LayerCache()
    first = np.arange(30 * 4, dtype=np.float32).reshape(30, 2, 2)
    keys, values = cache.append(first, first + 1)

    assert keys.shape == (30, 2, 2)
    assert cache.key is not None
    assert cache.key.shape == (64, 2, 2)
    storage = cache.key

    cache.append(np.ones((1, 2, 2), dtype=np.float32), np.ones((1, 2, 2), dtype=np.float32))
    assert cache.key is storage

    keys, _ = cache.append(
        np.full((40, 2, 2), 2, dtype=np.float32),
        np.full((40, 2, 2), 3, dtype=np.float32),
    )
    assert keys.shape == (71, 2, 2)
    assert cache.key is not None
    assert cache.key.shape == (128, 2, 2)
    np.testing.assert_array_equal(keys[:30], first)

    snapshot = cache.copy_active()
    assert snapshot.length == 71
    assert snapshot.key is not None
    assert snapshot.key.shape == (71, 2, 2)


def test_safetensors_engine_loads_fused_stage_and_executes_masked(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "model")
    manifest = load_hf_directory(root, model_id="tiny")
    assert manifest.stages[0].id == "layers.0.self_attn.qkv_proj"
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    stage = next(row for row in manifest.stages if row.id == "layers.0.self_attn.qkv_proj")
    metadata = run(engine.stage_metadata("tiny", stage.id))
    assert metadata.in_features == 32
    assert metadata.out_features == 64
    correlations = engine.create_local_correlations("tiny", stage.id, 3, seed=9)
    rng = np.random.default_rng(3)
    activation = rng.normal(size=(3, stage.in_features)).astype(np.float32)
    qa = quantize_activation_per_row(activation)
    masks = np.stack([row.mask for row in correlations])
    transformed = np.stack([row.transformed_mask for row in correlations])
    aggregate = StageCorrelation("aggregate", stage.id, masks, transformed, metadata.modulus)
    masked = (qa.values.astype(np.int64) + masks.astype(np.int64)) % metadata.modulus
    request = MaskedStageRequest(
        "tiny", stage.id, aggregate.id, masked.astype(np.uint32), qa.scales,
        metadata.modulus, metadata.wire_bits,
    )
    response = MaskedStageResponse.unpack(run(engine.execute_stage("tiny", stage, [request.pack()]))[0])
    actual = unmask_stage_output(response, aggregate)
    loaded = engine.models["tiny"].stages[stage.id]
    expected = qa.values.astype(np.int32) @ loaded.weight.values.astype(np.int32).T
    assert np.array_equal(actual, expected)
    dequantized = dequantize_matmul(actual, qa.scales, loaded.weight.scales)
    assert dequantized.shape == (3, 64)
    assert engine.stats()["execute_items"] == 3


@pytest.mark.parametrize(("ring", "bits"), [("u16", 16), ("u24", 24), ("u32", 32)])
def test_public_engine_executes_exact_unsigned_ring_shares(
    tmp_path: Path, ring: str, bits: int
):
    root = create_tiny_gemma4_checkpoint(tmp_path / "model")
    manifest = load_hf_directory(root, model_id="tiny-ring32")
    engine = MaskedTransformerEngine(threads=2)
    run(engine.load(manifest))
    stage_id = "layers.0.mlp.gate_up_proj"
    stage = next(row for row in manifest.stages if row.id == stage_id)
    metadata = run(engine.stage_metadata(manifest.id, stage_id))
    assert metadata.weight_digest
    activation = np.random.default_rng(7).normal(size=(3, metadata.in_features)).astype(np.float32)
    quantized = quantize_activation_per_row(activation, bits=metadata.activation_bits)
    modulus = 1 << bits
    clear = quantized.values.astype(np.int64) % modulus
    first = np.random.default_rng(11).integers(
        0, modulus, size=clear.shape, dtype=np.uint32
    )
    second = ((clear - first.astype(np.int64)) % modulus).astype(np.uint32)
    outputs = []
    for label, share in zip(("a", "b"), (first, second)):
        request = MaskedStageRequest(
            manifest.id,
            stage_id,
            label,
            share,
            np.ones(quantized.rows, dtype=np.float32),
            modulus,
            bits,
            ring=ring,
        )
        payload = run(engine.execute_stage(manifest.id, stage, [request.pack()]))[0]
        response = MaskedStageResponse.unpack(payload)
        assert response.ring == ring
        outputs.append(response.masked_output)
    combined = (
        outputs[0].astype(np.uint64) + outputs[1].astype(np.uint64)
    ) % modulus
    actual = np.where(combined >= 1 << (bits - 1), combined - modulus, combined).astype(
        np.int64
    )
    loaded = engine.models[manifest.id].stages[stage_id]
    expected = quantized.values.astype(np.int32) @ loaded.weight.values.astype(np.int32).T
    np.testing.assert_array_equal(actual, expected)


def test_public_bundle_exposes_only_local_boundary_weights(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "model")
    manifest = load_hf_directory(root, model_id="tiny")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("tiny"))
    assert bundle.model_id == "tiny"
    assert "token_lookup" in bundle.stages
    assert "layers.0.self_attn.qkv_proj" in bundle.stages
    assert bundle.stages["token_lookup"].client_weight is not None
    assert bundle.stages["lm_head"].client_weight is not None
    assert bundle.stages["layers.0.self_attn.qkv_proj"].client_weight is None
    assert bundle.privacy["local_quantized_stages"] == ["lm_head", "token_lookup"]
    assert any(key.endswith("input_layernorm.weight") for key in bundle.local_tensors)
    assert not any("q_proj.weight" in key for key in bundle.local_tensors)
    assert bundle.tokenizer_descriptor["type"] == "byte"
    for stage_id, runtime in engine.models["tiny"].stages.items():
        if stage_id in {"token_lookup", "lm_head"}:
            assert bundle.stages[stage_id].seeded_profile is None
        else:
            assert bundle.stages[stage_id].seeded_profile == seeded_ring_profile(
                runtime.signed_output_bound
            )


def test_public_boundary_stages_stay_local_and_head_projects_final_prefill_row(
    tmp_path: Path, monkeypatch
):
    from pllm.runtime.transformer_client import (
        MaskedTransformerClientRuntime,
        RemoteLinear,
    )

    root = create_tiny_gemma4_checkpoint(tmp_path / "model")
    manifest = load_hf_directory(root, model_id="tiny-local-boundaries")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("tiny-local-boundaries"))
    projected_rows: list[int] = []
    original = ClientBundle.local_linear

    def record_local_linear(self, stage_id, activation):
        if stage_id == "lm_head":
            projected_rows.append(int(activation.shape[0]))
        return original(self, stage_id, activation)

    monkeypatch.setattr(ClientBundle, "local_linear", record_local_linear)

    class Provider:
        model_id = "tiny-local-boundaries"

        def take_many(self, stage, count):
            return engine.create_local_correlations(self.model_id, stage.id, count)

    def exchange(stage_id, payloads):
        stage = next(row for row in manifest.stages if row.id == stage_id)
        return run(engine.execute_stage("tiny-local-boundaries", stage, payloads))

    runtime = MaskedTransformerClientRuntime(
        bundle,
        RemoteLinear(bundle.stages, Provider(), exchange),
    )
    runtime.prepare_ids([1, 2, 3])

    assert projected_rows == [1]
    assert engine.models["tiny-local-boundaries"].stages["token_lookup"].calls == 0
    assert engine.models["tiny-local-boundaries"].stages["lm_head"].calls == 0
    assert (
        engine.models["tiny-local-boundaries"]
        .stages["layers.0.self_attn.qkv_proj"]
        .rows
        == 3
    )

    qkv = engine.models["tiny-local-boundaries"].stages["layers.0.self_attn.qkv_proj"]
    calls = qkv.calls
    non_eos = (int(runtime.cfg["eos_token_id"]) + 1) % int(runtime.cfg["vocab_size"])
    monkeypatch.setattr(runtime, "sample", lambda *_args, **_kwargs: non_eos)

    assert len(list(runtime.generate_steps("a", max_output_tokens=1))) == 1
    assert qkv.calls == calls + 1


def test_engine_precision_applies_to_every_stage(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "model")
    manifest = load_hf_directory(root, model_id="tiny-w8")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    run(engine.load(manifest))

    assert {(stage.weight_bits, stage.activation_bits) for stage in manifest.stages} == {(8, 8)}
    values = engine.models["tiny-w8"].stages[manifest.stages[0].id].weight.values
    assert np.max(np.abs(values)) > 7
    bundle = ClientBundle.unpack(engine.client_bundle("tiny-w8"))
    assert bundle.privacy["protocol"] == "masked_w8a8"


def test_tied_w8_bundle_uses_one_canonical_boundary_matrix(tmp_path: Path):
    from safetensors.torch import load_file

    from pllm.runtime.quantization import quantize_weight_per_row

    root = create_tiny_gemma4_checkpoint(tmp_path / "tied", ple_dim=4)
    manifest = load_hf_directory(root, model_id="tied-w8")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    run(engine.load(manifest))

    payload = engine.client_bundle(manifest.id)
    raw = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    assert raw["v"] == 2
    assert set(raw["client_weights"]) == {"tied_embeddings", "token_lookup_aux"}
    assert raw["stages"]["token_lookup"]["client_weight"]["ref"] == "tied_embeddings"
    assert raw["stages"]["lm_head"]["client_weight"]["ref"] == "tied_embeddings"
    stored_bytes = sum(
        len(row["data"]) + len(row["scales"])
        for row in raw["client_weights"].values()
    )
    legacy_bytes = sum(
        runtime.weight.values.nbytes + runtime.weight.scales.nbytes
        for runtime in (
            engine.models[manifest.id].stages["token_lookup"],
            engine.models[manifest.id].stages["lm_head"],
        )
    )
    assert legacy_bytes - stored_bytes == manifest.hidden_size * (manifest.vocab_size + 4)

    bundle = ClientBundle.unpack(payload)
    lookup = bundle.stages["token_lookup"]
    head = bundle.stages["lm_head"]
    assert lookup.client_weight is head.client_weight
    assert lookup.client_weight_scales is head.client_weight_scales

    embedding = load_file(root / "model.safetensors")["model.embed_tokens.weight"].numpy()
    canonical = quantize_weight_per_row(embedding, bits=8)
    np.testing.assert_array_equal(head.client_weight, canonical.values)
    np.testing.assert_array_equal(head.client_weight_scales, canonical.scales)

    ids = np.asarray([0, 17, 257])
    expected_lookup = canonical.values[ids].astype(np.float32) * canonical.scales[ids, None]
    actual_lookup = bundle.local_token_lookup(ids)
    np.testing.assert_array_equal(actual_lookup[:, : manifest.hidden_size], expected_lookup)

    activation = np.random.default_rng(8).normal(size=(3, manifest.hidden_size)).astype(np.float32)
    quantized = quantize_activation_per_row(activation, bits=8)
    expected_integer = quantized.values.astype(np.int32) @ canonical.values.astype(np.int32).T
    expected_head = dequantize_matmul(expected_integer, quantized.scales, canonical.scales)
    np.testing.assert_array_equal(bundle.local_linear("lm_head", activation), expected_head)


def test_tied_w4_bundle_without_ple_rejects_malformed_weight_references(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "tied-w4-no-ple", ple_dim=0)
    manifest = load_hf_directory(root, model_id="tied-w4-no-ple")
    engine = MaskedTransformerEngine(threads=1, weight_bits=4, activation_bits=4)
    run(engine.load(manifest))
    raw = msgpack.unpackb(engine.client_bundle(manifest.id), raw=False, strict_map_key=False)

    assert raw["privacy"]["protocol"] == "masked_w4a4"
    assert set(raw["client_weights"]) == {"tied_embeddings"}
    assert "client_aux_weight" not in raw["stages"]["token_lookup"]

    missing = msgpack.unpackb(msgpack.packb(raw, use_bin_type=True), raw=False)
    missing["stages"]["token_lookup"]["client_weight"]["ref"] = "missing"
    with pytest.raises(TransformerClientError, match="unknown client weight reference"):
        ClientBundle.unpack(msgpack.packb(missing, use_bin_type=True))

    impossible_aux = msgpack.unpackb(msgpack.packb(raw, use_bin_type=True), raw=False)
    impossible_aux["stages"]["token_lookup"]["client_aux_weight"] = {
        "ref": "tied_embeddings"
    }
    with pytest.raises(TransformerClientError, match="invalid auxiliary client weight shape"):
        ClientBundle.unpack(msgpack.packb(impossible_aux, use_bin_type=True))


def test_untied_bundle_keeps_distinct_boundary_weights(tmp_path: Path):
    from safetensors.torch import load_file, save_file

    from pllm.runtime.tiny_gemma import create_tiny_llama_checkpoint

    root = create_tiny_llama_checkpoint(tmp_path / "untied", num_hidden_layers=1)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tie_word_embeddings"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    tensors = load_file(root / "model.safetensors")
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"] * 0.5
    save_file(tensors, root / "model.safetensors")

    manifest = load_hf_directory(root, model_id="untied")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    run(engine.load(manifest))
    raw = msgpack.unpackb(engine.client_bundle(manifest.id), raw=False, strict_map_key=False)
    assert set(raw["client_weights"]) == {"token_lookup", "lm_head"}
    assert raw["stages"]["token_lookup"]["client_weight"]["ref"] == "token_lookup"
    assert raw["stages"]["lm_head"]["client_weight"]["ref"] == "lm_head"

    bundle = ClientBundle.unpack(engine.client_bundle(manifest.id))
    token_stage = bundle.stages["token_lookup"]
    assert token_stage.client_weight is not bundle.stages["lm_head"].client_weight

    ids = np.asarray([2, 2, 57, 257])
    runtime = engine.models[manifest.id].stages["token_lookup"]
    qmax = 127
    integer = runtime.weight.values[:, ids].T.astype(np.int32) * qmax
    activation_scales = np.full(ids.size, np.float32(1.0 / qmax), dtype=np.float32)
    expected = dequantize_matmul(integer, activation_scales, runtime.weight.scales)
    np.testing.assert_array_equal(bundle.local_token_lookup(ids), expected)


def test_client_runtime_honors_tokenizer_add_bos_setting(tmp_path: Path):
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime

    root = create_tiny_gemma4_checkpoint(tmp_path / "model")
    tokenizer_config = root / "tokenizer_config.json"
    value = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    value["add_bos_token"] = False
    tokenizer_config.write_text(json.dumps(value), encoding="utf-8")
    manifest = load_hf_directory(root, model_id="tiny-no-bos")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("tiny-no-bos"))
    runtime = MaskedTransformerClientRuntime(bundle, lambda *_: None)

    assert bundle.tokenizer_descriptor["add_bos_token"] is False
    assert runtime.encode_prompt("A") == [65]


def test_masked_runtime_matches_clear_w4a4_runtime(tmp_path: Path):
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime, RemoteLinear

    root = create_tiny_gemma4_checkpoint(tmp_path / "model")
    manifest = load_hf_directory(root, model_id="tiny-parity")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("tiny-parity"))

    class Provider:
        model_id = "tiny-parity"

        def take_many(self, stage, count):
            return engine.create_local_correlations("tiny-parity", stage.id, count)

    def exchange(stage_id, payloads):
        stage = next(row for row in manifest.stages if row.id == stage_id)
        return run(engine.execute_stage("tiny-parity", stage, payloads))

    class ClearRemote:
        def __call__(self, stage_id, activation):
            stage = bundle.stages[stage_id]
            runtime = engine.models["tiny-parity"].stages[stage_id]
            qa = quantize_activation_per_row(activation, bits=stage.activation_bits)
            integer = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
            return dequantize_matmul(
                integer,
                qa.scales,
                runtime.weight.scales,
                output_shape=qa.original_shape[:-1] + (stage.out_features,),
            )

    masked = MaskedTransformerClientRuntime(
        bundle,
        RemoteLinear(bundle.stages, Provider(), exchange),
    )
    clear = MaskedTransformerClientRuntime(bundle, ClearRemote())
    masked_ids, masked_logits, _ = masked.prepare("parity")
    clear_ids, clear_logits, _ = clear.prepare("parity")
    assert masked_ids == clear_ids
    assert np.array_equal(masked_logits, clear_logits)


def test_client_bundle_embeds_tokenizer_assets(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "model")
    manifest = load_hf_directory(root, model_id="tiny-tokenizer")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("tiny-tokenizer"))
    assert bundle.tokenizer_descriptor["type"] == "byte"
    assert bundle.tokenizer_descriptor["vocab_size"] == 258
    assert "chat_template" in bundle.tokenizer_descriptor


def test_token_lookup_lru_deduplicates_and_reuses_private_rows(tmp_path: Path):
    from collections import OrderedDict
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime, RemoteLinear

    root = create_tiny_gemma4_checkpoint(
        tmp_path / "model-cache",
        num_hidden_layers=1,
        ple_dim=4,
    )
    manifest = load_hf_directory(root, model_id="tiny-cache")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(
        engine.client_bundle("tiny-cache", include_local_weights=False)
    )

    class Provider:
        model_id = "tiny-cache"

        def take_many(self, stage, count):
            return engine.create_local_correlations("tiny-cache", stage.id, count)

    def exchange(stage_id, payloads):
        stage = next(row for row in manifest.stages if row.id == stage_id)
        return run(engine.execute_stage("tiny-cache", stage, payloads))

    runtime = MaskedTransformerClientRuntime(
        bundle,
        RemoteLinear(bundle.stages, Provider(), exchange),
        token_cache=OrderedDict(),
        token_cache_size=8,
    )
    first_hidden, first_ple = runtime._token_lookup(np.asarray([5, 5, 6, 5]))
    lookup = engine.models["tiny-cache"].stages["token_lookup"]
    assert lookup.rows == 2
    second_hidden, second_ple = runtime._token_lookup(np.asarray([5, 6]))
    assert lookup.rows == 2
    assert np.array_equal(first_hidden[[0, 2]], second_hidden)
    assert np.array_equal(first_ple[[0, 2]], second_ple)
    # Hits count only entries already present in the LRU before the call.
    # Duplicate misses are still deduplicated in the private batch.
    assert runtime.token_cache_hits == 2
    assert runtime.token_cache_misses == 2


def test_client_bundle_sets_gemma4_runtime_semantics(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "model-semantics")
    manifest = load_hf_directory(root, model_id="tiny-semantics")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("tiny-semantics"))
    assert bundle.cfg["block_style"] == "gemma4"
    assert bundle.cfg["qk_norm"] is True
    assert bundle.cfg["v_norm"] is True
    assert bundle.cfg["norm_offset"] == 0.0
    assert bundle.cfg["attention_scaling"] == 1.0
    assert bundle.cfg["embedding_multiplier"] == np.sqrt(32)


def test_token_lookup_lru_avoids_repeat_remote_calls(tmp_path: Path):
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime

    root = create_tiny_gemma4_checkpoint(tmp_path / "model")
    manifest = load_hf_directory(root, model_id="tiny-cache")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(
        engine.client_bundle("tiny-cache", include_local_weights=False)
    )
    calls: dict[str, int] = {}

    class ClearRemote:
        def __call__(self, stage_id, activation):
            calls[stage_id] = calls.get(stage_id, 0) + 1
            stage = bundle.stages[stage_id]
            runtime = engine.models["tiny-cache"].stages[stage_id]
            qa = quantize_activation_per_row(activation, bits=stage.activation_bits)
            integer = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
            return dequantize_matmul(
                integer,
                qa.scales,
                runtime.weight.scales,
                output_shape=qa.original_shape[:-1] + (stage.out_features,),
            )

    runtime = MaskedTransformerClientRuntime(bundle, ClearRemote(), token_cache_size=8)
    ids = np.asarray([2, 2, 3, 2], dtype=np.int64)
    first, _ = runtime._token_lookup(ids)
    second, _ = runtime._token_lookup(ids)
    assert np.array_equal(first, second)
    assert calls["token_lookup"] == 1
    assert runtime.token_cache_misses == 2
    assert runtime.token_cache_hits == 4


def test_llama_compatible_checkpoint_uses_generic_client_graph(tmp_path: Path):
    from pllm.runtime.tiny_gemma import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime, RemoteLinear

    root = create_tiny_llama_checkpoint(tmp_path / "tiny-llama", num_hidden_layers=1)
    manifest = load_hf_directory(root, model_id="tiny-llama")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("tiny-llama"))
    assert bundle.cfg["block_style"] == "llama"
    assert bundle.cfg["qk_norm"] is False
    assert bundle.cfg["v_norm"] is False
    assert bundle.cfg["embedding_multiplier"] == 1.0

    class Provider:
        model_id = "tiny-llama"

        def take_many(self, stage, count):
            return engine.create_local_correlations("tiny-llama", stage.id, count)

    def exchange(stage_id, payloads):
        stage = next(row for row in manifest.stages if row.id == stage_id)
        return run(engine.execute_stage("tiny-llama", stage, payloads))

    runtime = MaskedTransformerClientRuntime(
        bundle,
        RemoteLinear(bundle.stages, Provider(), exchange),
    )
    ids, logits, _ = runtime.prepare("llama")
    assert ids
    assert logits.shape == (258,)
    assert np.isfinite(logits).all()


def test_gemma4_client_graph_matches_independent_quantized_reference(tmp_path: Path):
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime

    root = create_tiny_gemma4_checkpoint(
        tmp_path / "gemma-reference",
        num_hidden_layers=2,
        ple_dim=4,
    )
    manifest = load_hf_directory(root, model_id="gemma-reference")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("gemma-reference"))

    def qlinear(stage_id: str, value: np.ndarray) -> np.ndarray:
        stage = bundle.stages[stage_id]
        runtime = engine.models["gemma-reference"].stages[stage_id]
        qa = quantize_activation_per_row(value, bits=stage.activation_bits)
        integer = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
        return dequantize_matmul(
            integer,
            qa.scales,
            runtime.weight.scales,
            output_shape=qa.original_shape[:-1] + (stage.out_features,),
        )

    def tensor(suffix: str, width: int) -> np.ndarray:
        return bundle.tensor(suffix, default=np.ones(width, dtype=np.float32))

    def norm(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
        x = np.asarray(value, dtype=np.float32)
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * weight

    def rope(value: np.ndarray, positions: np.ndarray) -> np.ndarray:
        dim = value.shape[-1]
        frequency = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        angles = positions.astype(np.float32)[:, None] * frequency[None, :]
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[:, None, :]
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[:, None, :]
        half = dim // 2
        rotated = np.concatenate([-value[..., half:], value[..., :half]], axis=-1)
        return value * cos + rotated * sin

    tokenizer = bundle.tokenizer()
    ids = tokenizer.encode("reference", add_bos=True)
    token_stage = bundle.stages["token_lookup"]
    assert token_stage.client_weight is not None
    assert token_stage.client_weight_scales is not None
    lookup = (
        token_stage.client_weight[ids].astype(np.float32)
        * token_stage.client_weight_scales[ids, None]
    )
    assert token_stage.client_aux_weight is not None
    assert token_stage.client_aux_scales is not None
    auxiliary = (
        token_stage.client_aux_weight[:, ids].T.astype(np.float32)
        * token_stage.client_aux_scales[None, :]
    )
    lookup = np.concatenate((lookup, auxiliary), axis=-1)
    hidden = lookup[:, :32] * np.sqrt(32.0)
    token_ple = lookup[:, 32:].reshape(len(ids), 2, 4) * 2.0
    context_ple = qlinear("model.per_layer_model_projection", hidden)
    context_ple = (context_ple * (32.0 ** -0.5)).reshape(len(ids), 2, 4)
    context_ple = norm(context_ple, tensor("per_layer_projection_norm.weight", 4))
    ple = (token_ple + context_ple) * (2.0 ** -0.5)
    positions = np.arange(len(ids), dtype=np.int64)

    for index in range(2):
        residual = hidden
        x = norm(hidden, tensor(f"layers.{index}.input_layernorm.weight", 32))
        qkv = qlinear(f"layers.{index}.self_attn.qkv_proj", x)
        q = qkv[:, :32].reshape(len(ids), 4, 8)
        k = qkv[:, 32:48].reshape(len(ids), 2, 8)
        v = qkv[:, 48:64].reshape(len(ids), 2, 8)
        q = norm(q, tensor(f"layers.{index}.self_attn.q_norm.weight", 8))
        k = norm(k, tensor(f"layers.{index}.self_attn.k_norm.weight", 8))
        v = v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + 1e-6)
        q = rope(q, positions)
        k = rope(k, positions)
        rk = np.repeat(k, 2, axis=1)
        rv = np.repeat(v, 2, axis=1)
        attended = np.empty_like(q)
        for row in range(len(ids)):
            scores = np.einsum("hd,thd->ht", q[row], rk[: row + 1])
            scores -= scores.max(axis=-1, keepdims=True)
            probabilities = np.exp(scores).astype(np.float32)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            attended[row] = np.einsum("ht,thd->hd", probabilities, rv[: row + 1])
        attention = qlinear(
            f"layers.{index}.self_attn.o_proj",
            attended.reshape(len(ids), 32),
        )
        attention = norm(
            attention,
            tensor(f"layers.{index}.post_attention_layernorm.weight", 32),
        )
        hidden = residual + attention

        residual = hidden
        x = norm(hidden, tensor(f"layers.{index}.pre_feedforward_layernorm.weight", 32))
        gate_up = qlinear(f"layers.{index}.mlp.gate_up_proj", x)
        gate, up = np.split(gate_up, 2, axis=-1)
        mlp = gate / (1.0 + np.exp(-gate)) * up
        mlp = qlinear(f"layers.{index}.mlp.down_proj", mlp)
        mlp = norm(mlp, tensor(f"layers.{index}.post_feedforward_layernorm.weight", 32))
        hidden = residual + mlp

        residual = hidden
        ple_gate = qlinear(f"layers.{index}.per_layer_input_gate", hidden)
        ple_gate = ple_gate / (1.0 + np.exp(-ple_gate)) * ple[:, index, :]
        ple_update = qlinear(f"layers.{index}.per_layer_projection", ple_gate)
        ple_update = norm(
            ple_update,
            tensor(f"layers.{index}.post_per_layer_input_norm.weight", 32),
        )
        hidden = residual + ple_update

    reference = qlinear("lm_head", norm(hidden, tensor("model.norm.weight", 32)))

    class ClearRemote:
        def __call__(self, stage_id, activation):
            return qlinear(stage_id, activation)

    runtime = MaskedTransformerClientRuntime(bundle, ClearRemote(), token_cache_size=0)
    runtime_ids, actual, _ = runtime.prepare("reference")
    assert runtime_ids == ids
    assert np.allclose(actual, reference[-1], rtol=1e-5, atol=1e-5)


def test_sharded_safetensors_checkpoint_loads_through_engine(tmp_path: Path):
    import json
    from safetensors.torch import load_file, save_file

    root = create_tiny_gemma4_checkpoint(
        tmp_path / "sharded",
        num_hidden_layers=1,
        ple_dim=4,
    )
    tensors = load_file(root / "model.safetensors")
    keys = sorted(tensors)
    midpoint = len(keys) // 2
    shards = [keys[:midpoint], keys[midpoint:]]
    weight_map = {}
    for index, shard_keys in enumerate(shards, start=1):
        name = f"model-{index:05d}-of-00002.safetensors"
        save_file({key: tensors[key] for key in shard_keys}, root / name)
        weight_map.update({key: name for key in shard_keys})
    (root / "model.safetensors").unlink()
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = load_hf_directory(root, model_id="tiny-sharded")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("tiny-sharded"))
    assert bundle.manifest["id"] == "tiny-sharded"
    assert "layers.0.mlp.gate_up_proj" in bundle.stages


def test_token_lookup_cache_can_be_disabled(tmp_path: Path):
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime

    root = create_tiny_gemma4_checkpoint(tmp_path / "model-no-cache")
    manifest = load_hf_directory(root, model_id="tiny-no-cache")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(
        engine.client_bundle("tiny-no-cache", include_local_weights=False)
    )
    calls = 0

    class ClearRemote:
        def __call__(self, stage_id, activation):
            nonlocal calls
            calls += int(stage_id == "token_lookup")
            stage = bundle.stages[stage_id]
            runtime = engine.models["tiny-no-cache"].stages[stage_id]
            qa = quantize_activation_per_row(activation, bits=stage.activation_bits)
            integer = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
            return dequantize_matmul(integer, qa.scales, runtime.weight.scales)

    runtime = MaskedTransformerClientRuntime(bundle, ClearRemote(), token_cache_size=0)
    ids = np.asarray([2, 2, 3], dtype=np.int64)
    runtime._token_lookup(ids)
    runtime._token_lookup(ids)
    assert calls == 2
    assert len(runtime.token_cache) == 0
    assert runtime.token_cache_hits == 0
    assert runtime.token_cache_misses == 4


def test_streaming_bfloat16_compiler_uses_memmap_and_cache(tmp_path: Path):
    import json
    import torch
    from safetensors.torch import save_file

    root = tmp_path / "streaming-model"
    root.mkdir()
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 512,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 512,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "he_test_tokenizer": "byte",
        "bos_token_id": 256,
        "eos_token_id": 257,
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "he_tokenizer.json").write_text(json.dumps({
        "type": "byte", "vocab_size": 512, "bos_token_id": 256, "eos_token_id": 257,
    }))
    g = torch.Generator().manual_seed(99)
    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.bfloat16) * 0.05
    tensors = {
        "model.embed_tokens.weight": r(512, 32),
        "model.layers.0.self_attn.q_proj.weight": r(32, 32),
        "model.layers.0.self_attn.k_proj.weight": r(16, 32),
        "model.layers.0.self_attn.v_proj.weight": r(16, 32),
        "model.layers.0.self_attn.o_proj.weight": r(32, 32),
        "model.layers.0.mlp.gate_proj.weight": r(64, 32),
        "model.layers.0.mlp.up_proj.weight": r(64, 32),
        "model.layers.0.mlp.down_proj.weight": r(32, 64),
        "model.layers.0.input_layernorm.weight": torch.ones(32, dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(32, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(32, dtype=torch.bfloat16),
    }
    save_file(tensors, root / "model.safetensors")
    cache = tmp_path / "compiled"
    manifest = load_hf_directory(root, model_id="streaming-bf16")
    engine = MaskedTransformerEngine(
        threads=1,
        compiled_cache_dir=cache,
        streaming_threshold_elements=1,
        quantization_chunk_rows=7,
    )
    run(engine.load(manifest))
    token_weight = engine.models["streaming-bf16"].stages["token_lookup"].weight
    assert isinstance(token_weight.values, np.memmap)
    assert token_weight.values.shape == (32, 512)
    files = sorted(cache.rglob("*.i8"))
    assert files
    mtimes = {path: path.stat().st_mtime_ns for path in files}

    second = MaskedTransformerEngine(
        threads=1,
        compiled_cache_dir=cache,
        streaming_threshold_elements=1,
        quantization_chunk_rows=5,
    )
    awaitable = second.load(load_hf_directory(root, model_id="streaming-bf16"))
    run(awaitable)
    assert {path: path.stat().st_mtime_ns for path in files} == mtimes
    assert isinstance(second.models["streaming-bf16"].stages["token_lookup"].weight.values, np.memmap)


def test_safetensor_stream_reuses_one_open_handle(tmp_path: Path, monkeypatch):
    import json

    import safetensors
    from safetensors.numpy import save_file

    from pllm.runtime.safetensors_store import SafeTensorStore

    root = tmp_path / "checkpoint"
    root.mkdir()
    path = root / "model.safetensors"
    values = np.arange(48, dtype=np.float32).reshape(8, 6)
    save_file({"model.weight": values}, path)
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.weight": path.name},
    }))
    store = SafeTensorStore(root)
    original_safe_open = safetensors.safe_open
    opens = 0

    def counting_safe_open(*args, **kwargs):
        nonlocal opens
        opens += 1
        return original_safe_open(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", counting_safe_open)
    assert store.tensor_shape("model.weight") == (8, 6)
    assert store.tensor_dtype("model.weight") == "F32"
    assert store.tensor_shape("model.weight") == (8, 6)
    chunks = list(store.iter_slices(
        "model.weight",
        ((slice(start, start + 2), slice(None)) for start in range(0, 8, 2)),
    ))
    np.testing.assert_array_equal(np.concatenate(chunks), values)
    assert opens == 2


def test_streaming_compiler_serializes_concurrent_cache_writers(tmp_path: Path):
    import concurrent.futures
    import json
    import torch
    from safetensors.torch import save_file

    root = tmp_path / "concurrent-model"
    root.mkdir()
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 512,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 512,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "he_test_tokenizer": "byte",
        "bos_token_id": 256,
        "eos_token_id": 257,
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "he_tokenizer.json").write_text(json.dumps({
        "type": "byte", "vocab_size": 512, "bos_token_id": 256, "eos_token_id": 257,
    }))
    generator = torch.Generator().manual_seed(123)
    random = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float16) * 0.05
    save_file({
        "model.embed_tokens.weight": random(512, 32),
        "model.layers.0.self_attn.q_proj.weight": random(32, 32),
        "model.layers.0.self_attn.k_proj.weight": random(16, 32),
        "model.layers.0.self_attn.v_proj.weight": random(16, 32),
        "model.layers.0.self_attn.o_proj.weight": random(32, 32),
        "model.layers.0.mlp.gate_proj.weight": random(64, 32),
        "model.layers.0.mlp.up_proj.weight": random(64, 32),
        "model.layers.0.mlp.down_proj.weight": random(32, 64),
        "model.layers.0.input_layernorm.weight": torch.ones(32),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(32),
        "model.norm.weight": torch.ones(32),
    }, root / "model.safetensors")
    cache = tmp_path / "compiled"

    def compile_once(index: int):
        manifest = load_hf_directory(root, model_id=f"concurrent-{index}")
        # Cache identity intentionally ignores the model alias and follows source tensors + stage shape.
        engine = MaskedTransformerEngine(
            threads=1,
            compiled_cache_dir=cache,
            streaming_threshold_elements=1,
            quantization_chunk_rows=5 + index,
        )
        run(engine.load(manifest))
        return engine.models[manifest.id].stages["token_lookup"].weight.values.copy()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(compile_once, (0, 1)))
    assert np.array_equal(first, second)
    assert not list(cache.rglob("*.tmp"))
    assert list(cache.rglob("*.json"))
