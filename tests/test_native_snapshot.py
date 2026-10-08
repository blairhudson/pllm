from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import weakref

import numpy as np
import pytest

from pllm.runtime.native import MaskedGEMM


@pytest.mark.parametrize("reference", [False, True])
def test_snapshot_views_are_readonly_isolated_and_keep_storage_alive(monkeypatch, reference):
    from pllm.runtime import native

    if reference:
        monkeypatch.setattr(native, "extension", lambda: None)
    source = np.arange(-30, 30, dtype=np.int8).reshape(10, 6)
    expected = source.copy()
    source_ref = weakref.ref(source)
    compiled = MaskedGEMM(threads=1).compile(source)
    first, second = compiled.weight_view(), compiled.weight_view()
    assert np.shares_memory(first, second)
    assert not first.flags.owndata and not first.flags.writeable
    with pytest.raises(ValueError):
        first.setflags(write=True)
    with pytest.raises(TypeError):
        memoryview(first)[0, 0] = 0
    source[:] = 0
    del source
    gc.collect()
    assert source_ref() is None
    np.testing.assert_array_equal(first, expected)
    x = np.ones((2, 6), dtype=np.int8)
    np.testing.assert_array_equal(compiled.clear(x), x.astype(np.int32) @ expected.astype(np.int32).T)
    del compiled, second
    gc.collect()
    np.testing.assert_array_equal(first, expected)


@pytest.mark.parametrize("layout", ["bytes", "slice", "transpose"])
def test_snapshot_import_preserves_input_layout_and_signed_bytes(layout):
    source = np.frombuffer(bytes(range(256)), dtype=np.int8).reshape(16, 16)
    if layout == "slice":
        source = source[:, 2:9]
    elif layout == "transpose":
        source = source.T
    compiled = MaskedGEMM(threads=1).compile(source)
    np.testing.assert_array_equal(compiled.weight_view(), source)
    assert memoryview(compiled._matrix).readonly
    assert memoryview(compiled._matrix).nbytes == source.nbytes


def test_stage_metadata_and_residues_match_whole_matrix_control(monkeypatch):
    from pllm import _native
    from pllm.runtime import transformer_engine as engine
    from pllm.runtime.models import StageSpec
    from pllm.runtime.quantization import QuantizedWeight

    monkeypatch.setattr(engine, "WEIGHT_CHUNK_ELEMENTS", 73)
    values = np.arange(-95, 95, dtype=np.int8).reshape(10, 19)
    runtime = engine.StageRuntime(StageSpec("matrix", "linear", 19, 10, activation_bits=8),
        QuantizedWeight(values, np.ones(10, np.float32), 8), 65537, 24, ())
    assert runtime.weight_digest == hashlib.sha256(values.tobytes()).hexdigest()
    assert runtime.signed_output_bound == 127 * np.abs(values.astype(np.int64)).sum(1).max()
    assert runtime.output_residue_bits == _native.offset_row_bits(values.tobytes(), 19, 127)


@pytest.mark.parametrize("bits", [4, 8])
def test_chunked_fused_transposed_import_matches_whole_row_quantization(tmp_path, monkeypatch, bits):
    from safetensors.numpy import save_file
    from pllm.runtime import transformer_engine as engine
    from pllm.runtime.models import StageSpec
    from pllm.runtime.quantization import quantize_weight_per_row
    from pllm.runtime.safetensors_store import SafeTensorStore

    rng = np.random.default_rng(731)
    a, b = rng.normal(size=(13, 9)).astype(np.float32), rng.normal(size=(7, 13)).astype(np.float32)
    save_file({"a": a, "b": b}, tmp_path / "model.safetensors")
    store = SafeTensorStore(tmp_path)
    monkeypatch.setattr(store, "get_linear", lambda *a, **k: pytest.fail("whole float matrix read"))
    monkeypatch.setattr(engine, "WEIGHT_CHUNK_ELEMENTS", 40)
    original = store.iter_slices
    seen = []

    def bounded_slices(*a, **kw):
        for part in original(*a, **kw):
            seen.append(part.size)
            yield part

    monkeypatch.setattr(store, "iter_slices", bounded_slices)
    model = engine.MaskedTransformerEngine(weight_bits=bits, threads=1)
    stage = StageSpec("fused", "linear", 13, 16, weight_bits=bits)
    actual = model._quantize_sources(store, stage, [("a", True), ("b", False)])
    expected = quantize_weight_per_row(np.concatenate((a.T, b)), bits=bits)
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.scales, expected.scales)
    assert len(seen) > 2 and max(seen) <= 40


def test_loaded_stage_weights_alias_native_owner_and_survive_unload(tmp_path):
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(root, model_id="snapshot")))
    for stage in engine.models["snapshot"].stages.values():
        assert np.shares_memory(stage.weight.values, stage.compiled_weight.weight_view())
        assert not stage.weight.values.flags.writeable
    stage = engine.models["snapshot"].stages["lm_head"]
    view = stage.weight.values
    digest = hashlib.sha256(memoryview(view)).hexdigest()
    del stage
    asyncio.run(engine.unload("snapshot"))
    gc.collect()
    assert hashlib.sha256(memoryview(view)).hexdigest() == digest


def test_client_bundle_weights_reuse_immutable_bytes_and_preserve_tied_lookup(tmp_path):
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(root, model_id="client-snapshot")))
    bundle = ClientBundle.unpack(engine.client_bundle("client-snapshot"))
    table = bundle.stages["token_lookup"]
    head = bundle.stages["lm_head"]
    assert table.client_weight is head.client_weight
    assert not table.client_weight.flags.writeable
    backing = table.client_weight
    while isinstance(backing, np.ndarray):
        backing = backing.base
    assert type(backing) is bytes
    before = bundle.local_token_lookup(np.array([1, 2, 3]))
    value = np.ones((1, head.in_features), np.float32)
    result = bundle.local_linear("lm_head", value)
    assert table.client_weight is head.client_weight
    with pytest.raises(ValueError):
        head.client_weight.setflags(write=True)
    np.testing.assert_array_equal(bundle.local_token_lookup(np.array([1, 2, 3])), before)
    np.testing.assert_array_equal(bundle.local_linear("lm_head", value), result)


@pytest.mark.parametrize("tied", [False, True])
def test_client_snapshot_retires_import_owner_without_changing_binding(tmp_path, tied):
    from pllm import Model, lower_model
    from pllm.model_loader import resolve_model
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, tie_word_embeddings=tied,
    )
    source = resolve_model(Model.path(str(root), model_id="client-owner"))
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(source.manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("client-owner"))
    plan = lower_model(json.loads((root / "config.json").read_text()),
                       batch=1, max_input_tokens=4, max_new_tokens=2)
    binding = compile_runtime_model(plan, bundle)
    identity = binding.digest
    original = bundle.stages["lm_head"].client_weight
    imported = weakref.ref(original)
    expected = original.copy()
    inputs = np.ones((1, original.shape[1]), np.float32)
    before = bundle.local_token_lookup(np.array([1, 2, 3]))
    del original
    result = bundle.local_linear("lm_head", inputs)
    gc.collect()
    assert imported() is None
    view = bundle.stages["lm_head"].client_weight
    assert np.shares_memory(view, bundle._local_matrices["lm_head"].weight_view())
    assert (bundle.stages["token_lookup"].client_weight is view) is tied
    assert bundle.stages["embed_tokens"] is bundle.stages["token_lookup"]
    with pytest.raises(ValueError):
        view.setflags(write=True)
    np.testing.assert_array_equal(view, expected)
    np.testing.assert_allclose(result, expected.astype(np.int32).sum(1)[None].astype(np.float32)
                               * bundle.stages["lm_head"].client_weight_scales, rtol=2e-6)
    np.testing.assert_array_equal(bundle.local_token_lookup(np.array([1, 2, 3])), before)
    assert compile_runtime_model(plan, bundle).digest == identity
    del bundle, binding
    gc.collect()
    np.testing.assert_array_equal(view, expected)
