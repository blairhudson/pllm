from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.modeling import DecoderContinuationSchedule
from pllm.profiles import MaskedLinearCpu, VerifiedMaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, TransformerClientError
from pllm.runtime.transformer_engine import MaskedTransformerEngine
from pllm.runtime.prefill_cache import ExactPrefillCache, prefill_key


class QuantizedRemote:
    """Same integer clear kernel and per-row scales as prepared execution."""

    def __init__(self, engine, model_id):
        self.model = engine.models[model_id]
        self.calls = []
        self.inject = None

    def __call__(self, stage_id, activation):
        self.calls.append((stage_id, activation.shape))
        stage = self.model.stages[stage_id]
        quantized = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
        output = dequantize_matmul(
            stage.compiled_weight.clear(quantized.values), quantized.scales,
            stage.weight.scales,
            output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,),
        )
        if stage.bias is not None:
            output += stage.bias
        if self.inject == "nonfinite":
            output.flat[0] = np.nan
        elif self.inject == "shape":
            output = output[..., :-1]
        return np.ascontiguousarray(output, dtype=np.float32)


def bound_checkpoint(root, *, bound=64):
    model_id = "continuation-checkpoint"
    config = json.loads((root / "config.json").read_bytes())
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(root, model_id=model_id)))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    plan = lower_model(config, batch=1, max_input_tokens=bound, max_new_tokens=4)
    composition = MaskedLinearCpu(
        Model(model_id), quantization=SymmetricPerRow(weight_bits=8, activation_bits=8)
    )
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    contract = plan.continuation_schedule(composition)
    return engine, compiled, contract, composition


@pytest.fixture(params=[("qwen2", 97), ("qwen3", 131)])
def checkpoint(tmp_path, request):
    family, seed = request.param
    root = create_tiny_llama_checkpoint(
        tmp_path / "checkpoint", model_type=family, seed=seed,
        hidden_size=48, intermediate_size=96, head_dim=12, num_hidden_layers=3,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3",
        tie_word_embeddings=False,
    )
    return bound_checkpoint(root)


def runtime_for(compiled, engine):
    remote = QuantizedRemote(engine, compiled._bundle.model_id)
    return compiled.runtime(remote), remote


@pytest.mark.parametrize("length", [9, 31, 64])
def test_canonical_numeric_contract_all_splits_and_teacher_fold_are_bit_exact(checkpoint, length):
    engine, original, _, _ = checkpoint
    pipeline = MaskedLinearCpu(Model(original._bundle.model_id), quantization=SymmetricPerRow(
        causal_reduction="prefix_f32"))
    compiled = compile_runtime_model(original._plan, original._bundle, composition=pipeline)
    assert compiled.digest != original.digest
    contract = original._plan.continuation_schedule(pipeline)
    ids = np.random.default_rng(length).integers(2, 258, length).tolist()
    fresh, _ = runtime_for(compiled, engine)
    expected = fresh.prepare_ids(ids)[1]
    golden = fresh.snapshot()
    for prefix in range(1, length):
        candidate, _ = runtime_for(compiled, engine)
        candidate.prepare_ids(ids[:prefix])
        candidate.install_continuation(contract)
        actual = candidate.continue_ids(ids[prefix:])[-1]
        np.testing.assert_array_equal(actual, expected)
        for before, after in zip(golden.caches, candidate.snapshot().caches, strict=True):
            np.testing.assert_array_equal(before.key, after.key)
            np.testing.assert_array_equal(before.value, after.value)
    folded, _ = runtime_for(compiled, engine)
    actual = folded.prepare_ids(ids[:1])[1]
    for token in ids[1:]:
        actual = folded.decode_step(token, folded.caches)[0]
    np.testing.assert_array_equal(actual, expected)
    for before, after in zip(golden.caches, folded.snapshot().caches, strict=True):
        np.testing.assert_array_equal(before.key, after.key)
        np.testing.assert_array_equal(before.value, after.value)


def test_native_contract_binds_original_geometry_and_separate_semantics(checkpoint):
    _, compiled, contract, composition = checkpoint
    source = compiled._plan.runtime_schedule(composition)
    spec = contract.to_dict()
    assert spec["source_plan_digest"] == compiled.model_plan_digest
    assert spec["source_schedule_digest"] == source.digest
    assert contract.digest != source.digest
    assert spec["token_bound"] == 64
    assert spec["graph"]["state_inputs"]
    for before, after in zip(source.to_dict()["prefill"]["steps"], spec["schedule"]["steps"], strict=True):
        assert before["operation_ids"] == after["operation_ids"]
        assert before["outputs"] == after["outputs"]
        assert before["executor"] == after["executor"]
    assert contract.admit(8, 23) > spec["state_row_bytes"] * 31
    with pytest.raises(ValueError, match="fixed compiler input bound"):
        contract.admit(42, 23)
    with pytest.raises(ValueError, match="memory budget"):
        contract.admit(8, 23, memory_bytes=1)
    forged = spec.copy()
    forged["numeric_digest"] = "a" * 64
    with pytest.raises(ValueError, match="contract mismatch"):
        dataclasses.replace(contract, _canonical_bytes=json.dumps(forged).encode())
    with pytest.raises(ValueError, match="prepared public unverified"):
        compiled._plan.continuation_schedule(VerifiedMaskedLinearCpu(Model("continuation-checkpoint")))


@pytest.mark.parametrize("prefix,suffix", [(1, 1), (8, 23), (13, 7), (31, 33)])
def test_every_final_logit_and_teacher_fold_preserve_legacy_executor(checkpoint, prefix, suffix, record_property):
    engine, compiled, contract, _ = checkpoint
    ids = np.random.default_rng(prefix * 31 + suffix).integers(2, 258, prefix + suffix).tolist()
    fresh, _ = runtime_for(compiled, engine)
    reference = fresh.prepare_ids(ids)[1]
    initial, _ = runtime_for(compiled, engine)
    initial.prepare_ids(ids[:prefix])
    saved = initial.snapshot()
    batched, calls = runtime_for(compiled, engine)
    batched.install_continuation(contract)
    batched.restore(saved)
    actual = batched.continue_ids(ids[prefix:])[-1]
    record_property("array_equal_all_logits", bool(np.array_equal(actual, reference)))
    record_property("max_abs_logit_difference", float(np.max(np.abs(actual - reference))))
    # Compare every vocabulary logit. Float attention sums can change ULPs when
    # causal rows have differently sized zero-padded reductions; never call that bit parity.
    np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=0)
    assert np.argmax(actual) == np.argmax(reference)
    assert len(calls.calls) == 4 * batched.layers
    assert all(shape[0] == suffix for _, shape in calls.calls)
    np.testing.assert_array_equal(saved.caches[0].key, batched.caches[0].key[:prefix])
    assert batched.position == prefix + suffix
    sequential, seq_calls = runtime_for(compiled, engine)
    sequential.restore(saved)
    legacy = sequential.forward_ids(ids[prefix:])[-1]
    np.testing.assert_allclose(actual, legacy, atol=1e-5, rtol=0)
    assert len(seq_calls.calls) == suffix * 4 * sequential.layers
    # Fixed teacher tokens, not independent argmax feedback, isolate state drift.
    for token in (7, 101, 42):
        teacher = fresh.decode_step(token, fresh.caches)[0]
        candidate = batched.decode_step(token, batched.caches)[0]
        np.testing.assert_allclose(candidate, teacher, atol=1e-5, rtol=0)
        assert np.argmax(candidate) == np.argmax(teacher)


@pytest.mark.parametrize("failure", ["bound", "memory", "shape", "nonfinite", "state", "tokens"])
def test_preflight_and_invalid_execution_fail_closed(checkpoint, failure):
    engine, compiled, contract, _ = checkpoint
    runtime, remote = runtime_for(compiled, engine)
    runtime.prepare_ids([2] * 8)
    runtime.install_continuation(contract)
    before = runtime.snapshot()
    remote.calls.clear()
    query = [7] * 3
    memory = 2 << 30
    if failure == "bound":
        query = [7] * 57
    elif failure == "memory":
        memory = 1
    elif failure in {"shape", "nonfinite"}:
        remote.inject = failure
    elif failure == "state":
        runtime.caches[0].key[0, 0, 0] = np.nan
    else:
        query = [[7]]
    with pytest.raises((TransformerClientError, ValueError)):
        runtime.continue_ids(query, memory_bytes=memory)
    if failure in {"bound", "memory", "tokens"}:
        assert not remote.calls
        assert runtime.position == before.position
        np.testing.assert_array_equal(runtime.caches[0].key[:before.position], before.caches[0].key)
    with pytest.raises(TransformerClientError, match="burned"):
        runtime.decode_step(7, runtime.caches)


def test_restore_validates_before_mutating_runtime(checkpoint):
    engine, compiled, _, _ = checkpoint
    runtime, _ = runtime_for(compiled, engine)
    runtime.prepare_ids([2, 3])
    saved = runtime.snapshot()
    before = runtime.caches
    saved.caches[0].value = np.zeros((2, 1, 1), np.float32)
    with pytest.raises(TransformerClientError, match="shape/numeric"):
        runtime.restore(saved)
    assert runtime.caches is before and runtime.position == 2


def test_native_phase_basis_confines_generated_state_to_explicit_owner(checkpoint):
    engine, compiled, contract, _ = checkpoint
    source, _ = runtime_for(compiled, engine)
    ids = [2, 19, 45, 82, 12, 23, 104, 27]
    _, logits, _ = source.prepare_ids(ids)
    prefill = source.snapshot()
    assert prefill.state_basis.completed_prefill()
    cache = ExactPrefillCache(1 << 20)
    assert cache.put("prefill", prefill, logits)
    assert cache.put_prefixes(compiled.digest, "bundle", ids, prefill)
    # No implicit prefill basis for old/unqualified snapshot codecs.
    assert not cache.put("unknown", dataclasses.replace(prefill, state_basis=None), logits)
    source.decode_step(7, source.caches)
    generated = source.snapshot()
    assert generated.state_basis.phase == "incremental" and generated.state_basis.valid()
    assert generated.state_basis.native_phase_digest != prefill.state_basis.native_phase_digest
    assert generated.state_basis.execution_digest != prefill.state_basis.execution_digest
    assert not cache.put("generated", generated, logits)
    assert not cache.put_prefixes(compiled.digest, "bundle", [*ids, 7], generated)
    # Caller relabeling cannot turn a native decode result into fresh-prefill KV.
    forged = dataclasses.replace(generated, state_basis=dataclasses.replace(
        generated.state_basis, phase="completed_prefill"
    ))
    assert not cache.put("forged", forged, logits)
    assert not cache.put_prefixes(compiled.digest, "bundle", [*ids, 7], forged)
    restored, calls = runtime_for(compiled, engine)
    restored.install_continuation(contract)
    with pytest.raises(TransformerClientError, match="execution basis"):
        restored.restore(forged)
    owned = source.snapshot_for_response("response-A")
    with pytest.raises(TransformerClientError, match="execution basis"):
        restored.restore(owned)
    with pytest.raises(TransformerClientError, match="owner/source"):
        restored.restore_for_response(owned, "response-B")
    assert not calls.calls and restored.position == 0
    restored.restore_for_response(owned, "response-A")
    actual = restored.continue_ids([31, 44])[-1]
    assert np.all(np.isfinite(actual)) and len(calls.calls) == 4 * restored.layers
    continued = restored.snapshot()
    assert continued.state_basis.phase == "incremental"
    assert continued.state_basis.owner_response_id == "response-A"
    assert not cache.put("continued", continued, actual)
    assert not cache.put_prefixes(compiled.digest, "bundle", [*ids, 7, 31, 44], continued)
    # Local teacher-fold state is also incremental, not an implicit cache hit.
    source.install_continuation(contract)
    with pytest.raises(TransformerClientError, match="explicit prior-owner basis"):
        source.continue_ids([31])


def test_qualified_blocks_reuse_only_matching_geometry_and_keep_all_logits_exact(checkpoint):
    engine, compiled, contract, _ = checkpoint
    ids = np.random.default_rng(471).integers(2, 258, 64).tolist()
    source, _ = runtime_for(compiled, engine)
    logits = source.prepare_ids(ids)[1]
    saved = source.snapshot()
    payload = sum(row.key.nbytes + row.value.nbytes for row in saved.caches)
    cache = ExactPrefillCache(payload + logits.nbytes)
    assert cache.put_prefixes(compiled.digest, "bundle", ids, saved)
    assert cache.put(prefill_key(compiled.digest, "bundle", ids), saved, logits)
    assert cache.size_bytes == payload + logits.nbytes
    assert cache.block_count == 8 and cache.entry_count > cache.block_count
    assert cache.longest_prefix(compiled.digest, "bundle", ids + [7], layers=source.layers) is None
    branch = [*ids[:-1], (ids[-1] + 1) % 258]
    hit = cache.longest_prefix(compiled.digest, "bundle", branch, layers=source.layers)
    assert hit is not None and hit[0] == 63
    assert hit[1].state_basis.prefill_extent == 64
    fresh, _ = runtime_for(compiled, engine)
    expected = fresh.prepare_ids(branch)[1]
    continued, calls = runtime_for(compiled, engine)
    continued.install_continuation(contract)
    continued.restore(hit[1])
    actual = continued.continue_ids(branch[63:])[-1]
    np.testing.assert_array_equal(actual, expected)
    assert len(calls.calls) == 4 * source.layers
    assert continued.snapshot().state_basis.completed_prefill()
    # Domain-separated geometry cannot be relabeled or lose its seal in clipping.
    forged = dataclasses.replace(hit[1], state_basis=dataclasses.replace(
        hit[1].state_basis, prefill_extent=65
    ))
    assert not cache.put("forged-width", forged, actual)
    hit[1].caches[0].key.fill(0)
    independent = cache.longest_prefix(compiled.digest, "bundle", branch, layers=source.layers)
    np.testing.assert_array_equal(independent[1].caches[0].key, saved.caches[0].key[:63])
    cache.clear()
    assert cache.size_bytes == cache.entry_count == cache.block_count == 0


def test_client_requires_exact_provider_extension_and_never_downgrades(checkpoint):
    from pllm.runtime.client import RuntimeClient
    from pllm.runtime.masked_runtime import ModelError

    _, compiled, contract, _ = checkpoint
    client = object.__new__(RuntimeClient)
    expected = contract.handshake_spec()
    for invalid in (None, {}, {**expected, "digest": "a" * 64},
                    {**expected, "numeric_digest": "a" * 64},
                    {**expected, "token_bound": 65}, {**expected, "downgrade": True}):
        with pytest.raises(ModelError, match="did not admit"):
            client._admitted_decoder_continuation({"decoder_continuation": invalid}, compiled)
    assert client._admitted_decoder_continuation(
        {"decoder_continuation": expected}, compiled
    ).digest == contract.digest


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("PLLM_RUN_CACHED_CONTINUATION") != "1", reason="opt-in pinned offline checkpoint")
def test_pinned_cached_qwen_w8a8_three_branches():
    from huggingface_hub import hf_hub_download

    root = Path(hf_hub_download(
        "Qwen/Qwen2.5-0.5B-Instruct", "config.json", revision="7ae557604adf67be50417f59c2c2f167def9a775",
        cache_dir=os.getenv("PLLM_CACHED_HF_DIR", str(Path.home() / ".cache/huggingface/hub")),
        local_files_only=True,
    )).parent
    engine, compiled, contract, _ = bound_checkpoint(root, bound=128)
    initial, _ = runtime_for(compiled, engine)
    common = initial.encode_prompt("A short explanation of exact private inference begins with")
    initial.prepare_ids(common)
    saved = initial.snapshot()
    for text in (" public matrices.", " client-owned attention and fresh masks.", " one-use row tickets, then a longer suffix with changed content."):
        suffix = initial.tokenizer.encode(text, add_bos=False)
        fresh, _ = runtime_for(compiled, engine)
        reference = fresh.prepare_ids(common + suffix)[1]
        cached, calls = runtime_for(compiled, engine)
        cached.install_continuation(contract)
        cached.restore(saved)
        actual = cached.continue_ids(suffix)[-1]
        np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=0)
        assert np.argmax(actual) == np.argmax(reference)
        assert len(calls.calls) == cached.layers * 4
