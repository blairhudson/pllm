"""Native state admission and independent full-decoder retention controls."""
import asyncio
import dataclasses
import gc
import json
import weakref

import numpy as np
import pytest

from pllm import Model, _native, lower_model
from pllm.passes import KvCacheEviction
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.retained_execution import compile_retained_contract, retained_runtime
from pllm.runtime.retained_state import RetainedLayerCache
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, TransformerClientError
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def policy(keep=(3, 4)):
    return KvCacheEviction(static_keep=keep, observation_window=(1, 4), dynamic_keep=(1, 1), share_adjacent_layers=False)


@pytest.fixture(params=["qwen2", "qwen3", "llama"])
def decoder(tmp_path, request):
    family = request.param
    root = create_tiny_llama_checkpoint(tmp_path / family, model_type=family,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    manifest = load_hf_directory(root, model_id="retained-test")
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("retained-test"))
    pipeline = MaskedLinearCpu(Model("retained-test"), quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
    plan = lower_model((root / "config.json").read_bytes(), batch=1, max_input_tokens=16, max_new_tokens=4)
    compiled = compile_runtime_model(plan, bundle, composition=pipeline)

    def remote(stage_id, values):
        stage = engine.models["retained-test"].stages[stage_id]
        q = quantize_activation_per_row(values, bits=8)
        y = dequantize_matmul(stage.compiled_weight.clear(q.values), q.scales, stage.weight.scales,
            output_shape=q.original_shape[:-1] + (stage.spec.out_features,))
        return np.ascontiguousarray(y if stage.bias is None else y + stage.bias, dtype=np.float32)

    return compiled, remote


def test_full_retention_matches_every_logit_and_kv_bit(decoder):
    compiled, remote = decoder
    dense = compiled.runtime(remote)
    retained = retained_runtime(compiled, remote, policy=policy((1, 1)))
    assert retained._state_binding_digest != dense._state_binding_digest
    tokens = [0, 40, 2, 39, 6, 88, 12, 11, 21, 20, 19]
    _, a, _ = dense.prepare_ids(tokens)
    _, b, _ = retained.prepare_ids(tokens)
    for step in range(4):
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        for full, sparse in zip(dense.caches, retained.caches, strict=True):
            assert type(sparse) is RetainedLayerCache
            k, v, positions = sparse.gather()
            np.testing.assert_array_equal(positions, np.arange(len(tokens) + step))
            np.testing.assert_array_equal(k.view(np.uint32), full.key[:full.length].view(np.uint32))
            np.testing.assert_array_equal(v.view(np.uint32), full.value[:full.length].view(np.uint32))
        if step < 3:
            token = int(np.argmax(a))
            a, b = dense.forward_ids([token])[-1], retained.forward_ids([token])[-1]
    with pytest.raises(TransformerClientError, match="response bound"):
        retained.forward_ids([0])
    assert retained._retained.allocated_bytes == 0


def test_static_decoder_matches_independent_dense_view_oracle_and_releases_state(decoder):
    compiled, remote = decoder
    oracle = compiled.runtime(remote)
    sparse = retained_runtime(compiled, remote, policy=policy((1, 2)))
    original = oracle._local
    selected, dense_refs = {}, []

    def local(op, values, *state):
        kind, layer = op["operator"], op.get("layer")
        if kind in {"attention_scores", "attention_values"} and oracle.position:
            values = dict(values)
            positions = np.r_[selected[layer], np.arange(12, oracle.position + 1)]
            values[op["inputs"][1]] = values[op["inputs"][1]][:, :, positions, :].copy()
        result = original(op, values, *state)
        if kind == "softmax" and not oracle.position:
            scores = result[0, :, -3:, :].mean(axis=(0, 1))
            selected[layer] = sorted(sorted(range(9), key=lambda i: (-float(scores[i]), i))[:3] + [9, 10, 11])
        return result

    oracle._local = local
    sparse_local = sparse._local

    def watch(op, *args):
        result = sparse_local(op, *args)
        if op["operator"] == "attention_values" and not sparse.position:
            cache = sparse.caches[op["layer"]]
            dense_refs.extend([weakref.ref(cache.key), weakref.ref(cache.value)])
        return result

    sparse._local = watch
    tokens = list(range(12))
    _, a, _ = oracle.prepare_ids(tokens)
    _, b, _ = sparse.prepare_ids(tokens)
    gc.collect()
    assert all(ref() is None for ref in dense_refs)
    for layer, cache in enumerate(sparse.caches):
        np.testing.assert_array_equal(cache.gather()[2], selected[layer])
        assert cache.length == 6 and cache.capacity == 9
    for step in range(4):
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        for layer, cache in enumerate(sparse.caches):
            k, v, p = cache.gather()
            np.testing.assert_array_equal(k, oracle.caches[layer].key[p])
            np.testing.assert_array_equal(v, oracle.caches[layer].value[p])
        if step < 3:
            token = int(np.argmax(a))
            a, b = oracle.forward_ids([token])[-1], sparse.forward_ids([token])[-1]


def test_native_contract_binds_policy_geometry_and_workspace_before_execution(decoder):
    compiled, _ = decoder
    full, sparse = [compile_retained_contract(compiled, policy(keep)) for keep in [(1, 1), (1, 2)]]
    assert full.digest != sparse.digest
    doc = sparse.to_dict()
    assert doc["retained_payload_bound_bytes"] < full.to_dict()["retained_payload_bound_bytes"]
    assert doc["owned_array_bound_bytes"] == sum(doc[k] for k in (
        "retained_payload_bound_bytes", "dense_prefill_payload_bound_bytes", "conversion_export_bound_bytes", "workspace_bound_bytes"))
    with pytest.raises(ValueError, match="budget"):
        compile_retained_contract(compiled, policy(), memory_bytes=1)
    doc["layers"][0]["decode"]["mask"] = "forged"
    with pytest.raises(ValueError, match="mismatch"):
        dataclasses.replace(sparse, _canonical=json.dumps(doc).encode())
    with pytest.raises(TypeError, match="immutable bytes"):
        dataclasses.replace(sparse, _canonical=bytearray(sparse._canonical))
    for invalid in [KvCacheEviction(), policy().with_params(share_adjacent_layers=True)]:
        with pytest.raises(ValueError, match="static"):
            compile_retained_contract(compiled, invalid)
    graph = json.loads(sparse._plan)
    graph["decode"]["state_inputs"][0]["kind"] = "recurrent"
    with pytest.raises(ValueError):
        _native._retained_decoder_contract(json.dumps(graph).encode(), sparse._composition, sparse._policy, 2 << 30)


def test_state_failure_burns_all_owners_and_blocks_full_kv_interfaces(decoder):
    compiled, remote = decoder
    sparse = retained_runtime(compiled, remote, policy=policy())
    sparse.prepare_ids(list(range(12)))
    owners = list(sparse.caches)
    with pytest.raises(TransformerClientError, match="snapshot"):
        sparse.snapshot()
    dense = compiled.runtime(remote)
    dense.prepare_ids(list(range(12)))
    with pytest.raises(TransformerClientError, match="snapshot"):
        sparse.restore(dense.snapshot())
    from pllm.configuration import Pipeline
    pipeline = Pipeline.from_spec(json.loads(compiled._canonical_composition))
    with pytest.raises(TransformerClientError, match="continuation"):
        sparse.install_continuation(compiled._plan.continuation_schedule(pipeline))
    calls = 0

    def fail(stage, values):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("partial-layer failure")
        return remote(stage, values)

    sparse.remote = fail
    with pytest.raises(RuntimeError, match="partial-layer"):
        sparse.forward_ids([3])
    assert all(owner.allocated_bytes == 0 for owner in owners)
    with pytest.raises(TransformerClientError, match="burned"):
        sparse.forward_ids([3])


def test_replacing_response_owned_state_rejects_before_remote_work(decoder):
    compiled, remote = decoder
    a, b = [retained_runtime(compiled, remote, policy=policy()) for _ in range(2)]
    a.prepare_ids(list(range(12)))
    b.prepare_ids(list(reversed(range(12))))
    a.caches[0] = b.caches[0]
    a.remote = lambda *_: pytest.fail("must reject state before a stage call")
    with pytest.raises(TransformerClientError, match="owner/position"):
        a.forward_ids([3])
    assert b._retained.allocated_bytes > 0


@pytest.mark.parametrize("keep", [(1, 1), (1, 2)])
def test_pipeline_retention_matches_bound_reference_for_all_logits_and_kv(decoder, keep):
    from pllm.configuration import Pipeline
    from pllm.state import ClientLocalKv
    compiled, remote = decoder
    pipeline = Pipeline.from_spec(json.loads(compiled._canonical_composition))
    selected = Pipeline(pipeline.model, dict(pipeline.components) | {"state": ClientLocalKv(retention=policy(keep))})
    candidate = compile_runtime_model(compiled._plan, compiled._bundle, composition=selected)
    reference = retained_runtime(compiled, remote, policy=policy(keep))
    actual = candidate.runtime(remote)
    assert actual._retained is not None and candidate.runtime_schedule_digest != compiled.runtime_schedule_digest
    _, left, _ = reference.prepare_ids(list(range(12)))
    _, right, _ = actual.prepare_ids(list(range(12)))
    for step in range(3):
        np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
        for a, b in zip(reference.caches, actual.caches, strict=True):
            for x, y in zip(a.gather(), b.gather(), strict=True):
                np.testing.assert_array_equal(x.view(np.uint32), y.view(np.uint32))
        if step < 2:
            token = int(np.argmax(left))
            left, right = reference.forward_ids([token])[-1], actual.forward_ids([token])[-1]
