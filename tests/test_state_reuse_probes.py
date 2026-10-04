from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.metrics import GeneratedStateReuseProbe, StateCompatibilityProbe
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.protocols import ClientBundleTransport
from pllm.model_loader import resolve_model
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine
from pllm.state import ClientPrefixReuse


def fixture(root, family="qwen2"):
    root = create_tiny_llama_checkpoint(root, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, head_dim=8, model_type=family, with_qkv_bias=family == "qwen2",
        qk_norm=family == "qwen3")
    model_id = "state-probe"
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1)
    asyncio.run(engine.load(resolve_model(Model.path(str(root), model_id=model_id)).manifest))
    loaded = engine.models[model_id]
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    plan = lower_model(json.loads((root / "config.json").read_text()), batch=1,
        max_input_tokens=32, max_new_tokens=8)
    pipeline = MaskedLinearCpu(Model(model_id),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"))
    def remote(stage_id, value):
        stage = loaded.stages[stage_id]
        q = quantize_activation_per_row(value, bits=8)
        output = dequantize_matmul(stage.compiled_weight.clear(q.values), q.scales, stage.weight.scales,
            output_shape=q.original_shape[:-1] + (stage.spec.out_features,))
        return np.ascontiguousarray(output if stage.bias is None else output + stage.bias, dtype=np.float32)
    remote.weight_matrices = {key: stage.weight.values for key, stage in loaded.stages.items()}
    return compile_runtime_model(plan, bundle, composition=pipeline), remote, plan, bundle, pipeline


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_generated_probe_checks_every_kv_and_leaves_incremental_basis(tmp_path, family):
    compiled, remote, *_ = fixture(tmp_path / family, family)
    report = GeneratedStateReuseProbe(steps=4).run(compiled, remote, [3, 8, 7])
    assert report["numeric_gate_passed"]
    assert all(row["original_basis_is_incremental"] for row in report["samples"])
    assert report["pending_generated_tokens_excluded"] == 1
    assert not report["live_cache_promotion"]
    assert "token_ids" not in report
    with pytest.raises(ValueError, match="bound"):
        GeneratedStateReuseProbe(steps=32).run(compiled, remote, [3])
    with pytest.raises(ValueError):
        GeneratedStateReuseProbe(steps=True)


def test_state_compatibility_retains_numeric_and_weight_contracts(tmp_path):
    left, _, plan, bundle, pipeline = fixture(tmp_path / "source")
    right = compile_runtime_model(plan, bundle, composition=pipeline.with_params(
        delivery=ClientBundleTransport("artifacts", compression="zlib")))
    result = StateCompatibilityProbe().compare(left, right)
    assert result["contract_equal"]
    assert left.digest != right.digest
    assert not result["live_state_transfer_authorized"]
    old_numeric = compile_runtime_model(plan, bundle, composition=pipeline.with_params(
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8)))
    assert "quantization" in StateCompatibilityProbe().compare(left, old_numeric)["different_contract_fields"]


def test_checked_state_bridge_reexecutes_exact_suffix_and_rejects_owned_or_forged_state(tmp_path):
    from dataclasses import replace
    from pllm.runtime.model_binding import RuntimeBindingError
    from pllm.metrics.state_reuse import _same_state

    left, remote, plan, bundle, pipeline = fixture(tmp_path / "bridge")
    right = compile_runtime_model(plan, bundle, composition=pipeline.with_params(
        delivery=ClientBundleTransport("artifacts", compression="zlib")))
    a = left.runtime(remote)
    a.prepare_ids([3, 8, 7])
    snapshot = a.snapshot()
    b = right.runtime(remote)
    b.restore(left.transfer_snapshot(snapshot, right))
    b.install_continuation(plan.continuation_schedule(pipeline.with_params(
        delivery=ClientBundleTransport("artifacts", compression="zlib"))))
    actual = b.continue_ids([9, 10])[-1]
    fresh = right.runtime(remote)
    expected = fresh.prepare_ids([3, 8, 7, 9, 10])[1]
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert _same_state(b.snapshot(), fresh.snapshot())
    assert not np.shares_memory(snapshot.caches[0].key, b.caches[0].key)
    with pytest.raises(RuntimeBindingError, match="qualified"):
        left.transfer_snapshot(a.snapshot_for_response("owner"), right)
    snapshot.state_basis = replace(snapshot.state_basis, source_binding_digest="f" * 64)
    with pytest.raises(RuntimeBindingError, match="qualified"):
        left.transfer_snapshot(snapshot, right)


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_native_generated_qualification_preserves_state_and_separate_lineage(tmp_path, family):
    from dataclasses import replace
    from pllm.runtime.prefill_cache import ExactPrefillCache, prefill_key
    from pllm.runtime.transformer_client import TransformerClientError

    old, remote, plan, bundle, pipeline = fixture(tmp_path / family, family)
    assert "generated_prefix_canonical" not in plan.continuation_schedule(pipeline).to_dict()
    pipeline = pipeline.with_params(cache=ClientPrefixReuse(
        max_bytes=1 << 20, fixed_input_tokens=32, generated_prefixes=True))
    model = compile_runtime_model(plan, bundle, composition=pipeline)
    assert plan.continuation_schedule(pipeline).to_dict()["generated_prefix_canonical"]
    runtime = model.runtime(remote)
    ids = [3, 8, 7]
    logits = runtime.prepare_ids(ids)[1]
    for _ in range(4):
        ids.append(int(np.argmax(logits)))
        logits = runtime.decode_step(ids[-1], runtime.caches)[0]
    snapshot = runtime.canonical_generated_snapshot()
    assert runtime.snapshot().state_basis.phase == "incremental"
    assert snapshot.state_basis.phase == "canonical_incremental"
    assert not snapshot.state_basis.completed_prefill()
    legacy = ExactPrefillCache(1 << 20, causal_reduction="prefix_f32")
    assert not legacy.put("key", snapshot, logits)
    cache = ExactPrefillCache(1 << 20, causal_reduction="prefix_f32", generated_prefixes=True)
    key = prefill_key(model.digest, "bundle", ids, causal_reduction="prefix_f32")
    assert cache.put(key, snapshot, logits)
    restored = cache.get(key, position=len(ids), layers=2)[0]
    next_runtime = model.runtime(remote)
    next_runtime.restore(restored)
    owned = next_runtime.snapshot_for_response("rsp-canonical-hit")
    assert owned.state_basis.phase == "incremental"
    owned_runtime = model.runtime(remote)
    owned_runtime.restore_for_response(owned, "rsp-canonical-hit")
    from pllm.metrics.state_reuse import _same_state
    assert _same_state(next_runtime.snapshot(), owned_runtime.snapshot())
    actual = next_runtime.continue_ids([9, 10])[-1]
    fresh = model.runtime(remote)
    expected = fresh.prepare_ids([*ids, 9, 10])[1]
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    from pllm.metrics.state_reuse import _same_state
    assert _same_state(next_runtime.snapshot(), fresh.snapshot())
    with pytest.raises(TransformerClientError):
        old.runtime(remote).restore(snapshot)
    bad = runtime.snapshot()
    bad.state_basis = replace(snapshot.state_basis, native_phase_digest="f" * 64)
    with pytest.raises(TransformerClientError):
        model.runtime(remote).restore(bad)
    with pytest.raises(Exception, match="unsupported|canonical|runtime|composition|admitted"):
        bad_numeric = pipeline.with_params(quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
        plan.continuation_schedule(bad_numeric)
