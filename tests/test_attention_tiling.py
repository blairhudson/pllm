"""Complete native-bound decoder parity and tile/response failure retirement."""
import asyncio
import json
import weakref

import numpy as np
import pytest

from pllm import Model, _native, lower_model
from pllm.profiles import ClientOnlyCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.attention_tiling import AttentionTiling
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.transformer_client import ClientBundle, TransformerClientError
from pllm.runtime.transformer_engine import MaskedTransformerEngine
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.loaders import load_hf_directory
from pllm.schedulers import MlpPrefillTiling


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("profile", [None, "prefix_f32"])
@pytest.mark.parametrize("tile,heads,mlp", [(1, 1, False), (7, 1, True), (16, 2, True)])
def test_exact_generated_logits_kv_restore_and_tile_lifetime(tmp_path, monkeypatch, family, profile, tile, heads, mlp):
    path = create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(path, model_id="attention-test")))
    bundle = ClientBundle.unpack(engine.client_bundle("attention-test", placement="client"))
    plan = lower_model(json.loads((path / "config.json").read_text()), batch=1,
                       max_input_tokens=33, max_new_tokens=5)
    composition = ClientOnlyCpu(Model("attention-test"), quantization=SymmetricPerRow(causal_reduction=profile))
    if mlp:
        composition = composition.with_params(scheduler=MlpPrefillTiling(tile_rows=7))
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    executor = compiled.client_linear_executor(engine)
    results = []
    for enabled in (False, True):
        runtime = compiled.runtime(executor)
        tiling = AttentionTiling(compiled, runtime, tile, heads)
        if enabled:
            runtime._attention_tiling = tiling
        pending = []
        original_local = runtime._local
        def local(op, *args, **kwargs):
            if enabled and "attention_positions" in kwargs:
                if op["operator"] == "attention_scores":
                    assert all(ref() is None for ref in pending)
                    pending.clear()
                output = original_local(op, *args, **kwargs)
                pending.append(weakref.ref(output))
                return output
            return original_local(op, *args, **kwargs)
        monkeypatch.setattr(runtime, "_local", local)
        logits = [runtime._forward(np.asarray([0, 2, 3] * 11, np.int64)).tobytes()]
        prefill = runtime.snapshot()
        for token in (5, 7, 9):
            logits.append(runtime.forward_ids([token]).tobytes())
        state = [(c.key[:c.length].tobytes(), c.value[:c.length].tobytes()) for c in runtime.caches]
        restored = compiled.runtime(executor)
        if enabled:
            restored._attention_tiling = AttentionTiling(compiled, restored, tile, heads)
        restored.restore(prefill)
        for token in (5, 7, 9):
            restored.forward_ids([token])
        last = restored.forward_ids([11]).tobytes()
        state2 = [(c.key[:c.length].tobytes(), c.value[:c.length].tobytes()) for c in restored.caches]
        results.append((logits, state, last, state2))
        if enabled:
            assert tiling.calls == 2
            assert tiling.tiles == 2 * ((33 + tile - 1) // tile) * ((2 + heads - 1) // heads)
            assert tiling.peak_score_bytes <= 4 * heads * 2 * tile * 37
            assert all(ref() is None for ref in pending)
            assert restored._attention_tiling.calls == 0
    assert results[0] == results[1]


def test_native_bounds_and_research_target_identity(tmp_path):
    from pllm import Deployment, ExecutionBudget, Experiment
    from pllm.metrics import BenchmarkTarget
    from pllm.sources import _tiny_model_config
    from pllm.profiles import VerifiedMaskedLinearCpu

    pipeline = ClientOnlyCpu(Model.tiny())
    plan = lower_model(_tiny_model_config(), batch=1, max_input_tokens=16, max_new_tokens=2)
    for rows, heads in [(0, 1), (257, 1), (1, 0), (1, 65)]:
        with pytest.raises(ValueError):
            _native.decoder_attention_tiling(plan.canonical_bytes(), pipeline.canonical_bytes(), rows, heads)
    with pytest.raises(ValueError):
        _native.decoder_attention_tiling(plan.canonical_bytes(), VerifiedMaskedLinearCpu(Model.tiny()).canonical_bytes(), 16, 2)
    exp = Experiment("attention", pipeline, Deployment.local(root=str(tmp_path)), ExecutionBudget(1, 16, 2))
    control = BenchmarkTarget.response(exp, client_attention={"enabled": False, "query_rows": 16, "kv_heads": 2})
    candidate = BenchmarkTarget.response(exp, client_attention={"enabled": True, "query_rows": 16, "kv_heads": 2})
    assert control.digest != candidate.digest
    assert BenchmarkTarget.from_spec(candidate.to_spec()).digest == candidate.digest
    for selected in ({"enabled": 1, "query_rows": 16, "kv_heads": 2},
                     {"enabled": True, "query_rows": 0, "kv_heads": 2}):
        with pytest.raises(ValueError):
            BenchmarkTarget.response(exp, client_attention=selected)
    with pytest.raises(ValueError):
        BenchmarkTarget.response(exp, client_attention={"enabled": True, "query_rows": 16, "kv_heads": 2},
                                 client_mlp={"tile_rows": 7})


@pytest.mark.integration
@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_prepared_roles_gateway_exact_state_and_failure_burn(tmp_path, monkeypatch, family):
    from test_prepared_row_residues import experiment
    from pllm.runtime.model_binding import CompiledRuntimeModel
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    from pllm.runtime.servers import build_roles
    from starlette.testclient import TestClient
    from pllm.state import ClientPrefixReuse

    source = Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")), model_id="residue-test")
    original_runtime, original_forward = CompiledRuntimeModel.runtime, SemanticDecoderRuntime._forward
    results, observed, leases, runtimes = [], [], [], []
    enabled = False
    def bind(compiled, remote, **kwargs):
        runtime = original_runtime(compiled, remote, **kwargs)
        if enabled:
            runtime._attention_tiling = AttentionTiling(compiled, runtime, 7, 1)
            leases.append(remote.inventory)
            runtimes.append(runtime)
        return runtime
    def forward(runtime, *args, **kwargs):
        result = original_forward(runtime, *args, **kwargs)
        observed.append((result.tobytes(), [(c.key[:c.length].tobytes(), c.value[:c.length].tobytes())
                                          for c in runtime.caches]))
        return result
    monkeypatch.setattr(CompiledRuntimeModel, "runtime", bind)
    monkeypatch.setattr(SemanticDecoderRuntime, "_forward", forward)
    for enabled in (False, True):
        exp = experiment(source, tmp_path / str(enabled), "row_residues").with_params(
            pipeline__linear__request_encoding="stage_packed", pipeline__linear__prefill_chunk_rows=4,
            pipeline__cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=128),
            pipeline__scheduler=MlpPrefillTiling(tile_rows=7))
        with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / f"cache-{enabled}") as client:
            response = client.responses.create(input="abcde " * 8, max_output_tokens=3, temperature=0)
            results.append((response.output_text, observed.copy()))
            observed.clear()
            if enabled:
                with TestClient(roles.gateway_app(local_api_key="attention-gateway")) as gateway:
                    reply = gateway.post("/v1/responses", headers={"authorization": "Bearer attention-gateway"},
                        json={"model": "residue-test", "input": "abcde " * 8,
                              "max_output_tokens": 3, "temperature": 0})
                    assert reply.status_code == 200, reply.text
                    assert observed == results[-1][1]
                    observed.clear()
    assert results[0] == results[1]
    original_tile = AttentionTiling._tile
    def fail_after_tile(*args):
        original_tile(*args)
        raise RuntimeError("injected attention cancellation")
    monkeypatch.setattr(AttentionTiling, "_tile", staticmethod(fail_after_tile))
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cancel-cache") as client:
        with pytest.raises(RuntimeError, match="injected attention cancellation"):
            client.responses.create(input="abcde " * 8, max_output_tokens=3, temperature=0)
    lease = leases[-1]
    assert lease._closed and lease._owner._active == 0 and lease._owner._stage_burned > 0
    assert runtimes[-1]._attention_tiling.failed
    with pytest.raises(TransformerClientError, match="burned"):
        runtimes[-1].forward_ids([2])
