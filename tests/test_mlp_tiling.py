"""Exact row-independent scheduling, bounded tensor lifetimes and role transport."""
import asyncio
import json
import weakref

import numpy as np
import pytest

from pllm import Model, _native, lower_model
from pllm.profiles import ClientOnlyCpu
from pllm.runtime.mlp_tiling import MlpTiling
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.transformer_client import ClientBundle, TransformerClientError
from pllm.runtime.transformer_engine import MaskedTransformerEngine
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.loaders import load_hf_directory


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("tile", [1, 7, 16])
def test_tiled_mlp_exact_logits_kv_and_intermediate_retirement(tmp_path, family, tile):
    path = create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")
    model_id = "tile-test"
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(path, model_id=model_id)))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id, placement="client"))
    plan = lower_model(json.loads((path / "config.json").read_text()), batch=1,
                       max_input_tokens=33, max_new_tokens=4)
    composition = ClientOnlyCpu(Model(model_id))
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    results = []
    for tiled in (False, True):
        executor = compiled.client_linear_executor(engine)
        runtime = compiled.runtime(executor)
        contract = MlpTiling(compiled, runtime, tile)
        if tiled:
            runtime._mlp_tiling = contract
        gate_ids = {s.stage_id for s in compiled.stage_bindings if s.role == "mlp_gate_up"}
        down_ids = {s.stage_id for s in compiled.stage_bindings if s.role == "mlp_down"}
        pending = []
        def execute(stage, value):
            if tiled and runtime.position == 0 and stage in gate_ids:
                assert value.shape[-2] <= tile
                assert all(ref() is None for ref in pending)
                pending.clear()
            if tiled and runtime.position == 0 and stage in down_ids:
                pending.append(weakref.ref(value))
            return executor(stage, value)
        runtime.remote = execute
        logits = [runtime._forward(np.asarray([0, 2, 3] * 11, dtype=np.int64)).tobytes()]
        for token in (5, 7, 9):
            logits.append(runtime.forward_ids([token]).tobytes())
        state = [(c.key[:c.length].tobytes(), c.value[:c.length].tobytes()) for c in runtime.caches]
        results.append((logits, state))
        if tiled:
            assert contract.calls == 2 and contract.peak_tile_rows == tile
            assert contract.tiles == 2 * ((33 + tile - 1) // tile)
            assert all(ref() is None for ref in pending)
    assert results[0] == results[1]
    for bad in (0, 257):
        with pytest.raises(ValueError):
            _native.decoder_mlp_tiling(plan.canonical_bytes(), composition.canonical_bytes(), bad)


@pytest.mark.integration
@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_mlp_tiles_preserve_separate_role_logits_state_and_burn(tmp_path, monkeypatch, family):
    from test_prepared_row_residues import experiment
    from pllm.runtime.model_binding import CompiledRuntimeModel
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    from pllm.runtime.servers import build_roles
    source = Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")), model_id="residue-test")
    original_runtime, original_forward = CompiledRuntimeModel.runtime, SemanticDecoderRuntime._forward
    results, observed, leases = [], [], []

    def bind(compiled, remote, **kwargs):
        runtime = original_runtime(compiled, remote, **kwargs)
        if tile:
            runtime._mlp_tiling = MlpTiling(compiled, runtime, tile)
            leases.append(remote.inventory)
        return runtime

    def forward(runtime, *args, **kwargs):
        result = original_forward(runtime, *args, **kwargs)
        observed.append((result.tobytes(), [(c.key[:c.length].tobytes(), c.value[:c.length].tobytes())
                                          for c in runtime.caches]))
        return result

    monkeypatch.setattr(CompiledRuntimeModel, "runtime", bind)
    monkeypatch.setattr(SemanticDecoderRuntime, "_forward", forward)
    for tile in (0, 7):
        exp = experiment(source, tmp_path / str(tile), "row_residues").with_params(
            pipeline__linear__request_encoding="stage_packed", pipeline__linear__prefill_chunk_rows=4)
        with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / f"cache-{tile}") as client:
            response = client.responses.create(input="abcde " * 8, max_output_tokens=3, temperature=0)
            results.append((response.output_text, observed.copy()))
            observed.clear()
    assert results[0] == results[1]
    original_tile = MlpTiling._tile
    def fail_after_first_tile(*args):
        original_tile(*args)
        raise RuntimeError("injected tile cancellation")
    monkeypatch.setattr(MlpTiling, "_tile", staticmethod(fail_after_first_tile))
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cancel-cache") as client:
        with pytest.raises(RuntimeError, match="injected tile cancellation"):
            client.responses.create(input="abcde " * 8, max_output_tokens=3, temperature=0)
    lease = leases[-1]
    assert lease._closed and lease._owner._active == 0 and lease._owner._stage_burned > 0
    with pytest.raises(TransformerClientError, match="closed"):
        lease.take(next(iter(lease.stages)), 1)
