"""Combined serving choices preserve complete state and one-use retirement."""

from __future__ import annotations

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.native import capabilities
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles
from pllm.runtime.mlp_tiling import MlpTiling
from pllm.runtime.model_binding import CompiledRuntimeModel
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.semantic_stages import scheduled_prepared_phase_rows
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import TransformerClientError
from pllm.schedulers import MlpPrefillTiling
from pllm.state import ClientPrefixReuse


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not capabilities().get("digit8_cpu", False),
        reason="combined digit8 execution requires Apple Silicon Accelerate",
    ),
]

_VARIANTS = (
    ("control", False, False, False),
    ("tiling", True, False, False),
    ("digit8", False, True, False),
    ("both", True, True, False),
    ("both-placement", True, True, True),
)
_PROMPT = "abcde " * 14


def _experiment(source, root, variant):
    name, tiled, digit8, placed = variant
    pipeline = MaskedLinearCpu(
        source,
        kernels=Cpu(threads=1),
        quantization=SymmetricPerRow(
            weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"
        ),
        preparation=ModelAwareCorrections(
            storage="paged", **({"kernel": "digit8", "min_rows": 64} if digit8 else {})
        ),
        inventory=PreparedInventory(
            allocation="demand", refill="on-demand", stage_window=4,
            stage_window_bytes=64 << 20,
        ),
        linear=MaskedLinear(
            request_encoding="stage_packed", output_encoding="row_residues",
            prefill_pruning="terminal", prefill_chunk_rows=16,
        ),
        delivery=ClientBundleTransport(
            "artifacts", storage="paged", compression="zlib", batch_objects=64
        ),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=128),
    )
    if tiled:
        pipeline = pipeline.with_params(scheduler=MlpPrefillTiling(tile_rows=64))
    if placed:
        # A local gate/up group removes one tiled region; an independent local
        # attention group exercises mixed ownership inside the remaining layer.
        pipeline = pipeline.with_params(placement=ClientLinearRoles(
            layer_roles=((0, "mlp_gate_up"), (1, "attention_output"))
        ))
    return Experiment(
        name, pipeline, Deployment.local(root=str(root)),
        ExecutionBudget(requests=3, max_input_tokens=128, max_new_tokens=3),
    )


def _state_words(runtime, logits):
    return (
        logits.shape, logits.tobytes(),
        tuple((cache.length, cache.key[:cache.length].tobytes(),
               cache.value[:cache.length].tobytes()) for cache in runtime.caches),
    )


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_five_way_cold_repeat_state_and_gateway(tmp_path, monkeypatch, family):
    from starlette.testclient import TestClient

    source = Model.path(str(create_tiny_llama_checkpoint(
        tmp_path / "source", model_type=family, num_hidden_layers=3,
        qk_norm=family == "qwen3", with_qkv_bias=family == "qwen2",
    )), model_id="combination-test")
    observed, tiles, leases = [], [], []
    forward = SemanticDecoderRuntime._forward_phase
    bind = CompiledRuntimeModel.runtime
    phase_rows = CompiledRuntimeModel.prepared_stage_phase_rows

    def check_header_demand(compiled, prefill_rows, decode_steps, *, continuation=False):
        from pllm.configuration import Pipeline
        import json
        actual = phase_rows(compiled, prefill_rows, decode_steps, continuation=continuation)
        expected = scheduled_prepared_phase_rows(compiled._plan,
            Pipeline.from_spec(json.loads(compiled._canonical_composition)),
            prefill_rows, decode_steps, continuation=continuation)
        assert actual == expected
        return actual

    def capture(runtime, *args, **kwargs):
        logits = forward(runtime, *args, **kwargs)
        observed.append(_state_words(runtime, logits))
        if runtime._mlp_tiling is not None:
            owner = runtime._mlp_tiling
            tiles.append((owner.calls, owner.tiles, owner.peak_tile_rows))
        return logits

    def capture_owner(compiled, remote, **kwargs):
        runtime = bind(compiled, remote, **kwargs)
        if getattr(remote, "inventory", None) is not None:
            leases.append(remote.inventory)
        return runtime

    monkeypatch.setattr(SemanticDecoderRuntime, "_forward_phase", capture)
    monkeypatch.setattr(CompiledRuntimeModel, "runtime", capture_owner)
    monkeypatch.setattr(CompiledRuntimeModel, "prepared_stage_phase_rows", check_header_demand)
    results = []
    request = dict(input=_PROMPT, max_output_tokens=3, temperature=0)
    for variant in _VARIANTS:
        name, tiled, _digit8, placed = variant
        spec = _experiment(source, tmp_path / name, variant)
        pair = []
        with build_roles(spec, engine_threads=1) as roles:
            with roles.client(bundle_cache_dir=tmp_path / f"cache-{name}") as client:
                for warm in (False, True):
                    observed.clear()
                    tiles.clear()
                    response = client.responses.create(**request)
                    assert 64 < response.usage.input_tokens <= 128
                    assert response.usage.output_tokens == 3
                    pair.append((response.output_text, tuple(observed)))
                    if tiled and not warm:
                        region_count = 1 if placed else 2
                        assert max(tiles) == (region_count, region_count * 2, 64)
                    if tiled and warm:
                        assert all(value == (0, 0, 0) for value in tiles)
                audit = client.privacy_audit
                assert audit.prefill_cache_hits == 1
                assert audit.plaintext_prompt_bytes_sent == audit.plaintext_token_ids_sent == 0
                assert audit.preparation_requests_during_online == 0
                assert all(lease._closed and lease._owner._active == 0 for lease in leases)
            if placed:
                observed.clear()
                with TestClient(roles.gateway_app(local_api_key="combination-test")) as gateway:
                    reply = gateway.post(
                        "/v1/responses", headers={"Authorization": "Bearer combination-test"},
                        json={"model": roles.model_id, **request},
                    )
                    assert reply.status_code == 200, reply.text
                assert tuple(observed) == results[0][0][1]
        results.append(tuple(pair))
    assert all(result == results[0] for result in results[1:])
    assert len(results[0][0][1]) == 3
    assert len(results[0][1][1]) == 2


def test_combined_tile_failure_burns_reserved_material(tmp_path, monkeypatch):
    source = Model.path(str(create_tiny_llama_checkpoint(tmp_path / "source", num_hidden_layers=3)),
                        model_id="combination-failure")
    spec = _experiment(source, tmp_path / "roles", _VARIANTS[-1])
    leases = []
    bind = CompiledRuntimeModel.runtime
    tile = MlpTiling._tile

    def capture_owner(compiled, remote, **kwargs):
        runtime = bind(compiled, remote, **kwargs)
        if getattr(remote, "inventory", None) is not None:
            leases.append(remote.inventory)
        return runtime

    def fail_after_exchange(*args):
        tile(*args)
        raise RuntimeError("combined tile failure")

    monkeypatch.setattr(CompiledRuntimeModel, "runtime", capture_owner)
    monkeypatch.setattr(MlpTiling, "_tile", staticmethod(fail_after_exchange))
    with build_roles(spec, engine_threads=1) as roles:
        with roles.client(bundle_cache_dir=tmp_path / "cache") as client:
            with pytest.raises(RuntimeError, match="combined tile failure"):
                client.responses.create(input=_PROMPT, max_output_tokens=3, temperature=0)
            assert client.privacy_audit.preparation_requests_during_online == 0
    assert leases
    for lease in leases:
        assert lease._closed and lease._owner._active == 0
        assert lease._owner._stage_burned > 0
        with pytest.raises(TransformerClientError, match="closed"):
            lease.take(next(iter(lease.stages)), 1)
