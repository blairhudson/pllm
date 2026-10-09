"""Exact digit kernel, shape dispatch and ordinary prepared-role composition."""
import asyncio
from dataclasses import replace

import numpy as np
import pytest

import pllm
from pllm.native import capabilities
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.native import MaskedGEMM
from pllm.runtime.paged import PagedGEMM


available = capabilities().get("digit8_cpu", False)


def test_digit_component_preserves_defaults_and_binds_threshold():
    component = ModelAwareCorrections()
    assert dict(component.params) == {}
    selected = component.with_params(kernel="digit8", min_rows=32)
    assert dict(selected.params) == {"kernel": "digit8", "min_rows": 32}
    experiment = pllm.Experiment("digit", MaskedLinearCpu(pllm.Model.tiny(), preparation=selected),
        pllm.Deployment.local(root="local://digit-test"), pllm.ExecutionBudget(1, 64, 2))
    from pllm.profiles import resolve_runtime_composition
    assert resolve_runtime_composition(experiment.pipeline).preparation_kernel == "digit8"
    experiment.resolve()
    assert pllm.Experiment.from_spec(experiment.to_spec()).configuration_digest() == experiment.configuration_digest()
    with pytest.raises(ValueError):
        ModelAwareCorrections(min_rows=8)


@pytest.mark.skipif(not available, reason="requires Apple Silicon CPU Accelerate")
@pytest.mark.parametrize("storage", ["resident", "paged"])
@pytest.mark.parametrize("bits", [16, 24, 32])
def test_digit_native_integer_oracle_and_source_isolation(tmp_path, storage, bits):
    rng = np.random.default_rng(802)
    weights = rng.integers(-128, 128, (1031, 1024), dtype=np.int8)
    x = rng.integers(0, 1 << bits, (3, 1024), dtype=np.uint32)
    x[0, :6] = np.array([0, 127, 128, 255, (1 << bits) - 1, 1 << (bits - 1)], np.uint32)
    expected = (x.astype(np.int64) @ weights.astype(np.int64).T) % (1 << bits)
    if storage == "paged":
        path = tmp_path / "weights"
        digest = PagedGEMM.export(weights, path)
        matrix = PagedGEMM(path, digest)
    else:
        matrix = MaskedGEMM(threads=1).compile(weights)
    weights.fill(0)
    try:
        np.testing.assert_array_equal(matrix.digit8(x, bits), expected)
        with pytest.raises(ValueError):
            matrix.digit8(x, 17)
    finally:
        if storage == "paged":
            matrix.close()


@pytest.mark.skipif(not available, reason="requires Apple Silicon CPU Accelerate")
@pytest.mark.parametrize("storage", ["resident", "paged"])
def test_digit_preparation_frames_and_small_batch_fallback(tmp_path, monkeypatch, storage):
    from pllm.model_loader import resolve_model
    from pllm.runtime.preparation_protocol import PreparationRequest
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    manifest = resolve_model(pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model")))).manifest
    engines = [MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8,
        prepared_output_encoding="row_residues", weight_residency="provider", weight_storage=storage,
        preparation_kernel=kernel, preparation_min_rows=2) for kernel in ("native", "digit8")]
    try:
        for engine in engines:
            asyncio.run(engine.load(manifest))
        for sid in engines[0].seeded_stage_ids(manifest.id):
            stage = engines[0].models[manifest.id].stages[sid]
            profile = stage.seeded_profile
            for rows in (1, 3, 4097):
                # The residue codec has its own 4096-row ceiling. Exercise the
                # larger kernel fallback with the existing raw frame contract.
                for engine in engines:
                    engine.prepared_output_encoding = "row_residues" if rows <= 4096 else "raw"
                request = PreparationRequest(attempt_id="db" * 16, session_id="digit", model=manifest.id,
                    body_fingerprint=engines[0].models[manifest.id].manifest.metadata["body_fingerprint"],
                    stage_id=sid, weight_digest=stage.weight_digest, rows=rows,
                    in_features=stage.spec.in_features, out_features=stage.spec.out_features,
                    seed=b"d" * 32, weight_bits=8, activation_bits=8,
                    signed_output_bound=profile.signed_output_bound, ring=profile.ring,
                    modulus=profile.modulus, wire_bits=profile.wire_bits)
                calls = []
                matrix = engines[1].models[manifest.id].stages[sid].compiled_weight
                original = type(matrix).digit8
                def capture(self, *args):
                    calls.append(args[0].shape[0])
                    return original(self, *args)
                with monkeypatch.context() as patch:
                    patch.setattr(type(matrix), "digit8", capture)
                    output = [asyncio.run(e.prepare_seeded_stage(request)) for e in engines]
                assert replace(output[0], server_ns=0).pack() == replace(output[1], server_ns=0).pack()
                assert bool(calls) == (2 <= rows <= 4096)
    finally:
        for engine in engines:
            if manifest.id in engine.models:
                asyncio.run(engine.unload(manifest.id))


@pytest.mark.integration
@pytest.mark.skipif(not available, reason="requires Apple Silicon CPU Accelerate")
def test_digit_real_child_roles_sdk_logits_kv_and_gateway(tmp_path, monkeypatch):
    from pllm.runtime.servers import build_roles
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.protocols import ClientBundleTransport, MaskedLinear
    from fastapi.testclient import TestClient

    source = pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)))
    observed, results = [], []
    original = SemanticDecoderRuntime._forward_phase
    def capture(runtime, *args, **kwargs):
        logits = original(runtime, *args, **kwargs)
        observed.append((logits.copy(), [(c.key[:c.length].copy(), c.value[:c.length].copy()) for c in runtime.caches]))
        return logits
    monkeypatch.setattr(SemanticDecoderRuntime, "_forward_phase", capture)
    for kernel in ("native", "digit8"):
        pipeline = MaskedLinearCpu(source, preparation=ModelAwareCorrections(storage="paged", kernel=kernel),
            inventory=PreparedInventory(refill="on-demand", allocation="demand"),
            linear=MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues"),
            delivery=ClientBundleTransport("artifacts", storage="paged", compression="zlib"))
        if kernel == "digit8":
            pipeline = pipeline.with_params(preparation__min_rows=1)
        experiment = pllm.Experiment(kernel, pipeline, pllm.Deployment.local(root=str(tmp_path / kernel)),
                                     pllm.ExecutionBudget(2, 64, 2))
        with build_roles(experiment, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / f"cache-{kernel}") as client:
            response = client.responses.create(input="A", max_output_tokens=2, temperature=0)
            audit = client.privacy_audit
            assert audit.plaintext_prompt_bytes_sent == audit.plaintext_token_ids_sent == 0
            results.append((response.output_text, observed.copy()))
            observed.clear()
            with TestClient(roles.gateway_app(local_api_key="digit-test")) as http:
                reply = http.post("/v1/responses", headers={"Authorization": "Bearer digit-test"},
                    json={"model": response.model, "input": "A", "max_output_tokens": 2, "temperature": 0})
                assert reply.status_code == 200, reply.text
            observed.clear()
    assert results[0][0] == results[1][0]
    for (a, akv), (b, bkv) in zip(results[0][1], results[1][1], strict=True):
        np.testing.assert_array_equal(a, b)
        for pair, other in zip(akv, bkv, strict=True):
            for x, y in zip(pair, other, strict=True):
                np.testing.assert_array_equal(x, y)
