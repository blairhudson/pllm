"""Promoted options execute through ordinary role-backed SDK Experiments."""
import asyncio
from dataclasses import replace

import numpy as np
import pytest

import pllm
from pllm.model_loader import resolve_model
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.runtime.paged import PagedGEMM
from pllm.runtime.preparation_protocol import PreparationRequest, SessionAuthorization
from pllm.runtime.protocol import ProtocolError
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def test_paged_preparation_corrections_and_snapshot_lifetime(tmp_path):
    source = pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model")), model_id="paged-prep")
    manifest = resolve_model(source).manifest
    engines = [MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8,
                prepared_output_encoding="row_residues", weight_residency="provider", weight_storage=storage)
               for storage in ("resident", "paged")]
    handles = []
    try:
        for engine in engines:
            asyncio.run(engine.load(manifest))
        for sid in engines[0].seeded_stage_ids(manifest.id):
            stages = [e.models[manifest.id].stages[sid] for e in engines]
            assert stages[0].metadata.pack() == stages[1].metadata.pack()
            assert stages[1].quantized_weight is None
            assert isinstance(stages[1].compiled_weight, PagedGEMM)
            handles.append(stages[1].compiled_weight)
            stage = stages[0]
            profile = stage.seeded_profile
            request = PreparationRequest(attempt_id="ac" * 16, session_id="paged-session", model=manifest.id,
                body_fingerprint=engines[0].models[manifest.id].manifest.metadata["body_fingerprint"],
                stage_id=sid, weight_digest=stage.weight_digest, rows=3,
                in_features=stage.spec.in_features, out_features=stage.spec.out_features,
                seed=b"k" * 32, weight_bits=8, activation_bits=8, signed_output_bound=profile.signed_output_bound,
                ring=profile.ring, modulus=profile.modulus, wire_bits=profile.wire_bits)
            outputs = [asyncio.run(e.prepare_seeded_stage(request)) for e in engines]
            assert replace(outputs[0], server_ns=0).pack() == replace(outputs[1], server_ns=0).pack()
    finally:
        for engine in engines:
            if manifest.id in engine.models:
                asyncio.run(engine.unload(manifest.id))
    for handle in handles:
        with pytest.raises(ValueError):
            handle.wrap32(np.zeros((1, handle.shape[1]), np.uint32))


def test_stage_authorization_binds_exact_demand():
    auth = SessionAuthorization("demand", "model", "b" * 64, "c" * 64, 8, 8, 9, 7,
                                ("a", "b"), stage_rows=(7, 2))
    assert SessionAuthorization.unpack(auth.pack()) == auth
    assert auth.rows_for("b") == 2
    for counts in ((7,), (7, 0), (8, 1), (True, 8)):
        with pytest.raises(ProtocolError):
            replace(auth, stage_rows=counts).pack()


@pytest.mark.parametrize("cancel", [False, True])
def test_demand_rows_burn_once_without_uniform_overcount(monkeypatch, cancel):
    from collections import defaultdict
    from types import SimpleNamespace
    from pllm.runtime import transformer_client as tc
    monkeypatch.setattr(tc, "derive_online_attempt_id", lambda request, row: str(row))
    counts = defaultdict(int)
    def record(name, count):
        counts[name] += count
    stages = {name: tc.PreparedStageRows(SimpleNamespace(rows=n), np.ones((n, 2)), np.ones((n, 3)))
              for name, n in (("full", 5), ("terminal", 2))}
    inventory = tc.PreparedInventory("demand", 5, stages, demand=True, _audit=record)
    lease = inventory.reserve(5)
    lease.take("full", 3)
    lease.take("terminal", 1)
    assert inventory.available == 0
    if cancel:
        inventory.cancel()
        with pytest.raises(tc.TransformerClientError, match="cancelled"):
            lease.take("full", 1)
    else:
        with pytest.raises(tc.TransformerClientError, match="exhausted"):
            lease.take("terminal", 2)
    lease.close()
    lease.close()
    inventory.cancel()
    with pytest.raises(tc.TransformerClientError):
        inventory.reserve(1)
    assert {k.removeprefix("prepared_stage_rows_"): v for k, v in counts.items()} == {
        "issued": 7, "reserved": 7, "claimed": 4, "burned": 3, "discarded": 0}


@pytest.mark.integration
def test_demand_authorization_downgrade_cancels_before_issuance(tmp_path, monkeypatch):
    import httpx
    from pllm.runtime.masked_runtime import ModelError

    source = pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)),
                             model_id="demand-ack")
    pipeline = MaskedLinearCpu(source, linear=MaskedLinear(prefill_pruning="terminal"),
        inventory=PreparedInventory(refill="on-demand", allocation="demand"))
    experiment = pllm.Experiment("demand-ack", pipeline, pllm.Deployment.local(root=str(tmp_path)),
                                pllm.ExecutionBudget(2, 64, 3))
    with build_roles(experiment, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        original = client._core.http.post
        canceled = []
        def downgrade(path, **kwargs):
            response = original(path, **kwargs)
            if path == "/v1/runtime/inventories" and response.is_success:
                value = response.json()
                value["preparation_authorization"].pop("stage_rows")
                return httpx.Response(response.status_code, json=value, request=response.request)
            if path.endswith("/cancel"):
                canceled.append(path)
            return response
        with monkeypatch.context() as patch:
            patch.setattr(client._core.http, "post", downgrade)
            with pytest.raises(ModelError, match="acknowledge exact preparation demand"):
                client.prepare_response("A", 3)
        assert canceled
        assert client.privacy_audit.preparation_rows == client.privacy_audit.inference_stage_calls == 0
        client.prepare_response("A", 3)
        stream = client.responses.create(input="A", max_output_tokens=3, temperature=0, stream=True)
        for event in stream:
            if event.type == "response.output_text.delta":
                break
        stream.close()
        audit = client.privacy_audit
        assert audit.prepared_stage_rows_burned > 0
        assert audit.prepared_stage_rows_issued == audit.prepared_stage_rows_claimed + audit.prepared_stage_rows_burned
        assert client.responses.create(input="A", max_output_tokens=3, temperature=0).usage.output_tokens == 3


@pytest.mark.integration
def test_demand_paged_preparation_through_sdk(tmp_path, monkeypatch):
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    source = pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)),
                             model_id="incremental")
    observed = []
    original = SemanticDecoderRuntime._forward_phase
    def capture(runtime, *args, **kwargs):
        result = original(runtime, *args, **kwargs)
        observed.append((result.copy(), [(c.key[:c.length].copy(), c.value[:c.length].copy()) for c in runtime.caches]))
        return result
    monkeypatch.setattr(SemanticDecoderRuntime, "_forward_phase", capture)
    results = []
    for allocation, storage in (("uniform", "resident"), ("demand", "resident"), ("demand", "paged")):
        pipeline = MaskedLinearCpu(source, linear=MaskedLinear(prefill_pruning="terminal",
            request_encoding="stage_packed", output_encoding="row_residues"),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
            preparation=ModelAwareCorrections(storage=storage),
            inventory=PreparedInventory(refill="on-demand", allocation=allocation),
            delivery=ClientBundleTransport("artifacts", compression="zlib", storage="paged"))
        experiment = pllm.Experiment("incremental", pipeline, pllm.Deployment.local(root=str(tmp_path / f"{allocation}-{storage}")),
                                    pllm.ExecutionBudget(max_input_tokens=64, max_new_tokens=3, requests=2))
        assert pllm.Experiment.from_spec(experiment.to_spec()).configuration_digest() == experiment.configuration_digest()
        with build_roles(experiment, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / f"cache-{allocation}-{storage}") as client:
            client.prepare_response("A", 3)
            response = client.responses.create(input="A", max_output_tokens=3, temperature=0)
            audit = client.privacy_audit.to_dict()
            results.append((response.output_text, observed.copy(), audit))
            observed.clear()
            assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
    for candidate in results[1:]:
        assert candidate[0] == results[0][0]
        assert candidate[2]["prepared_stage_rows_issued"] < results[0][2]["prepared_stage_rows_issued"]
        assert candidate[2]["correction_push_bytes"] < results[0][2]["correction_push_bytes"]
        for (logits, states), (expected, before) in zip(candidate[1], results[0][1], strict=True):
            np.testing.assert_array_equal(logits, expected)
            for pair, old in zip(states, before, strict=True):
                for a, b in zip(pair, old, strict=True):
                    np.testing.assert_array_equal(a, b)
