import pytest
import numpy as np

import pllm
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear, TwoOnlineOffsetLinear
from pllm.quantization import SymmetricPerRow
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.semantic_executor import SemanticDecoderRuntime

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("topology", ["prepared", "verified", "offset", "duplex", "verified-duplex", "raw-duplex"])
def test_wan_choices_compose_through_normal_sdk_and_preserve_outputs(tmp_path, monkeypatch, topology):
    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)
    source = pllm.Model.path(str(root), model_id="wan-tps-tiny")
    outputs, snapshots, current = [], [], []
    original_prefill, original_decode = SemanticDecoderRuntime.prepare_ids, SemanticDecoderRuntime.decode_step
    def capture(runtime, logits):
        current.append((logits.copy(), [(c.key[:c.length].copy(), c.value[:c.length].copy()) for c in runtime.caches]))
    def prefill(runtime, *args, **kwargs):
        result = original_prefill(runtime, *args, **kwargs)
        capture(runtime, result[1])
        return result
    def decode(runtime, *args, **kwargs):
        result = original_decode(runtime, *args, **kwargs)
        capture(runtime, result[0])
        return result
    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", prefill)
    monkeypatch.setattr(SemanticDecoderRuntime, "decode_step", decode)
    for enabled in (False, True):
        common = dict(quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
            delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64 if enabled else 1))
        if topology == "offset":
            pipeline = TwoOnlineOffsetCpu(source, **common, linear=TwoOnlineOffsetLinear(
                input_encoding="seeded", output_encoding="row_residues",
                dispatch="seed_first" if enabled else "sequential"))
        else:
            profile = VerifiedMaskedLinearCpu if topology.startswith("verified") else MaskedLinearCpu
            pipeline = profile(source, **common, linear=MaskedLinear(
                output_encoding="raw" if topology == "raw-duplex" else "row_residues",
                prefill_chunk_rows=4 if "duplex" in topology and enabled else 0),
                inventory=PreparedInventory("request-sized", rows=1, refill="on-demand", stage_window=4 if enabled else 1))
        experiment = pllm.Experiment("wan-tps", pipeline, pllm.Deployment.local(root=str(tmp_path)),
            pllm.ExecutionBudget(max_input_tokens=64, max_new_tokens=2, requests=1))
        with build_roles(experiment, engine_threads=1) as roles:
            with roles.client(bundle_cache_dir=tmp_path / f"cache-{enabled}") as client:
                response = client.responses.create(input="A", max_output_tokens=2, temperature=0)
                outputs.append((response.output_text, response.usage))
                snapshots.append(current.copy())
                current.clear()
                assert client._core.artifact_cache_stats.object_batch_requests > 0 if enabled else True
                audit = client.privacy_audit.to_dict()
                assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
                assert audit["inference_stage_calls"] > 0
    assert outputs[0] == outputs[1]
    assert len(snapshots[0]) == len(snapshots[1]) >= 2
    for (left, caches), (right, others) in zip(*snapshots, strict=True):
        np.testing.assert_array_equal(left, right)
        for pair, other in zip(caches, others, strict=True):
            for a, b in zip(pair, other, strict=True):
                np.testing.assert_array_equal(a, b)


def test_duplex_reordered_authenticated_reply_burns_session_before_retry(tmp_path, monkeypatch):
    from pllm.runtime.client import _Channel
    from pllm.runtime.protocol import ProtocolEnvelope, pack_envelope, unpack_envelope
    from pllm.runtime.security import derive_session_key
    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)
    source = pllm.Model.path(str(root), model_id="duplex-burn")
    exp = pllm.Experiment("duplex-burn", MaskedLinearCpu(source,
        linear=MaskedLinear(output_encoding="row_residues", prefill_chunk_rows=4),
        inventory=PreparedInventory("request-sized", rows=1, refill="on-demand", stage_window=2)),
        pllm.Deployment.local(root=str(tmp_path)), pllm.ExecutionBudget(requests=2, max_input_tokens=64, max_new_tokens=2))
    original = _Channel.exchange_many
    captured = []
    def reorder(channel, payloads):
        replies = original(channel, payloads)
        if not captured:
            assert len(replies) > 1
            captured.append((channel.session_id, unpack_envelope(payloads[0]), len(payloads)))
            replies[-1] = replies[0]
        return replies
    monkeypatch.setattr(_Channel, "exchange_many", reorder)
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        with pytest.raises(Exception, match="ordered session frame"):
            client.responses.create(input="abc", max_output_tokens=2, temperature=0)
        session, first, sequence = captured[0]
        fresh_frame = ProtocolEnvelope.create(request_id="after-cancel", session_id=session,
            model=first.model, kind=first.kind, sequence=sequence, payload=first.payload,
            metadata=first.metadata, key=derive_session_key(client._core.api_key, session))
        rejected = client._core.http.post(f"/v1/runtime/sessions/{session}/execute",
            headers=client._core.headers, content=pack_envelope(fresh_frame))
        assert rejected.status_code == 400 and "terminal" in rejected.text
        assert client.responses.create(input="abc", max_output_tokens=2, temperature=0).usage.output_tokens == 2


@pytest.mark.parametrize("offset,aggregate", [(False, False), (True, False), (False, True)])
def test_wan_choices_use_the_existing_gateway(tmp_path, offset, aggregate):
    from fastapi.testclient import TestClient
    source = pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)),
                             model_id="wan-gateway-tiny")
    common = dict(quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64))
    pipeline = (TwoOnlineOffsetCpu(source, **common, linear=TwoOnlineOffsetLinear(
        input_encoding="seeded", output_encoding="row_residues", dispatch="seed_first")) if offset else
        MaskedLinearCpu(source, **common,
            linear=MaskedLinear(output_encoding="row_residues", prefill_chunk_rows=4,
                request_encoding="compact" if aggregate else "raw",
                prefill_pruning="terminal" if aggregate else "none"),
            inventory=PreparedInventory("request-sized", rows=1, refill="on-demand", stage_window=4)))
    experiment = pllm.Experiment("wan-gateway", pipeline, pllm.Deployment.local(root=str(tmp_path)),
        pllm.ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=2))
    with build_roles(experiment, engine_threads=1) as roles:
        with TestClient(roles.gateway_app(local_api_key="wan-local")) as api:
            response = api.post("/v1/responses", headers={"Authorization": "Bearer wan-local"},
                json={"model": roles.model_id, "input": "A", "max_output_tokens": 2, "temperature": 0})
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["output_tokens"] == 2
