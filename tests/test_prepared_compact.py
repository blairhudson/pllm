"""Compact decode keeps authenticated context, exact numbers and burn boundaries."""
from dataclasses import replace

import msgpack
import numpy as np
import pytest

import pllm
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, VerifiedMaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.runtime.protocol import ProtocolError
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.runtime.stage_protocol import (
    PreparedStageBatchRequest, PreparedStageBatchResponse, prepared_stage_batch_rows,
)
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.state import ClientPrefixReuse


def test_single_row_frames_reject_old_namespace_and_bad_bounds():
    values = np.array([[0, 65535]], np.uint32)
    request = PreparedStageBatchRequest("ab" * 16, ("cd" * 16,), values, 16).pack()
    response = PreparedStageBatchResponse("ab" * 16, values, 16).pack()
    assert prepared_stage_batch_rows(request) == 1
    for body, unpack, old_magic in (
        (request, PreparedStageBatchRequest.unpack, b"PLLMPSB1"),
        (response, PreparedStageBatchResponse.unpack, b"PLLMPSR1"),
    ):
        for bad in (old_magic + body[8:], body[:8] + (2).to_bytes(4, "big") + body[12:], body[:-1]):
            with pytest.raises(ProtocolError):
                unpack(bad, max_rows=1, max_tensor_elements=2)
        with pytest.raises(ProtocolError, match="allocation"):
            unpack(body, max_rows=1, max_tensor_elements=1)
    duplicate = PreparedStageBatchRequest("ab" * 16, ("cd" * 16, "ef" * 16),
                                         np.repeat(values, 2, axis=0), 16).pack()
    fields = msgpack.unpackb(duplicate[12:])
    fields[1] = bytes.fromhex("cd" * 32)
    with pytest.raises(ProtocolError, match="unique"):
        PreparedStageBatchRequest.unpack(duplicate[:12] + msgpack.packb(fields, use_bin_type=True))


def experiment(source, root, encoding, *, verified=False, output="raw", chunks=0, pruning="none"):
    profile = VerifiedMaskedLinearCpu if verified else MaskedLinearCpu
    return pllm.Experiment("compact-test", profile(source,
        linear=MaskedLinear(request_encoding=encoding, output_encoding=output, prefill_chunk_rows=chunks,
                            prefill_pruning=pruning),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        inventory=PreparedInventory("request-sized", rows=1, refill="on-demand", stage_window=4),
        delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64, storage="paged"),
        cache=ClientPrefixReuse(fixed_input_tokens=64, max_bytes=1 << 20, generated_prefixes=True)),
        pllm.Deployment.local(root=str(root)),
        pllm.ExecutionBudget(max_input_tokens=64, max_new_tokens=3, requests=3))


@pytest.mark.integration
@pytest.mark.parametrize("verified,output,chunks,transport", [
    (False, "raw", 0, "http"), (False, "row_residues", 4, "websocket"),
    (True, "row_residues", 4, "websocket"),
])
def test_compact_composes_with_cache_paging_duplex_and_verification(
    tmp_path, monkeypatch, verified, output, chunks, transport,
):
    source = pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)),
                            model_id="compact-test")
    observed, reports = [], []
    original_prefill, original_decode = SemanticDecoderRuntime.prepare_ids, SemanticDecoderRuntime.decode_step

    def save(runtime, logits):
        observed.append((logits.copy(), [(c.key[:c.length].copy(), c.value[:c.length].copy())
                                       for c in runtime.caches]))

    def prefill(runtime, *args, **kwargs):
        result = original_prefill(runtime, *args, **kwargs)
        save(runtime, result[1])
        return result

    def decode(runtime, *args, **kwargs):
        result = original_decode(runtime, *args, **kwargs)
        save(runtime, result[0])
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", prefill)
    monkeypatch.setattr(SemanticDecoderRuntime, "decode_step", decode)
    for encoding, pruning in (("raw", "none"), ("compact", "none"), ("raw", "terminal"), ("compact", "terminal")):
        key = f"{encoding}-{pruning}"
        exp = experiment(source, tmp_path / key, encoding, verified=verified, output=output, chunks=chunks, pruning=pruning)
        with build_roles(exp, engine_threads=1) as roles, roles.client(
            bundle_cache_dir=tmp_path / f"cache-{key}", session_transport=transport,
        ) as client:
            outputs = []
            for _ in range(2):
                response = client.responses.create(input="A", max_output_tokens=3, temperature=0)
                outputs.append((response.output_text, response.usage))
            assert client.privacy_audit.prefill_cache_hits == 1
            audit = client.privacy_audit.to_dict()
            assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
            reports.append((outputs, observed.copy(), audit))
            observed.clear()
    for candidate in reports[1:]:
        assert reports[0][0] == candidate[0]
        for (logits, states), (other, others) in zip(reports[0][1], candidate[1], strict=True):
            np.testing.assert_array_equal(logits, other)
            for pair, expected in zip(states, others, strict=True):
                for a, b in zip(pair, expected, strict=True):
                    np.testing.assert_array_equal(a, b)
        assert candidate[2]["masked_online_upload_bytes"] < reports[0][2]["masked_online_upload_bytes"]


@pytest.mark.integration
def test_compact_replay_and_wrong_stage_burn_material(tmp_path, monkeypatch):
    from pllm.runtime.client import _Channel
    from pllm.runtime.protocol import ProtocolEnvelope, pack_envelope, unpack_envelope
    from pllm.runtime.security import derive_session_key
    source = pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)),
                            model_id="compact-test")
    exp = experiment(source, tmp_path, "compact")
    original = _Channel.exchange
    captured = []

    def capture(channel, payload):
        result = original(channel, payload)
        frame = unpack_envelope(payload)
        if prepared_stage_batch_rows(frame.payload) == 1:
            captured.append(frame)
        return result

    monkeypatch.setattr(_Channel, "exchange", capture)
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        client.responses.create(input="A", max_output_tokens=3, temperature=0)
        assert len(captured) > 1
        first, second = captured[:2]
        # Different authenticated outer sequence cannot revive a completed lease.
        replay = ProtocolEnvelope.create(request_id="retry", session_id=first.session_id,
            model=first.model, kind=first.kind, sequence=10_000, payload=first.payload,
            metadata=first.metadata, key=derive_session_key(client._core.api_key, first.session_id))
        response = client._core.http.post(f"/v1/runtime/sessions/{first.session_id}/execute",
            headers=client._core.headers, content=pack_envelope(replay))
        assert response.status_code == 400 and "terminal" in response.text
        # A valid ticket is stage-scoped even within one live response.
        def wrong_stage(channel, payload):
            frame = unpack_envelope(payload)
            if prepared_stage_batch_rows(frame.payload) == 1:
                frame = replace(frame, metadata=second.metadata).sign(
                    derive_session_key(client._core.api_key, frame.session_id))
                payload = pack_envelope(frame)
            return original(channel, payload)
        monkeypatch.setattr(_Channel, "exchange", wrong_stage)
        with pytest.raises(Exception):
            client.responses.create(input="B", max_output_tokens=3, temperature=0)
        monkeypatch.setattr(_Channel, "exchange", original)
        assert client.responses.create(input="B", max_output_tokens=3, temperature=0).usage.output_tokens == 3
