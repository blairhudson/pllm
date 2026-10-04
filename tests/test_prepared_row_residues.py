"""Prepared output packing keeps complete decoder values and one-use contracts."""
import asyncio
import hashlib
import json

import msgpack
import numpy as np
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, lower_model
from pllm.kernels import Cpu
from pllm.model_loader import resolve_model
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, VerifiedMaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.residue_codec import pack_row_response, unpack_row_response
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.runtime.stage_protocol import MaskedStageResponse, ProtocolError
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, TransformerClientError
from pllm.runtime.transformer_engine import MaskedTransformerEngine
from pllm.state import ClientPrefixReuse


def experiment(source, root, encoding, verified=False):
    profile = VerifiedMaskedLinearCpu if verified else MaskedLinearCpu
    return Experiment("residue-test", profile(source,
        linear=MaskedLinear(output_encoding=encoding), kernels=Cpu(threads=1),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        inventory=PreparedInventory(rows=1, refill="on-demand"),
        delivery=ClientBundleTransport("artifacts", compression="zlib"),
        placement=ClientLinearRoles(["attention_output"]),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=128, generated_prefixes=True)),
        Deployment.local(root=str(root)), ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=3))


@pytest.mark.parametrize("family,verified", [("qwen2", False), ("qwen3", False), ("qwen2", True)])
def test_two_child_residue_execution_keeps_all_logits_kv_and_verification(tmp_path, monkeypatch, family, verified):
    source = Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        seed=71, num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")), model_id="residue-test")
    observed, results = [], []
    original_prefill, original_decode = SemanticDecoderRuntime.prepare_ids, SemanticDecoderRuntime.decode_step

    def save(runtime, logits):
        observed.append((logits.copy(), [(c.key[:c.length].copy(), c.value[:c.length].copy()) for c in runtime.caches]))

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
    for encoding in ("raw", "row_residues"):
        exp = experiment(source, tmp_path / encoding, encoding, verified)
        with build_roles(exp, engine_threads=1) as topology, topology.client(bundle_cache_dir=tmp_path / f"cache-{encoding}") as client:
            request = dict(model="residue-test", input="abcde " * 8, temperature=0, max_output_tokens=3)
            response = client.responses.create(**request)
            assert client.responses.create(**request).output_text == response.output_text
            assert client.privacy_audit.prefill_cache_hits == 1
            assert client.privacy_audit.plaintext_prompt_bytes_sent == client.privacy_audit.plaintext_token_ids_sent == 0
            if verified:
                state = client._core._transformer_conversations[response.id].snapshot.state_basis
                assert state.valid() and state.verification_failure_bits == 52
            results.append((response.output_text, observed.copy()))
            observed.clear()
    assert results[0][0] == results[1][0]
    assert len(results[0][1]) == len(results[1][1])
    for (logits, states), (other, others) in zip(results[0][1], results[1][1], strict=True):
        np.testing.assert_array_equal(logits, other)
        for pair, expected in zip(states, others, strict=True):
            for a, b in zip(pair, expected, strict=True):
                np.testing.assert_array_equal(a, b)


def test_layout_and_raw_downgrade_rejected_before_material(tmp_path):
    source = resolve_model(Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model")), model_id="residue-test"))
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, prepared_output_encoding="row_residues")
    asyncio.run(engine.load(source.manifest))
    pipeline = MaskedLinearCpu(Model.path(str(source.path), model_id="residue-test"), linear=MaskedLinear(output_encoding="row_residues"))
    payload = engine.client_bundle("residue-test")
    bundle = ClientBundle.unpack(payload)
    plan = lower_model(bundle.cfg, batch=1, max_input_tokens=4, max_new_tokens=2)
    assert compile_runtime_model(plan, bundle, composition=pipeline).model_plan_digest == plan.digest
    with pytest.raises(ValueError, match="encoding"):
        compile_runtime_model(plan, bundle, composition=MaskedLinearCpu(pipeline.model))
    forged = msgpack.unpackb(payload, raw=False)
    forged["manifest"]["metadata"]["prepared_residue_layout_digest"] = "0" * 64
    forged["manifest"]["fingerprint"] = hashlib.sha256(json.dumps(
        {k: v for k, v in forged["manifest"].items() if k not in {"fingerprint", "created_at"}},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="layout"):
        compile_runtime_model(plan, ClientBundle.unpack(msgpack.packb(forged, use_bin_type=True)), composition=pipeline)
    del forged["privacy"]["prepared_output_encoding"]
    with pytest.raises(TransformerClientError, match="layout"):
        ClientBundle.unpack(msgpack.packb(forged, use_bin_type=True))
    asyncio.run(engine.unload("residue-test"))


def test_packed_reply_cannot_cross_protocol_or_stage():
    response = MaskedStageResponse(correlation_id="ticket", stage_id="stage", ring="u16",
        modulus=65536, wire_bits=16, masked_output=np.array([[60000, 9]], dtype=np.uint32))
    widths = bytes([13, 7])
    raw = pack_row_response(response, widths, namespace="prepared")
    options = dict(ticket="ticket", stage="stage", widths=widths, rows=1, bits=16)
    assert unpack_row_response(raw, **options, namespace="prepared")[0]
    with pytest.raises(ProtocolError):
        unpack_row_response(raw, **options)
    with pytest.raises(ProtocolError):
        unpack_row_response(raw, **{**options, "stage": "another"}, namespace="prepared")
    with pytest.raises(ProtocolError):
        unpack_row_response(raw, **{**options, "widths": bytes([12, 7])}, namespace="prepared")
