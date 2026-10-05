"""Stage quotients preserve integer outputs and reject unadmitted layouts."""
from dataclasses import replace
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from pllm.protocols import MaskedLinear
from pllm.runtime.protocol import ProtocolError
from pllm.runtime.stage_protocol import PreparedStageBatchRequest


@pytest.mark.parametrize("rows,bits", [(1, 11), (5, 17), (3, 32)])
def test_packed_ingress_has_fixed_width_and_canonical_padding(rows, bits):
    wire = next(n for n in (16, 24, 32) if n >= bits)
    values = np.random.default_rng(351).integers(0, 1 << wire, (rows, 3), dtype=np.uint32)
    request = PreparedStageBatchRequest("ab" * 16, tuple(f"{i:032x}" for i in range(rows)), values, wire, bits)
    payload = request.pack()
    options = dict(expected_input_bits=bits, max_rows=rows, max_tensor_elements=values.size)
    decoded = PreparedStageBatchRequest.unpack(payload, **options)
    np.testing.assert_array_equal(decoded.masked_input, values & ((1 << bits) - 1))
    assert decoded.batch_id == request.batch_id and decoded.correlation_ids == request.correlation_ids
    for wrong in (None, bits - 1, bits + 1):
        with pytest.raises(ProtocolError):
            PreparedStageBatchRequest.unpack(payload, **{**options, "expected_input_bits": wrong})
    with pytest.raises(ProtocolError, match="admitted"):
        PreparedStageBatchRequest.unpack(replace(request, input_bits=None).pack(), **options)
    with pytest.raises(ProtocolError, match="allocation"):
        PreparedStageBatchRequest.unpack(payload, **{**options, "max_tensor_elements": values.size - 1})
    fields = msgpack.unpackb(payload[12:])
    for field, bad in ((2, True), (2, "3"), (2, 4_000_001), (4, True), (4, bits - 1),
                       (5, fields[5][:-1]), (5, fields[5] + b"\0")):
        modified = fields.copy()
        modified[field] = bad
        with pytest.raises(ProtocolError):
            PreparedStageBatchRequest.unpack(payload[:12] + msgpack.packb(modified, use_bin_type=True), **options)
    if values.size * bits % 8:
        fields[5] = fields[5][:-1] + bytes([fields[5][-1] | 128])
        with pytest.raises(ProtocolError, match="payload"):
            PreparedStageBatchRequest.unpack(payload[:12] + msgpack.packb(fields, use_bin_type=True), **options)


def test_packed_ingress_requires_residue_outputs_and_preflights_input_capacity(monkeypatch):
    from pllm.runtime import residue_codec as codec
    with pytest.raises(ValueError, match="row_residues"):
        MaskedLinear(request_encoding="stage_packed")
    stage = SimpleNamespace(out_features=2, in_features=5, seeded_profile=SimpleNamespace(wire_bits=16),
                            output_residue_bits=bytes([9, 7]), op="linear", activation_bits=8, weight_digest="w")
    digest = codec.row_layout_digest([("stage", "w", 5, 2, 8, "0907")], namespace="prepared")
    compiled = SimpleNamespace(_plan=SimpleNamespace(prefill={"query_sequence": 2}),
        _stages=[SimpleNamespace(client_weight_layout=None, stage_id="stage")],
        _bundle=SimpleNamespace(stages={"stage": stage}, manifest={"metadata": {"prepared_residue_layout_digest": digest}}))
    monkeypatch.setattr(codec, "MAX_VALUES", 8)
    assert codec.compiled_row_layout(compiled, namespace="prepared") == digest
    with pytest.raises(ProtocolError, match="input exceeds"):
        codec.compiled_row_layout(compiled, namespace="prepared", packed_input=True)


@pytest.mark.integration
def test_packed_ack_downgrade_burns_reservation_before_any_stage(tmp_path, monkeypatch):
    import httpx
    import pllm
    from test_prepared_compact import experiment
    from pllm.runtime.masked_runtime import ModelError
    from pllm.runtime.servers import build_roles
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    source = pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)),
                             model_id="packed-test")
    exp = experiment(source, tmp_path, "stage_packed", output="row_residues")
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        client.preprocess("packed-test", count=32)
        core = client._core
        inventory = core._transformer_states["packed-test"].prepared_inventory
        before = inventory.status()
        original = core.http.post

        def downgrade(path, **kwargs):
            response = original(path, **kwargs)
            if path == "/v1/runtime/sessions" and response.is_success:
                value = response.json()
                value["prepared_request_encoding"] = "compact"
                return httpx.Response(response.status_code, json=value, request=response.request)
            return response

        with monkeypatch.context() as patch:
            patch.setattr(core.http, "post", downgrade)
            with pytest.raises(ModelError, match="did not admit"):
                client.responses.create(input="A", max_output_tokens=3, temperature=0)
        after = inventory.status()
        assert after["burned"] > before["burned"]
        assert after["reserved"] == after["consumed"] == core.audit.inference_stage_calls == 0
        assert client.responses.create(input="A", max_output_tokens=3, temperature=0).usage.output_tokens == 3


@pytest.mark.integration
def test_packed_width_forgery_aborts_authenticated_session(tmp_path, monkeypatch):
    import pllm
    from test_prepared_compact import experiment
    from pllm.runtime.client import _Channel
    from pllm.runtime.protocol import pack_envelope, unpack_envelope
    from pllm.runtime.security import derive_session_key
    from pllm.runtime.servers import build_roles
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    source = pllm.Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)),
                             model_id="packed-test")
    exp = experiment(source, tmp_path, "stage_packed", output="row_residues")
    original = _Channel.exchange
    captured = []
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        def forge(channel, payload):
            frame = unpack_envelope(payload)
            if frame.payload.startswith(b"PLLMPSB3"):
                fields = msgpack.unpackb(frame.payload[12:])
                fields[4] -= 1
                frame = replace(frame, payload=frame.payload[:12] + msgpack.packb(fields, use_bin_type=True)).sign(
                    derive_session_key(client._core.api_key, frame.session_id))
                captured.append(frame.session_id)
                payload = pack_envelope(frame)
            return original(channel, payload)

        with monkeypatch.context() as patch:
            patch.setattr(_Channel, "exchange", forge)
            with pytest.raises(Exception, match="packed-input layout"):
                client.responses.create(input="A", max_output_tokens=3, temperature=0)
        assert captured
        inventory = client._core._transformer_states["packed-test"].prepared_inventory.status()
        assert inventory["reserved"] == 0 and inventory["burned"] > 0
        assert client.responses.create(input="A", max_output_tokens=3, temperature=0).usage.output_tokens == 3
