"""Explicit larger GKM ownership; complete vocabularies and original SDK path."""

import json
import asyncio
from concurrent.futures import CancelledError
from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import pytest

from pllm import Experiment, OpenAI, _native
from pllm.configuration import ConfigurationError
from pllm.protocols import OpenWeightVerifiableOutsourcing
from pllm.runtime import delegated_source as source, delegated_transport as transport, delegated_worker
from pllm.runtime.benchmark_memory import estimate_memory
from pllm.runtime.responses import normalize_input, prompt_text

from test_delegated_decoder import clear_oracle
from test_delegated_sdk import active_sessions, experiment_fixture
from test_delegated_interfaces import gateway, run_cli


def test_explicit_capacity_roundtrip_preserves_default_identity(tmp_path):
    original, _, _ = experiment_fixture(tmp_path)
    assert "capacity" not in original.pipeline.components["linear"].params
    assert original.pipeline.components["linear"] == OpenWeightVerifiableOutsourcing(capacity="capsule")
    expanded = original.with_params(pipeline__components__linear__capacity="decoder")
    assert expanded.pipeline.digest() != original.pipeline.digest()
    assert expanded.configuration_digest() != original.configuration_digest()
    assert expanded.with_params(pipeline__components__linear__capacity="capsule") == original
    assert Experiment.from_spec(json.loads(expanded.canonical_bytes())) == expanded
    assert expanded.resolve().client_runtime == "compiled_gkm_v1"
    for bad in (None, "unbounded", True, 1):
        with pytest.raises(ConfigurationError):
            OpenWeightVerifiableOutsourcing(capacity=bad)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("tied", [False, True])
def test_decoder_capacity_real_workers_preserve_full_logits_and_kv(tmp_path, family, tied, record_property):
    experiment, cfg, arrays = experiment_fixture(
        tmp_path, family=family, tied=tied, capacity="decoder", vocabulary=1100)
    captured = []
    observed = []
    original_post = transport._Peers.post

    def post(self, *args, **kwargs):
        result = original_post(self, *args, **kwargs)
        observed.append(len(result))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(transport._Peers, "post", post)
        with OpenAI(experiment=experiment) as client:
            client._core._result_hook = captured.append
            reply = client.responses.create(input="hello world", max_output_tokens=3)
            admitted = client._core._ensure_source()
            ids = admitted.tokenizer().encode(prompt_text(normalize_input("hello world")), add_bos=False)
            result, = captured
            assert (result.logits, result.kv, result.selected, result.executed, result.position) == clear_oracle(
                cfg, arrays, tuple(ids), 2)
            assert reply.usage.output_tokens == 3
            assert admitted.capacity == "decoder" and admitted.tokenizer().vocab_size == 1100
            assert all(len(logits) == 1100 * 4 for logits in result.logits)
            assert active_sessions(client._core) == [0, 0]
            assert not client._core._transformer_states
            record_property("checked_float32_values", (
                sum(map(len, result.logits)) + sum(len(k) + len(v) for _, k, v in result.kv)) // 4)
    assert max(observed) > 8192, "fixture must exercise a frame above capsule capacity"


def test_source_and_memory_admission_use_saved_native_owner_bounds(tmp_path):
    experiment, cfg, _ = experiment_fixture(tmp_path, capacity="decoder", vocabulary=1100)
    admitted = source.admit_source(experiment)
    with pytest.raises(ValueError):
        source.admit_source(experiment.with_params(pipeline__components__linear__capacity="capsule"))
    with pytest.raises(ValueError, match="stream"):
        admitted.load_values()
    estimate = estimate_memory(cfg, experiment.pipeline, max_input_tokens=8, max_output_tokens=3)
    owners = admitted.contract["decoder_resources"]
    assert owners["client_payload_bytes"] > owners["worker_payload_bytes"]
    assert transport.request_memory_bytes(admitted) == owners["client_payload_bytes"] + (64 << 20)
    assert transport.request_memory_bytes(admitted, worker=True) == owners["worker_payload_bytes"] + (64 << 20)
    assert estimate["client_peak_bytes"] > estimate["provider_peak_bytes"]["worker_a"]
    assert estimate["native_total_peak_bytes"] == estimate["client_peak_bytes"] + sum(estimate["provider_peak_bytes"].values())
    assert estimate["components"]["source_import_and_retention_bytes_per_role"] == 0
    with pytest.raises((ValueError, ConfigurationError)):
        source.admit_source(experiment.with_params(budget__max_input_tokens=513))


def test_decoder_tokenizer_metadata_can_exceed_capsule_limit_without_padding(tmp_path):
    experiment, _, _ = experiment_fixture(tmp_path, capacity="decoder", vocabulary=1100)
    path = tmp_path / "source" / "tokenizer.json"
    raw = path.read_bytes() + b" " * (9 << 20)
    path.write_bytes(raw)
    admitted = source.admit_source(experiment)
    assert admitted.tokenizer_bytes == raw
    assert admitted.tokenizer().vocab_size == 1100
    document = json.loads(raw)
    document["padding"] = {"strategy": {"Fixed": 1_000_000_000}, "direction": "Right",
                           "pad_to_multiple_of": None, "pad_id": 0, "pad_type_id": 0,
                           "pad_token": "[UNK]"}
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="padding"):
        source.admit_source(experiment)


def test_expanded_reply_corruption_burns_window_without_output(tmp_path, monkeypatch):
    experiment, _, _ = experiment_fixture(tmp_path, capacity="decoder", vocabulary=1100)
    original = transport._Peers.post
    corrupted = []

    def post(self, role, action, *args, **kwargs):
        result = original(self, role, action, *args, **kwargs)
        if action == "step" and len(result) > 8192 and not corrupted:
            corrupted.append(True)
            result = result[:-1] + bytes([result[-1] ^ 1])
        return result

    with OpenAI(experiment=experiment) as client:
        monkeypatch.setattr(transport._Peers, "post", post)
        emitted = []
        with pytest.raises((ValueError, transport.ProtocolError)):
            with client.responses.create(input="hello world", max_output_tokens=2, stream=True) as stream:
                emitted.extend(stream)
        assert corrupted and not emitted and not client._core.records
        assert active_sessions(client._core) == [0, 0]
        monkeypatch.setattr(transport._Peers, "post", original)
        assert client.responses.create(input="hello world", max_output_tokens=2).usage.output_tokens == 2


def test_decoder_capacity_cli_gateway_and_benchmark_share_serialized_pipeline(tmp_path):
    experiment, _, _ = experiment_fixture(tmp_path, capacity="decoder", vocabulary=1100)
    path = tmp_path / "experiment.json"
    path.write_bytes(experiment.canonical_bytes())
    with gateway(path, tmp_path) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        model_id = models.json()["data"][0]["id"]
        response = client.post("/v1/responses", json={"model": model_id, "input": "hello world",
            "max_output_tokens": 2, "temperature": 0})
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["output_tokens"] == 2
    report = json.loads(run_cli("benchmark", "run", "--experiment", path,
        "--prompt", "hello world", "--max-output-tokens", "2", "--warmups", "0",
        "--repetitions", "1", "--backend", "native", "--format", "json"))["data"]["report"]
    assert report["checks"]["passed"]
    assert report["runtime_contract"]["output_delivery"] == "buffered_until_verified"
    assert report["summary"]["median_tokens_per_second"] is None


def test_capacity_native_reconstruction_does_not_change_default_contract():
    from test_delegated_decoder import config
    from pllm.runtime.delegated_decoder import decoder_contract

    cfg = config()
    raw = decoder_contract(cfg, input_rows=3, decode_steps=2)
    assert raw == decoder_contract(cfg, input_rows=3, decode_steps=2, capacity="capsule")
    assert raw == _native.gkm_decoder_contract(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode(), 3, 2)
    assert "decoder_resources" not in json.loads(raw)


@pytest.mark.parametrize("outcome", ["accepted", "rejected", "cancelled", "cancelled_chained"])
def test_worker_admission_retires_decoded_body_before_temporary_reservation(tmp_path, monkeypatch, outcome):
    """Control-flow regression: source/native arithmetic is checked separately."""
    from starlette.testclient import TestClient
    from pllm.runtime.delegated_protocol import pack

    experiment, _, _ = experiment_fixture(tmp_path)
    app = delegated_worker.create_delegated_worker_app(experiment, "worker_a", "a" * 32)
    owner = app.state.delegated_owner
    decoded_refs, released = [], []
    original_unpack = delegated_worker.unpack

    class Decoded(dict):
        pass

    def unpack(*args, **kwargs):
        value = Decoded(original_unpack(*args, **kwargs))
        decoded_refs.append(weakref.ref(value))
        return value

    @contextmanager
    def reserve(size):
        try:
            yield
        finally:
            if size == 4 * owner.admission_limit:
                assert len(decoded_refs) == 1 and decoded_refs[0]() is None
            released.append(size)

    class Native:
        def request_id(self):
            return bytes.fromhex("a" * 64)

        def abort(self):
            pass

    def load(*args, **kwargs):
        # A failed callee can retain the received payload in its traceback.
        held = decoded_refs[-1]()
        assert held is not None
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        if outcome == "cancelled_chained":
            try:
                rejected()
            except ValueError:
                raise asyncio.CancelledError() from None
        if outcome == "rejected":
            raise ValueError("controlled native import rejection")
        return Native()

    def rejected():
        held = decoded_refs[-1]()
        assert held is not None
        raise ValueError("controlled nested import rejection")

    monkeypatch.setattr(delegated_worker, "unpack", unpack)
    monkeypatch.setattr(delegated_worker, "reserve_incremental_memory", reserve)
    owner.source = SimpleNamespace(capacity="decoder",
        source_lock_digest=owner.source.source_lock_digest,
        tokenizer_digest=owner.source.tokenizer_digest,
        runtime_contract=owner.source.runtime_contract,
        load_transport=load)
    monkeypatch.setattr(delegated_worker, "request_memory_bytes", lambda *args, **kwargs: 64 << 20)
    body = pack({"schema": "pllm.gkm.admit.v1", "request": "a" * 64,
        "epoch": owner.epoch, "configuration": experiment.configuration_digest(),
        "source_lock": owner.source.source_lock_digest, "tokenizer": owner.source.tokenizer_digest,
        "input_rows": 1, "output_tokens": 1, "publication": b"test-publication",
        "pin": bytes(32), "start": b"test-start"}, limit=owner.admission_limit)
    with TestClient(app, headers={"authorization": "Bearer " + "a" * 32}) as client:
        if outcome.startswith("cancelled"):
            with pytest.raises(CancelledError):
                client.post("/v1/delegation/admit", content=body)
        else:
            reply = client.post("/v1/delegation/admit", content=body)
            assert reply.status_code == (400 if outcome == "rejected" else 200)
    assert len(released) == 2 and owner.active is None and owner.reservation is None
