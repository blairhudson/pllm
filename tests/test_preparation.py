from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import httpx
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from pllm.runtime.client import OpenAI
from pllm.runtime.config import GatewayConfig
from pllm.runtime.correction_channel import (
    CORRECTION_CHANNEL_SUBPROTOCOL,
    CorrectionChannelError,
    CorrectionWebSocketClient,
)
from pllm.runtime.correction_rendezvous import CorrectionRendezvous, RendezvousError
from pllm.runtime.preparation_protocol import (
    CorrectionPush,
    PreparationAck,
    PreparationRequest,
    SeededRingProfile,
    SessionAuthorization,
    SessionAuthorizationAck,
    expand_output_mask,
    expand_preparation_mask,
    seeded_ring_profile,
)
from pllm.runtime.preparation_server import PreparationSessionRegistry, create_preparation_app
from pllm.runtime.privacy import PrivacyMode
from pllm.runtime.protocol import ProtocolError
from pllm.runtime.stage_protocol import MaskedStageRequest, MaskedStageResponse
from pllm.runtime.transformer_client import (
    PreparedRemoteLinear,
    StageMetadata,
    quantize_activation_per_row,
)
from pllm.runtime.transformer_engine import MaskedTransformerEngine


ATTEMPT = "00112233445566778899aabbccddeeff"


def request(**overrides: object) -> PreparationRequest:
    profile = seeded_ring_profile(12_345)
    values = {
        "attempt_id": ATTEMPT,
        "session_id": "hes-vector",
        "model": "model-v1",
        "body_fingerprint": "body-abc",
        "stage_id": "layer.0.qkv",
        "weight_digest": "weight-def",
        "rows": 2,
        "in_features": 3,
        "out_features": 2,
        "weight_bits": 8,
        "activation_bits": 8,
        "signed_output_bound": profile.signed_output_bound,
        "ring": profile.ring,
        "modulus": profile.modulus,
        "wire_bits": profile.wire_bits,
        "seed": bytes(range(32)),
    }
    values.update(overrides)
    return PreparationRequest(**values)


def ring_dot(activation: np.ndarray, weight: np.ndarray, modulus: int) -> np.ndarray:
    value = activation.astype(np.int64) @ weight.astype(np.int64).T
    return np.mod(value, modulus).astype(np.uint32)


def test_seed_expansion_has_stable_vectors_and_full_domain_separation():
    value = request()
    wire = msgpack.unpackb(value.pack(), raw=False)
    assert set(wire) == {"v", "a", "h", "r", "z"}
    assert value.pack() == PreparationRequest.unpack(
        value.pack(), context=value.context
    ).pack()
    assert expand_preparation_mask(value).tobytes().hex() == "44c00000ee6200006952000021bc0000ab3b000044340000"
    assert expand_output_mask(value).tobytes().hex() == "0a9a00009c2c000089f80000ea2a0000"
    assert not np.array_equal(expand_preparation_mask(value), expand_output_mask(value))
    for changed in (
        {"session_id": "hes-other"},
        {"stage_id": "layer.0.output"},
        {"attempt_id": "10112233445566778899aabbccddeeff"},
        {"out_features": 3},
        {"weight_digest": "weight-other"},
    ):
        assert not np.array_equal(
            expand_preparation_mask(value), expand_preparation_mask(request(**changed))
        )


@pytest.mark.parametrize(
    ("bound", "ring", "bits"),
    [
        ((1 << 15) - 1, "u16", 16),
        (1 << 15, "u24", 24),
        ((1 << 23) - 1, "u24", 24),
        (1 << 23, "u32", 32),
        ((1 << 31) - 1, "u32", 32),
    ],
)
def test_exact_ring_boundaries(bound: int, ring: str, bits: int):
    profile = seeded_ring_profile(bound)
    assert (profile.ring, profile.modulus, profile.wire_bits) == (ring, 1 << bits, bits)


def test_exact_ring_rejects_unsupported_bound_and_malformed_profile():
    with pytest.raises(ProtocolError, match="exceeds"):
        seeded_ring_profile(1 << 31)
    with pytest.raises(ProtocolError, match="insufficient"):
        SeededRingProfile.from_dict(
            {"signed_output_bound": 1 << 15, "ring": "u16", "modulus": 1 << 16, "wire_bits": 16}
        )


def test_preparation_request_rejects_short_attempt_seed_and_bad_shape():
    body = msgpack.unpackb(request().pack(), raw=False)
    for key, invalid, message in (
        ("a", "abcd", "128 random bits"),
        ("z", b"short", "32 bytes"),
        ("r", 0, "shape"),
    ):
        changed = dict(body)
        changed[key] = invalid
        with pytest.raises(ProtocolError, match=message):
            PreparationRequest.unpack(
                msgpack.packb(changed, use_bin_type=True), context=request().context
            )


def test_preparation_envelope_and_ack_are_small():
    value = request(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        body_fingerprint="a" * 64,
        stage_id="layers.23.mlp.gate_up_proj",
        weight_digest="b" * 64,
        rows=32,
        in_features=896,
        out_features=9728,
    )
    assert len(value.pack()) < 400
    ack = PreparationAck(value.attempt_id, value.stage_id, 999_999, 123).pack()
    assert len(ack) < 128
    authorization = session_authorization(
        session_id="hes_" + "c" * 24,
        model=value.model,
        body_fingerprint=value.body_fingerprint,
        stage_commitment="d" * 64,
        max_attempts=1_000_000,
    )
    assert len(authorization.pack()) < 256
    assert SessionAuthorization.unpack(authorization.pack()) == authorization
    assert len(SessionAuthorizationAck(authorization.session_id).pack()) < 64


@pytest.mark.parametrize("bound", [100, 40_000, 9_000_000])
def test_prepared_remote_linear_exact_algebra_and_payload_secrecy(bound: int):
    profile = seeded_ring_profile(bound)
    weight = np.array([[3, -2, 5], [-7, 1, 4]], dtype=np.int8)
    metadata = StageMetadata(
        id="layer.0.qkv", op="linear", in_features=3, out_features=2,
        weight_bits=8, activation_bits=8, modulus=65_537, wire_bits=32,
        weight_scales=np.ones(2, dtype=np.float32), bias=np.array([0.25, -0.5], dtype=np.float32),
        role="qkv", layer_index=0, ring="prime", weight_digest="weight-def",
        seeded_profile=profile,
    )
    corrections: dict[str, np.ndarray] = {}
    ready = threading.Condition()
    seen: dict[str, bytes] = {}
    preparation_context = request(
        signed_output_bound=profile.signed_output_bound,
        ring=profile.ring,
        modulus=profile.modulus,
        wire_bits=profile.wire_bits,
    ).context

    def prepare(_stage_id: str, payload: bytes) -> bytes:
        value = PreparationRequest.unpack(payload, context=preparation_context)
        r = expand_preparation_mask(value)
        s = expand_output_mask(value)
        correction = (ring_dot(r, weight, profile.modulus).astype(np.int64) - s) % profile.modulus
        packed = CorrectionPush(
            value.attempt_id, value.session_id, value.model, value.body_fingerprint,
            value.stage_id, value.weight_digest, value.rows, value.in_features,
            value.out_features, value.weight_bits, value.activation_bits,
            value.signed_output_bound, value.ring, value.modulus, value.wire_bits,
            correction.astype(np.uint32), 11,
        ).pack()
        with ready:
            corrections[value.attempt_id] = correction
            ready.notify_all()
        seen["seed"] = value.seed
        return PreparationAck(value.attempt_id, value.stage_id, len(packed), 11).pack()

    def infer(_stage_id: str, requests: list[bytes]) -> list[bytes]:
        value = MaskedStageRequest.unpack(requests[0])
        with ready:
            ready.wait_for(lambda: value.correlation_id in corrections, timeout=2)
            correction = corrections.pop(value.correlation_id)
        seen["inference"] = requests[0]
        output = (ring_dot(value.masked_input, weight, profile.modulus).astype(np.int64) + correction) % profile.modulus
        return [MaskedStageResponse(
            correlation_id=value.correlation_id, masked_output=output.astype(np.uint32),
            modulus=profile.modulus, wire_bits=profile.wire_bits, server_ns=13,
            stage_id=value.stage_id, ring=profile.ring,
        ).pack()]

    executor = ThreadPoolExecutor(max_workers=2)
    remote = PreparedRemoteLinear(
        "model-v1", "body-abc", {metadata.id: metadata}, prepare, infer, executor,
        session_id="hes-test",
    )
    clear = np.array([[5, -3, 2], [-8, 4, 7]], dtype=np.float32)
    actual = remote(metadata.id, clear)
    executor.shutdown()
    quantized = quantize_activation_per_row(clear, bits=8)
    expected = (quantized.values.astype(np.int64) @ weight.astype(np.int64).T) * quantized.scales[:, None] + metadata.bias
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    wire = msgpack.unpackb(seen["inference"], raw=False)
    assert len(wire["correlation_id"]) == 32
    assert "seed" not in wire and "z" not in wire and seen["seed"] not in seen["inference"]
    assert remote.stats.preparation_download_bytes < 128
    assert remote.stats.correction_push_bytes > remote.stats.preparation_download_bytes


def session_authorization(**overrides: object) -> SessionAuthorization:
    values = {
        "session_id": "hes",
        "model": "m",
        "body_fingerprint": "b",
        "stage_commitment": "stages",
        "weight_bits": 8,
        "activation_bits": 8,
        "max_attempts": 10,
    }
    values.update(overrides)
    return SessionAuthorization(**values)


def correction_and_activation(attempt: str = ATTEMPT):
    profile = seeded_ring_profile(123)
    req = MaskedStageRequest(
        model="m", stage_id="s", correlation_id=attempt,
        masked_input=np.ones((1, 2), dtype=np.uint32), activation_scales=1.0,
        modulus=profile.modulus, wire_bits=profile.wire_bits, ring=profile.ring,
        body_fingerprint="b", weight_digest="w", weight_bits=8, activation_bits=8,
        session_id="hes", out_features=3, signed_output_bound=profile.signed_output_bound,
    )
    correction = CorrectionPush(
        attempt, "hes", "m", "b", "s", "w", 1, 2, 3, 8, 8,
        profile.signed_output_bound, profile.ring, profile.modulus, profile.wire_bits,
        np.ones((1, 3), dtype=np.uint32),
    )
    return correction, req


def test_rendezvous_supports_both_orders_after_one_session_authorization():
    async def scenario():
        for correction_first in (True, False):
            attempt = ("1" if correction_first else "2") * 32
            correction, activation = correction_and_activation(attempt)
            rendezvous = CorrectionRendezvous(timeout=0.01, capacity=2, max_bytes=1000)
            authorization = session_authorization()
            rendezvous.register_session(authorization)
            rendezvous.authorize_session(authorization, len(authorization.pack()))
            if correction_first:
                rendezvous.push(correction, 100)
                entry = rendezvous.activate(activation)
                actual = await rendezvous.correction(activation, entry)
            else:
                entry = rendezvous.activate(activation)
                waiting = asyncio.create_task(rendezvous.correction(activation, entry))
                await asyncio.sleep(0)
                rendezvous.push(correction, 100)
                actual = await waiting
            assert actual.attempt_id == attempt
            await asyncio.sleep(0.02)
            with pytest.raises(RendezvousError, match="consumed or burned"):
                rendezvous.push(correction, 100)
            with pytest.raises(RendezvousError, match="consumed or burned"):
                rendezvous.activate(activation)
    asyncio.run(scenario())


def test_rendezvous_timeout_cancellation_cleanup_and_capacity():
    async def scenario():
        correction, activation = correction_and_activation()
        timeout = CorrectionRendezvous(timeout=0.01, capacity=1, max_bytes=100)
        authorization = session_authorization()
        timeout.register_session(authorization)
        with pytest.raises(RendezvousError, match="not authorized"):
            timeout.activate(activation)
        timeout.authorize_session(authorization, len(authorization.pack()))
        entry = timeout.activate(activation)
        with pytest.raises(RendezvousError, match="timed out"):
            await timeout.correction(activation, entry)
        assert timeout.stats()["entries"] == 0

        canceled = CorrectionRendezvous(timeout=1, capacity=2, max_bytes=100)
        canceled.register_session(authorization)
        canceled.authorize_session(authorization, len(authorization.pack()))
        entry = canceled.activate(activation)
        waiting = asyncio.create_task(canceled.correction(activation, entry))
        await asyncio.sleep(0)
        canceled.terminal("hes")
        with pytest.raises(RendezvousError, match="cancelled"):
            await waiting
        assert canceled.stats()["entries"] == 0

        capacity = CorrectionRendezvous(timeout=1, capacity=1, max_bytes=100)
        capacity.register_session(authorization)
        capacity.authorize_session(authorization, len(authorization.pack()))
        capacity.push(correction, 100)
        other, _ = correction_and_activation("3" * 32)
        with pytest.raises(RendezvousError, match="capacity"):
            capacity.push(other, 100)
        assert capacity.stats()["bytes"] == 100
    asyncio.run(scenario())


def test_rendezvous_bounds_and_expires_session_registrations(
    monkeypatch: pytest.MonkeyPatch,
):
    import pllm.runtime.correction_rendezvous as rendezvous_module

    now = 10.0
    monkeypatch.setattr(rendezvous_module.time, "monotonic", lambda: now)
    rendezvous = CorrectionRendezvous(
        timeout=1,
        capacity=4,
        max_bytes=1000,
        session_capacity=1,
        session_idle_timeout=5,
    )
    first = session_authorization(session_id="first")
    second = session_authorization(session_id="second")
    rendezvous.register_session(first)
    with pytest.raises(RendezvousError, match="session capacity"):
        rendezvous.register_session(second)

    now = 16.0
    rendezvous.register_session(second)
    assert rendezvous.stats()["sessions"] == 1
    with pytest.raises(RendezvousError, match="not active"):
        rendezvous.authorize_session(first, len(first.pack()))


def test_rendezvous_binds_session_authorization_replay_and_attempt_metadata():
    async def scenario():
        correction, activation = correction_and_activation()
        rendezvous = CorrectionRendezvous(timeout=0.01, capacity=4, max_bytes=1000)
        authorization = session_authorization()
        rendezvous.register_session(authorization)
        with pytest.raises(RendezvousError, match="metadata mismatch"):
            rendezvous.authorize_session(
                replace(authorization, stage_commitment="other"), 100
            )
        rendezvous.authorize_session(authorization, len(authorization.pack()))
        with pytest.raises(RendezvousError, match="already consumed"):
            rendezvous.authorize_session(authorization, len(authorization.pack()))

        mismatch_correction, mismatch_activation = correction_and_activation("4" * 32)
        rendezvous.push(replace(mismatch_correction, weight_digest="other"), 100)
        with pytest.raises(RendezvousError, match="metadata mismatch"):
            rendezvous.activate(mismatch_activation)

        correction_push, _ = correction_and_activation("6" * 32)
        rendezvous.push(correction_push, 100)
        with pytest.raises(RendezvousError, match="duplicate correction push"):
            rendezvous.push(correction_push, 100)
        assert rendezvous.stats()["entries"] == 0

        _, duplicate_activation = correction_and_activation("5" * 32)
        entry = rendezvous.activate(duplicate_activation)
        with pytest.raises(RendezvousError, match="duplicate activation"):
            rendezvous.activate(duplicate_activation)
        rendezvous.abort(duplicate_activation, entry)
        assert rendezvous.stats()["entries"] == 0

        rendezvous.terminal("hes")
        assert rendezvous.stats()["burned"] == 0
        with pytest.raises(RendezvousError, match="not active"):
            rendezvous.push(correction, 100)
        with pytest.raises(RendezvousError, match="not active"):
            rendezvous.authorize_session(authorization, len(authorization.pack()))
    asyncio.run(scenario())


def test_rendezvous_enforces_authorized_session_attempt_budget():
    authorization = session_authorization(max_attempts=1)
    rendezvous = CorrectionRendezvous(timeout=1, capacity=2, max_bytes=1000)
    rendezvous.register_session(authorization)
    rendezvous.authorize_session(authorization, len(authorization.pack()))
    first, _ = correction_and_activation("a" * 32)
    second, _ = correction_and_activation("b" * 32)
    rendezvous.push(first, 100)
    with pytest.raises(RendezvousError, match="attempt capacity"):
        rendezvous.push(second, 100)


def test_authorized_attempt_without_correction_times_out_and_shape_limits_precede_allocation():
    async def scenario():
        _, activation = correction_and_activation()
        rendezvous = CorrectionRendezvous(timeout=0.01, capacity=2, max_bytes=1000)
        authorization = session_authorization()
        rendezvous.register_session(authorization)
        rendezvous.authorize_session(authorization, len(authorization.pack()))
        entry = rendezvous.activate(activation)
        with pytest.raises(RendezvousError, match="correction rendezvous timed out"):
            await rendezvous.correction(activation, entry)
        assert rendezvous.stats()["entries"] == 0
    asyncio.run(scenario())

    activation = msgpack.unpackb(correction_and_activation()[1].pack(), raw=False)
    activation["shape"] = [10_000_000, 2]
    with pytest.raises(ProtocolError, match="row count"):
        MaskedStageRequest.unpack(msgpack.packb(activation), max_rows=513)

    correction = msgpack.unpackb(correction_and_activation()[0].pack(), raw=False)
    correction["r"] = 10_000_000
    with pytest.raises(ProtocolError, match="row count"):
        CorrectionPush.unpack(msgpack.packb(correction), max_rows=513)


def test_preparation_session_registry_bounds_owners_attempts_and_idle_lifetime(
    monkeypatch: pytest.MonkeyPatch,
):
    import pllm.runtime.preparation_server as preparation_server

    now = 10.0
    monkeypatch.setattr(preparation_server.time, "monotonic", lambda: now)
    registry = PreparationSessionRegistry(capacity=1, idle_timeout=5)
    first = session_authorization(session_id="first", max_attempts=1)
    registry.reserve(first, "owner")
    registry.activate(first.session_id, "owner")
    with pytest.raises(ProtocolError, match="capacity"):
        registry.reserve(session_authorization(session_id="second"), "owner")
    with pytest.raises(ProtocolError, match="not authorized"):
        registry.authorization(first.session_id, "other")
    registry.consume(first.session_id, ATTEMPT, "owner")
    with pytest.raises(ProtocolError, match="already consumed"):
        registry.consume(first.session_id, ATTEMPT, "owner")
    with pytest.raises(ProtocolError, match="attempt capacity"):
        registry.consume(first.session_id, "1" * 32, "owner")

    now = 16.0
    second = session_authorization(session_id="second")
    registry.reserve(second, "owner")
    assert registry.stats()["sessions"] == 1


def test_preparation_service_authorizes_once_then_computes_and_pushes_once():
    events: list[str] = []
    value = request(rows=513)
    correction = CorrectionPush(
        value.attempt_id, value.session_id, value.model, value.body_fingerprint,
        value.stage_id, value.weight_digest, value.rows, value.in_features,
        value.out_features, value.weight_bits, value.activation_bits,
        value.signed_output_bound, value.ring, value.modulus, value.wire_bits,
        np.zeros((value.rows, value.out_features), dtype=np.uint32),
    )

    class Engine:
        weight_bits = 8
        activation_bits = 8
        models = {
            value.model: SimpleNamespace(
                manifest=SimpleNamespace(
                    context_length=1024,
                    id=value.model,
                    metadata={
                        "body_fingerprint": value.body_fingerprint,
                        "seeded_stage_commitment": "stages",
                    },
                )
            )
        }

        def validate_seeded_session_authorization(self, actual):
            assert actual == authorization

        def validate_seeded_preparation(self, actual):
            assert actual == value

        async def prepare_seeded_stage(self, actual):
            assert actual == value
            events.append("compute")
            return correction

        async def stage_metadata(self, model_id, stage_id):
            assert (model_id, stage_id) == (value.model, value.stage_id)
            return SimpleNamespace(
                weight_digest=value.weight_digest,
                in_features=value.in_features,
                out_features=value.out_features,
            )

        def seeded_profile(self, model_id, stage_id):
            assert (model_id, stage_id) == (value.model, value.stage_id)
            return value.profile

    async def push(request):
        if request.url.path.endswith("/authorize"):
            assert request.content == authorization_payload
            events.append("authorization")
            return httpx.Response(
                200,
                content=SessionAuthorizationAck(authorization.session_id).pack(),
            )
        events.append("correction")
        return httpx.Response(
            200,
            content=PreparationAck(
                correction.attempt_id,
                correction.stage_id,
                len(correction.pack()),
                correction.server_ns,
            ).pack(),
        )

    push_client = httpx.AsyncClient(
        base_url="https://inference.example", transport=httpx.MockTransport(push)
    )
    app = create_preparation_app(
        GatewayConfig(
            api_keys=("preparation", "other-preparation"),
            preparation_inference_url="https://inference.example",
            preparation_push_api_key="push",
        ),
        Engine(),  # type: ignore[arg-type]
        push_client=push_client,
    )
    authorization = SessionAuthorization(
        session_id=value.session_id,
        model=value.model,
        body_fingerprint=value.body_fingerprint,
        stage_commitment="stages",
        weight_bits=value.weight_bits,
        activation_bits=value.activation_bits,
        max_attempts=100,
    )
    authorization_payload = authorization.pack()
    with TestClient(app) as client:
        compact_payload = value.pack()
        assert set(msgpack.unpackb(compact_payload, raw=False)) == {
            "v", "a", "h", "r", "z"
        }
        unauthorized = client.post(
            f"/v1/preparation/stages/{value.stage_id}",
            headers={"Authorization": "Bearer preparation"},
            content=compact_payload,
        )
        assert unauthorized.status_code == 400
        authorized = client.post(
            f"/v1/preparation/sessions/{value.session_id}/authorize",
            headers={"Authorization": "Bearer preparation"},
            content=authorization_payload,
        )
        assert authorized.status_code == 200, authorized.text
        replayed_authorization = client.post(
            f"/v1/preparation/sessions/{value.session_id}/authorize",
            headers={"Authorization": "Bearer preparation"},
            content=authorization_payload,
        )
        assert replayed_authorization.status_code == 400
        wrong_owner = client.post(
            f"/v1/preparation/stages/{value.stage_id}",
            headers={"Authorization": "Bearer other-preparation"},
            content=compact_payload,
        )
        assert wrong_owner.status_code == 400
        response = client.post(
            f"/v1/preparation/stages/{value.stage_id}",
            headers={"Authorization": "Bearer preparation"},
            content=compact_payload,
        )
        assert response.status_code == 200, response.text
        replayed_attempt = client.post(
            f"/v1/preparation/stages/{value.stage_id}",
            headers={"Authorization": "Bearer preparation"},
            content=compact_payload,
        )
        assert replayed_attempt.status_code == 400
        assert client.get("/metrics").status_code == 401
        assert client.get(
            "/metrics", headers={"Authorization": "Bearer preparation"}
        ).status_code == 200
        oversized = client.post(
            f"/v1/preparation/stages/{value.stage_id}",
            headers={
                "Authorization": "Bearer preparation",
                "Content-Length": "16385",
            },
            content=b"x",
        )
        assert oversized.status_code == 413
        oversized_authorization = client.post(
            f"/v1/preparation/sessions/{value.session_id}/authorize",
            headers={
                "Authorization": "Bearer preparation",
                "Content-Length": "16385",
            },
            content=b"x",
        )
        assert oversized_authorization.status_code == 413
        metrics = client.get(
            "/metrics", headers={"Authorization": "Bearer preparation"}
        ).json()["preparation"]
        assert metrics["authorization_calls"] == 1
        assert metrics["authorization_push_bytes"] == len(authorization_payload)
        assert metrics["correction_push_bytes"] == len(correction.pack())
        assert metrics["active_sessions"] == 1
        assert metrics["session_attempts"] == 1
    asyncio.run(push_client.aclose())
    assert events == ["authorization", "compute", "correction"]


def test_client_failure_cancels_and_drains_sibling_provider_future():
    profile = seeded_ring_profile(123)
    metadata = StageMetadata(
        id="s", op="linear", in_features=2, out_features=3,
        weight_bits=8, activation_bits=8, modulus=profile.modulus,
        wire_bits=profile.wire_bits, ring=profile.ring,
        weight_scales=np.ones(3, dtype=np.float32), weight_digest="w",
        seeded_profile=profile,
    )
    started = threading.Barrier(2)
    inference_finished = threading.Event()

    def prepare(_stage_id, _payload):
        started.wait(timeout=1)
        raise RuntimeError("preparation failed")

    def infer(_stage_id, _payloads):
        started.wait(timeout=1)
        try:
            threading.Event().wait(0.02)
            raise RuntimeError("inference failed")
        finally:
            inference_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        remote = PreparedRemoteLinear(
            "m", "b", {"s": metadata}, prepare, infer, executor, session_id="hes"
        )
        with pytest.raises(RuntimeError, match="failed"):
            remote("s", np.ones((1, 2), dtype=np.float32))
        assert inference_finished.is_set()
        assert remote.stats.failures == 1


def test_remote_services_require_https_and_three_distinct_credentials():
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAI(
            base_url="https://inference.example", api_key="inference",
            preparation_base_url="http://preparation.example", preparation_api_key="preparation",
        )
    with pytest.raises(ValueError, match="credentials must be distinct"):
        OpenAI(
            base_url="https://inference.example", api_key="same",
            preparation_base_url="https://preparation.example", preparation_api_key="same",
        )
    with pytest.raises(ValueError, match="correction-push credentials"):
        from pllm.runtime.server import create_app
        create_app(GatewayConfig(api_keys=("same",), provider_push_api_key="same"))
    with pytest.raises(ValueError, match="client and push credentials"):
        create_preparation_app(
            GatewayConfig(
                api_keys=("same",), preparation_inference_url="https://inference.example",
                preparation_push_api_key="same",
            ),
            MaskedTransformerEngine(),
        )


def test_correction_endpoint_accepts_only_push_credential():
    from pllm.runtime.server import create_app

    app = create_app(
        GatewayConfig(api_keys=("client-key",), provider_push_api_key="push-key")
    )
    with TestClient(app) as client:
        rejected = client.post(
            f"/v1/he/sessions/hes/corrections/{ATTEMPT}",
            headers={"Authorization": "Bearer client-key"},
            content=b"invalid",
        )
        assert rejected.status_code == 401
        authenticated = client.post(
            f"/v1/he/sessions/hes/corrections/{ATTEMPT}",
            headers={"Authorization": "Bearer push-key"},
            content=b"invalid",
        )
        assert authenticated.status_code == 409


def test_preparation_service_requires_https_fixed_endpoint_and_public_weights():
    with pytest.raises(ValueError, match="public-weight"):
        create_preparation_app(
            GatewayConfig(api_keys=("test",), privacy_mode=PrivacyMode.PROPRIETARY),
            MaskedTransformerEngine(),
        )
    with pytest.raises(ValueError, match="HTTPS"):
        create_preparation_app(
            GatewayConfig(
                api_keys=("preparation",), preparation_inference_url="http://inference.example",
                preparation_push_api_key="push",
            ),
            MaskedTransformerEngine(),
        )
    with pytest.raises(ValueError, match="origin"):
        create_preparation_app(
            GatewayConfig(
                api_keys=("preparation",),
                preparation_inference_url="https://inference.example/not-fixed",
                preparation_push_api_key="push",
            ),
            MaskedTransformerEngine(),
        )


def test_correction_websocket_accepts_only_push_credential_and_bounds_frames():
    from pllm.runtime.server import create_app

    app = create_app(
        GatewayConfig(
            api_keys=("client-key",),
            provider_push_api_key="push-key",
            prepared_payload_max_bytes=32,
        )
    )
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/v1/he/corrections/ws",
                headers={"Authorization": "Bearer client-key"},
                subprotocols=[CORRECTION_CHANNEL_SUBPROTOCOL],
            ):
                pass
        assert rejected.value.code == 4401

        with client.websocket_connect(
            "/v1/he/corrections/ws",
            headers={"Authorization": "Bearer push-key"},
            subprotocols=[CORRECTION_CHANNEL_SUBPROTOCOL],
        ) as websocket:
            websocket.send_bytes(b"x" * 555)
            with pytest.raises(WebSocketDisconnect) as oversized:
                websocket.receive_bytes()
            assert oversized.value.code == 4409


def test_correction_websocket_serializes_sends_and_never_retries_ambiguous_send(
    monkeypatch: pytest.MonkeyPatch,
):
    import pllm.runtime.correction_channel as channel_module

    class FakeSocket:
        def __init__(self, *, fail_send: bool = False) -> None:
            self.fail_send = fail_send
            self.close_code = None
            self.sent: list[str] = []
            self.received: asyncio.Queue[bytes] = asyncio.Queue()

        async def send(self, frame: bytes) -> None:
            correction = CorrectionPush.unpack(frame)
            self.sent.append(correction.attempt_id)
            if self.fail_send:
                raise ConnectionError("send lost")
            await self.received.put(
                PreparationAck(
                    correction.attempt_id,
                    correction.stage_id,
                    len(frame),
                    0,
                ).pack()
            )

        async def recv(self, *, decode: bool = False) -> bytes:
            assert decode is False
            return await self.received.get()

        async def close(self) -> None:
            self.close_code = 1000

    async def scenario() -> None:
        sockets = [
            FakeSocket(),
            FakeSocket(fail_send=True),
            FakeSocket(),
        ]
        connects = 0

        async def fake_connect(*_args, **_kwargs):
            nonlocal connects
            socket = sockets[connects]
            connects += 1
            return socket

        monkeypatch.setattr(channel_module, "connect", fake_connect)
        channel = CorrectionWebSocketClient(
            "wss://inference.example/v1/he/corrections/ws",
            "push-key",
            timeout=1,
            max_payload_bytes=1000,
        )
        first, _ = correction_and_activation("1" * 32)
        second, _ = correction_and_activation("2" * 32)
        sent = await asyncio.gather(
            channel.push(first.session_id, first.attempt_id, first.pack()),
            channel.push(second.session_id, second.attempt_id, second.pack()),
        )
        assert [value[0] for value in sent] == [b"", b""]
        assert sockets[0].sent == [first.attempt_id, second.attempt_id]
        assert connects == 1
        await channel.close()

        disconnected = CorrectionWebSocketClient(
            "wss://inference.example/v1/he/corrections/ws",
            "push-key",
            timeout=1,
            max_payload_bytes=1000,
        )
        third, _ = correction_and_activation("3" * 32)
        with pytest.raises(CorrectionChannelError, match="ambiguous"):
            await disconnected.push(third.session_id, third.attempt_id, third.pack())
        assert sockets[1].sent == [third.attempt_id]
        fourth, _ = correction_and_activation("4" * 32)
        ack, sent_bytes = await disconnected.push(
            fourth.session_id, fourth.attempt_id, fourth.pack()
        )
        assert ack == b""
        assert sent_bytes == len(fourth.pack())
        assert sockets[1].sent == [third.attempt_id]
        assert connects == 3
        await disconnected.close()

    asyncio.run(scenario())
