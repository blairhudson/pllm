import asyncio
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from conftest import start_gateway, start_preparation
from pllm.runtime import GatewayConfig, OpenAI, create_app
from pllm.runtime.client import ProtocolError
from pllm.runtime.correction_channel import (
    CORRECTION_CHANNEL_SUBPROTOCOL,
)
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.preparation_protocol import CorrectionPush, PreparationAck, SessionAuthorization
from pllm.runtime.protocol import encode_length_prefixed
from pllm.runtime.stage_protocol import MaskedStageRequest
from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def prepared_service(root: Path, model_id: str, gateway):
    engine = MaskedTransformerEngine(threads=1)
    asyncio.run(engine.load(load_hf_directory(root, model_id=model_id)))
    return start_preparation(engine, gateway.base_url, gateway.push_api_key), engine


def test_tiny_gemma_responses_api_keeps_prompt_local(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "tiny")
    engine = MaskedTransformerEngine(threads=1)
    gateway = start_gateway(engines={engine.capabilities.name: engine})
    preparation, preparation_engine = prepared_service(root, "tiny-gemma-pllm", gateway)
    canary = "PRIVATE-CANARY-4b6a8739"
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            loaded = admin.post(
                "/v1/runtime/models/load",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": "tiny-gemma-pllm",
                },
            )
            assert loaded.status_code == 200, loaded.text
            assert loaded.json()["status"] == "ready"

        with OpenAI(
            api_key=gateway.api_key,
            base_url=gateway.base_url,
            preparation_base_url=preparation.base_url,
            preparation_api_key=preparation.api_key,
            background_inventory_refill=False,
        ) as client:
            response = client.responses.create(
                model="tiny-gemma-pllm",
                input=canary,
                max_output_tokens=2,
                temperature=0,
            )
            assert response.status == "completed"
            assert response.usage is not None
            assert response.usage.output_tokens <= 2
            audit = client.privacy_audit.to_dict()
            assert audit["plaintext_prompt_bytes_sent"] == 0
            assert audit["plaintext_token_ids_sent"] == 0
            assert audit["online_steps"] > 0
            assert audit["correlation_count"] == 0
            assert audit["preparation_upload_bytes"] > 0

            stream = client.responses.create(
                model="tiny-gemma-pllm",
                input="abandoned",
                max_output_tokens=1,
                stream=True,
            )
            abandoned_id = next(stream).response["id"]
            stream.close()
            assert client.responses.retrieve(abandoned_id).status == "cancelled"

        raw_audit = b"\n".join(payload for _, payload in gateway.audit)
        assert canary.encode() not in raw_audit
        assert engine.stats()["execute_items"] > 0
        assert preparation_engine.stats()["execute_items"] > 0
    finally:
        gateway.close()
        preparation.close()


def test_seeded_preparation_executes_w8_without_sending_prompt(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "tiny-prepared",
        num_hidden_layers=1,
        ple_dim=0,
    )
    engine = MaskedTransformerEngine(threads=1)
    gateway = start_gateway(engines={engine.capabilities.name: engine})
    model_id = "tiny-prepared"
    preparation, preparation_engine = prepared_service(root, model_id, gateway)
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            loaded = admin.post(
                "/v1/runtime/models/load",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": model_id,
                },
            )
            assert loaded.status_code == 200, loaded.text
        prompt = "PRIVATE-PREPARED-INFERENCE-CANARY"
        with OpenAI(
            api_key=gateway.api_key,
            base_url=gateway.base_url,
            preparation_base_url=preparation.base_url,
            preparation_api_key=preparation.api_key,
            prepared_inventory_rows=128,
            background_inventory_refill=False,
        ) as client:
            response = client.responses.create(
                model=model_id,
                input=prompt,
                max_output_tokens=2,
                temperature=0,
            )
            assert response.status == "completed"
            assert response.usage is not None
            assert response.usage.output_tokens <= 2
            audit = client.privacy_audit.to_dict()
            assert audit["plaintext_prompt_bytes_sent"] == 0
            assert audit["plaintext_token_ids_sent"] == 0
            assert audit["correlation_count"] == 0
            assert audit["preparation_upload_bytes"] > 0
            assert audit["preparation_upload_bytes"] < 100_000
            assert audit["inference_upload_bytes"] > audit["preparation_upload_bytes"]
            assert audit["preparation_download_bytes"] > 0
            assert audit["preparation_download_bytes"] < audit["preparation_upload_bytes"]
            assert audit["correction_push_bytes"] > audit["preparation_download_bytes"]
            assert audit["inference_download_bytes"] > 0
            assert audit["preparation_attempts"] > 0
            assert audit["preparation_failures"] == 0
            assert audit["correction_push_ns"] > 0
            assert audit["encrypted_correlation_upload_bytes"] == 0
            assert audit["encrypted_correlation_download_bytes"] == 0
            assert audit["online_steps"] > 0
            prepared_items = preparation_engine.stats()["execute_items"]
            prepared_attempts = client.privacy_audit.preparation_attempts
            second = client.responses.create(
                model=model_id,
                input="second",
                max_output_tokens=2,
                temperature=0,
            )
            assert second.status == "completed"
            second_audit = client.privacy_audit.to_dict()
            assert preparation_engine.stats()["execute_items"] == prepared_items
            assert second_audit["preparation_attempts"] == prepared_attempts
            assert second_audit["inference_upload_bytes"] > audit["inference_upload_bytes"]
            inventory_id = client.prepared_inventory_status(model_id)["id"]
            canceled = httpx.post(
                f"{gateway.base_url}/v1/runtime/inventories/{inventory_id}/cancel",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
            )
            assert canceled.status_code == 200, canceled.text
            replacement = client.responses.create(
                model=model_id,
                input="stale inventory is replaced before online execution",
                max_output_tokens=1,
            )
            assert replacement.status == "completed"
            replacement_audit = client.privacy_audit.to_dict()
            assert preparation_engine.stats()["execute_items"] > prepared_items
        preparation_metrics = httpx.get(
            f"{preparation.base_url}/metrics",
            headers={"Authorization": f"Bearer {preparation.api_key}"},
        ).json()["preparation"]
        inference_metrics = httpx.get(
            f"{gateway.base_url}/metrics",
            headers={"Authorization": f"Bearer {gateway.api_key}"},
        ).json()
        assert (
            preparation_metrics["correction_push_attempts"]
            == replacement_audit["preparation_attempts"]
        )
        assert (
            preparation_metrics["correction_channel_upload_bytes"]
            == replacement_audit["correction_push_bytes"]
        )
        assert (
            preparation_metrics["correction_push_ns"]
            == replacement_audit["correction_push_ns"]
        )
        assert inference_metrics["correction_channel"]["connections"] == 1
        assert (
            inference_metrics["correction_channel"]["frames"]
            == replacement_audit["preparation_attempts"]
        )
        for service in (gateway, preparation):
            raw_audit = b"\n".join(payload for _, payload in service.audit)
            assert prompt.encode() not in raw_audit
        assert engine.stats()["execute_items"] > 0
        assert preparation_engine.stats()["execute_items"] > 0
    finally:
        gateway.close()
        preparation.close()


def test_seeded_preparation_accepts_prefill_larger_than_decode_scheduler_batch(
    tmp_path: Path,
):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "tiny-long-prefill",
        num_hidden_layers=1,
        ple_dim=0,
        max_position_embeddings=1024,
    )
    engine = MaskedTransformerEngine(threads=1)
    gateway = start_gateway(engines={engine.capabilities.name: engine})
    model_id = "tiny-long-prefill"
    preparation, _ = prepared_service(root, model_id, gateway)
    prompt = "P" * 520
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            loaded = admin.post(
                "/v1/runtime/models/load",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": model_id,
                },
            )
            assert loaded.status_code == 200, loaded.text

        with OpenAI(
            api_key=gateway.api_key,
            base_url=gateway.base_url,
            preparation_base_url=preparation.base_url,
            preparation_api_key=preparation.api_key,
            background_inventory_refill=False,
        ) as client:
            required = client.prepared_rows_for_response(prompt, 1, model=model_id)
            assert required > 32 * 16
            client.preprocess(model_id, count=required)
            inventory_id = client.prepared_inventory_status(model_id)["id"]
            response = client.responses.create(
                model=model_id,
                input=prompt,
                max_output_tokens=1,
                temperature=0,
            )
            assert response.status == "completed"
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
            retired = httpx.get(
                f"{gateway.base_url}/v1/runtime/inventories/{inventory_id}",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
            )
            assert retired.status_code == 404
            replacement = client.preprocess(model_id, count=required)
            assert replacement["generated"] == required
            assert client.prepared_inventory_status(model_id)["id"] != inventory_id
    finally:
        gateway.close()
        preparation.close()


def _legacy_activation_without_session_authorization_never_starts_gemm(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "no-permit", num_hidden_layers=1, ple_dim=0
    )
    engine = MaskedTransformerEngine(threads=1)
    model_id = "tiny-no-permit"
    engine_name = engine.capabilities.name
    app = create_app(
        GatewayConfig(
            api_keys=("client",),
            provider_push_api_key="push",
            rendezvous_timeout_seconds=0.01,
            prepared_session_capacity=1,
            prepared_session_idle_seconds=60,
            engine_models=({
                "engine": engine_name,
                "kind": "huggingface",
                "path": str(root),
                "model_id": model_id,
            },),
        ),
        engines={engine_name: engine},
    )
    headers = {"Authorization": "Bearer client"}
    with TestClient(app) as client:
        session_response = client.post(
            "/v1/runtime/sessions",
            headers=headers,
            json={"model": model_id, "execution": "seeded-preparation", "max_output_tokens": 1},
        )
        assert session_response.status_code == 200, session_response.text
        session_id = session_response.json()["id"]
        capacity = client.post(
            "/v1/runtime/sessions",
            headers=headers,
            json={"model": model_id, "execution": "seeded-preparation"},
        )
        assert capacity.status_code == 503
        model = engine.models[model_id]
        stage = next(
            item for item in model.manifest.stages
            if item.id not in {"token_lookup", "lm_head"}
        )
        runtime = model.stages[stage.id]
        profile = runtime.seeded_profile
        activation_request = MaskedStageRequest(
            model=model_id,
            stage_id=stage.id,
            correlation_id="7" * 32,
            masked_input=np.zeros((1, stage.in_features), dtype=np.uint32),
            activation_scales=np.ones(1, dtype=np.float32),
            modulus=profile.modulus,
            wire_bits=profile.wire_bits,
            ring=profile.ring,
            body_fingerprint=str(model.manifest.metadata["body_fingerprint"]),
            weight_digest=runtime.weight_digest,
            weight_bits=stage.weight_bits,
            activation_bits=stage.activation_bits,
            session_id=session_id,
            out_features=stage.out_features,
            signed_output_bound=profile.signed_output_bound,
        )
        activation = activation_request.pack()
        calls_before = runtime.calls
        response = client.post(
            f"/v1/runtime/sessions/{session_id}/stages/{stage.id}",
            headers=headers,
            content=encode_length_prefixed([activation]),
        )
        assert response.status_code == 400
        assert "not authorized" in response.text
        assert runtime.calls == calls_before

        oversized = client.post(
            f"/v1/runtime/sessions/{session_id}/stages/{stage.id}",
            headers={**headers, "Content-Length": str(268_435_457)},
            content=b"x",
        )
        assert oversized.status_code == 413
        authorization = engine.seeded_session_authorization(
            model_id,
            session_id,
            session_response.json()["preparation_authorization"]["max_attempts"],
        )
        authorization_payload = authorization.pack()
        rejected_authorization = client.post(
            f"/v1/runtime/sessions/{session_id}/authorize",
            headers=headers,
            content=authorization_payload,
        )
        assert rejected_authorization.status_code == 401
        oversized_authorization = client.post(
            f"/v1/runtime/sessions/{session_id}/authorize",
            headers={"Authorization": "Bearer push", "Content-Length": "16385"},
            content=b"x",
        )
        assert oversized_authorization.status_code == 413
        mismatched_authorization = client.post(
            f"/v1/runtime/sessions/{session_id}/authorize",
            headers={"Authorization": "Bearer push"},
            content=SessionAuthorization(
                session_id=session_id,
                model=model_id,
                body_fingerprint=authorization.body_fingerprint,
                stage_commitment="wrong",
                weight_bits=authorization.weight_bits,
                activation_bits=authorization.activation_bits,
                max_attempts=authorization.max_attempts,
            ).pack(),
        )
        assert mismatched_authorization.status_code == 409
        authorized = client.post(
            f"/v1/runtime/sessions/{session_id}/authorize",
            headers={"Authorization": "Bearer push"},
            content=authorization_payload,
        )
        assert authorized.status_code == 200, authorized.text

        with client.websocket_connect(
            "/v1/runtime/corrections/ws",
            headers={"Authorization": "Bearer push"},
            subprotocols=[CORRECTION_CHANNEL_SUBPROTOCOL],
        ) as corrections:
            correction = CorrectionPush(
                "9" * 32,
                session_id,
                model_id,
                str(model.manifest.metadata["body_fingerprint"]),
                stage.id,
                runtime.weight_digest,
                1,
                stage.in_features,
                stage.out_features,
                stage.weight_bits,
                stage.activation_bits,
                profile.signed_output_bound,
                profile.ring,
                profile.modulus,
                profile.wire_bits,
                np.zeros((1, stage.out_features), dtype=np.uint32),
            )
            correction_payload = correction.pack()
            frame = correction_payload
            corrections.send_bytes(frame)
            assert PreparationAck.unpack(corrections.receive_bytes()).attempt_id == correction.attempt_id
            corrections.send_bytes(frame)
            with pytest.raises(WebSocketDisconnect) as replayed_correction:
                corrections.receive_bytes()
            assert replayed_correction.value.code == 4409
        replayed = client.post(
            f"/v1/runtime/sessions/{session_id}/authorize",
            headers={"Authorization": "Bearer push"},
            content=authorization_payload,
        )
        assert replayed.status_code == 409
        assert "already consumed" in replayed.text
        mismatched_activation = client.post(
            f"/v1/runtime/sessions/{session_id}/stages/{stage.id}",
            headers=headers,
            content=encode_length_prefixed(
                [replace(activation_request, weight_digest="wrong").pack()]
            ),
        )
        assert mismatched_activation.status_code == 400
        assert "stage metadata mismatch" in mismatched_activation.text
        assert runtime.calls == calls_before

        race_attempt = "a" * 32
        race_activation = replace(activation_request, correlation_id=race_attempt)
        race_correction = replace(correction, attempt_id=race_attempt)
        pushed = client.post(
            f"/v1/runtime/sessions/{session_id}/corrections/{race_attempt}",
            headers={"Authorization": "Bearer push"},
            content=race_correction.pack(),
        )
        assert pushed.status_code == 200
        execute_stage = engine.execute_stage

        async def execute_then_cancel(model, stage_spec, payloads):
            results = await execute_stage(model, stage_spec, payloads)
            app.state.sessions[session_id].canceled = True
            app.state.correction_rendezvous.terminal(session_id)
            return results

        engine.execute_stage = execute_then_cancel
        raced = client.post(
            f"/v1/runtime/sessions/{session_id}/stages/{stage.id}",
            headers=headers,
            content=encode_length_prefixed([race_activation.pack()]),
        )
        assert raced.status_code == 400
        assert "already terminal" in raced.text
        unknown = client.post(
            "/v1/runtime/sessions/unknown/authorize",
            headers={"Authorization": "Bearer push"},
            content=authorization_payload,
        )
        assert unknown.status_code == 409
        oversized_correction = client.post(
            f"/v1/runtime/sessions/{session_id}/corrections/{'8' * 32}",
            headers={
                "Authorization": "Bearer push",
                "Content-Length": str(268_435_457),
            },
            content=b"x",
        )
        assert oversized_correction.status_code == 413
        completed = client.post(
            f"/v1/runtime/sessions/{session_id}/complete",
            headers=headers,
            json={"usage": {}},
        )
        assert completed.status_code == 200
        terminal = client.post(
            f"/v1/runtime/sessions/{session_id}/authorize",
            headers={"Authorization": "Bearer push"},
            content=authorization_payload,
        )
        assert terminal.status_code == 409
        assert "not active" in terminal.text
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers=headers).status_code == 200

        stale = client.post(
            "/v1/runtime/sessions",
            headers=headers,
            json={"model": model_id, "execution": "seeded-preparation"},
        ).json()
        app.state.sessions[stale["id"]].last_active -= 61
        replacement = client.post(
            "/v1/runtime/sessions",
            headers=headers,
            json={"model": model_id, "execution": "seeded-preparation"},
        )
        assert replacement.status_code == 200
        assert stale["id"] not in app.state.sessions
        assert app.state.correction_rendezvous.stats()["sessions"] == 1


def test_preparation_rejects_mismatched_body_weights(tmp_path: Path):
    roots = [
        create_tiny_gemma4_checkpoint(tmp_path / "primary", seed=17),
        create_tiny_gemma4_checkpoint(tmp_path / "secondary", seed=18),
    ]
    engine = MaskedTransformerEngine(threads=1)
    gateway = start_gateway(engines={engine.capabilities.name: engine})
    preparation, _ = prepared_service(roots[1], "tiny-mismatch", gateway)
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            loaded = admin.post(
                "/v1/runtime/models/load",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(roots[0]),
                    "model_id": "tiny-mismatch",
                },
            )
            assert loaded.status_code == 200, loaded.text
            legacy = admin.post(
                "/v1/runtime/sessions",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={"model": "tiny-mismatch"},
            )
            assert legacy.status_code == 410
            assert legacy.json()["detail"]["error"]["code"] == "seeded_preparation_required"
        with OpenAI(
            api_key=gateway.api_key,
            base_url=gateway.base_url,
            preparation_base_url=preparation.base_url,
            preparation_api_key=preparation.api_key,
        ) as client:
            with pytest.raises(ProtocolError, match="commitments do not match"):
                client.responses.create(
                    model="tiny-mismatch",
                    input="mismatched providers",
                    max_output_tokens=1,
                    temperature=0,
                )
    finally:
        gateway.close()
        preparation.close()


def test_client_authorization_failure_burns_inference_session(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "authorization-failure")
    model_id = "tiny-authorization-failure"
    inference_engine = MaskedTransformerEngine(threads=1)
    preparation_engine = MaskedTransformerEngine(threads=1)
    asyncio.run(preparation_engine.load(load_hf_directory(root, model_id=model_id)))
    gateway = start_gateway(
        engines={inference_engine.capabilities.name: inference_engine}
    )
    preparation = start_preparation(
        preparation_engine,
        gateway.base_url,
        "wrong-provider-push-key",
    )
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            loaded = admin.post(
                "/v1/runtime/models/load",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={
                    "engine": inference_engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": model_id,
                },
            )
            assert loaded.status_code == 200, loaded.text
        with OpenAI(
            api_key=gateway.api_key,
            base_url=gateway.base_url,
            preparation_base_url=preparation.base_url,
            preparation_api_key=preparation.api_key,
        ) as client:
            with pytest.raises(ProtocolError, match="401 Unauthorized"):
                client.preprocess(model_id, count=64)
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            metrics = admin.get(
                "/metrics",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
            ).json()
        assert metrics["sessions"]["active"] == 0
        assert metrics["correction_rendezvous"]["sessions"] == 0
    finally:
        gateway.close()
        preparation.close()


def test_previous_response_id_reuses_private_kv_and_token_cache(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "tiny-continuation",
        num_hidden_layers=1,
        ple_dim=4,
    )
    engine = MaskedTransformerEngine(threads=1)
    gateway = start_gateway(engines={engine.capabilities.name: engine})
    preparation, _ = prepared_service(root, "tiny-continuation-pllm", gateway)
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            loaded = admin.post(
                "/v1/runtime/models/load",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": "tiny-continuation-pllm",
                },
            )
            assert loaded.status_code == 200, loaded.text

        with OpenAI(
            api_key=gateway.api_key,
            base_url=gateway.base_url,
            preparation_base_url=preparation.base_url,
            preparation_api_key=preparation.api_key,
            correlation_mode="local-test",
            correlation_prefetch=1,
            token_cache_size=128,
            prepared_inventory_rows=256,
            background_inventory_refill=False,
        ) as client:
            client.preprocess("tiny-continuation-pllm", count=256)
            first = client.responses.create(
                model="tiny-continuation-pllm",
                input="repeat repeat repeat",
                max_output_tokens=1,
                temperature=0,
            )
            before_qkv_rows = engine.models["tiny-continuation-pllm"].stages[
                "layers.0.self_attn.qkv_proj"
            ].rows
            # The final emitted token is carried into a future continuation
            # instead of paying for a transformer pass after the response ends.
            assert before_qkv_rows == first.usage.input_tokens
            second = client.responses.create(
                model="tiny-continuation-pllm",
                previous_response_id=first.id,
                input="repeat again",
                max_output_tokens=1,
                temperature=0,
            )
            after_qkv_rows = engine.models["tiny-continuation-pllm"].stages[
                "layers.0.self_attn.qkv_proj"
            ].rows
            audit = client.privacy_audit.to_dict()
            assert second.status == "completed"
            assert audit["kv_continuation_hits"] == 1
            assert audit["kv_continuation_misses"] == 0
            assert audit["token_lookup_cache_hits"] == 0
            assert engine.models["tiny-continuation-pllm"].stages["token_lookup"].calls == 0
            assert engine.models["tiny-continuation-pllm"].stages["lm_head"].calls == 0
            # The continuation executes only the newly appended template suffix,
            # not the full context represented by the public usage count.
            assert after_qkv_rows - before_qkv_rows < second.usage.input_tokens
    finally:
        gateway.close()
        preparation.close()


def test_official_sdk_factory_forwards_transformer_cache_options(monkeypatch):
    import sys
    from types import SimpleNamespace
    import pllm.runtime.official as official

    captured = {}

    class FakeTransport:
        def __init__(self, **kwargs):
            captured["transport"] = kwargs

    class FakeOfficialClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

    monkeypatch.setattr(official, "PLLMTransport", FakeTransport)
    monkeypatch.setattr(official._httpx, "Client", lambda *, transport: ("http-client", transport))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOfficialClient))
    result = official.create_openai_client(
        gateway_url="http://gateway",
        gateway_api_key="secret",
        correlation_prefetch=9,
        token_cache_size=2048,
        bundle_cache_mode="refresh",
        bundle_cache_dir="/tmp/pllm-test-bundles",
    )
    assert isinstance(result, FakeOfficialClient)
    assert captured["transport"]["correlation_prefetch"] == 9
    assert captured["transport"]["token_cache_size"] == 2048
    assert captured["transport"]["bundle_cache_mode"] == "refresh"
    assert captured["transport"]["bundle_cache_dir"] == "/tmp/pllm-test-bundles"
    assert captured["client"]["http_client"][0] == "http-client"
