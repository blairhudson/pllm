from __future__ import annotations

import json

import httpx
import pytest

from pllm.runtime import OpenAI
from pllm.runtime.client import ProtocolError


def test_direct_sdk_nonstream_http(gateway):
    with OpenAI(
        base_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
        correlation_prefetch=16,
        session_transport="http",
    ) as client:
        response = client.responses.create(
            model="pllm-bigram-demo",
            input="TOP_SECRET_CANARY_73bce1",
            max_output_tokens=32,
        )
        assert response.output_text == "private\n"
        assert response.status == "completed"
        assert response.model == "pllm-bigram-demo"
        assert response.usage.output_tokens == len("private\n")
        assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
        assert client.privacy_audit.plaintext_token_ids_sent == 0
        assert client.privacy_audit.online_steps == len("private\n") + 1


def test_direct_sdk_stream_websocket(gateway):
    with OpenAI(
        base_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
        correlation_prefetch=16,
        session_transport="websocket",
    ) as client:
        stream = client.responses.create(
            model="pllm-bigram-demo",
            input="secret websocket prompt",
            max_output_tokens=32,
            stream=True,
        )
        events = list(stream)
        assert events[0].type == "response.created"
        assert events[-1].type == "response.completed"
        assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == "private\n"
        assert [event.sequence_number for event in events] == list(range(len(events)))


def test_previous_response_id_is_client_private(gateway):
    canary = "PREVIOUS_RESPONSE_CANARY_4f14"
    with OpenAI(
        base_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
        correlation_prefetch=16,
    ) as client:
        first = client.responses.create(model="pllm-bigram-demo", input=canary, max_output_tokens=32)
        second = client.responses.create(
            model="pllm-bigram-demo",
            input="follow up",
            previous_response_id=first.id,
            max_output_tokens=32,
        )
        assert second.previous_response_id == first.id
        assert second.output_text == "private\n"
        payloads = b"\n".join(payload for _, payload in gateway.audit)
        assert canary.encode() not in payloads
        assert b"follow up" not in payloads


def test_unknown_previous_response_id_rejected_locally(gateway):
    with OpenAI(
        base_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
    ) as client:
        with pytest.raises(ProtocolError, match="unknown previous_response_id"):
            client.responses.create(
                model="pllm-bigram-demo",
                input="x",
                previous_response_id="resp_missing",
            )


def test_retrieve_and_cancel(gateway):
    with OpenAI(
        base_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
        correlation_prefetch=16,
    ) as client:
        response = client.responses.create(model="pllm-bigram-demo", input="x", max_output_tokens=32)
        retrieved = client.responses.retrieve(response.id)
        assert retrieved.output_text == "private\n"
        cancelled = client.responses.cancel(response.id)
        assert cancelled.status == "cancelled"


def test_remote_plaintext_responses_rejected_for_private_model(gateway):
    response = httpx.post(
        f"{gateway.base_url}/v1/responses",
        headers={"Authorization": f"Bearer {gateway.api_key}"},
        json={"model": "pllm-bigram-demo", "input": "this must not be accepted"},
    )
    assert response.status_code == 426
    assert response.headers["Upgrade"] == "pllm-runtime/1"
    assert response.json()["detail"]["error"]["code"] == "runtime_client_required"


def test_models_capabilities_and_metrics(gateway):
    headers = {"Authorization": f"Bearer {gateway.api_key}"}
    models = httpx.get(f"{gateway.base_url}/v1/models", headers=headers).json()
    demo = next(item for item in models["data"] if item["id"] == "pllm-bigram-demo")
    assert demo["runtime"]["privacy_mode"] == "preprocessed"

    capabilities = httpx.get(f"{gateway.base_url}/v1/runtime/capabilities", headers=headers).json()
    assert capabilities["responses_api"] is True
    assert "websocket-binary" in capabilities["client_transports"]
    assert "local-test" in capabilities["correlation_modes"]

    with OpenAI(
        base_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
        correlation_prefetch=16,
    ) as client:
        client.responses.create(model="pllm-bigram-demo", input="x", max_output_tokens=32)

    assert httpx.get(f"{gateway.base_url}/metrics").status_code == 401
    metrics = httpx.get(f"{gateway.base_url}/metrics", headers=headers).json()
    assert metrics["sessions"]["total"] >= 1
    assert metrics["sessions"]["online_steps"] >= 9
    assert metrics["stage_schedulers"]["pllm-bigram-demo"]["items"] >= 9


def test_authentication_failure(gateway):
    response = httpx.get(
        f"{gateway.base_url}/v1/models",
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401


def test_server_audit_contains_no_prompt_or_output(gateway):
    prompt = "UNIQUE_PROMPT_CANARY_8c9964"
    with OpenAI(
        base_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
        correlation_prefetch=16,
    ) as client:
        client.responses.create(model="pllm-bigram-demo", input=prompt, max_output_tokens=32)
    captured = b"\n".join(payload for _, payload in gateway.audit)
    assert prompt.encode() not in captured
    assert b"private\n" not in captured
    # Session control plane contains only model/options, not OpenAI input.
    session_payload = next(payload for kind, payload in gateway.audit if kind == "session")
    session_json = json.loads(session_payload)
    assert set(session_json) == {"model", "max_output_tokens"}

@pytest.mark.asyncio
async def test_async_client_exposes_models_and_runtime_extensions(gateway):
    from pllm.runtime import AsyncOpenAI

    async with AsyncOpenAI(
        base_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
        correlation_prefetch=16,
    ) as client:
        models = await client.models.list()
        assert any(item["id"] == "pllm-bigram-demo" for item in models["data"])
        result = await client.runtime.preprocess(model="pllm-bigram-demo", correlations=16)
        assert result["available"] >= 16
        response = await client.responses.create(
            model="pllm-bigram-demo",
            input="ASYNC_CANARY_STAYS_LOCAL",
            max_output_tokens=32,
        )
        assert response.output_text == "private\n"
        assert client.runtime.privacy_audit.plaintext_prompt_bytes_sent == 0
