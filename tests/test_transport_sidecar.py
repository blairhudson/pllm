from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from pllm.runtime import AsyncPLLMTransport, PLLMTransport, create_sidecar_app


def _sse_events(text: str) -> list[dict]:
    events: list[dict] = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
    return events


def test_httpx_transport_intercepts_nonstream_response(gateway):
    transport = PLLMTransport(
        gateway_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
        correlation_prefetch=16,
    )
    try:
        with httpx.Client(transport=transport, base_url="https://api.openai.invalid") as client:
            response = client.post(
                "/v1/responses",
                json={"model": "pllm-bigram-demo", "input": "secret", "max_output_tokens": 32},
            )
            assert response.status_code == 200
            assert response.headers["X-PLLM-Transport"] == "1"
            assert response.json()["output"][0]["content"][0]["text"] == "private\n"
    finally:
        transport.close()


def test_httpx_transport_synthesizes_sse(gateway):
    transport = PLLMTransport(
        gateway_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
        correlation_prefetch=16,
        session_transport="websocket",
    )
    try:
        with httpx.Client(transport=transport, base_url="https://api.openai.invalid") as client:
            with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "pllm-bigram-demo", "input": "secret", "stream": True, "max_output_tokens": 32},
            ) as response:
                body = "".join(response.iter_text())
        events = _sse_events(body)
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"
        assert "".join(event.get("delta", "") for event in events) == "private\n"
    finally:
        transport.close()


@pytest.mark.asyncio
async def test_async_transport(gateway):
    transport = AsyncPLLMTransport(
        gateway_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
        correlation_prefetch=16,
    )
    try:
        async with httpx.AsyncClient(transport=transport, base_url="https://api.openai.invalid") as client:
            response = await client.post(
                "/v1/responses",
                json={"model": "pllm-bigram-demo", "input": "secret", "max_output_tokens": 32},
            )
            assert response.json()["status"] == "completed"
    finally:
        await transport.aclose()


def test_local_sidecar_is_openai_http_compatible(gateway):
    app = create_sidecar_app(
        remote_base_url=gateway.base_url,
        remote_api_key=gateway.api_key,
        local_api_key="sidecar-key",
        correlation_mode="local-test",
        correlation_prefetch=16,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sidecar-key"},
            json={"model": "pllm-bigram-demo", "input": "local secret", "max_output_tokens": 32},
        )
        assert response.status_code == 200
        assert response.json()["output"][0]["content"][0]["text"] == "private\n"

        streamed = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sidecar-key"},
            json={"model": "pllm-bigram-demo", "input": "local secret", "max_output_tokens": 32, "stream": True},
        )
        events = _sse_events(streamed.text)
        assert events[-1]["type"] == "response.completed"


def test_sidecar_auth(gateway):
    app = create_sidecar_app(
        remote_base_url=gateway.base_url,
        remote_api_key=gateway.api_key,
        local_api_key="sidecar-key",
        correlation_mode="local-test",
    )
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 401


def test_forwarded_control_plane_uses_gateway_credential(gateway):
    transport = PLLMTransport(
        gateway_url=gateway.base_url,
        api_key=gateway.api_key,
        correlation_mode="local-test",
    )
    try:
        with httpx.Client(transport=transport, base_url="https://api.openai.invalid") as client:
            # The local placeholder auth is deliberately wrong for the gateway.
            response = client.get("/v1/models", headers={"Authorization": "Bearer local-placeholder"})
            assert response.status_code == 200
            assert any(item["id"] == "pllm-bigram-demo" for item in response.json()["data"])
    finally:
        transport.close()
