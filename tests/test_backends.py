from __future__ import annotations

import json

import httpx
import pytest

from pllm.runtime.backends import LlamaCppAdapter, MLXLMAdapter, OllamaAdapter, VLLMAdapter
from pllm.runtime.backends.registry import BackendRegistry
from pllm.runtime.backends.translation import chat_completion_to_response, responses_to_chat


def _sse(events: list[dict]) -> bytes:
    return b"".join(
        f"data: {json.dumps(event)}\n\n".encode() for event in events
    ) + b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_vllm_native_responses_and_models():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "gemma", "owned_by": "vllm"}]})
        if request.url.path == "/v1/responses":
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "id": "resp_vllm", "object": "response", "status": "completed",
                "model": body["model"], "output": [], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })
        raise AssertionError(request.url)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = VLLMAdapter(base_url="http://backend", client=client)
    try:
        models = await adapter.list_models()
        response = await adapter.create_response({"model": "gemma", "input": "x"})
        assert models[0].privacy_mode == "trusted_backend"
        assert response["id"] == "resp_vllm"
        assert adapter.capabilities.responses is True
        assert adapter.capabilities.private_runtime is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ollama_strips_stateful_fields():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen"}]})
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"id": "r", "object": "response", "status": "completed", "output": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OllamaAdapter(base_url="http://backend", client=client)
    try:
        await adapter.create_response({
            "model": "qwen", "input": "x", "previous_response_id": "resp_old", "conversation": "conv_1"
        })
        assert "previous_response_id" not in bodies[0]
        assert "conversation" not in bodies[0]
        assert adapter.capabilities.stateful_responses is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_llamacpp_falls_back_to_chat_completions():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/v1/responses":
            return httpx.Response(404, json={"error": "unsupported"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={
                "id": "chat_1", "created": 1,
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            })
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "local"}]})
        raise AssertionError(request.url)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = LlamaCppAdapter(base_url="http://backend", client=client)
    try:
        result = await adapter.create_response({"model": "local", "input": "say hello"})
        assert paths == ["/v1/responses", "/v1/chat/completions"]
        assert result["output"][0]["content"][0]["text"] == "hello"
        assert result["usage"]["total_tokens"] == 4
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_mlx_translation_nonstream_and_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "mlx-model"}]})
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            if body.get("stream"):
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse([
                    {"choices": [{"delta": {"content": "he"}}]},
                    {"choices": [{"delta": {"content": "llo"}}]},
                ]))
            return httpx.Response(200, json={
                "id": "chat_mlx", "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            })
        raise AssertionError(request.url)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = MLXLMAdapter(base_url="http://backend", client=client)
    try:
        result = await adapter.create_response({"model": "mlx-model", "input": "x"})
        assert result["output"][0]["content"][0]["text"] == "hello"
        events = [event async for event in adapter.stream_response({"model": "mlx-model", "input": "x"})]
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"
        assert "".join(event.get("delta", "") for event in events) == "hello"
    finally:
        await client.aclose()


def test_responses_chat_translation_tools_and_text():
    chat = responses_to_chat({
        "model": "m",
        "instructions": "system",
        "input": [
            {"role": "user", "content": "hello"},
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        ],
        "max_output_tokens": 9,
        "tools": [{"type": "function", "function": {"name": "f"}}],
    })
    assert chat["messages"][0] == {"role": "system", "content": "system"}
    assert chat["messages"][-1]["role"] == "tool"
    assert chat["max_tokens"] == 9

    response = chat_completion_to_response({
        "choices": [{"message": {
            "content": "ok",
            "tool_calls": [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}],
        }}]
    }, model="m")
    assert [item["type"] for item in response["output"]] == ["message", "function_call"]


@pytest.mark.asyncio
async def test_backend_registry_routes_duplicate_model_ids():
    class Adapter:
        capabilities = VLLMAdapter.capabilities
        def __init__(self, name): self.name = name
        async def list_models(self):
            from pllm.runtime.backends.base import BackendModel
            return [BackendModel("same", self.name, self.name, "trusted_backend")]
        async def create_response(self, body): return body
        async def stream_response(self, body):
            if False: yield body

    registry = BackendRegistry()
    registry.add("a", Adapter("a"))
    registry.add("b", Adapter("b"))
    models = await registry.refresh()
    assert [m.id for m in models] == ["same", "b/same"]
    assert registry.adapter_for("same").name == "a"
    assert registry.adapter_for("b/same").name == "b"
