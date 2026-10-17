from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi.testclient import TestClient

from pllm.runtime.client import ProtocolError
from pllm.runtime.sidecar import _response_stream, create_sidecar_app


AUTH = {"Authorization": "Bearer local", "Content-Type": "application/json"}


def _response(*, output: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": 123.9,
        "status": "completed",
        "model": "private-model",
        "output": output
        if output is not None
        else [
            {
                "id": "msg_123",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_tokens_details": {"cached_tokens": 1},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class FakeStream:
    def __init__(self, events: list[Any]) -> None:
        self.events = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.events)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self) -> None:
        self.closed = True


class FakeRuntimeClient:
    default_model = None
    preparation_http = None

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.next_result: Any = _response()

    def create(self, body: dict[str, Any]) -> Any:
        self.created.append(body)
        return self.next_result

    def list_models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "private-model", "created": 2.7}]}

    def retrieve(self, response_id: str) -> dict[str, Any]:
        if response_id == "missing":
            raise ProtocolError("Response not found", 404)
        return {**_response(), "id": response_id}

    def cancel(self, response_id: str) -> dict[str, Any]:
        return {**_response(), "id": response_id, "status": "cancelled"}


class EphemeralRuntimeClient(FakeRuntimeClient):
    def __init__(self) -> None:
        super().__init__()
        self.saved = {
            **_response(
                output=[
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": "{}",
                    }
                ]
            ),
            "store": False,
        }
        self.next_result = self.saved

    def retrieve(self, response_id: str) -> dict[str, Any]:
        raise KeyError(response_id)

    def continuation_response(self, response_id: str) -> dict[str, Any]:
        if response_id != self.saved["id"]:
            raise KeyError(response_id)
        return self.saved


def _client(runtime: FakeRuntimeClient | None = None) -> tuple[TestClient, FakeRuntimeClient]:
    fake = runtime or FakeRuntimeClient()
    app = create_sidecar_app(
        remote_base_url="http://unused",
        remote_api_key="unused",
        local_api_key="local",
        client=fake,  # type: ignore[arg-type]
    )
    return TestClient(app, raise_server_exceptions=False), fake


def test_store_false_is_hidden_but_available_for_continuation() -> None:
    runtime = EphemeralRuntimeClient()
    client, _ = _client(runtime)
    created = client.post(
        "/v1/responses",
        headers=AUTH,
        json={"model": "private-model", "input": "call it", "store": False},
    )
    assert created.status_code == 200
    assert client.get("/v1/responses/resp_123", headers=AUTH).status_code == 404

    continued = client.post(
        "/v1/responses",
        headers=AUTH,
        json={
            "model": "private-model",
            "previous_response_id": "resp_123",
            "input": [{"type": "function_call_output", "call_id": "call_1", "output": "ok"}],
        },
    )
    assert continued.status_code == 200


def _sse(text: str) -> list[tuple[str | None, Any]]:
    output: list[tuple[str | None, Any]] = []
    for block in text.strip().split("\n\n"):
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        assert data is not None
        output.append((event, data if data == "[DONE]" else json.loads(data)))
    return output


def test_responses_nonstream_forwards_body_and_integer_timestamp():
    client, runtime = _client()
    body = {
        "model": "private-model",
        "input": [{"role": "user", "content": "secret"}],
        "temperature": 0.2,
        "parallel_tool_calls": False,
    }
    with client:
        response = client.post("/v1/responses", headers=AUTH, json=body)
    assert response.status_code == 200
    assert runtime.created == [body]
    assert response.json()["created_at"] == 123
    assert response.json()["output"][0]["content"][0]["text"] == "hello"


def test_responses_stream_is_named_sse_and_closes():
    runtime = FakeRuntimeClient()
    runtime.next_result = FakeStream(
        [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**_response(), "created_at": 123.8},
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "hello",
            },
            {
                "type": "response.completed",
                "sequence_number": 2,
                "response": _response(),
            },
        ]
    )
    client, _ = _client(runtime)
    with client:
        response = client.post(
            "/v1/responses",
            headers=AUTH,
            json={"model": "private-model", "input": "secret", "stream": True},
        )
    events = _sse(response.text)
    assert [item[0] for item in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
        None,
    ]
    assert events[0][1]["response"]["created_at"] == 123
    assert events[-1][1] == "[DONE]"
    assert runtime.next_result.closed is True


def test_responses_stream_emits_error_then_done():
    runtime = FakeRuntimeClient()
    runtime.next_result = FakeStream([RuntimeError("generation failed")])
    client, _ = _client(runtime)
    with client:
        response = client.post(
            "/v1/responses",
            headers=AUTH,
            json={"model": "private-model", "input": "secret", "stream": True},
        )
    events = _sse(response.text)
    assert events[0] == (
        "error",
        {
            "type": "error",
            "sequence_number": 0,
            "error": {
                "message": "generation failed",
                "type": "server_error",
                "param": None,
                "code": "runtime_error",
            },
        },
    )
    assert events[-1] == (None, "[DONE]")


def test_responses_stream_clean_eof_is_not_reported_as_success():
    runtime = FakeRuntimeClient()
    runtime.next_result = FakeStream([])
    client, _ = _client(runtime)
    with client:
        response = client.post(
            "/v1/responses",
            headers=AUTH,
            json={"model": "private-model", "input": "secret", "stream": True},
        )
    events = _sse(response.text)
    assert events[0][0] == "error"
    assert events[0][1]["error"]["code"] == "runtime_error"
    assert events[-1] == (None, "[DONE]")


def test_response_stream_close_on_consumer_disconnect():
    stream = FakeStream([{"type": "response.created", "sequence_number": 0}])
    generator = _response_stream(stream)
    next(generator)
    generator.close()
    assert stream.closed is True


def test_models_retrieve_cancel_and_error_envelopes():
    client, _ = _client()
    with client:
        assert client.get("/v1/models").status_code == 401
        models = client.get("/v1/models", headers=AUTH)
        retrieved = client.get("/v1/responses/resp_saved", headers=AUTH)
        cancelled = client.post("/v1/responses/resp_saved/cancel", headers=AUTH)
        missing = client.get("/v1/responses/missing", headers=AUTH)
    assert models.json()["data"][0]["created"] == 2
    assert retrieved.json()["id"] == "resp_saved"
    assert cancelled.json()["status"] == "cancelled"
    assert missing.status_code == 404
    assert missing.json()["error"]["message"] == "Response not found"
    assert missing.json()["error"]["type"] == "invalid_request_error"


def test_json_and_request_validation_errors_are_stable():
    client, _ = _client()
    with client:
        wrong_type = client.post(
            "/v1/responses", headers={**AUTH, "Content-Type": "text/plain"}, content="{}"
        )
        malformed = client.post("/v1/responses", headers=AUTH, content="{")
        array = client.post("/v1/responses", headers=AUTH, content="[]")
        stream = client.post(
            "/v1/responses",
            headers=AUTH,
            json={"model": "m", "input": "x", "stream": "yes"},
        )
        unknown = client.post(
            "/v1/responses", headers=AUTH, json={"model": "m", "input": "x", "seed": 1}
        )
    assert wrong_type.status_code == 415
    assert malformed.json()["error"]["code"] == "invalid_json"
    assert array.json()["error"]["message"] == "Request body must be a JSON object"
    assert stream.json()["error"]["param"] == "stream"
    assert unknown.json()["error"]["message"] == "Unsupported field: seed"


def test_compact_is_local_normalized_and_bounded():
    client, runtime = _client()
    input_items = [{"role": "user", "content": f"message {index}"} for index in range(70)]
    with client:
        response = client.post(
            "/v1/responses/compact",
            headers=AUTH,
            json={"model": "private-model", "instructions": "Keep policy", "input": input_items},
        )
    value = response.json()
    assert response.status_code == 200
    assert runtime.created == []
    assert value["object"] == "response.compaction"
    assert isinstance(value["created_at"], int)
    assert len(value["output"]) == 64
    assert value["output"][0]["content"] == [{"type": "input_text", "text": "Keep policy"}]
    assert "9 oldest items elided" in value["output"][1]["content"][0]["text"]
    assert value["output"][-2]["content"] == [{"type": "input_text", "text": "message 69"}]
    assert value["output"][-1]["type"] == "compaction"
    assert value["usage"]["total_tokens"] == 0


def test_compact_keeps_function_call_context_with_its_output() -> None:
    runtime = FakeRuntimeClient()
    client = TestClient(
        create_sidecar_app(client=runtime, remote_base_url="unused", remote_api_key="unused")
    )
    items = [{"role": "user", "content": f"before {index}"} for index in range(45)]
    items.extend(
        [
            {"type": "function_call", "call_id": "call_pair", "name": "lookup", "arguments": "{}"},
            {"role": "assistant", "content": "Calling the lookup now."},
            {"type": "function_call_output", "call_id": "call_pair", "output": "result"},
        ]
    )
    items.extend({"role": "user", "content": f"after {index}"} for index in range(25))
    with client:
        value = client.post(
            "/v1/responses/compact",
            headers=AUTH,
            json={"model": "gateway-model", "input": items},
        ).json()
    retained_types = [item.get("type") for item in value["output"]]
    assert ("function_call" in retained_types) == ("function_call_output" in retained_types)
    if "function_call" in retained_types:
        call_index = retained_types.index("function_call")
        output_index = retained_types.index("function_call_output")
        assert any(
            item.get("role") == "assistant"
            for item in value["output"][call_index + 1 : output_index]
        )


def test_nonstream_generation_does_not_block_health() -> None:
    class BlockingRuntime(FakeRuntimeClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def create(self, body):
            self.started.set()
            assert self.release.wait(timeout=2)
            return super().create(body)

    runtime = BlockingRuntime()
    app = create_sidecar_app(client=runtime, remote_base_url="unused", remote_api_key="unused")
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        response = pool.submit(
            client.post,
            "/v1/responses",
            headers=AUTH,
            json={"model": "gateway-model", "input": "hello"},
        )
        assert runtime.started.wait(timeout=1)
        health = pool.submit(client.get, "/healthz")
        assert health.result(timeout=1).status_code == 200
        runtime.release.set()
        assert response.result(timeout=2).status_code == 200


def test_websocket_failed_continuation_does_not_evict_stored_response() -> None:
    class StoredRuntime(FakeRuntimeClient):
        def __init__(self) -> None:
            super().__init__()
            self.evicted: list[str] = []

        def continuation_response(self, response_id: str):
            assert response_id == "resp_stored"
            return {
                **_response(
                    output=[
                        {
                            "id": "fc_1",
                            "type": "function_call",
                            "status": "completed",
                            "call_id": "call_expected",
                            "name": "lookup",
                            "arguments": "{}",
                        }
                    ]
                ),
                "store": True,
            }

        def evict_response(self, response_id: str) -> None:
            self.evicted.append(response_id)

    runtime = StoredRuntime()
    app = create_sidecar_app(client=runtime, remote_base_url="unused", remote_api_key="unused")
    with TestClient(app) as client:
        with client.websocket_connect("/v1/responses", headers=AUTH) as websocket:
            websocket.send_json(
                {
                    "type": "response.create",
                    "model": "gateway-model",
                    "previous_response_id": "resp_stored",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "wrong_call",
                            "output": "result",
                        }
                    ],
                }
            )
            assert websocket.receive_json()["type"] == "error"
    assert runtime.evicted == []


def test_chat_nonstream_maps_messages_tools_output_and_usage():
    runtime = FakeRuntimeClient()
    runtime.next_result = _response(
        output=[
            {
                "id": "msg_answer",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "calling"}],
            },
            {
                "id": "fc_internal",
                "type": "function_call",
                "call_id": "call_weather",
                "name": "weather",
                "arguments": '{"city":"Oslo"}',
            },
        ]
    )
    client, _ = _client(runtime)
    request = {
        "model": "private-model",
        "messages": [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_old", "content": "sunny"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "weather"}},
        "max_completion_tokens": 12,
        "temperature": 0.1,
        "top_p": 0.8,
        "parallel_tool_calls": False,
    }
    with client:
        response = client.post("/v1/chat/completions", headers=AUTH, json=request)
    forwarded = runtime.created[0]
    assert forwarded["max_output_tokens"] == 12
    assert forwarded["input"][2] == {
        "type": "function_call",
        "call_id": "call_old",
        "name": "weather",
        "arguments": '{"city":"Paris"}',
    }
    assert forwarded["input"][3] == {
        "type": "function_call_output",
        "call_id": "call_old",
        "output": "sunny",
    }
    assert forwarded["tools"][0]["name"] == "weather"
    assert forwarded["tool_choice"] == {"type": "function", "name": "weather"}
    value = response.json()
    assert value["id"] == "resp_123"
    assert value["created"] == 123
    assert value["choices"][0]["message"]["content"] == "calling"
    assert value["choices"][0]["message"]["tool_calls"][0]["id"] == "call_weather"
    assert value["choices"][0]["finish_reason"] == "tool_calls"
    assert value["usage"]["prompt_tokens"] == 3


def test_chat_stream_maps_text_function_arguments_and_done():
    runtime = FakeRuntimeClient()
    final = _response(
        output=[
            {
                "id": "fc_internal",
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"x"}',
            }
        ]
    )
    runtime.next_result = FakeStream(
        [
            {"type": "response.created", "response": {**_response(), "output": []}},
            {"type": "response.output_text.delta", "delta": "thinking"},
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_internal",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_internal",
                "delta": '{"q":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_internal",
                "delta": '"x"}',
            },
            {"type": "response.completed", "response": final},
        ]
    )
    client, _ = _client(runtime)
    with client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "private-model",
                "messages": [{"role": "user", "content": "lookup"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    events = _sse(response.text)
    chunks = [item[1] for item in events[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["id"] == "resp_123"
    assert chunks[1]["choices"][0]["delta"]["content"] == "thinking"
    assert chunks[2]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"
    arguments = "".join(
        chunk["choices"][0]["delta"]
        .get("tool_calls", [{}])[0]
        .get("function", {})
        .get("arguments", "")
        for chunk in chunks[3:5]
    )
    assert arguments == '{"q":"x"}'
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 2
    assert events[-1] == (None, "[DONE]")
    assert runtime.next_result.closed is True


def test_chat_rejects_unsupported_controls_and_invalid_stream():
    client, runtime = _client()
    with client:
        unknown = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "m", "messages": [{"role": "user", "content": "x"}], "n": 2},
        )
        conflict = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 2,
                "max_completion_tokens": 3,
            },
        )
        stream = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "stream": 1,
            },
        )
    assert unknown.json()["error"]["message"] == "Unsupported field: n"
    assert conflict.json()["error"]["param"] == "max_completion_tokens"
    assert stream.json()["error"]["param"] == "stream"
    assert runtime.created == []
