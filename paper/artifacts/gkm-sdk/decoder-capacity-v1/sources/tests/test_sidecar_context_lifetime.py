"""Legacy stream ContextVar ownership across worker resumes and retirement."""

import asyncio
from contextvars import ContextVar
import json
import threading
from types import SimpleNamespace

import anyio
from fastapi.testclient import TestClient
import pytest

from pllm.runtime import sidecar
from pllm.runtime.client import ResponseStream
from pllm.runtime.types import Response, response_events


def context_stream(*, failure=False):
    value = ContextVar("sidecar_stream_context", default="caller")
    retired = []
    response = Response.text_response(model="context-fixture", text="accepted",
                                      input_tokens=1, output_tokens=1)

    def events():
        token = value.set("stream")
        try:
            for index, event in enumerate(response_events(response, ["accepted"])):
                assert value.get() == "stream"
                if failure and index == 1:
                    raise ValueError("controlled stream failure")
                yield event
        finally:
            value.reset(token)
            retired.append(value.get())

    return ResponseStream(events()), value, retired


@pytest.mark.asyncio
@pytest.mark.parametrize("serializer", ["responses", "chat"])
@pytest.mark.parametrize("ending", ["completion", "failure", "early_close"])
async def test_stream_context_survives_resume_and_close(serializer, ending):
    source, value, retired = context_stream(failure=ending == "failure")
    iterator = (sidecar._response_stream(source) if serializer == "responses"
                else sidecar._chat_stream(source, "context-fixture", include_usage=True))
    stream = sidecar._closing_stream(iterator, source)
    chunks = [await anext(stream)]
    assert value.get() == "caller" and not retired
    try:
        if ending != "early_close":
            chunks.extend([chunk async for chunk in stream])
    finally:
        await stream.aclose()
    assert value.get() == "caller" and retired == ["caller"]
    payloads = [json.loads(line[6:]) for line in b"".join(chunks).splitlines()
                if line.startswith(b"data: ") and line != b"data: [DONE]"]
    errors = [payload["error"]["message"] for payload in payloads if "error" in payload]
    assert errors == (["controlled stream failure"] if ending == "failure" else [])
    if ending == "completion":
        assert b"data: [DONE]" in chunks[-1]


@pytest.mark.parametrize("failure", [False, True])
def test_websocket_uses_one_context_through_terminal_cleanup(failure):
    source, value, retired = context_stream(failure=failure)
    app = sidecar.create_sidecar_app(client=SimpleNamespace(create=lambda body: source))
    with TestClient(app) as client:
        with client.websocket_connect("/v1/responses", headers={"authorization": "Bearer local"}) as ws:
            ws.send_json({"type": "response.create", "model": "context-fixture", "input": "hello"})
            while True:
                event = ws.receive_json()
                if event["type"] in {"response.completed", "error"}:
                    break
            assert event["type"] == ("error" if failure else "response.completed")
            if failure:
                assert event["error"]["message"] == "controlled stream failure"
    assert value.get() == "caller" and retired == ["caller"]


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["asyncio", "anyio"])
async def test_disconnect_signals_then_drains_in_stream_context(scope):
    value = ContextVar("sidecar_cancel_context", default="caller")
    entered, cancelled = threading.Event(), threading.Event()
    retired, observed = [], []

    def events():
        token = value.set("stream")
        try:
            entered.set()
            assert cancelled.wait(2), "disconnect did not signal the executing iterator"
            yield b"unexposed"
        finally:
            value.reset(token)
            retired.append(value.get())

    iterator = events()
    source = SimpleNamespace(_cancel=cancelled.set, close=iterator.close)

    async def consume():
        async for chunk in sidecar._closing_stream(iterator, source):
            observed.append(chunk)

    if scope == "asyncio":
        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        async with anyio.create_task_group() as group:
            group.start_soon(consume)
            assert await asyncio.to_thread(entered.wait, 2)
            group.cancel_scope.cancel()
    assert cancelled.is_set() and retired == ["caller"] and not observed
    assert value.get() == "caller"
