from __future__ import annotations

import threading
from collections import deque

import httpx
import pytest

from pllm.runtime.client import RuntimeClient
from pllm.runtime.types import Response


def test_remote_inference_requires_https_without_preparation() -> None:
    with pytest.raises(ValueError, match="base_url must use HTTPS"):
        RuntimeClient(base_url="http://inference.example", api_key="secret")


def test_response_cache_only_bounds_ephemeral_continuations() -> None:
    client = object.__new__(RuntimeClient)
    stored = Response.text_response(model="m", text="stored", input_tokens=1, output_tokens=1)
    stored.store = True
    first = Response.text_response(model="m", text="first", input_tokens=1, output_tokens=1)
    first.store = False
    second = Response.text_response(model="m", text="second", input_tokens=1, output_tokens=1)
    second.store = False
    client.cache = {stored.id: stored, first.id: first, second.id: second}
    client.histories = {response_id: "history" for response_id in client.cache}
    client.message_histories = {response_id: [] for response_id in client.cache}
    client._transformer_sessions = {}
    client._transformer_conversations = {}
    client._transformer_conversation_lock = threading.Lock()
    client._response_cache_order = deque()
    client._response_cache_limit = 1
    client._response_cache_lock = threading.Lock()
    client._pending_response_store = {}
    client._ephemeral_response_ids = set()
    client._ephemeral_response_order = deque()

    client._remember_response(stored.id)
    client._remember_response(first.id)
    client._remember_response(second.id)

    assert stored.id in client.cache
    assert first.id not in client.cache
    assert second.id in client.cache
    try:
        client.retrieve(first.id)
    except KeyError:
        pass
    else:
        raise AssertionError("evicted store=false response fell through to the remote provider")


def test_store_false_is_hidden_but_available_for_continuation() -> None:
    client = object.__new__(RuntimeClient)
    response = Response.text_response(model="m", text="ephemeral", input_tokens=1, output_tokens=1)
    response.store = False
    client.cache = {response.id: response}
    client._ephemeral_response_ids = {response.id}

    try:
        client.retrieve(response.id)
    except KeyError:
        pass
    else:
        raise AssertionError("store=false response was publicly retrievable")
    assert client.continuation_response(response.id) is response


def test_tracked_stored_response_can_be_recovered_after_stream_abandonment() -> None:
    client = object.__new__(RuntimeClient)
    client.cache = {}
    client._ephemeral_response_ids = set()
    client._pending_response_store = {"resp_pending": True}
    client.headers = {"authorization": "Bearer test"}

    class Transport:
        def get(self, url, *, headers):
            assert url.endswith("/v1/responses/resp_pending")
            value = Response.text_response(
                model="m",
                text="",
                response_id="resp_pending",
                input_tokens=1,
                output_tokens=0,
            ).to_dict()
            value["status"] = "cancelled"
            return httpx.Response(
                200,
                json=value,
                request=httpx.Request("GET", url, headers=headers),
            )

    client.http = Transport()
    assert client.retrieve("resp_pending").status == "cancelled"
    with pytest.raises(KeyError):
        client.retrieve("resp_unknown")
