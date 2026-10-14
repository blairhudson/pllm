"""Public SDK transient-prefix integration; generated models, no timing cohorts.

Concurrency is ordered by callback events, never sleeps or polling. These tests
require the newly built native component admission as well as runtime plumbing.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import asyncio
import threading

import httpx
import numpy as np
import pytest

from pllm import AsyncOpenAI
from pllm.runtime.inflight_prefix import InflightPrefixOwner, InflightProducerFailed
from pllm.runtime.prepared_queue import PreparedCancellation, PreparedExecutionQueue
from pllm.runtime.protocol import ProtocolEnvelope, pack_envelope
from pllm.runtime.security import derive_session_key
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.runtime.transformer_client import PreparedRemoteLinear
from pllm.state import ClientPrefixReuse
from test_decode_epochs_sdk import epoch_experiment, numeric_state


pytestmark = pytest.mark.integration
TIMEOUT = 15
PROMPT = "shared prompt"
SAMPLING = (dict(temperature=0), dict(temperature=0.8, top_p=0.9))


def experiment(tmp_path, family="qwen2", *, epoch=None):
    base = epoch_experiment(tmp_path, family, epoch=epoch)
    return replace(base, pipeline=base.pipeline.with_params(cache=ClientPrefixReuse(
        max_bytes=1 << 20, fixed_input_tokens=128, inflight=True,
        inflight_max_requests=4, inflight_max_bytes=1 << 20)))


def create(client, prompt=PROMPT, sampling=None, **kwargs):
    return client.responses.create(input=prompt, max_output_tokens=4,
                                   **(sampling or SAMPLING[0]), **kwargs)


def freeze(client, response):
    saved = client._core._transformer_conversations[response.id]
    return dict(text=response.output_text, usage=response.usage.to_dict(), status=response.status,
                ids=list(saved.token_ids), pending=list(saved.pending_token_ids),
                position=saved.snapshot.position,
                claimed_stage_rows=client.privacy_audit.prepared_stage_rows_claimed,
                values=[value.view(np.uint32).copy() for value in numeric_state(saved)])


def exact(client, response, oracle):
    actual = freeze(client, response)
    for key in ("text", "usage", "status", "ids", "pending", "position"):
        assert actual[key] == oracle[key], key
    for value, expected in zip(actual["values"], oracle["values"], strict=True):
        np.testing.assert_array_equal(value, expected)


def controls(roles, tmp_path, prompts=(PROMPT, PROMPT)):
    # Separate SDK instances: no completed cache or transient owner can supply
    # either independent serial reference. Numeric/public composition is identical.
    results = []
    for index, (prompt, sampling) in enumerate(zip(prompts, SAMPLING, strict=True)):
        with roles.client(bundle_cache_dir=tmp_path / f"control-{index}") as client:
            results.append(freeze(client, create(client, prompt, sampling)))
    return results


def drained(core):
    counters = core._inflight_prefix_owner.counters()
    for name in ("reserved_bytes", "requests", "nodes", "pending_nodes", "active_producers"):
        assert counters[name] == 0, (name, counters)
    assert core._prepared_queue.requests == 0
    assert core._online_active == core._preparation_active == 0
    audit = core.audit
    assert audit.preparation_requests_during_online == 0
    assert audit.plaintext_prompt_bytes_sent == audit.plaintext_token_ids_sent == 0
    assert audit.prepared_stage_rows_reserved == (
        audit.prepared_stage_rows_claimed + audit.prepared_stage_rows_burned)


class Gate:
    """Hold first real prefill until another SDK request has completed lookup."""

    def __init__(self, monkeypatch, core, *, fail=False):
        self.owner = core._inflight_prefix_owner
        assert self.owner is not None
        assert isinstance(core._prepared_queue, PreparedExecutionQueue)
        self.blocked = threading.Event()
        self.second_lookup = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.requests = []
        self.prefills = 0
        lookup = InflightPrefixOwner.lookup_or_register
        prepare = SemanticDecoderRuntime.prepare_ids

        def traced_lookup(owner, *args, **kwargs):
            request = lookup(owner, *args, **kwargs)
            if owner is self.owner:
                with self.lock:
                    self.requests.append(request)
                    if len(self.requests) == 2:
                        self.second_lookup.set()
            return request

        def blocked_prepare(runtime, ids):
            with self.lock:
                self.prefills += 1
                first = self.prefills == 1
            if first:
                self.blocked.set()
                assert self.release.wait(TIMEOUT), "second lookup never released producer prefill"
                if fail:
                    raise RuntimeError("injected producer prefill failure")
            return prepare(runtime, ids)

        monkeypatch.setattr(InflightPrefixOwner, "lookup_or_register", traced_lookup)
        monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", blocked_prepare)

    def wait_first(self):
        assert self.blocked.wait(TIMEOUT), "producer never entered SemanticDecoderRuntime.prepare_ids"

    def wait_second(self, *, joined=True):
        assert self.second_lookup.wait(TIMEOUT), "waiter lookup blocked behind prepared queue"
        assert len(self.requests) == 2
        assert self.requests[0].is_producer
        assert self.requests[1].is_producer is not joined


def protocol_records(monkeypatch, core):
    sessions, inventories = [], []
    lock = threading.Lock()
    post = core.http.post
    send = core.preparation_http.send

    def traced_post(path, **kwargs):
        if path == "/v1/runtime/inventories":
            assert core._online_active == 0, "Preparation inventory issued during active inference"
        response = post(path, **kwargs)
        if response.is_success:
            with lock:
                if path == "/v1/runtime/sessions":
                    sessions.append((response.json()["id"], kwargs["json"]["inventory_id"]))
                elif path == "/v1/runtime/inventories":
                    inventories.append(response.json()["id"])
        return response

    def traced_send(request, **kwargs):
        if "/stages/" in request.url.path:
            assert core._online_active == 0, "seeded Preparation request issued while inference online"
        return send(request, **kwargs)

    monkeypatch.setattr(core.http, "post", traced_post)
    monkeypatch.setattr(core.preparation_http, "send", traced_send)
    return sessions, inventories


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("transport", ["http", "websocket"])
def test_public_sdk_one_pending_prefill_exact_independent_branches(tmp_path, monkeypatch, family, transport):
    exp = experiment(tmp_path, family)
    with build_roles(exp, engine_threads=1) as roles:
        reference = controls(roles, tmp_path)
        with roles.client(session_transport=transport, bundle_cache_dir=tmp_path / "shared") as client:
            core = client._core
            gate = Gate(monkeypatch, core)
            sessions, inventories = protocol_records(monkeypatch, core)
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(create, client, PROMPT, SAMPLING[0])
                try:
                    gate.wait_first()
                    second = pool.submit(create, client, PROMPT, SAMPLING[1])
                    gate.wait_second()
                finally:
                    gate.release.set()
                responses = [first.result(TIMEOUT), second.result(TIMEOUT)]
            assert responses[0].id != responses[1].id
            for response, oracle in zip(responses, reference, strict=True):
                exact(client, response, oracle)
            states = [core._transformer_conversations[r.id] for r in responses]
            for left, right in zip(numeric_state(states[0]), numeric_state(states[1]), strict=True):
                assert not np.shares_memory(left, right)
            counts = core._inflight_prefix_owner.counters()
            assert counts["producer_registrations"] == counts["joins"] == counts["publications"] == 1
            assert gate.prefills == 1
            assert len(sessions) == len({row[0] for row in sessions}) == 2
            assert len(inventories) == len(set(inventories)) == 2
            assert {row[1] for row in sessions} == set(inventories)
            assert core.audit.prepared_stage_rows_claimed < sum(
                row["claimed_stage_rows"] for row in reference)
            drained(core)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("epoch", [2, (2, 5)])
def test_gateway_shared_prefill_keeps_queue_owner_across_decode_epochs(tmp_path, monkeypatch, stream, epoch):
    import json
    from fastapi.testclient import TestClient
    from pllm.runtime.sidecar import create_sidecar_app
    from pllm.runtime.types import Response

    exp = experiment(tmp_path, epoch=epoch)
    with build_roles(exp, engine_threads=1) as roles:
        reference = controls(roles, tmp_path)
        with roles.client(session_transport="websocket", bundle_cache_dir=tmp_path / "shared") as client:
            core = client._core
            gate = Gate(monkeypatch, core)
            sessions, inventories = protocol_records(monkeypatch, core)
            with TestClient(create_sidecar_app(client=client, local_api_key="local")) as gateway:
                def request(sampling):
                    result = gateway.post("/v1/responses", headers={"Authorization": "Bearer local"},
                        json={"model": "epoch-sdk", "input": PROMPT, "max_output_tokens": 4,
                              "stream": stream, **sampling})
                    assert result.status_code == 200, result.text
                    if not stream:
                        return Response.from_dict(result.json())
                    events = [json.loads(line.removeprefix("data: ")) for line in result.text.splitlines()
                              if line.startswith("data: ") and line != "data: [DONE]"]
                    assert [event["sequence_number"] for event in events] == list(range(len(events)))
                    assert events[-1]["type"] in {"response.completed", "response.incomplete"}
                    return Response.from_dict(events[-1]["response"])

                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(request, SAMPLING[0])
                    try:
                        try:
                            gate.wait_first()
                        except AssertionError:
                            if first.done():
                                first.result()
                            raise
                        second = pool.submit(request, SAMPLING[1])
                        gate.wait_second()
                    finally:
                        gate.release.set()
                    responses = [first.result(TIMEOUT), second.result(TIMEOUT)]
                for response, oracle in zip(responses, reference, strict=True):
                    exact(client, response, oracle)
                counts = core._inflight_prefix_owner.counters()
                assert counts["producer_registrations"] == counts["joins"] == counts["publications"] == 1
                assert gate.prefills == 1
                assert core.audit.decode_epochs_started == 4
                assert core.audit.decode_epoch_pauses == 2
                assert len(sessions) == len({row[0] for row in sessions}) == 4
                assert len(inventories) == len(set(inventories)) == 4
                assert {row[1] for row in sessions} == set(inventories)
                drained(core)


def test_overlap_without_complete_original_prompt_match_does_not_split(tmp_path, monkeypatch):
    prompts = ("common prefix A", "common prefix B")
    with build_roles(experiment(tmp_path), engine_threads=1) as roles:
        reference = controls(roles, tmp_path, prompts)
        with roles.client(bundle_cache_dir=tmp_path / "shared") as client:
            core = client._core
            gate = Gate(monkeypatch, core)
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(create, client, prompts[0], SAMPLING[0])
                try:
                    gate.wait_first()
                    second = pool.submit(create, client, prompts[1], SAMPLING[1])
                    gate.wait_second(joined=False)
                finally:
                    gate.release.set()
                for future, oracle in zip((first, second), reference, strict=True):
                    exact(client, future.result(TIMEOUT), oracle)
            counts = core._inflight_prefix_owner.counters()
            assert counts["producer_registrations"] == counts["publications"] == 2
            assert counts["joins"] == 0
            assert gate.prefills == 2
            drained(core)


def test_completed_cache_priority_and_store_previous_response_bypasses(tmp_path, monkeypatch):
    with build_roles(experiment(tmp_path), engine_threads=1) as roles, roles.client(
        bundle_cache_dir=tmp_path / "shared") as client:
        core = client._core
        first = create(client)
        owner = core._inflight_prefix_owner
        before = owner.counters()
        lookups = []
        original = InflightPrefixOwner.lookup_or_register

        def trace(receiver, *args, **kwargs):
            result = original(receiver, *args, **kwargs)
            if receiver is owner:
                lookups.append((kwargs.get("completed_cache_hit", False), result))
            return result

        monkeypatch.setattr(InflightPrefixOwner, "lookup_or_register", trace)
        cached = create(client)
        exact(client, cached, freeze(client, first))
        after = owner.counters()
        assert after["producer_registrations"] == before["producer_registrations"]
        assert after["joins"] == before["joins"]
        assert all(hit and request is None for hit, request in lookups)
        lookup_count = len(lookups)
        untracked = create(client, store=False)
        assert untracked.output_text == first.output_text
        assert untracked.id not in core._transformer_conversations
        continued = create(client, prompt="new turn", previous_response_id=first.id)
        assert continued.previous_response_id == first.id
        assert len(lookups) == lookup_count
        assert owner.counters()["producer_registrations"] == before["producer_registrations"]
        drained(core)


def test_producer_failure_broadcasts_retires_and_fresh_retry_succeeds(tmp_path, monkeypatch):
    with build_roles(experiment(tmp_path), engine_threads=1) as roles:
        reference = controls(roles, tmp_path)[0]
        with roles.client(bundle_cache_dir=tmp_path / "shared") as client:
            core = client._core
            gate = Gate(monkeypatch, core, fail=True)
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(create, client)
                try:
                    gate.wait_first()
                    second = pool.submit(create, client)
                    gate.wait_second()
                finally:
                    gate.release.set()
                with pytest.raises(RuntimeError, match="injected producer prefill failure"):
                    first.result(TIMEOUT)
                with pytest.raises(InflightProducerFailed):
                    second.result(TIMEOUT)
            counts = core._inflight_prefix_owner.counters()
            assert counts["failures"] == 1 and counts["publications"] == 0
            assert core.audit.prepared_stage_rows_burned > 0
            drained(core)
            retry = create(client)
            exact(client, retry, reference)
            assert core._inflight_prefix_owner.counters()["producer_registrations"] == 2
            drained(core)


def test_public_stream_close_cancels_waiter_without_cancelling_producer(tmp_path, monkeypatch):
    with build_roles(experiment(tmp_path), engine_threads=1) as roles:
        reference = controls(roles, tmp_path)[0]
        with roles.client(bundle_cache_dir=tmp_path / "shared") as client:
            core = client._core
            gate = Gate(monkeypatch, core)
            stream = create(client, stream=True)
            with ThreadPoolExecutor(max_workers=3) as pool:
                producer = pool.submit(create, client)
                try:
                    gate.wait_first()
                    waiter = pool.submit(list, stream)
                    gate.wait_second()
                    closing = pool.submit(stream.close)
                    closing.result(TIMEOUT)
                    with pytest.raises(Exception, match="cancelled|retired"):
                        waiter.result(TIMEOUT)
                    assert not gate.requests[0].producer_cancel.is_set()
                    assert core._inflight_prefix_owner.counters()["requests"] == 1
                finally:
                    gate.release.set()
                exact(client, producer.result(TIMEOUT), reference)
            drained(core)


def async_client(roles, exp, tmp_path):
    # Public AsyncOpenAI owns independent HTTP clients and runtime; credentials
    # come from the real generated role topology, never a fake provider callback.
    return AsyncOpenAI(experiment=exp, api_key=roles._inference_key, base_url=roles._inference_url,
        preparation_base_url=roles._preparation_url, preparation_api_key=roles._preparation_key,
        bundle_cache_dir=tmp_path / "async")


@pytest.mark.parametrize("departing", ["waiter", "producer"])
def test_public_async_task_cancellation_preserves_surviving_branch(tmp_path, monkeypatch, departing):
    exp = experiment(tmp_path)
    with build_roles(exp, engine_threads=1) as roles:
        reference = controls(roles, tmp_path)

        async def run():
            async with async_client(roles, exp, tmp_path) as client:
                core = client.sync._core
                gate = Gate(monkeypatch, core)
                cancel_sent = threading.Event()
                cancel = PreparedCancellation.cancel

                def observe_cancel(token):
                    cancel(token)
                    cancel_sent.set()

                monkeypatch.setattr(PreparedCancellation, "cancel", observe_cancel)
                first = asyncio.create_task(client.responses.create(input=PROMPT, max_output_tokens=4, **SAMPLING[0]))
                second = None
                try:
                    assert await asyncio.to_thread(gate.blocked.wait, TIMEOUT)
                    second = asyncio.create_task(client.responses.create(input=PROMPT, max_output_tokens=4, **SAMPLING[1]))
                    assert await asyncio.to_thread(gate.second_lookup.wait, TIMEOUT)
                    gate.wait_second()
                    cancelled = second if departing == "waiter" else first
                    survivor = first if departing == "waiter" else second
                    cancelled.cancel()
                    assert await asyncio.to_thread(cancel_sent.wait, TIMEOUT), "async cancellation did not reach owned attempt"
                    assert not gate.requests[0].producer_cancel.is_set()
                    if departing == "producer":
                        # Its shared prefill must finish for the remaining waiter
                        # before the original producer caller can drain/cancel.
                        gate.release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(cancelled, TIMEOUT)
                    if departing == "waiter":
                        assert not gate.requests[0].producer_cancel.is_set()
                        gate.release.set()
                    response = await asyncio.wait_for(survivor, TIMEOUT)
                    exact(client.sync, response, reference[0 if departing == "waiter" else 1])
                    counts = core._inflight_prefix_owner.counters()
                    assert counts["producer_registrations"] == counts["joins"] == counts["publications"] == 1
                    assert counts["cancellations"] >= 1
                    drained(core)
                finally:
                    gate.release.set()
                    tasks = [task for task in (first, second) if task is not None]
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run(run())


def test_public_client_close_wakes_join_and_drains_active_producer(tmp_path, monkeypatch):
    with build_roles(experiment(tmp_path), engine_threads=1) as roles, roles.client(
        bundle_cache_dir=tmp_path / "shared") as client:
        core = client._core
        gate = Gate(monkeypatch, core)
        close_seen = threading.Event()
        original_close = InflightPrefixOwner.close

        def observe_close(owner, **kwargs):
            if owner is gate.owner:
                close_seen.set()
            return original_close(owner, **kwargs)

        monkeypatch.setattr(InflightPrefixOwner, "close", observe_close)
        with ThreadPoolExecutor(max_workers=3) as pool:
            producer = pool.submit(create, client)
            try:
                gate.wait_first()
                waiter = pool.submit(create, client)
                gate.wait_second()
                close = pool.submit(client.close)
                assert close_seen.wait(TIMEOUT), "client.close never reached transient owner shutdown"
            finally:
                gate.release.set()
            close.result(TIMEOUT)
            for future in (producer, waiter):
                with pytest.raises(Exception, match="closed|cancelled|failed|retired"):
                    future.result(TIMEOUT)
        drained(core)


def assert_provider_retired(core, session_id):
    """A valid authenticated frame must find no session or a terminal session.

    Initialization-failure sessions have never executed an online frame, so zero
    is fresh in their replay namespace. Empty stage payload is intentional: an
    active session would report an unknown stage, not the terminal rejection.
    """
    envelope = ProtocolEnvelope.create(request_id="retirement-regression", session_id=session_id,
        model=core.default_model, kind="masked.transformer.stage", sequence=0, payload=b"",
        metadata={"stage_id": "not-an-executable-stage"},
        key=derive_session_key(core.api_key, session_id))
    response = core.http.post(f"/v1/runtime/sessions/{session_id}/execute",
        headers={**core.headers, "Content-Type": "application/vnd.pllm.runtime+msgpack"},
        content=pack_envelope(envelope))
    assert response.status_code in {400, 404, 410}
    if response.status_code == 400:
        assert "session is already terminal" in response.text


@pytest.mark.parametrize("failure", ["response_id", "remote_constructor"])
def test_public_initialization_failure_burns_real_session_and_fresh_retry(tmp_path, monkeypatch, failure):
    with build_roles(experiment(tmp_path), engine_threads=1) as roles, roles.client(
        bundle_cache_dir=tmp_path / "shared") as client:
        core = client._core
        sessions, inventories = protocol_records(monkeypatch, core)
        with monkeypatch.context() as inject:
            if failure == "response_id":
                post = core.http.post

                def malformed_session(path, **kwargs):
                    response = post(path, **kwargs)
                    if path == "/v1/runtime/sessions" and response.is_success:
                        value = response.json()
                        value.pop("response_id")
                        return httpx.Response(response.status_code, json=value, request=response.request)
                    return response

                inject.setattr(core.http, "post", malformed_session)
                expected = "provider response identifier is invalid"
            else:
                def broken_constructor(self, *args, **kwargs):
                    raise RuntimeError("injected PreparedRemoteLinear constructor failure")

                inject.setattr(PreparedRemoteLinear, "__init__", broken_constructor)
                expected = "injected PreparedRemoteLinear constructor failure"
            with pytest.raises(Exception, match=expected):
                create(client)
        assert len(sessions) == len(inventories) == 1
        failed_session, failed_inventory = sessions[0]
        assert failed_inventory == inventories[0]
        assert core.audit.prepared_stage_rows_reserved > 0
        assert core.audit.prepared_stage_rows_claimed == 0
        assert core.audit.prepared_stage_rows_burned == core.audit.prepared_stage_rows_reserved
        assert core._transformer_states["epoch-sdk"].active_prepared_responses == 0
        assert not core._closing
        drained(core)
        assert_provider_retired(core, failed_session)
        retried = create(client)
        assert retried.usage.output_tokens > 0
        assert len(sessions) == len(inventories) == 2
        assert sessions[1][0] != failed_session
        assert inventories[1] != failed_inventory
        assert core._transformer_states["epoch-sdk"].active_prepared_responses == 0
        drained(core)


def test_public_failed_online_retirement_closes_response_and_explicit_preparation(tmp_path, monkeypatch):
    with build_roles(experiment(tmp_path), engine_threads=1) as roles, roles.client(
        bundle_cache_dir=tmp_path / "shared") as client:
        core = client._core
        post = core.http.post
        cancel_attempts, preparation_posts, accepted_stages, sessions = [], [], [], []

        def fail_after_accepted_stage(path, **kwargs):
            if path.endswith("/cancel"):
                cancel_attempts.append(path)
                return httpx.Response(503, json={"detail": "injected retirement endpoint failure"},
                                      request=httpx.Request("POST", core.base_url + path))
            if path == "/v1/runtime/inventories":
                preparation_posts.append(path)
            response = post(path, **kwargs)
            if path == "/v1/runtime/sessions" and response.is_success:
                sessions.append(response.json()["id"])
            if "/stages/" in path and path.startswith("/v1/runtime/sessions/") and response.is_success:
                accepted_stages.append(path)
                # Actual provider accepted/executed work. Caller loses its reply
                # and then cannot obtain a provider retirement acknowledgement.
                raise httpx.ReadTimeout("injected accepted-stage reply loss", request=response.request)
            return response

        with monkeypatch.context() as inject:
            inject.setattr(core.http, "post", fail_after_accepted_stage)
            with pytest.raises(httpx.ReadTimeout, match="injected accepted-stage reply loss"):
                create(client)
            assert len(sessions) == len(accepted_stages) == 1
            assert f"/v1/runtime/sessions/{sessions[0]}/cancel" in cancel_attempts
            assert core._closing
            assert core._transformer_states["epoch-sdk"].active_prepared_responses == 0
            drained(core)
            issued = core.audit.preparation_attempts
            inventory_calls = len(preparation_posts)
            for attempt in (lambda: create(client),
                            lambda: client.prepare_response(PROMPT, 4),
                            lambda: client.preprocess("epoch-sdk", count=4)):
                with pytest.raises(Exception, match="closed"):
                    attempt()
                assert len(preparation_posts) == inventory_calls
                assert core.audit.preparation_attempts == issued
            drained(core)
        # Fault is now removed solely to retire the real provider in test cleanup.
        # The poisoned SDK core must stay closed even if the network recovers.
        retired = post(f"/v1/runtime/sessions/{sessions[0]}/cancel", headers=core.headers)
        assert retired.status_code in {200, 404, 410}
        if retired.status_code == 200:
            assert retired.json()["status"] == "cancelled"
        assert core._closing


def test_public_async_full_text_stream_has_all_events_and_exact_saved_state(tmp_path):
    exp = experiment(tmp_path)
    with build_roles(exp, engine_threads=1) as roles:
        reference = controls(roles, tmp_path)[0]

        async def run():
            async with async_client(roles, exp, tmp_path) as client:
                stream = await client.responses.create(input=PROMPT, max_output_tokens=4,
                                                       temperature=0, stream=True)
                events = [event async for event in stream]
                assert [event.sequence_number for event in events] == list(range(len(events)))
                types = [event.type for event in events]
                assert types[:2] == ["response.created", "response.in_progress"]
                assert "response.output_item.added" in types
                assert "response.output_text.delta" in types
                assert "response.output_text.done" in types
                assert types[-1] in {"response.completed", "response.incomplete"}
                from pllm.runtime.types import Response

                response = Response.from_dict(events[-1].response)
                assert {event.response["id"] for event in events if "response" in event.data} == {response.id}
                exact(client.sync, response, reference)
                drained(client.sync._core)

        asyncio.run(run())
