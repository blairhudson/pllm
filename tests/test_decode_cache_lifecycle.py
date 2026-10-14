"""Exact lifecycle failure points; generated providers only, never performance runs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import asyncio
import threading

import httpx
import pytest

import pllm.runtime.client as implementation


class LifecycleFailure(RuntimeError):
    pass


@pytest.mark.parametrize("changes,match", [
    ({"preparation_api_key": None}, "required"),
    ({"preparation_api_key": "inference"}, "distinct"),
    ({"preparation_base_url": "http://127.0.0.1:8000"}, "distinct origins"),
    ({"preparation_base_url": "http://example.com:8001"}, "HTTPS"),
])
def test_constructor_rejects_credentials_origins_before_owned_http_creation(monkeypatch, changes, match):
    created = []
    def create(**kwargs):
        created.append(kwargs)
        raise AssertionError("rejection must precede owned HTTP allocation")
    monkeypatch.setattr(httpx, "Client", create)
    values = dict(base_url="http://127.0.0.1:8000", api_key="inference",
        preparation_base_url="http://127.0.0.1:8001", preparation_api_key="preparation") | changes
    with pytest.raises(ValueError, match=match):
        implementation.RuntimeClient(**values)
    assert created == []


def test_second_owned_http_constructor_failure_closes_first_and_preserves_failure(monkeypatch):
    failure = LifecycleFailure("second HTTP constructor")
    closed, calls = [], []
    first = SimpleNamespace(close=lambda: closed.append("inference"))
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise failure
        return first
    monkeypatch.setattr(httpx, "Client", create)
    with pytest.raises(LifecycleFailure) as caught:
        implementation.RuntimeClient(base_url="http://127.0.0.1:8000", api_key="inference",
            preparation_base_url="http://127.0.0.1:8001", preparation_api_key="preparation")
    assert caught.value is failure
    assert closed == ["inference"]


def close_owner(failure):
    core = object.__new__(implementation.RuntimeClient)
    calls = []
    def record(name):
        return lambda *args, **kwargs: calls.append(name)
    for name in ("_activity_lock", "_transformer_state_lock", "_transformer_conversation_lock"):
        setattr(core, name, threading.RLock())
    core._closing = False
    core._active_offset_transports = {}
    core._prepared_queue = None
    core._provider_executor = SimpleNamespace(shutdown=record("executor"))
    def corrupt(prior):
        assert prior == "prior"
        raise failure
    core.decode_cache = SimpleNamespace(head="current", _restore_head=corrupt, close=record("decode"))
    core._decode_publications = {"response": ("prior", "current")}
    core._decode_pending_responses = {"response"}
    core._decode_terminal_responses = {"response"}
    core._transformer_states = {"model": SimpleNamespace(bundle=SimpleNamespace(close=record("bundle")),
        prefill_cache=SimpleNamespace(clear=record("prefix")), prepared_inventory="inventory",
        prepared_inventory_spare=None, retired_inventories=[])}
    core._transformer_conversations = {"response": object()}
    core._model_manifests = {}
    core.cache, core.histories, core.message_histories = ({"response": object()} for _ in range(3))
    core._pending_response_store = {"response": True}
    core._cancel_prepared_inventory = record("inventory")
    core._owns_preparation_http = core._owns_http = True
    core.http = SimpleNamespace(close=record("http"))
    core.preparation_http = SimpleNamespace(close=record("preparation"))
    core._offset_workers = {"worker_b": (SimpleNamespace(close=record("worker")), "key")}
    return core, calls


def test_close_finishes_all_resources_then_reports_rollback_authentication_failure():
    failure = LifecycleFailure("prior-head authentication")
    core, calls = close_owner(failure)
    with pytest.raises(LifecycleFailure) as caught:
        core.close()
    assert caught.value is failure
    assert {"executor", "decode", "bundle", "prefix", "inventory", "http", "preparation", "worker"} <= set(calls)
    assert calls.count("decode") == 2
    assert core._closing
    for name in ("cache", "histories", "message_histories", "_transformer_states",
                 "_transformer_conversations", "_decode_publications", "_decode_pending_responses"):
        assert not getattr(core, name)


def test_close_preserves_original_exception_when_rollback_cleanup_also_fails():
    core, calls = close_owner(LifecycleFailure("rollback cleanup"))
    original = ValueError("original request failure")
    with pytest.raises(ValueError) as caught:
        try:
            raise original
        except ValueError:
            core.close()
            raise
    assert caught.value is original
    assert any("rollback cleanup" in note for note in original.__notes__)
    assert "http" in calls and "bundle" in calls


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_reserved_real_lease_burns_on_spec_serialization_failure(monkeypatch, cleanup_fails):
    burns, accounting = [], []
    stage = SimpleNamespace(request=SimpleNamespace(rows=3),
        burn=lambda start, rows: burns.append((start, rows)))
    inventory = implementation.PreparedInventory("inventory", 3, {"stage": stage}, demand=True,
        _audit=lambda name, count: accounting.append((name, count)))
    state = SimpleNamespace(privacy_mode="public", refill_in_progress=False,
        prepared_inventory=inventory, prepared_inventory_spare=None, context_ids={})
    core = object.__new__(implementation.RuntimeClient)
    core._transformer_state_lock = threading.RLock()
    core._activity_lock = threading.RLock()
    core._closing = False
    core._prepared_queue = None
    core._transformer_state = lambda model_id: state
    core._prepared_inventory_is_live = lambda value: True
    core.prepared_request_encoding = "raw"
    failure = LifecycleFailure("spec serialization after reservation")
    def serialize():
        assert inventory.status()["reserved"] == 1
        raise failure
    compiled = SimpleNamespace(model_plan_digest="plan", to_spec=serialize)
    if cleanup_fails:
        close = implementation.PreparedInventoryLease.close
        def close_then_fail(lease):
            close(lease)
            raise RuntimeError("secondary lease cleanup")
        monkeypatch.setattr(implementation.PreparedInventoryLease, "close", close_then_fail)
    with pytest.raises(LifecycleFailure) as caught:
        core._open_transformer_session("model", max_output_tokens=1, required_rows=1,
            compiled=compiled, semantic_input_tokens=1)
    assert caught.value is failure
    assert burns == [(0, 3)]
    assert inventory.status()["reserved"] == 0
    assert inventory.status()["stage_rows_burned"] == 3
    assert ("prepared_stage_rows_reserved", 3) in accounting
    assert ("prepared_stage_rows_burned", 3) in accounting
    if cleanup_fails:
        assert core._closing and any("secondary lease" in note for note in failure.__notes__)


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    from pllm.runtime.servers import build_roles
    from test_decode_cache_sdk import experiment_at, cache_at, SamplingRecorder
    model_id, experiment = experiment_at(tmp_path)
    cache, sampler = cache_at(tmp_path / "decode"), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles, roles.client(
            decode_cache=cache, background_inventory_refill=False) as client:
        yield client, cache, sampler, model_id


def settled(client, cache, model_id):
    core = client._core
    state = core._transformer_states[model_id]
    assert core._online_active == core._preparation_active == state.active_prepared_responses == 0
    assert core._prepared_queue.requests == 0
    assert not core._decode_publications and not core._decode_pending_responses
    assert not core._decode_epoch_responses
    assert cache.storage_metrics["staged_count"] == 0
    audit = core.audit
    assert audit.prepared_stage_rows_reserved == audit.prepared_stage_rows_claimed + audit.prepared_stage_rows_burned
    assert audit.preparation_requests_during_online == 0


def absent(core, response_id):
    for name in ("cache", "histories", "message_histories", "_transformer_conversations", "_pending_response_store"):
        assert response_id not in getattr(core, name), name
    assert response_id not in core._decode_terminal_responses


@pytest.mark.integration
@pytest.mark.parametrize("warm", [False, True])
def test_terminal_event_construction_failure_restores_head_and_evicts_response(sdk, monkeypatch, warm):
    from test_decode_cache_sdk import run_request, clear_resident
    client, cache, sampler, model_id = sdk
    if warm:
        run_request(client, sampler)
        clear_resident(client, model_id)
    prior = cache.head
    original = implementation.ResponseEvent.from_dict
    failure = LifecycleFailure("terminal event construction")
    failed_ids = []
    def construct(cls, value):
        if value["type"] in {"response.completed", "response.incomplete"}:
            failed_ids.append(value["response"]["id"])
            raise failure
        return original(value)
    monkeypatch.setattr(implementation.ResponseEvent, "from_dict", classmethod(construct))
    sampler.begin(42)
    with pytest.raises(LifecycleFailure) as caught:
        client.responses.create(input="abc" if warm else "different", max_output_tokens=4, temperature=0)
    assert caught.value is failure and cache.head == prior
    assert len(failed_ids) == 1
    absent(client._core, failed_ids[0])
    settled(client, cache, model_id)


@pytest.mark.integration
def test_terminal_response_conversion_failure_explicitly_retires(sdk, monkeypatch):
    client, cache, sampler, model_id = sdk
    core, prior = client._core, cache.head
    original_events = core.events
    original_parse = implementation.Response.from_dict
    retained_iterators, failed_ids = [], []

    def retain_events(body):
        events = original_events(body)
        retained_iterators.append(events)
        return events

    def fail_terminal(cls, value):
        if value.get("id") in core._decode_terminal_responses:
            assert cache.head != prior
            failed_ids.append(value["id"])
            raise LifecycleFailure("terminal response conversion")
        return original_parse(value)

    # Keep the iterator alive: cleanup must not rely on its finalizer.
    monkeypatch.setattr(core, "events", retain_events)
    monkeypatch.setattr(implementation.Response, "from_dict", classmethod(fail_terminal))
    sampler.begin(42)
    with pytest.raises(LifecycleFailure, match="terminal response conversion"):
        client.responses.create(input="abc", max_output_tokens=4, temperature=0)
    assert retained_iterators and len(failed_ids) == 1
    assert cache.head == prior
    assert not core._decode_cache_lock.locked()
    absent(core, failed_ids[0])
    settled(client, cache, model_id)


@pytest.mark.integration
def test_explicit_cancel_during_final_event_construction_rolls_back(sdk, monkeypatch):
    client, cache, sampler, model_id = sdk
    entered, release = threading.Event(), threading.Event()
    original = implementation.ResponseEvent.from_dict
    response_ids = []
    prior = cache.head
    def construct(cls, value):
        if value["type"] in {"response.completed", "response.incomplete"}:
            response_ids.append(value["response"]["id"])
            entered.set()
            assert release.wait(15)
        return original(value)
    monkeypatch.setattr(implementation.ResponseEvent, "from_dict", classmethod(construct))
    sampler.begin(42)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(client.responses.create, input="abc", max_output_tokens=4, temperature=0)
        try:
            assert entered.wait(15)
            assert client.responses.cancel(response_ids[0]).status == "cancelled"
        finally:
            release.set()
        with pytest.raises(Exception, match="cancel"):
            task.result(15)
    assert cache.head == prior
    absent(client._core, response_ids[0])
    settled(client, cache, model_id)


@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
def test_explicit_cancel_after_terminal_ready_returns_terminal_not_false_cancelled(sdk, monkeypatch, stream):
    client, cache, sampler, model_id = sdk
    entered, release = threading.Event(), threading.Event()
    original = client._core._accept_decode_response
    response_ids = []
    def accept(response_id):
        response_ids.append(response_id)
        entered.set()
        assert release.wait(15)
        return original(response_id)
    monkeypatch.setattr(client._core, "_accept_decode_response", accept)
    sampler.begin(42)
    def request():
        result = client.responses.create(input="abc", max_output_tokens=4, temperature=0, stream=stream)
        if not stream:
            return result
        with result:
            for event in result:
                if event.type in {"response.completed", "response.incomplete"}:
                    response = implementation.Response.from_dict(event.response)
        return response
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(request)
        try:
            assert entered.wait(15)
            terminal = client.responses.cancel(response_ids[0])
            assert terminal.status in {"completed", "incomplete"}
        finally:
            release.set()
        response = task.result(15)
    assert terminal.id == response.id and cache.head is not None
    settled(client, cache, model_id)


@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
def test_async_cancel_at_final_event_after_publication_drains_and_evicts(sdk, monkeypatch, stream):
    client, cache, sampler, model_id = sdk
    entered, release = threading.Event(), threading.Event()
    original = implementation.ResponseEvent.from_dict
    response_ids, delivered = [], []
    prior = cache.head
    def construct(cls, value):
        if value["type"] in {"response.completed", "response.incomplete"}:
            response_ids.append(value["response"]["id"])
            assert cache.head is not None and cache.head != prior
            assert value["response"]["id"] in client._core._decode_publications
            entered.set()
            assert release.wait(15)
        return original(value)
    monkeypatch.setattr(implementation.ResponseEvent, "from_dict", classmethod(construct))
    async def request():
        result = await implementation.AsyncResponsesResource(client.responses).create(
            input="abc", max_output_tokens=4, temperature=0, stream=stream)
        if stream:
            async for event in result:
                delivered.append(event.type)
        else:
            delivered.append(result.status)
    async def run():
        sampler.begin(42)
        task = asyncio.create_task(request())
        try:
            assert await asyncio.to_thread(entered.wait, 15)
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done(), "cancellation must drain terminal-event worker"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 15)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())
    assert not any(value in {"completed", "incomplete", "response.completed", "response.incomplete"}
                   for value in delivered)
    assert cache.head == prior
    absent(client._core, response_ids[0])
    settled(client, cache, model_id)


@pytest.mark.integration
def test_remote_executor_constructor_failure_retires_already_admitted_session(sdk, monkeypatch):
    client, cache, sampler, model_id = sdk
    core = client._core
    original = core._cancel_transformer_session
    cancelled = []
    def cancel(session_id):
        result = original(session_id)
        cancelled.append((session_id, result))
        return result
    monkeypatch.setattr(core, "_cancel_transformer_session", cancel)
    failure = LifecycleFailure("executor constructor")
    def construct(self, **kwargs):
        assert core._online_active == 1
        assert core._transformer_states[model_id].active_prepared_responses == 1
        raise failure
    monkeypatch.setattr(implementation.PreparedRemoteLinear, "__init__", construct)
    sampler.begin(42)
    with pytest.raises(LifecycleFailure) as caught:
        client.responses.create(input="abc", max_output_tokens=4, temperature=0)
    assert caught.value is failure
    assert len(cancelled) == 1 and cancelled[0][1] is True
    assert core.audit.prepared_stage_rows_reserved > 0
    assert cache.head is None and not core._transformer_conversations
    settled(client, cache, model_id)


@pytest.mark.integration
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_handshake_serialization_failure_burns_reserved_rows_preserving_exception(sdk, monkeypatch, cleanup_fails):
    client, cache, sampler, model_id = sdk
    core = client._core
    state = core._transformer_state(model_id)
    compiled = core._compiled_public_decoder(state, max_input_tokens=3, max_new_tokens=4)
    contract = core._decoder_continuation(compiled)
    failure = LifecycleFailure("handshake serialization")
    def handshake(self):
        assert state.prepared_inventory.status()["reserved"] > 0
        assert core._online_active == 0
        raise failure
    monkeypatch.setattr(type(contract), "handshake_spec", handshake)
    if cleanup_fails:
        close = implementation.PreparedInventoryLease.close
        def close_then_fail(lease):
            close(lease)
            raise RuntimeError("injected lease cleanup error")
        monkeypatch.setattr(implementation.PreparedInventoryLease, "close", close_then_fail)
    sampler.begin(42)
    with pytest.raises(LifecycleFailure) as caught:
        client.responses.create(input="abc", max_output_tokens=4, temperature=0)
    assert caught.value is failure
    if cleanup_fails:
        assert any("lease cleanup" in note for note in failure.__notes__)
        assert core._closing
    assert core.audit.prepared_stage_rows_reserved == core.audit.prepared_stage_rows_burned > 0
    assert core.audit.decode_epochs_started == 0
    assert cache.head is None and not core._transformer_conversations
    settled(client, cache, model_id)
