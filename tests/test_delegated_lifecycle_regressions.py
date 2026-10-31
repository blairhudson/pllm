"""Bounded reproducers for the SDK audit; fixtures identify controlled owners."""

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from types import SimpleNamespace

import anyio
from fastapi.testclient import TestClient
import httpx
import pytest

from pllm import Model
from pllm.deployment import resources
from pllm.runtime import delegated_source, delegated_transport, delegated_worker, sidecar
from pllm.runtime.client import AsyncResponsesResource, ProtocolError

from test_delegated_client import harness as harness
from test_delegated_sdk import experiment_fixture
from test_delegated_source import write_source_fixture


_LIFECYCLE_FAILED_LEDGERS = set()


def _forget_empty_ledger(capacity):
    if type(capacity._owners) is dict and not capacity._owners:
        _LIFECYCLE_FAILED_LEDGERS.discard(capacity)


@pytest.fixture
def _worker_lifecycle_capacity(monkeypatch):
    capacity = resources._ClientCapacity()
    _LIFECYCLE_FAILED_LEDGERS.add(capacity)
    try:
        monkeypatch.setattr(resources, "_capacity", capacity)
        monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))
        yield capacity
    finally:
        _forget_empty_ledger(capacity)


@pytest.mark.parametrize("padding", [
    {"strategy": {"Fixed": 1024}},
    {"strategy": {"Fixed": 1_000_000_000}},
    {"strategy": "BatchLongest", "pad_to_multiple_of": 1_000_000_000},
])
def test_source_rejects_padding_before_tokenizer_construction(tmp_path, monkeypatch, padding):
    write_source_fixture(tmp_path / "source")
    path = tmp_path / "source" / "tokenizer.json"
    value = json.loads(path.read_bytes())
    value["padding"] = {"direction": "Right", "pad_to_multiple_of": None,
                        "pad_id": 2, "pad_type_id": 0, "pad_token": "<unk>", **padding}
    path.write_text(json.dumps(value))
    import tokenizers

    class NoAllocation:
        @staticmethod
        def from_buffer(*args):
            raise AssertionError("tokenizer construction preceded padding admission")

    monkeypatch.setattr(tokenizers, "Tokenizer", NoAllocation)
    with pytest.raises(delegated_source.DelegatedSourceError, match="padding"):
        delegated_source.admit_source(Model.path(str(path.parent)), input_rows=8, decode_steps=2)


@pytest.mark.parametrize("route", ["responses", "chat/completions", "preprocess", "responses/compact"])
@pytest.mark.parametrize("chunked", [False, True])
def test_gateway_bounds_raw_ingress_before_json_or_source(harness, route, chunked):
    h = harness
    app = sidecar.create_sidecar_app(client=h.core)
    body = b'{"model":"gkm-fixture","input":"hello"}' + b" " * (128 << 10)
    content = (body[i:i + 4096] for i in range(0, len(body), 4096)) if chunked else body
    with TestClient(app) as client:
        result = client.post("/v1/" + route, content=content,
                             headers={"authorization": "Bearer local", "content-type": "application/json"})
    assert result.status_code == 413, result.text
    assert not h.admitted and not h.calls and h.core._active is None


@pytest.mark.asyncio
async def test_closing_unstarted_async_stream_releases_seat(harness):
    h = harness
    responses = AsyncResponsesResource(h.responses)
    stream = await responses.create(input="hello", stream=True)
    assert h.core._active is not None
    await stream.aclose()
    await stream.aclose()
    assert h.core._active is None and not h.admitted and not h.calls
    assert (await responses.create(input="hello")).usage.output_tokens == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close", [False, True])
async def test_concurrent_async_close_signals_and_drains_exclusive_request(harness, monkeypatch, cancel_close):
    from pllm.runtime import delegated_client

    h = harness
    entered, cancelled, release, retired = (threading.Event() for _ in range(4))

    def execute(*args):
        attempt = args[-1]
        entered.set()
        try:
            assert attempt.cancelled.wait(3), "close did not signal active execution"
            cancelled.set()
            assert release.wait(3)
            raise ProtocolError("cancelled", 499)
        finally:
            retired.set()

    original = delegated_client._execute_request
    monkeypatch.setattr(delegated_client, "_execute_request", execute)
    responses = AsyncResponsesResource(h.responses)
    stream = await responses.create(input="hello", stream=True)
    advance = asyncio.create_task(anext(stream))
    closers = []
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        closers = [asyncio.create_task(stream.aclose()) for _ in range(2)]
        assert await asyncio.to_thread(cancelled.wait, 3)
        assert h.core._active is not None  # No release while work still owns state.
        assert not any(task.done() for task in closers)
        if cancel_close:
            closers[0].cancel()
        release.set()
        outcomes = await asyncio.wait_for(asyncio.gather(*closers, return_exceptions=True), 3)
        assert outcomes[1] is None
        assert isinstance(outcomes[0], asyncio.CancelledError) if cancel_close else outcomes[0] is None
        with pytest.raises(ProtocolError, match="cancelled"):
            await advance
        await stream.aclose()
        assert retired.is_set() and h.core._active is None and not h.core.records
        assert h.core.audit.generation_completed_responses == 0
        monkeypatch.setattr(delegated_client, "_execute_request", original)
        assert (await responses.create(input="hello")).usage.output_tokens == 3
    finally:
        release.set()
        await stream.aclose()
        await asyncio.gather(advance, *closers, return_exceptions=True)


@pytest.mark.asyncio
async def test_async_close_after_terminal_keeps_accepted_response(harness):
    h = harness
    stream = await AsyncResponsesResource(h.responses).create(input="hello", stream=True)
    async for event in stream:
        if event.type == "response.incomplete":
            response_id = event.response["id"]
            break
    await stream.aclose()
    assert h.core._active is None
    assert h.core.retrieve(response_id).status == "incomplete"


@pytest.mark.asyncio
async def test_cancellation_racing_stream_creation_retires_unexposed_seat(harness, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    h = harness
    original = h.responses.create

    def create(**kwargs):
        result = original(**kwargs)
        entered.set()
        assert release.wait(3)
        return result

    monkeypatch.setattr(h.responses, "create", create)
    task = asyncio.create_task(AsyncResponsesResource(h.responses).create(input="hello", stream=True))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert h.core._active is None and not h.admitted and not h.calls
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["asyncio", "anyio"])
async def test_gateway_disconnect_signals_before_execution_drain(harness, monkeypatch, scope):
    from pllm.runtime import delegated_client

    h = harness
    entered, exited = threading.Event(), threading.Event()
    observed = []

    def execute(*args):
        cancelled = args[-1]
        entered.set()
        try:
            assert cancelled.cancelled.wait(3), "disconnect did not signal running execution"
            raise ProtocolError("cancelled", 499)
        finally:
            exited.set()

    monkeypatch.setattr(delegated_client, "_execute_request", execute)
    source = h.responses.create(input="hello", stream=True)

    async def consume():
        async for event in sidecar._closing_stream(sidecar._response_stream(source), source):
            observed.append(event)

    if scope == "asyncio":
        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
    else:
        async with anyio.create_task_group() as group:
            group.start_soon(consume)
            assert await asyncio.to_thread(entered.wait, 3)
            group.cancel_scope.cancel()
    assert exited.is_set() and not observed
    assert h.core._active is None and not h.core.records


@pytest.mark.parametrize("stale", ["0" * 64, "not-a-request"])
@pytest.mark.parametrize("action", ["step", "cancel", "finish"])
def test_worker_stale_routing_and_busy_admission_preserve_current_owner(
        tmp_path, _worker_lifecycle_capacity, stale, action):
    experiment, _, _ = experiment_fixture(tmp_path)
    app = delegated_worker.create_delegated_worker_app(experiment, "worker_a", "a" * 32)
    owner = app.state.delegated_owner
    capacity = _worker_lifecycle_capacity
    ledger = capacity._owners
    manager = resources.reserve_incremental_memory(64 << 20)
    token, backend = manager._token, manager._manager

    class Native:
        aborted = 0

        def abort(self):
            self.aborted += 1

        def execute(self, raw):
            if raw != b"current":
                raise ValueError("malformed current-window frame")
            return b"accepted-by-controlled-owner"

    native = Native()

    async def arm():
        async with owner.lock:
            retirement = delegated_worker._WorkerRetirement(
                manager, asyncio.get_running_loop(), owner.lock)
            owner.retirement = retirement
            retirement.root = owner.reservation = delegated_worker._Reservation(retirement)
            retirement.validate()
            retirement.native = owner.active = native
            owner.request = "a" * 64
            owner.seen.add(owner.request)
            owner.deadline = time.monotonic() + 10

    def assert_charged():
        assert capacity._owners is ledger and type(ledger) is dict
        assert manager._capacity is capacity and manager._token is token
        assert manager._manager is backend and manager._amount == 64 << 20
        assert ledger[token] == manager._amount and token.retained is manager
        assert manager._failed is custody and custody == (retirement,)
        assert retirement.root is root and retirement.manager is manager
        assert not retirement.done and not manager._exited

    with TestClient(app, headers={"authorization": "Bearer " + "a" * 32}) as client:
        client.portal.call(arm)
        retirement, root, custody = owner.retirement, owner.reservation, manager._failed
        assert_charged()
        assert client.post("/v1/delegation/admit", content=b"unrelated").status_code == 409
        assert client.post(f"/v1/delegation/{stale}/{action}", content=b"").status_code in (400, 404)
        assert owner.active is native and native.aborted == 0 and owner.seen == {"a" * 64}
        assert owner.reservation is root and owner.retirement is retirement
        assert_charged()
        result = client.post(f"/v1/delegation/{owner.request}/step", content=b"current")
        assert result.status_code == 200 and result.content == b"accepted-by-controlled-owner"
        assert_charged()
        result = client.post(f"/v1/delegation/{owner.request}/step", content=b"bad")
        assert result.status_code == 400 and owner.active is None and native.aborted == 1
        assert capacity._owners is ledger and token not in ledger
        assert manager._capacity is capacity and manager._token is token and manager._manager is backend
        assert token.retained is None and manager._exited and not manager._failed and not manager._draining
        assert retirement.done and retirement.terminal and retirement.abort_ok and retirement.native is None
        assert retirement.root is root and root._references == 0 and retirement.root_closed and retirement.exit_seen
        assert owner.reservation is None and owner.seen == {"a" * 64}
    assert native.aborted == 1  # Lifespan shutdown must not abort the retired shell again.
    assert capacity._owners is ledger and token not in ledger
    assert retirement.done and manager._exited and token.retained is None


@contextmanager
def slow_peer(*, truncated=False, cancelled=None):
    stop, finished = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                self.rfile.read(int(self.headers.get("content-length", "0")))
                self.send_response(200)
                self.send_header("content-length", "64")
                self.send_header("x-pllm-server-ns", "123456")
                self.end_headers()
                self.wfile.write(b"a")
                self.wfile.flush()
                if cancelled is not None:
                    cancelled.set()
                if not truncated:
                    for _ in range(63):
                        if stop.wait(0.05):
                            break
                        self.wfile.write(b"a")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                finished.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", finished
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert not thread.is_alive()


@pytest.mark.parametrize("reason", ["deadline", "cancellation", "retirement"])
def test_trickled_http_body_cannot_extend_exchange_or_cleanup(monkeypatch, reason):
    cancelled = threading.Event()
    monkeypatch.setattr(delegated_transport, "_RETIRE_TIMEOUT", 0.2)
    with slow_peer(cancelled=cancelled if reason == "cancellation" else None) as (origin, _):
        peers = delegated_transport._Peers()
        peers.start({"worker_a": (origin, "a" * 32),
            "worker_b": ("http://127.0.0.1:1", "b" * 32)}, 3, cancelled.is_set)
        peers.request_id = "a" * 64
        if reason == "deadline":
            peers.deadline = time.monotonic() + 0.2
        started = time.monotonic()
        try:
            with pytest.raises(ProtocolError, match="cancelled or timed out"):
                peers.post(0, "step", phase="online", limit=8192, retire=reason == "retirement")
            assert time.monotonic() - started < 1.0
            assert "inference_server_ns" not in peers.audit()
            assert "inference_stage_calls" not in peers.audit()
            assert "online_steps" not in peers.audit()
        finally:
            peers.close()


def test_truncated_online_reply_keeps_compute_unknown():
    with slow_peer(truncated=True) as (origin, _):
        peers = delegated_transport._Peers()
        peers.start({"worker_a": (origin, "a" * 32),
            "worker_b": ("http://127.0.0.1:1", "b" * 32)}, 3, lambda: False)
        peers.request_id = "a" * 64
        try:
            with pytest.raises(httpx.RemoteProtocolError):
                peers.post(0, "step", phase="online", limit=8192)
            audit = peers.audit()
            assert audit["role_link.worker_a.online_download_bytes"] == 1
            assert "inference_server_ns" not in audit and "inference_stage_calls" not in audit
            assert "online_steps" not in audit
        finally:
            peers.close()
