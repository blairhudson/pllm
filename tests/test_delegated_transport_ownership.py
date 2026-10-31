"""Bounded socket probes for literal origins and response-owned I/O retirement."""

import asyncio
from concurrent.futures import Future
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import ssl
import threading
import time
from types import SimpleNamespace
import weakref

import pytest

from pllm.runtime import delegated_transport as transport
from pllm.runtime.client import ProtocolError


def connections(origin):
    return {"worker_a": (origin, "a" * 32),
            "worker_b": ("http://127.0.0.1:1", "b" * 32)}


def no_dns(*args, **kwargs):
    raise AssertionError("an admitted transport must not invoke system DNS")


def assert_retired(peers):
    assert not peers.io.thread.is_alive()
    assert peers.io.loop.is_closed()
    assert not asyncio.all_tasks(peers.io.loop)
    assert peers.io.loop._default_executor is None
    assert all(client.is_closed for client in peers.clients)


@contextmanager
def peer_server(*, trickle=False, cancelled=None):
    stop = threading.Event()
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            observed.append((self.path, self.headers.get("authorization"), body))
            try:
                self.send_response(200)
                self.send_header("content-length", "64" if trickle else "2")
                self.send_header("x-pllm-server-ns", "123456")
                self.end_headers()
                self.wfile.write(b"x" if trickle else b"ok")
                self.wfile.flush()
                if cancelled is not None:
                    cancelled.set()
                if trickle:
                    for _ in range(63):
                        if stop.wait(0.03):
                            break
                        self.wfile.write(b"x")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", observed
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


@pytest.mark.parametrize("role", ["worker_a", "worker_b"])
@pytest.mark.parametrize("origin", [
    "http://worker.invalid", "https://worker.invalid", "https://localhost",
    "http://127.0.0.1.", "http://[::1%25lo0]",
])
def test_hostname_and_scoped_origins_reject_before_io(monkeypatch, role, origin):
    def forbidden():
        raise AssertionError("I/O ownership preceded complete origin admission")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    monkeypatch.setattr(transport, "_IOOwner", forbidden)
    selected = connections("http://127.0.0.1:2")
    selected[role] = (origin, selected[role][1])
    peers = transport._Peers()
    with pytest.raises(ValueError, match="IP|DNS|scoped"):
        peers.start(selected, 1, lambda: False)
    peers.close()


def test_literal_canonicalization_preserves_distinct_channels():
    assert transport._connections({
        "worker_a": ("http://LOCALHOST:80/v1", "a" * 32),
        "worker_b": ("https://[0:0:0:0:0:0:0:1]:443/v1/", "b" * 32),
    }) == [("http://127.0.0.1", "a" * 32), ("https://[::1]", "b" * 32)]
    with pytest.raises(ValueError, match="distinct"):
        transport._connections({"worker_a": ("http://localhost:80", "a" * 32),
                                "worker_b": ("http://127.0.0.1", "b" * 32)})
    with pytest.raises(ValueError, match="HTTPS"):
        transport._connections(connections("http://192.0.2.1"))


@pytest.mark.asyncio
async def test_sync_transport_inside_running_loop_never_allocates_dns_executor(monkeypatch, recwarn):
    caller_loop = asyncio.get_running_loop()
    with peer_server() as (origin, observed):
        monkeypatch.setattr(socket, "getaddrinfo", no_dns)
        peers = transport._Peers()
        peers.start(connections(origin.replace("127.0.0.1", "localhost")), 2, lambda: False)
        peers.request_id = "1" * 64
        assert peers.io.thread.is_alive() and not peers.io.thread.daemon
        try:
            assert peers.post(0, "step", b"frame", phase="online", limit=8192) == b"ok"
            assert peers.post(0, "step", b"again", phase="online", limit=8192) == b"ok"
            audit = peers.audit()
            assert audit["role_link.worker_a.online_upload_bytes"] == 10
            assert audit["role_link.worker_a.online_download_bytes"] == 4
            assert audit["inference_stage_calls"] == 2
            assert audit["inference_server_ns"] == 246912
        finally:
            peers.close()
            peers.close()
        assert observed == [(f"/v1/delegation/{peers.request_id}/step", "Bearer " + "a" * 32, body)
                            for body in (b"frame", b"again")]
        assert_retired(peers)
    assert asyncio.get_running_loop() is caller_loop
    assert not [warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)]


def test_https_literal_keeps_certificate_and_hostname_verification(monkeypatch):
    contexts = []
    create = ssl.create_default_context

    def observed(*args, **kwargs):
        context = create(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    monkeypatch.setattr(ssl, "create_default_context", observed)
    peers = transport._Peers()
    peers.start(connections("https://127.0.0.1:2"), 1, lambda: False)
    peers.close()
    assert contexts and all(context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
                            for context in contexts)
    assert_retired(peers)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["deadline", "cancel"])
async def test_timeout_and_cancellation_drain_socket_tasks_before_return(monkeypatch, reason):
    cancelled = threading.Event()
    with peer_server(trickle=True, cancelled=cancelled if reason == "cancel" else None) as (origin, _):
        monkeypatch.setattr(socket, "getaddrinfo", no_dns)
        peers = transport._Peers()
        peers.start(connections(origin), 2, cancelled.is_set)
        peers.request_id = "1" * 64
        if reason == "deadline":
            peers.deadline = time.monotonic() + 0.08
        started = time.monotonic()
        try:
            with pytest.raises(ProtocolError, match="cancelled or timed out"):
                peers.post(0, "step", b"frame", phase="online", limit=8192)
            assert time.monotonic() - started < 0.75

            async def outstanding():
                return asyncio.all_tasks() - {asyncio.current_task()}

            assert peers.io.run(outstanding(), timeout=0.1) == set()
            assert not {"inference_server_ns", "inference_stage_calls", "online_steps"} & peers.audit().keys()
        finally:
            peers.close()
        assert_retired(peers)


def test_both_remote_retirements_share_deadline_and_keep_role_credentials(monkeypatch):
    monkeypatch.setattr(transport, "_RETIRE_TIMEOUT", 0.08)
    with peer_server(trickle=True) as (zero, observed_zero), peer_server(trickle=True) as (one, observed_one):
        monkeypatch.setattr(socket, "getaddrinfo", no_dns)
        peers = transport._Peers()
        peers.start({"worker_a": (zero, "a" * 32), "worker_b": (one, "b" * 32)},
                                 2, lambda: True)
        peers.request_id = "1" * 64
        peers.live[:] = [True, True]
        started = time.monotonic()
        peers.close()
        assert time.monotonic() - started < 0.75
        assert not any(peers.live)
        for observed, key in ((observed_zero, "a" * 32), (observed_one, "b" * 32)):
            assert observed == [(f"/v1/delegation/{peers.request_id}/cancel", "Bearer " + key, b"")]
        assert_retired(peers)


def test_stalled_pool_close_retires_other_pool_and_joins_owner(monkeypatch):
    monkeypatch.setattr(transport, "_IO_DRAIN_TIMEOUT", 0.05)
    peers = transport._Peers()
    peers.start(connections("http://127.0.0.1:2"), 1, lambda: False)
    original = peers.clients[0].aclose

    async def stalled():
        try:
            await asyncio.sleep(60)
        finally:
            await original()

    monkeypatch.setattr(peers.clients[0], "aclose", stalled)
    started = time.monotonic()
    with pytest.raises(ProtocolError, match="pool retirement timed out"):
        peers.close()
    assert time.monotonic() - started < 0.75
    assert_retired(peers)


def test_partial_client_construction_failure_joins_io_owner(monkeypatch):
    owners, clients = [], []
    owner_type, client_type = transport._IOOwner, transport.httpx.AsyncClient

    def owner():
        value = owner_type()
        owners.append(value)
        return value

    def client(*args, **kwargs):
        if clients:
            raise RuntimeError("controlled second client construction failure")
        value = client_type(*args, **kwargs)
        clients.append(value)
        return value

    monkeypatch.setattr(transport, "_IOOwner", owner)
    monkeypatch.setattr(transport.httpx, "AsyncClient", client)
    peers = transport._Peers()
    try:
        with pytest.raises(RuntimeError, match="controlled second client"):
            peers.start(connections("http://127.0.0.1:2"), 1, lambda: False)
    finally:
        with pytest.raises(ProtocolError, match="client acquisition is unconfirmed"):
            peers.close()
    assert not peers._close_succeeded and peers.io._close_succeeded
    assert clients[0].is_closed
    assert len(owners) == 1 and not owners[0].thread.is_alive()
    assert owners[0].loop.is_closed() and not asyncio.all_tasks(owners[0].loop)


@pytest.mark.parametrize("mode", ["single-byte", "empty", "exact", "overflow", "cancel", "error", "status"])
def test_fixed_response_collector_retires_buffers_and_conserves_counts(monkeypatch, mode):
    """Retain failures while checking allocation retirement without GC."""
    buffers, allocations, closed = [], [], []

    class Buffer(bytearray):
        pass

    def allocate(size):
        value = Buffer(size)
        buffers.append(weakref.ref(value))
        allocations.append(size)
        return value

    monkeypatch.setattr(transport, "bytearray", allocate, raising=False)
    pieces = {"single-byte": [b"a", b"", b"b", b"c"], "empty": [b"", b""],
              "exact": [b"abc"], "overflow": [b"ab", b"cd"],
              "cancel": [b"a"], "error": [b"a"], "status": [b"abc"]}[mode]

    class Response:
        headers = {"x-pllm-server-ns": "123"}
        status_code = 400 if mode == "status" else 200
        request = SimpleNamespace(content=b"frame")

        async def aiter_raw(self):
            try:
                for piece in pieces:
                    yield piece
                if mode == "cancel":
                    raise asyncio.CancelledError("collector cancelled")
                if mode == "error":
                    raise ValueError("collector failed")
            finally:
                closed.append(True)

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *args):
            return False

    peers = object.__new__(transport._Peers)
    peers._close_gate = threading.Lock()
    peers._close_succeeded = False
    peers.roles, peers.namespace, peers.request_id = ("worker_a", "worker_b"), "delegation", "a" * 64
    peers.clients = [SimpleNamespace(stream=lambda *a, **k: Stream())]
    peers.counts = {f"role_link.worker_a.online_{direction}_bytes": 0 for direction in ("upload", "download")}
    peers.phase, peers.phase_bytes = "prefill", {"prefill": 0, "decode": 0}
    peers.unknown = set()
    peers.calls = peers.server_ns = 0
    peers.server_known = True

    async def run():
        if mode in {"overflow", "cancel", "error", "status"}:
            error = asyncio.CancelledError if mode == "cancel" else ValueError if mode == "error" else ProtocolError
            with pytest.raises(error) as caught:
                await peers._post(0, "step", b"frame", phase="online", limit=3, timeout=1)
            assert caught.value.__traceback__ is not None
            trace = caught.value.__traceback__
            while trace.tb_frame.f_code.co_name != "_post":
                trace = trace.tb_next
            assert all(trace.tb_frame.f_locals[name] is None
                       for name in ("buffer", "view", "prefix_view", "chunk", "payload"))
        else:
            assert await peers._post(0, "step", b"frame", phase="online", limit=3, timeout=1) == b"".join(pieces)
        assert all(ref() is None for ref in buffers)

    asyncio.run(run())
    downloaded = sum(map(len, pieces))
    assert allocations == [3] and closed == [True]
    assert peers.counts == {"role_link.worker_a.online_upload_bytes": 5,
                            "role_link.worker_a.online_download_bytes": downloaded}
    assert peers.phase_bytes == {"prefill": 5 + downloaded, "decode": 0}
    assert not peers.unknown
    assert peers.calls == (0 if mode in {"overflow", "cancel", "error", "status"} else 1)


@pytest.fixture
def shared_owner(monkeypatch):
    """Real shared owners and HTTPX clients; fake HTTP, no endpoint or ledger."""
    events = []
    client_type = transport.httpx.AsyncClient

    async def handler(request):
        events.append(("remote", request.url.port))
        return transport.httpx.Response(200, content=b"")

    def client(**options):
        value = client_type(transport=transport.httpx.MockTransport(handler), **options)
        close = value.aclose

        async def observed():
            events.append(("pool", options["base_url"]))
            await close()

        monkeypatch.setattr(value, "aclose", observed)
        return value

    monkeypatch.setattr(transport.httpx, "AsyncClient", client)
    peers = transport._Peers()
    peers.start(connections("http://127.0.0.1:2"), 1, lambda: False)
    peers.request_id = "a" * 64
    peers.live[:] = [True, True]
    try:
        yield peers, events
    finally:
        if not peers.closed:
            peers.close()
        # Failed logical close must not be retried to normalize fixture state.
        # Tests needing physical teardown perform it explicitly, without refund.
        assert_retired(peers)


def assert_sticky(close, first):
    errors = []
    for _ in range(2):
        with pytest.raises(ProtocolError, match="previously failed") as caught:
            close()
        errors.append(caught.value)
        assert caught.value is not first
        assert caught.value.__context__ is caught.value.__cause__ is None
    assert errors[0] is not errors[1]


def test_shared_owner_success_publishes_only_after_whole_sequence(shared_owner, monkeypatch):
    peers, events = shared_owner
    finalizer, join, close = peers.io.loop.shutdown_asyncgens, peers.io.thread.join, peers.io.loop.close

    async def finalized():
        events.append("finalizer")
        assert not peers._close_succeeded and not peers.io._close_succeeded
        await finalizer()

    def joined(timeout):
        events.append("join")
        assert not peers._close_succeeded and not peers.io._close_succeeded
        join(timeout)

    def closed():
        events.append("loop")
        assert not peers._close_succeeded and not peers.io._close_succeeded
        close()

    monkeypatch.setattr(peers.io.loop, "shutdown_asyncgens", finalized)
    monkeypatch.setattr(peers.io.thread, "join", joined)
    monkeypatch.setattr(peers.io.loop, "close", closed)
    peers.close()
    assert peers._close_succeeded and peers.io._close_succeeded
    assert events == [("remote", 2), ("remote", 1),
                      ("pool", "http://127.0.0.1:2"), ("pool", "http://127.0.0.1:1"),
                      "finalizer", "join", "loop"]
    before = list(events)
    peers.close()
    peers.io.close(peers.clients)
    assert events == before


@pytest.mark.parametrize("failure", ["pool", "finalizer", "cancelled_pool", "cancelled_finalizer"])
def test_shared_owner_failed_cleanup_stays_sticky(shared_owner, monkeypatch, failure):
    peers, events = shared_owner
    error = RuntimeError("controlled retirement failure")
    original = peers.clients[0].aclose if "pool" in failure else peers.io.loop.shutdown_asyncgens

    async def failed():
        await original()
        if failure.startswith("cancelled"):
            raise asyncio.CancelledError()
        raise error

    if "pool" in failure:
        monkeypatch.setattr(peers.clients[0], "aclose", failed)
    else:
        monkeypatch.setattr(peers.io.loop, "shutdown_asyncgens", failed)
    expected = ProtocolError if failure.startswith("cancelled") else RuntimeError
    with pytest.raises(expected) as caught:
        peers.close()
    if expected is RuntimeError:
        assert caught.value is error
    else:
        assert "task was cancelled" in str(caught.value)
    assert not peers._close_succeeded and not peers.io._close_succeeded
    before = list(events)

    def forbidden(*args, **kwargs):
        pytest.fail("sticky close entered cleanup or dispatched new work")

    monkeypatch.setattr(peers, "_close_owned", forbidden)
    monkeypatch.setattr(peers.io, "_close_owned", forbidden)
    monkeypatch.setattr(peers.io, "run", forbidden)
    monkeypatch.setattr(peers.io.thread, "join", forbidden)
    assert_sticky(peers.close, caught.value)
    assert_sticky(lambda: peers.io.close(peers.clients), caught.value)
    assert events == before
    assert_retired(peers)


def test_remote_and_io_failure_keep_io_primary(shared_owner, monkeypatch):
    peers, events = shared_owner
    remote_error, io_error = LookupError("remote dispatch"), RuntimeError("pool cleanup")

    def remote(coroutine, *, timeout):
        coroutine.close()
        events.append("remote failure")
        raise remote_error

    original = peers.clients[0].aclose

    async def pool():
        await original()
        raise io_error

    monkeypatch.setattr(peers.io, "run", remote)
    monkeypatch.setattr(peers.clients[0], "aclose", pool)
    with pytest.raises(RuntimeError) as caught:
        peers.close()
    assert caught.value is io_error and io_error.__context__ is remote_error
    before = list(events)
    assert_sticky(peers.close, io_error)
    assert events == before


def test_remote_only_failure_never_becomes_peer_success(shared_owner, monkeypatch):
    peers, events = shared_owner

    def remote(coroutine, *, timeout):
        coroutine.close()
        raise LookupError("remote dispatch")

    monkeypatch.setattr(peers.io, "run", remote)
    with pytest.raises(LookupError) as caught:
        peers.close()
    assert peers.io._close_succeeded and not peers._close_succeeded
    before = list(events)
    assert_sticky(peers.close, caught.value)
    assert events == before


def test_finite_cancellation_resistant_pool_keeps_timeout_precedence(shared_owner, monkeypatch):
    peers, events = shared_owner
    monkeypatch.setattr(transport, "_IO_DRAIN_TIMEOUT", .01)
    original = peers.clients[0].aclose
    cancellations = []

    async def resistant():
        try:
            while len(cancellations) < 4:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cancellations.append(True)
        finally:
            await original()

    monkeypatch.setattr(peers.clients[0], "aclose", resistant)
    with pytest.raises(ProtocolError, match="pool retirement timed out") as caught:
        peers.close()
    assert len(cancellations) == 4
    assert_retired(peers)
    before = list(events)
    assert_sticky(peers.close, caught.value)
    assert_sticky(lambda: peers.io.close(peers.clients), caught.value)
    assert events == before


def test_cancelled_finalizer_preserves_existing_pool_error(shared_owner, monkeypatch):
    peers, _ = shared_owner
    error = RuntimeError("pool error wins over cancelled finalizer")
    original = peers.clients[0].aclose

    async def failed_pool():
        await original()
        raise error

    async def cancelled_finalizer():
        raise asyncio.CancelledError()

    monkeypatch.setattr(peers.clients[0], "aclose", failed_pool)
    monkeypatch.setattr(peers.io.loop, "shutdown_asyncgens", cancelled_finalizer)
    with pytest.raises(RuntimeError) as caught:
        peers.close()
    assert caught.value is error
    assert not peers._close_succeeded and not peers.io._close_succeeded


def test_remembered_caller_interruption_stays_failed_after_physical_cleanup(shared_owner, monkeypatch):
    peers, events = shared_owner
    interrupted = KeyboardInterrupt("controlled caller interruption")
    injected = []
    result = Future.result

    def interrupt_once(future, timeout=None):
        if peers.io._closing and not injected:
            injected.append(True)
            # Keep the actual drain Future. An otherwise successful retirement
            # still cannot publish success after reporting caller interruption.
            raise interrupted
        return result(future, timeout=timeout)

    monkeypatch.setattr(Future, "result", interrupt_once)
    with pytest.raises(KeyboardInterrupt) as caught:
        peers.close()
    assert caught.value is interrupted and injected == [True]
    assert_retired(peers)
    assert not peers._close_succeeded and not peers.io._close_succeeded
    before = list(events)
    assert_sticky(peers.close, caught.value)
    assert_sticky(lambda: peers.io.close(peers.clients), caught.value)
    assert events == before


def test_concurrent_close_promptly_rejects_without_poisoning_producer_drain(shared_owner):
    peers, events = shared_owner
    ready, draining, release = threading.Event(), threading.Event(), threading.Event()

    async def producer():
        ready.set()
        while not release.is_set():
            try:
                await asyncio.sleep(.01)
            except asyncio.CancelledError:
                draining.set()

    errors = []

    def close():
        try:
            peers.close()
        except BaseException as error:
            errors.append(error)

    closer = None
    body_failed = True
    try:
        future = asyncio.run_coroutine_threadsafe(producer(), peers.io.loop)
        assert ready.wait(2)
        closer = threading.Thread(target=close)
        closer.start()
        assert draining.wait(2)
        before = list(events)
        started = time.monotonic()
        for attempt in (peers.close, lambda: peers.io.close(peers.clients)):
            with pytest.raises(ProtocolError, match="already active"):
                attempt()
        assert time.monotonic() - started < .5
        assert events == before
        assert not peers._close_succeeded and not peers.io._close_succeeded
        body_failed = False
    finally:
        release.set()
        # Standard Thread.start publishes ident even if a wrapper then raises.
        # No ident: never join an unstarted closer; fixture owns retirement.
        if closer is not None and closer.ident is not None:
            try:
                closer.join(3)
            except BaseException:
                if not body_failed:
                    raise
    assert not closer.is_alive() and not errors
    assert future.done() and peers._close_succeeded and peers.io._close_succeeded


def test_own_thread_and_reentrant_callback_reject_without_self_join(shared_owner, monkeypatch):
    peers, events = shared_owner

    async def own_thread():
        for close in (peers.close, lambda: peers.io.close(peers.clients)):
            with pytest.raises(ProtocolError, match="own thread|I/O thread"):
                close()
        assert not peers.closed and not peers.io._closing

    peers.io.run(own_thread(), timeout=1)
    pool = peers.clients[0].aclose

    async def callback():
        # Callback runs on I/O thread while both external close gates are held.
        for close in (peers.close, lambda: peers.io.close(peers.clients)):
            with pytest.raises(ProtocolError, match="own thread|I/O thread"):
                close()
        await pool()

    join = peers.io.thread.join

    def reentrant(timeout):
        for close in (peers.close, lambda: peers.io.close(peers.clients)):
            with pytest.raises(ProtocolError, match="already active"):
                close()
        join(timeout)

    monkeypatch.setattr(peers.clients[0], "aclose", callback)
    monkeypatch.setattr(peers.io.thread, "join", reentrant)
    peers.close()
    assert peers._close_succeeded and peers.io._close_succeeded


def test_join_failure_stays_sticky_after_later_physical_thread_death(shared_owner, monkeypatch):
    peers, events = shared_owner
    monkeypatch.setattr(transport, "_IO_DRAIN_TIMEOUT", .05)
    release, stopping = threading.Event(), threading.Event()
    stop, join, loop_close = peers.io.loop.stop, peers.io.thread.join, peers.io.loop.close
    joins = []

    def held_stop():
        stopping.set()
        assert release.wait(3), "controlled physical teardown not released"
        stop()

    def bounded_join(timeout):
        joins.append(timeout)
        assert stopping.wait(2)
        join(timeout)

    monkeypatch.setattr(peers.io.loop, "stop", held_stop)
    monkeypatch.setattr(peers.io.thread, "join", bounded_join)
    try:
        with pytest.raises(ProtocolError, match="thread did not retire") as caught:
            peers.close()
        assert peers.io.thread.is_alive() and not peers.io.loop.is_closed()
        assert_sticky(peers.close, caught.value)
        assert_sticky(lambda: peers.io.close(peers.clients), caught.value)
        assert joins == [.05]
    finally:
        # Physical teardown only: no flag normalization or logical refund claim.
        release.set()
        join(3)
        assert not peers.io.thread.is_alive()
        loop_close()
    before = list(events)
    assert_sticky(peers.close, None)
    assert_sticky(lambda: peers.io.close(peers.clients), None)
    assert joins == [.05] and events == before
    assert not peers._close_succeeded and not peers.io._close_succeeded


@pytest.mark.parametrize("owner_type,fallible", [
    (transport._Peers, "_connections"), (transport._IOOwner, "_NetworkLoop")])
def test_close_controls_precede_fallible_partial_construction(monkeypatch, owner_type, fallible):
    owner = owner_type()

    def fail(*args, **kwargs):
        assert not owner._close_succeeded
        assert owner._close_gate.acquire(blocking=False)
        owner._close_gate.release()
        raise ValueError("partial construction")

    monkeypatch.setattr(transport, fallible, fail)
    with pytest.raises(ValueError, match="partial construction"):
        if owner_type is transport._Peers:
            owner.start({}, 1, lambda: False)
        else:
            owner.start()
    # A rejected configuration owns nothing; an opaque failed loop acquisition
    # has no retirement witness. Neither path relies on missing attributes.
    close = owner.close if owner_type is transport._Peers else lambda: owner.close([])
    if owner_type is transport._Peers:
        close()
        assert owner._close_succeeded
        close()
    else:
        with pytest.raises(ProtocolError, match="acquisition is unconfirmed") as caught:
            close()
        assert not owner._close_succeeded
        assert_sticky(close, caught.value)


def _assert_physical_close_failure(shared_owner, monkeypatch, step, pool_failure):
    peers, events = shared_owner
    loop, thread = peers.io.loop, peers.io.thread
    dispatch, stop, join, loop_close = loop.call_soon_threadsafe, loop.stop, thread.join, loop.close
    physical_error = RuntimeError(f"controlled {step} failure")
    pool_error = LookupError("controlled preceding pool failure")
    calls = []
    pool_close = peers.clients[0].aclose

    def dispatched(callback, *args, **kwargs):
        calls.append("stop" if callback == stop else "dispatch")
        if step == "stop" and callback == stop:
            raise physical_error
        return dispatch(callback, *args, **kwargs)

    def joined(timeout):
        calls.append("join")
        if step == "join":
            raise physical_error
        return join(timeout)

    def closed():
        calls.append("loop close")
        if step == "loop close":
            raise physical_error  # Fail before touching the real loop close.
        return loop_close()

    async def failed_pool():
        await pool_close()
        raise pool_error

    with monkeypatch.context() as patch:
        patch.setattr(loop, "call_soon_threadsafe", dispatched)
        patch.setattr(thread, "join", joined)
        patch.setattr(loop, "close", closed)
        if pool_failure:
            patch.setattr(peers.clients[0], "aclose", failed_pool)
        try:
            with pytest.raises(RuntimeError) as caught:
                peers.close()
            assert caught.value is physical_error
            assert physical_error.__context__ is (pool_error if pool_failure else None)
            assert not peers._close_succeeded and not peers.io._close_succeeded
            assert all(client.is_closed for client in peers.clients)
            before = (list(events), list(calls))
            assert_sticky(peers.close, physical_error)
            assert_sticky(lambda: peers.io.close(peers.clients), physical_error)
            assert (events, calls) == before
        finally:
            # Physical teardown only. Always try the saved join, even if saved
            # stop dispatch fails; do not close a running loop or retry owners.
            failures = []
            try:
                if thread.is_alive():
                    dispatch(stop)
            except BaseException as error:
                failures.append(error)
            finally:
                try:
                    join(3)
                except BaseException as error:
                    failures.append(error)
            if thread.is_alive():
                failures.append(AssertionError("physical teardown left a live I/O thread"))
            else:
                # Inspect tasks after thread death and BEFORE physical loop close.
                # No coroutine dispatch to a dead thread, no hidden logical retry.
                try:
                    assert not asyncio.all_tasks(loop)
                except BaseException as error:
                    failures.append(error)
                try:
                    if not loop.is_closed():
                        loop_close()
                except BaseException as error:
                    failures.append(error)
            try:
                assert all(client.is_closed for client in peers.clients)
                assert not thread.is_alive() and loop.is_closed()
            except BaseException as error:
                failures.append(error)
            if failures:
                raise BaseExceptionGroup("controlled physical teardown failed", failures)

        assert_retired(peers)
        assert not peers._close_succeeded and not peers.io._close_succeeded
        before = (list(events), list(calls))
        assert_sticky(peers.close, physical_error)
        assert_sticky(lambda: peers.io.close(peers.clients), physical_error)
        assert (events, calls) == before
    # Restore patched methods before shared_owner's independent fixture teardown.


@pytest.mark.parametrize("pool_failure", [False, True])
def test_stop_dispatch_failure_stays_sticky_after_physical_cleanup(shared_owner, monkeypatch, pool_failure):
    _assert_physical_close_failure(shared_owner, monkeypatch, "stop", pool_failure)


@pytest.mark.parametrize("pool_failure", [False, True])
def test_join_exception_stays_sticky_after_physical_cleanup(shared_owner, monkeypatch, pool_failure):
    _assert_physical_close_failure(shared_owner, monkeypatch, "join", pool_failure)


@pytest.mark.parametrize("pool_failure", [False, True])
def test_loop_close_exception_stays_sticky_after_physical_cleanup(shared_owner, monkeypatch, pool_failure):
    _assert_physical_close_failure(shared_owner, monkeypatch, "loop close", pool_failure)


def test_gkm_execute_real_sticky_peers_keep_original_charge_after_gc(monkeypatch):
    """Real selector I/O and HTTPX, no network/native factory or decoder work.

    Normal module imports still load the native extension. Thread creation and
    GC occur only when this test runs, never as source-review evidence.
    """
    import gc
    import json

    from pllm.deployment import resources

    real_peers, real_io = transport._Peers, transport._IOOwner
    real_client = transport.httpx.AsyncClient
    real_reservation = resources.reserve_incremental_memory
    capacity = resources._ClientCapacity()
    refs, tokens, handles = {}, [], {}
    aborts, pool_closes, remote = [], [], []
    contract = {"input_rows": 1, "decode_steps": 0, "vocabulary": 2, "matrices": [{}]}

    def detach(error):
        pending, seen = [error], set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            pending.extend(item for item in (value.__cause__, value.__context__)
                           if item is not None)
            value.__traceback__ = value.__cause__ = value.__context__ = None

    class Native:
        def contract(self):
            return b"bound"

        def decoder(self):
            return json.dumps(contract).encode()

        def publication(self):
            return b"publication"

        def publication_digest(self):
            return b"pin"

        def starts(self):
            return b"a", b"b"

        def request_id(self):
            return bytes(32)

        def abort(self):
            aborts.append(True)

    class Source:
        capacity = "capsule"
        execution_payload_estimate_bytes = 1024

        def runtime_contract(self, *args):
            return b"bound"

        def load_transport(self, *args, **kwargs):
            value = Native()
            refs["native"] = weakref.ref(value)
            return value

    source = Source()
    source.contract = contract
    caller = lambda: False  # No owner/handoff seat: ledger must be the sole root.
    refs["source"], refs["caller"] = weakref.ref(source), weakref.ref(caller)
    amount = transport.request_memory_bytes(source)  # Unchanged production price.

    def observe():
        return resources.ClientResources(
            observed_at_ms=0, available_memory_bytes=amount,
            process_memory_bytes=1, reserve_bytes=0,
            reserved_memory_bytes=sum(capacity._owners.values()),
        )

    def reservation(size):
        value = real_reservation(size)
        refs["reservation"] = weakref.ref(value)
        tokens.append(id(value._token))
        return value

    async def handler(request):
        remote.append((request.url.port, request.url.path))
        return transport.httpx.Response(200, content=b"")

    def client(**options):
        index = len(pool_closes)
        pool_closes.append(0)
        value = real_client(transport=transport.httpx.MockTransport(handler), **options)
        close = value.aclose

        async def observed_close():
            await close()  # Physical HTTPX close precedes the injected failure.
            pool_closes[index] += 1
            if index == 0:
                raise RuntimeError("controlled HTTPX pool retirement failure")

        value.aclose = observed_close
        return value

    real_start = real_peers.start

    def peers_start(value, *args, **kwargs):
        real_start(value, *args, **kwargs)
        # Capture before assertions; never keep a strong peer collection.
        handles["thread"], handles["loop"] = value.io.thread, value.io.loop
        refs["peers"], refs["io"] = weakref.ref(value), weakref.ref(value.io)
        assert type(value) is real_peers and type(value.io) is real_io
        assert type(value).close is real_peers.close
        assert type(value.io).close is real_io.close

    def admit(self, bound, publication, pin, starts, experiment, source,
              request_id, rows, outputs):
        # Protocol boundary only: all execute_request validators remain real.
        self.request_id = request_id.hex()
        self.live[:] = [True, True]

    def operation(*args, **kwargs):
        raise ValueError("operation failed")

    def assert_charge():
        assert len(tokens) == 1
        assert len(capacity._owners) == 1
        token = next(iter(capacity._owners))
        value = refs["reservation"]()
        peer, io = refs["peers"](), refs["io"]()
        assert value is not None and peer is not None and io is not None
        assert id(token) == tokens[0] and value._token is token
        assert token.retained is value and value.pending_retirement
        assert capacity._owners == {token: amount}
        assert sum(capacity._owners.values()) == amount
        assert value._amount == amount and value._failed == (peer,)
        assert type(peer) is real_peers and type(io) is real_io and peer.io is io
        assert not peer._close_succeeded and not io._close_succeeded
        assert all(pool.is_closed for pool in peer.clients)
        assert not handles["thread"].is_alive() and handles["loop"].is_closed()
        assert refs["native"]() is None
        assert refs["source"]() is None
        assert refs["caller"]() is peer.cancelled
        assert aborts == [True] and pool_closes == [1, 1]
        assert sorted(remote) == [
            (1, "/v1/delegation/" + "0" * 64 + "/cancel"),
            (2, "/v1/delegation/" + "0" * 64 + "/cancel"),
        ]

    try:
        with monkeypatch.context() as patch:
            patch.setattr(resources, "_capacity", capacity)
            patch.setattr(capacity, "observe", observe)
            patch.setattr(transport, "reserve_incremental_memory", reservation)
            patch.setattr(transport.httpx, "AsyncClient", client)
            patch.setattr(real_peers, "start", peers_start)
            patch.setattr(real_peers, "admit", admit)
            patch.setattr(transport, "_execute_verified_window", operation)
            try:
                with pytest.raises(ValueError, match="^operation failed$") as caught:
                    transport.execute_request(
                        None, source, (0,), 1, connections("http://127.0.0.1:2"), 1, caller,
                    )
                assert type(caught.value) is ValueError
                assert caught.value.__notes__ == ["GKM cleanup failed: RuntimeError"]
                assert caught.value.__cause__ is caught.value.__context__ is None
                detach(caught.value)
                del caught, source, caller
                gc.collect()
                assert_charge()

                with pytest.raises(resources.NetworkError,
                                   match="CLIENT_MEMORY_CAPACITY_CHANGED") as caught:
                    with real_reservation(amount):
                        pytest.fail("sticky original charge must refuse follow-on admission")
                detach(caught.value)
                del caught
                gc.collect()
                assert_charge()

                for _ in range(2):
                    with pytest.raises(ProtocolError, match="retirement previously failed") as caught:
                        refs["reservation"]().drain_failed()
                    detach(caught.value)
                    del caught
                    gc.collect()
                    assert_charge()
            finally:
                # Independent physical attempts only. Never retry admitted close,
                # alter either witness, or treat emergency disposal as refund.
                failures = []
                if handles:
                    thread, loop = handles["thread"], handles["loop"]
                    try:
                        if thread.is_alive():
                            failures.append("emergency physical I/O stop required")
                            loop.call_soon_threadsafe(loop.stop)
                    except BaseException as error:
                        failures.append("physical stop: " + type(error).__name__)
                        detach(error)
                    try:
                        thread.join(2)
                    except BaseException as error:
                        failures.append("physical join: " + type(error).__name__)
                        detach(error)
                    try:
                        if thread.is_alive():
                            failures.append("physical I/O thread still alive")
                        elif not loop.is_closed():
                            failures.append("emergency physical loop disposal required")
                            loop.close()
                    except BaseException as error:
                        failures.append("physical loop disposal: " + type(error).__name__)
                        detach(error)
                assert not failures, "; ".join(failures)
    finally:
        # Discard isolated ledger only AFTER monkeypatch restoration. This is
        # test-state disposal, never successful retirement or a refund oracle.
        capacity = None
