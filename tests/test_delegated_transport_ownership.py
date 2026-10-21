"""Bounded socket probes for literal origins and response-owned I/O retirement."""

import asyncio
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
    with pytest.raises(ValueError, match="IP|DNS|scoped"):
        transport._Peers(selected, 1, lambda: False)


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
        peers = transport._Peers(connections(origin.replace("127.0.0.1", "localhost")), 2, lambda: False)
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
    peers = transport._Peers(connections("https://127.0.0.1:2"), 1, lambda: False)
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
        peers = transport._Peers(connections(origin), 2, cancelled.is_set)
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
        peers = transport._Peers({"worker_a": (zero, "a" * 32), "worker_b": (one, "b" * 32)},
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
    peers = transport._Peers(connections("http://127.0.0.1:2"), 1, lambda: False)
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
    with pytest.raises(RuntimeError, match="controlled second client"):
        transport._Peers(connections("http://127.0.0.1:2"), 1, lambda: False)
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
