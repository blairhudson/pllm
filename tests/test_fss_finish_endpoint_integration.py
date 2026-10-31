"""Real client I/O and authenticated ASGI/session/tape paths, bounded native doubles.

Only sockets, source/native arithmetic and host telemetry are doubled. No model,
cryptographic, TLS, physical-memory or trained-timeout qualification is claimed.
No ignored target artifacts are imported. Existing worker fixtures own the apps.
"""
import asyncio
from contextlib import contextmanager
import threading
from types import SimpleNamespace as NS

import httpx
import pytest
import pytest_asyncio

from pllm.deployment import resources
from pllm.runtime import delegated_transport
from pllm.runtime import fss_protocol as p
from pllm.runtime import fss_transport as client
from pllm.runtime import fss_worker as worker
from pllm.runtime.client import ProtocolError
from test_fss_control_boundary import core
from test_fss_worker import (  # noqa: F401 -- reusable worker/native/tape fixtures
    Frozen, HTTPClient, ORIGINS, client_key, cluster, configuration, double,
)


@pytest_asyncio.fixture
async def endpoint(cluster, monkeypatch):
    loop = asyncio.get_running_loop()
    state = NS(mode="job_failure", failing=1, frozen=None, peers=[], http_clients=[],
               http=[], server_tasks=set(), sessions=[], started=set(), exited=set(),
               finish_entered=set(), response_ready=set(), response_both=asyncio.Event(),
               fault_release=threading.Event(), observed_wait=None, reconstructed=0,
               parsed=[], valid_endpoint_receipts=set(), late_markers=[],
               cleanup=[], watchdog=False, native_close_marker=None)
    owner = core()
    state.attempt = owner.events({})

    def freeze(marker):
        if state.frozen is None:
            state.frozen = marker

    owner._fss_failure_observer = freeze

    # Keep the real shared ledger/leases, but make its host telemetry deterministic.
    # Existing fixture spies still check role/disk owner ordering; this is not RSS.
    capacity = resources._capacity
    before = dict(capacity._owners)
    monkeypatch.setattr(capacity, "observe", lambda: resources.ClientResources(
        1, 8 << 30, 1 << 20, 0, sum(capacity._owners.values())))
    fixture_reserve = worker.reserve_incremental_memory

    @contextmanager
    def role_reserve(amount):
        with resources.reserve_incremental_memory(amount), fixture_reserve(amount) as value:
            yield value

    monkeypatch.setattr(worker, "reserve_incremental_memory", role_reserve)

    @contextmanager
    def client_reserve(amount):
        with resources.reserve_incremental_memory(amount):
            try:
                yield
            finally:
                assert all(c.is_closed for c in state.http_clients)
                assert all(n.closed and not n.running for n in cluster.double.natives)
                assert all(s.closed for s in cluster.double.loaded)
                assert all(r["closed"] for r in cluster.double.storage_reservations)
                state.cleanup.append("client_reservation.release")

    monkeypatch.setattr(client, "reserve_incremental_memory", client_reserve)

    original_init = worker._Attempt.__init__

    def session_init(session, *args, **kwargs):
        original_init(session, *args, **kwargs)
        state.sessions.append(session)

    monkeypatch.setattr(worker._Attempt, "__init__", session_init)
    native_type = worker._native.FssRoleRequest
    original_run = native_type.run

    def native_run(native, frame, callbacks):
        if native.role == 2:
            return original_run(native, frame, callbacks)
        native.callbacks = callbacks
        state.started.add(native.role)
        if state.mode == "caller_cancel" or (
                state.mode == "job_failure" and native.role == state.failing):
            native.running = True
            cluster.double.native_threads.append(threading.get_ident())
            try:
                assert frame == f"input-{native.role}".encode()
                assert cluster.double.sealed_roles == set(p.HOLDERS)
                if state.mode == "caller_cancel":
                    assert native.stop.wait(2), "mock job cancellation was not delivered"
                    raise ValueError("bounded mock job cancelled")
                assert state.fault_release.wait(2), "finish/waiter gate was not reached"
                raise ValueError("bounded mock job failed before sending its peer frame")
            finally:
                native.running = False
                state.exited.add(native.role)
        try:
            result = original_run(native, frame, callbacks)
            if state.mode == "bad_native_result" and native.role == state.failing:
                return b""  # Rejected by real worker run/finish, not by a fake HTTP exception.
            return result
        finally:
            state.exited.add(native.role)

    monkeypatch.setattr(native_type, "run", native_run)

    class NativeClient:
        def __init__(self, contract, session, manifests, ids):
            assert ids == [1, 2, 3] and len(manifests) == 3
            self.session = session.hex()
            self.inputs = []

        def request_bytes(self):
            return b"request:" + self.session.encode()

        def take_input(self, index):
            assert all(n.closed for n in cluster.double.natives if n.role == 2)
            assert cluster.double.sealed_roles == set(p.HOLDERS)
            assert sum(capacity._owners.values()) - sum(before.values()) == (
                p.CLIENT_MEMORY + 2 * p.ROLE_MEMORY)
            self.inputs.append(index)
            return f"input-{index}".encode()

        def finish(self, *frames):
            state.reconstructed += 1
            assert state.mode == "success", "a partial/failed completion reached reconstruction"
            assert self.inputs == [0, 1] and frames == (b"final-0", b"final-1")
            assert set(p.HOLDERS) <= set(state.parsed)
            return [5, 6, 7]

        def close(self):
            state.native_close_marker = state.frozen
            state.cleanup.append("client_native.close")

    monkeypatch.setattr(client._native, "FssClientRequest", NativeClient, raising=False)
    original_receipt = client._receipt

    def parsed(raw, role, **options):
        state.parsed.append(role)
        return original_receipt(raw, role, **options)

    monkeypatch.setattr(client, "_receipt", parsed)

    class LoopASGI(httpx.AsyncBaseTransport):
        """Socket replacement only; actual endpoints execute on their owning loop."""

        def __init__(self, role):
            self.role = role

        async def handle_async_request(self, request):
            role = self.role
            action = request.url.path.rsplit("/", 1)[-1]
            is_finish = action == "finish" and role in p.HOLDERS

            async def dispatch():
                task = asyncio.current_task()
                state.server_tasks.add(task)
                state.http.append((role, action, request.headers["authorization"]))
                if is_finish:
                    state.finish_entered.add(role)
                try:
                    response = await httpx.ASGITransport(
                        app=cluster.apps[role]).handle_async_request(request)
                    body = await response.aread()
                    await response.aclose()
                    if is_finish and response.status_code == 200:
                        original_receipt(body, role)
                        state.valid_endpoint_receipts.add(role)
                        state.response_ready.add(role)
                        if state.response_ready == set(p.HOLDERS):
                            state.response_both.set()
                    if is_finish and state.mode == "late_receipt":
                        await state.response_both.wait()
                    if is_finish and role == p.HOLDERS[state.failing] and state.mode in (
                            "malformed_receipt", "late_receipt"):
                        body = p.pack({"invalid": True})  # Explicit corruption after real endpoint.
                    return response.status_code, response.headers, body
                except BaseException as error:
                    # The in-process bridge owns server exception frames too.
                    # Release an unsent _PayloadResponse retained by cancellation
                    # during endpoint retirement, just as ordinary transport
                    # failure hygiene releases its completed payload frames.
                    p.clear_failure_frames(error)
                    raise
                finally:
                    state.server_tasks.discard(task)

            future = asyncio.run_coroutine_threadsafe(dispatch(), loop)
            try:
                status, headers, body = await asyncio.wrap_future(future)
                if is_finish and state.mode == "late_receipt" and role != p.HOLDERS[state.failing]:
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        state.late_markers.append(state.frozen)
                        # Return bytes from the actual successful endpoint even
                        # after initial cancellation; the collector must ignore them.
                return httpx.Response(status, headers=headers, stream=httpx.ByteStream(body))
            finally:
                future.cancel()

        async def aclose(self):
            state.cleanup.append(f"client_http.close:{self.role}")

    worker_clients = worker.httpx.AsyncClient  # Existing real peer-handler Routing fixture.

    def http_client(**kwargs):
        if "base_url" not in kwargs:
            return worker_clients(**kwargs)
        role = next(r for r in p.ROLES if kwargs["base_url"] == ORIGINS[r])
        assert kwargs["headers"]["authorization"] == f"Bearer {client_key(role)}"
        value = HTTPClient(transport=LoopASGI(role), **kwargs)
        state.http_clients.append(value)
        return value

    monkeypatch.setattr(delegated_transport.httpx, "AsyncClient", http_client)
    actual_peers = delegated_transport._Peers
    actual_start = actual_peers.start

    def start(value, *args, **kwargs):
        actual_start(value, *args, **kwargs)
        assert type(value) is actual_peers and type(value.io) is delegated_transport._IOOwner
        state.peers.append(value)

    monkeypatch.setattr(actual_peers, "start", start)  # Observe start; close/post unchanged.

    async def gate():
        if state.mode not in {"job_failure", "caller_cancel"}:
            return
        async with asyncio.timeout(2):
            while True:
                natives = {n.role: n for n in cluster.double.natives if n.role in (0, 1)}
                if state.finish_entered == set(p.HOLDERS) and state.started == {0, 1}:
                    attempts = [natives[i].callbacks.attempt for i in (0, 1)]
                    if all(a.control_busy for a in attempts):
                        if state.mode == "caller_cancel":
                            state.attempt.cancel()
                            return
                        waiter = attempts[1 - state.failing]
                        if waiter.peer_sent == 1 and waiter.mailbox is None:
                            state.observed_wait = (waiter.peer_sent, waiter.peer_received, waiter.mailbox)
                            state.fault_release.set()
                            return
                await asyncio.sleep(.001)

    async def run():
        monitor = asyncio.create_task(gate())
        request = asyncio.create_task(asyncio.to_thread(client.execute_request,
            configuration(), Frozen("client"), [1, 2, 3], 3,
            {r: (ORIGINS[r], client_key(r)) for r in p.ROLES}, 30, state.attempt))
        try:
            return await asyncio.wait_for(asyncio.shield(request), 3)
        except TimeoutError:
            state.watchdog = True
            state.attempt.cancel()
            state.fault_release.set()
            await asyncio.wait_for(asyncio.gather(request, return_exceptions=True), 7)
            raise AssertionError("client hid failure until integration watchdog") from None
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)

    async def assert_retired():
        assert not state.watchdog and state.exited == {0, 1}
        async with asyncio.timeout(2):
            while state.server_tasks:
                await asyncio.sleep(.001)
        for role in p.ROLES:
            metrics = (await cluster.clients[role].get("/v1/fss/metrics")).json()
            assert metrics["active_sessions"] == 0 and metrics["retired_sessions"] == 1
            assert metrics["seen_requests"] == 1
        assert len(state.sessions) == 3
        for session in state.sessions:
            assert session.phase == "closed" and session.reservation is None
            assert session.close_task is not None and session.close_task.done()
            assert not session.handlers and not session.callback_tasks and not session.callback_futures
            assert session.job is session.timer is session.native is session.tape is session.loaded is None
        jobs = {"run", "native_call", "_close", "callback_job", "expire"}
        assert not [task for task in asyncio.all_tasks() if not task.done()
                    and task.get_coro().cr_code.co_name in jobs
                    and task.get_coro().cr_code.co_filename.endswith((
                        "/fss_worker.py", "/replicated_worker.py"))]
        for peer in state.peers:
            assert peer.closed and peer.io.closed and peer.io.loop.is_closed()
            assert not peer.io.thread.is_alive() and not asyncio.all_tasks(peer.io.loop)
            assert not peer.io._fss_finish_pending
        assert all(c.is_closed for c in state.http_clients)
        assert all(r["closed"] for r in cluster.double.reservations)
        assert all(r["closed"] for r in cluster.double.storage_reservations)
        assert dict(capacity._owners) == before
        assert state.cleanup[0] == "client_native.close"
        assert state.cleanup[-1] == "client_reservation.release"
        assert all(auth == f"Bearer {client_key(role)}" for role, _, auth in state.http)
        holders = [(r, action) for r, action, _ in state.http if action in {"input", "start"}]
        assert holders == [(r, action) for action in ("input", "start") for r in p.HOLDERS]
        assert state.finish_entered == set(p.HOLDERS)

    state.run, state.assert_retired = run, assert_retired
    try:
        yield state
    finally:
        state.attempt.cancel()
        state.fault_release.set()
        owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", [0, 1])
async def test_actual_failed_job_interrupts_other_holder_waiting_for_peer(endpoint, failing):
    endpoint.failing = failing
    with pytest.raises(ProtocolError) as caught:
        await endpoint.run()
    assert caught.value.status_code == 400 and caught.value.body is None
    assert endpoint.observed_wait == (1, 0, None)
    assert endpoint.frozen[:3] == (p.HOLDERS[failing], "finish", "failed")
    assert endpoint.native_close_marker is endpoint.frozen and endpoint.reconstructed == 0
    assert {(r, a) for r, a, _ in endpoint.http if a == "cancel"} == {
        (r, "cancel") for r in p.HOLDERS}
    await endpoint.assert_retired()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", [0, 1])
@pytest.mark.parametrize("mode", ["bad_native_result", "malformed_receipt"])
async def test_actual_completion_path_rejects_bad_native_or_corrupted_receipt(endpoint, failing, mode):
    endpoint.mode, endpoint.failing = mode, failing
    with pytest.raises(ProtocolError):
        await endpoint.run()
    assert endpoint.frozen[:3] == (p.HOLDERS[failing], "finish", "failed")
    assert endpoint.reconstructed == 0
    await endpoint.assert_retired()


@pytest.mark.asyncio
async def test_caller_cancellation_drains_actual_finish_handlers_and_mock_jobs(endpoint):
    endpoint.mode = "caller_cancel"
    with pytest.raises(ProtocolError) as caught:
        await endpoint.run()
    assert caught.value.status_code == 499 and endpoint.reconstructed == 0
    assert endpoint.native_close_marker is endpoint.frozen
    await endpoint.assert_retired()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", [0, 1])
async def test_late_actual_valid_receipt_after_other_failure_is_never_parsed(endpoint, failing):
    endpoint.mode, endpoint.failing = "late_receipt", failing
    with pytest.raises(ProtocolError, match="completion envelope"):
        await endpoint.run()
    assert endpoint.valid_endpoint_receipts == set(p.HOLDERS)
    assert endpoint.late_markers == [endpoint.frozen]
    assert endpoint.parsed == ["dealer", p.HOLDERS[failing]]
    assert endpoint.frozen[:3] == (p.HOLDERS[failing], "finish", "failed")
    assert endpoint.reconstructed == 0
    await endpoint.assert_retired()


@pytest.mark.asyncio
async def test_success_control_reconstructs_only_after_both_real_validated_receipts(endpoint):
    endpoint.mode = "success"
    assert (await endpoint.run()).selected == (5, 6, 7)
    assert endpoint.reconstructed == 1 and endpoint.frozen is None
    await endpoint.assert_retired()
