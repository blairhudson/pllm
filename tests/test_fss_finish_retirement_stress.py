"""Real selector owner/HTTP reader, controlled cancellation-resistant byte streams.

No native math/model, sockets or new watcher/thread owner. Response-buffer views
model the local HTTP lifetime, NOT a borrowed client-native input or remote role
capacity. The endpoint integration suite separately covers actual ASGI jobs.
"""
import asyncio
from concurrent.futures import Future
from contextlib import contextmanager
import gc
import threading
from types import SimpleNamespace as NS

import httpx
import pytest

from pllm.deployment import resources
from pllm.runtime import delegated_transport as io_transport
from pllm.runtime import fss_protocol as p
from pllm.runtime import fss_transport as transport
from pllm.runtime.client import ProtocolError
from test_fss_control_boundary import core
from test_fss_finish_barrier import receipt


@pytest.fixture
def requests(monkeypatch):
    monkeypatch.setattr(io_transport, "_IO_DRAIN_TIMEOUT", .01)
    capacity = resources._capacity
    original_owners = dict(capacity._owners)
    monkeypatch.setattr(capacity, "observe", lambda: resources.ClientResources(
        1, 8 << 30, 1 << 20, 0, sum(capacity._owners.values())))
    actual_client = httpx.AsyncClient
    constructed = []
    current = None

    class ResponseBytes(httpx.AsyncByteStream):
        def __init__(self, state):
            self.state = state

        async def __aiter__(self):
            state = self.state
            view = memoryview(state.data)
            state.active += 1
            state.ready.set()
            try:
                while True:
                    try:
                        await asyncio.Future()  # No external Future roots this Task.
                    except asyncio.CancelledError:
                        state.cancellations += 1
                        state.cancel_snapshots.append(state.frozen)
                        if state.cancellations >= 2:
                            state.global_cancel.set()
                        if (state.cancellations < state.required_cancels
                                or state.hold and not state.release.is_set()):
                            continue
                        if state.late == "return":
                            yield bytes(view)
                            return
                        if state.late == "error":
                            raise state.cleanup_error
                        raise
            finally:
                view.release()
                state.active -= 1
                state.view_retired.set()

        async def aclose(self):
            pass

    def clients(**kwargs):
        state = current
        role = p.ROLES[int(httpx.URL(kwargs["base_url"]).port) - 7300]

        async def handler(request):
            assert request.headers["authorization"] == "Bearer " + str(p.ROLES.index(role)) * 32
            action = request.url.path.rsplit("/", 1)[-1]
            state.http.append((role, action))
            if action == "cancel":
                return httpx.Response(200, content=b"")
            assert action == "finish" and role in p.HOLDERS
            if role == p.HOLDERS[state.failing]:
                await state.ready.wait()
                raise state.original
            return httpx.Response(200, stream=ResponseBytes(state))

        value = actual_client(transport=httpx.MockTransport(handler), **kwargs)
        state.clients.append(value)
        return value

    monkeypatch.setattr(io_transport.httpx, "AsyncClient", clients)
    original_receipt = transport._receipt

    def parsed(raw, role, **options):
        current.parsed.append(role)
        return original_receipt(raw, role, **options)

    monkeypatch.setattr(transport, "_receipt", parsed)

    @contextmanager
    def request(failing=1, late="cancel", required_cancels=3, *, hold=False, observer_error=False):
        nonlocal current
        owner = core()
        attempt = owner.events({})
        state = NS(owner=owner, attempt=attempt, failing=failing, late=late,
                   required_cancels=required_cancels, hold=hold, observer_error=observer_error,
                   ready=asyncio.Event(), release=threading.Event(), global_cancel=threading.Event(),
                   view_retired=threading.Event(), active=0, cancellations=0, cancel_snapshots=[],
                   frozen=None, probes=[], parsed=[], http=[], clients=[],
                   original=ValueError("first holder failure"), cleanup_error=RuntimeError("late read cleanup"),
                   close_failed=False,
                   data=bytearray(receipt(p.HOLDERS[1 - failing])))
        current = state

        def freeze(marker):
            if state.frozen is None:
                state.frozen = marker
            # Model the real cached observer's nonblocking lock acquisition.
            acquired = owner._lock.acquire(blocking=False)
            state.probes.append(acquired)
            if acquired:
                owner._lock.release()
            if observer_error:
                raise RuntimeError("optional cached observer failed")

        owner._fss_failure_observer = freeze
        with resources.reserve_incremental_memory(p.CLIENT_MEMORY):
            peers = io_transport._Peers()
            peers.start(
                {role: (f"http://127.0.0.1:{7300 + index}", str(index) * 32)
                 for index, role in enumerate(p.ROLES)}, 23, attempt, roles=p.ROLES, namespace="fss")
            peers.request_id = "a" * 64
            peers.live = [True, True, False]
            state.peers = peers
            state.fds = (peers.io.loop._ssock, peers.io.loop._csock)
            constructed.append(state)
            try:
                yield state
            finally:
                state.release.set()
                try:
                    if state.close_failed:
                        with pytest.raises(ProtocolError, match="previously failed"):
                            peers.close()
                    else:
                        peers.close()
                finally:
                    attempt.close()
                    owner.close()
                assert state.view_retired.is_set() and state.active == 0
                state.data.clear()  # Actual export-count oracle: no surviving view.
                assert all(value.is_closed for value in state.clients)
                assert peers.io.closed and not peers.io.thread.is_alive()
                assert not asyncio.all_tasks(peers.io.loop)
                assert not peers.io._fss_finish_pending
                assert all(sock.fileno() == -1 for sock in state.fds)
    yield request
    assert capacity._owners == original_owners
    assert all(not state.peers.io.thread.is_alive() for state in constructed)


def fail(state):
    # Hold the client lock on the caller thread. The real _Attempt diagnostic
    # plus optional observer must not wait for it on the I/O thread.
    with state.owner._lock:
        with pytest.raises(ValueError) as caught:
            transport._finish_receipts(state.peers, state.attempt.record_control_boundary,
                                       state.attempt.fail_control_boundary)
    assert caught.value is state.original
    assert state.frozen[:3] == (p.HOLDERS[state.failing], "finish", "failed")
    assert state.probes == [False]
    assert state.cancel_snapshots and all(marker is state.frozen for marker in state.cancel_snapshots)
    assert state.parsed == []
    p.clear_failure_frames(caught.value)


@pytest.mark.parametrize("failing", [0, 1])
@pytest.mark.parametrize("late", ["cancel", "return", "error"])
def test_more_than_five_failed_requests_retire_original_buffers_roots_threads_and_fds(requests, failing, late):
    for index in range(8):
        with requests(failing, late, required_cancels=index + 3, observer_error=bool(index % 2)) as state:
            fail(state)
            roots = state.peers.io._fss_finish_pending
            assert 0 < len(roots) <= 5
            gc.collect()

            async def remaining():
                return (asyncio.all_tasks() - {asyncio.current_task()}) <= roots

            assert state.peers.io.run(remaining(), timeout=1)
            assert state.active == 1
        assert state.cancellations >= index + 3
        assert state.parsed == []  # Late valid bytes never became accepted receipts.


@pytest.mark.parametrize("failing", [0, 1])
def test_same_io_owner_rejects_eight_recollections_without_replacing_pending_roots(requests, failing):
    with requests(failing, "return", required_cancels=4) as state:
        fail(state)
        roots = state.peers.io._fss_finish_pending
        before = len(state.http)
        frozen = state.frozen
        state.attempt.close()
        fresh = state.owner.events({})
        try:
            for _ in range(8):
                with pytest.raises(ProtocolError, match="one-use"):
                    transport._finish_receipts(state.peers, fresh.record_control_boundary,
                                               fresh.fail_control_boundary)
                assert state.peers.io._fss_finish_pending is roots and len(roots) <= 5
                assert fresh._fss_control_boundary is None and not fresh._fss_control_failed
            assert len(state.http) == before and state.frozen is frozen
            state.peers.close()
            assert state.frozen is frozen and fresh._fss_control_boundary is None
            assert not roots and state.parsed == []
        finally:
            fresh.close()


@pytest.mark.parametrize("late", ["return", "error"])
def test_global_close_keeps_original_charge_until_resistant_callback_retires_and_rejects_new_io(requests, late):
    with requests(1, late, hold=True) as state:
        fail(state)
        owners = dict(resources._capacity._owners)
        frozen = state.frozen
        close_errors = []

        def close():
            try:
                state.peers.close()
            except BaseException as error:
                close_errors.append(error)

        thread = threading.Thread(target=close, name="fss-retirement-test")
        thread.start()
        try:
            assert state.global_cancel.wait(2)
            assert state.active == 1 and not state.view_retired.is_set()
            assert resources._capacity._owners == owners and not state.peers.io.closed
            with pytest.raises(BufferError):
                state.data.clear()

            async def late_dispatch():
                pytest.fail("retiring I/O owner accepted new work")

            coroutine = late_dispatch()
            with pytest.raises(ProtocolError, match="retired"):
                state.peers.io.run(coroutine, timeout=1)
            assert coroutine.cr_frame is None
        finally:
            state.release.set()
            thread.join(timeout=3)
            assert not thread.is_alive(), "controlled callback did not drain"
        assert not close_errors and state.active == 0 and state.view_retired.is_set()
        assert state.frozen is frozen and state.parsed == []


def test_done_callback_error_during_retirement_does_not_change_first_role_or_leave_roots(requests):
    with requests(0, "error", required_cancels=4) as state:
        fail(state)
        observed = []

        async def install():
            def callback_error(future):
                raise RuntimeError("modeled unrelated done callback")

            def report(loop, context):
                observed.append(type(context["exception"]))

            state.peers.io.loop.set_exception_handler(report)
            for task in tuple(state.peers.io._fss_finish_pending):
                task.add_done_callback(callback_error)

        state.peers.io.run(install(), timeout=1)
        frozen = state.frozen
        state.peers.close()
        assert observed and set(observed) == {RuntimeError}
        assert state.frozen is frozen and not state.peers.io._fss_finish_pending
        assert state.parsed == []


def test_caller_interrupt_during_global_close_waits_for_exact_view_retirement(requests, monkeypatch):
    with requests(1, "return", required_cancels=5, hold=True) as state:
        fail(state)
        interrupted = KeyboardInterrupt("caller interrupted cleanup")
        injected = []
        result = Future.result

        def interrupt_once(future, timeout=None):
            if (state.peers.io._closing and not injected
                    and threading.current_thread() is threading.main_thread()):
                injected.append(state.active)
                state.release.set()
                raise interrupted
            return result(future, timeout=timeout)

        monkeypatch.setattr(Future, "result", interrupt_once)
        with pytest.raises(KeyboardInterrupt) as caught:
            state.peers.close()
        state.close_failed = True
        assert caught.value is interrupted and injected == [1]
        assert state.view_retired.is_set() and not state.active
        assert state.peers.io.closed and not state.peers.io._fss_finish_pending
        assert state.parsed == []


def test_pool_cleanup_resisting_repeated_cancels_is_joined_before_timeout_error(requests, monkeypatch):
    with requests(0, "cancel", required_cancels=3) as state:
        fail(state)
        client = state.clients[0]
        actual_close = client.aclose
        cancelled = []

        async def delayed_pool():
            try:
                while len(cancelled) < 4:
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        cancelled.append(True)
            finally:
                await actual_close()

        monkeypatch.setattr(client, "aclose", delayed_pool)
        with pytest.raises(ProtocolError, match="pool retirement timed out"):
            state.peers.close()
        state.close_failed = True
        assert len(cancelled) == 4 and client.is_closed
        assert state.view_retired.is_set() and state.active == 0
        assert not state.peers.io.thread.is_alive() and not state.peers.io._fss_finish_pending
