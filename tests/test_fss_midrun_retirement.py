"""Midrun retirement through authenticated ASGI and real client I/O owners.

Native arithmetic/source/jobs are existing bounded doubles. Socket replacement
uses independent server Tasks; an outgoing callback never owns its peer handler.
No trained-timeout cause, cryptographic or physical-memory claim is made.
"""
import asyncio
import threading
from types import SimpleNamespace as NS

import httpx
import pytest
import pytest_asyncio

from pllm.runtime import fss_protocol as p, fss_worker as worker
from pllm.runtime.client import ProtocolError
import test_fss_finish_endpoint_integration as integration
import test_fss_worker as topology
from test_fss_finish_endpoint_integration import endpoint as endpoint
from test_fss_receive_contracts import receive_contract as receive_contract
from test_fss_worker import cluster as cluster, double as double


@pytest.fixture
def midrun_setup(receive_contract, monkeypatch, request):
    state = NS(mode=request.param, fail=threading.Event(), cleanup_release=asyncio.Event(),
               io_entered=asyncio.Event(), io_cancelled=asyncio.Event(), io_drained=asyncio.Event(),
               space_entered=asyncio.Event(), space_drained=asyncio.Event(),
               both_finish=asyncio.Event(), peers_idle=asyncio.Event(), peer_tasks=set(),
               rounds={0: 0, 1: 0}, windows={0: [], 1: []}, finish_roles=set(), finish_exited=set(),
               callback_threads=[], callback_errors=[], native_errors=[], post_errors=[],
               refunded=[], roots={}, native_error=ValueError("synthetic midrun B failure"))
    state.peers_idle.set()
    # Existing real-window fixture supplies 17 original records, one kernel.
    # Four peer frames allow two good rounds, one overlap, then space pressure.
    monkeypatch.setattr(integration, "configuration", topology.configuration)

    async def cleanup_gate():
        state.io_cancelled.set()
        while not state.cleanup_release.is_set():
            try:
                await state.cleanup_release.wait()
            except asyncio.CancelledError:
                continue  # Bounded by test's finally; retain views/lease until drain.
        state.io_drained.set()

    async def blocked_io():
        state.io_entered.set()
        try:
            await asyncio.Future()
        finally:
            await cleanup_gate()

    state.cleanup_gate = cleanup_gate
    original_routing = topology.Routing.handle_async_request
    http_client = integration.HTTPClient

    class EndpointDisconnect(httpx.AsyncBaseTransport):
        def __init__(self, transport):
            self.transport = transport

        async def handle_async_request(self, request):
            try:
                return await self.transport.handle_async_request(request)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                # A remotely cancelled handler closes its socket. It cannot
                # cancel the independent client Task through an ASGI Future.
                raise httpx.ReadError("ASGI peer handler retired", request=request) from None

        async def aclose(self):
            await self.transport.aclose()

    def endpoint_client(*args, transport, **kwargs):
        return http_client(*args, transport=EndpointDisconnect(transport), **kwargs)

    monkeypatch.setattr(integration, "HTTPClient", endpoint_client)

    async def route(router, request):
        if not request.url.path.endswith("/peer"):
            return await original_routing(router, request)
        role = next(r for r in p.ROLES if request.url.port == 7300 + p.ROLES.index(r))
        value = p.unpack(request.content)

        async def server():
            app = router.apps[role]
            if state.mode == "inbound-body" and role == "worker_b" and value["sequence"] == 2:
                async def gated(scope, receive, send, *, target=app):
                    async def body():
                        await blocked_io()
                        return await receive()
                    return await target(scope, body, send)
                app = gated
            return await httpx.ASGITransport(app=app).handle_async_request(request)

        task = asyncio.create_task(server())
        state.peer_tasks.add(task)
        state.peers_idle.clear()

        def done(task):
            state.peer_tasks.discard(task)
            if not task.cancelled():
                error = task.exception()
                if error is not None:
                    p.clear_failure_frames(error)
            if not state.peer_tasks:
                state.peers_idle.set()

        task.add_done_callback(done)
        try:
            # Disconnect cancels server work but does not synchronously join it:
            # real socket client cancellation never joins remote retirement.
            response = await asyncio.shield(task)
            sender = next(r for r in p.ROLES
                          if request.headers["authorization"] == f"Bearer {topology.key(r, role)}")
            router.double.rpc_log.append((sender, role, "online", len(request.content),
                                          int(response.headers["content-length"])))
            if state.mode == "space-wait" and role == "worker_a" and value["sequence"] == 2:
                # Actual frame already delivered/ACKed. Hold B's ACK locally;
                # A can now send frame 3 while frame 2 fills B's mailbox.
                await blocked_io()
            return response
        finally:
            if not task.done():
                task.cancel()

    monkeypatch.setattr(topology.Routing, "handle_async_request", route)
    yield state
    state.fail.set()
    state.cleanup_release.set()


@pytest_asyncio.fixture
async def midrun(midrun_setup, endpoint, cluster, monkeypatch):
    s, c = midrun_setup, cluster
    endpoint.mode = "midrun"
    c.double.admission_changes["frames_per_party"] = 4
    native_type = worker._native.FssRoleRequest
    dealer_run = native_type.run
    loop = asyncio.get_running_loop()

    def meta(native, step):
        value = p.metadata(topology.callback_metadata(native.request_id, "peer", native.role))
        value["frame"] = step
        return p.canonical(value)

    def exchange(native, callbacks, step):
        assert callbacks.exchange(meta(native, step), f"peer{native.role}-{step}".encode()) == (
            f"peer{1-native.role}-{step}".encode())

    def native_run(native, frame, callbacks):
        if native.role == 2:
            return dealer_run(native, frame, callbacks)
        native.callbacks, native.running = callbacks, True
        c.double.native_threads.append(threading.get_ident())
        pump = None
        try:
            assert frame == f"input-{native.role}".encode()
            rows = tuple((p.canonical(p.metadata(topology.callback_metadata(
                native.request_id, "material", native.role)) | {"pc": pc, "packet_bytes_bound": bound}),
                f"packet-{native.role}-{pc}".encode()) for pc, bound in
                tuple((100 + (i * 7) % 17, (32, 48, 64)[i % 3]) for i in range(17)))
            for step, offset in enumerate((0, 8, 16)):
                batch = rows[offset:offset + 8]
                packets = callbacks.material_window(tuple(raw for raw, _ in batch))
                assert packets == tuple(packet for _, packet in batch)
                s.windows[native.role].append(packets)
                if step < 2:
                    exchange(native, callbacks, step)
                    s.rounds[native.role] += 1
            if native.role == 0:
                exchange(native, callbacks, 2)
                exchange(native, callbacks, 3)  # Only space-wait can reach here.
                raise AssertionError("failed B must not provide a final round/output")
            if s.mode == "space-wait":
                def callback_pump():
                    try:
                        exchange(native, callbacks, 2)
                    except BaseException as error:
                        s.callback_errors.append(error)
                pump = threading.Thread(target=callback_pump, name="bounded-midrun-native-double")
                s.callback_threads.append(pump)
                pump.start()
            assert s.fail.wait(4), "test did not reach in-flight midrun gate"
            # Model trusted native error propagation: signal BEFORE joining an
            # outstanding callback. Never hold a native mutex across this join.
            callbacks.cancel()
            if pump is not None:
                pump.join(2)
                assert not pump.is_alive()
            s.native_errors.append(s.native_error)
            raise s.native_error
        finally:
            native.running = False
            endpoint.exited.add(native.role)

    monkeypatch.setattr(native_type, "run", native_run)
    original_attempt = worker._Attempt.__init__

    def attempt_init(attempt, *args, **kwargs):
        original_attempt(attempt, *args, **kwargs)
        s.roots[attempt.reservation] = attempt
        available, empty = attempt.mail_available.wait, attempt.mail_empty.wait

        async def mail_wait():
            native = attempt.native
            if s.mode == "mailbox" and native is not None and native.role == 0 and attempt.peer_sent == 3:
                s.io_entered.set()
                try:
                    return await available()
                finally:
                    await s.cleanup_gate()
            return await available()

        async def space_wait():
            if (s.mode == "space-wait" and attempt.native is not None
                    and attempt.native.role == 1 and attempt.peer_received == 3
                    and attempt.mailbox is not None):
                s.space_entered.set()
                try:
                    return await empty()
                finally:
                    s.space_drained.set()
            return await empty()

        attempt.mail_available.wait, attempt.mail_empty.wait = mail_wait, space_wait

    monkeypatch.setattr(worker._Attempt, "__init__", attempt_init)
    release = worker._Reservation._release

    def refund(reservation):
        attempt = s.roots.get(reservation)
        if attempt is not None and reservation._references == 1:
            assert attempt.phase == "closed" and not attempt.callback_tasks and not attempt.handlers
            assert not attempt.callback_futures and not attempt.callback_pending
            assert attempt.native is attempt.loaded is attempt.tape is None
            assert all(not thread.is_alive() for thread in s.callback_threads)
            s.refunded.append(attempt.request_id)
        return release(reservation)

    monkeypatch.setattr(worker._Reservation, "_release", refund)
    post = integration.delegated_transport._Peers._post

    async def observe_post(peers, index, action, *args, **kwargs):
        try:
            return await post(peers, index, action, *args, **kwargs)
        except (ProtocolError, httpx.ReadError) as error:
            if action == "finish":
                s.post_errors.append((p.ROLES[index], error))
            raise
        finally:
            if action == "finish" and index < 2:
                s.finish_exited.add(p.ROLES[index])

    monkeypatch.setattr(integration.delegated_transport._Peers, "_post", observe_post)
    for role in p.HOLDERS:
        app = c.apps[role]

        async def finish_gate(scope, receive, send, *, app=app, role=role):
            if scope["path"].endswith("/finish"):
                s.finish_roles.add(role)
                if s.finish_roles == set(p.HOLDERS):
                    s.both_finish.set()
            return await app(scope, receive, send)

        c.apps[role] = finish_gate
    s.endpoint, s.cluster, s.loop = endpoint, c, loop
    try:
        yield s
    finally:
        s.fail.set()
        s.cleanup_release.set()
        endpoint.attempt.cancel()
        for thread in s.callback_threads:
            thread.join(2)
            assert not thread.is_alive()
        await asyncio.wait_for(s.peers_idle.wait(), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("midrun_setup", ["mailbox", "inbound-body", "space-wait"], indirect=True)
async def test_b_midrun_failure_drains_pending_peer_owners_before_refund(midrun):
    s, e = midrun, midrun.endpoint
    request = asyncio.create_task(e.run())
    try:
        async with asyncio.timeout(3):
            await s.both_finish.wait()
            await s.io_entered.wait()
            if s.mode == "space-wait":
                await s.space_entered.wait()
        assert s.rounds == {0: 2, 1: 2}
        assert [len(row) for row in s.windows[0]] == [8, 8, 1]
        assert [len(row) for row in s.windows[1]] == [8, 8, 1]
        holders = [a for a in e.sessions if a.native is not None and a.native.role in (0, 1)]
        assert all(a.control_busy and a.reservation._manager is not None for a in holders)
        s.fail.set()
        await asyncio.wait_for(s.io_cancelled.wait(), 2)
        blocked = next(a for a in holders if a.admission["role"] == (0 if s.mode == "mailbox" else 1))
        assert blocked.reservation is not None and blocked.reservation._manager is not None
        assert not request.done() and e.reconstructed == 0
        s.cleanup_release.set()
        with pytest.raises((ProtocolError, httpx.ReadError)) as caught:
            await asyncio.wait_for(request, 3)
        if isinstance(caught.value, ProtocolError):
            assert caught.value.status_code == 400 and caught.value.body is None
        assert s.native_errors == [s.native_error]
        assert s.post_errors and caught.value is s.post_errors[0][1]
        assert e.frozen[:3] == (s.post_errors[0][0], "finish", "failed")
        assert e.native_close_marker is e.frozen and e.reconstructed == 0
        assert s.io_drained.is_set()
        if s.mode == "space-wait":
            assert s.space_drained.is_set() and s.callback_errors
        await asyncio.wait_for(s.peers_idle.wait(), 2)
        await e.assert_retired()
        assert s.finish_exited == set(p.HOLDERS)
        assert len(s.refunded) == 3
    finally:
        s.fail.set()
        s.cleanup_release.set()
        e.attempt.cancel()
        await asyncio.wait_for(asyncio.gather(request, return_exceptions=True), 8)
