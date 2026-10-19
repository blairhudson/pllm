"""Bounded peer streams: no model/checkpoint or external network required."""
import asyncio
from contextlib import AsyncExitStack
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from pllm.runtime import replicated_peer as peer
from pllm.runtime import replicated_worker as worker
from pllm.runtime.delegated_protocol import pack, unpack
from pllm.runtime.replicated_protocol import FRAME_LIMIT, ROLES
from pllm.protocols import PumaThreePartyDecoder


class Socket:
    """Minimal binary WebSocket connector over the actual worker ASGI route."""

    def __init__(self, app, path, headers):
        self.incoming, self.outgoing = asyncio.Queue(1), asyncio.Queue(1)
        self.accepted = asyncio.Event()
        self.headers = headers
        self.sent = []
        self.phase = "setup"
        self.counts = dict.fromkeys((f"{p}_{d}_bytes" for p in ("setup", "online")
                                    for d in ("upload", "download")), 0)
        scope = {"type": "websocket", "asgi": {"version": "3.0"}, "scheme": "ws",
                 "path": path, "raw_path": path.encode(), "query_string": b"",
                 "root_path": "", "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
                 "client": ("127.0.0.1", 1), "server": ("test", 80), "subprotocols": []}
        self.task = asyncio.create_task(app(scope, self.incoming.get, self.emit))

    async def emit(self, event):
        if event["type"] == "websocket.accept":
            self.accepted.set()
        else:
            await self.outgoing.put(event)

    async def start(self):
        await self.incoming.put({"type": "websocket.connect"})
        await self.accepted.wait()
        return self

    async def send(self, raw):
        self.sent.append(raw)
        action = unpack(raw).get("action", "public")
        self.phase = "setup" if action in {"public", "receive_seed"} else "online"
        self.counts[f"{self.phase}_upload_bytes"] += len(raw)
        await self.incoming.put({"type": "websocket.receive", "bytes": raw})

    async def recv(self):
        event = await self.outgoing.get()
        if event["type"] != "websocket.send":
            raise ValueError("socket closed")
        self.counts[f"{self.phase}_download_bytes"] += len(event["bytes"])
        return event["bytes"]

    async def close(self):
        if not self.task.done():
            await self.incoming.put({"type": "websocket.disconnect", "code": 1000})
            await self.task


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_sequence", [99, False, 0.0])
async def test_stream_serializes_slow_peer_and_burns_failed_sequence(invalid_sequence):
    entered, release = asyncio.Event(), asyncio.Event()
    sent = []

    class SlowSocket:
        async def send(self, raw):
            sent.append(unpack(raw))

        async def recv(self):
            if len(sent) == 1:
                return pack({"accepted": True})
            entered.set()
            await release.wait()
            return pack({"sequence": invalid_sequence, "result": {}})

        async def close(self):
            release.set()

    async def connect(*args, **kwargs):
        assert kwargs["max_queue"] == 1 and kwargs["max_size"] == FRAME_LIMIT
        assert kwargs["compression"] is None and kwargs["proxy"] is None
        return SlowSocket()

    stream = peer.PeerStream("http://test", "key", {"request": "a" * 64},
        time.monotonic() + 5, dict.fromkeys((f"{p}_{d}_bytes" for p in ("setup", "online")
                                          for d in ("upload", "download")), 0), connect)
    first = asyncio.create_task(stream.call("advance", b""))
    await entered.wait()
    second = asyncio.create_task(stream.call("advance", b""))
    await asyncio.sleep(0)
    assert len(sent) == 2  # hello plus exactly one request; no queued serialization
    release.set()
    with pytest.raises(ValueError, match="sequence"):
        await first
    with pytest.raises(ValueError, match="burned"):
        await second
    await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["success", "source", "configuration", "sender", "recipient",
                                      "sender_epoch", "recipient_epoch", "request", "sequence", "reconnect",
                                      "client_credential", "stale_route", "sdk", "sdk_failure",
                                      "simultaneous_peer_failures", "role_shutdown",
                                      "publication_transport", "publication_selection", "unbound_setup", "unsealed_finish",
                                      "sealed_peer_frame"])
async def test_three_worker_stream_waves_bindings_conservation_and_retirement(monkeypatch, scenario):
    natives, reservations, sockets = [], [], []
    source = SimpleNamespace(source_lock_digest="b" * 64, tokenizer_digest="c" * 64,
        config_bytes=b"{}", contract={"weights": "public", "public_weight_snapshot_bytes_per_worker": 0},
        runtime_contract=lambda *args: None, populate_public_worker=lambda *args: None)
    experiment = SimpleNamespace(configuration_digest=lambda: "d" * 64,
        pipeline=SimpleNamespace(digest=lambda: "e" * 64,
            components={"linear": PumaThreePartyDecoder(weights="public", peer_transport="streams")}),
        budget=SimpleNamespace(requests=8))

    class Reservation:
        def __init__(self, size):
            self.size, self.closed = size, False
            reservations.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            assert not self.closed
            self.closed = True

    class Native:
        def __init__(self, manifest, index):
            self.index, self.started, self.initial, self.closed = index, False, False, False
            self.calls, self.finished = [], False
            natives.append(self)

        def call(self, action, payload, sender):
            self.calls.append(action)
            if action == "offer":
                return False, [(self.index, b"public")]
            if action == "seed":
                return False, [((self.index + 2) % 3, b"seed")]
            if action in {"public", "receive_seed"}:
                return False, []
            if action == "advance":
                if scenario == "unsealed_finish":
                    return True, []
                if self.started:
                    return True, []
                self.started = True
                return False, [((self.index + 1) % 3, b"initial")]
            assert action == "peer"
            if payload == b"initial":
                if self.index == 2:
                    if scenario == "sdk_failure":
                        raise ValueError("injected incoming native failure")
                    time.sleep(0.02)  # fast peers generate replies while this prerequisite waits
                self.initial = True
                return False, [(sender, b"reply")]
            assert payload == b"reply" and all(native.initial for native in natives)
            return False, []

        def activate(self):
            pass

        def finish(self):
            self.finished = True
            return b"final"

        def close(self):
            assert not self.closed
            self.closed = True

    monkeypatch.setattr(worker, "admit_source", lambda _: source)
    monkeypatch.setattr(worker, "reserve_incremental_memory", Reservation)
    monkeypatch.setattr(worker._native, "ReplicatedWorker", Native, raising=False)
    apps, clients = {}, {}
    keys = {role: f"client-{role}-12345678" for role in ROLES}
    directed = {(a, b): f"{a}-to-{b}-12345678" for a in ROLES for b in ROLES if a != b}
    for role in ROLES:
        peers = {other: (f"http://127.0.0.1:{9000 + ROLES.index(other)}", directed[role, other], directed[other, role])
                 for other in ROLES if other != role}
        apps[role] = worker.create_replicated_worker_app(experiment, role, keys[role], peers)

    async def connect(url, **kwargs):
        from urllib.parse import urlsplit
        parsed = urlsplit(url)
        socket = Socket(apps[ROLES[parsed.port - 9000]], parsed.path, kwargs["additional_headers"])
        sockets.append(socket)
        return await socket.start()

    # Default connector is captured by the helper constructor; wrap only here.
    original = peer.PeerStream
    monkeypatch.setattr(worker, "PeerStream", lambda *args: original(*args, connector=connect))
    request_id = "a" * 64
    manifest = json.dumps({"session": list(bytes.fromhex(request_id)),
        "source": list(bytes.fromhex(source.source_lock_digest)),
        "composition": list(bytes.fromhex(experiment.pipeline.digest())),
        "config": list(source.config_bytes), "rows": 1, "outputs": 1, "weights": "public",
        "execution": worker.execution_binding(experiment)}).encode()

    async def post(role, action, content=b""):
        result = await clients[role].post(f"/v1/replicated/{request_id}/{action}", content=content)
        assert result.status_code == 200, (role, action, result.content)
        return unpack(result.content)

    async with AsyncExitStack() as stack:
        for role, app in apps.items():
            await stack.enter_async_context(app.router.lifespan_context(app))
            clients[role] = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test",
                headers={"authorization": f"Bearer {keys[role]}"}))
        if scenario in {"sdk", "sdk_failure"}:
            from pllm.runtime import replicated_transport as transport

            loop = asyncio.get_running_loop()
            replies = {}

            class Publisher:
                def next_chunk(self):
                    return None

                def finish(self, *frames):
                    assert frames == (b"final",) * 3
                    return (4,)

                def close(self):
                    pass

            class BridgePeers:
                def __init__(self, *args, **kwargs):
                    self.request_id = None
                    self.live = [False] * 3
                    self.counts = {}

                def post(self, index, action, raw=b"", **kwargs):
                    async def request():
                        client = clients[ROLES[index]]
                        if action == "manifest":
                            response = await client.get("/v1/replicated/manifest")
                        else:
                            path = "/v1/replicated/admit" if action == "admit" else f"/v1/replicated/{self.request_id}/{action}"
                            response = await client.post(path, content=raw)
                        if action == "cancel" and response.status_code == 409:
                            return pack({"retired": True})
                        if response.status_code != 200:
                            raise ValueError(f"mock peer request rejected: {action}")
                        if action == "finish":
                            replies[ROLES[index]] = unpack(response.content)
                        return response.content

                    return asyncio.run_coroutine_threadsafe(request(), loop).result(5)

                def audit(self):
                    return {}

                def close(self):
                    for index, live in enumerate(self.live):
                        if live:
                            self.post(index, "cancel")

            source.publisher = lambda *args: (Publisher(), manifest)
            monkeypatch.setattr(transport, "_Peers", BridgePeers)
            monkeypatch.setattr(transport, "reserve_incremental_memory", Reservation)
            if scenario == "sdk_failure":
                with pytest.raises(ValueError, match="mock peer request rejected"):
                    await asyncio.wait_for(asyncio.to_thread(transport.execute_request, experiment, source, (3,), 1,
                                                            {}, 10, SimpleNamespace()), 5)
                assert not replies
            else:
                result = await asyncio.to_thread(transport.execute_request, experiment, source, (3,), 1,
                                                 {}, 10, SimpleNamespace())
                assert result.selected == (4,)
                assert len(replies) == 3
            assert len(sockets) == 6
            assert all(native.closed for native in natives)
            assert all(item.closed for item in reservations)
            assert sorted(item.size for item in reservations) == [worker.REQUEST_MEMORY] + [worker.REQUEST_MEMORY + peer.STREAM_MEMORY] * 3
            assert all(socket.task.done() for socket in sockets)
            return
        epochs = {}
        for role in ROLES:
            advertised = unpack((await clients[role].get("/v1/replicated/manifest")).content)
            assert advertised["peer_transport"] == peer.TRANSPORT
            epochs[role] = advertised["epoch"]
            if scenario in {"publication_transport", "publication_selection"}:
                substituted = json.loads(manifest)
                substituted["execution"]["peer_transport" if scenario == "publication_transport" else "selection"] = (
                    "http" if scenario == "publication_transport" else "tiled")
                response = await clients[role].post("/v1/replicated/admit", content=pack(
                    {"request": request_id, "epoch": epochs[role], "manifest": json.dumps(substituted).encode()}))
                assert response.status_code == 400 and not natives
                assert reservations[-1].closed
                # Even a corrected publication cannot reuse this admitted ID.
                response = await clients[role].post("/v1/replicated/admit", content=pack(
                    {"request": request_id, "epoch": epochs[role], "manifest": manifest}))
                assert response.status_code == 400 and not natives
                assert all(item.closed for item in reservations)
                return
            response = await clients[role].post("/v1/replicated/admit", content=pack(
                {"request": request_id, "epoch": epochs[role], "manifest": manifest}))
            assert response.status_code == 200
        if scenario == "unbound_setup":
            response = await clients[ROLES[1]].post(f"/v1/replicated/{request_id}/setup_public")
            assert response.status_code == 400 and response.content == b""
            assert not natives[1].calls and natives[1].closed and reservations[1].closed
            return
        for role in ROLES:
            await post(role, "bind_peers", pack({"transport": peer.TRANSPORT, "epochs": epochs}))
        if scenario == "sealed_peer_frame":
            for action in ("setup_public", "setup_dispatch", "setup_seed", "setup_dispatch", "activate"):
                for role in ROLES:
                    await post(role, action)
            await post(ROLES[0], "execute")
            before = tuple(natives[1].calls)
            socket, = [item for item in sockets if item.headers["authorization"]
                       == f"Bearer {directed[ROLES[0], ROLES[1]]}"]
            # Use the next sequence: rejection must be terminal-state enforcement,
            # not the replay predicate or a second native call after completion.
            sequence = max(unpack(item)["sequence"] for item in socket.sent if "sequence" in unpack(item)) + 1
            await socket.send(pack({"sequence": sequence, "action": "receive", "payload": b"reply"}))
            await asyncio.wait_for(asyncio.gather(socket.task, return_exceptions=True), 2)
            response = await clients[ROLES[1]].post(f"/v1/replicated/{request_id}/cancel")
            assert response.status_code in {200, 409}
            assert tuple(natives[1].calls) == before and natives[1].closed and reservations[1].closed
            return
        if scenario == "unsealed_finish":
            for action in ("setup_public", "setup_dispatch", "setup_seed", "setup_dispatch", "activate"):
                for role in ROLES:
                    await post(role, action)
            # Reuse coordinator's directed setup socket to make B locally
            # complete without the global seal. Final-share release must fail.
            socket, = [item for item in sockets if item.headers["authorization"]
                       == f"Bearer {directed[ROLES[0], ROLES[1]]}"]
            # A sent only its public offer on this direction during setup.
            await socket.send(pack({"sequence": 1, "action": "advance", "payload": b""}))
            assert unpack(await socket.recv())["result"] == {"complete": True}
            response = await clients[ROLES[1]].post(f"/v1/replicated/{request_id}/finish")
            assert response.status_code == 400 and response.content == b""
            assert not natives[1].finished and natives[1].closed and reservations[1].closed
            return
        if scenario == "role_shutdown":
            socket = await Socket(apps[ROLES[1]], f"/v1/replicated/{request_id}/stream",
                {"authorization": f"Bearer {directed[ROLES[0], ROLES[1]]}"}).start()
            await socket.send(pack(peer.binding(request_id, source.source_lock_digest,
                experiment.configuration_digest(), ROLES[0], ROLES[1], epochs)))
            assert unpack(await socket.recv()) == {"accepted": True}
            await asyncio.wait_for(stack.aclose(), 2)
            assert socket.task.done()
            assert all(native.closed for native in natives)
            assert all(item.closed for item in reservations)
            return
        if scenario == "simultaneous_peer_failures":
            failed = []
            for sender in (ROLES[0], ROLES[2]):
                socket = await Socket(apps[ROLES[1]], f"/v1/replicated/{request_id}/stream",
                    {"authorization": f"Bearer {directed[sender, ROLES[1]]}"}).start()
                await socket.send(pack(peer.binding(request_id, source.source_lock_digest,
                    experiment.configuration_digest(), sender, ROLES[1], epochs)))
                assert unpack(await socket.recv()) == {"accepted": True}
                failed.append(socket)
            await asyncio.gather(*(socket.send(pack({"sequence": 5, "action": "advance", "payload": b""}))
                                   for socket in failed))
            await asyncio.wait_for(asyncio.gather(*(socket.task for socket in failed), return_exceptions=True), 2)
            cancelled = await clients[ROLES[1]].post(f"/v1/replicated/{request_id}/cancel")
            assert cancelled.status_code in {200, 409}
            assert natives[1].closed and reservations[1].closed
            assert all(socket.task.done() for socket in failed)
            return
        if scenario in {"client_credential", "stale_route"}:
            path = f"/v1/replicated/{request_id if scenario == 'client_credential' else 'f' * 64}/stream"
            credential = keys[ROLES[1]] if scenario == "client_credential" else directed[ROLES[0], ROLES[1]]
            rejected = Socket(apps[ROLES[1]], path, {"authorization": f"Bearer {credential}"})
            await rejected.incoming.put({"type": "websocket.connect"})
            await asyncio.wait_for(rejected.task, 2)
            assert not rejected.accepted.is_set()
            assert not natives[1].closed and not reservations[1].closed
            return
        if scenario != "success":
            # Authenticated cross-session/source/epoch substitution, replay, and
            # reconnect all burn the native owner; no retry can reuse its tape.
            path = f"/v1/replicated/{request_id}/stream"
            headers = {"authorization": f"Bearer {directed[ROLES[0], ROLES[1]]}"}
            socket = await Socket(apps[ROLES[1]], path, headers).start()
            hello = peer.binding(request_id, source.source_lock_digest,
                                 experiment.configuration_digest(), ROLES[0], ROLES[1], epochs)
            if scenario not in {"sequence", "reconnect"}:
                hello[scenario] = "f" * 64
            await socket.send(pack(hello))
            if scenario in {"sequence", "reconnect"}:
                assert unpack(await socket.recv()) == {"accepted": True}
                if scenario == "sequence":
                    await socket.send(pack({"sequence": 1, "action": "advance", "payload": b""}))
                else:
                    duplicate = Socket(apps[ROLES[1]], path, headers)
                    await duplicate.incoming.put({"type": "websocket.connect"})
                    await asyncio.wait_for(duplicate.task, 2)
                    assert not duplicate.accepted.is_set()
            await asyncio.wait_for(asyncio.gather(socket.task, return_exceptions=True), 2)
            # Socket failure schedules tracked retirement; an explicit cancel
            # joins the same owner rather than starting competing cleanup.
            cancelled = await clients[ROLES[1]].post(f"/v1/replicated/{request_id}/cancel")
            assert cancelled.status_code in {200, 409}
            assert natives[1].closed and reservations[1].closed
            assert not natives[0].closed and not natives[2].closed
            response = await clients[ROLES[1]].post("/v1/replicated/admit", content=pack(
                {"request": request_id, "epoch": epochs[ROLES[1]], "manifest": manifest}))
            assert response.status_code == 400  # failed session stays burned
            return
        for action in ("setup_public", "setup_dispatch", "setup_seed", "setup_dispatch", "activate"):
            for role in ROLES:
                await post(role, action)
        assert await post(ROLES[0], "execute") == {"complete": True}
        assert len(sockets) == 6  # reused for every setup, frame, and control exchange
        results = {}
        # Receiver finish closes streams; all work must already be quiescent.
        for role in ROLES:
            results[role] = await post(role, "finish")
        assert all(native.closed for native in natives)
        assert all(item.closed and item.size == worker.REQUEST_MEMORY + peer.STREAM_MEMORY
                   for item in reservations)
        for a in ROLES:
            assert set(results[a]) == {"frame", "peer_bytes"}
            for b in ROLES:
                if a != b:
                    socket, = [item for item in sockets if item.headers["authorization"]
                               == f"Bearer {directed[a, b]}"]
                    assert results[a]["peer_bytes"][b] == socket.counts
                    assert results[a]["peer_bytes"][b]["setup_upload_bytes"] > 0
                    # Initiators alone report request + ACK bytes. Receiver must
                    # not duplicate those bytes in its reverse-direction totals.
                    assert results[a]["peer_bytes"][b]["setup_download_bytes"] > 0
            metrics = (await clients[a].get("/v1/replicated/metrics")).json()
            assert metrics["active_sessions"] == 0
            assert metrics["peer_counts"]["frames"] > 0
        assert all(socket.task.done() for socket in sockets)


@pytest.mark.parametrize("raw", [b"", b"x" * (FRAME_LIMIT + 1), "text"])
def test_stream_decode_rejects_unbounded_or_text_messages(raw):
    with pytest.raises(ValueError):
        peer.decode(raw)


@pytest.mark.asyncio
async def test_expired_stream_never_connects_or_retries():
    async def forbidden(*args, **kwargs):
        pytest.fail("expired session connected")

    stream = peer.PeerStream("http://test", "key", {"request": "a" * 64},
                             time.monotonic() - 1, {}, forbidden)
    with pytest.raises(TimeoutError):
        await stream.call("advance", b"")
    with pytest.raises(ValueError, match="burned"):
        await stream.call("advance", b"")
    await stream.aclose()


@pytest.mark.asyncio
async def test_attempt_refund_waits_for_peer_send_retirement():
    entered, release = asyncio.Event(), asyncio.Event()
    events = []

    class Stream:
        async def aclose(self):
            entered.set()
            await release.wait()
            events.append("send_drained")

    class Owner:
        def close(self):
            events.append("native_closed")

    class Reservation:
        def close(self):
            events.append("refunded")

    attempt = worker._Attempt(Owner(), Reservation(), "a" * 64)
    attempt.enqueue([(1, b"pending")])
    attempt.outgoing = ((2, b"outgoing"),)
    attempt.streams["worker_b"] = Stream()
    task = asyncio.create_task(attempt.close())
    await entered.wait()
    assert not task.done() and not events
    assert list(attempt.pending) == [(1, b"pending")] and attempt.outgoing == ((2, b"outgoing"),)
    release.set()
    await task
    assert events == ["send_drained", "native_closed", "refunded"]
    assert not attempt.pending and not attempt.outgoing


@pytest.mark.asyncio
async def test_attempt_cancellation_drains_incoming_native_before_refund():
    import threading

    entered, release = threading.Event(), threading.Event()
    events = []

    def operation():
        entered.set()
        assert release.wait(5)
        events.append("native_drained")

    class Owner:
        def close(self):
            events.append("native_closed")

    class Reservation:
        def close(self):
            events.append("refunded")

    attempt = worker._Attempt(Owner(), Reservation(), "a" * 64)

    async def receive():
        async with attempt.lock:
            await worker.native_call(operation)

    receiving = asyncio.create_task(receive())
    attempt.stream_tasks.add(receiving)
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        closing = asyncio.create_task(attempt.close())
        await asyncio.sleep(0)
        receiving.cancel()  # repeated cancellation cannot abandon the native call
        await asyncio.sleep(0)
        assert not closing.done() and not events
        release.set()
        await closing
        assert events == ["native_drained", "native_closed", "refunded"]
    finally:
        release.set()
        await asyncio.gather(receiving, *([closing] if closing else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_stream_close_cancels_connect_and_bounds_payload_waiters():
    entered = asyncio.Event()
    events = []

    async def connect(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            events.append("connect_drained")

    stream = peer.PeerStream("http://test", "key", {"request": "a" * 64},
                             time.monotonic() + 60, {}, connect)
    first = asyncio.create_task(stream.call("dispatch", b""))
    await entered.wait()
    second = asyncio.create_task(stream.call("receive", b"payload"))
    await asyncio.sleep(0)
    assert stream.users == 2 and stream.socket is None
    with pytest.raises(ValueError, match="queue exceeds admission"):
        await stream.call("receive", b"excess payload")
    await asyncio.wait_for(asyncio.gather(stream.aclose(), stream.aclose()), 1)
    outcomes = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(outcomes[0], asyncio.CancelledError)
    assert isinstance(outcomes[1], ValueError)
    assert stream.users == 0 and stream.socket is None and stream.closed
    assert events == ["connect_drained"] and stream.close_task.done()


@pytest.mark.asyncio
async def test_simultaneous_retirements_drain_native_and_connect_once_under_repeated_cancellation():
    import threading

    entered, release = threading.Event(), threading.Event()
    connecting = asyncio.Event()
    events = []

    def native_operation():
        entered.set()
        assert release.wait(5)
        events.append("native_drained")

    class Owner:
        def close(self):
            events.append("native_closed")

    class Manager:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append("refunded")

    async def connect(*args, **kwargs):
        connecting.set()
        try:
            await asyncio.Event().wait()
        finally:
            events.append("connect_drained")

    reservation = worker._Reservation(Manager())
    attempt = worker._Attempt(Owner(), reservation, "a" * 64)
    stream = peer.PeerStream("http://test", "key", {"request": attempt.request_id},
                             attempt.expires, {}, connect)
    attempt.streams["worker_b"] = stream

    async def receive():
        lease = reservation.lease()
        try:
            async with attempt.lock:
                await worker.native_call(native_operation)
        finally:
            lease.close()

    receiving = asyncio.create_task(receive())
    attempt.stream_tasks.add(receiving)
    attempt.task = sending = asyncio.create_task(stream.call("dispatch", b""))
    closing = []
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await connecting.wait()
        closing = [asyncio.create_task(attempt.close()), asyncio.create_task(attempt.close())]
        await asyncio.sleep(0)
        closing[0].cancel()
        await asyncio.sleep(0)
        closing[0].cancel()
        assert "refunded" not in events and "native_closed" not in events
        assert not any(task.done() for task in closing)
        release.set()
        outcomes = await asyncio.wait_for(asyncio.gather(*closing, return_exceptions=True), 2)
        assert isinstance(outcomes[0], asyncio.CancelledError) and outcomes[1] is None
        assert events.count("native_closed") == events.count("refunded") == 1
        assert events.index("native_drained") < events.index("native_closed") < events.index("refunded")
        assert events.index("connect_drained") < events.index("refunded")
        assert receiving.done() and sending.done() and attempt.close_task.done()
        assert stream.close_task.done() and not attempt.stream_tasks and not attempt.streams
    finally:
        release.set()
        await asyncio.gather(receiving, sending, *closing, return_exceptions=True)


def test_pending_and_frozen_wave_bounds_do_not_mutate_on_rejection():
    attempt = worker._Attempt(None, None, "a" * 64)
    wave = [(1, b"x")] * 16
    attempt.enqueue(wave)
    attempt.outgoing = tuple(attempt.pending)
    attempt.pending.clear()
    attempt.enqueue(wave)
    with pytest.raises(ValueError, match="queue exceeds admission"):
        attempt.enqueue([(2, b"overflow")])
    assert len(attempt.pending) == len(attempt.outgoing) == 16
    attempt.pending.clear()
    with pytest.raises(ValueError, match="queue exceeds admission"):
        attempt.enqueue([(1, b"x" * (FRAME_LIMIT + 1))])
    assert not attempt.pending and len(attempt.outgoing) == 16


@pytest.mark.asyncio
async def test_connect_completion_race_cannot_send_after_retirement():
    entered = asyncio.Event()
    events = []

    class LateSocket:
        async def send(self, raw):
            pytest.fail("retired stream sent a handshake")

        async def close(self):
            events.append("closed")

    async def connect(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return LateSocket()

    stream = peer.PeerStream("http://test", "key", {"request": "a" * 64},
                             time.monotonic() + 60, {}, connect)
    sending = asyncio.create_task(stream.call("advance", b""))
    await entered.wait()
    await asyncio.wait_for(stream.aclose(), 1)
    with pytest.raises(ValueError, match="burned"):
        await sending
    assert stream.socket is None and events == ["closed"] and stream.users == 0


@pytest.mark.asyncio
async def test_real_websocket_reuses_connection_accounts_envelopes_and_drains_close():
    from websockets.asyncio.server import serve

    retired = asyncio.Event()
    received, replied, connections = [], [], []
    hello = peer.binding("a" * 64, "b" * 64, "c" * 64, ROLES[0], ROLES[1],
                         {role: str(index + 1) * 64 for index, role in enumerate(ROLES)})

    async def handler(socket):
        connections.append(socket)
        try:
            assert socket.request.headers["authorization"] == "Bearer directed-key-12345678"
            raw = await socket.recv()
            assert unpack(raw) == hello
            received.append(("setup", len(raw)))
            raw = pack({"accepted": True})
            replied.append(("setup", len(raw)))
            await socket.send(raw)
            for sequence, phase in enumerate(("setup", "online")):
                raw = await socket.recv()
                value = unpack(raw)
                assert value["sequence"] == sequence
                received.append((phase, len(raw)))
                raw = pack({"sequence": sequence, "result": {"accepted": True}})
                replied.append((phase, len(raw)))
                await socket.send(raw)
            await socket.wait_closed()
        finally:
            retired.set()

    counters = dict.fromkeys((f"{p}_{d}_bytes" for p in ("setup", "online")
                              for d in ("upload", "download")), 0)
    async with serve(handler, "127.0.0.1", 0, compression=None, max_size=FRAME_LIMIT,
                     max_queue=1, write_limit=FRAME_LIMIT, ping_interval=None) as server:
        port = server.sockets[0].getsockname()[1]
        stream = peer.PeerStream(f"http://127.0.0.1:{port}", "directed-key-12345678", hello,
                                 time.monotonic() + 10, counters)
        try:
            assert await stream.call("public", b"setup-frame") == {"accepted": True}
            assert await stream.call("receive", b"online-frame") == {"accepted": True}
        finally:
            await stream.aclose()
        await asyncio.wait_for(retired.wait(), 1)
        assert len(connections) == 1 and stream.socket is None and stream.users == 0
    for phase in ("setup", "online"):
        assert counters[f"{phase}_upload_bytes"] == sum(size for p, size in received if p == phase)
        assert counters[f"{phase}_download_bytes"] == sum(size for p, size in replied if p == phase)


@pytest.mark.asyncio
async def test_socket_close_failure_waits_for_transport_abort_before_refund():
    aborted, lost = asyncio.Event(), asyncio.Event()
    events = []

    class Transport:
        def abort(self):
            events.append("aborted")
            aborted.set()

    class SocketFailure:
        transport = Transport()

        async def close(self):
            raise ValueError("injected close failure")

        async def wait_closed(self):
            await lost.wait()
            events.append("transport_drained")

    class Owner:
        def close(self):
            events.append("native_closed")

    class Reservation:
        def close(self):
            events.append("refunded")

    attempt = worker._Attempt(Owner(), Reservation(), "a" * 64)
    stream = peer.PeerStream("http://test", "key", {"request": attempt.request_id}, attempt.expires, {})
    stream.socket = SocketFailure()
    attempt.streams["worker_b"] = stream
    closing = asyncio.create_task(attempt.close())
    try:
        await asyncio.wait_for(aborted.wait(), 1)
        assert events == ["aborted"] and not closing.done()
        lost.set()
        await asyncio.wait_for(closing, 1)
        assert events == ["aborted", "transport_drained", "native_closed", "refunded"]
    finally:
        lost.set()
        await asyncio.gather(closing, return_exceptions=True)
