"""Real ASGI HTTP handlers + client orchestration, bounded native/source doubles.

No socket/TLS, rebuilt native, checkpoint, numeric or security qualification.
Reuse worker-test fixtures and real MaterialTape rather than duplicate tape tests.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from pllm.runtime import fss_protocol as p, fss_transport as client, fss_worker as worker
from pllm.runtime.delegated_transport import _IOOwner
from pllm.runtime.client import ProtocolError
from test_fss_worker import (
    Frozen, ORIGINS, Routing, callback_metadata, client_key, cluster, configuration,
    connections, contract, credentials, double, key,
)


async def complete(cluster):
    await cluster.ready()
    await cluster.prepare()
    dealer = await cluster.post("dealer", "finish")
    assert dealer.status_code == 200
    await cluster.inputs()
    for role in p.HOLDERS:
        assert (await cluster.post(role, "start")).status_code == 200
    holders = await asyncio.wait_for(asyncio.gather(*(cluster.post(r, "finish") for r in p.HOLDERS)), 5)
    assert all(r.status_code == 200 for r in holders)
    return dealer, holders


async def retired(cluster, roles=p.ROLES):
    for _ in range(100):
        if all([(await cluster.metrics(role))["active_sessions"] == 0 for role in roles]):
            return
        await asyncio.sleep(.005)
    raise AssertionError("bounded retirement did not complete")


@pytest.mark.asyncio
async def test_actual_holder_reply_preserves_tape_receipt_and_directed_conservation(cluster):
    dealer, holders = await complete(cluster)
    _, dealer_counts, storage = client._receipt(dealer.content, "dealer")
    assert storage == {}
    counts = dealer_counts.copy()
    for role, response in zip(p.HOLDERS, holders):
        value = p.unpack(response.content, limit=p.CONTROL_LIMIT)
        resources = worker.MaterialTape.price(records=2, packet_bytes=64, total_packet_bytes=128)
        receipt = value["storage"]
        assert receipt["state"] == "finished"
        assert receipt["records_written"] == receipt["records_read"] == resources.records == 2
        assert receipt["packet_bytes_written"] == receipt["packet_bytes_read"] <= resources.total_packet_bytes
        assert receipt["encrypted_file_bytes_written"] == receipt["encrypted_file_bytes_read"] == receipt["sealed_file_bytes"]
        assert receipt["sealed_file_bytes"] <= resources.max_file_bytes
        assert len(receipt["sealed_digest"]) == 64
        assert "material-" not in repr(receipt)
        assert len(response.content) <= p.CONTROL_LIMIT
        for name, number in p.collect_peer_bytes(value["peer_bytes"], role).items():
            counts[name] += number
    for sender in p.ROLES:
        for recipient in p.ROLES:
            if sender == recipient:
                continue
            for phase in ("setup", "online"):
                expected = sum(upload for a, b, q, upload, _ in cluster.double.rpc_log
                               if (a, b, q) == (sender, recipient, phase))
                expected += sum(download for a, b, q, _, download in cluster.double.rpc_log
                                if (a, b, q) == (recipient, sender, phase))
                assert counts[f"peer_link.{sender}.{recipient}.{phase}_bytes"] == expected
    assert all("dealer" not in (a, b) for a, b, phase, _, _ in cluster.double.rpc_log if phase == "online")


@pytest.mark.asyncio
async def test_actual_holder_finish_is_accepted_by_ordinary_client_parser(cluster):
    _, holders = await complete(cluster)
    for role, response in zip(p.HOLDERS, holders):
        client._receipt(response.content, role)


@pytest_asyncio.fixture
async def client_route(cluster, monkeypatch):
    """Bounded synchronous client dispatcher to actual app handlers on their loop.

    Does not replace worker peer HTTP handlers, tape, callbacks or client ordering.
    Does replace socket-owning _Peers, source import and native final reconstruction.
    """
    loop = asyncio.get_running_loop()
    events = []
    owners = []
    clients = []
    reservations = []
    transforms = []
    requests = []
    class NativeClient:
        def __init__(self, raw_contract, session, manifests, ids):
            assert raw_contract == contract(capacity=cluster.double.capacity)
            assert len(manifests) == 3 and ids == [1, 2, 3]
            assert all(p.metadata(m)["program"] == "e" * 64 for m in manifests)
            self.session, self.closed = session.hex(), False
            self.inputs = set()
            clients.append(self)
            events.append("native-client-created")
        def request_bytes(self): return b"request:" + self.session.encode()
        def take_input(self, role):
            assert role not in self.inputs
            assert all(n.closed for n in cluster.double.natives if n.role == 2)
            assert all(n.closed for n in cluster.double.loaded if n.role == "dealer")
            assert cluster.double.sealed_roles == set(p.HOLDERS)
            self.inputs.add(role)
            events.append(f"input-{role}-after-dealer-retirement")
            return f"input-{role}".encode()
        def finish(self, a, b):
            assert (a, b) == (b"final-0", b"final-1")
            events.append("native-final-reconstruction")
            return [5, 6, 7]
        def close(self): self.closed = True

    class Peers:
        def start(self, connections_map, timeout, cancelled, **kwargs):
            assert kwargs == {"roles": p.ROLES, "namespace": "fss"}
            assert set(connections_map) == set(p.ROLES)
            self.request_id, self.live, self.counts = None, [False] * 3, {}
            self.deadline, self.cancelled = client.time.monotonic() + timeout, cancelled
            self.closed = False
            # Borrow the fixture's ASGI loop, retaining the genuine synchronous
            # dispatch/cancellation contract. The fixture, not this mock, owns
            # loop retirement; never stop it or create a second I/O thread.
            class IO:
                closed = False
                _state = "running"
                run = _IOOwner.run
            self.io = IO()
            self.io.loop = loop
            owners.append(self)
        def check(self):
            if self.cancelled() or client.time.monotonic() >= self.deadline:
                raise ProtocolError("bounded review client cancelled", 499)
        def post(self, index, action, payload=b"", *, phase, limit):
            self.check()
            return self.io.run(self._post(index, action, payload, phase=phase, limit=limit,
                                          timeout=max(0.0, self.deadline - client.time.monotonic())),
                               timeout=max(0.0, self.deadline - client.time.monotonic()))

        async def _post(self, index, action, payload, *, phase, limit, timeout):
            self.check()
            role = p.ROLES[index]
            events.append((role, action, phase))
            requests.append((role, action, payload))
            async with asyncio.timeout(min(5.0, timeout)):
                if action == "manifest":
                    response = await cluster.clients[role].get("/v1/fss/manifest")
                else:
                    path = "/v1/fss/admit" if action == "admit" else f"/v1/fss/{self.request_id}/{action}"
                    response = await cluster.clients[role].post(path, content=payload)
            content = response.content
            for transform in transforms:
                content = transform(role, action, content)
            if len(content) > limit or response.status_code != 200:
                raise ProtocolError("bounded review HTTP reply rejected", response.status_code)
            return content
        def close(self):
            for index, live in enumerate(self.live):
                if live:
                    role = p.ROLES[index]
                    future = asyncio.run_coroutine_threadsafe(cluster.clients[role].post(
                        f"/v1/fss/{self.request_id}/cancel", content=b""), loop)
                    try: future.result(timeout=5)
                    finally: future.cancel()
            self.closed = True
            self.io.closed = True
        def audit(self): return dict(self.counts)

    @contextmanager
    def reserve(size):
        assert size == client.client_memory_bytes(cluster.double.capacity)
        record = {"closed": False}; reservations.append(record)
        try: yield
        finally: record["closed"] = True

    monkeypatch.setattr(client, "_Peers", Peers)
    monkeypatch.setattr(client, "reserve_incremental_memory", reserve)
    monkeypatch.setattr(client._native, "FssClientRequest", NativeClient, raising=False)
    class Attempt:
        cancelled = False
        audit = None
        def __call__(self): return self.cancelled
        def record_audit(self, value): self.audit = value
    source = Frozen("client", cluster.double.capacity)
    return SimpleNamespace(events=events, peers=owners, clients=clients,
        reservations=reservations, attempt=Attempt(), source=source, transforms=transforms, requests=requests)


@pytest.mark.asyncio
async def test_ordinary_client_handler_path_completes_after_dealer_retirement_and_drains(cluster, client_route):
    attempt = client_route.attempt
    result = await asyncio.wait_for(asyncio.to_thread(client.execute_request,
            configuration(), client_route.source, [1, 2, 3], 3,
            {role: (ORIGINS[role], client_key(role)) for role in p.ROLES}, 5, attempt), 10)
    assert result.selected == (5, 6, 7)
    assert all(c.closed for c in client_route.clients)
    assert all(p.closed for p in client_route.peers)
    assert all(r["closed"] for r in client_route.reservations)
    assert "input-0-after-dealer-retirement" in client_route.events
    assert "input-1-after-dealer-retirement" in client_route.events
    assert "native-final-reconstruction" in client_route.events
    for _, action, raw in client_route.requests:
        if action == "admit":
            value = p.unpack(raw, limit=p.ADMISSION_LIMIT)
            assert raw == p.pack({"request": value["request"], "epoch": value["epoch"],
                "epochs": value["epochs"], "rows": 3, "outputs": 3, "contract": contract()}, limit=p.ADMISSION_LIMIT)
    assert attempt.audit is not None
    await retired(cluster)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", [
    "kind", "context",
    "plan", "pc", "metadata_order"])
async def test_changed_offline_material_is_rejected_before_first_native_peer_reply(cluster, mutation):
    await cluster.ready()
    def transform(request):
        if request.url.path.endswith("/material") and request.url.port == 7300:
            value = p.unpack(request.content, limit=p.FRAME_LIMIT)
            meta = p.metadata(value["metadata"])
            if mutation == "kind": meta["kind"] = "peer"
            elif mutation == "context": meta["kernel"]["context"]["session"] = "9" * 64
            elif mutation == "plan": meta["kernel"]["plan_digest"] = "d" * 64
            elif mutation == "pc": meta["pc"] += 1
            else: meta["kernel"]["ordinal"] = 1 - meta["kernel"]["ordinal"]
            value["metadata"] = p.canonical(meta)
            return httpx.Request(request.method, request.url, headers=request.headers,
                                 content=p.pack(value, limit=p.FRAME_LIMIT))
        return request
    cluster.double.peer_transform = transform
    response = await cluster.post("dealer", "prepare")
    if mutation in ("kind", "context"):
        assert response.status_code == 400
        assert not cluster.double.calls
        await retired(cluster, ("worker_a", "dealer"))
    else:
        # Opaque offline records can be queued but must match exact native
        # material callback bytes when taken. No duplicate clear interpreter.
        assert response.status_code == 200
        await cluster.inputs()
        assert (await cluster.post("worker_a", "start")).status_code == 200
        await retired(cluster, ("worker_a",))
        assert not any(phase == "online" for _, _, phase, _, _ in cluster.double.rpc_log)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["recipient_epoch", "sender_epoch", "sender", "credential", "duplicate_auth"])
async def test_peer_credential_and_epoch_fingerprints_bound_to_direction(cluster, mutation):
    await cluster.ready()
    message = {"sender": "dealer", "sender_epoch": cluster.epochs["dealer"],
        "recipient_epoch": cluster.epochs["worker_a"], "sequence": 0,
        "metadata": callback_metadata(cluster.request_id, "material", 0), "packet": b"packet"}
    credential = key("dealer", "worker_a")
    if mutation == "recipient_epoch": message["recipient_epoch"] = cluster.epochs["worker_b"]
    elif mutation == "sender_epoch": message["sender_epoch"] = cluster.epochs["worker_b"]
    elif mutation == "sender": message["sender"] = "worker_b"
    elif mutation == "credential": credential = client_key("worker_a")
    if mutation == "duplicate_auth":
        response = await cluster.clients["worker_a"].post(f"/v1/fss/{cluster.request_id}/material",
            content=p.pack(message, limit=p.FRAME_LIMIT), headers=[("authorization", f"Bearer {credential}")] * 2)
    else:
        response = await cluster.post("worker_a", "material", message, credential=credential)
    assert response.status_code == (401 if mutation in ("credential", "duplicate_auth") else 400)
    assert not cluster.double.calls


@pytest.mark.asyncio
async def test_request_replay_rejected_inside_epoch_and_failed_admission_burns_id(cluster):
    await cluster.manifests()
    first = (await cluster.admit(roles=("worker_a",), contract=b"mismatch"))["worker_a"]
    assert first.status_code == 400
    assert (await cluster.metrics("worker_a"))["seen_requests"] == 1
    assert (await cluster.admit(roles=("worker_a",)))["worker_a"].status_code == 400
    assert not cluster.double.loaded
    assert (await cluster.admit(roles=("worker_a",), request_id="2" * 64))["worker_a"].status_code == 200
    assert (await cluster.post("worker_a", "cancel")).status_code == 200


@pytest.mark.asyncio
async def test_late_native_callbacks_after_cancel_cannot_fetch_tape_or_send_dealer_data(cluster):
    await cluster.ready(); await cluster.prepare(); await cluster.inputs()
    cluster.double.hold_role = "worker_a"
    assert (await cluster.post("worker_a", "start")).status_code == 200
    assert await asyncio.to_thread(cluster.double.entered.wait, 2)
    native = next(n for n in cluster.double.natives if n.role == 0)
    cancel = asyncio.create_task(cluster.post("worker_a", "cancel"))
    await asyncio.sleep(.01)
    calls_before = len(cluster.double.rpc_log)
    with pytest.raises(ValueError):
        await asyncio.to_thread(native.callbacks.material, callback_metadata(cluster.request_id, "material", 0))
    with pytest.raises(ValueError):
        await asyncio.to_thread(native.callbacks.exchange, callback_metadata(cluster.request_id, "peer", 0), b"peer0-0")
    assert len(cluster.double.rpc_log) == calls_before
    assert not native.closed  # run must still drain before source/native close
    cluster.double.release.set()
    assert (await asyncio.wait_for(cancel, 3)).status_code == 200
    assert native.closed


@pytest.mark.asyncio
async def test_startup_cancel_waits_for_late_owner_and_replay_is_not_reset(cluster):
    cluster.double.load_wait = True
    task = asyncio.create_task(cluster.admit(roles=("worker_a",)))
    assert await asyncio.to_thread(cluster.double.load_entered.wait, 2)
    held = [r for r in cluster.double.reservations if not r["closed"]]
    task.cancel()
    await asyncio.sleep(.01)
    assert not task.done() and all(not r["closed"] for r in held)
    cluster.double.load_release.set()
    with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(task, 3)
    assert all(r["closed"] for r in held)
    assert cluster.double.loaded[-1].closed
    assert (await cluster.admit(roles=("worker_a",)))["worker_a"].status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [.25], indirect=True)
async def test_deadline_drains_late_run_before_reservation_retirement(cluster):
    await cluster.ready(); await cluster.prepare(); await cluster.inputs()
    cluster.double.hold_role = "worker_a"
    assert (await cluster.post("worker_a", "start")).status_code == 200
    assert await asyncio.to_thread(cluster.double.entered.wait, 2)
    native = next(n for n in cluster.double.natives if n.role == 0)
    await asyncio.sleep(.3)
    assert not native.closed
    assert any(not r["closed"] for r in cluster.double.reservations)
    cluster.double.release.set()
    await retired(cluster)
    assert native.closed


@pytest.mark.asyncio
async def test_holder_finish_response_lease_lasts_until_send_drains(cluster):
    await cluster.ready(); await cluster.prepare(); await cluster.inputs()
    for role in p.HOLDERS:
        assert (await cluster.post(role, "start")).status_code == 200
    entered, release = asyncio.Event(), asyncio.Event()
    messages = []
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": f"/v1/fss/{cluster.request_id}/finish", "raw_path": b"/",
        "query_string": b"", "root_path": "", "server": ("127.0.0.1", 7300), "client": ("127.0.0.1", 1),
        "headers": [(b"authorization", f"Bearer {client_key('worker_a')}".encode()), (b"content-length", b"0")]}
    async def receive(): return {"type": "http.request", "body": b"", "more_body": False}
    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body":
            entered.set(); await release.wait()
    held = [r for r in cluster.double.reservations if not r["closed"]]
    task = asyncio.create_task(cluster.apps["worker_a"](scope, receive, send))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert all(not r["closed"] for r in held)
        assert next(n for n in cluster.double.natives if n.role == 0).closed
        body = next(m["body"] for m in messages if m["type"] == "http.response.body")
        assert len(body) <= p.CONTROL_LIMIT
    finally:
        release.set(); await asyncio.wait_for(task, 3)
    assert sum(r["closed"] for r in held) == 1
    assert (await cluster.post("worker_b", "finish")).status_code == 200


@pytest.mark.parametrize("mutation", ["bool_count", "negative_count", "dealer_online", "self_bytes", "holder_to_dealer"])
def test_client_receipt_counters_reject_impossible_directions(mutation):
    counters = p.peer_bytes()
    role = "worker_a"
    if mutation == "bool_count": counters["worker_b"]["online_upload_bytes"] = True
    elif mutation == "negative_count": counters["worker_b"]["online_upload_bytes"] = -1
    elif mutation == "dealer_online": role = "dealer"; counters["worker_a"]["online_upload_bytes"] = 1
    elif mutation == "self_bytes": counters["worker_a"]["online_upload_bytes"] = 1
    else: counters["dealer"]["setup_upload_bytes"] = 1
    with pytest.raises(ValueError): p.collect_peer_bytes(counters, role)


@pytest.mark.asyncio
@pytest.mark.parametrize("fingerprint", ["source_lock", "tokenizer", "configuration", "transport", "role"])
async def test_client_manifest_fingerprint_mismatch_never_creates_input_shares(cluster, client_route, fingerprint):
    def transform(role, action, raw):
        if action == "manifest" and role == "worker_a":
            value = p.unpack(raw)
            value[fingerprint] = "wrong" if fingerprint in ("transport", "role") else "9" * 64
            return p.pack(value)
        return raw
    client_route.transforms.append(transform)
    with pytest.raises(ProtocolError, match="source or transport"):
        await asyncio.to_thread(client.execute_request, configuration(), client_route.source,
            [1, 2, 3], 3, {r: (ORIGINS[r], client_key(r)) for r in p.ROLES}, 5, client_route.attempt)
    assert not client_route.clients and not cluster.double.natives and not cluster.double.loaded
    assert all(r["closed"] for r in client_route.reservations)


@pytest.mark.asyncio
async def test_client_rejects_integer_alias_in_real_handler_ack(cluster, client_route):
    def transform(role, action, raw):
        if role == "worker_a" and action == "bind":
            value = p.unpack(raw)
            assert value == {"accepted": True}
            return p.pack({"accepted": 1})
        return raw
    client_route.transforms.append(transform)
    with pytest.raises(ProtocolError):
        await asyncio.to_thread(client.execute_request, configuration(), client_route.source,
            [1, 2, 3], 3, {r: (ORIGINS[r], client_key(r)) for r in p.ROLES}, 5, client_route.attempt)


@pytest.mark.asyncio
async def test_whole_handler_reply_budget_rejects_oversized_manifest_and_drains_loaded_source(cluster, monkeypatch):
    original = worker.load_role_source
    def load(*args, **kwargs):
        value = original(*args, **kwargs)
        value.manifest_bytes = b"m" * (p.CONTROL_LIMIT + 1)
        return value
    monkeypatch.setattr(worker, "load_role_source", load)
    response = (await cluster.admit(roles=("worker_a",)))["worker_a"]
    assert response.status_code == 400 and len(response.content) <= p.CONTROL_LIMIT
    assert cluster.double.loaded[-1].closed
    await retired(cluster, ("worker_a",))


@pytest.mark.asyncio
async def test_holder_only_local_seal_barrier_while_dealer_is_still_preparing(cluster, monkeypatch):
    """Contract boundary, not a privacy break: global ordering trusts the client.

    Stall dealer's first seal ACK after the receiving handler has sealed A. A
    locally accepts input/start before B seals and before dealer prepare retires.
    No real native input/share arithmetic or opening is constructed in this test.
    """
    await cluster.ready()
    sealed, release = asyncio.Event(), asyncio.Event()
    original = Routing.handle_async_request
    async def stall(self, request):
        result = await original(self, request)
        if request.url.path.endswith("/seal") and request.url.port == 7300:
            sealed.set(); await release.wait()
        return result
    monkeypatch.setattr(Routing, "handle_async_request", stall)
    native = next(n for n in cluster.double.natives if n.role == 0)
    entered = threading.Event()
    def paused_run(frame, callbacks):
        native.running, native.callbacks = True, callbacks
        try:
            assert frame == b"input-0"
            entered.set()
            assert native.stop.wait(3)
            raise ValueError("review-only early start cancelled")
        finally:
            native.running = False
    native.run = paused_run
    preparing = asyncio.create_task(cluster.post("dealer", "prepare"))
    try:
        await asyncio.wait_for(sealed.wait(), 3)
        assert cluster.double.sealed_roles == {"worker_a"}
        assert not next(n for n in cluster.double.natives if n.role == 2).closed
        assert (await cluster.post("worker_a", "input", raw=b"input-0")).status_code == 200
        assert (await cluster.post("worker_a", "start")).status_code == 200
        assert await asyncio.to_thread(entered.wait, 2)
        assert (await cluster.post("worker_a", "cancel")).status_code == 200
        assert not any(phase == "online" for _, _, phase, _, _ in cluster.double.rpc_log)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(preparing, return_exceptions=True), 5)


@pytest.mark.asyncio
async def test_application_shutdown_drains_bound_native_tape_and_rejects_new_control(double):
    from test_fss_worker import HTTPClient
    double.making_role = "worker_a"
    app = worker.create_fss_worker_app(configuration(), "worker_a", client_key("worker_a"),
        peer_connections=connections("worker_a"), peer_credentials=credentials("worker_a"))
    async with HTTPClient(transport=httpx.ASGITransport(app=app), base_url=ORIGINS["worker_a"],
                          headers={"authorization": f"Bearer {client_key('worker_a')}"}) as http:
        async with app.router.lifespan_context(app):
            manifest = p.unpack((await http.get("/v1/fss/manifest")).content)
            epochs = {"worker_a": manifest["epoch"], "worker_b": "7" * 64, "dealer": "8" * 64}
            admitted = await http.post("/v1/fss/admit", content=p.pack({"request": "1" * 64,
                "epoch": epochs["worker_a"], "epochs": epochs, "rows": 3, "outputs": 3,
                "contract": contract()}, limit=p.ADMISSION_LIMIT))
            assert admitted.status_code == 200
            bound = await http.post(f"/v1/fss/{'1' * 64}/bind",
                content=p.pack({"request_frame": b"request:" + b"1" * 64}))
            assert bound.status_code == 200
            assert not double.loaded[-1].closed and not double.natives[-1].closed
        assert double.loaded[-1].closed and double.natives[-1].closed
        assert all(r["closed"] for r in double.reservations)
        assert (await http.get("/v1/fss/manifest")).status_code == 503


@pytest.mark.asyncio
async def test_new_process_epoch_rejects_old_epoch_and_replay_ledger_is_epoch_local(double):
    from test_fss_worker import HTTPClient
    async def exercise(old_epoch=None):
        double.making_role = "worker_a"
        app = worker.create_fss_worker_app(configuration(), "worker_a", client_key("worker_a"),
            peer_connections=connections("worker_a"), peer_credentials=credentials("worker_a"))
        async with app.router.lifespan_context(app):
            async with HTTPClient(transport=httpx.ASGITransport(app=app), base_url=ORIGINS["worker_a"],
                    headers={"authorization": f"Bearer {client_key('worker_a')}"}) as http:
                current = p.unpack((await http.get("/v1/fss/manifest")).content)["epoch"]
                if old_epoch is None:
                    return current
                assert current != old_epoch
                async def admit(epoch, request_id):
                    return await http.post("/v1/fss/admit", content=p.pack({"request": request_id,
                        "epoch": epoch, "epochs": {"worker_a": epoch, "worker_b": "7" * 64,
                        "dealer": "8" * 64}, "rows": 3, "outputs": 3,
                        "contract": contract()}, limit=p.ADMISSION_LIMIT))
                assert (await admit(old_epoch, "1" * 64)).status_code == 400
                assert not double.loaded
                # Failed old-epoch request has already burned its ID in THIS epoch.
                assert (await admit(current, "1" * 64)).status_code == 400
                assert (await admit(current, "2" * 64)).status_code == 200
    old = await exercise()
    await exercise(old)
    assert all(x.closed for x in double.loaded)
    assert all(r["closed"] for r in double.reservations)


@pytest.mark.asyncio
async def test_whole_private_body_declaration_rejected_before_receive(cluster):
    await cluster.ready()
    reads, messages = [], []
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": f"/v1/fss/{cluster.request_id}/material", "raw_path": b"/",
        "query_string": b"", "root_path": "", "server": ("127.0.0.1", 7300), "client": ("127.0.0.1", 1),
        "headers": [(b"authorization", f"Bearer {key('dealer', 'worker_a')}".encode()),
                    (b"content-length", str(p.FRAME_LIMIT + 1).encode())]}
    async def receive():
        reads.append(True)
        return {"type": "http.request", "body": b"unread-private-body", "more_body": False}
    async def send(message): messages.append(message)
    await cluster.apps["worker_a"](scope, receive, send)
    assert not reads
    assert next(m["status"] for m in messages if m["type"] == "http.response.start") == 400
    await retired(cluster, ("worker_a",))


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_at_boundary", [False, True])
async def test_more_than_4096_native_admitted_records_through_authenticated_lifecycle(cluster, replay_at_boundary):
    count = cluster.double.kernels = 4097
    if replay_at_boundary:
        def transform(request):
            if request.url.path.endswith("/material") and request.url.port == 7300:
                value = p.unpack(request.content, limit=p.FRAME_LIMIT)
                if value["sequence"] == 4096:
                    value["sequence"] = 4095
                    return httpx.Request(request.method, request.url, headers=request.headers,
                                         content=p.pack(value, limit=p.FRAME_LIMIT))
            return request
        cluster.double.peer_transform = transform
    await cluster.ready()
    prepared = await asyncio.wait_for(cluster.post("dealer", "prepare"), 30)
    if replay_at_boundary:
        assert prepared.status_code == 400
        assert not cluster.double.calls  # failure is still strictly offline
        await retired(cluster, ("dealer", "worker_a"))
        assert (await cluster.post("worker_b", "cancel")).status_code == 200
    else:
        assert prepared.status_code == 200
        assert cluster.double.sealed_roles == set(p.HOLDERS)
        assert (await cluster.post("dealer", "finish")).status_code == 200
        assert next(n for n in cluster.double.natives if n.role == 2).closed
        await cluster.inputs()
        for role in p.HOLDERS:
            assert (await cluster.post(role, "start")).status_code == 200
        replies = await asyncio.wait_for(asyncio.gather(*(cluster.post(r, "finish") for r in p.HOLDERS)), 30)
        assert all(r.status_code == 200 for r in replies)
        for role, reply in zip(p.HOLDERS, replies):
            _, _, receipt = client._receipt(reply.content, role)
            assert receipt["records_written"] == receipt["records_read"] == count
            assert receipt["state"] == "finished"
        assert len(cluster.double.calls) == 2 * count
    await retired(cluster)
    # The native double deliberately retains callback objects after close.
    # They must retain neither payload/native/source owners nor callback queues.
    for native in cluster.double.natives:
        if native.callbacks is None:
            continue
        attempt = native.callbacks.attempt
        assert attempt.reservation is attempt.native is attempt.tape is attempt.loaded is None
        assert attempt.input is attempt.output is attempt.mailbox is attempt.contract is None
        assert not attempt.callback_tasks and not attempt.callback_futures
        with pytest.raises(ValueError):
            await asyncio.to_thread(native.callbacks.finish)
        native.callbacks.cancel()  # retained cancellation owner remains idempotent


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [300, {"capacity": "decoder"}], indirect=True)
async def test_cancelled_callback_future_keeps_single_live_slot_until_coroutine_drains(cluster, monkeypatch):
    await cluster.ready()
    native = next(n for n in cluster.double.natives if n.role == 2)
    entered, release = asyncio.Event(), asyncio.Event()
    failed, native_release = threading.Event(), threading.Event()
    original = Routing.handle_async_request
    async def stalled(self, request):
        response = await original(self, request)
        entered.set()
        # Model a transport cancellation finalizer that still owns its body.
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        return response
    monkeypatch.setattr(Routing, "handle_async_request", stalled)
    def run(frame, callbacks):
        native.running, native.callbacks = True, callbacks
        try:
            try:
                callbacks.deliver(callback_metadata(cluster.request_id, "material", 0), b"packet")
            except BaseException:
                failed.set()
                assert native_release.wait(5)
                raise ValueError("test callback cancellation") from None
        finally:
            native.running = False
    native.run = run
    preparing = asyncio.create_task(cluster.post("dealer", "prepare"))
    cancelling = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        callbacks = native.callbacks  # intentionally retain the owner past cancellation
        attempt = callbacks.attempt
        with attempt.callback_lock:
            assert len(attempt.callback_futures) == 1
            next(iter(attempt.callback_futures)).cancel()
        assert await asyncio.to_thread(failed.wait, 2)
        assert not attempt.cancelled() and not attempt.callback_futures
        assert attempt.callback_pending and len(attempt.callback_tasks) == 1
        calls = len(cluster.double.rpc_log)
        with pytest.raises(ValueError, match="callback retired"):
            await asyncio.wait_for(asyncio.to_thread(callbacks.deliver,
                callback_metadata(cluster.request_id, "material", 1), b"second-packet"), 1)
        assert len(cluster.double.rpc_log) == calls
        held = [r for r in cluster.double.reservations if not r["closed"]]
        storage = [r for r in cluster.double.storage_reservations if r["role"] == "dealer"]
        loaded = next(x for x in cluster.double.loaded if x.role == "dealer")
        cancelling = asyncio.create_task(cluster.post("dealer", "cancel"))
        native_release.set()
        await asyncio.sleep(.02)
        assert not cancelling.done() and not loaded.closed
        assert all(not r["closed"] for r in held)
        assert all(not r["closed"] for r in storage)
        release.set()
        assert (await asyncio.wait_for(cancelling, 3)).status_code == 200
        assert native.closed and loaded.closed
        assert all(r["closed"] for r in storage)
        assert not attempt.callback_pending and not attempt.callback_tasks and not attempt.callback_futures
        assert attempt.reservation is attempt.native is attempt.loaded is None
        with pytest.raises(ValueError):
            await asyncio.to_thread(callbacks.deliver,
                callback_metadata(cluster.request_id, "material", 0), b"late-packet")
    finally:
        native_release.set()
        release.set()
        await asyncio.wait_for(asyncio.gather(preparing, *([cancelling] if cancelling else []),
                                              return_exceptions=True), 5)
