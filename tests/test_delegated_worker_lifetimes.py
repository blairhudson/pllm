"""Worker HTTP payload retirement, independent of native numerical execution."""

import asyncio
from concurrent.futures import CancelledError
from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from pllm.runtime import delegated_worker as worker


class _Buffer(bytearray):
    pass


class _Bytes(bytes):
    """bytes cannot be weak-referenced; observe their actual destruction instead."""

    def __new__(cls, value, tracker, label):
        result = super().__new__(cls, value)
        result.tracker, result.label = tracker, label
        tracker.live_bytes[id(result)] = label
        tracker.created.append(label)
        return result

    def __del__(self):
        self.tracker.live_bytes.pop(id(self))


class _Payloads:
    def __init__(self):
        self.buffers = []
        self.live_bytes = {}
        self.created = []

    def buffer(self, value=b"", label="partial-body"):
        result = _Buffer(value)
        self.buffers.append((label, weakref.ref(result)))
        self.created.append(label)
        return result

    def bytes(self, value=b"", label="received-body"):
        return _Bytes(value, self, label)

    def live(self):
        return [label for label, ref in self.buffers if ref() is not None] + list(self.live_bytes.values())


def _raise_failure(payloads, kind):
    held = payloads.buffer(b"callee payload", "failed-callee")
    assert held
    if kind == "error":
        raise ValueError("controlled worker failure")
    if kind == "cancel":
        raise asyncio.CancelledError("controlled cancellation")
    try:
        _raise_failure(payloads, "error")
    except ValueError as cause:
        if kind == "cancel_cause":
            raise asyncio.CancelledError("caused cancellation") from cause
        if kind == "cancel_context":
            raise asyncio.CancelledError("suppressed-context cancellation") from None
        if kind == "cancel_group":
            raise BaseExceptionGroup("grouped cancellation", [asyncio.CancelledError(), cause]) from None
        raise AssertionError(kind)


class _Incoming:
    def __init__(self, payloads, *, mode="complete", empty=False):
        self.payloads, self.mode, self.empty = payloads, mode, empty
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.calls == 1:
            partial = self.mode in {"cancel", "cancel_cause", "cancel_context", "cancel_group", "timeout", "disconnect"}
            return {"type": "http.request", "body": b"" if self.empty else self.payloads.buffer(b"x" * 64, "received-block"),
                    "more_body": partial}
        if self.mode == "timeout":
            held = self.payloads.buffer(b"waiting receive", "timed-out-callee")
            await asyncio.Future()
            assert held  # Keep the callee owner live until cancellation unwinds it.
        if self.mode.startswith("cancel"):
            _raise_failure(self.payloads, self.mode)
        if self.mode == "post_cancel":
            _raise_failure(self.payloads, "cancel_context")
        return {"type": "http.disconnect"}


def _request(incoming, *, action="step", length=64, rid="a" * 64, authorized=True):
    headers = [(b"content-length", str(length).encode())]
    if authorized:
        headers.append((b"authorization", b"Bearer " + b"k" * 32))
    return Request({"type": "http", "method": "POST", "path": f"/v1/delegation/{rid}/{action}",
                    "headers": headers, "path_params": {"request": rid, "action": action}}, incoming)


async def _connected():
    return False


async def _successful_operation(h, action):
    empty = action != "step"
    request = _request(_Incoming(h.payloads, empty=empty), action=action, length=0 if empty else 64)
    request.is_disconnected = _connected
    response = await h.operation(request)
    assert response.status_code == 200
    return response


@pytest.fixture
def harness(monkeypatch):
    """Use real routing, body parsing and retirement; stub only source/native work."""
    payloads = _Payloads()
    monkeypatch.setattr(worker, "bytearray", payloads.buffer, raising=False)
    monkeypatch.setattr(worker, "bytes", payloads.bytes, raising=False)
    monkeypatch.setattr(worker, "admit_source", lambda _: SimpleNamespace(capacity="decoder"))
    experiment = SimpleNamespace(
        resolve=lambda: SimpleNamespace(client_runtime="compiled_gkm_v1"),
        pipeline=SimpleNamespace(components={"linear": SimpleNamespace(params={"capacity": "decoder"})}),
        budget=SimpleNamespace(requests=64),
    )
    app = worker.create_delegated_worker_app(experiment, "worker_a", "k" * 32)
    owner = app.state.delegated_owner
    operation = next(route.endpoint for route in app.routes if route.path.endswith("/{request}/{action}"))
    retained, released, calls = [], [], []
    original_clear = worker._clear_failure_frames

    def clear(error):
        # Retain exceptions even when the endpoint converts them into HTTP 400.
        retained.append(error)
        original_clear(error)

    monkeypatch.setattr(worker, "_clear_failure_frames", clear)

    @contextmanager
    def reservation():
        try:
            yield
        finally:
            # Check at refund, without gc.collect or discarding the exception.
            released.append(payloads.live())
            assert not released[-1], f"payload survived reservation release: {released[-1]}"

    class Native:
        def execute(self, raw):
            calls.append("execute")
            assert raw == b"x" * 64
            return payloads.bytes(b"role-local-reply", "native-result")

        def finish(self):
            calls.append("finish")
            return payloads.bytes(b'{"completion":true}', "completion-receipt")

        def abort(self):
            calls.append("abort")

    owner.active = Native()
    owner.request = "a" * 64
    owner.deadline = worker.time.monotonic() + 60
    owner.reservation = worker._Reservation(reservation())
    return SimpleNamespace(owner=owner, operation=operation, payloads=payloads, retained=retained,
                           released=released, calls=calls)


@pytest.mark.parametrize("mode", ["cancel", "cancel_cause", "cancel_context", "cancel_group",
                                  "timeout", "disconnect", "oversize", "truncated"])
def test_partial_operation_body_drops_before_refund_with_retained_exception(harness, mode):
    h = harness

    async def run():
        if mode == "timeout":
            h.owner.deadline = worker.time.monotonic() + 0.05
        incoming = _Incoming(h.payloads, mode=mode)
        request = _request(incoming, length=32 if mode == "oversize" else 128)
        if mode.startswith("cancel"):
            expected = BaseExceptionGroup if mode == "cancel_group" else asyncio.CancelledError
            with pytest.raises(expected) as caught:
                await h.operation(request)
            assert caught.value is h.retained[0]
        else:
            assert (await h.operation(request)).status_code == 400
        assert h.released == [[]] and not h.payloads.live()
        assert h.owner.active is None and h.owner.reservation is None
        assert h.calls == ["abort"] and len(h.retained) == 1
        assert {"partial-body", "received-block"} <= set(h.payloads.created)
        if mode == "timeout":
            assert isinstance(h.retained[0], TimeoutError)
            assert isinstance(h.retained[0].__cause__, asyncio.CancelledError)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["complete", "post_cancel"])
def test_post_execution_disconnect_drains_input_and_result_before_refund(harness, mode):
    h = harness

    async def run():
        request = _request(_Incoming(h.payloads, mode=mode))
        if mode == "post_cancel":
            with pytest.raises(asyncio.CancelledError) as caught:
                await h.operation(request)
            assert caught.value is h.retained[0]
        else:
            assert (await h.operation(request)).status_code == 400
        assert h.calls == ["execute", "abort"]
        assert {"received-body", "native-result"} <= set(h.payloads.created)
        assert h.released == [[]] and not h.payloads.live()
        assert h.owner.active is None and h.owner.reservation is None

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["native_error", "native_cancel_group", "response"])
def test_finish_failures_drain_receipt_and_callee_frames_before_refund(harness, monkeypatch, failure):
    h = harness
    if failure == "response":
        original_headers = worker._PayloadResponse.init_headers

        def headers(self, headers=None):
            if self.media_type == "application/json":
                held = self.body
                assert held == b'{"completion":true}'
                _raise_failure(h.payloads, "error")
            return original_headers(self, headers)

        monkeypatch.setattr(worker._PayloadResponse, "init_headers", headers)
    else:
        def finish():
            h.calls.append("finish")
            receipt = h.payloads.bytes(b"incomplete receipt", "completion-receipt")
            assert receipt
            _raise_failure(h.payloads, "cancel_group" if failure == "native_cancel_group" else "error")

        h.owner.active.finish = finish

    async def run():
        request = _request(_Incoming(h.payloads, empty=True), action="finish", length=0)
        if failure == "native_cancel_group":
            with pytest.raises(BaseExceptionGroup) as caught:
                await h.operation(request)
            assert caught.value is h.retained[0]
        else:
            assert (await h.operation(request)).status_code == 400
        assert h.calls == ["finish", "abort"]
        assert "completion-receipt" in h.payloads.created
        assert h.released == [[]] and not h.payloads.live()
        assert h.owner.active is None and h.owner.reservation is None

    asyncio.run(run())


@pytest.mark.parametrize("mode,length", [("oversize", 32), ("truncated", 128)])
def test_body_itself_clears_buffers_and_closes_rejected_stream(monkeypatch, mode, length):
    payloads = _Payloads()
    monkeypatch.setattr(worker, "bytearray", payloads.buffer, raising=False)

    async def run():
        with pytest.raises(ValueError) as caught:
            await worker._body(_request(_Incoming(payloads, mode=mode), length=length), 1024)
        # No endpoint cleanup helper: its retained traceback must already have
        # released the partial buffer and the stream's suspended yielded block.
        assert caught.value.__traceback__ is not None
        assert {"partial-body", "received-block"} <= set(payloads.created)
        assert not payloads.live()

    asyncio.run(run())


@pytest.mark.parametrize("rid,authorized,status", [("b" * 64, True, 404), ("bad", True, 400),
                                                  ("a" * 64, False, 401)])
def test_unrelated_operation_cannot_retire_active_session(harness, rid, authorized, status):
    h = harness

    async def run():
        incoming = _Incoming(h.payloads)
        active = h.owner.active
        response = await h.operation(_request(incoming, rid=rid, authorized=authorized))
        assert response.status_code == status
        assert h.owner.active is active and h.owner.reservation is not None
        assert incoming.calls == 0 and not h.calls and not h.released and not h.retained
        h.owner.retire()
        assert h.released == [[]]

    asyncio.run(run())


@pytest.mark.parametrize("terminal", ["cancel", "finish"])
def test_retained_successful_replies_refund_only_after_last_unsent_drop(harness, terminal):
    h = harness

    async def run():
        first = await _successful_operation(h, "step")
        last = await _successful_operation(h, "step")
        completion = await _successful_operation(h, terminal)
        assert h.owner.active is None and h.owner.reservation is None
        assert not h.released
        assert first.body == last.body == b"role-local-reply"
        assert completion.body
        first.close()
        first.close()
        assert first.body == b"" and not h.released
        del completion  # No ASGI call: final destruction must drop its lease.
        assert not h.released
        del last
        assert h.released == [[]] and not h.payloads.live()
        h.owner.retire()
        assert h.released == [[]]

    asyncio.run(run())


def test_sent_or_closed_reply_does_not_refund_live_native_root(harness):
    h = harness

    async def send(message):
        assert message["type"] in {"http.response.start", "http.response.body"}

    async def run():
        reply = await _successful_operation(h, "step")
        await reply({"type": "http"}, None, send)
        assert reply.body == b"" and not h.payloads.live() and not h.released
        reply.close()
        h.owner.retire()
        assert h.calls == ["execute", "abort"] and h.released == [[]]
        with pytest.raises(RuntimeError, match="retired"):
            await reply({"type": "http"}, None, send)
        assert h.released == [[]]

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["complete", "cancel", "error", "cancel_context", "cancel_group"])
def test_stalled_send_owns_reply_through_completion_or_retained_failure(harness, outcome):
    h = harness

    async def run():
        started, proceed = asyncio.Event(), asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body":
                held = message["body"]
                assert held == b"role-local-reply"
                started.set()
                await proceed.wait()
                if outcome != "complete":
                    _raise_failure(h.payloads, outcome)

        reply = await _successful_operation(h, "step")
        task = asyncio.create_task(reply({"type": "http"}, None, send))
        await started.wait()
        h.owner.retire()
        reply.close()  # Must not release a body borrowed by the in-flight send.
        assert reply.body == b"role-local-reply" and not h.released
        if outcome == "cancel":
            task.cancel()
        else:
            proceed.set()
        if outcome == "complete":
            await task
            assert not h.retained
        else:
            expected = (ValueError if outcome == "error" else BaseExceptionGroup
                        if outcome == "cancel_group" else asyncio.CancelledError)
            with pytest.raises(expected) as caught:
                await task
            assert caught.value is h.retained[0]
            assert caught.value.__traceback__ is not None
        # Keep both response and failed task/exception alive across this check.
        assert reply.body == b"" and h.released == [[]] and not h.payloads.live()
        reply.close()
        assert h.released == [[]]

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["step_constructor", "cancel_constructor", "cancel_pack"])
def test_reply_constructor_and_pack_failures_retire_before_root_refund(harness, monkeypatch, failure):
    h = harness
    original_pack = worker.pack

    def pack(value, **kwargs):
        held = h.payloads.bytes(original_pack(value, **kwargs), "packed-reply")
        if failure == "cancel_pack":
            _raise_failure(h.payloads, "error")
        return held

    monkeypatch.setattr(worker, "pack", pack)
    if failure.endswith("constructor"):
        def headers(self, headers=None):
            held = self.body
            assert held
            _raise_failure(h.payloads, "error")

        monkeypatch.setattr(worker._PayloadResponse, "init_headers", headers)

    async def run():
        action = "step" if failure.startswith("step") else "cancel"
        request = _request(_Incoming(h.payloads, empty=action != "step"), action=action,
                           length=64 if action == "step" else 0)
        request.is_disconnected = _connected
        assert (await h.operation(request)).status_code == 400
        assert h.retained and h.released == [[]] and not h.payloads.live()
        assert h.owner.active is None and h.owner.reservation is None

    asyncio.run(run())


def test_native_abort_failure_drops_owner_before_root_refund(harness, monkeypatch):
    h = harness
    h.owner.active.payload = h.payloads.buffer(b"native owner", "native-owner")

    def abort(self):
        held = self.payload
        assert held
        _raise_failure(h.payloads, "error")

    monkeypatch.setattr(type(h.owner.active), "abort", abort)
    with pytest.raises(ValueError) as caught:
        h.owner.retire()
    assert caught.value is h.retained[0]
    assert h.owner.active is None and h.owner.reservation is None
    assert h.released == [[]] and not h.payloads.live()


def test_reply_limit_rollback_and_root_close_are_idempotent(harness):
    h = harness
    root = h.owner.reservation
    with pytest.raises(ValueError, match="bounded owner"):
        worker._PayloadResponse(b"too long", root, limit=1)
    assert not h.released
    h.owner.retire()
    root.close()
    with pytest.raises(ValueError, match="root is retired"):
        root.lease()
    assert h.released == [[]]


@pytest.fixture
def admission(monkeypatch):
    """A one-request host budget; source loading is observable and metadata-only."""
    payloads = _Payloads()
    retained, refunds, loads = [], [], []
    source = SimpleNamespace(capacity="decoder", source_lock_digest="c" * 64,
                             tokenizer_digest="d" * 64,
                             runtime_contract=lambda *args: b"runtime-contract")
    monkeypatch.setattr(worker, "admit_source", lambda _: source)
    monkeypatch.setattr(worker, "request_memory_bytes", lambda *args, **kwargs: 64 << 20)
    monkeypatch.setattr(worker, "bytearray", payloads.buffer, raising=False)
    original_pack = worker.pack
    monkeypatch.setattr(worker, "pack", lambda value, **kwargs:
                        payloads.bytes(original_pack(value, **kwargs), "packed-reply"))
    original_clear = worker._clear_failure_frames

    def clear(error):
        retained.append(error)
        original_clear(error)

    monkeypatch.setattr(worker, "_clear_failure_frames", clear)
    experiment = SimpleNamespace(
        resolve=lambda: SimpleNamespace(client_runtime="compiled_gkm_v1"),
        pipeline=SimpleNamespace(components={"linear": SimpleNamespace(params={"capacity": "decoder"})}),
        budget=SimpleNamespace(requests=64), configuration_digest=lambda: "b" * 64,
    )
    app = worker.create_delegated_worker_app(experiment, "worker_a", "k" * 32)
    owner = app.state.delegated_owner
    endpoint = next(route.endpoint for route in app.routes if route.path.endswith("/admit"))
    root_bytes, body_bytes = 64 << 20, 4 * owner.admission_limit
    ledger = {"used": 0, "limit": root_bytes + body_bytes}

    @contextmanager
    def reserve(size):
        if ledger["used"] + size > ledger["limit"]:
            raise ValueError("controlled host reservation exhaustion")
        ledger["used"] += size
        try:
            yield
        finally:
            if size == root_bytes:
                assert not payloads.live(), payloads.live()
            ledger["used"] -= size
            refunds.append(size)

    monkeypatch.setattr(worker, "reserve_incremental_memory", reserve)

    class Native:
        def __init__(self, rid):
            self.rid = rid
            self.payload = payloads.buffer(b"native owner", "native-owner")

        def request_id(self):
            return self.rid

        def abort(self):
            self.payload = None

    def load(*args, worker):
        assert ledger["used"] == root_bytes + body_bytes
        loads.append(worker[3])
        return Native(worker[3])

    source.load_transport = load

    def request(rid="a" * 64):
        raw = original_pack({"schema": "pllm.gkm.admit.v1", "request": rid, "epoch": owner.epoch,
            "configuration": experiment.configuration_digest(), "source_lock": source.source_lock_digest,
            "tokenizer": source.tokenizer_digest, "input_rows": 1, "output_tokens": 1,
            "publication": b"publication", "pin": bytes(32), "start": bytes.fromhex(rid)})

        async def receive():
            return {"type": "http.request", "body": raw, "more_body": False}

        result = _request(receive, action="admit", length=len(raw), rid=rid)
        result.is_disconnected = _connected
        return result

    return SimpleNamespace(app=app, owner=owner, source=source, endpoint=endpoint, request=request, payloads=payloads,
                           ledger=ledger, loads=loads, refunds=refunds, retained=retained,
                           root_bytes=root_bytes, body_bytes=body_bytes)


def test_pending_admission_reply_keeps_host_claim_before_another_source_import(admission):
    h = admission

    async def run():
        first = await h.endpoint(h.request())
        assert first.status_code == 200 and len(h.loads) == 1
        h.owner.retire()
        assert h.owner.reservation is None and h.ledger["used"] == h.root_bytes
        assert h.refunds == [h.body_bytes]
        blocked = await h.endpoint(h.request("b" * 64))
        assert blocked.status_code == 400 and len(h.loads) == 1
        assert "reservation exhaustion" in str(h.retained[-1])
        assert h.ledger["used"] == h.root_bytes
        del first
        assert h.ledger["used"] == 0 and not h.payloads.live()
        accepted = await h.endpoint(h.request("e" * 64))
        assert accepted.status_code == 200 and len(h.loads) == 2
        accepted.close()
        assert h.ledger["used"] == h.root_bytes
        h.owner.retire()
        assert h.ledger["used"] == 0 and not h.payloads.live()
        assert h.refunds.count(h.root_bytes) == 2

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["pack", "constructor"])
def test_admission_reply_failures_drain_before_host_refund(admission, monkeypatch, failure):
    h = admission
    if failure == "pack":
        original_pack = worker.pack

        def pack(value, **kwargs):
            held = original_pack(value, **kwargs)
            assert held
            _raise_failure(h.payloads, "error")

        monkeypatch.setattr(worker, "pack", pack)
    else:
        def headers(self, headers=None):
            held = self.body
            assert held
            _raise_failure(h.payloads, "error")

        monkeypatch.setattr(worker._PayloadResponse, "init_headers", headers)

    async def run():
        assert (await h.endpoint(h.request())).status_code == 400
        assert len(h.loads) == 1 and h.retained
        assert h.owner.active is None and h.owner.reservation is None
        assert h.ledger["used"] == 0 and not h.payloads.live()
        assert sorted(h.refunds) == sorted([h.root_bytes, h.body_bytes])

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["accepted", "error", "cancel", "cancel_context"])
def test_http_admission_delivery_and_retirement_preserve_existing_behavior(admission, monkeypatch, outcome):
    h = admission
    raw = asyncio.run(h.request().body())
    if outcome != "accepted":
        def load(*args, **kwargs):
            _raise_failure(h.payloads, outcome)

        monkeypatch.setattr(h.source, "load_transport", load)

    with TestClient(h.app, headers={"authorization": "Bearer " + "k" * 32}) as client:
        if outcome.startswith("cancel"):
            with pytest.raises(CancelledError):
                client.post("/v1/delegation/admit", content=raw)
        else:
            result = client.post("/v1/delegation/admit", content=raw)
            assert result.status_code == (200 if outcome == "accepted" else 400)
            if outcome == "accepted":
                assert worker.unpack(result.content)["schema"] == "pllm.gkm.ready.v1"
                assert h.payloads.live() == ["native-owner"]
                assert h.ledger["used"] == h.root_bytes
        if outcome != "accepted":
            assert h.retained and h.owner.active is None
            assert h.ledger["used"] == 0
    assert h.owner.active is None and h.owner.reservation is None
    assert h.ledger["used"] == 0 and not h.payloads.live()
    assert sorted(h.refunds) == sorted([h.root_bytes, h.body_bytes])


@pytest.mark.parametrize("blocks", [[b"a", b"", b"b", b"c"], [b"abc"], [b"", b""]])
def test_body_fixed_allocation_and_exact_length(monkeypatch, blocks):
    allocations = []
    expected = sum(map(len, blocks))

    def allocate(size):
        allocations.append(size)
        return bytearray(size)

    monkeypatch.setattr(worker, "bytearray", allocate, raising=False)

    async def run():
        iterator = iter(enumerate(blocks))

        async def receive():
            index, block = next(iterator)
            return {"type": "http.request", "body": block, "more_body": index < len(blocks) - 1}

        assert await worker._body(_request(receive, length=expected), 64) == b"".join(blocks)

    asyncio.run(run())
    assert allocations == [expected]


@pytest.mark.parametrize("authorized,busy,status", [(False, False, 401), (True, True, 409), (True, False, 400)])
def test_admission_preflight_preserves_auth_busy_and_native_load_order(admission, monkeypatch, authorized, busy, status):
    h = admission
    raw = b"\x8b\xabpublication\xdc\x00\x20"
    reads = []

    async def receive():
        reads.append(True)
        return {"type": "http.request", "body": raw, "more_body": False}

    def forbidden(*args, **kwargs):
        pytest.fail("invalid admission reached native or msgpack value loading")

    monkeypatch.setattr(worker, "unpack", forbidden)
    monkeypatch.setattr(worker.unpack_admission.__globals__["msgpack"], "unpackb", forbidden)
    monkeypatch.setattr(h.source, "load_transport", forbidden)
    sentinel = object()
    if busy:
        h.owner.active = sentinel
        h.owner.deadline = worker.time.monotonic() + 60
    try:
        response = asyncio.run(h.endpoint(_request(receive, action="admit", length=len(raw), authorized=authorized)))
        assert response.status_code == status
        assert bool(reads) == (status == 400)
        assert not h.loads and h.owner.seen == set()
        assert h.ledger["used"] == 0
        assert h.refunds == ([h.body_bytes] if status == 400 else [])
        assert h.owner.active is (sentinel if busy else None)
    finally:
        h.owner.active = None


def test_admission_replay_still_burns_only_valid_request(admission):
    h = admission

    async def run():
        response = await h.endpoint(h.request())
        assert response.status_code == 200
        response.close()
        h.owner.retire()
        assert (await h.endpoint(h.request())).status_code == 400
        assert len(h.loads) == 1 and h.owner.seen == {"a" * 64}
        assert h.ledger["used"] == 0

    asyncio.run(run())
