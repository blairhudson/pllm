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
from pllm.deployment import resources


# Same failed-ledger rooting policy as test_transport_construction_custody:
# teardown may forget verified empty dictionaries, never unresolved custody.
_LIFETIME_FAILED_LEDGERS = set()


def _forget_empty_ledger(capacity):
    if type(capacity._owners) is dict and not capacity._owners:
        _LIFETIME_FAILED_LEDGERS.discard(capacity)


def _validate_refund(snapshot):
    """Independent value-only oracle; never change a production ledger."""
    assert snapshot["ledger_same"] and snapshot["token_absent"]
    assert snapshot["backing_dead"]
    assert snapshot["usage"] == snapshot["remaining_usage"]
    if snapshot["worker_root"]:
        assert snapshot["retained_clear"] and snapshot["exited"]
        assert snapshot["failed_empty"] and snapshot["not_draining"]
        assert snapshot["done"]


class _RefundObservations:
    """Backend boundaries prove ordering; post-call checks observe refunds.

    These observations are NOT exact refund timestamps or native byte proofs.
    Keep the original plain dictionary. No pop override or shadow usage counter.
    Records contain weak manager/backend references and scalar observations only.
    Integer IDs label diagnostics; never use them to reconstruct token identity.
    """

    def __init__(self, capacity, payloads, monkeypatch):
        self.capacity, self.payloads = capacity, payloads
        self.ledger = capacity._owners
        assert type(self.ledger) is dict
        self.records, self.backend_exit, self.actual_refund_observed = [], [], []
        self.fail_backend = False
        self.physical_close = False
        self.monkeypatch = monkeypatch
        self.backend_type = None

    @property
    def used(self):
        assert self.capacity._owners is self.ledger
        return sum(self.ledger.values())

    def _manager_identity(self, record, manager):
        assert manager is not None and manager is record.manager(), "unresolved manager identity"
        assert manager._capacity is self.capacity and self.capacity._owners is self.ledger
        assert manager._manager is record.backend(), "unresolved backend identity"
        assert manager._amount == record.amount

    def watch(self, manager):
        assert not any(not r.completed and r.manager() is manager for r in self.records)
        record = SimpleNamespace(manager=weakref.ref(manager),
            backend=weakref.ref(manager._manager), token_id=id(manager._token),
            manager_id=id(manager), capacity_id=id(manager._capacity),
            ledger_id=id(self.ledger), amount=manager._amount, admitted=False,
            worker_root=False, completed=False, ordinary_removal_proved=False,
            boundary_ok=False, holds=[])
        self.records.append(record)
        if self.backend_type is None:
            self.backend_type = type(manager._manager)
            real_exit = self.backend_type.__exit__

            def backend_exit(backend, *exc):
                matches = [r for r in self.records if not r.completed and r.backend() is backend]
                if not matches:
                    return real_exit(backend, *exc)
                assert len(matches) == 1, "unresolved backend association"
                record = matches[0]
                manager = record.manager()
                self._manager_identity(record, manager)
                token = manager._token  # Strong identity witnesses live only in this callback.
                custody = manager._failed
                retirement = custody[0] if custody else None
                root = retirement.root if retirement is not None else None
                before = self._boundary(record, manager, token)
                self.backend_exit.append((record.token_id, "before", before))
                if self.fail_backend and record.worker_root:
                    if self.physical_close:
                        real_exit(backend, *exc)
                        self._manager_identity(record, manager)
                        assert manager._failed is custody and retirement.root is root
                        after = self._boundary(record, manager, token)
                        assert after == before
                        self.backend_exit.append((record.token_id, "after-ambiguous", after))
                    raise ValueError("backend exit ambiguous")
                result = real_exit(backend, *exc)
                self._manager_identity(record, manager)
                assert manager._token is token
                if record.worker_root:
                    assert manager._failed is custody and retirement.root is root
                    after = self._boundary(record, manager, token)
                    assert after == before
                    self.backend_exit.append((record.token_id, "after", after))
                else:
                    # Actual original-token removal, not backend return alone.
                    assert token not in self.ledger and token.retained is None
                    record.ordinary_removal_proved = True
                record.boundary_ok = True
                return result

            self.monkeypatch.setattr(self.backend_type, "__exit__", backend_exit)
        return manager

    def hold_snapshot(self, token, amount, phase):
        # Token slots are not weakrefable. Resolve ONLY through live managers.
        matches = []
        for record in self.records:
            if record.completed:
                continue
            manager = record.manager()
            if manager is not None and manager._token is token:
                self._manager_identity(record, manager)
                matches.append(record)
        assert len(matches) == 1, "unresolved hold association"
        record = matches[0]
        manager = record.manager()
        self._manager_identity(record, manager)
        assert manager._token is token and self.ledger[token] == amount == record.amount
        record.admitted = True
        record.holds.append((phase, self.used, self.ledger[token], amount))

    def _boundary(self, record, manager, token):
        self._manager_identity(record, manager)
        assert not self.capacity.lock._is_owned()
        assert manager._token is token and self.ledger[token] == record.amount
        record.admitted = True
        record.worker_root = bool(manager._failed)
        if not record.worker_root:
            return (record.token_id, record.amount, self.used)
        retirement = manager._failed[0]
        assert token.retained is manager and not manager._exited
        assert manager._failed is retirement.custody
        assert retirement.manager is manager and retirement.abort_ok
        assert retirement.terminal and retirement.root_closed and retirement.exit_seen
        assert retirement.native is None and not retirement.done
        assert not self.payloads.live(), self.payloads.live()
        return (id(self.capacity), id(self.ledger), record.token_id, id(manager),
                record.amount, id(manager._failed), id(retirement), id(retirement.root),
                id(manager._manager), self.used, True)

    def observe_refunds(self, retirement=None):
        # Called AFTER endpoint/retire/drain/reply/lifespan completion. Do not
        # demand terminal assignments at pop: production assigns them afterwards.
        assert self.capacity._owners is self.ledger
        for record in self.records:
            if record.completed:
                continue
            manager = record.manager()
            if manager is None:
                assert (not record.worker_root and record.admitted
                        and record.ordinary_removal_proved), "unresolved dead manager without removal proof"
                token_absent = True  # Prior actual-token check, never an ID lookup.
            else:
                self._manager_identity(record, manager)
                token = manager._token
                if token in self.ledger:
                    assert self.ledger[token] == record.amount
                    record.worker_root = bool(manager._failed)
                    continue
                if not record.admitted:
                    continue
                token_absent = token not in self.ledger
                if record.worker_root:
                    assert retirement is not None and retirement.manager is manager
                else:
                    assert record.ordinary_removal_proved
                    assert manager._exited and token.retained is None
                    assert not manager._failed and manager._draining is False
            # Usage is diagnostic only. Neither usage nor integer IDs establish
            # removal; the live token check or prior ordinary proof does that.
            usage = self.used
            snapshot = dict(ledger_same=self.capacity._owners is self.ledger,
                token_absent=token_absent, backing_dead=not self.payloads.live(),
                usage=usage, remaining_usage=usage, worker_root=record.worker_root,
                retained_clear=manager is not None and manager._token.retained is None,
                exited=manager is not None and manager._exited,
                failed_empty=manager is not None and not manager._failed,
                not_draining=manager is not None and manager._draining is False,
                done=bool(retirement is not None and retirement.done))
            # Ordinary body proof concerns its own hold, not a live native root.
            if not record.worker_root:
                snapshot["backing_dead"] = record.boundary_ok and record.ordinary_removal_proved
                assert [row[0] for row in record.holds] == ["entered", "leaving"]
            _validate_refund(snapshot)
            assert record.boundary_ok
            record.completed = True
            self.actual_refund_observed.append((record.token_id, record.amount))


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

    capacity = resources._ClientCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))
    hold = capacity.hold_bytes
    observations = _RefundObservations(capacity, payloads, monkeypatch)
    _LIFETIME_FAILED_LEDGERS.add(capacity)

    @contextmanager
    def reservation(amount, *, _token):
        with hold(amount, _token=_token) as observation:
            observations.hold_snapshot(_token, amount, "entered")
            yield observation
            observations.hold_snapshot(_token, amount, "leaving")

    monkeypatch.setattr(capacity, "hold_bytes", reservation)

    def arm():
        if owner.retirement is None:
            retirement = worker._WorkerRetirement(observations.watch(resources.reserve_incremental_memory(64 << 20)),
                                                   asyncio.get_running_loop(), owner.lock)
            owner.retirement = retirement
            retirement.root = owner.reservation = worker._Reservation(retirement)
            retirement.native = owner.active

    async def operation_endpoint(request):
        arm()
        try:
            return await operation(request)
        finally:
            observe_refunds()

    def observe_refunds():
        observations.observe_refunds(owner.retirement)
        released[:] = [[] for _ in observations.actual_refund_observed]

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
    try:
        yield SimpleNamespace(app=app, owner=owner, operation=operation_endpoint, payloads=payloads,
            retained=retained, released=released, calls=calls, arm=arm, capacity=capacity,
            observations=observations, observe_refunds=observe_refunds)
    finally:
        _forget_empty_ledger(capacity)


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
        h.observe_refunds()
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
        assert not h.released and h.owner.pending_retirement
        await h.owner._drain_retirement()
        h.observe_refunds()
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
        h.observe_refunds()
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
        assert h.owner.pending_retirement and not h.released
        await h.owner._drain_retirement()
        h.observe_refunds()
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


@pytest.mark.parametrize("keep_scratch", [False, True])
@pytest.mark.parametrize("backend_failure", [False, True])
@pytest.mark.parametrize("physical_close", [False, True])
def test_native_abort_failure_keeps_owner_and_full_root_charge(
        harness, monkeypatch, keep_scratch, backend_failure, physical_close):
    h = harness
    h.owner.active.payload = h.payloads.buffer(b"native owner", "native-owner")
    failure = ValueError("controlled abort failure")
    scratch = SimpleNamespace(payload=None)
    scratch_refs = []

    def abort(self):
        h.calls.append("abort-failed")
        held_owner = self  # Shell only, never a backing alias in the traceback.
        scratch.payload = h.payloads.buffer(b"temporary abort scratch", "abort-scratch")
        scratch_refs.append(weakref.ref(scratch.payload))
        try:
            assert held_owner.payload
            raise failure
        finally:
            if not keep_scratch:
                scratch.payload = None  # Explicit retirement, NOT frame clearing.

    async def run():
        h.arm()
        native = h.owner.active
        original = h.owner.retirement
        manager, root = original.manager, original.root
        token, capacity, amount = manager._token, manager._capacity, manager._amount
        ledger, custody, backend = capacity._owners, manager._failed, manager._manager
        concurrent = resources._ClientCapacity.hold_bytes(capacity, 12345)
        concurrent.__enter__()
        monkeypatch.setattr(type(native), "abort", abort)
        with pytest.raises(ValueError) as caught:
            h.owner.retire()
        h.observe_refunds()
        assert caught.value is failure and h.retained == []
        trace = failure.__traceback__
        while trace is not None and trace.tb_frame.f_code is not abort.__code__:
            trace = trace.tb_next
        assert trace is not None and trace.tb_frame.f_locals["held_owner"] is native
        original_trace = failure.__traceback__
        retirement = h.owner.retirement
        assert retirement is original and retirement.manager is manager
        assert retirement.root is root and manager._token is token
        assert manager._capacity is capacity and manager._amount == amount
        assert h.owner.active is None and retirement.native is native
        assert h.owner.pending_retirement and not h.released
        assert capacity._owners is ledger and ledger[token] == amount and token.retained is manager
        assert sum(ledger.values()) == amount + 12345
        assert manager._failed is custody and manager._manager is backend
        assert (scratch_refs[0]() is not None) is keep_scratch
        assert h.payloads.live() == (["native-owner", "abort-scratch"] if keep_scratch else ["native-owner"])
        # Ordinary shutdown calls retire(), NOT retry=True. It must leave failure
        # visible without a second native abort or backend attempt.
        h.owner.retire()
        h.observe_refunds()
        assert h.calls == ["abort-failed"] and h.owner.pending_retirement
        assert not h.observations.backend_exit and not h.released
        with pytest.raises(RuntimeError, match="retirement incomplete"):
            async with h.app.router.lifespan_context(h.app):
                pass
        h.observe_refunds()
        assert h.calls == ["abort-failed"] and h.owner.pending_retirement
        assert not h.observations.backend_exit and not h.released

        def successful_abort(self):
            h.calls.append("abort-success")
            self.payload = None
            scratch.payload = None

        monkeypatch.setattr(type(native), "abort", successful_abort)
        h.observations.fail_backend = backend_failure
        h.observations.physical_close = physical_close
        with pytest.raises(RuntimeError, match="incomplete or ambiguous"):
            await h.owner._drain_retirement()
        h.observe_refunds()
        assert (scratch_refs[0]() is not None) is keep_scratch
        assert h.payloads.live() == (["native-owner", "abort-scratch"] if keep_scratch else ["native-owner"])
        assert caught.value is failure and failure.__traceback__ is original_trace
        assert trace.tb_frame.f_locals["held_owner"] is native and native.payload is not None
        assert h.retained == [] and h.calls == ["abort-failed"]
        assert retirement is original and retirement.manager is manager
        assert retirement.root is root and manager._token is token
        assert manager._capacity is capacity and manager._amount == amount
        attempts = len(h.observations.backend_exit)
        assert not h.released and ledger[token] == amount and token.retained is manager
        assert manager._failed is custody and manager._manager is backend
        assert not manager._exited and not manager._draining and not retirement.done
        assert sum(ledger.values()) == amount + 12345
        with pytest.raises(RuntimeError, match="ambiguous"):
            await h.owner._drain_retirement()
        h.observe_refunds()
        assert len(h.observations.backend_exit) == attempts
        assert h.calls == ["abort-failed"]
        assert len(h.observations.actual_refund_observed) == 0
        concurrent.__exit__(None, None, None)
        assert sum(ledger.values()) == amount

    asyncio.run(run())


def test_reply_limit_rollback_and_root_close_are_idempotent(harness):
    h = harness

    async def run():
        h.arm()
        root = h.owner.reservation
        with pytest.raises(ValueError, match="bounded owner"):
            worker._PayloadResponse(b"too long", root, limit=1)
        assert not h.released
        h.owner.retire()
        h.observe_refunds()
        root.close()
        with pytest.raises(ValueError, match="root is retired"):
            root.lease()
        assert h.released == [[]]

    asyncio.run(run())


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
    ledger = {"limit": root_bytes + body_bytes}

    capacity = resources._ClientCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(
        headroom_bytes=ledger["limit"] - sum(capacity._owners.values())))
    hold = capacity.hold_bytes
    observations = _RefundObservations(capacity, payloads, monkeypatch)
    _LIFETIME_FAILED_LEDGERS.add(capacity)

    @contextmanager
    def charged(size, *, _token):
        with hold(size, _token=_token) as observation:
            observations.hold_snapshot(_token, size, "entered")
            yield observation
            observations.hold_snapshot(_token, size, "leaving")

    monkeypatch.setattr(capacity, "hold_bytes", charged)
    monkeypatch.setattr(worker, "reserve_incremental_memory",
                        lambda size: observations.watch(resources.reserve_incremental_memory(size)))

    def observe_refunds():
        observations.observe_refunds(owner.retirement)
        refunds[:] = [amount for _, amount in observations.actual_refund_observed]

    async def observed_endpoint(request):
        try:
            return await endpoint(request)
        finally:
            observe_refunds()

    class Native:
        def __init__(self, rid):
            self.rid = rid
            self.payload = payloads.buffer(b"native owner", "native-owner")

        def request_id(self):
            return self.rid

        def abort(self):
            self.payload = None

    def load(*args, worker):
        assert observations.used == root_bytes + body_bytes
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

    try:
        yield SimpleNamespace(app=app, owner=owner, source=source, endpoint=observed_endpoint,
            request=request, payloads=payloads, ledger=ledger, capacity=capacity,
            observations=observations, observe_refunds=observe_refunds, loads=loads,
            refunds=refunds, retained=retained, root_bytes=root_bytes, body_bytes=body_bytes)
    finally:
        _forget_empty_ledger(capacity)


def test_pending_admission_reply_keeps_host_claim_before_another_source_import(admission):
    h = admission

    async def run():
        first = await h.endpoint(h.request())
        assert first.status_code == 200 and len(h.loads) == 1
        h.owner.retire()
        h.observe_refunds()
        assert h.owner.reservation is None and h.observations.used == h.root_bytes
        assert h.refunds == [h.body_bytes]
        blocked = await h.endpoint(h.request("b" * 64))
        assert blocked.status_code == 409 and len(h.loads) == 1
        assert not h.retained
        assert h.observations.used == h.root_bytes
        del first
        assert h.observations.used == h.root_bytes and h.owner.pending_retirement
        await h.owner._drain_retirement()
        h.observe_refunds()
        assert h.observations.used == 0 and not h.payloads.live()
        accepted = await h.endpoint(h.request("e" * 64))
        assert accepted.status_code == 200 and len(h.loads) == 2
        accepted.close()
        h.observe_refunds()
        assert h.observations.used == h.root_bytes
        h.owner.retire()
        h.observe_refunds()
        assert h.observations.used == 0 and not h.payloads.live()
        assert h.refunds.count(h.root_bytes) == 2

    asyncio.run(run())


@pytest.mark.parametrize("terminal", ["finish", "cancel"])
def test_http_second_admission_after_terminal_send(admission, monkeypatch, terminal):
    h = admission
    first_raw = asyncio.run(h.request().body())
    second_raw = asyncio.run(h.request("b" * 64).body())
    load = h.source.load_transport
    original = None

    def checked_load(*args, **kwargs):
        if original is not None:
            manager = original.manager
            assert original.done and manager._token not in h.capacity._owners
            assert manager._token.retained is None and manager._exited
            assert manager._failed == ()
            h.observations.observe_refunds(original)
        native = load(*args, **kwargs)
        native.finish = lambda: b'{"completion":true}'
        return native

    monkeypatch.setattr(h.source, "load_transport", checked_load)
    with TestClient(h.app, headers={"authorization": "Bearer " + "k" * 32}) as client:
        assert client.post("/v1/delegation/admit", content=first_raw).status_code == 200
        original = h.owner.retirement
        token = original.manager._token
        assert client.post(f"/v1/delegation/{'a' * 64}/{terminal}", content=b"").status_code == 200
        assert original.exit_seen and original.root._references == 0
        assert token in h.capacity._owners and not original.done
        assert client.post("/v1/delegation/admit", content=second_raw).status_code == 200
        assert original.done and token not in h.capacity._owners
        assert len(h.loads) == 2 and h.owner.retirement is not original
        assert sum(amount == h.root_bytes for _, amount in
                   h.observations.actual_refund_observed) == 1
    h.observe_refunds()
    assert not h.capacity._owners


def test_http_second_admission_after_expiry_reply_release(admission, monkeypatch):
    h = admission

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        original = None
        load = h.source.load_transport

        def checked_load(*args, **kwargs):
            if original is not None:
                manager = original.manager
                assert original.done and manager._token not in h.capacity._owners
                assert manager._token.retained is None and manager._exited
                h.observations.observe_refunds(original)
            return load(*args, **kwargs)

        monkeypatch.setattr(h.source, "load_transport", checked_load)

        async def http(rid, *, withheld=False):
            raw = await h.request(rid).body()
            delivered = False
            statuses = []

            async def receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": raw, "more_body": False}
                await asyncio.Future()

            async def send(message):
                if message["type"] == "http.response.start":
                    statuses.append(message["status"])
                elif withheld:
                    entered.set()
                    await release.wait()

            scope = {"type": "http", "http_version": "1.1", "method": "POST",
                "scheme": "http", "path": "/v1/delegation/admit", "root_path": "",
                "query_string": b"", "server": ("testserver", 80), "client": ("testclient", 1),
                "headers": [(b"content-length", str(len(raw)).encode()),
                            (b"authorization", b"Bearer " + b"k" * 32)]}
            await h.app(scope, receive, send)
            return statuses

        async with h.app.router.lifespan_context(h.app):
            task = asyncio.create_task(http("a" * 64, withheld=True))
            try:
                await entered.wait()
                original = h.owner.retirement
                token = original.manager._token
                h.owner.deadline = 0
                assert await http("b" * 64) == [409]
                assert original.abort_ok and original.root_closed and not original.exit_seen
                assert original.root._references == 1 and token in h.capacity._owners
                assert len(h.loads) == 1 and not original.done
                release.set()
                assert await task == [200]
                assert original.exit_seen and original.root._references == 0
                assert token in h.capacity._owners and not original.done
                assert await http("b" * 64) == [200]
                assert original.done and token not in h.capacity._owners
                assert len(h.loads) == 2
            finally:
                release.set()
                await task
        h.observe_refunds()
        assert not h.capacity._owners

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
        assert h.observations.used == 0 and not h.payloads.live()
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
                assert h.observations.used == h.root_bytes
        if outcome != "accepted":
            assert h.retained and h.owner.active is None
            h.observe_refunds()
            assert h.observations.used == 0
    h.observe_refunds()
    assert h.owner.active is None and h.owner.reservation is None
    assert h.observations.used == 0 and not h.payloads.live()
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
        h.observe_refunds()
        assert response.status_code == status
        assert bool(reads) == (status == 400)
        assert not h.loads and h.owner.seen == set()
        assert h.observations.used == 0
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
        h.observe_refunds()
        assert (await h.endpoint(h.request())).status_code == 400
        h.observe_refunds()
        assert len(h.loads) == 1 and h.owner.seen == {"a" * 64}
        assert h.observations.used == 0

    asyncio.run(run())


@pytest.mark.parametrize("live_backing,stale_terminal", [(True, False), (False, True)])
def test_refund_validator_rejects_early_or_nonterminal_removal(live_backing, stale_terminal):
    snapshot = dict(ledger_same=True, token_absent=True, backing_dead=not live_backing,
        usage=123, remaining_usage=123, worker_root=True, retained_clear=not stale_terminal,
        exited=not stale_terminal, failed_empty=not stale_terminal,
        not_draining=True, done=not stale_terminal)
    with pytest.raises(AssertionError):
        _validate_refund(snapshot)
    snapshot.update(backing_dead=True, retained_clear=True, exited=True, failed_empty=True, done=True)
    _validate_refund(snapshot)


@pytest.mark.parametrize("omit_retirement", [False, True])
def test_failed_abort_scratch_cannot_authorize_retry_or_backend(harness, monkeypatch, omit_retirement):
    h = harness
    failure = ValueError("scratch first failure")
    scratch = SimpleNamespace(payload=None)
    refs = []

    def abort(self):
        held_owner = self
        scratch.payload = h.payloads.buffer(b"scratch", "abort-scratch")
        refs.append(weakref.ref(scratch.payload))
        assert held_owner is self
        raise failure

    async def run():
        h.arm()
        native, retirement = h.owner.active, h.owner.retirement
        manager = retirement.manager
        token, ledger, custody = manager._token, h.capacity._owners, manager._failed
        monkeypatch.setattr(type(native), "abort", abort)
        with pytest.raises(ValueError) as caught:
            h.owner.retire()
        h.observe_refunds()
        assert caught.value is failure and h.retained == [] and refs[0]() is not None
        assert ledger[token] == manager._amount and token.retained is manager

        def retry(self):
            if not omit_retirement:
                scratch.payload = None

        monkeypatch.setattr(type(native), "abort", retry)
        # Neither replacement abort behavior nor scratch state authorizes retry.
        with pytest.raises(RuntimeError, match="incomplete or ambiguous"):
            await h.owner._drain_retirement()
        h.observe_refunds()
        assert not h.released and refs[0]() is not None
        assert ledger[token] == manager._amount and token.retained is manager
        assert manager._failed is custody and not manager._exited
        assert manager._draining is False and not retirement.done
        assert not h.observations.backend_exit
        assert failure.__traceback__ is not None and caught.value is failure

    asyncio.run(run())


@pytest.mark.parametrize("fixture_factory", [harness, admission])
@pytest.mark.parametrize("charged", [False, True])
def test_lifetime_failed_ledger_teardown_preserves_nonempty_custody(monkeypatch, charged, fixture_factory):
    fixture = fixture_factory.__wrapped__(monkeypatch)
    h = next(fixture)
    capacity = h.capacity
    assert capacity in _LIFETIME_FAILED_LEDGERS

    async def run():
        if charged:
            if fixture_factory is harness:
                h.arm()
            else:
                response = await h.endpoint(h.request())
                response.close()
            failure = ValueError("keep original custody")

            def abort(self):
                raise failure

            monkeypatch.setattr(type(h.owner.active), "abort", abort)
            with pytest.raises(ValueError):
                h.owner.retire()
            h.observe_refunds()

    asyncio.run(run())
    ledger = capacity._owners
    before = [(id(token), amount, id(token.retained)) for token, amount in ledger.items()]
    refunds_before = list(h.observations.actual_refund_observed)
    backend_before = list(h.observations.backend_exit)
    with pytest.raises(StopIteration):
        next(fixture)
    assert capacity._owners is ledger and type(ledger) is dict
    assert before == [(id(token), amount, id(token.retained)) for token, amount in ledger.items()]
    assert (capacity in _LIFETIME_FAILED_LEDGERS) is charged
    assert h.observations.actual_refund_observed == refunds_before
    assert h.observations.backend_exit == backend_before


def test_original_user_payload_is_not_cleanup_scratch(harness, monkeypatch):
    h = harness
    user_payloads = _Payloads()  # Deliberately NOT the cleanup-backing tracker.
    failure = ValueError("original user failure")

    def user_frame():
        held_user = user_payloads.buffer(b"user-owned payload", "original-user")
        assert held_user
        raise failure

    def abort(self):
        held_owner = self
        assert held_owner is self
        user_frame()

    async def run():
        h.arm()
        native, retirement = h.owner.active, h.owner.retirement
        monkeypatch.setattr(type(native), "abort", abort)
        with pytest.raises(ValueError) as caught:
            h.owner.retire()
        h.observe_refunds()
        assert caught.value is failure and h.retained == []
        traceback = failure.__traceback__
        assert user_payloads.live() == ["original-user"]
        assert worker._retirement_error(failure, RuntimeError("secondary"), "shutdown") is failure
        assert failure.__traceback__ is traceback
        assert user_payloads.live() == ["original-user"] and not h.payloads.live()
        monkeypatch.setattr(type(native), "abort", lambda self: None)
        with pytest.raises(RuntimeError, match="incomplete or ambiguous"):
            await h.owner._drain_retirement()
        h.observe_refunds()
        assert not retirement.done and not h.released
        manager = retirement.manager
        assert h.capacity._owners[manager._token] == manager._amount
        assert manager._token.retained is manager and manager._failed is retirement.custody
        assert failure.__traceback__ is traceback and user_payloads.live() == ["original-user"]
        frame = traceback
        while frame is not None and frame.tb_frame.f_code is not user_frame.__code__:
            frame = frame.tb_next
        assert frame is not None
        assert frame.tb_frame.f_locals["held_user"] is user_payloads.buffers[0][1]()

    asyncio.run(run())


def test_sequential_ordinary_reservations_ignore_historical_token_ids(harness):
    h = harness
    observations = h.observations
    amount = 64 << 10
    first = observations.watch(resources.reserve_incremental_memory(amount))
    first.__enter__()
    first.__exit__(None, None, None)
    h.observe_refunds()
    first_record = observations.records[-1]
    assert first_record.completed and first_record.ordinary_removal_proved
    assert [row[0] for row in first_record.holds] == ["entered", "leaving"]
    first_ref = weakref.ref(first)
    del first
    assert first_ref() is None and first_record.backend() is None

    second = observations.watch(resources.reserve_incremental_memory(amount))
    second_record = observations.records[-1]
    # Deterministic collision in DIAGNOSTICS ONLY. No token/ledger/id patching.
    first_record.token_id = id(second._token)
    first_state = {**vars(first_record), "holds": list(first_record.holds)}
    second.__enter__()
    assert h.capacity._owners[second._token] == amount
    assert [row[0] for row in second_record.holds] == ["entered"]
    second.__exit__(None, None, None)
    assert second._token not in h.capacity._owners
    assert second_record.ordinary_removal_proved and second_record.boundary_ok
    assert not second_record.completed
    assert [row[0] for row in second_record.holds] == ["entered", "leaving"]
    assert second_record.holds == [("entered", amount, amount, amount),
                                   ("leaving", amount, amount, amount)]
    assert len(observations.backend_exit) == 2
    assert observations.backend_exit[-1] == (id(second._token), "before",
                                            (id(second._token), amount, amount))
    h.observe_refunds()
    assert second_record.completed and len(observations.actual_refund_observed) == 2
    assert observations.actual_refund_observed[-1] == (id(second._token), amount)
    assert vars(first_record) == first_state
    h.observe_refunds()
    assert len(observations.actual_refund_observed) == 2 and not h.capacity._owners


def test_dead_ordinary_manager_without_removal_proof_is_unresolved(harness):
    h = harness
    manager = h.observations.watch(resources.reserve_incremental_memory(64 << 10))
    record = h.observations.records[-1]
    manager_ref = weakref.ref(manager)
    del manager  # Never entered; no actual removal proof exists to observe.
    assert manager_ref() is None and record.backend() is None
    with pytest.raises(AssertionError, match="unresolved dead manager without removal proof"):
        h.observe_refunds()
    assert not record.completed and not record.ordinary_removal_proved
    assert not h.capacity._owners and not h.observations.actual_refund_observed
    assert not h.observations.backend_exit
