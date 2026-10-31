"""Source-only public caller custody tests, not arithmetic acceptance proof.

Native loading, peers and verified-window arithmetic are fake boundaries. The
reservation, original token, ledger and explicit drain are real. These generic
recoverable owners do NOT model recovery of sticky real shared transport owners.
Pre-assignment importer/partial-native/private-constructor custody is NOT PROVEN.
"""

import gc
import json
from types import SimpleNamespace as NS
import weakref

import pytest

from pllm.deployment import resources
from pllm.runtime import delegated_transport as transport


def _detach(*errors):
    """Remove all propagated traceback/cause/context roots, including cycles."""
    pending, seen = list(errors), set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        pending.extend((error.__cause__, error.__context__))
        error.__traceback__ = error.__cause__ = error.__context__ = None


@pytest.fixture
def custody(monkeypatch):
    capacity = resources._ClientCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    state = NS(capacity=capacity, amount=1 << 30, refs={}, tokens=[], events=[],
               errors={}, failed=set(), handoff=None, preassignment=None,
               sentinel=object(), snapshots=[], handoffs=[], window_hook=None,
               native_failures_remaining=None)
    monkeypatch.setattr(capacity, "observe", lambda: NS(
        headroom_bytes=state.amount - sum(capacity._owners.values())))

    def memory(amount):
        state.amount = amount
        reservation = resources.reserve_incremental_memory(amount)
        assert isinstance(reservation, resources._MemoryReservation)
        state.refs["reservation"] = weakref.ref(reservation)
        state.tokens.append(id(reservation._token))
        return reservation

    def fail(stage):
        if stage in state.failed:
            raise state.errors[stage]

    class Native:
        def __init__(self):
            self._fake_abort_attempts = 0
            self._fake_abort_succeeded = False
            state.refs["native"] = weakref.ref(self)

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
            assert not capacity.lock._is_owned()
            self._fake_abort_attempts += 1
            state.events.append("native.abort")
            if state.native_failures_remaining is not None:
                if state.native_failures_remaining > 0:
                    state.native_failures_remaining -= 1
                    fail("native")
            else:
                fail("native")
            self._fake_abort_succeeded = True

    class Peers:
        def __init__(self, *args, **options):
            if state.preassignment == "peers":
                # Resource-free shell allocation fails before publication.
                raise state.errors["construction"]
            self._fake_close_attempts = 0
            self._fake_close_succeeded = False
            state.refs["peers"] = weakref.ref(self)

        def start(self, *args, **options):
            pass

        def admit(self, *args, **options):
            pass

        def close(self):
            assert not capacity.lock._is_owned()
            self._fake_close_attempts += 1
            state.events.append("peers.close")
            fail("peer")
            self._fake_close_succeeded = True

        def audit(self):
            return {"fixture": 1}

    class Cancelled:
        def __call__(self):
            return False

        def record_audit(self, snapshot):
            state.events.append("audit")
            assert state.events.index("native.abort") < state.events.index("audit")
            assert state.events.index("peers.close") < state.events.index("audit")
            assert sum(capacity._owners.values()) == state.amount
            state.snapshots.append(snapshot)
            fail("audit")

    class Handoff:
        def _retain_failed_request_reservation(self, cancelled, reservation):
            token = reservation._token
            assert id(token) == state.tokens[0]
            assert token.retained is reservation
            assert capacity._owners == {token: state.amount}
            state.handoffs.append(state.handoff)
            if state.handoff == "after":
                self.reservation = reservation
            if state.handoff in {"before", "after"}:
                raise state.errors["handoff"]
            self.reservation = reservation

    contract = {"input_rows": 1, "decode_steps": 0,
                "vocabulary": 2, "matrices": [{}]}

    def load(*args, **options):
        if state.preassignment == "load":
            raise state.errors["construction"]
        return Native()

    source = NS(capacity="capsule", contract=contract,
                execution_payload_estimate_bytes=1024,
                runtime_contract=lambda *args: b"bound", load_transport=load)

    def window(native, admitted, token_ids, *, cancelled=None, decode_tokens=None):
        assert admitted == contract
        state.events.append("operation")
        if state.window_hook is not None:
            return state.window_hook(native)
        fail("operation")
        return state.sentinel

    def execute():
        return transport.execute_request(None, source, (0,), 1, {
            "worker_a": ("http://127.0.0.1:23001", "a" * 32),
            "worker_b": ("http://127.0.0.1:23002", "b" * 32),
        }, 5.0, state.cancelled)

    state.cancelled = Cancelled()
    state.make_handoff = Handoff
    state.execute = execute
    monkeypatch.setattr(transport, "reserve_incremental_memory", memory)
    monkeypatch.setattr(transport, "_Peers", Peers)
    monkeypatch.setattr(transport, "_execute_verified_window", window)
    try:
        yield state
    finally:
        # Teardown-only generic fake recovery, never real sticky-owner policy.
        # No resource collection is held across yield or the tests' GC checks.
        failures = []
        state.failed.clear()
        state.native_failures_remaining = 0
        state.window_hook = None
        # Recover originals even if an assertion interrupted handoff capture.
        # Token custody, not the optional client handoff, locates each charge.
        reservations = []
        for token in tuple(capacity._owners):
            if token.retained is not None:
                reservations.append(token.retained)
        reservation = state.refs.get("reservation", lambda: None)()
        if reservation is not None and all(
                reservation is not retained for retained in reservations):
            reservations.append(reservation)
        for reservation in reservations:
            try:
                reservation.drain_failed()
            except BaseException as error:
                failures.append("reservation drain: " + type(error).__name__)
        # Owned adapters get the first and only recovery attempt above. Never
        # bypass a still-failed adapter with a direct underlying-native abort.
        protected = set()
        for token in tuple(capacity._owners):
            retained = token.retained
            if retained is not None:
                for owner in retained._failed:
                    if owner is state.refs.get("peers", lambda: None)():
                        protected.add("peers")
                    else:
                        protected.add("native")
        for name, method, success in (
                ("native", "abort", "_fake_abort_succeeded"),
                ("peers", "close", "_fake_close_succeeded")):
            owner = state.refs.get(name, lambda: None)()
            if owner is not None and name not in protected and not getattr(owner, success):
                try:
                    getattr(owner, method)()
                except BaseException as error:
                    failures.append(name + " cleanup: " + type(error).__name__)
            del owner
        # No manual ledger clearing, token replacement or success flag writes.
        # Pytest reports teardown failures separately from the original failure.
        if resources._capacity is not capacity:
            failures.append("original capacity no longer installed")
        if capacity._owners:
            failures.append("original capacity retains " + str(len(capacity._owners))
                            + " tokens / " + str(sum(capacity._owners.values())) + " bytes")
        assert not failures, "fake custody teardown failures: " + "; ".join(failures)


def _errors(state, failed, hard=False):
    base = BaseException if hard else Exception
    state.failed = set(failed)
    for stage in (*failed, "handoff", "construction"):
        error_type = type(stage.title() + "Failure", (base,), {})
        state.errors[stage] = error_type("PRIVATE_PAYLOAD_" + stage)


def _charged(state, count):
    reservation = state.refs["reservation"]()
    assert reservation is not None
    token = reservation._token
    assert id(token) == state.tokens[0]
    assert token.retained is reservation
    assert state.capacity._owners == {token: state.amount}
    assert sum(state.capacity._owners.values()) == state.amount
    assert reservation.pending_retirement
    assert type(reservation._failed) is tuple and len(reservation._failed) == count
    assert all(callable(owner.close) for owner in reservation._failed)
    return reservation


def _forget_errors_and_callers(state, error):
    _detach(error, *state.errors.values())
    state.errors.clear()
    state.cancelled = lambda: False  # Discard optional partial handoff roots too.


@pytest.mark.parametrize("hard", [False, True], ids=["exception", "baseexception"])
@pytest.mark.parametrize("failed", [
    ("operation",), ("native",), ("peer",), ("audit",),
    ("native", "peer", "audit"),
    ("operation", "native", "peer", "audit"),
])
def test_primary_identity_secondary_notes_and_original_token_custody(custody, failed, hard):
    state = custody
    _errors(state, failed, hard)
    primary = state.errors[failed[0]]
    cause, context = LookupError("original cause"), ValueError("original context")
    primary.__cause__, primary.__context__ = cause, context
    with pytest.raises(type(primary)) as caught:
        state.execute()
    assert caught.value is primary
    assert primary.__cause__ is cause and primary.__context__ is context
    assert state.events.count("native.abort") == state.events.count("peers.close") == 1
    assert state.events[-1] == "audit"
    secondary = [type(state.errors[name]).__name__ for name in failed[1:]]
    notes = getattr(primary, "__notes__", [])
    assert len(notes) == len(secondary)
    assert all(name in note and "PRIVATE_PAYLOAD" not in note and len(note) <= 256
               for name, note in zip(secondary, notes, strict=True))
    failed_owners = tuple(name for name in ("native", "peers")
                          if ("peer" if name == "peers" else name) in failed)
    if failed_owners:
        reservation = _charged(state, len(failed_owners))
        owners = reservation._failed
        if "peers" in failed_owners:
            assert owners[-1] is state.refs["peers"]()
        # Native wrapper identity is intentionally private. Its public close
        # behavior below must reach the exact weak-referenced native once.
        del owners, reservation
    _forget_errors_and_callers(state, primary)
    del primary, caught, cause, context
    gc.collect()
    assert all((state.refs[name]() is not None) == (name in failed_owners)
               for name in ("native", "peers"))
    if not failed_owners:
        assert not state.capacity._owners
        assert state.refs["reservation"]() is None
        return
    reservation = _charged(state, len(failed_owners))
    with pytest.raises(resources.NetworkError, match="CLIENT_MEMORY_CAPACITY_CHANGED"):
        with resources.reserve_incremental_memory(state.amount):
            pytest.fail("failed retirement must block follow-on admission")
    # A failed drain must retain the exact tuple in original order and charge.
    before = reservation._failed
    _errors(state, tuple("peer" if name == "peers" else name for name in failed_owners), hard)
    with pytest.raises(BaseException) as caught:
        reservation.drain_failed()
    assert caught.value is state.errors["native" if "native" in failed_owners else "peer"]
    assert reservation._failed == before
    _forget_errors_and_callers(state, caught.value)
    del caught, before, reservation
    gc.collect()
    reservation = _charged(state, len(failed_owners))
    state.failed.clear()  # Generic fake recovery only, NOT real shared recovery.
    reservation.drain_failed()
    assert reservation._failed == () and reservation._token.retained is None
    assert not reservation.pending_retirement and not state.capacity._owners
    expected_native_calls = 3 if "native" in failed_owners else 1
    assert state.events.count("native.abort") == expected_native_calls
    assert state.events.count("peers.close") == (3 if "peers" in failed_owners else 1)
    events = list(state.events)
    reservation.drain_failed()
    assert state.events == events and not state.capacity._owners


@pytest.mark.parametrize("handoff", ["before", "after"])
@pytest.mark.parametrize("hard", [False, True], ids=["exception", "baseexception"])
def test_handoff_failure_cannot_replace_primary_or_remove_token_root(custody, handoff, hard):
    state = custody
    _errors(state, ("operation", "native", "peer"), hard)
    state.handoff = handoff
    state.cancelled.owner = state.make_handoff()
    primary = state.errors["operation"]
    with pytest.raises(type(primary)) as caught:
        state.execute()
    assert caught.value is primary
    assert state.handoffs == [handoff]
    notes = getattr(primary, "__notes__", [])
    assert len(notes) == 3
    assert all(name in note and "PRIVATE_PAYLOAD" not in note and len(note) <= 256
               for name, note in zip(("HandoffFailure", "NativeFailure", "PeerFailure"),
                                     notes, strict=True))
    _forget_errors_and_callers(state, primary)
    del caught, primary
    gc.collect()
    reservation = _charged(state, 2)
    assert state.refs["native"]() is not None
    assert reservation._failed[-1] is state.refs["peers"]()
    state.failed.clear()
    reservation.drain_failed()
    assert not state.capacity._owners and reservation._token.retained is None


def test_success_retires_before_audit_and_returns_window_sentinel(custody):
    state = custody
    assert state.execute() is state.sentinel
    assert state.events == ["operation", "native.abort", "peers.close", "audit"]
    assert state.snapshots == [{"fixture": 1}]
    assert not state.capacity._owners
    gc.collect()
    assert all(ref() is None for ref in state.refs.values())


@pytest.mark.parametrize("boundary", ["load", "peers"])
def test_preassignment_only_returned_owners_are_caller_responsibility(custody, boundary):
    """Partial native/importer/private constructor custody NOT PROVEN here."""
    state = custody
    _errors(state, ())
    state.preassignment = boundary
    primary = state.errors["construction"]
    with pytest.raises(type(primary)) as caught:
        state.execute()
    assert caught.value is primary
    assert state.events == ([] if boundary == "load" else ["native.abort"])
    assert "peers" not in state.refs
    assert not state.capacity._owners
    _forget_errors_and_callers(state, primary)
    del caught, primary
    gc.collect()
    assert all(ref() is None for ref in state.refs.values())


@pytest.mark.parametrize("operation_fails,repeats,secondary_failures,drain_fails", [
    pytest.param(True, 3, True, True, id="operation-repeated-secondary-drain-failure"),
    pytest.param(True, 1, False, False, id="operation-single-recover"),
    pytest.param(False, 3, True, False, id="success-repeated-secondary-recover"),
    pytest.param(False, 1, False, True, id="success-single-drain-failure"),
])
def test_window_finally_abort_capture_is_once_until_explicit_drain(
        custody, operation_fails, repeats, secondary_failures, drain_fails):
    """Exercise public adapter boundary, NOT the shared decoder algorithm.

    In particular, an abort that would succeed on its second actual call must
    not get that call from either a repeated callback or outer cleanup.
    """
    state = custody
    failed = (["operation"] if operation_fails else []) + ["native"]
    if secondary_failures:
        failed += ["peer", "audit"]
    _errors(state, failed)
    state.native_failures_remaining = 1
    primary = state.errors["operation" if operation_fails else "native"]
    cause, context = LookupError("original cause"), ValueError("original context")
    primary.__cause__, primary.__context__ = cause, context

    def window_finally(exchange):
        try:
            if operation_fails:
                raise state.errors["operation"]
            return state.sentinel
        finally:
            for _ in range(repeats):
                exchange.abort()

    state.window_hook = window_finally
    with pytest.raises(type(primary)) as caught:
        state.execute()
    assert caught.value is primary
    assert primary.__cause__ is cause and primary.__context__ is context
    assert state.events == ["operation", "native.abort", "peers.close", "audit"]
    secondary = (["NativeFailure"] if operation_fails else [])
    if secondary_failures:
        secondary += ["PeerFailure", "AuditFailure"]
    notes = getattr(primary, "__notes__", [])
    assert len(notes) == len(secondary)
    assert all(name in note and "PRIVATE_PAYLOAD" not in note and len(note) <= 256
               for name, note in zip(secondary, notes, strict=True))
    reservation = _charged(state, 2 if secondary_failures else 1)
    if secondary_failures:
        assert reservation._failed[-1] is state.refs["peers"]()
    del reservation
    _forget_errors_and_callers(state, primary)
    state.window_hook = None
    del primary, caught, cause, context, window_finally
    gc.collect()
    reservation = _charged(state, 2 if secondary_failures else 1)
    assert state.refs["native"]() is not None
    assert (state.refs["peers"]() is not None) == secondary_failures
    assert state.events.count("native.abort") == 1
    with pytest.raises(resources.NetworkError, match="CLIENT_MEMORY_CAPACITY_CHANGED"):
        with resources.reserve_incremental_memory(state.amount):
            pytest.fail("inner abort failure cannot refund the original charge")
    if drain_fails:
        _errors(state, ("native", "peer") if secondary_failures else ("native",))
        state.native_failures_remaining = 1
        owners = reservation._failed
        with pytest.raises(BaseException) as caught:
            reservation.drain_failed()
        assert caught.value is state.errors["native"]
        assert reservation._failed == owners
        assert state.events.count("native.abort") == 2
        _forget_errors_and_callers(state, caught.value)
        del caught, owners, reservation
        gc.collect()
        reservation = _charged(state, 2 if secondary_failures else 1)
    # Only explicit drain retries the fake native. This is generic recovery,
    # never evidence that a failed real shared owner may be recovered.
    state.failed.clear()
    state.native_failures_remaining = 0
    reservation.drain_failed()
    assert state.events.count("native.abort") == (3 if drain_fails else 2)
    assert state.events.count("peers.close") == (
        (3 if drain_fails else 2) if secondary_failures else 1)
    assert not state.capacity._owners
    assert reservation._token.retained is None and reservation._failed == ()
    events = list(state.events)
    reservation.drain_failed()
    assert state.events == events and not state.capacity._owners


def test_successful_repeated_window_finally_abort_fences_outer_cleanup(custody):
    """A successful callback also fences actual abort through outer cleanup."""
    state = custody

    def window_finally(exchange):
        try:
            return state.sentinel
        finally:
            exchange.abort()
            exchange.abort()
            exchange.abort()

    state.window_hook = window_finally
    assert state.execute() is state.sentinel
    assert state.events == ["operation", "native.abort", "peers.close", "audit"]
    assert not state.capacity._owners
    state.window_hook = None
    del window_finally
    gc.collect()
    assert all(ref() is None for ref in state.refs.values())
