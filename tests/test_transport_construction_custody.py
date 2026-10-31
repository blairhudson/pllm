"""UNRUN source-authored regressions, NOT native/thread/loop cleanup evidence.

Every loop, thread, dispatch future, HTTP client and native boundary here is a
fake. Importing production modules can still load native code; execution needs
separate authority. No destructor recovery or synthetic teardown refund.
"""

import gc
import json
import sys
from types import SimpleNamespace as NS
import weakref

import pytest

from pllm.deployment import resources
from pllm.runtime import delegated_transport as shared
from pllm.runtime import fss_transport as fss
from pllm.runtime import replicated_transport as replicated
from pllm.runtime.client import ProtocolError
from pllm.runtime.delegated_client import _Attempt


def _detach(error):
    pending, seen = [error], set()
    while pending:
        value = pending.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        pending.extend((value.__cause__, value.__context__))
        value.__traceback__ = value.__cause__ = value.__context__ = None


def _published(peers):
    """Assert the actual public caller already holds the shell, without rooting it."""
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if frame.f_code is shared.execute_request.__code__:
                assert frame.f_locals["peers"] is peers
                return
            frame = frame.f_back
        pytest.fail("acquisition escaped the actual GKM caller")
    finally:
        del frame


@pytest.fixture
def fake_resources(monkeypatch):
    state = NS(cut=None, cleanup=False, events=[], refs={}, primary=ValueError("construction"),
               entered=False, reentrant=None, native_failure=False)
    real_peers, real_io = shared._Peers, shared._IOOwner

    def check(stage):
        peers = state.refs.get("peers", lambda: None)()
        if peers is not None:
            _published(peers)
            if stage not in {"config"}:
                assert peers.io is state.refs["io"]()
        state.events.append(stage)
        if state.cut == stage:
            raise state.primary

    class Loop:
        def __init__(self):
            check("loop")
            state.refs["loop"] = weakref.ref(self)

        def stop(self):
            state.events.append("stop")

        def call_soon_threadsafe(self, callback):
            callback()

        def close(self):
            state.events.append("loop.close")
            if state.cleanup:
                raise LookupError("secondary loop close")

    class Thread:
        def __init__(self, *, target, name, daemon):
            assert not daemon
            owner = target.__self__
            assert owner.loop is state.refs["loop"]()
            check("thread")
            self.owner = owner
            state.refs["thread"] = weakref.ref(self)

        def start(self):
            assert self.owner.thread is self
            check("start-before-entry")
            # Fake target-entry witness, not an actual launch or runtime claim.
            self.owner._entered = True
            state.entered = True
            check("start-after-entry")
            if state.cut == "start-after-exit":
                self.owner._exited = True  # fake target-finally witness only
                check("start-after-exit")

        def join(self, timeout):
            assert self.owner._entered
            state.events.append("join")

        def is_alive(self):
            # Deliberately false even when startup is ambiguous: not proof.
            return False

    class Future:
        def result(self):
            state.events.append("drain")
            if state.reentrant is not None:
                state.reentrant()

        def done(self):
            return True

    def dispatch(coroutine, loop):
        # Do not execute the production drain. This tests custody/control only.
        coroutine.close()
        return Future()

    class IO(real_io):
        def __init__(self):
            super().__init__()
            state.refs["io"] = weakref.ref(self)

    class Peers(real_peers):
        def __init__(self):
            super().__init__()
            state.refs["peers"] = weakref.ref(self)

        def admit(self, *args):
            check("connect")

    client_count = []

    def client(**options):
        index = len(client_count)
        peers = state.refs["peers"]()
        assert peers.clients[index] is None
        assert all(value is not None for value in peers.clients[:index])
        check(f"client-{index}")
        value = NS(index=index)
        client_count.append(index)
        return value

    real_limits = shared.limits

    def limits(capacity):
        check("config")
        return real_limits(capacity)

    monkeypatch.setattr(shared, "_Peers", Peers)
    monkeypatch.setattr(shared, "_IOOwner", IO)
    monkeypatch.setattr(shared, "_NetworkLoop", Loop)
    monkeypatch.setattr(shared.threading, "Thread", Thread)
    monkeypatch.setattr(shared.asyncio, "run_coroutine_threadsafe", dispatch)
    monkeypatch.setattr(shared.httpx, "AsyncClient", client)
    monkeypatch.setattr(shared, "limits", limits)
    state.io_type, state.peer_type = IO, Peers
    yield state
    # No cleanup retry: unresolved shells remain under their original isolated
    # custodian. Fakes have no physical resources needing emergency disposal.


def _gkm(monkeypatch, state):
    capacity = resources._ClientCapacity()
    amount = 1 << 30
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: NS(
        headroom_bytes=amount - sum(capacity._owners.values())))
    tokens = []

    def reserve(size):
        nonlocal amount
        amount = size
        value = resources.reserve_incremental_memory(size)
        state.refs["reservation"] = weakref.ref(value)
        tokens.append(id(value._token))
        return value

    contract = {"input_rows": 1, "decode_steps": 0, "vocabulary": 2, "matrices": [{}]}

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
            state.events.append("native.abort")
            if state.native_failure:
                raise RuntimeError("secondary native abort")

    source = NS(capacity="capsule", contract=contract,
                execution_payload_estimate_bytes=1024,
                runtime_contract=lambda *args: b"bound",
                load_transport=lambda *args, **options: Native())
    monkeypatch.setattr(shared, "reserve_incremental_memory", reserve)

    def execute():
        return shared.execute_request(None, source, (0,), 1, {
            "worker_a": ("http://127.0.0.1:23001", "a" * 32),
            "worker_b": ("http://127.0.0.1:23002", "b" * 32)}, 5, lambda: False)

    return execute, capacity, tokens


@pytest.mark.parametrize("cut,cleanup,retained", [
    ("config", False, False),
    ("loop", False, True),
    ("thread", False, False),
    ("thread", True, True),
    ("start-before-entry", False, True),
    ("start-after-entry", False, False),
    ("start-after-entry", True, True),
    ("start-after-exit", False, True),
    ("client-0", False, True),
    ("client-0", True, True),
    ("client-1", False, True),
    ("client-1", True, True),
    ("connect", False, False),
    ("connect", True, True),
])
def test_gkm_publishes_shells_at_every_cut_and_keeps_original_custody(
        monkeypatch, fake_resources, cut, cleanup, retained):
    state = fake_resources
    state.cut, state.cleanup = cut, cleanup
    execute, capacity, tokens = _gkm(monkeypatch, state)
    with pytest.raises(ValueError) as caught:
        execute()
    assert caught.value is state.primary
    if retained:
        assert any("GKM cleanup failed:" in note for note in caught.value.__notes__)
    _detach(caught.value)
    del caught, execute
    gc.collect()
    if not retained:
        assert not capacity._owners
        assert state.refs["reservation"]() is None
        assert state.refs["peers"]() is None
        return
    assert len(tokens) == len(capacity._owners) == 1
    token = next(iter(capacity._owners))
    reservation = state.refs["reservation"]()
    assert reservation is token.retained and id(token) == tokens[0]
    assert reservation._token is token
    assert capacity._owners == {token: reservation._amount}
    assert reservation._failed == (state.refs["peers"](),)
    before = list(state.events)
    for _ in range(2):
        with pytest.raises(ProtocolError, match="previously failed") as caught:
            reservation.drain_failed()
        _detach(caught.value)
        del caught
        gc.collect()
        assert capacity._owners == {token: reservation._amount}
        assert token.retained is reservation
        assert state.events == before
    with pytest.raises(resources.NetworkError, match="CLIENT_MEMORY_CAPACITY_CHANGED"):
        with resources.reserve_incremental_memory(reservation._amount):
            pytest.fail("unconfirmed original charge must block admission")


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_startup_interruption_stays_primary_and_unconfirmed_owner_retained(
        monkeypatch, fake_resources, interruption):
    state = fake_resources
    state.cut = "start-before-entry"
    state.primary = interruption("interrupted startup")
    execute, capacity, _ = _gkm(monkeypatch, state)
    with pytest.raises(interruption) as caught:
        execute()
    assert caught.value is state.primary
    _detach(caught.value)
    del caught, execute
    gc.collect()
    assert len(capacity._owners) == 1
    assert "join" not in state.events and "stop" not in state.events
    assert "loop.close" not in state.events


def test_constructor_primary_precedes_native_and_peer_cleanup(monkeypatch, fake_resources):
    state = fake_resources
    state.cut, state.cleanup, state.native_failure = "client-1", True, True
    execute, capacity, _ = _gkm(monkeypatch, state)
    with pytest.raises(ValueError) as caught:
        execute()
    assert caught.value is state.primary
    assert caught.value.__notes__ == ["GKM cleanup failed: RuntimeError", "GKM cleanup failed: LookupError"]
    assert state.events.index("native.abort") < state.events.index("loop.close")
    _detach(caught.value)
    del caught, execute
    gc.collect()
    token = next(iter(capacity._owners))
    assert len(token.retained._failed) == 2


@pytest.mark.parametrize("kind", ["io", "peers"])
def test_resource_free_shell_retires_exactly_once(fake_resources, kind):
    state = fake_resources
    owner = state.io_type() if kind == "io" else state.peer_type()
    close = (lambda: owner.close([])) if kind == "io" else owner.close
    assert not state.events
    close()
    close()
    assert owner._close_succeeded and owner._state == "retired"
    assert not state.events
    with pytest.raises(ProtocolError, match="startup is unavailable"):
        owner.start() if kind == "io" else owner.start({}, 1, lambda: False)


def test_allocated_never_started_thread_is_not_joined(fake_resources):
    state = fake_resources
    owner = state.io_type()
    # Model interruption after thread assignment, before start-ambiguous/store
    # and start(). These are fake resources; no forced close/success witnesses.
    owner.loop = shared._NetworkLoop()
    owner._state = "acquired"
    owner.thread = shared.threading.Thread(target=owner._serve, name="fake", daemon=False)
    owner.close([])
    owner.close([])
    assert owner._state == "retired" and owner._close_succeeded
    assert state.events.count("loop.close") == 1
    assert "join" not in state.events and "stop" not in state.events


@pytest.mark.parametrize("kind", ["io", "peers"])
def test_rejected_close_does_not_poison_admitted_closer(fake_resources, kind):
    state = fake_resources
    owner = state.io_type() if kind == "io" else state.peer_type()
    close = (lambda: owner.close([])) if kind == "io" else owner.close
    assert owner._close_gate.acquire(blocking=False)
    try:
        for _ in range(2):
            with pytest.raises(ProtocolError, match="already active"):
                close()
        assert not owner.closed and not owner._close_succeeded
        assert owner._state == "never-acquired"
    finally:
        owner._close_gate.release()
    close()
    assert owner._close_succeeded


def test_own_thread_rejection_during_partial_start_does_not_admit_close(monkeypatch, fake_resources):
    owner = fake_resources.io_type()
    owner.thread = shared.threading.current_thread()
    with pytest.raises(ProtocolError, match="own thread"):
        owner.close([])
    assert not owner.closed and not owner._closing and not owner._close_succeeded
    owner.thread = None  # fake never-started reference, not a resource or close witness
    owner.close([])


@pytest.mark.parametrize("kind", ["io", "peers"])
def test_own_active_loop_rejection_does_not_admit_close(monkeypatch, fake_resources, kind):
    io = fake_resources.io_type()
    owner = io if kind == "io" else fake_resources.peer_type()
    if kind == "peers":
        owner.io = io
    sentinel = object()  # no loop acquired; model only the own-loop identity cut
    io.loop = sentinel
    with monkeypatch.context() as patch:
        patch.setattr(shared.asyncio, "get_running_loop", lambda: sentinel)
        with pytest.raises(ProtocolError, match="own loop|I/O loop"):
            owner.close([]) if kind == "io" else owner.close()
    assert not owner.closed and not owner._close_succeeded
    assert not io._closing
    io.loop = None  # Remove fake identity, not a physical cleanup witness.
    owner.close([]) if kind == "io" else owner.close()
    assert owner._close_succeeded


def test_reentrant_rejection_leaves_active_fake_drain_successful(monkeypatch, fake_resources):
    state = fake_resources
    state.cut = "connect"

    def reentrant():
        peers, io = state.refs["peers"](), state.refs["io"]()
        for close in (peers.close, lambda: io.close([])):
            with pytest.raises(ProtocolError, match="already active"):
                close()
        assert peers.closed and io._closing
        assert not peers._close_succeeded and not io._close_succeeded
        state.events.append("rejected")

    state.reentrant = reentrant
    execute, capacity, _ = _gkm(monkeypatch, state)
    with pytest.raises(ValueError) as caught:
        execute()
    assert caught.value is state.primary
    assert not capacity._owners
    assert state.events.count("drain") == state.events.count("rejected") == 1
    assert state.events.count("join") == state.events.count("loop.close") == 1
    _detach(caught.value)


def test_replicated_caller_publishes_failed_shell_on_same_token(monkeypatch):
    capacity = resources._ClientCapacity()
    amount = replicated.REQUEST_MEMORY
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: NS(headroom_bytes=amount))
    refs = {}
    primary = ValueError("replicated construction")

    class Peers:
        def __init__(self):
            refs["peers"] = weakref.ref(self)

        def start(self, *args, **options):
            frame = sys._getframe(1)
            try:
                assert frame.f_locals["peers"] is self
            finally:
                del frame
            raise primary

        def close(self):
            raise LookupError("sticky partial peers")

        def audit(self):
            return {}

    monkeypatch.setattr(replicated, "_Peers", Peers)
    with pytest.raises(ValueError) as caught:
        replicated.execute_request(None, None, (0,), 1, {}, 1, lambda: False)
    assert caught.value is primary
    _detach(caught.value)
    del caught
    gc.collect()
    token = next(iter(capacity._owners))
    assert capacity._owners == {token: amount}
    assert token.retained._token is token
    assert token.retained._failed == (refs["peers"](),)


@pytest.mark.parametrize("interruption", [ValueError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_type", [None, LookupError, KeyboardInterrupt, SystemExit])
def test_fss_construction_cleanup_precedence_and_original_shell_custody(
        monkeypatch, interruption, cleanup_type):
    capacity = resources._ClientCapacity()
    amount = fss.client_memory_bytes("capsule")
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: NS(
        headroom_bytes=amount - sum(capacity._owners.values())))
    refs, tokens, closes = {}, [], []
    primary = interruption("FSS construction")
    cleanup = cleanup_type("FSS partial close") if cleanup_type else None
    _FSS_POST_START_FAILED_LEDGERS.add(capacity)

    def reserve(size):
        assert size == amount
        value = resources.reserve_incremental_memory(size)
        refs["reservation"] = weakref.ref(value)
        tokens.append(id(value._token))
        return value

    class Peers(shared._Peers):
        def __init__(self):
            super().__init__()
            refs["peers"] = weakref.ref(self)

        def start(self, *args, **options):
            frame = sys._getframe(1)
            try:
                assert frame.f_locals["peers"] is self
            finally:
                del frame
            raise primary

        def _close_owned(self):
            closes.append("close")
            if cleanup is not None:
                raise cleanup

        def audit(self):
            return {}

    def handoff(attempt, reservation):
        assert reservation._token.retained is reservation
        assert id(reservation._token) == tokens[0]

    class Attempt:
        owner = NS(_retain_failed_request_reservation=handoff)

        def __call__(self):
            return False

    experiment = NS(pipeline=NS(components={"linear": NS(params={})}))
    contract = {"vocabulary": 8, "capacity": "capsule"}
    source = NS(contract=contract, runtime_contract=lambda *args: json.dumps(contract).encode())
    monkeypatch.setattr(fss, "_Peers", Peers)
    monkeypatch.setattr(fss, "reserve_incremental_memory", reserve)
    with pytest.raises(cleanup_type or interruption) as caught:
        fss.execute_request(experiment, source, [1], 1, {
            role: (f"http://127.0.0.1:{7300 + index}", str(index) * 32)
            for index, role in enumerate(fss.ROLES)}, 5, Attempt())
    assert caught.value is (cleanup if cleanup is not None else primary)
    assert closes == ["close"]
    if cleanup is not None:
        assert cleanup.__context__ is primary
    _detach(caught.value)
    del caught, experiment, source
    gc.collect()
    if cleanup is not None:
        token = next(iter(capacity._owners))
        assert id(token) == tokens[0]
        assert capacity._owners == {token: amount}
        assert token.retained is refs["reservation"]()
        assert token.retained._failed == (refs["peers"](),)
        for _ in range(2):
            with pytest.raises(ProtocolError, match="previously failed") as caught:
                token.retained.drain_failed()
            _detach(caught.value)
            del caught
            assert closes == ["close"]
            assert capacity._owners == {token: amount}
    else:
        assert not capacity._owners
        assert refs["reservation"]() is None and refs["peers"]() is None
        _FSS_POST_START_FAILED_LEDGERS.discard(capacity)


# Test-only ledger roots: fake failed custody lasts until test process exit.
# Never root fixture state or owners
# separately from the actual ledger -> token -> reservation -> failed-owner chain.
_FSS_POST_START_FAILED_LEDGERS = set()


@pytest.fixture
def fss_post_start_constructor(monkeypatch):
    """Fake manifest/admit path; sticky failed owners never retried at teardown."""
    capacity = resources._ClientCapacity()
    amount = fss.client_memory_bytes("capsule")
    state = NS(capacity=capacity, amount=amount, refs={}, tokens=[], events=[],
               calls=[], frozen=[], refunds=0, cleanup_failure=False,
               handoff_failure=False, primary=None, cleanup=None, start_failure=False)
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: NS(
        headroom_bytes=amount - sum(capacity._owners.values())))

    class ObservedLedger(dict):
        def pop(self, token, *default):
            charged = token in self
            value = super().pop(token, *default)
            if charged and token not in self:
                state.refunds += 1
            return value

    # Observe actual ordinary/direct-commit ledger removal, not reservation
    # __exit__ dispatch. Do not replace refund semantics with a fixture refund.
    capacity._owners = ObservedLedger(capacity._owners)

    def reserve(size):
        assert size == amount
        reservation = resources._MemoryReservation(capacity, size)
        state.refs["reservation"] = weakref.ref(reservation)
        state.tokens.append(id(reservation._token))
        return reservation

    class Peers(shared._Peers):
        def __init__(self):
            super().__init__()
            state.refs["peers"] = weakref.ref(self)

        def start(self, connections, timeout, cancelled, **options):
            assert options == {"roles": fss.ROLES, "namespace": "fss"}
            if state.start_failure:
                state.events.append("channels.failed")
                raise state.primary
            self.live = [True] * len(fss.ROLES)
            self.deadline = fss.time.monotonic() + timeout
            state.events.append("channels.started")

        def post(self, index, action, payload=b"", **options):
            role = fss.ROLES[index]
            state.calls.append((role, action))
            if action == "manifest":
                return fss.pack({"schema": "pllm.fss.worker.v1", "role": role,
                    "configuration": "a" * 64, "source_lock": "b" * 64,
                    "tokenizer": "c" * 64, "transport": fss.TRANSPORT,
                    "epoch": str(index + 1) * 64})
            assert action == "admit"
            return fss.pack({"manifest": b"native-manifest"})

        def close(self):
            state.events.append("peers.close")
            return super().close()

        def _close_owned(self):
            if state.cleanup_failure:
                raise state.cleanup

        def audit(self):
            return {}

    class Native:
        def __init__(self, contract, request_id, manifests, ids):
            assert state.events == ["channels.started"]
            assert manifests == (b"native-manifest",) * 3
            assert len(request_id) == 32 and ids == [1]
            state.events.append("native.constructor")
            raise state.primary

        def close(self):
            state.events.append("native.close")
            pytest.fail("failed constructor must not publish a native owner")

    def handoff(attempt, reservation):
        peers = state.refs["peers"]()
        assert reservation is state.refs["reservation"]()
        assert id(reservation._token) == state.tokens[0]
        assert reservation._token.retained is reservation
        assert reservation._failed == (peers,)
        assert peers.closed and not peers._close_succeeded
        assert capacity._owners == {reservation._token: amount}
        state.events.append("handoff")
        if state.handoff_failure:
            raise KeyboardInterrupt("handoff detail must not appear in notes")

    owner = NS(runtime_id="compiled_fss_v1", _active=None,
               _retain_failed_request_reservation=handoff,
               _fss_failure_observer=state.frozen.append)
    attempt = _Attempt(owner, {})
    owner._active = attempt
    attempt.record_audit = lambda audit: None
    experiment = NS(configuration_digest=lambda: "a" * 64,
                    pipeline=NS(components={"linear": NS(params={})}))
    contract = {"vocabulary": 8, "capacity": "capsule"}
    source = NS(contract=contract, source_lock_digest="b" * 64,
                tokenizer_digest="c" * 64,
                runtime_contract=lambda *args: json.dumps(contract).encode())
    monkeypatch.setattr(fss, "reserve_incremental_memory", reserve)
    monkeypatch.setattr(fss, "_Peers", Peers)
    monkeypatch.setattr(fss, "_native", NS(FssClientRequest=Native))

    def execute():
        return fss.execute_request(experiment, source, [1], 1, {
            role: (f"http://127.0.0.1:{7300 + index}", str(index) * 32)
            for index, role in enumerate(fss.ROLES)}, 5, attempt)

    state.execute = execute
    # No tested owners are acquired until execute() after yield. Register first,
    # before any test path can acquire them or fail an assertion.
    _FSS_POST_START_FAILED_LEDGERS.add(capacity)
    try:
        yield state
    finally:
        # Only a confirmed empty ledger may lose its durable test-owned root.
        # Nonempty/unknown custody stays intact even after assertion failures;
        # no close, drain, retry, state rewrite, or synthetic refund.
        owners = getattr(capacity, "_owners", None)
        if isinstance(owners, dict) and not owners:
            _FSS_POST_START_FAILED_LEDGERS.discard(capacity)


@pytest.mark.parametrize("interruption", [ValueError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_type", [None, LookupError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("handoff_failure", [False, True])
def test_fss_post_start_constructor_cleanup_precedence_and_original_custody(
        fss_post_start_constructor, interruption, cleanup_type, handoff_failure):
    state = fss_post_start_constructor
    cleanup_failure = cleanup_type is not None
    state.cleanup_failure, state.handoff_failure = cleanup_failure, handoff_failure
    cleanup = state.cleanup = cleanup_type("FSS post-start close") if cleanup_type else None
    primary = state.primary = interruption("FSS post-start construction")
    cause, context = RuntimeError("original cause"), OSError("original context")
    primary.__cause__, primary.__context__ = cause, context
    suppress_context = primary.__suppress_context__
    with pytest.raises(cleanup_type or interruption) as caught:
        state.execute()
    assert caught.value is (cleanup if cleanup_failure else primary)
    if cleanup_failure:
        assert cleanup.__context__ is primary
    assert primary.__cause__ is cause and primary.__context__ is context
    assert primary.__suppress_context__ is suppress_context
    assert state.calls == [(role, action) for action in ("manifest", "admit")
                           for role in fss.ROLES]
    assert state.events == ["channels.started", "native.constructor", "peers.close"] + (
        ["handoff"] if cleanup_failure else [])
    assert len(state.frozen) == 1
    assert state.frozen[0][:3] == ("client", "client_request", "failed")
    expected_notes = ["FuseFSS cleanup handoff failed: KeyboardInterrupt"] if (
        cleanup_failure and handoff_failure) else []
    assert getattr(caught.value, "__notes__", []) == expected_notes
    assert getattr(primary, "__notes__", []) == []
    _detach(caught.value)
    del caught
    gc.collect()
    if cleanup_failure:
        token = next(iter(state.capacity._owners))
        reservation = state.refs["reservation"]()
        peers = state.refs["peers"]()
        assert id(token) == state.tokens[0]
        assert token.retained is reservation and reservation._token is token
        assert reservation._failed == (peers,)
        assert state.capacity._owners == {token: state.amount}
        assert peers.closed and not peers._close_succeeded
        assert state.refunds == 0
        before = list(state.events)
        for _ in range(2):
            with pytest.raises(ProtocolError, match="previously failed") as drained:
                reservation.drain_failed()
            _detach(drained.value)
            assert state.events == before
            assert state.capacity._owners == {token: state.amount}
            assert token.retained is reservation and state.refunds == 0
    else:
        assert not state.capacity._owners and state.refunds == 1
        assert state.refs["reservation"]() is None and state.refs["peers"]() is None


@pytest.mark.parametrize("cut", ["channels", "client_request"])
@pytest.mark.parametrize("diagnostic", [
    "add_note_value", "add_note_interrupt", "add_note_exit", "attribute_interrupt",
    "noniterable", "hostile_iterable", "nonstring", "valid",
])
def test_fss_cleanup_handoff_diagnostic_failure_preserves_cleanup_primary(
        fss_post_start_constructor, cut, diagnostic):
    state = fss_post_start_constructor
    state.start_failure = cut == "channels"
    state.cleanup_failure = state.handoff_failure = True
    primary = state.primary = ValueError("FSS construction")
    diagnostics = []

    class HostileNotes:
        def __iter__(self):
            pytest.fail("foreign diagnostic iterator must not run")

    class Cleanup(LookupError):
        def __getattribute__(self, name):
            if name == "__notes__" and diagnostic == "attribute_interrupt":
                diagnostics.append("attribute")
                raise SystemExit("diagnostic access")
            return super().__getattribute__(name)

        def add_note(self, note):
            diagnostics.append("add_note")
            if diagnostic.startswith("add_note_"):
                kind = {"add_note_value": ValueError, "add_note_interrupt": KeyboardInterrupt,
                        "add_note_exit": SystemExit}[diagnostic]
                raise kind("diagnostic attachment")
            return super().add_note(note)

    cleanup = state.cleanup = Cleanup("FSS close")
    if diagnostic in {"noniterable", "hostile_iterable", "nonstring", "valid"}:
        cleanup.__notes__ = {"noniterable": 7, "hostile_iterable": HostileNotes(),
                             "nonstring": [7], "valid": ["existing note"]}[diagnostic]
    with pytest.raises(Cleanup) as caught:
        state.execute()
    assert caught.value is cleanup and cleanup.__context__ is primary
    assert state.events == (["channels.failed"] if state.start_failure else
                            ["channels.started", "native.constructor"]) + ["peers.close", "handoff"]
    assert state.calls == ([] if state.start_failure else [
        (role, action) for action in ("manifest", "admit") for role in fss.ROLES])
    assert len(state.frozen) == 1 and state.frozen[0][:3] == ("client", cut, "failed")
    if diagnostic == "valid":
        assert cleanup.__notes__ == ["existing note", "FuseFSS cleanup handoff failed: KeyboardInterrupt"]
    elif diagnostic in {"noniterable", "hostile_iterable", "nonstring"}:
        assert not diagnostics
    else:
        assert diagnostics == ["attribute" if diagnostic == "attribute_interrupt" else "add_note"]
    _detach(caught.value)
    token = next(iter(state.capacity._owners))
    reservation = state.refs["reservation"]()
    assert id(token) == state.tokens[0] and token.retained is reservation
    assert reservation._token is token and reservation._failed == (state.refs["peers"](),)
    assert state.capacity._owners == {token: state.amount} and state.refunds == 0
    before = list(state.events)
    for _ in range(2):
        with pytest.raises(ProtocolError, match="previously failed") as drained:
            reservation.drain_failed()
        _detach(drained.value)
        assert state.events == before and state.refunds == 0
        assert state.capacity._owners == {token: state.amount} and token.retained is reservation


@pytest.mark.parametrize("charged", [False, True])
def test_observed_ledger_fixture_teardown_only_discards_empty_root(monkeypatch, charged):
    # Drive the actual fixture generator to inspect AFTER teardown, without
    # execute(): no native constructor/transport/owner cleanup is called.
    fixture = fss_post_start_constructor.__wrapped__(monkeypatch)
    state = next(fixture)
    capacity = state.capacity
    assert isinstance(capacity._owners, dict) and type(capacity._owners) is not dict
    assert capacity in _FSS_POST_START_FAILED_LEDGERS
    reservation = None
    if charged:
        reservation = resources.reserve_incremental_memory(state.amount)
        reservation.__enter__()
        owner = NS(close=lambda: pytest.fail("teardown must not retire owners"))
        reservation.retain_failed((owner,))
        token = reservation._token
    fixture.close()
    assert state.refunds == 0
    if charged:
        assert capacity in _FSS_POST_START_FAILED_LEDGERS
        assert capacity._owners == {token: state.amount}
        assert token.retained is reservation and reservation._failed == (owner,)
    else:
        assert not capacity._owners
        assert capacity not in _FSS_POST_START_FAILED_LEDGERS
