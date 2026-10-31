"""UNRUN shared custody regressions; imports may load native code/read files.

Admission observations and backend/control-boundary faults are injected. No native,
model, telemetry or transport execution evidence is claimed by these tests.
"""

from contextlib import ExitStack, contextmanager
import gc
import linecache
import sys
import threading
from types import SimpleNamespace as NS
import weakref

import pytest

from pllm.deployment import resources
from pllm.runtime.delegated_client import DelegatedClient
from pllm.runtime.fss_client import FssClient
from pllm.runtime.replicated_client import ReplicatedClient
from pllm.runtime.responses import _PreparedPromptUse


AMOUNT = 64 << 10  # Existing allowance, not new cost/funding proof.
# Deliberately ambiguous backends are never synthetically refunded at teardown.
# Root only actual capacity ledgers, not separate owner/callback registries.
_FAILED_LEDGERS = set()


class _Payload:
    pass


class _Owner:
    def __init__(self, failures=0):
        self.backing = _Payload()
        self.failures = failures
        self.calls = 0
        self.before_close = None

    def close(self):
        self.calls += 1
        if self.before_close is not None:
            self.before_close()
        if self.failures:
            self.failures -= 1
            raise LookupError("owner still live")
        self.backing = None  # Successful return witnesses actual backing retirement.


class _NoTruthTest:
    def __bool__(self):
        raise AssertionError("direct backend result is not suppression")


class _Backend:
    def __init__(self, reservation):
        self.original = reservation._manager
        self.ref = weakref.ref(reservation)
        self.calls = 0
        self.fault = None
        self.before_exit = None

    def __enter__(self):
        return self.original.__enter__()

    def __exit__(self, *exc):
        self.calls += 1
        reservation = self.ref()
        if reservation._token.retained is reservation:
            assert exc == (None, None, None)
            assert reservation._manager is self
            assert reservation._failed
            assert reservation._draining
            assert reservation._drain_finish_state == "backend_attempted"
            assert reservation._capacity._owners == {reservation._token: AMOUNT}
            assert all(
                owner.backing is None if isinstance(owner, _Owner)
                else owner._closed and owner._metadata_owner is None
                for owner in reservation._failed)
        if self.before_exit is not None:
            self.before_exit()
        result = self.original.__exit__(*exc)
        if reservation._token.retained is reservation:
            # REAL generator finally deferred removal; no replacement refund.
            assert reservation._capacity._owners == {reservation._token: AMOUNT}
        if self.fault is not None:
            raise self.fault
        return result


class _CountingLedger(dict):
    refunds = 0

    def pop(self, token, *default):
        charged = token in self
        value = super().pop(token, *default)
        if charged and token not in self:
            self.refunds += 1
        return value


@pytest.fixture
def admitted(monkeypatch):
    capacity = resources._ClientCapacity()
    capacity._owners = _CountingLedger()
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: NS(
        headroom_bytes=AMOUNT - sum(capacity._owners.values())))
    reservation = resources.reserve_incremental_memory(AMOUNT)
    backend = _Backend(reservation)
    reservation._manager = backend
    reservation.__enter__()
    return NS(capacity=capacity, reservation=reservation,
              backend=backend, token=reservation._token)


def _assert_root(h):
    assert h.reservation._capacity is h.capacity
    assert h.reservation._token is h.token
    assert h.reservation._manager is h.backend
    assert h.reservation._amount == AMOUNT
    assert h.capacity._owners == {h.token: AMOUNT}
    assert h.token.retained is h.reservation
    assert h.reservation.pending_retirement
    assert h.reservation._failed


def _detach(error):
    error.__traceback__ = error.__context__ = error.__cause__ = None


def test_original_root_inside_backend_and_single_deferred_refund(admitted):
    h = admitted
    owner = _Owner()
    payload = weakref.ref(owner.backing)
    owners = (owner,)
    h.reservation.retain_failed(owners)

    def inspect_backend():
        _assert_root(h)
        assert h.reservation._failed is owners
        assert owner.backing is None and payload() is None
        assert not h.capacity.lock._is_owned()
        assert h.reservation._finish_lock.locked()
        with pytest.raises(resources.NetworkError, match="NOT_DRAINABLE"):
            h.reservation.drain_failed()

    h.backend.before_exit = inspect_backend
    # Truthiness MUST NOT be evaluated for the original backend return.
    original_exit = h.backend.original.__exit__

    def backend_exit(*exc):
        original_exit(*exc)
        return _NoTruthTest()

    h.backend.original.__exit__ = backend_exit
    h.reservation.drain_failed()
    assert not h.capacity._owners
    assert h.reservation._failed == () and h.token.retained is None
    assert h.reservation._exited and not h.reservation._draining
    assert h.reservation._drain_finish_state == "committed"
    h.reservation.drain_failed()
    assert h.backend.calls == owner.calls == 1


def test_backend_exception_gc_keeps_original_ledger_custody(admitted):
    h = admitted
    owner = _Owner()
    shell = weakref.ref(owner)
    payload = weakref.ref(owner.backing)
    h.reservation.retain_failed((owner,))
    h.backend.fault = RuntimeError("consumed backend, ambiguous completion")
    with pytest.raises(RuntimeError) as caught:
        h.reservation.drain_failed()
    _detach(caught.value)
    del caught
    _assert_root(h)
    assert owner.backing is None and payload() is None
    capacity, token = h.capacity, h.token
    reservation_ref = weakref.ref(h.reservation)
    _FAILED_LEDGERS.add(capacity)
    # Fixture namespace and all caller owner/reservation refs disappear.
    h.reservation = h.backend = h.token = h.capacity = None
    del h, owner
    gc.collect()
    assert capacity._owners == {token: AMOUNT}
    assert token.retained is reservation_ref()
    assert token.retained._failed == (shell(),)
    for _ in range(2):
        with pytest.raises(resources.NetworkError, match="NOT_DRAINABLE") as caught:
            token.retained.drain_failed()
        _detach(caught.value)
        del caught
        assert token.retained._manager.calls == shell().calls == 1
        assert capacity._owners == {token: AMOUNT}
        assert token.retained.pending_retirement


def test_two_owners_partial_failure_then_success(admitted):
    h = admitted
    first, second = _Owner(), _Owner(failures=1)
    h.reservation.retain_failed((first, second))
    with pytest.raises(LookupError):
        h.reservation.drain_failed()
    _assert_root(h)
    assert first.backing is None and second.backing is not None
    assert h.reservation._failed == (second,)
    assert h.backend.calls == 0
    assert not h.reservation._draining
    h.reservation.drain_failed()
    assert (first.calls, second.calls, h.backend.calls) == (1, 2, 1)
    assert not h.capacity._owners


def test_owner_reentrant_concurrent_drains_and_ordinary_exit_no_bypass(admitted):
    h = admitted
    entered, release = threading.Event(), threading.Event()
    errors = []
    owner = _Owner()

    def wait_owner():
        assert not h.capacity.lock._is_owned()
        assert not h.reservation._finish_lock.locked()
        with pytest.raises(resources.NetworkError, match="NOT_DRAINABLE"):
            h.reservation.drain_failed()
        assert h.reservation.__exit__(None, None, None) is False
        _assert_root(h)
        entered.set()
        assert release.wait(5)

    owner.before_close = wait_owner
    h.reservation.retain_failed((owner,))

    def drain():
        try:
            h.reservation.drain_failed()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(resources.NetworkError, match="NOT_DRAINABLE"):
            h.reservation.drain_failed()
        assert h.reservation.__exit__(None, None, None) is False
        assert h.backend.calls == 0
        _assert_root(h)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert owner.calls == h.backend.calls == 1
    assert not h.capacity._owners


@pytest.mark.parametrize("stage", ["owner", "backend"])
def test_precommit_validation_failure_is_sticky_full_charge(admitted, stage):
    h = admitted
    owner = _Owner()
    h.reservation.retain_failed((owner,))

    def invalidate():
        h.reservation._amount = AMOUNT + 1

    if stage == "owner":
        owner.before_close = invalidate
    else:
        h.backend.before_exit = invalidate
    with pytest.raises(resources.NetworkError, match="NOT_DRAINABLE"):
        h.reservation.drain_failed()
    h.reservation._amount = AMOUNT  # Restore test corruption, never ledger/root.
    _assert_root(h)
    _FAILED_LEDGERS.add(h.capacity)
    for _ in range(2):
        with pytest.raises(resources.NetworkError, match="NOT_DRAINABLE"):
            h.reservation.drain_failed()
        assert h.reservation.__exit__(None, None, None) is False
    assert owner.calls == 1
    assert h.backend.calls == (stage == "backend")


def test_interruption_after_refund_is_committed_not_retained_charge(admitted):
    h = admitted
    owner = _Owner()
    h.reservation.retain_failed((owner,))
    interrupted = []
    previous = sys.gettrace()

    def interrupt(frame, event, arg):
        if (frame.f_code is resources._MemoryReservation.drain_failed.__code__
                and event == "line" and not interrupted
                and h.reservation._drain_finish_state == "committing"
                and h.token not in h.capacity._owners):
            interrupted.append(True)
            raise KeyboardInterrupt("after irreversible ledger removal")
        return interrupt

    try:
        sys.settrace(interrupt)
        with pytest.raises(KeyboardInterrupt):
            h.reservation.drain_failed()
    finally:
        sys.settrace(previous)
    assert interrupted
    assert not h.capacity._owners
    assert not h.reservation.pending_retirement
    assert not h.reservation._draining
    assert h.reservation._failed == (owner,)
    h.reservation.drain_failed()  # Explicit terminal-only recovery.
    assert h.reservation._exited and h.token.retained is None
    assert h.reservation._failed == ()
    assert owner.calls == h.backend.calls == 1


def test_actual_count_owner_exitstack_reenters_same_reservation(admitted):
    h = admitted
    owner = _PreparedPromptUse._empty()
    tokenizer = _Owner(failures=1)
    owner._adopt_tokenizer(tokenizer)
    stack = ExitStack()
    # Already-entered fixture reservation: register its REAL retained __exit__.
    stack.push(h.reservation)
    owner._metadata_owner = stack
    with pytest.raises(LookupError):
        owner.close()
    h.reservation.retain_failed((owner,))
    errors = []

    def drain():
        try:
            h.reservation.drain_failed()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive(), "finish lock held across actual count-owner close"
    assert not errors
    assert tokenizer.calls == 2 and tokenizer.backing is None
    assert owner._closed and owner._metadata_owner is None
    assert h.backend.calls == 1
    assert not h.capacity._owners


@pytest.mark.parametrize("client_type", [DelegatedClient, FssClient, ReplicatedClient])
def test_shared_client_close_drains_original_reservation(admitted, client_type):
    h = admitted
    owner = _Owner()
    h.reservation.retain_failed((owner,))
    # Lifecycle-only shell: constructors would perform unrelated source/native
    # admission. Exercise REAL inherited close and owned-state clearing.
    client = client_type.__new__(client_type)
    client._lock = threading.Lock()
    client._active = None
    client._failed_request_cleanup = h.reservation
    client._compact_cleanup = None
    client._tokenizer_scopes = []
    client.records = {}
    client.close()
    assert client._failed_request_cleanup is None
    assert not h.capacity._owners
    assert h.backend.calls == owner.calls == 1


def test_concurrent_ordinary_exit_cannot_bypass_backend_transaction(admitted):
    h = admitted
    owner = _Owner()
    h.reservation.retain_failed((owner,))
    entered, done = threading.Event(), threading.Event()
    errors, threads = [], []

    def ordinary_exit():
        entered.set()
        try:
            assert h.reservation.__exit__(None, None, None) is False
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    def during_backend():
        thread = threading.Thread(target=ordinary_exit, daemon=True)
        threads.append(thread)
        thread.start()
        assert entered.wait(5)
        assert not done.is_set()  # finish lock held through backend + commit.
        _assert_root(h)

    h.backend.before_exit = during_backend
    try:
        h.reservation.drain_failed()
    finally:
        for thread in threads:
            thread.join(5)
    assert done.is_set() and not errors
    assert all(not thread.is_alive() for thread in threads)
    assert owner.calls == h.backend.calls == 1
    assert not h.capacity._owners


def test_retired_shell_last_reference_drops_outside_locks(admitted):
    h = admitted
    events = []
    capacity, reservation_ref = h.capacity, weakref.ref(h.reservation)

    class Owner(_Owner):
        def __del__(self):
            reservation = reservation_ref()
            events.append((self.backing is None, capacity.lock._is_owned(),
                           reservation._finish_lock.locked()))

    owner = Owner()
    shell = weakref.ref(owner)
    h.reservation.retain_failed((owner,))
    del owner
    h.reservation.drain_failed()
    gc.collect()
    assert shell() is None
    assert events == [(True, False, False)]


@pytest.mark.parametrize("exceptional", [False, True])
def test_ordinary_reservation_behavior_unchanged(admitted, exceptional):
    h = admitted
    error = ValueError("ordinary exceptional exit") if exceptional else None
    args = (type(error), error, None) if exceptional else (None, None, None)
    assert h.reservation.__exit__(*args) is False
    assert not h.capacity._owners and h.backend.calls == 1
    assert h.reservation.__exit__(None, None, None) is False
    assert h.backend.calls == 1
    assert h.reservation._drain_finish_state is None


class _Injected(BaseException):
    pass


class _ContextLock:
    """Fault hooks outside actual acquisition/after actual release, once only.

    Not a claim that a broken primitive can abandon physical lock ownership.
    Reset-line injection below uses REAL with-unwinding to release its lock.
    """
    def __init__(self, original, *, before_enter=None, after_release=None):
        self.original = original
        self.before_enter = before_enter
        self.after_release = after_release

    def __enter__(self):
        if self.before_enter is not None:
            self.before_enter()
        self.original.__enter__()
        return self

    def __exit__(self, *exc):
        result = self.original.__exit__(*exc)
        if self.after_release is not None:
            self.after_release()
        return result

    def _is_owned(self):
        return self.original._is_owned()


@contextmanager
def _inject_line(marker, error):
    # Source inspection only WHEN separately authorized tests execute. This
    # does read source files; neither imports nor collection are files-free.
    previous = sys.gettrace()
    hits = []

    def interrupt(frame, event, arg):
        if (frame.f_code is resources._MemoryReservation.drain_failed.__code__
                and event == "line" and not hits
                and (frame.f_lineno == marker if isinstance(marker, int)
                     else marker in linecache.getline(frame.f_code.co_filename, frame.f_lineno))):
            # Exact-line probes retain primitives only, never frames/errors/custody.
            hits.append((frame.f_lineno, frame.f_locals.get("body_error") is None)
                        if isinstance(marker, int) else True)
            raise error
        return interrupt

    try:
        sys.settrace(interrupt)
        yield hits
    finally:
        sys.settrace(previous)


@pytest.mark.parametrize("point", ["initial-enter", "after-publish", "initial-release"])
def test_admission_interruption_clears_only_own_flag_and_allows_retry(admitted, point):
    h = admitted
    owner = _Owner()
    h.reservation.retain_failed((owner,))
    interruption = _Injected(point)
    hits = []

    def boundary():
        if hits:
            return
        if point == "initial-release" and not h.reservation._draining:
            return
        hits.append(True)
        raise interruption

    if point == "initial-enter":
        h.capacity.lock = _ContextLock(h.capacity.lock, before_enter=boundary)
    elif point == "initial-release":
        h.capacity.lock = _ContextLock(h.capacity.lock, after_release=boundary)
    if point == "after-publish":
        with _inject_line("if recovering:", interruption) as trace_hits:
            with pytest.raises(_Injected) as caught:
                h.reservation.drain_failed()
        assert trace_hits
    else:
        with pytest.raises(_Injected) as caught:
            h.reservation.drain_failed()
        assert hits
    assert caught.value is interruption
    _detach(caught.value)
    del caught
    _assert_root(h)
    assert h.reservation._draining is False
    assert h.reservation._drain_finish_state is None
    assert owner.calls == h.backend.calls == h.capacity._owners.refunds == 0
    h.reservation.drain_failed()
    assert owner.calls == h.backend.calls == h.capacity._owners.refunds == 1
    assert not h.capacity._owners


@pytest.mark.parametrize("point", [
    "cleanup-control-entry", "cleanup-enter", "cleanup-reset", "cleanup-release",
    "cleanup-control-released",
])
def test_finalizer_interruption_propagates_and_terminal_recovery_never_refunds_twice(
        admitted, point):
    h = admitted
    owner = _Owner()
    h.reservation.retain_failed((owner,))
    interruption = _Injected(point)
    hits = []

    def boundary():
        if (hits or h.capacity._owners
                or h.reservation._drain_finish_state != "committed"):
            return
        if point == "cleanup-enter" and not h.reservation._draining:
            return
        if point == "cleanup-release" and h.reservation._draining:
            return
        hits.append(True)
        raise interruption

    if point == "cleanup-enter":
        h.capacity.lock = _ContextLock(h.capacity.lock, before_enter=boundary)
    elif point == "cleanup-release":
        h.capacity.lock = _ContextLock(h.capacity.lock, after_release=boundary)
    trace_markers = {
        "cleanup-control-entry": "# latch cleanup pass 1 acquire",
        "cleanup-reset": "# latch cleanup pass 1 reset",
        "cleanup-control-released": "# latch cleanup pass 1 released",
    }
    if point in trace_markers:
        with _inject_line(trace_markers[point], interruption) as trace_hits:
            with pytest.raises(_Injected) as caught:
                h.reservation.drain_failed()
        assert trace_hits
    else:
        with pytest.raises(_Injected) as caught:
            h.reservation.drain_failed()
        assert hits
    # A locally caught cleanup interruption must NOT disappear into success.
    assert caught.value is interruption
    _detach(caught.value)
    del caught
    assert h.reservation._draining is False
    assert not h.reservation.pending_retirement and not h.capacity._owners
    assert h.capacity._owners.refunds == h.backend.calls == owner.calls == 1
    h.reservation.drain_failed()
    assert h.reservation._exited and h.token.retained is None
    assert h.reservation._failed == () and h.reservation._draining is False
    assert h.capacity._owners.refunds == h.backend.calls == owner.calls == 1


def test_release_interruption_second_pass_cannot_clear_new_attempt_flag(admitted):
    h = admitted
    primary = LookupError("first owner attempt still live")
    interruption = _Injected("old cleanup release, new drain already owns flag")
    owner = _Owner()
    h.reservation.retain_failed((owner,))
    entered, release = threading.Event(), threading.Event()
    old_flags, new_flags, threads, worker_errors, hits = [], [], [], [], []

    def owner_step():
        if owner.calls == 1:
            old_flags.append(h.reservation._draining)
            raise primary
        new_flags.append(h.reservation._draining)
        entered.set()
        assert release.wait(5)

    owner.before_close = owner_step

    def new_drain():
        try:
            h.reservation.drain_failed()
        except BaseException as error:
            worker_errors.append(error)

    def after_release():
        if hits or h.reservation._draining or owner.calls != 1:
            return
        hits.append(True)
        thread = threading.Thread(target=new_drain, daemon=True)
        threads.append(thread)
        thread.start()
        assert entered.wait(5)
        raise interruption

    h.capacity.lock = _ContextLock(h.capacity.lock, after_release=after_release)
    try:
        with pytest.raises(LookupError) as caught:
            h.reservation.drain_failed()
        # This established guarded association is one bounded case, NOT global
        # primary-precedence proof for arbitrary Python interruptions.
        # Exact ordinary body primary survives here; exact cleanup cause retained.
        assert caught.value is primary and primary.__cause__ is interruption
        assert len(old_flags) == len(new_flags) == 1
        assert old_flags[0] is not new_flags[0]
        assert h.reservation._draining is new_flags[0]
        _assert_root(h)
        assert h.backend.calls == h.capacity._owners.refunds == 0
        _detach(primary)
        _detach(interruption)
        del caught
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
    assert threads and all(not thread.is_alive() for thread in threads)
    assert not worker_errors
    assert h.reservation._draining is False
    assert h.backend.calls == h.capacity._owners.refunds == 1
    assert owner.calls == 2 and not h.capacity._owners


@pytest.mark.parametrize("phase", ["commit", "recovery"])
@pytest.mark.parametrize("assignment", ["exited", "state", "token", "custody"])
def test_each_terminal_assignment_interruption_keeps_shell_until_explicit_recovery(
        admitted, phase, assignment):
    h = admitted
    capacity, reservation_ref = h.capacity, weakref.ref(h.reservation)
    events = []

    class Owner(_Owner):
        def __del__(self):
            reservation = reservation_ref()
            events.append((self.backing is None, capacity.lock._is_owned(),
                           reservation._finish_lock.locked()))

    owner = Owner()
    shell = weakref.ref(owner)
    payload = weakref.ref(owner.backing)
    h.reservation.retain_failed((owner,))
    del owner  # No foreign strong owner/backing references during either clear.
    if phase == "recovery":
        priming = _Injected("one interruption in priming operation")
        with _inject_line("# terminal exited", priming) as hits:
            with pytest.raises(_Injected) as caught:
                h.reservation.drain_failed()
        assert hits and caught.value is priming
        _detach(priming)
        del caught
        assert h.reservation._draining is False
    interruption = _Injected("one interruption in tested operation")
    with _inject_line("# terminal " + assignment, interruption) as hits:
        with pytest.raises(_Injected) as caught:
            h.reservation.drain_failed()
    assert hits and caught.value is interruption
    _detach(interruption)
    del caught
    gc.collect()
    assert payload() is None  # Backing truly retired before any refund.
    assert shell() is not None and h.reservation._failed == (shell(),)
    assert not events  # Interrupted clear leaves remaining field custody intact.
    assert h.reservation._draining is False
    assert not h.capacity._owners and not h.reservation.pending_retirement
    assert h.backend.calls == h.capacity._owners.refunds == shell().calls == 1
    h.reservation.drain_failed()  # Terminal-only recovery, including _exited=True.
    gc.collect()
    assert shell() is None and events == [(True, False, False)]
    assert h.reservation._failed == () and h.token.retained is None
    assert h.reservation._exited and h.reservation._draining is False
    assert h.backend.calls == h.capacity._owners.refunds == 1


class _OneShotPrimaryOwner(_Owner):
    def __init__(self, primary):
        super().__init__()
        self.primary = primary

    def close(self):
        if self.primary is not None:
            error, self.primary = self.primary, None
            self.calls += 1
            raise error
        return super().close()


def _handled_primary_source_line(site):
    """Resolve exact REAL drain source lines, including only OUTER bare raise."""
    code = resources._MemoryReservation.drain_failed.__code__
    source = linecache.getlines(code.co_filename)
    start = code.co_firstlineno - 1
    stop = next(index for index in range(start + 1, len(source))
                if source[index].startswith("class _CleanupSlot:"))
    exact = {
        "body-capture": "                body_error = error\n",
        "outer-guard": "            if not first_pass_done:\n",
        "outer-reraise": "            raise\n",
    }[site]
    matches = [index + 1 for index in range(start, stop) if source[index] == exact]
    assert len(matches) == 1, (site, matches)
    outer = [index + 1 for index in range(start, stop)
             if source[index] == "        except BaseException as cleanup_error:\n"]
    assert len(outer) == 1
    assert (matches[0] < outer[0]) == (site == "body-capture")
    return matches[0]


def _retry_one_shot_owner(reservation, backend, owner):
    # Acquire temporary caller refs only AFTER ledger-only GC proof. No closure
    # or probe retains them after this explicit retry returns.
    reservation.drain_failed()
    assert backend.calls == 1 and owner.calls == 2


@pytest.mark.parametrize("failure", ["owner", "backend"])
@pytest.mark.parametrize("site", [
    "body-capture", "outer-guard", "outer-reraise", None,
], ids=["before-capture", "before-outer-guard", "before-outer-reraise", "uninjected"])
def test_python_primary_interruption_policy_and_original_gc_custody(admitted, failure, site):
    # Six actual-source-line injections plus two uninjected controls. No claim
    # of universal bytecode precedence. One injected BaseException per operation.
    h = admitted
    primary = (LookupError("known one-shot owner failure") if failure == "owner"
               else RuntimeError("known backend failure after real backend exit"))
    owner = _OneShotPrimaryOwner(primary) if failure == "owner" else _Owner()
    payload_ref = weakref.ref(owner.backing)
    h.reservation.retain_failed((owner,))
    # Only actual ledger root; never automatic cleanup/retry/refund at teardown.
    _FAILED_LEDGERS.add(h.capacity)
    if failure == "backend":
        h.backend.fault = primary  # Existing wrapper raises AFTER real __exit__.
    interruption = _Injected("external interruption at " + site) if site else None
    expected = interruption if site else primary
    if site:
        line_number = _handled_primary_source_line(site)
        with _inject_line(line_number, interruption) as hits:
            with pytest.raises(_Injected) as caught:
                h.reservation.drain_failed()
        assert hits == [(line_number, site == "body-capture")]
        assert caught.value is interruption
        assert interruption.__context__ is primary
        # At before-capture, body_error is STILL None. Do not assume association.
        del hits
    else:
        with pytest.raises(type(primary)) as caught:
            h.reservation.drain_failed()
        assert caught.value is primary
    assert caught.value is expected
    _assert_root(h)
    assert h.reservation._draining is False
    assert h.reservation._failed == (owner,)
    assert owner.calls == 1 and h.capacity._owners.refunds == 0
    assert h.backend.calls == (1 if failure == "backend" else 0)
    assert h.reservation._drain_finish_state == (
        "backend_attempted" if failure == "backend" else None)
    assert (owner.backing is None) == (failure == "backend")

    capacity = h.capacity
    token_id, manager_id = id(h.token), id(h.backend)
    reservation_id, owner_id = id(h.reservation), id(owner)
    reservation_ref, backend_ref = weakref.ref(h.reservation), weakref.ref(h.backend)
    owner_ref = weakref.ref(owner)
    _detach(primary)
    if interruption is not None:
        _detach(interruption)
    h.backend.fault = None  # Sticky state, NOT a retained test exception, must root custody.
    assert owner.before_close is None and h.backend.before_exit is None
    if failure == "owner":
        assert owner.primary is None  # One-shot failure already consumed.
    capacity.observe = None  # Drop admission fixture closure; never sample telemetry.
    h.__dict__.clear()  # Pytest's cached fixture namespace now holds no custody.
    del h, admitted, owner, primary, interruption, expected, caught
    gc.collect()

    # Only capacity roots + weak references/primitive IDs remain in this test.
    assert len(capacity._owners) == 1
    assert id(next(iter(capacity._owners))) == token_id
    assert capacity._owners[next(iter(capacity._owners))] == AMOUNT
    assert next(iter(capacity._owners)).retained is reservation_ref()
    assert id(reservation_ref()) == reservation_id
    assert id(backend_ref()) == manager_id and id(owner_ref()) == owner_id
    assert reservation_ref()._capacity is capacity
    assert reservation_ref()._token is next(iter(capacity._owners))
    assert reservation_ref()._manager is backend_ref()
    assert reservation_ref()._amount == AMOUNT
    assert reservation_ref()._failed == (owner_ref(),)
    assert reservation_ref().pending_retirement and reservation_ref()._draining is False
    assert capacity._owners.refunds == 0
    if failure == "owner":
        assert payload_ref() is not None
        assert backend_ref().calls == 0 and reservation_ref()._drain_finish_state is None
        _retry_one_shot_owner(reservation_ref(), backend_ref(), owner_ref())
        assert not capacity._owners and capacity._owners.refunds == 1
        gc.collect()
        assert reservation_ref() is backend_ref() is owner_ref() is payload_ref() is None
        _FAILED_LEDGERS.discard(capacity)
    else:
        assert owner_ref().backing is None and payload_ref() is None
        for _ in range(2):
            with pytest.raises(resources.NetworkError, match="NOT_DRAINABLE") as retry:
                reservation_ref().drain_failed()
            _detach(retry.value)
            del retry
            assert backend_ref().calls == owner_ref().calls == 1
            assert capacity._owners.refunds == 0
            assert capacity._owners == {next(iter(capacity._owners)): AMOUNT}
            assert next(iter(capacity._owners)).retained is reservation_ref()
            assert reservation_ref()._manager is backend_ref()
            assert reservation_ref()._failed == (owner_ref(),)
            assert reservation_ref()._drain_finish_state == "backend_attempted"
            assert reservation_ref()._draining is False and reservation_ref().pending_retirement
        gc.collect()
        assert capacity in _FAILED_LEDGERS  # Unresolved original root stays charged.
        assert next(iter(capacity._owners)).retained is reservation_ref()
        assert reservation_ref()._failed == (owner_ref(),)
        assert backend_ref().calls == owner_ref().calls == 1


class _ThrowingNotePrimary(LookupError):
    def __init__(self, secondary):
        super().__init__("first owner failure; optional note must not replace it")
        self.secondary = secondary
        self.note_calls = 0

    def add_note(self, note):
        self.note_calls += 1
        raise self.secondary


@pytest.mark.parametrize("secondary_type", [MemoryError, RuntimeError])
def test_two_owner_throwing_note_keeps_first_primary_and_original_custody(
        admitted, secondary_type):
    h = admitted
    secondary = secondary_type("optional diagnostic insertion failed")
    primary = _ThrowingNotePrimary(secondary)
    other_error = LookupError("second owner failure")
    first, second = _OneShotPrimaryOwner(primary), _OneShotPrimaryOwner(other_error)
    owners = (first, second)
    h.reservation.retain_failed(owners)
    _FAILED_LEDGERS.add(h.capacity)
    with pytest.raises(_ThrowingNotePrimary) as caught:
        h.reservation.drain_failed()
    assert caught.value is primary and primary.__cause__ is secondary
    assert primary.note_calls == 1  # Actual existing add_note dispatch, not a fake refund.
    _assert_root(h)
    assert h.reservation._failed is owners
    assert owners[0] is first and owners[1] is second
    assert first.calls == second.calls == 1
    assert first.backing is not None and second.backing is not None
    assert h.reservation._draining is False and h.reservation._drain_finish_state is None
    assert h.backend.calls == h.capacity._owners.refunds == 0
    _detach(primary)
    _detach(secondary)
    _detach(other_error)
    del caught
    h.reservation.drain_failed()  # Both one-shot owners now retire normally.
    assert first.calls == second.calls == 2
    assert first.backing is second.backing is None
    assert h.backend.calls == h.capacity._owners.refunds == 1
    assert not h.capacity._owners and h.token.retained is None
    assert h.reservation._failed == () and h.reservation._draining is False
    h.reservation.drain_failed()
    assert h.backend.calls == h.capacity._owners.refunds == 1
    _FAILED_LEDGERS.discard(h.capacity)


def _owner_aggregation_source_line(site):
    # Actual builtin-list append/slice and tuple-construction statements. Never
    # mock a list or claim real memory exhaustion; inject at precise source line.
    code = resources._MemoryReservation.drain_failed.__code__
    source = linecache.getlines(code.co_filename)
    start = code.co_firstlineno - 1
    stop = next(index for index in range(start + 1, len(source))
                if source[index].startswith("class _CleanupSlot:"))
    exact = {
        "remaining-append": "                        remaining.append(owner)\n",
        "errors-append": "                        errors.append(error)\n",
        "error-slice": "                    for error in errors[1:]:\n",
        "failed-tuple": "                    failed_owners = tuple(remaining)\n",
    }[site]
    matches = [index + 1 for index in range(start, stop) if source[index] == exact]
    assert len(matches) == 1, (site, matches)
    capture = [index + 1 for index in range(start, stop)
               if source[index] == "                            owner_error = error\n"]
    assert len(capture) == 1 and capture[0] < matches[0]
    return matches[0]


@pytest.mark.parametrize("secondary_type", [MemoryError, RuntimeError])
@pytest.mark.parametrize("site", [
    "remaining-append", "errors-append", "error-slice", "failed-tuple",
])
def test_actual_owner_aggregation_fault_retains_first_primary_and_every_original_owner(
        admitted, site, secondary_type):
    h = admitted
    primary = LookupError("known owner failure before aggregation allocation")
    secondary = secondary_type("fault at actual owner aggregation statement")
    early = site in ("remaining-append", "errors-append")
    if early:
        first, second = _OneShotPrimaryOwner(primary), _Owner()
    else:
        first, second = _Owner(), _OneShotPrimaryOwner(primary)
    owners = (first, second)
    h.reservation.retain_failed(owners)
    _FAILED_LEDGERS.add(h.capacity)
    line_number = _owner_aggregation_source_line(site)
    with _inject_line(line_number, secondary) as hits:
        with pytest.raises(LookupError) as caught:
            h.reservation.drain_failed()
    assert hits == [(line_number, True)]
    assert caught.value is primary and primary.__cause__ is secondary
    _assert_root(h)
    assert h.reservation._failed is owners
    assert owners[0] is first and owners[1] is second
    assert h.reservation._draining is False and h.reservation._drain_finish_state is None
    assert h.backend.calls == h.capacity._owners.refunds == 0
    if early:
        # Fault occurred AFTER first primary capture but BEFORE visiting second.
        assert first.calls == 1 and second.calls == 0
        assert first.backing is not None and second.backing is not None
    else:
        # Completed first retirement must remain conservatively in ORIGINAL
        # custody when slicing/tuple construction prevents publication.
        assert first.calls == second.calls == 1
        assert first.backing is None and second.backing is not None
    _detach(primary)
    _detach(secondary)
    del caught
    h.reservation.drain_failed()
    assert first.calls == 2 and second.calls == (1 if early else 2)
    assert first.backing is second.backing is None
    assert h.backend.calls == h.capacity._owners.refunds == 1
    assert not h.capacity._owners and h.token.retained is None
    assert h.reservation._failed == () and h.reservation._draining is False
    h.reservation.drain_failed()
    assert h.backend.calls == h.capacity._owners.refunds == 1
    _FAILED_LEDGERS.discard(h.capacity)
