"""Compact selection and raw custody; no payload/native integrity activation."""
import asyncio
from contextlib import contextmanager
import gc
import inspect
import json
import sys
import threading
from types import SimpleNamespace
import weakref

import httpx
import pytest
from pllm.deployment import resources

from pllm.configuration import ComponentRef, Experiment
from pllm.profiles import resolve_runtime_composition
from pllm.protocols import PumaThreePartyDecoder
from pllm.runtime import replicated_source
from pllm.runtime.replicated_protocol import (
    execution_binding, execution_policy, storage_policy, validate_execution_binding,
)
from test_replicated_execution_policy import experiment_for


@pytest.fixture(autouse=True)
def _isolated_cleanup_admission(monkeypatch):
    capacity = resources._ClientCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))
    yield
    assert not capacity._cleanup, "test leaked durable cleanup; never discard live payloads"


def _scope():
    return resources.reserve_cleanup_slot()


@pytest.fixture
def raw_bridge(monkeypatch, tmp_path, _isolated_cleanup_admission):
    """Ownership choreography only, NOT synthetic Python source authority.

    Real process DiskLease/CM and cleanup ledger; native file factories below are
    test doubles, not a positive native integrity/activation test. Failed roots
    deliberately survive in these ISOLATED ledgers through all assertions. Their
    eventual test-state disposal is not product cleanup/retry certification.
    """
    capacity = resources._ClientCapacity()
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))
    monkeypatch.setattr(resources, "_capacity", capacity)
    disk = resources._DiskCapacity()
    monkeypatch.setattr(resources, "_disk_capacity", disk)
    events, contexts = [], []
    device = tmp_path.stat().st_dev
    reserve = resources.reserve_incremental_disk

    class RecordingCM:
        def __init__(self, amount, directory):
            self.inner = reserve(amount, directory=directory, minimum_free_bytes=1)
            self.exits = 0
            self.fail_exit = False
            contexts.append(self)

        def __enter__(self):
            events.append("lease_enter")
            return self.inner.__enter__()

        def __exit__(self, *args):
            self.exits += 1
            if self.fail_exit:
                raise RuntimeError("ambiguous lease exit")
            result = self.inner.__exit__(*args)
            events.append("lease_exit")
            return result

    def reservation(amount, *, directory):
        events.append("reserve")
        return RecordingCM(amount, directory)

    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", reservation)

    class Native:
        def __init__(self, fail=None):
            self.fail = fail
            self._raw_disk_bytes = 4096
            self._raw_disk_directory = str(tmp_path)
            self.expected_device = device
            self.files = False
            self.charged = True
            self.readers = 0
            self.retirement_reserved = False
            self.begin_calls = 0

        def _raw_start(self, observed_device):
            # No bool/mock/callback constitutes admission: check actual process
            # ledger membership and exact holder association at this factory.
            assert any(lease._raw_holder is not None
                       and lease._raw_holder._owner is self and not lease.closed
                       and lease._backing is None for lease in disk._owners.values())
            if observed_device != self.expected_device:
                raise ValueError("different device")
            events.append("factory")
            self.files = True
            if self.fail == "start":
                raise ValueError("start failed")

        def _raw_append(self, file, offset, data):
            events.append("append")
            if self.fail == "append":
                raise ValueError("append failed")

        def _raw_finish(self):
            events.append("finish")
            if self.fail == "finish":
                raise ValueError("finish failed")

        def _raw_begin_retire(self):
            # TEST ONLY typed ownership transition, never source authority.
            self.begin_calls += 1
            if self.readers:
                return False
            assert not self.retirement_reserved
            self.retirement_reserved = True
            return True

        def _raw_retire(self):
            assert self.retirement_reserved
            self.retirement_reserved = False
            assert all(lease._backing is None for lease in disk._owners.values()
                       if lease._raw_holder is not None and lease._raw_holder._owner is self)
            events.append("retire")
            if self.fail == "retire":
                raise ValueError("retire failed")
            self.files = False
            events.append("directory_removed")

        def _raw_release_final_charges(self):
            assert not self.files
            assert not any(lease._raw_holder is not None
                           and lease._raw_holder._owner is self for lease in disk._owners.values())
            events.append("release")
            if self.fail == "release":
                raise ValueError("release failed")
            self.charged = False

    monkeypatch.setattr(replicated_source._native, "ReplicatedPublisher", Native)

    def pending(fail=None):
        scope = _scope()
        holder = replicated_source._RawPublisherOwner(scope)
        native = Native(fail)
        holder._retain(native)
        # TEST ONLY: finish synthetic pending adoption. Product keeps this SAME
        # constructor-armed gate through native factory AND disk admission.
        holder._end_operation()
        return holder, native, scope

    return SimpleNamespace(pending=pending, native=Native, cm=RecordingCM, disk=disk, capacity=capacity,
                           events=events, contexts=contexts, directory=str(tmp_path), device=device)


def test_raw_bridge_real_lease_precedes_factory_and_exact_once_cleanup(raw_bridge):
    holder, native, scope = raw_bridge.pending()
    assert raw_bridge.events == [] and not raw_bridge.disk._owners
    holder._admit_disk()
    lease = holder._lease
    assert raw_bridge.events == ["reserve", "lease_enter", "factory"]
    assert lease._raw_holder is holder and lease._backing is None
    observed = raw_bridge.disk.observe(lease._directory, 1)
    assert observed.reserved_disk_bytes == 4096 and observed.backed_disk_bytes == 0
    assert observed.pending_disk_bytes == 4096
    with pytest.raises(Exception, match="DISK_BACKING_BINDING_INVALID"):
        lease.bind_backing(lambda: [])
    holder.append(0, 0, b"x")
    holder.finish()
    holder.close()
    scope.drain()
    assert raw_bridge.events[-4:] == ["retire", "directory_removed", "lease_exit", "release"]
    assert raw_bridge.contexts[0].exits == 1
    assert lease.closed and not native.charged and scope.closed
    assert not raw_bridge.disk._owners and not raw_bridge.capacity._cleanup


@pytest.mark.parametrize("failure", ["start", "append", "finish"])
def test_raw_bridge_ingestion_failure_retains_until_explicit_cleanup(raw_bridge, failure):
    holder, native, scope = raw_bridge.pending(failure)
    with pytest.raises(ValueError, match=f"{failure} failed"):
        holder._admit_disk()
        if failure == "append":
            holder.append(0, 0, b"x")
        elif failure == "finish":
            holder.finish()
    assert scope.owns(holder) and holder._owner is native and native.charged
    assert not holder._lease.closed and holder._disk_cm is raw_bridge.contexts[0]
    assert raw_bridge.contexts[0].exits == 0
    with pytest.raises(ValueError, match="stopped"):
        holder.append(0, 0, b"x")
    holder.close()
    assert scope.closed and not native.charged


def test_raw_bridge_pending_external_drop_stays_cleanup_rooted(raw_bridge):
    holder, native, scope = raw_bridge.pending()
    weak, weak_native = weakref.ref(holder), weakref.ref(native)
    del holder, native
    gc.collect()
    assert weak() is not None and weak()._owner is weak_native() and weak_native().charged
    assert not raw_bridge.events and not raw_bridge.disk._owners
    scope.drain()
    gc.collect()
    assert weak() is None and weak_native() is None
    assert raw_bridge.events == ["retire", "directory_removed", "release"]


def test_raw_bridge_failed_retirement_sticky_even_after_drop_and_repeated_drain(raw_bridge):
    holder, native, scope = raw_bridge.pending("retire")
    holder._admit_disk()
    lease, cm = holder._lease, holder._disk_cm
    weak = weakref.ref(holder)
    with pytest.raises(ValueError, match="retire failed") as caught:
        holder.close()
    error = caught.value
    del caught, holder
    gc.collect()
    for _ in range(3):
        with pytest.raises(ValueError) as repeated:
            scope.drain()
        assert repeated.value is error
    assert weak() is not None and scope.owns(weak())
    assert raw_bridge.events.count("retire") == 1 and "release" not in raw_bridge.events
    assert cm.exits == 0 and not lease.closed and native.charged
    assert raw_bridge.disk._owners[lease._token] is lease


def test_raw_bridge_ambiguous_lease_exit_never_retries_or_releases(raw_bridge):
    holder, native, scope = raw_bridge.pending()
    holder._admit_disk()
    cm = holder._disk_cm
    cm.fail_exit = True
    for _ in range(3):
        with pytest.raises(RuntimeError, match="ambiguous lease exit"):
            scope.drain()
    assert cm.exits == 1 and raw_bridge.events.count("retire") == 1
    assert native.charged and not holder._lease.closed and scope.owns(holder)
    assert "release" not in raw_bridge.events


def test_raw_bridge_final_release_failure_roots_owner_without_repeating_exit(raw_bridge):
    holder, native, scope = raw_bridge.pending("release")
    holder._admit_disk()
    lease, cm = holder._lease, holder._disk_cm
    for _ in range(3):
        with pytest.raises(ValueError, match="release failed"):
            scope.drain()
    assert lease.closed and cm.exits == 1 and native.charged
    assert scope.owns(holder) and holder._owner is native
    assert raw_bridge.events.count("retire") == raw_bridge.events.count("release") == 1


def test_raw_bridge_clears_later_backing_under_lock_before_native_retire(raw_bridge):
    holder, _, _ = raw_bridge.pending()
    holder._admit_disk()
    with raw_bridge.disk.lock:
        # TEST ONLY: model a future binding. Product raw bind_backing is refused.
        holder._lease._backing = lambda: pytest.fail("backing sampled during close")
    holder.close()  # fake native retire asserts callback already cleared


@pytest.mark.parametrize("finished", [False, True])
def test_raw_bridge_reader_busy_preserves_backing_and_retries(raw_bridge, monkeypatch, finished):
    # Genuine lease/ledger; synthetic reader count tests bridge choreography only.
    holder, native, scope = raw_bridge.pending()
    holder._admit_disk()
    if finished:
        holder.finish()
    lease, cm = holder._lease, holder._disk_cm
    assert type(lease) is resources.DiskLease
    backing = lambda: pytest.fail("backing sampled during reader refusal")
    with raw_bridge.disk.lock:
        lease._backing = backing  # TEST ONLY: public raw backing binding stays refused.
    native.readers = 1
    callback = scope._retire
    before = list(raw_bridge.events)
    cleanup_owners = dict(raw_bridge.capacity._owners)
    clear = replicated_source._clear_raw_disk_backing
    clears = []

    def clear_reserved(actual, owner):
        assert holder._gate.acquire(blocking=False), "gate mutex spans backing clear"
        holder._gate.release()
        assert native.retirement_reserved and holder._stopped
        clears.append(actual)
        return clear(actual, owner)

    monkeypatch.setattr(replicated_source, "_clear_raw_disk_backing", clear_reserved)
    for attempt in range(3):
        with pytest.raises(ValueError, match="raw host readers in flight"):
            scope.drain()
        assert native.begin_calls == attempt + 1 and not native.retirement_reserved
        assert raw_bridge.events == before and clears == []
        assert holder._stopped is finished and not holder._busy and not holder._closed
        assert holder._cleanup_failure is None and holder._entered
        assert holder._owner is native and holder._lease is lease and holder._disk_cm is cm
        assert lease._backing is backing and lease._raw_holder is holder and not lease.closed
        assert raw_bridge.disk._owners[lease._token] is lease and lease._bound == 4096
        assert native.files and native.charged and cm.exits == 0
        assert scope.owns(holder) and scope._payload is holder and scope._retire is callback
        assert not scope.closed and not scope._draining
        assert raw_bridge.capacity._cleanup[scope._token] is scope
        assert raw_bridge.capacity._owners == cleanup_owners
    native.readers = 0  # TEST ONLY last-reader drop; native owner proves real readiness later.
    holder.close()
    scope.drain()
    assert native.begin_calls == 4 and clears == [lease]
    assert raw_bridge.events[len(before):] == ["retire", "directory_removed", "lease_exit", "release"]
    assert cm.exits == 1 and lease.closed and lease._backing is None
    assert not native.charged and not native.files and not native.retirement_reserved
    assert holder._owner is holder._lease is holder._disk_cm is None
    assert holder._closed and holder._stopped and not holder._busy
    assert scope.closed and not raw_bridge.disk._owners and not raw_bridge.capacity._cleanup


@pytest.mark.parametrize("outcome", ["exception", "baseexception", "none", "zero", "one", "truthy"])
def test_raw_bridge_ambiguous_begin_retirement_stays_sticky(raw_bridge, monkeypatch, outcome):
    class BeginAborted(BaseException):
        pass

    holder, native, scope = raw_bridge.pending()
    holder._admit_disk()
    lease, cm = holder._lease, holder._disk_cm
    backing = lambda: pytest.fail("backing sampled after ambiguous begin")
    with raw_bridge.disk.lock:
        lease._backing = backing
    error = BeginAborted("begin interrupted") if outcome == "baseexception" else ValueError("begin failed")
    invalid = {"none": None, "zero": 0, "one": 1, "truthy": "ready"}
    before = list(raw_bridge.events)

    def ambiguous(owner):
        assert holder._gate.acquire(blocking=False), "gate mutex spans native begin"
        holder._gate.release()
        owner.begin_calls += 1
        owner.retirement_reserved = True  # may already have transitioned
        if outcome in ("exception", "baseexception"):
            raise error
        return invalid[outcome]

    monkeypatch.setattr(raw_bridge.native, "_raw_begin_retire", ambiguous)
    expected = type(error) if outcome in ("exception", "baseexception") else TypeError
    with pytest.raises(expected) as caught:
        holder.close()
    failure = caught.value
    if outcome in ("exception", "baseexception"):
        assert failure is error
    for _ in range(3):
        with pytest.raises(expected) as repeated:
            scope.drain()
        assert repeated.value is failure
    assert holder._cleanup_failure is failure and holder._stopped and not holder._busy
    assert native.begin_calls == 1 and native.retirement_reserved
    assert raw_bridge.events == before and native.charged and native.files
    assert holder._owner is native and holder._lease is lease and holder._disk_cm is cm
    assert holder._entered and not holder._closed and not lease.closed and cm.exits == 0
    assert lease._backing is backing and raw_bridge.disk._owners[lease._token] is lease
    assert scope.owns(holder) and scope._payload is holder and not scope.closed


@pytest.mark.parametrize("phase", ["clear", "retire", "exit", "release"])
def test_raw_bridge_ready_baseexception_stays_sticky(raw_bridge, monkeypatch, phase):
    class RetirementAborted(BaseException):
        pass

    holder, native, scope = raw_bridge.pending()
    holder._admit_disk()
    lease, cm = holder._lease, holder._disk_cm
    failure = RetirementAborted("reserved retirement interrupted")
    calls = []

    def interrupted(*args):
        assert holder._gate.acquire(blocking=False), "gate mutex spans cleanup callback"
        holder._gate.release()
        assert holder._stopped and holder._busy
        calls.append(phase)
        raise failure

    if phase == "clear":
        monkeypatch.setattr(replicated_source, "_clear_raw_disk_backing", interrupted)
    elif phase == "exit":
        monkeypatch.setattr(raw_bridge.cm, "__exit__", interrupted)
    else:
        method = "_raw_retire" if phase == "retire" else "_raw_release_final_charges"
        monkeypatch.setattr(raw_bridge.native, method, interrupted)
    for _ in range(3):
        with pytest.raises(RetirementAborted) as caught:
            scope.drain()
        assert caught.value is failure
    assert calls == [phase] and native.begin_calls == 1
    assert holder._cleanup_failure is failure and holder._stopped and not holder._busy
    assert holder._owner is native and holder._lease is lease and holder._disk_cm is cm
    assert native.charged and not holder._closed and scope.owns(holder)
    assert cm.exits == int(phase == "release")
    assert lease.closed is (phase == "release")
    assert raw_bridge.events.count("retire") == int(phase in ("exit", "release"))
    assert "release" not in raw_bridge.events


def test_raw_bridge_post_ready_pre_clear_interruption_stays_sticky(raw_bridge, monkeypatch):
    class ReadyInterrupted(BaseException):
        pass

    holder, native, scope = raw_bridge.pending()
    holder._admit_disk()
    lease, cm = holder._lease, holder._disk_cm
    backing = lambda: pytest.fail("backing sampled after Ready interruption")
    with raw_bridge.disk.lock:
        lease._backing = backing  # TEST ONLY; not a public backing/authority grant.
    before = list(raw_bridge.events)
    callback = scope._retire
    cleanup_owners = dict(raw_bridge.capacity._owners)
    failure = ReadyInterrupted("after Ready, before stopped/backing clear")
    retire = type(holder)._retire
    lines, first = inspect.getsourcelines(retire)
    clear_line = next(i for i, line in enumerate(lines) if "_clear_raw_disk_backing(" in line)
    # Last stopped assignment before clear selects the Ready branch, not an
    # earlier ambiguous-begin handler in the old two-handler implementation.
    target_line = first + max(i for i, line in enumerate(lines[:clear_line])
                              if line.strip() == "self._stopped = True")
    interruptions, clears = [], []
    clear = replicated_source._clear_raw_disk_backing

    def record_clear(*args):
        clears.append(args)
        return clear(*args)

    def interrupt(frame, event, arg):
        if frame.f_code is retire.__code__ and event == "line" and frame.f_lineno == target_line:
            assert native.begin_calls == 1 and native.retirement_reserved
            assert frame.f_locals["ready"] is True and not holder._stopped
            interruptions.append(frame.f_lineno)
            raise failure
        return interrupt

    monkeypatch.setattr(replicated_source, "_clear_raw_disk_backing", record_clear)
    previous_trace = sys.gettrace()
    try:
        sys.settrace(interrupt)  # Current thread only; no production injection seam.
        with pytest.raises(ReadyInterrupted) as caught:
            holder.close()
    finally:
        sys.settrace(previous_trace)
    assert sys.gettrace() is previous_trace
    assert caught.value is failure and interruptions == [target_line]
    for drain in (holder.close, scope.drain, holder.close):
        with pytest.raises(ReadyInterrupted) as repeated:
            drain()
        assert repeated.value is failure
    assert holder._cleanup_failure is failure and holder._stopped and not holder._busy
    assert native.begin_calls == 1 and native.retirement_reserved and clears == []
    assert raw_bridge.events == before and native.files and native.charged
    assert holder._owner is native and holder._lease is lease and holder._disk_cm is cm
    assert holder._entered and not holder._closed and cm.exits == 0 and not lease.closed
    assert lease._backing is backing and lease._raw_holder is holder and lease._bound == 4096
    assert raw_bridge.disk._owners[lease._token] is lease
    assert scope.owns(holder) and scope._payload is holder and scope._retire is callback
    assert not scope.closed and not scope._draining
    assert raw_bridge.capacity._cleanup[scope._token] is scope
    assert raw_bridge.capacity._owners == cleanup_owners


@pytest.mark.parametrize("ready_result", [False, True])
def test_raw_bridge_blocking_begin_gates_ingestion_and_concurrent_close(raw_bridge, monkeypatch, ready_result):
    holder, native, scope = raw_bridge.pending()
    holder._admit_disk()
    native.readers = int(not ready_result)
    entered, resume = threading.Event(), threading.Event()
    errors = []
    begin = raw_bridge.native._raw_begin_retire
    before = list(raw_bridge.events)

    def blocking(owner):
        assert holder._gate.acquire(blocking=False), "gate mutex spans native begin"
        holder._gate.release()
        result = begin(owner)
        entered.set()
        assert resume.wait(5), "retirement begin was not released"
        return result

    def drain():
        try:
            holder.close()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(raw_bridge.native, "_raw_begin_retire", blocking)
    thread = threading.Thread(target=drain)
    thread.start()
    try:
        assert entered.wait(5)
        for operation in (lambda: holder.append(0, 0, b"x"), holder.finish,
                          holder._admit_disk, holder._retire):
            with pytest.raises(ValueError, match="raw host operation in flight"):
                operation()
        with pytest.raises(Exception, match="CLIENT_CLEANUP_ALREADY_DRAINING"):
            holder.close()
        assert raw_bridge.events == before and native.charged and native.files
        assert native.retirement_reserved is ready_result and native.begin_calls == 1
        assert holder._busy and not holder._stopped and holder._cleanup_failure is None
        assert scope.owns(holder) and holder._disk_cm.exits == 0
    finally:
        resume.set()
        thread.join(5)
        assert not thread.is_alive(), "retirement begin must join"
    if ready_result:
        assert errors == [] and scope.closed
    else:
        assert len(errors) == 1 and type(errors[0]) is ValueError
        assert str(errors[0]) == "raw host readers in flight"
        assert not holder._busy and not holder._stopped and holder._cleanup_failure is None
        assert raw_bridge.events == before and not native.retirement_reserved
        native.readers = 0
        holder.close()
    scope.drain()
    assert scope.closed and not native.charged and raw_bridge.contexts[0].exits == 1
    assert raw_bridge.events[len(before):] == ["retire", "directory_removed", "lease_exit", "release"]


def test_raw_bridge_preserves_original_error_when_cleanup_fails(raw_bridge):
    holder, _, scope = raw_bridge.pending("retire")
    holder._admit_disk()
    original = ValueError("original ingestion error")
    holder._close_preserving(original)
    assert original.args == ("original ingestion error",)
    assert original.__notes__ == ["raw host cleanup failed: ValueError"]
    assert scope.owns(holder)


def test_raw_bridge_rejects_wrong_dead_and_foreign_leases(raw_bridge):
    holder = object()
    with pytest.raises(Exception, match="RAW_DISK_LEASE_INVALID"):
        with resources._admit_raw_disk_lease(True, holder, 4096, raw_bridge.directory):
            pytest.fail("bool is not a lease")
    cm = resources.reserve_incremental_disk(4096, directory=raw_bridge.directory, minimum_free_bytes=1)
    with cm as lease:
        with pytest.raises(Exception, match="RAW_DISK_LEASE_INVALID"):
            with resources._admit_raw_disk_lease(lease, holder, 4095, raw_bridge.directory):
                pytest.fail("wrong bound")
        with resources._admit_raw_disk_lease(lease, holder, 4096, raw_bridge.directory):
            pass
        with pytest.raises(Exception, match="RAW_DISK_LEASE_INVALID"):
            with resources._admit_raw_disk_lease(lease, object(), 4096, raw_bridge.directory):
                pytest.fail("another holder")
    with pytest.raises(Exception, match="RAW_DISK_LEASE_INVALID"):
        with resources._admit_raw_disk_lease(lease, holder, 4096, raw_bridge.directory):
            pytest.fail("dead lease")
    foreign = resources._DiskCapacity()
    with foreign.hold_bytes(4096, directory=raw_bridge.directory, minimum_free_bytes=1) as lease:
        with pytest.raises(Exception, match="RAW_DISK_LEASE_INVALID"):
            with resources._admit_raw_disk_lease(lease, holder, 4096, raw_bridge.directory):
                pytest.fail("foreign ledger")


def test_raw_bridge_different_device_starts_no_factory(raw_bridge):
    holder, native, scope = raw_bridge.pending()
    native.expected_device += 1
    with pytest.raises(ValueError, match="different device"):
        holder._admit_disk()
    assert "factory" not in raw_bridge.events and native.charged
    assert scope.owns(holder) and not holder._lease.closed
    holder.close()
    assert scope.closed and not native.charged


def test_raw_bridge_short_disk_budget_retains_pending_before_factory(raw_bridge, monkeypatch):
    holder, native, scope = raw_bridge.pending()
    monkeypatch.setattr(raw_bridge.disk, "observe", lambda *args:
                        resources.DiskResources(raw_bridge.device, 4096, 0, 1))
    with pytest.raises(Exception, match="DISK_CAPACITY_CHANGED"):
        holder._admit_disk()
    assert not raw_bridge.disk._owners and "factory" not in raw_bridge.events
    assert scope.owns(holder) and holder._owner is native and native.charged
    for _ in range(2):
        with pytest.raises(Exception, match="DISK_CAPACITY_CHANGED"):
            scope.drain()
    assert raw_bridge.contexts[0].exits == 0 and "release" not in raw_bridge.events


def test_raw_native_missing_authority_refuses_before_disk_reservation(monkeypatch):
    # Requires the later matching wheel; authored only. No fake expected-pin
    # provider, Python dictionary, payload authority or positive raw activation.
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk",
                        lambda *args, **kwargs: pytest.fail("reservation before authority"))
    scope = _scope()
    with pytest.raises(ValueError):
        replicated_source._prepare_raw_owner(scope, b"{}", b"{}", 1, 2,
                                             bytes([1])*32, bytes([2])*32, bytes([3])*32)
    assert scope.closed


def test_raw_bridge_empty_slot_then_pending_adoption_precedes_reservation(raw_bridge, monkeypatch):
    scope = _scope()
    native = raw_bridge.native()

    def prepare():
        assert scope._payload._owner is None and scope.owns(scope._payload)
        assert raw_bridge.events == []
        return native

    reserve = replicated_source.reserve_incremental_disk

    def reservation(amount, *, directory):
        assert scope._payload._owner is native
        return reserve(amount, directory=directory)

    monkeypatch.setattr(raw_bridge.native, "_raw_prepare", staticmethod(prepare), raising=False)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", reservation)
    holder = replicated_source._prepare_raw_owner(scope)
    assert holder._owner is native and scope.owns(holder)
    holder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["entry", "start"])
async def test_raw_bridge_to_thread_cancellation_keeps_slot_until_producer_drained(raw_bridge, monkeypatch, phase):
    scope = _scope()
    ready, unblock, done = threading.Event(), threading.Event(), threading.Event()
    start = raw_bridge.native._raw_start

    def pause():
        ready.set()
        assert unblock.wait(5)

    if phase == "entry":
        enter = raw_bridge.cm.__enter__

        def blocked_enter(cm):
            lease = enter(cm)  # genuine reservation acquired, NOT returned yet
            pause()
            return lease

        monkeypatch.setattr(raw_bridge.cm, "__enter__", blocked_enter)
    else:
        def blocked_start(native, device):
            start(native, device)
            pause()

        monkeypatch.setattr(raw_bridge.native, "_raw_start", blocked_start)
    monkeypatch.setattr(raw_bridge.native, "_raw_prepare", staticmethod(raw_bridge.native), raising=False)

    def prepare():
        try:
            replicated_source._prepare_raw_owner(scope)  # result deliberately lost
        finally:
            done.set()

    task = asyncio.create_task(asyncio.to_thread(prepare))
    try:
        assert await asyncio.to_thread(ready.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert scope.owns(scope._payload) and scope._payload._owner.charged
        assert raw_bridge.contexts[0].exits == 0
        for _ in range(3):
            with pytest.raises(ValueError, match="raw host operation in flight"):
                scope.drain()
        assert scope._payload._cleanup_failure is None and not scope.closed and not scope._draining
        assert "retire" not in raw_bridge.events and "release" not in raw_bridge.events
    finally:
        # Cancellation is NOT producer retirement. Join/drain thread BEFORE slot.
        unblock.set()
        assert await asyncio.to_thread(done.wait, 5)
        scope.drain()
    assert scope.closed and raw_bridge.contexts[0].exits == 1


@pytest.mark.parametrize("phase", ["publication", "factory", "adoption", "reservation", "entry", "start"])
def test_raw_bridge_early_drain_defers_entire_preparation_without_cleanup(raw_bridge, monkeypatch, phase):
    scope = _scope()
    ready, resume = threading.Event(), threading.Event()
    results, errors = [], []

    def pause():
        ready.set()
        assert resume.wait(5), "producer was not released"

    def prepare_native():
        native = raw_bridge.native()
        if phase == "factory":
            pause()
        return native

    monkeypatch.setattr(raw_bridge.native, "_raw_prepare", staticmethod(prepare_native), raising=False)
    if phase == "publication":
        retain = resources._CleanupSlot.retain

        def published(slot, payload, retire):
            retain(slot, payload, retire)
            if slot is scope:
                pause()  # holder published, constructor has not returned

        monkeypatch.setattr(resources._CleanupSlot, "retain", published)
    elif phase == "adoption":
        retain_native = replicated_source._RawPublisherOwner._retain

        def adopted(holder, native):
            retain_native(holder, native)
            pause()  # pending owner adopted, admission not called yet

        monkeypatch.setattr(replicated_source._RawPublisherOwner, "_retain", adopted)
    elif phase == "reservation":
        reserve = replicated_source.reserve_incremental_disk

        def reserved(amount, *, directory):
            cm = reserve(amount, directory=directory)
            pause()  # CM made, not returned/entered yet
            return cm

        monkeypatch.setattr(replicated_source, "reserve_incremental_disk", reserved)
    elif phase == "entry":
        enter = raw_bridge.cm.__enter__

        def acquired(cm):
            lease = enter(cm)
            pause()  # real ledger acquired, __enter__ has NOT returned
            return lease

        monkeypatch.setattr(raw_bridge.cm, "__enter__", acquired)
    elif phase == "start":
        start = raw_bridge.native._raw_start

        def started(native, device):
            start(native, device)
            pause()

        monkeypatch.setattr(raw_bridge.native, "_raw_start", started)

    def produce():
        try:
            results.append(replicated_source._prepare_raw_owner(scope))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=produce)
    thread.start()
    try:
        assert ready.wait(5), "producer did not reach pause"
        holder = scope._payload
        assert scope.owns(holder) and holder._busy and not holder._closed
        for _ in range(3):
            with pytest.raises(ValueError, match="raw host operation in flight"):
                scope.drain()
        assert scope.owns(holder) and holder._cleanup_failure is None and not scope._draining
        if phase not in ("publication", "factory"):
            assert holder._owner is not None and holder._owner.charged
        assert all(cm.exits == 0 for cm in raw_bridge.contexts)
        assert not {"retire", "directory_removed", "lease_exit", "release"}.intersection(raw_bridge.events)
        if phase == "entry":
            assert holder._disk_cm is raw_bridge.contexts[0]
            assert holder._lease is None and not holder._entered
            assert len(raw_bridge.disk._owners) == 1
            lease = next(iter(raw_bridge.disk._owners.values()))
            assert type(lease) is resources.DiskLease and not lease.closed
            observed = raw_bridge.disk.observe(raw_bridge.directory, 1)
            assert observed.pending_disk_bytes == 4096 and observed.backed_disk_bytes == 0
    finally:
        resume.set()
        thread.join(5)
        assert not thread.is_alive(), "producer must join before legitimate cleanup"
    assert errors == [] and results == [holder]
    assert scope.owns(holder) and holder._cleanup_failure is None and not holder._busy
    scope.drain()
    scope.drain()
    assert raw_bridge.contexts[0].exits == 1
    assert raw_bridge.events[-4:] == ["retire", "directory_removed", "lease_exit", "release"]
    assert scope.closed and not raw_bridge.disk._owners and not raw_bridge.capacity._cleanup


@pytest.mark.parametrize("operation", ["append", "finish"])
def test_raw_bridge_outstanding_native_operation_defers_drain(raw_bridge, monkeypatch, operation):
    holder, native, scope = raw_bridge.pending()
    holder._admit_disk()
    ready, resume = threading.Event(), threading.Event()
    errors = []
    method = f"_raw_{operation}"
    original = getattr(raw_bridge.native, method)

    def borrowed(owner, *args):
        ready.set()
        assert resume.wait(5), "native borrow was not released"
        return original(owner, *args)

    monkeypatch.setattr(raw_bridge.native, method, borrowed)

    def produce():
        try:
            if operation == "append":
                holder.append(0, 0, b"x")
            else:
                holder.finish()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=produce)
    thread.start()
    try:
        assert ready.wait(5)
        for _ in range(3):
            with pytest.raises(ValueError, match="raw host operation in flight"):
                scope.drain()
        with pytest.raises(ValueError, match="raw host operation in flight"):
            holder.append(0, 0, b"x")
        with pytest.raises(ValueError, match="raw host operation in flight"):
            holder.finish()
        assert scope.owns(holder) and holder._cleanup_failure is None and native.charged
        assert native.files and not holder._lease.closed
        assert holder._lease._backing is None and holder._disk_cm.exits == 0
        assert "retire" not in raw_bridge.events and "release" not in raw_bridge.events
    finally:
        resume.set()
        thread.join(5)
        assert not thread.is_alive(), "outstanding native borrow must join"
    assert errors == []
    holder.close()
    assert scope.closed and not native.charged and raw_bridge.contexts[0].exits == 1
    assert raw_bridge.events[-4:] == ["retire", "directory_removed", "lease_exit", "release"]


@pytest.mark.parametrize("acquired", [False, True])
def test_raw_bridge_baseexception_during_cm_entry_stays_sticky(raw_bridge, monkeypatch, acquired):
    class AcquisitionAborted(BaseException):
        pass

    scope, native = _scope(), raw_bridge.native()
    error = AcquisitionAborted("ambiguous reservation entry")
    enter = raw_bridge.cm.__enter__

    def interrupted(cm):
        if acquired:
            enter(cm)  # genuine lease stays held by CM/ledger, not holder._lease
        raise error

    monkeypatch.setattr(raw_bridge.cm, "__enter__", interrupted)
    monkeypatch.setattr(raw_bridge.native, "_raw_prepare", staticmethod(lambda: native), raising=False)
    with pytest.raises(AcquisitionAborted) as caught:
        replicated_source._prepare_raw_owner(scope)
    holder = scope._payload
    assert caught.value is error and holder._cleanup_failure is error
    assert error.__notes__ == ["raw host cleanup failed: AcquisitionAborted"]
    assert holder._lease is None and not holder._entered and not holder._busy
    assert scope.owns(holder) and holder._owner is native and native.charged
    for _ in range(3):
        with pytest.raises(AcquisitionAborted) as repeated:
            scope.drain()
        assert repeated.value is error
    assert raw_bridge.contexts[0].exits == 0
    assert not {"factory", "retire", "lease_exit", "release"}.intersection(raw_bridge.events)
    assert len(raw_bridge.disk._owners) == int(acquired)
    if acquired:
        lease = next(iter(raw_bridge.disk._owners.values()))
        assert type(lease) is resources.DiskLease and not lease.closed
        assert raw_bridge.disk.observe(raw_bridge.directory, 1).pending_disk_bytes == 4096


def test_raw_bridge_exit_raises_after_real_exit_never_retries_or_releases(raw_bridge, monkeypatch):
    holder, native, scope = raw_bridge.pending()
    holder._admit_disk()
    lease, cm = holder._lease, holder._disk_cm
    error = RuntimeError("exit raised after real release")
    exit_cm = raw_bridge.cm.__exit__

    def released_then_raised(context, *args):
        exit_cm(context, *args)
        raise error

    monkeypatch.setattr(raw_bridge.cm, "__exit__", released_then_raised)
    primary = ValueError("original ingestion error")
    with pytest.raises(ValueError) as caught:
        try:
            raise primary
        except BaseException as original:
            holder._close_preserving(original)
            raise
    assert caught.value is primary
    assert primary.args == ("original ingestion error",)
    assert primary.__notes__ == ["raw host cleanup failed: RuntimeError"]
    for _ in range(3):
        with pytest.raises(RuntimeError) as repeated:
            scope.drain()
        assert repeated.value is error
    assert holder._cleanup_failure is error and scope.owns(holder)
    assert holder._owner is native and native.charged and lease.closed
    assert cm.exits == 1 and not raw_bridge.disk._owners
    assert raw_bridge.events.count("retire") == raw_bridge.events.count("lease_exit") == 1
    assert "release" not in raw_bridge.events
    observed = raw_bridge.disk.observe(raw_bridge.directory, 1)
    assert observed.reserved_disk_bytes == observed.pending_disk_bytes == 0


@pytest.mark.parametrize("choices", [{}, {"weights": "public"},
    {"weights": "public", "range_policy": "checked"}])
def test_expanded_omitted_and_explicit_keep_historical_identity(choices):
    implicit = experiment_for(**choices)
    explicit = experiment_for(**choices, storage="expanded")
    assert implicit.canonical_bytes() == explicit.canonical_bytes()
    assert implicit.configuration_digest() == explicit.configuration_digest()
    assert implicit.pipeline.digest() == explicit.pipeline.digest()
    assert "storage" not in implicit.pipeline.components["linear"].params
    assert execution_policy(implicit) == execution_policy(explicit)
    assert storage_policy(implicit) == "expanded"


def test_historical_component_spec_golden():
    assert PumaThreePartyDecoder().to_spec() == {
        "component": "pllm/puma-three-party-decoder/v1", "params": {}}
    assert PumaThreePartyDecoder(weights="public", range_policy="checked").to_spec() == {
        "component": "pllm/puma-three-party-decoder/v1",
        "params": {"weights": "public", "range_policy": "checked", "max_abs_activation": 256}}


def test_compact_roundtrip_binds_composition_not_arithmetic_policy():
    expanded = experiment_for(weights="public", range_policy="checked")
    compact = experiment_for(weights="public", range_policy="checked", storage="compact")
    restored = Experiment.from_spec(json.loads(compact.canonical_bytes()))
    assert restored.canonical_bytes() == compact.canonical_bytes()
    assert compact.configuration_digest() != expanded.configuration_digest()
    assert compact.pipeline.digest() != expanded.pipeline.digest()
    assert execution_policy(compact) == execution_policy(expanded)
    assert execution_binding(compact) == execution_binding(expanded)
    assert "storage" not in execution_binding(compact)
    assert resolve_runtime_composition(compact.pipeline).numeric_profile == resolve_runtime_composition(expanded.pipeline).numeric_profile
    assert storage_policy(restored) == "compact"
    component = compact.pipeline.components["linear"]
    with pytest.raises((AttributeError, TypeError)):
        component.storage = "expanded"
    assert PumaThreePartyDecoder(**component.get_params()) == component


@pytest.mark.parametrize("storage", [None, False, 1, [], "Compact", "tiled", ""])
def test_unknown_storage_refuses(storage):
    with pytest.raises(ValueError):
        PumaThreePartyDecoder(weights="public", range_policy="checked", storage=storage)


@pytest.mark.parametrize("changes", [{"weights": "private"}, {"range_policy": "static"},
    {"lookup": "tiled"}, {"lookup": "trie"}, {"selection": "tiled"}, {"peer_transport": "streams"}])
def test_unsupported_compact_combination_refuses_before_admission(changes):
    choices = {"weights": "public", "range_policy": "checked", "storage": "compact", **changes}
    with pytest.raises(ValueError):
        experiment_for(**choices)


def test_redundant_storage_component_cannot_bypass_canonical_gate():
    component = ComponentRef(PumaThreePartyDecoder.descriptor.component,
                             {"weights": "public", "storage": "expanded"})
    malformed = SimpleNamespace(pipeline=SimpleNamespace(components={"linear": component}))
    with pytest.raises(ValueError, match="noncanonical"):
        storage_policy(malformed)


def test_storage_cannot_be_supplied_as_arithmetic_wire_permission():
    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")
    with pytest.raises(ValueError, match="execution binding"):
        validate_execution_binding(experiment, {
            "execution": {**execution_binding(experiment), "storage": "compact"}})


@pytest.mark.parametrize("native_boundary", ["missing", "refuses"])
def test_compact_never_uses_legacy_publisher_or_reads_source(monkeypatch, native_boundary):
    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("source_or_legacy")
        raise AssertionError("legacy native publisher/source access forbidden")

    def refusal(*args, **kwargs):
        calls.append("native_refusal")
        raise ValueError("native compact factory refused")

    class Native:
        __new__ = staticmethod(forbidden)

    if native_boundary == "refuses":
        Native._from_compact_source_request = staticmethod(refusal)
    monkeypatch.setattr(replicated_source, "_native", SimpleNamespace(ReplicatedPublisher=Native))
    monkeypatch.setattr(replicated_source.ReplicatedSource, "runtime_contract", forbidden)
    monkeypatch.setattr(replicated_source.ReplicatedSource, "_append_source", forbidden)
    @contextmanager
    def memory(*args):
        calls.append("caller_admitted")
        try:
            yield
        finally:
            calls.append("caller_refunded")

    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", memory)
    source = SimpleNamespace(experiment_bytes=experiment.canonical_bytes(), config_bytes=b"admitted config",
        source_lock_digest="a" * 64, runtime_contract=forbidden, _append_source=forbidden)
    error_type = RuntimeError if native_boundary == "missing" else ValueError
    message = "native publisher capability unavailable" if native_boundary == "missing" else "native compact factory refused"
    with pytest.raises(error_type, match=message):
        scope = _scope()
        try:
            replicated_source.ReplicatedSource.publisher(source, experiment, [2, 2, 3, 3], 2, lambda: False,
                cleanup_scope=scope)
        finally:
            scope.drain()
    assert calls == ([] if native_boundary == "missing" else
                     ["caller_admitted", "native_refusal", "caller_refunded"])


@pytest.mark.parametrize("outputs", [1, 3, True, 2.0])
def test_compact_unsupported_output_count_refuses_before_factory(monkeypatch, outputs):
    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported workload reached native/caller/source admission")

    monkeypatch.setattr(replicated_source, "_native", SimpleNamespace(ReplicatedPublisher=forbidden))
    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", forbidden)
    with pytest.raises(ValueError, match="exactly two outputs"):
        replicated_source.ReplicatedSource.publisher(SimpleNamespace(), experiment, [2, 2, 3, 3], outputs, forbidden)


@pytest.mark.parametrize("failure", [None, "filesystem", "directory_refused", "disk", "append", "cancel"])
def test_compact_actual_factory_signature_and_resource_order_mock(monkeypatch, failure):
    """ABI/ownership mocks only; no source validation, compilation or IDs proof."""
    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")
    encoded = experiment.canonical_bytes()
    config = b"exact admitted client config"
    events = []

    @contextmanager
    def memory(*args):
        events.append("caller_admitted")
        try:
            yield
        finally:
            events.append("caller_refunded")

    @contextmanager
    def disk(amount, *, directory):
        assert amount == 40 and directory == "/actual-native-directory-mocked"
        events.append("disk_admission")
        if failure == "disk":
            raise ValueError("disk refused")
        try:
            yield
        finally:
            events.append("disk_refunded")

    class NativeOwner:
        collection_disk_bytes = 40
        shapes = ()

        @property
        def collection_disk_directory(self):
            if failure == "filesystem":
                raise AttributeError("collection_disk_directory unavailable")
            if failure == "directory_refused":
                raise ValueError("directory refused")
            return "/actual-native-directory-mocked"

        def append(self, *args):
            events.append("append")
            if failure == "append":
                raise ValueError("append refused")

        def seal(self, ids):
            assert ids == b"".join(token.to_bytes(4, "little") for token in [2, 2, 3, 3])
            events.append("seal")
            return json.dumps({"execution": execution_binding(experiment)}).encode()

        def close(self):
            events.append("payload_closed")

    class Native:
        def __new__(cls, *args, **kwargs):
            raise AssertionError("legacy constructor fallback")

        @staticmethod
        def _from_compact_source_request(actual_experiment, actual_config, rows, outputs, source, composition):
            assert events == ["caller_admitted"]
            assert actual_experiment is encoded and actual_config is config
            assert (rows, outputs) == (4, 2)
            assert source == bytes.fromhex("a" * 64)
            assert composition == bytes.fromhex(experiment.pipeline.digest())
            events.append("factory")
            return NativeOwner()

    def runtime(*args):
        assert events.index("disk_admission") < len(events)
        events.append("runtime")

    def append(shapes, callback, cancelled):
        events.append("collection")
        callback()  # IO mock only, not an actual source frame

    source = SimpleNamespace(experiment_bytes=encoded, config_bytes=config,
        source_lock_digest="a" * 64, runtime_contract=runtime, _append_source=append)
    monkeypatch.setattr(replicated_source, "_native", SimpleNamespace(ReplicatedPublisher=Native))
    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", memory)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", disk)
    if failure:
        expected = AttributeError if failure == "filesystem" else ValueError
        message = {"filesystem": "collection_disk_directory unavailable", "directory_refused": "directory refused", "disk": "disk refused",
                   "append": "append refused", "cancel": "import cancelled"}[failure]
        with pytest.raises(expected, match=message):
            replicated_source.ReplicatedSource.publisher(source, experiment, [2, 2, 3, 3], 2,
                lambda: failure == "cancel", cleanup_scope=_scope())
        if failure in {"filesystem", "directory_refused", "disk"}:
            assert "collection" not in events and "runtime" not in events
        assert "seal" not in events
    else:
        owner, manifest = replicated_source.ReplicatedSource.publisher(source, experiment, [2, 2, 3, 3], 2, lambda: False,
            cleanup_scope=_scope())
        assert json.loads(manifest)["execution"] == execution_binding(experiment)
        assert "caller_refunded" not in events and "disk_refunded" not in events
        owner.close()
    assert events.index("payload_closed") < events.index("caller_refunded")
    if "disk_refunded" in events:
        assert events.index("payload_closed") < events.index("disk_refunded")


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["expanded_composition", "unsupported_execution"])
async def test_compact_server_rejects_binding_before_constructor_or_import(monkeypatch, tamper):
    from pllm.runtime import replicated_worker as worker
    from pllm.runtime.delegated_protocol import pack, unpack

    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")
    expanded = experiment_for(weights="public", range_policy="checked")
    events = []

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected binding reached native constructor/import")

    class Native:
        __new__ = staticmethod(forbidden)
        _from_source_request = staticmethod(forbidden)

    class Reservation:
        def __enter__(self):
            events.append("reserved")
            return self

        def __exit__(self, *exc):
            events.append("refunded")

    retained_experiment = experiment.canonical_bytes()
    source = SimpleNamespace(source_lock_digest="a" * 64, config_bytes=b"retained server config",
        experiment_bytes=retained_experiment,
        contract={"weights": "public", "public_weight_snapshot_bytes_per_worker": 1},
        runtime_contract=lambda *args: None, populate_public_worker=forbidden,
        runtime_contract_digest="b" * 64, tokenizer_digest="d" * 64)

    def admit(selected):
        assert selected.canonical_bytes() == retained_experiment
        return source  # admission IO/native contract deliberately mocked

    monkeypatch.setattr(worker, "admit_source", admit)
    monkeypatch.setattr(worker._native, "ReplicatedWorker", Native)
    monkeypatch.setattr(worker, "reserve_incremental_memory", lambda *args: Reservation())
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda *args: SimpleNamespace(free=1 << 40))
    key = "compact-test-client-12345678"
    peers = {"worker_a": ("http://127.0.0.1:1", "outgoing-a-12345678", "incoming-a-12345678"),
             "worker_c": ("http://127.0.0.1:2", "outgoing-c-12345678", "incoming-c-12345678")}
    app = worker.create_replicated_worker_app(experiment, "worker_b", key, peers)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test",
            headers={"authorization": f"Bearer {key}"}) as client:
        epoch = unpack((await client.get("/v1/replicated/manifest")).content)["epoch"]
        value = {"session": list(bytes.fromhex("c" * 64)), "source": list(bytes.fromhex("a" * 64)),
            "composition": list(bytes.fromhex(experiment.pipeline.digest())),
            "config": list(source.config_bytes), "weights": "public", "rows": 2, "outputs": 2,
            "execution": execution_binding(experiment),
            "storage": "expanded"}  # untrusted selector cannot override SERVER compact policy
        if tamper == "expanded_composition":
            value["composition"] = list(bytes.fromhex(expanded.pipeline.digest()))
        else:
            value["execution"]["lookup"] = "tiled"
        response = await client.post("/v1/replicated/admit", content=pack({
            "request": "c" * 64, "epoch": epoch, "manifest": json.dumps(value).encode()}))
        assert response.status_code == 400
        metrics = await client.get("/v1/replicated/metrics")
        metrics.raise_for_status()
        assert metrics.json()["active_sessions"] == 0
        assert events == ["reserved", "refunded"]
        assert "selected" not in response.text and "token_ids" not in response.text


@pytest.mark.parametrize("failure", [None, "disk", "append"])
def test_compact_caller_and_disk_owners_retire_payload_before_refund(monkeypatch, failure):
    events = []
    directory = "/not-opened/native-snapshot-filesystem"

    @contextmanager
    def memory(amount):
        from pllm.runtime.replicated_protocol import REQUEST_MEMORY
        assert amount == REQUEST_MEMORY
        events.append("caller_admitted")
        try:
            yield
        finally:
            events.append("caller_refunded")

    @contextmanager
    def disk(amount, *, directory):
        assert amount == 40 and directory == "/not-opened/native-snapshot-filesystem"
        events.append("disk_admission")
        if failure == "disk":
            raise ValueError("disk refused")
        try:
            yield
        finally:
            events.append("disk_refunded")

    class NativeOwner:
        def append(self):
            events.append("append")
            if failure == "append":
                raise ValueError("append refused")

        def seal(self):
            events.append("seal")

        def close(self):
            events.append("payload_closed")

    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", memory)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", disk)
    owner = replicated_source._CompactPublisherOwner(_scope())
    assert events == ["caller_admitted"]  # must precede future real native factory
    owner._retain(NativeOwner())
    with pytest.raises(ValueError, match="disk not admitted"):
        owner.append()
    try:
        if failure:
            with pytest.raises(ValueError, match=f"{failure} refused"):
                owner._admit_disk(40, directory=directory)
                owner.append()
        else:
            owner._admit_disk(40, directory=directory)
            owner.append()
            owner.seal()
            assert "caller_refunded" not in events and "disk_refunded" not in events
    finally:
        owner.close()
    assert events.index("payload_closed") < events.index("caller_refunded")
    if failure != "disk":
        assert events.index("payload_closed") < events.index("disk_refunded")
    else:
        assert "append" not in events
    previous = events[:]
    owner.close()
    assert events == previous
    with pytest.raises(ValueError, match="retired"):
        owner.append()
    with pytest.raises(ValueError, match="already retained"):
        owner._retain(NativeOwner())


@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
def test_compact_cleanup_failure_preserves_error_and_holds_charges(monkeypatch, error_type):
    refunded = []

    @contextmanager
    def lease(*args, **kwargs):
        try:
            yield
        finally:
            refunded.append(True)

    class NativeOwner:
        refuses = True

        def close(self):
            if self.refuses:
                raise RuntimeError("native close failed")

    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", lease)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", lease)
    owner = replicated_source._CompactPublisherOwner(_scope())
    native = NativeOwner()
    owner._retain(native)
    owner._admit_disk(40, directory="/not-opened")
    original = error_type("collection cancelled or failed")
    owner._close_preserving(original)
    assert not refunded and owner._owner is native
    assert "RuntimeError" in original.__notes__[0]
    with pytest.raises(error_type) as captured:
        raise original
    assert captured.value is original
    native.refuses = False
    owner.close()
    assert len(refunded) == 2


def test_compact_failed_close_survives_caller_and_traceback_gc_until_explicit_drain(monkeypatch):
    """H716 custody regression; mocks actual resource owner, NOT native evidence.

    No external owner/exception/traceback root may make this pass. The actual
    admission ledger must retain the slot, not a per-client cycle alone.
    """
    events = []

    @contextmanager
    def memory(*args):
        events.append("caller_admitted")
        try:
            yield
        finally:
            events.append("caller_refunded")

    @contextmanager
    def disk(*args, **kwargs):
        events.append("disk_admitted")
        try:
            yield
        finally:
            events.append("disk_refunded")

    class NativeOwner:
        refuses = True

        def close(self):
            if self.refuses:
                events.append("close_failed")
                raise RuntimeError("native close failed")
            events.append("payload_retired")

    class ObservedOwner(replicated_source._CompactPublisherOwner):
        # Permit weak observation, not strong custody, without changing the
        # production slots/destructor or substituting its resource methods.
        pass

    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", memory)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", disk)

    def fail_and_drop_all_caller_roots():
        owner = ObservedOwner(_scope())
        native = NativeOwner()
        owner._retain(native)
        owner._admit_disk(40, directory="/not-opened")
        references = weakref.ref(owner), weakref.ref(native)
        original = ValueError("original work failed")
        try:
            raise original
        except ValueError as caught:
            owner._close_preserving(caught)
            preserved = caught is original and str(caught) == "original work failed"
            assert "RuntimeError" in caught.__notes__[0]
        # Exceptions/tracebacks must NOT act as the hidden failed-owner root.
        original.__traceback__ = original.__context__ = original.__cause__ = None
        del original, owner, native
        return references, preserved

    (owner_ref, native_ref), preserved = fail_and_drop_all_caller_roots()
    gc.collect()
    assert preserved
    retained = owner_ref()
    try:
        assert retained is not None and native_ref() is not None, (
            f"failed native payload/leases lost durable custody after GC: {events}")
        assert "caller_refunded" not in events and "disk_refunded" not in events
        assert retained._cleanup_scope in resources._capacity._cleanup.values()
        assert sum(resources._capacity._owners.values()) == resources._capacity._owners[retained._cleanup_scope._token]
        assert sum(resources._capacity._owners.values()) >= 64 << 10
        native_ref().refuses = False
        retained.close()  # explicit successful cleanup, never timeout-as-success
        retained.close()  # idempotent cleanup must not refund twice
        assert events.count("payload_retired") == 1
        assert events.count("caller_refunded") == events.count("disk_refunded") == 1
        assert events.index("payload_retired") < events.index("disk_refunded") < events.index("caller_refunded")
    finally:
        # Test cleanup if an assertion fails after a genuine retained owner is
        # observed; does not establish the missing production handoff itself.
        if retained is not None and retained._owner is not None:
            retained._owner.refuses = False
            retained.close()


def _lifecycle_client():
    from pllm.runtime.delegated_client import DelegatedClient
    # Actual lifecycle methods; admission/tokenizer/model construction absent.
    client = object.__new__(DelegatedClient)
    client._lock = threading.RLock()
    client._closing = False
    client._active = client._compact_cleanup = client._failed_request_cleanup = None
    client._tokenizer_scopes = set()
    client.records = {}
    client._source = object()
    client._result_hook = None
    return client


def test_compact_pending_client_close_drains_outside_locks_and_blocks_new_request(monkeypatch):
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.delegated_client import _Attempt

    events = []
    client = _lifecycle_client()
    attempt = _Attempt(client, {})
    client._active = attempt
    slot = client._new_compact_cleanup_scope(attempt)
    original_source = client._source

    @contextmanager
    def memory(*args):
        try:
            yield
        finally:
            events.append("caller_refunded")

    @contextmanager
    def disk(*args, **kwargs):
        try:
            yield
        finally:
            events.append("disk_refunded")

    class Native:
        refuses = True

        def close(self):
            assert not client._lock._is_owned()
            assert not attempt.lock._is_owned()
            assert not resources._capacity.lock._is_owned()
            if client._closing:
                assert attempt.closed and client._active is None
            assert slot._lock.acquire(blocking=False)
            slot._lock.release()
            if self.refuses:
                raise RuntimeError("native drain refused")
            events.append("payload_retired")

    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", memory)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", disk)
    owner = replicated_source._CompactPublisherOwner(slot)
    native = Native()
    owner._retain(native)
    owner._admit_disk(40, directory="/not-opened")
    original = ValueError("original operation failed")
    owner._close_preserving(original)
    assert "RuntimeError" in original.__notes__[0]
    client._clear_owned_state()
    assert client._source is original_source and client._compact_cleanup is slot
    with pytest.raises(ProtocolError, match="pending compact cleanup"):
        client.events({})
    with pytest.raises(RuntimeError, match="native drain refused"):
        client.close()
    assert attempt.closed and client._active is None
    assert not events and not slot.closed and resources._capacity._cleanup
    native.refuses = False
    client.close()
    client.close()
    assert events == ["payload_retired", "disk_refunded", "caller_refunded"]
    assert client._compact_cleanup is client._source is None
    assert not resources._capacity._owners and not resources._capacity._cleanup


def test_cleanup_slot_control_is_admitted_before_allocation_and_one_short_refuses(monkeypatch):
    capacity = resources._capacity
    original = resources._CleanupSlot
    allocations = []
    quote = capacity._cleanup_control_bytes()

    def allocate(actual_capacity, token):
        assert actual_capacity is capacity
        assert capacity._owners[token] == quote
        allocations.append(token)
        return original(actual_capacity, token)

    monkeypatch.setattr(resources, "_CleanupSlot", allocate)
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=quote - 1))
    with pytest.raises(resources.NetworkError, match="CLIENT_MEMORY_CAPACITY_CHANGED"):
        resources.reserve_cleanup_slot()
    assert not allocations and not capacity._owners
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=quote))
    slot = resources.reserve_cleanup_slot()
    assert len(allocations) == 1 and capacity._cleanup
    slot.drain()
    slot.drain()
    assert not capacity._owners


def test_replicated_transport_preserves_work_error_and_cleans_peers_after_close_failure(monkeypatch):
    from pllm.runtime import replicated_transport as transport
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.delegated_client import _Attempt

    experiment = experiment_for(weights="public", range_policy="checked")
    client = _lifecycle_client()
    attempt = _Attempt(client, {})
    attempt_ref = weakref.ref(attempt)
    client._active = attempt
    original_source = client._source
    original = ValueError("original work exception")
    events = []
    references = []

    def memory(amount):
        reservation = resources.reserve_incremental_memory(amount)
        references.append(weakref.ref(reservation))
        return reservation

    class Publisher:
        refuses = True
        _storage_kind = "compact"  # a flag/echo is NOT independently rooted custody

        def __init__(self):
            references.append(weakref.ref(self))

        def close(self):
            pending = attempt_ref()
            assert not client._lock._is_owned()
            assert pending is None or not pending.lock._is_owned()
            assert not resources._capacity.lock._is_owned()
            if client._closing:
                assert (pending is None or pending.closed) and client._active is None
            if self.refuses:
                events.append("publisher_close_failed")
                raise RuntimeError("publisher cleanup failed")
            events.append("publisher_retired")

    class Peers:
        refuses = True

        def start(self, *args, **kwargs):
            references.append(weakref.ref(self))

        def post(self, *args, **kwargs):
            raise original

        def close(self):
            pending = attempt_ref()
            assert not client._lock._is_owned()
            assert pending is None or not pending.lock._is_owned()
            assert not resources._capacity.lock._is_owned()
            if client._closing:
                assert (pending is None or pending.closed) and client._active is None
            if self.refuses:
                events.append("peers_close_failed")
                raise RuntimeError("peers cleanup failed")
            events.append("peers_retired")

        def audit(self):
            events.append("audit")
            return {}

    source = SimpleNamespace(publisher=lambda *args: (Publisher(), json.dumps({"session": [0] * 32}).encode()))
    monkeypatch.setattr(transport, "reserve_incremental_memory", memory)
    monkeypatch.setattr(transport, "_Peers", Peers)
    with pytest.raises(ValueError) as caught:
        transport.execute_request(experiment, source, [2, 2, 3, 3], 2, {}, 1, attempt)
    assert caught.value is original and len(original.__notes__) == 2
    original.__traceback__ = original.__context__ = original.__cause__ = None
    del caught
    original = None
    gc.collect()
    reservation, peers, publisher = (reference() for reference in references)
    assert reservation is not None and publisher is not None and peers is not None
    assert events == ["publisher_close_failed", "peers_close_failed", "audit"]
    assert resources._capacity._owners[reservation._token] == transport.REQUEST_MEMORY
    assert reservation._token.retained is reservation
    assert client._failed_request_cleanup is reservation
    assert client._compact_cleanup is None
    reservation.__exit__(None, None, None)  # failed close is never successful refund
    assert resources._capacity._owners
    client._clear_owned_state()
    assert client._source is original_source
    with pytest.raises(ProtocolError, match="pending request cleanup"):
        client.events({})
    with pytest.raises(ProtocolError, match="pending request cleanup"):
        client._check_open()  # ordinary admission, not just iterator construction
    with pytest.raises(RuntimeError, match="publisher cleanup failed"):
        client.close()  # active request must drain before BOTH failed owners
    assert attempt.closed and client._active is None
    assert resources._capacity._owners and client._source is original_source
    publisher.refuses = peers.refuses = False
    # Caller/request/reservation/publisher references disappear. Production
    # client teardown, NOT a manual reservation recovery, must do the draining.
    del publisher, peers, reservation, attempt, source
    gc.collect()
    assert all(reference() is not None for reference in references)
    client.close()
    client.close()
    assert events[-2:] == ["publisher_retired", "peers_retired"]
    assert not resources._capacity._owners
    assert client._failed_request_cleanup is client._source is None


@pytest.mark.parametrize("peer_failure", [False, True])
def test_only_actual_rooted_compact_publisher_can_release_outer_after_close_failure(monkeypatch, peer_failure):
    from pllm.runtime import replicated_transport as transport
    from pllm.runtime.delegated_client import _Attempt

    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")
    original = ValueError("work failed")
    client = _lifecycle_client()
    attempt = _Attempt(client, {})
    client._active = attempt
    events, reservations = [], []

    @contextmanager
    def disk(*args, **kwargs):
        try:
            yield
        finally:
            events.append("disk_refunded")

    class Native:
        refuses = True

        def close(self):
            if self.refuses:
                raise RuntimeError("native close failed")
            events.append("payload_retired")

    class Peers:
        refuses = peer_failure

        def start(self, *args, **kwargs):
            pass

        def post(self, *args, **kwargs):
            raise original

        def close(self):
            events.append("peers_close")
            if self.refuses:
                raise RuntimeError("peers close failed")

        def audit(self):
            return {}

    def memory(amount):
        reservation = resources.reserve_incremental_memory(amount)
        reservations.append(reservation)
        return reservation

    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", disk)
    publisher = replicated_source._CompactPublisherOwner(client._new_compact_cleanup_scope(attempt))
    native = Native()
    publisher._retain(native)
    publisher._admit_disk(40, directory="/not-opened")
    before = sum(resources._capacity._owners.values())
    source = SimpleNamespace(publisher=lambda *args: (publisher, json.dumps({"session": [0] * 32}).encode()))
    monkeypatch.setattr(transport, "reserve_incremental_memory", memory)
    monkeypatch.setattr(transport, "_Peers", Peers)
    with pytest.raises(ValueError) as caught:
        transport.execute_request(experiment, source, [2, 2, 3, 3], 2, {}, 1, attempt)
    assert caught.value is original
    outer = reservations[0]
    assert replicated_source._independently_rooted_compact_publisher(publisher)
    assert sum(resources._capacity._owners.values()) == before + (transport.REQUEST_MEMORY if peer_failure else 0)
    assert "disk_refunded" not in events
    if peer_failure:
        assert outer._token.retained is outer and len(outer._failed) == 1
        assert client._failed_request_cleanup is outer
    with pytest.raises(RuntimeError, match="native close failed"):
        client.close()  # tries failed peer too, even when compact cleanup fails
    if peer_failure:
        assert events.count("peers_close") == 2
        assert outer.pending_retirement
        Peers.refuses = False
    native.refuses = False
    client.close()
    client.close()
    assert events.index("payload_retired") < events.index("disk_refunded")
    assert not resources._capacity._owners


def test_failed_outer_handoff_while_client_closing_uses_only_actual_active_seat():
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.delegated_client import _Attempt

    client = _lifecycle_client()
    active, foreign = _Attempt(client, {}), _Attempt(client, {})
    client._active = active
    retired = []

    class Payload:
        def close(self):
            assert active.closed and client._active is None
            assert not client._lock._is_owned() and not active.lock._is_owned()
            assert not resources._capacity.lock._is_owned()
            retired.append(True)

    reservation = resources.reserve_incremental_memory(replicated_source.REQUEST_MEMORY)
    reservation.__enter__()
    reservation.retain_failed((Payload(),))
    # Models close() having signalled cancellation while waiting for the
    # executing producer; its finally must still hand custody to this client.
    client._closing = True
    with pytest.raises(ProtocolError, match="active request seat"):
        client._retain_failed_request_reservation(foreign, reservation)
    assert reservation.pending_retirement and client._failed_request_cleanup is None
    client._retain_failed_request_reservation(active, reservation)
    foreign.close()
    client.close()  # ONLY production boundary invokes the explicit drain
    client.close()
    assert retired == [True] and not resources._capacity._owners
    assert client._failed_request_cleanup is None
