"""Deterministic mocked receipts on the real joined I/O owner; no HTTP/native work."""
import asyncio
from collections import OrderedDict
from contextlib import nullcontext
import gc
import json
import threading
from types import SimpleNamespace as NS

import pytest

from pllm.runtime.client import ProtocolError
from pllm.runtime import delegated_transport
from pllm.runtime import fss_protocol as p
from pllm.runtime import fss_transport as transport
from pllm.runtime.fss_client import FssClient


def receipt(role):
    storage = {} if role == "dealer" else {
        "state": "finished", "encrypted_file_bytes_written": 42,
        "encrypted_file_bytes_read": 42, "packet_bytes_written": 1,
        "packet_bytes_read": 1, "records_written": 1, "records_read": 1,
        "sealed_file_bytes": 42, "sealed_digest": "d" * 64}
    return p.pack({"frame": b"" if role == "dealer" else role.encode(),
                   "peer_bytes": p.peer_bytes(), "storage": storage})


@pytest.fixture
def barrier(monkeypatch):
    owner = FssClient.__new__(FssClient)
    owner._lock = threading.RLock()
    owner._active = None
    owner._closing = False
    owner._admit = lambda body: body
    owner._tokenizer_scopes = set()
    owner.records = OrderedDict()
    owner._source = owner._result_hook = None
    state = NS(owner=owner, mode="success", failing=1, error=ValueError("source failure"),
               calls=[], markers=[], frozen=None, cleanup=[], read_cancelled=set(),
               active=set(), arrived=set(), validated=set(), reconstructed=0,
               pending_at_close=None, retirement_errors={}, bad_callbacks=False,
               late_return=False, invalid=False, peers=None, withheld=None,
               both_started=threading.Event(), resist_cancel=False,
               late_cleanup="cancel", cleanup_error=RuntimeError("late cleanup"),
               pending_owned_at_close=None, remaining_at_close=None,
               cancel_markers=[], second_cancelled=set(), receipt_roles=[])

    def frozen(marker):
        if state.frozen is None:
            state.frozen = marker
        if state.bad_callbacks:
            raise KeyboardInterrupt("optional observer")

    owner._fss_failure_observer = frozen

    def new_attempt():
        attempt = owner.events({})
        record = attempt.record_control_boundary

        def observe(role, action, status):
            if state.bad_callbacks:
                raise KeyboardInterrupt("optional marker")
            record(role, action, status)
            state.markers.append((role, action, status))
            if role in p.HOLDERS and (action, status) == ("finish", "validated"):
                state.validated.add(role)
                if state.mode.startswith("hold_") and role == p.HOLDERS[1 - state.failing]:
                    state.withheld = (set(state.arrived), set(state.validated), state.reconstructed)
                    state.peers.release.set()

        attempt.record_control_boundary = observe
        return attempt

    state.attempt = new_attempt()
    state.new_attempt = new_attempt

    class Client:
        async def aclose(self):
            pass

    class Peers(delegated_transport._Peers):
        def start(self, connections, timeout, cancelled, **options):
            self._close_gate = threading.Lock()
            self._close_succeeded = False
            assert options == {"roles": p.ROLES, "namespace": "fss"}
            self.roles, self.namespace = p.ROLES, "fss"
            self.deadline = transport.time.monotonic() + timeout
            self.cancelled = cancelled
            self.io = delegated_transport._IOOwner()
            self.io.start()
            self.closed = False
            self.clients = [Client() for _ in p.ROLES]
            self.live, self.counts = [False] * 3, {}
            self.calls_known = self.server_known = True
            self.both = self.release = None
            state.peers = self
            run = self.io.run

            def interrupted(coroutine, *, timeout):
                finishing = coroutine.cr_code.co_qualname == "_finish_receipts.<locals>.owned"
                if state.mode == "expired_dispatch" and finishing:
                    self.deadline = transport.time.monotonic() - 1
                if state.mode == "interrupt" and finishing:
                    future = asyncio.run_coroutine_threadsafe(coroutine, self.io.loop)
                    assert state.both_started.wait(1)
                    future.cancel()
                    raise state.error
                if state.resist_cancel and finishing:
                    # Independent watchdog: an unbounded child join must fail
                    # this test quickly, not consume the 23-second request.
                    timeout = min(timeout, .5)
                return run(coroutine, timeout=timeout)

            self.io.run = interrupted

        async def _post(self, index, action, payload, *, phase, limit, timeout):
            role = p.ROLES[index]
            state.calls.append((role, action))
            assert timeout >= 0
            if action == "manifest":
                return p.pack({"schema": "pllm.fss.worker.v1", "role": role,
                    "configuration": "a" * 64, "source_lock": "b" * 64,
                    "tokenizer": "c" * 64, "transport": p.TRANSPORT,
                    "epoch": str(index + 1) * 64})
            if action == "admit":
                return p.pack({"manifest": b"native-manifest"})
            if action != "finish":
                return p.pack({"prepared" if action == "prepare" else
                               "started" if action == "start" else "accepted": True})
            if role == "dealer":
                return receipt(role)
            assert phase == "online" and limit == p.CONTROL_LIMIT and payload == b""
            assert state.calls[:11] == [
                *((r, "manifest") for r in p.ROLES),
                *((r, "admit") for r in p.ROLES),
                *((r, "bind") for r in p.ROLES),
                ("dealer", "prepare"), ("dealer", "finish")]
            assert state.calls[11:15] == [
                ("worker_a", "input"), ("worker_b", "input"),
                ("worker_a", "start"), ("worker_b", "start")]
            if self.both is None:
                self.both, self.release = asyncio.Event(), asyncio.Event()
            state.active.add(index)
            state.arrived.add(index)
            if len(state.arrived) == 2:
                self.both.set()
                state.both_started.set()
            try:
                await self.both.wait()
                if state.mode == "fail":
                    if index == state.failing:
                        if state.invalid:
                            return p.pack({"invalid": True})
                        raise state.error
                    await asyncio.Future()
                if state.mode == "cancel":
                    self.cancelled.cancel()
                    await asyncio.Future()
                if state.mode == "deadline":
                    self.deadline = transport.time.monotonic()
                    await asyncio.Future()
                if state.mode == "interrupt":
                    await asyncio.Future()
                if state.mode.startswith("hold_") and index == state.failing:
                    await self.release.wait()
                return receipt(role)
            except asyncio.CancelledError:
                state.read_cancelled.add(index)
                state.cancel_markers.append(state.frozen)
                if state.resist_cancel:
                    # No external Event/timer roots this Future. The request's
                    # I/O owner must retain the Task across bounded drain/GC.
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        state.second_cancelled.add(index)
                        if state.late_cleanup == "return":
                            return receipt(role)
                        if state.late_cleanup == "error":
                            raise state.cleanup_error
                        raise
                if state.late_return:
                    return receipt(role)
                raise
            finally:
                state.active.remove(index)

        def close(self):
            state.cleanup.append(("peers", state.frozen))
            try:
                super().close()
            finally:
                if "peers" in state.retirement_errors:
                    raise state.retirement_errors["peers"]

        def audit(self):
            return dict(self.counts)

    class Native:
        def __init__(self, *args):
            pass

        def request_bytes(self):
            return b"request-frame"

        def take_input(self, index):
            assert ("dealer", "finish") in state.calls
            return b"input-share"

        def finish(self, *frames):
            assert frames == (b"worker_a", b"worker_b")
            assert state.arrived == {0, 1} and not state.active
            if not state.bad_callbacks:
                assert state.validated == set(p.HOLDERS)
            assert state.frozen is None
            state.reconstructed += 1
            return [4]

        def close(self):
            async def pending():
                return asyncio.all_tasks() - {asyncio.current_task()}
            if state.resist_cancel:
                gc.collect()
            state.pending_at_close = state.peers.io.run(pending(), timeout=1)
            roots = state.peers.io._fss_finish_pending
            state.pending_owned_at_close = state.pending_at_close <= roots
            state.remaining_at_close = state.peers.deadline - transport.time.monotonic()
            state.cleanup.append(("native", state.frozen))
            if "native" in state.retirement_errors:
                raise state.retirement_errors["native"]

    monkeypatch.setattr(transport, "_Peers", Peers)
    monkeypatch.setattr(transport, "_native", NS(FssClientRequest=Native))
    class Reservation(nullcontext):
        # This joined-I/O fixture has no admission ledger. Real failed-token
        # custody is covered separately in test_fss_failed_retirement_custody.
        def retain_failed(self, owners):
            pass

    monkeypatch.setattr(transport, "reserve_incremental_memory", lambda size: Reservation())
    original_receipt = transport._receipt

    def parsed(raw, role, **options):
        state.receipt_roles.append(role)
        return original_receipt(raw, role, **options)

    monkeypatch.setattr(transport, "_receipt", parsed)
    experiment = NS(configuration_digest=lambda: "a" * 64,
                    pipeline=NS(components={"linear": NS(params={})}))
    contract = {"vocabulary": 8, "capacity": "capsule"}
    source = NS(contract=contract, source_lock_digest="b" * 64, tokenizer_digest="c" * 64,
                runtime_contract=lambda *args: json.dumps(contract).encode())

    def execute():
        return transport.execute_request(experiment, source, [1, 2, 3], 1,
            {role: (f"http://127.0.0.1:{7300 + i}", str(i) * 32)
             for i, role in enumerate(p.ROLES)}, 23, state.attempt)

    state.execute = execute
    yield state
    if state.peers is not None and not state.peers.closed:
        state.peers.close()
    owner.close()


def assert_retired(state):
    assert not state.active
    if state.resist_cancel:
        assert state.pending_at_close and state.pending_owned_at_close
    else:
        assert state.pending_at_close == set()
    assert [name for name, _ in state.cleanup] == ["native", "peers"]
    assert state.peers.closed and state.peers.io.closed
    assert not state.peers.io.thread.is_alive()
    assert not asyncio.all_tasks(state.peers.io.loop)
    assert not state.peers.io._fss_finish_pending


@pytest.mark.parametrize("failing", [0, 1])
def test_failing_role_interrupts_blocked_other_and_freezes_actual_role(barrier, failing):
    barrier.mode, barrier.failing = "fail", failing
    with pytest.raises(ValueError) as caught:
        barrier.execute()
    assert caught.value is barrier.error
    assert caught.value.__context__ is caught.value.__cause__ is None
    assert barrier.frozen[:3] == (p.HOLDERS[failing], "finish", "failed")
    assert barrier.arrived == {0, 1} and barrier.read_cancelled == {1 - failing}
    assert barrier.reconstructed == 0
    assert all(marker is not None for _, marker in barrier.cleanup)
    assert not barrier.peers.calls_known and not barrier.peers.server_known
    assert_retired(barrier)


@pytest.mark.parametrize("failing", [0, 1])
@pytest.mark.parametrize("late_cleanup", ["cancel", "return", "error"])
def test_first_failure_bounds_resistant_drain_and_global_close_owns_late_read(
        barrier, monkeypatch, failing, late_cleanup):
    monkeypatch.setattr(delegated_transport, "_IO_DRAIN_TIMEOUT", .05)
    barrier.mode, barrier.failing, barrier.resist_cancel = "fail", failing, True
    barrier.late_cleanup = late_cleanup
    with pytest.raises(ValueError) as caught:
        barrier.execute()
    assert caught.value is barrier.error
    assert caught.value.__context__ is caught.value.__cause__ is None
    assert barrier.remaining_at_close > 20
    assert barrier.cancel_markers == [barrier.frozen]
    assert barrier.frozen[:3] == (p.HOLDERS[failing], "finish", "failed")
    assert barrier.read_cancelled == barrier.second_cancelled == {1 - failing}
    assert barrier.receipt_roles == ["dealer"]  # Late bytes never reach validation.
    assert not barrier.validated and barrier.reconstructed == 0
    assert all(marker is barrier.frozen for _, marker in barrier.cleanup)
    assert_retired(barrier)


@pytest.mark.parametrize("held", [0, 1])
def test_both_receipts_validate_before_reconstruction_and_order_is_role_order(barrier, held):
    barrier.mode, barrier.failing = "hold_one", held
    assert barrier.execute().selected == (4,)
    assert barrier.reconstructed == 1 and barrier.frozen is None
    assert barrier.withheld == ({0, 1}, {p.HOLDERS[1 - held]}, 0)
    assert not barrier.read_cancelled
    assert_retired(barrier)


@pytest.mark.parametrize("failing", [0, 1])
def test_malformed_completion_also_interrupts_other_role(barrier, failing):
    barrier.mode, barrier.failing, barrier.invalid = "fail", failing, True
    with pytest.raises(ProtocolError, match="completion envelope"):
        barrier.execute()
    assert barrier.frozen[:3] == (p.HOLDERS[failing], "finish", "failed")
    assert barrier.read_cancelled == {1 - failing} and barrier.reconstructed == 0
    assert_retired(barrier)


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
def test_cancel_or_expired_absolute_deadline_drains_both_reads(barrier, mode):
    barrier.mode = mode
    with pytest.raises(ProtocolError, match="cancelled or timed out"):
        barrier.execute()
    assert barrier.read_cancelled == {0, 1} and barrier.reconstructed == 0
    assert all(marker is not None for _, marker in barrier.cleanup)
    assert_retired(barrier)


def test_caller_side_interruption_is_fenced_before_native_retirement(barrier):
    barrier.mode = "interrupt"
    with pytest.raises(ValueError) as caught:
        barrier.execute()
    assert caught.value is barrier.error and barrier.reconstructed == 0
    assert barrier.read_cancelled == {0, 1}
    assert_retired(barrier)


def test_expiry_during_dispatch_cannot_start_a_new_finish_read(barrier):
    barrier.mode = "expired_dispatch"
    with pytest.raises(ProtocolError, match="cancelled or timed out"):
        barrier.execute()
    assert not barrier.arrived and barrier.reconstructed == 0
    assert not any((role, "finish") in barrier.calls for role in p.HOLDERS)
    assert_retired(barrier)


@pytest.mark.parametrize("failing", [0, 1])
def test_late_response_after_failure_cannot_validate_or_reconstruct(barrier, failing):
    barrier.mode, barrier.failing, barrier.late_return = "fail", failing, True
    with pytest.raises(ValueError) as caught:
        barrier.execute()
    assert caught.value is barrier.error and barrier.reconstructed == 0
    assert not barrier.validated
    assert barrier.peers.live == [False, False, False]
    assert barrier.frozen[:3] == (p.HOLDERS[failing], "finish", "failed")
    assert_retired(barrier)


@pytest.mark.parametrize("fail", [False, True])
def test_optional_callback_errors_preserve_result_or_source_error(barrier, fail):
    barrier.bad_callbacks = True
    if fail:
        barrier.mode = "fail"
        with pytest.raises(ValueError) as caught:
            barrier.execute()
        assert caught.value is barrier.error and barrier.reconstructed == 0
    else:
        assert barrier.execute().selected == (4,)
    assert_retired(barrier)


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("step", ["native", "peers"])
def test_cleanup_failure_keeps_original_precedence_and_first_capture(barrier, primary, step):
    if primary:
        barrier.mode = "fail"
    error = RuntimeError(step)
    barrier.retirement_errors[step] = error
    with pytest.raises(RuntimeError) as caught:
        barrier.execute()
    assert caught.value is error
    assert error.__context__ is (barrier.error if primary else None)
    assert error.__cause__ is None and not error.__suppress_context__
    assert barrier.frozen[:3] == (("worker_b", "finish", "failed") if primary else
                                  ("client", "retire", "failed"))
    assert_retired(barrier)


def test_combined_cleanup_failures_keep_last_error_and_original_chain(barrier):
    barrier.mode = "fail"
    first, last = RuntimeError("native"), LookupError("peers")
    barrier.retirement_errors = {"native": first, "peers": last}
    with pytest.raises(LookupError) as caught:
        barrier.execute()
    assert caught.value is last and last.__context__ is first
    assert first.__context__ is barrier.error
    assert barrier.frozen[:3] == ("worker_b", "finish", "failed")
    assert_retired(barrier)


def test_next_attempt_has_fresh_marker_and_old_callbacks_have_no_authority(barrier):
    barrier.mode = "fail"
    with pytest.raises(ValueError):
        barrier.execute()
    old = barrier.attempt
    old.close()
    barrier.attempt = barrier.new_attempt()
    assert barrier.attempt._fss_control_boundary is None
    frozen = barrier.frozen
    old.record_control_boundary("worker_a", "finish", "entered")
    old.fail_control_boundary()
    assert barrier.frozen is frozen and barrier.attempt._fss_control_boundary is None
    barrier.mode, barrier.frozen = "success", None
    barrier.calls.clear()
    barrier.markers.clear()
    barrier.cleanup.clear()
    barrier.arrived.clear()
    barrier.read_cancelled.clear()
    assert barrier.execute().selected == (4,)
    assert barrier.frozen is None and not barrier.attempt._fss_control_failed
    assert_retired(barrier)
