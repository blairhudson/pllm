"""Mocked Python control route only: no models, native execution, HTTP or roles."""
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
import asyncio
import json
import sys
import threading
from types import CellType, FunctionType, SimpleNamespace as NS
import weakref

import pytest

from pllm.runtime import benchmark_failure as diagnostics
from pllm.runtime import delegated_client
from pllm.runtime import delegated_transport
from pllm.runtime import fss_protocol as p
from pllm.runtime import fss_transport as transport
from pllm.runtime.fss_client import FssClient


CONTROLS = ([ (role, "manifest") for role in p.ROLES]
            + [(role, "admit") for role in p.ROLES]
            + [(role, "bind") for role in p.ROLES]
            + [("dealer", "prepare"), ("dealer", "finish")]
            + [(role, "input") for role in p.HOLDERS]
            + [(role, "start") for role in p.HOLDERS]
            + [(role, "finish") for role in p.HOLDERS])
SECRET = "private/prompt/token/source/credential"


def core():
    value = FssClient.__new__(FssClient)
    value._lock = threading.RLock()
    value._active = None
    value._closing = False
    value._fss_failure_observer = None
    value._admit = lambda body: body
    value._tokenizer_scopes = set()
    value.records = OrderedDict()
    value._source = value._result_hook = None
    return value


def runtime_for(owner):
    return NS(_topology=None, _lock=threading.Lock(), _state={"phase": "online"},
              _active_run=object(), _has_started_run=True, _memory_guard=None,
              _client=NS(_core=owner))


@pytest.fixture
def route(monkeypatch):
    owner = core()
    runtime = runtime_for(owner)
    collector = diagnostics.FailureDiagnostics(runtime)
    collector.bind_fss_client(owner)
    attempt = owner.events({})
    state = NS(owner=owner, runtime=runtime, collector=collector, attempt=attempt,
               calls=[], boundaries=[], cleanup=[], fail=None, invalid=None,
               native_fail=None, original=ValueError(SECRET), check_markers=True,
               retirement=[], retirement_errors={})
    record = attempt.record_control_boundary

    def observe(role, action, status):
        record(role, action, status)
        state.boundaries.append(attempt._fss_control_boundary)

    attempt.record_control_boundary = observe

    def retiring(name):
        state.retirement.append((name, collector._snapshot))
        if name in state.retirement_errors:
            raise state.retirement_errors[name]

    record_audit = attempt.record_audit

    def audit(counters):
        retiring("record_audit")
        record_audit(counters)

    attempt.record_audit = audit

    class Peers:
        def __init__(self, connections, timeout, cancelled, **options):
            assert options == {"roles": p.ROLES, "namespace": "fss"}
            assert cancelled is attempt and 0 < timeout <= 23
            self.deadline = transport.time.monotonic() + timeout
            self.live, self.counts = [False] * 3, {}
            self.closed = False
            self.io = delegated_transport._IOOwner()

        def check(self):
            pass

        def post(self, index, action, payload=b"", *, phase, limit):
            role = p.ROLES[index]
            if state.check_markers:
                assert attempt._fss_control_boundary[:3] == (role, action, "entered")
            state.calls.append((role, action))
            assert phase == ("online" if action in ("input", "start") or
                             action == "finish" and role in p.HOLDERS else "setup")
            if state.fail == (role, action):
                raise state.original
            if state.invalid == (role, action):
                return p.pack({"untrusted": SECRET})
            if action == "manifest":
                return p.pack({"schema": "pllm.fss.worker.v1", "role": role,
                    "configuration": "a" * 64, "source_lock": "b" * 64,
                    "tokenizer": "c" * 64, "transport": p.TRANSPORT,
                    "epoch": str(index + 1) * 64})
            if action == "admit":
                return p.pack({"manifest": b"native-manifest"})
            if action == "finish":
                storage = {} if role == "dealer" else {
                    "state": "finished", "encrypted_file_bytes_written": 42,
                    "encrypted_file_bytes_read": 42, "packet_bytes_written": 1,
                    "packet_bytes_read": 1, "records_written": 1, "records_read": 1,
                    "sealed_file_bytes": 42, "sealed_digest": "d" * 64}
                return p.pack({"frame": b"" if role == "dealer" else role.encode(),
                               "peer_bytes": p.peer_bytes(), "storage": storage})
            return p.pack({"prepared" if action == "prepare" else
                            "started" if action == "start" else "accepted": True})

        async def _post(self, index, action, payload, *, phase, limit, timeout):
            # Immediate fake replies retain these tests' deterministic ordering;
            # blocked/concurrent readers have their own finish-barrier controls.
            return self.post(index, action, payload, phase=phase, limit=limit)

        def close(self):
            self.io.close(())
            self.closed = True
            state.cleanup.append(("peers", collector._snapshot))
            retiring("peers.close")

        def audit(self):
            retiring("peers.audit")
            return dict(self.counts)

    class Native:
        def __init__(self, *args):
            if state.native_fail == "client_request":
                raise state.original

        def request_bytes(self):
            return b"request-frame"

        def take_input(self, index):
            assert ("dealer", "finish") in state.calls
            if state.native_fail == "input_share":
                raise state.original
            return b"input-share"

        def finish(self, *frames):
            assert state.calls == CONTROLS
            assert frames == (b"worker_a", b"worker_b")
            if state.native_fail == "reconstruct":
                raise state.original
            return [] if state.native_fail == "incomplete" else [4]

        def close(self):
            state.cleanup.append(("native", collector._snapshot))
            retiring("native.close")

    monkeypatch.setattr(transport, "_Peers", Peers)
    monkeypatch.setattr(transport, "_native", NS(FssClientRequest=Native))
    monkeypatch.setattr(transport, "reserve_incremental_memory", lambda size: nullcontext())
    experiment = NS(configuration_digest=lambda: "a" * 64,
                    pipeline=NS(components={"linear": NS(params={})}))
    contract = {"vocabulary": 8, "capacity": "capsule"}
    source = NS(contract=contract, source_lock_digest="b" * 64, tokenizer_digest="c" * 64,
                runtime_contract=lambda *args: json.dumps(contract).encode())
    state.execute = lambda **options: transport.execute_request(experiment, source, [1, 2, 3], 1,
        {role: (f"http://127.0.0.1:{7300 + i}", str(i) * 32)
         for i, role in enumerate(p.ROLES)}, 23, attempt, **options)
    yield state
    owner.close()


def test_success_exact_control_order_and_native_boundaries(route):
    assert route.execute().selected == (4,)
    assert route.calls == CONTROLS and len(route.calls) == 17
    pairs = ([ ("client", "contract"), ("client", "channels")]
             + CONTROLS[:6] + [("client", "client_request")]
             + CONTROLS[6:11]
             + [("client", "input_share"), CONTROLS[11],
                ("client", "input_share"), CONTROLS[12]]
             + CONTROLS[13:] + [("client", "reconstruct"), ("client", "retire")])
    assert [marker[:3] for marker in route.boundaries] == [
        (*pair, status) for pair in pairs for status in ("entered", "validated")]
    times = [marker[3] for marker in route.boundaries]
    assert times[0] == 0 and times == sorted(times)
    assert route.collector._snapshot is None
    assert all(snapshot is None for _, snapshot in route.cleanup)
    assert type(route.attempt._fss_control_boundary) is tuple
    assert [name for name, _ in route.retirement] == [
        "native.close", "peers.close", "peers.audit", "record_audit"]


RETIREMENT = ("native.close", "peers.close", "peers.audit", "record_audit")


def trace_failure_capture(route):
    observer = route.owner._fss_failure_observer

    def capture(marker):
        observer(marker)
        route.retirement.append(("capture", route.collector._snapshot))

    route.owner._fss_failure_observer = capture


@pytest.mark.parametrize("via_iterator", [False, True])
@pytest.mark.parametrize("step", RETIREMENT)
def test_retirement_failure_after_reconstruction_freezes_at_first_boundary(route, step, via_iterator):
    error = RuntimeError(step)
    route.retirement_errors[step] = error
    trace_failure_capture(route)
    if via_iterator:
        route.owner._produce = lambda attempt: route.execute()
    with pytest.raises(RuntimeError) as caught:
        next(route.attempt) if via_iterator else route.execute()
    assert caught.value is error
    assert error.__context__ is error.__cause__ is None
    assert error.__suppress_context__ is False
    assert route.calls == CONTROLS
    expected = list(RETIREMENT if step == "native.close" else
                    RETIREMENT[:RETIREMENT.index(step) + 1])
    expected.insert(expected.index(step) + 1, "capture")
    assert [name for name, _ in route.retirement] == expected
    failure_index = expected.index("capture")
    assert all(snapshot is None for _, snapshot in route.retirement[:failure_index])
    assert all(snapshot is not None for _, snapshot in route.retirement[failure_index:])
    marker = route.collector.capture()["fss_control_boundary"]
    assert (marker["role"], marker["action"], marker["status"]) == ("client", "retire", "failed")
    assert any(value[:3] == ("client", "reconstruct", "validated") for value in route.boundaries)
    assert any(value[:3] == ("client", "retire", "entered") for value in route.boundaries)
    assert not any(value[:3] == ("client", "retire", "validated") for value in route.boundaries)
    assert (route.owner._active is None) is via_iterator
    if not via_iterator:
        assert route.attempt._fss_control_boundary[:3] == ("client", "retire", "failed")


@pytest.mark.parametrize("step", RETIREMENT)
def test_cleanup_exception_keeps_existing_precedence_over_primary_error(route, step):
    route.fail = ("dealer", "prepare")
    error = RuntimeError(step)
    route.retirement_errors[step] = error
    trace_failure_capture(route)
    with pytest.raises(RuntimeError) as caught:
        route.execute()
    assert caught.value is error
    assert error.__context__ is route.original
    assert error.__cause__ is None and not error.__suppress_context__
    assert route.original.__context__ is None
    expected = list(RETIREMENT if step == "native.close" else
                    RETIREMENT[:RETIREMENT.index(step) + 1])
    assert [name for name, _ in route.retirement] == ["capture", *expected]
    assert all(snapshot is not None for _, snapshot in route.retirement)
    marker = route.collector.capture()["fss_control_boundary"]
    assert (marker["role"], marker["action"], marker["status"]) == ("dealer", "prepare", "failed")


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("step", RETIREMENT[1:])
def test_multiple_cleanup_errors_keep_last_exception_and_original_chain(route, primary, step):
    if primary:
        route.fail = ("dealer", "prepare")
    first, last = RuntimeError("native.close"), LookupError(step)
    route.retirement_errors = {"native.close": first, step: last}
    trace_failure_capture(route)
    with pytest.raises(LookupError) as caught:
        route.execute()
    assert caught.value is last and last.__context__ is first
    assert first.__context__ is (route.original if primary else None)
    assert all(error.__cause__ is None and not error.__suppress_context__ for error in (first, last))
    expected = list(RETIREMENT[:RETIREMENT.index(step) + 1])
    expected.insert(0 if primary else 1, "capture")
    assert [name for name, _ in route.retirement] == expected
    marker = route.collector.capture()["fss_control_boundary"]
    assert marker["action"] == ("prepare" if primary else "retire")


@pytest.mark.parametrize("step", RETIREMENT)
def test_retirement_observer_failure_does_not_replace_cleanup_error(route, step):
    def broken(marker):
        raise KeyboardInterrupt("observer")
    route.owner._fss_failure_observer = broken
    error = RuntimeError(step)
    route.retirement_errors[step] = error
    with pytest.raises(RuntimeError) as caught:
        route.execute()
    assert caught.value is error and error.__context__ is None
    assert route.attempt._fss_control_boundary[:3] == ("client", "retire", "failed")
    assert [name for name, _ in route.retirement] == list(
        RETIREMENT if step == "native.close" else RETIREMENT[:RETIREMENT.index(step) + 1])


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("invalid", [False, True])
def test_failed_or_invalid_control_freezes_before_cleanup(route, control, invalid):
    if invalid:
        route.invalid = control
    else:
        route.fail = control
    with pytest.raises(Exception) as caught:
        route.execute()
    if not invalid:
        assert caught.value is route.original
    assert route.calls == CONTROLS[:CONTROLS.index(control) + 1]
    marker = route.collector.capture()["fss_control_boundary"]
    assert (marker["role"], marker["action"], marker["status"]) == (*control, "failed")
    assert route.cleanup and all(snapshot is not None for _, snapshot in route.cleanup)
    assert not any(value[:3] == (*control, "validated") for value in route.boundaries)
    assert route.attempt._fss_control_boundary[:3] == (*control, "failed")
    route.attempt.close()
    assert route.owner._active is None and route.owner._fss_control_snapshot() is None
    marker["action"] = SECRET  # Returned dictionaries cannot mutate first failure.
    assert route.collector.capture()["fss_control_boundary"]["action"] == control[1]


@pytest.mark.parametrize("action", ["client_request", "input_share", "reconstruct", "incomplete"])
def test_native_client_boundaries_are_not_misattributed_to_control_calls(route, action):
    route.native_fail = action
    with pytest.raises(Exception):
        route.execute()
    marker = route.collector.capture()["fss_control_boundary"]
    assert marker["role"] == "client"
    assert marker["action"] == ("reconstruct" if action == "incomplete" else action)
    assert marker["status"] == "failed"


def test_error_between_boundaries_does_not_relabel_validated_reconstruction(route):
    def broken(manifest):
        raise route.original
    with pytest.raises(ValueError) as caught:
        route.execute(completed_program=broken)
    assert caught.value is route.original
    marker = route.collector.capture()["fss_control_boundary"]
    assert (marker["action"], marker["status"]) == ("reconstruct", "validated")


def test_actual_iterator_failure_captures_before_clearing_active_attempt(route):
    route.fail = ("dealer", "prepare")
    route.owner._produce = lambda attempt: route.execute()
    with pytest.raises(ValueError) as caught:
        next(route.attempt)
    assert caught.value is route.original
    assert route.owner._active is None and route.attempt._fss_control_boundary is None
    assert route.collector.capture()["fss_control_boundary"]["action"] == "prepare"


def test_successful_iterator_retirement_has_no_last_request_marker(route):
    def produce(attempt):
        route.execute()
        yield NS(type="response.completed")
    route.owner._produce = produce
    assert next(route.attempt).type == "response.completed"
    assert route.owner._active is None and route.attempt._fss_control_boundary is None
    assert "fss_control_boundary" not in route.collector.observe()


def test_recording_callback_failure_does_not_change_success(route):
    def broken(*args):
        raise KeyboardInterrupt(SECRET)
    route.check_markers = False
    route.attempt.record_control_boundary = broken
    assert route.execute().selected == (4,)
    assert route.calls == CONTROLS


@pytest.mark.parametrize("fail", [False, True])
def test_diagnostic_clock_failure_preserves_entire_route_result_or_error(route, monkeypatch, fail):
    def broken():
        raise RuntimeError("diagnostic clock unavailable")
    # Replace only this module's diagnostic clock, not transport's deadline clock.
    monkeypatch.setattr(delegated_client, "time", NS(monotonic=broken))
    route.check_markers = False
    if fail:
        route.fail = ("dealer", "prepare")
        with pytest.raises(ValueError) as caught:
            route.execute()
        assert caught.value is route.original
    else:
        assert route.execute().selected == (4,)
    assert route.attempt._fss_control_boundary is None


def test_failure_observer_error_cannot_replace_original(route):
    def broken(marker):
        assert type(marker) is tuple
        raise KeyboardInterrupt(SECRET)
    route.owner._fss_failure_observer = broken
    route.fail = ("dealer", "prepare")
    with pytest.raises(ValueError) as caught:
        route.execute()
    assert caught.value is route.original
    assert route.attempt._fss_control_boundary[:3] == ("dealer", "prepare", "failed")
    assert [name for name, _ in route.cleanup] == ["native", "peers"]


@pytest.mark.parametrize("clock", [None, True, float("nan"), float("inf"), object()])
def test_invalid_diagnostic_clock_clears_marker_only(route, monkeypatch, clock):
    route.attempt.record_control_boundary("dealer", "prepare", "entered")
    monkeypatch.setattr(delegated_client.time, "monotonic", lambda: clock)
    route.attempt.record_control_boundary("worker_a", "start", "entered")
    assert route.attempt._fss_control_boundary is None


def test_clock_exception_and_precleanup_collector_failure_are_optional(route, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(delegated_client.time, "monotonic", broken)
    route.attempt.record_control_boundary("dealer", "prepare", "entered")
    assert route.attempt._fss_control_boundary is None
    monkeypatch.setattr(route.collector, "capture", broken)
    route.attempt.fail_control_boundary()  # Hook failures never escape.


def test_relative_clock_origin_and_backward_clock_are_not_exposed(route, monkeypatch):
    clock = iter([90_000_000.0, 90_000_001.5, 89_999_999.0])
    monkeypatch.setattr(delegated_client.time, "monotonic", lambda: next(clock))
    route.attempt.record_control_boundary("dealer", "prepare", "entered")
    assert route.attempt._fss_control_boundary[3] == 0
    route.attempt.record_control_boundary("dealer", "prepare", "validated")
    assert route.attempt._fss_control_boundary[3] == 1.5
    route.attempt.record_control_boundary("worker_a", "input", "entered")
    assert route.attempt._fss_control_boundary is None


class Hostile:
    def __str__(self):
        raise AssertionError("must not stringify")
    def __eq__(self, other):
        raise AssertionError("must not compare")
    def __iter__(self):
        raise AssertionError("must not iterate")


class CollidingKey(Hostile):
    def __hash__(self):
        return hash("role")


@pytest.mark.parametrize("marker", [
    None, Hostile(), ["dealer", "prepare", "entered", 0],
    (Hostile(), "prepare", "entered", 0), ("dealer", Hostile(), "entered", 0),
    ("dealer", "prepare", Hostile(), 0), ("dealer", "prepare", "entered", Hostile()),
    (SECRET * 1000, "prepare", "entered", 0), ("client", "prepare", "entered", 0),
    ("dealer", "prepare", "entered", -1), ("dealer", "prepare", "entered", True),
    ("dealer", "prepare", "entered", float("nan")),
    ("dealer", "prepare", "entered", 1 << 4096),
    {"role": "dealer", "action": "prepare", "status": "entered", "elapsed_seconds": 0, "payload": SECRET},
    {CollidingKey(): "dealer", "action": "prepare", "status": "entered", "elapsed_seconds": 0},
])
def test_malformed_hostile_marker_is_omitted(marker):
    safe = diagnostics.sanitize_failure_diagnostics({"schema_version": diagnostics.SCHEMA,
                                                    "fss_control_boundary": marker})
    assert "fss_control_boundary" not in safe
    assert SECRET not in json.dumps(safe)


def maximal_failure_value(elapsed):
    marker = ("client", "client_request", "validated", elapsed)
    maximum = (1 << 63) - 1
    return {"schema_version": diagnostics.SCHEMA, "phase": "qualification_retirement",
        "role": "preparation", "elapsed_seconds": elapsed,
        "qualification": dict.fromkeys(("priced", "completed", "retired"), True),
        "spawned_roles": list(diagnostics.ROLES), "ready_roles": list(diagnostics.ROLES),
        "host": {"pressure": "critical", "available_bytes": maximum, "swap_used_bytes": maximum},
        "process_rss": {"latest_total_bytes": maximum, "peak_total_bytes": maximum,
            "sample_age_seconds": elapsed, "sampling_incomplete": False,
            "roles": {role: {"latest_bytes": maximum, "peak_bytes": maximum}
                      for role in diagnostics.ROLES}}, "fss_control_boundary": marker}


@pytest.mark.parametrize("elapsed", [31_536_000.0, 31_535_999.999999996,
    1.2345678901234567e-100, 2.2250738585072014e-308, 0.00010000000000000002, 5e-324])
def test_full_schema_serialization_bound_includes_long_finite_floats(elapsed):
    safe = diagnostics.sanitize_failure_diagnostics(maximal_failure_value(elapsed))
    assert diagnostics.sanitize_failure_diagnostics(safe) == safe
    assert safe["elapsed_seconds"] == safe["process_rss"]["sample_age_seconds"] == elapsed
    assert safe["fss_control_boundary"]["elapsed_seconds"] == elapsed
    for indent in (None, 2):
        assert len(json.dumps(safe, indent=indent, allow_nan=False).encode()) < 4096


def enumerated_metadata_bytes(roots):
    """Object-size inventory only; no allocator/RSS, stacks or aggregate proof.

    Follow declared builtin containers and hook closure, not globals/code or
    weak referents (which would include the existing collector/runtime graph).
    """
    seen = set()

    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        total = sys.getsizeof(value)
        if type(value) is dict:
            return total + sum(size(key) + size(item) for key, item in value.items())
        if type(value) in (tuple, list):
            return total + sum(size(item) for item in value)
        if type(value) is FunctionType:
            return total + size(value.__dict__) + size(value.__closure__)
        if type(value) is CellType:
            return total + size(value.cell_contents)
        assert type(value) in (str, bytes, float, int, bool, type(None), weakref.ReferenceType)
        return total

    return size(roots)


def test_enumerated_marker_hook_and_snapshot_copy_overlap_against_coarse_allowances(route):
    attempt, collector = route.attempt, route.collector
    attempt.record_control_boundary("client", "reconstruct", "entered")
    reader_marker = attempt._fss_control_boundary
    attempt.record_control_boundary("client", "reconstruct", "validated")
    replaced_marker = attempt._fss_control_boundary
    attempt.record_control_boundary("client", "retire", "entered")
    current_marker = attempt._fss_control_boundary
    frozen = collector.capture(observation=maximal_failure_value(1.2345678901234567e-100))
    # Enumerate one writer/reader overlap and materialized snapshot/copy/output
    # objects, NOT deepcopy memo/encoder scratch, allocator overhead, arbitrary
    # concurrent callers, total metadata or RSS. Shared coarse allowances also
    # cover existing other owners; this inventory is not their aggregate proof.
    client_objects = {
        "current_marker": current_marker, "reader_marker": reader_marker,
        "replaced_marker": replaced_marker, "clock_origin": attempt._fss_control_origin,
        "failure_flag": attempt._fss_control_failed,
        "temporary_marker_map": diagnostics.sanitize_fss_control_boundary(current_marker),
    }
    supervisor_objects = {
        "weak_hook_and_closure": route.owner._fss_failure_observer,
        "cached_snapshot": collector._snapshot, "returned_deepcopy": frozen,
        "sanitized_snapshot": diagnostics.sanitize_failure_diagnostics(collector._snapshot),
        "serialized_snapshot": json.dumps(frozen, indent=2).encode(),
    }
    assert frozen is not collector._snapshot
    assert frozen["process_rss"]["roles"] is not collector._snapshot["process_rss"]["roles"]
    assert route.owner._fss_failure_observer.__closure__[0].cell_contents() is collector
    assert enumerated_metadata_bytes(client_objects) < 8 * p.CONTROL_LIMIT
    assert enumerated_metadata_bytes(supervisor_objects) < 1 << 20


def test_new_request_and_startup_cannot_observe_old_success(route):
    route.execute()
    route.attempt.close()
    assert route.owner._fss_control_snapshot() is None
    second = route.owner.events({})
    assert "fss_control_boundary" not in route.collector.observe()
    route.attempt.record_control_boundary("dealer", "prepare", "entered")
    route.attempt.fail_control_boundary()
    assert route.collector._snapshot is None  # Old callbacks have no authority.
    second.record_control_boundary("worker_b", "start", "entered")
    assert route.collector.observe()["fss_control_boundary"]["role"] == "worker_b"
    route.runtime._active_run = None
    route.runtime._state["phase"] = "ready"
    assert "fss_control_boundary" not in route.collector.observe()
    second.close()


def test_timeout_freezes_cached_entered_marker_before_attempt_cleanup(route):
    route.attempt.record_control_boundary("dealer", "prepare", "entered")
    frozen = route.collector.capture()
    route.attempt.close()
    route.runtime._state["phase"] = "ready"
    assert route.collector.capture() == frozen
    assert frozen["fss_control_boundary"]["status"] == "entered"


def test_capture_never_waits_for_execution_client_or_runtime_locks(route):
    route.attempt.record_control_boundary("dealer", "prepare", "entered")
    results = []
    with route.attempt.lock:
        thread = threading.Thread(target=lambda: results.append(route.collector.observe()))
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
    assert results[-1]["fss_control_boundary"]["action"] == "prepare"
    with route.owner._lock:
        thread = threading.Thread(target=lambda: results.append(route.collector.observe()))
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
    assert "fss_control_boundary" not in results[-1]
    with route.runtime._lock:
        assert "fss_control_boundary" not in route.collector.observe()
        route.attempt.fail_control_boundary()  # Explicit cached tuple needs no client lock.
    assert route.collector.capture()["fss_control_boundary"]["status"] == "failed"


def test_contended_first_failure_slot_never_blocks_original_error(route):
    route.fail = ("dealer", "prepare")
    with route.collector._lock:
        with pytest.raises(ValueError) as caught:
            route.execute()
    assert caught.value is route.original and route.collector._snapshot is None


def test_binding_is_weak_and_close_clears_observer():
    owner = core()
    collector = diagnostics.FailureDiagnostics(runtime_for(owner))
    reference = weakref.ref(collector)
    collector.bind_fss_client(owner)
    callback = owner._fss_failure_observer
    del collector
    assert reference() is None
    callback(("dealer", "prepare", "failed", 0))
    owner.close()
    assert owner._fss_failure_observer is None


@pytest.mark.parametrize("broken_hook", [False, True])
def test_dashboard_binds_only_after_client_construction(tmp_path, monkeypatch, broken_hook):
    from pllm.runtime import benchmark_memory, dashboard

    config = dashboard.DashboardConfig("127.0.0.1", 1, "unused", "unit", 1, backend="native")
    runtime = dashboard.DashboardRuntime(config, NS())
    runtime.config = replace(config, experiment=NS(
        pipeline=NS(components={"linear": NS(params={"capacity": "decoder"})})))
    runtime._fss = True
    runtime._delegated = False  # Skip unrelated experiment inspection; no model loads.
    owner = core()
    wrapper = NS(_core=owner)
    collector = diagnostics.FailureDiagnostics(runtime)
    runtime._benchmark_failure_diagnostics = collector
    calls = []
    bind = collector.bind_fss_client

    def observe(current):
        assert current is owner and runtime._client is wrapper
        calls.append("bind")
        if broken_hook:
            raise RuntimeError(SECRET)
        bind(current)

    def client(**options):
        assert owner._fss_failure_observer is None
        calls.append("construct")
        runtime._stopping.set()  # End mocked startup immediately AFTER binding seam.
        return wrapper

    monkeypatch.setattr(collector, "bind_fss_client", observe)
    monkeypatch.setattr(dashboard, "build_roles", lambda *args, **kwargs: NS(
        start=lambda: None, statuses=[], client=client, requires_preparation=False))
    monkeypatch.setattr(benchmark_memory, "execution_memory_admission", lambda *args, **kwargs: {
        "admitted": True, "selected_backend": "native", "estimate": {"provider_peak_bytes": {}}})
    monkeypatch.setattr(benchmark_memory, "MemoryWatchdog", lambda *args: NS(
        start=lambda: None, failure_observation=None))
    monkeypatch.setattr(dashboard.tempfile, "TemporaryDirectory", lambda **kwargs: NS(
        name=str(tmp_path), cleanup=lambda: None))
    asyncio.run(runtime.start())
    assert calls == ["construct", "bind"]
    assert runtime._state["error"].endswith("dashboard stopped during startup")
    assert (owner._fss_failure_observer is None) is broken_hook
    owner.close()
