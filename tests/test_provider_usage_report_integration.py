"""Real supervisor reap -> ordinary SDK adapter -> canonical result serialization.

Only telemetry is real: two quiet stdlib Python children, 8 MiB/25 ms CPU each.
Dashboard HTTP, model/source identities and request timings are explicit doubles;
these tests neither execute inference nor establish model/publication evidence.
No sampler supplies final HWM. Tests retain one raw OS exit tuple per child.
"""
from copy import deepcopy
import json
import os
import select
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import httpx
import pytest

import pllm
from pllm._benchmark_contract import canonical, digest
from pllm._benchmark_worker import Measurement
from pllm.evidence import BenchmarkResult, EvidenceReport
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario
from pllm.metrics.benchmark_runner import _result
from pllm.profiles import MaskedLinearCpu
from pllm.runtime import benchmark_cli, benchmark_memory, benchmark_resources, dashboard
from pllm.runtime import process_usage, servers
from pllm.runtime.benchmark_response_target import response_metrics
from pllm.runtime.benchmark_targets import execute_target


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "wait4") or sys.platform not in {"darwin", "linux"},
    reason="qualified POSIX wait4 owner required")

_ROLES = ("preparation", "inference")
_CHILD = """
import atexit, os, signal, sys, time
def cleanup():
    data = bytearray(8 << 20)
    for offset in range(0, len(data), 4096):
        data[offset] = 1
    end = time.process_time() + .025
    while time.process_time() < end:
        pass
    os.write(1, b'D')
atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
os.write(1, b'R')
os.read(FD, 1)
sys.exit(CODE)
"""


def _record():
    # Deliberately distinct old windows/values: none is lifetime OS evidence.
    return {"status": "completed", "cold": True, "warm": False,
        "model_fingerprint": "b" * 64, "max_output_tokens": 2,
        "tokens": {"authoritative": True, "input_tokens": 3, "output_tokens": 2},
        "generation": {"response_status": "completed", "output_text_digest": "c" * 64},
        "durations": {"full_seconds": .5, "online_seconds": .3, "ttft_seconds": .1,
                      "tokens_per_second": 5, "preparation_seconds": .2},
        "privacy": {"plaintext_prompt_bytes_sent": 0, "plaintext_token_ids_sent": 0,
                    "preparation_requests_during_online": 0},
        "processes": {"preparation": {"rss_peak_bytes": 321, "cpu_seconds": .012}}}


@pytest.fixture
def ordinary(monkeypatch, tmp_path):
    """Keep production launch, close, benchmark, metric and serializer owners."""
    state = NS(mode="normal", raw={}, calls={role: 0 for role in _ROLES}, processes={},
        gates={}, events=[], hold_role=None, exit_available=threading.Event(),
        release=threading.Event(), benchmark_record=_record(), last_request=None)
    topology = servers.LocalTopology.__new__(servers.LocalTopology)
    topology._process_lock, topology._close_lock = threading.Lock(), threading.Lock()
    topology._stopping, topology._closed = threading.Event(), False
    topology._processes, topology._role_ids = state.processes, _ROLES
    topology._child_source_status, topology._logs, topology._local_engine = {}, {}, None
    topology._inference_key = topology._preparation_key = topology._push_key = ""
    topology._role_credentials, topology._peer_credentials = {}, {}
    topology._source_pins = {"fixture": "declaration-only"}
    state.topology = topology

    def retired():
        assert all(p.returncode is not None for p in state.processes.values())
        assert set(state.raw) == set(_ROLES)
        assert topology._source_pins is None
        state.events.append("capacity_released_after_reap")

    topology._capacity_reservations = NS(close=retired)
    runtime = NS(_topology=topology, _client=None, _memory_guard=None,
        _lock=threading.Lock(), _state={"phase": "ready"}, _active_run=None,
        _has_started_run=True, memory_preflight=None, source_lock_digest="a" * 64,
        store=NS(stage_body_snapshot=lambda **_: {}), initial_preparation_audit=lambda: None,
        client_body_placement=lambda: None,
        cold_process_cpu=lambda: {
            "startup": {role: .05 for role in ("client", *_ROLES)},
            "first_response": {role: .125 for role in ("client", *_ROLES)}})
    state.runtime = runtime

    def observe(role, process):
        wait4 = process._usage_wait4

        def owned_wait4(pid, flags):
            assert threading.current_thread() is threading.main_thread()
            assert process._waitpid_lock.locked()
            state.calls[role] += 1
            value = wait4(pid, flags)  # Instrument same reaper; never issue another wait.
            if value[0] == pid:
                assert role not in state.raw
                state.raw[role] = value
                if state.hold_role == role:
                    state.exit_available.set()
                    assert state.release.wait(5)
            return value

        process._usage_wait4 = owned_wait4

    client_class = httpx.Client

    def transport(request):
        if request.url.path == "/api/run":
            state.last_request = json.loads(request.content)["request_id"]
            if state.mode == "cancel":
                raise KeyboardInterrupt
            if state.mode == "failure":
                raise benchmark_cli.LoopbackBenchmarkError("fixture request failed")
            if state.mode in {"nonzero", "signal"}:
                process = state.processes["preparation"]
                if state.mode == "signal":
                    process.send_signal(signal.SIGKILL)
                else:
                    os.write(state.gates["preparation"], b'x')
                process.wait(timeout=5)  # Same supervisor owner, not an outside watcher.
                state.benchmark_record["status"] = "failed"
            return httpx.Response(200, json={})
        if request.url.path == "/api/snapshot":
            return httpx.Response(200, json={"run": {"phase": "ready",
                "run_id": state.last_request, "active_run": None}})
        if request.url.path.startswith("/api/runs/"):
            return httpx.Response(200, json=state.benchmark_record)
        raise AssertionError("unexpected benchmark HTTP request")

    monkeypatch.setattr(benchmark_cli.httpx, "Client", lambda **kw:
        client_class(transport=httpx.MockTransport(transport), **kw))
    monkeypatch.setattr(dashboard, "create_dashboard_app", lambda _: NS(state=NS(dashboard_runtime=runtime)))
    monkeypatch.setattr(benchmark_cli, "_DASHBOARD_PORT", 7300)
    monkeypatch.setattr(benchmark_cli, "_DASHBOARD_TOKEN", "fixture-only")
    monkeypatch.setattr(benchmark_cli, "_wait_for_dashboard_listener", lambda *a: None)
    monkeypatch.setattr(benchmark_cli, "_DashboardHandle", lambda *a: NS(
        start=lambda: None, poll=lambda: None, close=topology.close, diagnostic=lambda: ""))
    # Explicit sampler double: impossible tiny RSS values must stay sampled only.
    monkeypatch.setattr(benchmark_resources, "ProcessMemorySampler", lambda _: NS(
        start=lambda: None, stop=lambda: {"sampled_total_peak_rss_bytes": 123},
        failure_snapshot=lambda: {"latest_total_bytes": 123, "sampling_incomplete": True}))
    monkeypatch.setattr(benchmark_memory, "process_memory", lambda: {"lifetime_peak_rss_bytes": 456})
    monkeypatch.setattr(pllm.metrics, "benchmark_memory", lambda *a, **kw: {"admitted": True})
    experiment = pllm.Experiment("provider-usage", MaskedLinearCpu(pllm.Model.tiny()),
        pllm.Deployment.local(root=str(tmp_path / "roles")), pllm.ExecutionBudget(1, 128, 2))
    target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=2))
    sampling = Sampling(processes=1, memory_bytes=2 << 30, timeout_seconds=5)
    state.case = BenchmarkCase("provider-usage", target)
    state.suite = BenchmarkSuite((state.case,), sampling)
    state.measured = Measurement({"sampling": sampling.to_spec(), "salt": "ab" * 32,
        "inputs": {"prompt": "private-fixture-prompt"}, "directory": str(tmp_path)})

    try:
        for role in _ROLES:
            read_fd, write_fd = os.pipe()
            state.gates[role] = write_fd
            try:
                command = _CHILD.replace("FD", str(read_fd)).replace("CODE", "7" if role == "preparation" else "0")
                process = servers._spawn_role_process([sys.executable, "-I", "-c", command],
                    environment=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    pass_fds=(read_fd,))
                state.processes[role] = process
                observe(role, process)
            finally:
                os.close(read_fd)
            assert select.select([process.stdout], [], [], 5)[0]
            assert process.stdout.read(1) == b'R'
            assert process.final_usage()["coverage"] == "unknown"
        yield state
    finally:
        state.release.set()
        try:
            topology.close()
        finally:
            for descriptor in state.gates.values():
                os.close(descriptor)
            for process in state.processes.values():
                if process.returncode is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                try:
                    assert process.stderr.read() == b''
                finally:
                    process.stdout.close()
                    process.stderr.close()
                with pytest.raises(ChildProcessError):
                    os.waitpid(process.pid, os.WNOHANG)  # Leak assertion after owner's reap only.


def _execute(state):
    identity = state.case.target.digest
    execute_target(state.case.target, state.measured)
    assert state.case.target.digest == identity
    return state.measured.details["report"]


def _serialized(state, *, status="completed", failure=None):
    m = state.measured
    m.finalize_workload(state.case.target, completed=status == "completed")
    attributes = {"fixture": "report-integration-not-an-installed-build"}
    # Existing runner serializer, not a replacement worker/runner or evidence archive.
    result = _result(state.case, state.suite, 0, "d" * 64,
        {"attributes": attributes, "digest": digest(attributes)},
        {"status": status, "failure_code": failure, "details": m.details,
         "comparison": m.comparison, "metrics": m.metrics, "samples": m.samples,
         "fixture": m.fixture, "oracle": m.oracle, "admission": m.admission,
         "limitations": m.limitations})
    return BenchmarkResult.from_dict(json.loads(result.canonical_bytes())).to_dict()


def test_real_owner_finalization_reaches_sdk_and_canonical_report_without_changing_old_windows(ordinary, monkeypatch):
    report = _execute(ordinary)
    value = report["provider_lifetime_usage"]
    assert value["complete"] is True
    assert "including reaped descendants" in value["cpu_scope"]
    assert "individual" in value["peak_rss_scope"] and "not simultaneous" in value["peak_rss_scope"]
    assert value["aggregate_peak_rss_bytes"] is None
    assert ordinary.events == ["capacity_released_after_reap"]
    for role in _ROLES:
        record = value["roles"][role]
        _, status, raw = ordinary.raw[role]
        assert os.WIFEXITED(status) and record["normal_exit"] is True
        assert record["source"] == "wait4" and record["coverage"] == "os_process_exit"
        assert record["user_cpu_seconds"] == raw.ru_utime
        assert record["system_cpu_seconds"] == raw.ru_stime
        assert record["cpu_seconds"] == raw.ru_utime + raw.ru_stime > 0
        assert record["peak_rss_bytes"] == raw.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        assert record["peak_rss_bytes"] > 0  # No fixed RSS floor or aggregate-memory claim.
        assert record["peak_rss_bytes"] not in (123, 321, 456)
        assert ordinary.processes[role].stdout.read() == b'D'  # Actual exit includes atexit.
    assert report["native_process_memory"] == {"sampled_total_peak_rss_bytes": 123}
    old_report = deepcopy(report)
    del old_report["provider_lifetime_usage"]
    baseline = Measurement({"sampling": ordinary.suite.sampling.to_spec(), "salt": "ab" * 32,
        "inputs": {}, "directory": "."})
    old_bytes = canonical(old_report)
    response_metrics(old_report, baseline)
    assert canonical(old_report) == old_bytes
    added = [row for row in ordinary.measured.metrics if ".lifetime_" in row["id"]]
    assert len(added) == 4
    assert [row for row in ordinary.measured.metrics if row not in added] == baseline.metrics
    metrics = {row["id"]: row for row in ordinary.measured.metrics}
    assert metrics["preparation.peak_rss"]["value"] == 321
    assert metrics["preparation.peak_rss"]["window"] == "request_windows"
    assert metrics["preparation.cold_cpu"]["value"] == .125
    assert metrics["preparation.cold_cpu"]["window"] == "role_birth_through_first_response"
    for row in added:
        role = row["role"]
        assert row["window"] == "role_birth_through_exit" and row["origin"] == "measured"
        if row["id"].endswith("lifetime_cpu"):
            assert row["unit"] == "seconds" and row["value"] == value["roles"][role]["cpu_seconds"]
            assert "including_reaped_descendants" in row["measurement_scope"]
        else:
            assert row["unit"] == "bytes" and row["value"] == value["roles"][role]["peak_rss_bytes"]
            assert "not_simultaneous_provider_tree_peak" in row["measurement_scope"]
    historical = Measurement({"sampling": ordinary.suite.sampling.to_spec(), "salt": "ab" * 32,
        "inputs": ordinary.measured.inputs, "directory": str(ordinary.measured.directory)})
    with monkeypatch.context() as patch:
        patch.setattr(benchmark_cli, "run_loopback_benchmark", lambda **kw: old_report)
        execute_target(ordinary.case.target, historical)
    assert historical.metrics == baseline.metrics
    assert historical.comparison == ordinary.measured.comparison
    assert historical.details["report"] == old_report
    serialized = _serialized(ordinary)
    assert serialized["status"] == "completed"
    assert serialized["execution"]["details"]["report"]["provider_lifetime_usage"] == value
    assert serialized["configuration_digest"] == ordinary.case.target.digest
    assert "private-fixture-prompt" not in json.dumps(serialized)


@pytest.mark.parametrize("scope", ["missing_source", "remote", "container", "missing_owner"])
def test_ordinary_report_preserves_unknown_coverage_without_borrowing_local_usage(ordinary, scope):
    if scope == "missing_source":
        ordinary.runtime.source_lock_digest = None
    elif scope == "remote":
        ordinary.runtime._topology = NS()  # No authority to borrow local PID/exit metadata.
    elif scope == "container":
        ordinary.runtime.memory_preflight = {"selected_backend": "docker"}
    else:
        ordinary.topology._processes = {"preparation": ordinary.processes["preparation"]}
        ordinary.topology._role_ids = ("preparation",)
        # Retire omitted child through its existing owner too, before any refund.
        servers._stop_role_processes((ordinary.processes["inference"],))
    report = _execute(ordinary)
    section = report["provider_lifetime_usage"]
    assert section["complete"] is False and section["aggregate_peak_rss_bytes"] is None
    missing = ("inference",) if scope == "missing_owner" else _ROLES
    for role in missing:
        record = section["roles"][role]
        assert record["coverage"] == "unknown" and record["source"] is None
        assert record["cpu_seconds"] is None and record["peak_rss_bytes"] is None
        metrics = [row for row in ordinary.measured.metrics if row["role"] == role and ".lifetime_" in row["id"]]
        assert len(metrics) == 2 and all(row["origin"] == "not_available" for row in metrics)
    # Canonical ordinary report retains nulls, including absent source identity.
    assert EvidenceReport(canonical(report)).to_dict()["provider_lifetime_usage"] == section


@pytest.mark.parametrize("mode", ["failure", "cancel", "nonzero", "signal", "privacy_failure"])
def test_actual_final_usage_does_not_turn_failed_or_cancelled_execution_into_valid_measurement(ordinary, mode):
    ordinary.mode = mode
    if mode == "privacy_failure":
        ordinary.benchmark_record["privacy"]["plaintext_token_ids_sent"] = 1
    with pytest.raises(benchmark_cli.LoopbackBenchmarkError):
        _execute(ordinary)
    assert set(ordinary.raw) == set(_ROLES)
    assert ordinary.topology.closed and ordinary.measured.oracle["outcome"] == "not_run"
    assert "report" not in ordinary.measured.details
    assert not any(".lifetime_" in row["id"] for row in ordinary.measured.metrics)
    if mode in {"nonzero", "signal"}:
        assert ordinary.processes["preparation"].returncode == (7 if mode == "nonzero" else -signal.SIGKILL)
        assert ordinary.processes["preparation"].final_usage()["normal_exit"] is False
    result = _serialized(ordinary, status="cancelled" if mode == "cancel" else "failed", failure="fixture_failed")
    assert result["status"] != "completed"
    assert "provider_lifetime_usage" not in json.dumps(result)


def test_concurrent_report_snapshot_cannot_reap_or_publish_half_a_final_record(ordinary):
    ordinary.hold_role = "preparation"
    observed, errors = {}, []

    def reader():
        try:
            assert ordinary.exit_available.wait(5)
            calls = dict(ordinary.calls)
            for _ in range(3):
                observed.update(process_usage.provider_lifetime_usage(ordinary.topology,
                    ("client", *_ROLES), local_cpu_loopback=True, source_present=True))
                record = observed["roles"]["preparation"]
                assert record["reason"] == "reap_in_progress"
                assert record["cpu_seconds"] is record["peak_rss_bytes"] is None
                assert ordinary.calls == calls
        except BaseException as error:
            errors.append(error)
        finally:
            ordinary.release.set()

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        report = _execute(ordinary)
    finally:
        ordinary.release.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert observed["complete"] is False
    assert report["provider_lifetime_usage"]["complete"] is True
