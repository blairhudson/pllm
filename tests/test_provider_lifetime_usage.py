"""Provider exit observations use the existing sole supervisor reaper only."""
from contextlib import nullcontext
import json
import os
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import pytest

from pllm.runtime import process_usage as usage
from pllm.runtime import servers


def owner(wait4=None, *, scale=1):
    """Real Popen wait/poll methods, no child/native execution in mock tests."""
    process = usage.UsagePopen.__new__(usage.UsagePopen)
    process._child_created = False
    process._waitpid_lock = threading.Lock()
    process.pid, process.returncode = 12345, None
    process.args = ["private-fixture-not-reportable"]
    process._usage_wait4 = wait4
    process._usage_scale = scale
    process._usage_final = None
    process._usage_unavailable = False
    return process


def topology(processes):
    result = servers.LocalTopology.__new__(servers.LocalTopology)
    result._process_lock = threading.Lock()
    result._processes = processes
    return result


def report(processes, roles=("client", "worker_a"), **kwargs):
    return usage.provider_lifetime_usage(topology(processes), roles,
        local_cpu_loopback=kwargs.get("local_cpu_loopback", True),
        source_present=kwargs.get("source_present", True))


@pytest.mark.parametrize("scale", [1, 1024])
@pytest.mark.parametrize("first_wait", ["poll", "wait"])
def test_exact_child_wait_is_single_reaper_and_normalizes_os_units(scale, first_wait):
    calls = []
    process = owner(scale=scale)

    def wait4(pid, flags):
        assert process._waitpid_lock.locked()
        calls.append((pid, flags))
        return pid, 0, NS(ru_utime=1.25, ru_stime=.5, ru_maxrss=4096)

    process._usage_wait4 = wait4
    assert process.final_usage()["reason"] == "not_reaped"
    assert not calls  # Reading metadata must not invoke a syscall or child watcher.
    assert getattr(process, first_wait)(**({"timeout": 1} if first_wait == "wait" else {})) == 0
    assert process.wait(timeout=1) == process.poll() == 0
    assert len(calls) == 1 and calls[0][0] == process.pid  # Never waitpid(-1).
    value = report({"worker_a": process})
    assert value["complete"] is True
    assert value["roles"]["worker_a"] == {
        "coverage": "os_process_exit", "reason": None, "exit_code": 0,
        "user_cpu_seconds": 1.25, "system_cpu_seconds": .5, "cpu_seconds": 1.75,
        "peak_rss_bytes": 4096 * scale, "source": "wait4", "normal_exit": True}
    assert value["aggregate_peak_rss_bytes"] is None
    assert "private-fixture" not in json.dumps(value)
    assert str(process.pid) not in json.dumps(value)


def test_nonblocking_poll_cannot_promote_last_live_snapshot_to_exit_usage():
    process = owner(lambda pid, flags: (0, 0, NS(ru_utime=999, ru_stime=999, ru_maxrss=999)))
    assert process.poll() is None
    assert process._usage_final is None
    assert process.final_usage()["peak_rss_bytes"] is None
    with pytest.raises(subprocess.TimeoutExpired):
        process.wait(timeout=0)
    assert report({"worker_a": process})["complete"] is False


def test_poll_and_wait_share_original_popen_lock_no_competing_reap():
    entered, finish = threading.Event(), threading.Event()
    calls = []
    process = owner()

    def wait4(pid, flags):
        calls.append((pid, flags))
        entered.set()
        assert finish.wait(2)
        return pid, 0, NS(ru_utime=0, ru_stime=0, ru_maxrss=0)

    process._usage_wait4 = wait4
    result = []
    waiter = threading.Thread(target=lambda: result.append(process.wait(timeout=2)))
    waiter.start()
    try:
        assert entered.wait(2)
        assert process.poll() is None
        assert process.final_usage()["reason"] == "reap_in_progress"
        assert len(calls) == 1
    finally:
        finish.set()
        waiter.join(3)
    assert not waiter.is_alive() and result == [0]
    assert process.final_usage()["cpu_seconds"] == 0  # Real zero, not missing.
    assert process.poll() == 0 and len(calls) == 1


@pytest.mark.parametrize("first_wait", ["poll", "wait"])
def test_outside_reaper_or_ignored_sigchld_keeps_usage_unknown(first_wait):
    def missing(*_):
        raise ChildProcessError

    process = owner(missing)
    assert getattr(process, first_wait)() == 0  # Existing Popen status convention.
    value = process.final_usage()
    assert value["coverage"] == "unknown" and value["reason"] == "exit_usage_unavailable"
    assert value["cpu_seconds"] is value["peak_rss_bytes"] is value["normal_exit"] is None


def test_signalled_exit_is_final_os_usage_not_clean_retirement():
    process = owner(lambda pid, flags: (
        pid, signal.SIGKILL, NS(ru_utime=.25, ru_stime=.125, ru_maxrss=100)))
    assert process.wait(timeout=1) == -signal.SIGKILL
    value = report({"worker_a": process})
    assert value["complete"] is True  # Complete OS observation, not protocol success.
    assert value["roles"]["worker_a"]["normal_exit"] is False
    assert value["roles"]["worker_a"]["cpu_seconds"] == .375
    assert any("not clean protocol" in line for line in value["limitations"])


def test_traced_child_stop_status_is_not_exit_usage():
    stopped = (signal.SIGSTOP << 8) | 0x7f
    process = owner(lambda pid, flags: (
        pid, stopped, NS(ru_utime=1, ru_stime=2, ru_maxrss=999)))
    process.poll()
    assert process.final_usage()["coverage"] == "unknown"
    assert process._usage_final is None


@pytest.mark.parametrize("values", [(float("nan"), 0, 4), (-1, 0, 4), (0, float("inf"), 4),
                                    (0, 0, -1), (0, 0, 1.5)])
def test_invalid_final_os_values_never_become_measurements(values):
    process = owner()
    process.returncode, process._usage_final = 0, values
    assert process.final_usage()["reason"] == "invalid_os_usage"


def test_scope_and_foreign_process_owners_stay_unknown(monkeypatch):
    process = owner(lambda *_: pytest.fail("report attempted to reap"))
    process.returncode, process._usage_final = 0, (1, 2, 4096)
    assert report({"worker_a": process}, local_cpu_loopback=False)["roles"]["worker_a"]["cpu_seconds"] is None
    assert report({"worker_a": process}, source_present=False)["roles"]["worker_a"]["reason"] == "source_identity_unavailable"
    # Existing async/multiprocessing/container-style owners may expose tags and
    # metrics, but cannot acquire this raw-Popen reaping authority from those tags.
    foreign = NS(pid=process.pid, returncode=0, final_usage=lambda: pytest.fail("foreign owner"))
    assert report({"worker_a": foreign})["roles"]["worker_a"]["coverage"] == "unknown"
    value = usage.provider_lifetime_usage(NS(_processes={"worker_a": process}),
        ("worker_a",), local_cpu_loopback=True, source_present=True)
    assert value["complete"] is False
    assert usage.provider_popen_factory(lambda: None) is not usage.UsagePopen
    monkeypatch.delattr(usage.os, "wait4")
    assert usage.provider_popen_factory(usage._POPEN) is usage._POPEN


def test_all_roles_must_be_observed_no_sum_of_independent_high_water_marks():
    process = owner()
    process.returncode, process._usage_final = 0, (0, 0, 100)
    value = report({"worker_a": process}, ("client", "worker_a", "worker_b"))
    assert value["complete"] is False
    assert value["roles"]["worker_b"]["peak_rss_bytes"] is None
    assert value["aggregate_peak_rss_bytes"] is None
    empty = report({}, ("client",))
    assert not empty["complete"] and empty["roles"] == {}
    with pytest.raises(ValueError, match="bounded"):
        report({}, ["worker_a"] * 10000)
    with pytest.raises(ValueError, match="bounded"):
        report({}, ["worker_a", "secret-request-id"])
    maximum = report({role: process for role in usage._PROVIDERS}, usage._PROVIDERS)
    assert len(json.dumps(maximum).encode()) < 8192  # Fixed public metadata, not a payload/history.


def test_supervisor_reaps_after_child_cleanup_before_capacity_refund(monkeypatch):
    events = []
    process = owner()
    live = [True]

    def wait4(pid, flags):
        assert "refund" not in events
        if live[0]:
            return 0, 0, None
        events.extend(("native_source_job_retired", "os_exit_usage"))
        return pid, 0, NS(ru_utime=.5, ru_stime=.25, ru_maxrss=8192)

    def terminate(pid, sig):
        assert pid == process.pid
        events.append("signal")
        live[0] = False

    def refund():
        assert process.returncode == 0 and process._usage_final is not None
        events.append("refund")

    process._usage_wait4 = wait4
    t = topology({"worker_a": process})
    t._stopping, t._close_lock = threading.Event(), threading.Lock()
    t._role_ids, t._closed = ("worker_a",), False
    t._capacity_reservations = NS(close=refund)
    t._child_source_status, t._logs, t._local_engine = {}, {}, None
    t._inference_key = t._preparation_key = t._push_key = ""
    t._role_credentials, t._peer_credentials, t._source_pins = {}, {}, object()
    monkeypatch.setattr(servers.os, "killpg", terminate)
    t.close()
    assert events == ["signal", "native_source_job_retired", "os_exit_usage", "refund"]
    assert t._source_pins is None and t._capacity_reservations is None
    t.close()  # Repeated close/polls cannot consume another child's usage.
    assert len(events) == 4


@pytest.mark.skipif(not hasattr(os, "wait4"), reason="POSIX wait4 required")
def test_common_launcher_reports_true_post_exit_usage_including_atexit():
    # Lightweight ordinary Python only: no PLLM native build/import/model/server.
    code = ("import atexit\n"
            "def cleanup():\n"
            "    data=bytearray(4*1024*1024)\n"
            "    for i in range(0,len(data),4096): data[i]=1\n"
            "atexit.register(cleanup)\n")
    process = servers._spawn_role_process([sys.executable, "-I", "-c", code],
        environment=dict(os.environ), discard_output=True)
    try:
        assert type(process) is usage.UsagePopen
        raw, wait4 = [], process._usage_wait4

        def observe(pid, flags):
            value = wait4(pid, flags)  # Same sole reaper, not a second observation.
            if value[0] == pid:
                raw.append(value)
            return value

        process._usage_wait4 = observe
        assert process.wait(timeout=5) == 0
        record = process.final_usage()
        assert record["coverage"] == "os_process_exit"
        assert record["cpu_seconds"] >= 0 and record["peak_rss_bytes"] > 0
        assert len(raw) == 1 and os.WIFEXITED(raw[0][1])
        assert record["cpu_seconds"] == raw[0][2].ru_utime + raw[0][2].ru_stime
        assert record["peak_rss_bytes"] == raw[0][2].ru_maxrss * process._usage_scale
        assert process.poll() == 0 and process.final_usage() == record
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_ordinary_benchmark_reports_usage_only_after_handle_retirement(monkeypatch):
    from pllm.runtime import benchmark_cli as benchmark, benchmark_memory, dashboard

    events = []
    process = owner()
    runtime = NS(_topology=topology({"preparation": process, "inference": owner()}),
        _client=None, _memory_guard=None, memory_preflight=None,
        source_lock_digest="1" * 64, store=NS(stage_body_snapshot=lambda **_: {}),
        initial_preparation_audit=lambda: None, cold_process_cpu=lambda: {},
        client_body_placement=lambda: None)

    def close():
        events.append("handle_retired")
        for child in runtime._topology._processes.values():
            child.returncode, child._usage_final = 0, (1, 2, 1024)

    monkeypatch.setattr(benchmark_memory, "process_memory", lambda: {})
    monkeypatch.setattr(dashboard, "create_dashboard_app", lambda _: NS(state=NS(dashboard_runtime=runtime)))
    monkeypatch.setattr(benchmark, "_DASHBOARD_PORT", 7300)
    monkeypatch.setattr(benchmark, "_DashboardHandle", lambda *a: NS(start=lambda: None, close=close))
    monkeypatch.setattr(benchmark, "_wait_for_dashboard_listener", lambda *a: None)
    monkeypatch.setattr(benchmark, "_wait_for_ready", lambda *a: None)
    monkeypatch.setattr(benchmark.httpx, "Client", lambda **k: nullcontext(object()))
    monkeypatch.setattr(benchmark, "_run_once", lambda *a, **k: {})
    monkeypatch.setattr(benchmark, "build_loopback_report", lambda **k: {
        "configuration": {}, "checks": {"passed": True}})
    from pllm.runtime import benchmark_resources
    monkeypatch.setattr(benchmark_resources, "ProcessMemorySampler", lambda _: NS(
        start=lambda: None, stop=lambda: {"sampled_total_peak_rss_bytes": 999999}))
    original = usage.provider_lifetime_usage

    def observed(*a, **k):
        assert events == ["handle_retired"]
        return original(*a, **k)

    monkeypatch.setattr(usage, "provider_lifetime_usage", observed)
    result = benchmark._run_loopback_benchmark(model="unused", model_id=None, tiny=False,
        prompt="private-not-for-report", max_output_tokens=1, warmups=0,
        repetitions=1, timeout_seconds=1)
    record = result["provider_lifetime_usage"]
    assert record["complete"] is True
    assert record["roles"]["preparation"]["peak_rss_bytes"] == 1024
    assert result["native_process_memory"]["sampled_total_peak_rss_bytes"] == 999999
    assert "private-not-for-report" not in json.dumps(record)
