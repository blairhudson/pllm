"""Owner-classified RSS accounting, not inference or physical-memory evidence."""
from copy import deepcopy
import json
import os
import resource
import select
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import psutil
import pytest

from pllm._benchmark_contract import Sampling, canonical
from pllm._benchmark_worker import Measurement
from pllm.runtime import benchmark_memory, benchmark_resources, servers
from pllm.runtime.client_memory_scope import (
    ClientMemoryScope, HELPERS, client_memory_report, client_memory_metrics,
)
from pllm.runtime.benchmark_response_target import response_metrics
from test_provider_usage_report_integration import ordinary, _execute, _serialized


_SAMPLER = benchmark_resources.ProcessMemorySampler
_PROCESS_MEMORY = benchmark_memory.process_memory


def _owner(pid):
    owner = subprocess.Popen.__new__(subprocess.Popen)
    owner._child_created = False
    owner.pid = pid
    owner.args = ["private-argv-not-reportable"]
    return owner


@pytest.fixture
def fixed(monkeypatch):
    state = NS(values={101: (1., 100), 201: (2., 60), 301: (3., 9000)}, visits=[])
    monkeypatch.setattr(benchmark_resources.os, "getpid", lambda: 101)

    def observe(pid):
        state.visits.append(pid)
        if pid not in state.values:
            raise psutil.NoSuchProcess(pid)
        birth, rss = state.values[pid]
        return NS(create_time=lambda: birth, memory_info=lambda: NS(rss=rss))

    monkeypatch.setattr(benchmark_resources.psutil, "Process", observe)
    state.scope = ClientMemoryScope()
    state.runtime = NS(_topology=None, _benchmark_client_memory_owner=state.scope)
    return state


def test_fixed_pass_sum_is_not_sum_of_individual_peaks_and_provider_children_stay_excluded(fixed):
    fixed.scope.bind_helper(HELPERS[0], _owner(201), created_at=2.)
    fixed.scope.seal()
    fixed.runtime._topology = NS(_process_lock=threading.Lock(), _processes={"inference": _owner(301)})
    sampler = _SAMPLER(fixed.runtime)
    sampler.sample()  # Client 100 + helper 60 = 160, provider 9000 excluded.
    fixed.values.update({101: (1., 40), 201: (2., 100)})
    sampler.sample()  # Client 40 + helper 100 = 140, never sum maxima 100+100.
    value = client_memory_report(sampler, {"lifetime_peak_rss_bytes": 700})
    assert value["sampled_peak_rss_sum_bytes"] == 160
    assert value["latest"]["same_pass_rss_sum_bytes"] == 140
    assert value["sampled_peak_rss_lower_bound_bytes"] == 100
    assert value["sampling_coverage"] == "all_declared_members_observed"
    assert value["latest"]["missing_members"] == []
    assert set(value["latest"]["members"]) == {"client", HELPERS[0]}
    assert fixed.visits == [101, 301, 201, 201] * 2
    assert sampler.peak == 9100  # Legacy client+provider diagnostic unchanged.
    assert "not_simultaneous_peak_bound" in value["sampled_peak_rss_sum_kind"]
    assert value["individual_client_hwm"]["bytes"] == 700 if sys.platform in {"linux", "darwin"} else value["individual_client_hwm"]["bytes"] is None
    assert value["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] is None
    assert value["physical_used_ram_peak_bytes"] is None


def test_no_owner_manifest_means_unknown_helpers_not_zero_or_complete(fixed):
    # Providers being this process's children is not a client-helper declaration.
    fixed.runtime._topology = NS(_process_lock=threading.Lock(), _processes={"preparation": _owner(301)})
    sampler = _SAMPLER(fixed.runtime)
    sampler.sample()
    value = client_memory_report(sampler, {})
    assert value["membership_declared"] is False
    assert value["sampling_coverage"] == "unknown_or_partial"
    assert value["sampled_peak_rss_sum_bytes"] is None
    assert value["sampled_peak_rss_lower_bound_bytes"] == 100
    assert value["latest"]["missing_members"] == ["undeclared_client_helpers"]
    assert fixed.visits == [101, 301]


@pytest.mark.parametrize("unavailable", ["missing", "remote", "replaced", "provider", "client_missing"])
def test_missing_reused_remote_or_provider_members_never_become_zero(fixed, unavailable):
    if unavailable == "remote":
        fixed.scope.expect_unavailable_helper(HELPERS[0], remote=True)
    else:
        fixed.scope.bind_helper(HELPERS[0], _owner(201), created_at=2.)
        if unavailable == "missing":
            del fixed.values[201]
        elif unavailable == "replaced":
            fixed.values[201] = (2.5, 60)
        elif unavailable == "provider":
            fixed.runtime._topology = NS(_process_lock=threading.Lock(), _processes={"worker_a": _owner(201)})
        else:
            del fixed.values[101]
    fixed.scope.seal()
    sampler = _SAMPLER(fixed.runtime)
    sampler.sample()
    value = client_memory_report(sampler, {})
    missing = "client" if unavailable == "client_missing" else HELPERS[0]
    assert value["sampling_coverage"] == "unknown_or_partial"
    assert value["sampled_peak_rss_sum_bytes"] is None
    assert value["latest"]["same_pass_rss_sum_bytes"] is None
    assert missing in value["latest"]["missing_members"]
    assert value["latest"]["members"][missing]["rss_bytes"] is None
    if unavailable == "remote":
        assert fixed.visits == [101]  # No arbitrary PID discovery or remote poll.
    if unavailable == "provider":
        assert fixed.visits == [101, 201]  # Never read same provider again as helper.


def test_later_provider_conflict_invalidates_earlier_false_client_bound(fixed):
    fixed.values[201] = (2., 5000)
    fixed.scope.bind_helper(HELPERS[0], _owner(201), created_at=2.)
    fixed.scope.seal()
    sampler = _SAMPLER(fixed.runtime)
    sampler.sample()
    assert fixed.scope.report()["sampled_peak_rss_lower_bound_bytes"] == 5000
    fixed.runtime._topology = NS(_process_lock=threading.Lock(), _processes={"worker_a": _owner(201)})
    sampler.sample()
    value = fixed.scope.report()
    assert value["sampled_peak_rss_sum_bytes"] is None
    assert value["sampled_peak_rss_lower_bound_bytes"] == 100
    assert value["sampling_coverage"] == "unknown_or_partial"


def test_nonatomic_pass_sum_cannot_be_promoted_to_simultaneous_peak_lower_bound(fixed):
    # Counterexample: two disjoint RSS spikes, no instant with sum 200.
    # Sequential sampler reads 100 from each. True simultaneous maximum is 100.
    fixed.scope.bind_helper(HELPERS[0], _owner(201), created_at=2.)
    fixed.scope.seal()
    fixed.scope.observe({"client": (101, 1., 100), HELPERS[0]: (201, 2., 100)}, span_seconds=.01)
    value = fixed.scope.report()
    assert value["sampled_peak_rss_sum_bytes"] == 200
    assert value["sampled_peak_rss_lower_bound_bytes"] == 100
    assert value["maximum_sample_span_seconds"] == .01
    assert value["lifetime_peak_rss_sum_interval"]["lower_bound_bytes"] is None


def test_late_manifest_does_not_retroactively_cover_startup_or_source_costs(fixed):
    sampler = _SAMPLER(fixed.runtime)
    sampler.sample()
    fixed.scope.bind_helper(HELPERS[0], _owner(201), created_at=2.)
    fixed.scope.seal()
    sampler.sample()
    value = fixed.scope.report()
    assert value["sampled_peak_rss_sum_bytes"] == 160
    assert value["membership_declared_before_sampling"] is False
    assert value["sampling_coverage"] == "unknown_or_partial"
    assert value["missing_member_samples"] == 1
    assert any("pre-sampler source" in line for line in value["limitations"])
    with pytest.raises(ValueError, match="sealed"):
        fixed.scope.bind_helper(HELPERS[1], _owner(202), created_at=2.)


def test_owner_contract_cannot_replay_another_benchmark_window(fixed):
    fixed.scope.seal()
    first = _SAMPLER(fixed.runtime)
    first.sample()
    assert first.client_memory_scope.report()["sampled_peak_rss_sum_bytes"] == 100
    second = _SAMPLER(fixed.runtime)
    assert second.client_memory_scope is not first.client_memory_scope
    assert second.client_memory_scope.report()["sampled_peak_rss_sum_bytes"] is None
    assert second.client_memory_scope.report()["membership_declared"] is False


def test_seal_after_sampling_begins_does_not_claim_before_sampling_even_before_first_result(fixed):
    fixed.scope.begin_sample()
    fixed.scope.seal()
    fixed.scope.observe({"client": (101, 1., 100)}, span_seconds=.001)
    value = fixed.scope.report()
    assert value["membership_declared_before_sampling"] is False
    assert value["sampling_coverage"] == "unknown_or_partial"


def test_roster_is_bounded_and_cannot_be_supplied_by_pid_tags_or_metadata(fixed):
    with pytest.raises(ValueError, match="process owner"):
        fixed.scope.bind_helper(HELPERS[0], NS(pid=201), created_at=2.)
    for slot, birth in (("private-secret-name", 2.), (HELPERS[0], float("nan"))):
        with pytest.raises(ValueError):
            fixed.scope.bind_helper(slot, _owner(201), created_at=birth)
    with pytest.raises(ValueError, match="duplicate"):
        fixed.scope.bind_helper(HELPERS[0], _owner(101), created_at=1.)
    for index, name in enumerate(HELPERS):
        fixed.scope.bind_helper(name, _owner(1000 + index), created_at=100. + index)
    fixed.scope.seal()
    fixed.scope.observe({"client": (101, 1., 100)}, span_seconds=0)
    text = json.dumps(fixed.scope.report())
    assert len(json.loads(text)["latest"]["members"]) == 9
    assert len(text.encode()) < 8192  # Serialization bound, not heap/physical proof.
    assert "private-secret-name" not in text and "private-argv" not in text
    assert '"pid"' not in text and '"created_at"' not in text


def test_report_and_failure_cache_reads_do_not_sample_or_reap(fixed, monkeypatch):
    sampler = _SAMPLER(fixed.runtime)
    sampler.sample()
    monkeypatch.setattr(benchmark_resources.psutil, "Process", lambda *_: pytest.fail("cache read queried OS"))
    monkeypatch.setattr(os, "waitpid", lambda *_: pytest.fail("cache read reaped a process"))
    first = client_memory_report(sampler, {})
    assert sampler.failure_snapshot()["latest_total_bytes"] == 100
    first["latest"]["members"]["client"]["rss_bytes"] = 0
    assert client_memory_report(sampler, {})["latest"]["members"]["client"]["rss_bytes"] == 100


def test_helper_reuse_during_rss_read_is_unknown_not_another_process_sample(fixed, monkeypatch):
    fixed.scope.bind_helper(HELPERS[0], _owner(201), created_at=2.)
    fixed.scope.seal()
    original = benchmark_resources.psutil.Process

    def racing(pid):
        process = original(pid)
        if pid == 201:
            read = process.memory_info

            def memory_info():
                value = read()
                fixed.values[201] = (2.5, 5000)
                return value

            process.memory_info = memory_info
        return process

    monkeypatch.setattr(benchmark_resources.psutil, "Process", racing)
    sampler = _SAMPLER(fixed.runtime)
    sampler.sample()
    value = fixed.scope.report()
    assert value["sampled_peak_rss_sum_bytes"] is None
    assert value["latest"]["members"][HELPERS[0]]["rss_bytes"] is None
    assert value["sampled_peak_rss_lower_bound_bytes"] == 100


@pytest.mark.parametrize("declared", [False, True])
def test_ordinary_sdk_report_wires_scope_without_changing_historical_metrics_or_identities(ordinary, monkeypatch, declared):
    # Reuse existing telemetry-only dashboard/HTTP fixture and real provider owners.
    # No inference/source resolution; enable the production sampler, not its double.
    monkeypatch.setattr(benchmark_resources, "ProcessMemorySampler", _SAMPLER)
    monkeypatch.setattr(benchmark_memory, "process_memory", _PROCESS_MEMORY)
    monkeypatch.setattr(benchmark_memory, "host_memory", lambda: NS(available=8 << 30, reserve=0))
    if declared:
        owner = ClientMemoryScope()
        owner.seal()  # This fixture has no client helpers, only explicit providers.
        ordinary.runtime._benchmark_client_memory_owner = owner
    report = _execute(ordinary)
    scope = report["client_owned_memory"]
    assert scope["membership_declared"] is declared
    assert scope["sampled_peak_rss_lower_bound_bytes"] > 0
    assert set(scope["latest"]["members"]) == {"client"}
    assert scope["sampling_coverage"] == ("all_declared_members_observed" if declared else "unknown_or_partial")
    if not declared:
        assert scope["sampled_peak_rss_sum_bytes"] is None
    assert scope["individual_client_hwm"]["bytes"] == report["client_process_memory"]["after"]["lifetime_peak_rss_bytes"]
    assert scope["individual_client_hwm"]["coverage"] == "individual_process_only_not_client_group"
    historical_report = deepcopy(report)
    del historical_report["client_owned_memory"]
    before = canonical(historical_report)
    baseline = Measurement({"sampling": Sampling(processes=1).to_spec(), "salt": "ab" * 32,
        "inputs": {}, "directory": "."})
    response_metrics(historical_report, baseline)
    extra = [row for row in ordinary.measured.metrics if row["id"].startswith("client.owned_")]
    assert len(extra) == 2
    assert [row for row in ordinary.measured.metrics if row not in extra] == baseline.metrics
    assert canonical(historical_report) == before
    assert extra[0]["origin"] == ("estimate" if declared else "not_available")
    assert extra[1]["origin"] == "lower_bound"
    assert all(row["window"] == scope["window"] and row["unit"] == "bytes" for row in extra)
    result = _serialized(ordinary)
    assert result["execution"]["details"]["report"]["client_owned_memory"] == scope
    assert result["configuration_digest"] == ordinary.case.target.digest


def test_real_sampler_does_not_publish_success_metrics_on_failed_sdk_execution(ordinary, monkeypatch):
    from pllm.runtime.benchmark_cli import LoopbackBenchmarkError

    monkeypatch.setattr(benchmark_resources, "ProcessMemorySampler", _SAMPLER)
    monkeypatch.setattr(benchmark_memory, "host_memory", lambda: NS(available=8 << 30, reserve=0))
    ordinary.mode = "failure"
    with pytest.raises(LoopbackBenchmarkError):
        _execute(ordinary)
    assert "report" not in ordinary.measured.details
    assert not any(row["id"].startswith("client.owned_") for row in ordinary.measured.metrics)
    assert ordinary.topology.closed


def test_unsupported_platform_hwm_and_unavailable_sampler_stay_unknown(monkeypatch):
    monkeypatch.setattr(sys, "platform", "unqualified-platform")
    value = client_memory_report(None, {"lifetime_peak_rss_bytes": 100})
    assert value["individual_client_hwm"]["bytes"] is None
    assert value["sampled_peak_rss_sum_bytes"] is None
    assert value["sampled_peak_rss_lower_bound_bytes"] is None
    measured = NS(metric=lambda *a, **k: pytest.fail("historical report gained metrics"))
    client_memory_metrics({}, measured)


@pytest.mark.skipif(os.name != "posix", reason="tiny POSIX owner smoke")
def test_two_real_owned_helpers_use_existing_sampler_and_reaper_without_orphans():
    processes, gates = [], []
    scope = ClientMemoryScope()
    sampler = None
    code = ("import os; data=bytearray(8<<20); "
            "data[::4096]=b'x'*len(data[::4096]); os.write(1,b'R'); os.read(FD,1)")
    try:
        for slot in HELPERS[:2]:
            read_fd, write_fd = os.pipe()
            gates.append(write_fd)
            try:
                process = servers._spawn_role_process([sys.executable, "-I", "-c", code.replace("FD", str(read_fd))],
                    environment=dict(os.environ), stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, pass_fds=(read_fd,))
                processes.append(process)
            finally:
                os.close(read_fd)
            assert select.select([process.stdout], [], [], 5)[0]
            assert process.stdout.read(1) == b'R'
            scope.bind_helper(slot, process, created_at=psutil.Process(process.pid).create_time())
        scope.seal()
        sampler = _SAMPLER(NS(_topology=None, _benchmark_client_memory_owner=scope))
        sampler.start()
        sampler.stop()
        hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        hwm_bytes = int(hwm * (1 if sys.platform == "darwin" else 1024))
        value = client_memory_report(sampler, {"lifetime_peak_rss_bytes": hwm_bytes})
        rows = value["latest"]["members"]
        assert set(rows) == {"client", *HELPERS[:2]}
        assert value["sampling_coverage"] == "all_declared_members_observed"
        assert value["latest"]["same_pass_rss_sum_bytes"] == sum(row["rss_bytes"] for row in rows.values())
        assert value["sampled_peak_rss_sum_bytes"] >= value["latest"]["same_pass_rss_sum_bytes"] > 0
        assert value["sampled_peak_rss_lower_bound_bytes"] >= max(row["rss_bytes"] for row in rows.values())
        assert value["individual_client_hwm"]["bytes"] == hwm_bytes if sys.platform in {"darwin", "linux"} else value["individual_client_hwm"]["bytes"] is None
        assert value["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] is None
    finally:
        if sampler is not None and sampler.thread.is_alive():
            sampler.stop()
        for descriptor in gates:
            os.write(descriptor, b'x')
            os.close(descriptor)
        for process in processes:
            try:
                assert process.wait(timeout=5) == 0
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                raise
            finally:
                process.stdout.close()
            with pytest.raises(ChildProcessError):
                os.waitpid(process.pid, os.WNOHANG)
