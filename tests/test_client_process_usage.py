"""Real tiny child/OS usage; protocol/model data are explicit test doubles.

No weights, model, native compute, build or Linux runtime claim. Mock controls
execute literal CPython poll code with copied globals, never patch os.waitpid.
"""
from copy import deepcopy
import errno
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import FunctionType, SimpleNamespace as NS

import psutil
import pytest

from pllm.runtime import client_process_usage as usage
from pllm.runtime.client_memory_scope import ClientMemoryScope, client_lifetime_manifest

pytestmark = pytest.mark.skipif(not usage.supported(), reason="Darwin CPython 3.11-3.13 qualification only")


def _topology():
    # Exact built-in class is tested without loading its native engine/model.
    from pllm.runtime.servers import LocalTopology
    owner = LocalTopology.__new__(LocalTopology)
    owner._process_lock = threading.Lock()
    owner._closed, owner._started = True, True
    owner._role_ids, owner._processes = (), {}
    from pllm.sources import TinyModel
    owner._model = TinyModel()
    return owner


def _scope_report(*, helper=False, missing=False, closed=False):
    from pllm.runtime.benchmark_cli import build_loopback_report
    from pllm.runtime.benchmark_memory import process_memory
    scope = ClientMemoryScope()
    root = psutil.Process()
    children = []
    try:
        if helper:
            child = usage.ClientUsagePopen([sys.executable, "-c",
                "import sys; sys.stdin.buffer.read(1); x=bytearray(8<<20); sum(range(50000))"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            children.append(child)
            scope.bind_helper("client_helper_1", child, created_at=psutil.Process(child.pid).create_time())
            scope.seal_lifetime_membership(closed_family=True)
            child.stdin.write(b"1")
            child.stdin.close()
            child.wait(timeout=10)
        if missing:
            scope.expect_unavailable_helper("client_helper_2", remote=True)
        if closed:
            scope.seal_lifetime_membership(closed_family=True)
        scope.observe({"client": (os.getpid(), root.create_time(), root.memory_info().rss)}, span_seconds=.001)
        row = {"status": "completed", "cold": True, "warm": False,
            "tokens": {"authoritative": True, "input_tokens": 3, "output_tokens": 2},
            "model_fingerprint": "b" * 64, "max_output_tokens": 2,
            "generation": {"response_status": "completed", "output_text_digest": "c" * 64},
            "privacy": {"plaintext_prompt_bytes_sent": 0, "plaintext_token_ids_sent": 0,
                        "preparation_requests_during_online": 0}}
        report = build_loopback_report(model_id="memory-telemetry-test-double", tiny=False,
            max_output_tokens=2, warmup_runs=[], runs=[row], source_lock_digest="a" * 64,
            capture_output_digest=True)
        report["client_owned_memory"] = scope.report(process_memory()["lifetime_peak_rss_bytes"])
        report["client_owned_memory"]["lifetime_owner"] = client_lifetime_manifest(
            NS(client_memory_scope=scope), _topology(), ("client",), local_cpu_loopback=True)
        return report
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()


def _child(connection, options):
    """Replacement benchmark body only; existing parent owner/pipe are real."""
    data = bytearray(8 << 20)
    for index in range(0, len(data), 4096):
        data[index] = 1
    sum(range(50000))
    mode = options.get("mode", "normal")
    if mode == "sleep":
        time.sleep(.2)
    if mode == "signal":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "cancel":
        connection.send(("ready", None))
        time.sleep(5)
    report = _scope_report(helper=mode == "helper", missing=mode == "missing", closed=mode == "closed")
    if connection is not None:
        connection.send(("report", report))
        connection.close()
    if mode == "nonzero":
        raise SystemExit(7)


@pytest.fixture
def children():
    processes = []
    yield processes
    for process in processes:
        if process._popen is not None and process._popen.returncode is None:
            process.kill()
        process.join(timeout=10)
        assert process._popen.returncode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
        process.close()


def launch(children, mode="normal"):
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = usage.client_process_factory(context.Process)(target=_child, args=(send, {"mode": mode}))
    process.start()
    children.append(process)
    send.close()
    return process, receive


def test_two_real_children_same_owner_join_postexit_bytes_and_no_orphans(children):
    pairs = [launch(children), launch(children)]
    for process, receive in pairs:
        kind, report = receive.recv()
        receive.close()
        assert kind == "report"
        prefix = report["client_owned_memory"]["individual_client_hwm"]["bytes"]
        # Report itself is still pre-exit. Parent has not silently reaped here.
        assert "client_process_exit_usage" not in report
        process.join(timeout=10)
        result = usage.finalize_client_report(report, process)
        observed = result["client_process_exit_usage"]
        assert observed["coverage"] == "os_process_exit" and observed["raw_hwm_unit"] == "bytes"
        assert observed["cpu_seconds"] == observed["user_cpu_seconds"] + observed["system_cpu_seconds"] >= 0
        assert observed["peak_rss_bytes"] > 0  # No fixed RSS claim: imports/OS differ.
        assert "including_reaped_descendants" == observed["cpu_scope"]
        bounds = result["client_owned_memory"]["lifetime_peak_rss_sum_interval"]
        assert 0 < bounds["lower_bound_bytes"] == prefix <= bounds["upper_bound_bytes"] == observed["peak_rss_bytes"]
        assert bounds["source"] == "owner_bound_individual_prefix_lower_and_wait4_family_upper"
        assert usage.owned_family_interval(result["client_owned_memory"]) is not None
        assert result["client_owned_memory"]["individual_client_hwm"]["bytes"] == prefix
        assert "unknown" in result["client_owned_memory"]["lifetime_owner"]["outside_isolated_role"]


def test_real_helper_family_uses_family_maxima_not_individual_or_simultaneous_peak(children):
    process, receive = launch(children, "helper")
    _, report = receive.recv()
    receive.close()
    process.join(timeout=10)
    usage.finalize_client_report(report, process)
    bounds = report["client_owned_memory"]["lifetime_peak_rss_sum_interval"]
    values = bounds["member_family_max_hwm_bytes"]
    assert set(values) == {"client", "client_helper_1"}
    assert bounds["lower_bound_bytes"] == max(values.values())
    assert bounds["upper_bound_bytes"] == sum(values.values())
    assert "not_self_only" in bounds["hwm_coverage"]


def test_real_timeout_then_same_owner_finish(children):
    process, receive = launch(children, "sleep")
    assert process.join(timeout=0) is None and process._popen.returncode is None
    assert usage.final_client_usage(process)["peak_rss_bytes"] is None
    receive.recv()
    receive.close()
    process.join(timeout=10)
    assert usage.final_client_usage(process)["normal_exit"] is True


def test_real_concurrent_join_poll_snapshot_share_one_reap(children):
    process, receive = launch(children, "sleep")
    owner = process._popen
    real_wait4 = owner._usage_wait4
    entered, release = threading.Event(), threading.Event()
    calls, results = [], []
    def wait4(pid, flags):
        calls.append((pid, flags))
        entered.set()
        assert release.wait(5)
        return real_wait4(pid, flags)
    owner._usage_wait4 = wait4  # Instance's original owner boundary, not global OS.
    first = threading.Thread(target=lambda: results.append(owner.wait()))
    second = threading.Thread(target=lambda: results.append(owner.wait()))
    first.start()
    assert entered.wait(5)
    second.start()
    try:
        assert owner.poll() is None
        assert usage.final_client_usage(process)["reason"] == "reap_in_progress"
    finally:
        release.set()
        receive.recv()
        receive.close()
        first.join(10)
        second.join(10)
    process.join(timeout=10)
    assert results == [0, 0] and calls == [(process.pid, 0)]


def test_real_cancel_reaped_not_made_admissible(children):
    process, receive = launch(children, "cancel")
    assert receive.recv()[0] == "ready"
    process.terminate()
    process.join(timeout=10)
    receive.close()
    report = _scope_report()
    usage.finalize_client_report(report, process, forced=True)
    assert report["client_process_exit_usage"]["exit_code"] == -signal.SIGTERM
    assert report["client_owned_memory"]["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] is None


def test_explicit_whole_lifetime_closed_single_process_may_have_degenerate_interval(children):
    process, receive = launch(children, "closed")
    _, report = receive.recv()
    receive.close()
    process.join(timeout=10)
    usage.finalize_client_report(report, process)
    bounds = report["client_owned_memory"]["lifetime_peak_rss_sum_interval"]
    assert bounds["lower_bound_bytes"] == bounds["upper_bound_bytes"] > 0
    assert bounds["source"] == "complete_owned_member_wait4_family_maxima"


@pytest.mark.parametrize("mode,code", [("nonzero", 7), ("signal", -signal.SIGKILL)])
def test_real_nonzero_and_signal_observed_but_no_successful_interval(children, mode, code):
    process, receive = launch(children, mode)
    if mode == "nonzero":
        _, report = receive.recv()
    else:
        report = _scope_report()
        with pytest.raises(EOFError):
            receive.recv()
    receive.close()
    process.join(timeout=10)
    usage.finalize_client_report(report, process)
    assert usage.final_client_usage(process)["exit_code"] == code
    assert report["client_owned_memory"]["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] is None


def test_existing_isolation_parent_attaches_only_after_normal_join(monkeypatch):
    from pllm.runtime import benchmark_isolation as isolation
    monkeypatch.setattr(isolation, "_worker", _child)
    report = isolation.run_isolated_loopback_benchmark(mode="normal")
    assert report["client_process_exit_usage"]["normal_exit"] is True
    assert report["client_owned_memory"]["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] is not None


def test_real_external_reap_stays_unknown_with_literal_multiprocessing_status():
    from multiprocessing.process import _children
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = usage.client_process_factory(context.Process)(target=_child, args=(send, {}))
    process.start()
    send.close()
    try:
        receive.recv()
        receive.close()
        # Deliberate external reaper: normal production never does this.
        assert os.waitpid(process.pid, 0)[0] == process.pid
        process.join(timeout=0)
        assert process._popen.returncode is None  # No fabricated subprocess zero.
        assert usage.final_client_usage(process)["reason"] == "external_reap_or_sigchld_ignored"
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
    finally:
        receive.close()
        # Original multiprocessing cannot infer the lost status either. Release
        # its inherited fd finalizer/stale registry entry, WITHOUT changing code.
        process._popen.close()
        _children.discard(process)


def _sdk_target(target, m):
    """Tiny source fixture in the real SDK worker protocol, not a new runner."""
    import pllm
    assert Path(pllm.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])
    from pllm.runtime.client_memory_scope import client_memory_metrics
    m.admit(256 << 20)
    def execute(_):
        data = bytearray(8 << 20)
        sum(range(50000))
        result = _scope_report()
        assert len(data) == 8 << 20
        return result
    report = m.run(execute)[-1]
    m.details["report"] = report
    client_memory_metrics(report, m)
    m.fixture = {"kind": "generated_checkpoint", "digest": "a" * 64, "metadata": {}}
    m.oracle = {"kind": "telemetry_test_double", "outcome": "passed", "checked_values": 1}
    m.comparison.update({key: "a" * 64 if key == "source_digest" else "b" * 64 for key in m.comparison})
    if m.inputs.get("terminal") == "cancelled":
        raise KeyboardInterrupt
    if m.inputs.get("terminal") == "failed":
        raise RuntimeError("telemetry test fixture failure")


@pytest.mark.parametrize("terminal", ["completed", "cancelled", "failed"])
def test_real_sdk_owner_report_serialization_and_opt_in_comparison(monkeypatch, tmp_path, terminal):
    import pllm
    from pllm._benchmark_contract import digest
    from pllm.evidence import EvidenceRegistry
    from pllm.metrics import BenchmarkCase, BenchmarkRun, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario
    from pllm.metrics import benchmark_runner as runner
    from pllm.profiles import MaskedLinearCpu

    experiment = pllm.Experiment("telemetry-sdk-fixture", MaskedLinearCpu(pllm.Model.tiny()),
        pllm.Deployment.local(root="target/telemetry-sdk-fixture"), pllm.ExecutionBudget(1, 128, 2))
    target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=2))
    cases = tuple(BenchmarkCase(name, target) for name in ("control", "candidate"))
    sampling = Sampling(warmups=0, repetitions=1, processes=1, timeout_seconds=10)
    suite = BenchmarkSuite(cases, sampling)
    real_factory = usage.client_popen_factory(subprocess.Popen)
    code = (f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'python')!r}); "
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
        "import test_client_process_usage as fixture; "
        "import pllm.runtime.benchmark_targets as targets; targets.execute_target=fixture._sdk_target; "
        "from pllm._benchmark_worker import main; main()")
    children = []
    def select(factory):
        assert factory is subprocess.Popen
        def spawn(_command, **kwargs):
            child = real_factory([sys.executable, "-c", code], **kwargs)
            children.append(child)
            return child
        return spawn
    monkeypatch.setattr(usage, "client_popen_factory", select)
    before = []
    real_finalize = usage.finalize_client_payload
    def finalize(payload, process):
        assert process.returncode == 0
        before.append(deepcopy(payload["metrics"]))
        real_finalize(payload, process)
        assert payload["metrics"] == before[-1]  # No duplicate/replaced old metric.
    monkeypatch.setattr(usage, "finalize_client_payload", finalize)
    rows = []
    for case in cases:
        payload = runner._execute(case, sampling, {"prompt": "private SDK fixture", "terminal": terminal}, b"x" * 32, tmp_path)
        assert payload["status"] == terminal
        report = payload["details"]["report"]
        assert (usage.owned_family_interval(report["client_owned_memory"]) is not None) == (terminal == "completed")
        assert "_root_identity" not in report["client_owned_memory"]["lifetime_owner"]
        attributes = {"fixture": "source-telemetry-only"}
        rows.append(runner._result(case, suite, 0, "d" * 64,
            {"attributes": attributes, "digest": digest(attributes)}, payload).to_dict())
    observed = BenchmarkRun({"schema": "pllm.benchmark_run.v1", "suite": suite.to_spec(),
        "cohort": "d" * 64, "complete": True, "results": rows})
    name = "client.owned_lifetime_peak_rss_sum_interval"
    result = observed.compare("control", "candidate", name)
    if terminal == "completed":
        assert result["status"] == "bounded" and result["reduction_status"] == "not_established"
        assert not result["physical_peak"] and "control_median" not in result
    else:
        assert result["status"] == "unranked"
    assert EvidenceRegistry(observed.results).query(metric=name)
    assert "private SDK fixture" not in json.dumps(observed.to_dict())
    assert len(children) == 2
    for child in children:
        assert child.returncode == 0
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)


@pytest.fixture
def owner(monkeypatch):
    # Real Popen initializer/lock; replace launch only, not global wait/reap APIs.
    monkeypatch.setattr(usage._SpawnPopen, "_launch", lambda self, _: setattr(self, "pid", 4242))
    return usage.UsageSpawnPopen(NS())


def literal_poll(owner, waitpid):
    namespace = dict(usage._POLL.__globals__)
    namespace["os"] = NS(waitpid=waitpid, waitstatus_to_exitcode=os.waitstatus_to_exitcode)
    poll = FunctionType(usage._POLL.__code__, namespace, argdefs=usage._POLL.__defaults__)
    return poll(owner)


@pytest.mark.parametrize("error", [errno.ECHILD, errno.EINTR, errno.EIO])
def test_errors_preserve_literal_multiprocessing_unknown_not_subprocess_zero(owner, error):
    def waitpid(*_):
        raise OSError(error, "test reap boundary")
    owner._usage_wait4 = waitpid
    baseline = NS(pid=owner.pid, returncode=None)
    assert literal_poll(baseline, waitpid) is None
    assert owner.poll() is None and owner.returncode is None
    assert owner.final_usage()["peak_rss_bytes"] is None
    # Same owner can recover a genuine exit after transient failure; no deadstate hack.
    owner._usage_wait4 = lambda *_: (owner.pid, 7 << 8, NS(ru_utime=.01, ru_stime=.02, ru_maxrss=12345))
    assert owner.wait() == 7 and owner.final_usage()["exit_code"] == 7


def test_same_owner_concurrent_wait_poll_snapshot_cannot_compete(owner):
    entered, finish = threading.Event(), threading.Event()
    calls = []
    def wait4(pid, flags):
        calls.append((pid, flags))
        entered.set()
        assert finish.wait(5)
        return pid, 0, NS(ru_utime=.01, ru_stime=.02, ru_maxrss=12345)
    owner._usage_wait4 = wait4
    result = []
    first = threading.Thread(target=lambda: result.append(owner.wait()))
    second = threading.Thread(target=lambda: result.append(owner.wait()))
    first.start()
    assert entered.wait(5)
    second.start()
    try:
        assert owner.poll() is None and owner.final_usage()["reason"] == "reap_in_progress"
        assert calls == [(4242, 0)]
    finally:
        finish.set()
        first.join(5)
        second.join(5)
    assert result == [0, 0] and len(calls) == 1


def test_gc_finalizer_remains_inherited_and_does_not_reap(owner):
    calls = []
    owner._usage_wait4 = lambda *_: calls.append(True)
    assert "__del__" not in usage.UsageSpawnPopen.__dict__
    assert usage.UsageSpawnPopen.close is usage._SpawnPopen.close
    owner.close()
    gc.collect()
    assert calls == [] and owner.final_usage()["coverage"] == "unknown"


@pytest.mark.parametrize("case", ["missing", "forced", "unclassified", "remote", "wrong_window", "wrong_units", "birth_mismatch"])
def test_incomplete_or_nonadmissible_cases_never_force_interval_available(owner, case):
    report = _scope_report(missing=case == "missing")
    manifest = report["client_owned_memory"]["lifetime_owner"]
    owner.pid = os.getpid()
    owner._usage_identity = (owner.pid, psutil.Process().create_time())
    owner._usage_wait4 = lambda *_: (owner.pid, 0, NS(ru_utime=.01, ru_stime=.02, ru_maxrss=1 << 50))
    owner.wait()
    if case == "unclassified":
        manifest["membership_complete"] = False
    if case == "remote":
        manifest["membership_complete"] = False
    if case == "birth_mismatch":
        manifest["_root_identity"] = (owner.pid, owner._usage_identity[1] + 1)
    usage.finalize_client_report(report, NS(_popen=owner), forced=case == "forced")
    bounds = report["client_owned_memory"]["lifetime_peak_rss_sum_interval"]
    if case in {"wrong_window", "wrong_units"}:
        assert bounds["upper_bound_bytes"] is not None
        bounds["window" if case == "wrong_window" else "unit"] = "mismatch"
    else:
        assert bounds["upper_bound_bytes"] is None
    assert usage.owned_family_interval(report["client_owned_memory"]) is None


def test_unsupported_factory_stays_foreign_without_global_patch(monkeypatch):
    monkeypatch.setattr(usage.sys, "platform", "linux")
    factory = mp.get_context("spawn").Process
    assert usage.client_process_factory(factory) is factory
    assert usage.client_popen_factory(subprocess.Popen) is subprocess.Popen
    assert usage.final_client_usage(NS())["coverage"] == "unknown"


def test_declared_provider_owner_never_becomes_client_helper_by_ancestry():
    scope = ClientMemoryScope()
    scope.observe({"client": (os.getpid(), psutil.Process().create_time(), 100)}, span_seconds=0)
    topology = _topology()
    provider = subprocess.Popen.__new__(subprocess.Popen)
    provider.pid, provider.returncode = 999999, 0
    topology._role_ids, topology._processes = ("inference",), {"inference": provider}
    manifest = client_lifetime_manifest(NS(client_memory_scope=scope), topology,
        ("client", "inference"), local_cpu_loopback=True)
    assert manifest["membership_complete"] and not manifest["closed_client_owned_family"]
    assert set(manifest["members"]) == {"client"}
    conflict_scope = ClientMemoryScope()
    conflict_scope.observe({"client": (os.getpid(), psutil.Process().create_time(), 100)}, span_seconds=0)
    conflict_scope.bind_helper("client_helper_1", provider, created_at=1.)
    conflict_scope.seal_lifetime_membership()
    conflict = client_lifetime_manifest(NS(client_memory_scope=conflict_scope), topology,
        ("client", "inference"), local_cpu_loopback=True)
    assert not conflict["membership_complete"]


def test_sampling_only_seal_never_implies_closed_lifetime_family():
    scope = ClientMemoryScope()
    scope.seal()
    scope.observe({"client": (os.getpid(), psutil.Process().create_time(), 100)}, span_seconds=0)
    manifest = client_lifetime_manifest(NS(client_memory_scope=scope), _topology(),
        ("client",), local_cpu_loopback=True)
    assert manifest["membership_complete"] and manifest["closed_client_owned_family"] is False
    # Only the audited builtin root-only owner route supplies this classification.
    foreign = client_lifetime_manifest(NS(client_memory_scope=scope), NS(),
        ("client",), local_cpu_loopback=True)
    assert not foreign["membership_complete"]


def test_fixed_member_cap_remains_unknown(owner):
    report = _scope_report()
    manifest = report["client_owned_memory"]["lifetime_owner"]
    owner.pid = os.getpid()
    owner._usage_identity = (owner.pid, psutil.Process().create_time())
    owner._usage_wait4 = lambda *_: (owner.pid, 0, NS(ru_utime=.01, ru_stime=.02, ru_maxrss=1 << 50))
    owner.wait()
    manifest["members"].update({f"client_helper_{i}": {} for i in range(1, 257)})
    usage.finalize_client_report(report, NS(_popen=owner))
    assert report["client_owned_memory"]["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] is None
    assert usage.owned_family_interval(report["client_owned_memory"]) is None


def test_unclassified_dynamic_source_helpers_stay_unknown_not_assumed_absent():
    import pllm
    scope = ClientMemoryScope()
    scope.observe({"client": (os.getpid(), psutil.Process().create_time(), 100)}, span_seconds=0)
    topology = _topology()
    topology._model = pllm.Model.hf("unresolved/source")
    manifest = client_lifetime_manifest(NS(client_memory_scope=scope), topology,
        ("client",), local_cpu_loopback=True)
    assert not manifest["membership_complete"]
    assert manifest["reason"] == "source_helper_lifetime_owner_contract_unavailable"
