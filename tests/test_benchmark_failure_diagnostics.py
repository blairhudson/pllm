"""Mocked failure observations only: no sources, native execution, children or servers."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
import json
import threading
from types import SimpleNamespace as NS

import pytest

from pllm._cli.benchmark_output import BenchmarkOutput
from pllm.runtime import benchmark_cli, benchmark_failure as diagnostics
from pllm.runtime import benchmark_memory as memory, benchmark_resources, dashboard, servers


SECRET = "private-prompt-token-credential-/tensor/path"


def request_context(role="worker_b", status=503):
    return {"stage": "transport", "role": role, "code": "worker_http_rejected", "http_status": status}


COUNT_CONTEXTS = [
    {"stage": "input_count", "role": "client", "code": code, "http_status": None}
    for code in ("count_owner_busy", "input_count_failed")
]


@pytest.mark.parametrize("context", COUNT_CONTEXTS)
def test_input_count_context_accepts_only_fixed_local_combinations(context):
    assert diagnostics.sanitize_request_failure_context(context) == context
    for changes in ({"stage": "transport"}, {"stage": "prepare_prompt"},
                    {"role": "worker_a"}, {"http_status": 409}, {"http_status": 500},
                    {"code": "gkm_stage_failed"}, {"code": "worker_http_rejected"},
                    {"code": SECRET}, {"body": SECRET}):
        assert diagnostics.sanitize_request_failure_context({**context, **changes}) is None


@pytest.mark.parametrize("changes", [
    {"stage": SECRET}, {"role": SECRET}, {"code": SECRET}, {"http_status": True},
    {"http_status": 99}, {"http_status": 600}, {"http_status": "503"},
    {"http_status": 503.0}, {"http_status": None}, {"role": "client"},
    {"stage": "accept_result"}, {"body": SECRET},
])
def test_request_failure_context_fails_closed(changes):
    from pllm.runtime.client import ProtocolError

    context = {**request_context(), **changes}
    assert diagnostics.sanitize_request_failure_context(context) is None
    error = ProtocolError(SECRET, body={"secret": SECRET})
    error.annotate_request_failure(context)
    assert error.request_failure_context is None
    safe = diagnostics.sanitize_failure_diagnostics({
        "schema_version": diagnostics.SCHEMA, "request_failure_context": context})
    assert "request_failure_context" not in safe
    assert SECRET not in json.dumps(safe)


def test_request_context_exact_builtins_and_nonthrowing_annotation(monkeypatch):
    from pllm.runtime.client import ProtocolError

    class DictSubclass(dict):
        pass

    class HostileKey:
        def __hash__(self):
            return hash("stage")

        def __eq__(self, other):
            raise AssertionError("unknown key must not be compared")

    class IntSubclass(int):
        pass

    class StrSubclass(str):
        pass

    for bad in (None, [], tuple(request_context().items()), DictSubclass(request_context()),
                {**request_context(), "http_status": IntSubclass(503)},
                {**request_context(), "stage": StrSubclass("transport")},
                {HostileKey(): "transport", "role": "worker_b", "code": "worker_http_rejected", "http_status": 503},
                {"stage": "transport", "role": "worker_b", "code": "worker_http_rejected"}):
        assert diagnostics.sanitize_request_failure_context(bad) is None
    for status in (100, 599):
        safe = diagnostics.sanitize_request_failure_context(request_context(status=status))
        assert safe == request_context(status=status)
    error = ProtocolError(SECRET, body={"private": SECRET})
    assert error.request_failure_context is None
    error.annotate_request_failure(request_context())
    original = error.request_failure_context
    error.annotate_request_failure({"stage": "accept_result", "role": "client",
                                    "code": "gkm_stage_failed", "http_status": None})
    error.annotate_request_failure(COUNT_CONTEXTS[1])
    assert error.request_failure_context is original
    assert error.body == {"private": SECRET}

    def unavailable(*args):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(diagnostics, "sanitize_request_failure_context", unavailable)
    error.annotate_request_failure(request_context())  # Optional observation cannot throw.
    assert error.request_failure_context is original


def test_request_capture_first_failure_and_unknown_error_fallback():
    from pllm.runtime.client import ProtocolError

    runtime = minimal_runtime()
    runtime._benchmark_failure_diagnostics = diagnostics.FailureDiagnostics(runtime)
    error = ProtocolError(SECRET, body=SECRET)
    error.annotate_request_failure(request_context())
    value = diagnostics.capture_failure(runtime, phase="request", error=error)
    assert value["request_failure_context"] == request_context()
    assert SECRET not in json.dumps(value)
    assert diagnostics.capture_failure(runtime, error=ValueError(SECRET)) == value
    value["request_failure_context"]["role"] = "worker_a"
    assert runtime._benchmark_failure_diagnostics.capture()["request_failure_context"] == request_context()
    runtime = minimal_runtime()
    runtime._benchmark_failure_diagnostics = diagnostics.FailureDiagnostics(runtime)
    unknown = ValueError(SECRET)
    unknown.request_failure_context = request_context()
    first = diagnostics.capture_failure(runtime, error=unknown)
    assert "request_failure_context" not in first
    assert diagnostics.capture_failure(runtime, error=error) == first
    assert diagnostics.capture_failure(minimal_runtime(), error=ValueError(SECRET)) is None


def test_outer_collector_merge_preserves_request_context_and_observations():
    outer = diagnostics.FailureDiagnostics(minimal_runtime()).capture()
    outer["host"]["pressure"] = "warning"
    inner = {**outer, "phase": "request", "request_failure_context": request_context(), "body": SECRET}
    merged = diagnostics.merge_failure_diagnostics(outer, inner)
    assert merged["host"]["pressure"] == "warning"
    assert merged["request_failure_context"] == request_context()
    assert "request_failure_context" not in outer
    assert SECRET not in json.dumps(merged)
    assert diagnostics.merge_failure_diagnostics(merged, {
        **inner, "request_failure_context": request_context("worker_a")}) == merged


def rss_snapshot():
    return {"latest_total_bytes": 150, "peak_total_bytes": 180, "sample_age_seconds": .05,
            "sampling_incomplete": False,
            "roles": {"client": {"latest_bytes": 100, "peak_bytes": 120},
                      "worker_a": {"latest_bytes": 50, "peak_bytes": 70}}}


@pytest.mark.parametrize("stage", ["pre_spawn", "role_startup", "request"])
def test_warning_snapshot_precedes_abort_cleanup(tmp_path, monkeypatch, stage):
    """Exercise the real dashboard abort closure and real watchdog check/run logic."""
    config = dashboard.DashboardConfig("127.0.0.1", 1, "unused", "unit", 1, backend="native")
    runtime = dashboard.DashboardRuntime(config, NS())
    calls, host = [], [NS(pressure="normal", available=10 << 30, swap_used=0)]
    breadcrumb = {"phase": "memory_preflight", "role": None,
                  "spawned_roles": [], "ready_roles": []}
    sampler = NS(failure_snapshot=lambda: calls.append("cached_rss") or rss_snapshot())
    collector = diagnostics.FailureDiagnostics(runtime, sampler)
    runtime._benchmark_failure_diagnostics = collector
    collector.started -= 2

    def close():
        calls.append("close")
        assert collector._snapshot is not None  # Capture must beat role teardown.
        breadcrumb.update(phase="unknown", role=None, spawned_roles=[], ready_roles=[])
        host[0] = NS(pressure="normal", available=30 << 30, swap_used=0)

    def start():
        breadcrumb.update(phase="role_startup", role="worker_b",
                          spawned_roles=["worker_a", "worker_b"], ready_roles=["worker_a"])
        runtime._set(startup_step="worker_b")
        if stage == "request":
            breadcrumb.update(phase="ready", role=None,
                              spawned_roles=["worker_a", "worker_b", "dealer"],
                              ready_roles=["worker_a", "worker_b", "dealer"])
            runtime._active_run = object()
            runtime._set(phase="online", prompt=SECRET)
        host[0] = NS(pressure="warning", available=3 << 30, swap_used=4096)
        runtime._memory_guard._stop = NS(wait=lambda _: False)
        runtime._memory_guard._run()  # Exactly one failed check; no thread.
        raise RuntimeError(SECRET)

    topology = NS(startup_breadcrumbs=breadcrumb, start=start, close=close)
    monkeypatch.setattr(dashboard, "build_roles", lambda *a, **k: topology)
    report = {"admitted": True, "selected_backend": "native",
              "host": {"available": 10 << 30, "swap_used": 0, "reserve_bytes": 1 << 30},
              "estimate": {"provider_peak_bytes": {}}}
    monkeypatch.setattr(memory, "execution_memory_admission", lambda *a, **k: report)

    guard_class = memory.MemoryWatchdog
    monkeypatch.setattr(memory, "MemoryWatchdog", lambda report, abort:
                        guard_class(report, abort, sample=lambda: host[0]))
    monkeypatch.setattr(guard_class, "start", lambda self: self.check())
    monkeypatch.setattr(dashboard.tempfile, "TemporaryDirectory", lambda **k:
                        NS(name=str(tmp_path), cleanup=lambda: None))
    if stage == "pre_spawn":
        host[0] = NS(pressure="warning", available=3 << 30, swap_used=4096)
    asyncio.run(runtime.start())
    if stage == "pre_spawn":
        topology.close()

    value = collector.capture()
    assert calls == ["cached_rss", "close"]
    assert value["host"] == {"pressure": "warning", "available_bytes": 3 << 30,
                             "swap_used_bytes": 4096}
    assert value["elapsed_seconds"] >= 2
    assert value["process_rss"] == rss_snapshot()
    if stage == "pre_spawn":
        assert value["phase"] == "memory_preflight"
        assert value["spawned_roles"] == value["ready_roles"] == []
    else:
        assert value["phase"] == stage
        assert value["role"] == ("worker_b" if stage == "role_startup" else None)
        assert value["spawned_roles"] == (["worker_a", "worker_b"] if stage == "role_startup"
                                          else ["worker_a", "worker_b", "dealer"])
        assert value["ready_roles"] == (["worker_a"] if stage == "role_startup"
                                        else ["worker_a", "worker_b", "dealer"])
    assert SECRET not in json.dumps(value)


REAL_GUARD = memory.MemoryWatchdog


def test_cached_sampler_values_are_same_window_and_need_no_abort_scan(monkeypatch):
    values = {1: 100, 2: 50}
    monkeypatch.setattr(benchmark_resources.os, "getpid", lambda: 1)
    monkeypatch.setattr(benchmark_resources.psutil, "Process", lambda pid: NS(
        create_time=lambda: 0, memory_info=lambda: NS(rss=values[pid])))
    runtime = NS(_topology=NS(_process_lock=threading.Lock(),
                            _processes={"worker_a": NS(pid=2)}))
    sampler = benchmark_resources.ProcessMemorySampler(runtime)
    assert sampler.failure_snapshot() is None
    sampler.sample()
    values.update({1: 50, 2: 100})
    sampler.sample()
    monkeypatch.setattr(benchmark_resources.psutil, "Process", lambda pid:
                        pytest.fail("failure capture must not scan processes"))
    value = sampler.failure_snapshot()
    assert value["latest_total_bytes"] == value["peak_total_bytes"] == 150
    assert value["roles"] == {"client": {"latest_bytes": 50, "peak_bytes": 100},
                              "worker_a": {"latest_bytes": 100, "peak_bytes": 100}}
    value["roles"]["client"]["peak_bytes"] = 999
    assert sampler.failure_snapshot()["roles"]["client"]["peak_bytes"] == 100
    sampler.error = True
    assert sampler.failure_snapshot()["sampling_incomplete"] is True


def minimal_runtime():
    return NS(_topology=None, _lock=threading.Lock(), _state={"phase": "starting"},
              _active_run=None, _has_started_run=False, _memory_guard=None)


def test_no_stale_observations_between_candidates():
    first = minimal_runtime()
    first._memory_guard = NS(failure_observation={"pressure": "warning", "available_bytes": 1})
    collector = diagnostics.FailureDiagnostics(first, NS(failure_snapshot=rss_snapshot))
    value = collector.capture()
    value["host"]["pressure"] = "critical"
    assert collector.capture()["host"]["pressure"] == "warning"
    second = diagnostics.FailureDiagnostics(minimal_runtime()).capture()
    assert second["host"] == {"pressure": None, "available_bytes": None, "swap_used_bytes": None}
    assert second["process_rss"]["roles"] == {}
    assert second["process_rss"]["peak_total_bytes"] is None
    assert second["qualification"] == {"priced": None, "completed": None, "retired": None}
    assert REAL_GUARD({"host": {"available": 1, "swap_used": 0}}, lambda _: None,
                      sample=lambda: None).failure_observation is None
    idle = minimal_runtime()
    idle._has_started_run = True
    idle._state["phase"] = "ready"
    assert diagnostics.FailureDiagnostics(idle).capture()["phase"] == "ready"


def test_fixed_schema_bounds_and_sensitive_fields_are_discarded():
    maximum = (1 << 63) - 1
    value = {"schema_version": diagnostics.SCHEMA, "phase": SECRET, "role": SECRET,
              "elapsed_seconds": 10 ** 1000, "spawned_roles": list(diagnostics.ROLES),
             "qualification": {"priced": SECRET, "completed": 1, "retired": False,
                               "source_path": SECRET},
             "ready_roles": list(diagnostics.ROLES), "prompt": SECRET, "exception": SECRET,
             "host": {"pressure": SECRET, "available_bytes": maximum, "swap_used_bytes": True,
                      "environment": SECRET},
             "process_rss": {"latest_total_bytes": maximum, "peak_total_bytes": maximum,
                             "sample_age_seconds": float("nan"), "token": SECRET,
                             "roles": {role: {"latest_bytes": maximum, "peak_bytes": maximum,
                                              "pid": SECRET, "tensor_path": SECRET}
                                       for role in diagnostics.ROLES}}}
    safe = diagnostics.sanitize_failure_diagnostics(value)
    encoded = json.dumps(safe, allow_nan=False)
    assert len(encoded.encode()) < 4096 and SECRET not in encoded
    assert safe["phase"] == "unknown" and safe["role"] is None
    assert safe["elapsed_seconds"] is None
    assert safe["qualification"] == {"priced": None, "completed": None, "retired": False}
    assert safe["host"]["swap_used_bytes"] is None
    assert safe["process_rss"]["sample_age_seconds"] is None
    assert diagnostics.sanitize_failure_diagnostics(safe) == safe
    value["spawned_roles"] = [SECRET] * 10000
    assert diagnostics.sanitize_failure_diagnostics(value)["spawned_roles"] == []
    value["spawned_roles"] = [SECRET, "dealer", "dealer"]
    value["process_rss"]["roles"][SECRET] = {"latest_bytes": maximum}
    safe = diagnostics.sanitize_failure_diagnostics(value)
    assert safe["spawned_roles"] == ["dealer"]
    assert SECRET not in json.dumps(safe, allow_nan=False)
    assert diagnostics.sanitize_failure_diagnostics({"schema_version": SECRET}) is None


def test_ordinary_json_preserves_failure_and_old_reports(tmp_path):
    path = tmp_path / "benchmark.json"
    output = BenchmarkOutput(path, force=False, configuration={}, experiments=[None, None])
    output.start(1)
    success = {"checks": {"passed": True}, "configuration": {"identity": "unchanged"}}
    output.completed(success)
    output.start(2)
    error = benchmark_cli.LoopbackBenchmarkError(SECRET,
        failure_diagnostics=diagnostics.FailureDiagnostics(minimal_runtime()).capture())
    output.failed(error)
    saved = json.loads(path.read_text())
    assert saved["failure"]["type"] == "LoopbackBenchmarkError"
    assert saved["failure"]["diagnostics"]["schema_version"] == diagnostics.SCHEMA
    assert saved["complete"] is saved["checks"]["complete"] is saved["checks"]["passed"] is False
    assert saved["candidates"][0]["report"] == success
    assert saved["candidates"][1]["report"] is None
    assert saved["candidates"][1]["status"] == "failed"
    assert SECRET not in path.read_text()
    # No diagnostics remain when an older failure without observations is written.
    output.failed(RuntimeError(SECRET))
    assert json.loads(path.read_text())["failure"] == {"type": "RuntimeError"}
    output.finish(success)
    assert json.loads(path.read_text()) == success


def test_topology_records_successful_spawn_only(monkeypatch):
    topology = servers.LocalTopology.__new__(servers.LocalTopology)
    topology._process_lock = threading.Lock()
    topology._stopping = threading.Event()
    topology._log_dir = None
    topology._processes = {}
    topology._startup_breadcrumbs = {"phase": "memory_preflight", "role": None,
                                    "spawned_roles": [], "ready_roles": []}
    monkeypatch.setattr(servers.LocalTopology, "_environment", lambda *a: {})
    def fail(*a, **k):
        raise OSError(SECRET)
    monkeypatch.setattr(servers, "_spawn_role_process", fail)
    topology._startup_step("role_startup", "worker_a")
    with pytest.raises(OSError):
        topology._spawn("worker_a", [])
    assert topology.startup_breadcrumbs["spawned_roles"] == []
    monkeypatch.setattr(servers, "_spawn_role_process", lambda *a, **k: NS(pid=1))
    topology._spawn("worker_a", [])
    assert topology.startup_breadcrumbs == {"phase": "role_startup", "role": "worker_a",
                                            "spawned_roles": ["worker_a"], "ready_roles": []}
    topology._spawn("worker_a", [])
    assert topology.startup_breadcrumbs["spawned_roles"] == ["worker_a"]


def test_loopback_failure_carries_snapshot_before_handle_cleanup(monkeypatch):
    runtime = minimal_runtime()
    runtime.store = NS(stage_body_snapshot=lambda **k: {})
    calls = []
    monkeypatch.setattr(memory, "process_memory", lambda: {})
    monkeypatch.setattr(dashboard, "create_dashboard_app", lambda _: NS(state=NS(dashboard_runtime=runtime)))
    monkeypatch.setattr(benchmark_resources, "ProcessMemorySampler", lambda _: NS(
        start=lambda: None, stop=lambda: calls.append("sampler_stop"),
        failure_snapshot=lambda: calls.append("capture") or rss_snapshot()))
    monkeypatch.setattr(benchmark_cli, "_free_port", lambda: 1)
    monkeypatch.setattr(benchmark_cli, "_DashboardHandle", lambda *a: NS(
        start=lambda: None, close=lambda: calls.append("handle_close"), diagnostic=lambda: ""))
    monkeypatch.setattr(benchmark_cli, "_wait_for_dashboard_listener", lambda *a: None)
    monkeypatch.setattr(benchmark_cli.httpx, "Client", lambda **k: nullcontext(object()))
    def fail(*a):
        raise benchmark_cli.LoopbackBenchmarkError(SECRET, failure_diagnostics={
            "schema_version": diagnostics.SCHEMA, "request_failure_context": request_context()})
    monkeypatch.setattr(benchmark_cli, "_wait_for_ready", fail)
    with pytest.raises(benchmark_cli.LoopbackBenchmarkError) as caught:
        benchmark_cli._run_loopback_benchmark(model="unused", model_id=None, tiny=False,
            prompt=SECRET, max_output_tokens=1, warmups=0, repetitions=1, timeout_seconds=1)
    assert caught.value.failure_diagnostics["phase"] == "dashboard_startup"
    assert caught.value.failure_diagnostics["request_failure_context"] == request_context()
    assert calls == ["capture", "sampler_stop", "handle_close"]


@pytest.mark.parametrize("context", [None, request_context(), *COUNT_CONTEXTS])
def test_optional_diagnostics_cross_isolation_pipe_without_new_process(monkeypatch, context):
    from pllm.runtime import benchmark_isolation as isolation
    value = diagnostics.FailureDiagnostics(minimal_runtime()).capture()
    if context is not None:
        value["request_failure_context"] = context
    messages = iter([("failure_diagnostics", value), ("error", "LoopbackBenchmarkError")])
    process = NS(start=lambda: None, join=lambda **k: None, is_alive=lambda: False)
    receive, send = NS(recv=lambda: next(messages), close=lambda: None), NS(close=lambda: None)
    context = NS(Pipe=lambda **k: (receive, send), Process=lambda **k: process)
    monkeypatch.setattr(isolation.multiprocessing, "get_context", lambda _: context)
    with pytest.raises(benchmark_cli.LoopbackBenchmarkError) as caught:
        isolation.run_isolated_loopback_benchmark()
    assert caught.value.failure_diagnostics == value


@pytest.mark.parametrize("phase", ["qualification_metadata", "qualification_import",
                                    "qualification_seal", "qualification_pricing",
                                    "qualification_retirement", "graph_admission",
                                    "source_config", "source_tokenizer"])
@pytest.mark.parametrize("warning_time", [False, True])
def test_qualification_failure_flags_are_live_and_captured_before_cleanup(monkeypatch, phase, warning_time):
    """Real coordinator lifecycle; mocked source/leases/telemetry, no tensor import."""
    from pllm.deployment import resources
    from pllm.runtime import fss_source

    topology = servers.LocalTopology.__new__(servers.LocalTopology)
    topology._process_lock = threading.Lock()
    topology._stopping = threading.Event()
    topology._startup_breadcrumbs = {"phase": "unknown", "role": None,
                                    "spawned_roles": [], "ready_roles": []}
    topology._memory_admission = None
    topology._memory_enforced_wan = False
    topology._memory_budget_bytes = None
    topology._memory_safety_margin_percent = 25
    topology._experiment = NS(pipeline=None, budget=NS(max_input_tokens=3, max_new_tokens=1))
    runtime = minimal_runtime()
    runtime._topology = topology
    runtime._memory_guard = NS(failure_observation=None)
    collector = diagnostics.FailureDiagnostics(runtime)
    topology._failure_observer = collector.capture
    calls = []

    def warning():
        if warning_time:
            runtime._memory_guard.failure_observation = {
                "pressure": "warning", "available_bytes": 1024, "swap_used_bytes": 32}
            topology.request_stop()
            collector.capture()
        raise RuntimeError(SECRET)

    source = NS(contract={"source_float32_bytes": 1, "source_disk_bytes": 0},
                snapshot_pins={"pin": "public"}, config_bytes=b"{}")
    estimate = {"incremental_owner_bounds": {"per_role": 1},
                "fss_admission": {"source_pins": source.snapshot_pins}}

    def resolve(*a, **k):
        if phase == "qualification_metadata":
            warning()
        if phase in ("source_config", "source_tokenizer"):
            k["lifecycle"](phase)
            warning()
        return source

    def close():
        calls.append("close")
        if phase == "qualification_retirement":
            warning()
        if phase in ("qualification_import", "qualification_seal", "qualification_pricing"):
            assert collector._snapshot is not None

    def load(*a, lifecycle, on_failure, **k):
        lifecycle("qualification_import")
        if phase == "qualification_import":
            try:
                warning()
            except RuntimeError:
                on_failure()
                raise
        lifecycle("qualification_seal")
        if phase == "qualification_seal":
            try:
                warning()
            except RuntimeError:
                on_failure()
                raise
        return NS(close=close)

    def price(*a, **k):
        if phase == "qualification_pricing":
            warning()
        return {"estimate": estimate}

    def admit(*a, **k):
        if topology.startup_breadcrumbs["phase"] == "graph_admission":
            warning()
        return {"admitted": True}

    monkeypatch.setattr(fss_source, "resolve_client_source", resolve)
    monkeypatch.setattr(fss_source, "load_role_source", load)
    monkeypatch.setattr(memory, "_fss_memory", lambda *a, **k: estimate)
    monkeypatch.setattr(memory, "_fss_source_qualification_estimate", lambda value: value)
    monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(4096, 4096, 0, 4096))
    monkeypatch.setattr(memory, "admit_memory", admit)
    monkeypatch.setattr(memory, "benchmark_memory", price)
    monkeypatch.setattr(resources, "reserve_incremental_memory", lambda *a, **k: nullcontext())
    with pytest.raises(RuntimeError, match=SECRET):
        topology._qualify_fss_decoder()
    value = collector.capture()
    assert value["phase"] == phase
    assert value["spawned_roles"] == value["ready_roles"] == []
    assert value["qualification"] == {
        "priced": phase in ("qualification_retirement", "graph_admission"),
        "completed": False, "retired": phase == "graph_admission"}
    assert value["process_rss"]["latest_total_bytes"] is None
    assert value["process_rss"]["sampling_incomplete"] is True
    assert value["host"]["pressure"] == ("warning" if warning_time else None)
    assert SECRET not in json.dumps(value, allow_nan=False)
    # Later finally/cleanup observations cannot rewrite warning-time lifecycle.
    topology._qualification_flags(priced=True, completed=True, retired=True)
    assert collector.capture() == value


@pytest.mark.parametrize("phase", ["qualification_import", "qualification_seal"])
def test_source_breadcrumbs_surround_real_builder_boundaries(monkeypatch, phase):
    """Call source orchestration with entirely fake metadata, IO and native builder."""
    from pathlib import Path
    from pllm.runtime import fss_source

    calls, observed = [], []
    record = {"source_float32_bytes": 1, "source_shapes": {}, "numeric_profile": "mock"}
    source = NS(runtime_contract=lambda *a: b"mock", checkpoint_digest="checkpoint",
                source_lock_digest="lock", _lock=b"mock", path=Path("."), model=None,
                _identities=(), _specs={}, config_bytes=b"mock")
    monkeypatch.setattr(fss_source, "FrozenFssSource", NS)
    monkeypatch.setattr(fss_source, "_check_pins", lambda *a: None)
    monkeypatch.setattr(fss_source, "_document", lambda *a: record)
    monkeypatch.setattr(fss_source, "_locked", lambda *a: ("checkpoint", "lock", record))
    monkeypatch.setattr(fss_source, "_store", lambda *a: (NS(clear_cache=lambda: calls.append("cache_close")), {}))
    monkeypatch.setattr(fss_source, "_rotary_specs", lambda *a: [])

    def fail():
        assert observed[-1] == phase
        raise RuntimeError(SECRET)

    def builder(*a, **k):
        if phase == "qualification_import":
            fail()
        return NS(seal=fail, close=lambda: calls.append("builder_close"))

    monkeypatch.setattr(fss_source, "_native", NS(FssSourceBuilder=builder))
    def capture():
        assert calls == []
        calls.append("capture")
    with pytest.raises(RuntimeError, match=SECRET):
        fss_source.load_role_source(NS(budget=NS(max_input_tokens=3, max_new_tokens=1)),
            expected_pins={}, source=source, lifecycle=observed.append, on_failure=capture)
    assert observed == (["qualification_metadata", "source_contract", "source_lock", "source_headers",
                         "qualification_import"] + (["source_weights", "source_rotary",
                                                     "source_recheck", "qualification_seal"]
                                                    if phase == "qualification_seal" else []))
    assert calls == (["capture"] + (["builder_close"] if phase == "qualification_seal" else [])
                     + ["cache_close"])


@pytest.mark.parametrize("context", [None, request_context(), *COUNT_CONTEXTS])
def test_actual_cli_writes_only_bounded_optional_failure_fields(tmp_path, monkeypatch, capsys, context):
    from pllm.cli import main
    value = diagnostics.FailureDiagnostics(minimal_runtime()).capture()
    if context is not None:
        value["request_failure_context"] = context
    def fail(**kwargs):
        raise benchmark_cli.LoopbackBenchmarkError(SECRET, failure_diagnostics=value)
    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", fail)
    path = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as caught:
        main(["benchmark", "run", "--tiny", "--prompt", SECRET, "--warmups", "0",
              "--max-output-tokens", "1", "--output", str(path), "--format", "json"])
    assert caught.value.code == 6
    saved = json.loads(path.read_text())
    assert saved["failure"] == {"type": "LoopbackBenchmarkError", "diagnostics": value}
    assert saved["complete"] is saved["checks"]["passed"] is False
    assert saved["candidates"][0]["report"] is None
    assert SECRET not in path.read_text()
    capsys.readouterr()


def test_existing_history_record_still_readable_without_diagnostics():
    from pllm.runtime.benchmark_history import BenchmarkHistory, BenchmarkRun
    history = BenchmarkHistory(":memory:")
    try:
        old = BenchmarkRun(run_id="old-run", status="failed", model_id="unit",
                           max_output_tokens=1, cold=True, warm=False,
                           started_at_ns=1, finished_at_ns=2)
        saved = history.add(old)
        assert history.get("old-run") == saved
        assert "diagnostics" not in saved
    finally:
        history.close()


@pytest.mark.parametrize("with_collector", [False, True])
@pytest.mark.parametrize("context", [request_context(), *COUNT_CONTEXTS])
def test_dashboard_request_context_survives_history_and_artifact(tmp_path, monkeypatch, with_collector, context):
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.benchmark_history import BenchmarkHistory

    config = dashboard.DashboardConfig("127.0.0.1", 1, "unused", "unit", 2, backend="native")
    store = NS(finish_run_window=lambda _: {}, snapshot=lambda: {
        "services": {"pllm-preparation": {}}, "traffic": {}})
    runtime = dashboard.DashboardRuntime(config, store)
    runtime._topology = NS(requires_preparation=False, statuses=[], startup_breadcrumbs={
        "phase": "ready", "spawned_roles": ["worker_a", "worker_b"],
        "ready_roles": ["worker_a", "worker_b"]})
    runtime._active_run = dashboard._RunCapture("context-run", 2, False, 1, {}, 1, 0)
    runtime._has_started_run = True
    runtime._state.update(phase="preparing", run_id="context-run", prompt=SECRET)
    runtime._audit_snapshot = lambda: {}
    runtime._services_healthy = lambda: True
    runtime._capture_online_telemetry = lambda: None
    if with_collector:
        runtime._benchmark_failure_diagnostics = diagnostics.FailureDiagnostics(runtime)
    error = ProtocolError(SECRET, 503, {"body": SECRET})
    error.annotate_request_failure(context)
    closed = []

    class Stream:
        def __iter__(self):
            raise error

        def close(self):
            closed.append(True)

    runtime._client = NS(tokenizer_scope=nullcontext, responses=NS(create=lambda **kw: Stream()), _core=NS())
    runtime.history = BenchmarkHistory(":memory:")
    monkeypatch.setattr(dashboard.time, "sleep", lambda _: None)
    try:
        runtime._execute_run("context-run", SECRET, 2)
        saved = runtime.history.get("context-run")
        assert saved["status"] == "failed" and saved["failure"]["type"] == "ProtocolError"
        value = saved["failure"]["diagnostics"]
        assert value["request_failure_context"] == context
        if with_collector:
            assert value["ready_roles"] == ["worker_a", "worker_b"]
        assert SECRET not in json.dumps(saved)
        assert error.body == {"body": SECRET} and closed == [True]
        path = tmp_path / "context-failure.json"
        output = BenchmarkOutput(path, force=False, configuration={}, experiments=[None])
        output.start(1)
        output.failed(benchmark_cli.LoopbackBenchmarkError(SECRET, failure_diagnostics=value))
        artifact = json.loads(path.read_text())
        assert artifact["failure"]["diagnostics"]["request_failure_context"] == context
        assert SECRET not in path.read_text()
    finally:
        runtime.history.close()


@pytest.mark.parametrize("failure_stage", ["tokenizer_enter", "preparation", "stream", "tokenizer_exit"])
def test_request_failure_frozen_before_finish_and_cleanup(monkeypatch, failure_stage):
    """Healthy services return the actual finished request to ready, not its diagnostic."""
    config = dashboard.DashboardConfig("127.0.0.1", 1, "unused", "unit", 1, backend="native")
    store = NS(finish_run_window=lambda _: {}, snapshot=lambda: {
        "services": {"pllm-preparation": {}}, "traffic": {}})
    runtime = dashboard.DashboardRuntime(config, store)
    runtime._topology = NS(requires_preparation=False, statuses=[], startup_breadcrumbs={
        "phase": "ready", "spawned_roles": ["worker_a"], "ready_roles": ["worker_a"]})
    runtime._active_run = dashboard._RunCapture("unit-run", 1, False, 1, {}, 1, 0)
    runtime._has_started_run = True
    runtime._state.update(phase="preparing", run_id="unit-run", prompt=SECRET)
    runtime._audit_snapshot = lambda: {}
    runtime._services_healthy = lambda: True
    runtime._capture_online_telemetry = lambda: None
    collector = diagnostics.FailureDiagnostics(runtime, NS(failure_snapshot=rss_snapshot))
    runtime._benchmark_failure_diagnostics = collector
    calls = []

    def fail():
        calls.append("failure")
        raise ValueError(SECRET)

    @contextmanager
    def tokenizer_scope():
        if failure_stage == "tokenizer_enter":
            fail()
        yield
        if failure_stage == "tokenizer_exit":
            assert runtime._active_run is None and runtime._state["phase"] == "ready"
            fail()

    def captured_cleanup(name):
        assert collector._snapshot is not None and collector._snapshot["phase"] == "request"
        calls.append(name)

    class Stream:
        def __iter__(self):
            yield {"type": "response.created", "response": {"id": "unit-response"}}
            if failure_stage == "stream":
                fail()
            yield {"type": "response.completed", "response": {
                "id": "unit-response", "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1}}}

        def close(self):
            if failure_stage == "stream":
                captured_cleanup("stream_close")

    runtime._client = NS(tokenizer_scope=tokenizer_scope, responses=NS(create=lambda **k: Stream()),
                         _core=NS(evict_response=lambda _: captured_cleanup("evict")
                                  if failure_stage == "stream" else None))
    if failure_stage == "preparation":
        runtime.memory_preflight = {"estimate": {"max_input_tokens": 1}}
        runtime._client._core._response_input_tokens = lambda *a: fail()
    finish_run = runtime._finish_run

    def finish(*args):
        if failure_stage != "tokenizer_exit":
            captured_cleanup("finish_run")
        finish_run(*args)
    runtime._finish_run = finish
    monkeypatch.setattr(dashboard.time, "sleep", lambda _: None)
    runtime._execute_run("unit-run", SECRET, 1)
    assert runtime._active_run is None and runtime._state["phase"] == "ready"
    assert collector.capture()["phase"] == "request"
    assert SECRET not in json.dumps(collector.capture())
    if failure_stage == "stream":
        assert calls == ["failure", "finish_run", "stream_close", "evict"]
    elif failure_stage != "tokenizer_exit":
        assert calls == ["failure", "finish_run"]


def test_abort_publishes_stop_before_nonblocking_snapshot_with_topology_lock_held(tmp_path, monkeypatch):
    config = dashboard.DashboardConfig("127.0.0.1", 1, "unused", "unit", 1, backend="native")
    runtime = dashboard.DashboardRuntime(config, NS())
    topology = servers.LocalTopology.__new__(servers.LocalTopology)
    topology._process_lock = threading.Lock()
    topology._stopping = threading.Event()
    topology._startup_breadcrumbs = {"phase": "role_startup", "role": "worker_a",
                                    "spawned_roles": [], "ready_roles": []}
    topology._failure_observer = None
    calls, errors = [], []

    def cached_sample():
        assert runtime._stopping.is_set() and topology._stopping.is_set()
        calls.append("capture")
        return rss_snapshot()
    collector = diagnostics.FailureDiagnostics(runtime, NS(failure_snapshot=cached_sample))
    runtime._benchmark_failure_diagnostics = collector
    monkeypatch.setattr(dashboard, "build_roles", lambda *a, **k: topology)
    monkeypatch.setattr(memory, "execution_memory_admission", lambda *a, **k: {
        "admitted": True, "selected_backend": "native", "estimate": {"provider_peak_bytes": {}}})
    monkeypatch.setattr(memory, "MemoryWatchdog", lambda report, abort:
                        NS(start=lambda: None, abort=abort, failure_observation=None))
    monkeypatch.setattr(dashboard.tempfile, "TemporaryDirectory", lambda **k:
                        NS(name=str(tmp_path), cleanup=lambda: None))

    def close(self):
        assert collector._snapshot is not None
        calls.append("teardown")
    monkeypatch.setattr(servers.LocalTopology, "close", close)

    def start(self):
        def abort():
            try:
                runtime._memory_guard.abort("warning")
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=abort, daemon=True)
        with self._process_lock:
            thread.start()
            assert runtime._stopping.wait(1), "runtime cancellation blocked by diagnostics"
            assert self._stopping.wait(1), "topology cancellation blocked by diagnostics"
            thread.join(timeout=1)
            assert not thread.is_alive(), "diagnostics waited for topology lock"
        thread.join(timeout=1)
        if errors:
            raise errors[0]
        raise RuntimeError("startup cancelled")
    monkeypatch.setattr(servers.LocalTopology, "start", start)
    asyncio.run(runtime.start())
    assert not errors
    assert calls == ["capture", "teardown"]
    assert collector.capture()["phase"] == "unknown"
    assert collector.capture()["qualification"] == {"priced": None, "completed": None, "retired": None}


def test_all_diagnostic_locks_are_nonblocking():
    runtime = minimal_runtime()
    collector = diagnostics.FailureDiagnostics(runtime)
    with runtime._lock:
        assert collector.observe()["phase"] == "unknown"
    with collector._lock:
        assert collector.capture() is None
    sampler = benchmark_resources.ProcessMemorySampler(runtime)
    with sampler._snapshot_lock:
        assert sampler.failure_snapshot() is None


@pytest.mark.parametrize("outcome", ["success", "privacy_failure", "guard_during_cleanup"])
def test_final_checks_use_pre_teardown_fallback_without_claiming_first_failure(monkeypatch, outcome):
    runtime = minimal_runtime()
    breadcrumb = {"phase": "ready", "spawned_roles": ["worker_a"], "ready_roles": ["worker_a"]}
    runtime._topology = NS(startup_breadcrumbs=breadcrumb)
    runtime._client = None
    runtime.memory_preflight = None
    runtime.source_lock_digest = None
    runtime.store = NS(stage_body_snapshot=lambda **k: {})
    runtime.initial_preparation_audit = lambda: {}
    runtime.cold_process_cpu = lambda: None
    runtime.client_body_placement = lambda: {}
    calls, closed, fallback = [], [False], []

    def cached_sample():
        assert not closed[0], "diagnostic read after teardown"
        calls.append("observe")
        return rss_snapshot()

    def stop():
        calls.append("sampler_stop")
        collector = runtime._benchmark_failure_diagnostics
        assert collector._snapshot is None, "fallback must not consume first-failure slot"
        if outcome == "guard_during_cleanup":
            runtime._memory_guard = NS(error="warning", minimum_available_bytes=3 << 30,
                maximum_swap_growth_bytes=0, failure_observation={
                    "pressure": "warning", "available_bytes": 3 << 30, "swap_used_bytes": 0})
            collector.capture(phase="request")
        return {"sampled_total_peak_rss_bytes": 180}

    def close():
        calls.append("handle_close")
        closed[0] = True
        runtime._state.update(phase="error", startup_step="unknown")
        breadcrumb.clear()

    monkeypatch.setattr(memory, "process_memory", lambda: {})
    monkeypatch.setattr(dashboard, "create_dashboard_app", lambda _: NS(state=NS(dashboard_runtime=runtime)))
    monkeypatch.setattr(benchmark_resources, "ProcessMemorySampler", lambda _: NS(
        start=lambda: None, stop=stop, failure_snapshot=cached_sample))
    monkeypatch.setattr(benchmark_cli, "_free_port", lambda: 1)
    monkeypatch.setattr(benchmark_cli, "_DashboardHandle", lambda *a: NS(
        start=lambda: None, close=close, diagnostic=lambda: ""))
    monkeypatch.setattr(benchmark_cli, "_wait_for_dashboard_listener", lambda *a: None)
    monkeypatch.setattr(benchmark_cli, "_wait_for_ready", lambda *a: runtime._state.update(phase="ready"))
    monkeypatch.setattr(benchmark_cli.httpx, "Client", lambda **k: nullcontext(object()))
    monkeypatch.setattr(benchmark_cli, "_run_once", lambda *a, **k: {"status": "completed"})

    def build(**kwargs):
        assert closed[0], "preserve existing post-teardown check ordering"
        return {"configuration": {"identity": "unchanged"},
                "checks": {"passed": outcome != "privacy_failure"}}
    monkeypatch.setattr(benchmark_cli, "build_loopback_report", build)
    observe_failure = diagnostics.observe_failure

    def observe(current):
        value = observe_failure(current)
        fallback.append(value)
        return value
    monkeypatch.setattr(diagnostics, "observe_failure", observe)

    def run():
        return benchmark_cli._run_loopback_benchmark(model="unused", model_id=None, tiny=False,
            prompt=SECRET, max_output_tokens=1, warmups=0, repetitions=1, timeout_seconds=1)
    if outcome == "success":
        report = run()
        assert report["configuration"]["identity"] == "unchanged"
        assert "diagnostics" not in json.dumps(report)
        assert runtime._benchmark_failure_diagnostics._snapshot is None
    else:
        with pytest.raises(benchmark_cli.LoopbackBenchmarkError,
                           match="runtime or privacy checks failed") as caught:
            run()
        value = caught.value.failure_diagnostics
        assert value["process_rss"] == rss_snapshot()
        assert value["spawned_roles"] == value["ready_roles"] == ["worker_a"]
        assert value["phase"] == ("request" if outcome == "guard_during_cleanup" else "ready")
        assert value["host"]["pressure"] == ("warning" if outcome == "guard_during_cleanup" else None)
    assert fallback[0]["phase"] == "ready"
    assert calls == (["observe", "sampler_stop", "observe", "handle_close"]
                     if outcome == "guard_during_cleanup" else ["observe", "sampler_stop", "handle_close"])
