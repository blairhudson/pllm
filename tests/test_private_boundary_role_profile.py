"""Model-free role bootstrap tests. No checkpoint, role process or timing jobs."""
import asyncio
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.research import private_boundary_role_profile as p
from benchmarks.research.sota_combinations import Rejected
from pllm._benchmark_contract import digest
from pllm.metrics import BuildRef
from test_private_boundary_decode_profile import profile_cohort, cohort  # noqa: F401


def head_row():
    return {"elapsed_ns": 1000, "process_cpu_ns": 100, "handler_thread_cpu_ns": 20,
        "completed": True, "spans": [{"label": label, "start_ns": 10 + i * 200,
            "end_ns": 110 + i * 200, "calling_thread_cpu_ns": 10} for i, label in enumerate(p.LABELS)]}


def ordinary_estimate():
    return {"client_peak_bytes": 100 << 20, "provider_peak_bytes": {
        "inference": 200 << 20, "preparation": 300 << 20,
        "boundary_a": 400 << 20, "boundary_b": 500 << 20},
        "native_total_peak_bytes": 1500 << 20, "components": {"synthetic_ordinary_bytes": 123}}


def admission(estimate):
    return {"schema": "pllm.benchmark_memory.v1", "admitted": True,
        "requested_backend": "native", "selected_backend": "native", "enforced_wan": True,
        "host": {"admission_budget_bytes": 8 << 30}, "estimate": deepcopy(estimate),
        "candidates": {"native": {"admitted": True, "reasons": [],
            "required_host_bytes": estimate["native_total_peak_bytes"] + (32 << 20)}}}


def admission_details():
    baseline = ordinary_estimate()
    charged = deepcopy(baseline)
    charged.update(client_peak_bytes=140 << 20, native_total_peak_bytes=1548 << 20)
    charged["provider_peak_bytes"].update(boundary_a=404 << 20, boundary_b=504 << 20)
    charged["components"].update(response_state_audit_bytes=32 << 20,
        private_boundary_profile_bytes=16 << 20, private_boundary_role_profile_owners=dict(p.OWNERS))
    charged[p.BASELINE_KEY] = deepcopy(baseline)
    charged[p.ORDER_KEY] = "research_then_audit"
    return {"topology_admission": admission(charged),
        "report": {"memory_preflight": admission(baseline)}}


@pytest.fixture
def role_cohort(profile_cohort):
    from pllm.metrics import BenchmarkSuite
    from pllm.metrics.benchmark_runner import _result
    old_suite = BenchmarkSuite.from_spec(profile_cohort["suite"])
    suite = p.matched_suite(build=old_suite.cases[0].build)
    result = {**profile_cohort, "suite": suite.to_spec(), "results": []}
    for row in profile_cohort["results"]:
        execution = deepcopy(row["execution"])
        name = execution["case"]["name"]
        if name.split("--", 1)[1] != p.NAME:
            continue
        case, = [c for c in suite.cases if c.name == name]
        details = execution["details"]
        memory = admission_details()
        details["topology_admission"] = memory["topology_admission"]
        details["report"]["memory_preflight"] = memory["report"]["memory_preflight"]
        details["research_profile_identity"]["schema"] = p.IDENTITY
        comparison = p.client.ordinary_comparison(case.target, details)
        execution["comparison"]["measurement_digest"] = digest({"ordinary": comparison["measurement_digest"],
            "research": p.IDENTITY, "mode": name.split("--")[0], "sources": "c" * 64})
        if name.startswith("probe--"):
            details["private_boundary_decode_profile"]["schema"] = p.IDENTITY
            details["private_boundary_role_profile"] = {"schema": p.IDENTITY,
                "clock_alignment": "role_local_only", "roles": {role: [head_row() for _ in range(8)] for role in p.ROLES}}
        commitment = p.telemetry_digest(details)
        details["research_profile_identity"]["telemetry_digest"] = commitment
        execution["comparison"]["measurement_digest"] = digest({
            "ordinary_research": execution["comparison"]["measurement_digest"], "role_telemetry": commitment})
        payload = {key: execution[key] for key in ("failure_code", "details", "comparison",
            "fixture", "admission", "oracle")}
        payload.update({key: row[key] for key in ("status", "metrics", "samples", "environment")})
        result["results"].append(_result(case, suite, execution["replicate"], result["cohort"],
            execution["build"], payload).to_dict())
    return result


def test_matched_six_clients_and_frozen_auditor_namespace():
    suite = p.matched_suite(build=BuildRef(expected_digest="a" * 64))
    assert len(suite.cases) == 2 and suite.sampling.processes == 3
    assert suite.sampling.threads == 1 and suite.sampling.warmups == 0
    assert suite.cases[0].target.to_spec() == suite.cases[1].target.to_spec()
    assert p.client.IDENTITY != p._engine().IDENTITY
    assert len(p.client.NAMES) == 3


def test_replay_and_nonadditive_role_local_output(role_cohort):
    result = p.audit(role_cohort, source_identity="c" * 64)
    assert result["clients"] == 6 and len(result["role_profiles"]) == 3
    role = result["role_profiles"][0]["paired_roles"]["boundary_a"]
    assert role["decode_denominator"] == 7
    assert role["decode_head_ordinals"] == list(range(1, 8))
    assert sum(role["heads"][0]["role_local"]["exclusive_wall_ns"].values()) == 1000


@pytest.mark.parametrize("mutation", ["build", "metrics", "cpu", "positive_cpu", "valid_span", "overlap", "label", "missing", "clock", "control", "admission", "window", "state", "source", "span_extra"])
def test_replay_rejects_tampering(role_cohort, mutation):
    execution = role_cohort["results"][-1]["execution"]
    details = execution["details"]
    record = details["private_boundary_role_profile"]["roles"]["boundary_a"][0]
    if mutation == "build":
        execution["build"]["digest"] = "b" * 64
    elif mutation == "metrics":
        next(m for m in role_cohort["results"][-1]["metrics"] if m["id"] == "response.decode_tps")["value"] += 1
    elif mutation == "cpu":
        record["process_cpu_ns"] = -1
    elif mutation == "positive_cpu":
        record["process_cpu_ns"] += 1
    elif mutation == "valid_span":
        record["spans"][0]["end_ns"] += 1
    elif mutation == "overlap":
        record["spans"][1]["start_ns"] = 11
    elif mutation == "label":
        record["spans"][0]["label"] = "pure_network"
    elif mutation == "missing":
        details["private_boundary_role_profile"]["roles"]["boundary_a"].pop()
    elif mutation == "clock":
        details["private_boundary_role_profile"]["clock_alignment"] = "same_host"
    elif mutation == "control":
        role_cohort["results"][0]["execution"]["details"]["private_boundary_role_profile"] = {}
    elif mutation == "admission":
        details["topology_admission"]["estimate"]["components"]["private_boundary_role_profile_owners"].pop("boundary_a_bytes")
    elif mutation == "window":
        details["private_boundary_decode_profile"]["windows_relative_ns"]["after_first_output"][0] += 10000
    elif mutation == "state":
        details["response_state_audit"]["bindings"][0]["cryptographic_randomness_frozen"] = True
    elif mutation == "source":
        details["research_profile_identity"]["sources"] = "d" * 64
    else:
        record["spans"][0]["payload"] = "forbidden"
    with pytest.raises(Rejected):
        p.audit(role_cohort, source_identity="c" * 64)


def test_rebound_invalid_role_evidence_still_rejected(role_cohort):
    execution = role_cohort["results"][-1]["execution"]
    details = execution["details"]
    details["private_boundary_role_profile"]["roles"]["boundary_a"][0]["spans"][0]["end_ns"] = 9999999
    commitment = p.telemetry_digest(details)
    details["research_profile_identity"]["telemetry_digest"] = commitment
    target = p.matched_suite(build=BuildRef(expected_digest="a" * 64)).cases[1].target
    base = digest({"ordinary": p.client.ordinary_comparison(target, details)["measurement_digest"],
        "research": p.IDENTITY, "mode": "probe", "sources": "c" * 64})
    execution["comparison"]["measurement_digest"] = digest({"ordinary_research": base, "role_telemetry": commitment})
    with pytest.raises(Rejected):
        p.audit(role_cohort, source_identity="c" * 64)


def test_control_worker_uses_original_executor_and_binds_telemetry(monkeypatch):
    from pllm.runtime import benchmark_targets
    target = p.matched_suite(build=BuildRef(expected_digest="a" * 64)).cases[0].target
    calls = []
    def execute(target, measurement):
        calls.append(target)
    @contextmanager
    def children(script, outputs, enabled):
        assert outputs == 8 and enabled is False
        yield {}
    monkeypatch.setattr(p, "role_children", children)
    monkeypatch.setattr(benchmark_targets, "execute_target", execute)
    measurement = SimpleNamespace(details=admission_details(), comparison={"workload_digest": "w", "measurement_digest": "m"}, limitations=[])
    with p.worker_bootstrap("control", "c" * 64, Path("driver.py")):
        benchmark_targets.execute_target(target, measurement)
    assert calls == [target] and benchmark_targets.execute_target is execute
    assert measurement.details["research_profile_identity"]["telemetry_digest"] == p.telemetry_digest(measurement.details)


def test_record_return_exception_and_bounded_collection(tmp_path):
    record = p.RequestRecord()
    secret = object()
    assert record.wrap("native_head_dispatch", lambda x: x)(secret) is secret
    error = ValueError("PRIVATE")
    def fail():
        raise error
    with pytest.raises(ValueError) as caught:
        record.wrap("reply_codec", fail)()
    assert caught.value is error
    row = record.finish(False)
    assert not row["completed"] and "PRIVATE" not in json.dumps(row)
    path = tmp_path / "role"
    path.touch(mode=0o600)
    sink = p.Sink(path)
    sink.emit(head_row())
    sink.close()
    assert p.collect(path, 1) == [head_row()]
    with pytest.raises(ValueError):
        p.collect(path, 2)
    path.write_bytes(b"{}")
    with pytest.raises(ValueError):
        p.collect(path, 1)


def test_safe_collection_rejects_symlink_permissions_and_size(tmp_path):
    path = tmp_path / "role"
    path.touch(mode=0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        p.Sink(link)
    path.chmod(0o644)
    with pytest.raises(ValueError):
        p.collect(path, 1)
    path.chmod(0o600)
    with path.open("wb") as stream:
        stream.truncate(p.MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError):
        p.collect(path, 1)


def test_sink_and_request_bounds(tmp_path, monkeypatch):
    path = tmp_path / "role"
    path.touch(mode=0o600)
    monkeypatch.setattr(p, "MAX_RECORDS", 1)
    sink = p.Sink(path)
    try:
        sink.emit(head_row())
        with pytest.raises(RuntimeError):
            sink.emit(head_row())
    finally:
        sink.close()
    record = p.RequestRecord()
    for _ in range(10):
        record.wrap("native_head_dispatch", lambda: None)()
    assert len(record.spans) == 8 and record.failed


def test_asgi_preserves_bytes_failure_cancellation_and_context():
    async def run(error=None):
        rows, sent = [], []
        secret = object()
        async def app(scope, receive, send):
            record = p._REQUEST.get()
            assert await asyncio.to_thread(record.wrap("native_head_dispatch", lambda: secret)) is secret
            if error:
                raise error
            record.wrap("reply_codec", lambda: secret)()
            record.wrap("reply_frame", lambda: secret)()
            await send(secret)
        async def send(value):
            sent.append(value)
        handler = p.Handling(app, SimpleNamespace(emit=rows.append))
        if error:
            with pytest.raises(type(error)) as caught:
                await handler({"type": "http", "path": "/private/execute"}, None, send)
            assert caught.value is error and not rows[0]["completed"]
        else:
            await handler({"type": "http", "path": "/private/execute"}, None, send)
            assert sent == [secret]
            p.summarize_role(rows[0])
        assert p._REQUEST.get() is None
    asyncio.run(run())
    asyncio.run(run(ValueError("PRIVATE")))
    asyncio.run(run(asyncio.CancelledError()))


def test_pending_native_cancellation_does_not_join_or_recycle():
    import threading
    async def run():
        rows, pending = [], []
        started, release = threading.Event(), threading.Event()
        def native():
            started.set()
            release.wait(timeout=5)
        async def app(scope, receive, send):
            record = p._REQUEST.get()
            task = asyncio.create_task(asyncio.to_thread(record.wrap("native_head_dispatch", native)))
            pending.append(task)
            await asyncio.shield(task)
        handler = p.Handling(app, SimpleNamespace(emit=rows.append))
        task = asyncio.create_task(handler({"type": "http", "path": "/execute"}, None, None))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not pending[0].done() and rows == []
        finally:
            release.set()
            await pending[0]
        assert p._REQUEST.get() is None
    asyncio.run(run())


def test_role_hooks_restore_and_native_optional_dispatch(monkeypatch):
    from pllm.runtime.native import CompiledMatrix
    from pllm.runtime import private_boundary_role as role
    monkeypatch.setattr(CompiledMatrix, "_execute", lambda self, operation, *args: args[0])
    original = CompiledMatrix._execute
    create = role.create_private_boundary_role_app
    record = p.RequestRecord()
    token = p._REQUEST.set(record)
    try:
        with pytest.raises(RuntimeError):
            with p.role_instrument(SimpleNamespace(emit=lambda row: None)):
                secret = object()
                assert CompiledMatrix._execute(None, "future_byte_dot", secret) is secret
                raise RuntimeError()
    finally:
        p._REQUEST.reset(token)
    assert CompiledMatrix._execute is original and role.create_private_boundary_role_app is create
    assert [s["label"] for s in record.spans] == ["native_head_dispatch"]


def test_child_allowlist_control_and_cleanup(monkeypatch):
    import subprocess
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        return "child"
    monkeypatch.setattr(subprocess, "Popen", popen)
    with p.role_children(Path("/research/driver.py"), 8, False) as records:
        for role in p.ROLES:
            assert subprocess.Popen(["python", "-m", "pllm.runtime.private_boundary_role",
                "--role", role, "--host", "127.0.0.1", "--port", "1234"]) == "child"
        subprocess.Popen(["python", "-m", "provider"])
    assert not records and commands[-1] == ["python", "-m", "provider"]
    assert commands[0][1:5] == ["-I", "/research/driver.py", "--role-worker", commands[0][4]]
    assert not Path(commands[0][4]).parent.exists()
    assert subprocess.Popen is popen
    with pytest.raises(ValueError, match="allowlisted"):
        with p.role_children(Path("/research/driver.py"), 8, True):
            subprocess.Popen(["python", "-m", "pllm.runtime.private_boundary_role", "--bad"])
    with pytest.raises(KeyboardInterrupt):
        with p.role_children(Path("/research/driver.py"), 8, True):
            raise KeyboardInterrupt()
    assert subprocess.Popen is popen


def test_both_arm_memory_owners(monkeypatch):
    from pllm.runtime import benchmark_memory as memory
    monkeypatch.setattr(memory, "host_memory", lambda: "host")
    monkeypatch.setattr(memory, "admit_memory", lambda estimate, host, **kwargs: estimate)
    report = {"estimate": ordinary_estimate()}
    result = p.priced_admission(report, 8 << 30)
    assert result["client_peak_bytes"] == 108 << 20
    assert result["provider_peak_bytes"]["boundary_a"] == 404 << 20
    assert result["provider_peak_bytes"]["boundary_b"] == 504 << 20
    assert result["native_total_peak_bytes"] == 1516 << 20
    assert result[p.BASELINE_KEY] == ordinary_estimate()
    assert report["estimate"] == ordinary_estimate()
    with pytest.raises(ValueError):
        p.priced_admission({"estimate": result}, 8 << 30)


def test_preflight_actual_order_audit_then_research(monkeypatch):
    from pllm import metrics
    from pllm.runtime import benchmark_memory as memory
    observed = []
    def admit(estimate, *args, **kwargs):
        observed.append(deepcopy(estimate))
        return admission(estimate)
    monkeypatch.setattr(metrics, "benchmark_memory", lambda *args, **kwargs: admission(ordinary_estimate()))
    monkeypatch.setattr(memory, "admit_memory", admit)
    monkeypatch.setattr(memory, "host_memory", lambda: None)
    suite = p.matched_suite(build=BuildRef(expected_digest="a" * 64))
    rows = p.preflight(suite)
    assert len(rows) == 2
    for i, row in enumerate(rows):
        audited, charged = observed[2 * i:2 * i + 2]
        assert audited["client_peak_bytes"] == 132 << 20
        assert audited["native_total_peak_bytes"] == 1532 << 20
        assert charged["client_peak_bytes"] == 140 << 20
        assert charged["native_total_peak_bytes"] == 1548 << 20
        assert charged[p.BASELINE_KEY] == ordinary_estimate()
        assert charged[p.ORDER_KEY] == "audit_then_research"
        assert charged == row["admission"]["estimate"]


def test_runtime_actual_order_research_then_audit(monkeypatch):
    from pllm import metrics
    from pllm.runtime import benchmark_memory as memory, benchmark_targets
    from pllm.runtime.benchmark_response_audit import state_audit_admission
    observed = []
    def admit(estimate, *args, **kwargs):
        observed.append(deepcopy(estimate))
        return admission(estimate)
    monkeypatch.setattr(metrics, "benchmark_memory", lambda *args, **kwargs: admission(ordinary_estimate()))
    monkeypatch.setattr(memory, "admit_memory", admit)
    monkeypatch.setattr(memory, "host_memory", lambda: None)
    def execute(target, measurement):
        report = metrics.benchmark_memory(None, memory_budget_bytes=8 << 30)
        measurement.details["topology_admission"] = state_audit_admission(report, 8 << 30, enforced_wan=True)
        measurement.details["report"] = {"memory_preflight": admission(ordinary_estimate())}
    @contextmanager
    def children(*args):
        yield {}
    monkeypatch.setattr(p, "role_children", children)
    monkeypatch.setattr(benchmark_targets, "execute_target", execute)
    target = p.matched_suite(build=BuildRef(expected_digest="a" * 64)).cases[0].target
    measurement = SimpleNamespace(details={}, comparison={"workload_digest": "w", "measurement_digest": "m"}, limitations=[])
    with p.worker_bootstrap("control", "c" * 64, Path("driver.py")):
        benchmark_targets.execute_target(target, measurement)
    research, audited = observed
    assert research["client_peak_bytes"] == 108 << 20
    assert research["native_total_peak_bytes"] == 1516 << 20
    assert audited["client_peak_bytes"] == 140 << 20
    assert audited["native_total_peak_bytes"] == 1548 << 20
    assert audited[p.BASELINE_KEY] == ordinary_estimate()
    assert audited[p.ORDER_KEY] == "research_then_audit"
    p.validate_admission(measurement.details)


@pytest.mark.parametrize("mutation", ["missing_client", "missing_providers", "missing_total", "client", "float_client", "boundary_a",
    "boundary_b", "inference", "total", "audit_omitted", "audit_twice", "baseline_missing", "baseline_client",
    "baseline_role", "baseline_total", "baseline_metadata", "coherent_baseline", "baseline_audited", "order", "raw", "candidate", "raw_candidate"])
def test_numerical_admission_rejects_correct_labels(role_cohort, mutation):
    execution = role_cohort["results"][-1]["execution"]
    details = execution["details"]
    estimate = details["topology_admission"]["estimate"]
    baseline = estimate[p.BASELINE_KEY]
    if mutation.startswith("missing_"):
        estimate.pop({"missing_client": "client_peak_bytes", "missing_providers": "provider_peak_bytes",
            "missing_total": "native_total_peak_bytes"}[mutation])
    elif mutation == "client":
        estimate["client_peak_bytes"] -= 8 << 20
    elif mutation == "float_client":
        estimate["client_peak_bytes"] = float(estimate["client_peak_bytes"])
    elif mutation in {"boundary_a", "boundary_b", "inference"}:
        estimate["provider_peak_bytes"][mutation] -= 4 << 20
    elif mutation == "total":
        estimate["native_total_peak_bytes"] -= 16 << 20
    elif mutation in {"audit_omitted", "audit_twice"}:
        delta = (-1 if mutation == "audit_omitted" else 1) * (32 << 20)
        estimate["client_peak_bytes"] += delta
        estimate["native_total_peak_bytes"] += delta
    elif mutation == "baseline_missing":
        estimate.pop(p.BASELINE_KEY)
    elif mutation == "baseline_client":
        baseline["client_peak_bytes"] -= 1
    elif mutation == "baseline_role":
        baseline["provider_peak_bytes"]["boundary_a"] -= 1
    elif mutation == "baseline_total":
        baseline["native_total_peak_bytes"] -= 1
    elif mutation == "baseline_metadata":
        baseline["components"]["synthetic_ordinary_bytes"] += 1
        estimate["components"]["synthetic_ordinary_bytes"] += 1
    elif mutation == "coherent_baseline":
        for field in ("client_peak_bytes", "native_total_peak_bytes"):
            baseline[field] += 1
            estimate[field] += 1
        details["topology_admission"]["candidates"]["native"]["required_host_bytes"] += 1
    elif mutation == "baseline_audited":
        baseline["components"]["response_state_audit_bytes"] = 32 << 20
    elif mutation == "order":
        estimate[p.ORDER_KEY] = "audit_then_research"
    elif mutation == "raw":
        details["report"]["memory_preflight"]["estimate"]["client_peak_bytes"] += 1
    elif mutation == "candidate":
        details["topology_admission"]["candidates"]["native"]["required_host_bytes"] -= 16 << 20
    else:
        details["report"]["memory_preflight"]["candidates"]["native"]["required_host_bytes"] += 1
    # Numerical validation must reject even when attacker refreshes unkeyed binding.
    commitment = p.telemetry_digest(details)
    details["research_profile_identity"]["telemetry_digest"] = commitment
    target = p.matched_suite(build=BuildRef(expected_digest="a" * 64)).cases[1].target
    base = digest({"ordinary": p.client.ordinary_comparison(target, details)["measurement_digest"],
        "research": p.IDENTITY, "mode": "probe", "sources": "c" * 64})
    execution["comparison"]["measurement_digest"] = digest({"ordinary_research": base, "role_telemetry": commitment})
    assert estimate["components"]["private_boundary_role_profile_owners"] == p.OWNERS
    with pytest.raises(Rejected):
        p.audit(role_cohort, source_identity="c" * 64)


def test_replay_does_not_call_memory_planner(role_cohort, monkeypatch):
    from pllm import metrics
    from pllm.runtime import benchmark_memory as memory
    def forbidden(*args, **kwargs):
        raise AssertionError("read-only replay called memory planner")
    for owner, names in ((memory, ("benchmark_memory", "estimate_memory", "estimate_plan_memory", "admit_memory", "host_memory")),
            (metrics, ("benchmark_memory",))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    assert p.audit(role_cohort, source_identity="c" * 64)["complete"]


def test_prefix_exclusive_and_help(tmp_path):
    from scripts.profile_private_boundary_roles import main, reserve_output_paths
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    output = tmp_path / "out.json"
    reserve_output_paths(output)
    with pytest.raises(FileExistsError):
        reserve_output_paths(output)
