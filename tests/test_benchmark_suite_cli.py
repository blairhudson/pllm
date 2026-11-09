"""Real CLI/factory/cohort engine; model, transport and build probes doubled.

No checkpoint loading, native admission, material issuance or measured RSS. The
ordinary response/count executor and public report serializer remain real.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from pllm._cli.app import build_parser, main
from pllm._cli.errors import ResolutionError
from pllm._cli.targets import resolve_target
from pllm._benchmark_contract import TargetUnavailable, canonical, digest
from pllm._benchmark_worker import Measurement, PROTOCOL
from pllm.metrics import BenchmarkRun
from pllm.metrics import benchmark_runner as engine
from pllm.runtime import benchmark_cli as cli


SUPERVISE = engine._execute


FACTORY = '''from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm._benchmark_contract import digest
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario, TargetBudget
experiment = Experiment("suite-cli-fixture", MaskedLinearCpu(Model.path("not-a-checkpoint", model_id="unit")),
    Deployment.local(root="unused-roles"), ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=8))
def campaign():
    target = BenchmarkTarget.response(experiment, backend="native",
        scenario=Scenario.cold(max_output_tokens=8), state_audit=True,
        expected_input_tokens=64, expected_output_tokens=8)
    return BenchmarkSuite(tuple(BenchmarkCase(name, target, BuildRef(expected_digest=digest({})),
        (TargetBudget("client.peak_rss", maximum=1000),)) for name in ("baseline", "candidate")),
        Sampling(processes=2, warmups=0, repetitions=1, timeout_seconds=60, memory_bytes=256 << 20),
        name="suite-cli-fixture")
def legacy():
    return experiment
def artifact_campaign():
    return BenchmarkSuite((BenchmarkCase("artifact", BenchmarkTarget.artifacts(experiment, seed=123)),),
        Sampling(processes=1), name="artifact-cli-fixture")
def secret_campaign():
    return BenchmarkSuite((BenchmarkCase("artifact", BenchmarkTarget.artifacts(experiment,
        config={"private_key": "not-a-real-credential"})),), Sampling(processes=1), name="reject-secret")
'''


@pytest.fixture
def factory(tmp_path):
    path = tmp_path / "campaign.py"
    path.write_text(FACTORY)
    return f"{path}:campaign"


def arguments(factory, *extra):
    return ["benchmark", "run", "--experiment", factory, "--factory", "--trust-python",
            "--no-input", "--format", "json", *map(str, extra)]


@pytest.fixture
def runtime(monkeypatch):
    """Fake only execution boundaries, not parser/dispatch/count/report code."""
    from pllm.configuration import Experiment
    from pllm.runtime import benchmark_response_target as executor

    calls, workers = [], []
    control = {"input": 64, "output": 8, "authoritative": True, "checks": True,
               "admitted": True, "audit_failed": False, "build": {}}
    monkeypatch.setattr(Experiment, "resolve", lambda *a, **kw: pytest.fail("controller resolved a model"))
    monkeypatch.setattr(engine, "inspect_build", lambda _: {"digest": digest(control["build"]), "attributes": control["build"]})
    delegated = importlib.import_module("pllm.runtime.benchmark_delegated")
    for name in ("is_buffered", "is_replicated", "is_fss"):
        monkeypatch.setattr(delegated, name, lambda _: False)
    audit = importlib.import_module("pllm.runtime.benchmark_response_audit")
    monkeypatch.setattr(audit, "state_audit_admission", lambda value, *a, **kw: value)
    monkeypatch.setattr(audit, "response_state_audit", lambda *a, **kw: nullcontext([
        {"failed": control["audit_failed"], "selections": [{"checked_values": 2}] * control["output"]}]))
    loader = importlib.import_module("pllm.model_loader")
    monkeypatch.setattr(loader, "expected_model_id", lambda _: "unit")
    boundary = importlib.import_module("pllm._search_benchmark")
    monkeypatch.setattr(boundary, "_boundary", lambda _: {"numeric": {"bits": 8}, "privacy": {}})
    import pllm.metrics
    monkeypatch.setattr(pllm.metrics, "benchmark_memory", lambda *a, **kw: {"admitted": control["admitted"]})
    monkeypatch.setattr(executor, "response_metrics", lambda report, m: m.metric(
        "client.peak_rss", None, kind="memory", reason="mock_boundary_not_measured"))

    def transport(**options):
        calls.append(options)
        return {"schema_version": "pllm.loopback_benchmark.v1", "checks": {"passed": control["checks"]},
            "configuration": {"source_lock_digest": "a" * 64, "provider_backend": "native",
                "sampling": {"temperature": 0}, "prompt_digest": hashlib.sha256(
                    b"pllm.benchmark.prompt.v1\0" + options["_cohort_salt"] + options["prompt"].encode()).hexdigest()},
            "topology_accounting": {}, "runs": [{"status": "completed", "model_fingerprint": "c" * 64,
                "tokens": {"authoritative": control["authoritative"], "input_tokens": control["input"],
                           "output_tokens": control["output"]},
                "generation": {"output_text_digest": "d" * 64}}]}

    monkeypatch.setattr(cli, "run_loopback_benchmark", transport)

    def execute(case, sampling, inputs, salt, directory):
        workers.append((case, sampling, deepcopy(inputs), salt, directory))
        m = Measurement({"sampling": sampling.to_spec(), "salt": salt.hex(), "inputs": inputs,
                         "directory": str(directory)})
        m.admit = lambda required: m.admission.update(  # Host admission doubled; no resource claim.
            admitted=True, required_bytes=required, host_headroom_bytes=256 << 20)
        status, code = "completed", None
        try:
            executor.execute_response(case.target.to_spec()["parameters"], m)
        except TargetUnavailable as error:
            status, code = error.status, error.code
        m.finalize_workload(case.target, completed=status == "completed")
        return {"status": status, "failure_code": code, "metrics": m.metrics, "samples": m.samples,
            "fixture": m.fixture, "oracle": m.oracle, "admission": m.admission, "comparison": m.comparison,
            "details": m.details, "limitations": ["Mock execution boundaries; no numeric or memory evidence."],
            "environment": {"scope": "mock"}}

    monkeypatch.setattr(engine, "_execute", execute)
    return SimpleNamespace(control=control, calls=calls, workers=workers, execute=execute)


def test_real_parser_factory_engine_count_gate_and_public_serializer(factory, runtime, tmp_path, capsys):
    prompt, output = tmp_path / "prompt.txt", tmp_path / "run.json"
    prompt.write_text("private shared fixture\n")
    main(arguments(factory, "--prompt-file", prompt, "--output", output))
    document = json.loads(capsys.readouterr().out)["data"]["report"]
    run = BenchmarkRun.load(output)
    assert run.to_dict() == document
    assert document["schema"] == "pllm.benchmark_run.v1" and "schema_version" not in document
    assert document["complete"] is True and len(document["results"]) == 4
    assert len({row["execution"]["cohort"] for row in document["results"]}) == 1
    assert [row["execution"]["case"]["name"] for row in document["results"]] == [
        "baseline", "candidate", "candidate", "baseline"]
    assert len({worker[3] for worker in runtime.workers}) == 1
    assert document["cohort"] == hashlib.sha256(runtime.workers[0][3]).hexdigest()
    assert len({worker[4] for worker in runtime.workers}) == 4
    assert all(not worker[4].exists() for worker in runtime.workers)
    assert all(worker[2] == {"prompt": "private shared fixture\n"} for worker in runtime.workers)
    assert "private shared fixture" not in json.dumps(document)
    for row in document["results"]:
        execution = row["execution"]
        parameters = execution["case"]["target"]["parameters"]
        assert parameters["expected_input_tokens"] == 64 and parameters["expected_output_tokens"] == 8
        assert parameters["state_audit"] is True
        assert execution["case"]["build"]["expected_digest"] == digest({})
        assert execution["isolation"]["fresh_process"] is True
        assert execution["details"]["report"]["checks"]["expected_output_tokens"] is True
        assert execution["details"]["response_state_audit"]["bindings"]
        assert execution["comparison"]["output_digest"]
        assert execution["comparison"]["source_digest"] == "a" * 64
        assert execution["budgets"][0]["outcome"] == "unknown"
    assert run.compare("baseline", "candidate", "client.peak_rss")["status"] == "unranked"
    assert len({row["execution"]["comparison"]["output_digest"] for row in document["results"]}) == 1
    assert len({row["execution"]["comparison"]["workload_digest"] for row in document["results"]}) == 1


@pytest.mark.parametrize("changes,code", [
    ({"input": 63}, "response_expected_token_counts_failed"),
    ({"output": 7}, "response_expected_token_counts_failed"),
    ({"authoritative": False}, "response_checks_failed"),
    ({"checks": False}, "response_checks_failed"),
    ({"admitted": False}, "topology_memory_limit"),
    ({"audit_failed": True}, "response_state_audit_incomplete"),
    ({"build": {"different": True}}, "build_mismatch"),
])
def test_failed_unknown_guard_and_early_eos_are_not_promoted(factory, runtime, tmp_path, capsys, changes, code):
    runtime.control.update(changes)
    output = tmp_path / "failed.json"
    main(arguments(factory, "--prompt", "private fixture", "--output", output))
    document = json.loads(capsys.readouterr().out)["data"]["report"]
    run = BenchmarkRun.load(output)
    assert document == run.to_dict() and document["complete"] is True
    assert all(row["status"] != "completed" and row["execution"]["failure_code"] == code
               for row in document["results"])
    assert run.compare("baseline", "candidate", "client.peak_rss")["status"] == "unranked"
    if code == "response_expected_token_counts_failed":
        for row in document["results"]:
            execution = row["execution"]
            assert execution["details"]["report"]["runs"][0]["tokens"]["output_tokens"] == runtime.control["output"]
            assert execution["details"]["response_state_audit"]["bindings"]
            assert execution["comparison"]["output_digest"]


@pytest.mark.parametrize("options", [
    ("--max-output-tokens", "9"), ("--warmups", "1"), ("--backend", "docker"),
    ("--temperature", "0.8"), ("--timeout", "61"), ("--repetitions", "2"),
    ("--memory-budget-mib", "128"), ("--prefill-cache-mib", "1"),
    ("--wan",), ("--isolate-candidates",), ("--save-best", "not-written.json"),
    ("--model", "not-a-checkpoint"),
])
def test_conflicting_overrides_reject_before_workers(factory, runtime, capsys, options):
    with pytest.raises(SystemExit) as error:
        main(arguments(factory, *options))
    assert error.value.code == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "BENCHMARK_SUITE_OVERRIDE"
    assert not runtime.workers and not runtime.calls


def test_matching_explicit_options_do_not_drop_target_params(factory, runtime, capsys):
    main(arguments(factory, "--prompt", "private fixture", "--max-output-tokens", "8",
                   "--warmups", "0", "--repetitions", "1", "--timeout", "60", "--backend", "native",
                   "--temperature", "0"))
    assert json.loads(capsys.readouterr().out)["data"]["report"]["complete"] is True
    assert all(worker[0].target.to_spec()["parameters"]["state_audit"] for worker in runtime.workers)


def test_resolver_opt_in_default_and_trust_policy_unchanged(factory, runtime):
    options = dict(factory=True, no_input=True, trust_python=True, output_format="json")
    with pytest.raises(ResolutionError, match="Experiment"):
        resolve_target(factory, **options)
    selected = resolve_target(factory, allow_benchmark_suite=True, **options).configuration
    assert type(selected).from_spec(selected.to_spec()).digest == selected.digest
    with pytest.raises(ResolutionError) as error:
        resolve_target(factory, allow_benchmark_suite=True, **{**options, "trust_python": False})
    assert error.value.code == "PYTHON_TRUST_REQUIRED"


def test_nonresponse_public_fixture_seed_is_retained_without_weakening_secret_scan(factory, runtime, capsys):
    module = factory.rsplit(":", 1)[0]
    target = module + ":artifact_campaign"
    main(arguments(target, "--dry-run"))
    suite = json.loads(capsys.readouterr().out)["data"]["suite"]
    assert suite["cases"][0]["target"]["parameters"]["seed"] == 123
    selected = resolve_target(target, factory=True, no_input=True, trust_python=True,
                              output_format="json", allow_benchmark_suite=True).configuration
    assert canonical(suite) == canonical(selected.to_spec())
    with pytest.raises(SystemExit):
        main(arguments(module + ":secret_campaign", "--dry-run"))
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "CONFIGURATION_SECRET_FIELD"
    with pytest.raises(SystemExit):
        main(arguments(target, "--prompt", "no response target", "--dry-run"))
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "BENCHMARK_SUITE_OVERRIDE"
    assert not runtime.workers


def test_no_factory_flag_does_not_call_callable(factory, runtime, capsys):
    args = arguments(factory)
    args.remove("--factory")
    with pytest.raises(SystemExit):
        main(args)
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "TARGET_TYPE"
    assert not runtime.workers


def test_multiple_suites_or_mixed_targets_reject_before_model_resolution(factory, runtime, capsys):
    for other in (factory, factory.rsplit(":", 1)[0] + ":legacy"):
        with pytest.raises(SystemExit):
            main(arguments(factory, "--experiment", other))
        assert json.loads(capsys.readouterr().err)["error"]["code"] == "BENCHMARK_SUITE_TARGETS"
    assert not runtime.workers


def test_dry_run_keeps_public_identity_without_running_or_leaking_prompt(factory, runtime, tmp_path, capsys):
    output = tmp_path / "dry.json"
    main(arguments(factory, "--dry-run", "--prompt", "not in public spec", "--output", output))
    data = json.loads(capsys.readouterr().out)["data"]
    selected = resolve_target(factory, factory=True, no_input=True, trust_python=True,
                              output_format="json", allow_benchmark_suite=True).configuration
    assert canonical(data["suite"]) == canonical(selected.to_spec())
    assert data["report_schema"] == "pllm.benchmark_run.v1"
    assert "not in public spec" not in json.dumps(data)
    assert not output.exists() and not runtime.workers


def test_parser_legacy_defaults_unchanged():
    args = build_parser().parse_args(["benchmark", "run"])
    assert (args.warmups, args.repetitions, args.max_output_tokens, args.timeout) == (0, 1, 24, 900)
    assert not hasattr(args, "benchmark_explicit_options")


def test_legacy_experiment_dispatch_keeps_schema_defaults_and_public_spec(factory, runtime, monkeypatch, capsys):
    from pllm.configuration import Experiment
    legacy = factory.rsplit(":", 1)[0] + ":legacy"
    selected = resolve_target(legacy, factory=True, no_input=True, trust_python=True,
                              output_format="json").configuration
    before = canonical(selected.to_spec())
    resolves = []
    monkeypatch.setattr(Experiment, "resolve", lambda self: resolves.append(self.name))
    monkeypatch.setattr(Experiment, "configuration_digest", lambda self: digest(self.to_spec()))
    monkeypatch.setattr(type(selected.pipeline), "digest", lambda self: digest(self.to_spec()))
    # Legacy's own output cap must still be explicit when smaller than 24.
    main(arguments(legacy, "--max-output-tokens", "8", "--dry-run"))
    data = json.loads(capsys.readouterr().out)["data"]
    assert "suite" not in data and data["dry_run"] is True
    assert (data["configuration"]["warmups"], data["configuration"]["repetitions"],
            data["configuration"]["timeout_seconds"]) == (0, 1, 900)
    assert resolves and not runtime.workers
    assert canonical(selected.to_spec()) == before


@pytest.fixture
def supervision(runtime, monkeypatch):
    """Isolate every supervisor thread/process/probe boundary for selected tests."""
    from pllm._benchmark_contract import BenchmarkTarget, Sampling
    monkeypatch.setattr(engine, "_execute", SUPERVISE)
    memory = importlib.import_module("pllm.runtime.benchmark_memory")
    usage = importlib.import_module("pllm.runtime.client_process_usage")
    state = SimpleNamespace(groups=[], requests=[], events=[], monitors=[], finalized=[],
        mode="normal", snapshot_fault=False, diagnostics={})

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected host probe or uncontrolled execution")

    class Event:
        def __init__(self):
            self.stopped = False

        def set(self):
            self.stopped = True

        def is_set(self):
            return self.stopped

        wait = forbidden

    class Tracker:
        def __init__(self, *, target, daemon):
            assert daemon is True

        def start(self):
            # Deliberately never execute the tracking target.
            state.events.append("tracker-start")

        def join(self):
            state.events.append("tracker-join")

    class Guard:
        def __init__(self, report, abort):
            assert report == {"host": {"reserve_bytes": 0, "swap_used": 0, "available": 1 << 30}}
            self.abort = abort
            self.error_class = state.diagnostics.get("error_class")
            self.error = state.diagnostics.get("error_message")
            self.observation = deepcopy(state.diagnostics.get("failure_observation"))
            self.minimum_available_bytes = state.diagnostics.get("minimum_available_bytes")
            self.maximum_swap_growth_bytes = state.diagnostics.get("maximum_swap_growth_bytes")
            state.monitors.append(self)

        @property
        def failure_observation(self):
            if state.snapshot_fault:
                raise RuntimeError("secondary snapshot failure")
            return self.observation

        def start(self):
            state.events.append("guard-start")

        def close(self):
            state.events.append("guard-close")

    class Process:
        returncode = 0

        def __init__(self, command, **options):
            self.pid = 2147483646 - len(state.groups)
            state.groups.append((command, options))

        def communicate(self, body, timeout):
            request = json.loads(body)
            assert request["protocol"] == PROTOCOL
            state.requests.append(request)
            if state.mode == "cancel":
                raise KeyboardInterrupt
            if state.mode == "abort":
                monitor = state.monitors[-1]
                monitor.abort(monitor.error)
                if monitor.observation is not None:
                    monitor.observation.update(pressure="normal", available_bytes=999, swap_used_bytes=999)
                return b"", b""
            payload = runtime.execute(SimpleNamespace(target=BenchmarkTarget.from_spec(request["target"])),
                Sampling(**request["sampling"]), request["inputs"], bytes.fromhex(request["salt"]),
                Path(request["directory"]))
            return canonical({"protocol": PROTOCOL, **payload}), b""

        def poll(self):
            return None if state.mode == "cancel" else 0

        def wait(self, timeout=None):
            state.events.append("wait")

    def finalize(payload, process):
        state.finalized.append((payload, process.pid))
        state.events.append("finalize")

    monkeypatch.setattr(memory, "MemoryWatchdog", Guard)
    monkeypatch.setattr(memory, "host_memory", lambda: SimpleNamespace(reserve=0, swap_used=0, available=1 << 30))
    monkeypatch.setattr(engine, "subprocess", SimpleNamespace(**(vars(engine.subprocess) | {
        "Popen": Process, "run": forbidden})))
    monkeypatch.setattr(engine, "os", SimpleNamespace(**(vars(engine.os) | {
        "killpg": lambda pid, sig: state.events.append(sig)})))
    monkeypatch.setattr(engine, "threading", SimpleNamespace(Event=Event, Thread=Tracker))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=forbidden, Error=RuntimeError))
    monkeypatch.setattr(usage, "client_popen_factory", lambda factory: factory)
    monkeypatch.setattr(usage, "finalize_client_payload", finalize)
    return state


def test_normal_worker_supervision_still_owns_fresh_isolated_process_groups(
        factory, supervision, tmp_path, capsys):
    output = tmp_path / "supervised.json"
    main(arguments(factory, "--prompt", "private fixture", "--output", output))
    assert json.loads(capsys.readouterr().out)["data"]["report"] == BenchmarkRun.load(output).to_dict()
    assert len(supervision.groups) == len(supervision.requests) == 4
    assert len({request["salt"] for request in supervision.requests}) == 1
    assert len({request["directory"] for request in supervision.requests}) == 4
    assert len({pid for _, pid in supervision.finalized}) == 4
    for command, options in supervision.groups:
        assert command[1:] == ["-I", "-m", "pllm._benchmark_worker"]
        assert options["start_new_session"] is True
        assert "private fixture" not in str(command)
    assert supervision.events == ["tracker-start", "guard-start", "guard-close",
        signal.SIGKILL, "wait", "tracker-join", "finalize"] * 4
    assert all(payload["status"] == "completed" for payload, _ in supervision.finalized)


def test_cancellation_uses_existing_guard_drain_and_incomplete_public_checkpoint(
        factory, supervision, tmp_path, capsys):
    supervision.mode = "cancel"
    output = tmp_path / "cancelled.json"
    with pytest.raises(SystemExit) as error:
        main(arguments(factory, "--prompt", "private fixture", "--output", output))
    assert error.value.code == 130
    document = BenchmarkRun.load(output).to_dict()
    assert document["complete"] is False and len(document["results"]) == 1
    assert document["results"][0]["status"] == "cancelled"
    assert supervision.events == ["tracker-start", "guard-start", "guard-close",
        signal.SIGTERM, "wait", signal.SIGKILL, "wait", "tracker-join", "finalize"]
    assert supervision.finalized == [(None, 2147483646)]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("cause", ["resource_failure", "telemetry_exception"])
def test_controller_watchdog_diagnostics_persist(factory, supervision, tmp_path, capsys, cause):
    supervision.mode = "abort"
    resource = cause == "resource_failure"
    observation = {"pressure": "critical", "available_bytes": 0, "swap_used_bytes": 0,
        "extra_sample_field": "not public"} if resource else None
    supervision.diagnostics = {"error_class": ("BenchmarkMemoryError" if resource else "RuntimeError") + "x" * 80,
        "error_message": ("primary resource failure" if resource else "sample unavailable") + "x" * 512,
        "failure_observation": observation, "minimum_available_bytes": 0 if resource else 1 << 30,
        "maximum_swap_growth_bytes": 257 << 20 if resource else 0}
    expected = {**supervision.diagnostics,
        "error_class": supervision.diagnostics["error_class"][:80],
        "error_message": supervision.diagnostics["error_message"][:512],
        "failure_observation": {key: observation[key] for key in (
            "pressure", "available_bytes", "swap_used_bytes")} if resource else None}
    output = tmp_path / "watchdog.json"
    main(arguments(factory, "--prompt", "private fixture", "--output", output))
    document = json.loads(capsys.readouterr().out)["data"]["report"]
    assert document == json.loads(output.read_text()) == BenchmarkRun.load(output).to_dict()
    assert document["complete"] is True and len(document["results"]) == 4
    for row in document["results"]:
        assert row["status"] == "blocked" and row["execution"]["failure_code"] == "host_pressure"
        assert row["execution"]["details"] == {"controller_memory_watchdog": expected}
    assert len(expected["error_class"]) == 80 and len(expected["error_message"]) == 512
    if resource:
        assert all(monitor.observation["available_bytes"] == 999 for monitor in supervision.monitors)
    assert supervision.events == ["tracker-start", "guard-start", signal.SIGTERM,
        "guard-close", signal.SIGKILL, "wait", "tracker-join", "finalize"] * 4
    assert len(supervision.finalized) == 4 and all(payload is None for payload, _ in supervision.finalized)

    # Legacy/invalid retained fields must not fabricate zero or coerce strings.
    empty = {key: None for key in expected}
    assert engine._controller_memory_watchdog_details(SimpleNamespace()) == empty
    for invalid in (None, True, -1, "0"):
        monitor = SimpleNamespace(error_class=123, error=object(), minimum_available_bytes=invalid,
            maximum_swap_growth_bytes=invalid, failure_observation={"pressure": None,
                "available_bytes": invalid, "swap_used_bytes": 0})
        assert engine._controller_memory_watchdog_details(monitor) == empty
        monitor.failure_observation = {"pressure": None, "available_bytes": 0, "swap_used_bytes": invalid}
        assert engine._controller_memory_watchdog_details(monitor) == empty
    for invalid_observation in ({}, {"available_bytes": 0}, {"available_bytes": 0, "swap_used_bytes": 0},
            {"pressure": False, "available_bytes": 0, "swap_used_bytes": 0}):
        assert engine._controller_memory_watchdog_details(SimpleNamespace(
            failure_observation=invalid_observation)) == empty
    zero_observation = {"pressure": None, "available_bytes": 0, "swap_used_bytes": 0}
    zero_details = engine._controller_memory_watchdog_details(SimpleNamespace(
        failure_observation=zero_observation, minimum_available_bytes=0, maximum_swap_growth_bytes=0))
    assert zero_details == {**empty, "failure_observation": zero_observation,
        "minimum_available_bytes": 0, "maximum_swap_growth_bytes": 0}
    assert zero_details["failure_observation"] is not zero_observation


def test_controller_watchdog_snapshot_failure_preserves_abort(factory, supervision, tmp_path, capsys):
    supervision.mode = "abort"
    supervision.snapshot_fault = True
    supervision.diagnostics = {"error_class": "BenchmarkMemoryError", "error_message": "primary abort",
        "minimum_available_bytes": 0, "maximum_swap_growth_bytes": 0}
    output = tmp_path / "snapshot-failure.json"
    main(arguments(factory, "--prompt", "private fixture", "--output", output))
    document = json.loads(capsys.readouterr().out)["data"]["report"]
    assert document == json.loads(output.read_text()) == BenchmarkRun.load(output).to_dict()
    assert document["complete"] is True and len(document["results"]) == 4
    expected = {"error_class": None, "error_message": None, "failure_observation": None,
        "minimum_available_bytes": None, "maximum_swap_growth_bytes": None}
    for row in document["results"]:
        assert row["status"] == "blocked" and row["execution"]["failure_code"] == "host_pressure"
        assert row["execution"]["details"] == {"controller_memory_watchdog": expected}
    assert supervision.events == ["tracker-start", "guard-start", signal.SIGTERM,
        "guard-close", signal.SIGKILL, "wait", "tracker-join", "finalize"] * 4
    assert len(supervision.finalized) == 4 and all(payload is None for payload, _ in supervision.finalized)
