"""Timeout policy/propagation with native capability and HTTP doubles; no models or workers."""
from contextlib import nullcontext
import json
from types import SimpleNamespace as NS

import httpx
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, _native
from pllm.profiles import FuseFssSourceCpu, OpenWeightVerifiableCpu
from pllm.protocols import OpenWeightVerifiableOutsourcing
from pllm.runtime import benchmark_cli as benchmark
from pllm.runtime.fss_transport import timeout_limit


PROFILE = "fusefss.dense-rope-checked-ranges-ring50-q12.v1"


def declaration(tmp_path, capacity="decoder", name="timeout"):
    return Experiment(name, FuseFssSourceCpu(Model.path(str(tmp_path / "source"), model_id="unit"),
        numeric_profile=PROFILE, capacity=capacity), Deployment.local(root=str(tmp_path / "roles")),
        ExecutionBudget(requests=2, max_input_tokens=1, max_new_tokens=1))


@pytest.fixture
def native_capability(monkeypatch):
    calls = []
    def resolve(raw):
        value = json.loads(raw)
        calls.append(value)
        return NS(canonical_composition=json.dumps(value["pipeline"]).encode(),
                  composition_digest="a" * 64, configuration_digest="b" * 64)
    monkeypatch.setattr(_native, "resolve_experiment", resolve)
    from pllm.runtime import fss_source
    monkeypatch.setattr(fss_source, "resolve_client_source", lambda *a, **k: pytest.fail("source inspection"))
    return calls


@pytest.mark.parametrize("timeout", [3601, 7200, timeout_limit("decoder")])
def test_extended_timeout_requires_native_exact_decoder_admission(tmp_path, native_capability, timeout):
    experiment = declaration(tmp_path)
    benchmark.validate_benchmark_timeout(timeout, experiment=experiment)
    assert native_capability == [json.loads(experiment.canonical_bytes())]


@pytest.mark.parametrize("kind", ["none", "capsule", "unknown", "gkm-decoder"])
def test_capacity_label_or_other_decoder_cannot_expand_legacy_ceiling(tmp_path, native_capability, kind):
    experiment = declaration(tmp_path)
    if kind == "none": experiment = None
    elif kind == "capsule": experiment = declaration(tmp_path, "capsule")
    elif kind == "unknown":
        record = experiment.to_spec()
        del record["pipeline"]["components"]["state"]
        experiment = Experiment.from_spec(record)
    else:
        experiment = experiment.with_params(pipeline=OpenWeightVerifiableCpu(experiment.pipeline.model,
            linear=OpenWeightVerifiableOutsourcing(capacity="decoder")))
    benchmark.validate_benchmark_timeout(3600, experiment=experiment)
    with pytest.raises(ValueError, match="3600"):
        benchmark.validate_benchmark_timeout(3601, experiment=experiment)
    assert not native_capability


@pytest.mark.parametrize("timeout", [True, False, None, "7200", 0, -1, .5,
    float("nan"), float("inf"), float("-inf"), 86401, 1 << 4096])
def test_invalid_decoder_timeout_rejects_before_native_resolution(tmp_path, native_capability, timeout):
    with pytest.raises(ValueError, match="timeout"):
        benchmark.validate_benchmark_timeout(timeout, experiment=declaration(tmp_path))
    assert not native_capability


def test_unavailable_native_capability_cannot_use_python_profile_alone(tmp_path, monkeypatch):
    def unavailable(_): raise ValueError("native decoder unavailable")
    monkeypatch.setattr(_native, "resolve_experiment", unavailable)
    with pytest.raises(ValueError, match="native decoder unavailable"):
        benchmark.validate_benchmark_timeout(7200, experiment=declaration(tmp_path))


def test_programmatic_entry_validates_before_runner_and_preserves_value(tmp_path, native_capability, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "_run_loopback_benchmark", lambda **kw: calls.append(kw) or {})
    options = dict(model="unused", model_id=None, tiny=False, prompt="hello",
                   max_output_tokens=1, warmups=0, repetitions=1)
    with pytest.raises(ValueError, match="3600"):
        benchmark.run_loopback_benchmark(**options, timeout_seconds=7200)
    assert not calls
    benchmark.run_loopback_benchmark(**options, timeout_seconds=7200, experiment=declaration(tmp_path))
    assert calls[0]["timeout_seconds"] == 7200


def cli_options(timeout):
    return ["benchmark", "run", "--timeout", str(timeout), "--max-output-tokens", "1",
            "--warmups", "0", "--repetitions", "1", "--backend", "native", "--format", "json"]


@pytest.mark.parametrize("factory", [False, True])
@pytest.mark.parametrize("preflight", [False, True])
def test_cli_decoder_extended_timeout_after_factory_before_preflight(tmp_path, native_capability,
                                                                    monkeypatch, capsys, factory, preflight):
    from pllm.cli import main
    from pllm import metrics
    experiment, calls = declaration(tmp_path), []
    marker = tmp_path / "factory-called"
    if factory:
        target = tmp_path / "target.py"
        target.write_text("from pathlib import Path\nfrom pllm import Experiment\n"
            "def make():\n"
            f"    Path({str(marker)!r}).write_text('called')\n"
            f"    return Experiment.from_spec({experiment.to_spec()!r})\n")
        target_args = ["--experiment", f"{target}:make", "--factory", "--trust-python"]
    else:
        target = tmp_path / "experiment.json"
        target.write_bytes(experiment.canonical_bytes())
        target_args = ["--experiment", str(target)]
    def run(**kwargs):
        calls.append(kwargs)
        assert kwargs["timeout_seconds"] == 7200
        return {"checks": {"passed": True}, "configuration": {}}
    def inspect(value, **kwargs):
        assert value == experiment
        calls.append(kwargs)
        return {"admitted": True, "selected_backend": "native", "candidates": {}}
    monkeypatch.setattr(benchmark, "run_loopback_benchmark", run)
    monkeypatch.setattr(metrics, "benchmark_memory", inspect)
    main([*cli_options(7200), *target_args, *(["--preflight-only"] if preflight else [])])
    assert len(calls) == 1 and bool(native_capability)
    assert marker.exists() is factory
    output = json.loads(capsys.readouterr().out)
    assert output["data"].get("preflight_only", False) is preflight


@pytest.mark.parametrize("timeout", [0, -1, "nan", "inf", 86401])
def test_invalid_cli_timeout_does_not_execute_factory(tmp_path, monkeypatch, capsys, timeout):
    from pllm.cli import main
    from pllm._cli import targets
    monkeypatch.setattr(targets, "resolve_target", lambda *a, **k: pytest.fail("invalid timeout executed factory"))
    with pytest.raises(SystemExit):
        main([*cli_options(timeout), "--experiment", f"{tmp_path / 'unused.py'}:make", "--factory", "--trust-python"])
    assert "BENCHMARK_TIMEOUT" in capsys.readouterr().err


@pytest.mark.parametrize("timeout", [3600, 3601])
def test_ordinary_cli_ceiling_unchanged(monkeypatch, capsys, timeout):
    from pllm.cli import main
    calls = []
    monkeypatch.setattr(benchmark, "run_loopback_benchmark", lambda **kw:
                        calls.append(kw) or {"checks": {"passed": True}, "configuration": {}})
    if timeout == 3600:
        main([*cli_options(timeout), "--tiny"])
        assert calls[0]["timeout_seconds"] == 3600
    else:
        with pytest.raises(SystemExit): main([*cli_options(timeout), "--tiny"])
        assert not calls and "BENCHMARK_TIMEOUT" in capsys.readouterr().err


def test_mixed_cohort_cannot_borrow_decoder_window(tmp_path, native_capability, monkeypatch, capsys):
    from pllm.cli import main
    from pllm import metrics
    args = cli_options(7200)
    for capacity in ("decoder", "capsule"):
        path = tmp_path / f"{capacity}.json"
        path.write_bytes(declaration(tmp_path, capacity, capacity).canonical_bytes())
        args += ["--experiment", str(path)]
    monkeypatch.setattr(metrics, "benchmark_memory", lambda *a, **k: pytest.fail("unqualified cohort preflight"))
    with pytest.raises(SystemExit): main([*args, "--preflight-only"])
    assert "BENCHMARK_TIMEOUT" in capsys.readouterr().err


def test_startup_has_one_deadline_and_dashboard_receives_complete_request_window(tmp_path, native_capability, monkeypatch):
    from pllm.runtime import benchmark_memory, benchmark_resources, dashboard
    clock, seen = [10.0], {}
    class StopBeforeExecution(Exception): pass
    monkeypatch.setattr(benchmark, "time", NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(benchmark, "_DASHBOARD_PORT", 7300)
    monkeypatch.setattr(benchmark_memory, "process_memory", lambda: {})
    monkeypatch.setattr(benchmark_resources, "ProcessMemorySampler", lambda _:
                        NS(start=lambda: None, stop=lambda: seen.update(sampler_closed=True)))
    def app(config):
        seen["config"] = config
        return NS(state=NS(dashboard_runtime=NS()))
    monkeypatch.setattr(dashboard, "create_dashboard_app", app)
    def start(): clock[0] += 200
    monkeypatch.setattr(benchmark, "_DashboardHandle", lambda *a:
                        NS(start=start, close=lambda: seen.update(closed=True)))
    monkeypatch.setattr(benchmark.httpx, "Client", lambda **kw: nullcontext(object()))
    def listener(handle, origin, deadline):
        seen["listener_deadline"] = deadline
        clock[0] += 400
    def ready(client, handle, deadline, progress):
        seen["ready_deadline"] = deadline
        raise StopBeforeExecution
    monkeypatch.setattr(benchmark, "_wait_for_dashboard_listener", listener)
    monkeypatch.setattr(benchmark, "_wait_for_ready", ready)
    with pytest.raises(StopBeforeExecution):
        benchmark.run_loopback_benchmark(model="unused", model_id=None, tiny=False, prompt="hello",
            max_output_tokens=1, warmups=0, repetitions=1, timeout_seconds=7200,
            experiment=declaration(tmp_path), backend="native")
    assert seen["config"].request_timeout_seconds == 7200
    assert seen["listener_deadline"] == seen["ready_deadline"] == 7210
    assert seen["closed"] and seen["sampler_closed"] and benchmark._DASHBOARD_PORT is None


@pytest.mark.parametrize("slow_phase", ["post", "snapshot", "record"])
def test_per_run_deadline_includes_post_poll_and_record_without_refresh(monkeypatch, slow_phase):
    clock, paths = [0.0], []
    monkeypatch.setattr(benchmark, "time", NS(monotonic=lambda: clock[0], sleep=lambda _: None))
    monkeypatch.setattr(benchmark.secrets, "token_hex", lambda _: "fixed")
    def respond(request):
        paths.append(request.url.path)
        phase = ("post" if request.url.path == "/api/run" else
                 "snapshot" if request.url.path == "/api/snapshot" else "record")
        if phase == slow_phase: clock[0] += 7201
        data = ({"run": {"phase": "ready", "run_id": "bench-fixed", "active_run": None}}
                if phase == "snapshot" else {"status": "completed"})
        return httpx.Response(200, json=data)
    with httpx.Client(transport=httpx.MockTransport(respond), base_url="http://unit") as client:
        with pytest.raises(benchmark.LoopbackBenchmarkError, match="before timeout"):
            benchmark._run_once(client, NS(poll=lambda: None), prompt="hello", max_output_tokens=1,
                                timeout_seconds=7200)
    assert len(paths) == {"post": 1, "snapshot": 2, "record": 3}[slow_phase]
