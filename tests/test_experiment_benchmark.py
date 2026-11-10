"""Facade contracts with mocked execution; no model or benchmark workers."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def facade(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from examples.papers.gkm import bounded_response
    from pllm import evidence
    from pllm.configuration import Experiment
    from pllm.metrics import benchmark_runner

    resolve = Mock()
    run = Mock(return_value=object.__new__(benchmark_runner.BenchmarkRun))
    monkeypatch.setattr(Experiment, "resolve", resolve)
    monkeypatch.setattr(benchmark_runner, "benchmark_target", run)
    return evidence, bounded_response(), resolve, run


@pytest.mark.parametrize("keyword_plan", [False, True])
def test_experiment_forwards_existing_defaults_and_same_run(facade, keyword_plan):
    from pllm._benchmark_contract import BenchmarkTarget, BuildRef, Sampling, Scenario
    from pllm.metrics.benchmark_runner import BenchmarkRun

    evidence, experiment, resolve, runner = facade
    scenario = Scenario.cold(max_output_tokens=2)
    inputs = MappingProxyType({"prompt": "Hello"})
    kwargs = {"scenario": scenario, "inputs": inputs}
    result = (evidence.benchmark(plan=experiment, **kwargs) if keyword_plan
              else evidence.benchmark(experiment, **kwargs))
    resolve.assert_called_once_with()
    target = BenchmarkTarget.response(experiment, scenario=scenario)
    runner.assert_called_once_with(target, sampling=Sampling(), build=BuildRef(),
                                   budgets=(), inputs=inputs, output=None)
    assert runner.call_args.kwargs["inputs"] is inputs
    assert isinstance(result, BenchmarkRun)
    assert result is runner.return_value
    assert runner.call_args.kwargs["sampling"].to_spec() == {
        "processes": 3, "repetitions": 1, "warmups": 0, "timeout_seconds": 300,
        "order": "balanced", "threads": 1, "memory_bytes": 512 << 20,
    }
    assert runner.call_args.kwargs["build"].expected_digest is None


def test_explicit_options_forward_unchanged_after_resolution(facade):
    from pllm import benchmark
    from pllm._benchmark_contract import BenchmarkTarget, BuildRef, Sampling, Scenario, TargetBudget

    _, experiment, resolve, runner = facade
    scenario = Scenario.cold(max_output_tokens=2)
    sampling = Sampling(processes=1)
    build = BuildRef(python="/approved/python", expected_digest="a" * 64)
    budgets = (TargetBudget("request.0.full", maximum=10),)
    inputs = {"prompt": "Hello"}
    output = Path("target/experiment.json")
    events = []
    resolve.side_effect = lambda: events.append("resolve")
    runner.side_effect = lambda *args, **kwargs: events.append("runner") or runner.return_value
    assert benchmark(experiment, scenario=scenario, inputs=inputs,
        sampling=sampling, build=build, budgets=budgets, output=output) is runner.return_value
    assert events == ["resolve", "runner"]
    runner.assert_called_once_with(BenchmarkTarget.response(experiment, scenario=scenario),
        sampling=sampling, build=build, budgets=budgets, inputs=inputs, output=output)
    for name, value in (("sampling", sampling), ("build", build), ("budgets", budgets),
                        ("inputs", inputs), ("output", output)):
        assert runner.call_args.kwargs[name] is value


@pytest.mark.parametrize("inputs", [None, False, 0, "", [], (), [("prompt", "Hello")],
                                    (("prompt", "Hello"),)])
def test_non_mapping_inputs_rejected_before_work(facade, inputs):
    from pllm._benchmark_contract import Scenario

    evidence, experiment, resolve, runner = facade
    assert not isinstance(inputs, Mapping)
    with pytest.raises(TypeError, match="inputs must be a mapping"):
        evidence.benchmark(experiment, scenario=Scenario.cold(max_output_tokens=2), inputs=inputs)
    resolve.assert_not_called()
    runner.assert_not_called()


def test_empty_mapping_is_valid_and_not_replaced(facade):
    from pllm._benchmark_contract import Scenario

    evidence, experiment, _, runner = facade
    inputs = {}
    evidence.benchmark(experiment, scenario=Scenario.cold(max_output_tokens=2), inputs=inputs)
    assert runner.call_args.kwargs["inputs"] is inputs


@pytest.mark.parametrize("kwargs", [
    {"inputs": {}}, {"scenario": "wrong"},
    {"scenario": "wrong", "inputs": {}, "weights": b""},
    {"scenario": "wrong", "inputs": {}, "backend": "native"},
    {"scenario": "wrong", "inputs": {}, "warmups": 0},
])
def test_missing_or_mixed_experiment_keywords_rejected_before_work(facade, kwargs):
    evidence, experiment, resolve, runner = facade
    with pytest.raises(TypeError):
        evidence.benchmark(plan=experiment, **kwargs)
    resolve.assert_not_called()
    runner.assert_not_called()


@pytest.mark.parametrize("scenario", [None, "wrong"])
def test_invalid_scenario_rejected_before_work(facade, scenario):
    evidence, experiment, resolve, runner = facade
    with pytest.raises(TypeError, match="scenario must be a Scenario"):
        evidence.benchmark(experiment, scenario=scenario, inputs={})
    resolve.assert_not_called()
    runner.assert_not_called()


@pytest.mark.parametrize("keyword_plan", [False, True])
def test_legacy_body_returns_report_and_preserves_defaults(monkeypatch, keyword_plan):
    import pllm
    from pllm import evidence
    from pllm.plan import CompiledPlan

    # Bypass native handle construction; exercise unchanged helper body only.
    plan = object.__new__(CompiledPlan)
    native = Mock()
    native.benchmark_wrap32.return_value = b'{"schema_version":"legacy.fixture"}'
    object.__setattr__(plan, "_native", native)
    monkeypatch.setitem(pllm.__dict__, "_native", ModuleType("pllm._native"))
    kwargs = dict(weights=b"weights", input=b"input", id="fixture",
        privacy_cohort="privacy", numeric_cohort="numeric", environment={"host": "fixture"})
    report = (evidence.benchmark(plan=plan, **kwargs) if keyword_plan
              else evidence.benchmark(plan, **kwargs))
    assert isinstance(report, evidence.EvidenceReport)
    assert report.canonical_bytes() == native.benchmark_wrap32.return_value
    native.benchmark_wrap32.assert_called_once_with(b"weights", b"input", "fixture",
        "privacy", "numeric", b'{"host":"fixture"}', 3, 10, 1, True)


@pytest.mark.parametrize("extra", [{"scenario": "wrong"}, {"inputs": {}}, {"sampling": None}])
def test_mixed_legacy_keywords_rejected_before_helper_body(monkeypatch, extra):
    import pllm
    from pllm import evidence
    from pllm.plan import CompiledPlan

    plan = object.__new__(CompiledPlan)
    native = Mock()
    object.__setattr__(plan, "_native", native)
    monkeypatch.setitem(pllm.__dict__, "_native", ModuleType("pllm._native"))
    with pytest.raises(TypeError):
        evidence.benchmark(plan=plan, weights=b"", input=b"", id="fixture",
            privacy_cohort="privacy", numeric_cohort="numeric", environment={}, **extra)
    native.benchmark_wrap32.assert_not_called()
