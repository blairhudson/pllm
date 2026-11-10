"""Mocked facade contracts plus real-native declaration rejection cases.

No model execution or benchmark workers; native negatives check unsupported
composition admission only, not complete coverage or security properties.
"""

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


def _partial_maverick_experiment(profile=None):
    from pllm.configuration import Deployment, ExecutionBudget, Experiment, Model, Pipeline
    from pllm.kernels import Cpu
    from pllm.protocols import MaverickDelegatedLinear

    return Experiment(
        name="partial-maverick-admission",
        pipeline=Pipeline(
            model=Model.tiny(),
            components={
                "linear": MaverickDelegatedLinear(public_code_seed="0d" * 32),
                "kernels": Cpu(threads=1),
            },
            profile=profile,
        ),
        deployment=Deployment.local(root="local://maverick-admission"),
        budget=ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=1),
    )


def test_partial_maverick_immutable_canonical_roundtrip_and_registered_recreation():
    from dataclasses import FrozenInstanceError

    from pllm.configuration import ComponentRef, Experiment, Pipeline
    from pllm.protocols import MaverickDelegatedLinear

    experiment = _partial_maverick_experiment()
    canonical = experiment.canonical_bytes()
    linear = experiment.pipeline.components["linear"]
    assert type(linear) is MaverickDelegatedLinear
    assert linear.params == {"public_code_seed": "0d" * 32}
    assert MaverickDelegatedLinear() == linear
    assert "profile" not in experiment.pipeline.to_spec()

    restored = Experiment.from_json(canonical.decode("utf-8"))
    assert restored == experiment
    assert restored is not experiment
    assert restored.canonical_bytes() == canonical
    assert restored.configuration_digest() == experiment.configuration_digest()
    assert restored.pipeline.profile is None
    assert type(restored.pipeline.components["linear"]) is MaverickDelegatedLinear

    reference = ComponentRef(linear.component, linear.params)
    assert type(reference) is ComponentRef
    recreated = Pipeline(model=experiment.pipeline.model, components={
        "linear": reference,
        "kernels": experiment.pipeline.components["kernels"],
    })
    assert type(recreated.components["linear"]) is MaverickDelegatedLinear
    assert recreated == experiment.pipeline
    assert recreated.canonical_bytes() == experiment.pipeline.canonical_bytes()

    with pytest.raises(FrozenInstanceError):
        restored.name = "mutated"
    with pytest.raises(TypeError):
        restored.pipeline.components["linear"] = reference
    with pytest.raises(TypeError):
        restored.pipeline.components["linear"].params["public_code_seed"] = "00" * 32
    detached = restored.to_spec()
    detached["pipeline"]["components"]["linear"]["params"]["public_code_seed"] = "00" * 32
    assert restored.canonical_bytes() == canonical
    assert experiment.canonical_bytes() == canonical


@pytest.mark.parametrize("profile", [
    None, "research.maverick", "research.open_weight_verifiable_cpu",
    "runtime.proprietary_guarded",
])
@pytest.mark.parametrize("roundtrip", [False, True], ids=["original", "canonical-json"])
def test_partial_maverick_native_rejection_precedes_benchmark_target(monkeypatch, profile, roundtrip):
    from pllm import _native, benchmark
    from pllm._benchmark_contract import BenchmarkTarget, Scenario
    from pllm.configuration import ConfigurationError, Experiment
    from pllm.metrics import benchmark_runner

    experiment = _partial_maverick_experiment(profile)
    canonical = experiment.canonical_bytes()
    assert experiment.pipeline.profile == profile
    assert canonical == _partial_maverick_experiment().canonical_bytes()
    if roundtrip:
        experiment = Experiment.from_json(canonical.decode("utf-8"))
        assert experiment.pipeline.profile is None
    assert experiment.canonical_bytes() == canonical

    response = Mock(side_effect=AssertionError("BenchmarkTarget.response must not be called"))
    runner = Mock(side_effect=AssertionError("benchmark_target must not be called"))
    monkeypatch.setattr(BenchmarkTarget, "response", response)
    monkeypatch.setattr(benchmark_runner, "benchmark_target", runner)

    # Real native declaration admission: unsupported linear identity reaches the
    # masked-linear fallback's first slot, not a complete-coverage/security audit.
    # Missing or stale native support must fail, never skip or use a substitute.
    message = "masked-linear composition requires linear component pllm/masked-linear with no parameters"
    with pytest.raises(ValueError) as native_error:
        _native.resolve_experiment(experiment.canonical_bytes())
    assert type(native_error.value) is ValueError
    assert str(native_error.value) == message

    with pytest.raises(ConfigurationError) as public_error:
        benchmark(experiment, scenario=Scenario.cold(max_output_tokens=1),
                  inputs={"prompt": "Hello"})
    assert str(public_error.value) == message
    assert type(public_error.value.__cause__) is ValueError
    assert str(public_error.value.__cause__) == message
    response.assert_not_called()
    runner.assert_not_called()


@pytest.fixture(params=["forward", "response"])
def target_facade(facade, monkeypatch, request):
    from pllm import Deployment, Experiment, ForwardBudget, Model, compiler, model_loader, sources
    from pllm._benchmark_contract import BenchmarkTarget, Scenario
    from pllm.kernels import Cpu
    from pllm.metrics import benchmark_runner
    from pllm.numeric import SourceFloat32
    from pllm.profiles import ClientOnlyCpu

    _, response, resolve, runner = facade
    guards = []
    for owner, name in (
        (Experiment, "compile_forward"), (compiler, "compile"),
        (model_loader, "resolve_model"), (model_loader, "load_model"),
        (sources, "_materialize_tiny_model"), (benchmark_runner, "inspect_build"),
    ):
        guard = Mock(side_effect=AssertionError(f"unexpected {name}"))
        monkeypatch.setattr(owner, name, guard)
        guards.append(guard)

    # Same inert factory pattern as test_benchmark_forward; no checkpoint files.
    if request.param == "forward":
        experiment = Experiment(
            "facade-forward",
            ClientOnlyCpu(Model.path("target/facade-checkpoint"),
                          kernels=Cpu(threads=2), numeric=SourceFloat32()),
            Deployment.local(root="target/facade-runtime"), ForwardBudget(1, 6),
        )
        target = BenchmarkTarget.forward(experiment)
    else:
        target = BenchmarkTarget.response(response, scenario=Scenario.cold(max_output_tokens=2))
    # Construction above is real; facade must not reconstruct the frozen target.
    for name in ("forward", "response", "from_spec"):
        guard = Mock(side_effect=AssertionError(f"unexpected target {name}"))
        monkeypatch.setattr(BenchmarkTarget, name, guard)
        guards.append(guard)
    payload, digest = target._bytes, target.digest
    yield target, runner
    assert target._bytes is payload
    assert target.digest == digest
    resolve.assert_not_called()
    for guard in guards:
        guard.assert_not_called()


@pytest.mark.parametrize("keyword_plan", [False, True])
@pytest.mark.parametrize("inputs_kwargs", [
    {}, {"inputs": None}, {"inputs": {}},
    {"inputs": MappingProxyType({"prompt": "Hello"})},
    {"inputs": {"input_ids": [[1, 2]]}},
], ids=["omitted", "none", "empty", "readonly", "populated"])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_target_public_defaults_and_input_identity(target_facade, keyword_plan,
                                                  inputs_kwargs, explicit_none):
    from pllm import benchmark
    from pllm._benchmark_contract import BuildRef, Sampling
    from pllm.metrics.benchmark_runner import BenchmarkRun

    target, runner = target_facade
    kwargs = dict(inputs_kwargs)
    if explicit_none:
        kwargs.update(sampling=None, build=None)
    result = (benchmark(plan=target, **kwargs) if keyword_plan
              else benchmark(target, **kwargs))
    runner.assert_called_once_with(target, sampling=Sampling(), build=BuildRef(),
                                   budgets=(), inputs=inputs_kwargs.get("inputs"), output=None)
    assert runner.call_args.args[0] is target
    assert runner.call_args.kwargs["inputs"] is inputs_kwargs.get("inputs")
    assert isinstance(result, BenchmarkRun)
    assert result is runner.return_value
    assert runner.call_args.kwargs["sampling"].to_spec() == {
        "processes": 3, "repetitions": 1, "warmups": 0, "timeout_seconds": 300,
        "order": "balanced", "threads": 1, "memory_bytes": 512 << 20,
    }
    assert runner.call_args.kwargs["build"].expected_digest is None


@pytest.mark.parametrize("keyword_plan", [False, True])
@pytest.mark.parametrize("output", ["target/target-run.json", Path("target/target-run.json")])
def test_target_public_explicit_options_preserve_identity(target_facade, keyword_plan, output):
    from pllm import benchmark
    from pllm._benchmark_contract import BuildRef, Sampling, TargetBudget

    target, runner = target_facade
    sampling = Sampling(processes=1, repetitions=2, threads=2)
    build = BuildRef(python="/approved/python", expected_digest="a" * 64)
    budgets = (TargetBudget("request.0.full", maximum=10),)
    inputs = MappingProxyType({"input_ids": [[1, 2]]})
    kwargs = dict(inputs=inputs, sampling=sampling, build=build, budgets=budgets, output=output)
    result = (benchmark(plan=target, **kwargs) if keyword_plan
              else benchmark(target, **kwargs))
    assert result is runner.return_value
    runner.assert_called_once_with(target, **kwargs)
    assert runner.call_args.args[0] is target
    for name, value in kwargs.items():
        assert runner.call_args.kwargs[name] is value


@pytest.mark.parametrize("inputs", [False, 0, "", [], (), [("prompt", "Hello")],
                                    (("prompt", "Hello"),)])
def test_target_public_invalid_inputs_precede_runner(target_facade, inputs):
    from pllm import benchmark

    target, runner = target_facade
    assert not isinstance(inputs, Mapping)
    with pytest.raises(TypeError, match="^inputs must be a mapping$"):
        benchmark(target, inputs=inputs)
    runner.assert_not_called()


@pytest.mark.parametrize("extra", [
    "scenario_same", "scenario_none", "backend", "wan", "config", "weights", "input", "id",
    "privacy_cohort", "numeric_cohort", "environment", "warmups", "repetitions", "threads",
    "simd", "unknown",
])
@pytest.mark.parametrize("inputs", [{}, False], ids=["valid-inputs", "bad-inputs"])
def test_target_public_extra_binding_precedes_input_validation(target_facade, extra, inputs):
    from pllm import benchmark
    from pllm._benchmark_contract import Scenario

    target, runner = target_facade
    name, value = extra, None
    if extra == "scenario_same":
        name = "scenario"
        value = (Scenario(**target.to_spec()["parameters"]["scenario"])
                 if target.kind == "response" else Scenario.cold(max_output_tokens=2))
    elif extra == "scenario_none":
        name = "scenario"
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
        benchmark(plan=target, inputs=inputs, **{name: value})
    runner.assert_not_called()


@pytest.mark.parametrize("mapping", [False, True], ids=["duck", "mapping"])
def test_target_lookalikes_keep_legacy_type_error(facade, monkeypatch, mapping):
    from types import SimpleNamespace

    import pllm
    from pllm import benchmark

    _, _, resolve, runner = facade
    to_spec = Mock()
    target = ({"kind": "forward", "digest": "a" * 64, "to_spec": to_spec} if mapping
              else SimpleNamespace(kind="forward", digest="a" * 64, to_spec=to_spec))
    monkeypatch.setitem(pllm.__dict__, "_native", ModuleType("pllm._native"))
    with pytest.raises(TypeError, match="^plan must be returned by pllm.compile$"):
        benchmark(target, weights=b"", input=b"", id="fixture", privacy_cohort="privacy",
                  numeric_cohort="numeric", environment={})
    to_spec.assert_not_called()
    resolve.assert_not_called()
    runner.assert_not_called()
