"""Lightweight ordinary response acceptance tests; no serving/native imports.

Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and pytest --noconftest to avoid the
repository's unrelated model/server fixtures.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm._benchmark_contract import BenchmarkSuite, BenchmarkTarget, Scenario, TargetUnavailable
from pllm.profiles import MaskedLinearCpu


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def executor():
    # Loading this stdlib-only executor by path avoids pllm.runtime.__init__,
    # which would otherwise import unrelated serving/native dependencies.
    spec = importlib.util.spec_from_file_location(
        "pllm.runtime.benchmark_response_target", ROOT / "python/pllm/runtime/benchmark_response_target.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def experiment():
    return Experiment("expected-counts", MaskedLinearCpu(Model.hf(
        "Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c",
        local_files_only=True)), Deployment.local(root=".pllm/expected-counts"),
        ExecutionBudget(requests=3, max_input_tokens=64, max_new_tokens=8))


def target(**options):
    return BenchmarkTarget.response(experiment(), **options)


def report(*counts):
    return {"checks": {"passed": True}, "runs": [
        {"status": "completed", "tokens": {"authoritative": True,
         "input_tokens": inputs, "output_tokens": outputs},
         "generation": {"output_text_digest": "d" * 64}}
        for inputs, outputs in (counts or ((64, 8),))]}


def test_none_is_byte_identical_and_with_params_adds_removes_expectations():
    original = target()
    explicit = target(expected_input_tokens=None, expected_output_tokens=None)
    assert original.to_spec() == explicit.to_spec()
    assert original.digest == explicit.digest
    assert not any(key.startswith("expected_") for key in original.to_spec()["parameters"])
    selected = original.with_params(expected_input_tokens=64, expected_output_tokens=8)
    assert selected.digest != original.digest
    assert BenchmarkTarget.from_spec(selected.to_spec()).digest == selected.digest
    assert selected.with_params(expected_input_tokens=None, expected_output_tokens=None).digest == original.digest


@pytest.mark.parametrize("field,value", [
    ("expected_input_tokens", True), ("expected_input_tokens", -1),
    ("expected_input_tokens", 65), ("expected_input_tokens", 64.0),
    ("expected_output_tokens", False), ("expected_output_tokens", 9),
    ("expected_output_tokens", "8"),
])
def test_invalid_expectations_are_not_coerced_or_allowed_above_caps(field, value):
    with pytest.raises(ValueError, match=field):
        target(**{field: value})


def test_output_expectation_cannot_exceed_selected_scenario_cap():
    with pytest.raises(ValueError, match="expected_output_tokens"):
        target(scenario=Scenario.cold(max_output_tokens=2), expected_output_tokens=8)
    with pytest.raises(ValueError, match="serial"):
        target(concurrent_requests=2, expected_input_tokens=64)


def test_unrequested_check_does_not_touch_legacy_report(executor):
    value = report((63, 1))
    before = deepcopy(value)
    assert executor._expected_token_count_checks({}, value) is True
    assert value == before


@pytest.mark.parametrize("counts", [((63, 8),), ((64, 7),), ((64, 0),), ((64, 8), (64, 7))])
def test_count_mismatch_and_early_eos_fail_without_rewriting_observations(executor, counts):
    value = report(*counts)
    runs = deepcopy(value["runs"])
    assert not executor._expected_token_count_checks(
        {"expected_input_tokens": 64, "expected_output_tokens": 8}, value)
    assert value["checks"]["passed"] is False
    assert value["runs"] == runs


@pytest.mark.parametrize("change", [
    lambda row: row.pop("tokens"),
    lambda row: row.update(tokens=None),
    lambda row: row["tokens"].update(authoritative=False),
    lambda row: row["tokens"].update(input_tokens=None),
    lambda row: row["tokens"].update(input_tokens="64"),
    lambda row: row["tokens"].update(output_tokens=8.0),
    lambda row: row["tokens"].update(output_tokens=True),
    lambda row: row.update(status="failed"),
])
def test_unknown_failed_or_nonauthoritative_telemetry_is_not_success(executor, change):
    value = report()
    change(value["runs"][0])
    assert not executor._expected_token_count_checks(
        {"expected_input_tokens": 64, "expected_output_tokens": 8}, value)


def test_zero_requires_explicit_authoritative_completed_observation(executor):
    parameters = target(expected_output_tokens=0).to_spec()["parameters"]
    assert executor._expected_token_count_checks(parameters, report((64, 0)))
    value = report((64, None))
    assert not executor._expected_token_count_checks(parameters, value)
    assert not executor._expected_token_count_checks(parameters, {"runs": [], "checks": {"passed": True}})


def test_both_expectations_check_every_request_in_sequence(executor):
    parameters = target(scenario=Scenario.sequence(count=2), expected_input_tokens=64,
                        expected_output_tokens=8).to_spec()["parameters"]
    value = report((64, 8), (64, 8))
    assert executor._expected_token_count_checks(parameters, value)
    assert value["checks"] == {"passed": True, "expected_input_tokens": True, "expected_output_tokens": True}


@pytest.fixture
def mocked_response(executor, monkeypatch, tmp_path):
    """Exercise the ordinary executor with mock transport/admission only."""
    def install(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    # No parent runtime import, checkpoint resolution, issuance or role launch.
    if "pllm.runtime" not in sys.modules:
        module = ModuleType("pllm.runtime")
        module.__path__ = []
        monkeypatch.setitem(sys.modules, "pllm.runtime", module)
    value = report()
    value.update(schema_version="pllm.loopback_benchmark.v1", configuration={
        "source_lock_digest": "s" * 64, "provider_backend": "native", "sampling": {"temperature": 0}},
        topology_accounting={})
    value["runs"][0]["model_fingerprint"] = "b" * 64
    records = [{"failed": False, "selections": [{"checked_values": 2}] * 8}]
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return value

    install("pllm.model_loader", expected_model_id=lambda _: "model")
    install("pllm.runtime.benchmark_cli", run_loopback_benchmark=transport)
    install("pllm.runtime.benchmark_context", validate_context=lambda prompt: prompt)
    install("pllm.runtime.benchmark_delegated", admit_options=lambda *a, **kw: None,
            is_buffered=lambda _: False, is_replicated=lambda _: False, is_fss=lambda _: False)
    install("pllm.runtime.benchmark_response_audit", GKM_AUDIT_SCHEMA="unused",
            state_audit_admission=lambda report, *a, **kw: report,
            response_state_audit=lambda *a, **kw: nullcontext(records))
    install("pllm._search_benchmark", _boundary=lambda _: {"numeric": {"bits": 8}, "privacy": {}})
    install("pllm.runtime.topology_accounting", ACCOUNTING_SCHEMA="accounting",
            accounting_coverage_identity=lambda _: "accounting")
    import pllm.metrics
    monkeypatch.setattr(pllm.metrics, "benchmark_memory", lambda *a, **kw: {"admitted": True})
    monkeypatch.setattr(executor, "response_metrics", lambda *a: None)
    m = SimpleNamespace(
        sampling=SimpleNamespace(repetitions=1, warmups=0, memory_bytes=12 << 30, timeout_seconds=60),
        inputs={"prompt": "private text"}, directory=tmp_path, salt=b"x" * 32,
        details={}, comparison={}, limitations=[], admit=lambda _: None,
        run=lambda action: [action(0)])
    return m, value, calls


def test_executor_failure_retains_actual_counts_digests_and_state_audit(executor, mocked_response):
    m, value, calls = mocked_response
    value["runs"][0]["tokens"]["output_tokens"] = 7
    parameters = target(state_audit=True, expected_input_tokens=64,
                        expected_output_tokens=8).to_spec()["parameters"]
    with pytest.raises(TargetUnavailable) as error:
        executor.execute_response(parameters, m)
    assert error.value.code == "response_expected_token_counts_failed"
    assert error.value.status == "failed"
    assert m.details["report"] is value
    assert value["runs"][0]["tokens"]["output_tokens"] == 7
    assert value["runs"][0]["generation"]["output_text_digest"] == "d" * 64
    assert m.details["response_state_audit"]["bindings"]
    assert m.comparison["output_digest"]
    assert m.oracle["outcome"] == "failed"
    assert value["checks"]["expected_output_tokens"] is False
    assert "expected_input_tokens" not in calls[0]  # Never changes actual generation.
    assert calls[0]["max_output_tokens"] == 8


def test_original_execution_exception_is_not_replaced_by_count_failure(executor, mocked_response):
    m, _, _ = mocked_response
    original = TargetUnavailable("injected_original_failure", status="blocked")

    def fail(_):
        raise original

    m.run = fail
    parameters = target(expected_input_tokens=64, expected_output_tokens=8).to_spec()["parameters"]
    with pytest.raises(TargetUnavailable) as error:
        executor.execute_response(parameters, m)
    assert error.value is original
    assert "report" not in m.details


@pytest.mark.parametrize("change", [
    lambda value: value["checks"].update(passed=False),
    lambda value: value["runs"][0].update(status="failed"),
    lambda value: value["runs"][0]["tokens"].update(authoritative=False),
])
def test_existing_failed_report_gate_keeps_its_original_failure(executor, mocked_response, change):
    m, value, _ = mocked_response
    change(value)
    parameters = target(expected_input_tokens=64, expected_output_tokens=8).to_spec()["parameters"]
    with pytest.raises(TargetUnavailable) as error:
        executor.execute_response(parameters, m)
    assert error.value.code == "response_checks_failed"
    assert "expected_input_tokens" not in value["checks"]


def test_opted_success_checks_bind_public_expectations_without_changing_output(executor, mocked_response):
    m, value, _ = mocked_response
    legacy = target(state_audit=True).to_spec()["parameters"]
    executor.execute_response(legacy, m)
    original = deepcopy(m.comparison)
    assert value["checks"] == {"passed": True}
    parameters = target(state_audit=True, expected_input_tokens=64,
                        expected_output_tokens=8).to_spec()["parameters"]
    executor.execute_response(parameters, m)
    assert value["checks"] == {"passed": True, "expected_input_tokens": True, "expected_output_tokens": True}
    assert m.oracle["outcome"] == "passed"
    assert m.details["response_state_audit"]["bindings"]
    assert m.comparison["workload_digest"] != original["workload_digest"]
    assert all(m.comparison[key] == original[key] for key in original if key != "workload_digest")


def test_public_suite_opts_in_idempotently_at_factory():
    from pllm.metrics import BenchmarkCase, BuildRef, Sampling

    # Generic count contract must not depend on another paper's uncommitted
    # factory. That factory's concrete 64/8 selection belongs in its own tests.
    selected_target = target(state_audit=True, expected_input_tokens=64, expected_output_tokens=8)
    original = BenchmarkSuite(tuple(BenchmarkCase(name, selected_target,
        BuildRef(expected_digest="a" * 64)) for name in ("baseline", "candidate")),
        Sampling(processes=2, warmups=0, repetitions=1, memory_bytes=12 << 30),
        name="public-count-opt-in")
    selected = replace(original, cases=tuple(replace(case, target=case.target.with_params(
        expected_input_tokens=64, expected_output_tokens=8)) for case in original.cases))
    assert selected.to_spec() == original.to_spec()
    assert selected.digest == original.digest
    assert BenchmarkSuite.from_spec(selected.to_spec()).digest == selected.digest
    assert selected.sampling == original.sampling
    for before, after in zip(original.cases, selected.cases, strict=True):
        params = after.target.to_spec()["parameters"]
        assert params["state_audit"] is True
        assert params["expected_input_tokens"] == 64
        assert params["expected_output_tokens"] == 8
        assert params == before.target.to_spec()["parameters"]
        assert after.target.digest == before.target.digest
