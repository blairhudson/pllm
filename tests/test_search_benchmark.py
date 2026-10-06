from __future__ import annotations

import gc
import hashlib
import json
import weakref

import pytest

import pllm
from pllm.kernels import Cpu
from pllm.profiles import ClientOnlyCpu
from test_search import experiment


def report_for(kwargs):
    value = kwargs["experiment"]
    count = kwargs["max_output_tokens"]
    full = 4 / value.pipeline.kernels.params.get("threads", 1)
    sampling = {"effective_temperature": 0, "mode": "greedy", "top_p": None}
    return {
        "schema_version": "pllm.loopback_benchmark.v1", "checks": {"passed": True},
        "configuration": {
            "prompt_digest": hashlib.sha256(kwargs["_cohort_salt"] + kwargs["prompt"].encode()).hexdigest(),
            "source_lock_digest": "a" * 64, "max_output_tokens": count,
            "sampling": sampling, "provider_backend": "native", "warmups": 0, "repetitions": 1,
        },
        "experiment": {"configuration_digest": value.configuration_digest(),
                       "pipeline_digest": value.pipeline.digest()},
        "runs": [{
            "model_id": value.pipeline.model.source, "model_fingerprint": "b" * 64,
            "status": "completed", "warm": False, "max_output_tokens": count,
            "tokens": {"authoritative": True, "input_tokens": 3, "output_tokens": count},
            "generation": {"output_text_digest": "c" * 64},
            "durations": {"full_seconds": full},
        }],
        "communication_per_token": {"summary": {"setup_inclusive_mb_per_output_token": 4,
                                                   "online_mb_per_output_token": 2}},
    }


@pytest.fixture
def driver(monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return report_for(kwargs)

    monkeypatch.setattr("pllm.runtime.benchmark_cli.run_loopback_benchmark", run)
    monkeypatch.setattr("pllm.metrics.benchmark_memory", lambda *args, **kwargs: {"admitted": True})
    return calls


@pytest.mark.parametrize("strategy", ["grid", "random", "beam"])
def test_shared_measured_driver_retains_factories_and_private_cohort_salt(driver, strategy):
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2, 3]})
    search = {"grid": lambda: pllm.GridSearch("trial", space),
              "random": lambda: pllm.RandomSearch("trial", space, seed=7, count=3),
              "beam": lambda: pllm.BeamSearch("trial", space, "request_tps", "max")}[strategy]()
    result = pllm.benchmark_search(search, "unique private prompt", max_output_tokens=2)
    data = result.to_dict()
    assert data["schema_version"] == "pllm.search_benchmark.v1"
    assert len(data["search"]["evaluations"]) == len(driver) == len(data["candidates"]) == 3
    assert len({value["_cohort_salt"] for value in driver}) == 1
    for call in driver:
        assert call["temperature"] == 0 and call["capture_output_digest"] is True
        assert call["warmups"] == 0 and call["repetitions"] == 1
        assert call["tiny"] is False  # The immutable Model, not the legacy --tiny flag, selects fixtures.
    winner = next(record for record in data["candidates"]
                  if record["configuration_digest"] == data["search"]["best_configuration_digest"])
    scope = {}
    exec(winner["python_source"], scope)
    assert scope["experiment"]().pipeline.components["kernels"].params["threads"] == 3
    assert scope["experiment"]().configuration_digest() == winner["configuration_digest"]
    assert "unique private prompt" not in json.dumps(data)
    assert driver[0]["_cohort_salt"].hex() not in json.dumps(data)
    pllm.benchmark_search(search, "unique private prompt", max_output_tokens=2)
    assert driver[0]["_cohort_salt"] != driver[-1]["_cohort_salt"]


def test_memory_and_topology_changes_reject_before_runtime(driver, monkeypatch):
    base = experiment()
    space = pllm.SearchSpace(base, {"pipeline": [base.pipeline,
        base.pipeline.with_params(kernels=Cpu(threads=2)), ClientOnlyCpu(base.pipeline.model)]})
    monkeypatch.setattr("pllm.metrics.benchmark_memory",
                        lambda x, **kwargs: {"admitted": x.pipeline.kernels.params["threads"] == 1})
    result = pllm.benchmark_search(pllm.BeamSearch("admit", space, "request_tps", "max"),
                                   "private", max_output_tokens=2).to_dict()
    assert len(driver) == 1
    assert len(result["search"]["rejections"]) == 2
    assert {r["reason"] for r in result["search"]["rejections"]} == {
        "host memory preflight rejected candidate", "model, numeric, topology or verification comparison boundary changed",
    }


@pytest.mark.parametrize("strategy", ["grid", "random", "beam"])
def test_combined_native_incompatibility_does_not_discard_other_trials(driver, strategy):
    from pllm.kernels import AppleMetal
    from pllm.preparation import ModelAwareCorrections

    base = experiment()
    space = pllm.SearchSpace(base, {
        "pipeline__kernels": [base.pipeline.kernels, AppleMetal()],
        "pipeline__preparation": [ModelAwareCorrections(), ModelAwareCorrections(storage="paged")],
    })
    search = {"grid": lambda: pllm.GridSearch("compatible", space),
              "random": lambda: pllm.RandomSearch("compatible", space, seed=7, count=4),
              "beam": lambda: pllm.BeamSearch("compatible", space, "request_tps", "max", max_trials=4)}[strategy]()
    data = pllm.benchmark_search(search, "private", max_output_tokens=2).to_dict()
    assert len(driver) == len(data["candidates"]) == 3
    assert len(data["search"]["rejections"]) == 1
    assert data["search"]["rejections"][0]["reason"] == "component composition is incompatible"


def test_role_failure_retained_and_other_candidates_still_execute(driver, monkeypatch):
    from pllm.runtime.benchmark_cli import LoopbackBenchmarkError

    def run(**kwargs):
        if kwargs["experiment"].pipeline.kernels.params["threads"] == 2:
            raise LoopbackBenchmarkError("benchmark failed: private text should never enter search report")
        driver.append(kwargs)
        return report_for(kwargs)

    monkeypatch.setattr("pllm.runtime.benchmark_cli.run_loopback_benchmark", run)
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2, 3]})
    result = pllm.benchmark_search(pllm.BeamSearch("failure", space, "request_tps", "max"),
                                   "private", max_output_tokens=2).to_dict()
    assert len(driver) == 2 and len(result["search"]["rejections"]) == 1
    assert result["search"]["best_configuration_digest"] == driver[-1]["experiment"].configuration_digest()
    assert "private text" not in json.dumps(result)


@pytest.mark.parametrize("strategy", ["grid", "random", "beam"])
def test_interrupted_benchmark_stops_search_after_owned_role_cleanup(driver, monkeypatch, strategy):
    from pllm.runtime.benchmark_cli import LoopbackBenchmarkError

    def run(**kwargs):
        driver.append(kwargs)
        # The ordinary role driver closes roles, then wraps the interrupt.
        raise LoopbackBenchmarkError("benchmark interrupted") from KeyboardInterrupt()

    monkeypatch.setattr("pllm.runtime.benchmark_cli.run_loopback_benchmark", run)
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2, 3]})
    search = {"grid": lambda: pllm.GridSearch("cancel", space),
              "random": lambda: pllm.RandomSearch("cancel", space, seed=7, count=3),
              "beam": lambda: pllm.BeamSearch("cancel", space, "request_tps", "max")}[strategy]()
    with pytest.raises(KeyboardInterrupt):
        pllm.benchmark_search(search, "private", max_output_tokens=2)
    assert len(driver) == 1


@pytest.mark.parametrize("mutation", ["source", "output", "body", "salt", "sampling", "duration", "identity", "tokens", "cap", "communication"])
def test_changed_cohorts_and_untrustworthy_counters_fail_closed(driver, monkeypatch, mutation):
    def run(**kwargs):
        value = report_for(kwargs)
        driver.append(kwargs)
        if len(driver) == 2:
            if mutation == "source":
                value["configuration"]["source_lock_digest"] = "d" * 64
            elif mutation == "output":
                value["runs"][0]["generation"]["output_text_digest"] = "d" * 64
            elif mutation == "body":
                value["runs"][0]["model_fingerprint"] = "d" * 64
            elif mutation == "salt":
                value["configuration"]["prompt_digest"] = "d" * 64
            elif mutation == "sampling":
                value["configuration"]["sampling"]["effective_temperature"] = 0.7
            elif mutation == "duration":
                value["runs"][0]["durations"]["full_seconds"] = float("nan")
            elif mutation == "identity":
                value["experiment"]["configuration_digest"] = "d" * 64
            elif mutation == "tokens":
                value["runs"][0]["tokens"]["authoritative"] = False
            elif mutation == "cap":
                value["runs"][0]["max_output_tokens"] = value["configuration"]["max_output_tokens"] = 99
            elif mutation == "communication":
                value["communication_per_token"]["summary"]["online_mb_per_output_token"] = -1
        return value

    monkeypatch.setattr("pllm.runtime.benchmark_cli.run_loopback_benchmark", run)
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2, 3]})
    with pytest.raises(pllm.SearchError):
        pllm.benchmark_search(pllm.BeamSearch("tamper", space, "request_tps", "max"), "private", max_output_tokens=2)
    assert len(driver) == 2


def test_unknown_body_costs_remain_null_not_zero(driver, monkeypatch):
    def run(**kwargs):
        value = report_for(kwargs)
        value["communication_per_token"]["summary"] = {}
        return value

    monkeypatch.setattr("pllm.runtime.benchmark_cli.run_loopback_benchmark", run)
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2]})
    result = pllm.benchmark_search(pllm.BeamSearch("unknown", space, "covered_bytes", "min"),
                                   "private", max_output_tokens=2).to_dict()
    assert result["search"]["best_configuration_digest"] is None
    assert len(result["search"]["evaluations"]) == 1


def test_conflicting_objective_and_budget_never_launch(driver):
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2]})
    search = pllm.BeamSearch("bounds", space, "request_tps", "max")
    with pytest.raises(pllm.SearchError, match="conflicts"):
        pllm.benchmark_search(search, "private", max_output_tokens=2, objective="full_seconds")
    data = pllm.benchmark_search(search, "private", max_output_tokens=3).to_dict()
    assert data["search"]["best_configuration_digest"] is None
    assert not driver


def test_previous_trial_cycles_retire_before_next_memory_admission(driver, monkeypatch):
    references = []

    class TrialOwner:
        pass

    def run(**kwargs):
        owner = TrialOwner()
        owner.cycle = owner
        references.append(weakref.ref(owner))
        return report_for(kwargs)

    def memory(*args, **kwargs):
        assert all(reference() is None for reference in references)
        return {"admitted": True}

    monkeypatch.setattr("pllm.runtime.benchmark_cli.run_loopback_benchmark", run)
    monkeypatch.setattr("pllm.metrics.benchmark_memory", memory)
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2, 3]})
    enabled = gc.isenabled()
    gc.disable()
    try:
        result = pllm.benchmark_search(pllm.GridSearch("lifetime", space), "private", max_output_tokens=2)
        assert len(result.to_dict()["candidates"]) == 3
    finally:
        if enabled:
            gc.enable()
        gc.collect()
