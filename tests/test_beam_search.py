from __future__ import annotations

import json
from dataclasses import replace

import pytest

import pllm
from pllm.preparation import PreparedInventory
from pllm.search import BeamSearch, CandidateRejected
from test_search import experiment, result_for


def test_beam_measures_interacting_components_and_exports_exact_factory(monkeypatch):
    space = pllm.SearchSpace(experiment(), {
        "pipeline__kernels__threads": [1, 2, 3],
        "pipeline__inventory": [None, PreparedInventory("request-sized", rows=1)],
    })
    monkeypatch.setattr("pllm.search._grid", lambda *args: pytest.fail("beam materialized grid"))
    seen = []

    def measure(candidate):
        threads = candidate.experiment.pipeline.kernels.params["threads"]
        inventory = candidate.experiment.pipeline.inventory is not None
        seen.append((threads, inventory))
        latency = {(1, False): 20, (2, False): 21, (3, False): 99,
                   (1, True): 19, (2, True): 2, (3, True): 70}[threads, inventory]
        return result_for(candidate, latency=latency)

    outcome = BeamSearch("interactions", space, "latency", "min", width=2, max_trials=8).evaluate(measure)
    assert (2, True) in seen
    assert len(set(seen)) == len(seen) == outcome.attempted == 6
    assert outcome.best.candidate.experiment.pipeline.kernels.params["threads"] == 2
    assert outcome.best.candidate.experiment.pipeline.inventory is not None
    assert outcome.stop_reason == "neighborhood_exhausted"
    namespace = {}
    exec(outcome.best.candidate.python_source(), namespace)
    replayed = namespace["experiment"]()
    assert replayed.configuration_digest() == outcome.best.candidate.configuration_digest
    assert replayed.pipeline.profile == outcome.best.candidate.experiment.pipeline.profile
    pllm.search.evaluate_search((replace(outcome.best.candidate, experiment=replayed),),
                                lambda _: outcome.best.result)
    assert json.loads(json.dumps(outcome.to_dict()))["global_optimum_established"] is False


def test_admission_rejections_and_constraints_consume_bounded_attempts():
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2, 3, 4]},
                            constraints=[pllm.Constraint("pipeline__kernels__threads", "le", 3)])
    observed = []

    def measure(candidate):
        observed.append(candidate.experiment.pipeline.kernels.params["threads"])
        if observed[-1] == 3:
            raise CandidateRejected("capacity unavailable")
        return result_for(candidate)

    outcome = BeamSearch("bounds", space, "latency", "min", max_trials=4).evaluate(
        measure, admit=lambda c: "memory admission" if c.experiment.pipeline.kernels.params["threads"] == 2 else None,
    )
    assert observed == [1, 3]
    assert outcome.attempted == 4 and outcome.stop_reason == "budget"
    assert {r.reason for r in outcome.rejections} == {"memory admission", "capacity unavailable", "search constraint rejected"}
    assert outcome.best.candidate.index == 0
    with pytest.raises(RuntimeError, match="unexpected"):
        BeamSearch("errors", space, "latency", "min").evaluate(lambda c: (_ for _ in ()).throw(RuntimeError("unexpected")))


@pytest.mark.parametrize("change", ["failed", "unavailable", "source_reported"])
def test_unmeasured_and_failed_objectives_cannot_guide_search(change):
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2]})

    def measure(candidate):
        if candidate.index == 0:
            return result_for(candidate)
        value = result_for(candidate, status="failed" if change == "failed" else "completed", latency=0.01).to_dict()
        if change == "unavailable":
            metric = next(m for m in value["metrics"] if m["id"] == "latency")
            metric.update(value=None, sample_count=0, origin="not_available", unavailable_reason="unmeasured")
        elif change == "source_reported":
            for metric in value["metrics"]:
                metric["origin"] = "source_reported"
        return pllm.BenchmarkResult.from_dict(value)

    result = BeamSearch("unknowns", space, "latency", "min").evaluate(measure)
    assert result.best.candidate.index == 0
    assert len(result.evaluations) == 2


def test_no_base_measurement_stops_without_inventing_winner():
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2]})
    result = BeamSearch("unknown-base", space, "missing", "max").evaluate(result_for)
    assert result.best is None and result.attempted == 1
    assert result.stop_reason == "no_measured_objective"


@pytest.mark.parametrize("field", ["workload_digest", "privacy_cohort", "numeric_cohort", "profile"])
def test_changed_cohort_or_candidate_identity_stops_before_further_trials(field):
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2, 3]})
    seen = []

    def measure(candidate):
        seen.append(candidate.index)
        value = result_for(candidate).to_dict()
        if candidate.index:
            value[field] = "f" * 64
        return pllm.BenchmarkResult.from_dict(value)

    with pytest.raises(pllm.SearchError, match="cohort|profile"):
        BeamSearch("cohort", space, "latency", "min").evaluate(measure)
    assert seen == [0, 1]


def test_objective_semantics_and_trial_identity_are_enforced():
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2]})

    def measure(candidate):
        value = result_for(candidate).to_dict()
        if candidate.index:
            metric = next(m for m in value["metrics"] if m["id"] == "latency")
            metric["statistic"] = metric["parameters"]["statistic"] = "p95"
        return pllm.BenchmarkResult.from_dict(value)

    with pytest.raises(pllm.SearchError, match="semantics"):
        BeamSearch("semantics", space, "latency", "min").evaluate(measure)


def test_maximizing_and_ties_preserve_first_measured_incumbent():
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2, 3]})
    result = BeamSearch("max", space, "communication", "max").evaluate(result_for)
    assert result.best.candidate.experiment.pipeline.kernels.params["threads"] == 3
    tie = BeamSearch("ties", space, "latency", "min").evaluate(lambda c: result_for(c, latency=1))
    assert tie.best.candidate.index == 0


def test_budget_checks_and_incumbent_membership_are_explicit():
    space = pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2]})
    for kwargs in ({"width": 0}, {"max_trials": True}, {"width": 3, "max_trials": 2}):
        with pytest.raises(pllm.SearchError):
            BeamSearch("bounds", space, "latency", "min", **kwargs)
    with pytest.raises(pllm.SearchError, match="incumbent"):
        BeamSearch("absent", pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [2]}), "latency", "min")
    with pytest.raises(pllm.SearchError, match="overlap"):
        BeamSearch("overlap", pllm.SearchSpace(experiment(), {
            "pipeline__kernels": [experiment().pipeline.kernels], "pipeline__kernels__threads": [1, 2],
        }), "latency", "min")


def test_large_space_visits_only_budgeted_proposals(monkeypatch):
    space = pllm.SearchSpace(experiment(), {
        "pipeline__kernels__threads": list(range(1, 101)), "budget__max_new_tokens": list(range(1, 101)),
    })
    monkeypatch.setattr("pllm.search._grid", lambda *args: pytest.fail("materialized grid"))
    result = BeamSearch("lazy", space, "latency", "min", width=1, max_trials=1).evaluate(result_for)
    assert result.attempted == len(result.evaluations) == 1


def test_evaluation_binds_resolved_tiny_model_identity():
    base = experiment().with_params(pipeline__model=pllm.Model.tiny())
    assert base.resolve().model == "pllm-tiny-qwen2"
    candidate, = pllm.GridSearch("tiny", pllm.SearchSpace(base, {})).candidates()
    value = result_for(candidate).to_dict()
    value["model"]["id"] = "pllm-tiny-qwen2"
    assert len(pllm.evaluate_search((candidate,), lambda c: pllm.BenchmarkResult.from_dict(value))) == 1
