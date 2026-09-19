from __future__ import annotations

import hashlib

import pytest

import pllm
from pllm.kernels import Cpu
from pllm.profiles import MaskedLinearCpu


def experiment() -> pllm.Experiment:
    return pllm.Experiment(
        name="search-base",
        pipeline=MaskedLinearCpu(pllm.Model("org/model"), kernels=Cpu(threads=1)),
        deployment=pllm.Deployment.local(root="local://search"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    )


def workload_digest(max_new_tokens: int) -> str:
    return hashlib.sha256(f"workload:{max_new_tokens}".encode()).hexdigest()


def result_for(
    candidate: pllm.SearchCandidate,
    *,
    status: str = "completed",
    latency: float | None = None,
    communication: float | None = None,
) -> pllm.BenchmarkResult:
    threads = candidate.experiment.pipeline.kernels.params["threads"]
    max_new_tokens = candidate.experiment.budget.max_new_tokens
    latency = 4.0 / threads if latency is None else latency
    communication = float(threads * 10) if communication is None else communication
    sample_status = "completed" if status == "completed" else "failed"
    metrics = [
        {
            "id": "latency",
            "component": "pllm/latency",
            "parameters": {"statistic": "median", "phase": "full"},
            "value": latency,
            "unit": "seconds",
            "unit_detail": None,
            "statistic": "median",
            "phase": "full",
            "role": "client",
            "origin": "native_executed",
            "unavailable_reason": None,
            "sample_count": 1,
        },
        {
            "id": "communication",
            "component": "pllm/communication",
            "parameters": {"direction": "total", "phase": "online"},
            "value": communication,
            "unit": "bytes",
            "unit_detail": None,
            "statistic": "sum",
            "phase": "online",
            "role": "client",
            "origin": "native_executed",
            "unavailable_reason": None,
            "sample_count": 1,
        },
    ]
    return pllm.BenchmarkResult.from_dict({
        "schema_version": "pllm.benchmark_result.v1",
        "id": candidate.trial_id,
        "created_at": "2026-09-19T12:00:00Z",
        "status": status,
        "scope": "deployment",
        "profile": candidate.experiment.pipeline.profile,
        "component_ids": sorted(
            component.component
            for component in candidate.experiment.pipeline.components.values()
        ),
        "model": {
            "id": candidate.experiment.pipeline.model.source,
            "checkpoint_digest": None,
            "source_lock_digest": None,
        },
        "plan_lock_digest": None,
        "configuration_digest": candidate.configuration_digest,
        "workload_digest": workload_digest(max_new_tokens),
        "privacy_cohort": "masked-linear",
        "numeric_cohort": "wrap32",
        "environment": {
            "digest": pllm.environment_digest({"cpu": "fixture"}),
            "attributes": {"cpu": "fixture"},
        },
        "warmups": 0,
        "repetitions": 1,
        "samples": [
            {
                "index": 0,
                "kind": "measurement",
                "status": sample_status,
                "duration_seconds": latency if sample_status == "completed" else None,
                "artifact_digest": None,
            }
        ],
        "metrics": metrics,
        "limitations": ["fixture search result"],
    })


def test_grid_search_is_exhaustive_typed_and_deterministic() -> None:
    space = pllm.SearchSpace(
        experiment(),
        {
            "pipeline__kernels__threads": [1, 2],
            "budget__max_new_tokens": [2, 4],
        },
    )
    search = pllm.GridSearch("grid", space)
    first = search.candidates()
    second = search.candidates()
    assert first == second
    assert len(first) == 4
    assert [candidate.index for candidate in first] == [0, 1, 2, 3]
    assert len({candidate.configuration_digest for candidate in first}) == 4
    assert {
        (
            candidate.experiment.pipeline.kernels.params["threads"],
            candidate.experiment.budget.max_new_tokens,
        )
        for candidate in first
    } == {(1, 2), (1, 4), (2, 2), (2, 4)}
    with pytest.raises(TypeError):
        first[0].parameters["changed"] = True


def test_constraints_filter_candidates_without_silent_invalid_substitution() -> None:
    space = pllm.SearchSpace(
        experiment(),
        {"pipeline__kernels__threads": [1, 2, 4]},
        constraints=[pllm.Constraint("pipeline__kernels__threads", "ge", 2)],
    )
    assert [
        candidate.experiment.pipeline.kernels.params["threads"]
        for candidate in pllm.GridSearch("constrained", space).candidates()
    ] == [2, 4]
    with pytest.raises(pllm.SearchError, match="unknown"):
        pllm.SearchSpace(experiment(), {"pipeline__missing": [1]})
    with pytest.raises(pllm.SearchError, match="invalid value"):
        pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [0]})
    overlapping = pllm.SearchSpace(
        experiment(),
        {
            "pipeline__kernels": [Cpu(threads=2)],
            "pipeline__kernels__threads": [4],
        },
    )
    with pytest.raises(pllm.SearchError, match="combined"):
        pllm.GridSearch("overlap", overlapping).candidates()


def test_seeded_random_search_is_reproducible_without_replacement() -> None:
    space = pllm.SearchSpace(
        experiment(), {"pipeline__kernels__threads": [1, 2, 3, 4, 5]}
    )
    first = pllm.RandomSearch("random", space, count=3, seed=17).candidates()
    second = pllm.RandomSearch("random", space, count=3, seed=17).candidates()
    different = pllm.RandomSearch("random", space, count=3, seed=18).candidates()
    assert first == second
    assert first != different
    assert len({candidate.configuration_digest for candidate in first}) == 3
    with pytest.raises(pllm.SearchError, match="exceeds"):
        pllm.RandomSearch("random", space, count=6, seed=1).candidates()


def test_evaluation_binds_trial_configuration_profile_model_and_components() -> None:
    candidates = pllm.GridSearch(
        "evaluate",
        pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2]}),
    ).candidates()
    evaluations = pllm.evaluate_search(candidates, result_for)
    assert len(evaluations) == 2
    assert all(evaluation.result.id == evaluation.candidate.trial_id for evaluation in evaluations)

    def wrong_id(candidate):
        document = result_for(candidate).to_dict()
        document["id"] = "wrong"
        return pllm.BenchmarkResult.from_dict(document)

    with pytest.raises(pllm.SearchError, match="trial id"):
        pllm.evaluate_search(candidates[:1], wrong_id)
    with pytest.raises(TypeError, match="BenchmarkResult"):
        pllm.evaluate_search(candidates[:1], lambda candidate: {})
    with pytest.raises(pllm.SearchError, match="more than once"):
        pllm.evaluate_search((candidates[0], candidates[0]), result_for)


def test_pareto_frontier_uses_explicit_directions_and_preserves_excluded_results() -> None:
    candidates = pllm.GridSearch(
        "pareto",
        pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2, 4]}),
    ).candidates()
    evaluations = (
        pllm.SearchEvaluation(candidates[0], result_for(candidates[0], latency=4, communication=10)),
        pllm.SearchEvaluation(candidates[1], result_for(candidates[1], latency=2, communication=20)),
        pllm.SearchEvaluation(candidates[2], result_for(candidates[2], status="failed")),
    )
    frontier = pllm.ParetoFrontier(
        evaluations,
        objectives={"communication": "min", "latency": "min"},
    )
    assert frontier.results() == evaluations[:2]
    assert frontier.excluded() == evaluations[2:]

    dominated = pllm.SearchEvaluation(
        candidates[1], result_for(candidates[1], latency=5, communication=20)
    )
    frontier = pllm.ParetoFrontier(
        (evaluations[0], dominated),
        objectives={"communication": "min", "latency": "min"},
    )
    assert frontier.results() == (evaluations[0],)


def test_pareto_frontier_rejects_mixed_cohorts_and_implicit_scalarization() -> None:
    candidates = pllm.GridSearch(
        "mixed",
        pllm.SearchSpace(
            experiment(),
            {
                "pipeline__kernels__threads": [1],
                "budget__max_new_tokens": [2, 4],
            },
        ),
    ).candidates()
    evaluations = tuple(
        pllm.SearchEvaluation(candidate, result_for(candidate)) for candidate in candidates
    )
    with pytest.raises(pllm.SearchError, match="exact comparison cohort"):
        pllm.ParetoFrontier(evaluations, objectives={"latency": "min"})
    with pytest.raises(pllm.SearchError, match="directions"):
        pllm.ParetoFrontier(evaluations[:1], objectives={"latency": "best"})

    comparable = pllm.GridSearch(
        "semantics",
        pllm.SearchSpace(experiment(), {"pipeline__kernels__threads": [1, 2]}),
    ).candidates()
    first = pllm.SearchEvaluation(comparable[0], result_for(comparable[0]))
    changed = result_for(comparable[1]).to_dict()
    latency = next(metric for metric in changed["metrics"] if metric["id"] == "latency")
    latency["parameters"]["statistic"] = "p95"
    latency["statistic"] = "p95"
    changed_result = pllm.BenchmarkResult.from_dict(changed)
    with pytest.raises(pllm.SearchError, match="objective semantics"):
        pllm.ParetoFrontier(
            (
                first,
                pllm.SearchEvaluation(comparable[1], changed_result),
            ),
            objectives={"latency": "min"},
        )
    assert not hasattr(pllm.ParetoFrontier, "best")
