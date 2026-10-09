import copy
import json

import pytest

import pllm
from pllm.evidence import BenchmarkResult, EvidenceRegistry
from pllm.metrics import (BenchmarkCase, BenchmarkRun, BenchmarkSuite, BenchmarkTarget,
                          BuildRef, Sampling, TargetBudget, benchmark_suite)
from pllm.profiles import MaskedLinearCpu
from pllm.search import FrontierStudy, Variant


@pytest.fixture(scope="module")
def study_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("frontier")
    experiment = pllm.Experiment("frontier", MaskedLinearCpu(pllm.Model.tiny()),
        pllm.Deployment.local(root=str(root)), pllm.ExecutionBudget(1, 16, 2))
    suite = BenchmarkSuite((BenchmarkCase("prepare", BenchmarkTarget.preparation(experiment),
                                         BuildRef.capture(), (TargetBudget("target.cpu", maximum=60),)),),
                           Sampling(processes=2))
    study = FrontierStudy(suite, (Variant("digit", {"kernel": "digit4"}),
                                 Variant("chunk", {"chunk_rows": 3})), (("digit", "chunk"),))
    work = study.plan(max_trials=8)
    run = benchmark_suite(work.suite, output=root / "run.json")
    return study, run


def test_frontier_interactions_ablations_and_exact_build_coverage(study_run):
    study, run = study_run
    assert len(run.results) == 8
    assert all(r.to_dict()["status"] == "completed" for r in run.results)
    assert run.compare("prepare", "prepare/digit+chunk", "target.cpu")["status"] == "comparable"
    registry = EvidenceRegistry().register_all(run.results)
    assert study.plan(registry).suite is None
    stale_case = BenchmarkCase("prepare", study.suite.cases[0].target, BuildRef(expected_digest="f" * 64))
    stale = FrontierStudy(BenchmarkSuite((stale_case,), study.suite.sampling))
    work = stale.plan(registry, changed_paths=["unknown/new-kernel.rs"])
    assert work.suite is not None
    assert work.coverage.to_dict()["cases"][0]["status"] == "stale_or_incomplete"
    assert work.coverage.to_dict()["cases"][0]["priority"] == "affected"
    assert FrontierStudy.from_spec(study.to_spec()).digest == study.digest


def test_snapshot_keeps_metric_gaps_and_never_combines_cohorts(study_run):
    study, run = study_run
    snapshot = study.snapshot(run, metrics={"target.cpu": "min", "process.peak_rss": "min"}).to_dict()
    assert snapshot["groups"][0]["nondominated"]
    assert snapshot["groups"][0]["status"] == "exploratory"
    missing = study.snapshot(run, metrics={"network.all_link_stream_bytes": "min"}).to_dict()
    assert not missing["groups"][0]["nondominated"]
    with pytest.raises(ValueError, match="separate cohort"):
        study.snapshot(run, confirmation=run)
    document = run.to_dict()
    document["results"][0]["execution"]["cohort"] = "f" * 64
    with pytest.raises(ValueError, match="cohort"):
        BenchmarkRun(document)
    document = run.to_dict()
    document["results"][0]["execution"]["sampling"]["threads"] = 2
    with pytest.raises(ValueError, match="cohort"):
        BenchmarkRun(document)


def test_budget_retest_always_includes_control_and_counts_processes(study_run):
    study, run = study_run
    registry = EvidenceRegistry(r for r in run.results if "+" not in r.to_dict()["execution"]["case"]["name"])
    work = study.plan(registry, max_trials=4)
    assert [c.name for c in work.suite.cases] == ["prepare", "prepare/digit+chunk"]
    assert work.coverage.to_dict()["selected_trials"] == 4
    assert study.plan(registry, max_trials=3).suite is None
    marker = object()
    assert work.inputs({"prepare": marker}) == {"prepare": marker, "prepare/digit+chunk": marker}


def test_result_rejects_forged_target_admission_budget_and_window(study_run):
    _, run = study_run
    raw = run.results[0].to_dict()
    for mutate in (
        lambda d: d.update(configuration_digest="f" * 64),
        lambda d: d["execution"]["admission"].update(required_bytes=1 << 60),
        lambda d: d["execution"]["budgets"][0].update(outcome="unknown"),
        lambda d: d["execution"]["oracle"].update(outcome="not_run"),
    ):
        bad = copy.deepcopy(raw)
        mutate(bad)
        with pytest.raises(ValueError):
            BenchmarkResult.from_dict(bad)
    document = run.to_dict()
    candidate = next(r for r in document["results"] if r["execution"]["case"]["name"] == "prepare/digit")
    next(m for m in candidate["metrics"] if m["id"] == "target.cpu")["window"] = "other_window"
    assert BenchmarkRun(document).compare("prepare", "prepare/digit", "target.cpu")["status"] == "unranked"


def test_v2_json_schema_and_existing_pareto_obey_scoped_contract(study_run):
    from pathlib import Path
    from jsonschema import Draft202012Validator
    from pllm.search import ParetoFrontier, SearchError, SearchEvaluation

    _, run = study_run
    schema = json.loads((Path(__file__).parents[1] / "schemas/benchmark-result-v2.schema.json").read_text())
    validator = Draft202012Validator(schema)
    for result in run.results:
        validator.validate(result.to_dict())
    left = run.results[0]
    right = left.to_dict()
    next(m for m in right["metrics"] if m["id"] == "target.cpu")["window"] = "other_window"
    with pytest.raises(SearchError, match="semantics"):
        ParetoFrontier([SearchEvaluation(None, left), SearchEvaluation(None, BenchmarkResult.from_dict(right))],
                       objectives={"target.cpu": "min"})


def test_private_salts_change_unrelated_run_commitments(study_run):
    study, previous = study_run
    again = benchmark_suite(BenchmarkSuite((study.suite.cases[0],), Sampling(processes=1)))
    assert again.results[0].to_dict()["workload_digest"] != previous.results[0].to_dict()["workload_digest"]
    assert "salt" not in json.dumps(again.to_dict())


def test_timeout_and_implementation_lock_mismatch_checkpoint(tmp_path):
    cases = (BenchmarkCase("timeout", BenchmarkTarget.runtime()),
             BenchmarkCase("wrong-build", BenchmarkTarget.runtime(), BuildRef(expected_digest="e" * 64)))
    run = benchmark_suite(BenchmarkSuite(cases, Sampling(processes=1, timeout_seconds=0.001)),
                          output=tmp_path / "run.json")
    assert [r.to_dict()["status"] for r in run.results] == ["timeout", "blocked"]
    assert BenchmarkRun.load(tmp_path / "run.json").digest == run.digest


def test_relative_interpreter_survives_isolated_working_directory(tmp_path, monkeypatch):
    import os
    import sys

    monkeypatch.chdir(tmp_path)
    relative = os.path.relpath(sys.executable, tmp_path)
    pinned = BuildRef.capture(relative)
    # resolve() would follow the interpreter symlink out of its virtualenv.
    assert pinned.python == os.path.abspath(sys.executable)
    run = benchmark_suite(BenchmarkSuite((
        BenchmarkCase("captured", BenchmarkTarget.runtime(), pinned),
        BenchmarkCase("unlocked", BenchmarkTarget.runtime(), BuildRef(relative)),
    ), Sampling(processes=1)))
    assert all(r.to_dict()["status"] == "completed" for r in run.results)
    assert BenchmarkRun(run.to_dict()).digest == run.digest


def test_implementation_lock_also_binds_observed_hardware(monkeypatch):
    from pllm import _benchmark_build

    monkeypatch.setattr(_benchmark_build, "hardware_identity", lambda: {"cpu_model": "fixture-a"})
    first = _benchmark_build.inspect()
    monkeypatch.setattr(_benchmark_build, "hardware_identity", lambda: {"cpu_model": "fixture-b"})
    second = _benchmark_build.inspect()
    assert first["attributes"]["sdk_sources"] == second["attributes"]["sdk_sources"]
    assert first["digest"] != second["digest"]


def test_coverage_does_not_freeze_private_inputs_or_external_files():
    from pllm.nonlinear import SharedRescale

    tiny = pllm.Experiment("coverage", MaskedLinearCpu(pllm.Model.tiny()),
        pllm.Deployment.local(root="local://coverage"), pllm.ExecutionBudget(1, 16, 2))
    external = tiny.with_params(pipeline__model=pllm.Model.hf("public/model", local_files_only=True))
    cases = (
        BenchmarkCase("response", BenchmarkTarget.response(tiny)),
        BenchmarkCase("operator", BenchmarkTarget.operator(SharedRescale())),
        BenchmarkCase("source", BenchmarkTarget.geometry(external)),
        BenchmarkCase("synthetic", BenchmarkTarget.preparation(tiny)),
    )
    rows = FrontierStudy(BenchmarkSuite(cases)).plan().coverage.to_dict()["cases"]
    reasons = {row["case"]: row["coverage_constraint"] for row in rows}
    assert reasons == {"response": "runtime_inputs_require_revalidation",
        "operator": "runtime_inputs_require_revalidation",
        "source": "external_source_requires_revalidation", "synthetic": None}


def test_completed_caller_owned_measurement_still_requires_revalidation():
    from pllm.nonlinear import SharedRescale

    case = BenchmarkCase("private", BenchmarkTarget.operator(SharedRescale(), lanes=2), BuildRef.capture())
    study = FrontierStudy(BenchmarkSuite((case,), Sampling(processes=1)))
    run = benchmark_suite(study.suite, inputs={"private": {"values": [-3, 4]}})
    assert run.results[0].to_dict()["status"] == "completed"
    assert run.results[0].to_dict()["execution"]["fixture"]["kind"] == "caller_owned"
    work = study.plan(EvidenceRegistry().register_all(run.results))
    assert work.suite is not None
    assert work.coverage.to_dict()["cases"][0]["status"] == "revalidation_required"


def test_confirmation_selects_comparable_finalists_with_fresh_controls(study_run):
    study, run = study_run
    work = study.confirm(run, cases=["prepare/digit+chunk"], metrics={"target.cpu": "min"})
    assert [case.name for case in work.suite.cases] == ["prepare", "prepare/digit+chunk"]
    assert work.suite.sampling.processes == 3
    assert work.coverage.to_dict()["selected_trials"] == 6
    marker = object()
    assert work.inputs({"prepare": marker}) == {"prepare": marker, "prepare/digit+chunk": marker}
    with pytest.raises(ValueError, match="measurements"):
        study.confirm(run, cases=["prepare/digit"], metrics={"network.all_link_stream_bytes": "min"})
    with pytest.raises(ValueError, match="replicates"):
        study.confirm(run, processes=2)


def test_maintained_prepared_study_expands_public_recipe_and_keeps_ablations():
    from benchmarks.research.prepared_frontier_study import study
    current = study(build=BuildRef(), output_tokens=2)
    work = current.plan()
    assert not [row for row in work.coverage.to_dict()["cases"] if row["status"] == "incompatible"]
    assert len(work.suite.cases) == 10
    assert "prepared/no-reuse+whole-prefill" in {c.name for c in work.suite.cases}
    for case in work.suite.cases:
        from pllm.configuration import Experiment
        p = case.target.to_spec()["parameters"]["model"]
        Experiment.from_spec(p["experiment"]).with_params(pipeline__profile=p["profile"]).resolve()
