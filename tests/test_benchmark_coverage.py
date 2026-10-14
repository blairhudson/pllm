"""Coverage must never manufacture evidence from unrelated or partial runs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pllm._benchmark_contract import digest
from pllm._benchmark_evidence import assess_budgets
from pllm.components import list_components, planned_components
from pllm.evidence import BenchmarkResult, EvidenceRegistry
from pllm.metrics import (
    BenchmarkCase, BenchmarkRequirement, BenchmarkRun, BenchmarkSuite,
    benchmark_capabilities, benchmark_coverage,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def archived():
    run = BenchmarkRun.load(ROOT / "docs/evidence/research-benchmark-targets.json")
    first = run.to_dict()["suite"]["cases"][0]
    records = [r.to_dict() for r in run.results if r.to_dict()["execution"]["case"] == first]
    assert len(records) == 3
    return run, records


def coverage(records, *, requirements=None, **kwargs):
    registry = EvidenceRegistry(BenchmarkResult.from_dict(r) for r in records)
    return benchmark_coverage(registry=registry, requirements=requirements or (
        BenchmarkRequirement("runtime", target={"kind": "runtime"}),), **kwargs).to_dict()


def test_every_component_parameter_enum_and_pending_gate_is_visible():
    specs = {r.name: r.to_spec() for r in benchmark_capabilities()}
    for descriptor in list_components():
        assert specs[descriptor.component]["component"] == descriptor.component
        for name, field in descriptor.to_dict()["parameter_schema"].get("properties", {}).items():
            key = f"{descriptor.component}/parameter/{name}"
            assert specs[key]["present"] == [name]
            for value in field.get("enum", [False, True] if field.get("type") == "boolean" else []):
                assert specs[f"{key}/choice/{digest(value)[:16]}"]["parameters"] == {name: value}
    for item in planned_components():
        if item.status == "pending":
            assert specs[item.identity]["disposition"] == "unsupported"
            assert specs[item.identity]["reason"] == item.gate


def test_requirements_are_immutable_exact_json_predicates_and_do_not_invent_defaults():
    params = {"decode": True}
    target = {"kind": "response"}
    requirement = BenchmarkRequirement("decode", component="pllm/client-prefix-reuse/v1",
        parameters=params, present=("decode",), target=target)
    params["decode"] = False
    target["kind"] = "operator"
    spec = requirement.to_spec()
    spec["parameters"]["decode"] = False
    assert BenchmarkRequirement.from_spec(requirement.to_spec()) == requirement
    component = {"component": "pllm/client-prefix-reuse/v1", "params": {"decode": True}}
    configured = {"kind": "response", "parameters": {"model": {"experiment": {"pipeline": {"components": {"state": component}}}}}}
    assert requirement.matches(configured)
    for wrong in (False, 1, None):
        component["params"]["decode"] = wrong
        assert not requirement.matches(configured)
    component["params"] = {}
    assert not requirement.matches(configured)
    configured["component_ids"] = ["pllm/client-prefix-reuse/v1"]
    assert not requirement.matches(configured)
    with pytest.raises(ValueError, match="reason"):
        BenchmarkRequirement("unavailable", disposition="unsupported")
    configured["parameters"]["model"]["config"] = {"component": "pllm/client-prefix-reuse/v1", "params": {"decode": True}}
    assert not requirement.matches(configured)


def test_individual_features_do_not_cover_their_interaction():
    requirement = BenchmarkRequirement("two-options", target={"kind": "response", "parameters": {"a": 1, "b": 2}})
    assert not requirement.matches({"kind": "response", "parameters": {"a": 1}})
    assert not requirement.matches({"kind": "response", "parameters": {"b": 2}})
    assert requirement.matches({"kind": "response", "parameters": {"a": 1, "b": 2, "c": 3}})
    blocked = BenchmarkRequirement("blocked", disposition="unsupported", reason="requires new continuation contract")
    assert not blocked.matches({})


def test_completed_exact_build_has_scoped_cells_and_unknown_metrics(archived):
    _, records = archived
    report = coverage(records, build_digest=records[0]["execution"]["build"]["digest"])
    assert report["requirements"][0]["status"] == "measured"
    cell, = report["cells"]
    assert cell["observed_replicates"] == [0, 1, 2]
    assert cell["oracle_passed"] and cell["build_locked"]
    assert cell["measured_metrics"]
    assert cell["environment_digest"] == records[0]["environment"]["digest"]
    assert cell["fixture"] == records[0]["execution"]["fixture"]
    assert cell["numeric_cohort"] == records[0]["numeric_cohort"]
    stale = coverage(records, build_digest="f" * 64)
    assert stale["requirements"][0]["status"] == "stale_build"
    assert stale["requirements"][0]["measured_kinds"] == []


def test_partial_runs_and_unrun_suites_remain_gaps(archived):
    run, records = archived
    suite = BenchmarkSuite.from_spec(run.to_dict()["suite"])
    report = coverage(records[:2], suites=[suite])
    assert report["requirements"][0]["status"] == "incomplete"
    assert report["requirements"][0]["unmeasured_cases"]
    empty = benchmark_coverage(suites=[suite], requirements=(BenchmarkRequirement("runtime", target={"kind":"runtime"}),)).to_dict()
    assert empty["requirements"][0]["status"] == "unmeasured"
    assert all(not c["measured"] for c in empty["case_coverage"])
    partial = coverage(records, suites=[suite])
    assert partial["requirements"][0]["status"] == "partial"
    assert partial["requirements"][0]["has_measurement"]


@pytest.mark.parametrize("boundary", ["cohort", "suite_digest", "source_digest", "environment", "fixture", "scope"])
def test_distinct_boundaries_cannot_fill_missing_replicates(archived, boundary):
    _, original = archived
    records = copy.deepcopy(original)
    row = records[-1]
    if boundary in {"cohort", "suite_digest"}:
        row["execution"][boundary] = "f" * 64
    elif boundary == "source_digest":
        row["execution"]["comparison"][boundary] = "f" * 64
    elif boundary == "environment":
        from pllm.evidence import environment_digest
        row["environment"]["attributes"]["coverage_fixture"] = "different-machine"
        row["environment"]["digest"] = environment_digest(row["environment"]["attributes"])
    elif boundary == "fixture":
        row["execution"]["fixture"]["digest"] = "f" * 64
    else:
        row["scope"] = "operator"
    report = coverage(records)
    assert len(report["cells"]) == 2
    assert all(c["status"] == "incomplete" for c in report["cells"])


def test_duplicate_replicas_are_not_extra_independent_samples(archived):
    _, records = archived
    extra = copy.deepcopy(records[0])
    extra["id"] = "duplicate-with-different-record-id"
    report = coverage([*records, extra])
    assert report["requirements"][0]["status"] == "duplicate_replicate"


@pytest.mark.parametrize("condition, status", [("unlocked", "unlocked_build"), ("reused", "not_isolated")])
def test_missing_build_lock_or_fresh_process_is_not_measurement(archived, condition, status):
    _, original = archived
    records = copy.deepcopy(original)
    for row in records:
        e = row["execution"]
        if condition == "unlocked":
            e["case"]["build"]["expected_digest"] = None
            e["case_digest"] = digest(e["case"])
        else:
            e["isolation"]["fresh_process"] = False
    assert coverage(records)["requirements"][0]["status"] == status


def test_missing_or_incompatible_metric_semantics_are_not_covered(archived):
    _, original = archived
    records = copy.deepcopy(original)
    metric = records[0]["metrics"][0]
    metric["window"] = "different-window"
    report = coverage(records)
    assert metric["id"] not in report["cells"][0]["measured_metrics"]
    assert metric["id"] in report["cells"][0]["unmeasured_metrics"]
    for row in records:
        for item in row["metrics"]:
            item.update(value=None, origin="not_available", sample_count=0, unavailable_reason="fixture missing")
        row["execution"]["budgets"] = assess_budgets(BenchmarkCase.from_spec(row["execution"]["case"]), row["metrics"])
    assert coverage(records)["requirements"][0]["status"] == "no_measured_metrics"


def test_sdk_cli_reads_archived_evidence_without_starting_work(capsys, monkeypatch):
    from pllm._cli.app import main
    import pllm.metrics.benchmark_runner as runner

    def forbidden(*args, **kwargs):
        raise AssertionError("coverage must not execute benchmarks")
    monkeypatch.setattr(runner, "benchmark_suite", forbidden)
    main(["benchmark", "coverage", "--evidence", str(ROOT / "docs/evidence/research-client-stage-tiny.json"), "--format", "json"])
    report = json.loads(capsys.readouterr().out)["data"]["report"]
    assert report["counts"]["measured"] > 0
    assert report["counts"]["unmapped"] > 0
    with pytest.raises(SystemExit) as rejected:
        main(["benchmark", "coverage", "--require", "mapped", "--format", "json"])
    assert rejected.value.code == 3
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "BENCHMARK_COVERAGE_GAP"


def test_repository_inventory_keeps_unbenchmarked_exports_and_static_links_separate(tmp_path, monkeypatch):
    import scripts.benchmark_coverage as inventory

    class NewCapability:
        def new_method(self, value=1): ...

        @property
        def read_only(self):
            raise AssertionError("inventory must not execute property getters")

    monkeypatch.setattr(inventory, "public_exports", lambda: [{"module":"pllm.example", "name":"NewCapability", "value":NewCapability}])
    first = inventory.surface()
    row = next(iter(first.values()))
    assert "new_method" in row["contract"]["members"]
    assert row["contract"]["attributes"]["read_only"]["kind"] == "property"
    aliases = {"pllm.example.NewCapability": "capability"}
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks/example.py").write_text("from pllm.example import NewCapability as C\ndef recipe(): return C()\n")
    assert inventory.source_references(tmp_path, aliases) == {"capability": ["benchmarks/example.py:2"]}
    NewCapability.new_method = lambda self, value=2: None
    second = inventory.surface()
    assert inventory.sha(row["contract"]) != inventory.sha(next(iter(second.values()))["contract"])


def test_decode_and_growing_evidence_use_existing_strict_cohort_gates():
    from scripts.benchmark_coverage import external_decode, read

    report, source_hash = read(ROOT / "docs/evidence/decode-cache-sdk-qwen25-2026-10-06.json")
    linked = external_decode(report, source_hash, None)
    assert linked["status"] == "validated_historical"
    assert external_decode(report, source_hash, "f" * 64)["status"] == "stale_build"
    broken = copy.deepcopy(report)
    broken["results"].pop()
    with pytest.raises(ValueError):
        external_decode(broken, source_hash, None)


def test_review_gate_detects_new_methods_lost_links_and_lost_evidence():
    from scripts.benchmark_coverage import review_changes

    previous = {"schema": "v1", "api": {"A":"1"}, "components": {}, "requirements": {},
                "coverage_links": {"A":"linked"}, "measured_requirements": ["old"]}
    changed = copy.deepcopy(previous)
    changed["api"]["A"] = "new-signature"
    changed["api"]["B"] = "new-export"
    changed["coverage_links"]["A"] = "unmapped"
    changed["measured_requirements"] = ["new"]
    assert review_changes(previous, changed) == {"api": ["A", "B"], "coverage_links": ["A"], "measurement_losses": ["old"]}
    progressed = copy.deepcopy(previous)
    progressed["measured_requirements"].append("new")
    assert review_changes(previous, progressed) == {}
