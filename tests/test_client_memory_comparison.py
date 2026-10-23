"""Synthetic observations in real ordinary/canonical Reports; no models executed.

Trusted-process registry metadata is the source, not a new receipt/authentication
scheme. Qualified interval fixtures exercise a future source contract; the current
live-client producer cannot supply it and never establishes a 10x result here.
"""
from copy import deepcopy
import os
from types import SimpleNamespace as NS

import pytest

import pllm
from pllm._benchmark_contract import canonical, digest
from pllm._benchmark_evidence import measurement
from pllm._benchmark_worker import Measurement
from pllm.evidence import BenchmarkResult, EvidenceRegistry
from pllm.metrics import (
    BenchmarkCase, BenchmarkRun, BenchmarkSuite, BenchmarkTarget, MetricSpec,
    Sampling, Scenario, client_memory_metric_specs,
)
from pllm.metrics.benchmark_runner import _result
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.benchmark_cli import build_loopback_report
from pllm.runtime.client_memory_scope import ClientMemoryScope, client_memory_metrics


SUM = "client.owned_sampled_peak_rss_sum"
LOWER = "client.owned_peak_rss_lower_bound"
INTERVAL = "client.owned_lifetime_peak_rss_sum_interval"


def scope(value=100):
    owner = ClientMemoryScope()
    owner.seal()
    owner.observe({"client": (os.getpid(), 1., value)}, span_seconds=.001)
    return owner.report(value)


def report(value=100):
    row = {"status": "completed", "cold": True, "warm": False,
        "tokens": {"authoritative": True, "input_tokens": 3, "output_tokens": 2},
        "model_fingerprint": "b" * 64, "max_output_tokens": 2,
        "generation": {"response_status": "completed", "output_text_digest": "c" * 64},
        "durations": {"full_seconds": .5, "online_seconds": .3, "ttft_seconds": .1,
                      "tokens_per_second": 5, "preparation_seconds": .2},
        "privacy": {"plaintext_prompt_bytes_sent": 0, "plaintext_token_ids_sent": 0,
                    "preparation_requests_during_online": 0}}
    value_report = build_loopback_report(model_id="telemetry-test-double", tiny=False,
        max_output_tokens=2, warmup_runs=[], runs=[row], source_lock_digest="a" * 64,
        capture_output_digest=True)
    value_report["client_owned_memory"] = scope(value)
    return value_report


def run(left=None, right=None, *, salt=None, mutation=None, status="completed", complete=True):
    # Declaration only; no checkpoint/model loading, native compute or subprocess.
    experiment = pllm.Experiment("memory-comparison", MaskedLinearCpu(pllm.Model.tiny()),
        pllm.Deployment.local(root="target/memory-comparison"), pllm.ExecutionBudget(1, 128, 2))
    target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=2))
    cases = tuple(BenchmarkCase(name, target) for name in ("control", "candidate"))
    sampling = Sampling(processes=1)
    suite = BenchmarkSuite(cases, sampling)
    results = []
    for index, (case, observed) in enumerate(zip(cases, (left or report(), right or report(1)), strict=True)):
        m = Measurement({"sampling": sampling.to_spec(), "salt": salt if index and salt else "ab" * 32,
            "inputs": {"prompt": "private-comparison-input"}, "directory": "."})
        m.details["report"] = deepcopy(observed)
        client_memory_metrics(observed, m)
        m.metrics.append(measurement("legacy", 1))
        m.samples.append({"index": 0, "kind": "measurement", "status": "completed",
            "duration_seconds": .5, "artifact_digest": None})
        m.fixture = {"kind": "generated_checkpoint", "digest": "a" * 64, "metadata": {}}
        m.oracle = {"kind": "test_double", "outcome": "passed", "checked_values": 1}
        m.admission.update(admitted=True, required_bytes=1, host_headroom_bytes=1 << 30)
        m.comparison.update({key: "a" * 64 if key == "source_digest" else "b" * 64 for key in m.comparison})
        m.finalize_workload(target)
        attributes = {"fixture": "synthetic-report-not-a-build-measurement"}
        result = _result(case, suite, 0, "d" * 64,
            {"attributes": attributes, "digest": digest(attributes)},
            {"status": status, "failure_code": None if status == "completed" else "fixture_failure",
             "metrics": m.metrics, "details": m.details, "fixture": m.fixture, "oracle": m.oracle,
             "comparison": m.comparison, "admission": m.admission, "samples": m.samples}).to_dict()
        if mutation is not None and index:
            mutation(result)
        results.append(BenchmarkResult.from_dict(result).to_dict())
    return BenchmarkRun({"schema": "pllm.benchmark_run.v1", "suite": suite.to_spec(),
        "cohort": "d" * 64, "complete": complete, "results": results})


def interval(observed, values):
    # Future, explicitly final per-individual HWM contract; never derived from
    # current root snapshot / wait4 helper maximum or assumed isolated by PID.
    observed["client_owned_memory"]["lifetime_peak_rss_sum_interval"] = {
        "lower_bound_bytes": max(values.values()), "upper_bound_bytes": sum(values.values()),
        "member_hwm_bytes": values, "source": "final_individual_os_hwm", "unit": "bytes",
        "hwm_coverage": "individual_process_only", "finalized_after_full_member_retirement": True,
        "metric": "client_owned_rss_sum_peak", "isolated_lifetime": True,
        "membership_complete": True, "reason": None,
        "window": "isolated_client_group_birth_through_complete_retirement"}
    return observed


def test_public_specs_discover_and_select_actual_ordinary_registry_metric_names():
    specs = client_memory_metric_specs()
    assert {spec.id for spec in specs} == {SUM, LOWER, INTERVAL}
    assert all(spec.to_spec()["unit"] == "bytes" and not spec.to_spec()["physical_peak"] for spec in specs)
    observed = run()
    registry = EvidenceRegistry(observed.results)
    for spec in specs:
        assert registry.query(metric=spec) == registry.query(metric=spec.id) == registry.list()
    # Existing component queries/defaults unchanged; discovery alone claims no availability.
    assert registry.query(metric="pllm/memory") == registry.list()
    assert observed.compare("control", "candidate", INTERVAL)["status"] == "unranked"


def test_complete_sampled_sums_are_descriptive_not_physical_point_rankings_or_tenfold_proof():
    observed = run(report(10000), report(1))
    before = observed.to_dict()
    result = observed.compare("control", "candidate", MetricSpec(SUM))
    assert result["status"] == "descriptive_only" and result["kind"] == "estimate"
    assert result["pairs"] == [{"control_bytes": 10000, "candidate_bytes": 1}]
    assert result["reduction_status"] == "not_established" and not result["physical_peak"]
    assert "paired_control_over_candidate" not in result and "winners" not in result
    assert observed.to_dict() == before and all(row["status"] == "completed" for row in before["results"])


@pytest.mark.parametrize("change", [
    lambda value: value.pop("client_owned_memory"),
    lambda value: value.update(client_owned_memory=None),
    lambda value: value["client_owned_memory"].update(membership_declared=False),
    lambda value: value["client_owned_memory"].update(membership_declared_before_sampling=False),
    lambda value: value["client_owned_memory"].update(sampling_coverage="unknown_or_partial"),
    lambda value: value["client_owned_memory"].update(missing_member_samples=1),
    lambda value: value["client_owned_memory"].update(complete_member_samples=0),
    lambda value: value["client_owned_memory"].update(samples=0),
    lambda value: value["client_owned_memory"].update(window="different_or_reset_window"),
    lambda value: value["client_owned_memory"]["latest"].update(missing_members=["client_helper_1"]),
    lambda value: value["client_owned_memory"]["latest"]["members"].update(
        client_helper_1={"rss_bytes": None, "reason": "member_identity_changed"}),
    lambda value: value["client_owned_memory"]["latest"]["members"].update(
        client_helper_1={"rss_bytes": None, "reason": "remote_helper_unobservable"}),
    lambda value: value["client_owned_memory"].update(sampled_peak_rss_sum_bytes=None),
])
def test_unknown_reset_missing_or_reused_members_do_not_rank_or_fail_accepted_response(change):
    candidate = report(1)
    change(candidate)
    # Missing/null section is an ordinary optional report field, not target failure.
    # The settled adapter accepts only a section of the correct schema.
    if candidate.get("client_owned_memory") is None:
        observed = run(mutation=lambda row: row["execution"]["details"]["report"].update(
            client_owned_memory=candidate.get("client_owned_memory")))
    else:
        observed = run(right=candidate)
    result = observed.compare("control", "candidate", SUM)
    assert result["status"] == "unranked"
    assert all(row["status"] == "completed" for row in observed.to_dict()["results"])


def test_partial_explicit_lower_bounds_remain_bounds_never_ratios():
    candidate = report(1)
    candidate["client_owned_memory"].update(membership_declared=False,
        sampling_coverage="unknown_or_partial", missing_member_samples=1)
    candidate["client_owned_memory"]["latest"]["missing_members"] = ["undeclared_client_helpers"]
    result = run(right=candidate).compare("control", "candidate", LOWER)
    assert result["status"] == "bounded" and result["kind"] == "lower_bound"
    assert result["reduction_status"] == "not_established"
    assert "paired_control_over_candidate" not in result


@pytest.mark.parametrize("key", ["numeric_digest", "state_digest", "output_digest", "measurement_digest"])
def test_existing_numeric_trajectory_output_and_measurement_guards_still_apply(key):
    result = run(mutation=lambda row: row["execution"]["comparison"].update({key: "e" * 64}))
    assert result.compare("control", "candidate", SUM) == {
        "status": "unranked", "reason": "cohort_contract_differs"}


def test_different_private_salts_cannot_compare_or_leak_raw_inputs():
    observed = run(salt="cd" * 32)
    assert observed.compare("control", "candidate", SUM)["reason"] == "cohort_contract_differs"
    assert b"private-comparison-input" not in canonical(observed.to_dict())


@pytest.mark.parametrize("change", [
    lambda row: row["execution"]["details"]["report"]["configuration"].update(source_lock_digest="e" * 64),
    lambda row: row["execution"]["details"]["report"]["runs"][0].update(model_fingerprint="e" * 64),
    lambda row: next(m for m in row["metrics"] if m["id"] == SUM).update(statistic="median"),
    lambda row: next(m for m in row["metrics"] if m["id"] == SUM).update(origin="measured"),
    lambda row: next(m for m in row["metrics"] if m["id"] == SUM).update(window="individual_process_lifetime"),
])
def test_source_body_statistics_and_individual_hwm_cannot_masquerade_as_owned_group(change):
    observed = run(mutation=change)
    assert observed.compare("control", "candidate", SUM)["status"] == "unranked"


def test_qualified_intervals_require_candidate_upper_times_factor_below_control_lower():
    observed = run(interval(report(1000), {"client": 1000}), interval(report(100), {"client": 100}))
    result = observed.compare("control", "candidate", INTERVAL, minimum_reduction=10)
    assert result["status"] == "bounded" and result["reduction_status"] == "established_in_scope"
    assert result["pairs"][0]["control"] == {"lower_bytes": 1000, "upper_bytes": 1000}
    assert not result["physical_peak"] and "control_median" not in result
    insufficient = run(interval(report(1000), {"client": 1000}), interval(report(101), {"client": 101}))
    assert insufficient.compare("control", "candidate", INTERVAL)["reduction_status"] == "not_established"


def test_helper_hwm_maximum_never_substitutes_for_group_sum_upper_bound():
    candidate = report(100)
    scoped = candidate["client_owned_memory"]
    scoped["latest"]["members"]["client_helper_1"] = {"rss_bytes": 200, "reason": None}
    scoped["latest"]["same_pass_rss_sum_bytes"] = scoped["sampled_peak_rss_sum_bytes"] = 300
    scoped["sampled_peak_rss_lower_bound_bytes"] = 200
    candidate = interval(candidate, {"client": 100, "client_helper_1": 200})
    observed = run(interval(report(1000), {"client": 1000}), candidate)
    assert observed.compare("control", "candidate", INTERVAL, minimum_reduction=4)["reduction_status"] == "not_established"
    assert observed.compare("control", "candidate", INTERVAL, minimum_reduction=3)["reduction_status"] == "established_in_scope"


@pytest.mark.parametrize("selection", [SUM, LOWER, INTERVAL, MetricSpec(SUM)])
def test_new_client_memory_selections_cannot_enter_ordinary_point_pareto_ranking(selection):
    from pllm.search import ParetoFrontier, SearchEvaluation

    evaluations = tuple(SearchEvaluation(NS(), result) for result in run().results)
    frontier = ParetoFrontier(evaluations, objectives={selection: "min"})
    assert frontier.results() == () and frontier.excluded() == evaluations
    legacy = ParetoFrontier(evaluations, objectives={"legacy": "min"})
    assert legacy.results() == evaluations


@pytest.mark.parametrize("change", [
    lambda bounds: bounds.update(isolated_lifetime=False),
    lambda bounds: bounds.update(membership_complete=False),
    lambda bounds: bounds.update(window="sampler_start_through_responses"),
    lambda bounds: bounds.update(source="wait4_maximum_individual_including_reaped_descendants"),
    lambda bounds: bounds.update(hwm_coverage="maximum_individual_including_reaped_descendants"),
    lambda bounds: bounds.update(finalized_after_full_member_retirement=False),
    lambda bounds: bounds.update(metric="individual_client_hwm"),
    lambda bounds: bounds.update(unit="KiB"),
    lambda bounds: bounds.update(upper_bound_bytes=1),
    lambda bounds: bounds.update(member_hwm_bytes={"client": 100, "unclassified_helper": 200}),
])
def test_no_interval_point_pretence_mismatched_windows_units_or_unclassified_hwm_members(change):
    candidate = interval(report(100), {"client": 100})
    change(candidate["client_owned_memory"]["lifetime_peak_rss_sum_interval"])
    result = run(interval(report(1000), {"client": 1000}), candidate).compare("control", "candidate", INTERVAL)
    assert result["status"] == "unranked"


def test_legacy_defaults_ids_and_historical_comparison_outputs_remain_exact():
    observed = run()
    result = observed.compare("control", "candidate", "legacy")
    assert result == {"status": "comparable", "control_median": 1., "candidate_median": 1.,
        "paired_control_over_candidate": [1.], "process_pairs": 1, "scope": "target-local",
        "confirmation": "separate finalist cohort required"}
    assert observed.compare("control", "candidate", "missing_old_metric") == {
        "status": "unranked", "reason": "measurement_contract_differs"}
    assert run(complete=False).compare("control", "candidate", SUM)["reason"] == "incomplete_cohort"
