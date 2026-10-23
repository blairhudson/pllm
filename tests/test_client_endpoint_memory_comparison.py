"""Independent synthetic ordinary reports: no native work, child, model or sampler.

Use real adapters/serialization/BenchmarkRun comparison. Cached owner and full-state
commitments below are explicit test doubles, not measured workload/exit evidence.
"""
import hashlib
import os
import subprocess
import threading
from types import SimpleNamespace as NS

import pytest

import pllm
from pllm._benchmark_contract import canonical, digest
from pllm._benchmark_evidence import client_memory_metric_id, measurement
from pllm._benchmark_worker import Measurement
from pllm.evidence import BenchmarkResult, EvidenceRegistry
from pllm.metrics import BenchmarkCase, BenchmarkRun, BenchmarkSuite, BenchmarkTarget, MetricSpec, Sampling, Scenario
from pllm.metrics.benchmark_runner import _result
from pllm.profiles import MaskedLinearCpu
from pllm.runtime import client_process_usage as usage
from pllm.runtime.benchmark_cli import build_loopback_report
from pllm.runtime.benchmark_response_target import _expected_token_count_checks, response_metrics
from pllm.runtime.client_memory_scope import ClientMemoryScope, client_lifetime_manifest
from pllm.runtime.servers import LocalTopology


ID = "client.peak_rss"
QUALIFICATION = "root_only_report_endpoint"
WINDOW = "client_process_birth_through_report_endpoint"
SELECT = MetricSpec(ID, qualification=QUALIFICATION)


@pytest.fixture(autouse=True)
def no_children(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("endpoint comparison controls may not launch children or sample OS birth")
    monkeypatch.setattr(usage, "_birth_identity", forbidden)
    monkeypatch.setattr(usage, "supported", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def report(value, salt, *, input_tokens=3, output_tokens=2):
    scope = ClientMemoryScope()
    scope.claim_sampling()
    scope.observe({"client": (os.getpid(), 1., value // 2)}, span_seconds=0)
    scope.seal_lifetime_membership()  # Actual producer declares after sampler start.
    scope.observe({"client": (os.getpid(), 1., value // 2)}, span_seconds=0)
    row = {"status": "completed", "cold": True, "warm": False,
        "tokens": {"authoritative": True, "input_tokens": input_tokens, "output_tokens": output_tokens},
        "model_fingerprint": "b" * 64, "max_output_tokens": output_tokens,
        "generation": {"response_status": "completed", "output_text_digest": "c" * 64},
        "durations": {"full_seconds": .5, "online_seconds": .3, "ttft_seconds": .1,
                      "tokens_per_second": 5, "preparation_seconds": .2},
        "privacy": {"plaintext_prompt_bytes_sent": 0, "plaintext_token_ids_sent": 0,
                    "preparation_requests_during_online": 0}}
    observed = build_loopback_report(model_id="synthetic-owner-endpoint", tiny=False,
        max_output_tokens=output_tokens, warmup_runs=[], runs=[row], source_lock_digest="a" * 64,
        prompt_digest=hashlib.sha256(b"pllm.benchmark.prompt.v1\0" + salt + b"private-input").hexdigest(),
        capture_output_digest=True)
    topology = LocalTopology.__new__(LocalTopology)
    topology._process_lock = threading.Lock()
    topology._started = topology._closed = True
    topology._model = pllm.Model.tiny()
    topology._role_ids = ("preparation", "inference")
    topology._processes = {"preparation": NS(pid=201), "inference": NS(pid=301)}
    observed["client_process_memory"] = {"after": {"lifetime_peak_rss_bytes": value}}
    observed["client_owned_memory"] = scope.report(value)
    observed["client_owned_memory"]["lifetime_owner"] = client_lifetime_manifest(
        NS(client_memory_scope=scope), topology, ("client", *topology._role_ids), local_cpu_loopback=True)
    owner = usage.ClientUsagePopen.__new__(usage.ClientUsagePopen)
    owner._child_created = False
    owner.pid, owner.returncode = os.getpid(), 0
    owner._usage_identity = (os.getpid(), 1.)
    owner._waitpid_lock = threading.Lock()
    owner._usage_scale, owner._usage_unavailable = 1, False
    # Excluded provider maxima inflate the FAMILY upper, not the endpoint point.
    owner._usage_final = (.01, .02, 1 << 30)
    usage.finalize_client_report(observed, owner)
    return observed


def run(*, change=None, other_salt=False, audited=True, input_tokens=3, output_tokens=2):
    experiment = pllm.Experiment("endpoint-test-double", MaskedLinearCpu(pllm.Model.tiny()),
        pllm.Deployment.local(root="target/endpoint-test-double"), pllm.ExecutionBudget(1, 128, output_tokens))
    target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=output_tokens),
        state_audit=audited).with_params(expected_input_tokens=input_tokens, expected_output_tokens=output_tokens)
    sampling = Sampling(processes=1)
    cases = tuple(BenchmarkCase(name, target) for name in ("control", "candidate"))
    suite = BenchmarkSuite(cases, sampling)
    results = []
    for index, case in enumerate(cases):
        salt = ("cd" if index and other_salt else "ab") * 32
        m = Measurement({"sampling": sampling.to_spec(), "salt": salt,
            "inputs": {"prompt": "private-input"}, "directory": "."})
        observed = report(1000 if index == 0 else 10, m.salt,
            input_tokens=input_tokens, output_tokens=output_tokens)
        assert _expected_token_count_checks(target.to_spec()["parameters"], observed)
        response_metrics(observed, m)
        binding = {"sampler_seed": 42, "temperature": 0, "numeric_body_fingerprint": "b" * 64,
            "cryptographic_randomness_frozen": False, "failed": False,
            "selections": [{"position": position, "cache_lengths": [position], "checked_values": 20,
                "logits_and_active_kv": "d" * 64, "selected_token": "e" * 64,
                "rng_before": "f" * 64, "rng_after": "a" * 64}
                for position in range(input_tokens, input_tokens + output_tokens)]}
        m.details.update(report=observed, response_state_audit={"schema": "pllm.response_state_audit.v1",
            "bindings": [binding]})
        m.metrics.append(measurement("legacy", 1))
        m.samples.append({"index": 0, "kind": "measurement", "status": "completed",
            "duration_seconds": .5, "artifact_digest": None})
        m.fixture = {"kind": "generated_checkpoint", "digest": "a" * 64, "metadata": {}}
        m.oracle = {"kind": "salted_full_logit_kv_sampler_commitments", "outcome": "passed",
            "checked_values": 20 * output_tokens}
        m.admission.update(admitted=True, required_bytes=1, host_headroom_bytes=1 << 30)
        m.comparison.update({key: "a" * 64 if key == "source_digest" else "b" * 64 for key in m.comparison})
        m.finalize_workload(target)
        attributes = {"fixture": "mock-ordinary-report-not-a-build-measurement"}
        result = _result(case, suite, 0, "d" * 64, {"attributes": attributes, "digest": digest(attributes)},
            {"status": "completed", "failure_code": None, "metrics": m.metrics, "samples": m.samples,
             "details": m.details, "fixture": m.fixture, "oracle": m.oracle,
             "comparison": m.comparison, "admission": m.admission}).to_dict()
        if change is not None and index:
            change(result)
        results.append(BenchmarkResult.from_dict(result).to_dict())
    return BenchmarkRun({"schema": "pllm.benchmark_run.v1", "suite": suite.to_spec(),
        "cohort": "d" * 64, "complete": True, "results": results})


def scope(row):
    return row["execution"]["details"]["report"]["client_owned_memory"]


def test_public_qualified_existing_id_serialization_and_ordinary_api():
    assert client_memory_metric_id(ID) is None  # Never intercept the legacy string.
    assert client_memory_metric_id(SELECT) == ID
    assert SELECT.to_spec() == {"id": ID, "qualification": QUALIFICATION, "unit": "bytes",
        "kind": "measured", "window": WINDOW,
        "source_setup_costs": "only_inside_isolated_birth_through_report_endpoint", "physical_peak": False}
    observed = run()
    before = observed.to_dict()
    compared = observed.compare("control", "candidate", SELECT)
    assert compared["status"] == "comparable_in_scope" and compared["window"] == WINDOW
    assert compared["control_median"] == 1000 and compared["candidate_median"] == 10
    assert compared["paired_control_over_candidate"] == [100]
    assert compared["reduction_status"] == "not_established"
    assert not compared["physical_peak"] and not compared["whole_lifetime"]
    assert "unknown" in compared["outside_isolated_role"]
    registry = EvidenceRegistry(observed.results)
    assert registry.query(metric=SELECT) == registry.list()
    assert observed.to_dict() == before and b"private-input" not in canonical(before)
    root = before["results"][0]["execution"]["details"]["report"]["client_owned_memory"]
    assert root["sampling_coverage"] == "unknown_or_partial"  # Not a sampled-SUM promotion.
    assert root["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] == 1 << 30


@pytest.mark.parametrize("change", [
    lambda row: scope(row)["lifetime_owner"].update(membership_complete=False),
    lambda row: scope(row)["lifetime_owner"].update(root_birth_bound=False),
    lambda row: scope(row)["lifetime_owner"].update(reason="unretired_topology"),
    lambda row: scope(row)["lifetime_owner"].update(window=WINDOW),
    lambda row: scope(row)["lifetime_owner"].update(classification="foreign_source"),
    lambda row: scope(row)["lifetime_owner"].update(_root_identity=[101, 1]),
    lambda row: scope(row)["lifetime_owner"]["members"].update(client_helper_1={"coverage": "unknown"}),
    lambda row: scope(row)["lifetime_owner"]["members"].update(preparation={}),
    lambda row: scope(row)["lifetime_owner"].update(outside_isolated_role="complete"),
    lambda row: scope(row).update(membership_declared=False),
    lambda row: scope(row)["latest"]["members"].update(client_helper_1={"reason": "missing"}),
    lambda row: scope(row)["latest"]["members"]["client"].update(reason="member_identity_changed"),
    lambda row: scope(row)["latest"].update(missing_members=["undeclared_client_helpers"]),
    lambda row: scope(row)["individual_client_hwm"].update(bytes=9),
    lambda row: scope(row)["individual_client_hwm"].update(bytes=None),
    lambda row: scope(row)["individual_client_hwm"].update(source="wait4_family_maximum"),
    lambda row: scope(row)["individual_client_hwm"].update(source="sampled_RSS_SUM"),
    lambda row: scope(row)["individual_client_hwm"].update(window="process_birth_through_exit"),
    lambda row: scope(row)["individual_client_hwm"].update(coverage="client_group"),
    lambda row: scope(row)["individual_client_hwm"].update(isolated_lifetime=False),
    lambda row: row["execution"]["details"]["report"]["client_process_memory"]["after"].update(lifetime_peak_rss_bytes=1),
    lambda row: row["execution"]["details"]["report"]["checks"].update(expected_output_tokens=False),
    lambda row: row["execution"]["details"]["report"]["runs"][0]["tokens"].update(output_tokens=1),
    lambda row: row["execution"]["details"]["report"]["runs"][0]["tokens"].update(authoritative=False),
    lambda row: row["execution"]["details"]["report"]["runs"][0]["generation"].update(response_status="cancelled"),
    lambda row: row["execution"]["details"]["report"]["configuration"].update(source_lock_digest="e" * 64),
    lambda row: row["execution"]["details"]["report"]["runs"][0].update(model_fingerprint="e" * 64),
    lambda row: row["execution"]["details"]["response_state_audit"]["bindings"][0].update(failed=True),
    lambda row: row["execution"]["details"]["response_state_audit"]["bindings"][0].update(selections=[]),
    lambda row: row["execution"]["details"]["response_state_audit"]["bindings"][0].update(selections=[{}, {}]),
    lambda row: row["execution"]["details"]["response_state_audit"]["bindings"][0].update(cryptographic_randomness_frozen=True),
    lambda row: row["execution"]["details"]["response_state_audit"]["bindings"][0]["selections"][0].update(rng_before=None),
    lambda row: row["execution"]["oracle"].update(checked_values=1),
    lambda row: row["execution"]["details"]["response_state_audit"]["bindings"][0]["selections"][0].update(logits_and_active_kv="f" * 64),
    lambda row: row["execution"]["details"].pop("response_state_audit"),
    lambda row: row["execution"]["isolation"].update(fresh_process=False),
    lambda row: row["execution"]["isolation"].update(controller_included=True),
    lambda row: next(m for m in row["metrics"] if m["id"] == ID).update(window=WINDOW),
    lambda row: next(m for m in row["metrics"] if m["id"] == ID).update(measurement_scope="whole_client_lifetime"),
    lambda row: next(m for m in row["metrics"] if m["id"] == ID).update(value=1),
    lambda row: row["metrics"].__setitem__(slice(None), [m for m in row["metrics"] if m["id"] != ID]),
])
def test_missing_or_conflicting_evidence_stays_unranked_without_legacy_fallback(change):
    observed = run(change=change)
    assert observed.compare("control", "candidate", SELECT)["status"] == "unranked"
    assert all(row["status"] == "completed" for row in observed.to_dict()["results"])


@pytest.mark.parametrize("key", ["numeric_digest", "state_digest", "output_digest", "measurement_digest"])
def test_original_comparison_guards_are_not_bypassed(key):
    observed = run(change=lambda row: row["execution"]["comparison"].update({key: "e" * 64}))
    assert observed.compare("control", "candidate", SELECT)["reason"] == "cohort_contract_differs"


def test_salt_and_missing_full_state_audit_cannot_qualify():
    assert run(other_salt=True).compare("control", "candidate", SELECT)["status"] == "unranked"
    assert run(audited=False).compare("control", "candidate", SELECT)["status"] == "unranked"


def test_legacy_identity_defaults_and_comparison_remain_exact_even_when_qualification_fails():
    old = MetricSpec("client.owned_sampled_peak_rss_sum")
    assert old.to_spec() == MetricSpec(old.id, qualification=None).to_spec()
    assert "qualification" not in old.to_spec()
    observed = run(change=lambda row: scope(row)["lifetime_owner"].update(root_birth_bound=False))
    identity = canonical(observed.to_dict()["suite"])
    legacy = observed.compare("control", "candidate", ID)
    assert legacy == {"status": "comparable", "control_median": 1000., "candidate_median": 10.,
        "paired_control_over_candidate": [100.], "process_pairs": 1, "scope": "target-local",
        "confirmation": "separate finalist cohort required"}
    assert observed.compare("control", "candidate", SELECT)["status"] == "unranked"
    assert canonical(observed.to_dict()["suite"]) == identity
    assert {m["window"] for row in observed.to_dict()["results"] for m in row["metrics"] if m["id"] == ID} == {
        "process_birth_through_scenario"}


def test_qualified_selection_cannot_be_stripped_to_unguarded_pareto_point():
    from pllm.search import ParetoFrontier, SearchEvaluation
    evaluations = tuple(SearchEvaluation(NS(), row) for row in run().results)
    frontier = ParetoFrontier(evaluations, objectives={SELECT: "min"})
    assert not frontier.results() and frontier.excluded() == evaluations
    assert ParetoFrontier(evaluations, objectives={ID: "min"}).results()


@pytest.mark.parametrize("id,qualification", [(ID, None), (ID, "whole_lifetime"),
    ("client.owned_peak_rss_lower_bound", QUALIFICATION)])
def test_unsupported_qualifications_fail_closed(id, qualification):
    with pytest.raises(ValueError):
        MetricSpec(id, qualification=qualification)
