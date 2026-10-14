"""Tier eligibility needs every full-owner peak and one exact matched cohort."""
from copy import deepcopy

import pytest

from benchmarks.research.private_boundary_tiers import METRICS, Rejected, audit, recipes


@pytest.fixture
def cohort():
    cases = [{"name": name, "target": {"parameters": {"model": {"experiment": {"name": name}}}}}
             for name in ("maintained", "tiling", "coupled-boundary")]
    document = {"complete": True, "suite": {"cases": cases, "sampling": {"processes": 3}}, "results": []}
    for index, case in enumerate(cases):
        for replicate in range(3):
            # The candidate's median fits Low; its final complete-owner outlier does not.
            peak = (300 if index == 2 and replicate == 2 else 220) << 20
            metrics = {key: 10 for key in METRICS}
            metrics.update({"client.peak_rss": 180 << 20, "process.peak_rss": peak,
                            "network.all_link_stream_bytes": 120 - 20 * index,
                            "network.covered_body_bytes": None})
            report = {"checks": {"passed": True}, "runs": [{"cold": True,
                "tokens": {"authoritative": True, "input_tokens": 150, "output_tokens": 8},
                "generation": {"output_text_digest": "same-output"}}],
                "configuration": {"source_lock_digest": "same-source", "prompt_digest": "same-cohort",
                    "wan_emulation_digest": "same-wan", "sampling": {"temperature": 0}},
                "client_process_memory": {"scope": "client and in-process dashboard",
                    "after": {"lifetime_peak_rss_bytes": 180 << 20}},
                "memory_guard": {"tripped": False, "maximum_host_swap_growth_bytes": 0},
                "wan_readiness": {"emulation": {"native_stream_rate_snapshots_checked": True}}}
            bindings = [{"failed": False, "numeric_body_fingerprint": "same-body",
                         "selections": [{"checked_values": 100, "logits_and_active_kv": "same-state"} for _ in range(8)]}]
            document["results"].append({"status": "completed", "metrics": [{"id": key, "value": value}
                for key, value in metrics.items()], "execution": {"case": deepcopy(case), "replicate": replicate,
                    "build": "same-build", "details": {"report": report, "response_state_audit": {"bindings": bindings}}}})
    return document


def test_maximum_complete_owner_peak_gates_tier_not_child_or_median(cohort):
    summary = audit(cohort)
    candidate = summary["groups"]["coupled-boundary"]
    assert candidate["medians"]["process.peak_rss"] == 220 << 20
    assert candidate["maximum_client_peak_bytes"] == 300 << 20
    assert candidate["observed_tiers"] == {"low": False, "medium": True, "high": True}
    assert not summary["tiers"]["low"]["comparisons"]["network.all_link_stream_bytes"]["candidate_wins"]
    win = summary["tiers"]["medium"]["comparisons"]["network.all_link_stream_bytes"]
    assert win["incumbents"] == ["tiling"]
    assert win["candidate_wins"] and win["improvement_fraction"] == pytest.approx(0.2)
    assert summary["tiers"]["medium"]["comparisons"]["network.covered_body_bytes"] is None
    assert summary["checked_values"] == 7200


@pytest.mark.parametrize("mutation", ["state", "prompt", "build", "child_only", "missing", "duplicate", "warm", "nan"])
def test_changed_or_incomplete_evidence_cannot_rank(cohort, mutation):
    row = cohort["results"][-1]
    details = row["execution"]["details"]
    if mutation == "state":
        details["response_state_audit"]["bindings"][0]["selections"][-1]["logits_and_active_kv"] = "changed-bit"
    elif mutation == "prompt":
        details["report"]["configuration"]["prompt_digest"] = "other-cohort"
    elif mutation == "build":
        row["execution"]["build"] = "other-build"
    elif mutation == "child_only":
        details["report"]["client_process_memory"]["scope"] = "client subprocess; parent/relay excluded"
    elif mutation == "missing":
        cohort["results"].pop()
    elif mutation == "duplicate":
        row["execution"]["replicate"] = 0
    elif mutation == "warm":
        details["report"]["runs"][0]["cold"] = False
    else:
        next(metric for metric in row["metrics"] if metric["id"] == "client.cold_cpu")["value"] = float("nan")
    with pytest.raises(Rejected):
        audit(cohort)


def test_incumbents_have_common_public_request_bounds(tmp_path):
    experiments = recipes(artifacts=tmp_path)
    assert len(experiments) == 8
    for exp in experiments.values():
        assert exp.budget.max_input_tokens == 150
        assert exp.budget.max_new_tokens == 8
        assert exp.budget.requests == 1
        cache = exp.pipeline.components.get("cache")
        if cache is not None:
            assert cache.get_params()["fixed_input_tokens"] == 150
