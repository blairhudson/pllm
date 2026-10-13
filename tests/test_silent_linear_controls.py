"""Reject forged control accounting before projecting cryptographic budgets."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/probe_silent_linear_controls.py"
SPEC = importlib.util.spec_from_file_location("silent_linear_controls", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture
def source():
    return json.loads(probe.SOURCE.read_text())


def selected(document):
    return next(r for r in document["results"] if r["execution"]["case"]["name"] == "150/both-placement")


def test_measured_controls_conserve_but_do_not_qualify_candidate():
    report = probe.build_report()
    assert len(report["cases"]) == 4
    assert all(c["measured_control_fits_low"] for c in report["cases"])
    assert report["candidate_low_tier_qualified"] is None
    assert report["complete_bytes_per_generated_token"] is None
    row = report["cases"][1]["samples"][0]
    assert row["covered_pair_bytes"] == 471829943
    assert row["correction_bytes"] == 121105112
    assert row["public_delivery_bytes"] == 155464774
    assert row["known_base_bytes"] + row["known_output_dependent_bytes"] + row["unattributed_bytes"] == 471829943
    for sensitivity in row["sensitivity"]:
        for target in (20, 35, 50):
            # Independent horizon inequality, with all additional costs grouped.
            for horizon in (1, 2, 4, 16, 64, 4096):
                common = sensitivity["common_public_bytes"]
                original = common + horizon * sensitivity["control_recurring_pair_bytes"]
                extra = Fraction(sensitivity["helper_plus_amortized_bootstrap_budget_bytes_per_pair"]
                                 [str(horizon)][str(target)]) * horizon
                candidate = original - horizon * row["correction_bytes"] + extra
                assert candidate * 100 == original * (100 - target)
        assert Fraction(sensitivity["correction_only_recurring_ceiling"]) < Fraction(1, 2)
    # First pair is identical at both public reuse endpoints.
    assert row["sensitivity"][0]["helper_plus_amortized_bootstrap_budget_bytes_per_pair"]["1"] == \
        row["sensitivity"][1]["helper_plus_amortized_bootstrap_budget_bytes_per_pair"]["1"]


def test_missing_or_duplicate_replicate_rejects(source):
    duplicate = deepcopy(selected(source))
    source["results"].append(duplicate)
    with pytest.raises(ValueError, match="replicate"):
        probe.reconcile(source)
    source["results"].remove(duplicate)
    source["results"].remove(selected(source))
    with pytest.raises(ValueError, match="replicate"):
        probe.reconcile(source)


def test_conserving_correction_substitution_rejects(source):
    result = selected(source)
    edges = result["execution"]["details"]["report"]["topology_accounting"]["runs"][0]["body_bytes_by_edge"]
    correction = next(e for e in edges if e["source"] == "preparation" and e["destination"] == "inference")
    output = next(e for e in edges if e["source"] == "inference" and e["destination"] == "client" and e["phase"] == "online")
    correction["serialized_body_bytes"] += 1000
    output["serialized_body_bytes"] -= 1000
    with pytest.raises(ValueError, match="category substitution"):
        probe.reconcile(source)


@pytest.mark.parametrize("mutation", ("scope", "partial", "identity", "swap", "duplicate_edge"))
def test_other_unsupported_evidence_rejects(source, mutation):
    result = selected(source)
    execution = result["execution"]
    if mutation == "scope":
        next(m for m in result["metrics"] if m["id"] == "network.covered_body_bytes")["measurement_scope"] = "tcp_stream_payload"
    elif mutation == "partial":
        source["complete"] = False
    elif mutation == "identity":
        execution["comparison"]["numeric_digest"] = "0" * 64
    elif mutation == "swap":
        execution["details"]["report"]["memory_guard"]["maximum_host_swap_growth_bytes"] = 1
    else:
        edges = execution["details"]["report"]["topology_accounting"]["runs"][0]["body_bytes_by_edge"]
        duplicate = deepcopy(edges[0])
        duplicate["serialized_body_bytes"] = 0
        edges.append(duplicate)
    with pytest.raises(ValueError):
        probe.reconcile(source)
