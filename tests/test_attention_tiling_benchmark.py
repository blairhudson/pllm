"""Evidence gates reject missing, mismatched or incompletely sampled responses."""
import copy

import pytest

from benchmarks.research.attention_tiling import audit
from benchmarks.research.sota_combinations import Rejected


class Run:
    def __init__(self):
        names = [f"tiny/tiling/{arm}" for arm in ("full", "tiled")]
        rows = []
        for name in names:
            for replicate in range(2):
                request = {"tokens": {"authoritative": True, "output_tokens": 2},
                    "generation": {"output_text_digest": "a" * 64}}
                report = {"checks": {"passed": True}, "runs": [dict(copy.deepcopy(request),
                    cold=warm == 0, privacy={"prefill_cache_hits": warm,
                        "prefill_cache_misses": 1 - warm}) for warm in (0, 1)],
                    "memory_guard": {"maximum_host_swap_growth_bytes": 0, "tripped": False},
                    "topology_accounting": {"startup": {"all_link_serialized_body_bytes": 0,
                        "lifecycle_control_serialized_body_bytes": 0, "control_body_counter_set_present": True},
                        "warmups": [], "runs": [{"all_link_serialized_body_bytes": 500,
                            "lifecycle_control_serialized_body_bytes": 10,
                            "control_body_counter_set_present": True} for _ in (0, 1)]}}
                enabled = name.endswith("/tiled")
                rows.append({"status": "completed", "metrics": [
                    {"id": "client.peak_rss", "value": 260 << 20 if replicate else 200 << 20},
                    {"id": "network.covered_body_bytes", "value": 1000}],
                    "execution": {"case": {"name": name}, "replicate": replicate,
                        "details": {"report": report, "client_attention_research": {
                            "selection": {"enabled": enabled, "query_rows": 16, "kv_heads": 2},
                            "bindings": [{"failed": False, "eligible_regions": 1,
                                "tiles": int(enabled), "phases": [{"position": 10,
                                    "checked_values": 1024, "state_commitment": "b" * 64}]}]}}}})
        self.document = {"complete": True, "suite": {"cases": [{"name": name} for name in names],
            "sampling": {"processes": 2}}, "results": rows}

    def to_dict(self):
        return self.document

    def compare(self, *_):
        return {"status": "comparable", "process_pairs": 2}


def test_largest_replicate_gates_observed_tier():
    result = audit(Run(), tiny=True)
    assert result["completed_clients"] == 4
    assert result["checked_values"] == 4096
    assert result["maximum_observed_swap_growth_bytes"] == 0
    assert all(value["medium"] and not value["low"] for value in result["observed_tiers"].values())


def test_variable_control_bodies_remain_charged_and_visible():
    run = Run()
    row = run.document["results"][-1]
    row["metrics"][1]["value"] += 1
    ledger = row["execution"]["details"]["report"]["topology_accounting"]["runs"][0]
    ledger["all_link_serialized_body_bytes"] += 1
    ledger["lifecycle_control_serialized_body_bytes"] += 1
    result = audit(run, tiny=True)
    assert result["noncontrol_application_bodies_match"] is True
    assert result["all_covered_application_bodies_match"] is False
    assert result["covered_body_ranges"]["tiny/tiling"] == {"minimum": 1000, "maximum": 1001}
    ledger["lifecycle_control_serialized_body_bytes"] -= 1
    with pytest.raises(Rejected, match="application_bodies_differ"):
        audit(run, tiny=True)


@pytest.mark.parametrize("mutation,code", [
    ("incomplete", "incomplete_or_failed_cohort"),
    ("missing", "incomplete_or_failed_cohort"),
    ("duplicate", "duplicate_or_unknown_cell"),
    ("state", "full_logits_or_active_kv_differ"),
    ("output", "matched_outputs_differ"),
    ("bodies", "peak_missing_or_application_bodies_differ"),
    ("swap", "swap_observation_missing"),
    ("tile", "response_or_attention_checks_failed"),
    ("cache", "cold_repeat_cache_contract_mismatch"),
])
def test_ineligible_evidence_rejects(mutation, code):
    run = Run()
    doc = run.document
    row = doc["results"][-1]
    details = row["execution"]["details"]
    if mutation == "incomplete":
        doc["complete"] = False
    elif mutation == "missing":
        doc["results"].pop()
    elif mutation == "duplicate":
        row["execution"]["replicate"] = 0
    elif mutation == "state":
        details["client_attention_research"]["bindings"][0]["phases"][0]["state_commitment"] = "c" * 64
    elif mutation == "output":
        details["report"]["runs"][0]["generation"]["output_text_digest"] = "c" * 64
    elif mutation == "bodies":
        row["metrics"][1]["value"] += 1
    elif mutation == "swap":
        del details["report"]["memory_guard"]["maximum_host_swap_growth_bytes"]
    elif mutation == "tile":
        details["client_attention_research"]["bindings"][0]["tiles"] = 0
    elif mutation == "cache":
        details["report"]["runs"][1]["privacy"]["prefill_cache_hits"] = 0
    with pytest.raises(Rejected, match=code):
        audit(run, tiny=True)
