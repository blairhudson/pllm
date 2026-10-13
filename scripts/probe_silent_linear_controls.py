"""Reconcile measured Low-tier controls for conditional correlation budgets.

No candidate cryptography or model execution. A horizon unit contains one cold
request and its exact repeat. Complete base/per-token rates remain unknown.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/research-sota-combinations-qwen25.json"
CASES = ("150/both", "150/both-placement", "288/both", "288/both-placement")
LOW_BYTES = 256 * 1024 * 1024


def integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a nonnegative byte count")
    if value < 0 or value != int(value):
        raise ValueError("expected a nonnegative byte count")
    return int(value)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def reconcile(document: dict) -> list[dict]:
    require(document.get("schema") == "pllm.benchmark_run.v1"
            and document.get("complete") is True, "incomplete source cohort")
    require(document["suite"]["sampling"]["processes"] == 3, "expected three replicates")
    rows: dict[str, list] = defaultdict(list)
    for result in document["results"]:
        execution = result["execution"]
        case = execution["case"]["name"]
        if case not in CASES:
            continue
        details = execution["details"]
        report = details["report"]
        require(result["status"] == "completed" and report["checks"]["passed"] is True,
                "failed control cannot qualify")
        require(len(report["runs"]) == 2 and not report["warmup_runs"], "wrong request lifecycle")
        require(all(run["status"] == "completed" for run in report["runs"]), "failed request")
        require(report["memory_guard"]["maximum_host_swap_growth_bytes"] == 0,
                "source reports swap growth")
        topology = report["topology_accounting"]
        require(len(topology["runs"]) == 2 and not topology["warmups"], "wrong accounting lifecycle")
        totals: dict[tuple, int] = defaultdict(int)
        body = 0
        for window in [topology["startup"], *topology["runs"]]:
            require(window["body_counter_coverage"] ==
                    "prepared_protocol_lifecycle_controls_and_correction_acks_v3", "wrong body coverage")
            edges = window["body_bytes_by_edge"]
            keys = [(edge["source"], edge["destination"], edge["phase"]) for edge in edges]
            require(len(keys) == len(set(keys)), "duplicate directed edge")
            count = integer(window["all_link_serialized_body_bytes"])
            require(count == sum(integer(edge["serialized_body_bytes"]) for edge in edges),
                    "directed bodies do not conserve")
            body += count
            for key, edge in zip(keys, edges, strict=True):
                totals[key] += integer(edge["serialized_body_bytes"])
        metrics = {metric["id"]: metric for metric in result["metrics"]}
        require(len(metrics) == len(result["metrics"]), "duplicate metric")
        measured = metrics["network.covered_body_bytes"]
        require(measured["measurement_scope"] ==
                "covered_application_bodies:prepared_protocol_lifecycle_controls_and_correction_acks_v3",
                "body/stream scope substitution")
        require(body == integer(measured["value"]) ==
                integer(report["summary"]["total_accounted_benchmark_body_bytes"]), "body total mismatch")
        attribution = details["traffic_decomposition"]["covered_lifecycle"]
        require(attribution["conservation_checked"] is True
                and attribution["attribution_valid"] is True, "unvalidated attribution")
        require(body == attribution["total_all_link_body_bytes"] == sum(integer(attribution[name]) for name in (
            "known_base_all_link_body_bytes", "known_output_dependent_all_link_body_bytes",
            "unattributed_all_link_body_bytes")), "attribution does not conserve")
        correction = totals[("preparation", "inference", "offline")]
        public = totals[("inference", "client", "cold")]
        audit = details["runtime_audit"]["totals"]
        require(correction == audit["correction_push_bytes"] and public == audit["bundle_network_bytes"],
                "counter/category substitution")
        peak = metrics["client.peak_rss"]
        require(peak["window"] == "process_birth_through_scenario" and
                peak["measurement_scope"] == "process_lifetime_high_water", "wrong client memory scope")
        require(0 < public < body and 0 < correction < body - public, "invalid partitions")
        rows[case].append({
            "replicate": integer(execution["replicate"]),
            "result_id": result["id"],
            "build_digest": execution["build"]["digest"],
            "configuration_digest": result["configuration_digest"],
            "comparison": execution["comparison"],
            "client_peak_bytes": integer(peak["value"]),
            "covered_pair_bytes": body,
            "public_delivery_bytes": public,
            "correction_bytes": correction,
            "known_base_bytes": attribution["known_base_all_link_body_bytes"],
            "known_output_dependent_bytes": attribution["known_output_dependent_all_link_body_bytes"],
            "unattributed_bytes": attribution["unattributed_all_link_body_bytes"],
        })
    require(set(rows) == set(CASES), "missing control case")
    output = []
    for case in CASES:
        samples = sorted(rows[case], key=lambda row: row["replicate"])
        require([row["replicate"] for row in samples] == [0, 1, 2], "missing/duplicate replicate")
        for key in ("build_digest", "configuration_digest", "comparison"):
            require(all(row[key] == samples[0][key] for row in samples), "replicate identity mismatch")
        for row in samples:
            row["sensitivity"] = []
            for reuse in (False, True):
                common = row["public_delivery_bytes"] if reuse else 0
                recurring = row["covered_pair_bytes"] - common
                saving = row["correction_bytes"]
                row["sensitivity"].append({
                    "public_reuse_assumption": "all_reused" if reuse else "all_recurs",
                    "common_public_bytes": common,
                    "control_recurring_pair_bytes": recurring,
                    "correction_only_first_pair_ceiling": str(Fraction(saving, row["covered_pair_bytes"])),
                    "correction_only_recurring_ceiling": str(Fraction(saving, recurring)),
                    "helper_plus_amortized_bootstrap_budget_bytes_per_pair": {
                        str(horizon): {
                            str(target): str(Fraction(saving) - Fraction(target, 100) *
                                             (Fraction(common, horizon) + recurring))
                            for target in (20, 35, 50)
                        } for horizon in (1, 2, 4, 16, 64, 4096)
                    },
                })
        peak = max(row["client_peak_bytes"] for row in samples)
        output.append({"case": case, "samples": samples, "maximum_control_client_peak_bytes": peak,
                       "measured_control_fits_low": peak <= LOW_BYTES})
    return output


def build_report() -> dict:
    return {
        "schema": "pllm.silent_linear_control_budgets.v1",
        "complete": True,
        "scope": "historical measured control reconciliation and conditional byte ceilings only",
        "horizon_unit": "one new-prefix eight-output request and its exact repeat",
        "candidate_implemented": False,
        "candidate_client_peak_bytes": None,
        "candidate_low_tier_qualified": None,
        "complete_base_bytes_per_request": None,
        "complete_bytes_per_generated_token": None,
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), SOURCE)},
        "cases": reconcile(json.loads(SOURCE.read_text())),
        "limitations": [
            "Public reuse endpoints are assumptions; the aggregate includes unclassified repeated metadata.",
            "The hypothetical saving removes correction bodies only; ACKs and other controls stay charged.",
            "Helper budgets jointly bound extra setup, renewal, frames, failure and recurring traffic.",
            "Missing provider/checkpoint distribution and physical wire are not priced by these bodies.",
            "Observed control eligibility does not qualify a candidate or certify runtime admission.",
            "Separate workloads and source cohorts are not pooled or ranked.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "control_cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
