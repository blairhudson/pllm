"""Read-only validation and summaries of matched private-boundary SDK cohorts.

Keep the original driver, source-present lifecycle and subprocess ownership.
This neither reruns a model nor converts historical reports into BenchmarkResults.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_private_boundary import (
    DRIVER_IMPLEMENTATION, matched_experiments, replicate_order, verify_results,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    return type(value) is int and value >= 0


def summarize(document, source_sha256, control, control_sha256):
    """Recheck raw parity, accounting, admission and the frozen control recipe."""
    from pllm import Experiment

    require(document.get("schema") == "pllm.private_boundary_sdk_benchmark.v1"
        and document.get("complete") is True and document.get("tiny") is False
        and document.get("scope") == "matched-sdk-response"
        and document.get("source_case") == "source-present"
        and document.get("wan_requested") is True, "complete source-present checkpoint cohort required")
    require(document.get("driver_implementation") == DRIVER_IMPLEMENTATION,
            "original measurement implementation changed")
    for value in (source_sha256, control_sha256, document.get("cohort_commitment"),
                  document.get("source_lock_digest"), document.get("build", {}).get("expected_digest")):
        require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                "missing source, cohort or build commitment")
    require(document.get("sampler_fixture") == {"seed": 42, "temperature": 0,
        "cryptographic_randomness_frozen": False, "diagnostic_cost_charged_equally": True},
        "sampler or equally charged diagnostic contract changed")
    rows = document.get("results", [])
    inputs = {row.get("input_tokens") for row in rows}
    require(len(inputs) == 1, "one independently retained workload required")
    tokens, = inputs
    replicates = document.get("replicates")
    order = replicate_order(replicates)
    require(document.get("planned_cells") == len(order) == len(rows)
        and [(row["replicate"], row["variant"]) for row in rows] == order
        and document.get("attempts") == [dict(variant=variant, replicate=replicate, status="executed")
                                        for replicate, variant in order], "incomplete or reordered paired cohort")
    verified = verify_results(rows, replicates=replicates, input_tokens=(tokens,), wan=True,
                              source_lock_digest=document["source_lock_digest"])
    require(document.get("verification") == verified, "cached verification differs from raw evidence")
    swap = document.get("swap", {})
    require(all(count(swap.get(key)) for key in ("initial_bytes", "maximum_growth_bytes", "minimum_available_bytes"))
        and swap["maximum_growth_bytes"] == 0, "zero-swap completed cohort required")
    experiments = matched_experiments(Experiment.from_spec(control), tokens)
    admissions = document.get("memory_admission", {})
    require(set(admissions) == {f"{tokens}/{variant}" for variant in experiments}, "missing admission arms")
    for variant, experiment in experiments.items():
        admitted = admissions[f"{tokens}/{variant}"]
        topology = admitted["topology"]
        estimate = topology["estimate"]
        require(admitted["admitted"] is True and topology["admitted"] is True
            and topology["selected_backend"] == "native" and topology["enforced_wan"] is True
            and estimate["pipeline_digest"] == experiment.pipeline.digest()
            and estimate["max_input_tokens"] == tokens and estimate["max_output_tokens"] == 8,
            "frozen Experiment differs from measured native admission")
    body_counters = ("inference_upload_bytes", "inference_download_bytes", "correction_push_bytes",
                     "correction_ack_bytes", "prepared_stage_rows_issued")
    body = {key: rows[0]["client"]["audit"][key] for key in body_counters}
    for row in rows:
        require({key: row["client"]["audit"][key] for key in body_counters} == body,
                "ordinary prepared arithmetic bodies changed")
        wan = row["wan"]
        roles = {"client", "inference", "preparation"}
        links = {(a, b) for a in roles for b in roles if a != b}
        if row["variant"] == "coupled":
            roles |= {"boundary_a", "boundary_b"}
            links |= {(a, b) for role in ("boundary_a", "boundary_b")
                      for a, b in (("client", role), (role, "client"))}
        directed = wan["directed_stream_bytes"]
        require(wan["schema"] == "pllm.native_wan.v1" and wan["backend"] == "native-shared-tcp-pacer"
            and wan["conditions"] == {"schema": "pllm.wan_conditions.v1", "download_mbps": 100.0,
                                     "upload_mbps": 40.0, "parties": [], "role_parties": []}
            and wan["role_parties"] == {role: role for role in roles}
            and set(directed) == {f"{a}->{b}" for a, b in links}
            and all(count(value) for value in directed.values())
            and wan["client_stream_bytes"] == sum(value for edge, value in directed.items()
                                                   if "client" in edge.split("->")),
            "directed client/whole-graph WAN coverage changed")
    groups = {}
    for variant, experiment in experiments.items():
        samples = [row for row in rows if row["variant"] == variant]
        observations = []
        for row in samples:
            client, response = row["client"], row["client"]["responses"][0]
            observations.append({
                "request_seconds": response["request_seconds"],
                "first_text_seconds": response["first_text_seconds"],
                "text_delta_decode_tps": response["decode_tps"],
                "complete_case_seconds": row["complete_case_seconds"],
                "client_request_cpu_seconds": response["client_request_cpu_seconds"],
                "client_lifetime_cpu_seconds": client["client_lifetime_cpu_seconds"],
                "aggregate_cleanup_cpu_seconds": row["aggregate_cleanup_cpu_seconds"],
                "preparation_birth_cpu_seconds": row["provider_birth_cpu_seconds"]["preparation"],
                "client_subprocess_peak_rss_bytes": client["client_peak_rss_bytes"],
                "preparation_sampled_peak_rss_bytes": row["provider_sampled_peak_rss_bytes"]["preparation"],
                "inference_sampled_peak_rss_bytes": row["provider_sampled_peak_rss_bytes"]["inference"],
                "provider_sampled_peak_rss_bytes": max(row["provider_sampled_peak_rss_bytes"].values()),
                "all_link_tcp_stream_bytes": row["wan"]["total_stream_bytes"],
                "client_tcp_stream_bytes": row["wan"]["client_stream_bytes"],
                "covered_body_bytes": client["ledger"]["all_link_serialized_body_bytes"],
                "online_body_bytes": client["ledger"]["online_all_link_serialized_body_bytes"],
                "bundle_delivery_bytes": client["audit"]["bundle_network_bytes"],
            })
        groups[variant] = {"experiment": experiment.to_spec(), "configuration_digest": experiment.configuration_digest(),
            "pipeline_digest": experiment.pipeline.digest(), "sample_count": len(samples),
            "medians": {key: median(row[key] for row in observations) for key in observations[0]},
            "maximum_client_subprocess_peak_rss_bytes": max(row["client"]["client_peak_rss_bytes"] for row in samples),
            "whole_client_peak_rss_bytes": None,
            "whole_client_memory_tier_qualified": False}
    first = rows[0]["client"]["responses"][0]
    return {"schema": "pllm.private_boundary_sdk_summary.v1",
        "source_sha256": source_sha256, "control_sha256": control_sha256,
        "summarizer_sha256": sha(Path(__file__).read_bytes()),
        "driver_implementation": deepcopy(document["driver_implementation"]),
        "build": deepcopy(document["build"]), "cohort_commitment": document["cohort_commitment"],
        "source_lock_digest": document["source_lock_digest"],
        "body_fingerprint": first["body_fingerprint"], "output_commitment": first["state"]["output"],
        "input_tokens": tokens, "output_tokens": 8, "output_cap": 8, "replicates": replicates,
        "wan_digest": rows[0]["wan"]["conditions_digest"], "source_case": "source-present",
        "evidence_class": "matched-source-present-sdk-screen", "verification": verified,
        "maximum_swap_growth_bytes": swap["maximum_growth_bytes"], "groups": groups,
        "limits": [*document["limits"],
            "client-subprocess RSS excludes parent/relay; no whole-client memory-tier qualification",
            "first-text delay includes request setup; decode uses eight observed text events for eight outputs",
            "independent workload cohorts and different role topologies retain separate rankings"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw, control = args.report.read_bytes(), args.control.read_bytes()
    result = summarize(json.loads(raw), sha(raw), json.loads(control), sha(control))
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
