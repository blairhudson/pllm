#!/usr/bin/env python3
"""Validate and summarize the retained five-probe screen without rerunning it."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    require(bool(values) and all(math.isfinite(x) and x > 0 for x in values),
            "positive finite timing samples required")
    return {"median": median(values), "min": min(values), "max": max(values)}


def summarize(archive):
    require(archive["schema"] == "pllm.research.boundary_decode_frontier.v1", "archive schema")
    reports = {}
    for item in archive["files"]:
        name, raw = item["name"], item["utf8"].encode()
        require(name not in reports, "duplicate report")
        require(hashlib.sha256(raw).hexdigest() == item["sha256"], "raw report digest")
        require(item["format"] == ("jsonl" if name == "dotprod.json" else "json"), "report format")
        reports[name] = ([json.loads(line) for line in raw.splitlines()]
                         if item["format"] == "jsonl" else json.loads(raw))
    expected = {"manifest.json", "dotprod.json", "residues.json", "runahead.json",
                *(f"duplex-{i}.json" for i in range(4)),
                *(f"fusion-{rows}-{i}.json" for rows in (1025, 4096) for i in range(3))}
    require(set(reports) == expected, "complete fixed screen required")
    manifest = reports["manifest.json"]
    require(len(manifest) == 13 and {r["stdout"] for r in manifest} == expected - {"manifest.json"},
            "complete measurement manifest required")
    require(all(r["status"] == 0 and r["guard_error"] is None for r in manifest), "failed job")
    require(all(r["maximum_swap_growth_bytes"] == 0 for r in manifest), "swap growth")

    modes = ("serial", "parallel", "striped", "pipelined")
    duplex = [reports[f"duplex-{i}.json"] for i in range(4)]
    require(all(r["provenance"] == duplex[0]["provenance"] for r in duplex), "duplex build identity")
    output_hashes = set()
    for i, report in enumerate(duplex):
        geometry = report["geometry"]
        require((geometry["vocabulary"], geometry["hidden"], geometry["stripe_rows"],
                 geometry["repeats"], geometry["order_offset"]) == (151936, 896, 8192, 4, i),
                "duplex geometry/order")
        require(geometry["download_mbps"] == 100 and geometry["upload_mbps"] == 40, "access caps")
        runs = report["runs"]
        require(len(runs) == 16 and {(r["repetition"], r["mode"]) for r in runs}
                == {(j, mode) for j in range(4) for mode in modes}, "complete duplex pairs")
        for row in runs:
            require(row["ok"] is True and row["error"] is None, "duplex failure")
            require(all(o["retired"] and not o["busy"] for o in row["owners"]), "live owner")
            require(row["wan"]["enforced"] is True, "unenforced transport")
            body = sum(sum(direction.values()) for direction in row["ledger"].values())
            require(body == row["wan"]["total_stream_bytes"]
                    == sum(row["wan"]["directed_stream_bytes"].values()), "body conservation")
            output_hashes.add(row["full_logits_sha256"])
    require(len(output_hashes) == 1 and None not in output_hashes, "full-logit mismatch")
    delivery = {}
    for mode in modes:
        groups = [[r for r in report["runs"] if r["mode"] == mode] for report in duplex]
        sizes = {r["wan"]["total_stream_bytes"] for group in groups for r in group}
        require(len(sizes) == 1, "variable delivered bytes")
        delivery[mode] = {
            "wall_seconds": distribution([median(r["online"]["wall_seconds"] for r in g) for g in groups]),
            "aggregate_fixture_cpu_seconds": distribution([median(r["online"]["cpu_seconds"] for r in g) for g in groups]),
            "application_stream_bytes": sizes.pop(),
        }

    dots = reports["dotprod.json"]
    require(len(dots) == 6, "complete dotprod workers")
    for i, row in enumerate(dots):
        require((row["rows"], row["cols"], row["repeats"]) == (151936, 896, 4), "dotprod geometry")
        require(row["exact"] is True and len(row["samples"]) == 4
                and all(s["exact"] is True for s in row["samples"]), "dotprod exactness")
        require(row["order_offset"] == i % 2, "dotprod counterbalance")
        require(row["provenance"]["binary_sha256"] == dots[0]["provenance"]["binary_sha256"]
                and row["provenance"]["source_sha256"] == dots[0]["provenance"]["source_sha256"]
                and row["provenance"]["native_command"][:-1] == dots[0]["provenance"]["native_command"][:-1],
                "dotprod fixture/build identity")
    byte_dot = {arm: {metric: distribution([median(s[arm][metric] for s in row["samples"])
                                           for row in dots])
                     for metric in ("cpu_seconds", "elapsed_seconds")}
                for arm in ("baseline", "candidate_including_packing")}
    byte_dot["cpu_reduction_percent"] = 100 * (1 - byte_dot["candidate_including_packing"]["cpu_seconds"]["median"]
                                               / byte_dot["baseline"]["cpu_seconds"]["median"])
    byte_dot["scope"] = "API-level: preallocated candidate versus allocating wrap32; packing included"

    fusion = {}
    binaries = set()
    for rows in (1025, 4096):
        records = [reports[f"fusion-{rows}-{i}.json"] for i in range(3)]
        cases = [r["native_fused_reference"] for r in records]
        binaries.update(r["provenance"]["test_binary_sha256"] for r in records)
        require(all(r["correct"] is True and r["fused_scan_executed"] is True
                    and (r["rows"], r["columns"], r["repetitions"]) == (rows, 896, 32)
                    for r in cases), "fusion exactness/geometry")
        ratios = [r["fused_ns_both_parties"] / r["separate_ns_both_parties"] for r in cases]
        require(all(math.isclose(ratio, r["fused_over_separate_elapsed_ratio"])
                    for ratio, r in zip(ratios, cases)), "fusion ratio")
        fusion[str(rows)] = {"worker_aggregate_elapsed_ratios": distribution(ratios),
                             "elapsed_reduction_percent": 100 * (1 - median(ratios))}
    require(len(binaries) == 1, "fusion build identity")

    residues = reports["residues.json"]
    uniform = residues["uniform_geometry_profile"]
    require(all(v is True for v in residues["correctness"].values())
            and uniform["full_float32_logits_byte_equal"] is True
            and uniform["native_vs_python_packing_and_integer_dot"] is True, "residue exactness")
    certificate, costs = uniform["certificate"], uniform["costs"]
    require(certificate["bias"] is False and certificate["uniform_width"] == 25
            and certificate["uniform_bound"] == 128 * 127 * 896
            and (costs["vocabulary"], costs["hidden"], costs["head_calls"]) == (151936, 896, 8),
            "uniform certificate geometry")
    saving = 8 * (costs["raw_u32_two_party_payload_bytes_per_call"]
                  - costs["two_party_framed_reply_bytes_per_call"]) - costs["cold_certificate_layout_delivery_bytes"]
    require(saving == costs["raw_payload_minus_new_frames_and_cold_delivery_bytes"], "residue ledger")
    runahead = reports["runahead.json"]
    equal = [r for r in runahead["modeled"] if r["a_service_ms"] == r["b_service_ms"]]
    require(runahead["exact_full_logits"] is True and equal
            and all(r["gain_vs_ready_parallel_ms"] == 0 for r in equal), "runahead equal-service control")
    return {
        "schema": "pllm.research.boundary_decode_frontier.summary.v1",
        "scope": "generated bounded operators; not SDK speed, privacy, quality or client-tier qualification",
        "duplex_aggregation": "four process medians, each containing four balanced repetitions",
        "duplex": delivery, "byte_dot": byte_dot,
        "fusion_aggregation": "three worker-level ratios of 32-repeat aggregate elapsed times per shape",
        "fusion": fusion, "uniform_residues": {"costs": costs, "cpu": residues["cpu"]},
        "runahead_equal_service_gain_ms": 0,
        "successful_monitored_jobs": len(manifest), "maximum_observed_swap_growth_bytes": 0,
        "whole_client_memory_tier": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.input.open("rb") as source:
        raw = source.read((2 << 20) + 1)
    require(len(raw) <= 2 << 20, "archive exceeds 2 MiB")
    result = summarize(json.loads(raw))
    result["archive_sha256"] = hashlib.sha256(raw).hexdigest()
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
