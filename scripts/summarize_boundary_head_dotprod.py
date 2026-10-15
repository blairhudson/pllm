#!/usr/bin/env python3
"""Validate/replay one frozen v3 allocating-output cohort; never run native code."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "crates/pllm-core/examples/boundary_head_dotprod.rs",
    "crates/pllm-core/src/kernels.rs", "scripts/probe_boundary_head_dotprod.py",
    "tests/test_boundary_head_dotprod_probe.py", "docs/research/boundary-head-dotprod-probe.md",
    "Cargo.lock", "Cargo.toml", "crates/pllm-core/Cargo.toml",
}
CLOCKS = ("cpu_seconds", "elapsed_seconds")
MATCHED = ("packing", "candidate_allocating", "candidate_allocating_including_packing", "baseline")
HISTORICAL = ("packing", "kernel", "candidate_including_packing", "baseline")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, lower, upper, message):
    require(type(value) in (int, float) and math.isfinite(value)
            and lower <= value <= upper, message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "SHA256 syntax")


def close(a, b):
    require(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), "phase timing sum")


def phases(row, keys, limit):
    for key in keys:
        require(set(row[key]) == set(CLOCKS), "timing clocks")
        for clock in CLOCKS:
            number(row[key][clock], 0, limit, "bounded finite timing")
        require(row[key]["elapsed_seconds"] > 0, "positive elapsed timing")
    for clock in CLOCKS:
        close(row[keys[2]][clock], row[keys[0]][clock] + row[keys[1]][clock])


def distribution(values):
    return {"median": median(values), "min": min(values), "max": max(values)}


def aggregation(workers, totals=False):
    arms = {}
    for arm in MATCHED:
        arms[arm] = {clock: distribution([
            row["allocation_matched"][arm][clock] if totals else
            median(s["allocation_matched"][arm][clock] for s in row["samples"])
            for row in workers]) for clock in CLOCKS}
    candidate = arms["candidate_allocating_including_packing"]["cpu_seconds"]["median"]
    baseline = arms["baseline"]["cpu_seconds"]["median"]
    require(candidate > 0 and baseline > 0, "positive aggregate CPU")
    return {"aggregation": "median of six worker four-call totals" if totals else
            "median of six worker medians of four per-repeat samples",
            "timings": arms, "cpu_reduction_percent": 100 * (1 - candidate / baseline)}


def summarize(archive):
    require(archive["schema"] == "pllm.research.boundary_head_dotprod.archive.v3", "archive schema")
    files = {}
    for item in archive["files"]:
        name = item["name"]
        require(name not in files, "duplicate raw file")
        raw = item["utf8"].encode()
        digest(item["sha256"])
        require(hashlib.sha256(raw).hexdigest() == item["sha256"], "raw file digest")
        require(item["format"] == {"results.jsonl": "jsonl", "guard.json": "json",
                                   "run.log": "text"}.get(name), "raw file format")
        files[name] = item
    require(set(files) == {"results.jsonl", "guard.json", "run.log"}, "complete raw files")
    workers = [json.loads(s) for s in files["results.jsonl"]["utf8"].splitlines()]
    guard = json.loads(files["guard.json"]["utf8"])
    require(len(workers) == 6, "six workers required")
    require(guard["exit_code"] == 0 and guard["guard_error"] is None, "successful guard")
    require(guard["maximum_swap_growth_bytes"] == 0, "swap growth")
    host = guard["host"]
    for key in ("total", "available", "swap_used", "disk_free", "reserve_bytes"):
        number(host[key], 0, 1 << 50, "bounded host resource")
    allowance = guard["preflight_allowance_bytes"]
    number(allowance, 512 << 20, 4 << 30, "preflight allowance")
    require(host["reserve_bytes"] >= 4 << 30 and
            host["available"] >= host["reserve_bytes"] + allowance and
            host["disk_free"] >= allowance, "preflight reserve")
    number(guard["minimum_available_bytes"], host["reserve_bytes"], host["total"], "guard reserve")
    command = guard["command"]
    require(len(command) == 13 and command[1:] == [
        "scripts/probe_boundary_head_dotprod.py", "--build", "--rows", "151936",
        "--cols", "896", "--repeats", "4", "--processes", "6", "--order-offset", "0"],
        "guard command")
    # Resolve only retained hashes, never today's mutable sources or executable.
    sources = archive["frozen_source_sha256"]
    require(set(sources) == SOURCES, "complete frozen sources")
    for value in sources.values():
        digest(value)
    first = workers[0]
    for i, row in enumerate(workers):
        require(row["schema"] == "boundary-head-dotprod-v3", "native schema")
        require((row["rows"], row["cols"], row["repeats"]) == (151936, 896, 4), "geometry")
        require(row["exact"] is True and row["allocation_matched"]["exact"] is True, "full exactness")
        require(row["order_offset"] == i % 2 and row["process_schedule"] == {
            "index": i, "processes": 6, "effective_order_offset": i % 2,
            "paired_start_balance": True, "paired_cohort_start_balance": True}, "worker counterbalance")
        provenance = row["provenance"]
        digest(provenance["binary_sha256"])
        require(provenance["source_sha256"] == sources and
                provenance["binary_sha256"] == first["provenance"]["binary_sha256"] and
                provenance["binary_path"] == first["provenance"]["binary_path"], "same build/source hashes")
        require(provenance["native_command"] == [provenance["binary_path"], "151936", "896",
                "4", "1234567", "auto", "quiet", str(i % 2)], "native command")
        require(row["compiled_source_sha256"] == sources["crates/pllm-core/examples/boundary_head_dotprod.rs"]
                and row["compiled_kernels_sha256"] == sources["crates/pllm-core/src/kernels.rs"],
                "compiled source locks")
        build = row["build_provenance"]
        require(build == first["build_provenance"] and build["profile"] == "release" and
                build["command"] == ["cargo", "build", "--locked", "-p", "pllm-core", "--release",
                                     "--example", "boundary_head_dotprod"] and
                all(build[k] for k in ("rustc_verbose", "cargo_verbose", "built_at_utc", "rustc_path")),
                "native build provenance")
        require(row["selected"] == "arm-sdot-inline-asm" and row["dotprod_available"] is True
                and row["baseline_simd"] is True and row["arch"] == "aarch64", "feature selection")
        require("preallocated" in row["comparison"] and
                "both APIs allocate output" in row["allocation_matched_comparison"], "comparison scope")
        require(all(row[k] is None for k in ("weights", "input", "output", "baseline_output")), "quiet report")
        buffers = row["owned_buffers"]
        require(buffers == first["owned_buffers"] and buffers["weights_bytes"] == 151936 * 896
                and buffers["extra_persistent_weight_bytes"] == 0 and
                buffers["matched_candidate_call_output_bytes"] == 4 * 151936, "owned buffers")
        fresh = row["fresh_process"]
        number(fresh["peak_rss_bytes"], buffers["weights_bytes"], 512 << 20, "bounded process RSS")
        for clock in ("user_cpu_seconds", "system_cpu_seconds"):
            number(fresh[clock], 0, 120, "bounded process CPU")
        require(all(fresh[k] == first["fresh_process"][k] for k in
                    ("machine", "platform", "cpu_brand", "native_runtime_features")) and
                fresh["native_runtime_features"] == {"dotprod": True, "baseline_simd": True}, "same host/features")
        for clock in CLOCKS:
            number(row["setup"][clock], 0, 60, "bounded setup timing")
        require(len(row["samples"]) == 4, "four samples required")
        for j, sample in enumerate(row["samples"]):
            candidate_first = (i + j) % 2 == 0
            order = "candidate-first" if candidate_first else "baseline-first"
            require(sample["index"] == j and sample["order"] == order and
                    sample["cohort_order"] == ("historical-first" if candidate_first else "allocation-matched-first")
                    and sample["allocation_matched"]["order"] == order, "sample counterbalance")
            require(sample["exact"] is True and sample["allocation_matched"]["exact"] is True, "sample exactness")
            phases(sample, HISTORICAL, 60)
            phases(sample["allocation_matched"], MATCHED, 60)
        phases(row, HISTORICAL, 240)
        phases(row["allocation_matched"], MATCHED, 240)
        for scope, keys in ((None, HISTORICAL), ("allocation_matched", MATCHED)):
            total = row if scope is None else row[scope]
            for key in keys:
                for clock in CLOCKS:
                    close(total[key][clock], sum((s if scope is None else s[scope])[key][clock]
                                                for s in row["samples"]))
    return {
        "schema": "pllm.research.boundary_head_dotprod.summary.v3",
        "scope": "synthetic same-weight allocating-output APIs; internal allocation/transposition differs; not isolated SDOT, SDK or whole-client gain",
        "workers": 6, "repeats_per_worker": 4, "rows": 151936, "cols": 896,
        "worker_medians": aggregation(workers), "worker_four_call_totals": aggregation(workers, totals=True),
        "maximum_process_rss_bytes": max(r["fresh_process"]["peak_rss_bytes"] for r in workers),
        "maximum_observed_swap_growth_bytes": guard["maximum_swap_growth_bytes"],
        "whole_client_memory_tier": None,
        "binary_sha256": first["provenance"]["binary_sha256"], "source_sha256": sources,
        "raw_sha256": {name: item["sha256"] for name, item in files.items()},
        "build_timestamp_utc": first["build_provenance"]["built_at_utc"],
    }


def replay(path):
    with path.open("rb") as source:
        raw = source.read((2 << 20) + 1)
    require(len(raw) <= 2 << 20, "archive exceeds 2 MiB")
    result = summarize(json.loads(raw))
    result["archive_sha256"] = hashlib.sha256(raw).hexdigest()
    return json.dumps(result, sort_keys=True, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = replay(args.input)
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
