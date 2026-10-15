#!/usr/bin/env python3
"""Strict read-only fusion audit; print freshly derived summary, never run native code."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

try:
    from scripts.probe_boundary_scan_fusion_full import (
        ALLOCATION, ARMS, COLS, RESERVE, ROWS, TEST, canonical, sha)
except ModuleNotFoundError:
    from probe_boundary_scan_fusion_full import (
        ALLOCATION, ARMS, COLS, RESERVE, ROWS, TEST, canonical, sha)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_pins(pins):
    require(set(pins) == {"schema", "rows", "columns", "workers", "repetitions", "sources",
                         "prebuild_sources_sha256", "binary", "binary_sha256", "build_log_sha256",
                         "build_log", "rustc", "cargo", "head", "allocation_bound_bytes", "reserve_bytes", "test"},
            "unexpected pins/cached fields")
    require(pins.get("schema") == "boundary-scan-fusion-full-pins-v1", "pins schema")
    require((pins.get("rows"), pins.get("columns")) == (ROWS, COLS), "changed geometry")
    require(type(pins.get("workers")) is int and 3 <= pins["workers"] <= 16, "independent workers")
    require(type(pins.get("repetitions")) is int and pins["repetitions"] in range(2, 17, 2), "balanced repetitions")
    require(pins.get("allocation_bound_bytes") == ALLOCATION and pins.get("reserve_bytes") == RESERVE,
            "changed admission")
    require(pins.get("test") == TEST, "changed fixture")
    require(digest(pins.get("binary_sha256")) and isinstance(pins.get("head"), str)
            and len(pins["head"]) in (40, 64)
            and all(c in "0123456789abcdef" for c in pins["head"]), "build pins")
    require(isinstance(pins.get("sources"), dict) and len(pins["sources"]) >= 3
            and all(digest(v) for v in pins["sources"].values()), "source pins")
    require(sha(canonical(pins["sources"]) + b"\n") == pins.get("prebuild_sources_sha256"),
            "changed prebuild sources")
    require(all(isinstance(pins.get(k), str) and pins[k] for k in ("binary", "rustc", "cargo", "build_log")),
            "missing build identity")
    require(sha(pins["build_log"].encode()) == pins.get("build_log_sha256"), "changed build log")
    records = [json.loads(line) for line in pins["build_log"].splitlines() if line.startswith("{")]
    require(any(r.get("reason") == "compiler-artifact" and r.get("profile", {}).get("test") is True
                and r.get("target", {}).get("name") == "pllm_garble"
                and r.get("executable") == pins["binary"] for r in records), "unbound binary")
    require(any(r.get("reason") == "build-finished" and r.get("success") is True for r in records), "failed build")


def parse_native(stdout):
    values = []
    for line in stdout.splitlines():
        if "FUSION_FULL_PROGRESS " in line:
            continue
        # libtest may print `test path ... ` before nocapture JSON.
        position = line.find('{"')
        if position >= 0:
            values.append(json.loads(line[position:]))
    require(len(values) == 1, "one native record required")
    return values[0]


def guard(g, initial):
    require(isinstance(g, dict) and type(g.get("available_bytes")) is int
            and g["available_bytes"] >= RESERVE + ALLOCATION, "bad headroom guard")
    require(type(g.get("swap_used_bytes")) is int and g["swap_used_bytes"] >= 0
            and g["swap_used_bytes"] == initial, "bad swap guard")


def audit_worker(report, pins, pins_hash):
    require(set(report) == {"schema", "worker", "pins_sha256", "observed_binary_sha256",
                           "observed_sources", "guards", "status", "mixed_fixture_sampled_peak_rss_bytes",
                           "pid", "returncode", "stdout", "stderr", "pins_unchanged_after", "failure"},
            "unexpected fields/cached summary forbidden")
    require(report.get("schema") == "boundary-scan-fusion-full-worker-v1"
            and report.get("status") == "complete" and report.get("returncode") == 0
            and report.get("failure") is None, "incomplete worker")
    require(report.get("pins_sha256") == pins_hash
            and report.get("observed_binary_sha256") == pins["binary_sha256"]
            and report.get("observed_sources") == pins["sources"]
            and report.get("pins_unchanged_after") is True, "changed build/source")
    require(type(report.get("pid")) is int and report["pid"] > 0, "missing process identity")
    require(type(report.get("mixed_fixture_sampled_peak_rss_bytes")) is int
            and report["mixed_fixture_sampled_peak_rss_bytes"] > 0, "missing mixed peak")
    observations = report.get("guards")
    require(isinstance(observations, list) and len(observations) >= 2, "missing guards")
    initial = observations[0].get("swap_used_bytes")
    for g in observations:
        guard(g, initial)
    require("running 1 test" in report["stdout"]
            and "test result: ok. 1 passed; 0 failed;" in report["stdout"], "incomplete libtest")
    native = parse_native(report["stdout"])
    require(set(native) == {"schema", "rows", "columns", "repetitions", "samples", "setup_wall_ns",
                           "allocation_bound_bytes", "work_bound_weight_values", "native_initial_guard",
                           "native_guards", "native_final_guard", "reserve_bytes", "native_simd",
                           "matrix_owners", "parties", "correct", "fault_controls",
                           "physical_memory_scans_measured", "sdk_selectable"}, "unexpected native/cached fields")
    require(native.get("schema") == "boundary-scan-fusion-full-v1"
            and (native.get("rows"), native.get("columns")) == (ROWS, COLS), "native geometry")
    require(native.get("repetitions") == pins["repetitions"] and native.get("parties") == 2
            and native.get("matrix_owners") == 1, "native geometry/owners")
    for key in ("correct", "fault_controls"):
        require(native.get(key) is True, "bad parity/fault controls")
    for key in ("physical_memory_scans_measured", "sdk_selectable"):
        require(native.get(key) is False, "invalid scope")
    require(native.get("allocation_bound_bytes") == ALLOCATION
            and native.get("reserve_bytes") == RESERVE
            and native.get("work_bound_weight_values") == ROWS * COLS * (2 * pins["repetitions"] * 12 + 8),
            "native admission")
    guard(native.get("native_initial_guard"), initial)
    guard(native.get("native_final_guard"), initial)
    require(len(native.get("native_guards", [])) == pins["repetitions"], "native guard gaps")
    for g in native["native_guards"]:
        guard(g, initial)
    samples = native.get("samples", [])
    require(len(samples) == 2 * pins["repetitions"], "incomplete orders")
    progress = [json.loads(line.split("FUSION_FULL_PROGRESS ", 1)[1])
                for line in report["stdout"].splitlines() if "FUSION_FULL_PROGRESS " in line]
    require(progress == samples, "changed/incomplete native progress")
    seen = set()
    for sample in samples:
        iteration, party = sample.get("iteration"), sample.get("party")
        require(type(iteration) is int and 0 <= iteration < pins["repetitions"]
                and type(party) is int and party in (0, 1), "sample identity")
        require((iteration, party) not in seen, "duplicate sample")
        seen.add((iteration, party))
        order = sample.get("order", [])
        require(sorted(order) == sorted(ARMS), "incomplete order")
        shift = (iteration // 2 + party) % 5
        expected = [ARMS[(i + shift) % 5] for i in range(5)]
        if iteration % 2:
            expected.reverse()
        a, b = expected.index("original_pair"), expected.index("fused_pair")
        if (a > b) == ((iteration + party) % 2 == 0):
            expected[a], expected[b] = expected[b], expected[a]
        require(order == expected, "changed counterbalance")
        require(sample.get("parity") is True and sample.get("replay_rejected") is True, "bad parity")
        require((sample.get("rows_visited"), sample.get("weight_values_visited"), sample.get("scale_bytes_visited"))
                == (ROWS, ROWS * COLS, 4 * ROWS), "bad work accounting")
        require(set(sample.get("timings", {})) == set(ARMS), "missing arm")
        for timing in sample["timings"].values():
            require(set(timing) == {"wall_ns", "process_cpu_ns"}
                    and all(type(v) is int and 0 < v < 10**15 for v in timing.values()), "invalid timing")
    return native


def schedule(lookup_a, lookup_b, random_head, fused, complementary_head, body):
    """Analytical local service schedule; no network, queue, codec or body observation."""
    start = max(lookup_a, lookup_b)
    fused_start = max(fused, lookup_b)
    return {"installed_ready_parallel_body_start_ns": start,
            "installed_ready_parallel_finish_ns": start + body + max(random_head, complementary_head),
            "fused_body_start_ns": fused_start,
            "fused_finish_ns": fused_start + body + complementary_head,
            "fused_body_start_delay_ns": fused_start - start,
            "separate_runahead_finish_ns": max(start + body + complementary_head, lookup_a + random_head)}


def summarize(paths, pins_raw, body_ns=200_000_000):
    require(type(body_ns) is int and 0 <= body_ns <= 10**12, "analytical body bounds")
    pins = json.loads(pins_raw)
    validate_pins(pins)
    require(len(paths) == pins["workers"], "incomplete cohort")
    workers, ids, pids = [], set(), set()
    for path in paths:
        raw = Path(path).read_bytes()
        report = json.loads(raw)
        worker = report.get("worker")
        require(type(worker) is int and 0 <= worker < pins["workers"] and worker not in ids, "worker identity")
        ids.add(worker)
        require(report.get("pid") not in pids, "independent processes required")
        pids.add(report.get("pid"))
        native = audit_worker(report, pins, sha(pins_raw))
        samples = native["samples"]
        ratios = {}
        for clock in ("wall_ns", "process_cpu_ns"):
            baseline = sum(s["timings"]["original_pair"][clock] for s in samples)
            candidate = sum(s["timings"]["fused_pair"][clock] for s in samples)
            ratios[clock] = candidate / baseline
        totals = {arm: {clock: sum(s["timings"][arm][clock] for s in samples)
                        for clock in ("wall_ns", "process_cpu_ns")} for arm in ARMS}
        schedules, token_cpu = [], []
        for iteration in range(pins["repetitions"]):
            a, b = [next(s for s in samples if (s["iteration"], s["party"]) == (iteration, p)) for p in (0, 1)]
            for paired, other in ((a, b), (b, a)):
                wall = lambda sample, arm: sample["timings"][arm]["wall_ns"]
                # Assign this sample's verified W(q-r) to logical opposite role;
                # never splice an unrelated sample's complementary input.
                schedules.append(schedule(wall(paired, "lookup"), wall(other, "lookup"),
                                          wall(paired, "head"), wall(paired, "fused_pair"),
                                          wall(paired, "complementary_head"), body_ns))
                cpu = lambda sample, arm: sample["timings"][arm]["process_cpu_ns"]
                token_cpu.append({"baseline_two_worker_cpu_ns": cpu(paired, "original_pair") + cpu(other, "lookup") + cpu(paired, "complementary_head"),
                                  "candidate_two_worker_cpu_ns": cpu(paired, "fused_pair") + cpu(other, "lookup") + cpu(paired, "complementary_head")})
        workers.append({"worker": worker, "report_sha256": sha(raw), "paired_aggregate_ratios": ratios,
                        "paired_sample_deltas": [
                            {"iteration": s["iteration"], "party": s["party"],
                             **{clock: s["timings"]["fused_pair"][clock] - s["timings"]["original_pair"][clock]
                                for clock in ("wall_ns", "process_cpu_ns")}} for s in samples],
                        "arm_totals": totals, "analytical_schedules": schedules,
                        "two_worker_token_cpu": token_cpu,
                        "mixed_fixture_sampled_peak_rss_bytes": report["mixed_fixture_sampled_peak_rss_bytes"]})
    wall_ratio = statistics.median(w["paired_aggregate_ratios"]["wall_ns"] for w in workers)
    cpu_ratio = statistics.median(w["paired_aggregate_ratios"]["process_cpu_ns"] for w in workers)
    return {"schema": "boundary-scan-fusion-full-summary-v1", "pins_sha256": sha(pins_raw),
            "workers": sorted(workers, key=lambda w: w["worker"]),
            "median_worker_paired_wall_ratio": wall_ratio, "median_worker_paired_cpu_ratio": cpu_ratio,
            "scalar_fusion_regresses": wall_ratio >= 1 or cpu_ratio >= 1,
            "promotion_gate": "blocked: SDK/real-checkpoint parity, authenticated tickets, schedule and physical-load review required",
            "schedule_scope": "analytical service-only, not observed model/SDK latency; equal-geometry roles; no network/codec/queue",
            "analytical_body_ns": body_ns, "memory_scope": "mixed-fixture sampled process peak; not per-arm or Low-client",
            "logical_two_worker_source_bytes": {"baseline": 4 * ROWS * COLS + 8 * ROWS,
                                                 "candidate": 3 * ROWS * COLS + 8 * ROWS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", required=True)
    parser.add_argument("--analytical-body-ms", type=float, default=200.)
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    require(math.isfinite(args.analytical_body_ms), "finite analytical body required")
    print(json.dumps(summarize(args.reports, Path(args.pins).read_bytes(),
                               round(args.analytical_body_ms * 1e6)), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
