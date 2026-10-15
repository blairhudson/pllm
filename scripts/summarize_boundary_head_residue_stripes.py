#!/usr/bin/env python3
"""Read-only audit of six completed generated-source residue-stripe workers.

Stdlib only: never imports the fixture, native extension or model runtime. Pins
must come from the frozen pre-run plan, not be guessed from a mixed cohort.
Hashes establish identity/integrity, not authenticity of a worker's assertions.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys

MODES = ("parallel", "sliced", "snapshots")
ROLES = ("boundary_a", "boundary_b")
ORDERS = [list(order) for order in itertools.permutations(MODES)]
V, H, STRIPE = 151936, 896, 8192
WEIGHTS = V * H
RANGES = [min(STRIPE, V - start) for start in range(0, V, STRIPE)]
MAX_FILE = 4 << 20
SOURCE_PATHS = {
    **{f"pllm.runtime.{name}": f"python/pllm/runtime/{name}.py" for name in (
        "native", "native_wan", "private_boundary_head_codec", "private_boundary_protocol",
        "private_boundary_transport", "private_boundary_worker", "quantization")},
    **{name: name for name in ("scripts/probe_boundary_head_residue_stripes.py",
                              "tests/test_boundary_head_residue_stripes.py",
                              "docs/research/boundary-head-residue-stripes.md")},
}


class Rejected(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise Rejected(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None, "invalid_digest")
    return value


def number(value, *, positive=False):
    require(type(value) in (int, float), "invalid_numeric_measurement")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and (value > 0 if positive else value >= 0), "invalid_numeric_measurement")
    return value


def count(value):
    require(type(value) is int and value >= 0, "invalid_counter")
    return value


def same(value, expected, code):
    # Canonical comparison also keeps True distinct from 1 and 0.0 from 0.
    require(canonical(value) == canonical(expected), code)


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def constant(_):
        raise Rejected("nonfinite_json")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        canonical(value)  # Reject exponent overflow too.
        require(type(value) is dict, "json_object_required")
        return value
    except (UnicodeError, json.JSONDecodeError, OverflowError, RecursionError, ValueError) as exc:
        if isinstance(exc, Rejected):
            raise
        raise Rejected("invalid_json") from None


def read_raw(path):
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_FILE + 1)
    except OSError:
        raise Rejected("missing_or_unreadable_input") from None
    require(len(raw) <= MAX_FILE, "input_size_limit")
    return raw


def geometry(read):
    return {"vocabulary": V, "hidden": H, "stripe_rows": STRIPE, "seed": 17,
            "activation_bits": 8, "modes": list(MODES), "repeats": 6,
            "read_chunk_bytes": read, "download_mbps": 100., "upload_mbps": 40.}


def codec_geometry(descriptor):
    return {"schema": "pllm.research.head-stripes.v1", "descriptor": descriptor,
            "rows": STRIPE, "vocabulary": V, "order": "ascending-complete",
            "codec": "installed-uniform-residues", "lookahead": 1}


def msgpack_size(value):
    """Independent length-only MessagePack subset; ('binary', n) allocates no body."""
    if type(value) is tuple:
        _, n = value
        return n + (2 if n <= 255 else 3 if n <= 65535 else 5)
    if type(value) is str:
        n = len(value.encode())
        return n + (1 if n <= 31 else 2 if n <= 255 else 3)
    if type(value) is int:
        require(0 <= value <= 127, "unsupported_msgpack_integer")
        return 1
    if type(value) is list:
        return 1 + sum(msgpack_size(item) for item in value)
    if type(value) is dict:
        return 1 + sum(msgpack_size(k) + msgpack_size(v) for k, v in value.items())
    raise Rejected("unsupported_msgpack_type")


def frame_bytes(role, size, *, response=False):
    return msgpack_size({"schema": "pllm.private_boundary.v1." + ("reply" if response else "frame"),
        "role": role, "session": "c" * 32, "epoch": 0, "sequence": 0,
        "operation": "head", "frames": [("binary", size)]})


def body_sizes(mode, role):
    reply = (frame_bytes(role, 74 + (V * 25 + 7) // 8, response=True) if mode == "parallel" else
             sum(4 + frame_bytes(role, 74 + (rows * 25 + 7) // 8, response=True) for rows in RANGES))
    return frame_bytes(role, H * 4), reply


def validate_guard(guard):
    host = guard["host"]
    for key in ("total", "available", "swap_used", "disk_free"):
        count(host[key])
    reserve = max(4 << 30, host["total"] // 8)
    allowance = max(1 << 30, 6 * WEIGHTS + (512 << 20))
    same(guard["reserve_bytes"], reserve, "host_reserve")
    same(guard["allocation_allowance_bytes"], allowance, "host_allowance")
    same(guard["maximum_swap_growth_bytes"], 0, "swap_growth")
    same(guard["sample_interval_seconds"], .1, "guard_interval")
    require(host["total"] >= host["available"] >= reserve + allowance, "host_admission")
    require(host["disk_free"] >= allowance, "disk_admission")
    require(reserve <= count(guard["minimum_available_bytes"]) <= host["available"], "runtime_reserve")


def validate_network(row, bodies):
    network = row["all_phases_network"]
    same(network["schema"], "pllm.native_wan.v1", "network_schema")
    require(network["enforced"] is True, "network_not_enforced")
    same(network["backend"], "native-shared-tcp-pacer", "network_backend")
    conditions = {"schema": "pllm.wan_conditions.v1", "download_mbps": 100., "upload_mbps": 40.,
                  "parties": [], "role_parties": []}
    same(network["conditions"], conditions, "network_conditions")
    same(network["conditions_digest"], sha(b"pllm.wan_conditions.v1\0" + canonical(conditions)), "network_digest")
    parties = ("client", *ROLES)
    same(network["role_parties"], {p: p for p in parties}, "party_colocation")
    same(network["burst_bytes"], 16384, "network_burst")
    same(network["read_chunk_bytes"], 8192, "relay_read_bound")
    same(network["max_connections"], 32, "network_connections")
    number(network["elapsed_seconds"], positive=True)
    online = row["online_tcp_payload_bytes"]
    all_links = network["directed_stream_bytes"]
    links = {link for role in ROLES for link in (f"client->{role}", f"{role}->client")}
    require(set(online) == set(all_links) == links, "link_set")
    for role in ROLES:
        upload, download = bodies[role]
        for link, body in ((f"client->{role}", upload), (f"{role}->client", download)):
            require(count(online[link]) > body, "missing_http_framing")
            require(count(all_links[link]) >= online[link], "phase_byte_conservation")
    total = sum(count(n) for n in all_links.values())
    same(network["total_stream_bytes"], total, "network_total")
    same(network["client_stream_bytes"], total, "client_network_total")
    require(set(network["parties"]) == set(parties), "party_set")
    require(network["elapsed_seconds"] >= row["setup_seconds"] + row["exchange_seconds"], "network_timing_scope")
    for party in parties:
        for direction, rate in (("upload", 5_000_000.), ("download", 12_500_000.)):
            bucket = network["parties"][party][direction]
            same(bucket["bytes_per_second"], rate, "party_capacity")
            size = sum(n for link, n in all_links.items() if link.split("->")[0 if direction == "upload" else 1] == party)
            same(bucket["admitted_bytes"], size, "party_byte_conservation")


def validate_row(row, read, descriptor):
    mode = row["mode"]
    require(mode in MODES and row["success"] is True and row["error"] is None, "failed_exchange")
    require(row["online_coverage_complete"] is True, "partial_coverage")
    same(row["geometry"], codec_geometry(descriptor), "row_geometry")
    same(row["candidate_read_chunk_bytes"], read, "client_read_geometry")
    same(row["remaining_native_tasks"], 0, "undrained_native")
    digest(row["logits_sha256"])
    elapsed = number(row["exchange_seconds"], positive=True)
    number(row["exchange_cpu_seconds"])
    number(row["setup_seconds"])
    baseline = mode == "parallel"
    same(row["replay_status"], [] if baseline else [409, 409], "replay_rejection")
    bodies = {}
    require(set(row["per_role_bodies"]) == set(ROLES), "body_roles")
    for index, role in enumerate(ROLES):
        request, reply = body_sizes(mode, role)
        ledger = row["per_role_bodies"][role]
        expected = ({"installed_online_upload_bytes": request, "installed_online_download_bytes": reply}
                    if baseline else {"attempted_request_body_bytes": request,
                                      "generated_reply_body_bytes": reply, "observed_reply_body_bytes": reply})
        same(ledger, expected, "framed_body_conservation")
        bodies[role] = request, reply
        timing = row["timing_breakdown"][index]
        same(timing["asgi_response_body_bytes"], reply, "asgi_body_conservation")
        for key in ("native_seconds", "encode_seconds", "first_asgi_body_seconds"):
            require(number(timing[key]) <= elapsed, "timing_outside_exchange")
        if baseline:
            # Zero native/encode values are placeholders, NOT measured baseline costs.
            same(timing["native_seconds"], 0., "baseline_native_placeholder")
            same(timing["encode_seconds"], 0., "baseline_encode_placeholder")
        else:
            for key in ("first_yield_seconds", "first_client_body_seconds", "first_paired_decoded_stripe_seconds"):
                require(number(timing[key]) <= elapsed, "timing_outside_exchange")
            require(timing["first_paired_decoded_stripe_seconds"] >= timing["first_client_body_seconds"], "decode_before_body")
            owner = row["owners"][index]
            same(owner, {"attempted": True, "retired": True, "native_calls": 1 if mode == "sliced" else 19,
                         "drained_native_calls": 0, "peak_output_slots": 1 if mode == "sliced" else 2,
                         "peak_output_bytes": V * 4 if mode == "sliced" else STRIPE * 8,
                         "peak_pending_native": 1, "replay_rejected": True}, "owner_retirement_or_bound")
    if baseline:
        require(row["owners"] is None and row["candidate_operation_counters"] is None, "baseline_owner_schema")
        same(row["baseline_ownership"], "installed worker/session and paired executor", "baseline_path")
    else:
        require(len(row["owners"]) == 2 and row["baseline_ownership"] is None, "candidate_owner_schema")
        same(row["candidate_operation_counters"], {r: {"head_calls_reserved": 1, "sequence_reserved": 0,
             "complete_stripes_expected": 19} for r in ROLES}, "candidate_admission_counters")
    same(row["installed_operation_counters"], {
        "lookup": {"calls": 0, "rows": 0, "upload_bytes": 0, "download_bytes": 0},
        "head": {"calls": int(baseline), "rows": int(baseline),
                 "upload_bytes": sum(v[0] for v in bodies.values()) if baseline else 0,
                 "download_bytes": sum(v[1] for v in bodies.values()) if baseline else 0}}, "installed_head_control")
    validate_network(row, bodies)


def worker(path, pins, checkout):
    paths = [Path(path), Path(str(path) + ".progress.json"), Path(str(path) + ".runs.jsonl")]
    raws = [read_raw(p) for p in paths]
    report, progress = parse(raws[0]), parse(raws[1])
    require(raws[2].endswith(b"\n"), "partial_jsonl")
    lines = raws[2].splitlines()
    require(len(lines) == 20 and all(lines), "jsonl_event_count")
    events = [parse(line) for line in lines]
    same(report["schema"], "pllm.research.head-residue-stripes.v1", "report_schema")
    same(progress, report["progress"], "progress_sidecar_mismatch")
    same(progress["schema"], "pllm.research.head-residue-stripes-progress.v1", "progress_schema")
    require(progress["status"] == "completed" and progress["measurement_complete"] is True,
            "unfinished_progress")
    require(not any(key in progress for key in ("failure_code", "retention_error")), "failure_progress")
    same(progress["planned_runs"], 18, "planned_run_count")
    same(progress["completed_runs"], 18, "completed_run_count")
    plan = report["plan"]
    require(plan["sampled_qwen"] is True and type(plan["sampled_worker"]) is int
            and plan["sampled_worker"] in (1, 2, 3), "not_sampled_qwen")
    read = plan["geometry"]["read_chunk_bytes"]
    require(type(read) is int and read in (8192, 1024), "read_cohort")
    same(plan["geometry"], geometry(read), "frozen_geometry")
    ghash = sha(canonical(plan["geometry"]))
    same(plan["geometry_sha256"], ghash, "geometry_digest")
    same(progress["geometry_sha256"], ghash, "progress_geometry")
    same(plan["repeat_orders"], ORDERS, "planned_orders")
    same(report["repeat_orders"], ORDERS, "actual_orders")
    provenance = plan["provenance"]
    for key in ("sources_sha256", "native_sha256"):
        digest(provenance[key])
        same(provenance[key], pins[key], "original_" + key)
    same(provenance["build"], pins["build"], "original_build")
    for value in provenance["build"].values():
        digest(value)
    same(sha(canonical(provenance["sources"])), provenance["sources_sha256"], "source_map_digest")
    require(set(provenance["sources"]) == set(SOURCE_PATHS), "source_map_keys")
    for key, relative in SOURCE_PATHS.items():
        same(digest(provenance["sources"][key]), sha(read_raw(checkout / relative)), "original_source_file")
    same(progress["sources_sha256"], provenance["sources_sha256"], "progress_sources")
    same(report["native_sha256"], provenance["native_sha256"], "native_identity")
    require(type(report["native_filename"]) is str and "/" not in report["native_filename"], "native_filename")
    for key in ("installed_matrix_methods", "installed_executor_methods"):
        methods = report[key]
        require(type(methods) is list and all(type(m) is str for m in methods)
                and methods == sorted(set(methods)), "native_api_record")
    require({"shape", "weight_bytes"} <= set(report["installed_matrix_methods"])
            and "wrap32" in report["installed_executor_methods"], "native_api_missing")
    same(report["source_sha256"], sha(canonical({"generated": 17, "vocabulary": V, "hidden": H})), "generated_source")
    for key in ("descriptor_sha256", "codec_geometry_sha256", "weight_sha256", "scales_sha256"):
        digest(report[key])
    same(report["codec_geometry_sha256"], sha(canonical(codec_geometry(report["descriptor_sha256"]))), "codec_geometry_hash")
    require(report["production_ready"] is False, "production_claim")
    setup = report["setup"]
    number(setup["wall_seconds"])
    for key in ("generated_weight_bytes", "full_native_weight_bytes", "additional_snapshot_weight_bytes"):
        same(setup[key], WEIGHTS, "weight_allocation")
    same(setup["codec_layout_bytes"], V, "layout_allocation")
    validate_guard(plan["host_admission"])
    validate_guard(report["host_guard"])
    same(plan["host_admission"]["host"], report["host_guard"]["host"], "guard_host_identity")
    same(events[0], {"kind": "fixture", "setup": setup,
         "descriptor_sha256": report["descriptor_sha256"], "geometry_sha256": report["codec_geometry_sha256"],
         "source_sha256": report["source_sha256"], "weight_sha256": report["weight_sha256"],
         "scales_sha256": report["scales_sha256"]}, "fixture_sidecar")
    same(events[-1].get("kind"), "complete", "complete_event")
    core = {k: v for k, v in report.items() if k not in ("plan", "host_guard", "progress")}
    same(events[-1], {"kind": "complete", "report": core}, "complete_sidecar")
    require(len(report["runs"]) == 18, "run_count")
    for index, (row, event) in enumerate(zip(report["runs"], events[1:-1])):
        same(event, {"kind": "run", "report": row}, "run_sidecar")
        repeat, position = divmod(index, 3)
        same([row["repeat"], row["position"], row["mode"]], [repeat, position, ORDERS[repeat][position]], "repeat_order")
        validate_row(row, read, report["descriptor_sha256"])
    require(len({r["logits_sha256"] for r in report["runs"]}) == 1, "logit_mismatch")
    hashes = {label: sha(raw) for label, raw in zip(("report", "progress", "runs"), raws)}
    # Detect concurrent change; never wait for or consume an ongoing run.
    require(all(read_raw(p) == raw for p, raw in zip(paths, raws)), "inputs_changed_during_read")
    return report, hashes


def med(values):
    return statistics.median(values)


def worker_metrics(report, hashes):
    rows = report["runs"]
    owners = [owner for row in rows if row["owners"] is not None for owner in row["owners"]]
    result = {"worker": report["plan"]["sampled_worker"], "raw_sha256": hashes, "arms": {},
              "validated_exposed_counts": {"successful_exchange_rows": len(rows),
                  "candidate_retired_owners": sum(owner["retired"] for owner in owners),
                  "candidate_native_calls": sum(owner["native_calls"] for owner in owners),
                  "replay_409_rejections": sum(len(row["replay_status"]) for row in rows),
                  "remaining_native_tasks": sum(row["remaining_native_tasks"] for row in rows),
                  "baseline_retired_owner_count": None, "context_fault_injection_count": None}}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        arm = {"samples": 6, "elapsed_seconds_median": med(r["exchange_seconds"] for r in selected),
               "aggregate_fixture_cpu_seconds_median": med(r["exchange_cpu_seconds"] for r in selected),
               "transport_setup_seconds_median": med(r["setup_seconds"] for r in selected),
               "online_tcp_payload_bytes_median": med(sum(r["online_tcp_payload_bytes"].values()) for r in selected),
               "total_application_body_bytes": sum(sum(body_sizes(mode, role)) for role in ROLES),
               "per_role_application_bodies": {role: dict(zip(("request_bytes", "reply_bytes"), body_sizes(mode, role)))
                                                for role in ROLES}}
        arm["encoding_elapsed_seconds_median_by_role"] = (None if mode == "parallel" else
            {role: med(r["timing_breakdown"][i]["encode_seconds"] for r in selected) for i, role in enumerate(ROLES)})
        arm["native_elapsed_seconds_median_by_role"] = (None if mode == "parallel" else
            {role: med(r["timing_breakdown"][i]["native_seconds"] for r in selected) for i, role in enumerate(ROLES)})
        arm["availability_seconds_median_by_role"] = {role: {
            key: (None if mode == "parallel" and key != "first_asgi_body_seconds" else
                  med(r["timing_breakdown"][i][key] for r in selected))
            for key in ("first_asgi_body_seconds", "first_client_body_seconds", "first_paired_decoded_stripe_seconds")}
            for i, role in enumerate(ROLES)}
        result["arms"][mode] = arm
    paired = []
    for repeat in range(6):
        by_mode = {r["mode"]: r for r in rows if r["repeat"] == repeat}
        deltas = {}
        for control, candidate in (("parallel", "sliced"), ("parallel", "snapshots"), ("sliced", "snapshots")):
            a, b = by_mode[control], by_mode[candidate]
            deltas[candidate + "_minus_" + control] = {
                "elapsed_seconds": b["exchange_seconds"] - a["exchange_seconds"],
                "aggregate_fixture_cpu_seconds": b["exchange_cpu_seconds"] - a["exchange_cpu_seconds"],
                "application_body_bytes": sum(sum(body_sizes(candidate, role)) - sum(body_sizes(control, role)) for role in ROLES),
                "online_tcp_payload_bytes": sum(b["online_tcp_payload_bytes"].values()) - sum(a["online_tcp_payload_bytes"].values())}
            for role in ROLES:
                deltas[candidate + "_minus_" + control][role + "_application_body_bytes"] = (
                    sum(body_sizes(candidate, role)) - sum(body_sizes(control, role)))
        paired.append({"repeat": repeat, "deltas": deltas})
    result["paired_repeats"] = paired
    result["paired_delta_medians"] = {pair: {metric: med(row["deltas"][pair][metric] for row in paired)
        for metric in paired[0]["deltas"][pair]} for pair in paired[0]["deltas"]}
    result["fixture_setup_wall_seconds"] = report["setup"]["wall_seconds"]
    return result


def summarize(paths, pins, *, checkout=None):
    require(len(paths) == 6 and len({str(Path(p).resolve()) for p in paths}) == 6, "six_distinct_reports_required")
    checkout = Path(checkout) if checkout else Path(__file__).resolve().parents[1]
    for key in ("sources_sha256", "native_sha256"):
        digest(pins[key])
    require(set(pins["build"]) == {"expected_digest", "interpreter_sha256"}, "build_pins")
    workers = [worker(path, pins, checkout) for path in paths]
    identities = [(r["plan"]["geometry"]["read_chunk_bytes"], r["plan"]["sampled_worker"]) for r, _ in workers]
    require(set(identities) == {(read, i) for read in (8192, 1024) for i in (1, 2, 3)}, "missing_or_duplicate_worker")
    first = workers[0][0]
    keys = ("source_sha256", "descriptor_sha256", "codec_geometry_sha256", "weight_sha256", "scales_sha256",
            "native_sha256", "native_filename", "installed_matrix_methods", "installed_executor_methods")
    for report, _ in workers:
        for key in keys:
            same(report[key], first[key], "mixed_" + key)
        same(report["plan"]["provenance"], first["plan"]["provenance"], "mixed_provenance")
        same(report["runs"][0]["logits_sha256"], first["runs"][0]["logits_sha256"], "mixed_logit_identity")
    cohorts = {}
    for read in (8192, 1024):
        selected = sorted(((r, h) for r, h in workers if r["plan"]["geometry"]["read_chunk_bytes"] == read),
                          key=lambda item: item[0]["plan"]["sampled_worker"])
        metrics = [worker_metrics(r, h) for r, h in selected]
        cohorts[str(read)] = {"workers": metrics,
            "across_worker_medians_of_worker_medians": {mode: {key: med(w["arms"][mode][key] for w in metrics)
                for key in ("elapsed_seconds_median", "aggregate_fixture_cpu_seconds_median",
                            "transport_setup_seconds_median", "online_tcp_payload_bytes_median")}
                for mode in MODES},
            "across_worker_medians_of_paired_worker_medians": {pair: {
                metric: med(w["paired_delta_medians"][pair][metric] for w in metrics)
                for metric in metrics[0]["paired_delta_medians"][pair]}
                for pair in metrics[0]["paired_delta_medians"]}}
    return {"schema": "pllm.research.head-residue-stripes-summary.v1", "read_cohorts": cohorts,
            "summarizer_sha256": sha(read_raw(Path(__file__))),
            "identity": {key: first[key] for key in keys}, "provenance": first["plan"]["provenance"],
            "repeat_orders": ORDERS, "production_ready": False,
            "qualification": "exposed fixture timings/counters only; full-integer comparative qualification blocked by missing baseline evidence",
            "scope": "generated-source co-located fixture CPU, not role CPU or SDK CPU; no performance promotion",
            "allocation": {"generated_weight_bytes": WEIGHTS, "full_native_weight_bytes": WEIGHTS,
                           "extra_persistent_snapshot_weight_bytes": WEIGHTS,
                           "all_three_weight_owners_bytes": 3 * WEIGHTS,
                           "source_declared_client_reader_bound": "one framed stripe plus read chunk per link; not a measured peak; framework buffers additional",
                           "native_outputs_per_role": {"sliced_bytes": 4 * V, "snapshots_current_plus_next_bytes": 8 * STRIPE}},
            "evidence_limits": {
                "integer_parity": {"sliced": "pinned source asserts full integer oracle before success",
                                   "snapshots": "pinned source asserts full integer oracle before success",
                                   "parallel": "unknown: only complete float32 logits equality emitted"},
                "direct_integer_digest": "not emitted; cannot independently reconstruct or certify baseline integer equality",
                "context_fault_runs": "not emitted by sampled plan; replay rejection only, no fault-test attestation",
                "baseline_retirement_and_output_counters": "not emitted; installed source lifecycle only, not measured counts",
                "rss": "not emitted; lifetime shared-worker RSS would not be per-arm or Low tier",
                "import_and_process_startup": "unknown, not timed by fixture setup or exchange",
                "setup": "fixture generation/compile/oracle wall and per-exchange transport setup measured separately; import/startup unknown",
                "setup_cpu": "unknown, not emitted",
                "baseline_native_and_encoding": "unknown: zero placeholders are not measurements",
                "http_framing": "body sizes independently reconstructed; aggregate relay conservation only, no raw HTTP capture",
                "buffer_copies": "logical native reservation counts only; packing, allocator and socket owners additional",
                "client_reader_peak": "not emitted; reader bound follows pinned source, not a measured buffer counter",
                "encoding": "candidate per-role elapsed encoding totals measured inside aggregate exchange; no per-role CPU attribution",
                "supervisor": "parent guard/provenance overhead outside worker process CPU and timed exchange",
                "hashes": "identity/integrity checks, not signatures or malicious-worker correctness"},
            "gate": "production no-go until borrowed immutable row-range execution and whole-SDK parity/resource evidence"}


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=6, type=Path, help="explicit completed report paths; never discovers live outputs")
    parser.add_argument("--expected-sources-sha256", required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--expected-build-digest", required=True)
    parser.add_argument("--expected-interpreter-sha256", required=True)
    args = parser.parse_args(argv)
    pins = {"sources_sha256": args.expected_sources_sha256, "native_sha256": args.expected_native_sha256,
            "build": {"expected_digest": args.expected_build_digest, "interpreter_sha256": args.expected_interpreter_sha256}}
    try:
        result = summarize(args.reports, pins)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (Rejected, KeyError, TypeError, IndexError, ValueError, OverflowError, RecursionError) as exc:
        code = str(exc) if isinstance(exc, Rejected) else "missing_or_malformed_evidence"
        print(json.dumps({"schema": "pllm.research.head-residue-stripes-summary.v1",
                          "status": "rejected", "failure_code": code}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
