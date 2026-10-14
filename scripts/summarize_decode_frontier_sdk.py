"""Read-only, single-cohort summaries for the frozen SDK frontier benchmarks.

No build inspection, source resolution, role launch, or cross-cohort comparison.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_decode_cache_sdk as decode  # noqa: E402
import benchmark_growing_epochs_sdk as growing  # noqa: E402

LOW_BYTES = 268435456
SCHEMAS = {
    "pllm.growing_epochs_sdk_benchmark.v1": False,
    "pllm.decode_cache_sdk_benchmark.v1": True,
}
WINDOWS = {
    False: {
        "client_cpu_rss": "process birth through close and state hashing",
        "provider_cpu": "process birth through client exit; excludes role shutdown",
        "provider_rss": "50 ms samples after startup through client exit; startup peak unknown",
        "tcp_streams": "native relays from topology startup through provider/network close",
        "covered_bodies": "client construction through close, v3 counters only",
        "aggregate_cpu": "complete cell including launcher, relays and reaped client/role children",
        "ttft": "request start through first text; all admission/preparation waiting charged",
        "decode_tps": "N-1 / first-to-last text delta; all epoch pauses charged",
    },
    True: {
        "client": "process birth through response state hashing, client and disk-owner close",
        "provider_cpu": "birth through client exit; shutdown excluded",
        "provider_rss": "50ms sampled after startup through client exit; startup peak unknown",
        "all_link_tcp": "native topology/relay startup through role and relay shutdown",
        "covered_bodies": "client construction through close, v3 controls/ACK included",
        "disk": "owner bind through close, every encrypted read/write including cold publication and orphans",
        "request": "ordinary SDK request start through terminal stream; hashing and publication separately charged in lifecycle",
        "aggregate_cpu": "launcher plus all reaped client/provider children through complete cleanup",
    },
}
COUNTERS = tuple("prepared_stage_rows_" + name for name in
                 ("issued", "reserved", "claimed", "burned", "discarded")) + (
    "decode_epochs_started", "decode_epoch_pauses", "decode_epoch_pause_ns")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, *, integer=False):
    require(type(value) in ({int} if integer else {int, float})
            and math.isfinite(value) and value >= 0, f"missing or invalid {label}")
    return value


def digest(value):
    require(isinstance(value, str) and len(value) == 64
            and set(value) <= set("0123456789abcdef"), "invalid identity digest")


def median(values):
    """Unknown in any sample remains unknown, never a zero or a smaller cohort."""
    return statistics.median(values) if values and all(v is not None for v in values) else None


def numeric_tree_summary(values):
    """Preserve nonnumeric scope/unknown fields; aggregate numeric leaves only."""
    if all(isinstance(v, dict) for v in values):
        keys = set().union(*(v.keys() for v in values))
        return {k: numeric_tree_summary([v.get(k) for v in values]) for k in sorted(keys)}
    if all(v is None or type(v) in {int, float} for v in values):
        return median(values)
    return deepcopy(values[0]) if all(v == values[0] for v in values) else {"samples": values}


def check_ledger(stored, calculated):
    require(stored == calculated, "tampered scoped body accounting or scope")
    part = calculated["traffic_attribution"]
    total, base, output, residual = (number(part[k], k, integer=True) for k in (
        "total_all_link_body_bytes", "known_base_all_link_body_bytes",
        "known_output_dependent_all_link_body_bytes", "unattributed_all_link_body_bytes"))
    require(total == base + output + residual, "traffic conservation failed")


def validate(report):
    """Validate report declarations and every cell before calculating any median."""
    require(report.get("schema") in SCHEMAS, "unsupported raw cohort schema")
    cached = SCHEMAS[report["schema"]]
    require(report.get("complete") is True and not report.get("failure"), "incomplete or failed cohort")
    require(type(report.get("tiny")) is bool, "tiny classification missing")
    require(report.get("memory_guard", {}).get("failed") is False, "missing or failed memory guard")
    for key in ("source_lock_digest", "cohort_commitment"):
        digest(report.get(key))
    digest(report.get("build", {}).get("expected_digest"))
    require(bool(report["build"].get("python")), "missing build interpreter identity")
    require(bool(report.get("frozen_sources")), "missing frozen source identities")
    for value in report["frozen_sources"].values():
        digest(value)
    require(report.get("windows") == WINDOWS[cached] and bool(report.get("limits")), "invalid measurement scopes")
    require(report.get("execution_isolation", {}).get("sdk_native_file_lock_checked_per_child") is True,
            "missing installed implementation lock")
    experiments = report["experiments"]
    keys = set(experiments)
    workloads = {key.split("/")[0] for key in keys}
    policies = decode.POLICIES if cached else growing.VARIANTS
    allowed = decode.SCENARIOS if cached else growing.WORKLOADS
    require(bool(workloads) and workloads <= set(allowed), "unknown workload")
    require(keys == {f"{w}/{p}" for w in workloads for p in policies}, "partial configuration coverage")
    if cached:
        require(workloads == set(decode.SCENARIOS), "partial scenario coverage")
        require(report["sampling"]["scenarios"] == {k: list(v) for k, v in decode.SCENARIOS.items()},
                "sampling scenarios changed")
        require(report["sampling"]["temperature"] == .7 and report["sampling"]["top_p"] == .9,
                "sampling configuration changed")
        require(report["storage_bounds"] == decode.OWNER_LIMITS, "storage bounds changed")
        require(report["api_boundary"].get("structurally_ready") is True, "failed API boundary")
    for name in ("config_hashes", "memory_admission", "semantic_geometry"):
        require(set(report[name]) == keys, f"partial {name} coverage")
    from pllm import Experiment
    normalized = []
    for key, spec in experiments.items():
        require(hashlib.sha256(Experiment.from_spec(spec).canonical_bytes()).hexdigest()
                == report["config_hashes"][key], "configuration digest mismatch")
        work, policy = key.split("/")
        clean = deepcopy(spec)
        components = clean["pipeline"]["components"]
        inventory = components["inventory"]["params"]
        epoch = inventory.pop("decode_epoch_tokens", None)
        schedule = inventory.pop("decode_epoch_schedule", None)
        if cached:
            require(epoch == decode.POLICIES[policy][0] and schedule is None, "policy epoch mismatch")
            cache = components["cache"]["params"]
            require(cache.pop("decode", False) == decode.POLICIES[policy][1], "decode policy mismatch")
            bound = cache.pop("decode_max_bytes", None)
            require(bound == (decode.DECODE_MAX_BYTES if policy == "decode" else None), "decode bound mismatch")
        else:
            expected = growing.VARIANTS[policy]
            require(epoch == expected.get("decode_epoch_tokens")
                    and schedule == (list(expected["decode_epoch_schedule"])
                                     if "decode_epoch_schedule" in expected else None), "growing policy mismatch")
        require(clean["budget"] == {"requests": 3 if cached else 1, "max_input_tokens": 128,
                                    "max_new_tokens": 8 if cached or report["tiny"] else 64},
                "declared request allowance mismatch")
        normalized.append(clean)
        admission = report["memory_admission"][key]
        require(admission.get("admitted") is True and not admission.get("reasons"), "failed admission")
        geometry = report["semantic_geometry"][key]
        for name in ("plan_digest", "schedule_digest"):
            digest(geometry.get(name))
        require(bool(geometry["stages"]), "missing semantic geometry")
    require(all(spec == normalized[0] for spec in normalized), "source or non-policy configurations differ")
    stage_geometries = [g["stages"] for g in report["semantic_geometry"].values()]
    require(all(g == stage_geometries[0] for g in stage_geometries), "semantic body geometries differ")
    results = report["results"]
    seen = set()
    wan_settings = None
    for cell in results:
        identity = (cell["workload"], cell["variant"], cell["replicate"])
        require(type(identity[2]) is int and identity[2] >= 0, "invalid replicate")
        require(identity not in seen, "duplicate cohort cell")
        seen.add(identity)
        key = "/".join(identity[:2])
        require(key in keys, "undeclared cell")
        client = cell["client"]
        require(not client.get("failure"), "failed client")
        for field in ("client_lifetime_cpu_seconds", "client_peak_rss_bytes", "client_window_seconds"):
            number(client.get(field), field)
        for field in ("aggregate_cleanup_cpu_seconds", "complete_case_seconds"):
            number(cell.get(field), field)
        for field in ("provider_birth_cpu_seconds", "provider_sampled_peak_rss_bytes"):
            require(set(cell[field]) == {"preparation", "inference"}, "partial provider resources")
            for value in cell[field].values():
                number(value, field)
        require(client.get("cleanup", {}).get("known_private_containers_empty") is True, "private cleanup failed")
        check_ledger(client["ledger"], growing.validate_audit(client["audit"]))
        for field in COUNTERS:
            number(client["audit"].get(field), field, integer=True)
        require(cell["admitted_host_bytes"] == report["memory_admission"][key]["required_bytes"],
                "cell admission mismatch")
        wan = cell["wan"]
        require(wan.get("enforced") is True, "WAN not enforced")
        settings = {k: wan[k] for k in ("schema", "backend", "conditions", "conditions_digest",
                    "role_parties", "burst_bytes", "read_chunk_bytes", "max_connections", "scope", "excluded")}
        settings["party_access_rates"] = {party: {direction: counters["bytes_per_second"]
            for direction, counters in access.items()} for party, access in wan.get("parties", {}).items()}
        digest(wan["conditions_digest"])
        wan_settings = settings if wan_settings is None else wan_settings
        require(settings == wan_settings, "mixed WAN settings or scopes")
        from pllm.deployment import WanConditions
        require(wan["conditions"] == WanConditions().to_spec()
                and wan["conditions_digest"] == WanConditions().digest, "WAN conditions identity mismatch")
        number(wan["total_stream_bytes"], "TCP total", integer=True)
        for value in wan["directed_stream_bytes"].values():
            number(value, "directed TCP bytes", integer=True)
        require(wan["total_stream_bytes"] == sum(wan["directed_stream_bytes"].values()), "TCP conservation failed")
        for index, row in enumerate(client["responses"]):
            require(row["index"] == index and row["status"] in {"completed", "incomplete"}, "invalid response status/order")
            number(row["input_tokens"], "input tokens", integer=True)
            number(row["output_tokens"], "output tokens", integer=True)
            for witness in ("bits", "tokens", "pending", "output", "history", "rendered_history"):
                digest(row["state"].get(witness))
            require(row["output_tokens"] <= experiments[key]["budget"]["max_new_tokens"], "output cap exceeded")
            number(row["request_seconds"], "request seconds")
            for rate in ("first_text_seconds", "decode_tps"):
                if row.get(rate) is not None:
                    number(row[rate], rate)
            if cached:
                from pllm.runtime.topology_accounting import prepared_body_accounting
                check_ledger(row["request_ledger"], prepared_body_accounting({"privacy": row["audit_delta"]}))
                old = deepcopy(row.get("semantic_work"))
                work = decode.validate_request(row, cell["variant"], report["semantic_geometry"][key]["stages"])
                require(old == work, "tampered actual work")
                for field in COUNTERS:
                    number(row["audit_delta"].get(field), field, integer=True)
            else:
                work = growing.work_counts(report["semantic_geometry"][key]["stages"], row["input_tokens"],
                                           client["audit"]["generation_executed_decode_steps"])
                require(cell["semantic_work"] == work and work["remote_90_percent_verified"] is True,
                        "tampered semantic geometry/work")
        if cached:
            for field in decode.STORAGE_FIELDS:
                number(client["storage_lifetime"].get(field), field, integer=True)
            for field in decode.STORAGE_COUNTERS:
                lifetime = number(client["storage_lifetime"].get(field), field, integer=True)
                residual = number(client["storage_setup_cleanup_or_failed_request_residual"].get(field), field, integer=True)
                scoped = [number(r["disk_io"].get(field), field, integer=True) for r in client["responses"]]
                require(lifetime == sum(scoped) + residual, "disk window conservation failed")
            for field in COUNTERS + tuple(decode.WORK_COUNTERS.values()) + tuple(decode.CACHE_COUNTERS.values()):
                require(sum(r["audit_delta"][field] for r in client["responses"]) <= client["audit"][field],
                        "request counters exceed lifetime")
            for field in tuple(decode.WORK_COUNTERS.values()) + tuple(decode.CACHE_COUNTERS.values()) + (
                    "generation_executed_decode_steps", "generation_output_tokens"):
                require(sum(r["audit_delta"][field] for r in client["responses"]) == client["audit"][field],
                        "lifetime work/request conservation failed")
    reps = {identity[2] for identity in seen}
    require(reps == set(range(len(reps))) and bool(reps), "missing replicate")
    require(len(reps) >= (1 if report["tiny"] else 3), "too few replicates")
    require(seen == {(w, p, r) for w in workloads for p in policies for r in reps}, "partial cohort coverage")
    planned = "planned_clients" if cached else "planned_cells"
    require(report[planned] == len(seen), "planned cell count mismatch")
    if cached:
        require(report["planned_requests"] == 3 * len(seen), "planned request count mismatch")
    attempts = report["attempts"]
    identities = [(a["scenario" if cached else "workload"], a["policy" if cached else "variant"], a["replicate"])
                  for a in attempts]
    require(len(identities) == len(seen) and set(identities) == seen
            and all(a["status"] == "completed" for a in attempts), "incomplete attempt coverage")
    if cached:
        checks = decode.verify_results(results, report["semantic_geometry"], require_complete=True, replicates=len(reps))
    else:
        checks = growing.verify_results(results, tiny=report["tiny"])
        checks["remote_90_percent_verified_all_cells"] = True
    require(report["checks"] == checks, "declared checks mismatch")
    return cached, wan_settings


def traffic(ledgers, output_tokens=None):
    parts = [ledger["traffic_attribution"] for ledger in ledgers]
    fields = ("total_all_link_body_bytes", "known_base_all_link_body_bytes",
              "known_output_dependent_all_link_body_bytes", "unattributed_all_link_body_bytes")
    # Sums conserve exactly; independent medians need not, so keep both explicitly.
    totals = {field: sum(part[field] for part in parts) for field in fields}
    uniform = bool(output_tokens) and len(set(output_tokens)) == 1 and output_tokens[0] > 0
    known_rate = median([p[fields[2]] / n for p, n in zip(parts, output_tokens)]) if uniform else None
    return {"sample_count": len(parts), "cohort_sums": totals,
            "medians_not_additive": {field: median([p[field] for p in parts]) for field in fields},
            "complete_base_body_bytes": None if totals[fields[-1]] else median([p[fields[1]] for p in parts]),
            "complete_output_body_bytes": None if totals[fields[-1]] else median([p[fields[2]] for p in parts]),
            "known_output_bytes_per_generated_token": known_rate,
            "complete_output_bytes_per_token": None if totals[fields[-1]] else known_rate,
            "per_token_ranking_eligible": uniform and not totals[fields[-1]],
            "scoped_ledger_medians": numeric_tree_summary(ledgers)}


def summarize(report, raw_sha256):
    cached, wan = validate(report)
    groups = {}
    for key in sorted(report["experiments"]):
        cells = [c for c in report["results"] if f'{c["workload"]}/{c["variant"]}' == key]
        clients = [c["client"] for c in cells]
        peaks = [c["client_peak_rss_bytes"] for c in clients]
        requests = []
        for index in range(3 if cached else 1):
            rows = [c["responses"][index] for c in clients]
            output_counts = sorted({r["output_tokens"] for r in rows})
            requests.append({"index": index, "window": "cold" if index == 0 else "later",
                "sample_count": len(rows), "output_token_counts": output_counts,
                "per_token_ranking_eligible": (len(output_counts) == 1 and output_counts[0] > 1
                                              and all(r.get("decode_tps") is not None for r in rows)),
                "median_request_seconds": median([r["request_seconds"] for r in rows]),
                "median_ttft_seconds": median([r.get("first_text_seconds") for r in rows]),
                "median_charged_n_minus_1_tps": median([r.get("decode_tps") for r in rows]) if len(output_counts) == 1 else None,
                "counter_medians": numeric_tree_summary([{k: r["audit_delta"].get(k) for k in COUNTERS} for r in rows]) if cached else None,
                "actual_work_medians": numeric_tree_summary([r["semantic_work"] for r in rows]) if cached else None,
                "disk_io_medians": numeric_tree_summary([r["disk_io"] for r in rows]) if cached else None,
                "covered_bodies": traffic([r["request_ledger"] for r in rows], [r["output_tokens"] for r in rows]) if cached else None})
        groups[key] = {"client_samples": len(cells), "request_samples": sum(len(c["responses"]) for c in clients),
            "median_client_lifetime_cpu_seconds": median([c["client_lifetime_cpu_seconds"] for c in clients]),
            "median_client_lifetime_peak_rss_bytes": median(peaks), "largest_client_lifetime_peak_rss_bytes": max(peaks),
            "low_rss_gate": max(peaks) <= LOW_BYTES, "low_threshold_bytes": LOW_BYTES,
            "low_tier_checkpoint_qualification": False if report["tiny"] else None,
            "median_preparation_birth_cpu_seconds": median([c["provider_birth_cpu_seconds"]["preparation"] for c in cells]),
            "median_preparation_sampled_peak_rss_bytes": median([c["provider_sampled_peak_rss_bytes"]["preparation"] for c in cells]),
            "median_complete_cell_seconds": median([c["complete_case_seconds"] for c in cells]),
            "median_aggregate_cleanup_cpu_seconds": median([c["aggregate_cleanup_cpu_seconds"] for c in cells]),
            "median_all_link_tcp_payload_bytes": median([c["wan"]["total_stream_bytes"] for c in cells]),
            "covered_bodies": traffic([c["ledger"] for c in clients], [sum(r["output_tokens"] for r in c["responses"]) for c in clients]),
            "lifetime_counter_medians": numeric_tree_summary([{k: c["audit"][k] for k in COUNTERS} for c in clients]),
            "semantic_work_medians": numeric_tree_summary([c["semantic_work"] for c in cells]) if not cached else None,
            "storage_lifetime_medians": numeric_tree_summary([c["storage_lifetime"] for c in clients]) if cached else None,
            "storage_setup_cleanup_residual_medians": numeric_tree_summary([c["storage_setup_cleanup_or_failed_request_residual"] for c in clients]) if cached else None,
            "requests": requests}
    policy_peaks = {policy: max(c["client"]["client_peak_rss_bytes"] for c in report["results"]
                               if c["variant"] == policy)
                    for policy in (decode.POLICIES if cached else growing.VARIANTS)}
    result = {"schema": "pllm.decode_frontier_sdk_summary.v1", "raw_evidence_sha256": raw_sha256,
            "raw_schema": report["schema"], "evidence_class": "functionality only; not Low-tier checkpoint qualification" if report["tiny"] else "checkpoint cohort",
            **{k: deepcopy(report[k]) for k in ("build", "source_lock_digest", "cohort_commitment", "frozen_sources",
                "config_hashes", "experiments", "semantic_geometry", "execution_isolation", "windows", "limits", "checks")},
            "wan_settings": wan, "client_samples": len(report["results"]),
            "priorities": ["client lifetime peak RSS", "client lifetime CPU", "Preparation birth CPU",
                           "Preparation sampled peak RSS", "time to first text", "charged N-1 decode throughput",
                           "all-link traffic, with base/output-dependent/residual attribution"],
            "largest_client_peak_by_policy": policy_peaks,
            "observed_low_rss_gate_by_policy": {policy: peak <= LOW_BYTES for policy, peak in policy_peaks.items()},
            "low_gate_scope": ("largest recorded client lifetime peak across every replicate and workload; "
                               "observed RSS only, not conservative runtime or whole-device admission; "
                               "tiny fixtures do not qualify checkpoint execution"),
            "unknown": {"physical_wire_bytes": None, "overall_peak_memory_bytes": None,
                        "publisher_artifact_distribution_bytes": None, "publisher_artifact_disk_bytes": None},
            "groups": groups}
    for key in ("sampling", "api_boundary", "storage_bounds", "memory_admission", "required_host_bytes", "memory_guard"):
        if key in report:
            result[key] = deepcopy(report[key])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("--output must be new")
    raw = args.input.read_bytes()
    try:
        result = summarize(json.loads(raw), hashlib.sha256(raw).hexdigest())
    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(f"cohort rejected: {error}")
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        with args.output.open("x") as destination:
            destination.write(text)


if __name__ == "__main__":
    main()
