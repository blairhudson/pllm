"""Validate canonical single-response WAN reports and retain scoped comparison data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pllm.metrics import communication_per_token, wan_readiness


def _candidates(paths):
    for path in paths:
        raw = path.read_bytes()
        document = json.loads(raw)
        if "candidates" not in document:
            raise ValueError("use canonical multi-Experiment reports with one shared private cohort salt")
        for candidate in document["candidates"]:
            yield path, raw, candidate


def summarize(paths):
    rows, identities, source_groups = [], set(), {}
    for path, raw, candidate in _candidates(paths):
        report = candidate["report"]
        if not report.get("checks", {}).get("passed") or len(report["runs"]) != 1:
            raise ValueError("comparison requires checked single-response reports")
        config, run = report["configuration"], report["runs"][0]
        graph = ",".join(sorted(config["roles"]))
        source_groups.setdefault(graph, set()).add(config["source_lock_digest"])
        if run["status"] != "completed" or not run["tokens"]["authoritative"]:
            raise ValueError("comparison requires completed authoritative token usage")
        measured = wan_readiness(report)["emulation"]["runs"][0]
        communication = communication_per_token(report)["summary"]
        output = run["generation"]["output_text_digest"]
        if not output:
            raise ValueError("comparison requires an output digest")
        identity = {"model": run["model_id"], "body_fingerprint": run["model_fingerprint"],
            "model_source": candidate["pipeline"]["model"], "prompt_digest": config["prompt_digest"],
            "quantization": candidate["pipeline"]["components"].get("quantization"),
            "input_tokens": run["tokens"]["input_tokens"], "output_tokens": run["tokens"]["output_tokens"],
            "output_cap": run["max_output_tokens"], "sampling": run["sampling"], "warm": run["warm"],
            "wan_conditions_digest": config["wan_emulation_digest"],
            "link_conditions_digest": config["link_conditions_digest"], "output_text_digest": output}
        identities.add(json.dumps(identity, sort_keys=True))
        processes = run.get("processes", {})
        cpus = {role: data.get("cpu_seconds") for role, data in processes.items()}
        placement = report.get("client_body_placement", {})
        rows.append({"name": candidate["name"], "file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), **measured,
            "source_lock_digest": config["source_lock_digest"], "role_graph": graph,
            "ttft_seconds": run["durations"]["ttft_seconds"],
            "preparation_seconds": run["durations"]["preparation_seconds"],
            "covered_setup_body_bytes": round(communication["setup_inclusive_mb_per_output_token"]
                                              * run["tokens"]["output_tokens"] * 1_000_000),
            "online_body_bytes": round(communication["online_mb_per_output_token"]
                                       * run["tokens"]["output_tokens"] * 1_000_000),
            "request_cpu_seconds_by_role": cpus,
            "request_cpu_seconds_total": sum(cpus.values()) if cpus and all(v is not None for v in cpus.values()) else None,
            "client_body_i8_weight_bytes": placement.get("client_body_i8_weight_bytes"),
            "client_body_scale_bytes": placement.get("client_body_scale_bytes"),
            "native_body_i8_snapshot_bytes": placement.get("native_body_i8_snapshot_bytes"),
            "peak_client_memory_bytes": placement.get("peak_client_memory_bytes"),
            "runtime_image_id": report["docker_accounting"]["samples"]["runs"][0]["after"]["image_id"]})
    if len(identities) != 1:
        values = [json.loads(value) for value in identities]
        differing = [key for key in values[0] if len({json.dumps(v[key], sort_keys=True) for v in values}) != 1]
        raise ValueError("WAN cohort fields differ: " + ", ".join(differing))
    if any(len(values) != 1 for values in source_groups.values()):
        raise ValueError("source lock changed within one role graph")
    return {"schema": "pllm.wan_tps_comparison.v1", "cohort": json.loads(identities.pop()), "runs": rows,
        "source_locks_by_role_graph": {key: next(iter(value)) for key, value in source_groups.items()},
        "all_outputs_match": True, "scope": "single-sample cold requests under local enforced shared WAN rates/delay",
        "limitations": ["Single samples and limited repeats, not a latency distribution",
            "Provider startup/checkpoint distribution excluded from request timing",
            "Request CPU excludes deployment startup, helpers and uncharged kernel/system work",
            "Application bodies and routed IP hooks are not full physical-wire measurements",
            "Client peak memory and independent provider operation are unmeasured"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.reports)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in result["runs"]:
        print(f"{row['name']}: full={row['full_seconds']:.2f}s ttft={row['ttft_seconds']:.2f}s "
              f"e2e={row['end_to_end_tokens_per_second']:.4f} decode={row['decode_tokens_per_second']:.4f} "
              f"client_cpu={row['request_cpu_seconds_by_role']['client']:.2f}s "
              f"aggregate_cpu={row['request_cpu_seconds_total']:.2f}s")
