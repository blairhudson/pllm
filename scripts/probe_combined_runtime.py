"""Bounded, source-locked current optimisation comparison using the canonical driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.benchmarks.combined_runtime import CANDIDATES, CONTEXTS, SOURCE
from pllm.metrics import communication_per_token
from pllm import _native
from pllm.runtime.benchmark_cli import (
    accounted_benchmark_body_totals,
    build_comparison_report,
    run_loopback_benchmark,
)


def summarise(paths):
    records = [json.loads(path.read_text()) for path in paths]
    names = [record["candidate"] for record in records]
    if len(set(names)) != len(names) or "baseline" not in names:
        raise ValueError("summary requires distinct candidates and baseline")
    candidates = [(CANDIDATES[row["candidate"]], row["report"]) for row in records]
    for record, (experiment, _) in zip(records, candidates, strict=True):
        if record["experiment"] != experiment.to_spec():
            raise ValueError("recorded Experiment does not match the current candidate")
        if record["source_revision"] != SOURCE.revision:
            raise ValueError("source revision mismatch")
    comparison = build_comparison_report(candidates)
    for key in ("all_candidates_passed", "unique_configurations", "matched_workload"):
        if not comparison["checks"][key]:
            raise ValueError(f"cohort mismatch: {key}")
    cpu_comparison = build_comparison_report(
        [
            (experiment, report)
            for experiment, report in candidates
            if experiment.pipeline.components["kernels"].component == "pllm/cpu"
        ]
    )
    baseline = records[names.index("baseline")]["report"]
    reference_outputs = [run["generation"]["output_text_digest"] for run in baseline["runs"]]
    rows = []
    for path, record in zip(paths, records, strict=True):
        report = record["report"]
        token_rates = communication_per_token(report)
        if [run["generation"]["output_text_digest"] for run in report["runs"]] != reference_outputs:
            raise ValueError(f"output text mismatch: {record['candidate']}")
        runs = []
        for run, accounting in zip(
            report["runs"], report["topology_accounting"]["runs"], strict=True
        ):
            runs.append(
                {
                    "input_tokens": run["tokens"]["input_tokens"],
                    "output_tokens": run["tokens"]["output_tokens"],
                    "output_text_digest": run["generation"]["output_text_digest"],
                    "full_seconds": run["durations"]["full_seconds"],
                    "online_seconds": run["durations"]["online_seconds"],
                    "ttft_seconds": run["durations"]["ttft_seconds"],
                    "decode_tokens_per_second": run["durations"]["tokens_per_second"],
                    "online_body_bytes": accounting["online_all_link_serialized_body_bytes"],
                    "all_link_body_bytes": accounting["all_link_serialized_body_bytes"],
                    "client_cpu_seconds": run["processes"]["client"]["cpu_seconds"],
                    "aggregate_cpu_seconds": accounting["aggregate_run_window_cpu_seconds"],
                    "prefix_tokens_reused": run["privacy"].get("prefill_prefix_tokens_reused", 0),
                    "cache_hits": run["privacy"].get("prefill_cache_hits", 0),
                    "batched_continuation_hits": run["privacy"].get(
                        "kv_continuation_batched_hits", 0
                    ),
                }
            )
        summary = report["summary"]
        rows.append(
            {
                "candidate": record["candidate"],
                "evidence_path": path.as_posix(),
                "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "runs": runs,
                "total_accounted_body_bytes": summary["total_accounted_benchmark_body_bytes"],
                "cold_first_response_body_bytes": summary[
                    "accounted_setup_through_first_response_body_bytes"
                ],
                "total_online_body_bytes": summary[
                    "total_run_online_all_link_serialized_body_bytes"
                ],
                "total_request_seconds": summary["total_run_full_seconds"],
                "cold_process_cpu": report.get("process_cpu_accounting"),
                "client_body_placement": report.get("client_body_placement"),
                "prepared_material_accounting": report.get("prepared_material_accounting"),
                "communication_per_token": token_rates,
            }
        )
    return {
        "schema": "pllm.combined_runtime_summary.v1",
        "scope": "One sample per fresh/repeated/extended request; exact token counts in each row; application bodies; cached checkpoint; co-located roles; request latency excludes role startup",
        "checks": comparison["checks"] | {"matching_output_text": True},
        "cpu_comparison_checks": cpu_comparison["checks"],
        "candidates": rows,
        "limitations": [
            "Output agreement is relative to the same W8A8 numeric mode, not independent generation-quality evidence.",
            "Client peak memory, full wire bytes and GPU compute are unmeasured.",
            "Offset setup-inclusive totals remain unknown in the canonical report.",
            "CPU and GPU candidates are not ranked together by the canonical comparator.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--candidate", choices=CANDIDATES)
    action.add_argument("--summarise", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, choices=range(1, 33), default=32)
    args = parser.parse_args()
    if args.summarise:
        result = summarise(args.summarise)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "checks": result["checks"],
                    "cpu_comparison_checks": result["cpu_comparison_checks"],
                    "candidates": [
                        {
                            key: row[key]
                            for key in (
                                "candidate",
                                "total_accounted_body_bytes",
                                "total_online_body_bytes",
                                "cold_first_response_body_bytes",
                                "total_request_seconds",
                            )
                        }
                        for row in result["candidates"]
                    ],
                },
                indent=2,
            )
        )
        return
    experiment = CANDIDATES[args.candidate]
    experiment.resolve()
    print(f"Running {experiment.name}: fresh, repeat, extension; {args.max_output_tokens} outputs each", flush=True)
    report = run_loopback_benchmark(
        model=SOURCE.source,
        model_id=experiment.resolve().model,
        tiny=False,
        prompt=CONTEXTS[0],
        prompt_sequence=CONTEXTS,
        max_output_tokens=args.max_output_tokens,
        warmups=0,
        repetitions=1,
        timeout_seconds=180,
        experiment=experiment,
        _cohort_salt=b"pllm.public.combined-runtime.2026-10-02.v1",
        temperature=0,
        capture_output_digest=True,
        progress=lambda message: (
            print(message, flush=True) if "running" not in message.lower() else None
        ),
    )
    if not report["checks"]["passed"]:
        raise ValueError("canonical benchmark checks failed")
    result = {
        "schema": "pllm.combined_runtime_candidate.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": args.candidate,
        "source_revision": SOURCE.revision,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "code_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "examples/benchmarks/combined_runtime.py",
                "scripts/probe_combined_runtime.py",
                "python/pllm/runtime/client.py",
                "python/pllm/runtime/model_binding.py",
                "python/pllm/runtime/transformer_client.py",
                "python/pllm/runtime/transformer_engine.py",
                "python/pllm/runtime/metal.py",
                "python/pllm/runtime/benchmark_cli.py",
                "python/pllm/metrics/communication.py",
                "python/pllm/runtime/bundle_artifacts.py",
                "python/pllm/runtime/bundle_compression.py",
                "python/pllm/runtime/continuation_admission.py",
                "python/pllm/runtime/offset_reference.py",
                "python/pllm/runtime/offset_worker.py",
                "python/pllm/runtime/prefill_cache.py",
                "python/pllm/runtime/semantic_executor.py",
                "python/pllm/profiles/__init__.py",
            )
        },
        "native_binary_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        "scope": "Three ordered requests, single sample per position, co-located roles; isolated cold client cache; cached checkpoint; application bodies",
        "experiment": experiment.to_spec(),
        "report": report,
        "bodies": accounted_benchmark_body_totals(report.get("topology_accounting") or {}),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {"candidate": args.candidate, "bodies": result["bodies"], "summary": report["summary"]}
        )
    )


if __name__ == "__main__":
    main()
