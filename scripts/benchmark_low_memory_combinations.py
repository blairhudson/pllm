#!/usr/bin/env python3
"""Six matched Low-RAM recipes through the installed SDK benchmark target."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock, Timeout
from pllm import Experiment, Model
from pllm.metrics import BuildRef, benchmark_suite

from benchmarks.research import low_memory_combinations as low
from benchmarks.research.sota_combinations import (
    Rejected, atomic_json, hashes, placement_pairs, preflight, public_input_counts,
)
from scripts.benchmark_sota_combinations import fresh_output_paths, read_fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--placement-plan", type=Path,
                        default=Path("docs/evidence/research-all-link-placement-sdk-confirmation.plan.json"))
    parser.add_argument("--workload", choices=("150", "288"), action="append")
    parser.add_argument("--processes", type=int, default=3, help="Fresh serial replicates per recipe")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-plan", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.processes <= 6 or args.tiny and args.workload:
        parser.error("require 1..6 processes and no production workload with --tiny")
    workloads = ("tiny",) if args.tiny else tuple(args.workload or ("150", "288"))
    if len(set(workloads)) != len(workloads):
        parser.error("workloads must be distinct")
    plan_path, progress_path, summary_path = fresh_output_paths(args.output)
    progress = {"schema": "pllm.low_memory_combinations_progress.v1", "status": "preflight",
                "failure_code": None, "measurement_complete": False}
    atomic_json(progress_path, progress)
    lock = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock), timeout=0):
            paths = [Path(__file__), Path("benchmarks/research/low_memory_combinations.py"),
                     Path("benchmarks/research/sota_combinations.py"), Path("examples/prepared_sdk.py"),
                     Path("scripts/benchmark_sota_combinations.py"),
                     Path("python/pllm/search/client_memory_tiers.json")]
            inputs, fixtures = {}, {}
            pairs = record = tokenizer = None
            if args.tiny:
                model = Model.tiny()
                inputs["tiny"] = {"prompt": "abcde " * 14}
                fixtures["tiny"] = {"kind": "public_generated_transport_fixture"}
            else:
                pairs, record = placement_pairs(args.placement_plan)
                indexed = args.artifacts / "indexed.json"
                source = Experiment.from_spec(json.loads(indexed.read_bytes()))
                model, tokenizer = source.pipeline.model, source.pipeline.components["tokenizer"]
                prompt_path = args.artifacts / "prompt.txt"
                text, source_hash = read_fixture(prompt_path)
                paths += [args.placement_plan, indexed, prompt_path]
                for workload in workloads:
                    prompt = text if workload == "150" else text + "\n" + text + "\n" + (
                        "Explain why the sky is blue, then describe how its colour changes at sunset.")
                    if len(prompt.encode()) > 4096:
                        raise Rejected("public_fixture_bound")
                    inputs[workload] = {"prompt": prompt}
                    fixtures[workload] = {"path": str(prompt_path), "source_sha256": source_hash,
                        "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "derivation": "identity" if workload == "150" else "double-medium-plus-sky-v1",
                        "expected_input_tokens": int(workload)}
            frozen = hashes(paths)
            study = low.study(model=model, tokenizer=tokenizer, build=BuildRef.capture(), placement=pairs,
                              workloads=workloads, processes=args.processes, tiny=args.tiny)
            work = study.plan(max_trials=256)
            if work.suite is None or len(work.suite.cases) != len(workloads) * len(low.VARIANTS):
                raise Rejected("six_way_compatibility")
            plan = {"study": study.to_spec(), "suite": work.suite.to_spec(), "coverage": work.coverage.to_dict(),
                    "public_fixtures": fixtures, "frozen_sources": frozen, "tiny": args.tiny,
                    "source_lock_digest": None if args.tiny else record["source_lock_digest"],
                    "client_memory_tier": {"id": "low", "max_bytes": low.LOW_BYTES,
                        "qualification": "largest_fresh_client_peak_across_all_replicates"},
                    "execution": "serial balanced fresh workers; private salt generated by benchmark_suite"}
            atomic_json(plan_path, plan)
            if args.replay_plan:
                old = json.loads(args.replay_plan.read_bytes())
                if any(old.get(key) != plan[key] for key in (
                        "study", "suite", "public_fixtures", "frozen_sources", "tiny", "source_lock_digest",
                        "client_memory_tier")):
                    raise Rejected("exact_replay_mismatch")
            counts, source_lock = public_input_counts(model, tokenizer, inputs)
            if not args.tiny and source_lock != record["source_lock_digest"]:
                raise Rejected("placement_source_lock_mismatch")
            plan["public_input_counts"] = counts

            def checkpoint(records):
                plan["preflight"] = records
                atomic_json(plan_path, plan)

            preflight(work.suite, tiny=args.tiny, placement_record=record,
                      input_counts=counts, checkpoint=checkpoint)
            if hashes(paths) != frozen:
                raise Rejected("sources_changed_during_preflight")
            progress["status"] = "admitted"
            atomic_json(progress_path, progress)
            if args.preflight_only:
                print(json.dumps({"status": "admitted", "cases": len(work.suite.cases), "plan": str(plan_path)}))
                return 0
            progress["status"] = "running"
            atomic_json(progress_path, progress)
            matched_inputs = {case.name: inputs[case.name.split("/")[0]] for case in work.suite.cases}
            run = benchmark_suite(work.suite, inputs=matched_inputs, output=args.output)
            if hashes(paths) != frozen:
                raise Rejected("sources_changed_during_run")
            summary = low.audit(run, tiny=args.tiny, placement_record=record)
            atomic_json(summary_path, summary)
            progress.update(status="completed", measurement_complete=True, ranking=summary["ranking"])
            atomic_json(progress_path, progress)
            print(json.dumps({"status": "completed", "summary": str(summary_path)}))
            return 0
    except BaseException as exc:
        code = ("frontier_cohort_lock_busy" if isinstance(exc, Timeout) else str(exc) if isinstance(exc, Rejected)
                else "interrupted" if isinstance(exc, KeyboardInterrupt) else "preflight_or_execution_error")
        if not isinstance(exc, (Rejected, Timeout, KeyboardInterrupt)):
            traceback.print_exc(file=sys.stderr)
        progress.update(status="rejected", failure_code=code)
        if args.output.exists():
            partial = json.loads(args.output.read_bytes())
            progress.update(retained_results=len(partial["results"]), measurement_complete=partial["complete"])
        atomic_json(progress_path, progress)
        print(json.dumps(progress))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
