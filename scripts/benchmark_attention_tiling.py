"""Serial fresh-client attention ablations; installed SDK and shared 100/40 caps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock, Timeout
from pllm import Experiment, Model
from pllm.metrics import BuildRef, benchmark_suite

from benchmarks.research.attention_tiling import RECIPES, admitted_suite, audit, matched_suite
from benchmarks.research.sota_combinations import (
    PLACEMENT_PLAN, Rejected, atomic_json, hashes, placement_pairs, public_input_counts,
)
from scripts.benchmark_sota_combinations import fresh_output_paths, read_fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--recipe", choices=RECIPES, action="append")
    parser.add_argument("--workload", choices=("150", "288"), action="append")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.processes <= 6 or args.tiny and args.workload:
        parser.error("use 1..6 serial replicates; tiny cannot select checkpoint workloads")
    plan_path, progress_path, summary_path = fresh_output_paths(args.output)
    progress = {"schema": "pllm.attention_tiling_progress.v1", "status": "preflight"}
    atomic_json(progress_path, progress)
    lock = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock), timeout=0):
            workloads = ("tiny",) if args.tiny else tuple(args.workload or ("150", "288"))
            recipes = tuple(args.recipe or RECIPES)
            sources = [Path(__file__), Path("benchmarks/research/attention_tiling.py"),
                Path("benchmarks/research/sota_combinations.py"), Path("examples/prepared_sdk.py")]
            inputs, fixtures = {}, {}
            pairs = placement = tokenizer = None
            if args.tiny:
                model = Model.tiny()
                inputs["tiny"] = {"prompt": "abcde " * 14}
                fixtures["tiny"] = {"kind": "public_generated_transport_fixture"}
            else:
                pairs, placement = placement_pairs()
                indexed = args.artifacts / "indexed.json"
                experiment = Experiment.from_spec(json.loads(indexed.read_bytes()))
                model, tokenizer = experiment.pipeline.model, experiment.pipeline.components["tokenizer"]
                prompt_path = args.artifacts / "prompt.txt"
                text, _ = read_fixture(prompt_path)
                sources += [PLACEMENT_PLAN, indexed, prompt_path]
                for workload in workloads:
                    prompt = text if workload == "150" else text + "\n" + text + "\n" + (
                        "Explain why the sky is blue, then describe how its colour changes at sunset.")
                    inputs[workload] = {"prompt": prompt}
                    fixtures[workload] = {"sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "expected_input_tokens": int(workload)}
            frozen = hashes(sources)
            suite = matched_suite(model=model, tokenizer=tokenizer, build=BuildRef.capture(),
                workloads=workloads, processes=args.processes, placement=pairs, tiny=args.tiny, recipes=recipes)
            counts, source_lock = public_input_counts(model, tokenizer, inputs)
            if not args.tiny and source_lock != placement["source_lock_digest"]:
                raise Rejected("placement_source_lock_mismatch")
            plan = {"schema": "pllm.attention_tiling_plan.v1", "suite": suite.to_spec(),
                "public_fixtures": fixtures, "public_input_counts": counts, "frozen_sources": frozen,
                "source_lock_digest": source_lock, "tiny": args.tiny,
                "execution": "serial balanced fresh workers; common private salt; shared native 100/40 caps"}
            def checkpoint(records):
                plan["preflight"] = records
                atomic_json(plan_path, plan)
            admitted_suite(suite, tiny=args.tiny, placement_record=placement,
                           input_counts=counts, checkpoint=checkpoint)
            if hashes(sources) != frozen:
                raise Rejected("sources_changed_during_preflight")
            if args.preflight_only:
                progress["status"] = "admitted"
            else:
                progress["status"] = "running"
                atomic_json(progress_path, progress)
                run = benchmark_suite(suite, inputs={case.name: inputs[case.name.split("/")[0]]
                    for case in suite.cases}, output=args.output)
                if hashes(sources) != frozen:
                    raise Rejected("sources_changed_during_run")
                summary = audit(run, tiny=args.tiny, placement_record=placement)
                atomic_json(summary_path, summary)
                progress.update(status="completed", clients=summary["completed_clients"])
            atomic_json(progress_path, progress)
            print(json.dumps(progress, indent=2))
            return 0
    except BaseException as exc:
        code = ("frontier_cohort_lock_busy" if isinstance(exc, Timeout) else str(exc)
                if isinstance(exc, Rejected) else "interrupted" if isinstance(exc, KeyboardInterrupt)
                else "preflight_or_execution_error")
        progress.update(status="rejected", failure_code=code)
        atomic_json(progress_path, progress)
        print(json.dumps(progress, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
