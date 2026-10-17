"""Serial five-way matched installed-SDK screen. Preflight never loads tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

# Import only repository study/recipe helpers; never put python/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock, Timeout
from pllm import Experiment, Model
from pllm.metrics import BuildRef, benchmark_suite

from benchmarks.research.sota_combinations import (
    PLACEMENT_PLAN, Rejected, atomic_json, audit, hashes, placement_pairs, preflight,
    public_input_counts, study,
)


def read_fixture(path):
    with path.open("rb") as source:
        data = source.read(4097)
    if not data or len(data) > 4096:
        raise Rejected("public_fixture_bound")
    return data.decode("utf-8"), hashlib.sha256(data).hexdigest()


def fresh_output_paths(output):
    """Never reuse initial-cohort evidence or partially written rerun artifacts."""
    paths = (output, output.with_suffix(".plan.json"), output.with_suffix(".progress.json"),
             output.with_suffix(".summary.json"))
    if any(path.exists() for path in paths):
        raise FileExistsError("output or companion evidence already exists")
    return paths[1:]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--placement-plan", type=Path, default=PLACEMENT_PLAN)
    parser.add_argument("--prompt-150-file", type=Path)
    parser.add_argument("--prompt-288-file", type=Path)
    parser.add_argument("--workload", choices=("150", "288"), action="append")
    parser.add_argument("--processes", type=int, default=3,
                        help="Fresh serial replicates per case, not parallel concurrency")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-plan", type=Path, help="Require exact public replay identities")
    args = parser.parse_args(argv)
    if not 1 <= args.processes <= 6:
        parser.error("--processes must be in [1,6]")
    if args.tiny and args.workload:
        parser.error("--tiny cannot select production workloads")
    plan_path, progress_path, summary_path = fresh_output_paths(args.output)
    progress = {"schema": "pllm.sota_combinations_progress.v1", "status": "preflight",
                "failure_code": None, "measurement_complete": False}
    atomic_json(progress_path, progress)
    lock_path = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Fail fast even for admission: other serialized cohorts may own source caches.
        with FileLock(str(lock_path), timeout=0):
            workloads = ("tiny",) if args.tiny else tuple(args.workload or ("150", "288"))
            source_paths = [Path(__file__), Path("benchmarks/research/sota_combinations.py"),
                            Path("examples/prepared_sdk.py")]
            inputs, fixtures = {}, {}
            pairs = record = tokenizer = None
            if args.tiny:
                model = Model.tiny()
                inputs["tiny"] = {"prompt": "abcde " * 14}
                fixtures["tiny"] = {"kind": "public_generated_transport_fixture"}
            else:
                pairs, record = placement_pairs(args.placement_plan)
                indexed = args.artifacts / "indexed.json"
                base = Experiment.from_spec(json.loads(indexed.read_bytes()))
                model, tokenizer = base.pipeline.model, base.pipeline.components["tokenizer"]
                source_paths += [args.placement_plan, indexed]
                for workload in workloads:
                    path = ((args.prompt_150_file or args.artifacts / "prompt.txt")
                            if workload == "150" or args.prompt_288_file is None else args.prompt_288_file)
                    text, digest = read_fixture(path)
                    if workload == "288" and args.prompt_288_file is None:
                        text = text + "\n" + text + "\n" + (
                            "Explain why the sky is blue, then describe how its colour changes at sunset.")
                    if len(text.encode()) > 4096:
                        raise Rejected("public_fixture_bound")
                    inputs[workload] = {"prompt": text}
                    fixtures[workload] = {"path": str(path), "source_sha256": digest,
                        "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "derivation": "double-medium-plus-sky-v1" if workload == "288" and args.prompt_288_file is None else "identity",
                        "expected_input_tokens": int(workload)}
                    source_paths.append(path)
            frozen = hashes(source_paths)
            current = study(model=model, tokenizer=tokenizer, build=BuildRef.capture(),
                workloads=workloads, processes=args.processes, placement=pairs, tiny=args.tiny)
            work = current.plan(max_trials=256)
            if work.suite is None or len(work.suite.cases) != len(workloads) * 5:
                raise Rejected("five_way_compatibility")
            # benchmark_suite generates one private salt for all cases/replicates.
            # Only salted commitments and public fixture hashes enter evidence.
            plan = {"study": current.to_spec(), "suite": work.suite.to_spec(),
                    "coverage": work.coverage.to_dict(), "public_fixtures": fixtures,
                    "frozen_sources": frozen, "tiny": args.tiny,
                    "source_lock_digest": None if args.tiny else record["source_lock_digest"],
                    "execution": "serial balanced fresh workers; private salt generated by benchmark_suite"}
            atomic_json(plan_path, plan)
            if args.replay_plan:
                old = json.loads(args.replay_plan.read_bytes())
                if any(old.get(key) != plan[key] for key in
                       ("study", "suite", "public_fixtures", "frozen_sources", "tiny", "source_lock_digest")):
                    raise Rejected("exact_replay_mismatch")
            if hashes(source_paths) != frozen:
                raise Rejected("sources_changed_during_preflight")
            counts, actual_source = public_input_counts(model, tokenizer, inputs)
            if not args.tiny and actual_source != record["source_lock_digest"]:
                raise Rejected("placement_source_lock_mismatch")
            plan["public_input_counts"] = counts

            def checkpoint_admission(records):
                plan["preflight"] = records
                atomic_json(plan_path, plan)

            preflight(work.suite, tiny=args.tiny, placement_record=record,
                      input_counts=counts, checkpoint=checkpoint_admission)
            if hashes(source_paths) != frozen:
                raise Rejected("sources_changed_during_preflight")
            atomic_json(plan_path, plan)
            progress["status"] = "admitted"
            atomic_json(progress_path, progress)
            if args.preflight_only:
                print(json.dumps({"status": "admitted", "plan": str(plan_path),
                                  "cases": len(work.suite.cases)}, indent=2))
                return 0
            progress["status"] = "running"
            atomic_json(progress_path, progress)
            matched_inputs = {case.name: inputs[case.name.split("/")[0]] for case in work.suite.cases}
            run = benchmark_suite(work.suite, inputs=matched_inputs, output=args.output)
            if hashes(source_paths) != frozen:
                raise Rejected("sources_changed_during_run")
            summary = audit(run, tiny=args.tiny, placement_record=record)
            atomic_json(summary_path, summary)
            progress.update(status="completed", measurement_complete=True)
            progress["ranking"] = summary["ranking"]
            atomic_json(progress_path, progress)
            print(json.dumps({"status": "completed", "summary": str(summary_path)}, indent=2))
            return 0
    except BaseException as exc:
        code = ("frontier_cohort_lock_busy" if isinstance(exc, Timeout) else
                str(exc) if isinstance(exc, Rejected) else
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "preflight_or_execution_error")
        progress.update(status="rejected", failure_code=code)
        # Actual run atomically checkpoints each cell itself, including interruptions.
        if args.output.exists():
            partial = json.loads(args.output.read_bytes())
            progress["retained_results"] = len(partial["results"])
            progress["measurement_complete"] = partial["complete"]
        atomic_json(progress_path, progress)
        print(json.dumps(progress, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
