"""Run a fresh matched six-client private-boundary head-kernel screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock, Timeout
from pllm import Experiment
from pllm.metrics import BuildRef, benchmark_suite

from benchmarks.research.private_boundary_byte_dot import audit, matched_suite, preflight
from benchmarks.research.sota_combinations import Rejected, atomic_json, hashes, public_input_counts
from scripts.benchmark_sota_combinations import fresh_output_paths, read_fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--tiny", action="store_true", help="generated 128+2 transport smoke")
    parser.add_argument("--preflight-only", action="store_true", help="metadata admission only; no tensor loading or timings")
    args = parser.parse_args(argv)
    plan_path, progress_path, summary_path = fresh_output_paths(args.output)
    model_path = args.output.with_suffix(".model")
    if args.tiny and model_path.exists():
        raise FileExistsError("tiny model output already exists")
    progress = {"schema": "pllm.private_boundary_byte_dot_progress.v1",
        "status": "preflight", "measurement_complete": False}
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("x") as stream:
        json.dump(progress, stream, indent=2)
        stream.write("\n")
    lock = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock), timeout=0):
            sources = [Path(__file__), Path("benchmarks/research/private_boundary_byte_dot.py"),
                Path("benchmarks/research/private_boundary_decode_confirmation.py"),
                Path("benchmarks/research/private_boundary_decode.py"), Path("benchmarks/research/private_boundary_tiers.py"),
                Path("benchmarks/research/historical_evidence.py"), Path("benchmarks/research/sota_combinations.py"),
                Path("scripts/benchmark_sota_combinations.py"), Path("examples/prepared_sdk.py"),
                Path("docs/evidence/private-boundary-qwen-control.json")]
            if args.tiny:
                from pllm import Model
                from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
                if args.preflight_only:
                    generated_source = Model.tiny()
                else:
                    fixture = create_tiny_llama_checkpoint(model_path, model_type="qwen2", with_qkv_bias=True)
                    generated_source = Model.path(str(fixture.resolve()), model_id="boundary-byte-dot-tiny")
                prompt = "abcde " * 14
            else:
                generated_source = None
                prompt, _ = read_fixture(args.artifacts / "prompt.txt")
                sources += [args.artifacts / "prompt.txt", args.artifacts / "indexed.json"]
            frozen = hashes(sources)
            suite = matched_suite(build=BuildRef.capture(), tiny=args.tiny,
                artifacts=args.artifacts, generated_source=generated_source)
            exp = Experiment.from_spec(suite.cases[0].target.to_spec()["parameters"]["model"]["experiment"])
            counts, source = public_input_counts(exp.pipeline.model, exp.pipeline.components.get("tokenizer"),
                {"tiny" if args.tiny else "150": {"prompt": prompt}})
            if any(count > exp.budget.max_input_tokens for count in counts.values()):
                raise Rejected("public_fixture_exceeds_request_bound")
            if not args.tiny and set(counts.values()) != {150}:
                raise Rejected("public_fixture_not_150_tokens")
            plan = {"schema": "pllm.private_boundary_byte_dot_plan.v1", "suite": suite.to_spec(),
                "frozen_sources": frozen, "tiny": args.tiny, "public_input_counts": counts,
                "output_cap": 2 if args.tiny else 8,
                "source_lock_digest": source, "preflight": preflight(suite),
                "execution": "three balanced serial fresh clients per kernel; one new private salt; native shared 100/40 Mbps caps",
                "measurement_scope": "source-present; complete SDK/supervisor/dashboard/relay/full-state audit worker",
                "output_policy": "natural EOS; actual outputs and issued/claimed/burned/discarded material retained"}
            atomic_json(plan_path, plan)
            if hashes(sources) != frozen:
                raise Rejected("sources_changed_during_preflight")
            if any(not row["admission"]["admitted"] for row in plan["preflight"]):
                raise Rejected("topology_memory_limit")
            if args.preflight_only:
                progress["status"] = "admitted"
            else:
                progress["status"] = "running"
                atomic_json(progress_path, progress)
                run = benchmark_suite(suite, inputs={case.name: {"prompt": prompt} for case in suite.cases}, output=args.output)
                if hashes(sources) != frozen:
                    raise Rejected("sources_changed_during_run")
                summary = audit(run.to_dict(), tiny=args.tiny)
                if source is not None and summary["identity"]["source_lock_digest"] != source:
                    raise Rejected("planned_source_lock_mismatch")
                atomic_json(summary_path, summary)
                progress.update(status="completed", measurement_complete=True, clients=summary["clients"],
                    maximum_swap_growth_bytes=summary["maximum_observed_swap_growth_bytes"])
            atomic_json(progress_path, progress)
            print(json.dumps(progress, indent=2))
            return 0
    except BaseException as exc:
        code = ("frontier_cohort_lock_busy" if isinstance(exc, Timeout) else str(exc) if isinstance(exc, Rejected)
            else "interrupted" if isinstance(exc, KeyboardInterrupt) else "preflight_or_execution_error")
        progress.update(status="rejected", failure_code=code)
        atomic_json(progress_path, progress)
        print(json.dumps(progress, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
