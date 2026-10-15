"""Fresh five-cell ordinary SDK private-boundary decode comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock, Timeout
from pllm.metrics import BuildRef, benchmark_suite

from benchmarks.research.private_boundary_decode import audit, matched_suite, preflight
from benchmarks.research.sota_combinations import Rejected, atomic_json, hashes, public_input_counts
from scripts.benchmark_sota_combinations import fresh_output_paths, read_fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--tiny", action="store_true", help="generated source, 128+2 smoke; not 150+8 evidence")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    plan_path, progress_path, summary_path = fresh_output_paths(args.output)
    progress = {"schema": "pllm.private_boundary_decode_progress.v1", "status": "preflight"}
    atomic_json(progress_path, progress)
    lock = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock), timeout=0):
            sources = [Path(__file__), Path("benchmarks/research/private_boundary_decode.py"),
                Path("benchmarks/research/private_boundary_tiers.py"),
                Path("benchmarks/research/historical_evidence.py"), Path("benchmarks/research/sota_combinations.py"),
                Path("scripts/benchmark_sota_combinations.py"), Path("examples/prepared_sdk.py"),
                Path("docs/evidence/private-boundary-qwen-control.json")]
            if args.tiny:
                from pllm import Model
                from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
                fixture = create_tiny_llama_checkpoint(args.output.with_suffix(".model"),
                    model_type="qwen2", with_qkv_bias=True)
                generated_source = Model.path(str(fixture.resolve()), model_id="decode-audit-tiny")
                prompt = "abcde " * 14
            else:
                generated_source = None
                prompt, _ = read_fixture(args.artifacts / "prompt.txt")
                sources += [args.artifacts / "prompt.txt", args.artifacts / "indexed.json"]
            frozen = hashes(sources)
            suite = matched_suite(build=BuildRef.capture(), tiny=args.tiny,
                artifacts=args.artifacts, generated_source=generated_source)
            from pllm import Experiment
            exp = Experiment.from_spec(suite.cases[0].target.to_spec()["parameters"]["model"]["experiment"])
            counts, source = public_input_counts(exp.pipeline.model, exp.pipeline.components.get("tokenizer"),
                {"tiny" if args.tiny else "150": {"prompt": prompt}})
            if any(count > exp.budget.max_input_tokens for count in counts.values()):
                raise Rejected("public_fixture_exceeds_request_bound")
            plan = {"schema": "pllm.private_boundary_decode_plan.v1", "suite": suite.to_spec(),
                "frozen_sources": frozen, "tiny": args.tiny, "public_input_counts": counts,
                "source_lock_digest": source, "preflight": preflight(suite),
                "execution": "three balanced serial fresh clients per cell; one private salt; native shared 100/40 caps",
                "measurement_scope": "source-present; SDK/supervisor/dashboard/relay/full-state audit charged to worker"}
            atomic_json(plan_path, plan)
            if any(not row["admission"]["admitted"] for row in plan["preflight"]):
                raise Rejected("topology_memory_limit")
            if args.preflight_only:
                progress["status"] = "admitted"
            else:
                progress["status"] = "running"
                atomic_json(progress_path, progress)
                run = benchmark_suite(suite, inputs={case.name: {"prompt": prompt} for case in suite.cases},
                    output=args.output)
                if hashes(sources) != frozen:
                    raise Rejected("sources_changed_during_run")
                summary = audit(run.to_dict(), tiny=args.tiny)
                atomic_json(summary_path, summary)
                progress.update(status="completed", clients=summary["clients"],
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
