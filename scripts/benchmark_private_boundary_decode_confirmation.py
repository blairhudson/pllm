"""Run one independent three-arm private-boundary output-cap cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock, Timeout
from pllm import Experiment
from pllm.metrics import BuildRef, benchmark_suite

from benchmarks.research.private_boundary_decode_confirmation import audit, matched_suite, preflight
from benchmarks.research.sota_combinations import Rejected, atomic_json, hashes, public_input_counts
from scripts.benchmark_sota_combinations import fresh_output_paths, read_fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--max-output-tokens", choices=(8, 32, 64), type=int, default=8)
    parser.add_argument("--tiny", action="store_true", help="128+2 generated-source smoke, not production evidence")
    parser.add_argument("--preflight-only", action="store_true", help="source/tokenizer metadata only; no tensors or timing")
    args = parser.parse_args(argv)
    if args.tiny and args.max_output_tokens != 8:
        parser.error("--tiny cannot select production output workloads")
    plan_path, progress_path, summary_path = fresh_output_paths(args.output)
    model_path = args.output.with_suffix(".model")
    if args.tiny and model_path.exists():
        raise FileExistsError("tiny model output already exists")
    progress = {"schema": "pllm.private_boundary_decode_confirmation_progress.v1",
        "status": "preflight", "measurement_complete": False}
    # Exclusive reservation closes the race between the shared freshness check
    # and lock acquisition. A competing attempt must never overwrite progress.
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("x") as stream:
        json.dump(progress, stream, indent=2)
        stream.write("\n")
    lock = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock), timeout=0):
            sources = [Path(__file__), Path("benchmarks/research/private_boundary_decode_confirmation.py"),
                Path("benchmarks/research/private_boundary_decode.py"), Path("benchmarks/research/private_boundary_tiers.py"),
                Path("benchmarks/research/historical_evidence.py"), Path("benchmarks/research/sota_combinations.py"),
                Path("scripts/benchmark_sota_combinations.py"), Path("examples/prepared_sdk.py"),
                Path("docs/evidence/private-boundary-qwen-control.json")]
            if args.tiny:
                from pllm import Model
                from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
                if args.preflight_only:
                    # Metadata-only plan must not generate checkpoint tensors.
                    generated_source = Model.tiny()
                else:
                    fixture = create_tiny_llama_checkpoint(model_path, model_type="qwen2", with_qkv_bias=True)
                    generated_source = Model.path(str(fixture.resolve()), model_id="decode-confirmation-tiny")
                prompt = "abcde " * 14
            else:
                generated_source = None
                prompt, _ = read_fixture(args.artifacts / "prompt.txt")
                sources += [args.artifacts / "prompt.txt", args.artifacts / "indexed.json"]
            frozen = hashes(sources)
            suite = matched_suite(build=BuildRef.capture(), max_output_tokens=args.max_output_tokens,
                tiny=args.tiny, artifacts=args.artifacts, generated_source=generated_source)
            exp = Experiment.from_spec(suite.cases[0].target.to_spec()["parameters"]["model"]["experiment"])
            counts, source = public_input_counts(exp.pipeline.model, exp.pipeline.components.get("tokenizer"),
                {"tiny" if args.tiny else "150": {"prompt": prompt}})
            if any(count > exp.budget.max_input_tokens for count in counts.values()):
                raise Rejected("public_fixture_exceeds_request_bound")
            if not args.tiny and set(counts.values()) != {150}:
                raise Rejected("public_fixture_not_150_tokens")
            plan = {"schema": "pllm.private_boundary_decode_confirmation_plan.v1", "suite": suite.to_spec(),
                "frozen_sources": frozen, "tiny": args.tiny, "output_cap": exp.budget.max_new_tokens,
                "public_input_counts": counts, "source_lock_digest": source, "preflight": preflight(suite),
                "execution": "three balanced serial fresh clients per arm; one new private salt; native shared 100/40 Mbps caps",
                "measurement_scope": "source-present; SDK/supervisor/dashboard/relay/full-state audit charged to worker",
                "serializer": "original numeric MessagePack bodies, variable-width server_ns included",
                "output_policy": "natural EOS; report actual outputs, selections, decode steps and preparation rows; never force cap"}
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
                run = benchmark_suite(suite, inputs={case.name: {"prompt": prompt} for case in suite.cases},
                    output=args.output)
                if hashes(sources) != frozen:
                    raise Rejected("sources_changed_during_run")
                summary = audit(run.to_dict(), max_output_tokens=args.max_output_tokens, tiny=args.tiny)
                if source is not None and summary["identity"]["source_lock_digest"] != source:
                    raise Rejected("planned_source_lock_mismatch")
                atomic_json(summary_path, summary)
                progress.update(status="completed", measurement_complete=True, clients=summary["clients"],
                    long_output_cap_reached=summary["long_output_cap_reached"],
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
