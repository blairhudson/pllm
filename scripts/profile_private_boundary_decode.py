"""Separate research-only timing cohort over unchanged ordinary SDK responses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.research import private_boundary_decode_profile as profile


def reserve_output_paths(output):
    """Exclusive durable prefix claim; failed/interrupted runs are never resumed."""
    from scripts.benchmark_sota_combinations import fresh_output_paths
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation = output.with_suffix(".reservation")
    with reservation.open("x") as owner:
        owner.write(profile.IDENTITY + "\n")
    paths = fresh_output_paths(output)
    if output.with_suffix(".model").exists():
        raise FileExistsError("generated model prefix already exists")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--tiny", action="store_true", help="128+2 generated-source smoke, not Qwen evidence")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    from filelock import FileLock, Timeout
    from pllm.metrics import BuildRef, benchmark_suite
    from pllm._benchmark_contract import digest
    from benchmarks.research.sota_combinations import Rejected, atomic_json, hashes, public_input_counts
    from scripts.benchmark_sota_combinations import read_fixture

    args.output = args.output.resolve()
    plan_path, progress_path, summary_path = reserve_output_paths(args.output)
    progress = {"schema": profile.IDENTITY, "status": "preflight"}
    atomic_json(progress_path, progress)
    lock = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock), timeout=0):
            artifacts = args.artifacts.resolve()
            output = args.output.resolve()
            sources = [Path(__file__).resolve(), ROOT / "benchmarks/research/private_boundary_decode_profile.py",
                ROOT / "benchmarks/research/private_boundary_decode.py",
                ROOT / "benchmarks/research/private_boundary_tiers.py",
                ROOT / "benchmarks/research/historical_evidence.py",
                ROOT / "benchmarks/research/sota_combinations.py",
                ROOT / "scripts/benchmark_sota_combinations.py", ROOT / "examples/prepared_sdk.py",
                ROOT / "docs/evidence/private-boundary-qwen-control.json"]
            if args.tiny:
                from pllm import Model
                if args.preflight_only:
                    generated_source = Model.tiny()
                else:
                    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
                    fixture = create_tiny_llama_checkpoint(output.with_suffix(".model"),
                        model_type="qwen2", with_qkv_bias=True)
                    generated_source = Model.path(str(fixture.resolve()), model_id="decode-profile-tiny")
                prompt = "abcde " * 14
            else:
                generated_source = None
                prompt, _ = read_fixture(artifacts / "prompt.txt")
                sources += [artifacts / "prompt.txt", artifacts / "indexed.json"]
            frozen = hashes(sources)
            identity = digest({"schema": profile.IDENTITY, "sources": frozen})
            suite = profile.matched_suite(build=BuildRef.capture(), tiny=args.tiny,
                artifacts=artifacts, generated_source=generated_source)
            from pllm import Experiment
            exp = Experiment.from_spec(suite.cases[0].target.to_spec()["parameters"]["model"]["experiment"])
            counts, source = public_input_counts(exp.pipeline.model, exp.pipeline.components.get("tokenizer"),
                {"tiny" if args.tiny else "150": {"prompt": prompt}})
            if any(count > exp.budget.max_input_tokens for count in counts.values()):
                raise Rejected("public_fixture_exceeds_request_bound")
            plan = {"schema": profile.IDENTITY, "suite": suite.to_spec(), "frozen_sources": frozen,
                "research_source_identity": identity, "tiny": args.tiny, "public_input_counts": counts,
                "source_lock_digest": source, "preflight": profile.preflight(suite),
                "execution": "18 serial fresh workers: three selected recipes x control/probe x three balanced replicates; "
                    "one private salt; native shared 100/40 caps; ordinary memory watchdog and full-state audit",
                "measurement_scope": "Research bootstrap only; no installed SDK, masks or cryptographic RNG changes"}
            atomic_json(plan_path, plan)
            if any(not row["admission"]["admitted"] for row in plan["preflight"]):
                raise Rejected("topology_memory_limit")
            if args.preflight_only:
                progress["status"] = "admitted"
            else:
                progress["status"] = "running"
                atomic_json(progress_path, progress)
                with profile.research_workers(Path(__file__).resolve(), identity):
                    run = benchmark_suite(suite, inputs={case.name: {"prompt": prompt} for case in suite.cases},
                        output=output)
                if hashes(sources) != frozen:
                    raise Rejected("sources_changed_during_run")
                summary = profile.audit(run.to_dict(), source_identity=identity, tiny=args.tiny)
                atomic_json(summary_path, summary)
                progress.update(status="completed", clients=summary["clients"])
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
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        from pllm._benchmark_worker import main as ordinary_worker
        with profile.worker_bootstrap(sys.argv[2], sys.argv[3]):
            ordinary_worker()
    else:
        raise SystemExit(main())
