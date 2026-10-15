"""Six serial full-SDK clients measuring exact head HTTP/shared-pacer owners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.research import private_boundary_pacing_profile as profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    from filelock import FileLock, Timeout
    from pllm.metrics import BuildRef, benchmark_suite
    from pllm._benchmark_contract import digest
    from benchmarks.research.sota_combinations import Rejected, atomic_json, hashes, public_input_counts
    from scripts.benchmark_sota_combinations import read_fixture
    from scripts.profile_private_boundary_roles import reserve_output_paths
    args.output = args.output.resolve()
    plan_path, progress_path, summary_path = reserve_output_paths(args.output)
    progress = {"schema": profile.IDENTITY, "status": "preflight"}
    atomic_json(progress_path, progress)
    lock = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock), timeout=0):
            sources = [Path(__file__).resolve(),
                *(ROOT / "benchmarks/research" / name for name in (
                    "private_boundary_pacing_profile.py", "private_boundary_role_profile.py",
                    "private_boundary_decode_profile.py", "private_boundary_decode.py",
                    "private_boundary_tiers.py", "historical_evidence.py", "sota_combinations.py")),
                ROOT / "scripts/profile_private_boundary_roles.py", ROOT / "scripts/benchmark_sota_combinations.py",
                ROOT / "examples/prepared_sdk.py", ROOT / "docs/evidence/private-boundary-qwen-control.json"]
            artifacts = args.artifacts.resolve()
            if args.tiny:
                from pllm import Model
                if args.preflight_only:
                    generated = Model.tiny()
                else:
                    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
                    fixture = create_tiny_llama_checkpoint(args.output.with_suffix(".model"),
                        model_type="qwen2", with_qkv_bias=True)
                    generated = Model.path(str(fixture.resolve()), model_id="pacing-profile-tiny")
                prompt = "abcde " * 14
            else:
                generated = None
                prompt, _ = read_fixture(artifacts / "prompt.txt")
                sources += [artifacts / "prompt.txt", artifacts / "indexed.json"]
            # HTTP trace event implementation participates in the source lock too.
            import httpcore
            import httpx
            import pllm.runtime.native_wan as wan
            sources += [Path(wan.__file__), Path(httpx.__file__).with_name("_client.py"),
                Path(httpcore.__file__).with_name("_trace.py"),
                Path(httpcore.__file__).parent / "_sync/http11.py"]
            frozen = hashes(sources)
            identity = digest({"schema": profile.IDENTITY, "sources": frozen})
            suite = profile.matched_suite(build=BuildRef.capture(), tiny=args.tiny,
                artifacts=artifacts, generated_source=generated)
            from pllm import Experiment
            exp = Experiment.from_spec(suite.cases[0].target.to_spec()["parameters"]["model"]["experiment"])
            counts, source = public_input_counts(exp.pipeline.model, exp.pipeline.components.get("tokenizer"),
                {"tiny" if args.tiny else "150": {"prompt": prompt}})
            if any(count > exp.budget.max_input_tokens for count in counts.values()):
                raise Rejected("public_fixture_exceeds_request_bound")
            plan = {"schema": profile.IDENTITY, "suite": suite.to_spec(), "frozen_sources": frozen,
                "research_source_identity": identity, "tiny": args.tiny, "public_input_counts": counts,
                "source_lock_digest": source, "preflight": profile.preflight(suite),
                "telemetry_admission": {"total_extra_bytes": profile.EXTRA_MEMORY,
                    "client_and_collection_bytes": (8 << 20) + profile.EXTRA_TRACE_MEMORY,
                    "per_role_bytes": 4 << 20, "both_control_and_probe": True},
                "execution": "6 serial fresh clients, control/probe x 3 balanced replicates; native shared 100/40; exact state; zero swap",
                "scope": "Original installed SDK/HTTP/relay execution with bounded opt-in timing wrappers"}
            atomic_json(plan_path, plan)
            if any(not row["admission"]["admitted"] for row in plan["preflight"]):
                raise Rejected("topology_memory_limit")
            if args.preflight_only:
                progress["status"] = "admitted"
            else:
                progress["status"] = "running"
                atomic_json(progress_path, progress)
                with profile.research_workers(Path(__file__).resolve(), identity):
                    run = benchmark_suite(suite, inputs={c.name: {"prompt": prompt} for c in suite.cases}, output=args.output)
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
        with profile.worker_bootstrap(sys.argv[2], sys.argv[3], Path(__file__).resolve()):
            ordinary_worker()
    elif len(sys.argv) > 4 and sys.argv[1] == "--role-worker":
        from pllm.runtime.private_boundary_role import main as ordinary_role
        if sys.argv[3] == "probe":
            sink = profile.roles.Sink(sys.argv[2])
            try:
                with profile.roles.role_instrument(sink):
                    ordinary_role(sys.argv[4:])
            finally:
                sink.close()
        elif sys.argv[3] == "control":
            ordinary_role(sys.argv[4:])
        else:
            raise SystemExit("invalid pacing profile mode")
    else:
        raise SystemExit(main())
