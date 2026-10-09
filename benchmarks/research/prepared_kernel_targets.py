"""Reusable runtime decomposition, response and exact Preparation-kernel screens.

Geometry is compiler-derived; stage weights are public synthetic fixtures.
Separate resident/paged interactions retain integer-oracle checks and process
peaks. Import children measure independent entry points, not additive costs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.metrics import (BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef,
                          Sampling, benchmark_suite)
from pllm.profiles import MaskedLinearCpu
from pllm.search import FrontierStudy, Variant

from benchmarks.research.common import MODEL, REVISION


def kernel_study(build, *, processes=3):
    from huggingface_hub import hf_hub_download
    config = json.loads(Path(hf_hub_download(MODEL, "config.json", revision=REVISION,
                                            local_files_only=True)).read_bytes())
    experiment = Experiment("prepared-kernels", MaskedLinearCpu(
        Model.hf(MODEL, revision=REVISION, local_files_only=True)),
        Deployment.local(root="local://prepared-kernels"), ExecutionBudget(2, 150, 8))
    cases = tuple(BenchmarkCase(f"{role}-{rows}", BenchmarkTarget.preparation(
        experiment, role=role, rows=rows, config=config), build)
        for role in ("qkv_projection", "mlp_gate_up", "mlp_down") for rows in (16, 150))
    return FrontierStudy(BenchmarkSuite(cases, Sampling(processes=processes,
        timeout_seconds=300, memory_bytes=768 << 20), "prepared-native-kernels"),
        (Variant("digit", {"kernel": "digit8_native"}), Variant("paged", {"storage": "paged"})),
        (("digit", "paged"),))


def runtime_suite(build, *, control=None, processes=3):
    cases = [BenchmarkCase(role, BenchmarkTarget.runtime(role), build) for role in
             ("python", "sdk", "numpy", "transport", "native", "client", "prepared_client", "preparation")]
    if control is not None:
        cases += [BenchmarkCase(f"control-{role}", BenchmarkTarget.runtime(role), control)
                  for role in ("client", "preparation")]
    return BenchmarkSuite(tuple(cases), Sampling(processes=processes, memory_bytes=512 << 20),
                          "prepared-runtime-decomposition")


def runtime_response_suite(build, control, *, artifacts, processes=3):
    from benchmarks.research.prepared_frontier_study import study

    base = Experiment.from_spec(json.loads((artifacts / "indexed.json").read_text()))
    target = study(model=base.pipeline.model, tokenizer=base.pipeline.components["tokenizer"],
                   build=build).suite.cases[0].target
    suite = BenchmarkSuite((BenchmarkCase("control", target, control),
                            BenchmarkCase("candidate", target, build)),
                           Sampling(processes=processes, timeout_seconds=900, memory_bytes=8 << 30),
                           "prepared-runtime-response")
    private_input = {"prompt": (artifacts / "prompt.txt").read_text()}
    return suite, {case.name: private_input for case in suite.cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python")
    parser.add_argument("--control-python")
    parser.add_argument("--processes", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--runtime", action="store_true")
    mode.add_argument("--response", action="store_true", help="Matched installed-build cold/repeat responses")
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    args = parser.parse_args(argv)
    build = BuildRef.capture(args.python) if args.python else BuildRef.capture()
    current = inputs = None
    if args.response:
        if not args.control_python:
            parser.error("--response requires --control-python")
        suite, inputs = runtime_response_suite(build, BuildRef.capture(args.control_python),
                                               artifacts=args.artifacts, processes=args.processes)
    elif args.runtime:
        control = BuildRef.capture(args.control_python) if args.control_python else None
        suite = runtime_suite(build, control=control, processes=args.processes)
    else:
        if args.control_python:
            parser.error("--control-python applies to --runtime or --response")
        current = kernel_study(build, processes=args.processes)
        suite = current.plan(max_trials=256).suite
    run = benchmark_suite(suite, inputs=inputs, output=args.output)
    if current is not None:
        snapshot = current.snapshot(run, metrics={"target.cpu": "min", "process.peak_rss": "min"})
        args.output.with_suffix(".frontier.json").write_text(json.dumps(snapshot.to_dict(), indent=2) + "\n")
    print(json.dumps({"run": str(args.output), "trials": len(run.results),
        "completed": sum(r.to_dict()["status"] == "completed" for r in run.results)}, indent=2))


if __name__ == "__main__":
    main()
