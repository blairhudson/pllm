"""Matched compiler-shaped full-KV storage and restart mechanics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, benchmark_suite
from pllm.profiles import MaskedLinearCpu
from pllm.search import FrontierStudy, Variant

from benchmarks.research.common import MODEL, REVISION


def study(*, processes=3, tokens=(256, 2048, 8192)):
    from huggingface_hub import hf_hub_download
    config = json.loads(Path(hf_hub_download(MODEL, "config.json", revision=REVISION,
                                            local_files_only=True)).read_bytes())
    experiment = Experiment("state-working-set", MaskedLinearCpu(Model.hf(MODEL, revision=REVISION,
        local_files_only=True)), Deployment.local(root="local://state-working-set"),
        ExecutionBudget(1, max(tokens), 8))
    build = BuildRef.capture()
    cases = [BenchmarkCase(f"attention-{count}", BenchmarkTarget.state(experiment, tokens=count,
        operation="attention", config=config), build) for count in tokens]
    attention = tuple(case.name for case in cases)
    cases.append(BenchmarkCase("restart", BenchmarkTarget.state(experiment, tokens=max(tokens),
        operation="restart", storage="sealed_local", config=config), build))
    return FrontierStudy(BenchmarkSuite(tuple(cases), Sampling(processes=processes,
        timeout_seconds=300, memory_bytes=768 << 20), "state-working-set-v1"),
        (Variant("sealed", {"storage": "sealed_local"}, targets=attention),
         Variant("stream", {"storage": "sealed_stream"})))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--tokens", type=int, action="append")
    args = parser.parse_args(argv)
    current = study(processes=args.processes, tokens=tuple(args.tokens or (256, 2048, 8192)))
    plan = current.plan(max_trials=64)
    result = benchmark_suite(plan.suite, output=args.output)
    args.output.with_suffix(".plan.json").write_text(json.dumps(current.to_spec(), indent=2) + "\n")
    args.output.with_suffix(".frontier.json").write_text(json.dumps(current.snapshot(result,
        metrics={"target.cpu": "min", "process.peak_rss": "min"}).to_dict(), indent=2) + "\n")
    print(json.dumps({"processes": len(result.results), "completed": sum(
        row.to_dict()["status"] == "completed" for row in result.results)}))


if __name__ == "__main__":
    main()
