"""Actual generated-decoder restart, fresh material and separately charged lifecycle."""
import argparse
import json
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, benchmark_suite
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.search import FrontierStudy


def study(*, processes=3, tokens=(8, 32, 128)):
    experiment = Experiment("decoder-restart", MaskedLinearCpu(Model.tiny(),
        quantization=SymmetricPerRow(causal_reduction="prefix_f32")),
        Deployment.local(root="local://decoder-restart"), ExecutionBudget(1, max(tokens) + 4, 4))
    build = BuildRef.capture()
    cases = tuple(BenchmarkCase(f"restart-{count}", BenchmarkTarget.state(experiment,
        tokens=count, storage="sealed_local", operation="decoder_restart"), build) for count in tokens)
    return FrontierStudy(BenchmarkSuite(cases, Sampling(processes=processes, timeout_seconds=180,
        memory_bytes=768 << 20), "decoder-restart-v1"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--tokens", type=int, action="append")
    args = parser.parse_args(argv)
    current = study(processes=args.processes, tokens=tuple(args.tokens or (8, 32, 128)))
    plan = current.plan(max_trials=64)
    result = benchmark_suite(plan.suite, output=args.output)
    args.output.with_suffix(".plan.json").write_text(json.dumps(current.to_spec(), indent=2) + "\n")
    args.output.with_suffix(".frontier.json").write_text(json.dumps(current.snapshot(result,
        metrics={"target.cpu": "min", "restore_child.peak_rss": "min"}).to_dict(), indent=2) + "\n")
    print(json.dumps({"processes": len(result.results), "completed": sum(
        row.to_dict()["status"] == "completed" for row in result.results)}))


if __name__ == "__main__":
    main()
