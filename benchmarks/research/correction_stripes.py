"""Compiler-shaped correction batches through separate local push/ack peers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, benchmark_suite
from pllm.profiles import MaskedLinearCpu
from pllm.search import FrontierStudy, Variant
from benchmarks.research.common import MODEL, REVISION


def study(*, processes=3, rows=(16, 150), credit_screen=False):
    from huggingface_hub import hf_hub_download
    config = json.loads(Path(hf_hub_download(MODEL, "config.json", revision=REVISION,
                                            local_files_only=True)).read_bytes())
    experiment = Experiment("correction-stripes", MaskedLinearCpu(Model.hf(MODEL,
        revision=REVISION, local_files_only=True)), Deployment.local(root="local://correction-stripes"),
        ExecutionBudget(1, max(rows), 8))
    build = BuildRef.capture()
    cases = [BenchmarkCase(f"{kernel}-{count}", BenchmarkTarget.preparation(experiment,
        rows=count, kernel=kernel, output_columns=256 if credit_screen else 0,
        **({"credit_window": 1} if credit_screen else {}), config=config), build)
        for kernel in ("native", "digit8_native") for count in rows]
    return FrontierStudy(BenchmarkSuite(tuple(cases), Sampling(processes=processes,
        timeout_seconds=300, memory_bytes=1 << 30), "correction-stripes-v1"),
        ((Variant("credit4", {"credit_window": 4}), Variant("credit8", {"credit_window": 8}))
         if credit_screen else (Variant("columns256", {"output_columns": 256}),
                               Variant("columns1024", {"output_columns": 1024}))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--rows", type=int, action="append")
    parser.add_argument("--credit-screen", action="store_true")
    args = parser.parse_args(argv)
    current = study(processes=args.processes, rows=tuple(args.rows or (16, 150)), credit_screen=args.credit_screen)
    plan = current.plan(max_trials=64)
    result = benchmark_suite(plan.suite, output=args.output)
    args.output.with_suffix(".plan.json").write_text(json.dumps(current.to_spec(), indent=2) + "\n")
    args.output.with_suffix(".frontier.json").write_text(json.dumps(current.snapshot(result,
        metrics={"target.cpu": "min", "process.peak_rss": "min", "receiver.peak_rss": "min"}).to_dict(), indent=2) + "\n")
    print(json.dumps({"processes": len(result.results), "completed": sum(
        row.to_dict()["status"] == "completed" for row in result.results)}))


if __name__ == "__main__":
    main()
