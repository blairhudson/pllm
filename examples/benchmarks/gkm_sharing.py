"""Compare full and seeded GKM SDK responses through ordinary benchmark targets.

Pinned Qwen, source-present workers, shared native 100/40 Mbps access caps and
complete accepted-state audits. Fixed-count greedy output has no streaming decode
rate. Full and seeded sharing retain different privacy identities. No added RTT.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pllm import ExecutionBudget, Model
from pllm.deployment import WanConditions
from pllm.metrics import (
    BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario,
    benchmark_suite,
)

from examples.papers.common import MODEL
from examples.papers.gkm import experiment


def suite(*, model=MODEL, processes=3, input_tokens=8, output_tokens=2, memory_bytes=16 << 30):
    build = BuildRef.capture()
    budget = ExecutionBudget(requests=1, max_input_tokens=input_tokens, max_new_tokens=output_tokens)
    cases = []
    for sharing in ("full", "seeded"):
        selected = experiment(model=model, budget=budget).with_params(
            name=f"gkm-{sharing}", pipeline__linear__sharing=sharing,
            deployment__root=f".pllm/gkm-sharing/{sharing}")
        cases.append(BenchmarkCase(f"gkm-{sharing}", BenchmarkTarget.response(
            selected, scenario=Scenario.cold(max_output_tokens=output_tokens),
            backend="native", wan=WanConditions(), state_audit=True,
        ), build=build))
    return BenchmarkSuite(cases, Sampling(processes=processes, memory_bytes=memory_bytes,
                                         timeout_seconds=1200))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--input-tokens", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=2)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--memory-mib", type=int, default=16384,
                        help="whole-topology admission budget; not a client memory tier")
    args = parser.parse_args(argv)
    model = Model.path(str(args.model_dir.resolve())) if args.model_dir else MODEL
    prompt = args.prompt_file.read_text() if args.prompt_file else "Hello world"
    selected = suite(model=model, processes=args.processes, input_tokens=args.input_tokens,
                     output_tokens=args.output_tokens, memory_bytes=args.memory_mib << 20)
    report = benchmark_suite(selected, inputs={case.name: {"prompt": prompt} for case in selected.cases},
                             output=args.output)
    statuses = [result.to_dict()["status"] for result in report.results]
    print({"output": args.output, "statuses": statuses})
    if any(status != "completed" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
