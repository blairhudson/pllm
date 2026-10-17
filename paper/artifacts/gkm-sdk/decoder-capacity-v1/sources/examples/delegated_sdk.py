"""Run the bounded full-share GKM composition through ordinary PLLM APIs.

Requires a complete local Safetensors source with tokenizer.json. Default capsule
capacity admits at most 256 vocabulary entries; explicit decoder capacity uses
source-derived native ownership bounds. Every source and workload still requires
numerical and host-memory admission. Two local workers demonstrate functionality,
not operator independence. Output is buffered until request-end verification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model, OpenAI
from pllm.metrics import (
    BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario,
    benchmark_suite,
)
from pllm.profiles import OpenWeightVerifiableCpu
from pllm.protocols import OpenWeightVerifiableOutsourcing


def experiment(model_dir, *, root=".pllm/delegated", max_input_tokens=8, max_new_tokens=3,
               capacity="capsule"):
    return Experiment(
        "bounded-delegated", OpenWeightVerifiableCpu(Model.path(str(Path(model_dir).resolve())),
            linear=OpenWeightVerifiableOutsourcing(capacity=capacity)),
        Deployment.local(root=str(root)),
        ExecutionBudget(requests=4, max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(".pllm/delegated"))
    parser.add_argument("--prompt", default="hello world")
    parser.add_argument("--max-input-tokens", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=3)
    parser.add_argument("--capacity", choices=("capsule", "decoder"), default="capsule")
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument("--memory-bytes", type=int, default=2 << 30)
    parser.add_argument("--write-experiment", type=Path)
    parser.add_argument("--benchmark-output", type=Path)
    args = parser.parse_args(argv)
    selected = experiment(args.model_dir, root=args.root, max_input_tokens=args.max_input_tokens,
                          max_new_tokens=args.max_output_tokens, capacity=args.capacity)
    if args.write_experiment:
        with args.write_experiment.open("xb") as output:
            output.write(selected.canonical_bytes())
    if args.benchmark_output:
        case = BenchmarkCase("bounded-delegated", BenchmarkTarget.response(selected,
            scenario=Scenario.cold(max_output_tokens=args.max_output_tokens), backend="native"),
            build=BuildRef.capture())
        result = benchmark_suite(BenchmarkSuite([case], Sampling(processes=1,
            timeout_seconds=args.timeout_seconds, memory_bytes=args.memory_bytes)),
            inputs={case.name: {"prompt": args.prompt}}, output=args.benchmark_output)
        status = result.results[0].to_dict()["status"]
        print({"status": status, "output": str(args.benchmark_output)})
        if status != "completed":
            raise SystemExit(1)
    else:
        with OpenAI(experiment=selected, timeout=args.timeout_seconds) as client:
            response = client.responses.create(input=args.prompt,
                max_output_tokens=args.max_output_tokens, temperature=0)
            print(response.output_text)
            print({"status": response.status, "output_tokens": response.usage.output_tokens})


if __name__ == "__main__":
    main()
