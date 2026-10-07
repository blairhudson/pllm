"""Installed complete LogRow product and replay, including every boundary."""
import argparse
import json
from pathlib import Path

from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import LogRowGatedMultiplyQ9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("choose a fresh output path")
    report = benchmark_operator(LogRowGatedMultiplyQ9(), lanes=args.lanes, samples=args.samples)
    assert replay_operator_benchmark(report.configuration).resources == report.resources
    payload = json.dumps({"schema": "pllm.operator_benchmark_collection.v1", "reports": [report.to_dict()],
        "scope": "bounded complete LogRow product; local garbler and opaque arithmetic"},
        sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        with args.output.open("x") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
