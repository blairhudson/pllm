"""Installed complete Compact operator and exact configuration replay."""
import argparse
import json
from pathlib import Path

from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import CompactGatedMultiplyQ12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-pieces", type=int, default=8)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("choose a fresh output path")
    report = benchmark_operator(CompactGatedMultiplyQ12(max_pieces=args.max_pieces, degree=args.degree),
                                lanes=args.lanes, samples=args.samples)
    replay = replay_operator_benchmark(report.configuration)
    assert report.resources == replay.resources and report.quality == replay.quality
    raw = json.dumps({"schema": "pllm.operator_benchmark_collection.v1", "reports": [report.to_dict()], "replay_exact": True},
                     indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
