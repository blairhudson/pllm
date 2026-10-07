"""Benchmark complete mixed arithmetic blocks and replay their SDK configurations."""
import argparse
import json
from pathlib import Path

from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import CompactGatedMultiplyQ12, CurlWaveletGatedMultiplyQ9, LogRowGatedMultiplyQ9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for component in (CurlWaveletGatedMultiplyQ9(levels=4, backend="mixed"),
                      LogRowGatedMultiplyQ9(backend="mixed"), CompactGatedMultiplyQ12(backend="mixed")):
        report = benchmark_operator(component, lanes=args.lanes, samples=args.samples)
        replay = replay_operator_benchmark(report.configuration)
        assert report.resources == replay.resources and report.quality == replay.quality
        reports.append(report.to_dict())
    with args.output.open("x") as handle:
        json.dump({"schema": "pllm.operator_benchmark_collection.v1", "reports": reports,
                   "scope": "complete mixed arithmetic blocks; bounded local trusted dealer"},
                  handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Three complete mixed blocks and replay pass; local trusted dealer and numeric limits remain explicit")


if __name__ == "__main__":
    main()
