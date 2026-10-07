"""Run the installed Maverick algebra operator and replay its exact configuration."""
import argparse
from pathlib import Path

from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.protocols import MaverickDelegatedLinear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new path")
    report = benchmark_operator(MaverickDelegatedLinear(), lanes=args.lanes, samples=args.samples,
                                geometry={"rows": 64, "columns": 64})
    assert report.quality.exact_integer_outputs == 64 * args.lanes * args.samples
    replay = replay_operator_benchmark(report.configuration)
    assert replay.resources == report.resources
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.to_json() + "\n")
    print("Exact signed matvec, asymmetric ownership and replay passed; privacy/verifier remain uncertified")


if __name__ == "__main__":
    main()
