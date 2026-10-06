"""Installed SDK operator benchmark; run with ordinary Python, no repository imports."""

import argparse
from pathlib import Path

from pllm.metrics import benchmark_operator
from pllm.nonlinear import MaskedWireGatedMultiplyQ12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark_operator(MaskedWireGatedMultiplyQ12(), lanes=args.lanes, samples=args.samples)
    if args.output is None:
        print(report.to_json(), end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report.to_json())


if __name__ == "__main__":
    main()
