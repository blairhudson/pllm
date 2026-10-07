"""Installed complete Curl/Haar gated-block controls with explicit numeric identities."""
import argparse
import json
from pathlib import Path

from pllm.metrics import benchmark_operator
from pllm.nonlinear import CurlWaveletGatedMultiplyQ9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output exists; choose a fresh report path")
    reports = [benchmark_operator(CurlWaveletGatedMultiplyQ9(levels=level),
               lanes=args.lanes, samples=args.samples).to_dict() for level in (0, 2, 4)]
    payload = json.dumps({"schema": "pllm.operator_benchmark_collection.v1",
        "scope": "complete Q18-to-Q9 gated operators; exact rescaling adaptation; distinct Haar numeric identities",
        "reports": reports}, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
