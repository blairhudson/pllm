"""Installed SDK matched controls; no Cargo, repository imports or private inputs."""

import argparse
import json
from pathlib import Path

from pllm.metrics import benchmark_operator
from pllm.nonlinear import SharedRescale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--shift", type=int, default=7)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output exists; choose a fresh report path")
    reports = []
    for rounding in ("floor", "ties_even"):
        for backend in ("prefix_dpf", "compact_dcf", "interval_dcf"):
            for layout in ("unfused", "fused"):
                component = SharedRescale(
                    bits=args.bits,
                    shift=args.shift,
                    rounding=rounding,
                    layout=layout,
                    backend=backend,
                )
                reports.append(benchmark_operator(component, lanes=args.lanes, samples=args.samples).to_dict())
    value = {
        "schema": "pllm.operator_benchmark_collection.v1",
        "scope": "public-fixture, in-process helper controls; rank only within one rounding contract",
        "reports": reports,
    }
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
