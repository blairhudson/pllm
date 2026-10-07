"""Installed Compact, Curl and LogRow operator controls on a common Q7 profile."""

import argparse
import json
from pathlib import Path

from pllm.metrics import benchmark_operator
from pllm.nonlinear import CompactPiecewiseActivation, CurlWaveletEncodedTable
from pllm.protocols import LogRowGarbledLookup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=17)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output exists; choose a fresh report path")
    methods = [CompactPiecewiseActivation(), LogRowGarbledLookup()]
    methods += [CurlWaveletEncodedTable(levels=level) for level in (0, 2, 4)]
    reports = [benchmark_operator(method, lanes=args.lanes, samples=args.samples).to_dict() for method in methods]
    value = {
        "schema": "pllm.operator_benchmark_collection.v1",
        "scope": "public Q7 activation controls; compressed numerics and garbler/worker ownership differ",
        "reports": reports,
    }
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
