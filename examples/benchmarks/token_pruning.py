"""Benchmark and replay bounded CipherPrune threshold/count/compaction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.passes import CipherPruneTokenPruning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=8, help="complete 32-bit words per token")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--backend", choices=("half_gates", "mixed"), default="half_gates")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("provide a new output path")
    report = benchmark_operator(CipherPruneTokenPruning(backend=args.backend), lanes=args.lanes, samples=args.samples,
                                geometry={"tokens": 4, "heads": 2})
    replay = replay_operator_benchmark(report.to_dict()["configuration"])
    assert report.resources == replay.resources
    assert report.quality == replay.quality
    assert report.configuration_digest == replay.configuration_digest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("Exact protected count, stable compaction, degree selection and replay passed")


if __name__ == "__main__":
    main()
