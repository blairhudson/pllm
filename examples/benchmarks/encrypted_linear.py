"""Installed SDK encrypted matrix benchmark, configuration replay and saved report."""
import argparse
from pathlib import Path

from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.protocols import PackedEncryptedLinear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--rotation-keys", choices=("single", "full"), default="single")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = benchmark_operator(PackedEncryptedLinear(rotation_keys=args.rotation_keys),
        lanes=args.lanes, samples=args.samples, geometry={"rows": 8, "columns": 8})
    replay = replay_operator_benchmark(report.configuration)
    assert report.resources == replay.resources
    assert report.numeric_contract_digest == replay.numeric_contract_digest
    assert report.quality.outputs_within_tolerance == args.lanes * args.samples * 8
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.to_json() + "\n")
    print(f"checked {report.quality.checked_outputs} approximate outputs; max error {report.quality.maximum_absolute_error:.8g}")


if __name__ == "__main__":
    main()
