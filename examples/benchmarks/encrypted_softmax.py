"""Installed SDK encrypted softmax benchmark and independent configuration replay."""
import argparse
from pathlib import Path

from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import PackedEncryptedSoftmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--lanes", type=int, default=3)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--method", choices=("repeated_squaring", "square_normalize"), default="repeated_squaring")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = benchmark_operator(PackedEncryptedSoftmax(method=args.method), lanes=args.lanes,
                                samples=args.samples, geometry={"width": args.width})
    replay = replay_operator_benchmark(report.configuration)
    assert report.resources == replay.resources
    assert report.numeric_contract_digest == replay.numeric_contract_digest
    assert report.quality.outputs_within_tolerance == args.lanes * args.samples * args.width
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.to_json())
    print(f"checked {report.quality.checked_outputs} approximate outputs; max error {report.quality.maximum_absolute_error:.8g}")


if __name__ == "__main__":
    main()
