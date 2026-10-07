"""Installed SDK example: expanded/punctured scalar-vector PCF key layouts."""
import json

from pllm.correlation import SecretReplicationPcf
from pllm.metrics import benchmark_operator, replay_operator_benchmark


def main() -> None:
    reports = [benchmark_operator(SecretReplicationPcf(encoding=encoding), lanes=64, samples=3)
               for encoding in ("expanded", "punctured")]
    for report in reports:
        replay = replay_operator_benchmark(report.configuration)
        assert replay.resources == report.resources
        assert replay.quality == report.quality
    print(json.dumps({"schema": "pllm.operator_benchmark_collection.v1", "reports": [r.to_dict() for r in reports]},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
