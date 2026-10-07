"""Installed FastQuery-inspired coefficient packing and conversion comparison."""
import json

from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.protocols import PrivateEmbeddingLookup


def main() -> None:
    reports = [benchmark_operator(PrivateEmbeddingLookup(vocabulary_size=256, precision=precision),
                                 lanes=64, samples=3)
               for precision in ("int8", "int4", "mixed_4_3_3")]
    for report in reports:
        replay = replay_operator_benchmark(report.configuration)
        assert replay.resources == report.resources
        assert replay.quality == report.quality
    print(json.dumps({"schema": "pllm.operator_benchmark_collection.v1", "reports": [r.to_dict() for r in reports]},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
