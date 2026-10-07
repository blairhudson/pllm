"""Fresh-process Secret Replication key-layout controls and consumer boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "benchmarks/research/secret_replication.py",
    "crates/pllm-garble/src/secret_replication.rs",
    "crates/pllm-python/src/secret_replication.rs",
    "python/pllm/metrics/secret_replication.py",
)


def leakage_witness(field: int, domain: int) -> dict:
    """Exact support enumeration, independent of the native closed-form witness."""
    shift = min(domain, field - domain)
    before = [sum((0 - mask) % field == opening for mask in range(domain)) for opening in range(field)]
    after = [sum((shift - mask) % field == opening for mask in range(domain)) for opening in range(field)]
    distance_numerator = sum(abs(a - b) for a, b in zip(before, after))
    return {"field": field, "scalar_domain": domain, "input_shift": shift,
            "distance_numerator": distance_numerator, "distance_denominator": 2 * domain,
            "total_variation_distance": distance_numerator / (2 * domain),
            "arbitrary_field_consumer_admitted": field == domain}


def symbolic_width(field: int, vector_length: int) -> dict:
    depth = (field - 1).bit_length()
    return {"field": field, "vector_length": vector_length,
        "punctured_keys_both_parties_bytes": 2 * (4 + 16 * (1 + depth)),
        "prf_field_evaluations_both_parties": 2 * vector_length * (2 * field - 1),
        "native_admitted": field in (2, 17, 257) and 1 <= vector_length <= 4096,
        "scope": "symbolic full-field PCF count only; no material or outputs issued at unadmitted widths"}


def run() -> dict:
    reports = []
    for field in (2, 17, 257):
        for lanes in (1, 64, 256):
            for encoding in ("expanded", "punctured"):
                completed = subprocess.run([sys.executable, "-m", "benchmarks.research.secret_replication",
                    "--worker", encoding, "--field", str(field), "--lanes", str(lanes)],
                    check=True, capture_output=True, text=True, timeout=120,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                reports.append(json.loads(completed.stdout))
            a, b = reports[-2:]
            if (a["numeric_contract_digest"] != b["numeric_contract_digest"] or a["quality"] != b["quality"]
                    or a["resources"]["peer_frame_bytes_by_direction"] != b["resources"]["peer_frame_bytes_by_direction"]):
                raise ValueError("PCF key-layout controls differ in numeric or online contract")
    witnesses = [leakage_witness(p, d) for p, d in ((17, 2), (17, 12), (17, 17), (257, 16), (257, 256), (257, 257))]
    return {"schema": "pllm.secret_replication_cost.v1", "fresh_processes": len(reports), "reports": reports,
        "checked_outputs": sum(row["quality"]["checked_outputs"] for row in reports),
        "restricted_scalar_opening_witnesses": witnesses,
        "elementwise_reuse_counterexample": {"field": 17, "input": [3, 9], "mask": 5,
            "openings": [15, 4], "revealed_input_difference_mod_field": 11,
            "scope": "synthetic witness: opening both elements with one scalar mask exposes their difference"},
        "field_ring_conversion_counterexample": {"a": 256, "b": 1, "prime_field_257_sum": 0,
            "ring32_sum": 257, "scope": "matching word containers do not identify field and wrap32 arithmetic"},
        "full_field_width_counts": [symbolic_width(p, 64) for p in (257, 65537, 4294967291)],
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES},
        "scope": "bounded local modular scalar/vector operator and explicit incompatible-consumer witnesses",
        "decoder_executable": False, "ring32_elementwise_compatible": False,
        "full_wire_bytes": None, "checkpoint_quality": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("expanded", "punctured"))
    parser.add_argument("--field", type=int, default=257)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        from pllm.correlation import SecretReplicationPcf
        from pllm.metrics import benchmark_operator
        print(benchmark_operator(SecretReplicationPcf(field=args.field, encoding=args.worker),
                                 lanes=args.lanes, samples=3).to_json())
        return
    if args.output is None or args.output.exists():
        parser.error("choose a fresh output path")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"checked_outputs": report["checked_outputs"],
                      "fresh_processes": report["fresh_processes"],
                      "restricted_scalar_opening_witnesses": report["restricted_scalar_opening_witnesses"]}))


if __name__ == "__main__":
    main()
