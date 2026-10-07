"""Fresh-process CKKS matrix controls with complete key and frame accounting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "benchmarks/research/encrypted_linear.py", "crates/pllm-core/src/ckks_layout.rs",
    "crates/pllm-python/src/ckks.rs", "python/pllm/runtime/ckks_reference.py",
    "python/pllm/runtime/seal_frames.py", "python/pllm/protocols/packed_encrypted.py",
    "python/pllm/metrics/encrypted_linear.py", "python/pllm/metrics/operator_benchmarks.py", "Cargo.lock",
)
CASES = ((3, 5), (16, 16), (32, 32))


def fingerprints():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def worker(rows, columns, keys):
    from pllm.metrics import benchmark_operator
    from pllm.protocols import PackedEncryptedLinear
    return benchmark_operator(PackedEncryptedLinear(rotation_keys=keys), lanes=4, samples=1,
                              geometry={"rows": rows, "columns": columns}).to_dict()


def run(samples):
    sources = fingerprints()
    cases = []
    for rows, columns in CASES:
        for keys in ("single", "full"):
            reports = []
            for _ in range(samples):
                process = subprocess.run([sys.executable, "-m", "benchmarks.research.encrypted_linear",
                    "--worker", str(rows), str(columns), keys], cwd=ROOT, check=True,
                    capture_output=True, text=True, timeout=180,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                reports.append(json.loads(process.stdout))
            if any(r["resources"] != reports[0]["resources"] for r in reports):
                raise ValueError("CKKS fixed public geometry changed resource counts")
            cases.append({"rows": rows, "columns": columns, "rotation_keys": keys, "reports": reports,
                "median_seconds": {field: statistics.median(r["samples"][0][field] for r in reports)
                    for field in ("issuance_seconds", "online_seconds", "evaluation_seconds", "native_call_process_cpu_seconds")}})
    for start in range(0, len(cases), 2):
        single, full = [c["reports"][0] for c in cases[start:start+2]]
        if (single["numeric_contract_digest"] != full["numeric_contract_digest"]
                or single["resources"]["peer_frame_bytes_by_direction"] != full["resources"]["peer_frame_bytes_by_direction"]):
            raise ValueError("key-policy controls changed numeric contract or online frames")
    if sources != fingerprints():
        raise ValueError("CKKS implementation changed during measurement")
    return {"schema": "pllm.encrypted_linear_cost.v1", "source_sha256": sources, "cases": cases,
        "fresh_processes": len(cases)*samples, "samples_per_case": samples,
        "checked_outputs": sum(r["quality"]["checked_outputs"] for c in cases for r in c["reports"]),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "scope": {"inputs": "public synthetic float64 bounded matrices/vectors", "client_key_ownership": True,
            "provider_public_context_only": True, "decoded_intermediate": False, "approximate_ckks": True,
            "setup_frames_separate": True, "public_matrix_prepositioned": True,
            "bootstrapping": False, "decoder_execution": False, "checkpoint_quality": None,
            "provider_transport": False, "measured_peak_memory": None, "full_wire": None,
            "moai_reproduction": False, "nexus_system_reproduction": False}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        rows, columns, keys = args.worker
        print(json.dumps(worker(int(rows), int(columns), keys)))
    else:
        if args.output is None or args.output.exists():
            parser.error("provide a new --output path")
        result = run(args.samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: result[key] for key in ("fresh_processes", "checked_outputs")}))
