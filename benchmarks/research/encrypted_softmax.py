"""Fresh-process complete encrypted softmax, setup and approximation costs."""
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
    "benchmarks/research/encrypted_softmax.py", "crates/pllm-core/src/ckks_layout.rs",
    "crates/pllm-python/src/ckks.rs", "python/pllm/runtime/ckks_reference.py",
    "python/pllm/runtime/ckks_softmax.py", "python/pllm/runtime/seal_frames.py",
    "python/pllm/nonlinear/__init__.py", "python/pllm/metrics/encrypted_softmax.py",
    "python/pllm/metrics/operator_benchmarks.py", "Cargo.lock",
)


def fingerprints():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def worker(width, method):
    from pllm.metrics import benchmark_operator
    from pllm.nonlinear import PackedEncryptedSoftmax
    return benchmark_operator(PackedEncryptedSoftmax(method=method), lanes=3, samples=1,
                              geometry={"width": width}).to_dict()


def run(samples):
    sources, cases = fingerprints(), []
    for width in (1, 8, 32):
        for method in ("repeated_squaring", "square_normalize"):
            reports = []
            for _ in range(samples):
                process = subprocess.run([sys.executable, "-m", "benchmarks.research.encrypted_softmax",
                    "--worker", str(width), method], cwd=ROOT, check=True, capture_output=True,
                    text=True, timeout=180,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                reports.append(json.loads(process.stdout))
            if any(r["resources"] != reports[0]["resources"] for r in reports):
                raise ValueError("CKKS fixed public geometry changed resources")
            if any(r["quality"]["outputs_within_tolerance"] != r["quality"]["checked_outputs"] for r in reports):
                raise ValueError("observed softmax approximation exceeds tolerance")
            cases.append({"width": width, "method": method, "reports": reports,
                "median_seconds": {field: statistics.median(r["samples"][0][field] for r in reports)
                    for field in ("issuance_seconds", "online_seconds", "evaluation_seconds", "native_call_process_cpu_seconds")}})
    for start in range(0, len(cases), 2):
        a,b = [case["reports"][0] for case in cases[start:start+2]]
        if (a["numeric_contract_digest"] == b["numeric_contract_digest"]
                or any(a["resources"][field] != b["resources"][field]
                       for field in ("peer_frame_bytes_by_direction", "setup_frame_bytes_by_operation",
                                     "homomorphic_coefficient_bits"))):
            raise ValueError("different softmax profiles must retain distinct numerics with equal frame geometry")
    if sources != fingerprints():
        raise ValueError("encrypted softmax implementation changed during measurement")
    return {"schema": "pllm.encrypted_softmax_cost.v1", "source_sha256": sources, "cases": cases,
        "fresh_processes": len(cases)*samples, "samples_per_case": samples,
        "checked_outputs": sum(r["quality"]["checked_outputs"] for c in cases for r in c["reports"]),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "scope": {"inputs": "public synthetic scores in fixed [-1/8,1/8]", "client_key_ownership": True,
            "provider_public_context_only": True, "decoded_intermediate": False, "approximate_ckks": True,
            "setup_frames_separate": True, "independent_decimal_profile_oracle": True,
            "independent_decimal_softmax_oracle": True, "bootstrapping": False,
            "decoder_execution": False, "checkpoint_quality": None, "provider_transport": False,
            "measured_peak_memory": None, "full_wire": None, "paper_system_reproduction": False}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        width, method = args.worker
        print(json.dumps(worker(int(width), method)))
    else:
        if args.output is None or args.output.exists():
            parser.error("provide a new --output path")
        result = run(args.samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: result[key] for key in ("fresh_processes", "checked_outputs")}))
