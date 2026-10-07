"""Fresh-process complete LogRow/Curl controls under one fixed numeric profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
QUALITY = "docs/evidence/research-wavelet-gated-quality-qwen25.json"


def project(report):
    lanes = report["configuration"]["lanes"]
    elements = 24 * 4864 * (39 + 8 - 1)
    if elements % lanes:
        raise ValueError("projection requires complete chunks")
    batches = elements // lanes
    r = report["resources"]
    return {"method": report["method"], "elements": elements, "layers": 24, "width": 4864,
        "input_tokens": 39, "output_tokens": 8, "role_names": r["role_names"],
        "retained_key_bytes_by_role": [batches * v for v in r["raw_key_payload_bytes_per_party"]],
        "keys_by_operation_and_role": [[name, batches * a, batches * b]
            for name, a, b in r["key_payload_bytes_by_operation"]],
        "peer_frame_bytes": batches * sum(r["peer_frame_bytes_by_direction"]),
        "public_metadata_bytes_estimate": batches * r["public_metadata_bytes_estimate"],
        "client_boundary_payload_estimate_bytes": batches * r["client_entry_exit_payload_estimate_bytes"],
        "executable_decoder": False, "full_wire_bytes": None,
        "scope": "hypothetical complete MLP-only geometry; client-retained keys are not network traffic"}


def run():
    from pllm.metrics.wavelet_gated import _profile
    quality_bytes = (ROOT / QUALITY).read_bytes()
    quality = json.loads(quality_bytes)
    numeric = _profile(0)[1]
    if numeric != quality["numeric_contract_digests"]["haar-0"]:
        raise ValueError("historical quality has a different numeric contract")
    reports = []
    for lanes in (1, 64, 256):
        for method in ("curl", "logrow"):
            raw = subprocess.run([sys.executable, "-m", "benchmarks.research.logrow_product",
                "--worker", method, "--lanes", str(lanes)], check=True, capture_output=True,
                text=True, timeout=180, env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
            reports.append(json.loads(raw.stdout))
        left, right = reports[-2:]
        if (left["numeric_contract_digest"] != right["numeric_contract_digest"]
                or left["numeric_contract_digest"] != numeric or left["quality"] != right["quality"]):
            raise ValueError("matched operator numeric controls differ")
    return {"schema": "pllm.logrow_product_cost.v1", "fresh_processes": len(reports),
        "reports": reports, "checked_outputs": sum(r["quality"]["checked_outputs"] for r in reports),
        "projections": [project(r) for r in reports if r["configuration"]["lanes"] == 256],
        "quality_reference": {"path": QUALITY, "sha256": hashlib.sha256(quality_bytes).hexdigest(),
            "numeric_contract_digest": numeric, "teacher_forced": quality["teacher_forced"]["summary"]["haar-0"],
            "greedy_generation": quality["greedy_generation"]["summary"]["haar-0"],
            "scope": "unchanged numeric profile; retained clear Qwen diagnostic, not a new protected checkpoint run"},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("logrow", "curl"))
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        from pllm.metrics import benchmark_operator
        from pllm.nonlinear import CurlWaveletGatedMultiplyQ9, LogRowGatedMultiplyQ9
        method = LogRowGatedMultiplyQ9() if args.worker == "logrow" else CurlWaveletGatedMultiplyQ9()
        print(benchmark_operator(method, lanes=args.lanes, samples=3).to_json())
        return
    if args.output is None or args.output.exists():
        parser.error("choose a fresh output path")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["projections"]))


if __name__ == "__main__":
    main()
