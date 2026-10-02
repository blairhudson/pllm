"""SDK microbenchmark for combined correlation compression; no model downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

from pllm.metrics import ProjectedPolynomialCostProbe


def run() -> dict:
    cases = [
        ProjectedPolynomialCostProbe(
            mode=mode, ring_bits=bits, hidden=32, channels=128, outputs=32, rows=8
        ).run()
        for bits in (24, 32, 64)
        for mode in ("dense", "derived", "contracted", "seeded")
    ]
    root = Path(__file__).resolve().parents[2]
    files = [
        "crates/pllm-garble/src/projected_polynomial.rs",
        "crates/pllm-python/src/projected_polynomial.rs",
        "python/pllm/metrics/projected_polynomial.py",
        "examples/benchmarks/projected_polynomial.py",
    ]
    return {
        "schema": "pllm.projected_polynomial_ablation.v1",
        "complete": True,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "code_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
        },
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for row in report["cases"]:
        print(
            f"{row['ring_bits']:2} bits {row['mode']:10}: {row['offline_body_bytes']:6} offline + "
            f"{row['online_body_bytes']:5} online bytes; exact={row['exact_modular_parity']}"
        )
