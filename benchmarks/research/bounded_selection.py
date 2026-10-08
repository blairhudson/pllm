"""Matched full-ring and public-bounded metadata conversions for complete top-k."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import time

from benchmarks.research.shared_selection import ROOT, SOURCES, spec

FILES = SOURCES + ("crates/pllm-garble/src/shared_selection/batch.rs",
                   "benchmarks/research/bounded_selection.py")
CASES = ((16, 257), (39, 257), (256, 257))


def worker(rows, width, bounded):
    from pllm import _native

    p = {**spec(rows, width, "top-k"), "batch_size": 16, "bounded_metadata": bounded}
    words = [(i * 0x87abcdef) & 0xffffffff for i in range(rows * width)]
    scores = [float(i * 7 % 11) / 16 for i in range(rows)]
    for i in range(rows):
        words[i * width] = 2 * i + 1
    keep, recent = p["selection"]["keep"], p["selection"]["recent"]
    indices = sorted(sorted(range(rows), key=lambda i: (-(i >= rows-recent), -scores[i], i))[:keep])
    expected = [v for i in indices for v in words[i * width:(i+1) * width]]
    start = time.process_time()
    output, degrees, raw = _native.shared_selection_benchmark(json.dumps(p),
        struct.pack(f"<{rows}f", *scores), struct.pack(f"<{len(words)}I", *words))
    cpu = time.process_time() - start
    assert not degrees and output == struct.pack(f"<{len(expected)}I", *expected)
    measured = json.loads(raw)
    projected = json.loads(_native.shared_selection_projection(json.dumps(p), keep))
    assert {k: v for k, v in measured["resources"].items() if k != "phases"} == {
        k: v for k, v in projected["resources"].items() if k != "phases"}
    return {"specification": p, "measurement": measured, "checked_outputs": len(expected),
            "cpu_seconds": cpu, "clear_state_bytes_retired": (rows - keep) * width * 4}


def run(samples):
    def hashes():
        return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}
    sources, cases = hashes(), []
    for rows, width in CASES:
        for bounded in (False, True):
            reports = []
            for _ in range(samples):
                proc = subprocess.run([sys.executable, "-m", "benchmarks.research.bounded_selection",
                    "--worker", str(rows), str(width), str(int(bounded))], cwd=ROOT, check=True,
                    capture_output=True, text=True, timeout=900,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                reports.append(json.loads(proc.stdout))
            resources = [r["measurement"]["resources"] for r in reports]
            assert all(r == resources[0] for r in resources)
            cases.append({"rows": rows, "words": width, "batch_size": 16,
                "bounded_metadata": bounded, "reports": reports,
                "median_seconds": {k: statistics.median(r["measurement"][k] for r in reports)
                                   for k in ("issuance_seconds", "online_seconds")}})
    assert sources == hashes(), "implementation changed during measurement"
    return {"schema": "pllm.bounded_selection_cost.v1", "source_sha256": sources, "cases": cases,
        "fresh_processes": len(cases) * samples, "samples_per_case": samples,
        "checked_outputs": sum(r["checked_outputs"] for c in cases for r in c["reports"]),
        "scope": {"trusted_local_dealer": True, "party_local_state": True, "provider_transport": False,
                  "measured_peak_memory": None, "checkpoint_quality": None, "full_wire": None,
                  "decoder_execution": False, "scoring_protected": False, "live_owner_binding": False,
                  "numeric_contract": "exact nonnegative finite f32 scores, complete ring32 records",
                  "rounds": "in-process exchanges; no WAN measurement"},
        "environment": {"platform": platform.platform(), "python": platform.python_version()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        rows, width, bounded = map(int, args.worker)
        print(json.dumps(worker(rows, width, bool(bounded))))
    else:
        if args.output is None or args.output.exists():
            parser.error("provide a new output path")
        result = run(args.samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("fresh_processes", "checked_outputs")}))
