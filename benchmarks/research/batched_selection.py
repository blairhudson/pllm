"""Matched serial/batched complete selection; independent fresh-process oracles."""
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

FILES = SOURCES + ("crates/pllm-garble/src/shared_selection/batch.rs", "benchmarks/research/batched_selection.py")
CASES = (("top-k", 16, 257), ("top-k", 256, 257), ("threshold", 8, 16), ("threshold", 39, 896))


def worker(method, rows, width, batch):
    from pllm import _native
    from pllm.metrics import benchmark_operator
    from pllm.passes import CipherPruneTokenPruning
    words = [(i * 0x87abcdef) & 0xffffffff for i in range(rows * width)]
    if method == "threshold":
        row = [0 if i % 2 == 0 else 65536 // (rows // 2) for i in range(rows)]
        row[1] += 65536 - sum(row)
        report = benchmark_operator(CipherPruneTokenPruning(backend="mixed", batch_size=batch,
            threshold_q16=1), lanes=width, samples=1, geometry={"tokens": rows, "heads": 14},
            inputs={"attention_q16": row * (14 * rows), "words": words}).to_dict()
        return {"report": report, "checked_outputs": report["quality"]["exact_integer_outputs"]}
    p = {**spec(rows, width, method), "batch_size": batch}
    scores = [float(i * 7 % 11) / 16 for i in range(rows)]
    for i in range(rows): words[i * width] = 2 * i + 1
    keep, recent = p["selection"]["keep"], p["selection"]["recent"]
    indices = sorted(sorted(range(rows), key=lambda i: (-(i >= rows-recent), -scores[i], i))[:keep])
    expected = [v for i in indices for v in words[i * width:(i+1) * width]]
    start = time.process_time()
    output, degrees, raw = _native.shared_selection_benchmark(json.dumps(p),
        struct.pack(f"<{rows}f", *scores), struct.pack(f"<{len(words)}I", *words))
    cpu = time.process_time() - start
    assert not degrees and output == struct.pack(f"<{len(expected)}I", *expected)
    return {"measurement": json.loads(raw), "checked_outputs": len(expected), "cpu_seconds": cpu}


def run(samples):
    hashes = lambda: {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}
    sources = hashes()
    cases = []
    for method, rows, width in CASES:
        for batch in (1, 4, 16):
            reports = []
            for _ in range(samples):
                proc = subprocess.run([sys.executable, "-m", "benchmarks.research.batched_selection",
                    "--worker", method, str(rows), str(width), str(batch)], cwd=ROOT, check=True,
                    capture_output=True, text=True, timeout=900,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                reports.append(json.loads(proc.stdout))
            resources = [r.get("report", r.get("measurement"))["resources"] for r in reports]
            assert all(r == resources[0] for r in resources)
            timings = [r["report"]["samples"][0] if "report" in r else r["measurement"] for r in reports]
            cases.append({"method": method, "rows": rows, "words": width, "batch_size": batch, "reports": reports,
                "median_seconds": {k: statistics.median(t[k] for t in timings) for k in ("issuance_seconds", "online_seconds")}})
    assert sources == hashes(), "implementation changed during measurement"
    return {"schema": "pllm.batched_selection_cost.v1", "source_sha256": sources, "cases": cases,
        "fresh_processes": len(cases) * samples, "samples_per_case": samples,
        "checked_outputs": sum(r["checked_outputs"] for c in cases for r in c["reports"]),
        "scope": {"trusted_local_dealer": True, "party_local_state": True, "provider_transport": False,
                  "measured_peak_memory": None, "checkpoint_quality": None, "full_wire": None,
                  "decoder_execution": False, "rounds": "actual coalesced in-process phase exchanges; no WAN latency claim"},
        "environment": {"platform": platform.platform(), "python": platform.python_version()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=4, help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        method, rows, width, batch = args.worker
        print(json.dumps(worker(method, int(rows), int(width), int(batch))))
    else:
        if args.output is None or args.output.exists(): parser.error("provide a new output path")
        result = run(args.samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: result[k] for k in ("fresh_processes", "checked_outputs")}))
