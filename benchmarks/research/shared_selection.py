"""Fresh-process mixed private selection with complete record movement costs.

Synthetic scores/records only. Top-k preserves complete position/K/V words;
CipherPrune controls use normalized Q16 attention. No decoder is executed.
"""
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

ROOT = Path(__file__).resolve().parents[2]
CASES = (
    ("prune-half-gates", 4, 16, "half"), ("prune-mixed", 4, 16, "half"),
    ("prune-half-gates", 4, 16, "all"), ("prune-mixed", 4, 16, "all"),
    ("prune-half-gates", 4, 16, "none"), ("prune-mixed", 4, 16, "none"),
    ("prune-mixed", 39, 896, "half"),
    ("top-k", 8, 5, "half"), ("top-k", 16, 257, "half"),
    ("top-k", 39, 257, "half"), ("top-k", 256, 257, "half"),
)
SOURCES = (
    "benchmarks/research/shared_selection.py", "crates/pllm-garble/src/shared_selection.rs",
    "crates/pllm-garble/src/shared_circuit.rs", "crates/pllm-garble/src/shared_arithmetic_reference.rs",
    "crates/pllm-garble/src/boolean.rs", "crates/pllm-garble/src/token_pruning.rs",
    "crates/pllm-core/src/token_pruning.rs", "crates/pllm-garble/src/retained_selection.rs",
    "crates/pllm-garble/src/retained_selection/cost.rs", "crates/pllm-python/src/shared_selection.rs",
    "python/pllm/metrics/shared_selection.py", "python/pllm/metrics/token_pruning.py",
    "python/pllm/metrics/operator_benchmarks.py", "python/pllm/passes/__init__.py", "Cargo.lock",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec(rows, words, mode, keep=None):
    if mode == "top-k":
        selection = {"method": "top_k", "keep": keep or max(1, rows // 2), "recent": min(2, rows // 2)}
    else:
        selection = {"method": "threshold", "threshold": 8192 * 14 * rows, "reduction": 24576 * 14 * rows}
    return {"rows": rows, "words": words, "selection": selection}


def worker(method, rows, words, case):
    from pllm import _native
    from pllm.metrics import benchmark_operator
    from pllm.passes import CipherPruneTokenPruning

    if method.startswith("prune-"):
        theta, beta = (0, 65536) if case == "all" else (65535, 65536) if case == "none" else (1, 24576)
        selected = list(range(1, rows, 2)) if case == "half" else list(range(rows))
        row = [0] * rows
        for i in selected:
            row[i] = 65536 // len(selected)
        row[selected[-1]] += 65536 - sum(row)
        assert sum(row) == 65536
        backend = "mixed" if method == "prune-mixed" else "half_gates"
        report = benchmark_operator(CipherPruneTokenPruning(
            threshold_q16=theta, reduction_threshold_q16=beta, backend=backend),
            geometry={"tokens": rows, "heads": 14}, lanes=words, samples=1,
            inputs={"attention_q16": row * (14 * rows),
                    "words": [(i * 0x87ABCDEF) & 0xFFFFFFFF for i in range(rows * words)]}).to_dict()
        retained = 0 if case == "none" else rows if case == "all" else rows // 2
        expected_checks = 1 + retained * words + (retained if retained < rows else 0)
        assert report["quality"]["exact_integer_outputs"] == expected_checks
        return {"checked_outputs": expected_checks, "retained": retained, "report": report}
    p = spec(rows, words, method)
    scores = [float(i * 7 % 11) / 16 for i in range(rows)]
    values = [(i * 0x87ABCDEF) & 0xFFFFFFFF for i in range(rows * words)]
    for i in range(rows):
        values[i * words] = 2 * i + 1
    keep, recent = p["selection"]["keep"], p["selection"]["recent"]
    selected = sorted(sorted(range(rows), key=lambda i: (-(i >= rows - recent), -scores[i], i))[:keep])
    expected = [v for i in selected for v in values[i * words:(i + 1) * words]]
    start = time.process_time()
    output, degrees, metadata = _native.shared_selection_benchmark(
        json.dumps(p), struct.pack(f"<{rows}f", *scores), struct.pack(f"<{len(values)}I", *values))
    cpu = time.process_time() - start
    measured = json.loads(metadata)
    if output != struct.pack(f"<{len(expected)}I", *expected) or degrees != b"" or measured["retained"] != keep:
        raise ValueError("complete retained words differ from independent stable top-k oracle")
    return {"specification": p, "checked_outputs": len(expected), "retained": keep,
            "measurement": measured, "native_call_cpu_seconds": cpu,
            "clear_kv_words_retired": (rows - keep) * words,
            "retained_owner_committed": False}


def projections():
    from pllm import _native
    result = []
    for rows, words, mode in ((39, 896, "threshold"), (150, 896, "threshold"),
                             (39, 257, "top-k"), (256, 257, "top-k")):
        p = spec(rows, words, mode)
        if mode == "threshold":
            # Fixed public theta selects no particular fraction on a checkpoint;
            # these are only cost geometries for the explicitly stated count.
            legacy = json.loads(_native.token_pruning_projection(rows, 14, words, 8192, 24576, rows // 2))
        else:
            legacy = None
        r = json.loads(_native.shared_selection_projection(json.dumps(p), rows // 2))
        result.append({"specification": p, "retained": rows // 2, "mixed": r,
                       "legacy_pruning": legacy, "payload_words_retired": (rows - rows // 2) * words,
                       "material_issued": False, "scope": "one synthetic full-width invocation"})
    return result


def run(samples=3):
    if type(samples) is not int or not 1 <= samples <= 8:
        raise ValueError("samples must be between one and eight")
    sources = {p: sha(ROOT / p) for p in SOURCES}
    cases = []
    for method, rows, words, case in CASES:
        reports = []
        for _ in range(samples):
            proc = subprocess.run([sys.executable, "-m", "benchmarks.research.shared_selection", "--worker",
                method, str(rows), str(words), case], cwd=ROOT, check=True, capture_output=True,
                text=True, timeout=900, env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
            reports.append(json.loads(proc.stdout))
        resources = [r["report"]["resources"] if "report" in r else r["measurement"]["resources"] for r in reports]
        if any(r != resources[0] for r in resources):
            raise ValueError("fixed public geometry/count changed resource counts")
        timings = [r["report"]["samples"][0] if "report" in r else r["measurement"] for r in reports]
        cases.append({"method": method, "rows": rows, "words": words, "fixture": case, "reports": reports,
                      "median_seconds": {key: statistics.median(r[key] for r in timings)
                                         for key in ("issuance_seconds", "online_seconds")}})
    if sources != {p: sha(ROOT / p) for p in SOURCES}:
        raise ValueError("selection implementation changed during measurement")
    return {"schema": "pllm.shared_selection_cost.v1", "source_sha256": sources,
            "cases": cases, "fresh_processes": len(cases) * samples, "samples_per_case": samples,
            "checked_outputs": sum(r["checked_outputs"] for c in cases for r in c["reports"]),
            "full_width_projections": projections(),
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "scope": {"inputs": "synthetic public normalized attention and canonical nonnegative float32 scores",
                      "payload": "arbitrary complete ring32 words, including full K/V and absolute positions for top-k",
                      "trusted_local_dealer": True, "party_local_primitive_state": True,
                      "distributed_scoring": False, "retained_owner_binding": False,
                      "decoder_execution": False, "checkpoint_quality": None, "provider_transport": False,
                      "independent_cryptographic_review": False, "measured_peak_memory": None, "full_wire": None}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--worker", nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        method, rows, words, case = args.worker
        print(json.dumps(worker(method, int(rows), int(words), case)))
    else:
        if args.output is None or args.output.exists():
            parser.error("provide a new --output path")
        result = run(args.samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: result[key] for key in ("fresh_processes", "checked_outputs")}))
