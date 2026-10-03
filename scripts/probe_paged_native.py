#!/usr/bin/env python3
"""Isolated-process exact native head kernels with bounded file-backed weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from pllm.native import MaskedGEMM, PagedGEMM


def fingerprint(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def inputs(cols):
    rng = np.random.default_rng(981)
    return (
        rng.integers(-128, 128, (1, cols), dtype=np.int8),
        rng.integers(0, 2**32, (1, cols), dtype=np.uint32),
    )


def worker(args):
    config = json.loads(args.manifest.read_text())
    rows, cols = config["shape"]
    signed, masked = inputs(cols)
    initial_rss = rss()
    start = time.process_time_ns()
    weights = None
    if args.worker == "resident":
        weights = np.fromfile(config["raw_weights"], dtype=np.int8).reshape(rows, cols)
        matrix = MaskedGEMM(threads=1).compile(weights)
        storage = {
            "source_array_bytes": weights.nbytes,
            "native_weight_snapshot_bytes": weights.nbytes,
            "private_snapshot_disk_bytes": 0,
            "page_metadata_bytes": 0,
            "transient_weight_buffer_bound": 0,
        }
    else:
        artifact = config["artifacts"][args.worker]
        matrix = PagedGEMM(artifact["path"], artifact["digest"], threads=1)
        storage = {
            "source_array_bytes": 0,
            "native_weight_snapshot_bytes": 0,
            "private_snapshot_disk_bytes": matrix.artifact_bytes,
            "page_metadata_bytes": matrix.metadata_bytes,
            "transient_weight_buffer_bound": matrix.transient_weight_bytes,
        }
    import_cpu = (time.process_time_ns() - start) / 1e9
    if fingerprint(matrix.clear(signed)) != config["expected"]["clear"]:
        raise AssertionError("paged warmup differs from native source arithmetic")
    cpu, wall = [], []
    for _ in range(5):
        c, w = time.process_time_ns(), time.perf_counter_ns()
        result = matrix.clear(signed)
        cpu.append((time.process_time_ns() - c) / 1e9)
        wall.append((time.perf_counter_ns() - w) / 1e9)
        if fingerprint(result) != config["expected"]["clear"]:
            raise AssertionError("paged clear result differs")
    if fingerprint(matrix.wrap32(masked)) != config["expected"]["wrap32"]:
        raise AssertionError("paged wrap32 result differs")
    start = time.process_time_ns()
    if args.worker == "resident":
        assert weights is not None
        gathered = weights[[0, rows // 2, rows - 1]]
    else:
        gathered = matrix.gather_rows([0, rows // 2, rows - 1])
    gather_cpu = (time.process_time_ns() - start) / 1e9
    if fingerprint(gathered) != config["expected"]["gather"]:
        raise AssertionError("paged local token lookup differs")
    result = {
        "storage": args.worker,
        "shape": [rows, cols],
        "batch_rows": 1,
        "exact_clear_wrap32_and_gather": True,
        "client_import_cpu_seconds": import_cpu,
        "clear_cpu_seconds_median": statistics.median(cpu),
        "clear_wall_seconds_median": statistics.median(wall),
        "three_row_gather_cpu_seconds": gather_cpu,
        "process_peak_rss_bytes": rss(),
        "process_initial_peak_rss_bytes": initial_rss,
        "measured_repetitions": 5,
        **storage,
    }
    if args.worker != "resident":
        matrix.close()
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinned", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("resident", "raw", "zlib"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.output is None:
        parser.error("--output is required")
    if args.pinned:
        from probe_private_pages import pinned_boundary

        owner, weights, _, source = pinned_boundary()
    else:
        owner = None
        weights = np.random.default_rng(823).integers(-32, 33, (8192, 128), dtype=np.int8)
        source = {"kind": "public_generated_integer_weights"}
    rows, cols = weights.shape
    signed, masked = inputs(cols)
    reference = MaskedGEMM(threads=1).compile(weights)
    expected = {
        "clear": fingerprint(reference.clear(signed)),
        "wrap32": fingerprint(reference.wrap32(masked)),
        "gather": fingerprint(weights[[0, rows // 2, rows - 1]]),
    }
    del reference
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "pllm.paged_native_probe.v1",
        "source": source,
        "scope": "isolated-process one-row exact head kernel; warm filesystem cache",
        "platform": platform.platform(),
        "artifacts": {},
        "candidates": [],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            b"".join(
                (Path(__file__).resolve().parents[1] / name).read_bytes()
                for name in (
                    "crates/pllm-core/src/paged.rs",
                    "crates/pllm-core/src/kernels.rs",
                    "crates/pllm-python/src/paged.rs",
                    "python/pllm/runtime/paged.py",
                )
            )
        ).hexdigest(),
        "limitations": [
            "Public artifact sizes are serialized-file counts, not measured network transfers",
            "RSS measures processes, excluding kernel filesystem cache and total device memory",
            "Source artifact and private immutable snapshot consume separate disk storage",
            "Weight-buffer bound excludes I/O/output buffers, kernel metadata, stacks and OS cache",
            "Matrix arithmetic and gathers only; whole decoder and Pipeline integration remain unmeasured",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="pllm-paged-probe-") as directory:
        root = Path(directory)
        raw_path = root / "weights.i8"
        weights.tofile(raw_path)
        manifest = {
            "shape": [rows, cols],
            "raw_weights": str(raw_path),
            "artifacts": {},
            "expected": expected,
        }
        report["raw_weight_bytes"] = weights.nbytes
        for compression in ("raw", "zlib"):
            path = root / f"{compression}.pllm"
            start = time.process_time_ns()
            digest = PagedGEMM.export(weights, path, page_rows=512, compression=compression)
            report["artifacts"][compression] = {
                "digest": digest,
                "serialized_bytes": path.stat().st_size,
                "publisher_export_cpu_seconds": (time.process_time_ns() - start) / 1e9,
                "page_rows": 512,
            }
            manifest["artifacts"][compression] = {"path": str(path), "digest": digest}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        for mode in ("resident", "raw", "zlib"):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    mode,
                    "--manifest",
                    str(manifest_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
                close_fds=False,
                env=os.environ.copy(),
            )
            result = json.loads(completed.stdout)
            report["candidates"].append(result)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(
                mode,
                "exact",
                "peak RSS",
                result["process_peak_rss_bytes"],
                "clear CPU",
                result["clear_cpu_seconds_median"],
                flush=True,
            )
    _ = owner


if __name__ == "__main__":
    main()
