"""Fresh-process synthetic public-weight import and shared-snapshot ablation.

Both choices stream and authenticate identical raw i8 weights and re-hash actual
snapshot bytes for bundle validation. The control then makes the previous second
private native copy; the candidate binds the same immutable native file. Public
source files are pre-positioned and OS cache is uncontrolled. Integer outputs use
an independent bounded NumPy int64 oracle. This excludes checkpoint values, full
SDK responses, provider traffic, filesystem-cache and whole-device memory costs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

SHAPES = {"qwen25-shaped": (151936, 896), "qwen3-4b-shaped": (151936, 2560)}


def publish(root, shape):
    import numpy as np

    rows, cols = shape
    rng = np.random.default_rng(81307)
    inputs = ((np.arange(cols, dtype=np.int64) * 37) % 255 - 127).astype(np.int8)
    oracle = np.empty(rows, dtype="<i4")
    digest = hashlib.sha256()
    with (root / "weights").open("xb") as output:
        for start in range(0, rows, 128):
            weights = rng.integers(-128, 128, (min(128, rows - start), cols), dtype=np.int8)
            data = weights.tobytes()
            output.write(data)
            digest.update(data)
            oracle[start:start + len(weights)] = weights.astype(np.int64) @ inputs.astype(np.int64)
    (root / "oracle").write_bytes(oracle.tobytes())
    (root / "descriptor.json").write_text(json.dumps({
        "shape": shape, "sha256": digest.hexdigest(),
        "oracle_sha256": hashlib.sha256(oracle.tobytes()).hexdigest(), "seed": 81307}))


def worker(root, choice):
    import numpy as np
    import psutil
    from pllm import _native
    from pllm.runtime.bundle_storage import FilePayload, binary_digest
    from pllm.runtime.paged import PagedGEMM

    descriptor = json.loads((root / "descriptor.json").read_bytes())
    rows, cols = descriptor["shape"]
    size = rows * cols
    swap_before = psutil.swap_memory().used
    started, cpu = time.perf_counter(), time.process_time()
    with (root / "weights").open("rb") as source:
        payload = FilePayload(iter(lambda: source.read(1 << 20), b""), size=size,
                              digest=descriptor["sha256"], native=choice == "shared")
    payload_cpu = time.process_time() - cpu
    matrix = None
    try:
        phase = time.process_time()
        if binary_digest(payload) != descriptor["sha256"]:
            raise AssertionError("actual snapshot bytes differ")
        validation_cpu = time.process_time() - phase
        phase = time.process_time()
        executor = _native.Executor(1, True)
        matrix = PagedGEMM._from_payload_executor(payload, (rows, cols), executor)
        binding_cpu = time.process_time() - phase
        import_cpu = time.process_time() - cpu
        import_elapsed = time.perf_counter() - started
        payload.close()
        inputs = ((np.arange(cols, dtype=np.int64) * 37) % 255 - 127).astype(np.int8).reshape(1, -1)
        phase = time.process_time()
        values = matrix.clear(inputs)
        execution_cpu = time.process_time() - phase
        expected = np.frombuffer((root / "oracle").read_bytes(), "<i4")
        np.testing.assert_array_equal(values[0], expected)
        if matrix.weight_digest != descriptor["sha256"]:
            raise AssertionError("native weight identity differs")
        return {"choice": choice, "shape": [rows, cols], "weight_sha256": descriptor["sha256"],
                "output_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "checked_outputs": rows, "public_source_bytes": size,
                "peak_private_snapshot_logical_bytes": size * (1 if choice == "shared" else 2),
                "retained_private_snapshot_logical_bytes": size,
                "payload_cpu_seconds": payload_cpu, "validation_cpu_seconds": validation_cpu,
                "binding_cpu_seconds": binding_cpu, "import_cpu_seconds": import_cpu,
                "import_elapsed_seconds": import_elapsed, "execution_cpu_seconds": execution_cpu,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if platform.system() == "Darwin" else 1024),
                "swap_growth_bytes": max(0, psutil.swap_memory().used - swap_before)}
    finally:
        payload.close()
        if matrix is not None:
            matrix.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--worker", choices=("copied", "shared"), help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.root, args.worker)))
        return
    if args.output is None or args.output.exists():
        parser.error("a new --output path is required")
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory

    host = host_memory()
    if host.available - host.reserve < GiB or host.disk_free < 3 * GiB:
        raise RuntimeError("probe requires 1 GiB post-reserve RAM and 3 GiB disk headroom")
    active, samples = [], []

    def abort(_reason):
        for child in tuple(active):
            if child.poll() is None:
                child.terminate()

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    guard.start()
    try:
        with tempfile.TemporaryDirectory(prefix="pllm-public-snapshot-", dir=args.output.parent) as directory:
            for name, shape in SHAPES.items():
                guard.check()
                root = Path(directory) / name
                root.mkdir()
                publish(root, shape)
                for repeat in range(args.samples):
                    for choice in (("copied", "shared") if repeat % 2 == 0 else ("shared", "copied")):
                        guard.check()
                        if guard.error:
                            raise RuntimeError(guard.error)
                        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                            "--worker", choice, "--root", str(root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        active.append(child)
                        try:
                            stdout, stderr = child.communicate(timeout=180)
                            guard.check()
                            if child.returncode or guard.error:
                                raise RuntimeError(guard.error or stderr.decode())
                            row = json.loads(stdout)
                            row.update(shape_name=name, repeat=repeat)
                            samples.append(row)
                        finally:
                            if child.poll() is None:
                                child.kill()
                                child.communicate()
                            active.remove(child)
    finally:
        abort("cleanup")
        guard.close()
    for name in SHAPES:
        cohort = [row for row in samples if row["shape_name"] == name]
        for field in ("weight_sha256", "output_sha256"):
            if len({row[field] for row in cohort}) != 1:
                raise AssertionError(f"{name} {field} differs")
    metrics = ("import_cpu_seconds", "import_elapsed_seconds", "execution_cpu_seconds", "peak_rss_bytes")
    medians = {name: {choice: {metric: statistics.median(row[metric] for row in samples
        if row["shape_name"] == name and row["choice"] == choice) for metric in metrics}
        for choice in ("copied", "shared")} for name in SHAPES}
    repo = Path(__file__).resolve().parents[1]
    paths = ("scripts/probe_public_snapshot.py", "crates/pllm-core/src/public_file.rs",
             "crates/pllm-core/src/paged.rs", "crates/pllm-python/src/paged.rs",
             "python/pllm/runtime/bundle_storage.py", "python/pllm/runtime/paged.py")
    result = {"schema": "pllm.public_snapshot_probe.v1", "scope": __doc__,
              "environment": {"python": sys.version, "platform": platform.platform()},
              "host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
              "implementation_sha256": {path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in paths},
              "samples": samples, "medians": medians}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
