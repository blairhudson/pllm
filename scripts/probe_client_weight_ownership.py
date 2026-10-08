"""Fresh-process client snapshot ownership probe with exact boundary oracles.

Synthetic signed-i8 weights exercise the ordinary ClientBundle token/head path.
The retained-import control holds the source metadata across execution, matching
the old bundle's duplicate ownership. Its numeric kernels are unchanged. Phase
RSS samples and process peaks are separate; this is not whole-decoder evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import weakref


def worker(mode, rows, columns):
    import gc
    import numpy as np
    import psutil
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime.transformer_client import ClientBundle, StageMetadata

    phases = []
    process = psutil.Process()
    start, cpu = time.perf_counter(), time.process_time()

    def sample(name):
        phases.append({"phase": name, "rss_bytes": process.memory_info().rss,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
            "elapsed_seconds": time.perf_counter() - start,
            "process_cpu_seconds": time.process_time() - cpu})

    sample("imports")
    # Construct immutable import bytes in bounded rows. The row formula gives
    # an independent integer oracle without a second full-width weight owner.
    row = (np.arange(columns, dtype=np.int32) % 255 - 127).astype(np.int8)
    weights = np.frombuffer(row.tobytes() * rows, dtype=np.int8).reshape(rows, columns)
    scales = np.full(rows, np.float32(0.125), dtype=np.float32)
    head = StageMetadata("lm_head", "linear", columns, rows, 8, 8, "pow2", 1 << 32, 32,
                         scales, client_weight=weights, client_weight_scales=scales)
    table = StageMetadata("token_lookup", "embedding", rows, columns, 8, 8, "pow2", 1 << 32, 32,
                          np.ones(columns, np.float32), client_weight=weights,
                          client_weight_scales=scales, client_weight_layout="embedding")
    retained = (head, table) if mode == "retained-import" else None
    bundle = ClientBundle("synthetic", {}, {}, {},
        {"lm_head": head, "token_lookup": table, "embed_tokens": table}, {}, {}, schema_version=2)
    bundle._local_kernel = MaskedGEMM(threads=4)
    if not bundle._local_kernel.native:
        raise RuntimeError("ownership measurement requires the native snapshot backend")
    imported = weakref.ref(weights)
    weight_digest = hashlib.sha256(memoryview(weights)).hexdigest()
    del weights, head, table
    sample("bundle_import")
    tokens = np.array([0, rows // 2, rows - 1])
    expected_lookup = np.tile(row.astype(np.float32) * np.float32(0.125), (3, 1))
    np.testing.assert_array_equal(bundle.local_token_lookup(tokens), expected_lookup)
    sample("token_lookup")
    inputs = np.zeros((1, columns), dtype=np.float32)
    inputs[0, 0] = 127  # exact activation scale 1; remaining inputs are zero
    expected_head = np.full((1, rows), int(row[0]) * 127 * np.float32(0.125), np.float32)
    first = bundle.local_linear("lm_head", inputs)
    np.testing.assert_array_equal(first, expected_head)
    gc.collect()
    if (imported() is None) != (mode == "single-snapshot"):
        raise RuntimeError("imported owner did not follow the selected lifetime")
    sample("first_head")
    for _ in range(3):
        np.testing.assert_array_equal(bundle.local_linear("lm_head", inputs), expected_head)
        np.testing.assert_array_equal(bundle.local_token_lookup(tokens), expected_lookup)
    sample("repeated_boundaries")
    assert bundle.stages["token_lookup"].client_weight is bundle.stages["lm_head"].client_weight
    assert np.shares_memory(bundle.stages["lm_head"].client_weight,
                            bundle._local_matrices["lm_head"].weight_view())
    result = {"mode": mode, "weight_sha256": weight_digest,
        "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
        "matrix_bytes": rows * columns,
        "retained_matrix_payload_bytes": rows * columns * (2 if retained else 1),
        "import_array_retired": imported() is None,
        "checked_outputs": 4 * (rows + 3 * columns), "phases": phases}
    del bundle, retained
    gc.collect()
    sample("retired")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("retained-import", "single-snapshot"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (16 <= args.rows <= 200000 and 8 <= args.columns <= 4096
            and args.rows * args.columns <= 512 << 20 and 1 <= args.repeats <= 5):
        parser.error("bounded geometry requires <=512 MiB weights and 1..5 repeats")
    if args.worker:
        print(json.dumps(worker(args.worker, args.rows, args.columns)))
        return

    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    host = host_memory()
    required = max(2 * GiB, 4 * args.rows * args.columns + GiB)
    if host.available - host.reserve < required:
        raise RuntimeError(f"probe requires {required} bytes beyond host reserve")
    active = []

    def abort(_reason):
        for process in tuple(active):
            if process.poll() is None:
                process.terminate()

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    guard.start()
    try:
        samples = []
        for repeat in range(args.repeats):
            for mode in ("retained-import", "single-snapshot"):
                guard.check()
                if guard.error:
                    raise RuntimeError(guard.error)
                process = subprocess.Popen([sys.executable, __file__, "--worker", mode,
                    "--rows", str(args.rows), "--columns", str(args.columns)],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                active.append(process)
                try:
                    stdout, stderr = process.communicate(timeout=120)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    active.remove(process)
                guard.check()
                if process.returncode or guard.error:
                    raise RuntimeError(guard.error or stderr)
                samples.append(json.loads(stdout) | {"repeat": repeat})
        if len({(s["weight_sha256"], s["output_sha256"]) for s in samples}) != 1:
            raise RuntimeError("control/candidate numeric identity mismatch")
        from pllm.runtime import transformer_client
        report = {"schema": "pllm.client_weight_ownership_probe.v1", "scope": __doc__,
            "shape": [args.rows, args.columns], "samples": samples, "parity": True,
            "host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])):
                hashlib.sha256(path.read_bytes()).hexdigest() for path in
                (Path(__file__).resolve(), Path(transformer_client.__file__).resolve())}}
        if args.output:
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(json.dumps(report, indent=2))
    finally:
        abort("cleanup")
        guard.close()


if __name__ == "__main__":
    main()
