"""Bounded isolated-process loader/snapshot memory control; no decoder/WAN claim.

Run from repository root with the installed native extension. Four 4096-square
stages have 64 MiB of quantized weights and 256 MiB of float32 source artifacts.
The legacy control recreates whole-float materialization and duplicate retained
i8 storage. It is a mechanism control, not a historical full-engine benchmark.
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
import tempfile
import time


def worker(root: Path, mode: str, width: int, count: int) -> dict:
    import numpy as np
    import psutil
    from pllm.runtime.models import ModelManifest, StageSpec
    from pllm.runtime.quantization import quantize_weight_per_row
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    store = SafeTensorStore(root)
    manifest = ModelManifest("weight-memory-probe", "isolated-linear-stages", "safetensors", str(root),
                             width, width, width, count, 1, 1, width, 1,
                             metadata={"stage_origin": "semantic_schedule_v1"})
    x = np.random.default_rng(710).integers(-7, 8, (1, width), dtype=np.int8)
    retained, outputs, weights, bounds = [], [], [], []
    start_cpu, start = time.process_time(), time.perf_counter()
    for index in range(count):
        key = f"matrix.{index}.weight"
        stage = StageSpec(f"matrix_{index}", "linear", width, width,
                          weight_keys=(key,), weight_bits=8, activation_bits=8)
        if mode == "legacy":
            weight = quantize_weight_per_row(store.get_linear((key,)), bits=8)
            matrix = engine.kernel.compile(weight.values)
            digest = hashlib.sha256(weight.values.tobytes()).hexdigest()
            bound = int(np.abs(weight.values.astype(np.int16)).sum(axis=1, dtype=np.int64).max()) * 127
            retained.append((weight, matrix))
        else:
            loaded = engine._load_stage(store, stage, manifest)
            matrix, weight = loaded.compiled_weight, loaded.weight
            digest, bound = loaded.weight_digest, loaded.signed_output_bound
            assert np.shares_memory(weight.values, matrix.weight_view())
            retained.append(loaded)
        store.drop(key)
        weights.append(digest)
        bounds.append(bound)
        outputs.append(hashlib.sha256(matrix.clear(x).tobytes()).hexdigest())
    elapsed, cpu = time.perf_counter() - start, time.process_time() - start_cpu
    return {
        "mode": mode, "weight_digests": weights, "output_digests": outputs, "signed_bounds": bounds,
        "quantized_weight_bytes": count * width * width,
        "distinct_weight_storage_bytes": count * width * width * (2 if mode == "legacy" else 1),
        "end_rss_bytes": psutil.Process().memory_info().rss,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if platform.system() == "Darwin" else 1024),
        "load_and_one_row_cpu_seconds": cpu, "load_and_one_row_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("legacy", "snapshot"), help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 16 <= args.width <= 4096 or not 1 <= args.stages <= 4:
        parser.error("probe bound: width 16..4096 and stages 1..4")
    if args.worker:
        print(json.dumps(worker(args.root, args.worker, args.width, args.stages)))
        return

    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory

    host = host_memory()
    if host.available - host.reserve < 2 * GiB or host.disk_free < 3 * GiB:
        raise RuntimeError("probe needs 2 GiB of admitted RAM and 3 GiB of free disk")
    active = []

    def abort(_reason):
        for process in tuple(active):
            if process.poll() is None:
                process.terminate()

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    guard.start()
    try:
        import numpy as np
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory(prefix="pllm-weight-memory-") as directory:
            root = Path(directory)
            rng = np.random.default_rng(710)
            for index in range(args.stages):
                # Separate artifacts bound fixture construction to one float matrix.
                values = rng.standard_normal((args.width, args.width), dtype=np.float32)
                save_file({f"matrix.{index}.weight": values}, root / f"weights-{index}.safetensors")
            del values
            samples = []
            for mode in ("legacy", "snapshot"):
                guard.check()
                if guard.error:
                    raise RuntimeError(guard.error)
                process = subprocess.Popen([sys.executable, __file__, "--worker", mode, "--root", str(root),
                    "--width", str(args.width), "--stages", str(args.stages)],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                active.append(process)
                try:
                    stdout, stderr = process.communicate(timeout=120)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    active.remove(process)
                if process.returncode or guard.error:
                    raise RuntimeError(guard.error or stderr)
                samples.append(json.loads(stdout))
        for key in ("weight_digests", "output_digests", "signed_bounds"):
            assert samples[0][key] == samples[1][key], key
        report = {"schema": "pllm.weight_memory_probe.v1", "scope": __doc__,
                  "width": args.width, "stages": args.stages, "samples": samples,
                  "source_float_bytes": args.width**2 * args.stages * 4,
                  "host": asdict(host), "max_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                  "measured_at": datetime.now(timezone.utc).isoformat(),
                  "parity": True, "python": platform.python_version(), "platform": platform.platform()}
        if args.output:
            with args.output.open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        print(json.dumps(report, indent=2))
    finally:
        abort("cleanup")
        guard.close()


if __name__ == "__main__":
    main()
