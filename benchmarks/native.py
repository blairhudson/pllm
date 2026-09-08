"""Profile exact modular matrix execution. This is not a token throughput benchmark."""
from __future__ import annotations
import argparse
import cProfile
import json
import platform
import statistics
import time
from pathlib import Path
import numpy as np
from pllm.runtime.native import MaskedGEMM

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("inputs", 768), ("outputs", 256), ("batch", 16), ("samples", 10), ("threads", 1)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--output", type=Path, default=Path("results/native.json"))
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if min(args.inputs, args.outputs, args.batch, args.samples, args.threads) < 1:
        parser.error("Dimensions, samples and threads must be positive")
    rng = np.random.default_rng(2026)  # Public benchmark fixtures, never cryptographic masks.
    w = rng.integers(-7, 8, (args.outputs, args.inputs), dtype=np.int8)
    x = rng.integers(0, 786433, (args.batch, args.inputs), dtype=np.uint32)
    kernel = MaskedGEMM(threads=args.threads)
    reference = lambda: ((x.astype(np.int64) @ w.astype(np.int64).T) % 786433).astype(np.uint32)
    expected = reference()
    stage = kernel.compile(w)
    np.testing.assert_array_equal(stage.modular(x, 786433), expected)
    report = {"scope": "modular matrix kernel", "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "native": kernel.native, "backend": kernel.backend, "weight_compile_excluded": True, "inputs": args.inputs, "outputs": args.outputs, "batch": args.batch, "threads": args.threads, "warmups": 2, "results": {}}
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    for name, function in (("numpy", reference), ("selected", lambda: stage.modular(x, 786433))):
        for _ in range(2):
            function()
        raw = []
        for _ in range(args.samples):
            start = time.perf_counter_ns()
            actual = function()
            raw.append((time.perf_counter_ns() - start) / 1e6)
            np.testing.assert_array_equal(actual, expected)
        report["results"][name] = {"raw_ms": raw, "median_ms": statistics.median(raw), "p95_ms": float(np.percentile(raw, 95))}
    if profiler:
        profiler.disable()
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
