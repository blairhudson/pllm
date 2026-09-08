"""Matched Rust/C++/NumPy experiments; never report a fallback as Rust performance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def timed(function, samples, warmups, expected=None):
    for _ in range(warmups):
        result = function()
        if expected is not None:
            np.testing.assert_array_equal(result, expected)
    raw = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        result = function()
        raw.append((time.perf_counter_ns() - start) / 1e6)
        if expected is not None:
            np.testing.assert_array_equal(result, expected)
    return {"median_ms": statistics.median(raw), "p95_ms": float(np.percentile(raw, 95)), "samples_ms": raw}


def cpp_control(directory: Path, threads: int):
    import shutil
    if platform.machine() != "x86_64" or not shutil.which("g++"):
        return None
    if importlib.util.find_spec("pllm._native") is not None:
        from pllm import _native
        if not _native.capabilities()["avx2"]:
            return None
    elif Path("/proc/cpuinfo").exists():
        if "avx2" not in Path("/proc/cpuinfo").read_text().lower():
            return None
    else:
        return None  # Do not execute the old AVX2 binary without a feature check.
    directory.mkdir(parents=True, exist_ok=True)
    lib = directory / "previous-cpp.so"
    subprocess.run(["g++", "-O3", "-mavx2", "-fopenmp", "-shared", "-fPIC",
                    str(ROOT / "benchmarks/reference/masked_gemm.cpp"), "-o", str(lib)], check=True)
    spec = importlib.util.spec_from_file_location("cpp_control", ROOT / "benchmarks/reference/cpp_native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MaskedGEMM(lib, threads=threads)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("results/rust-migration.json"))
    p.add_argument("--cpp-reference", action="store_true")
    p.add_argument("--allow-missing-rust", action="store_true")
    p.add_argument("--qwen", action="store_true", help="include large synthetic projection dimensions, not model generation")
    p.add_argument("--profile", type=Path)
    args = p.parse_args()
    if args.samples < 1 or args.warmups < 0:
        p.error("samples must be positive and warmups nonnegative")
    available = importlib.util.find_spec("pllm._native") is not None
    if not available and not args.allow_missing_rust:
        p.error("compiled Rust extension missing; run uv sync first")
    if available:
        os.environ["PLLM_KERNEL_BACKEND"] = "rust"
    else:
        os.environ["PLLM_KERNEL_BACKEND"] = "python"
    from pllm.runtime.native import MaskedGEMM
    args.output.parent.mkdir(parents=True, exist_ok=True)
    control = cpp_control(args.output.parent, args.threads) if args.cpp_reference else None
    shapes = [(64, 64, 1), (768, 256, 1), (1536, 4096, 1), (1536, 4096, 16), (17408, 512, 1)]
    if args.qwen:
        shapes += [(5120, 34816, 1), (17408, 5120, 1), (5120, 34816, 16)]
    executor = MaskedGEMM(threads=args.threads)
    assert executor.native == available, "Never label a reference kernel as Rust"
    profiler = None
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    rows = []
    for n, m, b in shapes:
        rng = np.random.default_rng(n + m + b)
        w = rng.integers(-7, 8, (m, n), dtype=np.int8)
        modulus = 33554467 if n >= 8192 else 786433
        x = rng.integers(0, modulus, (b, n), dtype=np.uint32)
        expected = (x.astype(np.int64) @ w.astype(np.int64).T % modulus).astype(np.uint32)
        start = time.perf_counter_ns()
        compiled = executor.compile(w)
        compile_ms = (time.perf_counter_ns() - start)/1e6
        np.testing.assert_array_equal(compiled.modular(x, modulus), expected)
        row = {"shape": [b, n, m], "modulus": modulus, "matrix_compile_ms": compile_ms,
               "native_weight_bytes": compiled.weight_bytes, "exact": True,
               "timing_scope": "Python call through compiled stage, including validation and buffer conversion"}
        if available:
            row["rust"] = timed(lambda: compiled.modular(x, modulus), args.samples, args.warmups, expected)
            scalar = MaskedGEMM(threads=args.threads, simd=False).compile(w)
            np.testing.assert_array_equal(scalar.modular(x, modulus), expected)
            row["rust_scalar"] = timed(lambda: scalar.modular(x, modulus), args.samples, args.warmups, expected)
        else:
            row["rust"] = {"status": "not_run", "reason": "Rust toolchain and compiled extension unavailable"}
        row["numpy_reference"] = timed(lambda: (x.astype(np.int64) @ w.astype(np.int64).T % modulus), min(args.samples,5), args.warmups, expected)
        if control:
            observed = control.modular(w, x, modulus)
            row["cpp_exact"] = bool(np.array_equal(observed, expected))
            row["cpp"] = timed(lambda: control.modular(w,x,modulus), args.samples,args.warmups, expected if row["cpp_exact"] else None)
            if available and row["cpp_exact"]:
                row["speedup_vs_cpp"] = row["cpp"]["median_ms"]/row["rust"]["median_ms"]
        rows.append(row)
    if profiler:
        profiler.disable()
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.profile)
    # Keep a concrete arithmetic regression beside the performance comparison.
    regression = None
    if control:
        w=np.full((1,8),7,np.int8); x=np.full((1,8),2147483646,np.uint32); p=2147483647
        regression={"cpp":control.modular(w,x,p).tolist(), "exact":((x.astype(np.int64)@w.astype(np.int64).T)%p).tolist()}
    result={"scope":"Synthetic matrix kernels, not encrypted generation or model TPS", "rust_executed":available,
            "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"threads":args.threads,
                           "cpu":platform.processor(),"cpu_quota":Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None},
            "matrices":rows,"old_cpp_overflow":regression}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"rust_executed":available,"matrix_cases":len(rows)},indent=2))

if __name__=="__main__":
    main()
