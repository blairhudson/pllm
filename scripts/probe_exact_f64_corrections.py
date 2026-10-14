#!/usr/bin/env python3
"""Bounded exact-integer FP64 correction experiment; not a production backend."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_ELEMENTS = 4 << 20
MAX_ORACLE_MACS = 2_000_000
WORKSPACE = 64 << 20
EXACT = 1 << 53
FIXTURE = ROOT / "scripts/research_native/exact_f64_corrections/accelerate.py"


def bounds(bits):
    if type(bits) is not int or bits not in (16, 24, 32):
        raise ValueError("only ring16/24/32 supported")
    mask = (1 << bits) - 1
    return {"bits": bits, "mask": mask, "product_min": -128 * mask,
            "product_max": 127 * mask, "exact_integer_limit": EXACT,
            "universal_width": EXACT // (128 * mask),
            "source_l1_limit": EXACT // mask,
            "float_subtraction_universal_width": (EXACT - mask) // (128 * mask)}


def validate(weights, inputs, bits, output_masks=None):
    proof = bounds(bits)
    if (not isinstance(weights, np.ndarray) or weights.dtype != np.int8
            or weights.ndim != 2 or not isinstance(inputs, np.ndarray)
            or inputs.dtype != np.uint32 or inputs.ndim != 2):
        raise ValueError("weights must be rank-2 i8; inputs rank-2 u32")
    rows, cols = weights.shape
    batch = inputs.shape[0]
    if (min(rows, cols, batch) <= 0 or batch > 4096 or inputs.shape[1] != cols
            or max(weights.size, inputs.size, batch * rows) > MAX_ELEMENTS):
        raise ValueError("bounded geometry exceeded")
    if np.any(inputs > proof["mask"]):
        raise ValueError("input outside ring; no implicit narrowing")
    if output_masks is not None and (
            not isinstance(output_masks, np.ndarray) or output_masks.dtype != np.uint32
            or output_masks.shape != (batch, rows)
            or np.any(output_masks > proof["mask"])):
        raise ValueError("output mask outside ring or shape")
    return proof


def integer_oracle(weights, inputs, bits, output_masks=None):
    """Independent Python big-integer dot/subtract, then ring narrowing."""
    proof = validate(weights, inputs, bits, output_masks)
    if inputs.shape[0] * weights.size > MAX_ORACLE_MACS:
        raise ValueError("oracle MAC budget exceeded")
    result = np.empty((inputs.shape[0], weights.shape[0]), np.uint32)
    for sample, x in enumerate(inputs):
        for row, w in enumerate(weights):
            total = sum(int(a) * int(b) for a, b in zip(w, x))
            if output_masks is not None:
                total -= int(output_masks[sample, row])
            result[sample, row] = total & proof["mask"]
    return result


def source_bound(weights, bits):
    """Computed from actual immutable snapshot, never a caller-provided claim."""
    proof = bounds(bits)
    if (not isinstance(weights, np.ndarray) or weights.dtype != np.int8
            or weights.ndim != 2 or min(weights.shape) <= 0 or weights.size > MAX_ELEMENTS):
        raise ValueError("source certificate requires bounded rank-2 signed-i8 snapshot")
    # Widen before abs: abs(int8(-128)) is NOT 128.
    l1 = np.abs(weights.astype(np.int64)).sum(axis=1, dtype=np.int64)
    return int(l1.max(initial=0)) * proof["mask"]


def checked_integer(partial, limit):
    if (partial.dtype != np.float64 or not np.all(np.isfinite(partial))
            or np.any(np.abs(partial) > limit)
            or np.any(partial != np.trunc(partial)) or limit > EXACT):
        raise ValueError("FP64 result violates certified integer domain")
    # [-2^53,2^53] is safely inside int64; never cast FP64 directly to u32/i32.
    return partial.astype(np.int64)


def accelerate_module():
    spec = importlib.util.spec_from_file_location("exact_f64_accelerate", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project(weights, inputs, bits, output_masks=None, *, block=None, tile=1024,
            certified=False, backend="numpy", handle=None):
    proof = validate(weights, inputs, bits, output_masks)
    rows, cols = weights.shape
    batch = inputs.shape[0]
    if block is None:
        block = min(cols, proof["universal_width"])
    if type(block) is not int or not 1 <= block <= cols:
        raise ValueError("invalid block width")
    if type(tile) is not int or not 1 <= tile <= 1024:
        raise ValueError("invalid output tile")
    if block > proof["universal_width"] and not certified:
        raise ValueError("unsound width; source certificate required")
    if backend not in ("numpy", "accelerate"):
        raise ValueError("unknown FP64 backend")
    n = min(rows, tile)
    # Conservative live Python-array accounting incl. finite/fract checks and
    # int64 conversion/reconstruction temporaries, not only DGEMM operands.
    work = max(32 * batch * rows,
               4 * batch * rows + 8 * (batch + n) * block + 48 * batch * n
               + (8 * n * block if certified else 0))
    if work > WORKSPACE:
        raise ValueError("fixture workspace exceeds 64 MiB")
    native = accelerate_module() if backend == "accelerate" else None
    if native is not None and handle is None:
        handle = native.load()
    out = np.zeros((batch, rows), np.uint32)
    for start in range(0, cols, block):
        stop = min(cols, start + block)
        slab = weights[:, start:stop]
        # Universal path needs no source scan. Certified path binds copied f64
        # slab and its bound to the SAME local snapshot, preventing stale proofs.
        a = inputs[:, start:stop].astype(np.float64)
        for first in range(0, rows, tile):
            last = min(rows, first + tile)
            b = slab[first:last].astype(np.float64)
            limit = (int(np.abs(b).sum(axis=1).max()) * proof["mask"]
                     if certified else 128 * (stop - start) * proof["mask"])
            if limit > EXACT:
                raise ValueError("source certificate exceeds FP64 exact range")
            c = (native.product(a, b, handle=handle) if native else a @ b.T)
            v = checked_integer(c, limit)
            # Python fixture uses bounded int64 addition. Production equivalent
            # is (v as i64 as u32), wrapping_add, then ring mask.
            out[:, first:last] = ((out[:, first:last].astype(np.int64) + v)
                                  & proof["mask"]).astype(np.uint32)
            del b, c, v
        del a
    if output_masks is not None:
        # Subtraction before narrowing, after exact float -> signed integer.
        out = ((out.astype(np.int64) - output_masks.astype(np.int64))
               & proof["mask"]).astype(np.uint32)
    return out


def cost(rows, cols, batch, bits, *, block=None, tile=1024):
    proof = bounds(bits)
    if (any(type(v) is not int or v <= 0 for v in (rows, cols, batch, tile))
            or batch > 4096 or tile > 1024 or max(rows * cols, batch * cols,
                                                batch * rows) > MAX_ELEMENTS):
        raise ValueError("invalid bounded cost geometry")
    k = min(cols, proof["universal_width"]) if block is None else block
    if type(k) is not int or not 1 <= k <= min(cols, proof["universal_width"]):
        raise ValueError("unsound cost width")
    n = min(rows, tile)
    blocks = (cols + k - 1) // k
    tiles = (rows + n - 1) // n
    digit_blocks = (cols + 1022) // 1023
    digit_tiles = (rows + 1023) // 1024
    digits = bits // 8
    return {"geometry": [rows, cols, batch], "bits": bits, "block": k,
            "dgemm_calls": blocks * tiles,
            "digit8_sgemm_calls": digits * digit_blocks * digit_tiles,
            "f64_macs": batch * rows * cols,
            "digit8_macs": digits * batch * rows * cols,
            "i8_weight_owner_bytes": rows * cols,
            "u32_input_and_output_mask_owner_bytes": 4 * batch * (cols + rows),
            "persistent_f64_weight_bytes": 0,
            "f64_weight_conversion_elements": rows * cols,
            "f64_weight_conversion_read_write_bytes": 9 * rows * cols,
            "f64_input_conversion_elements": batch * cols,
            "f64_input_conversion_read_write_bytes": 12 * batch * cols,
            "f64_partial_reconstruction_elements": blocks * batch * rows,
            "candidate_rust_workspace_bytes": 4 * batch * rows + 8 * (batch * k + n * k + batch * n),
            "fixture_workspace_bound_bytes": max(32 * batch * rows,
                4 * batch * rows + 8 * (batch + n) * k + 48 * batch * n),
            "digit8_workspace_bytes": 4 * (batch * rows + batch * min(cols, 1023)
                + min(rows, 1024) * min(cols, 1023) + batch * min(rows, 1024)),
            "digit8_weight_conversion_read_write_bytes": 5 * rows * cols,
            "digit8_input_conversion_read_write_bytes": 8 * digits * batch * cols * digit_tiles,
            "digit8_partial_reconstruction_elements": digits * digit_blocks * batch * rows,
            "binding_input_serialization_and_decode_bytes": 8 * batch * cols,
            "binding_output_serialize_and_python_bytes": 8 * batch * rows,
            "weight_setup_serialization_and_native_copy_bytes": 2 * rows * cols,
            "correction_u32_owner_bytes": 4 * batch * rows,
            "correction_encoded_body_bytes": (bits // 8) * batch * rows,
            "blas_scratch_and_thread_stacks_bytes": None,
            "existing_digit8_admission_allowance_bytes": 224 << 20,
            "measured_cpu_s": None, "whole_client_tier_qualified": False}


def synthetic():
    rng = np.random.default_rng(5005)
    cases = 0
    native = platform.system() == "Darwin" and platform.machine() == "arm64"
    for bits in (16, 24, 32):
        mask = bounds(bits)["mask"]
        for cols in (1, 7, 1023, 1024, 16384, 16385):
            w = rng.integers(-128, 128, (3, cols), dtype=np.int8)
            w[0] = -128
            x = rng.integers(0, mask + 1, (2, cols), dtype=np.uint32)
            x[0] = mask
            s = rng.integers(0, mask + 1, (2, 3), dtype=np.uint32)
            expected = integer_oracle(w, x, bits, s)
            for backend in (("numpy", "accelerate") if native else ("numpy",)):
                np.testing.assert_array_equal(project(w, x, bits, s, backend=backend), expected)
                cases += 1
    paths = [Path(__file__).resolve(), FIXTURE, ROOT / "tests/test_exact_f64_corrections_probe.py",
             ROOT / "crates/pllm-core/src/digit_gemm.rs", ROOT / "crates/pllm-core/src/kernels.rs",
             ROOT / "crates/pllm-core/src/codec.rs", ROOT / "crates/pllm-python/src/lib.rs",
             ROOT / "python/pllm/runtime/correction_batch.py"]
    controls = installed_controls()
    return {"scope": "synthetic bounded parity only; no performance/tier claim",
            "seed": 5005, "backend_case_count": cases, "accelerate_observed": native,
            "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "proofs": [bounds(b) for b in (16, 24, 32)],
            "installed_controls": controls,
            "cost_examples": [cost(256, 4096, b, 32) for b in (1, 32)],
            "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "gates": ["vendor reduction algorithm certification", "serial admitted native timing",
                      "checkpoint source bounds", "complete-client exactness/RSS/traffic"]}


def installed_controls():
    """Installed SDK only: no checkout imports, builds, or reference fallback."""
    import pllm
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime._native_support import capabilities
    location = str(Path(pllm.__file__).resolve())
    if "target/frontier-sdk-env" not in location:
        raise RuntimeError("installed frontier SDK required; checkout shadow detected")
    engine = MaskedGEMM(threads=1, build=False)
    if not engine.native:
        raise RuntimeError("installed native control unavailable")
    caps = capabilities()
    checked = 0
    for bits in (16, 24, 32):
        mask = bounds(bits)["mask"]
        w = np.resize(np.array([-128, 127, 0, -1, 1], np.int8), (3, 1025))
        x = np.resize(np.array([0, mask, 128, 255, 32768], np.uint32), (2, 1025))
        s = np.resize(np.array([mask, 0, 1], np.uint32), (2, 3))
        expected = integer_oracle(w, x, bits, s)
        matrix = engine.compile(w)
        for method in (["wrap32", "digit8"] if caps.get("digit8_cpu") else ["wrap32"]):
            result = matrix.digit8(x, bits) if method == "digit8" else matrix.wrap32(x)
            result = np.subtract(result, s, dtype=np.uint32) & mask
            np.testing.assert_array_equal(result, expected)
            checked += 1
    path = Path(engine.library_path)
    return {"installed_package": location, "native_library": str(path),
            "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "digit8_available": bool(caps.get("digit8_cpu")), "parity_cases": checked}


def timing_worker(mode, rows, cols, batch, bits, repeats):
    """Small opt-in worker. Setup and all boundary copies are charged separately."""
    if max(rows, cols, batch) <= 0 or rows > 1024 or cols > 16384 or batch > 64:
        raise ValueError("timing geometry cap exceeded")
    if not 1 <= repeats <= 5 or repeats * rows * cols * batch > 100_000_000:
        raise ValueError("timing work cap exceeded")
    rng = np.random.default_rng(5005)
    w = rng.integers(-128, 128, (rows, cols), dtype=np.int8)
    x = rng.integers(0, 1 << bits, (batch, cols), dtype=np.uint32)
    s = rng.integers(0, 1 << bits, (batch, rows), dtype=np.uint32)
    if batch * w.size > MAX_ORACLE_MACS:
        raise ValueError("timing fixture must remain independently oracle-checkable")
    expected = integer_oracle(w, x, bits, s)
    t0, c0 = time.perf_counter(), time.process_time()
    if mode in ("wrap32", "digit8"):
        from pllm.runtime.native import MaskedGEMM
        import pllm
        if "target/frontier-sdk-env" not in str(Path(pllm.__file__).resolve()):
            raise RuntimeError("installed frontier SDK required; checkout shadow detected")
        engine = MaskedGEMM(threads=1, build=False)
        if not engine.native:
            raise RuntimeError("installed native kernel required; no fallback/build")
        matrix = engine.compile(w)
        def run():
            result = matrix.digit8(x, bits) if mode == "digit8" else matrix.wrap32(x)
            return np.subtract(result, s, dtype=np.uint32) & bounds(bits)["mask"]
    else:
        snapshot = np.frombuffer(w.tobytes(), np.int8).copy().reshape(w.shape)
        handle = accelerate_module().load() if mode == "accelerate" else None
        def run():
            # Transfer-inclusive emulation of immutable-byte boundary + decode.
            decoded = np.frombuffer(x.tobytes(), np.uint32).copy().reshape(x.shape)
            result = project(snapshot, decoded, bits, s, backend=mode, handle=handle)
            return np.frombuffer(result.tobytes(), np.uint32).copy().reshape(result.shape)
    setup = {"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0}
    samples = []
    for _ in range(repeats):
        t0, c0 = time.perf_counter(), time.process_time()
        result = run()
        samples.append({"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0})
        np.testing.assert_array_equal(result, expected)
    return {"mode": mode, "setup": setup, "transfer_inclusive": samples,
            "maxrss_bytes_darwin": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cost": cost(rows, cols, batch, bits), "parity": True,
            "threads_requested": 1, "vendor_threads_verified": False,
            "scope": "standalone fixture, not complete-client timing"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing", action="store_true", help="parent admission required; serial fresh workers")
    parser.add_argument("--worker", choices=("numpy", "accelerate", "wrap32", "digit8"))
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--bits", type=int, choices=(16, 24, 32), default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        report = timing_worker(args.worker, args.rows, args.cols, args.batch, args.bits, args.repeats)
    elif args.timing:
        env = dict(os.environ, VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        report = {"scope": "serial fresh-process transfer-inclusive fixture only", "workers": []}
        for mode in ("numpy", "accelerate", "wrap32", "digit8"):
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                       "--rows", str(args.rows), "--cols", str(args.cols), "--batch", str(args.batch),
                       "--bits", str(args.bits), "--repeats", str(args.repeats)]
            run = subprocess.run(command, env=env, cwd=sys.prefix, capture_output=True, text=True, timeout=90)
            report["workers"].append(json.loads(run.stdout) if run.returncode == 0 else
                                     {"mode": mode, "gate": "failed", "stderr": run.stderr})
    else:
        report = synthetic()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
