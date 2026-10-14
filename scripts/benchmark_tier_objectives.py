"""Parent-only, serial small-operator gates for the ten-objective screen.

Uses installed SDK controls and the existing frontier admission/watchdog. This
does not run a decoder, qualify client tiers, or measure network overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "target/limb_stationary_corrections/fixture.dylib"
MEMORY_BYTES = 384 << 20
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")
STEMS = {
    "mask": "mask_runahead", "f64": "exact_f64_corrections",
    "limb": "limb_stationary_corrections", "limb-tail": "limb_stationary_corrections",
}


def cases(selection):
    """Fixed geometry bounds are checked before any fixture allocation."""
    if selection == "mask":
        return [dict(kind="mask", rows=rows, width=width, output=output)
                for rows, width, output in ((4, 2048, 4096), (32, 896, 9728))]
    if selection == "f64":
        return [dict(kind="f64", rows=rows, cols=cols, batch=batch, bits=32, mode=mode)
                for rows, cols, batch in ((16, 2048, 32), (256, 4864, 1))
                for mode in ("wrap32", "digit8", "accelerate")]
    if selection in ("limb", "limb-tail"):
        shapes = ((32, 4096, 1024, 32), (1, 4096, 1024, 32)) if selection == "limb" else (
            (2, rows, cols, bits) for bits in (16, 24, 32)
            for rows, cols in ((1025, 3), (3, 1024)))
        return [dict(kind=selection, batch=b, rows=r, cols=c, bits=bits)
                for b, r, c, bits in shapes]
    raise ValueError("unknown fixed screen")


def identity(selection):
    import pllm
    from pllm.metrics import BuildRef
    if not Path(pllm.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("installed SDK required; checkout shadow rejected")
    stem = STEMS[selection]
    paths = [Path(__file__), ROOT / "scripts/frontier_probe_support.py",
             ROOT / f"scripts/probe_{stem}.py", ROOT / f"tests/test_{stem}_probe.py"]
    if selection.startswith("limb"):
        paths += [ROOT / "scripts/research_native/limb_stationary_corrections/fixture.c", LIBRARY]
    elif selection == "f64":
        paths += [ROOT / "scripts/research_native/exact_f64_corrections/accelerate.py"]
    return {"build": BuildRef.capture().to_spec(), "files": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def require_identity(actual, expected):
    if actual != expected:
        raise RuntimeError("research or installed-build identity changed")


def load_probe(selection):
    stem = STEMS[selection]
    spec = importlib.util.spec_from_file_location(stem, ROOT / f"scripts/probe_{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module


def full_limb_oracle(probe, case, measurements):
    """Large C fixtures also compare every output against installed exact Rust."""
    import numpy as np
    from pllm.runtime.native import MaskedGEMM
    cpu = time.process_time()
    engine = MaskedGEMM(threads=1, build=False)
    if not engine.native:
        raise RuntimeError("native integer oracle unavailable")
    weights, _ = probe.synthetic_case(case["rows"], case["cols"], case["batch"], case["bits"])
    matrix = engine.compile(weights)
    checked = 0
    for sample in measurements:
        _, inputs = probe.synthetic_case(1, case["cols"], case["batch"], case["bits"])
        inputs ^= np.uint32(sample["rep"])
        expected = matrix.wrap32(inputs) & ((1 << case["bits"]) - 1)
        if hashlib.sha256(expected.tobytes()).hexdigest() != sample["output_sha256"]:
            raise RuntimeError("standalone C differs from full installed integer oracle")
        checked += expected.size
    return {"checked_values": checked, "process_cpu_seconds": time.process_time() - cpu,
            "scope": "additional complete-output audit, outside native operation timing"}


def worker(payload):
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    case = payload["case"]
    selection = case["kind"]
    if case not in cases(selection):
        raise ValueError("worker geometry differs from fixed admitted screen")
    require_identity(identity(selection), payload["identity"])
    probe = load_probe(selection)
    if selection == "mask":
        result = probe.bounded_timing(case["width"], case["output"], case["rows"], repeats=3)
    elif selection == "f64":
        result = probe.timing_worker(case["mode"], case["rows"], case["cols"],
                                     case["batch"], case["bits"], repeats=3)
    else:
        result = probe.benchmark(SimpleNamespace(**case, native_library=str(LIBRARY),
                                                repeats=1 if selection == "limb-tail" else 3))
        result["full_installed_integer_audit"] = full_limb_oracle(probe, case, result["measurements"])
    frozen = identity(selection)
    require_identity(frozen, payload["identity"])
    return {"case": case, "identity": frozen, "result": result,
            "complete_worker_cpu_seconds": time.process_time(),
            "complete_worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
            "threads_requested": 1, "vendor_threads_verified": False}


def measure(selection, replicates, *, runner=None):
    if type(replicates) is not int or not 1 <= replicates <= 3:
        raise ValueError("replicates must be 1..3")
    plan = cases(selection)
    frozen = identity(selection)
    if runner is None:
        from frontier_probe_support import fresh_workers
        runner = fresh_workers
    records = []
    with runner(Path(__file__).resolve(), memory_bytes=MEMORY_BYTES, timeout=180) as (run, guard, host):
        for replicate in range(replicates):
            for case in (plan if replicate % 2 == 0 else list(reversed(plan))):
                record = run({"case": case, "identity": frozen})
                require_identity(record["identity"], frozen)
                if record["case"] != case:
                    raise RuntimeError("worker substituted admitted case")
                records.append({"replicate": replicate, **record})
        guard.check()
        if guard.error:
            raise RuntimeError(guard.error)
        watchdog = {"baseline_swap_used_bytes": host.swap_used,
                    "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                    "minimum_available_bytes": guard.minimum_available_bytes, "error": guard.error}
        require_identity(identity(selection), frozen)
    return {"schema": "pllm.tier_objectives_operator_screen.v1", "selection": selection,
            "replicates": replicates, "memory_admission_bytes": MEMORY_BYTES,
            "identity": frozen, "records": records, "watchdog": watchdog,
            "scope": "isolated operator fixtures; peak RSS includes setup and audits",
            "serving_gain": None, "whole_client_tier_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=tuple(STEMS))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        result = worker(json.load(sys.stdin))
    else:
        if args.case is None:
            parser.error("--case required")
        result = measure(args.case, args.replicates)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
