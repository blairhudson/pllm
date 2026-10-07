"""Fresh installed-SDK RAA costs; public synthetic data, never decoder execution."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "benchmarks/research/coded_delegation.py", "benchmarks/research/paper_baseline.py",
    "benchmarks/research/common.py", "python/pllm/runtime/coded_resources.py",
    "crates/pllm-core/src/coded_delegation_reference.rs", "crates/pllm-core/src/coded_linear.rs",
    "crates/pllm-core/src/kernels.rs", "crates/pllm-python/src/coded_delegation.rs",
    "python/pllm/metrics/coded_delegation.py", "python/pllm/metrics/operator_benchmarks.py",
    "python/pllm/protocols/__init__.py", "Cargo.lock",
)


def source_digests():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}


def worker(rows, columns, queries):
    from pllm.metrics import benchmark_operator
    from pllm.protocols import MaverickDelegatedLinear

    return benchmark_operator(MaverickDelegatedLinear(), lanes=queries, samples=1,
                              geometry={"rows": rows, "columns": columns}).to_dict()


def run(samples=3):
    from pllm import _native, lower_model
    from pllm.model_loader import resolve_model
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from pllm.runtime.coded_resources import coded_delegation_storage_lower_bound
    from benchmarks.research.paper_baseline import experiment

    if type(samples) is not int or not 1 <= samples <= 8:
        raise ValueError("samples must be between one and eight")
    sources = source_digests()
    control = experiment()
    resolved = resolve_model(control.pipeline.model)
    config = (resolved.path / "config.json").read_bytes()
    plan = lower_model(json.loads(config), batch=1, max_input_tokens=64, max_new_tokens=8)
    storage = coded_delegation_storage_lower_bound(plan, control.pipeline,
        resident_budget_bytes=2 << 30, cached_budget_bytes=6 << 30)
    shapes = Counter((s["output_width"], s["input_width"]) for s in storage["stage_dimensions"])
    host = host_memory()
    if host.available - host.reserve < GiB:
        raise RuntimeError("RAA costs require 1 GiB headroom beyond the host reserve")
    active = []
    def abort(_reason):
        for child in tuple(active):
            if child.poll() is None:
                child.terminate()
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    cases = []
    guard.start()
    try:
        # The tiny case is an SDK control, not a compiler-derived stage.
        for rows, columns in [(64, 64), *sorted(shapes)]:
            queries = 8
            projection = json.loads(_native.coded_delegation_projection(rows, columns, queries))
            reports = []
            if projection["admitted"]:
                for _ in range(samples):
                    guard.check()
                    if guard.error:
                        raise RuntimeError(guard.error)
                    child = subprocess.Popen([sys.executable, "-m", "benchmarks.research.coded_delegation",
                        "--worker", str(rows), str(columns), str(queries)], cwd=ROOT,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                    active.append(child)
                    try:
                        stdout, stderr = child.communicate(timeout=120)
                    finally:
                        if child.poll() is None:
                            child.kill()
                            child.wait()
                        active.remove(child)
                    guard.check()
                    if child.returncode or guard.error:
                        raise RuntimeError(guard.error or stderr)
                    reports.append(json.loads(stdout))
                if any(r["resources"] != reports[0]["resources"] for r in reports):
                    raise ValueError("fresh samples changed public resource counts")
            cases.append({"rows": rows, "columns": columns, "queries": queries,
                "compiled_stage_count": shapes[(rows, columns)], "projection": projection, "reports": reports,
                "median_seconds": {key: statistics.median(r["samples"][0][key] for r in reports)
                    for key in ("public_setup_seconds", "issuance_seconds", "evaluation_seconds",
                                "verification_seconds", "online_seconds", "clear_seconds")} if reports else None})
    finally:
        guard.close()
    if sources != source_digests():
        raise ValueError("implementation changed during measurement")
    return {"schema": "pllm.coded_delegation_cost.v1", "source_sha256": sources,
        "model": control.pipeline.model.to_spec(), "source_lock_digest": resolved.source_lock_digest,
        "model_config_sha256": hashlib.sha256(config).hexdigest(), "storage_lower_bound": storage,
        "cases": cases, "fresh_processes": sum(len(c["reports"]) for c in cases), "samples_per_case": samples,
        "checked_outputs": sum(r["quality"]["checked_outputs"] for c in cases for r in c["reports"]),
        "admitted_reference_stages": sum(c["compiled_stage_count"] for c in cases if c["projection"]["admitted"]),
        "memory_guard": {"host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "scope": {"component": "RAA standard-mode masked matrix-vector reference with conditional coded checks",
            "inputs": "synthetic public signed-i8 weights and vectors; source-bound stage shapes only",
            "distance_certified": False, "privacy_certified": False,
            "source_distance_theorem_applies_to_babybear": False,
            "full_model_storage_admitted": False, "checkpoint_quality": None,
            "whole_model_executable": False, "provider_transport": False,
            "peak_memory_bytes": None, "full_wire_bytes": None}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--worker", nargs=3, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker)))
    else:
        if args.output is None or args.output.exists():
            parser.error("provide a new --output path")
        result = run(args.samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(f"{result['checked_outputs']} exact outputs in {result['fresh_processes']} fresh processes")
