"""Fresh-process matched complete mixed-mode costs and frozen-profile Qwen quality."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("curl-0", "curl-4", "logrow", "compact-4-2", "compact-8-3", "compact-16-3")
# Frozen before measurements; separate from earlier fitting and quality cohorts.
PUBLIC_HELDOUT = (
    "Why does a bicycle need spokes between its hub and rim?",
    "Explain why shadows change length during the day.",
    "Six boxes each contain nine pencils. How many pencils are there?",
    "How do tree roots help soil stay in place during rain?",
    "Why does a glass jar sometimes open more easily after warming its lid?",
    "What allows a compass needle to point toward north?",
    "Describe why a parachute slows a falling object.",
    "How can a thermometer measure temperature without a battery?",
    "Twenty-eight biscuits are shared equally among seven children. How many each?",
    "Why is it useful to repeat a scientific measurement?",
    "What happens to the pitch when the vibrating part of a guitar string gets shorter?",
    "How can a sundial tell the approximate time on a sunny day?",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def component(name, backend):
    from pllm.nonlinear import CompactGatedMultiplyQ12, CurlWaveletGatedMultiplyQ9, LogRowGatedMultiplyQ9
    extra = {"backend": "mixed"} if backend == "mixed" else {}
    if name.startswith("curl-"):
        return CurlWaveletGatedMultiplyQ9(levels=int(name.split("-")[1]), **extra)
    if name.startswith("compact-"):
        _, p, d = name.split("-")
        return CompactGatedMultiplyQ12(max_pieces=int(p), degree=int(d), **extra)
    return LogRowGatedMultiplyQ9(**extra)


def costs():
    from .logrow_product import project
    reports, pairs = [], []
    for name in PROFILES:
        for lanes in (1, 16, 64):
            matched = []
            for backend in ("original", "mixed"):
                run = subprocess.run([sys.executable, "-m", "benchmarks.research.mixed_gated",
                    "--worker", name, backend, str(lanes)], check=True, capture_output=True,
                    text=True, timeout=900, env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                matched.append(json.loads(run.stdout))
            a, b = matched
            if a["numeric_contract_digest"] != b["numeric_contract_digest"] or a["quality"] != b["quality"]:
                raise ValueError("matched numeric controls differ")
            reports.extend(matched)
            pairs.append({"profile": name, "lanes": lanes,
                "original_configuration_digest": a["configuration_digest"], "mixed_configuration_digest": b["configuration_digest"],
                "combined_material_ratio": sum(b["resources"]["raw_key_payload_bytes_per_party"]) / sum(a["resources"]["raw_key_payload_bytes_per_party"]),
                "local_online_ratio": b["median_online_seconds"] / a["median_online_seconds"],
                "issuance_ratio": b["median_issuance_seconds"] / a["median_issuance_seconds"]})
    return {"schema": "pllm.mixed_gated_cost.v1", "reports": reports, "pairs": pairs,
        "fresh_processes": len(reports), "checked_outputs": sum(r["quality"]["checked_outputs"] for r in reports),
        "projections": [{"configuration_digest": r["configuration_digest"], **project(r)}
            for r in reports if r["configuration"]["lanes"] == 64],
        "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in
            ("benchmarks/research/mixed_gated.py", "benchmarks/research/logrow_product.py")},
        "scope": "matched numeric profiles; differing role ownership; hypothetical MLP-only geometry, no decoder transport"}


def quality():
    from pllm import _native
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from scripts.decoder_probe_support import DecoderFixture
    from .compact_product import NativeMapping as Compact
    from .wavelet_gated import NativeMapping as Curl
    from .gated_numeric_screen import screen
    from .gated_trajectories import generation
    host = host_memory()
    if host.available - host.reserve < 4 * GiB:
        raise RuntimeError("quality screen needs 4 GiB beyond host reserve")
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, lambda _: os.kill(os.getpid(), signal.SIGTERM))

    def check():
        guard.check()
        if guard.error:
            raise RuntimeError(guard.error)

    guard.start()
    try:
        fixture = DecoderFixture(inputs=96, outputs=8)
        bridges = {"curl-0-and-logrow": Curl(0), "curl-4": Curl(4),
                   **{f"compact-{p}-{d}": Compact(p, d) for p, d in ((4, 2), (8, 3), (16, 3))}}
        teacher = screen(fixture, bridges, PUBLIC_HELDOUT, check)
        greedy = generation(fixture, bridges, PUBLIC_HELDOUT, check)
        if (teacher["summary"]["control"]["bit_exact_positions"] != 48
                or teacher["summary"]["control"]["exact_kv_positions"] != 48
                or greedy["summary"]["control"]["exact_trajectories"] != 12):
            raise ValueError("fresh-runtime control is not exact")
        check()
    finally:
        guard.close()
    paths = ("benchmarks/research/mixed_gated.py", "benchmarks/research/compact_product.py", "benchmarks/research/wavelet_gated.py",
             "benchmarks/research/gated_numeric_screen.py", "benchmarks/research/gated_trajectories.py", "scripts/decoder_probe_support.py")
    return {"schema": "pllm.mixed_gated_quality.v1", "source": fixture.lock(),
        "numeric_contract_digests": {name: b.digest for name, b in bridges.items()},
        "independent_integer_checked_elements": {name: b.checked for name, b in bridges.items()},
        "installed_native_sha256": sha(Path(_native.__file__).read_bytes()),
        "public_heldout_sha256": sha(json.dumps(PUBLIC_HELDOUT).encode()),
        "teacher_forced": teacher, "greedy_generation": greedy,
        "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in paths},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "memory_guard": {"host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "scope": "fresh clear Qwen numeric diagnostic; protected operator parity is a separate cohort",
        "protected_checkpoint_execution": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cost", "quality"), default="cost")
    parser.add_argument("--worker", nargs=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        from pllm.metrics import benchmark_operator
        name, backend, lanes = args.worker
        if name not in PROFILES or backend not in ("original", "mixed"):
            parser.error("unsupported worker configuration")
        print(benchmark_operator(component(name, backend), lanes=int(lanes), samples=3).to_json())
        return
    if args.output is None or args.output.exists():
        parser.error("choose a fresh evidence output path")
    result = costs() if args.mode == "cost" else quality()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["pairs"] if args.mode == "cost" else result["teacher_forced"]["summary"]))


if __name__ == "__main__":
    main()
