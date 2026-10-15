#!/usr/bin/env python3
"""Serial v3 API probe: historical preallocated and allocating-output pairs; no SDK."""
import argparse
import datetime
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/release/examples/boundary_head_dotprod"
SCHEMA = "boundary-head-dotprod-v3"
SOURCES = ("crates/pllm-core/examples/boundary_head_dotprod.rs",
           "crates/pllm-core/src/kernels.rs", "scripts/probe_boundary_head_dotprod.py",
           "tests/test_boundary_head_dotprod_probe.py", "docs/research/boundary-head-dotprod-probe.md",
           "Cargo.lock", "Cargo.toml", "crates/pllm-core/Cargo.toml")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_binary():
    env = os.environ.copy()
    env["RUSTC"] = shutil.which(env.get("RUSTC", "rustc")) or "rustc"
    command = ["cargo", "build", "--locked", "-p", "pllm-core", "--release", "--example", "boundary_head_dotprod"]
    provenance = {
        "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command, "cwd": str(ROOT), "profile": "release",
        "rustc_verbose": subprocess.check_output([env["RUSTC"], "-vV"], text=True),
        "cargo_verbose": subprocess.check_output(["cargo", "-vV"], text=True),
        "rustc_path": env["RUSTC"],
        "build_environment": {k: v for k, v in env.items() if k in (
            "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR",
            "RUSTUP_TOOLCHAIN", "MACOSX_DEPLOYMENT_TARGET", "SDKROOT", "CC", "CFLAGS")
            or k.startswith(("CARGO_PROFILE_RELEASE_", "CARGO_TARGET_AARCH64_APPLE_DARWIN_"))},
        "cargo_config_sha256": {str(p): sha(p) for p in
            (ROOT / ".cargo/config", ROOT / ".cargo/config.toml",
             Path.home() / ".cargo/config", Path.home() / ".cargo/config.toml") if p.is_file()},
    }
    env["BOUNDARY_DOTPROD_BUILD_PROVENANCE"] = json.dumps(provenance, sort_keys=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=97)
    p.add_argument("--cols", type=int, default=896)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234567)
    p.add_argument("--processes", type=int, default=2)
    p.add_argument("--order-offset", type=int, choices=(0, 1), default=0,
                   help="0 starts candidate-first; 1 starts baseline-first; alternate each repeat and process")
    p.add_argument("--process-index", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--mode", choices=("auto", "scalar"), default="auto")
    p.add_argument("--emit", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if not 1 <= a.processes <= 20:
        p.error("processes must be in 1..20")
    if a.build:
        build_binary()
    if a.worker:
        # Each measurement gets its own Python parent, so RUSAGE_CHILDREN's
        # high-water mark cannot inherit an earlier measurement or cargo build.
        source_hashes = {name: sha(ROOT / name) for name in SOURCES}
        binary_hash = sha(BINARY)
        command = [str(BINARY), str(a.rows), str(a.cols), str(a.repeats),
                   str(a.seed), a.mode, "emit" if a.emit else "quiet", str(a.order_offset)]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode:
            sys.exit(proc.stderr)
        answer = json.loads(proc.stdout)
        if answer.get("schema") != SCHEMA:
            sys.exit("unexpected native report schema; rebuild with --build (do not relabel archives)")
        if binary_hash != sha(BINARY) or source_hashes != {name: sha(ROOT / name) for name in SOURCES}:
            sys.exit("binary/source changed during measurement; discard and rerun")
        for native_key, name in (("compiled_source_sha256", SOURCES[0]), ("compiled_kernels_sha256", SOURCES[1])):
            if answer[native_key] != source_hashes[name]:
                sys.exit("stale binary source hash; rebuild with --build")
        if not answer["build_provenance"]:
            sys.exit("missing embedded build provenance; rebuild with --build")
        answer["build_provenance"] = json.loads(answer["build_provenance"])
        answer["provenance"] = {"binary_sha256": binary_hash, "source_sha256": source_hashes,
                                "binary_path": str(BINARY), "native_command": command}
        answer["process_schedule"] = {"index": a.process_index, "processes": a.processes,
                                      "effective_order_offset": a.order_offset,
                                      "paired_start_balance": a.processes % 2 == 0,
                                      "paired_cohort_start_balance": a.processes % 2 == 0}
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        answer["fresh_process"] = {
            "peak_rss_bytes": int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cpu_brand": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                         if sys.platform == "darwin" else platform.processor(),
            "native_runtime_features": {"dotprod": answer["dotprod_available"],
                                        "baseline_simd": answer["baseline_simd"]},
        }
        print(json.dumps(answer))
        return
    for index in range(a.processes):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--rows", str(a.rows), "--cols", str(a.cols),
                   "--repeats", str(a.repeats), "--seed", str(a.seed), "--mode", a.mode,
                   "--order-offset", str((a.order_offset + index) % 2),
                   "--process-index", str(index), "--processes", str(a.processes)]
        if a.emit:
            command.append("--emit")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
