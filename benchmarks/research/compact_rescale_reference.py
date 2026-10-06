"""Replay compact-DCF versus prefix-DPF signed helpers; no SDK decoder ranking."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys

from benchmarks.research.shared_rescale_reference import (
    CONFIGURATION as HELPER_CONFIGURATION,
    measure_child,
    oracle_digest,
    projections,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION = {
    **HELPER_CONFIGURATION,
    "papers": {
        **HELPER_CONFIGURATION["papers"],
        "fss-mixed": {
            "url": "https://eprint.iacr.org/2020/1392",
            "sha256": "dc94c63a77a5cb52cfd12b186a327d045de4a1fc662060c1de6412b334f4fcf4",
            "sections": ["3", "Figure 1", "Theorem 2"],
        },
    },
    "backends": ["prefix_dpf", "compact_dcf"],
    "backend": "matched quadratic prefix-DPF control and linear-size Figure 1 DCF over Z/(2^32)",
    "source_deviations": [
        *HELPER_CONFIGURATION["source_deviations"],
        "compact DCF has separate byte controls (20 + 22n bytes/party), not the paper's bit-packed key format",
    ],
    "execution_order": "adjacent backend pairs, alternating which backend runs first; fresh process per case",
}


def run() -> dict:
    from pllm import Model
    from pllm.model_loader import resolve_model
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory

    resolved = resolve_model(Model(**CONFIGURATION["model"]))
    config_bytes = (resolved.path / "config.json").read_bytes()
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pllm-garble", "--example", "shared_rescale_probe"], cwd=ROOT, check=True)
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))
    binary = Path(metadata["target_directory"]) / "release/examples/shared_rescale_probe"
    host = host_memory()
    if host.available - host.reserve < GiB:
        raise RuntimeError("compact FSS probe needs 1 GiB headroom beyond the host reserve")
    active = []

    def abort(_reason):
        for child in tuple(active):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    cases = []
    guard.start()
    try:
        pair = 0
        for bits, shift in CONFIGURATION["shapes"]:
            for rounding in CONFIGURATION["rounding"]:
                for layout in CONFIGURATION["layouts"]:
                    backends = CONFIGURATION["backends"][::1 if pair % 2 == 0 else -1]
                    pair += 1
                    for backend in backends:
                        guard.check()
                        if guard.error:
                            raise RuntimeError(guard.error)
                        arguments = [str(bits), str(shift), rounding, layout, str(CONFIGURATION["lanes"]), str(CONFIGURATION["samples"]), backend]
                        child = subprocess.Popen(
                            [sys.executable, "-m", "benchmarks.research.compact_rescale_reference", "--child", str(binary), *arguments],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
                        )
                        active.append(child)
                        try:
                            stdout, stderr = child.communicate(timeout=180)
                        finally:
                            if child.poll() is None:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                            active.remove(child)
                        guard.check()
                        if child.returncode or guard.error:
                            raise RuntimeError(guard.error or stderr)
                        case = json.loads(stdout)
                        expected = oracle_digest(bits, shift, rounding, CONFIGURATION["lanes"])
                        if case["output_sha256"] != expected or case["executable_sdk"] is not False:
                            raise ValueError("native output disagrees with independent Python oracle or changed scope")
                        if [case[k] for k in ("bits", "shift", "rounding", "layout", "lanes", "samples", "backend")] != [bits, shift, rounding, layout, CONFIGURATION["lanes"], CONFIGURATION["samples"], backend]:
                            raise ValueError("native backend or workload acknowledgement changed")
                        case["median_seconds"] = {phase: statistics.median(case[f"{phase}_ns"]) / 1e9 for phase in ("issuance", "online", "clear")}
                        cases.append(case)
    finally:
        guard.close()

    geometry = projections(json.loads(config_bytes), cases)
    for workload in geometry:
        for projected, measured in zip(workload["cases"], cases, strict=True):
            projected["backend"] = measured["backend"]
    return {
        "schema": "pllm.compact_rescale_reference.v1", "configuration": CONFIGURATION,
        "source_lock_digest": resolved.source_lock_digest, "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "cases": cases, "geometry_projections": geometry,
        "executable_sdk": False, "privacy_reviewed": False, "whole_decoder_numeric_parity": False,
        "memory_guard": {"host": asdict(host), "required_headroom_bytes": GiB, "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()},
        "source_sha256": {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in [
            "benchmarks/research/compact_rescale_reference.py", "benchmarks/research/shared_rescale_reference.py",
            "crates/pllm-garble/src/compact_dcf.rs", "crates/pllm-garble/src/point_fss.rs",
            "crates/pllm-garble/src/shared_rescale_reference.rs", "crates/pllm-garble/examples/shared_rescale_probe.rs",
            "crates/pllm-garble/src/lib.rs", "Cargo.toml", "Cargo.lock",
        ]},
        "limitations": [
            "bounded published-DCF reimplementation, not a complete SIGMA/FuseFSS system or security review",
            "phase timers are local elapsed time; process CPU includes issuance, evaluation, clear oracle and lifecycle checks",
            "source resolution and native build are outside timings; no model values, logits or KV are evaluated",
            "key payload counts roots, corrections, mask and constant shares; offline metadata/transport and allocator overhead are extra",
            "process peak RSS contains trusted local dealer plus both parties; not independent-role or client memory",
            "no GPU, WAN, distributed dealer, authenticated transport, malicious-output verification or whole-response measurement",
            "Qwen projections are hypothetical helper counts; there is no Qwen W8A8 numeric mapping or admitted tensor schedule",
        ],
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(measure_child(sys.argv[2:])))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new reproduction path")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(report["cases"]), "checked_outputs": sum(c["checked_outputs"] for c in report["cases"]),
                      "maximum_swap_growth_bytes": report["memory_guard"]["maximum_swap_growth_bytes"]}))


if __name__ == "__main__":
    main()
