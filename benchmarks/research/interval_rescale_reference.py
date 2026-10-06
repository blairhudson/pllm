"""Measure public-interval key sharing against independent compact comparisons."""
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

from benchmarks.research import compact_rescale_reference as compact
from benchmarks.research.shared_rescale_reference import ROOT, measure_child, oracle_digest, projections


CONFIGURATION = {
    **compact.CONFIGURATION,
    "papers": {
        **compact.CONFIGURATION["papers"],
        "fss-mixed": {
            **compact.CONFIGURATION["papers"]["fss-mixed"],
            "sections": ["Section 3 Figure 1", "Section 4 Lemma 1 and Figure 3", "Appendix G.5 Figure 14"],
        },
    },
    "backends": ["compact_dcf", "interval_dcf"],
    "backend": "independent compact thresholds versus one universal DCF per input width per lane/phase",
    "interval_contract": "public-boundary shift corrections fold into secret constant shares; keys never cross lanes, phases or issuances",
    "source_deviations": [
        *compact.CONFIGURATION["source_deviations"],
        "Section 4 interval algebra is lifted to Z/(2^32), including narrower input rings",
        "linear combinations of interval corrections fold into the existing three constant shares per lane/phase",
        "bit-width sharing reduces issuance and key storage, not the number of online comparison evaluations",
    ],
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
        raise RuntimeError("Interval FSS probe needs 1 GiB headroom beyond the host reserve")
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
    pair = 0
    guard.start()
    try:
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
                        child = subprocess.Popen([sys.executable, "-m", "benchmarks.research.interval_rescale_reference", "--child", str(binary), *arguments],
                                                 cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
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
                        if case["output_sha256"] != oracle_digest(bits, shift, rounding, CONFIGURATION["lanes"]) or case["executable_sdk"] is not False:
                            raise ValueError("Native output differs from independent Python oracle or changed scope")
                        keys = ("bits", "shift", "rounding", "layout", "lanes", "samples", "backend")
                        if [case[k] for k in keys] != [bits, shift, rounding, layout, CONFIGURATION["lanes"], CONFIGURATION["samples"], backend]:
                            raise ValueError("Native case identity differs")
                        case["median_seconds"] = {phase: statistics.median(case[f"{phase}_ns"]) / 1e9 for phase in ("issuance", "online", "clear")}
                        cases.append(case)
    finally:
        guard.close()
    geometry = projections(json.loads(config_bytes), cases)
    for workload in geometry:
        for projected, measured in zip(workload["cases"], cases, strict=True):
            projected["backend"] = measured["backend"]
    return {
        "schema": "pllm.interval_rescale_reference.v1", "configuration": CONFIGURATION,
        "source_lock_digest": resolved.source_lock_digest, "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "cases": cases, "geometry_projections": geometry,
        "executable_sdk": False, "privacy_reviewed": False, "whole_decoder_numeric_parity": False,
        "memory_guard": {"host": asdict(host), "required_headroom_bytes": GiB, "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()},
        "source_sha256": {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in [
            "benchmarks/research/interval_rescale_reference.py", "benchmarks/research/compact_rescale_reference.py",
            "benchmarks/research/shared_rescale_reference.py", "crates/pllm-garble/src/interval_fss.rs",
            "crates/pllm-garble/src/compact_dcf.rs", "crates/pllm-garble/src/point_fss.rs",
            "crates/pllm-garble/src/shared_rescale_reference.rs", "crates/pllm-garble/examples/shared_rescale_probe.rs",
            "crates/pllm-garble/src/lib.rs", "crates/pllm-garble/Cargo.toml", "Cargo.toml", "Cargo.lock",
        ]},
        "limitations": [
            "bounded component reference; no full SIGMA/FuseFSS system, decoder or reviewed cryptography",
            "public-boundary queries share a key only within one lane/phase; no cross-request material reuse",
            "threshold shifts and constants remain secret-shared; public metadata depends only on descriptor and layout",
            "fresh native processes include dealer and both evaluators: CPU/RSS are not client or independent-provider costs",
            "phase timers are elapsed time; full process CPU includes issuance, frames, reconstruction and lifecycle checks",
            "all roots, corrections, mask and constant shares count; offline headers, distribution and allocator overhead are extra",
            "no WAN, authenticated transport, malicious-output verification, distributed dealer, checkpoint values, logits or KV",
            "Qwen geometry is a hypothetical helper placement, not admission of its W8A8 floating-point nonlinear execution",
            "projections omit attention, polynomial evaluation, other rescaling, input/output conversion, token feedback and full response work",
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
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(report["cases"]), "checked_outputs": sum(case["checked_outputs"] for case in report["cases"]),
                      "maximum_swap_growth_bytes": report["memory_guard"]["maximum_swap_growth_bytes"]}))


if __name__ == "__main__":
    main()
