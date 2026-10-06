"""Replay the scoped SIGMA/FuseFSS native helper pair, outside SDK rankings."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION = {
    "papers": {
        "sigma": {
            "url": "https://petsymposium.org/popets/2024/popets-2024-0107.php",
            "sha256": "f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf",
            "sections": ["2.4", "4.2.2", "4.2.3", "4.3"],
        },
        "fusefss": {
            "url": "https://arxiv.org/abs/2606.09551",
            "sha256": "6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0",
            "sections": ["3.5", "4.3", "4.5", "Appendix E"],
        },
    },
    "shapes": [[16, 7], [24, 12], [32, 12]],
    "rounding": ["floor", "ties_even"],
    "layouts": ["unfused", "fused"],
    "lanes": 64,
    "samples": 5,
    "backend": "arithmetic prefix-DPF, quadratic in comparison bit width; same backend for both layouts",
    "function": "signed rescale plus original-input nonnegative predicate",
    "source_deviations": [
        "SIGMA-inspired truncate/reduce and sign extension, not the paper's optimized DPF/B2A protocol",
        "FuseFSS-inspired arithmetic masked comparisons; no general operator compiler or coefficient lookup",
        "ties-to-even is an explicit PLLM extension; paper ARS uses floor",
        "128-bit AES-keyed counter expansion with separate seed/control bits; no early termination or GPU backend",
    ],
    "model": {"source": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775"},
    "workloads": [{"input_tokens": 39, "output_tokens": 8}, {"input_tokens": 39, "output_tokens": 32}],
    "projection": "hypothetical one helper per semantic SiLU element; no checkpoint values or Qwen numeric mapping",
}


def public_values(bits: int, shift: int, lanes: int) -> list[int]:
    """Public synthetic vector shared with the native probe, never mask material."""
    mask = (1 << bits) - 1
    half, sign = 1 << (shift - 1), 1 << (bits - 1)
    fixed = [0, mask, sign, sign - 1, half, half - 1, half + 1, -half, -half - 1, -half + 1, 3 * half, -3 * half]
    values = []
    for index in range(lanes):
        word = (index * 2_654_435_761) & 0xFFFFFFFF
        rotation = index % 31
        word = ((word << rotation) | (word >> (32 - rotation))) & 0xFFFFFFFF
        values.append((fixed[index] if index < len(fixed) else word) & mask)
    return values


def oracle_digest(bits: int, shift: int, rounding: str, lanes: int) -> str:
    output = bytearray()
    for code in public_values(bits, shift, lanes):
        signed = code if code < 1 << (bits - 1) else code - (1 << bits)
        # Python's exact rational integer division is independent of the FSS
        # implementation's masked carries and prefix functions.
        quotient, remainder = divmod(signed, 1 << shift)
        if rounding == "ties_even" and (2 * remainder > 1 << shift or (2 * remainder == 1 << shift and quotient % 2)):
            quotient += 1
        output.extend((quotient % (1 << bits)).to_bytes(4, "little"))
        output.extend(int(signed >= 0).to_bytes(4, "little"))
    return hashlib.sha256(output).hexdigest()


def measure_child(command: list[str]) -> dict:
    """Fresh helper has exactly one child: rusage peak belongs to this probe."""
    process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = json.loads(process.stdout)
    result["process_cpu_seconds"] = usage.ru_utime + usage.ru_stime
    result["process_peak_rss_bytes"] = int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    return result


def projections(config: dict, cases: list[dict]) -> list[dict]:
    from pllm import lower_model

    projections = []
    for workload in CONFIGURATION["workloads"]:
        plan = lower_model(config, batch=1, max_input_tokens=workload["input_tokens"], max_new_tokens=workload["output_tokens"])
        tensors = {phase: [op for op in plan.to_dict()[phase]["operations"] if op["operator"] == "silu"] for phase in ("prefill", "decode")}
        count = {phase: sum(math.prod(op["output_shape"]) for op in operations) for phase, operations in tensors.items()}
        evaluations = count["prefill"] + (workload["output_tokens"] - 1) * count["decode"]
        batches = {phase: sum(math.ceil(math.prod(op["output_shape"]) / CONFIGURATION["lanes"]) for op in operations) for phase, operations in tensors.items()}
        total_batches = batches["prefill"] + (workload["output_tokens"] - 1) * batches["decode"]
        assert evaluations and count["prefill"] == workload["input_tokens"] * count["decode"]
        projected = []
        for case in cases:
            # Only a helper-count hypothesis, not a compiled rescaling placement.
            key_bytes = 2 * case["party_key_payload_bytes"] * evaluations // case["lanes"]
            peer_bytes = case["rounds"] * (8 * evaluations + 2 * 78 * total_batches)
            projected.append({
                "bits": case["bits"], "shift": case["shift"], "rounding": case["rounding"], "layout": case["layout"],
                "both_party_key_payload_bytes": key_bytes,
                "peer_frame_bytes_at_64_lane_batches": peer_bytes,
                "issuance_seconds_linear_projection": case["median_seconds"]["issuance"] * evaluations / case["lanes"],
                "two_party_local_seconds_linear_projection": case["median_seconds"]["online"] * evaluations / case["lanes"],
                "executable_decoder": False,
            })
        projections.append({
            **workload, "semantic_plan_digest": plan.digest, "semantic_silu_operations_per_phase": len(tensors["prefill"]),
            "silu_elements": count, "hypothetical_helper_evaluations": evaluations,
            "hypothetical_batches": total_batches, "cases": projected,
        })
    return projections


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
        raise RuntimeError("FSS probe needs 1 GiB headroom beyond the host reserve")
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
        for bits, shift in CONFIGURATION["shapes"]:
            for rounding in CONFIGURATION["rounding"]:
                for layout in CONFIGURATION["layouts"]:
                    guard.check()
                    if guard.error:
                        raise RuntimeError(guard.error)
                    arguments = [str(bits), str(shift), rounding, layout, str(CONFIGURATION["lanes"]), str(CONFIGURATION["samples"])]
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", str(binary), *arguments],
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
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
                    assert [case[key] for key in ("bits", "shift", "rounding", "layout", "lanes", "samples")] == [bits, shift, rounding, layout, CONFIGURATION["lanes"], CONFIGURATION["samples"]]
                    case["median_seconds"] = {phase: statistics.median(case[f"{phase}_ns"]) / 1e9 for phase in ("issuance", "online", "clear")}
                    cases.append(case)
    finally:
        guard.close()
    return {
        "schema": "pllm.shared_rescale_reference.v1", "configuration": CONFIGURATION,
        "source_lock_digest": resolved.source_lock_digest, "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "cases": cases, "geometry_projections": projections(json.loads(config_bytes), cases),
        "executable_sdk": False, "privacy_reviewed": False, "whole_decoder_numeric_parity": False,
        "memory_guard": {"host": asdict(host), "required_headroom_bytes": GiB, "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()},
        "source_sha256": {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in [
            "benchmarks/research/shared_rescale_reference.py", "crates/pllm-garble/src/point_fss.rs",
            "crates/pllm-garble/src/shared_rescale_reference.rs", "crates/pllm-garble/examples/shared_rescale_probe.rs", "Cargo.lock",
        ]},
        "limitations": [
            "component reference, not either paper's complete protocol/backend or optimized performance reproduction",
            "fresh-process single-thread native CPU: no GPU, WAN, role transport, model values, logits or KV",
            "phase timers are elapsed time; process CPU includes issuance, evaluation, clear oracle and lifecycle checks",
            "party key payload counts every seed, correction, mask and constant; offline metadata/transport and allocator overhead are extra",
            "observed process RSS contains dealer and both parties together; not independent-role or client memory",
            "framing is context binding, not authentication or malicious-output verification; no distributed dealer or reviewed cryptography",
            "Qwen geometry is a hypothetical helper placement, not compatibility with its current W8A8 floating-point nonlinear execution",
            "projections exclude attention, polynomial/nonlinear evaluation, token feedback and complete response work",
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
    print(json.dumps({"cases": len(report["cases"]), "checked_outputs": sum(case["checked_outputs"] for case in report["cases"]),
                      "maximum_swap_growth_bytes": report["memory_guard"]["maximum_swap_growth_bytes"]}))


if __name__ == "__main__":
    main()
