"""Bounded client-garbled top-k/gather: fresh native processes and a Python oracle.

This fixture does not execute a model or MPCache's distributed scoring protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CASES = ((4, 2, 2, 1), (8, 4, 4, 2), (16, 2, 8, 2), (8, 4, 8, 0))
SOURCE_FILES = (
    "benchmarks/research/retained_selection.py",
    "crates/pllm-garble/examples/retained_selection_probe.rs",
    "crates/pllm-garble/src/retained_selection.rs",
    "crates/pllm-garble/src/retained_selection/tests.rs",
    "crates/pllm-garble/src/boolean.rs",
    "crates/pllm-core/src/retained_kv.rs", "Cargo.lock",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def oracle(rows, width, keep, recent, variant):
    scores = [1.0 if variant == 0 else ((i * 7 + variant * 3) % rows) / 16 for i in range(rows)]
    ranked = sorted(range(rows - recent), key=lambda i: (-scores[i], i))
    selected = sorted(ranked[:keep - recent] + list(range(rows - recent, rows)))
    positions = [2 * i + 1 for i in selected]
    data = b"".join(struct.pack("<I", p) for p in positions)
    for multiplier, modulus, offset, denominator in ((17, 37, 18, 8), (11, 53, 26, 16)):
        data += b"".join(struct.pack("<f", ((p * width + j) * multiplier % modulus - offset) / denominator)
                         for p in positions for j in range(width))
    return {"output_sha256": sha(data), "positions": positions,
            "checked_output_words": keep * (1 + 2 * width)}


def validate_case(case):
    args = [case[key] for key in ("rows", "width", "keep", "recent", "variant")]
    expected = oracle(*args)
    if case["output_sha256"] != expected["output_sha256"]:
        raise ValueError("native retained selection disagrees with independent Python oracle")
    rows, width, keep, _, _ = args
    m, allocation = case["material"], case["allocation"]
    if (m["evaluator_ciphertext_bytes"] != 32 * m["and_gates"]
            or m["input_label_bytes"] != rows * (64 + 64 * width) * 16
            or m["output_label_bytes"] != keep * (32 + 64 * width) * 16
            or m["client_encoding_bytes"] != 2 * (m["input_label_bytes"] + m["output_label_bytes"])
            or m["client_to_evaluator_material_bytes"] != sum(m[k] for k in (
                "evaluator_ciphertext_bytes", "constant_label_bytes", "input_label_bytes"))
            or m["evaluator_to_client_material_bytes"] != m["output_label_bytes"]):
        raise ValueError("retained selection material owners do not conserve")
    if (allocation["state_before_bytes"] != (rows + 2) * (8 * width + 4)
            or allocation["state_after_bytes"] != (keep + 2) * (8 * width + 4)
            or allocation["actual_wires"] > allocation["wire_bound"]
            or not 0 < allocation["combined_preflight_bound_bytes"] <= 64 << 20):
        raise ValueError("retained selection allocation bound mismatch")
    if not all(case["geometry_gates"].values()):
        raise ValueError("unsupported retained selection geometry was admitted")
    return expected["checked_output_words"]


def worker(binary, case, sample):
    # One native child in each fresh Python worker: ru_maxrss is its own lifetime
    # peak, never the cumulative maximum over earlier candidates.
    result = subprocess.run([str(binary), *map(str, (*case, sample % 3))],
                            check=True, capture_output=True, text=True, timeout=60)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = json.loads(result.stdout)
    report["native_process_cpu_seconds"] = usage.ru_utime + usage.ru_stime
    report["native_process_peak_rss_bytes"] = int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    report["checked_output_words"] = validate_case(report)
    return report


def run(samples=3):
    if type(samples) is not int or not 1 <= samples <= 8:
        raise ValueError("samples must be between 1 and 8")
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pllm-garble",
                    "--example", "retained_selection_probe"], cwd=ROOT, check=True)
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))
    binary = Path(metadata["target_directory"]) / "release/examples/retained_selection_probe"
    implementation = {path: sha((ROOT / path).read_bytes()) for path in SOURCE_FILES}
    binary_hash = sha(binary.read_bytes())
    cases = []
    for geometry in CASES:
        observations = []
        for sample in range(samples):
            output = subprocess.check_output([sys.executable, "-m", "benchmarks.research.retained_selection",
                "--worker", str(binary), "--geometry", *map(str, geometry), "--sample", str(sample)], cwd=ROOT)
            observation = json.loads(output)
            validate_case(observation)
            observations.append(observation)
        if len({r["public_schedule_sha256"] for r in observations}) != 1:
            raise ValueError("public access schedule changed with private fixture inputs")
        if any(r["material"] != observations[0]["material"] for r in observations):
            raise ValueError("material size changed with private fixture inputs")
        cases.append({"geometry": dict(zip(("rows", "width", "keep", "recent"), geometry, strict=True)),
            "samples": observations,
            "median": {
                "native_process_cpu_seconds": statistics.median(r["native_process_cpu_seconds"] for r in observations),
                "native_process_peak_rss_bytes": statistics.median(r["native_process_peak_rss_bytes"] for r in observations),
                "timings_seconds": {key: statistics.median(r["timings_seconds"][key] for r in observations)
                                    for key in observations[0]["timings_seconds"]}}})
    if implementation != {path: sha((ROOT / path).read_bytes()) for path in SOURCE_FILES} or binary_hash != sha(binary.read_bytes()):
        raise ValueError("retained selection implementation changed during measurement")
    return {"schema": "pllm.retained_selection_reference.v1", "samples_per_case": samples,
        "source_sha256": implementation, "native_binary_sha256": binary_hash,
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
            "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()},
        "sources": ["https://arxiv.org/abs/2501.06807v2", "https://eprint.iacr.org/2014/756"],
        "scope": {"method": "client-garbled stable top-k and complete-record bitonic movement",
            "inputs": "public synthetic finite float32 fixtures standing in for client-owned private scores/KV",
            "scores": "client-owned nonnegative float32; calculation excluded",
            "garbler": "trusted client with complete private state",
            "evaluator": "one opaque one-use native circuit handle; no output decoder",
            "output": "trusted client; exact owner/revision commit and physical compaction",
            "mpcache_distributed_scoring": False, "provider_resident_state": False,
            "compiler_validated_context": False, "whole_decoder": False, "transport": False,
            "independent_cryptographic_review": False, "full_wire_bytes": None,
            "material_bytes": "logical label/ciphertext bodies, not an implemented codec",
            "memory": "fresh native child process lifetime peak; excludes Python launcher",
            "allocation_bound": "64 MiB including native owner/compaction overlap; not an RSS ceiling"},
        "checked_output_words": sum(r["checked_output_words"] for c in cases for r in c["samples"]),
        "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--geometry", nargs=4, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sample", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.geometry, args.sample)))
        return
    if args.output is None or args.output.exists():
        parser.error("provide a new --output path")
    result = run(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
    print(f"{result['checked_output_words']} exact words across {len(result['cases']) * args.samples} fresh native processes")


if __name__ == "__main__":
    main()
