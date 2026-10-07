"""Matched full-record versus compact-index sorting and complete oblivious gather.

Public synthetic KV bit patterns; one fresh native process/material issuance per
sample. Complete-state Qwen shapes are resource projections, never issued keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys

from benchmarks.research.retained_selection import CASES, ROOT, oracle, validate_case

MATCHED = (*CASES, (2, 128, 1, 0))
INDEX_ONLY = ((4, 128, 2, 1), (32, 1, 1, 0))
SOURCE_FILES = (
    "benchmarks/research/retained_selection_indices.py", "benchmarks/research/retained_selection.py",
    "crates/pllm-garble/examples/retained_selection_probe.rs",
    "crates/pllm-garble/src/retained_selection.rs",
    "crates/pllm-garble/src/retained_selection/indexed.rs",
    "crates/pllm-garble/src/retained_selection/cost.rs",
    "crates/pllm-garble/src/retained_selection/tests.rs",
    "crates/pllm-garble/src/boolean.rs", "crates/pllm-core/src/retained_kv.rs", "Cargo.lock",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_projection(row):
    n, w, k = row["rows"], row["width"], row["keep"]
    if (row["material_issued"] is not False
            or row["input_label_bytes"] != n * (64 + 64 * w) * 16
            or row["output_label_bytes"] != k * (32 + 64 * w) * 16
            or row["client_encoding_bytes"] != 2 * (row["input_label_bytes"] + row["output_label_bytes"])
            or row["label_and_ciphertext_body_bytes"] != row["and_gates"] * 32
                + row["constant_label_bytes"] + row["input_label_bytes"] + row["output_label_bytes"]
            or row["state_before_bytes"] != (n + 2) * (8 * w + 4)
            or row["state_after_bytes"] != (k + 2) * (8 * w + 4)
            or row["actual_wires"] > row["wire_bound"]):
        raise ValueError("compact selection projection omits material or state owners")
    expected = (row["combined_preflight_bound_bytes"] <= 64 << 20
                and row["input_label_bytes"] <= 131072 * 16 and row["output_label_bytes"] <= 131072 * 16
                and row["wire_bound"] <= 2000000 and row["and_gates"] <= 1000000)
    if row["admitted"] is not expected:
        raise ValueError("compact selection resource admission mismatch")


def worker(binary, geometry, sample, layout):
    result = subprocess.run([str(binary), *map(str, geometry), str(sample % 3), str(layout)],
                            check=True, capture_output=True, text=True, timeout=60)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = json.loads(result.stdout)
    report["native_process_cpu_seconds"] = usage.ru_utime + usage.ru_stime
    report["native_process_peak_rss_bytes"] = int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    report["checked_output_words"] = validate_case(report)
    return report


def run(samples):
    if type(samples) is not int or not 1 <= samples <= 8:
        raise ValueError("samples must be between 1 and 8")
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pllm-garble",
                    "--example", "retained_selection_probe"], cwd=ROOT, check=True)
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))
    binary = Path(metadata["target_directory"]) / "release/examples/retained_selection_probe"
    sources = {p: sha(ROOT / p) for p in SOURCE_FILES}
    binary_hash = sha(binary)
    cases = []
    for geometry in (*MATCHED, *INDEX_ONLY):
        for layout in (0, 1) if geometry in MATCHED else (1,):
            observations = []
            for sample in range(samples):
                output = subprocess.check_output([sys.executable, "-m", "benchmarks.research.retained_selection_indices",
                    "--worker", str(binary), "--geometry", *map(str, geometry), "--sample", str(sample),
                    "--layout", str(layout)], cwd=ROOT)
                row = json.loads(output)
                validate_case(row)
                observations.append(row)
            if (len({r["public_schedule_sha256"] for r in observations}) != 1
                    or any(r["material"] != observations[0]["material"] for r in observations)):
                raise ValueError("private choices changed public access schedule or material geometry")
            projection = json.loads(subprocess.check_output([str(binary), *map(str, geometry), "0", str(layout), "1"]))
            validate_projection(projection)
            measured = observations[0]
            m = measured["material"]
            if (not projection["admitted"] or projection["actual_wires"] != measured["allocation"]["actual_wires"]
                    or any(projection[k] != m[k] for k in ("and_gates", "constant_label_bytes", "input_label_bytes",
                        "output_label_bytes", "client_encoding_bytes", "public_metadata_bytes_estimate"))):
                raise ValueError("geometry-only resource counter disagrees with issued circuit")
            cases.append({"geometry": dict(zip(("rows", "width", "keep", "recent"), geometry, strict=True)),
                "layout": layout, "samples": observations, "projection": projection,
                "median": {"native_process_cpu_seconds": statistics.median(r["native_process_cpu_seconds"] for r in observations),
                    "native_process_peak_rss_bytes": statistics.median(r["native_process_peak_rss_bytes"] for r in observations),
                    "timings_seconds": {k: statistics.median(r["timings_seconds"][k] for r in observations)
                                        for k in observations[0]["timings_seconds"]}}})
    # Derive complete-state geometry from an existing compiler-bound pinned
    # checkpoint contract, without loading values or pretending to execute it.
    source_report_path = ROOT / "docs/evidence/research-retained-decoder-qwen25.json"
    source_report = json.loads(source_report_path.read_bytes())
    contract = source_report["candidates"][1]["contract"]
    widths = {layer["kv_heads"] * layer["head_dim"] for layer in contract["layers"]}
    if len(widths) != 1:
        raise ValueError("reference cost screen requires one declared KV width")
    width = widths.pop()
    gates = []
    for rows in (16, 64, 256, 512):
        for layout in (0, 1):
            row = json.loads(subprocess.check_output([str(binary), str(rows), str(width), str(rows // 2),
                                                      str(rows // 4), "0", str(layout), "1"]))
            validate_projection(row)
            row["whole_body_label_and_ciphertext_body_bytes"] = row["label_and_ciphertext_body_bytes"] * len(contract["layers"])
            gates.append(row)
    if sources != {p: sha(ROOT / p) for p in SOURCE_FILES} or binary_hash != sha(binary):
        raise ValueError("selection implementation changed during measurement")
    return {"schema": "pllm.retained_selection_indices.v1", "samples_per_case": samples,
        "source_sha256": sources, "native_binary_sha256": binary_hash, "cases": cases,
        "checked_output_words": sum(r["checked_output_words"] for c in cases for r in c["samples"]),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
            "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()},
        "geometry_source": {"path": str(source_report_path.relative_to(ROOT)), "sha256": sha(source_report_path),
            "checkpoint": source_report["source"], "layers": len(contract["layers"]), "kv_width": width},
        "full_width_projections": gates,
        "scope": {"method": "compact-index bitonic sorting plus full-scan one-hot oblivious KV gather",
            "inputs": "synthetic public float32 fixtures standing in for trusted-client private scores and KV",
            "scores": "supplied by trusted client; protected scoring absent", "output_decoder": "trusted client only",
            "layouts": {"0": "complete records", "1": "compact indices and oblivious gather"},
            "material_bytes": "complete logical label/ciphertext bodies; public instructions separately estimated",
            "memory": "fresh native process peak; preflight separately bounds payload allocation",
            "full_width_projections": "exact layout counts at declared geometry; no issuance or execution",
            "provider_resident_state": False, "compiler_validated_context": False, "whole_decoder": False,
            "transport": False, "independent_cryptographic_review": False, "full_wire_bytes": None},
        "sources": ["https://arxiv.org/abs/2501.06807v2", "https://eprint.iacr.org/2014/756"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--geometry", nargs=4, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sample", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--layout", type=int, choices=(0, 1), default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.geometry, args.sample, args.layout)))
    else:
        if args.output is None or args.output.exists():
            parser.error("provide a new --output path")
        result = run(args.samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(f"{result['checked_output_words']} exact output words in {len(result['cases']) * args.samples} fresh processes")
