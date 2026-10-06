"""Common encoded-profile component screen with explicit missing block costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from benchmarks.research.nonlinear_block_reference import ROOT, sha
from benchmarks.research.shared_rescale_reference import measure_child

CONFIGURATION = {
    "methods": ["compact", "logrow", "curl0", "curl2", "curl4"],
    "profile": "uniform public Compact Q7 calibration, four requested pieces",
    "samples": 17, "domain": "signed Q7 [-128,128], representing [-1,1]",
    "curl_padding": "unused table entries repeat the public upper endpoint before Haar compression",
    "scope": "activation component lower bounds; different role/output contracts do not rank",
}


def run(output):
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pllm-garble", "--example", "nonlinear_frontier_probe"], cwd=ROOT, check=True)
    target = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))["target_directory"]
    binary = Path(target) / "release/examples/nonlinear_frontier_probe"
    cases = []
    for method in CONFIGURATION["methods"]:
        raw = subprocess.check_output([sys.executable, "-m", "benchmarks.research.nonlinear_frontier", "--child", str(binary), method], cwd=ROOT, text=True)
        case = json.loads(raw)
        rows = case["samples"]
        if len(rows) != CONFIGURATION["samples"] or len({row["material_body_bytes"] for row in rows}) != 1:
            raise ValueError("reference sample shape changed")
        case["summary"] = {
            "material_body_bytes_per_element": rows[0]["material_body_bytes"],
            "peer_frame_bytes_per_element": rows[0]["peer_frame_bytes"],
            "encoded_matches": sum(row["actual_q7"] == row["expected_q7"] for row in rows),
            "worst_encoded_error": max(abs(row["actual_q7"] - row["expected_q7"]) for row in rows),
            "median_issuance_seconds": statistics.median(row["issuance_seconds"] for row in rows),
            "median_local_online_seconds": statistics.median(row["local_online_seconds"] for row in rows),
        }
        cases.append(case)
    domain_path = "docs/evidence/research-nonlinear-block-reference-qwen25.json"
    domain = json.loads((ROOT / domain_path).read_text())
    projections = []
    for geometry in domain["geometry_projections"]:
        elements = geometry["gated_elements"]
        projections.append({
            "input_tokens": geometry["input_tokens"], "output_tokens": geometry["output_tokens"],
            "gated_elements": elements,
            "cases": [{"method": case["method"],
                "activation_material_body_bytes_lower_bound": elements * case["summary"]["material_body_bytes_per_element"],
                "activation_peer_frame_bytes_lower_bound": None if case["summary"]["peer_frame_bytes_per_element"] is None else elements * case["summary"]["peer_frame_bytes_per_element"]}
                for case in cases],
        })
    sources = ["benchmarks/research/nonlinear_frontier.py", "crates/pllm-garble/examples/nonlinear_frontier_probe.rs",
               "crates/pllm-core/src/compact.rs", "crates/pllm-garble/src/compact_polynomial.rs",
               "crates/pllm-garble/src/logrow.rs", "crates/pllm-garble/src/shared_lut_reference.rs"]
    report = {"schema": "pllm.nonlinear_frontier.v1", "configuration": CONFIGURATION,
        "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in sources},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "domain_evidence": {"path": domain_path, "sha256": sha((ROOT / domain_path).read_bytes())},
        "cases": cases, "geometry_projections": projections,
        "complete_block_cost": None, "executable_sdk": False,
        "missing_costs": ["protected input conversion and range enforcement", "output conversion between labels and additive shares",
            "secret multiplication by up and final rescaling", "complete authenticated transport, offline metadata and key distribution",
            "full tensor schedule, attention, normalization and private token feedback"],
        "limitations": ["fixed Q7 domain already fails the source-locked Qwen gate; no real checkpoint execution here",
            "Curl leaves output shared; Compact/LogRow output labels decode only at the trusted client",
            "compression changes the encoded function", "no reviewed cryptography or whole-response benchmark"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": [{"method": c["method"], **c["summary"]} for c in cases], "executable_sdk": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", nargs=2)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(measure_child(args.child)))
    elif args.output:
        run(args.output)
    else:
        parser.error("--output or --child required")
