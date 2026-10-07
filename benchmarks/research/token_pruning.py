"""Fresh-process CipherPrune threshold/compaction costs and disclosure witnesses.

Synthetic public row-stochastic attention and complete arbitrary 32-bit records.
Full-width counts are symbolic layouts; no Qwen pruning or decoder is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CASES = ((4, 2, 1, "half"), (4, 2, 16, "half"), (8, 4, 4, "half"),
         (4, 2, 16, "all"), (4, 2, 16, "none"))
SOURCES = (
    "benchmarks/research/token_pruning.py", "crates/pllm-core/src/token_pruning.rs",
    "crates/pllm-garble/src/token_pruning.rs", "crates/pllm-garble/src/boolean.rs",
    "crates/pllm-python/src/token_pruning.rs", "python/pllm/metrics/token_pruning.py",
    "python/pllm/metrics/operator_benchmarks.py", "python/pllm/passes/__init__.py", "Cargo.lock",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tokens, heads, features, case):
    from pllm.passes import CipherPruneTokenPruning

    threshold, reduction = (0, 65536) if case == "all" else (65535, 65536) if case == "none" else (8192, 24576)
    row = [65536 // tokens] * tokens
    if case == "half":
        row = [0 if i % 2 == 0 else 65536 // (tokens // 2) for i in range(tokens)]
    if sum(row) != 65536:
        raise ValueError("public attention fixture is not normalized")
    words = [[0, 0xFFFFFFFF, 0x80000000, i][i % 4] for i in range(tokens * features)]
    return (CipherPruneTokenPruning(threshold_q16=threshold, reduction_threshold_q16=reduction),
            {"attention_q16": row * (heads * tokens), "words": words})


def worker(tokens, heads, features, case):
    from pllm.metrics import benchmark_operator

    component, inputs = fixture(tokens, heads, features, case)
    report = benchmark_operator(component, geometry={"tokens": tokens, "heads": heads},
                                lanes=features, samples=1, inputs=inputs).to_dict()
    expected_count = 0 if case == "none" else tokens if case == "all" else tokens // 2
    expected_checks = 1 + expected_count * features + (expected_count if case != "all" else 0)
    if (report["quality"]["checked_outputs"] != expected_checks
            or report["quality"]["exact_integer_outputs"] != expected_checks):
        raise ValueError("protected compaction does not preserve complete checked records")
    return report


def projections():
    from pllm import _native

    rows = []
    # Declared Qwen2.5-0.5B-shaped dimensions, not a compiler binding or workload.
    for tokens in (39, 150):
        for retained in (tokens // 2, tokens):
            row = json.loads(_native.token_pruning_projection(tokens, 14, 896, 8192, 24576, retained))
            row.update(tokens=tokens, heads=14, features=896, material_issued=False,
                       scope="one hypothetical full-width pruning invocation")
            row["evaluator_material_bytes"] = sum(p["evaluator_material_bytes"] for p in row["phases"])
            row["issuer_key_bytes"] = sum(p["issuer_key_bytes"] for p in row["phases"])
            row["boundary_label_bytes_estimate"] = sum(p["boundary_payload_estimate_bytes"] for p in row["phases"])
            row["public_metadata_bytes_estimate"] = sum(p["public_metadata_bytes_estimate"] for p in row["phases"])
            row["two_party_payload_bytes_retired"] = 2 * (tokens - retained) * 896 * 4
            if row["admitted"]:
                raise ValueError("full-width screen unexpectedly became admitted; reassess scope")
            rows.append(row)
    return rows


def disclosure_witnesses():
    # These witnesses concern source-disclosed outputs, not a garbling attack.
    # A one-token count is its keep bit, but normalized one-token attention fixes
    # that bit. The separate two-token witness varies within normalized attention.
    count = [{"private_keep": b, "public_retained": int(b)} for b in (False, True)]
    degrees = [{"private_score_q16": s, "public_degree_bit": s > 24576} for s in (16384, 32768)]
    normalized = []
    for row in ([32768, 32768], [49152, 16384]):
        assert sum(row) == 65536
        normalized.append({"private_repeated_attention_row_q16": row, "threshold_q16": 32768,
                           "public_retained": sum(2 * v > 2 * 32768 for v in row)})
    assert [v["public_retained"] for v in normalized] == [0, 1]
    return {"algebraic_one_token_count_identity": count,
            "normalized_one_token_attention_fixes_keep_bit": True,
            "two_token_normalized_attention_count_disclosure": normalized,
            "compacted_degree_reveals_threshold_predicate": degrees,
            "constant_schedule_hides_original_positions_only_within_equal_public_counts": True,
            "independent_cryptographic_review": False}


def run(samples=3):
    if type(samples) is not int or not 1 <= samples <= 8:
        raise ValueError("samples must be between one and eight")
    sources = {p: sha(ROOT / p) for p in SOURCES}
    cases = []
    for tokens, heads, features, case in CASES:
        reports = []
        for _ in range(samples):
            process = subprocess.run([sys.executable, "-m", "benchmarks.research.token_pruning", "--worker",
                str(tokens), str(heads), str(features), case], cwd=ROOT, check=True, capture_output=True,
                text=True, timeout=120, env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
            reports.append(json.loads(process.stdout))
        if any(r["resources"] != reports[0]["resources"] for r in reports):
            raise ValueError("fixed public geometry changed resources across fresh issuances")
        cases.append({"fixture": case, "tokens": tokens, "heads": heads, "features": features,
                      "reports": reports, "median_seconds": {
                          key: statistics.median(r["samples"][0][key] for r in reports)
                          for key in ("issuance_seconds", "online_seconds", "input_sharing_seconds")}})
    if sources != {p: sha(ROOT / p) for p in SOURCES}:
        raise ValueError("pruning implementation changed during measurement")
    return {"schema": "pllm.token_pruning_cost.v1", "source_sha256": sources,
            "cases": cases, "fresh_processes": samples * len(cases), "samples_per_case": samples,
            "checked_outputs": sum(r["quality"]["checked_outputs"] for c in cases for r in c["reports"]),
            "full_width_projections": projections(), "disclosure_witnesses": disclosure_witnesses(),
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "scope": {"component": "threshold/count plus stable full-record compaction and degree selection",
                      "inputs": "synthetic public normalized Q16 attention and arbitrary complete words",
                      "full_width_geometry": "declared 14-head/896-feature shapes; no source-bound decoder",
                      "source_ot_primitives": False, "trained_thresholds": False,
                      "selected_polynomial_execution": False, "causal_kv_mapping": False,
                      "checkpoint_quality": None, "whole_model_executable": False,
                      "distributed_input_encoding": False, "provider_transport": False,
                      "peak_memory_bytes": None, "full_wire_bytes": None}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--worker", nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        n, h, w, case = args.worker
        print(json.dumps(worker(int(n), int(h), int(w), case)))
    else:
        if args.output is None or args.output.exists():
            parser.error("provide a new --output path")
        result = run(args.samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(f"{result['checked_outputs']} exact outputs in {result['fresh_processes']} fresh processes")
