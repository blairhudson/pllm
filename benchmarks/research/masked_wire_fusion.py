"""Matched complete-block SDK measurements with fresh processes and fixed numerics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
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
    "layouts": ["unfused", "fused"], "lanes": [1, 64, 256],
    "samples": 3, "repetitions": 3,
    "input_source": "installed public-boundary-fixture-v1",
    "function": "complete signed Q24-to-Q12 SiLU(gate) * up, fixed floor/residual-table profile",
    "fusion": "same masked gate input supplies Q12 shift, Q6 index shift and sign/interval predicates",
    "scope": "CPU component adaptation of FuseFSS multi-view rewriting on the SIGMA-derived block",
}
GEOMETRY = "docs/evidence/research-masked-wire-reference-qwen25.json"


def child(layout: str, lanes: int) -> dict:
    from pllm.metrics import benchmark_operator
    from pllm.nonlinear import MaskedWireGatedMultiplyQ12

    report = benchmark_operator(
        MaskedWireGatedMultiplyQ12(layout=layout), lanes=lanes, samples=CONFIGURATION["samples"],
    ).to_dict()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "layout": layout, "lanes": lanes, "report": report,
        "process_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "process_peak_rss_bytes": int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
    }


def summarize(cases: list[dict]) -> list[dict]:
    results = []
    for lanes in CONFIGURATION["lanes"]:
        groups = {layout: [c for c in cases if c["layout"] == layout and c["lanes"] == lanes]
                  for layout in CONFIGURATION["layouts"]}
        records = [c["report"] for group in groups.values() for c in group]
        if len(records) != 2 * CONFIGURATION["repetitions"]:
            raise ValueError("incomplete matched layout cohort")
        if any(r["numeric_contract_digest"] != records[0]["numeric_contract_digest"]
               or r["quality"] != records[0]["quality"] for r in records):
            raise ValueError("layout pair changed numeric contract or independent oracle results")
        for layout, group in groups.items():
            first = group[0]["report"]
            if any(c["report"]["resources"] != first["resources"] for c in group):
                raise ValueError("public resource shape changed across fresh masks")
            results.append({
                "lanes": lanes, "layout": layout, "resources": first["resources"],
                "numeric_contract_digest": first["numeric_contract_digest"],
                "configuration": first["configuration"],
                "median_issuance_seconds": statistics.median(c["report"]["median_issuance_seconds"] for c in group),
                "median_online_seconds": statistics.median(c["report"]["median_online_seconds"] for c in group),
                "median_process_cpu_seconds": statistics.median(c["process_cpu_seconds"] for c in group),
                "median_process_peak_rss_bytes": statistics.median(c["process_peak_rss_bytes"] for c in group),
            })
    return results


def projections(summary: list[dict]) -> list[dict]:
    source = json.loads((ROOT / GEOMETRY).read_text())
    result = []
    for geometry in source["geometry_projections"]:
        count, batches = geometry["gated_elements"], geometry["bounded_256_lane_batches"]
        # The retained, compiler-derived Qwen2.5 intermediate width is divisible
        # by 256. No unpriced partial-frame/key padding is hidden in this screen.
        if count != batches * 256:
            raise ValueError("source geometry needs explicit partial-batch accounting")
        layouts = []
        for row in summary:
            if row["lanes"] != 256:
                continue
            r = row["resources"]
            layouts.append({
                "layout": row["layout"],
                "both_party_raw_key_payload_bytes": sum(r["raw_key_payload_bytes_per_party"]) * batches,
                "peer_frame_bytes": sum(r["peer_frame_bytes_by_direction"]) * batches,
                "client_mask_payload_estimate_bytes": r["client_mask_payload_estimate_bytes"] * batches,
                "client_entry_exit_payload_estimate_bytes": r["client_entry_exit_payload_estimate_bytes"] * batches,
                "public_table_bytes_per_party": r["public_table_bytes_per_party"],
                "local_online_seconds_linear_projection": row["median_online_seconds"] * batches,
                "local_issuance_seconds_linear_projection": row["median_issuance_seconds"] * batches,
                "full_wire_bytes": None, "whole_decoder_execution": False,
            })
        result.append({
            **{key: geometry[key] for key in ("input_tokens", "output_tokens", "semantic_plan_digest", "gated_elements", "bounded_256_lane_batches")},
            "scope": "historical semantic shapes only; complete operator costs, not a measured decoder",
            "layouts": layouts,
        })
    return result


def run(output: Path) -> None:
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory

    if output.exists():
        raise ValueError("choose a new output path; historical measurements are immutable")
    host = host_memory()
    if host.available - host.reserve < GiB:
        raise RuntimeError("operator cohort needs 1 GiB headroom beyond host reserve")
    active = []

    def abort(_reason):
        for process in tuple(active):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)
    cases = []
    guard.start()
    try:
        for repetition in range(CONFIGURATION["repetitions"]):
            for lanes in CONFIGURATION["lanes"]:
                for layout in CONFIGURATION["layouts"][::1 if repetition % 2 == 0 else -1]:
                    guard.check()
                    if guard.error:
                        raise RuntimeError(guard.error)
                    process = subprocess.Popen(
                        [sys.executable, "-m", "benchmarks.research.masked_wire_fusion", "--child", layout, str(lanes)],
                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
                    )
                    active.append(process)
                    try:
                        stdout, stderr = process.communicate(timeout=120)
                        if process.returncode:
                            raise RuntimeError(stderr)
                        cases.append(json.loads(stdout) | {"repetition": repetition})
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        active.remove(process)
        guard.check()
        if guard.error:
            raise RuntimeError(guard.error)
    finally:
        guard.close()
    summary = summarize(cases)
    source_files = [
        "benchmarks/research/masked_wire_fusion.py", "python/pllm/nonlinear/__init__.py",
        "python/pllm/metrics/masked_wire.py", "python/pllm/metrics/operator_benchmarks.py",
        "crates/pllm-python/src/masked_wire.rs", "crates/pllm-garble/src/masked_wire_reference.rs",
        "crates/pllm-garble/src/compact_dpf.rs", "crates/pllm-core/src/residual_silu_reference.rs", "Cargo.lock",
    ]
    report = {
        "schema": "pllm.masked_wire_fusion_sdk.v1", "configuration": CONFIGURATION,
        "cases": cases, "summary": summary, "geometry_projections": projections(summary),
        "checked_outputs": sum(c["report"]["quality"]["checked_outputs"] for c in cases),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files},
        "geometry_source": {"path": GEOMETRY, "sha256": hashlib.sha256((ROOT / GEOMETRY).read_bytes()).hexdigest()},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "memory_guard": {"host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "limitations": [
            "Three fresh-process repetitions, alternating layout order; no paper GPU or Internet measurement",
            "Trusted in-process dealer/client; two one-use party-local native evaluators; no independent privacy review",
            "Fusion changes public-view widths and round schedule, not integer or approximation semantics",
            "Key payloads exclude dealer framing and metadata; boundary mask/entry-exit sizes are estimates",
            "RSS and process CPU include Python import/runtime; native per-sample timings remain separate",
            "The semantic projections are operator-only; protected float conversion and other decoder operations are absent",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"checked_outputs": report["checked_outputs"], "summary": summary,
                      "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", nargs=2, metavar=("LAYOUT", "LANES"))
    args = parser.parse_args()
    if args.child:
        layout, lanes = args.child
        print(json.dumps(child(layout, int(lanes))))
    elif args.output:
        run(args.output)
    else:
        parser.error("--output required")


if __name__ == "__main__":
    main()
