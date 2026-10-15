#!/usr/bin/env python3
"""Bounded cfg(test) fusion, separate-call baseline, analytical payload budget."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "private_pages::boundary_scan_fusion_probe::fused_fixture"


def build_fused_fixture() -> Path:
    command = ["cargo", "test", "--locked", "--offline", "--release", "-p", "pllm-garble", "--lib", "--no-run", "--message-format=json"]
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    artifacts = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    binaries = [item["executable"] for item in artifacts if item.get("reason") == "compiler-artifact" and item.get("executable") and item.get("target", {}).get("name") == "pllm_garble" and item.get("profile", {}).get("test")]
    if len(binaries) != 1:
        raise RuntimeError("expected exactly one pllm_garble test executable")
    return Path(binaries[0])


def run_fused_fixture(binary: Path, rows: int, columns: int, index: int, repetitions: int) -> dict:
    env = os.environ.copy()
    env.update(PLLM_FUSION_ROWS=str(rows), PLLM_FUSION_COLUMNS=str(columns), PLLM_FUSION_INDEX=str(index), PLLM_FUSION_REPETITIONS=str(repetitions))
    p = subprocess.run([str(binary.resolve()), FIXTURE, "--exact", "--ignored", "--nocapture", "--test-threads=1"], cwd=ROOT, env=env, check=True, text=True, capture_output=True)
    # libtest may prefix test name on same line as JSON with one test thread.
    values = []
    for line in p.stdout.splitlines():
        if '{"' in line:
            values.append(json.loads(line[line.index('{"'):]))
    if len(values) != 1 or values[0].get("status") != "test-only-fused-reference":
        raise RuntimeError("missing unique fused fixture JSON")
    return values[0]


def provenance(binary: Path) -> dict:
    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                value.update(chunk)
        return value.hexdigest()

    sources = (
        "scripts/probe_boundary_scan_fusion.py",
        "crates/pllm-garble/src/boundary_scan_fusion_probe.rs",
        "crates/pllm-garble/src/private_pages.rs",
        "crates/pllm-core/src/kernels.rs",
        "Cargo.lock",
    )
    return {
        "test_binary_sha256": digest(binary),
        "source_sha256": {name: digest(ROOT / name) for name in sources},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "scope": "fixture binary plus listed sources; not a complete hermetic build manifest",
    }


def material_model(rows: int, columns: int, tile_rows: int = 128, pending: int = 1) -> dict:
    """Per-worker payload accounting; never present this as total process RSS."""
    if any(type(x) is not int or x < 1 for x in (rows, columns, tile_rows, pending)):
        raise ValueError("dimensions, tile rows and pending count must be positive integers")
    if rows > 262144 or columns + 4 > 65536 or rows * (columns + 4) > 512 * 1024**2:
        raise ValueError("dimensions exceed private-page table policy")
    if tile_rows % 128 or tile_rows > 4096 or pending > 4096:
        raise ValueError("tile rows must be 128-aligned and <=4096; pending <=4096")
    depth = ((rows + 127) // 128 - 1).bit_length()
    query = 97 + 17 * depth
    reply = 65 + columns + 4
    weights, scales = rows * columns, 4 * rows
    # Optimistic candidate: each weight byte loaded once into register feeding
    # both operations. Current public API cannot realize this estimate.
    return {
        "scope": "analytical per worker per paired lookup/random-head; payload bytes only",
        "physical_read_instrumentation": False,
        "baseline_weight_scan_bytes": 2 * weights,
        "candidate_weight_scan_bytes_lower_bound": weights,
        "scale_scan_bytes_both": scales,
        "weight_scan_saving_fraction": weights / (2 * weights + scales),
        "two_role_token_baseline_scan_bytes": 4 * weights + 2 * scales,
        "two_role_token_candidate_scan_bytes_lower_bound": 3 * weights + 2 * scales,
        "two_role_token_scan_saving_fraction": weights / (4 * weights + 2 * scales),
        "matrix_owner_bytes": weights,
        "native_scale_owner_bytes": scales,
        "retained_caller_scale_bytes_if_any": scales,
        "setup_source_plus_native_matrix_bytes": 2 * weights,
        "descriptor_hash_row_scratch_bytes": columns,
        "descriptor_hash_logical_bytes": weights + scales,
        "query_packet_bytes": query,
        "decoded_key_seed_control_payload_bytes": 32 + 17 * depth,
        "test_joint_preflight_peak_seed_control_payload_bytes": 2 * (32 + 17 * depth),
        "lookup_accumulator_bytes": columns + 4,
        "lookup_reply_bytes": reply,
        "random_head_input_bytes": 4 * columns,
        "head_result_bytes": 4 * rows,
        "current_single_query_batch_mask_bytes": rows,
        "candidate_tile_mask_bytes_upper_bound": min(rows, tile_rows),
        "candidate_fused_result_payload_bytes": reply + 4 * rows,
        "pending_client_r_and_Wr_bytes": pending * (4 * columns + 4 * rows),
        "context_commitment_payload_bytes": 32,
        "query_id_payload_bytes_each": 16,
        "test_fused_head_ticket_id_payload_bytes_each": 16,
        "test_fused_max_head_ledger_payload_bytes": 4096 * 16,
        "test_fused_terminal_mask_bytes": 16,
        "max_query_ledger_id_payload_bytes": 4096 * 16,
        "max_context_ledger_payload_bytes": 4096 * 32,
        "unaccounted_exact_bytes": "allocator/tree nodes, Arc/Vec/Server/Key headers, DPF stack, executor, Python/HTTP buffers; measure RSS separately",
        "cancel_after_dispatch_worst_case_wasted_weight_scan_bytes": weights,
        "cancel_after_dispatch_worst_case_wasted_scale_scan_bytes": scales,
        "cancel_after_dispatch_worst_case_discarded_result_bytes": reply + 4 * rows,
        "fused_scan_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=129)
    parser.add_argument("--columns", type=int, default=19)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument("--pending", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--native", action="store_true", help="run old separate-call baseline only")
    modes.add_argument("--fused", action="store_true", help="run actual cfg(test)-only fused reference")
    modes.add_argument("--build-fused", action="store_true", help="compile fixture and print binary path; no execution")
    parser.add_argument("--test-binary", type=Path, help="prebuilt libtest binary; with --fused skips build")
    args = parser.parse_args()
    try:
        model = material_model(args.rows, args.columns, args.tile_rows, args.pending)
    except ValueError as exc:
        parser.error(str(exc))
    index = args.rows - 1 if args.index is None else args.index
    if not 0 <= index < args.rows or not 1 <= args.repetitions <= 32:
        parser.error("index outside table or repetitions outside 1..32")
    if args.test_binary and not args.fused:
        parser.error("--test-binary requires --fused")
    result = {"schema": "boundary-scan-fusion-probe-v2", "status": "test-hook-available-not-executed", "fused_scan_executed": False, "sdk_selectable": False, "material": model}
    if args.build_fused:
        result["test_binary"] = str(build_fused_fixture())
    if args.native or args.fused:
        if max(args.rows, args.columns) > 4096 or args.rows * args.columns > 4_194_304:
            parser.error("native fixtures capped at dimensions <=4096 and 4 MiB weights")
    if args.native:
        command = ["cargo", "run", "--locked", "--offline", "--quiet", "--release", "-p", "pllm-garble", "--example", "boundary_scan_fusion", "--", str(args.rows), str(args.columns), str(index), str(args.repetitions)]
        completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
        result["native_baseline"] = json.loads(completed.stdout)
    if args.fused:
        binary = args.test_binary or build_fused_fixture()
        result["status"] = "test-only-fused-reference"
        result["native_fused_reference"] = run_fused_fixture(binary, args.rows, args.columns, index, args.repetitions)
        result["provenance"] = provenance(binary)
        result["fused_scan_executed"] = True
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
