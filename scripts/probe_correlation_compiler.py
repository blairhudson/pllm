"""Preparation-only subset tables for fresh ring32 correlations (research only)."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MASK = (1 << 32) - 1
BUDGET = 256 << 20
RUNTIME_ALLOWANCE = 96 << 20
MODES = ("native_wrap32", "native_digit8", "subset")


def preflight(rows, columns, block, *, budget=BUDGET):
    if any(type(x) is not int or x <= 0 for x in (rows, columns, block, budget)):
        raise ValueError("positive integer geometry required")
    if block > 12 or max(rows, columns) > 16384:
        raise ValueError("bounded geometry exceeded")
    widths = [min(block, columns - start) for start in range(0, columns, block)]
    entries = sum(1 << width for width in widths)
    table_bytes = rows * entries * 2  # signed-i8 subset sums fit signed-i16
    # Weight snapshot, tables, setup vector, expansion vector and lookup scratch;
    # client r/s and provider correction plus output reconstruction also priced.
    # Independent oracle may transiently convert W to i64. Price that too.
    arrays = 9 * rows * columns + table_bytes + rows * 8 * 6 + 4 * (columns + rows) * 3
    estimate = arrays + RUNTIME_ALLOWANCE
    dense = rows * columns
    return {
        "rows": rows, "columns": columns, "block": block,
        "table_bytes": table_bytes, "weight_bytes": dense,
        "estimated_process_bytes": estimate, "budget_bytes": budget,
        "dense_control_estimated_process_bytes": 17 * rows * columns + rows * 64 + columns * 32 + RUNTIME_ALLOWANCE,
        "native_control_estimated_process_bytes": 11 * rows * columns + rows * 64 + columns * 32 + RUNTIME_ALLOWANCE,
        "digit8_control_estimated_process_bytes": 15 * rows * columns + rows * 64 + columns * 32 + RUNTIME_ALLOWANCE,
        "status": "admitted_geometry" if estimate <= budget else "rejected_before_weights",
        "setup_subset_vector_additions": rows * sum((1 << w) - 1 for w in widths),
        "candidate_vector_lane_accumulations_per_sample": rows * 32 * len(widths),
        "dense_multiply_accumulations_per_sample": dense,
        "mask_index_bit_extractions_per_sample": columns * 32,
        "raw_client_seed_delivery_per_sample_bytes": 64,
        "raw_correction_delivery_per_sample_bytes": 4 * rows,
        "expanded_client_material_per_sample_bytes": 4 * (columns + rows),
        "preparation_xof_output_per_sample_bytes": 4 * (columns + rows),
        "client_xof_output_per_sample_bytes": 4 * (columns + rows),
        "client_online_ring_operations_per_sample": columns + rows,
        "client_cpu_s": None,
        "client_rss_bytes": None,
        "seed_delivery_implemented": False,
        "setup_table_delivery_bytes": 0,  # tables never leave Preparation
        "checkpoint_weight_delivery_bytes": None,
        "authenticated_framing_bytes": None,
        "arithmetic_count_crossover_samples": None,
    }


class SubsetCompiler:
    """Public weight-derived tables, owned exclusively by trusted Preparation."""

    def __init__(self, weights, block):
        import numpy as np
        if weights.ndim != 2 or weights.dtype != np.int8:
            raise ValueError("signed-i8 matrix required")
        self.cost = preflight(*weights.shape, block)
        if self.cost["status"] != "admitted_geometry":
            raise MemoryError("table preflight rejects before weight snapshot")
        self.rows, self.columns = weights.shape
        self.tables = []
        for start in range(0, self.columns, block):
            width = min(block, self.columns - start)
            table = np.zeros((1 << width, self.rows), dtype=np.int16)
            for index in range(1, 1 << width):
                low = index & -index
                table[index] = table[index ^ low] + weights[:, start + low.bit_length() - 1]
            self.tables.append((start, width, table))

    def project(self, r):
        import numpy as np
        if r.shape != (self.columns,) or r.dtype != np.uint32:
            raise ValueError("canonical ring32 mask required")
        # Sum each bit-plane before shifting; unsigned reduction is exact mod 2^32.
        result = np.zeros(self.rows, dtype=np.uint64)
        for bit in range(32):
            plane = np.zeros(self.rows, dtype=np.int64)
            for start, width, table in self.tables:
                index = sum(((int(r[start + k]) >> bit) & 1) << k for k in range(width))
                plane += table[index]
            result += (plane.astype(np.uint64) << np.uint64(bit))
        return (result & np.uint64(MASK)).astype(np.uint32)


def expand(seed, role, binding, width):
    import numpy as np
    if len(seed) != 32 or role not in (b"input-mask", b"output-mask") or not binding:
        raise ValueError("invalid seed domain")
    context = b"pllm/correlation-subset/v1\0" + role + len(binding).to_bytes(4, "big") + binding
    return np.frombuffer(hashlib.shake_256(context + seed).digest(4 * width), dtype="<u4").copy()


class ClientMaterial:
    """Test-local one-use lifecycle; Python cannot guarantee secret zeroization."""

    def __init__(self, r, s, binding):
        self.r, self.s, self.binding = r, s, binding
        self.live = True

    def cancel(self):
        self.live = False
        self.r.fill(0)
        self.s.fill(0)

    def finish(self, reply, binding):
        import numpy as np
        if not self.live:
            raise ValueError("material already burned")
        try:
            if binding != self.binding or reply.shape != self.s.shape or reply.dtype != np.uint32:
                raise ValueError("malformed or wrong-binding response")
            return ((reply.astype(np.uint64) + self.s) & np.uint64(MASK)).astype(np.uint32)
        finally:
            self.cancel()


def dense_project(weights, r):
    import numpy as np
    # Width <=16384: signed-i8 times uint32 sum fits signed-i64.
    return ((weights.astype(np.int64) @ r.astype(np.int64)) & MASK).astype(np.uint32)


def native_projector(weights, mode="native_wrap32"):
    """Existing SIMD exact-wrap32 Preparation kernel; never permit fallback/build."""
    from pllm.runtime.native import MaskedGEMM, capabilities
    executor = MaskedGEMM(threads=4, build=False, simd=True)
    if not executor.native:
        raise RuntimeError("installed native wrap32 kernel required; no fallback or build")
    matrix = executor.compile(weights)
    if mode == "native_digit8":
        if not capabilities().get("digit8_cpu"):
            raise RuntimeError("native exact digit8 unavailable")
        return lambda r: matrix.digit8(r.reshape(1, -1), 32)[0], executor.library_path
    return lambda r: matrix.wrap32(r.reshape(1, -1))[0], executor.library_path


def freeze_sources(extra_paths=()):
    from pllm.runtime import coded_resources, native, _native_support
    paths = [ROOT / name for name in (
        "scripts/probe_correlation_compiler.py", "scripts/frontier_probe_support.py",
        "tests/test_correlation_compiler_probe.py", "benchmarks/research/paper_baseline.py",
        "benchmarks/research/common.py")]
    # Bind actual imported installed modules, not similarly named checkout files.
    paths += [Path(coded_resources.__file__), Path(native.__file__), Path(_native_support.__file__), *map(Path, extra_paths)]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def assert_sources_unchanged(frozen):
    for path, digest in frozen.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"cohort source changed: {path}")


def issue(projector, rows, columns, binding):
    import numpy as np
    # Only test dealer/client see either seed or either mask. Inference receives c.
    r = expand(secrets.token_bytes(32), b"input-mask", binding, columns)
    s = expand(secrets.token_bytes(32), b"output-mask", binding, rows)
    c = ((projector(r).astype(np.uint64) - s.astype(np.uint64)) & np.uint64(MASK)).astype(np.uint32)
    return ClientMaterial(r, s, binding), c


def oracle_check(weights, projector, samples=3):
    import numpy as np
    outputs = 0
    for _ in range(samples):
        binding = secrets.token_bytes(32)
        material, correction = issue(projector, *weights.shape, binding)
        x = np.arange(weights.shape[1], dtype=np.uint32) * np.uint32(101)
        masked = ((x.astype(np.uint64) - material.r) & np.uint64(MASK)).astype(np.uint32)
        reply = ((dense_project(weights, masked).astype(np.uint64) + correction) & np.uint64(MASK)).astype(np.uint32)
        result = material.finish(reply, binding)
        expected = [sum(int(w) * int(v) for w, v in zip(row, x)) & MASK for row in weights]
        if result.tolist() != expected:
            raise AssertionError("independent integer oracle mismatch")
        outputs += len(expected)
    return outputs


def worker(payload):
    import numpy as np
    if payload.get("mode") not in MODES or not 1 <= payload.get("samples", 0) <= 16:
        raise ValueError("invalid bounded worker request")
    assert_sources_unchanged(payload.get("frozen_sources", {}))
    rows, columns, block = (payload[key] for key in ("rows", "columns", "block"))
    cost = preflight(rows, columns, block)
    if cost["status"] != "admitted_geometry":
        return cost
    mode_estimate = cost["estimated_process_bytes"] if payload["mode"] == "subset" else cost[
        "digit8_control_estimated_process_bytes" if payload["mode"] == "native_digit8" else "native_control_estimated_process_bytes"]
    if mode_estimate > BUDGET:
        raise MemoryError("native control resource admission failed before weights")
    fixture_start, fixture_cpu = time.perf_counter(), time.process_time()
    weights = np.random.default_rng(20261011).integers(-128, 128, (rows, columns), dtype=np.int16).astype(np.int8)
    fixture_elapsed, fixture_cpu = time.perf_counter() - fixture_start, time.process_time() - fixture_cpu
    start, setup_cpu = time.perf_counter(), time.process_time()
    native_path = None
    if payload["mode"] == "subset":
        compiler = SubsetCompiler(weights, block)
        projector = compiler.project
    else:
        projector, native_path = native_projector(weights, payload["mode"])
    setup = time.perf_counter() - start
    setup_cpu = time.process_time() - setup_cpu
    oracle_outputs = oracle_check(weights, projector, 1)
    observations = []
    for _ in range(payload["samples"]):
        start, cpu = time.perf_counter(), time.process_time()
        material, correction = issue(projector, rows, columns, secrets.token_bytes(32))
        observations.append({"issuance_elapsed_s": time.perf_counter() - start,
                             "issuance_cpu_s": time.process_time() - cpu})
        material.cancel()
        correction.fill(0)
    import resource
    import os
    assert_sources_unchanged(payload.get("frozen_sources", {}))
    return cost | {"mode": payload["mode"], "setup_elapsed_s": setup,
                   "synthetic_fixture_setup_elapsed_s": fixture_elapsed,
                   "synthetic_fixture_setup_cpu_s": fixture_cpu,
                   "measured_mode_estimated_process_bytes": mode_estimate,
                   "worker_pid": os.getpid(), "python_version": sys.version,
                   "numpy_version": np.__version__,
                   "process_repetition": payload.get("process_repetition"),
                   "execution_order": payload.get("execution_order"),
                   "native_library_path": str(native_path) if native_path else None,
                   "native_simd": True if native_path else None,
                   "native_threads": 4 if native_path else None,
                   "setup_cpu_s": setup_cpu,
                   "observations": observations, "oracle_outputs": oracle_outputs,
                   "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                       (1 if sys.platform == "darwin" else 1024),
                   "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
                   "numeric_contract": "synthetic signed-i8 weights; exact ring32"}


def geometry(config_path):
    from pllm import lower_model
    from pllm.runtime.coded_resources import coded_delegation_storage_lower_bound
    from benchmarks.research.paper_baseline import experiment
    source_lock = model = None
    if config_path is None:
        from pllm.model_loader import resolve_model
        resolved = resolve_model(experiment().pipeline.model)
        config_path = resolved.path / "config.json"
        source_lock = resolved.source_lock_digest
        model = experiment().pipeline.model.to_spec()
    data = config_path.read_bytes()
    plan = lower_model(json.loads(data), batch=1, max_input_tokens=64, max_new_tokens=8)
    storage = coded_delegation_storage_lower_bound(plan, experiment().pipeline,
        resident_budget_bytes=2 << 30, cached_budget_bytes=6 << 30)
    shapes = Counter((row["output_width"], row["input_width"]) for row in storage["stage_dimensions"])
    shape_rows = [preflight(m, n, b) | {"compiled_stage_count": count}
                  for (m, n), count in sorted(shapes.items()) for b in (4, 8, 12)]
    return {"config_sha256": hashlib.sha256(data).hexdigest(), "plan_digest": plan.digest,
            "source_lock_digest": source_lock, "default_fixture_model": model,
            "schedule_digest": storage["schedule_digest"],
            "all_stage_retained_table_bytes_by_block": {
                str(b): sum(r["table_bytes"] * r["compiled_stage_count"] for r in shape_rows if r["block"] == b)
                for b in (4, 8, 12)},
            "all_stage_setup_lane_additions_by_block": {
                str(b): sum(r["setup_subset_vector_additions"] * r["compiled_stage_count"] for r in shape_rows if r["block"] == b)
                for b in (4, 8, 12)}, "shapes": shape_rows}


def paired_horizons(measurements):
    import math
    import statistics
    results = []
    cells = {}
    for row in measurements:
        key = (row["rows"], row["columns"], row["block"])
        cells.setdefault(key, {}).setdefault(row["mode"], []).append(row)
    for groups, control_mode in ((groups, mode) for groups in cells.values()
                                 for mode in groups if mode != "subset"):
        controls, candidates = groups[control_mode], groups["subset"]
        if len(controls) < 3 or len(controls) != len(candidates):
            raise ValueError("at least three fresh matched processes required")
        control, candidate = controls[0], candidates[0]
        keys = ("rows", "columns", "block", "weights_sha256")
        if any(row[k] != control[k] for row in controls + candidates for k in keys):
            raise ValueError("unmatched correlation cohort")
        dense = statistics.median(statistics.median(r["issuance_elapsed_s"] for r in row["observations"]) for row in controls)
        subset = statistics.median(statistics.median(r["issuance_elapsed_s"] for r in row["observations"]) for row in candidates)
        setup_delta = statistics.median(r["setup_elapsed_s"] for r in candidates) - statistics.median(r["setup_elapsed_s"] for r in controls)
        results.append({k: control[k] for k in keys} | {
            "control": control_mode, "fresh_processes_per_mode": len(controls),
            "native_median_issuance_elapsed_s": dense, "subset_median_issuance_elapsed_s": subset,
            "empirical_elapsed_amortization_samples": max(1, math.ceil(max(0, setup_delta) / (dense - subset)))
                if subset < dense else None,
            "empirical_horizon_scope": "component median projection, not measured multi-request throughput"})
    return results


def timing_selection(shape_rows):
    """One low-storage table cell per shape; other tradeoffs remain explicit gates."""
    gates = []
    for row in shape_rows:
        reason = "retained_low_storage_cell" if row["block"] == 4 else (
            "pruned_no_arithmetic_crossover_and_larger_setup_tables: query additions trade off; no timing dominance claimed")
        gates.append(row | {"timing_gate": reason if row["status"] == "admitted_geometry" else "resource_preflight_rejection",
                            "timing_retained": row["block"] == 4 and row["status"] == "admitted_geometry"})
    return gates


def cohort_jobs(cases, samples, repetitions=3, modes=("native_wrap32", "subset")):
    if repetitions < 3:
        raise ValueError("at least three fresh processes per mode required")
    for cell, (rows, columns, block) in enumerate(cases):
        for repetition in range(repetitions):
            # Two modes alternate AB/BA; three use balanced Latin rotations.
            base = modes if cell % 2 == 0 else modes[::-1]
            rotation = repetition % len(base)
            order = base[rotation:] + base[:rotation]
            for position, mode in enumerate(order):
                yield dict(rows=rows, columns=columns, block=block, mode=mode, samples=samples,
                           process_repetition=repetition, execution_order=position)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--processes", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if not args.output or not 1 <= args.samples <= 16 or not 3 <= args.processes <= 6:
        parser.error("--output, 1..16 samples and 3..6 processes required")
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    sys.path.insert(0, str(ROOT))
    import os
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    report = {"schema": "pllm.correlation_compiler_probe.v1", "geometry": geometry(args.config),
              "measurements": [], "checkpoint_values_loaded": False, "production_admitted": False,
              "security": "trusted Preparation, SHAKE256 PRG assumption; test-local paired ownership only",
              "whole_response_metrics": None}
    report["timing_gates"] = timing_selection(report["geometry"]["shapes"])
    from pllm.runtime.native import extension, capabilities
    module = extension()
    if args.serial and module is None:
        raise RuntimeError("installed native extension absent; serial cohort blocked")
    frozen = freeze_sources([module.__file__] if module else [])
    report["frozen_source_sha256"] = frozen
    modes = MODES if capabilities().get("digit8_cpu") else ("native_wrap32", "subset")
    report["measured_control_modes"] = [mode for mode in modes if mode != "subset"]
    report["native_digit8_gate"] = "admitted" if "native_digit8" in modes else "capability_unavailable"
    if args.serial:
        from scripts.frontier_probe_support import fresh_workers
        cases = [(64, 64, 4)]
        cases += [(r["rows"], r["columns"], r["block"]) for r in report["timing_gates"]
                  if r["timing_retained"]]
        with fresh_workers(Path(__file__), memory_bytes=512 << 20, timeout=240) as (run, guard, host):
            report["build"] = run.build
            for payload in cohort_jobs(cases, args.samples, args.processes, modes):
                report["measurements"].append(run(payload | {"frozen_sources": frozen}))
            report["maximum_swap_growth_bytes"] = guard.maximum_swap_growth_bytes
        report["paired_horizons"] = paired_horizons(report["measurements"])
    assert_sources_unchanged(frozen)
    report["source_freeze_verified_after_cohort"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
