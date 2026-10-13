"""Public histogram threshold certificate; default never imports model/runtime.

Bounds describe a specified raw-i8 artifact layout, not every possible codec.
Optional source scan is parent-only, guarded, offline and never serving.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/research-narrow-ingress.json"
SOURCE_SHA = "8c55c563422a38ddc0c6d0187379c38450b0702b4031912166e7f4bd4c7e105d"
FRAME = 64
MAP_FRAME = 16
HEADER = 64
CLIENT_LIMIT = 256 << 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities():
    paths = [SOURCE, Path(__file__), ROOT / "tests/test_column_residue_placement_probe.py",
             ROOT / "docs/research/column-residue-placement-reference.md",
             ROOT / "scripts/network_algebra_hypotheses.py",
             ROOT / "scripts/decoder_probe_support.py",
             ROOT / "scripts/frontier_probe_support.py",
             ROOT / "python/pllm/roles/__init__.py",
             ROOT / "crates/pllm-core/src/offset_transport.rs"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def width(s):
    if s < 0:
        raise ValueError("negative coefficient sum")
    return max(1, (127 * s).bit_length() + 1)


def minimum_sum(bits):
    """Smallest integer S consistent with compiler h=bit_length(127S)+1."""
    if bits == 1:
        return 0
    if bits < 1:
        raise ValueError("invalid width")
    return ((1 << (bits - 2)) + 126) // 127


def centered(value, bits):
    ring = 1 << bits
    return (value + ring // 2) % ring - ring // 2


def split_product(weights, x, selected, r, s):
    """Toy one-use delegation: exact remote integer, then exact local integer.

    Input contract [-127,127], weight contract signed-i8 (including -128).
    No float partials, bias or scales; caller applies original dequantization once.
    """
    n, m = len(weights), len(x)
    if not n or not m or any(len(row) != m for row in weights):
        raise ValueError("invalid matrix")
    if any(not -127 <= v <= 127 for v in x):
        raise ValueError("outside original activation domain")
    if any(not -128 <= v <= 127 for row in weights for v in row):
        raise ValueError("outside i8 weight domain")
    if len(set(selected)) != len(selected) or any(i < 0 or i >= m for i in selected):
        raise ValueError("invalid columns")
    if len(r) != m or len(s) != n:
        raise ValueError("invalid masks")
    remote = [i for i in range(m) if i not in selected]
    widths = [width(sum(abs(row[i]) for i in remote)) for row in weights]
    request_bits = max(widths)
    a = {i: (x[i] - r[i]) % (1 << request_bits) for i in remote}
    out = []
    for j, row in enumerate(weights):
        correction = sum(row[i] * r[i] for i in remote) - s[j]
        response = sum(row[i] * a[i] for i in remote) + correction
        remote_integer = centered(response + s[j], widths[j])
        local_integer = sum(row[i] * x[i] for i in selected)
        out.append(remote_integer + local_integer)
    return out


def baseline(stage):
    n, m = stage["shape"]
    hist = {int(h): count for h, count in stage["row_residue_width_histogram"].items()}
    if sum(hist.values()) != n or max(hist) != stage["stage_bits"]:
        raise ValueError("invalid histogram")
    output = (sum(h * c for h, c in hist.items()) + 7) // 8
    return (m * max(hist) + 7) // 8 + 2 * output + 3 * FRAME


def option(stage, k, rows, cold=True):
    """Optimistic per-stage savings. Row crossings can be mutually impossible.

    Removing k i8 coefficients removes at most 128k absolute mass per row.
    Raw delivery, map and modeled frames are charged even in this relaxation.
    """
    n, m = stage["shape"]
    if not 0 <= k <= m or rows < 0:
        raise ValueError("invalid placement")
    if k == 0:
        return {"columns": 0, "macs_per_row": 0, "delivery_bytes": 0,
                "saving_per_row_upper": 0, "net_upper": 0}
    full = k == m
    hist = {int(h): c for h, c in stage["row_residue_width_histogram"].items()}
    lower = {h: width(max(0, minimum_sum(h) - 128 * k)) for h in hist}
    remaining = 0 if full else (
        ((m - k) * max(lower.values()) + 7) // 8
        + 2 * ((sum(lower[h] * c for h, c in hist.items()) + 7) // 8)
        + 3 * (FRAME + MAP_FRAME))
    delivery = n * k + HEADER + (0 if full else 4 * k)
    saving = baseline(stage) - remaining
    return {"columns": k, "macs_per_row": n * k, "delivery_bytes": delivery,
            "selected_file_bytes": delivery, "selected_native_weight_bytes": n * k,
            "remote_request_bits_lower": 0 if full else max(lower.values()),
            "remote_output_bits_sum_lower": 0 if full else sum(lower[h] * c for h, c in hist.items()),
            "saving_per_row_upper": saving,
            "net_upper": rows * saving - (delivery if cold else 0)}


def group_optimum(stages, rows, budget, cold=True):
    """Exact 0/1 semantic-group optimum for specified identical raw layout."""
    costs = [math.prod(s["shape"]) for s in stages]
    unit = math.gcd(*costs)
    dp = {0: (0, ())}
    for j, (stage, cost) in enumerate(zip(stages, costs, strict=True)):
        gain = option(stage, stage["shape"][1], rows, cold)["net_upper"]
        for used, (value, selected) in list(dp.items()):
            new = used + cost // unit
            candidate = (value + gain, selected + (j,))
            if new * unit <= budget and candidate[0] > dp.get(new, (-1, ()))[0]:
                dp[new] = candidate
    used, (gain, selected) = max(dp.items(), key=lambda kv: (kv[1][0], -kv[0]))
    delivery = sum(option(stages[i], stages[i]["shape"][1], rows, cold)["delivery_bytes"]
                   for i in selected)
    return {"net_saving": gain, "macs_per_row": used * unit,
            "delivery_bytes": delivery, "source_stage_indices": list(selected),
            "optimal_for": "all layer/semantic groups; raw i8 artifacts; modeled frames"}


def certificate(stages, rows, budget, cold=True):
    """Integer upper bound on every column placement under common MAC budget.

    min(sum stage maxima, B*maximum benefit/cost) relaxes simultaneous row
    crossings, stage choices and integral allocation. Never a feasible placement.
    """
    summed, ratio = 0, Fraction()
    for stage in stages:
        n, m = stage["shape"]
        maximum = 0
        for k in range(1, min(m, budget // n) + 1):
            candidate = option(stage, k, rows, cold)
            gain = candidate["net_upper"]
            maximum = max(maximum, gain)
            ratio = max(ratio, Fraction(gain, n * k))
        summed += maximum
    upper = min(summed, (ratio.numerator * budget) // ratio.denominator)
    group = group_optimum(stages, rows, budget, cold)
    return {"executed_rows_each_stage": rows, "column_net_saving_upper": upper,
            "additional_saving_over_group_upper": upper - group["net_saving"],
            "whole_group": group, "unplaced_bytes": rows * sum(map(baseline, stages)),
            "classification": "rejected" if upper <= group["net_saving"] else "unresolved optimistic ceiling"}


def exact_toy_optimum(weights, rows, budget):
    """Exhaustive public-column optimizer, bounded fixtures only."""
    n, m = len(weights), len(weights[0])
    if m > 12:
        raise ValueError("toy enumeration limited to 12 columns")
    widths = [width(sum(map(abs, row))) for row in weights]
    stage = {"shape": [n, m], "stage_bits": max(widths),
             "row_residue_width_histogram": {str(h): widths.count(h) for h in set(widths)}}
    best = (0, ())
    for k in range(1, min(m, budget // n) + 1):
        for selected in itertools.combinations(range(m), k):
            remote_widths = [width(sum(abs(w) for i, w in enumerate(row) if i not in selected))
                             for row in weights]
            remaining = 0 if k == m else (((m - k) * max(remote_widths) + 7) // 8
                         + 2 * ((sum(remote_widths) + 7) // 8) + 3 * (FRAME + MAP_FRAME))
            delivery = n * k + HEADER + (0 if k == m else 4 * k)
            gain = rows * (baseline(stage) - remaining) - delivery
            if gain > best[0]:
                best = (gain, selected)
    return {"net_saving": best[0], "columns": list(best[1])}


def checks():
    count = 0
    weights = [[-128, 0, 127], [1, -2, 3], [0, 0, 0]]
    for x in itertools.product((-127, -1, 0, 1, 127), repeat=3):
        oracle = [sum(w * v for w, v in zip(row, x, strict=True)) for row in weights]
        for mask in range(8):
            selected = [i for i in range(3) if mask & (1 << i)]
            actual = split_product(weights, x, selected, [991, -719, 8192], [317, -91, 0])
            if actual != oracle:
                raise AssertionError("independent integer oracle mismatch")
            count += len(weights)
    return {"independent_integer_outputs": count, "activation_endpoints": [-127, 127],
            "weight_endpoints": [-128, 127], "all_column_subsets": True,
            "float_partial_outputs_added": False, "protected_protocol_executed": False}


def source_scan():
    """Only opt-in fresh child imports checkpoint helper; no concurrent launch."""
    import numpy as np
    from decoder_probe_support import DecoderFixture
    fixture = DecoderFixture(inputs=150, outputs=8)
    saved = {s["stage"]: s for s in json.loads(SOURCE.read_text())["checkpoint_geometry"]["stages"]}
    result = []
    for name, stage in fixture.body.items():
        w = stage.weight.values
        if stage.weight_digest != saved[name]["weight_sha256"]:
            raise ValueError("source weight mismatch")
        # Fixed public rule. No activation trace; even layers calibration, odd held out.
        score = np.abs(w.astype(np.int16)).sum(axis=0, dtype=np.int64)
        order = sorted(range(w.shape[1]), key=lambda i: (-int(score[i]), i))
        sums = np.abs(w.astype(np.int16)).sum(axis=1, dtype=np.int64)
        original = [width(int(v)) for v in sums]
        hist = {str(h): original.count(h) for h in set(original)}
        if hist != saved[name]["row_residue_width_histogram"]:
            raise ValueError("source compiler histogram mismatch")
        points = []
        sampled_weights = w[[0, w.shape[0] // 2, w.shape[0] - 1]].tolist()
        oracle_outputs = 0
        for k in (1, max(1, w.shape[1] // 100), max(1, w.shape[1] // 20), w.shape[1] // 10):
            remaining = sums.copy()
            for i in order[:k]:
                remaining -= np.abs(w[:, i].astype(np.int16))
            actual = [width(int(v)) for v in remaining]
            bound = option(saved[name], k, 1)
            if sum(actual) < bound["remote_output_bits_sum_lower"] or max(actual) < bound["remote_request_bits_lower"]:
                raise AssertionError("certificate unsound")
            for pattern in range(3):
                x = [(-127, 0, 127)[(i + pattern) % 3] for i in range(w.shape[1])]
                oracle = [sum(a * b for a, b in zip(row, x, strict=True)) for row in sampled_weights]
                reconstructed = split_product(sampled_weights, x, order[:k],
                                              [991 + 19 * i for i in range(len(x))], [17, 201, 999])
                if reconstructed != oracle:
                    raise AssertionError("held-out source integer oracle mismatch")
                oracle_outputs += len(oracle)
            points.append({"columns": k, "remote_bits_sum": sum(actual), "request_bits": max(actual),
                           "selected_column_sha256": hashlib.sha256(json.dumps(order[:k]).encode()).hexdigest()})
        layer = int(name.split(".")[1])
        result.append({"stage": name, "weight_sha256": stage.weight_digest,
                       "split": "calibration" if layer % 2 == 0 else "held_out", "points": points,
                       "independent_integer_outputs": oracle_outputs})
    return {"lock": fixture.lock(), "stages": result, "selection_rule": "descending public column L1; index tie break"}


def report():
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("histogram evidence source lock mismatch")
    source = json.loads(SOURCE.read_text())["checkpoint_geometry"]
    stages = source["stages"]
    if len(stages) != 96 or sum(s["shape"][0] for s in stages) != 304128:
        raise ValueError("incomplete geometry")
    total = sum(math.prod(s["shape"]) for s in stages)
    budget = total // 10
    # Both file + i8 native copies fit this optimistic artifact-only envelope.
    # Existing client allocation, page cache and scratch are deliberately unknown.
    worst_map_bytes = 4 * sum(s["shape"][1] for s in stages)
    if 2 * budget + worst_map_bytes + len(stages) * HEADER > CLIENT_LIMIT:
        raise ValueError("artifact-only client envelope exceeds budget")
    comparisons = []
    group_recipes = {}
    for horizon in (1, 4, 16, 64):
        for cold in (True, False):
            phases = {phase: certificate(stages, rows * horizon, budget, cold)
                      for phase, rows in (("prefill", 150), ("decode", 7), ("combined", 157))}
            for phase in phases.values():
                group = phase["whole_group"]
                selected = group.pop("source_stage_indices")
                key = hashlib.sha256(json.dumps(selected).encode()).hexdigest()
                group_recipes[key] = selected
                group["recipe_sha256"] = key
            comparisons.append({"horizon": horizon, "initial_cache": "empty" if cold else "same selected artifacts preowned; historical delivery excluded",
                                "phases": phases})
    ingress_counts = {}
    best_ingress_only = None
    for s in stages:
        n = s["shape"][0]
        b = s["stage_bits"]
        key = (n, b, (8 * n + b - 1) // b)
        ingress_counts[key] = ingress_counts.get(key, 0) + 1
        m = s["shape"][1]
        for k in range(1, min(m, budget // n) + 1):
            ingress_saving = 157 * (((m * b + 7) // 8) - (((m - k) * b + 7) // 8))
            delivery = n * k + HEADER + (0 if k == m else 4 * k)
            net = ingress_saving - delivery
            if best_ingress_only is None or net > best_ingress_only["net_bytes"]:
                best_ingress_only = {"stage": s["stage"], "columns": k, "net_bytes": net,
                                     "executed_rows": 157,
                                     "scope": "coordinate removal only, original request width and correction/reply/frame costs fixed"}
    ingress = [{"output_rows": n, "request_bits": b, "raw_rows_break_even": r, "stages": count}
               for (n, b, r), count in sorted(ingress_counts.items())]
    threshold_break_even = []
    for s in stages:
        for k in (1, max(1, s["shape"][1] // 10)):
            candidate = option(s, k, 1)
            saving = candidate["saving_per_row_upper"]
            threshold_break_even.append({"stage": s["stage"], "columns": k,
                                        "delivery_bytes": candidate["delivery_bytes"],
                                        "saving_per_row_upper": saving,
                                        "optimistic_rows_break_even_lower": None if saving <= 0 else
                                        (candidate["delivery_bytes"] + saving - 1) // saving})
    best_thresholds = sorted((point for point in threshold_break_even
                              if point["optimistic_rows_break_even_lower"] is not None),
                             key=lambda point: (point["optimistic_rows_break_even_lower"], point["stage"], point["columns"]))[:4]
    return {"schema": "pllm.column_residue_placement.v1", "scope": "optimistic raw-layout histogram certificate, not measured all-link traffic",
            "source_lock": source["lock"], "identities": identities(), "checks": checks(),
            "body_macs_per_executed_row": total, "local_macs_per_row_limit": budget,
            "client_limit_bytes": CLIENT_LIMIT, "selected_artifact_limit_bytes": CLIENT_LIMIT,
            "whole_client_peak_rss_bytes": None, "whole_client_cpu_seconds": None,
            "native_scratch_bytes": None, "full_wire_bytes": None,
            "frame_model": {"bytes_per_edge_row": FRAME, "column_map_binding_extra_per_edge_row": MAP_FRAME,
                            "artifact_header_bytes": HEADER, "column_map_bytes_per_column": 4},
            "original_scale_bias_delivery": "retained common client bundle, cancels; no free new coefficients",
            "ownership": "raw selected file + native i8 coefficients; file cache/page/scratch/full-client peak unknown",
            "comparisons": comparisons, "ingress_only_break_even": ingress,
            "group_recipe_source_stage_indices": group_recipes,
            "best_nonempty_cold_ingress_only": best_ingress_only,
            "best_optimistic_threshold_break_even": best_thresholds,
            "threshold_break_even_scope": "best four of all stages at one column and floor(input_columns/10); lower bound, not achievable repayment",
            "held_out_source_column_validation": None,
            "decision": "screen only: histogram ceiling cannot identify columns; source scan and exact matched serving gate remain"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scan-source", action="store_true", help="parent-only serial fresh worker; never default")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        request = json.load(sys.stdin)
        if request["identities"] != identities():
            raise ValueError("worker source drift")
        result = source_scan()
        if request["identities"] != identities():
            raise ValueError("worker source drift")
        print(json.dumps(result))
        return
    if args.output is None or args.output.exists():
        raise ValueError("provide fresh --output path")
    before = identities()
    result = report()
    if args.scan_source:
        from frontier_probe_support import fresh_workers
        with fresh_workers(__file__, memory_bytes=6 << 30, timeout=600) as (run, guard, host):
            result["held_out_source_column_validation"] = run({"identities": before})
            result["fresh_worker_build"] = run.build
    if identities() != before:
        raise ValueError("parent source drift")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
