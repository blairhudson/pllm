"""Bounded public RAA cartridge algebra; no performance or privacy admission."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[1]
FIELD = 2_013_265_921
MAX_ATTEMPTS = 65_536
MAX_LIVE = 8
LOW = 268_435_456


class PublicSampler:
    """Exact SHA-256/rejection public sampler from coded_linear.rs."""

    def __init__(self, seed):
        self.seed, self.counter = seed, 0

    def below(self, upper):
        limit = (1 << 32) - (1 << 32) % upper
        while True:
            blob = hashlib.sha256(b"pllm.public-raa-code.v1" + self.seed
                                  + self.counter.to_bytes(8, "little")).digest()
            self.counter += 1
            value = int.from_bytes(blob[:4], "little")
            if value < limit:
                return value % upper


class Code:
    def __init__(self, n, seed):
        if not 1 <= n <= 16384 or len(seed) != 32:
            raise ValueError("bounded code geometry/seed required")
        self.n, self.length = n, 8 * n
        rng = PublicSampler(seed)
        self.p1, self.p2 = [], []
        for target in (self.p1, self.p2):
            target.extend(range(self.length))
            for i in range(self.length - 1, 0, -1):
                j = rng.below(i + 1)
                target[i], target[j] = target[j], target[i]
        pairs = [(rng.below(FIELD - 1) + 1, rng.below(FIELD - 1) + 1)
                 for _ in range(self.length)]
        self.s1, self.s2 = map(list, zip(*pairs))

    def accumulators(self, row):
        if len(row) != self.n:
            raise ValueError("row shape")
        first, second, value = [], [], 0
        for j in range(self.length):
            value = (value + row[self.p1[j] // 8] * self.s1[j]) % FIELD
            first.append(value)
        value = 0
        for j in range(self.length):
            value = (value + first[self.p2[j]] * self.s2[j]) % FIELD
            second.append(value)
        return first, second

    def encode(self, row):
        return self.accumulators(row)[1]

    def syndrome(self, sparse):
        if not sparse or len({j for j, _ in sparse}) != len(sparse):
            raise ValueError("distinct nonempty support required")
        current = [0] * self.length
        for j, value in sparse:
            if not 0 <= j < self.length or not 0 < value < FIELD:
                raise ValueError("noncanonical sparse entry")
            current[j] = value
        total, first = 0, [0] * self.length
        for j in reversed(range(self.length)):
            total = (total + current[j]) % FIELD
            first[self.p2[j]] = total * self.s2[j] % FIELD
        total, result = 0, [0] * self.n
        for j in reversed(range(self.length)):
            total = (total + first[j]) % FIELD
            k = self.p1[j] // 8
            result[k] = (result[k] + total * self.s1[j]) % FIELD
        return result


class CheckpointCartridge:
    """Public W plus both accumulator boundary states. No query tape."""

    def __init__(self, weights, code, stride):
        if stride < 1 or not weights or any(len(row) != code.n for row in weights):
            raise ValueError("checkpoint geometry")
        self.weights, self.code, self.stride = weights, code, stride
        self.a, self.b = [], []
        self.boundaries = list(range(stride - 1, code.length - 1, stride))
        for row in weights:
            first, second = code.accumulators(row)
            self.a.append([first[j] for j in self.boundaries])
            self.b.append([second[j] for j in self.boundaries])

    def selected(self, positions):
        """Union reuse is public, client-local; no support sent to evaluator."""
        positions = sorted(set(positions))
        if any(not 0 <= j < self.code.length for j in positions):
            raise ValueError("column index")
        count, columns = Counter(), {j: [] for j in positions}
        d, code = self.stride, self.code
        for r, row in enumerate(self.weights):
            a_cache = {}
            for j in positions:
                block = j // d
                value = self.b[r][block - 1] if block else 0
                for k in range(block * d, j + 1):
                    index = code.p2[k]
                    if index not in a_cache:
                        ablock = index // d
                        avalue = self.a[r][ablock - 1] if ablock else 0
                        for h in range(ablock * d, index + 1):
                            avalue = (avalue + row[code.p1[h] // 8] * code.s1[h]) % FIELD
                            count["first_mul_add_lanes"] += 1
                        a_cache[index] = avalue
                    else:
                        count["first_cache_hits"] += 1
                    value = (value + a_cache[index] * code.s2[k]) % FIELD
                    count["second_mul_add_lanes"] += 1
                columns[j].append(value)
        count["selected_columns"] = len(positions)
        return columns, dict(count)


def fresh_sparse(code, weight):
    if not 1 <= weight <= code.length:
        raise ValueError("sparse weight")
    # Uniform support without replacement; fresh OS randomness, never public seed.
    pool = list(range(code.length))
    result = []
    for k in range(weight):
        j = k + secrets.randbelow(code.length - k)
        pool[k], pool[j] = pool[j], pool[k]
        result.append((pool[k], secrets.randbelow(FIELD - 1) + 1))
    return result


def dot(row, vector):
    return sum(w * x for w, x in zip(row, vector)) % FIELD


def exact_oracle(weights, x):
    return [sum(int(w) * int(v) for w, v in zip(row, x)) for row in weights]


def heuristic_weight(n):
    return math.ceil(2 * math.log(2) * (128 - math.log2(8 * n)))


def horizon(cold, recurring, bootstrap, saving, overhead, target):
    """Smallest integer R: (R*(S-H)-B)/(C0+R*T0) >= target."""
    target = Fraction(str(target))
    gap = Fraction(saving) - Fraction(str(overhead)) - target * recurring
    if gap <= 0:
        return None
    return max(1, math.ceil((bootstrap + target * cold) / gap))


def geometry(m, n, stride):
    length, t = 8 * n, heuristic_weight(n)
    boundaries = (length - 1) // stride
    return {
        "rows": m, "columns": n, "stride": stride, "privacy_weight_heuristic": t,
        "weight_bytes": m * n, "p_only_bytes": 32 * m * n,
        "checkpoint_bytes": 8 * m * boundaries,
        "bootstrap_bytes": m * n + 8 * m * boundaries + 96,
        "setup_mul_add_lanes": 16 * m * n,
        "per_query_reconstruction_mul_add_upper": m * t * (stride * stride + stride),
        "per_query_sparse_correction_mul_add_lanes": m * t,
        "per_query_syndrome_mul_lanes": 16 * n,
        "per_query_syndrome_add_upper": 24 * n,
        "reference_pq_payload_passes": 64 * m * n + 192 * (m + n) + 32 * (m + n) <= LOW,
        "p_only_stage_payload_exceeds_low": 32 * m * n > LOW,
    }


def run_probe():
    cases, totals = [], Counter()
    for m, n in ((5, 11), (7, 17), (8, 64)):
        code = Code(n, bytes([n]) * 32)
        weights = [[((r * n + j) * 31 + r * 7) % 256 - 128 for j in range(n)]
                   for r in range(m)]
        # Independent dense generator from basis encodings, not syndrome/DAG logic.
        generator = [code.encode([int(k == j) for k in range(n)]) for j in range(n)]
        full = [code.encode(row) for row in weights]
        queries = []
        weight = min(code.length, heuristic_weight(n))
        for sample in range(8):
            sparse = fresh_sparse(code, weight)
            mask = code.syndrome(sparse)
            oracle_mask = [sum(generator[k][j] * v for j, v in sparse) % FIELD
                           for k in range(n)]
            assert mask == oracle_mask
            x = [-128 if sample == 0 else 127 if sample == 1 else
                 ((j * 17 + sample * 29) % 256 - 128) for j in range(n)]
            masked = [(v + r) % FIELD for v, r in zip(x, mask)]
            correction = [sum(row[j] * v for j, v in sparse) % FIELD for row in full]
            assert correction == [dot(row, oracle_mask) for row in weights]
            reply = [dot(row, masked) for row in weights]
            decoded = [(y - c) % FIELD for y, c in zip(reply, correction)]
            centered = [v - FIELD if v > FIELD // 2 else v for v in decoded]
            assert centered == exact_oracle(weights, x)
            queries.append(sparse)
            totals["fresh_queries"] += 1
            totals["mask_values_checked"] += n
            totals["correction_values_checked"] += m
            totals["signed_output_values_checked"] += m
        strides = []
        for stride in (1, 4, 16, 64):
            cartridge = CheckpointCartridge(weights, code, stride)
            selected, counts = cartridge.selected(range(code.length))
            assert all(selected[j] == [row[j] for row in full] for j in selected)
            totals["checkpoint_column_values_checked"] += m * code.length
            union = {j for sparse in queries for j, _ in sparse}
            batched, batch_counts = cartridge.selected(union)
            for sparse in queries:
                correction = [sum(batched[j][r] * v for j, v in sparse) % FIELD
                              for r in range(m)]
                assert correction == [sum(row[j] * v for j, v in sparse) % FIELD
                                      for row in full]
                totals["batched_correction_values_checked"] += m
            strides.append({"stride": stride, "all_columns_counts": counts,
                            "eight_fresh_support_union_counts": batch_counts,
                            "checkpoint_field_elements": 2 * m * len(cartridge.boundaries)})
        cases.append({"rows": m, "columns": n, "fresh_queries": 8,
                      "sparse_weight": weight, "strides": strides})
    return {"scope": "stdlib synthetic, exact algebra and instrumented loop counts; no timings",
            "totals": dict(totals), "cases": cases}


def symbolic_report():
    source = json.loads((ROOT / "docs/evidence/research-maverick-reference-qwen25.json").read_text())
    stages = source["storage_lower_bound"]["stage_dimensions"]
    shapes = Counter((s["output_width"], s["input_width"]) for s in stages)
    frontier = json.loads((ROOT / "docs/evidence/research-all-link-frontier.json").read_text())
    classes = frontier["control"]["classes"]
    cold = classes["public_bundle"]
    recurring = sum(classes.values()) - cold
    saving = classes["corrections"]
    # Sensitivities only: historical class bodies are not raw 24-bit lane counts.
    overheads = {
        "ideal_no_expansion_no_new_framing": Fraction(0),
        "31bit_ingress_only_output_codec_dependency": Fraction(classes["inputs"] * 7, 24),
        "31bit_ingress_and_output_vs_assumed24": Fraction((classes["inputs"] + classes["outputs"]) * 7, 24),
        "32bit_ingress_and_output_vs_assumed24": Fraction(classes["inputs"] + classes["outputs"], 3),
    }
    candidates = []
    for stride in (1, 4, 16, 64, 256):
        cells = [dict(geometry(m, n, stride), multiplicity=k) for (m, n), k in shapes.items()]
        bootstrap = sum(c["bootstrap_bytes"] * c["multiplicity"] for c in cells)
        candidates.append({"name": f"two_accumulator_checkpoint_d{stride}", "cells": cells,
                           "bootstrap_bytes": bootstrap,
                           "network_sensitivities": network_cases(cold, recurring, bootstrap, saving, overheads)})
    p_bytes = sum(32 * m * n * k for (m, n), k in shapes.items()) + 96 * len(stages)
    candidates.append({"name": "p_only_no_verification", "bootstrap_bytes": p_bytes,
                       "network_sensitivities": network_cases(cold, recurring, p_bytes, saving, overheads)})
    weights = sum(m * n * k for (m, n), k in shapes.items())
    candidates.append({"name": "weights_only_regenerate_per_stage_batch", "bootstrap_bytes": weights + 96 * len(stages),
                       "setup_mul_add_lanes_per_stage_batch": 16 * weights,
                       "network_sensitivities": network_cases(cold, recurring, weights + 96 * len(stages), saving, overheads)})
    candidates.append({"name": "weights_only_dense_mask_correction", "bootstrap_bytes": weights + 96 * len(stages),
                       "matrix_mul_add_lanes_per_query_full_schedule": weights,
                       "mechanism": "Compute W*(G*e) directly per fresh mask, not regenerate P; one dense client matvec per use.",
                       "network_sensitivities": network_cases(cold, recurring, weights + 96 * len(stages), saving, overheads)})
    batching = []
    for (m, n), multiplicity in shapes.items():
        length, t = 8 * n, heuristic_weight(n)
        rows = []
        for h in (1, 8, 64, 158, 512, 4096, 8192, 65536):
            union = length * (1 - (1 - t / length) ** h)
            # Each query's support is sampled without replacement, queries independent.
            rows.append({"batch_queries": h, "expected_distinct_columns": union,
                         "expected_selected_column_bytes": 4 * m * union,
                         "expected_column_delivery_vs_corrections_ratio": union / h,
                         "full_encode_mul_add_lanes_per_query": 16 * m * n / h,
                         "sparse_correction_mul_add_lanes_per_query": m * t,
                         "mask_and_correction_live_payload_bytes": 4 * h * (m + n),
                         "within_native_live_query_limit": h <= MAX_LIVE,
                         "within_native_issuance_limit": h <= MAX_ATTEMPTS})
        batching.append({"rows": m, "columns": n, "multiplicity": multiplicity, "horizons": rows})
    return {"scope": "historical application-body sensitivity, not latest Low control or physical wire",
            "model": frontier["model"], "shape_provenance": "maverick compiler schedule evidence",
            "remote_stages": len(stages), "shape_multiplicities": [list(s) + [k] for s, k in shapes.items()],
            "cold_shared_public_bytes": cold, "warm_recurring_bytes": recurring,
            "arithmetic_subtotal_bytes": classes["corrections"] + classes["outputs"] + classes["inputs"],
            "removable_corrections_upper_bytes": saving, "public_weight_bytes": weights,
            "formula": "control=C0+R*T0; candidate=C0+B+R*(T0-S+H); saving=R*(S-H)-B",
            "warm_assumption": "all public_bundle delivered once to BOTH; controls recur; no correction removability proof",
            "public_redelivery_sensitivity": [
                {"fraction_of_bundle_recurring": f,
                 "shared_cold_bytes": (1 - f) * cold,
                 "recurring_bytes": recurring + f * cold,
                 "ideal_correction_only_ceiling_percent": 100 * saving / (recurring + f * cold)}
                for f in (0, .1, 1)],
            "max_total_overhead_for_positive_finite_horizon_bytes": {
                str(p): saving - p / 100 * recurring for p in (20, 35, 50)},
            "extra_framing_and_challenge_bytes": None, "actual_raw_lane_expansion_bytes": None,
            "output_codec_dependency": "separate prime-field side-information worker; no result assumed",
            "candidates": candidates, "stage_batching": batching}


def network_cases(cold, recurring, bootstrap, saving, overheads):
    result = []
    for name, overhead in overheads.items():
        net = Fraction(saving) - overhead
        horizons = {str(p): horizon(cold, recurring, bootstrap, saving, overhead, p / 100)
                    for p in (20, 35, 50)}
        result.append({"assumption": name, "overhead_bytes_per_pair": float(overhead),
                       "steady_state_reduction_percent": float(100 * net / recurring),
                       "horizons_pairs": horizons,
                       "native_attempt_budget_sensitivity": [
                           {"executed_rows_per_stage_per_pair_assumption": q,
                            "max_pairs_before_attempt_exhaustion": MAX_ATTEMPTS // q,
                            "horizon_within_budget": {p: r is not None and r * q <= MAX_ATTEMPTS
                                                       for p, r in horizons.items()}}
                           for q in (158, 316)],
                       "reduction_percent_at_pairs": {str(r): float(100 * (r * net - bootstrap) / (cold + r * recurring))
                                                       for r in (1, 8, 64, 158, 512, 4096)}})
    return result


SOURCE_PATHS = (
    "scripts/probe_bootstrap_cartridge.py", "tests/test_bootstrap_cartridge_probe.py",
    "docs/research/bootstrap-cartridge-reference.md", "docs/research/maverick-reference.md",
    "docs/research/correlation-compiler-reference.md", "docs/research/secret-replication-reference.md",
    "crates/pllm-core/src/coded_delegation_reference.rs", "crates/pllm-core/src/coded_linear.rs",
    "docs/plans/low-tier-all-link-hypotheses.md", "docs/evidence/research-all-link-frontier.json",
    "docs/evidence/research-maverick-reference-qwen25.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("immutable evidence exists; choose a new output path")
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    report = {
        "schema": "pllm.bootstrap_cartridge_probe.v1", "source_sha256": hashes,
        "measured": run_probe(), "symbolic": symbolic_report(),
        "native_reference_bounds": {"max_dimensions": 16384, "min_delegation_dimensions": 64,
                                    "max_issuance_attempts_including_failures": MAX_ATTEMPTS,
                                    "max_live_queries": MAX_LIVE, "payload_ceiling_bytes": LOW},
        "gates": [
            "Exact synthetic signed outputs pass; checkpoint values and batch corrections match independent dense generator.",
            "P-only intentionally has no verification, Freivalds guarantee, or integrity admission.",
            "BabyBear distance applicability and seeded ensemble/dual-LPN concrete security remain unproved.",
            "Native implementation unchanged: P+Q preflight still rejects both MLP shapes; cartridges are algebra probes only.",
            "No reusable fresh masks: fresh OS support/coefficient samples for every query; only public matrix artifacts reused.",
            "Checkpoint regeneration, disk reads, code expansion, masks, framing, authentication, failures all require charging.",
            "No peak RSS, whole decoder numeric identity, privacy, physical wire, or performance claims.",
            "All public body weights on client disk and support-private disk accesses require new ownership/cache contracts.",
            "Remote evaluator retains all original body MACs symbolically, but added local matrix work is not free.",
            "Stage batching over eight live handles requires a new reviewed lifecycle; decode future inputs cannot be assumed ready.",
            "Historical correction class is only an upper bound on removable linear bytes; nonlinear/other portions need decomposition.",
            "50 percent impossible from corrections alone; 35 percent only survives ideal zero-expansion upper bound; output codec is separate dependency.",
        ],
    }
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != v for p, v in hashes.items()):
        raise RuntimeError("source changed during probe")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report["measured"]["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
