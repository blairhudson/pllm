"""Bounded bit-plane algebra, exact toy views, and conditional payload gates.

No crypto implementation, secure parameters, native execution or RSS measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = {
    "docs/evidence/research-selective-correlation-budget.json":
        "e039f5f864311137b2e48923684059196738314b6d2fec94c9125d721b6a9e97",
    "docs/evidence/compact-setup-fulltext-review.json":
        "4797390ba0ed1d42eebeccb5add20d829b3dc4938a05301dc57b6e99c555f38c",
    "docs/research/correlation-cartridge-frontier-reference.md":
        "d79a55de6b07946d3ed64c199faab9fc90d45a6fbebb1f68aa97dcfba113e52b",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packed(bits):
    if type(bits) is not int or bits < 0:
        raise ValueError("nonnegative integer bit count required")
    return (bits + 7) // 8


def matvec(w, x):
    return [sum(a * b for a, b in zip(row, x, strict=True)) for row in w]


def binary_correction(w, a, u, e):
    """Materialized toy P; real setup/streaming costs remain additional."""
    au = [v % 2 for v in matvec(a, u)]
    low = [v ^ z for v, z in zip(au, e, strict=True)]
    columns = list(zip(*a, strict=True))
    p = [[sum(v * z for v, z in zip(row, col, strict=True)) % 2
          for col in columns] for row in w]
    image = [(v + z) % 2 for v, z in zip(matvec(p, u), matvec(w, e), strict=True)]
    return low, image


def join_correction(w, low, high, q_low, s_high, bits):
    """Trusted dealer retains high computation and sends q_high, not low bits."""
    modulus = 1 << bits
    r = [l + 2 * h for l, h in zip(low, high, strict=True)]
    wr = [v % modulus for v in matvec(w, r)]
    s_low = [(v - q) % 2 for v, q in zip(wr, q_low, strict=True)]
    s = [l + 2 * h for l, h in zip(s_low, s_high, strict=True)]
    q_high = [((v - z) % modulus) // 2 for v, z in zip(wr, s, strict=True)]
    q = [l + 2 * h for l, h in zip(q_low, q_high, strict=True)]
    return r, s, q


def provider_view(x, r, q, bits):
    return tuple((v - z) % (1 << bits) for v, z in zip(x, r, strict=True)) + tuple(q)


def toy_view(x, *, noise="uniform", disclose_low=False, disclose_high_share=False):
    """n=2,k=3,W=[1,-1]. Integer weights enumerate exact probabilities."""
    counts = Counter()
    w = [[1, -1]]
    for u in (0, 1):
        for e in product((0, 1), repeat=2):
            weight = 1 if noise == "uniform" else (3 if e[0] == 0 else 1) * (3 if e[1] == 0 else 1)
            low, _ = binary_correction(w, [[1], [1]], [u], list(e))
            for high in product(range(4), repeat=2):
                for sh, ql in product(range(4), range(2)):
                    r, s, q = join_correction(w, low, high, [ql], [sh], 3)
                    view = provider_view(x, r, q, 3)
                    if disclose_low:
                        view += tuple(low)
                    if disclose_high_share:
                        view += (s[0] // 2,)
                    counts[view] += weight
    return counts


def tv(left, right):
    nl, nr = sum(left.values()), sum(right.values())
    return sum((abs(Fraction(left[k], nl) - Fraction(right[k], nr))
                for k in left.keys() | right.keys()), Fraction()) / 2


def checks():
    arithmetic = 0
    # Independent oracle is direct signed integer dot product, then ring reduction.
    for coeffs in product(range(-2, 2), repeat=2):
        w = [list(coeffs)]
        for x in product(range(-2, 2), repeat=2):
            for r in product(range(8), repeat=2):
                for ql, sh in product(range(2), range(4)):
                    low, high = [v % 2 for v in r], [v // 2 for v in r]
                    mask, s, q = join_correction(w, low, high, [ql], [sh], 3)
                    d = [(v - z) % 8 for v, z in zip(x, mask, strict=True)]
                    actual = (sum(v * z for v, z in zip(coeffs, d, strict=True)) + q[0] + s[0]) % 8
                    expected = sum(coeffs[i] * x[i] for i in range(2)) % 8
                    if actual != expected:
                        raise AssertionError("exact ring oracle mismatch")
                    arithmetic += 1
    binary = 0
    for ws in product((-1, 0, 1), repeat=2):
        for aa in product((0, 1), repeat=4):
            a = [list(aa[:2]), list(aa[2:])]
            for u, e in product(product((0, 1), repeat=2), repeat=2):
                low, image = binary_correction([list(ws)], a, u, e)
                if image != [sum(ws[i] * low[i] for i in range(2)) % 2]:
                    raise AssertionError("binary oracle mismatch")
                binary += 1
    reference = toy_view((0, 0))
    perfect = max(tv(reference, toy_view(x)) for x in product(range(8), repeat=2))
    biased = tv(toy_view((0, 0), noise="biased"), toy_view((1, 0), noise="biased"))
    disclosed = tv(toy_view((0, 0), disclose_low=True), toy_view((1, 0), disclose_low=True))
    high_leak = tv(toy_view((0, 0), disclose_high_share=True),
                   toy_view((1, 0), disclose_high_share=True))
    # W=[1,1], l=[1,1], h=0, s_low=q_low=0: omitted carry gives 0, correct gives 2.
    return {
        "ring_oracle_outputs": arithmetic, "binary_oracle_outputs": binary,
        "full_joint_uniform_input_distributions": 64,
        "full_joint_uniform_max_tv": str(perfect),
        "biased_noise_joint_tv": str(biased), "disclosed_low_joint_tv": str(disclosed),
        "disclosed_high_share_joint_tv": str(high_leak),
        "omitted_lift_counterexample": {"w": [1, 1], "low": [1, 1], "correct": 2, "naive": 0},
        "borrow_counterexample": {"w": [1], "r": [0], "q_low": 1,
                                 "s_low": 1, "correct_q_high_mod4": 3, "naive": 0},
        "certifies_security": False,
    }


def split_saving(bits, lanes):
    if lanes < 0 or bits < lanes:
        raise ValueError("invalid one-bit slice")
    return packed(bits) - packed(bits - lanes)


def finite_cost(pairs, rows_per_pair, capacity, burned, setup, delivery, per_row):
    if any(type(v) is not int or v < 0 for v in
           (pairs, rows_per_pair, burned, setup, delivery, per_row)) or capacity <= 0:
        raise ValueError("invalid finite issuance budget")
    rows = pairs * rows_per_pair + burned
    pools = (rows + capacity - 1) // capacity
    return {"rows": rows, "pools": pools, "unused": pools * capacity - rows,
            "extra_bytes": setup + pools * delivery + rows * per_row}


def budget(control, lanes, pairs, target, extra, *, public_reused=True):
    """Necessary optimistic ceiling, original public reuse symmetric."""
    total = (control["public_delivery_bytes"] + pairs *
             (control["covered_pair_bytes"] - control["public_delivery_bytes"])) if public_reused else pairs * control["covered_pair_bytes"]
    return Fraction(pairs * packed(lanes)) - target * total - extra


def cost_report(source):
    result = []
    columns = source["subset_columns"]
    for case in source["cases"]:
        values = dict(zip(columns, case["subsets"]["gate_up_bulk_prefill"], strict=True))
        # Column labels are pinned; geometry lane position additionally reconciles widths.
        lanes = case["subsets"]["gate_up_bulk_prefill"][2]
        sample = next(s for s in case["samples"] if s["replicate"] == case["representative_replicate"])
        ceiling = packed(lanes)
        row = {"case": case["case"], "output_lanes": lanes,
               "optimistic_one_bit_removed_bytes": ceiling,
               "first_pair_zero_cost_saving_fraction": str(Fraction(ceiling, sample["covered_pair_bytes"])),
               "warm_zero_cost_saving_fraction": str(Fraction(ceiling, sample["covered_pair_bytes"] - sample["public_delivery_bytes"])),
               "first_pair_20pct_extra_ceiling": str(budget(sample, lanes, 1, Fraction(1, 5), 0)),
               "horizons": []}
        for horizon in (1, 2, 4, 16, 64, 4096):
            row["horizons"].append({"pairs": horizon,
                                    "positive_saving_extra_ceiling": horizon * ceiling,
                                    "20pct_extra_ceiling": str(budget(sample, lanes, horizon, Fraction(1, 5), 0))})
        # Ensure imported payload has not been silently treated as bit-plane payload.
        if not values or case["subsets"]["gate_up_bulk_prefill"][0] <= ceiling:
            raise ValueError("invalid source payload")
        result.append(row)
    return {
        "bulk_controls": result,
        "bulk_binary_w_publication_bytes": 23 * packed(9728 * 896),
        "all_96_stage_binary_w_bytes": 357826560 // 8,
        "shared_A_dense_publication_formula": "ceil(896*d/8)",
        "all_23_P_dense_publication_formula": "23*ceil(9728*d/8)",
        "sparse_W_column_read_bytes_per_row_stage": "t*ceil(m/8), absent cache reuse",
        "dense_P_read_bytes_per_row_stage": "ceil(m*d/8), absent cache reuse",
        "A_expansion_bit_operations_per_row": "n*d for dense A; unknown optimized generator",
        "P_expansion_bit_operations_per_row": "m*d for dense P; unknown optimized generator",
        "P_creation_bit_operations_per_stage": "m*n*d for naive binary product",
        "fresh_rows_per_pair_bulk": {"150": 3450, "288": 6624},
        "complete_gate": "E(R) < R*S for any positive saving; E(R) <= R*S-g*(C+R*T) for target g",
        "E_components": ["A/P/W publication to every new owner", "creation inputs/outputs",
                         "private mask issuance and both role keys", "high-plane retained preparation",
                         "frame splitting delta", "burns/renewals/rollback protection", "authentication/control"],
        "unmeasured": ["secure d/noise/horizon", "CPU", "whole-client RSS", "disk/cache/I/O", "serving saving"],
    }


def report(root=ROOT):
    sources = {}
    for path, expected in LOCKS.items():
        if sha(root / path) != expected:
            raise ValueError(f"source hash drift: {path}")
        if path.endswith(".json"):
            sources[path] = json.loads((root / path).read_text())
    selective = sources["docs/evidence/research-selective-correlation-budget.json"]
    return {
        "schema": "binary-correction-slices-v1",
        "verdict": "conditional one-bit algebra; security unknown; 20pct rejected by free byte ceiling",
        "source_hashes": LOCKS,
        "implementation_hashes": {path: sha(root / path) for path in
                                  ("scripts/probe_binary_correction_slices.py", "tests/test_binary_correction_slices_probe.py")},
        "primary_source_pins_not_binary_parameter_certificates": selective["papers"],
        "checks": checks(), "cost": cost_report(selective),
        "security": {"assumption": "multi-sample noisy-code pseudorandomness of (A, A*u_j+e_j) over GF(2)",
                     "concrete_parameters": None, "security_bits": None,
                     "fresh_multi_query_requirement": "independent u,e,high masks,high output shares; bound all stages/tickets under same A",
                     "seed_custody": "inference owns q_low seed only; client/dealer own private low/high and high-share seeds",
                     "dealer": "trusted noncolluding offline party still computes exact high correction",
                     "active_security": "not supplied; preserve original verifier separately"},
        "native_real_geometry_runs": "not launched; parent-only serialized opt-in fresh_workers required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = report()
    with args.output.open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
