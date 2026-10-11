"""Bounded narrow-ingress falsification; never issues protected material.

Default: tiny exhaustive integer/privacy checks plus frozen evidence projections.
Optional serial geometry worker loads the pinned checkpoint, not live inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "docs/evidence/network-algebra-qwen25-2026-10-06.json"
CONTROL = ROOT / "docs/evidence/research-prepared-study-qwen25-confirm.json"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identities():
    """Actual executable/runtime/native/dependency identities, not git HEAD alone."""
    import pllm
    from pllm import _native
    paths = [Path(__file__), GEOMETRY, CONTROL,
             ROOT / "tests/test_narrow_ingress_probe.py",
             ROOT / "docs/research/narrow-ingress-reference.md",
             ROOT / "scripts/frontier_probe_support.py",
             ROOT / "scripts/decoder_probe_support.py",
             ROOT / "scripts/network_algebra_hypotheses.py",
             ROOT / "scripts/probe_all_link_frontier.py", Path(sys.executable).resolve(),
             Path(_native.__file__)]
    package = Path(pllm.__file__).parent
    paths.extend(sorted(package.rglob("*.py")))
    return {"files": {str(p): digest(p) for p in paths},
            "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                        for d in importlib.metadata.distributions()))}


def centered(value, bits):
    modulus = 1 << bits
    return (value + modulus // 2) % modulus - modulus // 2


def lift(x, r, bits=8):
    return (x - r) % (1 << bits), int(x < r)


def repaired_product(weights, x, r, q, s, t, widths, bits=8):
    """Toy trusted-dealer oracle; q independent uniform in carry quotient.

    Caller values are deterministic test fixtures, never cryptographic masks.
    Provider gets a,c,Wr-s,Wq-t; client alone retains s,t and reconstructs.
    """
    modulus = 1 << bits
    a, b = zip(*(lift(xi, ri, bits) for xi, ri in zip(x, r, strict=True)))
    carry_bits = max(widths) - bits
    if carry_bits < 1:
        raise ValueError("probe requires a nontrivial carry quotient")
    c = [(bi - qi) % (1 << carry_bits) for bi, qi in zip(b, q, strict=True)]
    outputs = []
    for row, sj, tj, width in zip(weights, s, t, widths, strict=True):
        ring = 1 << width
        quotient = 1 << (width - bits)
        main = (sum(w * (ai + ri) for w, ai, ri in zip(row, a, r, strict=True)) - sj) % ring
        repair = (sum(w * (ci + qi) for w, ci, qi in zip(row, c, q, strict=True)) - tj) % quotient
        outputs.append(centered(main - modulus * repair + sj - modulus * tj, width))
    return outputs


def fused_product(weights, x, r, q, s, widths, bits=8):
    """Fully fused fresh wide-mask decomposition; no second correction or t.

    R=r-2^bits*q is uniform wide-ring mask when r,q are independent uniform.
    This is a recoding of ordinary delegation, not free new correlated material.
    """
    low = 1 << bits
    a, b = zip(*(lift(xi, ri, bits) for xi, ri in zip(x, r, strict=True)))
    c = [(bi - qi) % (1 << (max(widths) - bits))
         for bi, qi in zip(b, q, strict=True)]
    out = []
    for row, sj, width in zip(weights, s, widths, strict=True):
        correction = sum(w * (ri - low * qi) for w, ri, qi in zip(row, r, q, strict=True)) - sj
        response = sum(w * (ai - low * ci) for w, ai, ci in zip(row, a, c, strict=True)) + correction
        out.append(centered(response + sj, width))
    return out


def tv(left, right):
    nl, nr = sum(left.values()), sum(right.values())
    return sum((abs(Fraction(left[k], nl) - Fraction(right[k], nr))
                for k in left.keys() | right.keys()), Fraction()) / 2


def distributions(bits=3, quotient_bits=3, q_domain=None, disclose=False):
    m, l = 1 << bits, 1 << quotient_bits
    domain = range(l) if q_domain is None else q_domain
    result = []
    for x in range(-m // 2, m // 2):
        histogram = Counter()
        for r in range(-m // 2, m // 2):
            a, b = lift(x, r, bits)
            for q in domain:
                histogram[(a, b if disclose else (b - q) % l)] += 1
        result.append(histogram)
    return result


def privacy_checks():
    cases = {}
    for name, kwargs in (("full_quotient", {}), ("disclosed_carry", {"disclose": True}),
                         ("binary_mask_in_wide_ring", {"q_domain": range(2)}),
                         ("missing_carry", {"q_domain": range(1)})):
        views = distributions(**kwargs)
        maximum, witness = Fraction(), None
        for i, left in enumerate(views):
            for j, right in enumerate(views):
                distance = tv(left, right)
                if distance > maximum:
                    maximum, witness = distance, [i - 4, j - 4]
        cases[name] = {"maximum_tv": str(maximum), "input_witness": witness,
                       "domain": "signed3 inputs/masks; quotient ring8; exact enumeration"}
    # One scalar pad on every table entry does not hide its transition pattern.
    patterns = {}
    for r in range(-4, 4):
        values = []
        for a in range(8):
            x = centered(a + r, 3)
            values.append(int(x < r))
        patterns[r] = tuple((values[a] - values[0]) % 8 for a in range(8))
    cases["single_pad_carry_lookup"] = {"distinct_patterns": len(set(patterns.values())),
        "possible_masks": 8, "result": "table differences recover r, then a recovers x"}
    cases["reused_byte_mask"] = {"identity": "(a1-a0) mod256 = (x1-x0) mod256",
        "witness": {"inputs": [0, 1], "r": 17, "openings": [239, 240]}}
    # Include provider's actual fused correction, not only ingress marginals.
    reference = None
    for x in range(-4, 4):
        view = Counter()
        for r in range(-4, 4):
            a, b = lift(x, r, 3)
            for q in range(8):
                for s in range(64):
                    view[(a, (b - q) % 8, (3 * (r - 8 * q) - s) % 64)] += 1
        if reference is None:
            reference = view
        elif tv(reference, view):
            raise AssertionError("full fused provider transcript leaks")
    cases["full_fused_provider_transcript"] = {"maximum_tv": "0",
        "transcript": "(a,c,3R-s mod64)", "enumerated_private_inputs": 8,
        "mask_combinations_per_input": 4096,
        "assumption": "independent full-domain one-use r,q,s; trusted noncolluding offline dealer"}
    return cases


def algebra_checks():
    checked = 0
    for x in range(-128, 128):
        for r in range(-128, 128):
            a, b = lift(x, r)
            if a + r - 256 * b != x:
                raise AssertionError("signed lift mismatch")
            checked += 1
    # Independent clear Python products, varying per-output widths, all tiny x/r.
    weights, widths = [[3, -5], [-7, 2]], [7, 8]
    products = 0
    for x0 in range(-4, 4):
        for x1 in range(-4, 4):
            x = [x0, x1]
            oracle = [sum(w * xi for w, xi in zip(row, x, strict=True)) for row in weights]
            for r0 in range(-4, 4):
                for r1 in range(-4, 4):
                    for q in ([0, 0], [1, 7], [31, 17]):
                        actual = repaired_product(weights, x, [r0, r1], q,
                                                  [39, 201], [3, 11], widths, bits=3)
                        if actual != oracle:
                            raise AssertionError((actual, oracle))
                        if fused_product(weights, x, [r0, r1], q, [39, 201], widths, bits=3) != oracle:
                            raise AssertionError("fused oracle mismatch")
                        products += len(oracle)
    # Full byte endpoints with wide output rings (including -128, outside W8A8's -127 bound).
    for x in ([-128, 127], [127, -128], [-1, 0]):
        expected = [sum(w * xi for w, xi in zip(row, x, strict=True)) for row in weights]
        assert repaired_product(weights, x, [127, -128], [255, 91],
                                [100, 777], [2, 11], [12, 13]) == expected
    return {"signed_byte_lifts": checked, "tiny_product_outputs": products,
            "byte_endpoint_vectors": 3, "dequantization": "integer equality only; retain original scales/bias unchanged",
            "distributed_security_proof": False, "protected_primitives_executed": 0}


def geometry_projection():
    from scripts.probe_all_link_frontier import conserved_control
    geometry = json.loads(GEOMETRY.read_text())
    control = conserved_control(json.loads(CONTROL.read_text()))
    if not geometry["complete"] or geometry["source"]["body_fingerprint"] != control["body_fingerprint"]:
        raise ValueError("control/geometry identity mismatch")
    stages = []
    for stage in geometry["stages"]:
        m, n = stage["shape"]
        rows = geometry["executed_rows_per_stage"]
        k = stage["annihilator"]["uniform_input_bits"]
        old_input = rows * n * stage["ring_bits"] // 8
        output = (stage["control_arithmetic_bytes"] - old_input) // 2
        sum_bits = output * 8 // rows
        if (sum_bits * rows + 7) // 8 != output:
            raise ValueError("cannot recover residue width sum")
        baseline = (rows * n * k + 7) // 8
        narrow = rows * n
        carry_input = (rows * n * (k - 8) + 7) // 8
        carry_output = (rows * (sum_bits - 8 * m) + 7) // 8
        stages.append({"stage": stage["stage"], "shape": [m, n], "rows": rows,
            "stage_bits": k, "quotient_stage_bits": k - 8,
            "row_residue_sum_bits": sum_bits, "row_residue_max_bits": k,
            "row_residue_mean_bits": sum_bits / m,
            "row_residue_histogram": "not recoverable from saved aggregate; optional checkpoint worker derives it",
            "baseline_input_payload": baseline, "byte_input_payload": narrow,
            "repair_input_payload": carry_input,
            "extra_repair_correction_payload": carry_output,
            "extra_repair_output_payload_separate": carry_output,
            "fused_extra_repair_output_payload": 0,
            "output_fused_net_extra_payload": narrow + carry_input - baseline + carry_output,
            "fully_fused_net_extra_payload": narrow + carry_input - baseline})
    sums = {key: sum(s[key] for s in stages) for key in (
        "baseline_input_payload", "byte_input_payload", "repair_input_payload",
        "extra_repair_correction_payload", "extra_repair_output_payload_separate",
        "output_fused_net_extra_payload", "fully_fused_net_extra_payload")}
    coordinates = sum(s["rows"] * s["shape"][1] for s in stages)
    return {"source": geometry["source"], "historical_46_row_projection": sums,
        "stages": stages,
        "geometry_stage_width_histogram": dict(sorted(Counter(str(s["stage_bits"]) for s in stages).items())),
        "measured_control": {k: v for k, v in control.items() if k != "stages"},
        "hypothetical_uniform24_to8_gross_bytes": control["classes"]["inputs"] * 2 / 3,
        "complete_cost_gate": {
            "fully_fused": "prepare C=W(r-256q)-s; provider sends Wa-256Wc+C; client adds s; original correction/output sizes",
            "fully_fused_net_saving_bytes": 0,
            "fully_fused_masks": "fresh original wide mask R decomposes uniquely into signed byte r and quotient q; no extra free randomness or reused ticket",
            "output_fused_only": "provider returns main-256*repair once; client adds s-256*t",
            "additional_correction_separate_delegation": "one fresh Wq-t row-residue payload per issued row; cannot reuse Wr-s",
            "additional_output": "zero if fused, quotient row payload if separate",
            "input_and_repair": "8+(k-8)=k; no saving for stage-packed control",
            "setup_seed_bytes": ">=16 independent carry seed bytes per fresh issuance if specified PRG replaces q,t distribution; plus binding/framing; not measured",
            "fully_fused_setup_seed_bytes": "same existing fresh wide-mask seed/ticket material; no extra carry seed, no setup saving",
            "raw_dealer_to_client_alternative_46rows": sums["repair_input_payload"] + sums["extra_repair_correction_payload"],
            "party_key_ot_peer_bytes": 0,
            "why_zero": "direct carry delegation: client already knows b; no FSS/OT/B2A called",
            "external_framing_transport_manifest_bytes": None,
            "gate": "fully fused ties payload, may add framing/CPU; separate delegation loses even with zero new framing and optimistically free seed distribution"},
        "alternative_layouts": {
            "classic_128bit_garbling_one_selected_label_floor_46rows": 16 * coordinates,
            "classic_128bit_garbling_both_client_input_labels_46rows": 32 * coordinates,
            "classic_128bit_garbling": "specified fresh-label layout only, not universal lower bound; tables/OT/outputs/dealer frames/setup extra",
            "uniform_B2A": "arithmetic masks must cover carry quotient, not {0,1}; same delegated ingress cost",
            "shared_rescale": "sign of privately shared x-r needs signed9 domain; installed helper has two evaluators, keys and openings; client already knows b, so no compatible free conversion",
            "secret_replication": "scalar/vector field2/17/257 contract does not supply independent full-ring lane masks or exact ring lifting",
            "online_preparation": "changed topology; explicitly excluded from one-online-provider result",
            "clear_client_Wb": "extra full dense body MAC pass and public body-weight client storage; violates >90% remote body MAC target"}}


def checkpoint_geometry():
    import numpy as np
    from decoder_probe_support import DecoderFixture
    from network_algebra_hypotheses import row_widths
    fixture = DecoderFixture(inputs=150, outputs=8)
    expected = json.loads(GEOMETRY.read_text())
    rows = []
    for name, stage in fixture.body.items():
        widths = row_widths(stage.weight.values)
        if bytes(widths) != stage.output_residue_bits:
            raise ValueError("compiler residue mismatch")
        w = stage.weight.values.astype(np.int64)
        positive = np.maximum(w, 0).sum(axis=1)
        negative_magnitude = -np.minimum(w, 0).sum(axis=1)
        bit_widths = [max(1, (max(int(p) + 1, int(n)) - 1).bit_length() + 1)
                      for p, n in zip(positive, negative_magnitude, strict=True)]
        shifted_unsigned_widths = [int(p + n).bit_length()
                                   for p, n in zip(positive, negative_magnitude, strict=True)]
        if any(d < int(h) - 8 for d, h in zip(bit_widths, widths, strict=True)):
            raise AssertionError("exact bit-vector bound unexpectedly beats quotient")
        if any(d < int(h) - 8 for d, h in zip(shifted_unsigned_widths, widths, strict=True)):
            raise AssertionError("shifted unsigned bit-vector bound unexpectedly beats quotient")
        rows.append({"stage": name, "weight_sha256": stage.weight_digest,
                     "shape": list(stage.weight.values.shape),
                     "row_residue_width_histogram": dict(Counter(map(str, map(int, widths)))),
                     "quotient_width_histogram": dict(Counter(str(int(b) - 8) for b in widths)),
                     "exact_Wb_width_histogram": dict(Counter(map(str, bit_widths))),
                     "exact_Wb_stage_bits": max(bit_widths),
                     "shifted_unsigned_Wb_width_histogram": dict(Counter(map(str, shifted_unsigned_widths))),
                     "shifted_unsigned_Wb_stage_bits": max(shifted_unsigned_widths),
                     "shifted_unsigned_Wb_bounds": "Wb+sum negative magnitudes in [0,sum abs coefficients]; width bit_length(S), zero width when S=0",
                     "exact_Wb_bounds": "[-sum negative magnitudes, sum positive coefficients]; includes b=0/1 independently",
                     "stage_bits": int(widths.max())})
    saved = {s["stage"]: s for s in expected["stages"]}
    if {s["stage"] for s in rows} != set(saved):
        raise ValueError("stage set mismatch")
    for stage in rows:
        if stage["weight_sha256"] != saved[stage["stage"]]["weight_sha256"]:
            raise ValueError("weight identity mismatch")
    return {"scope": "compiler-derived checkpoint geometry, no serving/crypto timings",
            "lock": fixture.lock(), "stages": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--serial-geometry", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        request = json.load(sys.stdin)
        before = identities()
        if request["identities"] != before:
            raise ValueError("worker identity changed before execution")
        result = checkpoint_geometry()
        if before != identities():
            raise ValueError("worker identity changed during execution")
        print(json.dumps(result))
        return
    if args.output is None or args.output.exists():
        raise ValueError("provide fresh --output path")
    before = identities()
    result = {"schema": "pllm.narrow_ingress.v1", "complete": True,
              "scope": "bounded deterministic algebra/privacy falsification; no distributed security claim",
              "identities": before, "algebra": algebra_checks(), "privacy": privacy_checks(),
              "cost": geometry_projection(), "full_wire_bytes": None,
              "whole_client_cpu_seconds": None, "whole_client_peak_rss_bytes": None,
              "decision": "fully fused repair recodes baseline with zero saving; separate quotient delegation loses bytes; invalid shortcuts leak; not impossibility of every private lifting scheme"}
    if args.serial_geometry:
        from frontier_probe_support import fresh_workers
        with fresh_workers(__file__, memory_bytes=6 << 30, timeout=300) as (run, guard, host):
            result["checkpoint_geometry"] = run({"identities": before})
            result["fresh_worker_build"] = run.build
    if identities() != before:
        raise ValueError("parent identity changed during probe")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
