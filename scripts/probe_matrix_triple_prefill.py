#!/usr/bin/env python3
"""Bounded ideal-dealer matrix-triple algebra and historical-shape byte screen.

No PCG, cryptographic PRNG, checkpoint execution, or production ownership is
implemented. random.Random is deliberately a reproducible test-only dealer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELD = 2**31 - 1
TILES = (1, 4, 16, 64, 150)
SOURCES = (
    "docs/evidence/research-all-link-frontier.json",
    "docs/evidence/research-narrow-ingress.json",
    "docs/evidence/research-bootstrap-ingress.json",
    "docs/evidence/network-algebra-qwen25-2026-10-06.json",
)
PRIMARY = {
    "url": "https://eprint.iacr.org/2026/2298",
    "title": "Pseudorandom Correlation Generators for Matrix Triples",
    "authors": ["Maxime Bombar", "Geoffroy Couteau", "Nikolas Melissaris", "Anuja Modi"],
    "received": "2026-10-01",
    "approved": "2026-10-04",
    "landing_modified_timestamp": "2026-10-01T16:31:24+00:00",
    "landing_html_sha256": "9d38ef537bed564d1fdfddc3a6687086e1b3b477ecdd7ebff62472edbb800e00",
    "landing_html_utf8_bytes": 18583,
    "version": "current landing page; numbered version unverified (versions HTTP 403)",
    "access": "primary landing HTML and abstract only; PDF HTTP 403; author-text search empty",
    "claims": [
        "Random square field matrix triples (A,B,AB), not chosen fixed W correlations.",
        "First: QASD, seed poly(lambda,log n,log N), Theta(n^3 log(n^3 N)) field operations/triple.",
        "Second: new Weyl-Pauli twisted QASD variant, same seed asymptotics, small constant over clear triples.",
    ],
    "not_verified": ["finite constants", "field restrictions", "dimension restrictions", "rectangular support", "reusable A", "malicious security", "finite setup and renewal capacity"],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add(a, b, q, sign=1):
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b, strict=True)):
        raise ValueError("matrix shape mismatch")
    return [[(x + sign * y) % q for x, y in zip(ar, br, strict=True)]
            for ar, br in zip(a, b, strict=True)]


def multiply(a, b, q):
    if not a or not b or any(len(row) != len(b) for row in a):
        raise ValueError("product shape mismatch")
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) % q
             for j in range(len(b[0]))] for i in range(len(a))]


def sample(rng, m, n, q):
    return [[rng.randrange(q) for _ in range(n)] for _ in range(m)]


def split(rng, matrix, q):
    left = sample(rng, len(matrix), len(matrix[0]), q)
    return left, add(matrix, left, q, -1)


class UseRegistry:
    """Shared in-process spent set catches shallow clones, not durable rollback."""

    def __init__(self):
        self.spent = set()

    def burn(self, label):
        if label in self.spent:
            raise ValueError("spent correlation")
        self.spent.add(label)


class ToyTriple:
    """Functional role shares held together only for independent test orchestration."""

    def __init__(self, rng, m, n, t, q, registry, label, fixed_a=None):
        self.q, self.registry, self.label = q, registry, label
        self.a = fixed_a if fixed_a is not None else sample(rng, m, n, q)
        self.b = sample(rng, n, t, q)
        self.as_ = split(rng, self.a, q)
        self.bs = split(rng, self.b, q)
        self.cs = split(rng, multiply(self.a, self.b, q), q)

    def cancel(self):
        self.registry.burn(self.label)

    def consume(self, w, x):
        # Burn before validation: failures cannot return a correlation to inventory.
        self.registry.burn(self.label)
        q = self.q
        # Public W is assigned entirely to Inference's additive share.
        zero = [[0] * len(w[0]) for _ in w]
        dc = add(zero, self.as_[0], q, -1)
        di = add(w, self.as_[1], q, -1)
        d = add(dc, di, q)  # two directed dense opening-share messages
        hc = add(self.cs[0], multiply(d, self.bs[0], q), q)
        hi = add(self.cs[1], multiply(d, self.bs[1], q), q)
        ec = add(x, self.bs[0], q, -1)
        ei = [[(-v) % q for v in row] for row in self.bs[1]]
        # Only ec is sent Client->Inference; ei and E stay at Inference because
        # the entire public W*E term is assigned there. No reverse E opening.
        e = add(ec, ei, q)
        yi = add(hi, multiply(w, e, q), q)
        return add(hc, yi, q), {"d": d, "e": e, "hc": hc, "hi": hi}


def clear_signed(w, x):
    # Independent oracle: direct signed integers, no triples/modular helper.
    return [[sum(v * x[k][j] for k, v in enumerate(row))
             for j in range(len(x[0]))] for row in w]


def bounded_algebra():
    rng = random.Random(2298)
    checked = cases = 0
    by_domain = {}
    for q in (2**16, 2**24, 2**32, FIELD):
        domain_count = 0
        for tile in TILES:
            # Exercise full tile and complete non-divisible remainder, bounded m,n.
            w = [[-128, 127, -3], [9, -7, 0], [1, 1, 1]]
            x = [[rng.randrange(-128, 128) for _ in range(tile + 3)] for _ in range(3)]
            x[0][0], x[1][0] = -128, 127
            expected = clear_signed(w, x)
            for offset in range(0, tile + 3, tile):
                chunk = [row[offset:offset + tile] for row in x]
                triple = ToyTriple(rng, 3, 3, len(chunk[0]), q, UseRegistry(), (q, tile, offset))
                got, view = triple.consume(w, chunk)
                assert view["d"] == add(w, triple.a, q, -1)
                assert add(view["hc"], view["hi"], q) == multiply(w, triple.b, q)
                for i, row in enumerate(got):
                    for j, value in enumerate(row):
                        oracle = expected[i][offset + j]
                        assert value == oracle % q
                        if q == FIELD:
                            centered = value if value <= q // 2 else value - q
                            assert centered == oracle
                            for bits in (16, 24, 32):
                                assert centered % (2**bits) == oracle % (2**bits)
                        checked += 1
                        domain_count += 1
                cases += 1
        by_domain[str(q)] = domain_count
    # Reusing A is a separately specified ideal-dealer variant, not paper support.
    a = sample(rng, 2, 3, FIELD)
    w = [[-128, 127, 3], [0, -2, 1]]
    ds = []
    for use in range(3):
        x = [[rng.randrange(-128, 128) for _ in range(4)] for _ in range(3)]
        triple = ToyTriple(rng, 2, 3, 4, FIELD, UseRegistry(), use, fixed_a=a)
        got, view = triple.consume(w, x)
        assert got == [[v % FIELD for v in row] for row in clear_signed(w, x)]
        ds.append(view["d"])
        checked += 8
    assert ds[0] == ds[1] == ds[2]
    return {"independent_output_values": checked, "fresh_a_cases": cases,
            "fresh_a_output_values_by_modulus": by_domain,
            "reused_a_ideal_dealer_cases": 3,
            "field_to_ring_checks_per_field_output": 3,
            "test_only_ideal_dealer": True, "pcg_implemented": False}


def leakage_witnesses():
    q, b, x1, x2 = 257, 193, -17, 42
    e1, e2 = (x1 - b) % q, (x2 - b) % q
    assert (e2 - e1) % q == (x2 - x1) % q
    # Public A and a single role's B_i,C_i remain masked by independent C_j.
    # If total C or Client's other B share is disclosed, scalar invertible A breaks it.
    a = 7
    c = a * b % q
    recovered = c * pow(a, -1, q) % q
    assert recovered == b
    # A small field loses signed integer output, even though field equations hold.
    integer = 127 * 127 * 2
    wrong = integer % q
    assert wrong % 65536 != integer % 65536
    return {"reused_B_opening_difference": (e2 - e1) % q,
            "actual_input_difference_mod257": (x2 - x1) % q,
            "public_A_plus_total_C_recovers_B": True,
            "public_A_plus_single_role_C_share_is_not_total_C": True,
            "small_field257_integer": integer, "small_field257_residue": wrong}


def unique_json_object(pairs):
    """Reject duplicate JSON object keys before stage mappings can hide them."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON key")
    return result


def geometry(root=ROOT):
    sources = [json.loads((root / p).read_text(), object_pairs_hook=unique_json_object)
               for p in SOURCES]
    return validate_geometry(*sources)


def unique_stages(stages, source):
    names = [s["stage"] for s in stages]
    assert len(names) == len(set(names)), f"duplicate stage in {source}"
    return {s["stage"]: s for s in stages}


def validate_geometry(all_link, narrow, bootstrap, source_geometry):
    """Validate independently saved source identities before any projections."""
    control = all_link["control"]
    stages = bootstrap["cost"]["stages"]
    indexed = unique_stages(narrow["checkpoint_geometry"]["stages"], "narrow")
    bootstrap_index = unique_stages(stages, "bootstrap")
    source_index = unique_stages(source_geometry["stages"], "original geometry")
    names = set(control["stages"])
    assert names == set(indexed) == set(bootstrap_index) == set(source_index), "stage set mismatch"
    assert len(names) == 96, "stage count mismatch"
    source_lock = source_geometry["source"]
    narrow_lock = narrow["checkpoint_geometry"]["lock"]
    assert (control["body_fingerprint"] == narrow_lock["body_fingerprint"]
            == bootstrap["cost"]["control_body_fingerprint"] == source_lock["body_fingerprint"]), "body lock mismatch"
    for key in ("checkpoint_digest", "composition_digest", "source_lock_digest", "model", "revision"):
        assert narrow_lock[key] == source_lock[key], f"source lock mismatch: {key}"
    assert control["pair_bytes"] == 493575247
    assert control["classes"]["public_bundle"] == 124905091
    assert sum(s["rows"] for s in stages) == 15297
    assert sum(s["old_input_payload"] for s in stages) == 90104112
    assert sum(s["output_payload"] for s in stages) == 138864059
    for s in stages:
        other = indexed[s["stage"]]
        original = source_index[s["stage"]]
        assert s["shape"] == other["shape"] == original["shape"], "stage shape mismatch"
        assert s["weight_sha256"] == other["weight_sha256"] == original["weight_sha256"], "stage weight mismatch"
        assert s["stage_bits"] == other["stage_bits"] == original["annihilator"]["uniform_input_bits"], "encoded width mismatch"
        assert s["original_ring_bits"] == original["ring_bits"], "original ring mismatch"
        assert s["original_ring_bits"] in (24, 32)
        terminal = s["stage"].startswith("layers.23.") and "qkv_proj" not in s["stage"]
        assert s["rows"] == (15 if terminal else 164), "stage executed rows mismatch"
        m, n = s["shape"]
        widths = other["row_residue_width_histogram"]
        assert all(isinstance(count, int) and count > 0 for count in widths.values()), "invalid width histogram"
        assert sum(widths.values()) == m, "width histogram row count mismatch"
        assert max(map(int, widths)) == s["stage_bits"], "residue/encoded width mismatch"
        assert 0 < min(map(int, widths)) <= s["stage_bits"] <= s["original_ring_bits"], "width exceeds ring"
        assert s["old_input_payload"] == packed(s["rows"] * n, s["stage_bits"]), "encoded input payload mismatch"
        assert s["output_payload"] == packed(s["rows"] * sum(int(h) * count for h, count in widths.items()), 1), "encoded output payload mismatch"
        assert s["body_macs"] == math.prod(s["shape"]) * s["rows"]
        for category, link, payload in (
            ("input", "client->inference", s["old_input_payload"]),
            ("output", "inference->client", s["output_payload"]),
            ("correction", "preparation->inference", s["output_payload"]),
        ):
            assert control["stages"][s["stage"]][link] == payload + s["retained_body_overhead"][category], "per-stage ledger mismatch"
        s["row_residue_width_histogram"] = dict(widths)
    return control, stages


def packed(elements, bits):
    return (elements * bits + 7) // 8


def project(stages, tile, field=True):
    """Optimistic rectangular triples; exact remainder and serial decode underfill.

    Every A is fresh. Dense D opens in two directions. Only Client's E share
    is sent to Inference. Output share goes Inference->Client. Setup is unknown.
    """
    dbytes = ebytes = ybytes = macs = padded_macs = generated = peak = 0
    stage_rows = []
    square_d = square_e = square_triples = square_expansion_macs = 0
    prefill_macs = decode_macs = 0
    static_d = 0
    for s in stages:
        m, n = s["shape"]
        bits = 31 if field else s["original_ring_bits"]
        prefill = s["rows"] - 14
        batches = math.ceil(prefill / tile) + 14
        remainder = prefill % tile
        slots = (math.ceil(prefill / tile) + 14) * tile
        # Ring quotient is a valid homomorphism; grant public per-row width packing.
        # Odd-field shares cannot be truncated this way without conversion.
        d_one = packed(m * n, bits) if field else packed(
            n * sum(int(h) * count for h, count in s["row_residue_width_histogram"].items()), 1)
        d = 2 * batches * d_one
        e = packed(n * s["rows"], bits) if field else s["old_input_payload"]
        y = packed(m * s["rows"], bits) if field else s["output_payload"]
        static_d += 2 * d_one
        dbytes += d
        ebytes += e
        ybytes += y
        macs += m * n * s["rows"]
        prefill_macs += m * n * prefill
        decode_macs += m * n * 14
        padded_macs += m * n * slots
        # Each endpoint generates A_i,B_i,C_i; both endpoints charged explicitly.
        per_role = batches * m * n + (n + m) * s["rows"]
        generated += 2 * packed(per_role, bits)
        # Logical packed storage only, NOT whole-process memory or PCG workspace.
        stage_peak = packed(3 * m * n + 3 * (m + n) * tile, bits) + m * n
        peak = max(peak, stage_peak)
        # Conservative power-of-two square tiling interpretation, not source theorem.
        b = 1 << (tile - 1).bit_length()
        mb, nb = math.ceil(m / b), math.ceil(n / b)
        batch_count = math.ceil(prefill / b) + 14
        triples = mb * nb * batch_count
        square_triples += triples
        square_d += 2 * packed(triples * b * b, bits)
        square_e += 2 * packed(triples * b * b, bits)
        square_expansion_macs += triples * b**3
        stage_rows.append({"stage": s["stage"], "shape": s["shape"], "rows": s["rows"],
                           "prefill_rows": prefill, "decode_rows": 14, "batches": batches,
                           "complete_remainder": remainder, "decode_active_columns": 1,
                           "decode_unused_columns_each": tile - 1, "fixed_width_slots": slots,
                           "D_two_direction_bytes": d, "E_client_to_inference_bytes": e,
                           "output_share_bytes": y, "logical_packed_stage_peak": stage_peak})
    # Historical non-payload body cost retained; deleted correction includes metadata.
    retained = 368670156 - 139060689 - 90104112 - 138864059
    recurring_floor = retained + dbytes + ebytes + ybytes
    return {"tile": tile, "domain": "field31 bounded lift" if field else "exact original ring24/32",
            "projection_scope": "hypothetical ideal rectangular triple lower bound, not paper instantiation or measurement",
            "D_two_direction_bytes": dbytes, "E_client_to_inference_bytes": ebytes,
            "generic_two_role_full_E_opening_bytes": 2 * ebytes,
            "output_share_bytes": ybytes, "retained_historical_nonpayload_bytes": retained,
            "recurring_floor_before_setup_helpers": recurring_floor,
            "net_saved_before_setup_helpers": 368670156 - recurring_floor,
            "all_original_body_macs_remote_prefill": prefill_macs,
            "all_original_body_macs_remote_decode": decode_macs,
            "remote_original_body_fraction_prefill": 1.0,
            "remote_original_body_fraction_decode": 1.0,
            "extra_client_dense_crypto_macs": macs,
            "extra_inference_dense_crypto_macs": macs,
            "extra_each_role_if_fixed_width_padded_macs": padded_macs,
            "both_endpoint_generated_packed_matrix_bytes": generated,
            "logical_packed_peak_one_stage_bytes": peak,
            "whole_client_256MiB_admitted": False,
            "client_peak_rss_bytes": None, "client_cpu_seconds": None,
            "static_D_both_endpoint_delivery_if_reusable_A": static_d,
            "reusable_A_unsupported_recurring_floor": retained + ebytes + ybytes,
            "reusable_A_unsupported_saving": 368670156 - retained - ebytes - ybytes,
            "two_online_providers_changed_topology_recurring_floor": retained + dbytes + 4 * ebytes + 2 * ybytes,
            "two_online_providers_cost_scope": "Client sends two X shares, providers exchange two E shares, both send Y shares; Preparation now online and shares private X; no lightweight-client claim in original topology",
            "static_D_status": "unsupported reusable-A ideal-dealer variant; not charged as attained saving",
            "reusable_A_unsupported_headroom_before_static_D": headroom(368670156 - retained - ebytes - ybytes),
            "square_power2_block": 1 << (tile - 1).bit_length(),
            "square_tiling_status": "conditional padding map only; source dimension/field admissibility unverified",
            "square_triples": square_triples, "square_D_two_direction_bytes": square_d,
            "square_E_client_to_inference_bytes": square_e // 2,
            "square_generic_E_two_direction_bytes": square_e,
            "square_clear_triple_macs": square_expansion_macs,
            "square_each_role_D_B_adaptation_macs": square_expansion_macs,
            "square_both_endpoint_generated_matrix_elements": 6 * square_triples * (1 << (tile - 1).bit_length())**2,
            "square_pcg_expansion_field_ops": None, "stages": stage_rows}


def headroom(saving, horizons=(1, 2, 4, 16, 1024)):
    result = []
    for h in horizons:
        baseline = 124905091 + h * 368670156
        result.append({"pairs": h, "matched_control": baseline,
                       "additional_setup_and_helper_budget_per_pair": {
                           str(p): str(Fraction(saving) - p * Fraction(baseline, h))
                           for p in (Fraction(1, 5), Fraction(7, 20), Fraction(1, 2))}})
    result.append({"pairs": "infinite warm", "additional_setup_and_helper_budget_per_pair": {
        str(p): str(Fraction(saving) - p * 368670156)
        for p in (Fraction(1, 5), Fraction(7, 20), Fraction(1, 2))}})
    return result


def report():
    control, stages = geometry()
    projections = [project(stages, tile, field) for field in (True, False) for tile in TILES]
    for cell in projections:
        cell["matched_horizon_headroom"] = headroom(cell["net_saved_before_setup_helpers"])
        assert cell["net_saved_before_setup_helpers"] < 0
    files = (*SOURCES, "scripts/probe_matrix_triple_prefill.py",
             "tests/test_matrix_triple_prefill_probe.py",
             "docs/research/matrix-triple-prefill-reference.md")
    return {
        "schema": "pllm.matrix_triple_prefill.v1", "complete": True,
        "decision": "reject explicit fresh random-to-fixed matrix Beaver layout: dense D openings alone erase seed savings",
        "scope": "bounded functional toy and saved checkpoint-shape projections; no PCG/decoder/performance measurement",
        "primary_source": PRIMARY,
        "ring_context_primary_source": {
            "url": "https://eprint.iacr.org/2025/1223",
            "title": "Efficient Pseudorandom Correlation Generators over Z/p^k Z",
            "authors": ["Zhe Li", "Chaoping Xing", "Yizhou Yao", "Chen Yuan"],
            "received": "2025-07-01", "approved": "2025-07-07",
            "landing_html_sha256": "1b77ecdec75fac21664c7e6d5202713466d9c7dc37d26aa5c43e5fca78a0fa5a",
            "access": "primary abstract only; not an implementation or finite-parameter transfer",
            "claim": "Galois ring OLE and authenticated triples, circuit-dependent preprocessing and matrix multiplication applications; does not establish 2026 paper over ring2^k",
        },
        "primary_verified_metadata_snapshot_sha256": hashlib.sha256(
            json.dumps(PRIMARY, sort_keys=True).encode()).hexdigest(),
        "primary_sha_scope": "canonical verified landing metadata/claim snapshot, NOT full PDF or raw HTML hash",
        "source_hashes": {p: digest(ROOT / p) for p in files},
        "algebra": bounded_algebra(), "leakage_controls": leakage_witnesses(),
        "focused_verification_record": {"pytest_passed": 39, "ruff": "All checks passed",
                                        "scope": "recorded external focused commands, not a test run inside this probe"},
        "geometry": {"stages": 96, "executed_rows": 15297, "ordinary_prefill_rows": 150,
                     "terminal_prefill_rows": 1, "terminal_prefill_stages": 3,
                     "decode_ops_per_request": 7, "requests_per_pair": 2,
                     "exact_repeat_prefill_rows": 0, "control_classes": control["classes"]},
        "free_correction_removal_only_headroom": headroom(139060689),
        "projections": projections,
        "unknown_all_link_costs": {name: None for name in (
            "PCG_setup_both_roles", "seed_delivery_authentication", "renewal_capacity",
            "renewal_bytes", "security_failure_budget", "malicious_verification",
            "new_opening_frames", "failed_cancelled_burned_material", "checkpoint_distribution",
            "whole_wire_transport", "PCG_endpoint_CPU_workspace")},
        "lifecycle_formula": "control=P+H*R; candidate=Pnew+H*(R-S+helper+failure+frames+verify)+ceil(H/L)*B; P common assumed124905091, R368670156; L/security unverified",
        "promotion": False, "qualifies_Low": False,
        "next_gate": "obtain full paper/version; determine whether A-only opening seeds can be safely disclosed independently of B/C; then reviewed fixed-W private consumer with no fresh dense D, no client dense adaptation, exact domain/packing and finite setup/lifetime parameters",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"decision": result["decision"], "algebra": result["algebra"]}, sort_keys=True))


if __name__ == "__main__":
    main()
