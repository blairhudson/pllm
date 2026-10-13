"""Small exact-ring algebra and saved-source cost screen; no SDK or crypto import."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = "docs/evidence/research-all-link-frontier.json"
CONTROL = "docs/evidence/research-prepared-study-qwen25-confirm.json"
NARROW = "docs/evidence/research-narrow-ingress.json"
GEOMETRY = "docs/evidence/network-algebra-qwen25-2026-10-06.json"
PRIMARY = [
    {"url": "https://eprint.iacr.org/2019/1159", "scope": "Silent random OT preprocessing; not chosen matrix-vector correlations"},
    {"url": "https://eprint.iacr.org/2025/1223", "scope": "Ring OLE/PCG and circuit-dependent preprocessing exist; no concrete PLLM consumer derived"},
    {"url": "https://eprint.iacr.org/2025/1637", "scope": "Generic arithmetic-circuit PCF via Ring-LWR/BFV; bootstrap and local evaluation not free"},
    {"url": "https://eprint.iacr.org/2026/196", "scope": "Prime-field OLE/VOLE; published throughput not exact power-of-two-ring consumer cost"},
    {"url": "https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITC.2026.7", "scope": "Secret replication scalar/vector; not independent byte/carry lanes"},
]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(name):
    return json.loads((ROOT / name).read_text())


def request_frame_size(rows, columns, width, wire_bits):
    """Independent byte enumeration of stage_protocol.pack_encoded schema.

    Includes 8-byte magic, u32 rows, array6, IDs, dimensions, bin headers.
    Not HTTP/TCP overhead or outer stage/session envelopes.
    """
    def bin_size(n):
        return n + (2 if n <= 255 else 3 if n <= 65535 else 5)

    def uint_size(n):
        return 1 if n <= 127 else 2 if n <= 255 else 3 if n <= 65535 else 5

    data = (rows * columns * width + 7) // 8
    return (12 + 1 + bin_size(16) + bin_size(16 * rows) + uint_size(columns)
            + uint_size(wire_bits) + uint_size(width) + bin_size(data))


def lift(x, r, bits=8):
    """Unsigned pad; signed input. d is signed wrap, not merely unsigned carry."""
    base = 1 << bits
    if not -base // 2 <= x < base // 2 or not 0 <= r < base:
        raise ValueError("input/pad outside declared domain")
    u, sign = x % base, int(x < 0)
    z = (u + r) % base
    carry = (u + r) // base
    biased = (x + base // 2 + r) % base
    bias_carry = (x + base // 2 + r) // base
    return z, carry, sign, carry - sign, biased, bias_carry


def tv(a, b):
    na, nb = sum(a.values()), sum(b.values())
    return sum((abs(Fraction(a[k], na) - Fraction(b[k], nb))
                for k in a.keys() | b.keys()), Fraction()) / 2


def algebra():
    count = 0
    for x in range(-128, 128):
        for r in range(256):
            z, c, s, d, zb, t = lift(x, r)
            assert x == z - r + 256 * d
            assert x == zb - r - 128 + 256 * t
            assert t == d + int(z >= 128)
            assert d == (x + r) // 256 == c - s
            count += 1
    rng = random.Random(681)
    outputs = 0
    # Independent oracle computes clear integer products, never using lift.
    for width in (16, 24, 32):
        for _ in range(128):
            x = [rng.randrange(-128, 128) for _ in range(5)]
            r = [rng.randrange(256) for _ in x]
            w = [[rng.randrange(-128, 128) for _ in x] for _ in range(3)]
            opening = [lift(a, b) for a, b in zip(x, r)]
            for row in w:
                oracle = sum(a * b for a, b in zip(row, x)) % (1 << width)
                actual = sum(a * (z - pad + 256 * d)
                             for a, pad, (z, _, _, d, _, _) in zip(row, r, opening)) % (1 << width)
                assert actual == oracle
                outputs += 1
    return {"exhaustive_byte_pairs": count, "independent_ring_outputs": outputs,
            "rings": [16, 24, 32], "scales_bias_rounding": "unchanged; no float/decoder execution"}


def leakage():
    # Provider sees byte opening and complete W*d; W=[1] is a legal odd i8 row.
    views = [Counter((lift(x, r, 3)[0], lift(x, r, 3)[3]) for r in range(8))
             for x in range(-4, 4)]
    hidden = [Counter(lift(x, r, 3)[0] for r in range(8)) for x in range(-4, 4)]
    patterns = []
    for r in range(8):
        table = [((z - r + 4) % 8 - 4 - 19) % 64 for z in range(8)]
        patterns.append(tuple((v - table[0]) % 64 for v in table))
    assert len(set(patterns)) == 8
    # Field257 sum and ring16 sum disagree, even with byte inputs and weights.
    field_witness = {"W": [127, 127], "x": [127, 127], "integer": 32258,
                     "field257_residue": 133, "ring16_residue": 32258}
    assert 32258 % 257 == 133
    return {"byte_only_max_tv": str(max(tv(a, b) for a in hidden for b in hidden)),
            "byte_plus_Wd_max_tv": str(max(tv(a, b) for a in views for b in views)),
            "domain": "signed3/full unsigned3 pad; W=[1]; complete repair contribution",
            "shared_pad_table_distinct_patterns": len(set(patterns)),
            "shared_pad_table_masks": 8,
            "inference_receives_pad_seed": "expands r; x=centered(z-r,8), exact disclosure",
            "reuse": {"x": [0, 1], "r": 17, "z": [17, 18], "difference": 1},
            "prime_field_substitution": field_witness}


def ideal_vector_ot(w, x, pads, width):
    """Functional ideal-OT specification, not an implementation/security proof.

    Sender masks each whole vector independently; receiver gets one vector per
    input bit. Sender finally sends aggregate mask vector. No W work at client.
    """
    ring = 1 << width
    receiver = [0] * len(w)
    sender = [0] * len(w)
    for i, xi in enumerate(x):
        for bit in range(8):
            coeff = -(1 << bit) if bit == 7 else 1 << bit
            choice = (xi % 256 >> bit) & 1
            pad = pads[i * 8 + bit]
            for j, row in enumerate(w):
                messages = ((-pad[j]) % ring, (coeff * row[i] - pad[j]) % ring)
                receiver[j] = (receiver[j] + messages[choice]) % ring
                sender[j] = (sender[j] + pad[j]) % ring
    return [(a + b) % ring for a, b in zip(receiver, sender)]


def ideal_biased_byte_ot(w, x, r, width, rng):
    """Functional optimized byte consumer in ideal-OT model; no crypto calls."""
    ring = 1 << width
    opening = [lift(a, b) for a, b in zip(x, r)]
    receiver = [0] * len(w)
    sender = [sum(a * (item[4] - 128) for a, item in zip(row, opening)) % ring for row in w]
    for i, ri in enumerate(r):
        # Eight unsigned-pad bits with negative coefficients, one biased carry.
        choices = [(k, -1, (ri >> k) & 1) for k in range(8)] + [(8, 1, opening[i][5])]
        for shift, sign, choice in choices:
            quotient = 1 << (width - shift)
            for j, row in enumerate(w):
                pad = rng.randrange(quotient)
                messages = ((-pad) % quotient, (sign * row[i] - pad) % quotient)
                receiver[j] = (receiver[j] + (messages[choice] << shift)) % ring
                sender[j] = (sender[j] + (pad << shift)) % ring
    return [(a + b) % ring for a, b in zip(receiver, sender)]


def ot_checks():
    rng = random.Random(619)
    checked = 0
    for width in (16, 24, 32):
        w = [[127, -128], [-3, 2], [1, 1]]
        for a in (-128, -127, -1, 0, 1, 126, 127):
            for b in (-128, -1, 0, 127):
                pads = [[rng.randrange(1 << width) for _ in w] for _ in range(16)]
                result = ideal_vector_ot(w, [a, b], pads, width)
                assert result == [sum(wi * xi for wi, xi in zip(row, [a, b])) % (1 << width) for row in w]
                byte_result = ideal_biased_byte_ot(w, [a, b], [rng.randrange(256), rng.randrange(256)], width, rng)
                assert byte_result == result
                checked += len(w)
    return {"outputs": checked, "status": "functional ideal-OT construction only",
            "optimized_biased_byte_outputs": checked,
            "client_work": "8*n*m additions; chosen vectors are not compact scalar ROT seeds",
            "distribution": "fresh independent uniform m-vector per bit OT; private choices assumed ideal"}


def cost_screen():
    frontier, geometry, narrow = load(FRONTIER), load(GEOMETRY), load(NARROW)
    control = frontier["control"]
    report = next(r["execution"]["details"]["report"] for r in load(CONTROL)["results"]
                  if r["execution"]["case"]["name"] == "prepared/digit8")
    assert len(report["runs"]) == 2
    assert [r["tokens"]["input_tokens"] for r in report["runs"]] == [150, 150]
    assert [r["tokens"]["output_tokens"] for r in report["runs"]] == [8, 8]
    assert report["runs"][1]["privacy"]["generation_prefill_online_body_bytes"] == 0
    assert geometry["source"]["body_fingerprint"] == control["body_fingerprint"]
    exact = {s["stage"]: s for s in narrow["checkpoint_geometry"]["stages"]}
    assert narrow["checkpoint_geometry"]["lock"]["body_fingerprint"] == control["body_fingerprint"]
    last_layer = max(int(s["stage"].split(".")[1]) for s in geometry["stages"])
    cold_prefill = report["runs"][0]["tokens"]["input_tokens"]
    decode = sum(r["network"]["decode_output_tokens"] for r in report["runs"])
    stages = []
    for stage in geometry["stages"]:
        name = stage["stage"]
        m, n = stage["shape"]
        # Saved ledger plus source terminal pruning: cold 150+7, cached repeat 7.
        terminal = int(name.split(".")[1]) == last_layer and "qkv_proj" not in name
        prefill = 1 if terminal else cold_prefill
        rows = prefill + decode
        x = exact[name]
        assert x["shape"] == [m, n] and x["weight_sha256"] == stage["weight_sha256"]
        k = x["stage_bits"]
        widths = x["row_residue_width_histogram"]
        assert sum(widths.values()) == m
        output_bits = sum(int(h) * count for h, count in widths.items())
        old = (rows * n * k + 7) // 8
        new = rows * n
        output = (rows * output_bits + 7) // 8
        traffic = control["stages"][name]
        residual = {"input": traffic["client->inference"] - old,
                    "output": traffic["inference->client"] - output,
                    "correction": traffic["preparation->inference"] - output}
        assert min(residual.values()) >= 0
        inner = request_frame_size(prefill, n, k, stage["ring_bits"]) + decode * request_frame_size(1, n, k, stage["ring_bits"])
        byte_inner = request_frame_size(prefill, n, 8, stage["ring_bits"]) + decode * request_frame_size(1, n, 8, stage["ring_bits"])
        assert inner <= traffic["client->inference"]
        # Optimistic selected-vector floor for ideal OT: 8 scalar bits each
        # select an entire output vector. Actual two-message wrapper doubles it.
        vector_floor = rows * n * output_bits
        # Stronger biased layout needs just one carry bit. Public factors 2^k
        # let each r-bit OT use h-k bits and the carry OT h-8 bits per row.
        # Sum k=0..7(h-k)+(h-8) = 9h-36. All fixture h exceed eight.
        assert min(map(int, widths)) > 8
        carry_floor = (rows * n * (9 * output_bits - 36 * m) + 7) // 8
        stages.append({"stage": name, "shape": [m, n], "rows": rows,
                       "stage_bits": k, "original_ring_bits": stage["ring_bits"],
                       "weight_sha256": stage["weight_sha256"],
                       "old_input_payload": old, "byte_input_payload": new,
                       "output_payload": output, "retained_body_overhead": residual,
                       "source_schema_inner_input_frames": inner,
                       "proposed_byte_inner_input_frames": byte_inner,
                       "outer_envelope_residual": traffic["client->inference"] - inner,
                       "selected_vector_ot_floor": vector_floor,
                       "selected_Wr_and_carry_ot_floor": carry_floor,
                       "body_macs": rows * m * n})
    totals = {key: sum(s[key] for s in stages) for key in (
        "old_input_payload", "byte_input_payload", "output_payload", "selected_vector_ot_floor",
        "selected_Wr_and_carry_ot_floor", "body_macs")}
    assert sum(s["rows"] for s in stages) == report["prepared_material_accounting"]["counts"]["claimed"] == 15297
    assert set(control["stages"]) == {s["stage"] for s in stages}
    classes = control["classes"]
    assert sum(classes.values()) == control["pair_bytes"]
    arithmetic = classes["inputs"] + classes["outputs"] + classes["corrections"]
    # Optimistic removal of *all* correction bodies, including their metadata.
    saving = classes["corrections"] + totals["old_input_payload"] - totals["byte_input_payload"]
    pair = control["pair_bytes"]
    recurring = pair - classes["public_bundle"]
    horizons = []
    for h in (1, 2, 4, 16, 1024):
        baseline = classes["public_bundle"] + h * recurring
        minimum = baseline - h * saving
        horizons.append({"pairs": h, "matched_control_bytes": baseline,
                         "optimistic_candidate_bytes": minimum,
                         "maximum_saving_fraction": str(Fraction(h * saving, baseline)),
                         "helper_and_amortized_bootstrap_budget_per_pair": {
                             str(t): str((baseline * (1 - t) - minimum) / h)
                             for t in (Fraction(2, 5), Fraction(1, 2))}})
    tables = []
    for b in (4, 8, 12):
        storage = adds = 0
        for s in stages:
            m, n = s["shape"]
            blocks = [min(b, n - i) for i in range(0, n, b)]
            storage += 2 * m * sum(1 << k for k in blocks)
            adds += 9 * s["rows"] * m * len(blocks)
        tables.append({"block_bits": b, "public_table_bytes": storage,
                       "fresh_Wr_and_carry_lane_additions": adds,
                       "addition_count_over_dense_mac_count": str(Fraction(adds, totals["body_macs"])),
                       "private_selection": "client-local only; public lookup indices disclose pads/carries"})
    return {"control_pair_bytes": pair, "classes": classes, "arithmetic_subtotal": arithmetic,
            "control_body_fingerprint": control["body_fingerprint"],
            "source_derived_rows": {"ordinary_stages": 93, "rows_each": 164,
                                    "terminal_stages": 3, "terminal_rows_each": 15,
                                    "total_claimed_issued_reserved": 15297},
            "totals": totals, "optimistic_removed_bytes": saving,
            "input_frame_transport_overhead_retained": classes["inputs"] - totals["old_input_payload"],
            "arithmetic_only_helper_budget_50_percent": str(Fraction(arithmetic, 2) - (arithmetic - saving)),
            "horizons": horizons, "subset_layouts": tables, "stages": stages,
            "direct_chosen_message_vector_ot_both_sender_payloads": 2 * totals["selected_vector_ot_floor"],
            "standard_ROT_private_choice_switch_bits": 8 * totals["byte_input_payload"],
            "biased_byte_ROT_private_choice_switch_bits": 9 * totals["byte_input_payload"],
            "byte_Wr_carry_selected_vector_floor_gap_to_cold_50_percent": str(
                Fraction(pair, 2) - saving + totals["selected_Wr_and_carry_ot_floor"]),
            "steady_matched_helper_budget_50_percent": str(Fraction(saving) - Fraction(recurring, 2)),
            "unknown_not_zero": ["base OT both directions and role setup", "silent expansion parameters/renewal",
                                  "public compiled programs/delivery", "verification if selected",
                                  "new authenticated frames", "failed/unused fresh material", "client CPU/RSS", "physical wire"],
            "bootstrap_formula": "candidate=P_new+H*(recurring-saving+helper+fresh+failure+verify+frames)+ceil(H/L)*B; matched control=P+H*recurring; B includes both directions, L justified security capacity"}


def build_report():
    paths = [FRONTIER, CONTROL, NARROW, GEOMETRY,
             "scripts/probe_bootstrap_ingress.py", "tests/test_bootstrap_ingress_probe.py",
             "docs/research/bootstrap-ingress-reference.md",
             "python/pllm/runtime/stage_protocol.py", "python/pllm/runtime/semantic_stages.py",
             "python/pllm/runtime/prepared_stage.py", "python/pllm/runtime/model_binding.py",
             "scripts/probe_narrow_ingress.py", "scripts/probe_all_link_frontier.py",
             "scripts/network_algebra_hypotheses.py", "scripts/probe_correlation_compiler.py",
             "scripts/probe_network_first_principles.py", "docs/research/network-algebra.md",
             "docs/research/network-first-principles.md", "docs/plans/all-link-next-five.md",
             "docs/research/correlation-compiler-reference.md", "docs/research/secret-replication-reference.md"]
    return {"schema": "pllm.bootstrap_ingress_probe.v1", "complete": True,
            "scope": "bounded synthetic algebra plus saved checkpoint/source cost enumeration; no SDK activation",
            "decision": "reject tested vector-OT/table layouts; circuit-specific ring PCF remains missing concrete consumer/cost, not no-go",
            "source_hashes": {p: digest(ROOT / p) for p in paths},
            "runtime": {"version": sys.version, "executable": sys.executable,
                        "executable_sha256": digest(Path(sys.executable).resolve())},
            "algebra": algebra(), "counterexamples": leakage(), "ideal_ot": ot_checks(),
            "cost": cost_screen(), "primary_sources": PRIMARY,
            "protected_primitives_executed": 0, "full_wire_bytes": None,
            "client_cpu_seconds": None, "client_peak_rss_bytes": None,
            "measured_low_memory_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report()
    # Never replace another worker's evidence, including our previous report.
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
