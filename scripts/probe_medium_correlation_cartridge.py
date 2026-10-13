#!/usr/bin/env python3
"""Small exact-ring algebra and saved-geometry ledgers, not a secure PCG."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_SOURCES = (
    "docs/evidence/research-all-link-frontier.json",
    "docs/evidence/research-narrow-ingress.json",
    "docs/evidence/research-bootstrap-ingress.json",
    "docs/evidence/network-algebra-qwen25-2026-10-06.json",
)
OWNED = (
    "scripts/probe_medium_correlation_cartridge.py",
    "tests/test_medium_correlation_cartridge_probe.py",
    "docs/research/medium-correlation-cartridge-reference.md",
)
HISTORICAL = (
    "docs/evidence/research-matrix-triple-prefill.json",
    "docs/evidence/research-streaming-matrix-correlation.json",
    "docs/evidence/research-bootstrap-cartridge.json",
    "docs/research/silent-linear-frontier-reference.md",
    "docs/research/streaming-matrix-correlation-reference.md",
    "docs/research/matrix-triple-prefill-reference.md",
    "docs/research/bootstrap-cartridge-reference.md",
    "scripts/probe_streaming_matrix_correlation.py",
    "scripts/probe_matrix_triple_prefill.py",
    "docs/research/compact-setup-fulltext-review.md",
    "docs/evidence/compact-setup-fulltext-review.json",
    "papers/2025-1223.pdf", "papers/2026-2298.pdf",
    "target/compact-setup-fulltext/2025-1223.txt",
    "target/compact-setup-fulltext/2026-2298.txt",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate JSON key")
    return obj


def load(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def validate_stages(stages):
    names = [s["stage"] for s in stages]
    if len(names) != len(set(names)) or len(stages) != 96:
        raise ValueError("stage identities")
    for s in stages:
        if (len(s["shape"]) != 2 or any(type(v) is not int or v <= 0 for v in s["shape"])
                or type(s["rows"]) is not int or s["rows"] <= 14):
            raise ValueError("invalid stage geometry")
        m, n = s["shape"]
        k, h = s["original_ring_bits"], s["stage_bits"]
        widths = s["row_residue_width_histogram"]
        if (type(k) is not int or k not in (24, 32) or type(h) is not int
                or not widths or any(not key.isdecimal() or not 0 < int(key) <= k
                                     or type(v) is not int or v <= 0 for key, v in widths.items())
                or sum(widths.values()) != m or max(map(int, widths)) != h):
            raise ValueError("invalid residue geometry")
        if s["body_macs"] != m * n * s["rows"]:
            raise ValueError("MAC geometry mismatch")
    return stages


def geometry(root=ROOT):
    """Check frozen historical hash locks BEFORE trusting its geometry validator."""
    record = load(root / "docs/evidence/research-matrix-triple-prefill.json")
    for path in (*GEOMETRY_SOURCES, "scripts/probe_matrix_triple_prefill.py"):
        if sha(root / path) != record["source_hashes"][path]:
            raise ValueError(f"historical source hash mismatch: {path}")
    manifest = load(root / "docs/evidence/compact-setup-fulltext-review.json")
    for paper in manifest["papers"]:
        if paper["id"] in ("2025/1223", "2026/2298"):
            path = root / paper["path"]
            if sha(path) != paper["sha256"] or path.stat().st_size != paper["bytes"]:
                raise ValueError("PDF source hash mismatch")
    spec = importlib.util.spec_from_file_location("historical_medium_geometry", root / "scripts/probe_matrix_triple_prefill.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    control, stages = module.validate_geometry(*(load(root / p) for p in GEOMETRY_SOURCES))
    return control, validate_stages(stages)


def mv(w, x, bits):
    if (type(bits) is not int or not 1 <= bits <= 32 or not w or not x
            or any(len(row) != len(x) for row in w)
            or any(type(v) is not int for row in w for v in row)
            or any(type(v) is not int for v in x)):
        raise ValueError("invalid algebra geometry")
    return [sum(a * b for a, b in zip(row, x, strict=True)) % (1 << bits) for row in w]


def direct_control(w, x, r, q, bits, widths):
    """Fixtures supplied explicitly; no PRG, Gen or secret keys implemented."""
    if len(q) != len(w) or len(widths) != len(w) or any(type(h) is not int or not 1 <= h <= bits for h in widths):
        raise ValueError("output geometry")
    wr = mv(w, r, bits)
    z = [(a - b) % (1 << bits) for a, b in zip(x, r, strict=True)]
    if len(z) != len(r):
        raise ValueError("input geometry")
    s = [(a - b) % (1 << h) for a, b, h in zip(wr, q, widths, strict=True)]
    wz = mv(w, z, bits)
    reply = [(a + b) % (1 << h) for a, b, h in zip(wz, q, widths, strict=True)]
    return [(a + b) % (1 << h) for a, b, h in zip(reply, s, widths, strict=True)]


def bounded_column_control(w, xs, bits, capacity, rng):
    """Independent ideal-dealer algebra only, fresh B columns in ONE A triple."""
    if type(capacity) is not int or capacity <= 0 or not xs or len(xs) > capacity:
        raise ValueError("column capacity")
    if not w or not w[0]:
        raise ValueError("invalid algebra geometry")
    mv(w, [0] * len(w[0]), bits)
    m, n, modulus = len(w), len(w[0]), 1 << bits
    a = [[rng.randrange(modulus) for _ in range(n)] for _ in range(m)]
    d = [[(v - mask) % modulus for v, mask in zip(row, ar, strict=True)] for row, ar in zip(w, a, strict=True)]
    got = []
    for x in xs:
        bc, bi = ([rng.randrange(modulus) for _ in range(n)] for _ in range(2))
        ab = mv(a, [(c + i) % modulus for c, i in zip(bc, bi, strict=True)], bits)
        cc = [rng.randrange(modulus) for _ in range(m)]
        ci = [(v - c) % modulus for v, c in zip(ab, cc, strict=True)]
        s = [(c + v) % modulus for c, v in zip(cc, mv(d, bc, bits), strict=True)]
        q = [(c - v) % modulus for c, v in zip(ci, mv(a, bi, bits), strict=True)]
        z = [(v - c) % modulus for v, c in zip(x, bc, strict=True)]
        got.append([(v + mask + pad) % modulus for v, mask, pad in zip(mv(w, z, bits), q, s, strict=True)])
        if q != [(v - pad) % modulus for v, pad in zip(mv(w, bc, bits), s, strict=True)]:
            raise AssertionError("cartridge relation")
    return got


def algebra():
    rng = random.Random(12232298)
    w = [[-128, 127, -3], [0, 1, 7], [8, -16, 32]]
    checks = 0
    for bits in (3, 16, 24, 32):
        xs = [[rng.randrange(-128, 128) for _ in range(3)] for _ in range(7)]
        xs[0] = [-128, 127, -128]
        outputs = bounded_column_control(w, xs, bits, 7, rng)
        for x, got in zip(xs, outputs, strict=True):
            # Oracle uses signed integers, not mv or triple routines.
            expected = [sum(row[i] * x[i] for i in range(3)) for row in w]
            if got != [v % (1 << bits) for v in expected]:
                raise AssertionError("signed integer oracle")
            r = [rng.randrange(1 << bits) for _ in range(3)]
            q = [rng.randrange(1 << bits) for _ in range(3)]
            widths = [bits, max(1, bits - 1), max(1, bits - 3)]
            if direct_control(w, x, r, q, bits, widths) != [v % (1 << h) for v, h in zip(expected, widths, strict=True)]:
                raise AssertionError("quotient oracle")
            checks += 3
    # Full inference view (q,z) is uniform for independent r,q in scalar toy.
    views = [{(q, (x - r) % 8) for r in range(8) for q in range(8)} for x in range(8)]
    assert all(view == views[0] and len(view) == 64 for view in views)
    return {"bounded_column_outputs": checks, "direct_quotient_outputs": checks,
            "exhaustive_inference_views": 512, "rings": [3, 16, 24, 32],
            "secure_pcg_implemented": False, "toy_rng_only": True}


def packed(bits):
    if type(bits) is not int or bits < 0:
        raise ValueError("invalid bit count")
    return (bits + 7) // 8


def totals(stages):
    result = {name: 0 for name in ("weight_i8_bytes", "prefill_body_macs", "decode_body_macs",
              "full_ring_ingress_bytes", "full_ring_reply_bytes", "encoded_ingress_bytes",
              "residue_reply_bytes", "full_ring_client_generated_bytes", "residue_inference_generated_bytes")}
    for s in stages:
        m, n = s["shape"]
        rows, k = s["rows"], s["original_ring_bits"]
        result["weight_i8_bytes"] += m * n
        result["prefill_body_macs"] += m * n * (rows - 14)
        result["decode_body_macs"] += m * n * 14
        result["full_ring_ingress_bytes"] += packed(rows * n * k)
        result["full_ring_reply_bytes"] += packed(rows * m * k)
        result["encoded_ingress_bytes"] += s["old_input_payload"]
        result["residue_reply_bytes"] += s["output_payload"]
        result["full_ring_client_generated_bytes"] += packed(rows * (m + n) * k)
        result["residue_inference_generated_bytes"] += s["output_payload"]
    return result


def logical_resources(stages):
    """Packed arrays only. Not RSS, allocator size, CPU time, or RAM admission."""
    return {"persistent_public_weight_bytes": sum(m * n for m, n in (s["shape"] for s in stages)),
            "private_program_root_bytes_all_stages": 96 * 64,
            "public_manifest_hypothesis_bytes": 96 * 96,
            "maximum_one_stage_weight_bytes": max(math.prod(s["shape"]) for s in stages),
            "maximum_one_row_packed_five_vector_buffers_bytes": max(
                packed((2 * s["shape"][1] + 3 * s["shape"][0]) * s["original_ring_bits"]) for s in stages),
            "client_PRG_full_ring_output_bytes_per_pair": sum(
                packed(s["rows"] * sum(s["shape"]) * s["original_ring_bits"]) for s in stages),
            "inference_PRG_full_ring_output_bytes_per_pair": sum(
                packed(s["rows"] * s["shape"][0] * s["original_ring_bits"]) for s in stages),
            "client_PRG_ring_lanes_per_pair": sum(s["rows"] * sum(s["shape"]) for s in stages),
            "client_vector_add_sub_lanes_per_pair": sum(
                s["rows"] * (s["shape"][1] + 2 * s["shape"][0]) for s in stages),
            "SDK_state_workspace_live_queue_renewal_overlap_bytes": None,
            "whole_client_peak_RSS_bytes": None, "CPU_seconds": None}


def ledger(control, horizon, removed, additional_public, issuance, recurring_extra):
    """Exact all-link horizon. issuance includes INITIAL issue and every renewal."""
    if (type(horizon) is not int or horizon <= 0
            or any(type(v) is not int or v < 0 for v in (removed, additional_public, issuance, recurring_extra))):
        raise ValueError("invalid ledger")
    public = control["classes"]["public_bundle"]
    recurring = control["pair_bytes"] - public
    base = public + horizon * recurring
    candidate = base - horizon * removed + additional_public + issuance + horizon * recurring_extra
    # Target 20% is tested with integers, not rounded fractions.
    return {"pairs": horizon, "control_bytes": base, "candidate_conditional_bytes": candidate,
            "additional_public_bytes": additional_public, "both_role_issuance_bytes": issuance,
            "new_recurring_bytes_per_pair": recurring_extra, "saved_bytes": base - candidate,
            "remaining_20pct_budget_times_five": 4 * base - 5 * candidate,
            "conditional_20pct_survives": 5 * candidate <= 4 * base,
            "unknown_auth_frames_failure_bytes": None}


def direct_projection(control, stages, horizon, capacity=65536, quotient=True, burned_per_stage=0,
                      weights_preexisting_client=False):
    if type(capacity) is not int or not 0 < capacity <= 2**64:
        raise ValueError("invalid finite capacity")
    if type(burned_per_stage) is not int or burned_per_stage < 0:
        raise ValueError("invalid burned attempts")
    t = totals(stages)
    # Two roots Client, shared q root Inference: three 32-byte deliveries per stage epoch.
    epochs = sum((horizon * s["rows"] + burned_per_stage + capacity - 1) // capacity for s in stages)
    manifests = 2 * 96 * epochs  # both-role stage-epoch delivery hypothesis
    width_extra = 0 if quotient else (t["full_ring_ingress_bytes"] + t["full_ring_reply_bytes"]
                                     - t["encoded_ingress_bytes"] - t["residue_reply_bytes"])
    new_weight_bytes = 0 if weights_preexisting_client else t["weight_i8_bytes"]
    cell = ledger(control, horizon, 139060689, new_weight_bytes + manifests, 96 * epochs, width_extra)
    cell.update({"candidate": "local W*r with separate private-r/shared-q PRG roots",
                 "quotient_codec_conditional": quotient, "capacity_per_stage_epoch": capacity,
                 "capacity_status": "ledger hypothesis, not paper-certified concrete security",
                 "stage_epochs": epochs, "unused_indices_at_final_retirement": sum(
                     ((horizon * s["rows"] + burned_per_stage + capacity - 1) // capacity) * capacity
                     - horizon * s["rows"] - burned_per_stage for s in stages),
                 "burned_attempts": 96 * burned_per_stage,
                 "public_manifest_deliveries_all_stage_epochs_bytes": manifests,
                 "failed_attempt_body_and_expansion_costs": None,
                 "extra_client_Wr_macs": horizon * (t["prefill_body_macs"] + t["decode_body_macs"]),
                 "extra_client_Wr_macs_prefill": horizon * t["prefill_body_macs"],
                 "extra_client_Wr_macs_decode": horizon * t["decode_body_macs"],
                 "client_root_delivery_bytes": 64 * epochs,
                 "inference_root_delivery_bytes": 32 * epochs,
                 "public_weight_delivery_client_bytes": new_weight_bytes,
                 "weights_preexisting_client_conditional": weights_preexisting_client,
                 "earlier_checkpoint_delivery_if_preexisting_not_in_control_bytes": None,
                 "public_weight_inference_already_present_bytes": t["weight_i8_bytes"],
                 "extra_if_new_inference_weight_distribution_required_bytes": t["weight_i8_bytes"],
                 "remote_original_body_fraction_prefill": 1, "remote_original_body_fraction_decode": 1,
                 "PRG_expansion_CPU_seconds": None, "whole_client_peak_RSS_bytes": None,
                 "qualifies_Medium": False})
    return cell


def bounded_projection(control, stages, horizon, columns=128, square_field=False):
    """Ring rectangular optimistic control OR published field64 square geometry."""
    if type(columns) is not int or columns <= 0:
        raise ValueError("invalid columns")
    dbytes = ingress = reply = triples = generated = unused = client_macs = padded = 0
    weights = 0
    for s in stages:
        m, n = s["shape"]
        rows, k = horizon * s["rows"], s["original_ring_bits"]
        batches = (rows + columns - 1) // columns
        residue_bits = sum(int(h) * count for h, count in s["row_residue_width_histogram"].items())
        weights += m * n
        if square_field:
            mb, nb = (m + columns - 1) // columns, (n + columns - 1) // columns
            count = mb * nb * batches
            triples += count
            dbytes += 2 * count * columns**2 * 8
            # Different block triples own different B: no free cross-block sharing.
            ingress += mb * nb * rows * columns * 8
            reply += mb * nb * rows * columns * 8
            generated += 6 * count * columns**2 * 8
            unused += count * columns - mb * nb * rows
            client_macs += mb * nb * rows * columns**2
            padded += count * columns**3
        else:
            triples += batches
            dbytes += 2 * batches * packed(n * residue_bits)
            ingress += horizon * s["old_input_payload"]
            reply += horizon * s["output_payload"]
            generated += 2 * (packed(batches * m * n * k)
                              + packed(batches * columns * (n + m) * k))
            unused += batches * columns - rows
            client_macs += rows * m * n
            padded += batches * columns * m * n
    t = totals(stages)
    # Known all-link retained baseline excludes existing input/reply/correction payloads.
    existing_io = horizon * (t["encoded_ingress_bytes"] + t["residue_reply_bytes"])
    cell = ledger(control, horizon, 139060689, weights, dbytes, 0)
    cell["candidate_conditional_bytes"] += ingress + reply - existing_io
    cell["saved_bytes"] = cell["control_bytes"] - cell["candidate_conditional_bytes"]
    cell["remaining_20pct_budget_times_five"] = 4 * cell["control_bytes"] - 5 * cell["candidate_conditional_bytes"]
    cell["conditional_20pct_survives"] = cell["remaining_20pct_budget_times_five"] >= 0
    cell.update({"candidate": "published field64 square adapter" if square_field else "optimistic exact-ring rectangular adapter",
                 "columns_per_triple": columns, "triples_or_rectangular_batches": triples,
                 "D_both_role_deliveries_bytes": dbytes, "ingress_bytes": ingress, "reply_bytes": reply,
                 "both_role_generated_matrix_bytes": generated, "unused_columns_retired": unused,
                 "extra_client_dense_adaptation_macs": client_macs,
                 "extra_inference_dense_adaptation_macs": client_macs,
                 "padded_each_role_adaptation_macs": padded,
                 "remote_original_body_fraction_prefill": 1, "remote_original_body_fraction_decode": 1,
                 "whole_client_peak_RSS_bytes": None, "qualifies_Medium": False,
                 "ring_construction_supplied": False, "setup_omitted_not_zero": True})
    if square_field:
        cell.update({name: None for name in (
            "PCG_epochs_4096_triples", "unused_PCG_triples_at_retirement",
            "estimated_dealer_both_role_seed_delivery_bytes",
            "estimated_distributed_setup_MiB_combined", "natural_per_party_workspace_bytes",
            "candidate_with_estimated_dealer_seed_delivery_bytes",
            "candidate_with_estimated_distributed_setup_bytes",
            "dealer_delivery_client_estimate_bytes", "dealer_delivery_inference_estimate_bytes",
            "field_to_exact_ring_conversion_bytes_CPU")})
        cell["published_setup_workspace_parameter_point"] = columns == 128
        cell["setup_modes_not_additive"] = True
        if columns != 128:
            cell["candidate"] = "field64 square dimension sensitivity"
            return cell
        # Table 4 rounded amortized estimates, not exact key serialized lengths.
        pcg_epochs = (triples + 4095) // 4096
        cell.update({"PCG_epochs_4096_triples": pcg_epochs,
                     "unused_PCG_triples_at_retirement": pcg_epochs * 4096 - triples,
                     "estimated_dealer_both_role_seed_delivery_bytes": 2 * pcg_epochs * 4096 * 5453,
                     "estimated_distributed_setup_MiB_combined": pcg_epochs * 249.8,
                     "setup_modes_not_additive": True,
                     "natural_per_party_workspace_bytes": 2 * 2**30,
                     "field_to_exact_ring_conversion_bytes_CPU": None})
        seed_estimate = cell["estimated_dealer_both_role_seed_delivery_bytes"]
        distributed_estimate = (pcg_epochs * 1249 * 2**20 + 4) // 5
        cell["candidate_with_estimated_dealer_seed_delivery_bytes"] = cell["candidate_conditional_bytes"] + seed_estimate
        cell["candidate_with_estimated_distributed_setup_bytes"] = cell["candidate_conditional_bytes"] + distributed_estimate
        cell["dealer_delivery_client_estimate_bytes"] = seed_estimate // 2
        cell["dealer_delivery_inference_estimate_bytes"] = seed_estimate // 2
    return cell


def report():
    paths = (*OWNED, *GEOMETRY_SOURCES, *HISTORICAL)
    before = {p: sha(ROOT / p) for p in paths}
    control, stages = geometry()
    t = totals(stages)
    result = {"schema": "pllm.medium_correlation_cartridge.v1", "complete": True,
              "scope": "bounded independent algebra and deterministic historical application-body accounting; no crypto or RAM measurement",
              "source_sha256": before, "algebra": algebra(), "totals_per_pair": t,
              "local_cartridge_logical_resources": logical_resources(stages),
              "direct_controls": [direct_projection(control, stages, h, quotient=q)
                                  for q in (True, False) for h in (1, 4, 6, 10, 16, 64, 1024)],
              "already_paid_local_weight_sensitivity": [
                  direct_projection(control, stages, h, weights_preexisting_client=True)
                  for h in (1, 6, 1024)],
              "bounded_column_controls": [bounded_projection(control, stages, h, b, f)
                                          for f in (False, True) for b in (128, 256) for h in (1, 16, 1024)],
              "Medium": {"whole_client_limit_bytes": 2 * 2**30,
                         "hypothetical_workspace_bytes": [512 * 2**20, 2**30],
                         "program_memory_storage_and_RAM_are_distinct": True,
                         "measured_whole_client_RSS_bytes": None, "qualified": False},
              "fixed_W_programmable_construction": None,
              "decision": "No reviewed cheap-client fixed-W construction. Local generation is exact but dense; bounded columns amortize D only within fresh finite triples and still reject 20pct here.",
              "security_or_performance_claim": False,
              "unknowns": ["fixed-W masked-linear-map compiler", "concrete aggregate PRG/PCG security bound",
                           "authenticated manifests and keys", "failed/burned attempts", "rollback-safe index authority",
                           "whole-client peak including SDK and renewal overlap", "CPU and physical wire"]}
    if before != {p: sha(ROOT / p) for p in paths}:
        raise ValueError("source changed during probe")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "complete": True, "algebra": result["algebra"]}))


if __name__ == "__main__":
    main()
