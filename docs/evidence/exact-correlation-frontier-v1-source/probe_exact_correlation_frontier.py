"""Exact algebra and conditional ledgers only; no cryptography or model execution."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = {
    "docs/evidence/network-algebra-qwen25-2026-10-06.json": "8c276df98557694ae5c54ee97d7d42e78b3fa146a5bf272b3f1826fb0bb23e3e",
    "docs/evidence/research-narrow-ingress.json": "8c55c563422a38ddc0c6d0187379c38450b0702b4031912166e7f4bd4c7e105d",
    "docs/evidence/research-attested-linear-bootstrap.json": "f8ea98839aaf19ec698cc381b35fc92a7fe54e48477fdcf4929ea99fff66ad78",
    "docs/evidence/private-boundary-decode-confirmation-150x8.summary.json": "be59ead505959085152b1401ca7fb413b05760d128dd3967b4963421afccd975",
    "docs/evidence/private-boundary-decode-confirmation-150x64.summary.json": "6938389d3d5798ac4758d5a3edbdd0caa46d0218b0221dab4979496452168cb3",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mv(a, x, bits):
    """General ring matrix product, including ring-valued compiled coefficients."""
    require(1 <= bits <= 32 and a and x, "invalid ring/shape")
    require(all(len(row) == len(x) for row in a), "dimension mismatch")
    return tuple(sum(v * z for v, z in zip(row, x, strict=True)) % (1 << bits)
                 for row in a)


def image_matrix(w, a, bits):
    require(a and len(a) == len(w[0]), "dimension mismatch")
    columns = tuple(mv(w, column, bits) for column in zip(*a, strict=True))
    return tuple(zip(*columns, strict=True))


def sparse_image(w, a, u, noise, bits):
    """r=A*u+e; image=(W*A)*u+W*e. Noise has distinct coordinate/value pairs.

    This is a deterministic algebraic interface, NOT a secure noise sampler.
    """
    n, modulus = len(a), 1 << bits
    require(len({i for i, _ in noise}) == len(noise), "duplicate sparse index")
    require(all(type(i) is int and 0 <= i < n for i, _ in noise), "invalid sparse index")
    r = list(mv(a, u, bits))
    result = list(mv(image_matrix(w, a, bits), u, bits))
    for i, value in noise:
        r[i] = (r[i] + value) % modulus
        for j, row in enumerate(w):
            result[j] = (result[j] + row[i] * value) % modulus
    return tuple(r), tuple(result)


def reconstruct(w, x, r, image, q, bits):
    modulus = 1 << bits
    require(len(x) == len(r) and len(q) == len(image) == len(w), "dimension mismatch")
    z = tuple((v - mask) % modulus for v, mask in zip(x, r, strict=True))
    reply = tuple((v + correction) % modulus
                  for v, correction in zip(mv(w, z, bits), q, strict=True))
    s = tuple((v - correction) % modulus for v, correction in zip(image, q, strict=True))
    return tuple((v + pad) % modulus for v, pad in zip(reply, s, strict=True))


def residual(w, u, v, bits):
    """Compile W=U*V+E exactly; dense E is the universal, unhelpful fallback."""
    require(len(u) == len(w) and len(v[0]) == len(w[0]), "dimension mismatch")
    product = image_matrix(u, v, bits)
    return tuple(tuple((a - b) % (1 << bits) for a, b in zip(row, prow, strict=True))
                 for row, prow in zip(w, product, strict=True))


def circuit_image(u, v, e, r, bits):
    left, right = mv(u, mv(v, r, bits), bits), mv(e, r, bits)
    return tuple((a + b) % (1 << bits) for a, b in zip(left, right, strict=True))


def sparse_toy_tv():
    """Exact mod-4 chosen-input view distance for A=(1,1,1), one noise entry."""
    counts = []
    for x in ((0, 0, 0), (0, 1, 2)):
        view = Counter()
        for secret, position, error in itertools.product(range(4), range(3), range(4)):
            r = [secret] * 3
            r[position] = (r[position] + error) % 4
            view[tuple((a - b) % 4 for a, b in zip(x, r, strict=True))] += 1
        counts.append(view)
    a, b = counts
    return Fraction(sum(abs(a[k] - b[k]) for k in a.keys() | b.keys()), 2 * a.total())


def homomorphic_leak(secret, x, bits):
    """Transparent two-word linear key evaluator H(j)=(1,j), three fresh labels."""
    modulus = 1 << bits
    masks = [(secret[0] + j * secret[1]) % modulus for j in range(3)]
    openings = [(a - b) % modulus for a, b in zip(x, masks, strict=True)]
    return (openings[0] - 2 * openings[1] + openings[2]) % modulus


def packed(bits):
    return (bits + 7) // 8


def geometry(root=ROOT):
    docs = {}
    for path, expected in LOCKS.items():
        raw = (root / path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, f"source hash drift: {path}")
        docs[path] = json.loads(raw)
    algebra = docs["docs/evidence/network-algebra-qwen25-2026-10-06.json"]
    narrow = docs["docs/evidence/research-narrow-ingress.json"]["checkpoint_geometry"]
    demand = docs["docs/evidence/research-attested-linear-bootstrap.json"]["budget"]
    ni = {s["stage"]: s for s in narrow["stages"]}
    di = {s["stage"]: s for s in demand["stages"]}
    require(len(ni) == len(di) == len(algebra["stages"]) == 96, "stage count mismatch")
    require(algebra["source"]["body_fingerprint"] == narrow["lock"]["body_fingerprint"],
            "body identity mismatch")
    stages = []
    for s in algebra["stages"]:
        n, d = ni[s["stage"]], di[s["stage"]]
        require(s["shape"] == n["shape"] == d["shape"], "shape mismatch")
        require(s["weight_sha256"] == n["weight_sha256"] == d["weight_sha256"],
                "weight identity mismatch")
        m, width = s["shape"]
        hist = n["row_residue_width_histogram"]
        require(sum(hist.values()) == m and max(map(int, hist)) <= s["ring_bits"],
                "invalid output widths")
        stages.append({"stage": s["stage"], "m": m, "n": width, "bits": s["ring_bits"],
                       "input_bits": n["stage_bits"],
                       "output_bits": sum(int(b) * count for b, count in hist.items()),
                       "terminal": d["terminal_prefill"]})
    require(sum(s["terminal"] for s in stages) == 3, "terminal count mismatch")
    for outputs in (8, 64):
        summary = docs[f"docs/evidence/private-boundary-decode-confirmation-150x{outputs}.summary.json"]
        require(summary["complete"] and summary["identity"]["tokens"]["input_tokens"] == 150
                and summary["identity"]["tokens"]["output_tokens"] == outputs, "workload mismatch")
        require(summary["identity"]["body"] == algebra["source"]["body_fingerprint"],
                "workload body mismatch")
        rows = sum((1 if s["terminal"] else 150) + outputs - 1 for s in stages)
        require(all(w["preparation_rows"] == rows for w in
                    summary["groups"]["maintained"]["executed_work"]), "demand mismatch")
    return stages


def ledger(stages, outputs, *, dimension=128, noise=27, capacity=256, requests=1, burned=0,
           request_sized=False):
    """Specified projected stage protocol. Parameters are NOT security calibrated.

    All stages kept remote. One request is 150 prefill plus outputs-1 decode;
    three terminal-prefill stages use one prefill row. Burns are per stage.
    """
    require(outputs in (8, 64), "unsupported workload")
    require(all(type(v) is int and v > 0 for v in (dimension, capacity, requests)), "invalid count")
    require(type(noise) is int and noise >= 0 and type(burned) is int and burned >= 0,
            "invalid count")
    total = Counter()
    max_live = 0
    for s in stages:
        m, n, k = s["m"], s["n"], s["bits"]
        require(dimension <= n and noise <= n, "invalid sensitivity parameters")
        rows = requests * ((1 if s["terminal"] else 150) + outputs - 1)
        stage_capacity = ((1 if s["terminal"] else 150) + outputs - 1) if request_sized else capacity
        epochs = (rows + burned + stage_capacity - 1) // stage_capacity
        issued = epochs * stage_capacity
        # Every row is separately packed; no cross-row bit packing or free metadata.
        correction = packed(s["output_bits"])
        delivery = m * n + packed(k * m * dimension) + 64  # W, WA, A seed to both roles
        total.update({
            "successful_stage_rows": rows, "issued_stage_rows": issued,
            "epochs": epochs,
            "baseline_preparation_to_client": 128 * epochs,
            "baseline_preparation_to_inference": 64 * epochs + issued * correction,
            "candidate_preparation_to_client": 128 * epochs + delivery - 32,
            "candidate_preparation_to_inference": 96 * epochs + 32,
            "client_to_inference": rows * (40 + packed(n * s["input_bits"])),
            "inference_to_client": rows * (40 + correction),
            "new_public_delivery": delivery,
            "setup_wa_ring_macs": m * n * dimension,
            "baseline_preparation_ring_macs": issued * m * n,
            "worker_online_ring_macs": rows * m * n,
            "client_a_u_ring_macs": rows * n * dimension,
            "client_wa_u_ring_macs": rows * m * dimension,
            "client_sparse_w_e_ring_macs": rows * m * noise,
            "client_mask_and_reconstruction_additions": rows * (n + m),
            "client_image_and_noise_additions": rows * (m + noise),
            "client_public_a_expansion_bytes": rows * packed(k * n * dimension),
            "client_private_prg_value_bytes_minimum": rows * packed(k * (dimension + noise + m)),
            "worker_q_prg_bytes": rows * packed(k * m),
            "baseline_client_prg_bytes": rows * packed(k * (n + m)),
            "baseline_preparation_prg_bytes": issued * packed(k * (n + m)),
            "client_sparse_support_index_bytes_minimum": rows * packed(noise * (n - 1).bit_length()),
            "client_streamed_wa_read_bytes": rows * packed(k * m * dimension),
            "client_secret_index_w_read_bytes": rows * m * noise,
            "client_public_artifact_writes": delivery - 64,
            "preparation_setup_paged_w_reads": dimension * m * n,
            "baseline_preparation_paged_w_reads": issued * m * n,
            "worker_online_paged_w_reads": rows * m * n,
            "baseline_correction_tape_writes": issued * correction,
            "baseline_correction_tape_reads": rows * correction,
        })
        # Payload-only live vectors, not object/RSS bounds. Public matrices are streamed.
        max_live = max(max_live, packed(k * (3 * n + 4 * m + dimension + noise)) + 64)
    baseline = (total["baseline_preparation_to_client"] + total["baseline_preparation_to_inference"]
                + total["client_to_inference"] + total["inference_to_client"])
    candidate = (total["candidate_preparation_to_client"] + total["candidate_preparation_to_inference"]
                 + total["client_to_inference"] + total["inference_to_client"])
    return {"outputs": outputs, "requests": requests, "dimension_not_security_calibrated": dimension,
            "noise_not_security_calibrated": noise, "capacity": capacity, "burned_per_stage": burned,
            "capacity_policy": "per-stage one-request rows" if request_sized else "fixed Q",
            "known_counts": dict(total), "baseline_known_all_link_bytes": baseline,
            "candidate_known_all_link_bytes": candidate,
            "known_byte_delta": candidate - baseline,
            "additional_bytes_allowed_for_20pct_target": str(Fraction(4 * baseline, 5) - candidate),
            "payload_vector_workspace_bytes": max_live,
            "whole_client_limit_bytes": 256 << 20,
            "unknown": {k: None for k in (
                "sdk_state_transport_allocator_peak_bytes", "complete_client_peak_bytes",
                "renewal_and_active_stage_overlap_bytes", "public_stream_scratch_bytes",
                "private_support_sampler_rejection_bytes", "malicious_verification_bytes",
                "durable_authority_rpc_bytes", "failed_online_bytes", "tls_and_physical_wire_bytes",
                "common_checkpoint_and_original_public_distribution_bytes",
                "client_expansion_seconds", "worker_expansion_seconds", "setup_seconds",
                "external_storage_authentication_bytes", "multiquery_security_bits")},
            "scope": "specified application payload projection; unknowns excluded, not a serving win"}


def alternative_ledgers(stages, outputs, capacity=256):
    """Price the exact transparent-key fallback; leave unobserved circuit structure null."""
    counts = Counter()
    factor_only = 0
    for s in stages:
        m, n, k = s["m"], s["n"], s["bits"]
        rows = (1 if s["terminal"] else 150) + outputs - 1
        epochs = (rows + capacity - 1) // capacity
        counts.update({
            "preparation_to_client": epochs * (64 + packed(k * capacity * (n + m))),
            "preparation_to_inference": epochs * (64 + packed(k * capacity * m)),
            "setup_ring_macs": epochs * capacity * m * n,
            "client_key_tape_reads": rows * packed(k * (n + m)),
            "inference_key_tape_reads": rows * packed(k * m),
        })
        factor_only += packed(k * 128 * (m + n))
    base = ledger(stages, outputs)
    counts["client_to_inference"] = base["known_counts"]["client_to_inference"]
    counts["inference_to_client"] = base["known_counts"]["inference_to_client"]
    return {
        "outputs": outputs,
        "transparent_exact_key_fallback": {
            "capacity_and_key_dimension": capacity,
            "evaluator": "H(j)=unit vector j; independent ring-word key tapes",
            "known_counts": dict(counts),
            "known_all_link_bytes": sum(counts[k] for k in (
                "preparation_to_client", "preparation_to_inference",
                "client_to_inference", "inference_to_client")),
            "client_streamed_io_bytes": counts["client_key_tape_reads"],
            "complete_client_peak_bytes": None, "setup_seconds": None,
            "unknown_costs": base["unknown"],
            "break_even": "none against seeded baseline: client tape is additional, inference tape no smaller",
        },
        "exact_weight_circuit": {
            "rank_sensitivity_not_a_factorization": 128,
            "new_public_factor_bytes_excluding_residual_and_metadata": factor_only,
            "exact_residual_nonzeros": None, "new_public_residual_bytes": None,
            "preparation_to_client": None,
            "preparation_to_inference": base["known_counts"]["candidate_preparation_to_inference"] - 32 * len(stages),
            "client_to_inference": counts["client_to_inference"],
            "inference_to_client": counts["inference_to_client"],
            "client_circuit_ring_operations": None, "client_streamed_io_bytes": None,
            "known_all_link_bytes": None, "complete_client_peak_bytes": None,
            "unknown_costs": base["unknown"],
            "break_even": "sum factor+residual+metadata bytes < removed corrections minus new key/framing costs",
        },
    }


def report():
    stages = geometry()
    w = ((-128, 127, -3, 19, 0), (7, -11, 0, 1, 23), (2, 5, -17, 31, -9))
    a = ((1, 5), (7, 2), (-3, 1), (0, 11), (9, -4))
    cases = []
    for bits in (3, 24, 32):
        r, image = sparse_image(w, a, (17, (1 << bits) - 1), ((0, 13), (4, -5)), bits)
        x = (-128, 127, 0, -1, 63)
        result = reconstruct(w, x, r, image, (4, 5, 6), bits)
        # Separate signed-integer oracle; reduction only at the final output.
        oracle = tuple(sum(row[i] * x[i] for i in range(5)) & ((1 << bits) - 1) for row in w)
        require(result == oracle, "integer oracle mismatch")
        cases.append({"bits": bits, "exact_output": result})
    ledgers = [ledger(stages, outputs, requests=horizon)
               for outputs in (8, 64) for horizon in (1, 2, 4, 16)]
    thresholds = {}
    for outputs in (8, 64):
        thresholds[str(outputs)] = {
            "first_known_payload_break_even_request": next(
                (r for r in range(1, 129) if ledger(stages, outputs, requests=r)["known_byte_delta"] <= 0), None),
            "first_known_20pct_request": next(
                (r for r in range(1, 129) if Fraction(ledger(stages, outputs, requests=r)[
                    "additional_bytes_allowed_for_20pct_target"]) >= 0), None),
            "scope": "search 1..128; artifacts reused, roots renewed; unknown costs excluded"}
    return {"schema": "pllm.exact_correlation_frontier.v1", "source_sha256": LOCKS,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tests_sha256": hashlib.sha256((ROOT / "tests/test_exact_correlation_frontier.py").read_bytes()).hexdigest(),
            "algebra_cases": cases, "sparse_mod4_chosen_input_tv": str(sparse_toy_tv()),
            "key_linear_three_index_leak": homomorphic_leak((7, 13), (3, 8, 21), 32),
            "stages": stages, "ledgers": ledgers, "conditional_thresholds": thresholds,
            "request_sized_capacity_ledgers": [ledger(stages, o, requests=r, request_sized=True)
                                               for o in (8, 64) for r in (1, 4, 16)],
            "alternative_ledgers": [alternative_ledgers(stages, o) for o in (8, 64)],
            "security_proven": False, "real_checkpoint_loaded": False,
            "serving_or_rss_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
