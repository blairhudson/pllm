"""Bounded transciphering budget falsification, not a transcipher implementation."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "target/transciphered-linear"
SOURCES = [
    "docs/evidence/research-bootstrap-ingress.json",
    "docs/evidence/research-narrow-ingress.json",
    "docs/evidence/research-all-link-frontier.json",
    "docs/evidence/network-algebra-qwen25-2026-10-06.json",
    "scripts/probe_transciphered_linear_budget.py",
    "tests/test_transciphered_linear_budget_probe.py",
    "docs/research/transciphered-linear-reference.md",
]
C0, T0 = 124905091, 368670156
OLD_INPUT, BYTE_INPUT, OUTPUT = 90104112, 28732928, 138864059
REMOVED = 139060689 + OLD_INPUT - BYTE_INPUT
DEGREE, PLAIN = 8192, 268369921


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ceildiv(n, d):
    if type(n) is not int or type(d) is not int or n < 0 or d <= 0:
        raise ValueError("nonnegative numerator and positive integer divisor required")
    return (n + d - 1) // d


def signed_byte(value):
    if type(value) is not int or not 0 <= value < 256:
        raise ValueError("byte required")
    return value if value < 128 else value - 256


def prime_lift(residue, prime, bound, ring_bits):
    """Recover bounded integer before mapping to power-of-two ring, not vice versa."""
    if not 0 <= residue < prime or bound < 0 or prime <= 2 * bound:
        raise ValueError("ambiguous centered lift")
    if ring_bits not in (16, 24, 32):
        raise ValueError("unsupported original ring")
    integer = residue if residue <= prime // 2 else residue - prime
    if abs(integer) > bound:
        raise ValueError("outside certified public bound")
    return integer % (1 << ring_bits)


def ledger(pairs, expansion, *, public_reuse=True, setup=0, recurring_extra=0,
           renewal=0, lifetime=1, failures=0):
    """Explicit optimistic scenario. Unknown actual costs are never passed as zero."""
    if pairs < 1 or lifetime < 1 or min(setup, recurring_extra, renewal, failures) < 0:
        raise ValueError("invalid horizon or costs")
    expansion = Fraction(expansion)
    if expansion < 0:
        raise ValueError("negative expansion")
    common = C0 if public_reuse else 0
    recurring = T0 if public_reuse else C0 + T0
    control = common + pairs * recurring
    # Initial issuance is setup; renewal is ONLY subsequent epoch issuance.
    epochs = ceildiv(pairs, lifetime)
    candidate = (common + pairs * (recurring - REMOVED + (expansion - 1) * OUTPUT
                                  + recurring_extra) + setup + (epochs - 1) * renewal + failures)
    return {"control": Fraction(control), "candidate": candidate,
            "saving": Fraction(control) - candidate, "saving_fraction": (control - candidate) / control,
            "issuances": epochs}


def expansion_limit(target, pairs=None, public_reuse=True):
    denominator = T0 if public_reuse else C0 + T0
    if pairs is not None:
        denominator = Fraction(C0 + pairs * T0, pairs) if public_reuse else C0 + T0
    return 1 + (REMOVED - Fraction(target) * denominator) / OUTPUT


def first_horizon(target, expansion, setup, recurring_extra=0):
    gain = REMOVED - (Fraction(expansion) - 1) * OUTPUT - recurring_extra - Fraction(target) * T0
    if gain <= 0:
        return None
    return max(1, ceildiv((Fraction(target) * C0 + setup).numerator * gain.denominator,
                          (Fraction(target) * C0 + setup).denominator * gain.numerator))


def packing_count(outputs, prefill_rows, decode_rows, slots):
    """Within-stage prefill batching; each dependent decode step is separate."""
    if outputs < 1:
        raise ValueError("positive output width required")
    return ceildiv(outputs * prefill_rows, slots) + decode_rows * ceildiv(outputs, slots)


def admit_placement(client_dense_macs, original_macs):
    if original_macs <= 0 or client_dense_macs < 0 or client_dense_macs > original_macs:
        raise ValueError("invalid ownership")
    return Fraction(original_macs - client_dense_macs, original_macs) >= Fraction(9, 10)


def algebra_control():
    assert (PLAIN - 1) % (2 * DEGREE) == 0
    assert not any(PLAIN % divisor == 0 for divisor in range(2, math.isqrt(PLAIN) + 1))
    checked = 0
    for byte in range(256):
        x = signed_byte(byte)
        assert x % 256 == byte
        for w in (-128, -127, -1, 0, 1, 127):
            y = w * x
            for ring in (16, 24, 32):
                assert prime_lift(y % PLAIN, PLAIN, 16384, ring) == y % (1 << ring)
                checked += 1
    # Wrap in ring16 is legal and differs from a small-prime substitution.
    assert prime_lift(65536, PLAIN, 65536, 16) == 0
    assert prime_lift(32258, PLAIN, 32258, 16) == 32258
    assert 32258 % 257 != 32258 % 65536
    return {"outputs_checked": checked, "signed_bytes": 256, "rings": [16, 24, 32],
            "plaintext_prime": PLAIN, "prime_independently_trial_divided": True,
            "batching_congruence_checked": True,
            "contract": "bounded integer lift then ring reduction; no CKKS"}


def validate_stage_contracts(stages, narrow_stages, saved_stages):
    """Reject duplicate/missing identities and mismatched original numeric rings."""
    maps = []
    for records in (stages, narrow_stages, saved_stages):
        mapping = {s["stage"]: s for s in records}
        if len(mapping) != len(records):
            raise ValueError("duplicate stage identity")
        maps.append(mapping)
    if not maps[0].keys() == maps[1].keys() == maps[2].keys():
        raise ValueError("stage identity sets differ")
    for name, stage in maps[0].items():
        narrow, saved = maps[1][name], maps[2][name]
        if not stage["shape"] == narrow["shape"] == saved["shape"]:
            raise ValueError("stage shapes differ")
        if not stage["weight_sha256"] == narrow["weight_sha256"] == saved["weight_sha256"]:
            raise ValueError("stage weight identities differ")
        if (stage["original_ring_bits"] not in (16, 24, 32)
                or stage["original_ring_bits"] != saved["ring_bits"]
                or stage["stage_bits"] != narrow["stage_bits"]
                or not 0 < stage["stage_bits"] <= stage["original_ring_bits"]):
            raise ValueError("stage original ring/residue contract differs")
    return maps[1]


def geometry_screen(measurement):
    ingress = json.loads((ROOT / SOURCES[0]).read_text())["cost"]
    stages = ingress["stages"]
    narrow = json.loads((ROOT / SOURCES[1]).read_text())["checkpoint_geometry"]
    saved = json.loads((ROOT / SOURCES[3]).read_text())["stages"]
    identities = validate_stage_contracts(stages, narrow["stages"], saved)
    assert len(stages) == len(identities) == 96
    assert sum(s["rows"] for s in stages) == 15297
    assert sum(s["old_input_payload"] for s in stages) == OLD_INPUT
    assert sum(s["byte_input_payload"] for s in stages) == BYTE_INPUT
    assert sum(s["output_payload"] for s in stages) == OUTPUT
    assert sum(ingress["classes"].values()) == C0 + T0
    output_values = sum(s["rows"] * s["shape"][0] for s in stages)
    valid_levels, rejected_levels = [], []
    for level in measurement.get("levels", []):
        if level["exact"] is not True or level["noise_bits"] <= 0:
            rejected_levels.append({**level, "rejection": "wrong output or nonpositive noise budget"})
        else:
            valid_levels.append(level)
    layouts = []
    for slots in (4096, 8192):
        counts = []
        for s in stages:
            n = identities[s["stage"]]
            assert n["shape"] == s["shape"] and n["weight_sha256"] == s["weight_sha256"]
            m, width = s["shape"]
            assert PLAIN > 2 * width * 127 * 127
            prefill = s["rows"] - 14
            count = packing_count(m, prefill, 14, slots)
            counts.append({"stage": s["stage"], "output_width": m, "input_width": width,
                           "prefill_rows": prefill, "decode_rows": 14, "ciphertexts": count,
                           "decode_utilization": str(Fraction(m, ceildiv(m, slots) * slots))})
        total = sum(c["ciphertexts"] for c in counts)
        sizes = []
        for level in valid_levels:
            output_bytes = total * (level["serialized_bytes"] + 96)
            expansion = Fraction(output_bytes, OUTPUT)
            sizes.append({"data_primes": level["data_primes"], "toy_noise_bits": level["noise_bits"],
                          "toy_exact": level["exact"], "ciphertext_bytes": level["serialized_bytes"],
                          "hypothetical_output_bytes": output_bytes,
                          "expansion": str(expansion),
                          "optimistic_first_pair_saving_fraction": str(ledger(1, expansion)["saving_fraction"]),
                          "optimistic_recurring_saving_fraction": str((REMOVED - (expansion - 1) * OUTPUT) / T0),
                          "public_padding_certified": False,
                          "coefficient_storage_only_uncompressed_bytes": 2 * DEGREE * level["data_primes"] * 8,
                          "full_width_noise_certified": False})
        layouts.append({"slots": slots, "implemented_toy_layout": slots == 4096,
                        "ciphertexts": total, "output_values": output_values,
                        "utilization": str(Fraction(output_values, total * slots)),
                        "20_percent_max_reply_bytes_per_ciphertext_before_new_costs": {
                            "first_pair": str(expansion_limit(Fraction(1, 5), 1) * OUTPUT / total),
                            "recurring": str(expansion_limit(Fraction(1, 5)) * OUTPUT / total)},
                        "impossible_cross_dependency_pooling_floor": ceildiv(output_values, slots),
                        "stages": counts, "measured_serialization_projection": sizes})
    return {"stage_count": 96, "executed_rows": 15297, "output_values": output_values,
            "rejected_modulus_levels": rejected_levels,
            "stage_identity_sets_and_ring_contracts_validated": True,
            "body_macs": sum(s["body_macs"] for s in stages), "layouts": layouts,
            "fixed_ciphertext_frame_assumption_bytes": 96,
            "setup_context_lifecycle_projection": {
                "one_reused_context": measurement.get("public_setup_delivery_bytes"),
                "one_context_per_stage": (96 * measurement["public_setup_delivery_bytes"]
                                          if measurement.get("executed") else None),
                "one_context_per_stage_row": (15297 * measurement["public_setup_delivery_bytes"]
                                              if measurement.get("executed") else None),
                "scope": "BFV public matvec setup ONLY; excludes encrypted cipher key, TFHE, conversion and renewals"},
            "packed_full_width_validity": None, "transcipher_to_BFV_conversion_cost": None}


def seal_control():
    """Independent tiny BFV evaluator. Direct HE input is a downstream control only."""
    try:
        import tenseal
        import tenseal.sealapi as seal
    except ImportError:
        return {"executed": False, "reason": "existing SEAL binding unavailable"}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    parms = seal.EncryptionParameters(seal.SCHEME_TYPE.BFV)
    parms.set_poly_modulus_degree(DEGREE)
    parms.set_coeff_modulus(seal.CoeffModulus.Create(DEGREE, [55, 54, 54, 55]))
    parms.set_plain_modulus(PLAIN)
    client = seal.SEALContext(parms, True, seal.SEC_LEVEL_TYPE.TC128)
    assert client.parameters_set()
    keygen = seal.KeyGenerator(client)
    sk, pk, gk = keygen.secret_key(), seal.PublicKey(), seal.GaloisKeys()
    keygen.create_public_key(pk)
    steps = [1 << i for i in range(12)]
    # Binding's list overload accepts odd Galois elements, not rotation steps.
    keygen.create_galois_keys([pow(3, step, 2 * DEGREE) for step in steps], gk)
    sizes = {}
    for name, obj in (("parameters", parms), ("public_key", pk), ("galois_keys", gk), ("secret_key", sk)):
        path = SCRATCH / (name + ".seal")
        obj.save(str(path))
        sizes[name] = path.stat().st_size
    # Separate parameter/context and public evaluation-key objects; never load sk.
    provider_parms = seal.EncryptionParameters(seal.SCHEME_TYPE.BFV)
    provider_parms.load(str(SCRATCH / "parameters.seal"))
    provider = seal.SEALContext(provider_parms, True, seal.SEC_LEVEL_TYPE.TC128)
    provider_gk = seal.GaloisKeys()
    provider_gk.load(provider, str(SCRATCH / "galois_keys.seal"))
    encryptor, decryptor = seal.Encryptor(client, pk), seal.Decryptor(client, sk)
    encoder, evaluator = seal.BatchEncoder(provider), seal.Evaluator(provider)
    setup_wall = time.perf_counter() - started
    weights = [[127, -127, 1, -1, 0, 3, -7, 11], [-128] * 8,
               [1] * 8, [0] * 8, [0, 0, 0, 0, 0, 0, 0, -127]]
    levels, trials = [], []
    for trial, x in enumerate(([-128, 127, -1, 0, 1, 126, -127, 3], [127] * 8)):
        start = time.perf_counter()
        plain, request = seal.Plaintext(), seal.Ciphertext()
        encoder.encode([v % PLAIN for v in x] + [0] * (DEGREE - len(x)), plain)
        encryptor.encrypt(plain, request)
        request.save(str(SCRATCH / "request.seal"))
        received = seal.Ciphertext()
        received.load(provider, str(SCRATCH / "request.seal"))
        result = None
        for j, row in enumerate(weights):
            if not any(row):
                continue  # public zero row; avoid SEAL transparent-ciphertext rejection
            wplain, scalar = seal.Plaintext(), seal.Ciphertext()
            encoder.encode([v % PLAIN for v in row] + [0] * (DEGREE - len(row)), wplain)
            evaluator.multiply_plain(received, wplain, scalar)
            for step in steps:
                rotated = seal.Ciphertext()
                evaluator.rotate_rows(scalar, step, provider_gk, rotated)
                evaluator.add_inplace(scalar, rotated)
            mask = [0] * DEGREE
            mask[j] = 1
            encoder.encode(mask, wplain)
            evaluator.multiply_plain_inplace(scalar, wplain)
            if result is None:
                result = scalar
            else:
                evaluator.add_inplace(result, scalar)
        expected = [sum(w * v for w, v in zip(row, x)) for row in weights]
        trial_levels = []
        while True:
            path = SCRATCH / "reply.seal"
            result.save(str(path))
            raw = path.read_bytes()
            loaded = seal.Ciphertext()
            loaded.load(client, str(path))
            budget = decryptor.invariant_noise_budget(loaded)
            decoded = seal.Plaintext()
            decryptor.decrypt(loaded, decoded)
            values = encoder.decode_int64(decoded)
            exact = values[:5] == expected and all(v == 0 for v in values[5:])
            level = {"data_primes": result.coeff_modulus_size(), "noise_bits": budget,
                     "exact": exact, "serialized_bytes": len(raw), "compression_mode_byte": raw[5],
                     "size_two": result.size() == 2, "serialized_sha256": digest(path)}
            trial_levels.append(level)
            if provider.get_context_data(result.parms_id()).next_context_data() is None:
                break
            evaluator.mod_switch_to_next_inplace(result)
        assert trial_levels[0]["exact"] and trial_levels[0]["noise_bits"] > 0
        if trial == 0:
            levels = trial_levels
        else:
            for a, b in zip(levels, trial_levels):
                a["noise_bits"] = min(a["noise_bits"], b["noise_bits"])
                a["exact"] = a["exact"] and b["exact"]
                a["serialized_bytes"] = max(a["serialized_bytes"], b["serialized_bytes"])
                a.pop("serialized_sha256", None)  # aggregate size is not a single frame digest
        trials.append({"exact_outputs_at_initial_level": 5, "wall_seconds": time.perf_counter() - start,
                       "direct_HE_input_bytes": (SCRATCH / "request.seal").stat().st_size})
    return {"executed": True, "transciphering_implemented": False, "tenseal_version": tenseal.__version__,
            "degree": DEGREE, "plain_prime": PLAIN, "coeff_bits": [55, 54, 54, 55],
            "rotation_steps": steps, "source_shaped_full_width_evaluated": False,
            "security_admission": "SEAL TC128 (classical), not PETCHA PQ parameter certification",
            "slot_capacity_used": 4096, "matrix_shape": [5, 8], "trials": trials,
            "levels": levels, "setup_serialized_bytes": sizes, "setup_wall_seconds": setup_wall,
            "public_setup_delivery_bytes": sum(sizes[k] for k in ("parameters", "public_key", "galois_keys")),
            "provider_receives_secret_key": False, "symmetric_FHE_key_bytes": None,
            "TFHE_bootstrap_conversion_keys_bytes": None, "seeded_evaluated_output_bytes": None,
            "bounded_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024), "whole_client_peak_rss_bytes": None}


def build(measure=False):
    measurement = seal_control() if measure else {"executed": False, "reason": "--measure-seal not selected"}
    sensitivity = []
    for reuse in (True, False):
        for pairs in (1, 2, 4, 16, 1024):
            row = ledger(pairs, Fraction(3, 2), public_reuse=reuse)
            sensitivity.append({"pairs": pairs, "public_reuse": reuse, **{k: str(v) for k, v in row.items()},
                                "20_percent_extra_budget_total": str(row["saving"] - Fraction(1, 5) * row["control"])})
    snapshots = {str(p.relative_to(ROOT)): digest(p) for p in SCRATCH.glob("*-primary.html")}
    snapshots.update({str(p.relative_to(ROOT)): digest(p) for p in SCRATCH.glob("seal-*.h")})
    snapshots.update({str(p.relative_to(ROOT)): digest(p) for p in SCRATCH.glob("bumblebee.pdf")})
    return {"complete": True, "scope": "bounded falsification/cost screen only; no candidate protocol",
            "source_sha256": {p: digest(ROOT / p) for p in SOURCES}, "primary_snapshot_sha256": snapshots,
            "primary_access": {"PETCHA": "2026/112; received 2026-01-23, revised 2026-06-08; ChaCha; abstract-only direct access",
                               "Coral": "2026/1917; received 2026-09-08, approved 2026-09-10; TFHE; Trivium/AES/ChaCha20/ZUC; abstract-only direct access"},
            "algebra": algebra_control(), "seal_downstream_control": measurement,
            "geometry": geometry_screen(measurement), "historical_ledger": {"common_public": C0, "recurring": T0,
            "first_pair": C0 + T0, "old_input": OLD_INPUT, "byte_input": BYTE_INPUT,
            "output_payload": OUTPUT, "removed_before_output_expansion": REMOVED},
            "hypothetical_r_1_5_sensitivity": sensitivity,
            "20_percent_max_expansion": {"first_pair": str(expansion_limit(Fraction(1, 5), 1)),
            "recurring_reusable_public": str(expansion_limit(Fraction(1, 5))),
            "recurring_no_public_reuse": str(expansion_limit(Fraction(1, 5), public_reuse=False))},
            "unknown_costs": ["TFHE cipher/keys/setup/renewal", "TFHE-to-exact-matvec conversion", "full-width packing/noise",
                              "provider/checkpoint artifacts", "new frames/authentication", "failures/retries", "whole-client RSS/CPU", "physical wire", "malicious verification"],
            "unknown_costs_are_zero": False, "full_decoder_executed": False, "source_checkpoint_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure-seal", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a fresh path")
    report = build(args.measure_seal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "complete": True,
                      "seal_executed": report["seal_downstream_control"]["executed"]}))


if __name__ == "__main__":
    main()
