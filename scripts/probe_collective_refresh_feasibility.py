"""Bounded protocol arithmetic/accounting oracle; no HE or secure refresh.

Uses already locked TenSEAL evidence, Lattigo v6.1.1's nominal mask-level
formula, and ePrint 2023/1203 Appendix D's distributed-rounding bound. No
upstream implementation is imported, no keys are generated, no network I/O.
The scalar rounding check is not a cryptographic protocol implementation.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/he-token-boundary-feasibility-qwen25-2026-09-30.json"


def minimum_mask_limbs(
    bits: tuple[int, ...], scale_bits: int, security_bits: int = 128, parties: int = 2
) -> int | None:
    """Nominal-bit version of mpckks.GetMinimumLevelForRefresh, not prime arithmetic."""
    if not bits or any(type(b) is not int or b <= 0 for b in bits):
        raise ValueError("positive active-Q bit sizes required")
    if min(scale_bits, security_bits) < 0 or parties < 2:
        raise ValueError("invalid scale, security or party count")
    required = scale_bits + security_bits + math.ceil(math.log2(parties))
    total = 0
    for count, bit_size in enumerate(bits, 1):
        total += bit_size
        if total >= required:
            return count
    return None


def refresh_epochs(depth: int, usable_products: int) -> int:
    """No refresh after final product; assumes refresh resets usable depth."""
    if depth <= 0 or usable_products <= 0:
        raise ValueError("positive depth and usable products required")
    return (depth - 1) // usable_products


def coefficient_payload(degree: int, limbs: int, polynomials: int = 1) -> int:
    """Raw uint64 RNS coefficients; excludes headers, not universal wire lower bound."""
    if min(degree, limbs, polynomials) <= 0:
        raise ValueError("positive representation dimensions required")
    return 8 * degree * limbs * polynomials


def worker_refresh_payload(
    degree: int, input_limbs: int, output_limbs: int, protocol: str
) -> dict[str, int]:
    """A evaluates/coordinates and owns one key share; B owns other key share."""
    request = coefficient_payload(degree, input_limbs)
    if protocol == "masked_pair":
        response = coefficient_payload(degree, input_limbs + output_limbs)
    elif protocol == "distributed_rounding":
        response = coefficient_payload(degree, output_limbs, 2)
    else:
        raise ValueError("unknown refresh protocol")
    return {
        "A_to_B": request,
        "B_to_A": response,
        "both_links": request + response,
        "messages_per_ciphertext_without_batching": 2,
        "sequential_one_way_flights": 2,
    }


def required_rounding_modulus_bits(
    degree: int, beta: int, refreshes: int, failure_bits: int
) -> int:
    """q >= 2*N*beta*R*2^kappa suffices by Appendix D Theorem 1 + union bound."""
    if min(degree, beta, refreshes) <= 0 or failure_bits < 0:
        raise ValueError("invalid correctness bound")
    target = 2 * degree * beta * refreshes * (1 << failure_bits)
    return (target - 1).bit_length()


def distributed_round_coefficient(residue: int, modulus: int) -> int:
    """OpenFHE PolynomialRound's interval rule, then centered integer lift.

    Odd moduli use floor(q/2), as upstream code does. Tiny test moduli have no
    RLWE interpretation. This function neither decrypts nor encrypts anything.
    """
    if modulus < 8:
        raise ValueError("rounding test modulus too small")
    value = residue % modulus
    if modulus // 4 < value <= (3 * modulus) // 4:
        value = (value + modulus // 2) % modulus
    if value > modulus // 2:
        value -= modulus
    return value


def rounding_oracle(modulus: int, bound: int = 8) -> dict[str, object]:
    """Exhaust all uniformly distributed scalar shares; no PRNG/masking protocol."""
    if modulus > 1024 or not 0 < bound < modulus // 4:
        raise ValueError("exhaustive oracle must stay small and bounded")
    worst_failures = 0
    small_errors: set[int] = set()
    for message in range(-bound, bound + 1):
        failures = 0
        for first in range(modulus):
            result = distributed_round_coefficient(first, modulus) + distributed_round_coefficient(
                message - first, modulus
            )
            error = result - message
            # Odd q/2 floor can introduce one coefficient unit of extra error.
            if abs(error) <= 1 and modulus % 2:
                small_errors.add(error)
            elif error == 0:
                small_errors.add(0)
            else:
                failures += 1
        # Integer endpoint conventions add at most two bad residues. The paper's
        # asymptotic 2*N*beta/q expression is not an exact tiny-modulus count.
        if modulus % 4 == 0 and failures > 2 * (abs(message) + 1):
            raise AssertionError("scalar rounding violates discrete endpoint guard")
        worst_failures = max(worst_failures, failures)
    return {
        "modulus": modulus,
        "messages_checked": 2 * bound + 1,
        "share_pairs_checked": (2 * bound + 1) * modulus,
        "worst_large_lift_failures": worst_failures,
        "small_coefficient_errors": sorted(small_errors),
        "is_secure_refresh": False,
    }


def interval_shift_tv(length: int, shift: int) -> Fraction:
    """Exact TV of two shifted uniform integer intervals, not full transcript privacy."""
    if length <= 0:
        raise ValueError("positive interval length required")
    return Fraction(min(abs(shift), length), length)


def run() -> dict[str, object]:
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["depth_contract"]["prefill_or_decode_one_forward_minimum"] == 96
    parameters = (
        (8192, (60, 40, 40), "8192_degree_60_40_40_60"),
        (16384, (60, 40, 40, 40, 40), "16384_degree_60_40_40_40_40_60"),
    )
    cohorts = []
    for outputs in (8, 32):
        locked = evidence["cohorts"][str(outputs)]
        budgets = {
            "online": locked["covered_online_body_tenfold_budget_bytes"],
            "all_link": locked["covered_all_link_body_tenfold_budget_bytes"],
        }
        for degree, q_bits, name in parameters:
            sample = evidence["sampled_backend"][name]
            slots = degree // 2
            prefill_hidden_chunks = math.ceil(39 * 896 / slots)
            live_hidden_chunks = prefill_hidden_chunks + outputs - 1
            mask_limbs = minimum_mask_limbs(q_bits, 40)
            # SEAL's last 60-bit special key-switch prime is not active Q.
            usable_masked = None if mask_limbs is None else len(q_bits) - mask_limbs
            protocols = []
            for protocol, reserved in (
                ("masked_pair", mask_limbs),
                ("distributed_rounding_fixed", 2),
                ("distributed_rounding_flexible", 3),
            ):
                usable = None if reserved is None else len(q_bits) - reserved
                entry: dict[str, object] = {"protocol": protocol, "usable_products": usable}
                if usable is not None and usable > 0:
                    epochs = refresh_epochs(96, usable)
                    kind = "masked_pair" if protocol == "masked_pair" else "distributed_rounding"
                    # Flexible adjust-scale needs three limbs before reducing to two.
                    input_limbs = mask_limbs if kind == "masked_pair" else 2
                    assert input_limbs is not None
                    payload = worker_refresh_payload(degree, input_limbs, len(q_bits), kind)
                    minimum_calls = outputs * epochs
                    proxy_calls = live_hidden_chunks * epochs
                    entry.update(
                        {
                            "epochs_per_forward": epochs,
                            "one_live_ciphertext_calls_floor": minimum_calls,
                            "one_live_ciphertext_raw_link_floor_bytes": minimum_calls
                            * payload["both_links"],
                            "two_flights_per_epoch_response_floor": 2 * minimum_calls,
                            "hidden_sized_proxy_calls_not_schedule": proxy_calls,
                            "hidden_sized_proxy_bytes_not_floor": proxy_calls
                            * payload["both_links"],
                            "payload_per_call": payload,
                        }
                    )
                    if kind == "distributed_rounding":
                        entry["q_bits_for_global_2^-50_at_assumed_beta_2^40"] = (
                            required_rounding_modulus_bits(degree, 1 << 40, proxy_calls, 50)
                        )
                        entry["nominal_100_bit_q_global_failure_bound"] = float(
                            Fraction(2 * degree * (1 << 40) * proxy_calls, 1 << 100)
                        )
                    else:
                        # Simple interval-TV sufficient bound, not Lattigo theorem.
                        entry["sufficient_mask_bits_global_tv_2^-128_assumed_beta_2^40"] = (
                            required_rounding_modulus_bits(degree, 1 << 40, proxy_calls, 128)
                        )
                protocols.append(entry)
            uploaded = live_hidden_chunks
            boundary = (
                uploaded * sample["one_896_value_input_ciphertext_body_bytes_approx"]
                + outputs
                * sample[
                    f"depth{sample['sampled_serial_ciphertext_product_depth']}_output_ciphertext_body_bytes_approx"
                ]
            )
            # One refresh after each of first 23 layers: useful only if a full layer fits.
            layer_calls = 23 * live_hidden_chunks
            single_layer_allowance = (budgets["online"] - boundary) // layer_calls
            kv_tokens = 39 + outputs - 1
            packed_kv_cts = 24 * math.ceil(2 * kv_tokens * 128 / slots)
            naive_kv_cts = 2 * 24 * kv_tokens
            cohorts.append(
                {
                    "outputs": outputs,
                    "degree": degree,
                    "active_Q_nominal_bits": list(q_bits),
                    "special_key_switch_prime_excluded_bits": 60,
                    "sampled_boundary_bytes": boundary,
                    "budgets": budgets,
                    "executed_rows": kv_tokens,
                    "prefill_hidden_chunks": prefill_hidden_chunks,
                    "live_hidden_chunks": live_hidden_chunks,
                    "mask_input_limbs_nominal": mask_limbs,
                    "usable_masked_products": usable_masked,
                    "protocols": protocols,
                    "once_per_layer_hidden_calls": layer_calls,
                    "once_per_layer_remaining_bytes_per_call": single_layer_allowance,
                    "encrypted_KV_combined_ideal_chunks": packed_kv_cts,
                    "encrypted_KV_combined_raw_full_level_bytes": packed_kv_cts
                    * coefficient_payload(degree, len(q_bits), 2),
                    "encrypted_KV_naive_raw_full_level_bytes": naive_kv_cts
                    * coefficient_payload(degree, len(q_bits), 2),
                }
            )
    return {
        "schema": "pllm.collective_refresh_arithmetic_oracle.v1",
        "scope": "nominal parameter transfer and scalar correctness; no HE runtime",
        "evidence_source": str(EVIDENCE.relative_to(ROOT)),
        "cohorts": cohorts,
        "rounding_checks": [rounding_oracle(q) for q in (64, 256, 65, 257)],
        "interval_TV_shift_2^40_width_2^168": str(interval_shift_tv(1 << 168, 1 << 40)),
        "secure_collective_refresh_executed": False,
        "real_Qwen_numeric_fidelity_validated": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
