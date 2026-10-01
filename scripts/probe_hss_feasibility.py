"""Bounded HSS class/traffic audit; no cryptography or model execution.

Standard library only. Computes projections from published parameters, checks
adaptive token packing, and certifies degrees of two finite-domain lookup
polynomials. These are representation screens, not secure HSS implementations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/compiler-region-contract-qwen25-2026-09-30.json"


def ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def interpolation_certificate(values: list[int], prime: int = 257) -> dict:
    """Newton interpolation at consecutive field points; exhaustive parity."""
    if len(values) != 256 or prime != 257:
        raise ValueError("this audit admits exactly 256 points over F_257")
    differences = [v % prime for v in values]
    coefficients = []
    inverse_factorial = 1
    for order in range(len(values)):
        if order:
            inverse_factorial = inverse_factorial * pow(order, -1, prime) % prime
        coefficients.append(differences[0] * inverse_factorial % prime)
        differences = [(b - a) % prime for a, b in zip(differences, differences[1:])]
    degree = max(i for i, c in enumerate(coefficients) if c)
    for x, expected in enumerate(values):
        actual, basis = 0, 1
        for order, coefficient in enumerate(coefficients):
            actual = (actual + coefficient * basis) % prime
            basis = basis * (x - order) % prime
        if actual != expected % prime:
            raise AssertionError("interpolation certificate failed")
    return {
        "field": prime,
        "points": len(values),
        "degree": degree,
        "leading_newton_coefficient": coefficients[degree],
        "exhaustive_reconstruction": True,
    }


def ties_even_half(x: int) -> int:
    q, remainder = divmod(x, 2)
    return q + int(bool(remainder) and bool(q % 2))


def traffic_projection(
    cohort: dict, *, dimension: int, modulus_bits: int, slots: int, input_polynomials: int
) -> dict:
    generated = cohort["output_tokens"]
    prompt = cohort["input_tokens"]
    width = 896
    rows = prompt + generated - 1
    assert rows == cohort["executed_rows"]
    # First response uses all prompt embeddings; each later token is adaptive.
    prefill_blocks = ceildiv(prompt * width, slots)
    decode_blocks = (generated - 1) * ceildiv(width, slots)
    ring_bytes = ceildiv(dimension * modulus_bits, 8)
    encoding_bytes = input_polynomials * ring_bytes
    per_worker_upload = (prefill_blocks + decode_blocks) * encoding_bytes
    # Optimistic 32-bit terminal shares, conditional on a valid output conversion.
    per_worker_download = generated * width * 4
    upload = 2 * per_worker_upload
    download = 2 * per_worker_download
    online = upload + download
    matched = cohort["matched_prepared"]
    return {
        "cohort": f"{prompt}+{generated}",
        "dimension": dimension,
        "modulus_bits": modulus_bits,
        "slots": slots,
        "input_polynomials_per_encoding": input_polynomials,
        "ring_element_bytes_bitpacked": ring_bytes,
        "input_encoding_bytes_per_worker": encoding_bytes,
        "prefill_blocks": prefill_blocks,
        "adaptive_decode_blocks": decode_blocks,
        "impossible_future_token_packed_blocks": ceildiv(rows * width, slots),
        "client_to_worker_a_bytes": per_worker_upload,
        "client_to_worker_b_bytes": per_worker_upload,
        "worker_a_to_client_bytes_conditional_32bit": per_worker_download,
        "worker_b_to_client_bytes_conditional_32bit": per_worker_download,
        "input_upload_all_links_bytes": upload,
        "online_with_conditional_terminal_shares_bytes": online,
        "input_alone_exceeds_online_budget": upload > matched["tenfold_online_budget_bytes"],
        "input_alone_exceeds_all_link_budget": upload > matched["tenfold_all_link_budget_bytes"],
        "unpriced": [
            "setup and evaluation keys to each worker",
            "dealer links",
            "worker-to-worker conversion or hybrid protocol",
            "exact terminal conversion",
            "persistent KV and state refresh",
            "numeric semantics and full wire",
            "cold model distribution",
        ],
    }


def run() -> dict:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["source"]["checkpoint"] == (
        "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"
    )
    assert [c["output_tokens"] for c in evidence["cohorts"]] == [8, 32]
    sign = interpolation_certificate([int(x >= 128) for x in range(256)])
    rounding = interpolation_certificate([ties_even_half(x - 128) for x in range(256)])
    assert ties_even_half(-3) == -2 and ties_even_half(-1) == 0
    assert ties_even_half(1) == 0 and ties_even_half(3) == 2
    assert sign["degree"] == 255
    assert rounding["degree"] == 255
    assert ceildiv(39 * 896, 8192) == 5
    degree = 3**24
    return {
        "schema": "pllm.hss_independent_math_screen.v1",
        "secure_hss_implemented": False,
        "model_or_runtime_executed": False,
        "locked_source": evidence["source"],
        "controls": [c["matched_prepared"] for c in evidence["cohorts"]],
        "degree_examples_not_qwen_lower_bounds": {
            "24_composed_integer_cubes_degree": degree,
            "rms_input_times_state_chain_multiplications_at_least": degree - 1,
            "ordinary_cube_circuit_multiplications": 48,
            "12_squarings_degree": 2**12,
            "12_squarings_rms_chain_multiplications_at_least": 2**12 - 1,
            "24_layer_attention_linear_probability_quadratic_gate_proxy_degree": 9**24,
        },
        "finite_domain_polynomial_certificates_not_runtime_operators": {
            "signed_i8_negative_threshold_complement": sign,
            "signed_i8_ties_even_divide_by_two": rounding,
        },
        "unseeded_public_key_encoding_projections_not_universal_floors": {
            # BKS 2019, Table 5: Bmax=2^32, N=8192, log(q)=203.
            # Full N-slot packing is deliberately optimistic and not validated.
            "bks_2019_optimistic_full_ring_slots": [
                traffic_projection(
                    c, dimension=8192, modulus_bits=203, slots=8192, input_polynomials=4
                )
                for c in evidence["cohorts"]
            ],
            # Tensor HSS 2026/1569, Section 5, Construction 2.
            "tensor_2026_reported_c2_instance": [
                traffic_projection(
                    c, dimension=45360, modulus_bits=540, slots=495, input_polynomials=4
                )
                for c in evidence["cohorts"]
            ],
        },
        "bks_40bit_per_operation_union_bound_illustration": {
            "operations_2pow20": math.ldexp(2**20, -40),
            "operations_2pow35": math.ldexp(2**35, -40),
            "note": "union upper bounds, not measured failure rates or Qwen operation counts",
        },
        "tensor_c2_single_495slot_input_both_workers_bytes": 8 * ceildiv(45360 * 540, 8),
        "tensor_c2_single_896value_input_both_workers_bytes": 16 * ceildiv(45360 * 540, 8),
        "roy_singh_dcr_unpacked_input_interface_projections": [
            {
                "cohort": f"39+{c['output_tokens']}",
                "rsa_modulus_bits": 3072,
                "s2_input_ciphertext_bytes_each_worker": 1152,
                "s2_scalar_embedding_upload_both_workers_bytes": c["executed_rows"]
                * 896
                * 2
                * 1152,
                "s3_additive_variant_scalar_embedding_upload_both_workers_bytes": c["executed_rows"]
                * 896
                * 2
                * 1536,
                "s2_18bit_token_input_upload_both_workers_bytes": c["executed_rows"]
                * 18
                * 2
                * 1152,
                "s3_18bit_token_input_upload_both_workers_bytes": c["executed_rows"]
                * 18
                * 2
                * 1536,
                "note": "token-bit interface has unimplemented lookup, exact full RMS/BP, KV and output conversion",
            }
            for c in evidence["cohorts"]
        ],
        "checks": "256-point sign and rounding parity; signed ties; cohort rows; adaptive packing",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
