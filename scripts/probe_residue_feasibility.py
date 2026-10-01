"""Bounded exact-integer CRT oracle and optional pinned semantic cost screen.

No protected protocol is implemented. Reconstruction is an offline oracle only;
private multiplication, conversion, comparisons and rounded edges stay unpriced.
Default mode uses only the standard library, tiny integers and synthetic inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import json
from math import gcd, log2, prod
from pathlib import Path


@dataclass(frozen=True)
class Basis:
    moduli: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.moduli) is not tuple
            or not 1 <= len(self.moduli) <= 8
            or any(type(m) is not int or not 2 <= m <= 65536 for m in self.moduli)
            or sum((m - 1).bit_length() for m in self.moduli) > 96
            or any(
                gcd(a, b) != 1
                for a, b in (
                    (self.moduli[i], self.moduli[j])
                    for i in range(len(self.moduli))
                    for j in range(i)
                )
            )
        ):
            raise ValueError("bounded pairwise-coprime integer basis required")

    @property
    def modulus(self) -> int:
        return prod(self.moduli)

    @property
    def send_bits(self) -> int:
        return sum((m - 1).bit_length() for m in self.moduli)

    def encode(self, value: int) -> tuple[int, ...]:
        if type(value) is not int:
            raise ValueError("integer residue source required")
        return tuple(value % m for m in self.moduli)

    def canonical(self, residues: tuple[int, ...]) -> int:
        if (
            type(residues) is not tuple
            or len(residues) != len(self.moduli)
            or any(
                type(r) is not int or not 0 <= r < m
                for r, m in zip(residues, self.moduli, strict=True)
            )
        ):
            raise ValueError("noncanonical residue tuple")
        total = 0
        for r, m in zip(residues, self.moduli, strict=True):
            cofactor = self.modulus // m
            total += r * cofactor * pow(cofactor, -1, m)
        return total % self.modulus

    def lift(self, residues: tuple[int, ...], low: int, high: int) -> int:
        """Offline oracle; reject any public domain lacking global injectivity."""
        if (
            type(low) is not int
            or type(high) is not int
            or low > high
            or high - low >= self.modulus
        ):
            raise ValueError("ambiguous or invalid public lift domain")
        canonical = self.canonical(residues)
        candidate = (
            canonical + ((low - canonical + self.modulus - 1) // self.modulus) * self.modulus
        )
        if candidate > high:
            raise ValueError("residue outside public lift domain")
        return candidate

    def mixed_radix(self, residues: tuple[int, ...]) -> tuple[int, ...]:
        """Independent Garner-style oracle; private digit extraction is not free."""
        self.canonical(residues)  # validation only; do not use reconstructed value
        digits: list[int] = []
        for i, (r, m) in enumerate(zip(residues, self.moduli, strict=True)):
            prior, place = 0, 1
            for j in range(i):
                prior += digits[j] * place
                place *= self.moduli[j]
            digits.append(((r - prior) * pow(place, -1, m)) % m)
        return tuple(digits)


def round_even(numerator: int, denominator: int) -> int:
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError("integer numerator and positive denominator required")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    return sign * (
        quotient
        + int(2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 == 1))
    )


def linear_residues(
    basis: Basis, weights: tuple[int, ...], inputs: tuple[int, ...], bias: int = 0
) -> tuple[int, ...]:
    if len(weights) != len(inputs) or not 1 <= len(inputs) <= 8:
        raise ValueError("bounded matching linear operands required")
    encoded = [basis.encode(x) for x in inputs]
    return tuple(
        (bias + sum(w * x[j] for w, x in zip(weights, encoded, strict=True))) % m
        for j, m in enumerate(basis.moduli)
    )


def arithmetic_probe() -> dict:
    basis = Basis((16, 17, 19))
    linear_cases = quadratic_cases = 0
    for values in product(range(-7, 8), repeat=2):
        for weights in ((-3, 5), (0, -7), (7, 7)):
            for bias in (-9, 0, 9):
                expected = sum(w * x for w, x in zip(weights, values, strict=True)) + bias
                actual = basis.lift(linear_residues(basis, weights, values, bias), -107, 107)
                assert actual == expected
                linear_cases += 1
        # Exact rational quadratic island: (3x² - 2xy + 5y + 7)/12.
        x, y = values
        expected_numerator = 3 * x * x - 2 * x * y + 5 * y + 7
        residues = tuple(
            (3 * (x % m) ** 2 - 2 * (x % m) * (y % m) + 5 * (y % m) + 7) % m for m in basis.moduli
        )
        actual = basis.lift(residues, -287, 287)
        assert Fraction(actual, 12) == (3 * Fraction(x) ** 2 - 2 * Fraction(x) * y + 5 * y + 7) / 12
        assert round_even(actual, 12) == round(Fraction(expected_numerator, 12))
        quadratic_cases += 1
    # Linear -> rational quadratic -> public linear, reconstruct only at end.
    # Independent Fraction oracle preserves exact common denominator throughout.
    island_basis = Basis((251, 241, 239))
    delayed_cases = 0
    for x, y in product(range(-7, 8), repeat=2):
        a, b = -3 * Fraction(x) + 2 * y + 1, Fraction(x) - y - 2
        oracle = 2 * (3 * a * a - 2 * a * b + 5 * b + 7) / 12 + a / 3
        result = []
        for m in island_basis.moduli:
            ar, br = (-3 * (x % m) + 2 * (y % m) + 1) % m, ((x % m) - (y % m) - 2) % m
            qr = (3 * ar * ar - 2 * ar * br + 5 * br + 7) % m
            result.append((2 * qr + 4 * ar) % m)
        numerator = island_basis.lift(tuple(result), -16384, 16384)
        assert Fraction(numerator, 12) == oracle
        assert round_even(numerator, 12) == round(oracle)
        delayed_cases += 1
    # Public-linear operations on additive residue shares need no lift mid-island.
    shared_cases = 0
    small = Basis((3, 5))
    for x in range(-7, 8):
        for a in range(small.modulus):
            b = (x - a) % small.modulus
            left = linear_residues(small, (-3,), (a,), bias=2)
            right = linear_residues(small, (-3,), (b,))
            combined = tuple((l + r) % m for l, r, m in zip(left, right, small.moduli, strict=True))
            # Output domain larger than basis: only modular parity is certified.
            assert small.canonical(combined) == (-3 * x + 2) % 15
            shared_cases += 1
    carry_counts = Counter()
    for x in (-127, -1, 0, 1, 127):
        for a in range(4096):
            b = (x - a) % 4096
            carry, remainder = divmod(a + b - x, 4096)
            assert remainder == 0 and carry in (0, 1, 2)
            assert -7 * (a + b - 4096 * carry) == -7 * x
            assert carry != 0 or x >= 0
            assert carry != 2 or x < 0
            carry_counts[carry] += 1
    # Full mixed-radix reconstruction independently checks CRT oracle.
    for x in range(-2583, 2585):
        digits = basis.mixed_radix(basis.encode(x))
        place, result = 1, 0
        for d, m in zip(digits, basis.moduli, strict=True):
            result += d * place
            place *= m
        assert result == x % basis.modulus
    return {
        "linear_exact_cases": linear_cases,
        "rational_quadratic_exact_cases": quadratic_cases,
        "delayed_linear_quadratic_linear_exact_cases": delayed_cases,
        "additive_shared_public_linear_modular_cases": shared_cases,
        "mixed_radix_full_domain_cases": basis.modulus,
        "narrow_share_signed_carry_cases": sum(carry_counts.values()),
        "secret_carry_histogram_offline_only": dict(carry_counts),
        "private_quadratic_multiplication_implemented": False,
        "private_reconstruction_implemented": False,
        "counterexamples": {
            "small_basis_alias": {"basis": [3, 5], "values": [-1, 14], "same_residues": [2, 4]},
            "unsigned_share_local_extension": {
                "source_modulus": 15,
                "target_modulus": 17,
                "shares": [14, 2],
                "secret": 1,
                "naive_target_sum": 16,
            },
            "signed_local_extension": {
                "source_modulus": 15,
                "target_modulus": 17,
                "shares": [14, 0],
                "secret": -1,
                "naive_target_sum": 14,
                "correct_target": 16,
            },
            "local_share_squares": {
                "modulus": 15,
                "shares": [1, 1],
                "sum_of_local_squares": 2,
                "correct_square": 4,
            },
            "modular_inverse_not_rounding": {
                "modulus": 5,
                "numerator": 1,
                "denominator": 2,
                "modular_quotient": 3,
                "ties_even_integer": 0,
            },
            "deferred_sum_rounding": {
                "numerators": [1, 1],
                "denominator": 2,
                "round_then_sum": 0,
                "sum_then_round": 1,
            },
            "deferred_linear_rounding": {
                "numerator": 1,
                "denominator": 2,
                "public_weight": 3,
                "round_then_map": 0,
                "map_then_round": 2,
            },
            "per_channel_sign_insufficient": {
                "channel_modulus": 3,
                "values": [-2, 1],
                "same_channel_residue": 1,
            },
            "noisypair_lift_wrap": {
                "p": 5,
                "q": 17,
                "x": 4,
                "r": 3,
                "opened_masked": 2,
                "lifted_mod_q": 16,
                "correct_mod_q": 4,
            },
        },
    }


def basis_report(moduli: tuple[int, ...]) -> dict:
    basis = Basis(moduli)
    return {
        "moduli": moduli,
        "product": basis.modulus,
        "capacity_bits": log2(basis.modulus),
        "packed_bits_per_share": basis.send_bits,
        "joint_canonical_share_encoding_bits": (basis.modulus - 1).bit_length(),
        "byte_aligned_bits_per_share": sum(8 * (((m - 1).bit_length() + 7) // 8) for m in moduli),
        "u32_channel_storage_bits_per_share": 32 * len(moduli),
    }


def semantic_screen(output_tokens: int) -> dict:
    """Read cached config only; bind graph and schedule to locked region evidence."""
    import os
    from huggingface_hub import hf_hub_download
    from pllm import Model, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.region_contract_cost import compiler_region_contract_cost

    root = Path(__file__).resolve().parents[1]
    locked = json.loads(
        (root / "docs/evidence/compiler-region-contract-qwen25-2026-09-30.json").read_text()
    )
    cohort = next(c for c in locked["cohorts"] if c["output_tokens"] == output_tokens)
    model, revision = locked["source"]["checkpoint"].split("@")
    cache = os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    path = hf_hub_download(
        model, "config.json", revision=revision, local_files_only=True, cache_dir=cache
    )
    plan = lower_model(
        json.loads(Path(path).read_text()),
        batch=1,
        max_input_tokens=39,
        max_new_tokens=output_tokens,
    )
    composition = MaskedLinearCpu(
        Model.hf(model, revision=revision),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    schedule = plan.runtime_schedule(composition)
    if (
        plan.digest != cohort["plan_digest"]
        or schedule.digest != cohort["schedule_digest"]
        or schedule.composition_digest != locked["source"]["composition_digest"]
    ):
        raise ValueError("semantic plan/schedule/composition differs from pinned control")
    budgets = cohort["matched_prepared"]
    region = compiler_region_contract_cost(
        plan,
        composition,
        response_new_tokens=output_tokens,
        maximum_online_all_link_body_bytes=budgets["tenfold_online_budget_bytes"],
        maximum_total_all_link_body_bytes=budgets["tenfold_all_link_budget_bytes"],
    )
    graph = plan.to_dict()
    expected = Counter(
        {
            "linear": 7,
            "rms_norm": 2,
            "softmax": 1,
            "silu": 1,
            "rotary_embedding": 2,
            "attention_scores": 1,
            "attention_scale": 1,
            "attention_values": 1,
            "residual_add": 2,
            "multiply": 1,
            "reshape": 4,
            "kv_cache_append": 2,
            "cache_suffix": 2,
            "causal_mask": 1,
        }
    )
    phases = {}
    for phase in ("prefill", "decode"):
        ops = graph[phase]["operations"]
        by_id = {op["id"]: op for op in ops}
        seen = set()
        for op in ops:
            if any(source in by_id and source not in seen for source in op["inputs"]):
                raise ValueError("semantic graph not dependency ordered")
            seen.add(op["id"])
        for layer in range(region["layer_count"]):
            if Counter(op["operator"] for op in ops if op.get("layer") == layer) != expected:
                raise ValueError("unclassified or incomplete semantic layer")
        first_layer = [op for op in ops if op.get("layer") == 0]
        softmax_op = next(op for op in first_layer if op["operator"] == "softmax")
        silu_op = next(op for op in first_layer if op["operator"] == "silu")
        dynamic_inputs = []
        for step in schedule.to_dict()[phase]["steps"]:
            if step.get("layer") is not None and step["operators"] == ["linear"] * len(
                step["operators"]
            ):
                if len(step["input_ids"]) != 1:
                    raise ValueError("projection group lacks unique semantic quantization source")
                dynamic_inputs.append(by_id[step["input_ids"][0]]["output_shape"][-1])
        if len(dynamic_inputs) != 4 * region["layer_count"]:
            raise ValueError("dynamic quantization group coverage differs")
        # Nonlinear/order-sensitive internals and floating rounded polynomial ops
        # are distinct obligations, not claims that plaintext must be opened.
        rounded_kinds = {
            "linear",
            "rotary_embedding",
            "attention_scores",
            "attention_scale",
            "attention_values",
            "residual_add",
            "multiply",
        }
        phases[phase] = {
            "per_layer_declared_operator_counts": dict(sorted(expected.items())),
            "body_nonlinear_sites_per_forward": 4 * region["layer_count"],
            "dynamic_quantization_groups_per_forward": len(dynamic_inputs),
            "dynamic_max_abs_binary_tournament_comparators_per_query_row_all_layers": sum(
                w - 1 for w in dynamic_inputs
            ),
            "rounded_polynomial_operator_sites_per_forward": sum(
                op.get("layer") is not None and op["operator"] in rounded_kinds for op in ops
            ),
            "max_w8a8_public_dot_input_width": max(dynamic_inputs),
            "query_heads_from_softmax_shape": softmax_op["output_shape"][1],
            "intermediate_width_from_silu_shape": silu_op["output_shape"][-1],
        }
    if phases["prefill"] != phases["decode"]:
        raise ValueError("phase semantic barrier classifications disagree")
    layers, rows = region["layer_count"], region["executed_rows"]
    source_elements = region["two_worker_resident"]["12"]["independent_source_elements_per_worker"]
    token_bytes = cohort["two_worker_resident"]["token_boundary_body_bytes"]
    source_scenarios = []
    for moduli in ((16, 17), (16, 17, 19), (251, 241, 239, 13)):
        report = basis_report(moduli)
        # Each worker sends own masked residue tuple. Neither message omitted.
        peer = 2 * ((source_elements * report["packed_bits_per_share"] + 7) // 8)
        online = peer + token_bytes
        source_scenarios.append(
            {
                **report,
                "duplicated_peer_opening_bytes": peer,
                "token_boundary_bytes_24_32_bit_contract": token_bytes,
                "known_online_floor_bytes": online,
                "remaining_online_budget_bytes": budgets["tenfold_online_budget_bytes"] - online,
                "remaining_all_link_budget_bytes": budgets["tenfold_all_link_budget_bytes"]
                - online,
                "private_scale_or_lift_cost_bytes": None,
                "admitted": False,
            }
        )
        joint_peer = 2 * (
            (source_elements * report["joint_canonical_share_encoding_bits"] + 7) // 8
        )
        source_scenarios[-1]["joint_share_codec_known_online_floor_bytes"] = (
            joint_peer + token_bytes
        )
        source_scenarios[-1]["joint_share_codec_remaining_online_budget_bytes"] = (
            budgets["tenfold_online_budget_bytes"] - joint_peer - token_bytes
        )
        source_scenarios[-1]["joint_share_codec_cpu_measured"] = False
    widest = phases["prefill"]["max_w8a8_public_dot_input_width"]
    dot_bound = widest * 127 * 127
    obligations_per_forward = sum(
        phases["prefill"][key]
        for key in (
            "body_nonlinear_sites_per_forward",
            "dynamic_quantization_groups_per_forward",
            "rounded_polynomial_operator_sites_per_forward",
        )
    )
    return {
        "output_tokens": output_tokens,
        "executed_rows": rows,
        "source": locked["source"],
        "plan_digest": plan.digest,
        "schedule_digest": schedule.digest,
        "budgets": budgets,
        "semantic_phases": phases,
        "body_norm_query_rows": 2 * layers * rows,
        "body_softmax_head_query_rows": phases["prefill"]["query_heads_from_softmax_shape"]
        * layers
        * rows,
        "body_silu_elements": phases["prefill"]["intermediate_width_from_silu_shape"]
        * layers
        * rows,
        "body_dynamic_quantization_query_rows": 4 * layers * rows,
        "body_representation_obligation_sites_executed": obligations_per_forward * output_tokens,
        "body_representation_obligation_layer_query_row_sites": obligations_per_forward * rows,
        "widest_bias_free_i8_dot_bound": dot_bound,
        "minimum_information_bits_for_symmetric_dot_domain": (2 * dot_bound).bit_length(),
        "source_opening_scenarios_not_protocols": source_scenarios,
        "private_boundary_cost_bytes": None,
        "plaintext_internal_reconstructions_allowed": 0,
        "trusted_client_final_hidden_reconstructions": output_tokens,
        "whole_decoder_executable": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic", action="store_true", help="include cached-config compiler cost screens"
    )
    args = parser.parse_args()
    result = {
        "scope": "synthetic exact-integer oracle; no secure runtime or quality claim",
        "arithmetic": arithmetic_probe(),
        "bases": [
            basis_report(m)
            for m in (
                (16, 17),
                (16, 17, 19),
                (251, 241, 239),
                (251, 241, 239, 3),
                (251, 241, 239, 13),
                (251, 241, 239, 233),
            )
        ],
    }
    if args.semantic:
        result["semantic_cohorts"] = [semantic_screen(n) for n in (8, 32)]
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
