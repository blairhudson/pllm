"""Bounded projective identities, rounded-edge falsification, and compiler costs.

Synthetic tensors only. Optional pinned config is read from local cache; no
weights, network execution, material issuance, or protected runtime activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from pllm.runtime.quantization import quantize_activation_per_row
from pllm.runtime.semantic_executor import SemanticDecoderRuntime

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 20261001


def local(kind: str, *values: np.ndarray, epsilon: str = "1e-6") -> np.ndarray:
    """Execute actual scalar semantic operators without a model/weight bundle."""
    attrs = {"epsilon": epsilon, "weight": "synthetic", "weight_offset": 0}
    runtime = SimpleNamespace(
        nonlinear_evaluator=None,
        _causal_reduction=None,
        _weight=lambda _: np.ones(values[0].shape[-1], dtype=np.float32),
    )
    inputs = [f"source{i}" for i in range(len(values))]
    operation = {
        "operator": kind,
        "inputs": inputs,
        "attributes": attrs if kind == "rms_norm" else {},
        "layer": 0,
    }
    return SemanticDecoderRuntime._local(
        runtime, operation, dict(zip(inputs, values, strict=True)), {}, {}
    )


def round_ratio_comparisons(numerator: int, denominator: int, limit: int = 127) -> int:
    """Clipped rational ties-even via bounded comparisons, no division/inverse.

    Reference arithmetic only: Python branches are not a private implementation.
    A protected implementation must hide comparisons and use fixed-size traffic.
    """
    if denominator <= 0 or limit < 1:
        raise ValueError("positive denominator and clipping limit required")
    magnitude = abs(numerator)
    lower, upper = 0, limit
    for _ in range((limit + 1).bit_length()):
        middle = (lower + upper) >> 1
        threshold = (2 * middle + 1) * denominator
        above = 2 * magnitude > threshold or (2 * magnitude == threshold and middle % 2 == 1)
        if lower < upper:
            if above:
                lower = middle + 1
            else:
                upper = middle
    return -lower if numerator < 0 else lower


def rational_silu(value: Fraction) -> Fraction:
    """SiLU from [2/2] Pade exp; degree 3/2, approximation, not checkpoint SiLU."""
    return value * (value * value + 6 * value + 12) / (2 * value * value + 24)


def merge_summary(left: tuple, right: tuple) -> tuple:
    """Exact base-two exponential summary, validating merge independently."""
    lm, ll, ln = left
    rm, rl, rn = right
    maximum = max(lm, rm)
    a, b = Fraction(2) ** (lm - maximum), Fraction(2) ** (rm - maximum)
    return maximum, a * ll + b * rl, a * ln + b * rn


def exact_algebra_probe() -> dict[str, Any]:
    count = 0
    for d in range(1, 12):
        for n in range(-400, 401):
            expected = max(-127, min(127, round(Fraction(n, d))))
            assert round_ratio_comparisons(n, d) == expected
            count += 1
    # Independently evaluate rational/homogeneous forms, including residuals.
    homogeneous = squared_norms = 0
    for n in range(-8, 9):
        for d in range(1, 6):
            numerator = n**3 + 6 * n * n * d + 12 * n * d * d
            denominator = 2 * n * n * d + 24 * d**3
            assert Fraction(numerator, denominator) == rational_silu(Fraction(n, d))
            for u, e in ((-3, 2), (5, 7)):
                assert Fraction(n * e + u * d, d * e) == Fraction(n, d) + Fraction(u, e)
            # Carry x_i = a_i/sqrt(s): compare squared normalized values using
            # independent rational variance evaluation, retaining epsilon.
            vector = [Fraction(n), Fraction(n + 1), Fraction(-2 * n)]
            s, epsilon = Fraction(d * d), Fraction(1, 1_000_000)
            source_squares = [a * a / s for a in vector]
            variance = sum(source_squares) / len(vector)
            next_s = sum(a * a for a in vector) / len(vector) + epsilon * s
            assert [a / (variance + epsilon) for a in source_squares] == [
                a * a / next_s for a in vector
            ]
            squared_norms += 1
            homogeneous += 1
    # Powers of two are exact rational stand-ins for exp(i*log(2)), not float exp.
    summaries = [(i, Fraction(1), Fraction(v)) for i, v in ((-3, 7), (2, -4), (1, 9))]
    a, b, c = summaries
    merged = merge_summary(merge_summary(a, b), c)
    assert merged == merge_summary(a, merge_summary(b, c))
    assert merged == merge_summary(c, merge_summary(b, a))
    raw_l = sum(Fraction(2) ** i for i, _, _ in summaries)
    raw_n = sum(Fraction(2) ** i * v for i, _, v in summaries)
    assert merged[2] / merged[1] == raw_n / raw_l
    return {
        "rational_quantization_comparison_cases": count,
        "homogeneous_rational_silu_and_residual_cases": homogeneous,
        "squared_denominator_rmsnorm_cases": squared_norms,
        "exact_base_two_summary_associative_and_commutative": True,
        "exact_summary_value": str(merged[2] / merged[1]),
        "protected_comparison_protocol_available": False,
    }


def stable_summary(scores: np.ndarray, values: np.ndarray) -> tuple:
    if not np.any(np.isfinite(scores)):
        raise ValueError("summary requires at least one visible score")
    maximum = np.max(scores)
    weights = np.exp(scores - maximum)
    return maximum, weights.sum(), weights @ values


def merge_float(left: tuple, right: tuple) -> tuple:
    lm, ll, ln = left
    rm, rl, rn = right
    maximum = np.maximum(lm, rm)
    a, b = np.exp(lm - maximum), np.exp(rm - maximum)
    return maximum, a * ll + b * rl, a * ln + b * rn


def rounded_probe() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    summary_error = 0.0
    float_mismatches = q7_mismatches = 0
    first_attention = None
    # Four value features; one KV head, one query, eight visible scores.
    for _ in range(256):
        scores = rng.uniform(-8, 8, 8).astype(np.float32)
        values = rng.uniform(-1, 1, (8, 4)).astype(np.float32)
        probabilities = local("softmax", scores)
        # Use the runtime's grouped einsum operator with its declared group size.
        operation = {
            "operator": "attention_values",
            "inputs": ["p", "v"],
            "attributes": {"group_size": 1},
            "layer": 0,
        }
        baseline = SemanticDecoderRuntime._local(
            SimpleNamespace(_causal_reduction=None),
            operation,
            {"p": probabilities.reshape(1, 1, 1, 8), "v": values.reshape(1, 1, 8, 4)},
            {},
            {},
        ).reshape(4)
        weights = np.exp(scores - scores.max()).astype(np.float32)
        delayed = np.einsum("t,td->d", weights, values) / weights.sum(dtype=np.float32)
        float_mismatches += int(not np.array_equal(baseline, delayed))
        # Deliberately stress a Q7 half-ulp: constant first value is 1/256.
        # This is synthetic threshold evidence, not frequency on real prompts.
        values[:, 0] = np.float32(1 / 256)
        near = np.einsum("t,td->d", probabilities, values)
        late = np.einsum("t,td->d", weights, values) / weights.sum(dtype=np.float32)
        q_near = quantize_activation_per_row(near, bits=8, scales=1 / 128).values
        q_late = quantize_activation_per_row(late, bits=8, scales=1 / 128).values
        if not np.array_equal(q_near, q_late):
            q7_mismatches += 1
            if first_attention is None:
                first_attention = {
                    "scores": scores.tolist(),
                    "baseline": near.tolist(),
                    "delayed": late.tolist(),
                    "q7_baseline": q_near.tolist(),
                    "q7_delayed": q_late.tolist(),
                }
        full = stable_summary(scores.astype(np.float64), values.astype(np.float64))
        chunks = [
            stable_summary(scores[s].astype(np.float64), values[s].astype(np.float64))
            for s in (slice(0, 3), slice(3, 5), slice(5, 8))
        ]
        merged = merge_float(merge_float(chunks[0], chunks[1]), chunks[2])
        summary_error = max(
            summary_error, float(np.max(np.abs(full[2] / full[1] - merged[2] / merged[1])))
        )
    # Positive shared-denominator cancellation at dynamic W8A8 half thresholds.
    numerators = np.column_stack(
        (
            np.ones(127, np.float32),
            (np.arange(127, dtype=np.float32) + np.float32(0.5)) / np.float32(127),
        )
    )
    q_n = quantize_activation_per_row(numerators, bits=8).values
    scaling_cases = []
    for denominator in (3.0, 10.0, 0.001, 128.0):
        divided = numerators / np.float32(denominator)
        q_div = quantize_activation_per_row(divided, bits=8).values
        indices = np.flatnonzero(np.any(q_n != q_div, axis=1))
        first = int(indices[0]) if indices.size else None
        scaling_cases.append(
            {
                "denominator": denominator,
                "rows": len(numerators),
                "changed_code_rows": int(indices.size),
                "first_numerator": None if first is None else numerators[first].tolist(),
                "first_q_n": None if first is None else q_n[first].tolist(),
                "first_q_div": None if first is None else q_div[first].tolist(),
            }
        )
    n = np.asarray([1e-4, -2e-4, 3e-4], dtype=np.float32)
    d = np.float32(3)
    norm = local("rms_norm", n / d)
    legal = n / np.sqrt(np.mean(n * n) + np.float32(1e-6) * d * d)
    bogus = local("rms_norm", n)
    # Shift invariance cannot restore bits lost by adding a large public constant.
    small = np.asarray([0, 1, 2], dtype=np.float32)
    shifted = small + np.float32(2**25)
    grid = np.linspace(-8, 8, 2049, dtype=np.float32)
    exact_silu = local("silu", grid)
    rational = grid * (grid * grid + 6 * grid + 12) / (2 * grid * grid + 24)
    q_exact = quantize_activation_per_row(exact_silu, bits=8, scales=1 / 128).values
    q_rational = quantize_activation_per_row(rational, bits=8, scales=1 / 128).values
    return {
        "attention_trials": 256,
        "attention_float32_changed_rows": float_mismatches,
        "attention_targeted_q7_changed_rows": q7_mismatches,
        "attention_first_q7_counterexample": first_attention,
        "float64_stable_summary_max_absolute_error": summary_error,
        "dynamic_w8a8_shared_denominator_cases": scaling_cases,
        "rmsnorm_nonzero_epsilon": {
            "numerator": n.tolist(),
            "denominator": float(d),
            "runtime": norm.tolist(),
            "legal_real_identity_float32": legal.tolist(),
            "bogus_drop_denominator": bogus.tolist(),
            "bogus_max_absolute_error": float(np.max(np.abs(norm - bogus))),
        },
        "softmax_shift_counterexample": {
            "original": small.tolist(),
            "shifted": shifted.tolist(),
            "original_probabilities": local("softmax", small).tolist(),
            "shifted_probabilities": local("softmax", shifted).tolist(),
        },
        "pade_silu": {
            "grid_points": len(grid),
            "domain": [-8, 8],
            "max_absolute_error": float(np.max(np.abs(exact_silu - rational))),
            "q7_changed_elements": int(np.count_nonzero(q_exact != q_rational)),
            "denominator_min": 24,
            "degree_numerator": 3,
            "degree_denominator": 2,
        },
        "synthetic_only": True,
        "checkpoint_quality_measured": False,
    }


def compiler_inventory(plan: Any, composition: Any, output_tokens: int) -> dict[str, Any]:
    """Count compiler operators/shapes, validating all schedule operation IDs."""
    from pllm.runtime.he_layer_feasibility import token_boundary_he_layer_gate
    from pllm.runtime.region_contract_cost import compiler_region_contract_cost
    from pllm.runtime.semantic_stages import scheduled_stage_specs

    all_budget, online_budget = {8: (11684396, 7383174), 32: (17897055, 11354502)}[output_tokens]
    region = compiler_region_contract_cost(
        plan,
        composition,
        response_new_tokens=output_tokens,
        maximum_online_all_link_body_bytes=online_budget,
        maximum_total_all_link_body_bytes=all_budget,
    )
    # Complete attention/norm/KV/MLP and causal depth validation, not HE execution.
    depth = token_boundary_he_layer_gate(
        plan,
        composition,
        response_new_tokens=output_tokens,
        covered_all_link_body_budget_bytes=all_budget,
        covered_online_body_budget_bytes=online_budget,
    )
    graph, schedule = plan.to_dict(), plan.runtime_schedule(composition).to_dict()
    totals: Counter[str] = Counter()
    phase_counts = {}
    for phase, repetitions in (("prefill", 1), ("decode", output_tokens - 1)):
        ops = graph[phase]["operations"]
        seen = Counter(
            identity for step in schedule[phase]["steps"] for identity in step["operation_ids"]
        )
        if seen != Counter(row["id"] for row in ops):
            raise ValueError("runtime schedule operation coverage is not exact")
        phase_counts[phase] = dict(sorted(Counter(row["operator"] for row in ops).items()))
        for row in ops:
            if row.get("layer") is None:
                continue
            kind, shape = row["operator"], row["output_shape"]
            if kind == "rms_norm":
                totals["body_norm_rows"] += repetitions * math.prod(shape[:-1])
            if kind == "softmax":
                totals["attention_head_rows"] += repetitions * math.prod(shape[:-1])
            if kind == "silu":
                totals["silu_elements"] += repetitions * math.prod(shape)
            if kind == "residual_add":
                totals["residual_rows"] += repetitions * math.prod(shape[:-1])
                totals["residual_elements"] += repetitions * math.prod(shape)
    stages = scheduled_stage_specs(plan, composition)
    width = stages[0].out_features
    per_layer = [r for r in graph["prefill"]["operations"] if r.get("layer") == 0]
    score = next(r for r in per_layer if r["operator"] == "attention_scores")
    value = next(r for r in per_layer if r["operator"] == "attention_values")
    gated = next(r for r in per_layer if r["operator"] == "silu")
    heads, head_dim, intermediate = (
        score["output_shape"][1],
        value["output_shape"][-1],
        gated["output_shape"][-1],
    )
    prompt, layers = graph["prefill"]["query_sequence"], region["layer_count"]
    key_lengths = range(prompt + 1, prompt + output_tokens)
    # Visible causal domain vs dense scheduled prefill computation; decode uses
    # actual valid prefixes, not the maximum-capacity shape every step.
    visible_pairs = layers * heads * (prompt * (prompt + 1) // 2 + sum(key_lengths))
    dense_pairs = layers * heads * (prompt * prompt + sum(key_lengths))
    residual_rows = totals["residual_rows"]
    broadcast_online = 2 * (width + 1) * 4
    broadcast_dealer = 2 * (2 * width + 1) * 4
    floor12 = region["two_worker_resident"]["12"]["known_online_body_floor_bytes"]
    return {
        "output_tokens": output_tokens,
        "executed_rows": region["executed_rows"],
        "plan_digest": plan.digest,
        "schedule_digest": region["schedule_digest"],
        "composition_digest": region["composition_digest"],
        "semantic_layers": layers,
        "hidden": width,
        "heads": heads,
        "head_dim": head_dim,
        "intermediate": intermediate,
        "phase_operator_counts": phase_counts,
        "response_tensor_counts": dict(totals),
        "visible_causal_score_pairs": visible_pairs,
        "dense_runtime_score_pairs": dense_pairs,
        "qk_and_probability_v_macs_each": dense_pairs * head_dim,
        "body_projection_macs_per_worker": region["two_worker_resident"]["12"][
            "remote_body_projection_integer_macs_per_worker"
        ],
        "body_quantized_stage_row_boundaries": 4 * layers * region["executed_rows"],
        "optimistic_nonaffine_product_serial_depth": depth["semantic_phase_contracts"]["prefill"][
            "optimistic_serial_ciphertext_product_depth"
        ],
        "all_link_budget_bytes": all_budget,
        "online_budget_bytes": online_budget,
        "resident_12_known_online_floor_bytes": floor12,
        "resident_12_remaining_online_bytes": online_budget - floor12,
        "resident_12_remaining_all_link_bytes": all_budget - floor12,
        "projected_residual_alignment_two_secret_denominators": {
            "broadcast_products": 2 * residual_rows,
            "online_peer_bodies_bytes": 2 * residual_rows * broadcast_online,
            "dealer_to_both_bodies_bytes": 2 * residual_rows * broadcast_dealer,
            "excluded": [
                "denominator products",
                "private inverses/exponents/sqrt",
                "rounding",
                "all other layer work",
                "wire/control",
            ],
        },
        "attention_denominator_32bit_two_share_storage_bytes": totals["attention_head_rows"] * 8,
        "kv_32bit_share_storage_bytes_per_worker": sum(
            math.prod(r["shape"]) * 4 for r in graph["decode"]["state_inputs"]
        ),
        "complete_protected_layer_available": False,
    }


def pinned_inventory(output_tokens: int) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    from pllm import Model, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow

    cache = os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    source = Path(
        hf_hub_download(
            MODEL, "config.json", revision=REVISION, local_files_only=True, cache_dir=cache
        )
    )
    config_body = source.read_bytes()
    plan = lower_model(
        json.loads(config_body), batch=1, max_input_tokens=39, max_new_tokens=output_tokens
    )
    composition = MaskedLinearCpu(
        Model.hf(MODEL, revision=REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    evidence = json.loads(
        (ROOT / "docs/evidence/latent-response-network-qwen25-2026-09-28.json").read_text()
    )
    control = next(c for c in evidence["cohorts"] if c["output_tokens"] == output_tokens)
    report = compiler_inventory(plan, composition, output_tokens)
    if (
        report["plan_digest"] != control["official_plan_digest"]
        or report["schedule_digest"] != control["official_schedule_digest"]
        or report["composition_digest"] != evidence["source"]["pipeline_digest"]
    ):
        raise ValueError("compiler/source/composition differs from pinned control")
    return {
        **report,
        "checkpoint": f"{MODEL}@{REVISION}",
        "config_sha256": hashlib.sha256(config_body).hexdigest(),
        "body_fingerprint_from_control": evidence["source"]["body_fingerprint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-pinned-config", action="store_true")
    args = parser.parse_args()
    result = {
        "schema": "pllm.projective_feasibility.v1",
        "seed": SEED,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "exact_algebra": exact_algebra_probe(),
        "rounded_edges": rounded_probe(),
    }
    if args.with_pinned_config:
        result["compiler_cohorts"] = [pinned_inventory(n) for n in (8, 32)]
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
