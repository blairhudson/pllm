"""Independent exact oracles and rounded counterexamples for projective claims."""

from __future__ import annotations

import copy
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.modeling import ModelPlan
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow

_PATH = Path(__file__).parents[1] / "scripts/probe_projective_feasibility.py"
_SPEC = importlib.util.spec_from_file_location("projective_probe", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_cross_comparison_rounding_handles_sign_ties_and_saturation() -> None:
    # Independent exact Fraction round, not the comparison implementation.
    for denominator in (1, 2, 3, 7, 128):
        for numerator in range(-2048, 2049):
            expected = np.clip(round(Fraction(numerator, denominator)), -127, 127)
            assert probe.round_ratio_comparisons(numerator, denominator) == expected
    assert probe.round_ratio_comparisons(5, 2) == 2
    assert probe.round_ratio_comparisons(7, 2) == 4
    assert probe.round_ratio_comparisons(-7, 2) == -4
    with pytest.raises(ValueError):
        probe.round_ratio_comparisons(1, 0)


def test_summary_composition_matches_independent_rational_weighted_average() -> None:
    leaves = [(i, Fraction(1), Fraction(v)) for i, v in ((-9, 4), (-2, -7), (6, 2), (1, 9))]
    left = probe.merge_summary(
        probe.merge_summary(leaves[0], leaves[1]), probe.merge_summary(leaves[2], leaves[3])
    )
    right = probe.merge_summary(
        leaves[0], probe.merge_summary(leaves[1], probe.merge_summary(leaves[2], leaves[3]))
    )
    raw_weights = [Fraction(2) ** row[0] for row in leaves]
    independent = sum(w * row[2] for w, row in zip(raw_weights, leaves, strict=True)) / sum(
        raw_weights
    )
    assert left == right
    assert left[2] / left[1] == independent
    with pytest.raises(ValueError):
        probe.stable_summary(np.asarray([-np.inf]), np.asarray([[1.0]]))


def test_epsilon_and_rounded_edges_falsify_whole_decoder_cancellation() -> None:
    report = probe.rounded_probe()
    assert report["attention_float32_changed_rows"] > 0
    assert report["attention_targeted_q7_changed_rows"] > 0
    assert report["float64_stable_summary_max_absolute_error"] < 1e-12
    norm = report["rmsnorm_nonzero_epsilon"]
    assert norm["bogus_max_absolute_error"] > 0.1
    np.testing.assert_allclose(norm["runtime"], norm["legal_real_identity_float32"], rtol=1e-6)
    assert any(
        case["changed_code_rows"] for case in report["dynamic_w8a8_shared_denominator_cases"]
    )
    assert report["pade_silu"]["q7_changed_elements"] > 0
    shift = report["softmax_shift_counterexample"]
    assert shift["original_probabilities"] != shift["shifted_probabilities"]


def test_compiler_shapes_and_coverage_drive_projective_inventory() -> None:
    config = {
        "model_type": "qwen2",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "max_position_embeddings": 32768,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
    }
    plan = lower_model(config, batch=1, max_input_tokens=3, max_new_tokens=8)
    composition = MaskedLinearCpu(
        Model("synthetic/projective-inventory"),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    report = probe.compiler_inventory(plan, composition, 8)
    assert report["executed_rows"] == 10
    assert report["response_tensor_counts"]["body_norm_rows"] == 40
    assert report["response_tensor_counts"]["attention_head_rows"] == 80
    assert report["response_tensor_counts"]["silu_elements"] == 5120
    assert report["body_quantized_stage_row_boundaries"] == 80
    assert report["visible_causal_score_pairs"] == 2 * 4 * (6 + sum(range(4, 11)))
    assert report["dense_runtime_score_pairs"] == 2 * 4 * (9 + sum(range(4, 11)))
    assert report["optimistic_nonaffine_product_serial_depth"] == 8
    assert not report["complete_protected_layer_available"]
    # Corrupt semantic lineage; no stage-name heuristic may repair it.
    altered = copy.deepcopy(plan.to_dict())
    score = next(
        row for row in altered["prefill"]["operations"] if row["operator"] == "attention_scores"
    )
    score["inputs"][0] = "missing.query"
    with pytest.raises(ValueError):
        probe.compiler_inventory(ModelPlan(json.dumps(altered).encode()), composition, 8)
