"""Fail-fast cost gate for the bounded share-resident MLP reference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.shared_resources import SharedResourceError, resident_mlp_resource_gate


CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
}


def test_reference_gated_mlp_exceeds_both_pinned_qwen25_budgets() -> None:
    plan = lower_model(CONFIG, batch=1, max_input_tokens=30, max_new_tokens=1)
    composition = MaskedLinearCpu(Model("Qwen/Qwen2.5-0.5B-Instruct"))
    report = resident_mlp_resource_gate(
        plan, composition, response_new_tokens=1, fixed_scale_bits=8,
        maximum_material_bytes_per_party=256 << 20,
        maximum_online_all_link_body_bytes=95_805_056,
    )
    assert report["plan_digest"] == plan.digest
    assert report["schedule_digest"] == plan.runtime_schedule(composition).digest
    assert report["prefill_elements"] == 24 * 30 * 4864
    assert report["decode_elements"] == 0
    assert report["minimum_material_body_bytes_per_party"] == 735_436_800
    assert report["minimum_material_body_bytes_two_parties"] == 1_470_873_600
    assert report["minimum_online_all_link_body_bytes"] == 168_975_360
    assert report["material_within_budget"] is False
    assert report["online_within_budget"] is False
    assert report["executable"] is False
    assert report["full_wire_measured"] is False
    evidence = json.loads(
        (Path(__file__).parents[1] / "docs/evidence/shared-resident-mlp-cost-gate-2026-09-27.json")
        .read_text(encoding="utf-8")
    )
    for key in ("schema", "response_new_tokens", "fixed_scale_bits", "prefill_elements", "decode_elements"):
        evidence_key = {
            "response_new_tokens": "output_tokens",
            "fixed_scale_bits": "fixed_scale_bits",
            "prefill_elements": "prefill_gated_product_elements",
            "decode_elements": "decode_gated_product_elements",
        }.get(key, key)
        source = evidence["cohort"] if key != "schema" else evidence
        assert report[key] == source[evidence_key]
    for key in (
        "minimum_material_body_bytes_per_party", "minimum_material_body_bytes_two_parties",
        "minimum_online_all_link_body_bytes", "material_within_budget", "executable",
        "full_wire_measured",
    ):
        assert report[key] == evidence[key]
    assert report["plan_digest"] == evidence["source"]["plan_digest"]
    assert report["schedule_digest"] == evidence["source"]["schedule_digest"]
    assert report["composition_digest"] == evidence["source"]["composition_digest"]


def test_small_plan_is_distinct_and_unsupported_scale_fails_closed() -> None:
    source = Model("org/small")
    plan = lower_model({**CONFIG, "hidden_size": 128, "intermediate_size": 256,
                        "num_hidden_layers": 2, "num_attention_heads": 4,
                        "num_key_value_heads": 2},
                       batch=1, max_input_tokens=2, max_new_tokens=2)
    composition = MaskedLinearCpu(source)
    report = resident_mlp_resource_gate(
        plan, composition, response_new_tokens=2, fixed_scale_bits=8,
        maximum_material_bytes_per_party=256 << 20,
        maximum_online_all_link_body_bytes=95_805_056,
    )
    assert report["prefill_elements"] == 2 * 2 * 256
    assert report["decode_elements"] == 2 * 256
    assert report["material_within_budget"] is True
    assert report["online_within_budget"] is True
    with pytest.raises(SharedResourceError, match="FSS reference"):
        resident_mlp_resource_gate(
            plan, composition, response_new_tokens=2, fixed_scale_bits=16,
            maximum_material_bytes_per_party=256 << 20,
            maximum_online_all_link_body_bytes=95_805_056,
        )
    with pytest.raises(SharedResourceError, match="budget"):
        resident_mlp_resource_gate(
            plan, composition, response_new_tokens=2, fixed_scale_bits=8,
            maximum_material_bytes_per_party=0,
            maximum_online_all_link_body_bytes=95_805_056,
        )
    with pytest.raises(SharedResourceError, match="W8A8"):
        resident_mlp_resource_gate(
            plan, MaskedLinearCpu(source, quantization=SymmetricPerRow(
                weight_bits=4, activation_bits=4,
            )), response_new_tokens=2, fixed_scale_bits=8,
            maximum_material_bytes_per_party=256 << 20,
            maximum_online_all_link_body_bytes=95_805_056,
        )
    with pytest.raises(SharedResourceError, match="decode key bound"):
        resident_mlp_resource_gate(
            plan, composition, response_new_tokens=3, fixed_scale_bits=8,
            maximum_material_bytes_per_party=256 << 20,
            maximum_online_all_link_body_bytes=95_805_056,
        )
