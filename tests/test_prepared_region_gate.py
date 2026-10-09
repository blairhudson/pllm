from __future__ import annotations

from dataclasses import replace

import pytest

from pllm import Model, lower_model
from pllm.metrics import benchmark_operator
from pllm.nonlinear import LogRowGatedMultiplyQ9
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from scripts.probe_prepared_region_gate import project


@pytest.fixture(scope="module")
def scalar_report():
    return benchmark_operator(LogRowGatedMultiplyQ9(), lanes=1, samples=1)


def inputs(model_type="qwen2", outputs=3):
    config = {
        "model_type": model_type, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "vocab_size": 128, "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu",
        "tie_word_embeddings": True, "attention_dropout": 0.0, "attention_bias": False,
        "use_cache": True, "use_sliding_window": False, "sliding_window": None,
        "max_window_layers": 2, "layer_types": ["full_attention"] * 2,
        "rope_scaling": None,
    }
    plan = lower_model(config, batch=1, max_input_tokens=5, max_new_tokens=outputs)
    composition = MaskedLinearCpu(Model("fixture/model"),
                                 quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
    return plan, composition


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
@pytest.mark.parametrize("outputs", [1, 3])
def test_one_evaluator_cost_conserves_fresh_material_and_counts_n_minus_one_decode(
        scalar_report, model_type, outputs):
    plan, composition = inputs(model_type, outputs)
    result = project(plan, composition, scalar_report, output_tokens=outputs, covered_body_budget=100)
    elements = 2 * 64 * (5 + outputs - 1)
    assert result["gated_elements"] == elements
    assert result["prefill_elements"] == 2 * 64 * 5
    assert result["decode_elements"] == 2 * 64 * (outputs - 1)
    client, evaluator = scalar_report.resources.raw_key_payload_bytes_per_party
    assert result["client_owned_encoding_bytes"] == elements * client
    assert result["client_issued_hypothesis"]["required_material_payload_bytes"] == elements * evaluator
    offloaded = result["offline_preparation_issued_hypothesis"]
    assert offloaded["required_material_payload_bytes"] == elements * (client + evaluator)
    assert {(row["source"], row["destination"]) for row in offloaded["hypothetical_material_links"]} == {
        ("preparation", "inference"), ("preparation", "client")}
    # Constant labels are already in material; metadata and boundary estimates
    # must not be silently added again to the exact material-only floor.
    assert sum(row[2] for row in result["material_by_operation_and_owner"]) == elements * evaluator
    assert all(value == "veto_material_alone" for value in offloaded["budget_decisions"].values())
    assert not result["same_numeric_identity_as_control"]
    assert not result["offline_preparation_implemented"]
    assert not result["whole_decoder_executable"]
    assert result["full_wire_bytes"] is None
    generous = project(plan, composition, scalar_report, output_tokens=outputs, covered_body_budget=1 << 40)
    assert all(value == "inconclusive_missing_work"
               for value in generous["offline_preparation_issued_hypothesis"]["budget_decisions"].values())


def test_one_evaluator_screen_rejects_substituted_ownership_or_unchecked_material(scalar_report):
    plan, composition = inputs()
    resources = scalar_report.resources
    bad_reports = (
        replace(scalar_report, resources=replace(resources, role_names=("worker_a", "worker_b"))),
        replace(scalar_report, resources=replace(resources, peer_frame_bytes_by_direction=(1, 0))),
        replace(scalar_report, resources=replace(resources, raw_key_payload_bytes_per_party=(1, 1))),
        replace(scalar_report, resources=replace(resources, public_metadata_bytes_estimate=None)),
        replace(scalar_report, quality=replace(scalar_report.quality, exact_integer_outputs=0)),
        replace(scalar_report, configuration_digest="0" * 64),
        benchmark_operator(LogRowGatedMultiplyQ9(), lanes=2, samples=1),
    )
    for report in bad_reports:
        with pytest.raises(ValueError):
            project(plan, composition, report, output_tokens=3, covered_body_budget=1 << 20)
    for budget in (0, True, 1 << 41):
        with pytest.raises(ValueError):
            project(plan, composition, scalar_report, output_tokens=3, covered_body_budget=budget)
    with pytest.raises(ValueError):
        project(plan, composition, scalar_report, output_tokens=4, covered_body_budget=1 << 20)
