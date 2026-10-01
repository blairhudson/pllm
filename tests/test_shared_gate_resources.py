"""Compiler-bound resource veto for the dense two-input FSS reference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllm import Model, lower_model
from pllm.metrics import ResidentFusedGateCostProbe, ResidentQuadraticGateCostProbe
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientPrefixLayers
from pllm.runtime.shared_resources import SharedResourceError

from test_shared_resources import CONFIG
from test_logrow_tensor_native import QWEN3


@pytest.mark.parametrize("bits", [4, 8])
def test_pinned_qwen_layer_fails_online_and_material_budgets(bits: int) -> None:
    plan = lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=1)
    composition = MaskedLinearCpu(Model("Qwen/Qwen2.5-0.5B-Instruct"))
    probe = ResidentFusedGateCostProbe(
        domain_bits=bits,
        maximum_material_bytes_per_party=256 << 20,
        maximum_online_all_link_body_bytes=6_000_000,
        maximum_online_body_bytes_per_layer=250_000,
    )
    report = probe.run(plan, composition, response_new_tokens=1)
    assert report["plan_digest"] == plan.digest
    assert report["schedule_digest"] == plan.runtime_schedule(composition).digest
    assert report["domain_bits"] == bits
    assert len(report["layers"]) == 24
    assert {row["gated_elements"] for row in report["layers"]} == {39 * 4864}
    expected_online_per_layer = 39 * 4864 * (2 if bits == 4 else 4)
    expected_key_per_layer = 39 * 4864 * ((1 << bits) ** 2 * 8 + 2)
    assert all(row["minimum_online_all_link_body_bytes"] == expected_online_per_layer for row in report["layers"])
    assert all(row["minimum_material_body_bytes_per_party"] == expected_key_per_layer for row in report["layers"])
    assert report["minimum_online_all_link_body_bytes"] == 24 * expected_online_per_layer
    assert report["minimum_material_body_bytes_per_party"] == 24 * expected_key_per_layer
    assert report["material_within_budget"] is False
    assert report["online_within_budget"] is False
    assert report["whole_layer_executable"] is False
    assert report["decoder_material_issuance_admitted"] is False


def test_small_plan_can_pass_optimistic_gate_without_claiming_coverage() -> None:
    source = Model("org/small")
    configuration = {
        **CONFIG, "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
    }
    plan = lower_model(configuration, batch=1, max_input_tokens=2, max_new_tokens=2)
    composition = MaskedLinearCpu(source)
    probe = ResidentFusedGateCostProbe(4, 256 << 20, 6_000_000, 250_000)
    report = probe.run(plan, composition, response_new_tokens=2)
    assert [row["gated_elements"] for row in report["layers"]] == [3 * 256] * 2
    assert report["online_within_budget"] and report["material_within_budget"]
    assert report["whole_layer_executable"] is False


def test_fused_projection_rejects_incompatible_numeric_and_placement() -> None:
    plan = lower_model(CONFIG, batch=1, max_input_tokens=2, max_new_tokens=1)
    source = Model("Qwen/Qwen2.5-0.5B-Instruct")
    probe = ResidentFusedGateCostProbe(4, 256 << 20, 6_000_000, 250_000)
    with pytest.raises(SharedResourceError, match="W8A8"):
        probe.run(
            plan,
            MaskedLinearCpu(source, quantization=SymmetricPerRow(weight_bits=4, activation_bits=4)),
            response_new_tokens=1,
        )
    with pytest.raises(SharedResourceError, match="baseline"):
        probe.run(
            plan, MaskedLinearCpu(source, placement=ClientPrefixLayers(layers=1)),
            response_new_tokens=1,
        )
    with pytest.raises(SharedResourceError, match="decode key bound"):
        probe.run(plan, MaskedLinearCpu(source), response_new_tokens=5)


def test_pinned_qwen_quadratic_gate_passes_only_optimistic_gate_budget() -> None:
    plan = lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=1)
    composition = MaskedLinearCpu(Model("Qwen/Qwen2.5-0.5B-Instruct"))
    report = ResidentQuadraticGateCostProbe(
        256 << 20, 6_000_000, 250_000,
    ).run(plan, composition, response_new_tokens=1)
    assert report["schedule_digest"] == plan.runtime_schedule(composition).digest
    assert len(report["layers"]) == 24
    assert {row["hidden_elements"] for row in report["layers"]} == {39 * 896}
    assert {row["gated_elements"] for row in report["layers"]} == {39 * 4864}
    assert {row["minimum_online_all_link_body_bytes"] for row in report["layers"]} == {209_664}
    assert {row["minimum_material_body_bytes_per_party"] for row in report["layers"]} == {2_950_272}
    assert report["minimum_online_all_link_body_bytes"] == 5_031_936
    assert report["minimum_material_body_bytes_per_party"] == 70_806_528
    assert report["minimum_dealer_to_both_parties_body_bytes"] == 141_613_056
    assert report["gate_only_material_within_budget"] is True
    assert report["gate_only_online_within_budget"] is True
    assert report["source_scale_and_range_verified_for_model"] is False
    assert report["protected_numerator_to_model_scale"] is False
    assert report["whole_layer_executable"] is False
    assert report["decoder_material_issuance_admitted"] is False


def test_quadratic_gate_vetoes_different_budget_and_source_contract() -> None:
    plan = lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=1)
    composition = MaskedLinearCpu(Model("Qwen/Qwen2.5-0.5B-Instruct"))
    too_small = ResidentQuadraticGateCostProbe(1 << 20, 1 << 20, 200_000)
    report = too_small.run(plan, composition, response_new_tokens=1)
    assert not report["gate_only_material_within_budget"]
    assert not report["gate_only_online_within_budget"]
    assert not report["decoder_material_issuance_admitted"]
    with pytest.raises(SharedResourceError, match="baseline"):
        too_small.run(
            plan,
            MaskedLinearCpu(composition.model, placement=ClientPrefixLayers(layers=1)),
            response_new_tokens=1,
        )


def test_two_distinct_semantic_sources_veto_qwen_layer_before_attention_work() -> None:
    plan = lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=1)
    composition = MaskedLinearCpu(Model("Qwen/Qwen2.5-0.5B-Instruct"))
    report = ResidentQuadraticGateCostProbe(
        256 << 20, 6_000_000, 250_000,
    ).run_two_source_layer_bound(plan, composition, response_new_tokens=1)
    assert {row["independent_attention_source_elements"] for row in report["layers"]} == {39 * 896}
    assert {row["minimum_two_sources_online_all_link_body_bytes"] for row in report["layers"]} == {419_328}
    assert report["minimum_online_all_link_body_bytes"] == 10_063_872
    assert report["minimum_material_body_bytes_per_party"] == 73_322_496
    assert report["online_within_budget"] is False
    assert report["material_within_budget"] is True
    assert report["decoder_material_issuance_admitted"] is False


def test_semantic_attention_source_works_for_dense_qwen3_without_id_parsing() -> None:
    plan = lower_model(QWEN3, batch=1, max_input_tokens=2, max_new_tokens=1)
    composition = MaskedLinearCpu(Model("org/small"))
    report = ResidentQuadraticGateCostProbe(
        256 << 20, 6_000_000, 250_000,
    ).run_two_source_layer_bound(plan, composition, response_new_tokens=1)
    assert [row["hidden_elements"] for row in report["layers"]] == [2 * 8]
    assert [row["independent_attention_source_elements"] for row in report["layers"]] == [2 * 8]
    assert report["minimum_online_all_link_body_bytes"] == 2 * 2 * 8 * 3 * 2


def test_reported_qwen_cost_evidence_matches_locked_compiler_estimates() -> None:
    evidence = json.loads(
        (Path(__file__).parents[1] / "docs/evidence/resident-quadratic-layer-gate-2026-09-28.json")
        .read_text(encoding="utf-8")
    )
    plan = lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=1)
    composition = MaskedLinearCpu(Model("Qwen/Qwen2.5-0.5B-Instruct"))
    budget = evidence["budgets"]
    probe = ResidentQuadraticGateCostProbe(
        budget["one_use_material_bytes_per_party"],
        budget["online_all_link_body_bytes"],
        budget["online_body_bytes_per_layer"],
    )
    reports = {
        "dense_q3_two_input_lookup": ResidentFusedGateCostProbe(
            4, budget["one_use_material_bytes_per_party"],
            budget["online_all_link_body_bytes"], budget["online_body_bytes_per_layer"],
        ).run(plan, composition, response_new_tokens=1),
        "dense_q7_two_input_lookup": ResidentFusedGateCostProbe(
            8, budget["one_use_material_bytes_per_party"],
            budget["online_all_link_body_bytes"], budget["online_body_bytes_per_layer"],
        ).run(plan, composition, response_new_tokens=1),
        "quadratic_q7_correlated_mlp_source": probe.run(plan, composition, response_new_tokens=1),
        "quadratic_q7_distinct_attention_and_mlp_sources": probe.run_two_source_layer_bound(
            plan, composition, response_new_tokens=1,
        ),
    }
    assert evidence["source"]["plan_digest"] == plan.digest
    for name, report in reports.items():
        assert evidence["source"]["schedule_digest"] == report["schedule_digest"]
        assert evidence["source"]["composition_digest"] == report["composition_digest"]
        row = evidence["candidates"][name]
        assert row["minimum_online_all_link_body_bytes"] == report["minimum_online_all_link_body_bytes"]
        assert row["minimum_material_body_bytes_per_party"] == report["minimum_material_body_bytes_per_party"]
        assert row["minimum_dealer_to_both_parties_body_bytes"] == report["minimum_dealer_to_both_parties_body_bytes"]
        per_layer_key = (
            "minimum_two_sources_online_all_link_body_bytes"
            if name == "quadratic_q7_distinct_attention_and_mlp_sources"
            else "minimum_online_all_link_body_bytes"
        )
        assert row["minimum_online_body_bytes_per_layer"] == report["layers"][0][per_layer_key]
