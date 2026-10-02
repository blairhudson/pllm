"""Whole-response and per-token necessities for bounded network research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan


@dataclass(frozen=True, slots=True)
class TokenNetworkBudgetProbe:
    """Price necessary interaction/material budgets; never admit a protocol."""

    reduction_factor: int = 100
    ring_bits: int = 24

    def __post_init__(self) -> None:
        if type(self.reduction_factor) is not int or not 2 <= self.reduction_factor <= 1000:
            raise ValueError("network reduction factor must be an integer from 2 to 1000")
        if type(self.ring_bits) is not int or self.ring_bits not in (24, 32, 64):
            raise ValueError("network budget scenario requires a 24/32/64-bit ring")

    def project(
        self,
        plan: ModelPlan,
        composition: Pipeline,
        *,
        response_new_tokens: int,
        baseline_online_body_bytes: int,
        baseline_total_body_bytes: int,
    ) -> dict[str, Any]:
        from pllm.runtime.region_contract_cost import compiler_region_contract_cost
        from pllm.runtime.semantic_stages import scheduled_stage_specs

        if (
            any(
                type(v) is not int or not self.reduction_factor <= v <= 1 << 40
                for v in (baseline_online_body_bytes, baseline_total_body_bytes)
            )
            or baseline_online_body_bytes > baseline_total_body_bytes
        ):
            raise ValueError("network budget requires bounded ordered baseline body costs")
        online = baseline_online_body_bytes // self.reduction_factor
        total = baseline_total_body_bytes // self.reduction_factor
        base = compiler_region_contract_cost(
            plan,
            composition,
            response_new_tokens=response_new_tokens,
            maximum_online_all_link_body_bytes=online,
            maximum_total_all_link_body_bytes=total,
        )
        rows = base["executed_rows"]
        sources = [
            layer[key]
            for layer in base["layer_provenance"]
            for key in ("attention_query_source_elements", "post_attention_mlp_source_elements")
        ]
        widths = [value // rows for value in sources]
        word = self.ring_bits // 8
        source_openings = 2 * sum(sources) * word
        minimum_opening_per_row = 2 * min(widths) * word
        stages = scheduled_stage_specs(plan, composition)
        gate_elements = rows * sum(
            stage.in_features for stage in stages if stage.role == "mlp_down"
        )
        token_width = next(stage.out_features for stage in stages if stage.role == "token_lookup")
        feedback = 2 * token_width * word * (rows + response_new_tokens)
        return {
            "schema": "pllm.token_network_budget.v1",
            "scope": "necessary body budgets from compiler dimensions, not executable protocol or full wire",
            "plan_digest": plan.digest,
            "composition_digest": base["composition_digest"],
            "schedule_digest": base["schedule_digest"],
            "reduction_factor": self.reduction_factor,
            "assumed_ring_bits": self.ring_bits,
            "generated_tokens": response_new_tokens,
            "input_tokens": base["input_tokens"],
            "executed_rows": rows,
            "baseline_online_body_bytes": baseline_online_body_bytes,
            "baseline_total_body_bytes": baseline_total_body_bytes,
            "response_online_budget_bytes": online,
            "response_all_link_budget_bytes": total,
            "amortized_online_budget_bytes_per_generated_token": online / response_new_tokens,
            "amortized_all_link_budget_bytes_per_generated_token": total / response_new_tokens,
            "online_budget_bytes_per_executed_row": online / rows,
            "all_link_budget_bytes_per_executed_row": total / rows,
            "gated_elements": gate_elements,
            "all_link_bytes_per_gated_element_if_everything_else_free": total / gate_elements,
            "material_budget_if_online_budget_spent_bytes": total - online,
            "declared_independent_sources_per_row": len(sources),
            "declared_source_opening_body_floor_bytes": source_openings,
            "opening_floor_divided_by_online_budget": source_openings / online,
            "minimum_full_width_opening_bytes_per_row": minimum_opening_per_row,
            "maximum_full_width_openings_per_row_if_everything_else_free": online
            // (rows * minimum_opening_per_row),
            "full_width_client_feedback_scenario": {
                "optimistic_body_bytes": feedback,
                "remaining_online_body_budget_bytes": online - feedback,
                "maximum_openings_per_row_after_feedback": max(0, online - feedback)
                // (rows * minimum_opening_per_row),
                "scope": "full-width input embedding shares and final hidden output shares; head/selection client-local; no framing",
            },
            "whole_decoder_executable": False,
            "complete_cost_admitted": False,
            "cold_source_distribution_bytes": None,
            "complete_aggregate_compute": None,
            "limitations": [
                "Per-generated-token averages include prefill; marginal decode needs a separately matched cohort pair",
                "Opening budget omits framing, nonlinear protocols, correlations, state conversions and control",
                "Removing declared openings requires a new enforcing contract, not subtracting them from the report",
                "Baseline amounts are caller-supplied body evidence; source/workload matching remains mandatory",
            ],
        }
