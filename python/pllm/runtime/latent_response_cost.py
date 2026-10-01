"""Fail-closed compiler-bound costs for non-streaming private token feedback.

These are projections from a semantic schedule and bounded reference primitives,
not measurements of a server-resident decoder or an executable topology.
"""

from __future__ import annotations

import math
from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan

from .shared_gate_resources import (
    resident_quadratic_gate_resource_gate,
    resident_quadratic_layer_resource_gate,
)
from .shared_resources import SharedResourceError


def _feedback_contract(plan: ModelPlan, composition: Pipeline) -> int:
    graph = plan.to_dict()
    if graph.get("token_feedback") is not True:
        raise SharedResourceError("latent-loop probe requires semantic token feedback")
    try:
        schedule = plan.runtime_schedule(composition).to_dict()
    except (TypeError, ValueError) as exc:
        raise SharedResourceError("latent-loop probe requires an admitted semantic schedule") from exc
    vocabulary: int | None = None
    for phase in ("prefill", "decode"):
        operators = graph[phase]["operations"]
        required = {
            "output_head": [1, -1],
            "greedy_token_selection": [1],
            "token_feedback": [1, 1],
        }
        for operator, expected in required.items():
            found = [row for row in operators if row.get("operator") == operator]
            if len(found) != 1:
                raise SharedResourceError(f"{phase} has no unique {operator}")
            shape = found[0].get("output_shape")
            if not isinstance(shape, list) or len(shape) != len(expected):
                raise SharedResourceError(f"{phase} has an invalid {operator} shape")
            if expected[-1] == -1:
                if shape[0] != 1 or type(shape[1]) is not int or not 2 <= shape[1] <= 1 << 20:
                    raise SharedResourceError(f"{phase} has an invalid output-head vocabulary")
                if vocabulary is not None and vocabulary != shape[1]:
                    raise SharedResourceError("prefill/decode output-head vocabularies disagree")
                vocabulary = shape[1]
            elif shape != expected:
                raise SharedResourceError(f"{phase} has an invalid {operator} shape")
            if operator in ("greedy_token_selection", "token_feedback"):
                steps = [
                    step for step in schedule[phase]["steps"]
                    if found[0]["id"] in step.get("operation_ids", [])
                ]
                if len(steps) != 1 or steps[0].get("executor") != "client_local":
                    raise SharedResourceError("feedback placement no longer matches this probe")
    if vocabulary is None:
        raise SharedResourceError("the output-head vocabulary is not declared")
    return vocabulary


def latent_response_cost_probe(
    plan: ModelPlan,
    composition: Pipeline,
    *,
    response_new_tokens: int,
    maximum_online_all_link_body_bytes: int,
    maximum_total_all_link_body_bytes: int,
    maximum_material_bytes_per_party: int,
) -> dict[str, Any]:
    """Compare optimistic gates with the existing selection reference.

    That reference converts 17-bit 65537-field logits using at least one
    secret multiplication per logit/bit. Opening two i64 operands between
    parties costs at least 32 body bytes per multiplication. This is specific
    to this reference, not a lower bound on all secure argmax protocols.
    """
    for name, limit in (
        ("online", maximum_online_all_link_body_bytes),
        ("all-link", maximum_total_all_link_body_bytes),
        ("material", maximum_material_bytes_per_party),
    ):
        if type(limit) is not int or not 0 < limit <= 1 << 40:
            raise SharedResourceError(f"{name} budget must be a bounded positive integer")
    vocab = _feedback_contract(plan, composition)
    common = dict(
        response_new_tokens=response_new_tokens,
        maximum_material_bytes_per_party=maximum_material_bytes_per_party,
        maximum_online_all_link_body_bytes=maximum_online_all_link_body_bytes,
        maximum_online_body_bytes_per_layer=maximum_online_all_link_body_bytes,
    )
    gate = resident_quadratic_gate_resource_gate(plan, composition, **common)
    resident = resident_quadratic_layer_resource_gate(plan, composition, **common)
    if gate["plan_digest"] != resident["plan_digest"] or gate["composition_digest"] != resident["composition_digest"]:
        raise SharedResourceError("quadratic gate and resident layer plans disagree")
    hidden = sum(row["hidden_elements"] for row in gate["layers"])
    if hidden <= 0 or any(
        row["hidden_elements"] != other["hidden_elements"]
        for row, other in zip(gate["layers"], resident["layers"], strict=True)
    ):
        raise SharedResourceError("latent MLP input and resident layer shapes disagree")
    # Fixed *hypothetical* u24/u32 narrow boundary. Real checkpoint ring
    # admission and protected fixed-scale rescaling remain unimplemented.
    narrow_online = 7 * hidden
    feedback_steps = response_new_tokens - 1
    per_feedback_reference_lower = vocab * 17 * 32
    # A bounded lookup oracle currently requires an issuer who knows the
    # future token: its dense keys cannot privately feed an online loop.
    dense_lookup_body_per_feedback = 2 * vocab * 8
    key_both = gate["minimum_dealer_to_both_parties_body_bytes"]
    return {
        "schema": "pllm.latent_response_cost_probe.v1",
        "scope": "non-executable compiler-bound response-size comparisons; all links, not client-facing only",
        "plan_digest": plan.digest,
        "composition_digest": gate["composition_digest"],
        "schedule_digest": gate["schedule_digest"],
        "response_new_tokens": response_new_tokens,
        "vocabulary_size": vocab,
        "protected_feedback_steps_before_final_client_selection": feedback_steps,
        "remote_mlp_cut": {
            "assumed_masked_input_ring_bits": 24,
            "assumed_masked_output_ring_bits": 32,
            "minimum_online_input_output_body_bytes": narrow_online,
            "quadratic_gate_one_use_dealer_body_bytes_both_parties": key_both,
            "optimistic_all_link_body_bytes": narrow_online + key_both,
            "within_online_budget": narrow_online <= maximum_online_all_link_body_bytes,
            "within_all_link_budget": narrow_online + key_both <= maximum_total_all_link_body_bytes,
            "material_per_party_within_budget": gate["gate_only_material_within_budget"],
            "protected_down_projection_and_rescale_executable": False,
        },
        "resident_two_source": {
            "minimum_online_opening_body_bytes": resident["minimum_online_all_link_body_bytes"],
            "minimum_dealer_body_bytes_both_parties": resident["minimum_dealer_to_both_parties_body_bytes"],
            "within_online_budget": resident["online_within_budget"],
            "material_per_party_within_budget": resident["material_within_budget"],
            "optimistic_online_with_existing_feedback_reference_bytes": (
                resident["minimum_online_all_link_body_bytes"]
                + feedback_steps * per_feedback_reference_lower
            ),
        },
        "feedback_reference": {
            "field_modulus": 65_537,
            "optimistic_peer_only_opened_body_bytes_per_required_selection": per_feedback_reference_lower,
            "optimistic_peer_only_opened_body_bytes_for_response": feedback_steps * per_feedback_reference_lower,
            "dense_issued_lookup_body_bytes_per_feedback": dense_lookup_body_per_feedback,
            "actual_argmax_reveals_index_to_client": True,
            "lookup_requires_future_token_known_to_issuer": True,
            "point_fss_comparison_maximum_domain": 1024,
            "share_indexed_embedding_lookup_available": False,
            "client_local_feedback_in_compiled_schedule": True,
            "decimal_log10_exhaustive_branches_without_private_feedback": feedback_steps * math.log10(vocab),
        },
        "maximum_online_all_link_body_bytes": maximum_online_all_link_body_bytes,
        "maximum_total_all_link_body_bytes": maximum_total_all_link_body_bytes,
        "maximum_material_bytes_per_party": maximum_material_bytes_per_party,
        "whole_response_executable": False,
        "whole_response_byte_admission": False,
        "unmeasured": [
            "full HTTP/TLS/control wire bytes and remote-party checkpoint distribution",
            "private token selection and share-indexed embedding lookup for this vocabulary",
            "exact protected rescaling, attention, KV and output-head execution",
            "Qwen numeric range and model-quality parity for the quadratic Q7 method",
        ],
    }
