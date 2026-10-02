"""Compiler-bound directed-link lower bounds for non-executable decoder regions.

An unknown peer protocol, material generator, numeric contract or client CPU
cost is never counted as zero. This report can veto an interface on its known
floor; it cannot admit a protected layer, topology or whole response.
"""

from __future__ import annotations

from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan

from .semantic_stages import scheduled_stage_specs
from .shared_gate_resources import resident_quadratic_layer_resource_gate
from .shared_resources import SharedResourceError

_ROLES = frozenset({"qkv_projection", "attention_output", "mlp_gate_up", "mlp_down"})
_BODY_LIMIT = 1 << 40


def _bytes(elements: int, bits: int) -> int:
    return (elements * bits + 7) // 8


def _edge(source: str, destination: str, tensor: str, elements: int, bits: int) -> dict[str, Any]:
    return {
        "phase": "online",
        "source": source,
        "destination": destination,
        "tensor": tensor,
        "elements": elements,
        "assumed_bits_per_element": bits,
        "optimistic_body_bytes": _bytes(elements, bits),
    }


def _unknown(phase: str, link: str, requirement: str) -> dict[str, str | None]:
    return {"phase": phase, "link": link, "requirement": requirement, "body_bytes": None}


def _placement(
    edges: list[dict[str, Any]],
    unknown: list[dict[str, str | None]],
    *,
    online_budget: int,
    all_link_budget: int,
) -> dict[str, Any]:
    online_floor = sum(edge["optimistic_body_bytes"] for edge in edges)
    return {
        "directed_known_online_links": edges,
        "known_online_body_floor_bytes": online_floor,
        "known_all_link_body_floor_bytes": online_floor,
        "unknown_required_links_and_work": unknown,
        "known_floor_exceeds_online_budget": online_floor > online_budget,
        "known_floor_exceeds_all_link_budget": online_floor > all_link_budget,
        "all_link_cost_known": False,
        "byte_admitted": False,
        "numeric_contract_validated": False,
        "executable": False,
    }


def compiler_region_contract_cost(
    plan: ModelPlan,
    composition: Pipeline,
    *,
    response_new_tokens: int,
    maximum_online_all_link_body_bytes: int,
    maximum_total_all_link_body_bytes: int,
    mlp_input_share_bits: int = 24,
    mlp_output_share_bits: int = 32,
    resident_source_bits: tuple[int, ...] = (24, 12),
    token_input_share_bits: int = 24,
    token_output_share_bits: int = 32,
) -> dict[str, Any]:
    """Veto candidate placements from compiler roles and independent sources.

    Two-worker MLP boundaries send a separate share to/from *each* worker.
    Resident peer openings count attention-query and MLP sources separately;
    all bits and tensor-packing choices are assumptions, not numeric support.
    """
    if not isinstance(plan, ModelPlan) or not isinstance(composition, Pipeline):
        raise TypeError("region cost needs a semantic ModelPlan and Pipeline")
    if type(response_new_tokens) is not int or not 1 <= response_new_tokens <= 256:
        raise SharedResourceError("response length exceeds bounded region contract")
    for name, value in (
        ("online budget", maximum_online_all_link_body_bytes),
        ("all-link budget", maximum_total_all_link_body_bytes),
    ):
        if type(value) is not int or not 0 < value <= _BODY_LIMIT:
            raise SharedResourceError(f"{name} must be a bounded positive integer")
    if maximum_online_all_link_body_bytes > maximum_total_all_link_body_bytes:
        raise SharedResourceError("online budget cannot exceed all-link budget")
    for name, value, widths in (
        ("MLP input", mlp_input_share_bits, (16, 24, 32)),
        ("MLP output", mlp_output_share_bits, (16, 24, 32)),
        ("token input", token_input_share_bits, (16, 24, 32)),
        ("token output", token_output_share_bits, (16, 24, 32)),
    ):
        if type(value) is not int or value not in widths:
            raise SharedResourceError(f"{name} width is not an admitted cost scenario")
    if (
        type(resident_source_bits) is not tuple
        or not resident_source_bits
        or len(set(resident_source_bits)) != len(resident_source_bits)
        or any(type(width) is not int or width not in (12, 24) for width in resident_source_bits)
    ):
        raise SharedResourceError("resident sources require distinct bounded cost scenarios")

    # The pre-existing gate traces query inputs through rotary/reshape to the
    # attention linear producer and follows SiLU×up back to its *different*
    # post-attention source. It checks full schedule coverage and the W8A8
    # composition before issuing any cost report. Its quadratic bytes are only
    # a comparator for that existing method, not a floor on new protocols.
    lineage = resident_quadratic_layer_resource_gate(
        plan,
        composition,
        response_new_tokens=response_new_tokens,
        maximum_material_bytes_per_party=256 << 20,
        maximum_online_all_link_body_bytes=maximum_online_all_link_body_bytes,
        maximum_online_body_bytes_per_layer=maximum_online_all_link_body_bytes,
    )
    if lineage["plan_digest"] != plan.digest or len(lineage["layers"]) > 128:
        raise SharedResourceError("layer provenance differs from the compiler plan")
    document = plan.to_dict()
    prefill = document["prefill"].get("query_sequence")
    if type(prefill) is not int or not 1 <= prefill <= 4096:
        raise SharedResourceError("bounded prompt length is not declared")
    executed_rows = prefill + response_new_tokens - 1
    stages = scheduled_stage_specs(plan, composition)
    body: dict[int, dict[str, Any]] = {}
    boundaries: dict[str, Any] = {}
    for stage in stages:
        if stage.role in {"token_lookup", "lm_head"}:
            if stage.role in boundaries:
                raise SharedResourceError("duplicate client token boundary")
            boundaries[stage.role] = stage
            continue
        layer = stage.layer_index
        if stage.role not in _ROLES or type(layer) is not int or not 0 <= layer < 128:
            raise SharedResourceError("region lacks a supported semantic projection role")
        if stage.in_features <= 0 or stage.out_features <= 0:
            raise SharedResourceError("region projection has an invalid weight shape")
        row = body.setdefault(layer, {})
        if stage.role in row:
            raise SharedResourceError("duplicate semantic projection in layer")
        row[stage.role] = stage
    if set(boundaries) != {"token_lookup", "lm_head"}:
        raise SharedResourceError("client token boundary is incomplete")
    if (
        sorted(body) != list(range(len(body)))
        or set(body) != {row["index"] for row in lineage["layers"]}
        or any(set(row) != _ROLES for row in body.values())
    ):
        raise SharedResourceError("source and projection roles do not cover each layer")
    hidden = attention = 0
    client_projection_macs = remote_mlp_macs = 0
    client_attention_snapshot = remote_mlp_snapshot = 0
    layer_details: list[dict[str, int]] = []
    for source in lineage["layers"]:
        layer = source["index"]
        row = body[layer]
        qkv, attn_out = row["qkv_projection"], row["attention_output"]
        gate, down = row["mlp_gate_up"], row["mlp_down"]
        mlp_elements = source["hidden_elements"]
        query_elements = source["independent_attention_source_elements"]
        if (
            gate.in_features * executed_rows != mlp_elements
            or down.out_features * executed_rows != mlp_elements
            or qkv.in_features * executed_rows != query_elements
            or attn_out.out_features * executed_rows != query_elements
        ):
            raise SharedResourceError("semantic sources differ from projection dimensions")
        hidden += mlp_elements
        attention += query_elements
        local_macs = executed_rows * (
            qkv.in_features * qkv.out_features + attn_out.in_features * attn_out.out_features
        )
        remote_macs = executed_rows * (
            gate.in_features * gate.out_features + down.in_features * down.out_features
        )
        client_projection_macs += local_macs
        remote_mlp_macs += remote_macs
        client_attention_snapshot += (
            qkv.in_features * qkv.out_features
            + 4 * qkv.out_features
            + attn_out.in_features * attn_out.out_features
            + 4 * attn_out.out_features
        )
        remote_mlp_snapshot += (
            gate.in_features * gate.out_features
            + 4 * gate.out_features
            + down.in_features * down.out_features
            + 4 * down.out_features
        )
        layer_details.append(
            {
                "semantic_layer": layer,
                "executed_rows": executed_rows,
                "attention_query_source_elements": query_elements,
                "post_attention_mlp_source_elements": mlp_elements,
                "client_attention_projection_integer_macs": local_macs,
                "remote_mlp_projection_integer_macs_per_worker": remote_macs,
            }
        )
    if hidden <= 0 or attention <= 0 or client_projection_macs + remote_mlp_macs <= 0:
        raise SharedResourceError("region has no bounded semantic work")
    width = boundaries["token_lookup"].out_features
    head = boundaries["lm_head"]
    if head.in_features != width:
        raise SharedResourceError("client token and output-head widths disagree")
    head_macs = response_new_tokens * head.in_features * head.out_features
    base = {
        "schema": "pllm.compiler_region_contract_cost.v1",
        "scope": "non-executable optimistic body floors and explicit unpriced obligations",
        "plan_digest": plan.digest,
        "schedule_digest": lineage["schedule_digest"],
        "composition_digest": lineage["composition_digest"],
        "response_new_tokens": response_new_tokens,
        "input_tokens": prefill,
        "executed_rows": executed_rows,
        "layer_count": len(body),
        "layer_provenance": layer_details,
        "maximum_online_all_link_body_bytes": maximum_online_all_link_body_bytes,
        "maximum_total_all_link_body_bytes": maximum_total_all_link_body_bytes,
        "client_output_head_integer_macs": head_macs,
        "protected_composition_selected": False,
        "numeric_scale_and_range_validated": False,
        "whole_decoder_executable": False,
    }
    mlp_edges = [
        _edge("client", worker, "post_attention_mlp_input_share", hidden, mlp_input_share_bits)
        for worker in ("worker_a", "worker_b")
    ] + [
        _edge(worker, "client", "complete_mlp_output_share", hidden, mlp_output_share_bits)
        for worker in ("worker_a", "worker_b")
    ]
    mlp_unknown = [
        _unknown("online", "client↔workers", "secret per-row activation scales and numeric metadata"),
        _unknown("online", "worker_a↔worker_b",
                 "dynamic max-absolute quantization, reciprocal, ties-to-even and range rejection"),
        _unknown(
            "online", "worker_a↔worker_b", "protected SiLU×up, exact rescaling and share refresh"
        ),
        _unknown("offline", "dealer→worker_a", "one-use joint nonlinear/scale material"),
        _unknown(
            "offline", "dealer→worker_b", "independent one-use joint nonlinear/scale material"
        ),
        _unknown(
            "cold", "source→client", "attention projection weight distribution beyond i8/scales"
        ),
        _unknown("cold", "source→workers", "two provider MLP checkpoint snapshots and disk"),
        _unknown("online", "client↔workers", "framing, authentication and complete wire"),
        _unknown("local", "client", "attention/norm/KV CPU, output head, peak memory and latency"),
        _unknown("control", "all roles", "bounded admission, replay, cancellation and failure burns"),
    ]
    mlp = _placement(
        mlp_edges,
        mlp_unknown,
        online_budget=maximum_online_all_link_body_bytes,
        all_link_budget=maximum_total_all_link_body_bytes,
    )
    mlp.update(
        {
            "placement": "client_attention_remote_two_worker_mlp",
            "assumed_input_share_bits": mlp_input_share_bits,
            "assumed_output_share_bits": mlp_output_share_bits,
            "additional_client_attention_i8_weight_and_f32_scale_bytes_once_per_model": client_attention_snapshot,
            "provider_mlp_i8_weight_and_f32_scale_bytes_per_worker_once_per_model": remote_mlp_snapshot,
            "client_attention_projection_integer_macs": client_projection_macs,
            "remote_mlp_projection_integer_macs_per_worker": remote_mlp_macs,
            "client_output_head_integer_macs": head_macs,
            "remote_fraction_of_tracked_body_projection_macs": remote_mlp_macs
            / (client_projection_macs + remote_mlp_macs),
            "client_attention_full_compute_measured": False,
            "current_quadratic_numerator_dealer_bodies_both_parties_comparator_bytes": lineage[
                "minimum_dealer_to_both_parties_body_bytes"
            ],
        }
    )
    resident: dict[str, Any] = {}
    for bits in resident_source_bits:
        edges = (
            [
                _edge(
                    worker, peer, "attention_query_and_mlp_source_opening", attention + hidden, bits
                )
                for worker, peer in (("worker_a", "worker_b"), ("worker_b", "worker_a"))
            ]
            + [
                _edge(
                    "client",
                    worker,
                    "token_embedding_share",
                    executed_rows * width,
                    token_input_share_bits,
                )
                for worker in ("worker_a", "worker_b")
            ]
            + [
                _edge(
                    worker,
                    "client",
                    "final_hidden_share_for_client_token_choice",
                    response_new_tokens * width,
                    token_output_share_bits,
                )
                for worker in ("worker_a", "worker_b")
            ]
        )
        unknown = [
            _unknown("online", "worker_a↔worker_b",
                     "RMSNorm inverse square root with epsilon, dynamic quantization and private scales"),
            _unknown(
                "online",
                "worker_a↔worker_b",
                "protected attention scores/softmax, norms, MLP and exact truncation",
            ),
            _unknown(
                "offline", "dealer→worker_a", "full-layer one-use attention and MLP correlations"
            ),
            _unknown("offline", "dealer→worker_b", "independent full-layer one-use correlations"),
            _unknown("online", "workers", "private KV/cache share maintenance and scale changes"),
            _unknown(
                "cold", "source→workers", "full checkpoint distribution to both workers and disk"
            ),
            _unknown("online", "client↔workers", "framing, authentication and complete wire"),
            _unknown("local", "client", "output-head CPU and token feedback latency"),
            _unknown("control", "all roles", "bounded admission, replay, cancellation and failure burns"),
        ]
        scenario = _placement(
            edges,
            unknown,
            online_budget=maximum_online_all_link_body_bytes,
            all_link_budget=maximum_total_all_link_body_bytes,
        )
        scenario.update(
            {
                "placement": "two_worker_resident_complete_layers",
                "assumed_source_opening_bits": bits,
                "assumed_client_token_input_share_bits": token_input_share_bits,
                "assumed_client_final_hidden_share_bits": token_output_share_bits,
                "independent_source_elements_per_worker": attention + hidden,
                "remote_body_projection_integer_macs_per_worker": client_projection_macs
                + remote_mlp_macs,
                "client_output_head_integer_macs": head_macs,
                "current_quadratic_numerator_dealer_bodies_both_parties_comparator_bytes": lineage[
                    "minimum_dealer_to_both_parties_body_bytes"
                ],
            }
        )
        if bits == 24:
            two_source_opening = sum(edge["optimistic_body_bytes"] for edge in edges[:2])
            if two_source_opening != lineage["minimum_online_all_link_body_bytes"]:
                raise SharedResourceError("24-bit peer opening differs from semantic lineage")
        resident[str(bits)] = scenario
    return {
        **base,
        "client_attention_remote_mlp": mlp,
        "two_worker_resident": resident,
    }


def compiler_region_reduction_gates(
    plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    baseline_online_body_bytes: int, baseline_all_link_body_bytes: int,
) -> dict[str, Any]:
    """Screen complete obligations at 25%, 50% and tenfold covered-body targets.

    Unknown steps prevent passage, even under a generous byte budget. The existing
    quadratic key comparator is charged only to explicitly named reference-derived
    candidates; it is not a floor on every possible future correlation generator.
    Caller must bind a measured comparator's source, schedule and numeric cohort.
    """
    report = compiler_region_contract_cost(plan, composition,
        response_new_tokens=response_new_tokens,
        maximum_online_all_link_body_bytes=baseline_online_body_bytes,
        maximum_total_all_link_body_bytes=baseline_all_link_body_bytes)
    cases = [
        ("client_attention_remote_mlp_unknown_correlations", report["client_attention_remote_mlp"], 0),
        ("client_attention_remote_mlp_current_quadratic_keys", report["client_attention_remote_mlp"],
         report["client_attention_remote_mlp"]["current_quadratic_numerator_dealer_bodies_both_parties_comparator_bytes"]),
        ("resident_24_unknown_correlations", report["two_worker_resident"]["24"], 0),
        ("resident_24_current_quadratic_keys", report["two_worker_resident"]["24"],
         report["two_worker_resident"]["24"]["current_quadratic_numerator_dealer_bodies_both_parties_comparator_bytes"]),
        ("resident_12_unvalidated_numeric_and_correlations", report["two_worker_resident"]["12"], 0),
    ]
    decisions = []
    for name, placement, material in cases:
        online = placement["known_online_body_floor_bytes"]
        total = placement["known_all_link_body_floor_bytes"] + material
        targets = {}
        for label, numerator, denominator in (("25_percent", 3, 4), ("50_percent", 1, 2), ("10x", 1, 10)):
            online_limit = baseline_online_body_bytes * numerator // denominator
            all_limit = baseline_all_link_body_bytes * numerator // denominator
            veto = online > online_limit or total > all_limit
            targets[label] = {
                "online_budget_bytes": online_limit, "all_link_budget_bytes": all_limit,
                "known_online_floor_bytes": online, "known_all_link_floor_bytes": total,
                "remaining_online_budget_bytes": online_limit - online,
                "remaining_all_link_budget_bytes": all_limit - total,
                "decision": "veto_known_body_floor" if veto else "inconclusive_unpriced_requirements",
                "numeric_fidelity_established": False, "complete_cost_known": False,
                "byte_admitted": False, "executable": False,
            }
        decisions.append({
            "candidate": name, "known_quadratic_reference_key_body_bytes": material,
            "required_unpriced_obligations": placement["unknown_required_links_and_work"],
            "targets": targets,
        })
    report["reduction_gate_scope"] = "fresh response; covered bodies; all missing work remains unknown"
    report["baseline_online_body_bytes"] = baseline_online_body_bytes
    report["baseline_all_link_body_bytes"] = baseline_all_link_body_bytes
    report["reduction_gates"] = decisions
    return report


__all__ = ["compiler_region_contract_cost", "compiler_region_reduction_gates"]
