"""Semantic per-layer cost gate for the dense fused two-input lookup reference.

This table-only projection is *optimistic*: attention, normalization, client
share delivery, and full transport can only add cost. No material is issued.
"""

from __future__ import annotations

from math import prod
from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan

from .shared_resources import SharedResourceError, resident_mlp_resource_gate


def resident_fused_gate_resource_gate(
    plan: ModelPlan,
    composition: Pipeline,
    *,
    response_new_tokens: int,
    domain_bits: int,
    maximum_material_bytes_per_party: int,
    maximum_online_all_link_body_bytes: int,
    maximum_online_body_bytes_per_layer: int,
) -> dict[str, Any]:
    """Veto decoder-scale Q3/Q7 lookup before a one-use key is written."""
    if type(domain_bits) is not int or domain_bits not in (4, 8):
        raise SharedResourceError("fused table probe requires the checked Q3 or Q7 width")
    for name, limit in (
        ("material", maximum_material_bytes_per_party),
        ("online", maximum_online_all_link_body_bytes),
        ("per-layer online", maximum_online_body_bytes_per_layer),
    ):
        if type(limit) is not int or not 0 < limit <= 1 << 40:
            raise SharedResourceError(f"{name} budget must be a positive bounded integer")
    if any(slot in composition.components for slot in ("placement", "boundary", "cache")):
        raise SharedResourceError("fused gate projection needs a baseline semantic placement")
    # Reuse fail-closed schedule and gated-product admission. Its Beaver costs
    # are the old comparator, not part of this candidate's body estimate.
    baseline = resident_mlp_resource_gate(
        plan, composition,
        response_new_tokens=response_new_tokens, fixed_scale_bits=8,
        maximum_material_bytes_per_party=maximum_material_bytes_per_party,
        maximum_online_all_link_body_bytes=maximum_online_all_link_body_bytes,
    )
    document = plan.to_dict()
    counts: dict[int, int] = {}
    for phase, factor in (("prefill", 1), ("decode", response_new_tokens - 1)):
        if factor == 0:
            continue
        rows = document[phase]["operations"]
        by_id = {op["id"]: op for op in rows}
        for op in rows:
            if op.get("operator") != "multiply":
                continue
            sources = op.get("inputs")
            if not isinstance(sources, list) or len(sources) != 2:
                continue
            if {by_id.get(source, {}).get("operator") for source in sources} != {"silu", "linear"}:
                continue
            layer = op.get("layer")
            shape = op.get("output_shape")
            if (
                type(layer) is not int or not 0 <= layer < 128
                or not isinstance(shape, list) or not 2 <= len(shape) <= 4
                or any(type(dim) is not int or not 0 < dim <= 1 << 20 for dim in shape)
            ):
                raise SharedResourceError("gated MLP layer or shape is not declared")
            counts[layer] = counts.get(layer, 0) + factor * prod(shape)
    if (
        not counts or sorted(counts) != list(range(len(counts)))
        or sum(counts.values()) != baseline["prefill_elements"] + baseline["decode_elements"]
    ):
        raise SharedResourceError("fused gate layer counts do not match the compiler schedule")
    domain = 1 << domain_bits
    key_body_per_element_per_party = domain * domain * 8 + 2
    # Two parties each send their masked gate and up input shares. Excludes
    # frame headers, state/share delivery, and every other layer operator.
    online_body_per_element = 2 * (1 if domain_bits == 4 else 2)
    layers = [
        {
            "index": layer,
            "gated_elements": elements,
            "minimum_online_all_link_body_bytes": elements * online_body_per_element,
            "minimum_material_body_bytes_per_party": elements * key_body_per_element_per_party,
            "online_within_layer_budget": elements * online_body_per_element
            <= maximum_online_body_bytes_per_layer,
        }
        for layer, elements in sorted(counts.items())
    ]
    online = sum(row["minimum_online_all_link_body_bytes"] for row in layers)
    material = sum(row["minimum_material_body_bytes_per_party"] for row in layers)
    return {
        "schema": "pllm.resident_fused_gate_resource_gate.v1",
        "scope": "one dense one-use fused SiLU(gate)×up table per semantic MLP element",
        "plan_digest": plan.digest,
        "schedule_digest": baseline["schedule_digest"],
        "composition_digest": baseline["composition_digest"],
        "domain_bits": domain_bits,
        "response_new_tokens": response_new_tokens,
        "layers": layers,
        "minimum_material_body_bytes_per_party": material,
        "minimum_online_all_link_body_bytes": online,
        "minimum_dealer_to_both_parties_body_bytes": 2 * material,
        "maximum_material_bytes_per_party": maximum_material_bytes_per_party,
        "maximum_online_all_link_body_bytes": maximum_online_all_link_body_bytes,
        "maximum_online_body_bytes_per_layer": maximum_online_body_bytes_per_layer,
        "material_within_budget": material <= maximum_material_bytes_per_party,
        "online_within_budget": online <= maximum_online_all_link_body_bytes
        and all(row["online_within_layer_budget"] for row in layers),
        "whole_layer_executable": False,
        "decoder_material_issuance_admitted": False,
        "unmeasured": [
            "attention, normalization, linear tensor boundaries, output shares and transport framing",
            "actual distributed dealer transport and runtime CPU",
            "Qwen activation domain and model-quality parity with Q3/Q7 fitted SiLU",
        ],
    }


def resident_projected_fused_gate_resource_gate(
    plan: ModelPlan,
    composition: Pipeline,
    *,
    response_new_tokens: int,
    domain_bits: int,
    source_ring_bits: int,
    maximum_material_bytes_per_party: int,
    maximum_online_all_link_body_bytes: int,
    maximum_online_body_bytes_per_layer: int,
) -> dict[str, Any]:
    """Project online traffic after opening each common masked MLP source once.

    Only equal-width Q3/Q7 sources have an executable *toy* reference. Wider
    source rings are traffic scenarios, NOT a protected Qwen rescale contract.
    """
    if type(source_ring_bits) is not int or source_ring_bits not in (4, 8, 16, 24, 32):
        raise SharedResourceError("projected masked-source ring width is unsupported")
    if source_ring_bits < domain_bits:
        raise SharedResourceError("projected source ring cannot be narrower than lookup domain")
    dense = resident_fused_gate_resource_gate(
        plan, composition,
        response_new_tokens=response_new_tokens,
        domain_bits=domain_bits,
        maximum_material_bytes_per_party=maximum_material_bytes_per_party,
        maximum_online_all_link_body_bytes=maximum_online_all_link_body_bytes,
        maximum_online_body_bytes_per_layer=maximum_online_body_bytes_per_layer,
    )
    source_elements: dict[int, int] = {}
    document = plan.to_dict()
    for phase, factor in (("prefill", 1), ("decode", response_new_tokens - 1)):
        if factor == 0:
            continue
        rows = document[phase]["operations"]
        by_id = {op["id"]: op for op in rows}
        for op in rows:
            if op.get("operator") != "multiply":
                continue
            inputs = op.get("inputs")
            if not isinstance(inputs, list) or len(inputs) != 2:
                continue
            roots = [by_id.get(source) for source in inputs]
            if any(not isinstance(root, dict) for root in roots):
                raise SharedResourceError("gated MLP input is absent from the semantic graph")
            if {root.get("operator") for root in roots if isinstance(root, dict)} != {"silu", "linear"}:
                continue
            nonlinear = next(root for root in roots if isinstance(root, dict) and root["operator"] == "silu")
            up = next(root for root in roots if isinstance(root, dict) and root["operator"] == "linear")
            gate_inputs = nonlinear.get("inputs")
            if not isinstance(gate_inputs, list) or len(gate_inputs) != 1:
                raise SharedResourceError("gated activation lacks its projection")
            gate = by_id.get(gate_inputs[0])
            if (
                not isinstance(gate, dict) or gate.get("operator") != "linear"
                or not isinstance(gate.get("inputs"), list) or len(gate["inputs"]) != 1
                or up.get("inputs") != gate["inputs"]
            ):
                raise SharedResourceError("gate and up do not share one semantic source")
            source = by_id.get(gate["inputs"][0])
            shape = source.get("output_shape") if isinstance(source, dict) else None
            if (
                not isinstance(shape, list) or not 2 <= len(shape) <= 4
                or any(type(dim) is not int or not 0 < dim <= 1 << 20 for dim in shape)
                or shape[:-1] != op["output_shape"][:-1]
            ):
                raise SharedResourceError("projected common source lacks an exact shape")
            layer = op["layer"]
            source_elements[layer] = source_elements.get(layer, 0) + factor * prod(shape)
    if set(source_elements) != {row["index"] for row in dense["layers"]}:
        raise SharedResourceError("projected source counts do not cover all MLP layers")
    layers = [
        {
            "index": row["index"],
            "hidden_elements": source_elements[row["index"]],
            "gated_elements": row["gated_elements"],
            "minimum_online_all_link_body_bytes": 2 * (
                (source_elements[row["index"]] * source_ring_bits + 7) // 8
            ),
            "minimum_material_body_bytes_per_party": (
                row["gated_elements"] * (1 << domain_bits) ** 2 * 8
                + source_elements[row["index"]] * ((source_ring_bits + 7) // 8)
            ),
        }
        for row in dense["layers"]
    ]
    online = sum(row["minimum_online_all_link_body_bytes"] for row in layers)
    material = sum(row["minimum_material_body_bytes_per_party"] for row in layers)
    return {
        "schema": "pllm.resident_projected_fused_gate_resource_gate.v1",
        "scope": "one common-source masked opening for public gate/up projections and dense one-use output tables",
        "plan_digest": dense["plan_digest"],
        "schedule_digest": dense["schedule_digest"],
        "composition_digest": dense["composition_digest"],
        "domain_bits": domain_bits,
        "source_ring_bits": source_ring_bits,
        "source_width_has_executable_toy_reference": source_ring_bits == domain_bits,
        "source_scale_and_range_verified_for_model": False,
        "response_new_tokens": response_new_tokens,
        "layers": layers,
        "minimum_online_all_link_body_bytes": online,
        "minimum_material_body_bytes_per_party": material,
        "minimum_dealer_to_both_parties_body_bytes": 2 * material,
        "maximum_material_bytes_per_party": maximum_material_bytes_per_party,
        "maximum_online_all_link_body_bytes": maximum_online_all_link_body_bytes,
        "maximum_online_body_bytes_per_layer": maximum_online_body_bytes_per_layer,
        "material_within_budget": material <= maximum_material_bytes_per_party,
        "online_within_budget": online <= maximum_online_all_link_body_bytes
        and all(row["minimum_online_all_link_body_bytes"] <= maximum_online_body_bytes_per_layer for row in layers),
        "whole_layer_executable": False,
        "decoder_material_issuance_admitted": False,
        "unmeasured": [
            "attention, normalization, linear tensor boundaries, output shares and transport framing",
            "Qwen-scale weight bounds, exact linear rescale and real-model quality",
            "dealer distribution, source-mask expansion and full-wire traffic",
        ],
    }


def resident_quadratic_gate_resource_gate(
    plan: ModelPlan,
    composition: Pipeline,
    *,
    response_new_tokens: int,
    maximum_material_bytes_per_party: int,
    maximum_online_all_link_body_bytes: int,
    maximum_online_body_bytes_per_layer: int,
) -> dict[str, Any]:
    """Project correlated-source masked Q7 quadratic gate-only bodies.

    Polynomial coefficients give an exact 24-bit *numerator*, not the Q7
    rescaled model output. This cost gate cannot authorize a layer or decoder.
    """
    source = resident_projected_fused_gate_resource_gate(
        plan, composition,
        response_new_tokens=response_new_tokens,
        domain_bits=8,
        source_ring_bits=24,
        maximum_material_bytes_per_party=maximum_material_bytes_per_party,
        maximum_online_all_link_body_bytes=maximum_online_all_link_body_bytes,
        maximum_online_body_bytes_per_layer=maximum_online_body_bytes_per_layer,
    )
    layers = [
        {
            "index": row["index"],
            "hidden_elements": row["hidden_elements"],
            "gated_elements": row["gated_elements"],
            "minimum_online_all_link_body_bytes": row["minimum_online_all_link_body_bytes"],
            "minimum_material_body_bytes_per_party": (
                3 * row["hidden_elements"] + 15 * row["gated_elements"]
            ),
        }
        for row in source["layers"]
    ]
    online = sum(row["minimum_online_all_link_body_bytes"] for row in layers)
    material = sum(row["minimum_material_body_bytes_per_party"] for row in layers)
    return {
        "schema": "pllm.resident_quadratic_gate_resource_gate.v1",
        "scope": "one-use Q7 quadratic SiLU×up numerator from correlated 24-bit masked MLP source",
        "plan_digest": source["plan_digest"],
        "schedule_digest": source["schedule_digest"],
        "composition_digest": source["composition_digest"],
        "response_new_tokens": response_new_tokens,
        "source_ring_bits": 24,
        "polynomial_coefficients_per_element": 5,
        "output_numerator_scale": 65536,
        "layers": layers,
        "minimum_online_all_link_body_bytes": online,
        "minimum_material_body_bytes_per_party": material,
        "minimum_dealer_to_both_parties_body_bytes": 2 * material,
        "maximum_material_bytes_per_party": maximum_material_bytes_per_party,
        "maximum_online_all_link_body_bytes": maximum_online_all_link_body_bytes,
        "maximum_online_body_bytes_per_layer": maximum_online_body_bytes_per_layer,
        "gate_only_material_within_budget": material <= maximum_material_bytes_per_party,
        "gate_only_online_within_budget": online <= maximum_online_all_link_body_bytes
        and all(row["minimum_online_all_link_body_bytes"] <= maximum_online_body_bytes_per_layer for row in layers),
        "source_scale_and_range_verified_for_model": False,
        "protected_numerator_to_model_scale": False,
        "whole_layer_executable": False,
        "decoder_material_issuance_admitted": False,
        "unmeasured": [
            "exact protected truncation and nonlinear stage scale transitions",
            "attention, normalization, linear tensor boundaries, output shares and transport framing",
            "Qwen weights, activation range and model-quality parity with bounded Q7 quadratic SiLU",
            "distributed dealer transfer and full-wire traffic",
        ],
    }


def resident_quadratic_layer_resource_gate(
    plan: ModelPlan,
    composition: Pipeline,
    *,
    response_new_tokens: int,
    maximum_material_bytes_per_party: int,
    maximum_online_all_link_body_bytes: int,
    maximum_online_body_bytes_per_layer: int,
) -> dict[str, Any]:
    """Charge a distinct attention-query source to the polynomial MLP gate.

    Only this *specific* two-source masked-opening construction is vetoed;
    other protected attention protocols need their own independent bounds.
    The estimate omits the actual attention arithmetic and all rescaling.
    """
    gate_report = resident_quadratic_gate_resource_gate(
        plan, composition,
        response_new_tokens=response_new_tokens,
        maximum_material_bytes_per_party=maximum_material_bytes_per_party,
        maximum_online_all_link_body_bytes=maximum_online_all_link_body_bytes,
        maximum_online_body_bytes_per_layer=maximum_online_body_bytes_per_layer,
    )
    extra: dict[int, int] = {}
    for phase, factor in (("prefill", 1), ("decode", response_new_tokens - 1)):
        if factor == 0:
            continue
        rows = plan.to_dict()[phase]["operations"]
        by_id = {op["id"]: op for op in rows}
        mlp_sources: dict[int, str] = {}
        for op in rows:
            if op.get("operator") != "multiply":
                continue
            sources = [by_id.get(source) for source in op.get("inputs", ())]
            if {root.get("operator") for root in sources if isinstance(root, dict)} != {"silu", "linear"}:
                continue
            nonlinear = next(root for root in sources if isinstance(root, dict) and root["operator"] == "silu")
            gate = by_id[nonlinear["inputs"][0]]
            mlp_sources[op["layer"]] = gate["inputs"][0]
        for op in rows:
            if op.get("operator") != "attention_scores":
                continue
            layer = op.get("layer")
            inputs = op.get("inputs")
            if type(layer) is not int or layer not in mlp_sources or not isinstance(inputs, list) or not inputs:
                raise SharedResourceError("attention query lacks a matching MLP layer")
            # Trace the query input semantically through rotary/reshape to its
            # single nearest linear producer. Do not inspect IDs/weight paths.
            frontier = [inputs[0]]
            visited: set[str] = set()
            query_linears: list[dict[str, Any]] = []
            while frontier:
                current = frontier.pop()
                if current in visited:
                    continue
                visited.add(current)
                if len(visited) > 64:
                    raise SharedResourceError("attention source lineage exceeds bound")
                ancestor = by_id.get(current)
                if not isinstance(ancestor, dict):
                    continue
                if ancestor.get("operator") == "linear":
                    query_linears.append(ancestor)
                else:
                    frontier.extend(ancestor.get("inputs", ()))
            if len(query_linears) != 1 or query_linears[0].get("layer") != layer:
                raise SharedResourceError("attention query lacks one semantic linear producer")
            query_inputs = query_linears[0].get("inputs")
            if not isinstance(query_inputs, list) or len(query_inputs) != 1:
                raise SharedResourceError("attention query source is not unique")
            query_source = query_inputs[0]
            if query_source == mlp_sources[layer]:
                raise SharedResourceError("attention and MLP sources are not independent")
            query_op = by_id.get(query_source)
            shape = query_op.get("output_shape") if isinstance(query_op, dict) else None
            if (
                not isinstance(shape, list) or not 2 <= len(shape) <= 4
                or any(type(dim) is not int or not 0 < dim <= 1 << 20 for dim in shape)
            ):
                raise SharedResourceError("attention query input lacks a bounded source shape")
            extra[layer] = extra.get(layer, 0) + factor * prod(shape)
    if set(extra) != {row["index"] for row in gate_report["layers"]}:
        raise SharedResourceError("attention source lower bound misses a semantic layer")
    layers = [
        {
            **row,
            "independent_attention_source_elements": extra[row["index"]],
            "minimum_two_sources_online_all_link_body_bytes": (
                row["minimum_online_all_link_body_bytes"] + 6 * extra[row["index"]]
            ),
            "minimum_two_sources_material_body_bytes_per_party": (
                row["minimum_material_body_bytes_per_party"] + 3 * extra[row["index"]]
            ),
        }
        for row in gate_report["layers"]
    ]
    online = sum(row["minimum_two_sources_online_all_link_body_bytes"] for row in layers)
    material = sum(row["minimum_two_sources_material_body_bytes_per_party"] for row in layers)
    return {
        "schema": "pllm.resident_quadratic_layer_resource_gate.v1",
        "scope": "at least two independent semantic hidden-source openings per layer, before attention arithmetic",
        "plan_digest": gate_report["plan_digest"],
        "schedule_digest": gate_report["schedule_digest"],
        "composition_digest": gate_report["composition_digest"],
        "layers": layers,
        "response_new_tokens": response_new_tokens,
        "source_ring_bits": 24,
        "minimum_online_all_link_body_bytes": online,
        "minimum_material_body_bytes_per_party": material,
        "minimum_dealer_to_both_parties_body_bytes": 2 * material,
        "maximum_online_all_link_body_bytes": maximum_online_all_link_body_bytes,
        "maximum_online_body_bytes_per_layer": maximum_online_body_bytes_per_layer,
        "maximum_material_bytes_per_party": maximum_material_bytes_per_party,
        "material_within_budget": material <= maximum_material_bytes_per_party,
        "online_within_budget": online <= maximum_online_all_link_body_bytes
        and all(row["minimum_two_sources_online_all_link_body_bytes"] <= maximum_online_body_bytes_per_layer for row in layers),
        "whole_layer_executable": False,
        "decoder_material_issuance_admitted": False,
        "unmeasured": gate_report["unmeasured"] + [
            "attention projections, score products, protected softmax, KV, exact rescale and normalization",
        ],
    }
