"""Non-executable, compiler-bound depth gate for token-boundary CKKS.

Counts only ciphertext-ciphertext products forced by Q/K attention, attention
values, a non-affine SiLU approximation and gated MLP multiplication. Norms,
softmax, public matrix operations, scale management and private KV are not
free: their numeric, depth and resource costs are deliberately unknown.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan

from .semantic_stages import scheduled_stage_specs


class EncryptedLayerGateError(ValueError):
    """Incomplete semantic coverage cannot authorize an HE feasibility claim."""


_LAYER_KINDS = frozenset(
    {
        "rms_norm",
        "linear",
        "reshape",
        "rotary_embedding",
        "kv_cache_append",
        "cache_suffix",
        "attention_scores",
        "attention_scale",
        "causal_mask",
        "softmax",
        "attention_values",
        "residual_add",
        "silu",
        "multiply",
    }
)
_REQUIRED_COUNTS = {
    "rms_norm": 2,
    "attention_scores": 1,
    "softmax": 1,
    "attention_values": 1,
    "silu": 1,
    "multiply": 1,
    "residual_add": 2,
    "kv_cache_append": 2,
}
_CLIENT_BOUNDARY_KINDS = frozenset(
    {
        "token_lookup",
        "rms_norm",
        "last_token",
        "output_head",
        "greedy_token_selection",
        "token_feedback",
    }
)
_PRODUCT_KINDS = frozenset({"attention_scores", "attention_values", "silu", "multiply"})
_BODY_ROLES = frozenset({"qkv_projection", "attention_output", "mlp_gate_up", "mlp_down"})


def _reaches(operations: dict[str, dict[str, Any]], source: str, destination: str) -> bool:
    pending = [destination]
    seen: set[str] = set()
    while pending:
        node = pending.pop()
        if node == source:
            return True
        if node in seen:
            continue
        seen.add(node)
        pending.extend(
            item for item in operations.get(node, {}).get("inputs", ()) if item in operations
        )
    return False


def _positive_shape(operation: dict[str, Any]) -> tuple[int, ...]:
    shape = operation.get("output_shape")
    if (
        type(shape) is not list
        or not shape
        or len(shape) > 5
        or any(type(size) is not int or not 0 < size <= 1 << 20 for size in shape)
    ):
        raise EncryptedLayerGateError("encrypted operation lacks a bounded tensor shape")
    return tuple(shape)


def _phase_depth(graph: dict[str, Any], layer_count: int) -> dict[str, Any]:
    rows = graph.get("operations")
    if not isinstance(rows, list) or not rows:
        raise EncryptedLayerGateError("encrypted phase lacks ordered semantic operators")
    operations = {row["id"]: row for row in rows}
    if len(operations) != len(rows):
        raise EncryptedLayerGateError("encrypted phase duplicates operation identities")
    depths: dict[str, int] = {}
    layers: dict[int, list[dict[str, Any]]] = {index: [] for index in range(layer_count)}
    for operation in rows:
        kind = operation["operator"]
        identity = operation["id"]
        sources = operation.get("inputs")
        if not isinstance(sources, list):
            raise EncryptedLayerGateError("encrypted operator lacks declared input dependencies")
        for source in sources:
            if source in operations and source not in depths:
                raise EncryptedLayerGateError("encrypted schedule is not in dependency order")
        layer = operation.get("layer")
        if layer is not None:
            if type(layer) is not int or layer not in layers or kind not in _LAYER_KINDS:
                raise EncryptedLayerGateError(
                    "encrypted layer contains an unpriced semantic operator"
                )
            layers[layer].append(operation)
        depths[identity] = max((depths.get(source, 0) for source in sources), default=0) + int(
            kind in _PRODUCT_KINDS and layer is not None
        )

    per_layer: list[dict[str, Any]] = []
    gate_ids: dict[int, str] = {}
    for layer, members in layers.items():
        counts = Counter(row["operator"] for row in members)
        if any(counts.get(kind) != count for kind, count in _REQUIRED_COUNTS.items()):
            raise EncryptedLayerGateError("encrypted layer lacks complete attention/MLP semantics")
        if counts["linear"] != 7 or counts["rotary_embedding"] != 2:
            raise EncryptedLayerGateError("encrypted layer lacks full projection/rotary coverage")
        selection = {
            kind: [row for row in members if row["operator"] == kind]
            for kind in ("attention_scores", "softmax", "attention_values", "silu", "multiply")
        }
        score = selection["attention_scores"][0]["id"]
        softmax = selection["softmax"][0]["id"]
        values = selection["attention_values"][0]["id"]
        silu = selection["silu"][0]["id"]
        gate = selection["multiply"][0]["id"]
        if not all(
            _reaches(operations, source, destination)
            for source, destination in (
                (score, softmax),
                (softmax, values),
                (values, silu),
                (silu, gate),
            )
        ):
            raise EncryptedLayerGateError("attention-to-MLP dependency chain is not established")
        if layer and not any(
            _reaches(operations, prior["id"], score)
            for prior in layers[layer - 1]
            if prior["operator"] == "residual_add"
            and _reaches(operations, gate_ids[layer - 1], prior["id"])
        ):
            raise EncryptedLayerGateError("encrypted layer is not causally fed by previous layer")
        per_layer.append(
            {
                "layer": layer,
                "operator_counts": dict(sorted(counts.items())),
                "score_shape": _positive_shape(selection["attention_scores"][0]),
                "attention_value_shape": _positive_shape(selection["attention_values"][0]),
                "silu_shape": _positive_shape(selection["silu"][0]),
                "forced_chain_depth_at_gate": depths[gate],
            }
        )
        gate_ids[layer] = gate
    boundary = [row for row in rows if row.get("layer") is None]
    boundary_counts = Counter(row["operator"] for row in boundary)
    if set(boundary_counts) != _CLIENT_BOUNDARY_KINDS or any(
        count != 1 for count in boundary_counts.values()
    ):
        raise EncryptedLayerGateError("encrypted boundary has an unpriced or duplicated operator")
    terminal = [row for row in boundary if row["operator"] == "output_head"]
    final_norm = next(row for row in boundary if row["operator"] == "rms_norm")
    if (
        depths[terminal[0]["id"]] < 4 * layer_count
        or not _reaches(operations, gate_ids[layer_count - 1], final_norm["id"])
        or not _reaches(operations, final_norm["id"], terminal[0]["id"])
    ):
        raise EncryptedLayerGateError("client output boundary does not depend on complete decoder")
    return {
        "layer_count": layer_count,
        "operator_count": len(rows),
        "layers": per_layer,
        "outside_layer_operator_counts": dict(sorted(boundary_counts.items())),
        "optimistic_serial_ciphertext_product_depth": depths[terminal[0]["id"]],
    }


def token_boundary_he_layer_gate(
    plan: ModelPlan,
    composition: Pipeline,
    *,
    response_new_tokens: int,
    slots_per_ciphertext: int = 4096,
    covered_all_link_body_budget_bytes: int,
    covered_online_body_budget_bytes: int,
) -> dict[str, Any]:
    """Check pinned semantic depth and ideal boundary counts, never admit HE.

    Output head/token choice stay at client. No encrypted operator is executable
    merely because its product appears in this deliberately optimistic depth.
    """
    if not isinstance(plan, ModelPlan) or not isinstance(composition, Pipeline):
        raise TypeError("HE gate requires a semantic ModelPlan and Pipeline")
    if type(response_new_tokens) is not int or not 1 <= response_new_tokens <= 256:
        raise EncryptedLayerGateError("response length is outside bounded HE gate")
    if type(slots_per_ciphertext) is not int or not 128 <= slots_per_ciphertext <= 16384:
        raise EncryptedLayerGateError("HE slot bound is invalid")
    for budget in (covered_all_link_body_budget_bytes, covered_online_body_budget_bytes):
        if type(budget) is not int or not 0 < budget <= 1 << 40:
            raise EncryptedLayerGateError("HE evidence budget is not a bounded positive integer")
    if covered_online_body_budget_bytes > covered_all_link_body_budget_bytes:
        raise EncryptedLayerGateError("online budget exceeds all-link budget")
    graph = plan.to_dict()
    if graph.get("token_feedback") is not True:
        raise EncryptedLayerGateError("HE token boundary lacks committed client feedback")
    prefill = graph["prefill"].get("query_sequence")
    if type(prefill) is not int or not 1 <= prefill <= 4096:
        raise EncryptedLayerGateError("HE graph lacks bounded input length")
    if response_new_tokens > graph["decode"]["maximum_key_sequence"] - prefill + 1:
        raise EncryptedLayerGateError("HE response exceeds declared key-state bound")
    schedule = plan.runtime_schedule(composition)
    if not schedule.complete or schedule.protected_execution:
        raise EncryptedLayerGateError("HE gate needs a complete semantic baseline schedule")
    placed = schedule.to_dict()
    for phase in ("prefill", "decode"):
        for kind in ("output_head", "greedy_token_selection", "token_feedback"):
            found = [row for row in graph[phase]["operations"] if row.get("operator") == kind]
            if len(found) != 1:
                raise EncryptedLayerGateError("HE token/head/feedback boundary is not unique")
            matches = [
                step
                for step in placed[phase]["steps"]
                if found[0]["id"] in step.get("operation_ids", [])
            ]
            if len(matches) != 1 or (
                kind != "output_head" and matches[0]["executor"] != "client_local"
            ):
                raise EncryptedLayerGateError("HE token/head/feedback boundary is not client-local")
    stages = scheduled_stage_specs(plan, composition)
    if stages[0].role != "token_lookup" or stages[-1].role != "lm_head":
        raise EncryptedLayerGateError("HE gate lacks client token boundary")
    body: dict[int, set[str]] = {}
    for stage in stages[1:-1]:
        if stage.role not in _BODY_ROLES or type(stage.layer_index) is not int:
            raise EncryptedLayerGateError("HE gate lacks complete body stage roles")
        row = body.setdefault(stage.layer_index, set())
        if stage.role in row:
            raise EncryptedLayerGateError("HE gate duplicates a body stage")
        row.add(stage.role)
    if (
        not body
        or len(body) > 128
        or sorted(body) != list(range(len(body)))
        or any(row != _BODY_ROLES for row in body.values())
    ):
        raise EncryptedLayerGateError("HE gate does not cover all semantic layers")
    width = stages[0].out_features
    if width != stages[-1].in_features or width <= 0:
        raise EncryptedLayerGateError("HE token input/output widths disagree")
    prefill_depth = _phase_depth(graph["prefill"], len(body))
    decode_depth = _phase_depth(graph["decode"], len(body))
    if (
        prefill_depth["optimistic_serial_ciphertext_product_depth"]
        != decode_depth["optimistic_serial_ciphertext_product_depth"]
    ):
        raise EncryptedLayerGateError("encrypted prefill/decode product dependency differs")
    import math

    initial = math.ceil(prefill * width / slots_per_ciphertext)
    feedback = response_new_tokens - 1
    output = response_new_tokens
    ciphertexts = initial + feedback + output
    return {
        "schema": "pllm.token_boundary_he_layer_gate.v1",
        "scope": "non-executable optimistic CKKS body-depth and client ciphertext boundaries",
        "plan_digest": plan.digest,
        "schedule_digest": schedule.digest,
        "composition_digest": schedule.composition_digest,
        "input_tokens": prefill,
        "output_tokens": response_new_tokens,
        "layer_count": len(body),
        "hidden_width": width,
        "semantic_phase_contracts": {"prefill": prefill_depth, "decode": decode_depth},
        "assumed_slots_per_ciphertext": slots_per_ciphertext,
        "idealized_client_to_provider_ciphertexts": initial + feedback,
        "idealized_provider_to_client_ciphertexts": output,
        "idealized_both_direction_ciphertexts": ciphertexts,
        "maximum_average_ciphertext_bytes_for_online_tenfold": covered_online_body_budget_bytes
        // ciphertexts,
        "covered_online_body_budget_bytes": covered_online_body_budget_bytes,
        "covered_all_link_body_budget_bytes": covered_all_link_body_budget_bytes,
        "omitted_cryptographic_work": [
            "two encrypted body RMSNorm reciprocal-square-roots and elementwise scales per layer",
            "stable causally masked softmax, exponentiation and normalization",
            "non-affine SiLU approximation and numeric error across real W8A8 decode",
            "encrypted real-checkpoint public projections, rotations and rescaling",
            "encrypted persistent KV ownership/updates and context-dependent refresh",
            "packing/repacking between attention heads and token boundary",
            "trusted-client final RMSNorm, output head and per-token selection after decryption",
            "evaluation key delivery, bootstraps, authentication and full wire",
        ],
        "client_key_only": True,
        "client_final_norm_and_output_head_required_but_not_he_placed": True,
        "numeric_fidelity_validated": False,
        "complete_encrypted_layer_available": False,
        "whole_decoder_executable": False,
    }


__all__ = ["EncryptedLayerGateError", "token_boundary_he_layer_gate"]
