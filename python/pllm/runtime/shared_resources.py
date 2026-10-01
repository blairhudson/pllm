"""Cost gate for the existing two-party fixed-scale MLP reference.

Only the mandatory gated product and its exact rescale are counted. This
optimistic projection omits attention, RMSNorm, activation, transport framing,
and all other decoder work. It never authorizes a protected semantic plan.
"""

from __future__ import annotations

from math import prod
from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan

class SharedResourceError(ValueError):
    """Malformed semantic schedule or unsupported fixed-scale projection."""


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise SharedResourceError(f"{name} must be a positive integer")
    return value


def resident_mlp_resource_gate(
    plan: ModelPlan,
    composition: Pipeline,
    *,
    response_new_tokens: int,
    fixed_scale_bits: int,
    maximum_material_bytes_per_party: int,
    maximum_online_all_link_body_bytes: int,
) -> dict[str, Any]:
    """Lower bound the existing Beaver-product + FSS-rescale MLP path.

    The semantic plan supplies tensor shapes only; its float32 SiLU and
    softmax are not the fixed-scale reference's numeric operators. The
    projection assumes one Beaver multiplication and one exact FSS shift
    per gated MLP output, ignoring every other decoder primitive.
    """
    if type(plan) is not ModelPlan or not isinstance(composition, Pipeline):
        raise TypeError("plan and composition must be ModelPlan and Pipeline values")
    if type(fixed_scale_bits) is not int or not 1 <= fixed_scale_bits <= 10:
        raise SharedResourceError("bounded FSS reference requires one to ten scale bits")
    if type(response_new_tokens) is not int or not 1 <= response_new_tokens <= 256:
        raise SharedResourceError("response new-token count exceeds bounded projection")
    material_limit = _positive(maximum_material_bytes_per_party, "material budget")
    online_limit = _positive(maximum_online_all_link_body_bytes, "online body budget")
    if plan.to_dict().get("transformations"):
        raise SharedResourceError("transformed decoder has no bound reference projection")
    numeric = composition.components.get("quantization")
    if numeric is not None and (
        numeric.component != "pllm/symmetric-per-row-quantization/v1"
        or numeric.params.get("weight_bits") != 8
        or numeric.params.get("activation_bits") != 8
    ):
        raise SharedResourceError("matched comparator requires the W8A8 integer cohort")
    try:
        schedule = plan.runtime_schedule(composition)
    except Exception as exc:
        raise SharedResourceError("decoder has no complete semantic schedule") from exc
    if not schedule.complete or schedule.protected_execution:
        raise SharedResourceError("reference projection needs a complete unprotected schedule")
    document, scheduled = plan.to_dict(), schedule.to_dict()
    prefill_graph, decode_graph = document["prefill"], document["decode"]
    input_tokens = _positive(prefill_graph.get("query_sequence"), "prefill query length")
    maximum_decode_key = _positive(decode_graph.get("maximum_key_sequence"), "decode key length")
    if maximum_decode_key < input_tokens + response_new_tokens - 1:
        raise SharedResourceError("response length exceeds declared decode key bound")
    counts: dict[str, int] = {}
    for phase in ("prefill", "decode"):
        graph = document.get(phase)
        bound = scheduled.get(phase)
        if not isinstance(graph, dict) or not isinstance(bound, dict):
            raise SharedResourceError("decoder phase lacks bound operations")
        rows = graph.get("operations")
        steps = bound.get("steps")
        if not isinstance(rows, list) or not isinstance(steps, list) or len(steps) > 4096:
            raise SharedResourceError("decoder phase exceeds bounded schedule")
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in by_id:
                raise SharedResourceError("decoder has malformed or repeated operation")
            by_id[row["id"]] = row
        scheduled_ids = [name for step in steps for name in step.get("operation_ids", ())]
        if len(scheduled_ids) != len(set(scheduled_ids)) or set(scheduled_ids) != set(by_id):
            raise SharedResourceError("semantic schedule does not cover operations exactly")
        seen: set[str] = set()
        total = 0
        for step in steps:
            for name in step.get("operation_ids", ()):
                op = by_id[name]
                if op.get("operator") != "multiply":
                    continue
                if step.get("executor") != "client_local":
                    raise SharedResourceError("gated MLP placement is not local in source schedule")
                inputs = op.get("inputs")
                if (
                    name in seen or not isinstance(inputs, list) or len(inputs) != 2
                    or {by_id.get(source, {}).get("operator") for source in inputs}
                    != {"silu", "linear"}
                ):
                    raise SharedResourceError("gated MLP product is not a supported semantic pair")
                seen.add(name)
                shape = op.get("output_shape")
                if (
                    not isinstance(shape, list) or not 2 <= len(shape) <= 4
                    or any(type(size) is not int or not 0 < size <= 1 << 20 for size in shape)
                ):
                    raise SharedResourceError("gated MLP output has invalid shape")
                total += prod(shape)
                if total > 1 << 30:
                    raise SharedResourceError("gated MLP element projection exceeds bound")
        if not seen:
            raise SharedResourceError("decoder phase has no bounded gated MLP product")
        counts[phase] = total
    decoded_steps = response_new_tokens - 1
    elements = counts["prefill"] + counts["decode"] * decoded_steps
    # The same full-session material arithmetic used by ExactSessionAdmission:
    # three uint64 triple shares, four uint64 FSS mask shares, one Boolean
    # mask share, and a (17 + 17*b)-byte one-use point key per party/element.
    mask_body_per_element = 33 + 17 + 17 * fixed_scale_bits
    material_per_party = elements * (24 + mask_body_per_element)
    # Each party sends two uint64 Beaver openings and one uint64 masked
    # truncation opening. Each sends a packed FSS bit share. Frame headers,
    # input/output shares, setup, checkpoint delivery and other ops omitted.
    minimum_online_bodies = 2 * (24 * elements + (elements + 7) // 8)
    return {
        "schema": "pllm.resident_mlp_resource_gate.v1",
        "scope": "one Beaver gated product and one exact FSS rescale per semantic MLP element",
        "plan_digest": plan.digest,
        "schedule_digest": schedule.digest,
        "composition_digest": schedule.composition_digest,
        "fixed_scale_bits": fixed_scale_bits,
        "response_new_tokens": response_new_tokens,
        "prefill_elements": counts["prefill"],
        "decode_elements": counts["decode"] * decoded_steps,
        "minimum_material_body_bytes_per_party": material_per_party,
        "minimum_material_body_bytes_two_parties": 2 * material_per_party,
        "minimum_online_all_link_body_bytes": minimum_online_bodies,
        "maximum_material_bytes_per_party": material_limit,
        "maximum_online_all_link_body_bytes": online_limit,
        "material_within_budget": material_per_party <= material_limit,
        "online_within_budget": minimum_online_bodies <= online_limit,
        "executable": False,
        "full_wire_measured": False,
        "unmeasured": [
            "other layers, norms, softmax, attention, input/output, metadata and checkpoint delivery",
            "correlation issuance CPU, retained object overhead, actual network framing and quality",
        ],
    }
