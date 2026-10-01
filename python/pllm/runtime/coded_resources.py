"""Public, plan-bound lower bounds for coded delegated-linear preprocessing.

This is a storage feasibility calculation, not a code-distance certificate,
privacy proof, source-fidelity claim, or executable kernel admission.
"""

from __future__ import annotations

from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan

_BABYBEAR_MODULUS = 2_013_265_921
_CODE_RATE_DENOMINATOR = 8
_FIELD_ELEMENT_BYTES = 4
_MAX_STAGES = 4096


class CodedResourceError(ValueError):
    """Malformed or unsupported semantic stage in a coded-resource projection."""


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise CodedResourceError(f"{name} must be a positive integer")
    return value


def _width(shape: Any, name: str) -> int:
    if not isinstance(shape, list) or not shape:
        raise CodedResourceError(f"{name} must have a bounded output shape")
    return _positive_int(shape[-1], f"{name} width")


def coded_delegation_storage_lower_bound(
    plan: ModelPlan,
    composition: Pipeline,
    *,
    resident_budget_bytes: int,
    cached_budget_bytes: int,
) -> dict[str, Any]:
    """Bound public P=M Gx and Q=Gy^T M for each compiled remote W8 stage.

    Maverick's rate-1/8 privacy and verification codes each expand one
    matrix dimension eightfold. Each BabyBear field element occupies four
    bytes. Counts exclude batch-recursion auxiliary preprocessing, protocol
    state, actual cache metadata, mask generation, and model distribution.
    They are *lower bounds* on persistent preprocessing, not estimates of
    runtime memory or transport. No weight values or token data are read.
    """
    if type(plan) is not ModelPlan or not isinstance(composition, Pipeline):
        raise TypeError("plan and composition must be native ModelPlan and Pipeline values")
    resident_budget_bytes = _positive_int(resident_budget_bytes, "resident budget")
    cached_budget_bytes = _positive_int(cached_budget_bytes, "cache budget")
    numeric = composition.components.get("quantization")
    if numeric is not None and (
        numeric.component != "pllm/symmetric-per-row-quantization/v1"
        or numeric.params.get("weight_bits") != 8
        or numeric.params.get("activation_bits") != 8
    ):
        raise CodedResourceError("coded projection requires the bounded W8A8 integer cohort")
    try:
        schedule = plan.runtime_schedule(composition)
    except Exception as exc:
        raise CodedResourceError("composition has no complete unprotected runtime schedule") from exc
    if schedule.protected_execution or not schedule.complete:
        raise CodedResourceError("coded projection requires complete unprotected scheduling")
    document = plan.to_dict()
    prefill = document.get("prefill")
    operations = prefill.get("operations") if isinstance(prefill, dict) else None
    if not isinstance(operations, list):
        raise CodedResourceError("semantic prefill operators are missing")
    by_id: dict[str, dict[str, Any]] = {}
    for operation in operations:
        if not isinstance(operation, dict) or not isinstance(operation.get("id"), str):
            raise CodedResourceError("semantic prefill operation is malformed")
        if operation["id"] in by_id:
            raise CodedResourceError("semantic prefill operation is duplicated")
        by_id[operation["id"]] = operation

    rows: list[dict[str, int | str]] = []
    seen_orders: set[int] = set()
    steps = schedule.to_dict()["prefill"]["steps"]
    if len(steps) > _MAX_STAGES:
        raise CodedResourceError("coded projection stage count exceeds bounded limit")
    for step in steps:
        if step.get("executor") not in {"remote_stage", "verified_remote_stage"}:
            continue
        operators = step.get("operators")
        if not isinstance(operators, list) or not operators:
            raise CodedResourceError("remote stage semantic operators are malformed")
        if set(operators) <= {"token_lookup", "output_head"}:
            # The public token boundary is client-owned in the live role
            # protocol even when the native scheduler records a stage.
            continue
        if any(operator in {"token_lookup", "output_head"} for operator in operators):
            raise CodedResourceError("remote stage mixes client and provider ownership")
        order = step.get("order")
        if type(order) is not int or order < 0:
            raise CodedResourceError("remote stage order must be a nonnegative integer")
        if order in seen_orders:
            raise CodedResourceError("remote stage order is duplicated")
        seen_orders.add(order)
        input_ids = step.get("input_ids")
        outputs = step.get("outputs")
        if not isinstance(input_ids, list) or len(input_ids) != 1 or not isinstance(outputs, list) or not outputs:
            raise CodedResourceError("remote stage is missing its single input or output slices")
        source = by_id.get(input_ids[0])
        if source is None:
            raise CodedResourceError(
                f"remote stage input {input_ids[0]!r} has no semantic producer "
                f"for {step.get('operators')!r}"
            )
        input_width = _width(source.get("output_shape"), "remote stage input")
        output_width = sum(
            _positive_int(output.get("stage_width"), "remote output width")
            for output in outputs
            if isinstance(output, dict)
        )
        if len(outputs) != sum(isinstance(output, dict) for output in outputs) or output_width == 0:
            raise CodedResourceError("remote stage outputs are malformed")
        if input_width * 128 * 128 >= _BABYBEAR_MODULUS // 2:
            raise CodedResourceError("signed W8A8 output bound exceeds centered BabyBear field")
        elements = input_width * output_width
        if elements > 1 << 40:
            raise CodedResourceError("remote stage matrix exceeds bounded analysis")
        bytes_per_auxiliary = elements * _CODE_RATE_DENOMINATOR * _FIELD_ELEMENT_BYTES
        rows.append({
            "order": order,
            "input_width": input_width,
            "output_width": output_width,
            "public_w8_weight_bytes": elements,
            "privacy_p_bytes_lower_bound": bytes_per_auxiliary,
            "verification_q_bytes_lower_bound": bytes_per_auxiliary,
        })
    if not rows:
        raise CodedResourceError("composition has no remote public-weight stages")
    weight_bytes = sum(int(row["public_w8_weight_bytes"]) for row in rows)
    total = sum(
        int(row["privacy_p_bytes_lower_bound"]) + int(row["verification_q_bytes_lower_bound"])
        for row in rows
    )
    return {
        "schema": "pllm.coded_delegation_resource_lower_bound.v1",
        "scope": "public W8A8 remote stages in one complete semantic prefill schedule",
        "plan_digest": plan.digest,
        "schedule_digest": schedule.digest,
        "composition_digest": schedule.composition_digest,
        "field_modulus": _BABYBEAR_MODULUS,
        "privacy_code_rate_denominator": _CODE_RATE_DENOMINATOR,
        "verification_code_rate_denominator": _CODE_RATE_DENOMINATOR,
        "remote_stages": len(rows),
        "stage_dimensions": rows,
        "public_w8_weight_bytes": weight_bytes,
        "privacy_p_bytes_lower_bound": total // 2,
        "verification_q_bytes_lower_bound": total // 2,
        "preprocessing_bytes_lower_bound": total,
        "resident_budget_bytes": resident_budget_bytes,
        "cached_budget_bytes": cached_budget_bytes,
        "fully_resident_within_budget": total <= resident_budget_bytes,
        "fully_cached_within_budget": total <= cached_budget_bytes,
        "executable": False,
        "full_response_compute_cap_checked": False,
        "total_wire_bytes_measured": False,
        "unmeasured": [
            "batch-verification auxiliary matrix and code-distance certificate",
            "preprocessing CPU, memory-mapping overhead, distribution and cache misses",
            "dual-LPN security, finite-field checkpoint parity, transport and session burn",
        ],
    }
