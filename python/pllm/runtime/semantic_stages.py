"""Derive remote weight artifacts from the model-neutral decoder schedule."""

from __future__ import annotations

from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan

from .models import StageSpec


def _graph_reaches(
    start: str,
    target: str,
    operations: dict[str, dict[str, Any]],
    *,
    reverse: bool = False,
) -> bool:
    consumers: dict[str, list[str]] = {}
    if not reverse:
        for operation_id, operation in operations.items():
            for source in operation.get("inputs") or ():
                if source in operations:
                    consumers.setdefault(source, []).append(operation_id)
    frontier = [start]
    visited: set[str] = set()
    for _ in range(5):
        next_frontier: list[str] = []
        for operation_id in frontier:
            if operation_id in visited:
                continue
            visited.add(operation_id)
            operation = operations.get(operation_id)
            if operation is None:
                continue
            if operation.get("operator") == target:
                return True
            if reverse:
                next_frontier.extend(
                    source for source in operation.get("inputs") or () if source in operations
                )
            else:
                next_frontier.extend(consumers.get(operation_id, ()))
        frontier = next_frontier
    return False


def semantic_stage_role(step: dict[str, Any], operations: dict[str, dict[str, Any]]) -> str:
    operators = step.get("operators")
    if isinstance(operators, list) and operators and set(operators) == {"token_lookup"}:
        return "token_lookup"
    if operators == ["output_head"]:
        return "lm_head"
    operation_ids = step.get("operation_ids")
    if not isinstance(operation_ids, list) or not operation_ids:
        raise ValueError("remote stage is missing semantic operation identities")
    if (
        len(operation_ids) == 3
        and sum(
            _graph_reaches(operation_id, "rotary_embedding", operations)
            for operation_id in operation_ids
        )
        == 2
    ):
        return "qkv_projection"
    if (
        len(operation_ids) == 2
        and all(
            _graph_reaches(operation_id, "multiply", operations) for operation_id in operation_ids
        )
        and sum(_graph_reaches(operation_id, "silu", operations) for operation_id in operation_ids)
        == 1
    ):
        return "mlp_gate_up"
    if len(operation_ids) == 1:
        input_ids = step.get("input_ids")
        if not isinstance(input_ids, list) or len(input_ids) != 1:
            raise ValueError("remote linear stage must have one semantic input")
        source = input_ids[0]
        if _graph_reaches(source, "attention_values", operations, reverse=True):
            return "attention_output"
        if _graph_reaches(source, "multiply", operations, reverse=True):
            return "mlp_down"
    raise ValueError("remote stage topology is not implemented")


_STAGE_NAMES = {
    "qkv_projection": "self_attn.qkv_proj",
    "attention_output": "self_attn.o_proj",
    "mlp_gate_up": "mlp.gate_up_proj",
    "mlp_down": "mlp.down_proj",
}

_FUSED_SEMANTIC_ROLES = {
    "qkv_projection": ("q_proj", "k_proj", "v_proj"),
    "mlp_gate_up": ("gate_proj", "up_proj"),
}


def semantic_fused_roles(role: str) -> tuple[str, ...]:
    """Ordered operator roles; never infer them from checkpoint weight paths."""
    return _FUSED_SEMANTIC_ROLES.get(role, ())


def scheduled_stage_specs(plan: ModelPlan, composition: Pipeline) -> list[StageSpec]:
    """Issue the exact stage table required by a complete baseline schedule."""
    schedule = plan.runtime_schedule(composition)
    if not schedule.complete or schedule.protected_execution:
        raise ValueError("decoder schedule is not complete public baseline execution")
    graph = plan.prefill
    operations = {operation["id"]: operation for operation in graph["operations"]}
    output_heads = [
        int(operation["output_shape"][-1])
        for operation in operations.values()
        if operation["operator"] == "output_head"
    ]
    if len(output_heads) != 1:
        raise ValueError("decoder schedule requires one output head")
    vocabulary = output_heads[0]
    prefill = schedule.to_dict()["prefill"]
    stages: list[StageSpec] = []
    seen: set[tuple[str, int | None]] = set()
    for step in prefill["steps"]:
        if step["executor"] != "remote_stage":
            continue
        role = semantic_stage_role(step, operations)
        layer = step["layer"]
        key = (role, layer)
        if key in seen:
            raise ValueError("decoder schedule contains a duplicate stage role")
        seen.add(key)
        op_ids = step["operation_ids"]
        weights = tuple(step["weight_ids"])
        if len(weights) != len(op_ids) or not weights:
            raise ValueError("decoder stage is missing its ordered weight artifacts")
        declared_biases = tuple(
            operations[op_id]["attributes"].get("bias") or "" for op_id in op_ids
        )
        if any(declared_biases) and len(declared_biases) != len(weights):
            raise ValueError("semantic stage bias declaration does not cover each weight")
        if role == "token_lookup":
            spec = StageSpec(
                id="token_lookup",
                op="embedding",
                in_features=vocabulary,
                out_features=int(operations[op_ids[0]]["output_shape"][-1]),
                weight_keys=weights,
                bias_keys=declared_biases if any(declared_biases) else (),
                transpose_weight=True,
                role=role,
                metadata={"ple_width": 0},
            )
        elif role == "lm_head":
            input_id = step["input_ids"][0]
            spec = StageSpec(
                id="lm_head",
                op="lm_head",
                in_features=int(operations[input_id]["output_shape"][-1]),
                out_features=int(operations[op_ids[0]]["output_shape"][-1]),
                weight_keys=weights,
                bias_keys=declared_biases if any(declared_biases) else (),
                role=role,
            )
        else:
            if type(layer) is not int or role not in _STAGE_NAMES:
                raise ValueError("decoder schedule declares an unsupported stage layout")
            inputs = step["input_ids"]
            if len(inputs) != 1 or inputs[0] not in operations:
                raise ValueError("decoder stage lacks one semantic input")
            spec = StageSpec(
                id=f"layers.{layer}.{_STAGE_NAMES[role]}",
                op="linear",
                in_features=int(operations[inputs[0]]["output_shape"][-1]),
                out_features=sum(int(operations[op_id]["output_shape"][-1]) for op_id in op_ids),
                fused_from=semantic_fused_roles(role),
                weight_keys=weights,
                bias_keys=declared_biases if any(declared_biases) else (),
                layer_index=layer,
                role=role,
            )
        stages.append(spec)
    if not stages or stages[0].role != "token_lookup" or stages[-1].role != "lm_head":
        raise ValueError("decoder schedule lacks its token boundary stages")
    return stages


__all__ = ["scheduled_stage_specs", "semantic_fused_roles", "semantic_stage_role"]
