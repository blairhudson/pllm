"""Derive remote weight artifacts from the model-neutral decoder schedule."""

from __future__ import annotations

from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan

from .models import StageSpec


def client_owns_linear(
    stage: Any, *, client_prefix_layers: int = 0, client_linear_roles: tuple[str, ...] = ()
) -> bool:
    """Shared ownership rule for compiled body stages and validated bundle specs."""
    get = stage.get if isinstance(stage, dict) else lambda key: getattr(stage, key, None)
    layer = get("layer_index")
    return get("op") == "linear" and (
        (type(layer) is int and 0 <= layer < client_prefix_layers)
        or get("role") in client_linear_roles
    )


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
            if (
                reverse
                and operation_id != start
                and operation.get("operator") in {"linear", "token_lookup", "output_head"}
            ):
                continue
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
        and sum(
            any(
                _graph_reaches(operation_id, activation, operations)
                for activation in ("silu", "gelu_tanh")
            )
            for operation_id in operation_ids
        )
        == 1
    ):
        return "mlp_gate_up"
    if operators == ["linear"] * len(operation_ids):
        input_ids = step.get("input_ids")
        if not isinstance(input_ids, list) or len(input_ids) != 1:
            raise ValueError("grouped remote linear stage must have one semantic input")
        if len(operation_ids) > 1:
            if any(
                tuple(operations[operation_id].get("inputs") or ()) != tuple(input_ids)
                for operation_id in operation_ids
            ):
                raise ValueError("grouped linear projections must share one semantic input")
            return "semantic_linear"
        source = input_ids[0]
        if _graph_reaches(source, "attention_values", operations, reverse=True):
            return "attention_output"
        producer = operations.get(source)
        if (
            producer is not None
            and producer.get("operator") == "multiply"
            and len(producer.get("inputs") or ()) == 2
            and sum(
                operations.get(input_id, {}).get("operator") in {"silu", "gelu_tanh"}
                for input_id in producer["inputs"]
            )
            == 1
            and sum(
                operations.get(input_id, {}).get("operator") == "linear"
                for input_id in producer["inputs"]
            )
            == 1
        ):
            return "mlp_down"
        return "semantic_linear"
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


def _token_lookup_stage(
    step: dict[str, Any], operations: dict[str, dict[str, Any]], vocabulary: int
) -> StageSpec:
    op_ids = step["operation_ids"]
    outputs = step["outputs"]
    weights = step["weight_ids"]
    if (
        not isinstance(op_ids, list)
        or not op_ids
        or not isinstance(outputs, list)
        or len(outputs) != len(op_ids)
        or not isinstance(weights, list)
        or len(weights) != len(op_ids)
        or step.get("input_ids") != ["input.tokens"]
        or step.get("layer") is not None
    ):
        raise ValueError("semantic token lookup does not declare ordered boundary artifacts")
    width = 0
    for op_id, output, weight in zip(op_ids, outputs, weights, strict=True):
        operation = operations.get(op_id)
        if (
            operation is None
            or operation.get("operator") != "token_lookup"
            or tuple(operation.get("inputs") or ()) != ("input.tokens",)
            or not isinstance(output, dict)
            or output.get("operation_id") != op_id
            or output.get("stage_offset") != width
            or output.get("stage_width") != operation["output_shape"][-1]
            or operation["attributes"].get("weight") != weight
        ):
            raise ValueError(f"semantic token lookup stage offsets or weights disagree: {op_id}")
        width += int(output["stage_width"])
    primary_width = int(outputs[0]["stage_width"])
    if primary_width <= 0 or width < primary_width:
        raise ValueError("semantic token lookup width is invalid")
    return StageSpec(
        id="token_lookup",
        op="embedding",
        in_features=vocabulary,
        out_features=width,
        weight_keys=tuple(weights),
        transpose_weight=True,
        role="token_lookup",
        metadata={"ple_width": width - primary_width},
    )


def _linear_stage(
    step: dict[str, Any], operations: dict[str, dict[str, Any]], role: str
) -> StageSpec:
    layer = step["layer"]
    inputs = step["input_ids"]
    op_ids = step["operation_ids"]
    if len(inputs) != 1 or inputs[0] not in operations:
        raise ValueError("decoder stage lacks one semantic input")
    if role == "semantic_linear":
        order = step.get("order")
        if type(order) is not int or order < 0 or (layer is not None and type(layer) is not int):
            raise ValueError("semantic linear stage is missing its bound order")
        prefix = "boundary" if layer is None else f"layer.{layer}"
        stage_id = f"semantic.{prefix}.linear.{order}"
    elif type(layer) is int and role in _STAGE_NAMES:
        stage_id = f"layers.{layer}.{_STAGE_NAMES[role]}"
    else:
        raise ValueError("decoder schedule declares an unsupported stage layout")
    declared_biases = tuple(operations[op_id]["attributes"].get("bias") or "" for op_id in op_ids)
    return StageSpec(
        id=stage_id,
        op="linear",
        in_features=int(operations[inputs[0]]["output_shape"][-1]),
        out_features=sum(int(operations[op_id]["output_shape"][-1]) for op_id in op_ids),
        fused_from=semantic_fused_roles(role),
        weight_keys=tuple(step["weight_ids"]),
        bias_keys=declared_biases if any(declared_biases) else (),
        layer_index=layer,
        role=role,
    )


def scheduled_stage_specs(plan: ModelPlan, composition: Pipeline) -> list[StageSpec]:
    """Issue all compiled linear stages, including client-owned prefix stages."""
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
        if step["executor"] not in {"remote_stage", "verified_remote_stage", "client_linear"}:
            continue
        role = semantic_stage_role(step, operations)
        layer = step["layer"]
        key = (role, step["order"] if role == "semantic_linear" else layer)
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
            if any(declared_biases):
                raise ValueError("token boundary lookup cannot declare projection biases")
            spec = _token_lookup_stage(step, operations, vocabulary)
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
            spec = _linear_stage(step, operations, role)
        stages.append(spec)
    if not stages or stages[0].role != "token_lookup" or stages[-1].role != "lm_head":
        raise ValueError("decoder schedule lacks its token boundary stages")
    return stages


__all__ = ["scheduled_stage_specs", "semantic_fused_roles", "semantic_stage_role"]
