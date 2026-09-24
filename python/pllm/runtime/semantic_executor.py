"""Bound semantic decoder execution; model adapters do not dispatch here."""

from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from pllm.modeling import ModelPlan

from .transformer_client import ClientBundle, MaskedTransformerClientRuntime, TransformerClientError


class SemanticDecoderRuntime(MaskedTransformerClientRuntime):
    """Execute validated semantic steps with client-owned attention and cache state."""

    def __init__(
        self,
        bundle: ClientBundle,
        remote: Callable[[str, np.ndarray], np.ndarray],
        *,
        plan: ModelPlan,
        schedule: dict[str, Any],
        stages: Mapping[str, str],
        tensors: Mapping[str, str],
        token_cache: OrderedDict[int, np.ndarray] | None = None,
        token_cache_size: int = 512,
        token_cache_lock: threading.Lock | None = None,
        nonlinear_evaluator: Callable[[int, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        super().__init__(
            bundle,
            remote,
            token_cache=token_cache,
            token_cache_size=token_cache_size,
            token_cache_lock=token_cache_lock,
            nonlinear_evaluator=nonlinear_evaluator,
        )
        self._graphs = plan.to_dict()
        self._schedule = schedule
        self._stages = dict(stages)
        self._tensors = dict(tensors)

    def forward_ids(self, ids: list[int] | np.ndarray) -> np.ndarray:
        """Expose per-token logits via the bound prefill and decode phases."""
        tokens = np.asarray(ids, dtype=np.int64).reshape(-1)
        if tokens.size == 0:
            raise TransformerClientError("at least one token is required")
        if tokens.size == 1:
            return self._forward(tokens)
        phase = "prefill" if self.position == 0 else "decode"
        graph = self._graphs[phase]
        if (
            phase == "prefill" and tokens.size > graph["query_sequence"]
        ) or self.position + tokens.size > graph["maximum_key_sequence"]:
            raise TransformerClientError("semantic execution exceeds compiled workload bounds")
        return np.concatenate(
            [self._forward(tokens[index : index + 1]) for index in range(tokens.size)], axis=0
        )

    def _weight(self, name: str) -> np.ndarray:
        key = self._tensors.get(name)
        if key is None:
            raise TransformerClientError("semantic operation lacks a bound local tensor")
        return np.asarray(self.bundle.arrays[key], dtype=np.float32)

    @staticmethod
    def _rotary(value: np.ndarray, positions: np.ndarray, attributes: dict[str, Any]) -> np.ndarray:
        rotary_dim = int(attributes["rotary_dimensions"])
        theta = float(attributes["theta"])
        if rotary_dim == 0:
            return value
        head = value.shape[-1]
        if rotary_dim > head or rotary_dim % 2 or theta <= 0:
            raise TransformerClientError("semantic rotary dimensions are invalid")
        frequency = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        angles = positions.astype(np.float32)[:, None] * frequency[None, :]
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[None, None, :, :]
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[None, None, :, :]
        current = value[..., :rotary_dim]
        half = rotary_dim // 2
        rotated = np.concatenate([-current[..., half:], current[..., :half]], axis=-1)
        result = value.copy()
        result[..., :rotary_dim] = current * cos + rotated * sin
        return result

    def _local(
        self,
        operation: dict[str, Any],
        values: dict[str, Any],
        state_kinds: dict[str, str],
        pending_keys: dict[int, np.ndarray],
    ) -> Any:
        kind = operation["operator"]
        inputs = operation["inputs"]
        attrs = operation["attributes"]
        layer = operation.get("layer")
        source = values[inputs[0]]
        if kind == "reshape":
            layout = attrs.get("layout")
            if layout == "batch_heads_sequence_feature":
                shape = operation["output_shape"]
                return source.reshape(1, source.shape[0], shape[1], shape[-1]).transpose(0, 2, 1, 3)
            if layout == "batch_sequence_hidden":
                return source.transpose(0, 2, 1, 3).reshape(source.shape[2], -1)
        elif kind == "rms_norm":
            weight = self._weight(str(attrs["weight"]))
            epsilon = float(attrs["epsilon"])
            offset = int(attrs["weight_offset"])
            return (
                source
                / np.sqrt(np.mean(source * source, axis=-1, keepdims=True) + epsilon)
                * (weight + offset)
            )
        elif kind == "rotary_embedding":
            return self._rotary(source, values[inputs[1]], attrs)
        elif kind == "kv_cache_append":
            if type(layer) is not int:
                raise TransformerClientError("semantic cache operation lacks a layer")
            cache = self.caches[layer]
            tensor = values[inputs[1] if attrs["mode"] == "append" else inputs[0]][0].transpose(
                1, 0, 2
            )
            state_kind = state_kinds.get(operation["id"])
            if state_kind == "key":
                pending_keys[layer] = tensor
                prior = cache.key[: cache.length] if cache.key is not None else tensor[:0]
                return np.concatenate((prior, tensor), axis=0).transpose(1, 0, 2)[None]
            if state_kind == "value" and layer in pending_keys:
                key, value = cache.append(pending_keys.pop(layer), tensor)
                return value.transpose(1, 0, 2)[None]
        elif kind == "cache_suffix":
            return source
        elif kind == "attention_scores":
            query, keys = source, values[inputs[1]]
            group = int(attrs["group_size"])
            batch, heads, sequence, width = query.shape
            kv_heads = keys.shape[1]
            if batch != 1 or heads != kv_heads * group:
                raise TransformerClientError("semantic attention group shape is invalid")
            grouped = query.reshape(batch, kv_heads, group, sequence, width)
            scores = np.empty((batch, kv_heads, group, sequence, keys.shape[2]), dtype=np.float32)
            for row in range(sequence):
                scores[0, :, :, row, :] = np.einsum(
                    "kgd,tkd->kgt", grouped[0, :, :, row, :], keys[0].transpose(1, 0, 2)
                )
            return scores.reshape(batch, heads, sequence, keys.shape[2])
        elif kind == "attention_scale":
            return source * float(1.0 / math.sqrt(int(attrs["head_dim"])))
        elif kind == "causal_mask":
            positions = values[inputs[1]]
            indices = np.arange(source.shape[-1])[None, None, None, :]
            return np.where(indices <= positions[None, None, :, None], source, -np.inf)
        elif kind == "softmax":
            centered = source - source.max(axis=-1, keepdims=True)
            probabilities = np.exp(centered).astype(np.float32)
            return probabilities / probabilities.sum(axis=-1, keepdims=True)
        elif kind == "attention_values":
            probabilities, cache = source, values[inputs[1]]
            batch, heads, sequence, length = probabilities.shape
            kv_heads = cache.shape[1]
            group = int(attrs["group_size"])
            if batch != 1 or heads != kv_heads * group or length != cache.shape[2]:
                raise TransformerClientError("semantic attention value shape is invalid")
            grouped = probabilities.reshape(batch, kv_heads, group, sequence, length)
            result = np.empty((batch, kv_heads, group, sequence, cache.shape[-1]), dtype=np.float32)
            for row in range(sequence):
                result[0, :, :, row, :] = np.einsum(
                    "kgt,tkd->kgd", grouped[0, :, :, row, :], cache[0].transpose(1, 0, 2)
                )
            return result.reshape(batch, heads, sequence, cache.shape[-1])
        elif kind == "residual_add":
            return source + values[inputs[1]]
        elif kind == "silu":
            if self.nonlinear_evaluator is not None:
                if type(layer) is not int:
                    raise TransformerClientError("semantic nonlinear operation lacks a layer")
                evaluated = np.asarray(self.nonlinear_evaluator(layer, source))
                if (
                    evaluated.dtype != np.float32
                    or evaluated.shape != source.shape
                    or not np.all(np.isfinite(evaluated))
                ):
                    raise TransformerClientError(
                        "nonlinear evaluator returned an invalid activation"
                    )
                return evaluated
            result = np.empty_like(source)
            nonnegative = source >= 0
            result[nonnegative] = source[nonnegative] / (1.0 + np.exp(-source[nonnegative]))
            negative_exp = np.exp(source[~nonnegative])
            result[~nonnegative] = source[~nonnegative] * negative_exp / (1.0 + negative_exp)
            return result
        elif kind == "multiply":
            return source * values[inputs[1]]
        elif kind == "last_token":
            return source[-1:]
        elif kind == "greedy_token_selection":
            return np.argmax(source, axis=-1).astype(np.int64)
        elif kind == "token_feedback":
            return source
        raise TransformerClientError(f"semantic operator {kind!r} is not executable")

    def _forward(self, ids: np.ndarray, *, final_logits_only: bool = False) -> np.ndarray:
        phase = "prefill" if self.position == 0 else "decode"
        graph = self._graphs[phase]
        if (
            ids.size < 1
            or (phase == "prefill" and ids.size > graph["query_sequence"])
            or (phase == "decode" and ids.size != 1)
            or self.position + ids.size > graph["maximum_key_sequence"]
        ):
            raise TransformerClientError("semantic execution exceeds compiled workload bounds")
        positions = np.arange(self.position, self.position + ids.size, dtype=np.int64)
        values: dict[str, Any] = {
            "input.tokens": ids,
            "input.positions": positions,
            "input.sequence_lengths": np.asarray([self.position + ids.size], dtype=np.int64),
            "input.attention_mask": np.ones((1, ids.size), dtype=np.bool_),
        }
        for state in graph["state_inputs"]:
            cache = self.caches[int(state["layer"])]
            stored = cache.key if state["kind"] == "key" else cache.value
            if stored is None or cache.length != self.position:
                raise TransformerClientError("semantic cache state is unavailable")
            values[state["id"]] = stored[: cache.length].transpose(1, 0, 2)[None]
        operations = {op["id"]: op for op in graph["operations"]}
        state_kinds = {state["id"]: state["kind"] for state in graph["state_outputs"]}
        steps = self._schedule[phase]["steps"]
        remaining = Counter(input_id for step in steps for input_id in step["input_ids"])
        pending_keys: dict[int, np.ndarray] = {}
        for step in steps:
            op_ids = step["operation_ids"]
            if step["executor"] == "client_local":
                operation = operations[op_ids[0]]
                values[op_ids[0]] = self._local(operation, values, state_kinds, pending_keys)
            elif step["executor"] == "remote_stage":
                stage_ids = {self._stages[f"{phase}:{op_id}"] for op_id in op_ids}
                if len(stage_ids) != 1:
                    raise TransformerClientError("semantic remote step has multiple stage bindings")
                stage_id = stage_ids.pop()
                if operations[op_ids[0]]["operator"] == "token_lookup":
                    if op_ids != ["token_lookup"]:
                        raise TransformerClientError("semantic token lookup cannot be fused")
                    values[op_ids[0]] = self._token_lookup(ids)[0]
                elif operations[op_ids[0]]["operator"] == "output_head":
                    input_value = values[step["input_ids"][0]]
                    stage = self.bundle.stages[stage_id]
                    values[op_ids[0]] = (
                        self.bundle.local_linear(stage_id, input_value)
                        if stage.client_weight is not None
                        else self.remote(stage_id, input_value)
                    )
                else:
                    input_value = values[step["input_ids"][0]]
                    output = self.remote(stage_id, input_value)
                    for op_id, row in zip(op_ids, step["outputs"], strict=True):
                        offset = int(row["stage_offset"])
                        width = int(row["stage_width"])
                        values[op_id] = output[..., offset : offset + width]
            else:
                raise TransformerClientError("semantic runtime executor is unsupported")
            for input_id in step["input_ids"]:
                remaining[input_id] -= 1
                if remaining[input_id] == 0 and input_id != "output_head":
                    values.pop(input_id, None)
        if pending_keys:
            raise TransformerClientError("semantic cache key has no matching value")
        self.position += ids.size
        return np.asarray(values["output_head"], dtype=np.float32)


__all__ = ["SemanticDecoderRuntime"]
