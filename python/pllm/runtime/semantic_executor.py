"""Bound semantic decoder execution; model adapters do not dispatch here."""

from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np

from pllm.modeling import ModelPlan

from .semantic_attention import (
    SemanticAttentionError,
    mask_causal_scores,
    semantic_attention_scores,
    semantic_attention_values,
)
from .semantic_numeric import (
    SemanticNumericError,
    bfloat16_gelu_tanh,
    bfloat16_rms_norm,
    bfloat16_rotary,
    bfloat16_scale,
    bfloat16_softmax,
    bfloat16_softcap,
    float32_rotary_per_frequency,
    float32_rotary_text_mrope,
    float32_rotary_wavelength,
    round_bfloat16,
)
from .semantic_state import SemanticStateError, WindowedLayerCache
from .semantic_hybrid import (
    HybridStateError, bounded_gated_delta_decay, execute_declared_causal_convolution,
    execute_declared_gated_delta, execute_declared_gated_rms_norm,
)
from . import semantic_state
from .transformer_client import (
    ClientBundle,
    LayerCache,
    MaskedTransformerClientRuntime,
    TransformerClientError,
)


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
        self._text_only_tokens = self._excluded_multimodal_tokens(self._graphs, bundle.cfg)
        self._schedule = schedule
        self._stages = dict(stages)
        self._tensors = dict(tensors)
        self._window_contracts = self._declared_windows(self._graphs)
        self._window_state_bytes = self._window_resource_bytes(
            self._graphs, self._window_contracts
        )
        self._hybrid_contracts = self._declared_hybrid_state(self._graphs)
        self.reset()

    @staticmethod
    def _excluded_multimodal_tokens(graphs: dict[str, Any], cfg: dict[str, Any]) -> frozenset[int]:
        text_only = [
            op for graph in (graphs["prefill"], graphs["decode"])
            for op in graph["operations"] if op["operator"] == "rotary_embedding"
            and op["attributes"].get("position_policy") == "text_replicated_axes"
        ]
        if not text_only:
            return frozenset()
        source = cfg.get("semantic_source_config")
        if not isinstance(source, dict):
            raise TransformerClientError("text-only rotary requires the exact nested source lock")
        names = ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
        ids = [source.get(name) for name in names]
        if any(type(value) is not int or value < 0 for value in ids):
            raise TransformerClientError("text-only rotary lacks its public multimodal token boundary")
        return frozenset(cast(int, value) for value in ids)

    @staticmethod
    def _declared_hybrid_state(graphs: dict[str, Any]) -> dict[tuple[int, str], tuple[int, ...]]:
        def declared(rows: list[dict[str, Any]]) -> dict[tuple[int, str], tuple[int, ...]]:
            by_owner: dict[tuple[int, str], tuple[int, ...]] = {}
            for row in rows:
                kind = row["kind"]
                if kind not in {"convolution", "recurrent"}:
                    continue
                layer, shape = row.get("layer"), row.get("shape")
                if (
                    type(layer) is not int or not isinstance(shape, list)
                    or (layer, kind) in by_owner
                    or any(type(value) is not int or value < 1 for value in shape)
                    or (kind == "convolution" and (len(shape) != 3 or shape[0] != 1))
                    or (kind == "recurrent" and (len(shape) != 4 or shape[0] != 1))
                ):
                    raise TransformerClientError("semantic hybrid state has invalid ownership")
                by_owner[(layer, kind)] = tuple(shape)
            return by_owner

        if graphs["prefill"]["state_inputs"]:
            raise TransformerClientError("semantic prefill requires uninitialized hybrid state")
        prefill = declared(graphs["prefill"]["state_outputs"])
        decode = declared(graphs["decode"]["state_inputs"])
        if prefill != decode or declared(graphs["decode"]["state_outputs"]) != prefill:
            raise TransformerClientError("semantic hybrid state differs across phases")
        if sum(math.prod(shape) * 4 for shape in prefill.values()) > 2 << 30:
            raise TransformerClientError("semantic hybrid state exceeds client memory budget")
        return prefill

    @staticmethod
    def _declared_windows(graphs: dict[str, Any]) -> dict[int, int]:
        phase_windows: list[dict[int, dict[str, int]]] = []
        for phase in ("prefill", "decode"):
            windowed: dict[int, dict[str, int]] = {}
            for operation in graphs[phase]["operations"]:
                if operation["operator"] != "kv_cache_append":
                    continue
                attrs = operation["attributes"]
                domain = attrs.get("attention_domain", {})
                layout = domain.get("layout")
                if layout == "batch_kv_heads_sequence_feature":
                    continue
                window = domain.get("maximum_sequence")
                layer = operation.get("layer")
                kind = operation.get("state_kind")
                if (
                    layout != "batch_kv_heads_query_window_feature"
                    or type(window) is not int
                    or window < 1
                    or type(layer) is not int
                    or kind not in {"key", "value"}
                    or attrs.get("state_capacity") != window - 1
                    or attrs.get("state_layout") != "batch_kv_heads_sequence_feature"
                ):
                    raise TransformerClientError("semantic windowed state declaration is invalid")
                by_kind = windowed.setdefault(layer, {})
                if kind in by_kind:
                    raise TransformerClientError("semantic windowed state has duplicate owners")
                by_kind[kind] = window
            phase_windows.append(windowed)
        if phase_windows[0] != phase_windows[1] or any(
            set(by_kind) != {"key", "value"} or by_kind["key"] != by_kind["value"]
            for by_kind in phase_windows[0].values()
        ):
            raise TransformerClientError("semantic windowed state differs across phases")
        return {layer: by_kind["key"] for layer, by_kind in phase_windows[0].items()}

    @staticmethod
    def _window_resource_bytes(graphs: dict[str, Any], windows: dict[int, int]) -> int:
        if not windows:
            return 0
        used: dict[tuple[int, str], int] = {}
        for state in graphs["decode"]["state_inputs"]:
            layer, kind = state.get("layer"), state.get("kind")
            if layer not in windows:
                continue
            shape = state.get("shape")
            if (
                kind not in {"key", "value"}
                or (layer, kind) in used
                or not isinstance(shape, list)
                or len(shape) != 4
                or any(type(value) is not int or value < 1 for value in shape)
                or shape[0] != 1
                or shape[2] != windows[layer] - 1
                or state.get("maximum_sequence") != windows[layer] - 1
            ):
                raise TransformerClientError("semantic windowed state shape is invalid")
            used[(layer, kind)] = shape[1] * shape[2] * shape[3] * np.dtype(np.float32).itemsize
        if set(used) != {(layer, kind) for layer in windows for kind in ("key", "value")}:
            raise TransformerClientError("semantic windowed state lacks key/value capacity")
        total = sum(used.values())
        if total > semantic_state.MAX_WINDOW_STATE_BYTES:
            raise TransformerClientError("semantic windowed state exceeds client memory budget")
        return total

    def reset(self) -> None:
        super().reset()
        self._hybrid_states: dict[tuple[int, str], np.ndarray] = {}
        for layer, window in getattr(self, "_window_contracts", {}).items():
            self.caches[layer] = WindowedLayerCache(window=window)

    def prepare_ids(self, ids: list[int]) -> tuple[list[int], np.ndarray, list[LayerCache]]:
        if len(ids) == 0 and self.cfg.get("bos_token_policy") == "nonempty_only":
            raise TransformerClientError("model without a BOS token requires nonempty input")
        if not self._window_contracts:
            return super().prepare_ids(ids)
        if len(ids) == 0:
            ids = [int(self.cfg["bos_token_id"])]
        if len(ids) > int(self._graphs["prefill"]["query_sequence"]):
            raise TransformerClientError("semantic prefill exceeds its compiled query bound")
        self.reset()
        logits: np.ndarray | None = None
        for token in ids:
            logits = self._forward(np.asarray([token], dtype=np.int64), final_logits_only=True)[-1]
        if logits is None:
            raise TransformerClientError("semantic prefill produced no logits")
        return list(ids), logits, self.caches

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
    def _declared_state_kinds(graph: dict[str, Any]) -> dict[str, str]:
        """State ownership follows operator declarations, including suffix producers."""
        kinds: dict[str, str] = {}
        for operation in graph["operations"]:
            kind = operation.get("state_kind")
            if kind is None:
                continue
            if not (
                (kind in {"key", "value"} and operation["operator"] in {"kv_cache_append", "cache_suffix"})
                or (kind == "convolution" and operation["operator"] == "convolution_state_update")
                or (kind == "recurrent" and operation["operator"] == "gated_delta_state_update")
            ):
                raise TransformerClientError("semantic operation has an unsupported state kind")
            kinds[operation["id"]] = kind
        for state in graph["state_outputs"]:
            if kinds.get(state["id"]) != state["kind"]:
                raise TransformerClientError("semantic output state lacks declared ownership")
        return kinds

    @staticmethod
    def _numeric_output(operation: dict[str, Any], value: np.ndarray) -> np.ndarray:
        """Enforce an operator's declared representation after local or remote work."""
        attributes = operation["attributes"]
        factor = attributes.get("factor")
        dtype = (
            factor.get("output_dtype")
            if operation["operator"] in {"scale", "attention_scale"} and isinstance(factor, dict)
            else attributes.get("output_dtype")
        )
        if dtype is None:
            return value
        if dtype == "model_native":
            if np.asarray(value).dtype != np.float32:
                raise TransformerClientError("semantic operator has an unsupported numeric representation")
            return value
        if dtype != "bfloat16" or np.asarray(value).dtype != np.float32:
            raise TransformerClientError("semantic operator has an unsupported numeric representation")
        try:
            return round_bfloat16(value)
        except SemanticNumericError as exc:
            raise TransformerClientError(str(exc)) from exc

    @staticmethod
    def _rotary(value: np.ndarray, positions: np.ndarray, attributes: dict[str, Any]) -> np.ndarray:
        if "mrope_interleaved" in attributes or "mrope_section" in attributes:
            try:
                return float32_rotary_text_mrope(value, positions, attributes)
            except SemanticNumericError as exc:
                raise TransformerClientError(str(exc)) from exc
        if attributes.get("output_dtype") == "bfloat16":
            try:
                return bfloat16_rotary(value, positions, attributes)
            except SemanticNumericError as exc:
                raise TransformerClientError(str(exc)) from exc
        if "frequency_scaling" in attributes:
            try:
                scaling = attributes["frequency_scaling"]
                if isinstance(scaling, dict) and scaling.get("kind") == "per_frequency_context":
                    return float32_rotary_per_frequency(value, positions, attributes)
                return float32_rotary_wavelength(value, positions, attributes)
            except SemanticNumericError as exc:
                raise TransformerClientError(str(exc)) from exc
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
        pending_keys: dict[int, tuple[str, np.ndarray]],
    ) -> Any:
        kind = operation["operator"]
        inputs = operation["inputs"]
        attrs = operation["attributes"]
        layer = operation.get("layer")
        if kind == "state_initialize":
            shape = operation.get("output_shape")
            if (
                inputs != []
                or type(layer) is not int
                or attrs not in (
                    {"initial_value": 0, "dtype": "float32", "state_kind": "convolution"},
                    {"initial_value": 0, "dtype": "float32", "state_kind": "recurrent"},
                )
                or not isinstance(shape, list)
                or len(shape) not in (3, 4)
                or any(type(width) is not int for width in shape)
                or shape[0] != 1
                or (
                    attrs["state_kind"] == "convolution"
                    and (len(shape) != 3 or not 1 <= shape[1] <= 8192 or not 1 <= shape[2] <= 16)
                )
                or (
                    attrs["state_kind"] == "recurrent"
                    and (len(shape) != 4 or not 1 <= shape[1] <= 64 or not 1 <= shape[2] <= 256
                         or not 1 <= shape[3] <= 256 or shape[1] * shape[2] * shape[3] > 2 * 1024 * 1024)
                )
            ):
                raise TransformerClientError("semantic hybrid zero state declaration is invalid")
            return np.zeros(shape, dtype=np.float32)
        if kind in {"causal_convolution", "convolution_state_update"}:
            weight = attrs.get("weight")
            if not isinstance(weight, str) or not weight:
                raise TransformerClientError("semantic convolution has no bound weight")
            try:
                return execute_declared_causal_convolution(
                    operation, values, self._weight(weight)
                )
            except HybridStateError as exc:
                raise TransformerClientError(str(exc)) from exc
        if kind in {"gated_delta_rule", "gated_delta_state_update"}:
            try:
                return execute_declared_gated_delta(operation, values)
            except HybridStateError as exc:
                raise TransformerClientError(str(exc)) from exc
        if kind == "rms_norm_gated":
            weight = attrs.get("weight")
            if not isinstance(weight, str) or not weight:
                raise TransformerClientError("gated RMSNorm has no bound coefficient")
            try:
                return execute_declared_gated_rms_norm(operation, values, self._weight(weight))
            except HybridStateError as exc:
                raise TransformerClientError(str(exc)) from exc
        source = values[inputs[0]]
        if kind == "gated_delta_decay":
            if (
                set(attrs) != {"a_log", "dt_bias", "formula", "compute_dtype"}
                or not isinstance(attrs["a_log"], str)
                or not isinstance(attrs["dt_bias"], str)
                or not attrs["a_log"] or not attrs["dt_bias"]
                or attrs["formula"] != "-exp(A_log)*softplus(a+dt_bias)"
                or attrs["compute_dtype"] != "float32"
            ):
                raise TransformerClientError("gated-delta decay has no bound coefficient contract")
            source = np.asarray(source)
            declared = operation.get("output_shape")
            if (
                source.ndim == 2 and isinstance(declared, list) and len(declared) == 3
                and declared[0] == 1 and declared[2] == source.shape[1]
                and type(declared[1]) is int and 0 < source.shape[0] <= declared[1]
            ):
                source = source[None, :, :]
            try:
                return bounded_gated_delta_decay(
                    source, self._weight(attrs["a_log"]), self._weight(attrs["dt_bias"])
                )
            except HybridStateError as exc:
                raise TransformerClientError(str(exc)) from exc
        if kind == "sigmoid":
            source = np.asarray(source)
            declared = operation.get("output_shape")
            if (
                source.ndim == 2 and isinstance(declared, list) and len(declared) == 3
                and declared[0] == 1 and declared[2] == source.shape[1]
                and type(declared[1]) is int and 0 < source.shape[0] <= declared[1]
            ):
                source = source[None, :, :]
            if (
                attrs != {} or source.dtype != np.float32 or source.ndim != 3
                or source.shape[0] != 1 or not 1 <= source.shape[1] <= 256
                or source.size > 1 << 22 or not np.all(np.isfinite(source))
            ):
                raise TransformerClientError("semantic sigmoid numeric domain is invalid")
            output = np.empty_like(source)
            positive = source >= 0
            output[positive] = 1.0 / (1.0 + np.exp(-source[positive]))
            neg_exp = np.exp(source[~positive])
            output[~positive] = neg_exp / (1.0 + neg_exp)
            return output
        if kind == "scale" or (kind == "attention_scale" and "factor" in attrs):
            factor = attrs.get("factor")
            if not isinstance(factor, dict):
                raise TransformerClientError("semantic scale factor is missing")
            try:
                weight = (
                    self._weight(factor["weight"])
                    if factor.get("kind") == "checkpoint_scalar"
                    and isinstance(factor.get("weight"), str)
                    else None
                )
                return bfloat16_scale(source, factor, weight)
            except SemanticNumericError as exc:
                raise TransformerClientError(str(exc)) from exc
        if kind == "gelu_tanh":
            if attrs != {
                "approximation": "tanh",
                "compute_dtype": "bfloat16",
                "output_dtype": "bfloat16",
            }:
                raise TransformerClientError("semantic GELU numeric contract is unsupported")
            try:
                return bfloat16_gelu_tanh(source)
            except SemanticNumericError as exc:
                raise TransformerClientError(str(exc)) from exc
        if kind == "softcap":
            if (
                attrs.get("formula") != "cap*tanh(input/cap)"
                or attrs.get("compute_dtype") != "bfloat16"
                or attrs.get("output_dtype") != "bfloat16"
            ):
                raise TransformerClientError("semantic softcap numeric contract is unsupported")
            try:
                return bfloat16_softcap(source, attrs.get("cap"))
            except SemanticNumericError as exc:
                raise TransformerClientError(str(exc)) from exc
        if kind == "permute":
            order = attrs.get("permutation")
            if order != [0, 2, 1, 3] or source.ndim != 4:
                raise TransformerClientError("semantic permutation is unsupported")
            return source.transpose(order)
        if kind == "slice":
            axis, start, end = attrs.get("axis"), attrs.get("start"), attrs.get("end")
            squeezed = attrs.get("squeeze")
            if (
                set(attrs) != {"axis", "start", "end", "squeeze"}
                or type(axis) is not int
                or type(start) is not int
                or type(end) is not int
                or type(squeezed) is not bool
                or start < 0 or start >= end
                or source.ndim < 1
            ):
                raise TransformerClientError("semantic slice is unsupported")
            declared = operation["output_shape"]
            if source.ndim == 2 and len(declared) == 3 and declared[0] == 1 and not squeezed:
                if axis not in {-1, 2} or end > source.shape[1] or declared[2] != end - start:
                    raise TransformerClientError("semantic row-major slice shape is invalid")
                return source[:, start:end]
            index = axis % source.ndim
            if (
                axis < -source.ndim or axis >= source.ndim
                or end > source.shape[index]
                or (squeezed and end - start != 1)
            ):
                raise TransformerClientError("semantic slice exceeds its input")
            selection: list[int | slice] = [slice(None)] * source.ndim
            selection[index] = start if squeezed else slice(start, end)
            result = source[tuple(selection)]
            if len(declared) != result.ndim or any(
                (value != bound if dimension != 1 or result.ndim < 3 else not 0 < value <= bound)
                for dimension, (value, bound) in enumerate(zip(result.shape, declared, strict=True))
            ):
                raise TransformerClientError("semantic slice output shape is invalid")
            return result
        if kind == "reshape":
            layout = attrs.get("layout")
            if layout == "batch_heads_sequence_feature":
                if attrs.get("input_layout") == "batch_sequence_heads_feature":
                    shape = operation["output_shape"]
                    if (
                        source.ndim != 4 or source.shape[0] != 1
                        or source.shape[2] != shape[1] or source.shape[3] != shape[3]
                        or not 0 < source.shape[1] <= shape[2]
                    ):
                        raise TransformerClientError("semantic head permutation shape is invalid")
                    return source.transpose(0, 2, 1, 3)
                if attrs.get("input_layout") is not None:
                    raise TransformerClientError("semantic head input layout is unsupported")
                if source.ndim != 2:
                    raise TransformerClientError("semantic head reshape expects a row-major tensor")
                shape = operation["output_shape"]
                return source.reshape(1, source.shape[0], shape[1], shape[-1]).transpose(0, 2, 1, 3)
            if layout in {"batch_sequence_heads_feature", "batch_sequence_layer_feature"}:
                shape = operation["output_shape"]
                if (
                    len(shape) != 4
                    or shape[0] != 1
                    or source.ndim not in {2, 3}
                    or source.shape[-1] != shape[2] * shape[3]
                    or (source.ndim == 3 and source.shape[0] != 1)
                ):
                    raise TransformerClientError("semantic sequence reshape shape is invalid")
                return source.reshape(1, source.shape[-2], shape[2], shape[3])
            if layout == "batch_sequence_hidden":
                if source.ndim != 4 or source.shape[0] != 1:
                    raise TransformerClientError("semantic attention flatten shape is invalid")
                if attrs.get("input_layout") == "batch_sequence_heads_feature":
                    return source.reshape(source.shape[1], -1)
                if attrs.get("input_layout") not in {None, "batch_heads_sequence_feature"}:
                    raise TransformerClientError("semantic attention flatten layout is unsupported")
                return source.transpose(0, 2, 1, 3).reshape(source.shape[2], -1)
        elif kind == "rms_norm":
            if attrs.get("output_dtype") == "bfloat16" and attrs.get("compute_dtype") == "float32":
                declared_weight = attrs.get("weight")
                if attrs.get("with_scale") is not (declared_weight is not None):
                    raise TransformerClientError("semantic normalization weight contract is inconsistent")
                try:
                    return bfloat16_rms_norm(
                        source,
                        epsilon=attrs.get("epsilon"),
                        weight=self._weight(declared_weight) if declared_weight is not None else None,
                        offset=attrs.get("weight_offset"),
                    )
                except SemanticNumericError as exc:
                    raise TransformerClientError(str(exc)) from exc
            if attrs.get("output_dtype") is not None or attrs.get("compute_dtype") is not None:
                raise TransformerClientError("semantic RMSNorm dtype is unsupported")
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
            if attrs.get("mode") == "append":
                current_index = 1
            elif attrs.get("mode") == "initialize":
                current_index = 0
            else:
                raise TransformerClientError("semantic cache append domain is unsupported")
            tensor = values[inputs[current_index]][0].transpose(
                1, 0, 2
            )
            state_kind = state_kinds.get(operation["id"])
            if state_kind == "key":
                pending_keys[layer] = (operation["id"], tensor)
                if isinstance(cache, WindowedLayerCache):
                    return source
                prior = cache.key[: cache.length] if cache.key is not None else tensor[:0]
                return np.concatenate((prior, tensor), axis=0).transpose(1, 0, 2)[None]
            if state_kind == "value" and layer in pending_keys:
                key_id, key_tensor = pending_keys.pop(layer)
                if isinstance(cache, WindowedLayerCache):
                    try:
                        key_view, value_view = cache.append_windows(key_tensor, tensor)
                    except SemanticStateError as exc:
                        raise TransformerClientError(str(exc)) from exc
                    values[key_id] = key_view
                    return value_view
                key, value = cache.append(key_tensor, tensor)
                values[key_id] = key.transpose(1, 0, 2)[None]
                return value.transpose(1, 0, 2)[None]
        elif kind == "cache_suffix":
            if attrs.get("axis") == 2 and attrs.get("semantics") == "visible_valid_prefix":
                return source
            if (
                attrs.get("axis") != 3
                or attrs.get("output_axis") != 2
                or attrs.get("semantics") != "persist_last_valid_past_tokens"
                or type(layer) is not int
                or layer not in self._window_contracts
                or attrs.get("maximum_sequence") != self._window_contracts[layer] - 1
            ):
                raise TransformerClientError("semantic cache suffix is not yet bound")
            cache = self.caches[layer]
            if not isinstance(cache, WindowedLayerCache) or source.ndim != 5:
                raise TransformerClientError("semantic cache suffix lacks a windowed state view")
            stored = cache.key if state_kinds.get(operation["id"]) == "key" else cache.value
            if stored is None:
                raise TransformerClientError("semantic cache suffix lacks a retained prefix")
            return stored[: cache.length].transpose(1, 0, 2)[None]
        elif kind == "attention_scores":
            if "key_layout" in attrs:
                try:
                    return semantic_attention_scores(
                        source,
                        values[inputs[1]],
                        group=attrs.get("group_size"),
                        layout=attrs["key_layout"],
                    )
                except SemanticAttentionError as exc:
                    raise TransformerClientError(str(exc)) from exc
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
            try:
                return mask_causal_scores(
                    source,
                    values[inputs[1]],
                    attrs,
                    padding_mask=values[inputs[2]] if len(inputs) > 2 else None,
                    valid_lengths=values[inputs[3]] if len(inputs) > 3 else None,
                )
            except SemanticAttentionError as exc:
                raise TransformerClientError(str(exc)) from exc
        elif kind == "softmax":
            if attrs.get("output_dtype") == "bfloat16" and attrs.get("compute_dtype") == "float32":
                if attrs.get("axis") != -1:
                    raise TransformerClientError("semantic softmax axis is unsupported")
                try:
                    return bfloat16_softmax(source)
                except SemanticNumericError as exc:
                    raise TransformerClientError(str(exc)) from exc
            if attrs not in ({}, {"axis": -1}, {"axis": -1, "compute_dtype": "float32"}):
                raise TransformerClientError("semantic softmax dtype is unsupported")
            centered = source - source.max(axis=-1, keepdims=True)
            probabilities = np.exp(centered).astype(np.float32)
            return probabilities / probabilities.sum(axis=-1, keepdims=True)
        elif kind == "attention_values":
            if "value_layout" in attrs:
                try:
                    return semantic_attention_values(
                        source,
                        values[inputs[1]],
                        group=attrs.get("group_size"),
                        layout=attrs["value_layout"],
                    )
                except SemanticAttentionError as exc:
                    raise TransformerClientError(str(exc)) from exc
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
            other = values[inputs[1]]
            if (
                source.ndim == 2 and other.ndim == 3 and other.shape[0] == 1
                and source.shape == other.shape[1:]
                and isinstance(operation.get("output_shape"), list)
                and len(operation["output_shape"]) == 3
                and operation["output_shape"][0] == 1
                and operation["output_shape"][2] == source.shape[1]
                and type(operation["output_shape"][1]) is int
                and 0 < source.shape[0] <= operation["output_shape"][1]
            ):
                return source * other[0]
            return source * other
        elif kind == "last_token":
            if attrs.get("axis") == 1:
                if source.ndim not in {2, 3} or (source.ndim == 3 and source.shape[0] != 1):
                    raise TransformerClientError("semantic last-token layout is invalid")
                sequence = source.shape[0] if source.ndim == 2 else source.shape[1]
                if attrs.get("selection") == "last_valid" and len(inputs) == 2:
                    valid = np.asarray(values[inputs[1]])
                    if (
                        attrs.get("valid_lengths_input") != inputs[1]
                        or valid.shape != (1,)
                        or not np.issubdtype(valid.dtype, np.integer)
                        or int(valid[0]) != self.position + sequence
                    ):
                        raise TransformerClientError("semantic last-valid token state is invalid")
                elif (
                    attrs.get("selection") is not None
                    or attrs.get("valid_lengths_input") is not None
                    or len(inputs) != 1
                    or sequence != 1
                ):
                    raise TransformerClientError("semantic last-token state is invalid")
                return source[-1:] if source.ndim == 2 else source[:, -1, :]
            if attrs.get("axis") not in {None, 0}:
                raise TransformerClientError("semantic last-token axis is unsupported")
            return source[-1:]
        elif kind == "greedy_token_selection":
            return np.argmax(source, axis=-1).astype(np.int64)
        elif kind == "token_feedback":
            return source
        raise TransformerClientError(
            f"semantic operator {kind!r} ({operation.get('id')!r}, layout={attrs.get('layout')!r}) is not executable"
        )

    def _forward(self, ids: np.ndarray, *, final_logits_only: bool = False) -> np.ndarray:
        phase = "prefill" if self.position == 0 else "decode"
        graph = self._graphs[phase]
        if any(int(value) in self._text_only_tokens for value in ids):
            raise TransformerClientError("multimodal tokens require an unimplemented position policy")
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
            if state["kind"] in {"convolution", "recurrent"}:
                identity = (int(state["layer"]), state["kind"])
                retained = self._hybrid_states.get(identity)
                if (
                    retained is None or retained.dtype != np.float32
                    or retained.shape != self._hybrid_contracts.get(identity)
                    or not np.all(np.isfinite(retained))
                ):
                    raise TransformerClientError("semantic hybrid state is unavailable")
                values[state["id"]] = retained
                continue
            cache = self.caches[int(state["layer"])]
            stored = cache.key if state["kind"] == "key" else cache.value
            absolute_position = (
                cache.position if isinstance(cache, WindowedLayerCache) else cache.length
            )
            if (
                stored is None
                or absolute_position != self.position
                or cache.length > int(state["maximum_sequence"])
                or stored.dtype != np.float32
                or len(state["shape"]) != 4
                or stored.shape[0] < cache.length
                or stored.shape[1:] != (int(state["shape"][1]), int(state["shape"][3]))
                or not np.all(np.isfinite(stored[: cache.length]))
                or (
                    isinstance(cache, WindowedLayerCache)
                    and np.any((stored[: cache.length].view(np.uint32) & 0xFFFF) != 0)
                )
            ):
                raise TransformerClientError("semantic cache state is unavailable")
            values[state["id"]] = stored[: cache.length].transpose(1, 0, 2)[None]
        operations = {op["id"]: op for op in graph["operations"]}
        state_kinds = self._declared_state_kinds(graph)
        steps = self._schedule[phase]["steps"]
        remaining = Counter(input_id for step in steps for input_id in step["input_ids"])
        selections = [
            op for op in graph["operations"] if op["operator"] == "greedy_token_selection"
        ]
        if len(selections) != 1 or len(selections[0]["inputs"]) != 1:
            raise TransformerClientError("semantic plan has no unique logit selection source")
        logits_id = selections[0]["inputs"][0]
        pending_keys: dict[int, tuple[str, np.ndarray]] = {}
        produced_hybrid = {
            row["id"]: (int(row["layer"]), row["kind"])
            for row in graph["state_outputs"] if row["kind"] in {"convolution", "recurrent"}
        }
        pending_hybrid: dict[tuple[int, str], np.ndarray] = {}
        for step in steps:
            op_ids = step["operation_ids"]
            if step["executor"] == "client_local":
                operation = operations[op_ids[0]]
                values[op_ids[0]] = self._numeric_output(
                    operation, self._local(operation, values, state_kinds, pending_keys)
                )
            elif step["executor"] in {"remote_stage", "verified_remote_stage", "client_linear"}:
                stage_ids = {self._stages[f"{phase}:{op_id}"] for op_id in op_ids}
                if len(stage_ids) != 1:
                    raise TransformerClientError("semantic remote step has multiple stage bindings")
                stage_id = stage_ids.pop()
                if operations[op_ids[0]]["operator"] == "token_lookup":
                    primary, auxiliary = self._token_lookup(ids)
                    combined = (
                        primary if auxiliary is None else np.concatenate((primary, auxiliary), axis=-1)
                    )
                    if combined.shape[-1] != self.bundle.stages[stage_id].out_features:
                        raise TransformerClientError("semantic token lookup has an invalid width")
                    for op_id, row in zip(op_ids, step["outputs"], strict=True):
                        offset = int(row["stage_offset"])
                        width = int(row["stage_width"])
                        values[op_id] = combined[..., offset : offset + width]
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
                    if step["executor"] == "client_linear" and self.bundle.stages[stage_id].client_weight is not None:
                        output = self.bundle.local_linear(stage_id, input_value)
                    else:
                        output = self.remote(stage_id, input_value)
                    for op_id, row in zip(op_ids, step["outputs"], strict=True):
                        offset = int(row["stage_offset"])
                        width = int(row["stage_width"])
                        values[op_id] = output[..., offset : offset + width]
                for op_id in op_ids:
                    values[op_id] = self._numeric_output(operations[op_id], values[op_id])
            else:
                raise TransformerClientError("semantic runtime executor is unsupported")
            for op_id in op_ids:
                identity = produced_hybrid.get(op_id)
                if identity is None:
                    continue
                tensor = np.asarray(values[op_id])
                if (
                    identity in pending_hybrid
                    or tensor.dtype != np.float32
                    or tensor.shape != self._hybrid_contracts.get(identity)
                    or not np.all(np.isfinite(tensor))
                ):
                    raise TransformerClientError("semantic hybrid state output is malformed")
                pending_hybrid[identity] = np.ascontiguousarray(tensor.copy())
            for input_id in step["input_ids"]:
                remaining[input_id] -= 1
                if remaining[input_id] == 0 and input_id != logits_id:
                    values.pop(input_id, None)
        if pending_keys:
            raise TransformerClientError("semantic cache key has no matching value")
        if set(pending_hybrid) != set(self._hybrid_contracts):
            raise TransformerClientError("semantic hybrid state output is incomplete")
        self._hybrid_states = pending_hybrid
        self.position += ids.size
        return np.asarray(values[logits_id], dtype=np.float32)


__all__ = ["SemanticDecoderRuntime"]
