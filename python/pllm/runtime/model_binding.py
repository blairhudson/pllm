from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np

from pllm.modeling import ModelPlan, lower_model
from pllm.runtime.quantization import (
    choose_plain_modulus,
    choose_wire_bits,
    signed_dot_bound,
)
from pllm.runtime.transformer_client import ClientBundle, MaskedTransformerClientRuntime

if TYPE_CHECKING:
    from pllm.runtime.model_execution import CompiledRuntimeSession

BINDING_SCHEMA = "pllm.runtime_model_binding.v1"
BINDING_DOMAIN = b"pllm.runtime_model_binding.v1\0"
REQUIRED_MODEL_FAMILY = "qwen2"
REQUIRED_ADAPTER = "pllm.qwen2.v1"

REMOTE_OPERATORS = frozenset({"token_lookup", "linear", "output_head"})
LOCAL_OPERATORS = frozenset({
    "reshape",
    "rms_norm",
    "rotary_embedding",
    "kv_cache_append",
    "attention_scores",
    "attention_scale",
    "causal_mask",
    "softmax",
    "attention_values",
    "residual_add",
    "silu",
    "multiply",
    "last_token",
    "greedy_token_selection",
    "token_feedback",
    "cache_suffix",
})
BOUNDARY_STAGE_IDS = frozenset({"token_lookup", "lm_head"})


class RuntimeBindingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeStageBinding:
    stage_id: str
    role: str
    layer_index: int | None
    semantic_operations: tuple[str, ...]
    in_features: int
    out_features: int
    weight_bits: int
    activation_bits: int
    ring: str
    modulus: int
    wire_bits: int
    weight_digest: str
    weight_scales_digest: str
    bias_digest: str | None
    client_weight_layout: str | None = None
    client_weight_digest: str | None = None
    client_weight_scales_digest: str | None = None
    client_aux_weight_digest: str | None = None
    client_aux_scales_digest: str | None = None


class CompiledRuntimeModel:
    __slots__ = (
        "_plan",
        "_bundle",
        "_canonical",
        "_digest",
        "_fingerprint",
        "_local_operations",
        "_runtime_config_digest",
        "_runtime_schedule_digest",
        "_stages",
        "_tokenizer_digest",
    )

    def __init__(self) -> None:
        raise RuntimeBindingError(
            "CompiledRuntimeModel must be created by compile_runtime_model"
        )

    @classmethod
    def _create(
        cls,
        *,
        plan: ModelPlan,
        bundle: ClientBundle,
        canonical: bytes,
        digest: str,
        fingerprint: str,
        stages: tuple[RuntimeStageBinding, ...],
        local_operations: tuple[str, ...],
        runtime_config_digest: str,
        runtime_schedule_digest: str,
        tokenizer_digest: str,
    ) -> CompiledRuntimeModel:
        self = object.__new__(cls)
        self._plan = plan
        self._bundle = bundle
        self._canonical = canonical
        self._digest = digest
        self._fingerprint = fingerprint
        self._stages = stages
        self._local_operations = local_operations
        self._runtime_config_digest = runtime_config_digest
        self._runtime_schedule_digest = runtime_schedule_digest
        self._tokenizer_digest = tokenizer_digest
        return self

    @property
    def complete(self) -> bool:
        return True

    @property
    def completeness_scope(self) -> str:
        return "runtime_binding"

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def model_plan_digest(self) -> str:
        return self._plan.digest

    @property
    def bundle_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def runtime_config_digest(self) -> str:
        return self._runtime_config_digest

    @property
    def runtime_schedule_digest(self) -> str:
        return self._runtime_schedule_digest

    @property
    def tokenizer_digest(self) -> str:
        return self._tokenizer_digest

    @property
    def stage_bindings(self) -> tuple[RuntimeStageBinding, ...]:
        return self._stages

    @property
    def local_operations(self) -> tuple[str, ...]:
        return self._local_operations

    def canonical_bytes(self) -> bytes:
        return self._canonical

    def to_spec(self) -> dict[str, Any]:
        return json.loads(self._canonical)

    def validate(self) -> None:
        refreshed = compile_runtime_model(self._plan, self._bundle)
        if refreshed.digest != self._digest or refreshed.to_spec() != self.to_spec():
            raise RuntimeBindingError("bound plan or bundle changed since compilation")

    def runtime(
        self, remote: Callable[[str, np.ndarray], np.ndarray]
    ) -> MaskedTransformerClientRuntime:
        self.validate()
        return MaskedTransformerClientRuntime(self._bundle, remote)

    def session(
        self, remote: Callable[[str, np.ndarray], np.ndarray]
    ) -> "CompiledRuntimeSession":
        from pllm.runtime.model_execution import CompiledRuntimeSession

        return CompiledRuntimeSession._create(self, remote)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _binding_digest(payload: bytes) -> str:
    return _sha256(BINDING_DOMAIN + payload)


def _f32_bytes(value: np.ndarray) -> bytes:
    return np.asarray(value, dtype="<f4").tobytes()


def _normalize_weight_key(key: Any) -> str:
    if not isinstance(key, str):
        raise RuntimeBindingError("stage weight key must be a string")
    return key[len("model."):] if key.startswith("model.") else key


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeBindingError(f"{name} must be an integer")
    return int(value)


def _last_dim(shape: Any, name: str) -> int:
    if not isinstance(shape, (list, tuple)) or not shape:
        raise RuntimeBindingError(f"{name} must have a nonempty output shape")
    return _require_int(shape[-1], name)


def _phase_operations(document: dict[str, Any], phase: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    graph = document.get(phase)
    if not isinstance(graph, dict):
        raise RuntimeBindingError(f"plan is missing the {phase} phase")
    operations = graph.get("operations")
    if not isinstance(operations, list):
        raise RuntimeBindingError(f"plan {phase} operations must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for operation in operations:
        if not isinstance(operation, dict):
            raise RuntimeBindingError(f"plan {phase} operation must be a mapping")
        operation_id = operation.get("id")
        operator = operation.get("operator")
        if not isinstance(operation_id, str) or not isinstance(operator, str):
            raise RuntimeBindingError(f"plan {phase} operation requires id and operator")
        if operation_id in by_id:
            raise RuntimeBindingError(f"duplicate {phase} operation {operation_id!r}")
        by_id[operation_id] = operation
    return graph, by_id


def _plan_dimensions(
    document: dict[str, Any],
    phases: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]],
) -> dict[str, int]:
    prefill = phases["prefill"][1]
    layers = sorted(
        {
            operation["layer"]
            for operation in prefill.values()
            if isinstance(operation.get("layer"), int) and not isinstance(operation["layer"], bool)
        }
    )
    if not layers or layers != list(range(len(layers))):
        raise RuntimeBindingError("plan layers must be contiguous from zero")
    token_ops = [op for op in prefill.values() if op["operator"] == "token_lookup"]
    head_ops = [op for op in prefill.values() if op["operator"] == "output_head"]
    if len(token_ops) != 1 or len(head_ops) != 1:
        raise RuntimeBindingError("plan requires exactly one token lookup and output head")
    hidden = _last_dim(token_ops[0].get("output_shape"), "token_lookup")
    vocab = _last_dim(head_ops[0].get("output_shape"), "output_head")
    def producers(operation: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            prefill[input_id]
            for input_id in operation.get("inputs") or []
            if input_id in prefill
        ]

    def hop(source: dict[str, Any], operator: str, layer: int, name: str) -> dict[str, Any]:
        found = [op for op in producers(source) if op["operator"] == operator]
        if len(found) != 1 or found[0].get("layer") != layer:
            raise RuntimeBindingError(f"plan layer {layer} requires exactly one {name}")
        return found[0]

    def head_shape(operation: dict[str, Any], name: str) -> list[int]:
        shape = operation.get("output_shape")
        if not isinstance(shape, list) or len(shape) != 4:
            raise RuntimeBindingError(f"plan {name} reshape is malformed")
        return shape

    heads = kv_heads = head_dim = intermediate = 0
    for layer in layers:
        layer_ops = [op for op in prefill.values() if op.get("layer") == layer]

        def only(operator: str) -> dict[str, Any]:
            found = [op for op in layer_ops if op["operator"] == operator]
            if len(found) != 1:
                raise RuntimeBindingError(
                    f"plan layer {layer} requires exactly one {operator} operation"
                )
            return found[0]

        scores = only("attention_scores")
        rope_q = hop(scores, "rotary_embedding", layer, "query rotary input")
        q_heads = hop(rope_q, "reshape", layer, "query head reshape")
        q_linear = hop(q_heads, "linear", layer, "query projection")
        key_view = hop(scores, "cache_suffix", layer, "key cache view")
        key_append = hop(key_view, "kv_cache_append", layer, "key cache append")
        if key_append.get("state_kind") != "key":
            raise RuntimeBindingError(f"plan layer {layer} key append is not a key state")
        rope_k = hop(key_append, "rotary_embedding", layer, "key rotary input")
        k_heads = hop(rope_k, "reshape", layer, "key head reshape")
        k_linear = hop(k_heads, "linear", layer, "key projection")
        values = only("attention_values")
        value_view = hop(values, "cache_suffix", layer, "value cache view")
        value_append = hop(value_view, "kv_cache_append", layer, "value cache append")
        if value_append.get("state_kind") != "value":
            raise RuntimeBindingError(f"plan layer {layer} value append is not a value state")
        v_heads = hop(value_append, "reshape", layer, "value head reshape")
        v_linear = hop(v_heads, "linear", layer, "value projection")
        silu = only("silu")
        gate = hop(silu, "linear", layer, "gate projection")
        multiply = only("multiply")
        multiply_producers = producers(multiply)
        multiply_silu = [op for op in multiply_producers if op["operator"] == "silu"]
        multiply_linear = [op for op in multiply_producers if op["operator"] == "linear"]
        if (
            len(multiply_silu) != 1
            or multiply_silu[0]["id"] != silu["id"]
            or len(multiply_linear) != 1
            or multiply_linear[0].get("layer") != layer
        ):
            raise RuntimeBindingError(f"plan layer {layer} gated multiply is malformed")
        up = multiply_linear[0]
        norm_inputs = {
            op["id"]
            for source in (q_linear, k_linear, v_linear)
            for op in producers(source)
        }
        if len(norm_inputs) != 1:
            raise RuntimeBindingError(
                f"plan layer {layer} q/k/v projections must share one normalized input"
            )
        shared_norm = prefill[norm_inputs.pop()]
        if shared_norm["operator"] != "rms_norm":
            raise RuntimeBindingError(
                f"plan layer {layer} projections must consume a single rms norm"
            )
        if _last_dim(shared_norm.get("output_shape"), shared_norm["id"]) != hidden:
            raise RuntimeBindingError(f"plan layer {layer} norm width does not match hidden")

        q_shape = head_shape(q_heads, "query head")
        k_shape = head_shape(k_heads, "key head")
        v_shape = head_shape(v_heads, "value head")
        if q_shape[-1] != k_shape[-1] or k_shape[-1] != v_shape[-1]:
            raise RuntimeBindingError(f"plan layer {layer} head dims differ")
        if k_shape[1] != v_shape[1]:
            raise RuntimeBindingError(f"plan layer {layer} kv head counts differ")
        if heads and (heads != q_shape[1] or kv_heads != k_shape[1] or head_dim != q_shape[-1]):
            raise RuntimeBindingError("plan attention geometry differs across layers")
        heads, kv_heads, head_dim = q_shape[1], k_shape[1], q_shape[-1]
        if heads * head_dim != _last_dim(q_linear["output_shape"], "query projection"):
            raise RuntimeBindingError("plan query projection width does not match heads")
        if kv_heads * head_dim != _last_dim(k_linear["output_shape"], "key projection"):
            raise RuntimeBindingError("plan key projection width does not match heads")
        if kv_heads * head_dim != _last_dim(v_linear["output_shape"], "value projection"):
            raise RuntimeBindingError("plan value projection width does not match heads")
        width = _last_dim(gate["output_shape"], "gate projection")
        if width != _last_dim(up["output_shape"], "up projection"):
            raise RuntimeBindingError("plan gate and up widths differ")
        if intermediate and intermediate != width:
            raise RuntimeBindingError("plan intermediate width differs across layers")
        intermediate = width
    return {
        "layers": len(layers),
        "hidden": hidden,
        "vocab": vocab,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate": intermediate,
    }


def _resolve_array(arrays: dict[str, np.ndarray], weight_id: str) -> tuple[str, np.ndarray]:
    if weight_id in arrays:
        return weight_id, arrays[weight_id]
    marker = "." + weight_id
    matches = [(key, value) for key, value in arrays.items() if key.endswith(marker)]
    if not matches:
        raise RuntimeBindingError(f"missing client tensor for {weight_id!r}")
    if len(matches) > 1:
        raise RuntimeBindingError(f"ambiguous client tensor suffix for {weight_id!r}")
    return matches[0]


def _canonical_stage_fingerprints(stages: dict[str, Any]) -> tuple[str, str]:
    body = []
    commitment = []
    for stage_id, stage in sorted(stages.items()):
        if stage_id in BOUNDARY_STAGE_IDS:
            continue
        scales = stage.weight_scales.astype("<f4", copy=False).tobytes()
        bias = None if stage.bias is None else stage.bias.astype("<f4", copy=False).tobytes()
        body.append({
            "id": stage_id,
            "op": stage.op,
            "in": stage.in_features,
            "out": stage.out_features,
            "weight_bits": stage.weight_bits,
            "activation_bits": stage.activation_bits,
            "weight_digest": stage.weight_digest,
            "weight_scales": scales,
            "bias": bias,
        })
        commitment.append({
            "id": stage_id,
            "weight": stage.weight_digest,
            "in": stage.in_features,
            "out": stage.out_features,
            "wb": stage.weight_bits,
            "ab": stage.activation_bits,
            "profile": stage.seeded_profile.to_dict(),
        })
    return (
        _sha256(msgpack.packb(body, use_bin_type=True)),
        _sha256(msgpack.packb(commitment, use_bin_type=True)),
    )


def _runtime_config(cfg: dict[str, Any]) -> dict[str, Any]:
    hidden = _require_int(cfg.get("hidden_size"), "config hidden_size")
    layers = _require_int(cfg.get("num_hidden_layers"), "config num_hidden_layers")
    heads = _require_int(cfg.get("num_attention_heads"), "config num_attention_heads")
    kv_heads = _require_int(
        cfg.get("num_key_value_heads", heads), "config num_key_value_heads"
    )
    head_dim = _require_int(cfg.get("head_dim", hidden // heads), "config head_dim")
    eps = cfg.get("rms_norm_eps", 1e-6)
    norm_offset = cfg.get("norm_offset", 0.0)
    multiplier = cfg.get("embedding_multiplier", math.sqrt(hidden))
    theta = cfg.get("rope_theta", 10000.0)
    for name, value in (
        ("rms_norm_eps", eps),
        ("norm_offset", norm_offset),
        ("embedding_multiplier", multiplier),
        ("rope_theta", theta),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise RuntimeBindingError(f"config {name} must be a finite number")
    layer_types = cfg.get("layer_types") or ["full_attention"] * layers
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != layers
        or not all(isinstance(item, str) for item in layer_types)
    ):
        raise RuntimeBindingError("config layer_types must be a per-layer string list")
    sliding_window = cfg.get("sliding_window")
    if sliding_window is not None and _require_int(
        sliding_window, "config sliding_window"
    ) <= 0:
        raise RuntimeBindingError("config sliding_window must be positive when present")
    shared_count = _require_int(
        cfg.get("num_kv_shared_layers", 0) or 0, "config num_kv_shared_layers"
    )
    k_eq_v = bool(cfg.get("attention_k_eq_v", False))
    ple_dim = _require_int(
        cfg.get("hidden_size_per_layer_input", 0) or 0,
        "config hidden_size_per_layer_input",
    )
    block_style = str(cfg.get("block_style", "gemma4"))
    qk_norm = bool(cfg.get("qk_norm", True))
    v_norm = bool(cfg.get("v_norm", False))
    token_lookup_batch = max(1, _require_int(
        cfg.get("token_lookup_batch", 16), "config token_lookup_batch"
    ))
    output_multiplier = cfg.get("output_multiplier")
    softcap = cfg.get("final_logit_softcapping")
    attention_scaling = cfg.get("attention_scaling")
    rope_parameters = cfg.get("rope_parameters") or {}
    per_layer = cfg.get("per_layer_config") or {}
    hidden_activation = str(cfg.get("hidden_activation", "silu"))
    model_family = str(cfg.get("model_family", ""))
    rope_scaling = cfg.get("rope_scaling")
    if rope_scaling in ({}, None):
        rope_scaling = None
    elif rope_scaling in ({"type": "default"}, {"rope_type": "default"}):
        rope_scaling = None
    else:
        raise RuntimeBindingError("base qwen2 runtime profile forbids rope scaling")
    use_sliding_window = bool(cfg.get("use_sliding_window", False))
    if use_sliding_window:
        raise RuntimeBindingError("base qwen2 runtime profile forbids sliding windows")
    sliding_window = None
    attention_bias = bool(cfg.get("attention_bias", True))
    if not attention_bias:
        raise RuntimeBindingError("base qwen2 runtime profile requires attention bias")
    if block_style != "llama":
        raise RuntimeBindingError("base qwen2 runtime profile requires llama block style")
    if model_family != "llama-compatible":
        raise RuntimeBindingError("base qwen2 runtime profile requires llama-compatible family")
    if float(norm_offset) != 0.0:
        raise RuntimeBindingError("base qwen2 runtime profile requires zero norm offset")
    if float(multiplier) != 1.0:
        raise RuntimeBindingError("base qwen2 runtime profile requires unit embedding scale")
    if qk_norm or v_norm:
        raise RuntimeBindingError("base qwen2 runtime profile forbids qk/v norms")
    if layer_types != ["full_attention"] * layers:
        raise RuntimeBindingError("base qwen2 runtime profile requires full attention layers")
    if shared_count != 0:
        raise RuntimeBindingError("base qwen2 runtime profile forbids shared kv layers")
    if k_eq_v:
        raise RuntimeBindingError("base qwen2 runtime profile forbids shared k/v weights")
    if ple_dim != 0:
        raise RuntimeBindingError("base qwen2 runtime profile forbids per-layer inputs")
    if output_multiplier is not None or softcap is not None:
        raise RuntimeBindingError("base qwen2 runtime profile forbids logit transforms")
    if attention_scaling is not None:
        raise RuntimeBindingError("base qwen2 runtime profile forbids attention scaling")
    if not isinstance(rope_parameters, dict) or rope_parameters:
        raise RuntimeBindingError("base qwen2 runtime profile forbids rope parameters")
    if not isinstance(per_layer, dict) or per_layer:
        raise RuntimeBindingError("base qwen2 runtime profile forbids per-layer config")
    if hidden_activation != "silu":
        raise RuntimeBindingError("base qwen2 runtime profile requires silu activation")
    if token_lookup_batch <= 0:
        raise RuntimeBindingError("config token_lookup_batch must be positive")
    return {
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rms_norm_eps": float(eps),
        "norm_offset": float(norm_offset),
        "layer_types": list(layer_types),
        "sliding_window": sliding_window,
        "num_kv_shared_layers": shared_count,
        "attention_k_eq_v": k_eq_v,
        "hidden_size_per_layer_input": ple_dim,
        "embedding_multiplier": float(multiplier),
        "block_style": block_style,
        "qk_norm": qk_norm,
        "v_norm": v_norm,
        "token_lookup_batch": token_lookup_batch,
        "output_multiplier": output_multiplier,
        "final_logit_softcapping": softcap,
        "attention_scaling": attention_scaling,
        "rope_parameters": rope_parameters,
        "rope_theta": float(theta),
        "rope_scaling": rope_scaling,
        "use_sliding_window": use_sliding_window,
        "attention_bias": attention_bias,
        "per_layer_config": per_layer,
        "hidden_activation": hidden_activation,
        "bos_token_id": _require_int(cfg.get("bos_token_id"), "config bos_token_id"),
        "eos_token_id": _require_int(cfg.get("eos_token_id"), "config eos_token_id"),
        "vocab_size": _require_int(cfg.get("vocab_size"), "config vocab_size"),
        "model_type": str(cfg.get("model_type", "")),
        "model_family": model_family,
    }


def _canonicalize_descriptor(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeBindingError("tokenizer descriptor has a non-finite number")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return {"$bytes_sha256": _sha256(data), "length": len(data)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_descriptor(item) for item in value]
    if isinstance(value, dict):
        canonical = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeBindingError("tokenizer descriptor keys must be strings")
            canonical[key] = _canonicalize_descriptor(item)
        return canonical
    raise RuntimeBindingError(
        f"tokenizer descriptor value {type(value).__name__!r} is unsupported"
    )


def _tokenizer_digest(descriptor: Any, runtime_config: dict[str, Any]) -> str:
    if not isinstance(descriptor, dict):
        raise RuntimeBindingError("client bundle tokenizer descriptor must be a mapping")
    kind = str(descriptor.get("kind", descriptor.get("type", "")))
    if kind not in {"byte", "sentencepiece", "tokenizer_json"}:
        raise RuntimeBindingError(f"unsupported tokenizer type {kind!r}")
    vocab = descriptor.get("vocab_size")
    if vocab is not None and _require_int(vocab, "tokenizer vocab_size") != runtime_config[
        "vocab_size"
    ]:
        raise RuntimeBindingError("tokenizer vocabulary does not match the model config")
    expected = {
        "bos_token_id": int(runtime_config["bos_token_id"] or 2),
        "eos_token_id": int(runtime_config["eos_token_id"] or 1),
    }
    for key, wanted in expected.items():
        present = descriptor.get(key)
        if present is not None and _require_int(present, f"tokenizer {key}") != wanted:
            raise RuntimeBindingError(f"tokenizer {key} does not match the model config")
    return _sha256(_canonical_json(_canonicalize_descriptor(descriptor)))


def compile_runtime_model(plan: ModelPlan, bundle: ClientBundle) -> CompiledRuntimeModel:
    if type(plan) is not ModelPlan:
        raise RuntimeBindingError("plan must be a ModelPlan")
    if type(bundle) is not ClientBundle:
        raise RuntimeBindingError("bundle must be a ClientBundle")
    try:
        plan.coverage()
    except Exception as exc:
        raise RuntimeBindingError("native model plan validation failed") from exc
    document = plan.to_dict()
    if document.get("model_family") != REQUIRED_MODEL_FAMILY:
        raise RuntimeBindingError("model plan family is not qwen2")
    if document.get("adapter") != REQUIRED_ADAPTER:
        raise RuntimeBindingError("model plan adapter is not pllm.qwen2.v1")
    if document.get("transformations"):
        raise RuntimeBindingError("transformed model plans cannot be bound")

    manifest = bundle.manifest if isinstance(bundle.manifest, dict) else {}
    cfg = bundle.cfg if isinstance(bundle.cfg, dict) else {}
    if str(cfg.get("model_type", "")).lower() != REQUIRED_MODEL_FAMILY:
        raise RuntimeBindingError("client bundle config is not a qwen2 model")
    if REQUIRED_MODEL_FAMILY not in str(manifest.get("architecture", "")).lower():
        raise RuntimeBindingError("client bundle manifest is not a qwen2 model")
    if manifest.get("id") != bundle.model_id:
        raise RuntimeBindingError("client bundle model id does not match its manifest")
    official_fingerprint = manifest.get("fingerprint")
    if official_fingerprint is not None:
        if not isinstance(official_fingerprint, str):
            raise RuntimeBindingError("manifest fingerprint must be a string")
        official_payload = {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "fingerprint"}
        }
        recomputed = _sha256(
            json.dumps(official_payload, sort_keys=True, separators=(",", ":")).encode()
        )
        if recomputed != official_fingerprint:
            raise RuntimeBindingError("manifest fingerprint does not match its content")

    prefill_graph = document.get("prefill") or {}
    decode_graph = document.get("decode") or {}
    batch = _require_int(prefill_graph.get("batch"), "prefill batch")
    max_input_tokens = _require_int(
        prefill_graph.get("query_sequence"), "prefill query_sequence"
    )
    decode_key_sequence = _require_int(
        decode_graph.get("maximum_key_sequence"), "decode maximum_key_sequence"
    )
    max_new_tokens = decode_key_sequence - max_input_tokens + 1
    if batch <= 0 or max_input_tokens <= 0 or max_new_tokens <= 0:
        raise RuntimeBindingError("model plan workload bounds must be positive")
    if _require_int(
        decode_graph.get("query_sequence"), "decode query_sequence"
    ) != 1:
        raise RuntimeBindingError("decode query_sequence must be one")
    try:
        reconstructed = lower_model(
            cfg,
            batch=batch,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        raise RuntimeBindingError(
            "client bundle config cannot reproduce the model plan"
        ) from exc
    if reconstructed.digest != plan.digest:
        raise RuntimeBindingError("client bundle config does not reproduce the model plan")
    if reconstructed.to_dict().get("config_digest") != document.get("config_digest"):
        raise RuntimeBindingError("client bundle config digest does not match the plan")
    try:
        runtime_schedule = plan.runtime_schedule("baseline.masked_linear_cpu")
    except Exception as exc:
        raise RuntimeBindingError("native whole-decoder runtime scheduling failed") from exc
    runtime_schedule_spec = runtime_schedule.to_dict()
    if (
        not runtime_schedule.complete
        or runtime_schedule.protected_execution
        or runtime_schedule_spec.get("model_plan_digest") != plan.digest
        or runtime_schedule_spec.get("model_config_digest") != document.get("config_digest")
    ):
        raise RuntimeBindingError("native whole-decoder runtime schedule is inconsistent")
    runtime_schedule_digest = runtime_schedule.digest

    runtime_config = _runtime_config(cfg)
    runtime_config_digest = _sha256(_canonical_json(runtime_config))
    tokenizer_digest = _tokenizer_digest(bundle.tokenizer_descriptor, runtime_config)

    phases = {
        phase: _phase_operations(document, phase) for phase in ("prefill", "decode")
    }
    prefill_ops, decode_ops = phases["prefill"][1], phases["decode"][1]
    if set(prefill_ops) != set(decode_ops):
        raise RuntimeBindingError("prefill and decode operations differ")
    if any(prefill_ops[op_id]["operator"] != decode_ops[op_id]["operator"] for op_id in prefill_ops):
        raise RuntimeBindingError("prefill and decode operators differ")

    dims = _plan_dimensions(document, phases)
    manifest_dimensions = {
        "hidden_size": dims["hidden"],
        "intermediate_size": dims["intermediate"],
        "num_hidden_layers": dims["layers"],
        "num_attention_heads": dims["heads"],
        "num_key_value_heads": dims["kv_heads"],
        "head_dim": dims["head_dim"],
        "vocab_size": dims["vocab"],
    }
    for key, expected in manifest_dimensions.items():
        manifest_value = _require_int(manifest.get(key), f"manifest {key}")
        config_source = cfg.get(key)
        if key == "head_dim" and config_source is None:
            config_source = _require_int(cfg.get("hidden_size"), "config hidden_size") // _require_int(
                cfg.get("num_attention_heads"), "config num_attention_heads"
            )
        config_value = _require_int(config_source, f"config {key}")
        if manifest_value != expected:
            raise RuntimeBindingError(f"bundle manifest {key} does not match the model plan")
        if config_value != expected:
            raise RuntimeBindingError(f"bundle config {key} does not match the model plan")
        if config_value != manifest_value:
            raise RuntimeBindingError(f"bundle config and manifest {key} disagree")
    if _require_int(
        cfg.get("max_position_embeddings"), "config max_position_embeddings"
    ) != _require_int(manifest.get("context_length"), "manifest context_length"):
        raise RuntimeBindingError("bundle config and manifest context disagree")

    bound = 0
    for graph, _ in phases.values():
        bound = max(bound, _require_int(graph.get("maximum_key_sequence"), "maximum_key_sequence"))
        for state in list(graph.get("state_inputs") or []) + list(graph.get("state_outputs") or []):
            if isinstance(state, dict) and state.get("maximum_sequence") is not None:
                bound = max(bound, _require_int(state["maximum_sequence"], "maximum_sequence"))
    max_positions = _require_int(
        cfg.get("max_position_embeddings") or manifest.get("context_length"),
        "max_position_embeddings",
    )
    if bound > max_positions:
        raise RuntimeBindingError("model plan position bounds exceed the bundle context")

    privacy = bundle.privacy if isinstance(bundle.privacy, dict) else {}
    if privacy.get("mode") != "public":
        raise RuntimeBindingError("client bundle privacy mode must be public")
    if not str(privacy.get("protocol", "")).startswith("masked_w"):
        raise RuntimeBindingError("client bundle protocol must be a masked weight protocol")
    if not privacy.get("preprocessed"):
        raise RuntimeBindingError("client bundle must be preprocessed")
    if not privacy.get("client_intermediate_activations"):
        raise RuntimeBindingError("client bundle must keep intermediate activations local")
    body_fingerprint = privacy.get("body_fingerprint")
    stage_commitment = privacy.get("stage_commitment")
    if not isinstance(body_fingerprint, str) or not body_fingerprint:
        raise RuntimeBindingError("client bundle is missing a body fingerprint")
    if not isinstance(stage_commitment, str) or not stage_commitment:
        raise RuntimeBindingError("client bundle is missing a stage commitment")
    weight_bits = _require_int(privacy.get("weight_bits"), "privacy weight_bits")
    activation_bits = _require_int(privacy.get("activation_bits"), "privacy activation_bits")
    if not (2 <= weight_bits <= 8 and 2 <= activation_bits <= 8):
        raise RuntimeBindingError("client bundle bit widths must be in [2, 8]")

    canonical: dict[str, Any] = {}
    for key, stage in bundle.stages.items():
        if key != stage.id:
            continue
        if stage.id in canonical:
            raise RuntimeBindingError(f"duplicate bundle stage id {stage.id!r}")
        canonical[stage.id] = stage
    if len({(stage.role, stage.layer_index) for stage in canonical.values()}) != len(canonical):
        raise RuntimeBindingError("bundle stage roles and layers must be unique")

    spec_rows: dict[str, dict[str, Any]] = {}
    for row in manifest.get("stages") or []:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise RuntimeBindingError("bundle manifest stage rows must carry string ids")
        if row["id"] in spec_rows:
            raise RuntimeBindingError(f"duplicate manifest stage row {row['id']!r}")
        spec_rows[row["id"]] = row
    if set(spec_rows) != set(canonical):
        raise RuntimeBindingError("manifest stage rows must match the canonical bundle stages")
    for stage_id, stage in canonical.items():
        row = spec_rows[stage_id]
        if row.get("role") != stage.role or row.get("layer_index") != stage.layer_index:
            raise RuntimeBindingError(f"bundle stage {stage_id!r} role or layer drifted")
        row_keys = {_normalize_weight_key(key) for key in row.get("weight_keys") or ()}
        row_fused = list(row.get("fused_from") or [])
        row_op = row.get("op")
        layered = stage.layer_index is not None
        if stage.role == "token_lookup":
            valid = (
                row_op == "embedding"
                and not row_fused
                and stage.layer_index is None
                and row_keys == {"embed_tokens.weight"}
            )
        elif stage.role == "lm_head":
            valid = (
                row_op == "lm_head"
                and not row_fused
                and stage.layer_index is None
                and bool(row_keys)
                and row_keys <= {"embed_tokens.weight", "lm_head.weight"}
            )
        elif stage.role == "qkv_projection":
            valid = (
                row_op == "linear"
                and row_fused == ["q_proj", "k_proj", "v_proj"]
                and len(row_keys) == 3
                and layered
            )
        elif stage.role == "attention_output":
            valid = row_op == "linear" and not row_fused and len(row_keys) == 1 and layered
        elif stage.role == "mlp_gate_up":
            valid = (
                row_op == "linear"
                and row_fused == ["gate_proj", "up_proj"]
                and len(row_keys) == 2
                and layered
            )
        elif stage.role == "mlp_down":
            valid = row_op == "linear" and not row_fused and len(row_keys) == 1 and layered
        else:
            valid = False
        if not valid:
            raise RuntimeBindingError(f"bundle stage {stage_id!r} spec row is inconsistent")

    expected_roles = {("token_lookup", None), ("lm_head", None)}
    for layer in range(dims["layers"]):
        expected_roles |= {
            ("qkv_projection", layer),
            ("attention_output", layer),
            ("mlp_gate_up", layer),
            ("mlp_down", layer),
        }
    if {(stage.role, stage.layer_index) for stage in canonical.values()} != expected_roles:
        raise RuntimeBindingError("bundle stage role set does not match the qwen2 surface")

    manifest_metadata = manifest.get("metadata")
    if not isinstance(manifest_metadata, dict):
        raise RuntimeBindingError("bundle manifest is missing metadata")
    stage_specific_moduli = manifest_metadata.get("stage_specific_moduli")
    if not isinstance(stage_specific_moduli, bool):
        raise RuntimeBindingError("bundle manifest must declare a modulus policy")
    plain_moduli = manifest_metadata.get("plain_moduli")
    if not isinstance(plain_moduli, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 2
        for value in plain_moduli
    ):
        raise RuntimeBindingError("bundle manifest plain_moduli must be a list of integers")
    advertised_moduli = set(plain_moduli)

    hidden, intermediate, vocab = dims["hidden"], dims["intermediate"], dims["vocab"]
    q_width = dims["heads"] * dims["head_dim"]
    kv_width = dims["kv_heads"] * dims["head_dim"]
    expected_dims = {
        "token_lookup": (vocab, hidden),
        "lm_head": (hidden, vocab),
        "qkv_projection": (hidden, q_width + 2 * kv_width),
        "attention_output": (q_width, hidden),
        "mlp_gate_up": (hidden, 2 * intermediate),
        "mlp_down": (intermediate, hidden),
    }

    ownership: dict[str, str | None] = {}
    for stage_id, stage in canonical.items():
        row = spec_rows[stage_id]
        for key in set(row.get("weight_keys") or ()):
            normalized = _normalize_weight_key(key)
            if normalized in ownership:
                ownership[normalized] = None
            else:
                ownership[normalized] = stage_id

    for stage in canonical.values():
        expected = expected_dims.get(str(stage.role))
        if expected is None:
            raise RuntimeBindingError(f"unsupported bundle stage role {stage.role!r}")
        if (stage.in_features, stage.out_features) != expected:
            raise RuntimeBindingError(f"bundle stage {stage.id!r} dimensions are inconsistent")
        if stage.weight_bits != weight_bits or stage.activation_bits != activation_bits:
            raise RuntimeBindingError(f"bundle stage {stage.id!r} bit widths are inconsistent")
        if stage.ring != "prime":
            raise RuntimeBindingError(f"bundle stage {stage.id!r} must use the prime ring")
        modulus = _require_int(stage.modulus, f"stage {stage.id} modulus")
        if modulus <= 2 or choose_wire_bits(modulus) != _require_int(
            stage.wire_bits, f"stage {stage.id} wire_bits"
        ):
            raise RuntimeBindingError(f"bundle stage {stage.id!r} modulus and wire bits disagree")
        if modulus not in advertised_moduli:
            raise RuntimeBindingError(f"bundle stage {stage.id!r} modulus is not advertised")
        features = 1 if stage.op == "embedding" else stage.in_features
        if stage_specific_moduli:
            expected_modulus = choose_plain_modulus(
                features,
                weight_bits=stage.weight_bits,
                activation_bits=stage.activation_bits,
            )
            if modulus != expected_modulus:
                raise RuntimeBindingError(
                    f"bundle stage {stage.id!r} modulus does not match its bound"
                )
        elif modulus <= signed_dot_bound(
            features, stage.weight_bits, stage.activation_bits
        ) * 2 + 1:
            raise RuntimeBindingError(
                f"bundle stage {stage.id!r} modulus does not cover its bound"
            )
        if not isinstance(stage.weight_digest, str) or not stage.weight_digest:
            raise RuntimeBindingError(f"bundle stage {stage.id!r} is missing a weight digest")
        scales = np.asarray(stage.weight_scales)
        if scales.dtype != np.float32 or scales.shape != (stage.out_features,):
            raise RuntimeBindingError(f"bundle stage {stage.id!r} weight scales are malformed")
        if not np.all(np.isfinite(scales)) or not np.all(scales > 0):
            raise RuntimeBindingError(f"bundle stage {stage.id!r} weight scales must be positive")
        if stage.bias is not None:
            bias = np.asarray(stage.bias)
            if bias.dtype != np.float32 or bias.shape != (stage.out_features,):
                raise RuntimeBindingError(f"bundle stage {stage.id!r} bias is malformed")
            if not np.all(np.isfinite(bias)):
                raise RuntimeBindingError(f"bundle stage {stage.id!r} bias must be finite")
        if stage.id in BOUNDARY_STAGE_IDS:
            continue
        if (
            stage.client_weight is not None
            or stage.client_weight_scales is not None
            or stage.client_aux_weight is not None
            or stage.client_aux_scales is not None
        ):
            raise RuntimeBindingError(
                f"bundle stage {stage.id!r} must not carry client weights"
            )
        if stage.seeded_profile is None:
            raise RuntimeBindingError(f"bundle stage {stage.id!r} is missing a ring profile")

    if not stage_specific_moduli and len({stage.modulus for stage in canonical.values()}) != 1:
        raise RuntimeBindingError("bundle stages must share one fixed modulus")

    stage_by_role = {(stage.role, stage.layer_index): stage for stage in canonical.values()}
    token_stage = stage_by_role[("token_lookup", None)]
    head_stage = stage_by_role[("lm_head", None)]

    client_fields: dict[str, dict[str, Any]] = {}
    tied = bool(manifest.get("tied_embeddings"))
    for stage in (head_stage, token_stage):
        client_weight = stage.client_weight
        client_scales = stage.client_weight_scales
        if client_weight is None or client_scales is None:
            raise RuntimeBindingError(
                f"bundle stage {stage.id!r} must carry a local client weight"
            )
        weight_values = np.asarray(client_weight)
        if weight_values.dtype != np.int8 or weight_values.ndim != 2:
            raise RuntimeBindingError(f"bundle stage {stage.id!r} client weight is malformed")
        scale_values = np.asarray(client_scales)
        if scale_values.dtype != np.float32 or scale_values.ndim != 1:
            raise RuntimeBindingError(
                f"bundle stage {stage.id!r} client weight scales are malformed"
            )
        if not np.all(np.isfinite(scale_values)) or not np.all(scale_values > 0):
            raise RuntimeBindingError(
                f"bundle stage {stage.id!r} client weight scales must be positive"
            )
        layout = str(stage.client_weight_layout)
        digest = _sha256(np.ascontiguousarray(weight_values, dtype=np.int8).tobytes())
        if stage.id == "lm_head":
            if layout != "linear" or weight_values.shape != (
                stage.out_features,
                stage.in_features,
            ):
                raise RuntimeBindingError("lm_head client weight must be a linear layout")
            if digest != stage.weight_digest:
                raise RuntimeBindingError("lm_head client weight digest mismatch")
            if not np.array_equal(scale_values, np.asarray(stage.weight_scales)):
                raise RuntimeBindingError("lm_head client weight scales mismatch")
        elif layout == "embedding":
            if not tied:
                raise RuntimeBindingError("embedding token layout requires tied embeddings")
            if weight_values.shape[0] != stage.in_features or not (
                0 < weight_values.shape[1] <= stage.out_features
            ):
                raise RuntimeBindingError("token_lookup client weight shape is inconsistent")
            if digest != head_stage.weight_digest:
                raise RuntimeBindingError("token_lookup client weight must match lm_head")
            if not np.array_equal(
                scale_values, np.asarray(head_stage.client_weight_scales)
            ):
                raise RuntimeBindingError("token_lookup client scales must match lm_head")
        elif layout == "transposed_embedding":
            if tied:
                raise RuntimeBindingError("tied token_lookup must use embedding layout")
            if weight_values.shape != (stage.out_features, stage.in_features):
                raise RuntimeBindingError("token_lookup client weight shape is inconsistent")
            if digest != stage.weight_digest:
                raise RuntimeBindingError("token_lookup client weight digest mismatch")
            if not np.array_equal(scale_values, np.asarray(stage.weight_scales)):
                raise RuntimeBindingError("token_lookup client weight scales mismatch")
        else:
            raise RuntimeBindingError(f"unsupported token_lookup layout {layout!r}")
        aux_weight = stage.client_aux_weight
        aux_scales = stage.client_aux_scales
        aux_digest = aux_scales_digest = None
        if stage.id == "lm_head" and (aux_weight is not None or aux_scales is not None):
            raise RuntimeBindingError("lm_head must not carry auxiliary client weights")
        if (aux_weight is None) != (aux_scales is None):
            raise RuntimeBindingError(
                f"bundle stage {stage.id!r} auxiliary weight is incomplete"
            )
        if aux_weight is not None:
            aux_values = np.asarray(aux_weight)
            aux_scale_values = np.asarray(aux_scales)
            if (
                aux_values.dtype != np.int8
                or aux_values.ndim != 2
                or aux_values.shape[1] != stage.in_features
            ):
                raise RuntimeBindingError(
                    f"bundle stage {stage.id!r} auxiliary weight is malformed"
                )
            if (
                aux_scale_values.dtype != np.float32
                or aux_scale_values.shape != (aux_values.shape[0],)
                or not np.all(np.isfinite(aux_scale_values))
                or not np.all(aux_scale_values > 0)
            ):
                raise RuntimeBindingError(
                    f"bundle stage {stage.id!r} auxiliary scales are malformed"
                )
            aux_digest = _sha256(np.ascontiguousarray(aux_values, dtype=np.int8).tobytes())
            aux_scales_digest = _sha256(_f32_bytes(aux_scale_values))
        client_fields[stage.id] = {
            "client_weight_layout": layout,
            "client_weight_digest": digest,
            "client_weight_scales_digest": _sha256(_f32_bytes(scale_values)),
            "client_aux_weight_digest": aux_digest,
            "client_aux_scales_digest": aux_scales_digest,
        }

    actual_body, actual_commitment = _canonical_stage_fingerprints(canonical)
    if actual_body != body_fingerprint:
        raise RuntimeBindingError("bundle body fingerprint does not match stage metadata")
    if actual_commitment != stage_commitment:
        raise RuntimeBindingError("bundle stage commitment does not match stage metadata")

    stage_semantics: dict[str, set[str]] = {stage_id: set() for stage_id in canonical}
    local_operations: set[str] = set()
    norm_weights: dict[str, int] = {}
    output_sums: dict[tuple[str, str], int] = {
        (phase, stage_id): 0 for phase in ("prefill", "decode") for stage_id in canonical
    }
    mapped_shapes: dict[str, dict[str, Any]] = {"prefill": {}, "decode": {}}
    bias_expectations: dict[tuple[str, str], set[bool]] = {
        (phase, stage_id): set()
        for phase in ("prefill", "decode")
        for stage_id in canonical
    }
    for phase in ("prefill", "decode"):
        _, by_id = phases[phase]
        for operation_id, operation in by_id.items():
            qualified = f"{phase}:{operation_id}"
            operator = operation["operator"]
            attributes = operation.get("attributes")
            if not isinstance(attributes, dict):
                raise RuntimeBindingError(f"operation {operation_id!r} requires attributes")
            if operator in LOCAL_OPERATORS:
                local_operations.add(qualified)
                if operator == "rms_norm":
                    weight_id = attributes.get("weight")
                    if not isinstance(weight_id, str):
                        raise RuntimeBindingError(f"rms norm {operation_id!r} requires a weight")
                    width = _last_dim(operation.get("output_shape"), operation_id)
                    if weight_id in norm_weights and norm_weights[weight_id] != width:
                        raise RuntimeBindingError(f"norm weight {weight_id!r} width drifted")
                    norm_weights[weight_id] = width
                continue
            if operator not in REMOTE_OPERATORS:
                raise RuntimeBindingError(f"unsupported plan operator {operator!r}")
            if operator == "token_lookup":
                stage = token_stage
            elif operator == "output_head":
                stage = head_stage
            else:
                weight_id = attributes.get("weight")
                if not isinstance(weight_id, str):
                    raise RuntimeBindingError(f"linear {operation_id!r} requires a weight")
                owner = ownership.get(_normalize_weight_key(weight_id), "")
                if owner is None:
                    raise RuntimeBindingError(f"linear {operation_id!r} weight ownership is ambiguous")
                if not owner:
                    raise RuntimeBindingError(f"linear {operation_id!r} weight has no stage owner")
                stage = canonical[owner]
            weight_id = attributes.get("weight")
            if isinstance(weight_id, str):
                row_keys = {
                    _normalize_weight_key(key)
                    for key in spec_rows[stage.id].get("weight_keys") or ()
                }
                if _normalize_weight_key(weight_id) not in row_keys:
                    raise RuntimeBindingError(
                        f"operation {operation_id!r} weight is not owned by stage {stage.id!r}"
                    )
            stage_semantics[stage.id].add(qualified)
            producers = [
                by_id[input_id]
                for input_id in operation.get("inputs") or []
                if input_id in by_id
            ]
            if operator == "token_lookup":
                producers = []
            if operator in {"linear", "output_head"} and len(producers) != 1:
                raise RuntimeBindingError(f"operation {operation_id!r} requires one producer")
            for producer in producers:
                if _last_dim(producer.get("output_shape"), producer["id"]) != stage.in_features:
                    raise RuntimeBindingError(
                        f"operation {operation_id!r} input width does not match {stage.id!r}"
                    )
            output_sums[(phase, stage.id)] += _last_dim(
                operation.get("output_shape"), operation_id
            )
            bias_expectations[(phase, stage.id)].add(
                operator == "linear" and bool(attributes.get("bias", False))
            )
            mapped_shapes[phase][operation_id] = (
                stage.id,
                _last_dim(operation.get("output_shape"), operation_id),
                tuple(
                    _last_dim(producer.get("output_shape"), producer["id"])
                    for producer in producers
                ),
                tuple(producer["id"] for producer in producers),
            )
    for stage_id, stage in canonical.items():
        if not stage_semantics[stage_id]:
            raise RuntimeBindingError(f"bundle stage {stage_id!r} has no semantic operations")
        for phase in ("prefill", "decode"):
            if output_sums[(phase, stage_id)] != stage.out_features:
                raise RuntimeBindingError(
                    f"bundle stage {stage_id!r} {phase} output sum is inconsistent"
                )
        phase_bias: dict[str, bool] = {}
        for phase in ("prefill", "decode"):
            flags = bias_expectations[(phase, stage_id)]
            if len(flags) != 1:
                raise RuntimeBindingError(
                    f"bundle stage {stage_id!r} mixes biased and unbiased operations"
                )
            phase_bias[phase] = flags.pop()
        if phase_bias["prefill"] != phase_bias["decode"]:
            raise RuntimeBindingError(
                f"bundle stage {stage_id!r} bias expectations differ across phases"
            )
        if (stage.bias is not None) != phase_bias["prefill"]:
            raise RuntimeBindingError(
                f"bundle stage {stage_id!r} bias does not match the model plan"
            )
    if mapped_shapes["prefill"] != mapped_shapes["decode"]:
        raise RuntimeBindingError("prefill and decode stage mappings differ")
    for stage_id, qualified in stage_semantics.items():
        prefill_ids = {item.split(":", 1)[1] for item in qualified if item.startswith("prefill:")}
        decode_ids = {item.split(":", 1)[1] for item in qualified if item.startswith("decode:")}
        if prefill_ids != decode_ids or len(qualified) != len(prefill_ids) + len(decode_ids):
            raise RuntimeBindingError(f"stage {stage_id!r} prefill and decode mappings differ")

    scheduled_operations: set[str] = set()
    for phase in ("prefill", "decode"):
        phase_schedule = runtime_schedule_spec.get(phase)
        if not isinstance(phase_schedule, dict) or not isinstance(
            phase_schedule.get("steps"), list
        ):
            raise RuntimeBindingError(f"native {phase} runtime schedule is malformed")
        phase_operations = phases[phase][1]
        for order, step in enumerate(phase_schedule["steps"]):
            if (
                not isinstance(step, dict)
                or step.get("order") != order
                or not isinstance(step.get("operation_ids"), list)
                or not isinstance(step.get("operators"), list)
                or not isinstance(step.get("input_ids"), list)
            ):
                raise RuntimeBindingError(f"native {phase} runtime step is malformed")
            operation_ids = step["operation_ids"]
            if not operation_ids or not all(isinstance(item, str) for item in operation_ids):
                raise RuntimeBindingError(f"native {phase} runtime step operations are malformed")
            operations = [phase_operations.get(operation_id) for operation_id in operation_ids]
            if any(operation is None for operation in operations):
                raise RuntimeBindingError(f"native {phase} runtime operation is unknown")
            if step["operators"] != [operation["operator"] for operation in operations]:
                raise RuntimeBindingError(f"native {phase} runtime operators are malformed")
            if step.get("layer") != operations[0].get("layer") or any(
                operation.get("layer") != step.get("layer") for operation in operations
            ):
                raise RuntimeBindingError(f"native {phase} runtime step layer is malformed")
            if step["input_ids"] != operations[0].get("inputs") or any(
                operation.get("inputs") != step["input_ids"] for operation in operations
            ):
                raise RuntimeBindingError(f"native {phase} runtime step inputs are malformed")
            outputs = step.get("outputs")
            if (
                not isinstance(outputs, list)
                or not all(isinstance(row, dict) for row in outputs)
                or [row.get("operation_id") for row in outputs] != operation_ids
            ):
                raise RuntimeBindingError(f"native {phase} runtime outputs are malformed")
            executor = step.get("executor")
            layer = step.get("layer")
            stage_role = step.get("stage_role")
            stage_offset = 0
            remote_stage = None
            if executor == "remote_stage":
                remote_stage = stage_by_role.get((stage_role, layer))
                if remote_stage is None:
                    raise RuntimeBindingError(f"native {phase} runtime stage is unknown")
            elif executor != "client_local" or stage_role is not None or len(operation_ids) != 1:
                raise RuntimeBindingError(f"native {phase} runtime executor is unsupported")
            for operation_id, operation, output in zip(
                operation_ids, operations, outputs, strict=True
            ):
                qualified = f"{phase}:{operation_id}"
                if qualified in scheduled_operations:
                    raise RuntimeBindingError(f"native runtime operation {qualified!r} is duplicated")
                scheduled_operations.add(qualified)
                width = _last_dim(operation.get("output_shape"), operation_id)
                if (
                    output.get("output_shape") != operation.get("output_shape")
                    or output.get("stage_offset") != stage_offset
                    or output.get("stage_width") != width
                ):
                    raise RuntimeBindingError(
                        f"native runtime operation {qualified!r} has the wrong output slice"
                    )
                stage_offset += width
                if executor == "client_local":
                    if qualified not in local_operations or stage_offset != width:
                        raise RuntimeBindingError(
                            f"native runtime operation {qualified!r} has the wrong local executor"
                        )
                elif qualified not in stage_semantics[remote_stage.id]:
                    raise RuntimeBindingError(
                        f"native runtime operation {qualified!r} has the wrong remote stage"
                    )
            if remote_stage is not None and stage_offset != remote_stage.out_features:
                raise RuntimeBindingError(
                    f"native {phase} runtime stage {remote_stage.id!r} has the wrong output width"
                )
    expected_scheduled = local_operations | {
        operation for operations in stage_semantics.values() for operation in operations
    }
    if scheduled_operations != expected_scheduled:
        raise RuntimeBindingError("native runtime schedule and bundle binding differ")

    covered = len(local_operations) + sum(len(ids) for ids in stage_semantics.values())
    if covered != len(prefill_ops) + len(decode_ops):
        raise RuntimeBindingError("plan operation coverage is incomplete")

    local_tensors = []
    for weight_id in sorted(norm_weights):
        key, array = _resolve_array(bundle.arrays, weight_id)
        value = np.asarray(array)
        if value.dtype != np.float32:
            raise RuntimeBindingError(f"client tensor {key!r} must be float32")
        if list(value.shape) != [norm_weights[weight_id]]:
            raise RuntimeBindingError(f"client tensor {key!r} shape does not match its norm")
        if not np.all(np.isfinite(value)):
            raise RuntimeBindingError(f"client tensor {key!r} must be finite")
        local_tensors.append({
            "weight_id": weight_id,
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _sha256(value.astype("<f4", copy=False).tobytes()),
        })

    bindings = []
    for stage in canonical.values():
        bindings.append(RuntimeStageBinding(
            stage_id=stage.id,
            role=str(stage.role),
            layer_index=stage.layer_index,
            semantic_operations=tuple(sorted(stage_semantics[stage.id])),
            in_features=stage.in_features,
            out_features=stage.out_features,
            weight_bits=stage.weight_bits,
            activation_bits=stage.activation_bits,
            ring=str(stage.ring),
            modulus=stage.modulus,
            wire_bits=stage.wire_bits,
            weight_digest=stage.weight_digest,
            weight_scales_digest=_sha256(_f32_bytes(stage.weight_scales)),
            bias_digest=None if stage.bias is None else _sha256(_f32_bytes(stage.bias)),
            client_weight_layout=client_fields.get(stage.id, {}).get("client_weight_layout"),
            client_weight_digest=client_fields.get(stage.id, {}).get("client_weight_digest"),
            client_weight_scales_digest=client_fields.get(stage.id, {}).get(
                "client_weight_scales_digest"
            ),
            client_aux_weight_digest=client_fields.get(stage.id, {}).get(
                "client_aux_weight_digest"
            ),
            client_aux_scales_digest=client_fields.get(stage.id, {}).get(
                "client_aux_scales_digest"
            ),
        ))
    bindings.sort(
        key=lambda item: (
            item.layer_index is None,
            item.layer_index or -1,
            item.role,
            item.stage_id,
        )
    )

    manifest_identity = _binding_digest(_canonical_json({
        "id": manifest.get("id"),
        "architecture": manifest.get("architecture"),
        "source_format": manifest.get("source_format"),
        "vocab_size": manifest.get("vocab_size"),
        "hidden_size": manifest.get("hidden_size"),
        "intermediate_size": manifest.get("intermediate_size"),
        "num_hidden_layers": manifest.get("num_hidden_layers"),
        "num_attention_heads": manifest.get("num_attention_heads"),
        "num_key_value_heads": manifest.get("num_key_value_heads"),
        "head_dim": manifest.get("head_dim"),
        "context_length": manifest.get("context_length"),
        "quantization": manifest.get("quantization"),
        "tied_embeddings": manifest.get("tied_embeddings"),
        "stages": manifest.get("stages"),
        "metadata": {
            key: manifest_metadata[key]
            for key in (
                "model_type",
                "architectures",
                "text_config",
                "client_tensor_policy",
                "model_family",
                "block_style",
                "weight_bits",
                "activation_bits",
            )
            if key in manifest_metadata
        },
    }))
    fingerprint_payload = {
        "model_id": bundle.model_id,
        "manifest_identity": manifest_identity,
        "runtime_config_digest": runtime_config_digest,
        "runtime_schedule_digest": runtime_schedule_digest,
        "tokenizer_digest": tokenizer_digest,
        "privacy": {
            "mode": privacy["mode"],
            "protocol": privacy["protocol"],
            "body_fingerprint": body_fingerprint,
            "stage_commitment": stage_commitment,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
        },
        "stages": [
            {
                "id": stage.id,
                "role": stage.role,
                "layer_index": stage.layer_index,
                "in_features": stage.in_features,
                "out_features": stage.out_features,
                "weight_bits": stage.weight_bits,
                "activation_bits": stage.activation_bits,
                "ring": stage.ring,
                "modulus": stage.modulus,
                "wire_bits": stage.wire_bits,
                "weight_digest": stage.weight_digest,
                "weight_scales_digest": _sha256(_f32_bytes(stage.weight_scales)),
                "bias_digest": None if stage.bias is None else _sha256(_f32_bytes(stage.bias)),
                "client_weight_layout": client_fields.get(stage.id, {}).get(
                    "client_weight_layout"
                ),
                "client_weight_digest": client_fields.get(stage.id, {}).get(
                    "client_weight_digest"
                ),
                "client_weight_scales_digest": client_fields.get(stage.id, {}).get(
                    "client_weight_scales_digest"
                ),
                "client_aux_weight_digest": client_fields.get(stage.id, {}).get(
                    "client_aux_weight_digest"
                ),
                "client_aux_scales_digest": client_fields.get(stage.id, {}).get(
                    "client_aux_scales_digest"
                ),
            }
            for stage in (canonical[key] for key in sorted(canonical))
        ],
        "local_tensors": [
            {
                "key": key,
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
                "sha256": _sha256(_f32_bytes(np.asarray(value))),
            }
            for key, value in sorted(bundle.arrays.items())
        ],
    }
    fingerprint = _binding_digest(_canonical_json(fingerprint_payload))

    spec = {
        "schema": BINDING_SCHEMA,
        "model_plan_digest": plan.digest,
        "model_family": REQUIRED_MODEL_FAMILY,
        "adapter": REQUIRED_ADAPTER,
        "bundle_fingerprint": fingerprint,
        "model_id": bundle.model_id,
        "body_fingerprint": body_fingerprint,
        "stage_commitment": stage_commitment,
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "stages": [dataclasses.asdict(binding) for binding in bindings],
        "local_tensors": local_tensors,
        "local_operations": sorted(local_operations),
        "operation_count": covered,
        "complete": True,
        "completeness_scope": "runtime_binding",
        "runtime_config_digest": runtime_config_digest,
        "runtime_schedule_digest": runtime_schedule_digest,
        "tokenizer_digest": tokenizer_digest,
    }
    canonical_bytes = _canonical_json(spec)
    return CompiledRuntimeModel._create(
        plan=plan,
        bundle=bundle,
        canonical=canonical_bytes,
        digest=_binding_digest(canonical_bytes),
        fingerprint=fingerprint,
        stages=tuple(bindings),
        local_operations=tuple(sorted(local_operations)),
        runtime_config_digest=runtime_config_digest,
        runtime_schedule_digest=runtime_schedule_digest,
        tokenizer_digest=tokenizer_digest,
    )


__all__ = [
    "CompiledRuntimeModel",
    "RuntimeBindingError",
    "RuntimeStageBinding",
    "compile_runtime_model",
]
