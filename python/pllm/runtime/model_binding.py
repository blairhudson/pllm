from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np

from pllm.configuration import Model, Pipeline
from pllm.modeling import ModelPlan, lower_model
from pllm.runtime.quantization import (
    choose_plain_modulus,
    choose_wire_bits,
    dequantize_matmul,
    quantize_activation_per_row,
    signed_dot_bound,
)
from pllm.runtime.public_equalization import equalize_activation
from pllm.runtime.semantic_stages import (
    client_owns_linear,
    semantic_fused_roles,
    semantic_stage_role,
)
from pllm.runtime.semantic_source import semantic_source_config
from pllm.runtime.semantic_tensors import SemanticTensorError, required_client_tensors
from pllm.runtime.transformer_client import ClientBundle

if TYPE_CHECKING:
    from pllm.runtime.model_execution import CompiledRuntimeSession
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

BINDING_SCHEMA = "pllm.runtime_model_binding.v1"
BINDING_DOMAIN = b"pllm.runtime_model_binding.v1\0"

LINEAR_OPERATORS = frozenset({"token_lookup", "linear", "output_head"})
LOCAL_OPERATORS = frozenset(
    {
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
        "scale",
        "gelu_tanh",
        "softcap",
        "permute",
        "slice",
        "state_initialize",
        "causal_convolution",
        "convolution_state_update",
        "gated_delta_decay",
        "sigmoid",
        "gated_delta_rule",
        "gated_delta_state_update",
        "rms_norm_gated",
    }
)
BOUNDARY_STAGE_IDS = frozenset({"token_lookup", "lm_head"})


class RuntimeBindingError(ValueError):
    pass


class ClientLinearExecutor:
    """Plan-bound client-owned native matrices behind a linear-stage callback."""

    __slots__ = ("_binding_digest", "_stages", "_integer_macs", "_metal_stages", "_min_metal_rows")

    def __init__(self) -> None:
        raise RuntimeBindingError("client linear execution requires a compiled binding")

    @classmethod
    def _create(
        cls,
        binding: CompiledRuntimeModel,
        engine: MaskedTransformerEngine,
    ) -> ClientLinearExecutor:
        binding.validate()
        composition = Pipeline.from_spec(json.loads(binding._canonical_composition))
        from pllm.profiles import resolve_runtime_composition

        options = resolve_runtime_composition(composition)
        if options is None or options.client_runtime != "compiled_client_local_v1":
            raise RuntimeBindingError("client-owned kernel requires the client-only topology")
        from pllm.kernels import AppleMetal

        kernels = composition.components["kernels"]
        metal = kernels.component == AppleMetal.descriptor.component
        model = engine.models.get(binding._bundle.model_id)
        if model is None:
            raise RuntimeBindingError("client-owned kernel has no loaded checkpoint")
        metadata = model.manifest.metadata
        privacy = binding._bundle.privacy
        if (
            metadata.get("body_fingerprint") != privacy.get("body_fingerprint")
            or metadata.get("seeded_stage_commitment") != privacy.get("stage_commitment")
            or engine.weight_bits != privacy.get("weight_bits")
            or engine.activation_bits != privacy.get("activation_bits")
        ):
            raise RuntimeBindingError("client-owned kernel differs from the bound model body")
        stages: dict[
            str, tuple[Any, int, int, int, np.ndarray, np.ndarray | None, np.ndarray | None]
        ] = {}
        for row in binding._stages:
            if row.stage_id in BOUNDARY_STAGE_IDS:
                continue
            runtime = model.stages.get(row.stage_id)
            if runtime is None or runtime.compiled_weight is None:
                raise RuntimeBindingError("client-owned kernel is missing a compiled stage")
            bound = binding._bundle.stages[row.stage_id]
            if (
                runtime.weight_digest != row.weight_digest
                or runtime.spec.in_features != row.in_features
                or runtime.spec.out_features != row.out_features
                or runtime.spec.weight_bits != row.weight_bits
                or runtime.spec.activation_bits != row.activation_bits
                or runtime.modulus != row.modulus
                or runtime.wire_bits != row.wire_bits
                or not np.array_equal(runtime.weight.scales, bound.weight_scales)
                or (runtime.bias is None) != (bound.bias is None)
                or (runtime.bias is not None and not np.array_equal(runtime.bias, bound.bias))
                or runtime.equalization_profile_digest != bound.equalization_profile_digest
                or (runtime.input_equalization is None) != (bound.input_equalization is None)
                or (
                    runtime.input_equalization is not None
                    and not np.array_equal(runtime.input_equalization, bound.input_equalization)
                )
            ):
                raise RuntimeBindingError(f"client-owned stage {row.stage_id!r} differs from plan")
            stages[row.stage_id] = (
                runtime.compiled_weight,
                row.in_features,
                row.out_features,
                row.activation_bits,
                bound.weight_scales.copy(),
                None if bound.bias is None else bound.bias.copy(),
                None if bound.input_equalization is None else bound.input_equalization.copy(),
            )
        if not stages:
            raise RuntimeBindingError("client-owned kernel has no bound body stages")
        metal_stages: dict[str, Any] = {}
        if metal:
            from pllm.runtime.metal import MetalGEMM
            from pllm.runtime.native import NativeKernelError

            try:
                metal_stages = MetalGEMM().bind_stages(
                    {stage_id: model.stages[stage_id].weight.values for stage_id in stages}
                )
            except NativeKernelError as exc:
                raise RuntimeBindingError(f"client Metal stage admission failed: {exc}") from exc
        self = object.__new__(cls)
        self._binding_digest = binding.digest
        self._stages = stages
        self._integer_macs = 0
        self._metal_stages = metal_stages
        self._min_metal_rows = kernels.params["min_rows"] if metal else 0
        return self

    @property
    def integer_macs(self) -> int:
        return self._integer_macs

    def __call__(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        try:
            matrix, in_features, out_features, bits, scales, bias, equalization = self._stages[
                stage_id
            ]
        except KeyError as exc:
            raise RuntimeBindingError("stage is not in the client-owned plan") from exc
        values = np.asarray(activation)
        if (
            values.dtype != np.float32
            or values.ndim not in (1, 2, 3)
            or values.shape[-1] != in_features
            or not np.all(np.isfinite(values))
        ):
            raise RuntimeBindingError("client-owned stage input has an invalid shape")
        if equalization is not None:
            values = equalize_activation(values, equalization)
        quantized = quantize_activation_per_row(values, bits=bits)
        metal_matrix = self._metal_stages.get(stage_id)
        integer = (
            metal_matrix.clear(quantized.values)
            if metal_matrix is not None and quantized.rows >= self._min_metal_rows
            else matrix.clear(quantized.values)
        )
        output = dequantize_matmul(
            integer,
            quantized.scales,
            scales,
            output_shape=quantized.original_shape[:-1] + (out_features,),
        )
        if bias is not None:
            output += bias
        if not np.all(np.isfinite(output)):
            raise RuntimeBindingError("client-owned stage output is not finite")
        self._integer_macs += quantized.rows * in_features * out_features
        return np.ascontiguousarray(output, dtype=np.float32)


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
        "_canonical_composition",
        "_runtime_schedule_digest",
        "_stages",
        "_tokenizer_digest",
    )

    def __init__(self) -> None:
        raise RuntimeBindingError("CompiledRuntimeModel must be created by compile_runtime_model")

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
        canonical_composition: bytes,
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
        self._canonical_composition = canonical_composition
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

    def client_linear_executor(self, engine: MaskedTransformerEngine) -> ClientLinearExecutor:
        return ClientLinearExecutor._create(self, engine)

    def to_spec(self) -> dict[str, Any]:
        return json.loads(self._canonical)

    def validate(self) -> None:
        refreshed = compile_runtime_model(
            self._plan,
            self._bundle,
            composition=Pipeline.from_spec(json.loads(self._canonical_composition)),
        )
        if refreshed.digest != self._digest or refreshed.to_spec() != self.to_spec():
            raise RuntimeBindingError("bound plan or bundle changed since compilation")

    def runtime(
        self,
        remote: Callable[[str, np.ndarray], np.ndarray],
        *,
        token_cache: OrderedDict[int, np.ndarray] | None = None,
        token_cache_size: int = 512,
        token_cache_lock: threading.Lock | None = None,
        nonlinear_evaluator: Callable[[int, np.ndarray], np.ndarray] | None = None,
    ) -> SemanticDecoderRuntime:
        self.validate()
        from .semantic_executor import SemanticDecoderRuntime

        composition = Pipeline.from_spec(json.loads(self._canonical_composition))
        from pllm.profiles import resolve_runtime_composition

        options = resolve_runtime_composition(composition)
        if (
            options is not None
            and options.client_runtime == "compiled_client_local_v1"
            and (type(remote) is not ClientLinearExecutor or remote._binding_digest != self._digest)
        ):
            raise RuntimeBindingError("client-owned plan requires its bound local kernel")
        if options is not None and options.client_runtime == "compiled_offset_v1":
            from .offset_reference import TwoOnlineOffsetTransport

            if (
                type(remote) is not TwoOnlineOffsetTransport
                or remote._compiled is not self
                or remote._closed
            ):
                raise RuntimeBindingError(
                    "offset plan requires its authenticated two-worker session"
                )
        if options is not None and options.verification_component is not None:
            from .transformer_client import PreparedInventoryLease, PreparedRemoteLinear

            if (
                type(remote) is not PreparedRemoteLinear
                or remote.verification_component != options.verification_component
                or remote.model_id != self._bundle.model_id
                or remote.body_fingerprint != self._bundle.privacy.get("body_fingerprint")
                or remote.stages is not self._bundle.stages
                or type(remote.inventory) is not PreparedInventoryLease
                or remote.inventory._closed
                or any(
                    stage.id not in remote.inventory.stages
                    or remote.inventory.stages[stage.id].verification is None
                    for stage in self._bundle.stages.values()
                    if stage.client_weight is None and stage.id != "embed_tokens"
                )
            ):
                raise RuntimeBindingError(
                    "verified plan requires a matching one-use prepared verifier-bound executor"
                )
        schedule = self._plan.runtime_schedule(composition).to_dict()
        bindings = {
            operation: stage.stage_id
            for stage in self._stages
            for operation in stage.semantic_operations
        }
        local_tensors = {row["weight_id"]: row["key"] for row in self.to_spec()["local_tensors"]}
        return SemanticDecoderRuntime(
            self._bundle,
            remote,
            plan=self._plan,
            schedule=schedule,
            stages=bindings,
            tensors=local_tensors,
            token_cache=token_cache,
            token_cache_size=token_cache_size,
            token_cache_lock=token_cache_lock,
            nonlinear_evaluator=nonlinear_evaluator,
        )

    def _runtime_with_nonlinear(
        self,
        remote: Callable[[str, np.ndarray], np.ndarray],
        evaluator: Callable[[int, np.ndarray], np.ndarray],
    ) -> SemanticDecoderRuntime:
        return self.runtime(remote, nonlinear_evaluator=evaluator)

    def session(
        self,
        remote: Callable[[str, np.ndarray], np.ndarray],
        *,
        research_logrow_material: object | None = None,
    ) -> "CompiledRuntimeSession":
        from pllm.runtime.model_execution import CompiledRuntimeSession

        return CompiledRuntimeSession._create(
            self, remote, research_logrow_material=research_logrow_material
        )


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
    return key[len("model.") :] if key.startswith("model.") else key


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeBindingError(f"{name} must be an integer")
    return int(value)


def _last_dim(shape: Any, name: str) -> int:
    if not isinstance(shape, (list, tuple)) or not shape:
        raise RuntimeBindingError(f"{name} must have a nonempty output shape")
    return _require_int(shape[-1], name)


def _phase_operations(
    document: dict[str, Any], phase: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
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


def _canonical_stage_fingerprints(
    stages: dict[str, Any],
    *,
    remote_output_head: bool = False,
    client_prefix_layers: int = 0,
    client_linear_roles: tuple[str, ...] = (),
) -> tuple[str, str]:
    body = []
    commitment = []
    for stage_id, stage in sorted(stages.items()):
        if stage_id == "token_lookup" or (stage_id == "lm_head" and not remote_output_head):
            continue
        local_prefix = client_owns_linear(
            stage,
            client_prefix_layers=client_prefix_layers,
            client_linear_roles=client_linear_roles,
        )
        scales = stage.weight_scales.astype("<f4", copy=False).tobytes()
        bias = None if stage.bias is None else stage.bias.astype("<f4", copy=False).tobytes()
        body_row = {
            "id": stage_id,
            "op": stage.op,
            "in": stage.in_features,
            "out": stage.out_features,
            "weight_bits": stage.weight_bits,
            "activation_bits": stage.activation_bits,
            "weight_digest": stage.weight_digest,
            "weight_scales": scales,
            "bias": bias,
        }
        equalization = None
        if stage.input_equalization is not None:
            equalization = stage.input_equalization.astype("<f4", copy=False).tobytes()
            body_row["input_equalization"] = equalization
            body_row["equalization_profile_digest"] = stage.equalization_profile_digest
        if stage_id != "lm_head":
            body.append(body_row)
        if local_prefix:
            continue
        commitment_row = {
            "id": stage_id,
            "weight": stage.weight_digest,
            "in": stage.in_features,
            "out": stage.out_features,
            "wb": stage.weight_bits,
            "ab": stage.activation_bits,
            "profile": stage.seeded_profile.to_dict(),
        }
        if stage.input_equalization is not None:
            assert equalization is not None
            commitment_row["input_equalization"] = equalization
            commitment_row["equalization_profile_digest"] = stage.equalization_profile_digest
        commitment.append(commitment_row)
    return (
        _sha256(msgpack.packb(body, use_bin_type=True)),
        _sha256(msgpack.packb(commitment, use_bin_type=True)),
    )


def _runtime_config(
    cfg: dict[str, Any],
    *,
    nested_source: bool = False,
    tokenizer_descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hidden = _require_int(cfg.get("hidden_size"), "config hidden_size")
    intermediate = _require_int(cfg.get("intermediate_size"), "config intermediate_size")
    layers = _require_int(cfg.get("num_hidden_layers"), "config num_hidden_layers")
    heads = _require_int(cfg.get("num_attention_heads"), "config num_attention_heads")
    kv_heads = _require_int(cfg.get("num_key_value_heads", heads), "config num_key_value_heads")
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
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise RuntimeBindingError(f"config {name} must be a finite number")
    layer_types = cfg.get("layer_types") or ["full_attention"] * layers
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != layers
        or not all(isinstance(item, str) for item in layer_types)
    ):
        raise RuntimeBindingError("config layer_types must be a per-layer string list")
    source_sliding_window = cfg.get("sliding_window")
    if source_sliding_window is not None:
        if _require_int(source_sliding_window, "config sliding_window") <= 0:
            raise RuntimeBindingError("config sliding_window must be positive")
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
    token_lookup_batch = max(
        1, _require_int(cfg.get("token_lookup_batch", 16), "config token_lookup_batch")
    )
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
    elif not nested_source and not (
        isinstance(rope_scaling, dict)
        and (
            (
                set(rope_scaling)
                == {
                    "rope_type",
                    "factor",
                    "low_freq_factor",
                    "high_freq_factor",
                    "original_max_position_embeddings",
                }
                and rope_scaling.get("rope_type") == "llama3"
            )
            or (
                set(rope_scaling)
                in (
                    {"type", "short_factor", "long_factor"},
                    {"type", "rope_type", "short_factor", "long_factor"},
                )
                and rope_scaling.get("type") == "longrope"
                and rope_scaling.get("rope_type", "longrope") == "longrope"
            )
        )
    ):
        raise RuntimeBindingError("compiled runtime profile does not support this rope scaling")
    if (
        not nested_source
        and isinstance(rope_scaling, dict)
        and rope_scaling.get("type") == "longrope"
    ):
        original = _require_int(cfg.get("original_max_position_embeddings"), "original context")
        maximum = _require_int(cfg.get("max_position_embeddings"), "maximum context")
        if original < 2 or maximum < original:
            raise RuntimeBindingError("per-frequency rotary context is invalid")
        rope_scaling = {
            "type": "longrope",
            "original_max_position_embeddings": original,
            "factor": maximum / original,
            "short_factor": rope_scaling["short_factor"],
            "long_factor": rope_scaling["long_factor"],
        }
    use_sliding_window = bool(cfg.get("use_sliding_window", False))
    if not nested_source and (
        use_sliding_window or any(layer != "full_attention" for layer in layer_types)
    ):
        raise RuntimeBindingError("compiled runtime profile does not support sliding windows")
    # Some dense checkpoint configs declare a dormant window. The exact value
    # participates in the native source-plan digest; no window is applied at
    # runtime when the declared mode and every layer are full attention.
    attention_bias = bool(cfg.get("attention_bias", True))
    if block_style not in {"llama", "gemma4"}:
        raise RuntimeBindingError("compiled runtime profile does not implement this block style")
    if not isinstance(rope_parameters, dict) or not isinstance(per_layer, dict):
        raise RuntimeBindingError("compiled runtime model parameters must be objects")
    if hidden_activation not in {"silu", "gelu_pytorch_tanh"}:
        raise RuntimeBindingError("compiled runtime profile does not implement this activation")
    if token_lookup_batch <= 0:
        raise RuntimeBindingError("config token_lookup_batch must be positive")
    source_bos = cfg.get("bos_token_id")
    if nested_source:
        raw_source = cfg.get("semantic_source_config")
        if isinstance(raw_source, dict) and isinstance(raw_source.get("text_config"), dict):
            raw_bos = raw_source["text_config"].get("bos_token_id")
            if raw_bos is None:
                if (
                    cfg.get("bos_token_policy") != "nonempty_only"
                    or not isinstance(tokenizer_descriptor, dict)
                    or tokenizer_descriptor.get("add_bos_token") is not False
                    or source_bos != tokenizer_descriptor.get("bos_token_id")
                ):
                    raise RuntimeBindingError(
                        "absent source BOS requires a bound nonempty-input policy"
                    )
            elif source_bos != raw_bos or "bos_token_policy" in cfg:
                raise RuntimeBindingError("runtime BOS policy differs from its source")
    if source_bos is None:
        # A tokenizer without an automatic BOS can still serve nonempty text.
        # Preserve that fact in the source lock; its descriptor's fallback is
        # used only for the existing integer runtime field, not for prefill.
        if (
            not isinstance(tokenizer_descriptor, dict)
            or tokenizer_descriptor.get("add_bos_token") is not False
        ):
            raise RuntimeBindingError(
                "source without BOS requires a tokenizer that does not add BOS"
            )
        source_bos = tokenizer_descriptor.get("bos_token_id")
    bos_token_id = _require_int(source_bos, "config bos_token_id")
    if not 0 <= bos_token_id < _require_int(cfg.get("vocab_size"), "config vocab_size"):
        raise RuntimeBindingError("config BOS fallback is outside the vocabulary")
    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rms_norm_eps": float(eps),
        "norm_offset": float(norm_offset),
        "layer_types": list(layer_types),
        "sliding_window": source_sliding_window if nested_source else None,
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
        "bos_token_id": bos_token_id,
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
    raise RuntimeBindingError(f"tokenizer descriptor value {type(value).__name__!r} is unsupported")


def _validate_runtime_semantics(
    runtime_config: dict[str, Any],
    phases: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]],
    *,
    nested_source: bool = False,
) -> None:
    rotary_head_dims: set[int] = set()
    rotary_norm_inputs: list[bool] = []
    source_scaling = runtime_config.get("rope_scaling")
    for phase, (_, operations) in phases.items():
        for operation in operations.values():
            if operation.get("operator") != "rotary_embedding":
                continue
            source = next(
                (
                    operations[value]
                    for value in operation.get("inputs") or ()
                    if value in operations
                ),
                None,
            )
            if source is None:
                raise RuntimeBindingError(f"{phase} rotary input is not produced by the plan")
            rotary_head_dims.add(_last_dim(source.get("output_shape"), f"{phase} rotary input"))
            rotary_norm_inputs.append(source.get("operator") == "rms_norm")
            if not nested_source:
                attributes = operation.get("attributes") or {}
                expected_scaling = (
                    {
                        "kind": (
                            "per_frequency_context"
                            if source_scaling.get("type") == "longrope"
                            else "wavelength_transition"
                        ),
                        **{
                            key: value
                            for key, value in source_scaling.items()
                            if key not in {"rope_type", "type"}
                        },
                    }
                    if isinstance(source_scaling, dict)
                    else None
                )
                if attributes.get("frequency_scaling") != expected_scaling:
                    raise RuntimeBindingError("rotary frequency scaling diverges from the source")
            if nested_source:
                attributes = operation.get("attributes", {})
                width = _last_dim(source.get("output_shape"), f"{phase} rotary input")
                head_dim = attributes.get("head_dim")
                partial_dim = attributes.get("rotary_dimensions")
                if (
                    (head_dim is not None and head_dim != width)
                    or (head_dim is None and partial_dim is None)
                    or (
                        partial_dim is not None
                        and (type(partial_dim) is not int or not 0 < partial_dim <= width)
                    )
                ):
                    raise RuntimeBindingError(
                        "runtime rotary width diverges from the semantic plan"
                    )
    if nested_source:
        # Native re-lowering and numeric-flow checks validate every operator;
        # flattened transport controls are constrained to the locked source.
        return
    if rotary_head_dims != {_require_int(runtime_config.get("head_dim"), "runtime head_dim")}:
        raise RuntimeBindingError("runtime head dimension diverges from the semantic plan")
    if rotary_norm_inputs and any(rotary_norm_inputs) != all(rotary_norm_inputs):
        raise RuntimeBindingError("runtime Q/K normalization topology is inconsistent")
    if bool(runtime_config.get("qk_norm")) != bool(rotary_norm_inputs and all(rotary_norm_inputs)):
        raise RuntimeBindingError("runtime Q/K normalization diverges from the semantic plan")
    expected_defaults = {
        "block_style": "llama",
        "embedding_multiplier": 1.0,
        "output_multiplier": None,
        "final_logit_softcapping": None,
        "attention_scaling": None,
        "rope_parameters": {},
        "v_norm": False,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "attention_k_eq_v": False,
        "per_layer_config": {},
        "hidden_activation": "silu",
    }
    mismatched = [
        key for key, value in expected_defaults.items() if runtime_config.get(key) != value
    ]
    layer_types = runtime_config.get("layer_types")
    if not isinstance(layer_types, list) or any(
        layer_type != "full_attention" for layer_type in layer_types
    ):
        mismatched.append("layer_types")
    if runtime_config.get("sliding_window") is not None:
        mismatched.append("sliding_window")
    if runtime_config.get("local_attention_window") is not None:
        mismatched.append("local_attention_window")
    if mismatched:
        raise RuntimeBindingError(
            "runtime numeric controls diverge from supported semantics: " + ", ".join(mismatched)
        )


def _tokenizer_digest(descriptor: Any, runtime_config: dict[str, Any]) -> str:
    if not isinstance(descriptor, dict):
        raise RuntimeBindingError("client bundle tokenizer descriptor must be a mapping")
    kind = str(descriptor.get("kind", descriptor.get("type", "")))
    if kind not in {"byte", "sentencepiece", "tokenizer_json"}:
        raise RuntimeBindingError(f"unsupported tokenizer type {kind!r}")
    vocab = descriptor.get("vocab_size")
    if (
        vocab is not None
        and _require_int(vocab, "tokenizer vocab_size") != runtime_config["vocab_size"]
    ):
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


def compile_runtime_model(
    plan: ModelPlan,
    bundle: ClientBundle,
    *,
    composition: Pipeline | None = None,
) -> CompiledRuntimeModel:
    if type(plan) is not ModelPlan:
        raise RuntimeBindingError("plan must be a ModelPlan")
    if type(bundle) is not ClientBundle:
        raise RuntimeBindingError("bundle must be a ClientBundle")
    if composition is None:
        from pllm.profiles import MaskedLinearCpu

        composition = MaskedLinearCpu(Model(bundle.model_id))
    if type(composition) is not Pipeline and not isinstance(composition, Pipeline):
        raise RuntimeBindingError("composition must be a Pipeline")
    from pllm.profiles import resolve_runtime_composition

    runtime_options = resolve_runtime_composition(composition)
    if runtime_options is None or runtime_options.client_runtime not in {
        "masked_transformer_v1",
        "compiled_client_local_v1",
        "compiled_offset_v1",
    }:
        raise RuntimeBindingError("compiled runtime component composition is unsupported")
    client_owned = runtime_options.client_runtime == "compiled_client_local_v1"
    offset_public = runtime_options.client_runtime == "compiled_offset_v1"
    verified_public = runtime_options.verification_component is not None
    if verified_public and (
        client_owned
        or offset_public
        or runtime_options.verification_component != "pllm/freivalds-verify/v1"
        or not 1 <= runtime_options.verification_target_failure_bits <= 80
    ):
        raise RuntimeBindingError(
            "compiled verifier requires the bounded prepared Freivalds executor"
        )
    from pllm.model_loader import expected_model_id

    if expected_model_id(composition.model) != bundle.model_id:
        raise RuntimeBindingError("composition model differs from the imported checkpoint")
    canonical_composition = composition.canonical_bytes()
    try:
        plan.coverage(composition)
    except Exception as exc:
        raise RuntimeBindingError("native model plan validation failed") from exc
    document = plan.to_dict()
    model_family = document.get("model_family")
    adapter = document.get("adapter")
    if not isinstance(model_family, str) or not model_family:
        raise RuntimeBindingError("model plan is missing its family identity")
    if not isinstance(adapter, str) or not adapter:
        raise RuntimeBindingError("model plan is missing its adapter identity")
    if document.get("transformations"):
        raise RuntimeBindingError("transformed model plans cannot be bound")

    manifest = bundle.manifest if isinstance(bundle.manifest, dict) else {}
    cfg = bundle.cfg if isinstance(bundle.cfg, dict) else {}
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
    max_input_tokens = _require_int(prefill_graph.get("query_sequence"), "prefill query_sequence")
    decode_key_sequence = _require_int(
        decode_graph.get("maximum_key_sequence"), "decode maximum_key_sequence"
    )
    max_new_tokens = decode_key_sequence - max_input_tokens + 1
    if batch <= 0 or max_input_tokens <= 0 or max_new_tokens <= 0:
        raise RuntimeBindingError("model plan workload bounds must be positive")
    if _require_int(decode_graph.get("query_sequence"), "decode query_sequence") != 1:
        raise RuntimeBindingError("decode query_sequence must be one")
    try:
        source_config = semantic_source_config(cfg)
        reconstructed = lower_model(
            source_config,
            batch=batch,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        raise RuntimeBindingError("client bundle config cannot reproduce the model plan") from exc
    if reconstructed.digest != plan.digest:
        raise RuntimeBindingError("client bundle config does not reproduce the model plan")
    if reconstructed.to_dict().get("config_digest") != document.get("config_digest"):
        raise RuntimeBindingError("client bundle config digest does not match the plan")
    try:
        runtime_schedule = plan.runtime_schedule(composition)
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

    nested_source = source_config is not cfg
    runtime_config = _runtime_config(
        cfg,
        nested_source=nested_source,
        tokenizer_descriptor=bundle.tokenizer_descriptor,
    )
    runtime_config_digest = _sha256(_canonical_json(cfg))
    tokenizer_digest = _tokenizer_digest(bundle.tokenizer_descriptor, runtime_config)

    phases = {phase: _phase_operations(document, phase) for phase in ("prefill", "decode")}
    _validate_runtime_semantics(runtime_config, phases, nested_source=nested_source)
    prefill_ops, decode_ops = phases["prefill"][1], phases["decode"][1]
    prefill_only = set(prefill_ops) - set(decode_ops)
    prefill_initializers = {
        op_id
        for op_id, op in prefill_ops.items()
        if op.get("operator") in {"state_initialize", "kv_cache_initialize"}
        or (op.get("operator") == "last_token" and op_id not in decode_ops)
    }
    if (
        set(decode_ops) - set(prefill_ops)
        or prefill_only != prefill_initializers
        or any(
            prefill_ops[op_id].get("operator") == "state_initialize"
            and prefill_ops[op_id].get("attributes")
            not in (
                {"initial_value": 0, "dtype": "float32", "state_kind": "convolution"},
                {"initial_value": 0, "dtype": "float32", "state_kind": "recurrent"},
            )
            for op_id in prefill_only
        )
    ):
        raise RuntimeBindingError(
            "prefill and decode operations differ: "
            f"decode-only={sorted(set(decode_ops) - set(prefill_ops))}, "
            f"unexpected-prefill={sorted(prefill_only - prefill_initializers)}, "
            f"missing-initializers={sorted(prefill_initializers - prefill_only)}"
        )
    if any(prefill_ops[op_id]["operator"] != decode_ops[op_id]["operator"] for op_id in decode_ops):
        raise RuntimeBindingError("prefill and decode operators differ")

    dims = {
        "hidden": runtime_config["hidden_size"],
        "intermediate": runtime_config["intermediate_size"],
        "layers": runtime_config["num_hidden_layers"],
        "heads": runtime_config["num_attention_heads"],
        "kv_heads": runtime_config["num_key_value_heads"],
        "head_dim": runtime_config["head_dim"],
        "vocab": runtime_config["vocab_size"],
    }
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
            config_source = _require_int(
                cfg.get("hidden_size"), "config hidden_size"
            ) // _require_int(cfg.get("num_attention_heads"), "config num_attention_heads")
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
    if privacy.get("mode") != ("offset_public" if offset_public else "public"):
        raise RuntimeBindingError("client bundle privacy mode differs from its topology")
    if privacy.get("preprocessed") is not (not client_owned and not offset_public):
        raise RuntimeBindingError("client bundle preprocessing does not match the role topology")
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
    expected_protocol = (
        f"local_clear_w{weight_bits}a{activation_bits}"
        if client_owned
        else (
            f"two_online_offset_w{weight_bits}a{activation_bits}"
            if offset_public
            else f"masked_w{weight_bits}a{activation_bits}"
        )
    )
    if privacy.get("protocol") != expected_protocol:
        raise RuntimeBindingError("client bundle protocol differs from the composed topology")
    verification_component = privacy.get("verification_component", "none")
    verification_failure_bits = _require_int(
        privacy.get("verification_target_failure_bits", 0), "verification failure bits"
    )
    if (
        verification_component != (runtime_options.verification_component or "none")
        or verification_failure_bits != runtime_options.verification_target_failure_bits
    ):
        raise RuntimeBindingError("masked-linear verification differs from its composition")
    equalization_digest = runtime_options.public_equalization_digest
    if privacy.get("public_equalization_digest") != equalization_digest or (
        equalization_digest is not None and (weight_bits, activation_bits) != (8, 8)
    ):
        raise RuntimeBindingError("client bundle equalization profile differs from its composition")
    remote_output_head = runtime_options.remote_output_head
    client_prefix_layers = runtime_options.client_prefix_layers
    client_linear_roles = runtime_options.client_linear_roles
    if (client_prefix_layers or client_linear_roles) and (
        client_owned or offset_public or verified_public
    ):
        raise RuntimeBindingError("client-owned prefix requires baseline prepared execution")
    if remote_output_head and (client_owned or offset_public or verified_public):
        raise RuntimeBindingError("remote output head requires baseline prepared execution")
    if bool(manifest.get("tied_embeddings")) and remote_output_head:
        raise RuntimeBindingError("remote output head cannot share client token weights")

    canonical: dict[str, Any] = {}
    for key, stage in bundle.stages.items():
        if key != stage.id:
            continue
        if stage.id in canonical:
            raise RuntimeBindingError(f"duplicate bundle stage id {stage.id!r}")
        canonical[stage.id] = stage
    named_roles = [
        (stage.role, stage.layer_index)
        for stage in canonical.values()
        if stage.role != "semantic_linear"
    ]
    if len(set(named_roles)) != len(named_roles):
        raise RuntimeBindingError("bundle named stage roles and layers must be unique")
    for stage_id, stage in canonical.items():
        expected_digest = None if stage_id in BOUNDARY_STAGE_IDS else equalization_digest
        if stage.equalization_profile_digest != expected_digest or (
            stage.input_equalization is None
        ) != (expected_digest is None):
            raise RuntimeBindingError("stage equalization differs from composed numeric profile")

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
        row_keys = row.get("weight_keys") or ()
        row_fused = list(row.get("fused_from") or [])
        row_op = row.get("op")
        valid = (
            row_op == stage.op
            and row_op in {"embedding", "linear", "lm_head"}
            and isinstance(row_keys, (list, tuple))
            and bool(row_keys)
            and all(isinstance(key, str) and key for key in row_keys)
            and all(isinstance(name, str) and name for name in row_fused)
            and (
                (row_op in {"embedding", "lm_head"} and stage.layer_index is None)
                or (
                    row_op == "linear"
                    and (stage.layer_index is not None or stage.role == "semantic_linear")
                )
            )
        )
        if not valid:
            raise RuntimeBindingError(f"bundle stage {stage_id!r} spec row is inconsistent")
        if len(row_fused) not in {0, len(row_keys)}:
            raise RuntimeBindingError(f"bundle stage {stage_id!r} fused order is inconsistent")

    manifest_metadata = manifest.get("metadata")
    if not isinstance(manifest_metadata, dict):
        raise RuntimeBindingError("bundle manifest is missing metadata")
    if (client_owned or offset_public) and (
        manifest_metadata.get("client_runtime") != runtime_options.client_runtime
        or manifest_metadata.get("privacy_mode") != runtime_options.privacy_mode
        or manifest_metadata.get("privacy_protocol") != expected_protocol
    ):
        raise RuntimeBindingError("manifest lacks the compiled topology contract")
    if manifest_metadata.get("runtime_config_digest") != runtime_config_digest:
        raise RuntimeBindingError("client runtime config does not match its manifest commitment")
    if manifest_metadata.get("public_equalization_digest") != equalization_digest:
        raise RuntimeBindingError("manifest equalization differs from the bound numeric profile")
    if (
        manifest_metadata.get("remote_output_head", False) is not remote_output_head
        or privacy.get("remote_output_head", False) is not remote_output_head
    ):
        raise RuntimeBindingError("output-head ownership differs from the bound composition")
    if (
        manifest_metadata.get("client_prefix_layers", 0) != client_prefix_layers
        or privacy.get("client_prefix_layers", 0) != client_prefix_layers
        or manifest_metadata.get("client_linear_roles", []) != list(client_linear_roles)
        or privacy.get("client_linear_roles", []) != list(client_linear_roles)
    ):
        raise RuntimeBindingError("client-owned prefix differs from the bound composition")
    if any(
        manifest_metadata.get(key) != privacy.get(key)
        for key in (
            "verification_component",
            "verification_target_failure_bits",
        )
    ):
        raise RuntimeBindingError("client verification profile does not match its manifest")
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

    ownership: dict[str, str | None] = {}
    for stage_id, stage in canonical.items():
        row = spec_rows[stage_id]
        for key in set(row.get("weight_keys") or ()):
            normalized = _normalize_weight_key(key)
            if normalized in ownership and ownership[normalized] != stage_id:
                ownership[normalized] = None
            else:
                ownership[normalized] = stage_id

    stage_by_role = {(stage.role, stage.layer_index): stage for stage in canonical.values()}
    token_stage = stage_by_role.get(("token_lookup", None))
    head_stage = stage_by_role.get(("lm_head", None))
    if token_stage is None or head_stage is None:
        raise RuntimeBindingError("bundle must provide token lookup and output head stages")

    schedule_stage_ids: dict[tuple[str, int], str] = {}
    expected_dims: dict[str, tuple[int, int]] = {}
    for phase in ("prefill", "decode"):
        phase_schedule = runtime_schedule_spec.get(phase)
        if not isinstance(phase_schedule, dict):
            raise RuntimeBindingError(f"native {phase} runtime schedule is malformed")
        used_stages: set[str] = set()
        for step in phase_schedule.get("steps") or ():
            if not isinstance(step, dict):
                continue
            layer = step.get("layer")
            role = (
                semantic_stage_role(step, prefill_ops if phase == "prefill" else decode_ops)
                if step.get("weight_ids")
                else None
            )
            expected_executor = (
                "client_linear"
                if client_owned
                or (type(layer) is int and layer < client_prefix_layers)
                or role in client_linear_roles
                else "verified_remote_stage"
                if verified_public
                else "remote_stage"
            )
            if step.get("executor") != expected_executor:
                continue
            order = _require_int(step.get("order"), f"native {phase} runtime step order")
            operators = step.get("operators")
            if operators and set(operators) == {"token_lookup"}:
                remote_stage = token_stage
                input_width = dims["vocab"]
            elif operators == ["output_head"]:
                remote_stage = head_stage
                input_ids = step.get("input_ids")
                if not isinstance(input_ids, list) or len(input_ids) != 1:
                    raise RuntimeBindingError(f"native {phase} output head must have one input")
                source = (
                    prefill_ops.get(input_ids[0])
                    if phase == "prefill"
                    else decode_ops.get(input_ids[0])
                )
                if source is None:
                    raise RuntimeBindingError(f"native {phase} output head input is unknown")
                input_width = _last_dim(
                    source.get("output_shape"), f"native {phase} output head input"
                )
            else:
                weight_ids = step.get("weight_ids")
                if not isinstance(weight_ids, list) or not weight_ids:
                    raise RuntimeBindingError(
                        f"native {phase} remote step is missing weight identities"
                    )
                owners = {
                    ownership.get(_normalize_weight_key(weight_id), "")
                    for weight_id in weight_ids
                    if isinstance(weight_id, str)
                }
                if None in owners or "" in owners or len(owners) != 1:
                    raise RuntimeBindingError(
                        f"native {phase} runtime weight ownership is ambiguous"
                    )
                owner = next(iter(owners))
                if not isinstance(owner, str):
                    raise RuntimeBindingError(
                        f"native {phase} runtime weight ownership is malformed"
                    )
                remote_stage = canonical[owner]
                input_ids = step.get("input_ids")
                if not isinstance(input_ids, list) or len(input_ids) != 1:
                    raise RuntimeBindingError(
                        f"native {phase} remote linear stage must have one input"
                    )
                operations = prefill_ops if phase == "prefill" else decode_ops
                source = operations.get(input_ids[0])
                if source is None:
                    raise RuntimeBindingError(f"native {phase} remote linear input is unknown")
                input_width = _last_dim(
                    source.get("output_shape"), f"native {phase} remote linear input"
                )

            expected_stage_op = (
                "embedding"
                if isinstance(operators, list) and operators and set(operators) == {"token_lookup"}
                else "lm_head"
                if operators == ["output_head"]
                else "linear"
            )
            if remote_stage.op != expected_stage_op or remote_stage.layer_index != step.get(
                "layer"
            ):
                raise RuntimeBindingError(
                    f"native {phase} runtime stage does not match its semantic operation"
                )
            operations = prefill_ops if phase == "prefill" else decode_ops
            try:
                expected_role = semantic_stage_role(step, operations)
            except ValueError as exc:
                raise RuntimeBindingError("remote stage topology is not implemented") from exc
            if remote_stage.role != expected_role:
                raise RuntimeBindingError(
                    f"native {phase} runtime stage role does not match plan topology"
                )
            if spec_rows[remote_stage.id].get("fused_from") != list(
                semantic_fused_roles(expected_role)
            ):
                raise RuntimeBindingError("remote stage fused order disagrees with semantic roles")
            semantic_weights = step.get("weight_ids")
            manifest_weights = spec_rows[remote_stage.id].get("weight_keys")
            if (
                not isinstance(semantic_weights, list)
                or not isinstance(manifest_weights, list)
                or [_normalize_weight_key(value) for value in semantic_weights]
                != [_normalize_weight_key(value) for value in manifest_weights]
            ):
                raise RuntimeBindingError(
                    f"native {phase} runtime stage {remote_stage.id!r} weight order "
                    f"does not match the plan: {manifest_weights!r} != {semantic_weights!r}"
                )
            outputs = step.get("outputs")
            if not isinstance(outputs, list) or not outputs:
                raise RuntimeBindingError(f"native {phase} remote outputs are malformed")
            output_width = sum(
                _require_int(output.get("stage_width"), "remote stage output width")
                for output in outputs
                if isinstance(output, dict)
            )
            if output_width <= 0 or len(outputs) != sum(
                isinstance(output, dict) for output in outputs
            ):
                raise RuntimeBindingError(f"native {phase} remote outputs are malformed")
            contract = (input_width, output_width)
            previous = expected_dims.setdefault(remote_stage.id, contract)
            if previous != contract or remote_stage.id in used_stages:
                raise RuntimeBindingError(f"native {phase} remote stage binding is inconsistent")
            used_stages.add(remote_stage.id)
            schedule_stage_ids[(phase, order)] = remote_stage.id
        if used_stages != set(canonical):
            raise RuntimeBindingError(f"native {phase} remote stages do not match the bundle")

    for stage in canonical.values():
        expected = expected_dims.get(stage.id)
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
        elif (
            modulus <= signed_dot_bound(features, stage.weight_bits, stage.activation_bits) * 2 + 1
        ):
            raise RuntimeBindingError(f"bundle stage {stage.id!r} modulus does not cover its bound")
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
        if (
            stage.id == "token_lookup"
            or (stage.id == "lm_head" and not remote_output_head)
            or client_owns_linear(
                stage,
                client_prefix_layers=client_prefix_layers,
                client_linear_roles=client_linear_roles,
            )
        ):
            continue
        if (
            stage.client_weight is not None
            or stage.client_weight_scales is not None
            or stage.client_aux_weight is not None
            or stage.client_aux_scales is not None
        ):
            raise RuntimeBindingError(f"bundle stage {stage.id!r} must not carry client weights")
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
        if stage.id == "lm_head" and remote_output_head:
            if (
                any(
                    value is not None
                    for value in (
                        stage.client_weight,
                        stage.client_weight_scales,
                        stage.client_aux_weight,
                        stage.client_aux_scales,
                    )
                )
                or stage.seeded_profile is None
            ):
                raise RuntimeBindingError(
                    "remote output head must carry only a committed ring profile"
                )
            continue
        client_weight = stage.client_weight
        client_scales = stage.client_weight_scales
        if client_weight is None or client_scales is None:
            raise RuntimeBindingError(f"bundle stage {stage.id!r} must carry a local client weight")
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
            if not np.array_equal(scale_values, np.asarray(head_stage.client_weight_scales)):
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
            raise RuntimeBindingError(f"bundle stage {stage.id!r} auxiliary weight is incomplete")
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

    local_bytes = 0
    for stage in canonical.values():
        if not client_owns_linear(
            stage,
            client_prefix_layers=client_prefix_layers,
            client_linear_roles=client_linear_roles,
        ):
            continue
        weight = np.asarray(stage.client_weight)
        scales = np.asarray(stage.client_weight_scales)
        if (
            stage.client_weight_layout != "linear"
            or stage.seeded_profile is not None
            or stage.client_aux_weight is not None
            or stage.client_aux_scales is not None
            or weight.dtype != np.int8
            or weight.shape != (stage.out_features, stage.in_features)
            or scales.dtype != np.float32
            or scales.shape != (stage.out_features,)
            or not np.all(np.isfinite(scales))
            or not np.all(scales > 0)
            or _sha256(np.ascontiguousarray(weight).tobytes()) != stage.weight_digest
            or not np.array_equal(scales, np.asarray(stage.weight_scales))
        ):
            raise RuntimeBindingError("client-owned prefix weight or stage commitment is invalid")
        local_bytes += int(weight.nbytes + scales.nbytes)
        client_fields[stage.id] = {
            "client_weight_layout": "linear",
            "client_weight_digest": stage.weight_digest,
            "client_weight_scales_digest": _sha256(_f32_bytes(scales)),
            "client_aux_weight_digest": None,
            "client_aux_scales_digest": None,
        }
    if local_bytes > 512 << 20:
        raise RuntimeBindingError("client-owned prefix exceeds its 512 MiB weight bound")

    actual_body, actual_commitment = _canonical_stage_fingerprints(
        canonical,
        remote_output_head=remote_output_head,
        client_prefix_layers=client_prefix_layers,
        client_linear_roles=client_linear_roles,
    )
    if actual_body != body_fingerprint:
        raise RuntimeBindingError("bundle body fingerprint does not match stage metadata")
    if actual_commitment != stage_commitment:
        raise RuntimeBindingError("bundle stage commitment does not match stage metadata")

    stage_semantics: dict[str, set[str]] = {stage_id: set() for stage_id in canonical}
    local_operations: set[str] = set()
    try:
        client_weights = required_client_tensors(plan)
    except SemanticTensorError as exc:
        raise RuntimeBindingError(str(exc)) from exc
    output_sums: dict[tuple[str, str], int] = {
        (phase, stage_id): 0 for phase in ("prefill", "decode") for stage_id in canonical
    }
    mapped_shapes: dict[str, dict[str, Any]] = {"prefill": {}, "decode": {}}
    bias_expectations: dict[tuple[str, str], set[bool]] = {
        (phase, stage_id): set() for phase in ("prefill", "decode") for stage_id in canonical
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
                continue
            if operator not in LINEAR_OPERATORS:
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
                    raise RuntimeBindingError(
                        f"linear {operation_id!r} weight ownership is ambiguous"
                    )
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
                by_id[input_id] for input_id in operation.get("inputs") or [] if input_id in by_id
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
            output_sums[(phase, stage.id)] += _last_dim(operation.get("output_shape"), operation_id)
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
    if set(mapped_shapes["prefill"]) != set(mapped_shapes["decode"]):
        raise RuntimeBindingError("prefill and decode stage mappings differ")
    for op_id, bound in mapped_shapes["prefill"].items():
        decoded = mapped_shapes["decode"][op_id]
        if bound == decoded:
            continue
        prefill_source = bound[3][0] if len(bound[3]) == 1 else None
        decode_source = decoded[3][0] if len(decoded[3]) == 1 else None
        selector = prefill_ops.get(prefill_source, {})
        if not (
            prefill_ops[op_id]["operator"] == "output_head"
            and bound[0] == decoded[0] == "lm_head"
            and bound[:3] == decoded[:3]
            and selector.get("operator") == "last_token"
            and selector.get("inputs") == [decode_source, "input.sequence_lengths"]
            and selector.get("attributes")
            == {
                "axis": 1,
                "selection": "last_valid",
                "valid_lengths_input": "input.sequence_lengths",
            }
        ):
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
            weight_ids = step.get("weight_ids")
            if not isinstance(weight_ids, list) or not all(
                isinstance(weight_id, str) for weight_id in weight_ids
            ):
                raise RuntimeBindingError(f"native {phase} runtime weights are malformed")
            stage_offset = 0
            remote_stage = None
            layer = step.get("layer")
            role = semantic_stage_role(step, phase_operations) if weight_ids else None
            expected_executor = (
                "client_linear"
                if client_owned
                or (type(layer) is int and layer < client_prefix_layers)
                or role in client_linear_roles
                else "verified_remote_stage"
                if verified_public
                else "remote_stage"
            )
            if executor == expected_executor:
                expected_weights = [
                    operation["attributes"].get("weight") for operation in operations
                ]
                if weight_ids != expected_weights:
                    raise RuntimeBindingError(f"native {phase} runtime weights are malformed")
                stage_id = schedule_stage_ids.get(
                    (phase, _require_int(step.get("order"), "step order"))
                )
                if stage_id is None:
                    raise RuntimeBindingError(f"native {phase} runtime stage binding is missing")
                remote_stage = canonical[stage_id]
            elif executor != "client_local" or weight_ids or len(operation_ids) != 1:
                raise RuntimeBindingError(f"native {phase} runtime executor is unsupported")
            for operation_id, operation, output in zip(
                operation_ids, operations, outputs, strict=True
            ):
                qualified = f"{phase}:{operation_id}"
                if qualified in scheduled_operations:
                    raise RuntimeBindingError(
                        f"native runtime operation {qualified!r} is duplicated"
                    )
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
    for weight_id in sorted(client_weights):
        key, array = _resolve_array(bundle.arrays, weight_id)
        value = np.asarray(array)
        if value.dtype != np.float32:
            raise RuntimeBindingError(f"client tensor {key!r} must be float32")
        if tuple(value.shape) != client_weights[weight_id]:
            raise RuntimeBindingError(
                f"client tensor {key!r} shape does not match its semantic use"
            )
        if not np.all(np.isfinite(value)):
            raise RuntimeBindingError(f"client tensor {key!r} must be finite")
        local_tensors.append(
            {
                "weight_id": weight_id,
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _sha256(value.astype("<f4", copy=False).tobytes()),
            }
        )

    bindings = []
    for stage in canonical.values():
        bindings.append(
            RuntimeStageBinding(
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
            )
        )
    bindings.sort(
        key=lambda item: (
            item.layer_index is None,
            item.layer_index or -1,
            item.role,
            item.stage_id,
        )
    )

    manifest_identity = _binding_digest(
        _canonical_json(
            {
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
                        "runtime_config_digest",
                        "verification_component",
                        "verification_target_failure_bits",
                    )
                    if key in manifest_metadata
                },
            }
        )
    )
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
            "composition_digest": composition.digest(),
            "verification_component": privacy.get("verification_component", "none"),
            "verification_target_failure_bits": int(
                privacy.get("verification_target_failure_bits", 0)
            ),
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
                "client_weight_layout": client_fields.get(stage.id, {}).get("client_weight_layout"),
                "client_weight_digest": client_fields.get(stage.id, {}).get("client_weight_digest"),
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
        "model_family": model_family,
        "adapter": adapter,
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
        canonical_composition=canonical_composition,
        runtime_schedule_digest=runtime_schedule_digest,
        tokenizer_digest=tokenizer_digest,
    )


__all__ = [
    "CompiledRuntimeModel",
    "RuntimeBindingError",
    "RuntimeStageBinding",
    "compile_runtime_model",
]
