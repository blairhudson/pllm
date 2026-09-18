from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np

from pllm.modeling import ModelPlan
from pllm.runtime.quantization import choose_wire_bits
from pllm.runtime.transformer_client import ClientBundle, MaskedTransformerClientRuntime

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


class CompiledRuntimeModel:
    __slots__ = (
        "_plan",
        "_bundle",
        "_canonical",
        "_digest",
        "_fingerprint",
        "_local_operations",
        "_stages",
    )

    def __init__(
        self,
        *,
        plan: ModelPlan,
        bundle: ClientBundle,
        canonical: bytes,
        digest: str,
        fingerprint: str,
        stages: tuple[RuntimeStageBinding, ...],
        local_operations: tuple[str, ...],
    ) -> None:
        self._plan = plan
        self._bundle = bundle
        self._canonical = canonical
        self._digest = digest
        self._fingerprint = fingerprint
        self._stages = stages
        self._local_operations = local_operations

    @property
    def complete(self) -> bool:
        return True

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
    def stage_bindings(self) -> tuple[RuntimeStageBinding, ...]:
        return self._stages

    @property
    def local_operations(self) -> tuple[str, ...]:
        return self._local_operations

    def canonical_bytes(self) -> bytes:
        return self._canonical

    def to_spec(self) -> dict[str, Any]:
        return json.loads(self._canonical)

    def runtime(
        self, remote: Callable[[str, np.ndarray], np.ndarray]
    ) -> MaskedTransformerClientRuntime:
        refreshed = compile_runtime_model(self._plan, self._bundle)
        if refreshed.digest != self._digest or refreshed.to_spec() != self.to_spec():
            raise RuntimeBindingError("bound plan or bundle changed since compilation")
        return MaskedTransformerClientRuntime(self._bundle, remote)


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
    heads = kv_heads = head_dim = intermediate = 0
    for layer in layers:
        linears = {
            op["attributes"]["weight"]: op
            for op in prefill.values()
            if op["operator"] == "linear"
            and op.get("layer") == layer
            and isinstance(op.get("attributes"), dict)
            and isinstance(op["attributes"].get("weight"), str)
        }
        q_op = linears.get(f"model.layers.{layer}.self_attn.q_proj.weight")
        k_op = linears.get(f"model.layers.{layer}.self_attn.k_proj.weight")
        gate_op = linears.get(f"model.layers.{layer}.mlp.gate_proj.weight")
        if q_op is None or k_op is None or gate_op is None:
            raise RuntimeBindingError(f"plan layer {layer} is missing dense q/k/gate projections")

        def head_shape(source: dict[str, Any]) -> list[int]:
            consumers = [
                op for op in prefill.values()
                if op["operator"] == "reshape" and source["id"] in (op.get("inputs") or [])
            ]
            if len(consumers) != 1 or len(consumers[0].get("output_shape") or []) != 4:
                raise RuntimeBindingError("plan attention head reshape is malformed")
            return list(consumers[0]["output_shape"])

        q_shape = head_shape(q_op)
        k_shape = head_shape(k_op)
        if heads and (heads != q_shape[1] or kv_heads != k_shape[1] or head_dim != q_shape[-1]):
            raise RuntimeBindingError("plan attention geometry differs across layers")
        heads, kv_heads, head_dim = q_shape[1], k_shape[1], q_shape[-1]
        if heads * head_dim != _last_dim(q_op["output_shape"], "q_proj"):
            raise RuntimeBindingError("plan q projection width does not match heads")
        if kv_heads * head_dim != _last_dim(k_op["output_shape"], "k_proj"):
            raise RuntimeBindingError("plan k projection width does not match heads")
        width = _last_dim(gate_op["output_shape"], "gate_proj")
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
        if _require_int(manifest.get(key), f"manifest {key}") != expected:
            raise RuntimeBindingError(f"bundle manifest {key} does not match the model plan")

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
    for stage_id, stage in canonical.items():
        row = spec_rows.get(stage_id)
        if row is None:
            raise RuntimeBindingError(f"bundle stage {stage_id!r} has no manifest row")
        if row.get("role") != stage.role or row.get("layer_index") != stage.layer_index:
            raise RuntimeBindingError(f"bundle stage {stage_id!r} role or layer drifted")

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
            if (
                stage.client_weight is not None
                and stage.client_weight_layout in {"linear", "transposed_embedding"}
            ):
                digest = _sha256(
                    np.ascontiguousarray(stage.client_weight, dtype=np.int8).tobytes()
                )
                if digest != stage.weight_digest:
                    raise RuntimeBindingError(
                        f"bundle stage {stage.id!r} client weight digest mismatch"
                    )
                if stage.client_weight_scales is not None and not np.array_equal(
                    np.asarray(stage.client_weight_scales, dtype=np.float32), scales
                ):
                    raise RuntimeBindingError(
                        f"bundle stage {stage.id!r} client weight scales mismatch"
                    )
        elif stage.seeded_profile is None:
            raise RuntimeBindingError(f"bundle stage {stage.id!r} is missing a ring profile")

    actual_body, actual_commitment = _canonical_stage_fingerprints(canonical)
    if actual_body != body_fingerprint:
        raise RuntimeBindingError("bundle body fingerprint does not match stage metadata")
    if actual_commitment != stage_commitment:
        raise RuntimeBindingError("bundle stage commitment does not match stage metadata")

    stage_by_role = {(stage.role, stage.layer_index): stage for stage in canonical.values()}
    token_stage = stage_by_role[("token_lookup", None)]
    head_stage = stage_by_role[("lm_head", None)]

    stage_semantics: dict[str, set[str]] = {stage_id: set() for stage_id in canonical}
    local_operations: set[str] = set()
    norm_weights: dict[str, int] = {}
    output_sums: dict[str, int] = {stage_id: 0 for stage_id in canonical}
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
            output_sums[stage.id] += _last_dim(operation.get("output_shape"), operation_id)
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
    for stage_id, stage in canonical.items():
        if not stage_semantics[stage_id]:
            raise RuntimeBindingError(f"bundle stage {stage_id!r} has no semantic operations")
        per_phase = 2 * (output_sums[stage_id] // 2)
        if output_sums[stage_id] != per_phase or output_sums[stage_id] // 2 != stage.out_features:
            raise RuntimeBindingError(f"bundle stage {stage_id!r} fused output sum is inconsistent")
    for stage_id, qualified in stage_semantics.items():
        prefill_ids = {item.split(":", 1)[1] for item in qualified if item.startswith("prefill:")}
        decode_ids = {item.split(":", 1)[1] for item in qualified if item.startswith("decode:")}
        if prefill_ids != decode_ids or len(qualified) != len(prefill_ids) + len(decode_ids):
            raise RuntimeBindingError(f"stage {stage_id!r} prefill and decode mappings differ")

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
        ))
    bindings.sort(
        key=lambda item: (
            item.layer_index is None,
            item.layer_index or -1,
            item.role,
            item.stage_id,
        )
    )

    manifest_fingerprint = manifest.get("fingerprint")
    if not isinstance(manifest_fingerprint, str) or not manifest_fingerprint:
        manifest_fingerprint = _sha256(_canonical_json({
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "fingerprint"}
        }))
    fingerprint_payload = {
        "model_id": bundle.model_id,
        "manifest_fingerprint": manifest_fingerprint,
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
    }
    canonical_bytes = _canonical_json(spec)
    return CompiledRuntimeModel(
        plan=plan,
        bundle=bundle,
        canonical=canonical_bytes,
        digest=_binding_digest(canonical_bytes),
        fingerprint=fingerprint,
        stages=tuple(bindings),
        local_operations=tuple(sorted(local_operations)),
    )


__all__ = [
    "CompiledRuntimeModel",
    "RuntimeBindingError",
    "RuntimeStageBinding",
    "compile_runtime_model",
]
