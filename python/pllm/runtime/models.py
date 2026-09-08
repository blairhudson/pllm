from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class StageSpec:
    id: str
    op: Literal["linear", "embedding", "lm_head"]
    in_features: int
    out_features: int
    weight_bits: int = 4
    activation_bits: int = 4
    fused_from: tuple[str, ...] = ()
    local_after: tuple[str, ...] = ()
    weight_keys: tuple[str, ...] = ()
    bias_keys: tuple[str, ...] = ()
    layer_index: int | None = None
    role: str | None = None
    transpose_weight: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class ModelManifest:
    id: str
    architecture: str
    source_format: str
    source: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    context_length: int
    tokenizer_path: str | None = None
    chat_template: str | None = None
    quantization: dict[str, Any] | None = None
    tied_embeddings: bool = False
    stages: list[StageSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def fingerprint(self) -> str:
        # Creation time is transport metadata, not model identity. Excluding it
        # keeps manifests stable across repeated inspection/import runs.
        payload = self.to_dict(include_fingerprint=False)
        payload.pop("created_at", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "id": self.id,
            "architecture": self.architecture,
            "source_format": self.source_format,
            "source": self.source,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "context_length": self.context_length,
            "tokenizer_path": self.tokenizer_path,
            "chat_template": self.chat_template,
            "quantization": self.quantization,
            "tied_embeddings": self.tied_embeddings,
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelManifest":
        value = dict(value)
        value.pop("fingerprint", None)
        value["stages"] = [StageSpec(**row) for row in value.get("stages", [])]
        return cls(**value)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ModelManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))



def gemma4_stage_plan(config: dict[str, Any], *, include_lm_head: bool = False) -> list[StageSpec]:
    """Build the exact remote-linear surface for Hugging Face Gemma 4 text checkpoints.

    The plan follows the official Gemma 4 implementation: fused Q/K/V where KV
    is not shared, Q-only stages in late KV-sharing layers, fused SwiGLU
    gate/up projections, PLE projections, and optional tied vocabulary head.
    """

    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    default_kv_heads = int(config.get("num_key_value_heads", heads))
    default_head_dim = int(config.get("head_dim", hidden // heads))
    layer_types = list(config.get("layer_types") or ["full_attention"] * layers)
    if len(layer_types) != layers:
        raise ValueError("Gemma 4 layer_types length does not match num_hidden_layers")
    shared_count = int(config.get("num_kv_shared_layers", 0) or 0)
    first_shared = layers - shared_count
    attention_k_eq_v = bool(config.get("attention_k_eq_v", False))
    double_wide = bool(config.get("use_double_wide_mlp", False))
    ple = int(config.get("hidden_size_per_layer_input", 0) or 0)
    per_layer = config.get("per_layer_config") or {}

    def layer_value(index: int, key: str, default: int) -> int:
        candidates = (str(index), index, layer_types[index])
        for candidate in candidates:
            row = per_layer.get(candidate) if isinstance(per_layer, dict) else None
            if isinstance(row, dict) and row.get(key) is not None:
                return int(row[key])
        return default

    stages: list[StageSpec] = []
    if ple:
        stages.extend([
            StageSpec(
                id="embed_tokens_per_layer",
                op="embedding",
                in_features=int(config.get("vocab_size_per_layer_input", config["vocab_size"])),
                out_features=layers * ple,
                weight_keys=("embed_tokens_per_layer.weight",),
                transpose_weight=True,
                role="ple_token_embedding",
                local_after=("reshape_ple", "ple_embedding_scale"),
            ),
            StageSpec(
            id="model.per_layer_model_projection",
            op="linear",
            in_features=hidden,
            out_features=layers * ple,
            weight_keys=("model.per_layer_model_projection.weight",),
            role="ple_context_projection",
            local_after=("reshape_ple", "ple_rms_norm", "combine_token_ple"),
            ),
        ])

    for index in range(layers):
        prefix = f"layers.{index}"
        head_dim = layer_value(index, "head_dim", default_head_dim)
        kv_heads = layer_value(index, "num_key_value_heads", default_kv_heads)
        q_width = heads * head_dim
        kv_width = kv_heads * head_dim
        shared_kv = shared_count > 0 and index >= first_shared
        if shared_kv:
            stages.append(StageSpec(
                id=f"layers.{index}.self_attn.q_proj",
                op="linear",
                in_features=hidden,
                out_features=q_width,
                weight_keys=(f"{prefix}.self_attn.q_proj.weight",),
                layer_index=index,
                role="q_proj_shared_kv",
                local_after=("q_norm", "rope", "attention"),
            ))
        else:
            keys = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
            ]
            out = q_width + kv_width
            fused = ["q_proj", "k_proj"]
            if not (attention_k_eq_v and layer_types[index] == "full_attention"):
                keys.append(f"{prefix}.self_attn.v_proj.weight")
                out += kv_width
                fused.append("v_proj")
            stages.append(StageSpec(
                id=f"layers.{index}.self_attn.qkv_proj",
                op="linear",
                in_features=hidden,
                out_features=out,
                fused_from=tuple(fused),
                weight_keys=tuple(keys),
                layer_index=index,
                role="qkv_proj",
                local_after=("qkv_norm", "rope", "attention"),
            ))
        stages.append(StageSpec(
            id=f"layers.{index}.self_attn.o_proj",
            op="linear",
            in_features=q_width,
            out_features=hidden,
            weight_keys=(f"{prefix}.self_attn.o_proj.weight",),
            layer_index=index,
            role="attention_output",
            local_after=("post_attention_rms_norm", "residual"),
        ))
        mlp_width = intermediate * (2 if double_wide and shared_kv else 1)
        stages.append(StageSpec(
            id=f"layers.{index}.mlp.gate_up_proj",
            op="linear",
            in_features=hidden,
            out_features=2 * mlp_width,
            fused_from=("gate_proj", "up_proj"),
            weight_keys=(
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
            ),
            layer_index=index,
            role="mlp_gate_up",
            local_after=("silu", "multiply"),
        ))
        stages.append(StageSpec(
            id=f"layers.{index}.mlp.down_proj",
            op="linear",
            in_features=mlp_width,
            out_features=hidden,
            weight_keys=(f"{prefix}.mlp.down_proj.weight",),
            layer_index=index,
            role="mlp_down",
            local_after=("post_feedforward_rms_norm", "residual"),
        ))
        if ple:
            stages.extend([
                StageSpec(
                    id=f"layers.{index}.per_layer_input_gate",
                    op="linear",
                    in_features=hidden,
                    out_features=ple,
                    weight_keys=(f"{prefix}.per_layer_input_gate.weight",),
                    layer_index=index,
                    role="ple_gate",
                    local_after=("activation", "multiply_ple"),
                ),
                StageSpec(
                    id=f"layers.{index}.per_layer_projection",
                    op="linear",
                    in_features=ple,
                    out_features=hidden,
                    weight_keys=(f"{prefix}.per_layer_projection.weight",),
                    layer_index=index,
                    role="ple_projection",
                    local_after=("post_ple_rms_norm", "residual"),
                ),
            ])
    if include_lm_head:
        key = "lm_head.weight"
        stages.append(StageSpec(
            id="lm_head",
            op="lm_head",
            in_features=hidden,
            out_features=int(config["vocab_size"]),
            weight_keys=(key,),
            role="lm_head",
        ))
    return stages

def transformer_stage_plan(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    vocab_size: int,
    fuse_qkv: bool = True,
    fuse_gate_up: bool = True,
    include_lm_head: bool = True,
    include_embedding: bool = False,
) -> list[StageSpec]:
    stages: list[StageSpec] = []
    if include_embedding:
        stages.append(StageSpec(
            id="embed_tokens",
            op="embedding",
            in_features=vocab_size,
            out_features=hidden_size,
            weight_keys=("embed_tokens.weight",),
            transpose_weight=True,
            local_after=("embedding_scale",),
        ))
    q_width = num_attention_heads * head_dim
    kv_width = num_key_value_heads * head_dim
    for layer in range(num_hidden_layers):
        prefix = f"layers.{layer}"
        if fuse_qkv:
            stages.append(StageSpec(
                id=f"{prefix}.self_attn.qkv_proj",
                op="linear",
                in_features=hidden_size,
                out_features=q_width + 2 * kv_width,
                fused_from=("q_proj", "k_proj", "v_proj"),
                weight_keys=(
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                ),
                local_after=("qk_norm", "rope", "attention"),
            ))
        else:
            stages.extend([
                StageSpec(f"{prefix}.self_attn.q_proj", "linear", hidden_size, q_width, weight_keys=(f"{prefix}.self_attn.q_proj.weight",)),
                StageSpec(f"{prefix}.self_attn.k_proj", "linear", hidden_size, kv_width, weight_keys=(f"{prefix}.self_attn.k_proj.weight",)),
                StageSpec(f"{prefix}.self_attn.v_proj", "linear", hidden_size, kv_width, weight_keys=(f"{prefix}.self_attn.v_proj.weight",)),
            ])
        stages.append(StageSpec(
            id=f"{prefix}.self_attn.o_proj",
            op="linear",
            in_features=q_width,
            out_features=hidden_size,
            weight_keys=(f"{prefix}.self_attn.o_proj.weight",),
            local_after=("residual", "rms_norm"),
        ))
        if fuse_gate_up:
            stages.append(StageSpec(
                id=f"{prefix}.mlp.gate_up_proj",
                op="linear",
                in_features=hidden_size,
                out_features=2 * intermediate_size,
                fused_from=("gate_proj", "up_proj"),
                weight_keys=(
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                ),
                local_after=("activation", "multiply"),
            ))
        else:
            stages.extend([
                StageSpec(f"{prefix}.mlp.gate_proj", "linear", hidden_size, intermediate_size, weight_keys=(f"{prefix}.mlp.gate_proj.weight",)),
                StageSpec(f"{prefix}.mlp.up_proj", "linear", hidden_size, intermediate_size, weight_keys=(f"{prefix}.mlp.up_proj.weight",)),
            ])
        stages.append(StageSpec(
            id=f"{prefix}.mlp.down_proj",
            op="linear",
            in_features=intermediate_size,
            out_features=hidden_size,
            weight_keys=(f"{prefix}.mlp.down_proj.weight",),
            local_after=("residual",),
        ))
    if include_lm_head:
        stages.append(StageSpec(
            "lm_head", "lm_head", hidden_size, vocab_size,
            weight_keys=("lm_head.weight", "embed_tokens.weight"),
        ))
    return stages
