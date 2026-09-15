from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from .loaders import load_hf_directory
from .models import ModelManifest, StageSpec
from .safetensors_store import SafeTensorStore


DEFAULT_TINY_ALPHABET = " private\n"


def build_tiny_gemma_checkpoint(
    path: str | Path,
    *,
    model_id: str = "tiny-gemma-pllm",
    alphabet: str = DEFAULT_TINY_ALPHABET,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_hidden_layers: int = 2,
    seed: int = 7,
) -> Path:
    """Create a deterministic sharded Safetensors checkpoint for E2E tests.

    The residual path encodes a robust byte-sized next-token cycle while every
    Transformer projection is non-zero, so the test exercises embedding, QKV,
    attention output, gated MLP, down projection and LM head stages.
    """

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    vocab_size = len(alphabet) + 2
    if hidden_size < vocab_size:
        raise ValueError("hidden_size must be at least tokenizer vocabulary size")
    rng = np.random.default_rng(seed)
    config = {
        "architectures": ["PLLMGemmaForCausalLM"],
        "model_type": "pllm_gemma",
        "name_or_path": model_id,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": hidden_size // 2,
        "max_position_embeddings": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6,
        "rms_norm_centered": False,
        "embedding_multiplier": 1.0,
        "attention_scaling": (hidden_size // 2) ** -0.5,
        "layer_types": ["full_attention"] * num_hidden_layers,
        "tie_word_embeddings": False,
        "bos_token_id": len(alphabet),
        "eos_token_id": len(alphabet) + 1,
        "final_logit_softcapping": None,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (root / "pllm_tokenizer.json").write_text(
        json.dumps({"type": "alphabet", "alphabet": alphabet}, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, indent=2) + "\n",
        encoding="utf-8",
    )

    tensors: dict[str, np.ndarray] = {}
    prefix = "model.language_model"
    embedding = np.zeros((vocab_size, hidden_size), dtype=np.float32)
    embedding[:, :vocab_size] = np.eye(vocab_size, dtype=np.float32)
    tensors[f"{prefix}.embed_tokens.weight"] = embedding

    for layer in range(num_hidden_layers):
        lp = f"{prefix}.layers.{layer}"
        head_dim = hidden_size // 2
        tensors[f"{lp}.self_attn.q_proj.weight"] = rng.normal(0, 0.035, (hidden_size, hidden_size)).astype(np.float32)
        tensors[f"{lp}.self_attn.k_proj.weight"] = rng.normal(0, 0.035, (head_dim, hidden_size)).astype(np.float32)
        tensors[f"{lp}.self_attn.v_proj.weight"] = rng.normal(0, 0.035, (head_dim, hidden_size)).astype(np.float32)
        tensors[f"{lp}.self_attn.o_proj.weight"] = rng.normal(0, 0.004, (hidden_size, hidden_size)).astype(np.float32)
        tensors[f"{lp}.mlp.gate_proj.weight"] = rng.normal(0, 0.04, (intermediate_size, hidden_size)).astype(np.float32)
        tensors[f"{lp}.mlp.up_proj.weight"] = rng.normal(0, 0.04, (intermediate_size, hidden_size)).astype(np.float32)
        tensors[f"{lp}.mlp.down_proj.weight"] = rng.normal(0, 0.003, (hidden_size, intermediate_size)).astype(np.float32)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            tensors[f"{lp}.{name}.weight"] = np.ones(hidden_size, dtype=np.float32)
        tensors[f"{lp}.self_attn.q_norm.weight"] = np.ones(head_dim, dtype=np.float32)
        tensors[f"{lp}.self_attn.k_norm.weight"] = np.ones(head_dim, dtype=np.float32)
    tensors[f"{prefix}.norm.weight"] = np.ones(hidden_size, dtype=np.float32)

    lm_head = np.zeros((vocab_size, hidden_size), dtype=np.float32)
    mapping = {char: index for index, char in enumerate(alphabet)}
    phrase = "private\n"
    transitions: dict[int, int] = {}
    for left, right in zip(phrase, phrase[1:]):
        transitions[mapping[left]] = mapping[right]
    transitions[mapping["\n"]] = mapping["p"]
    transitions[mapping[" "]] = mapping["p"]
    transitions[config["bos_token_id"]] = mapping["p"]
    transitions[config["eos_token_id"]] = config["eos_token_id"]
    for source, target in transitions.items():
        lm_head[target, source] = 7.0
    # A small negative baseline avoids ties without changing the intended cycle.
    lm_head -= 0.01
    tensors[f"{prefix}.lm_head.weight"] = lm_head

    keys = sorted(tensors)
    split = max(1, len(keys) // 2)
    shards = [keys[:split], keys[split:]]
    weight_map: dict[str, str] = {}
    for index, shard_keys in enumerate(shards, start=1):
        name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        save_file({key: tensors[key] for key in shard_keys}, root / name)
        weight_map.update({key: name for key in shard_keys})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(v.nbytes for v in tensors.values())}, "weight_map": weight_map}, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


class ClearStageCaller:
    def __init__(self, path: str | Path, manifest: ModelManifest) -> None:
        self.store = SafeTensorStore(path)
        self.stages = {stage.id: self._matrix(stage) for stage in manifest.stages}

    def _matrix(self, stage: StageSpec) -> np.ndarray:
        keys = stage.weight_keys
        if stage.op == "lm_head":
            for key in keys:
                try:
                    matrix = self.store.get_linear((key,))
                    break
                except Exception:
                    continue
            else:
                raise ValueError("LM head not found")
        else:
            parts = [self.store.get_linear((key,)) for key in keys]
            matrix = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        return matrix.T if stage.transpose_weight else matrix

    def __call__(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        matrix = self.stages[stage_id]
        return np.asarray(activation, dtype=np.float32) @ matrix.T


def tiny_manifest(path: str | Path, *, model_id: str = "tiny-gemma-pllm") -> ModelManifest:
    return load_hf_directory(path, model_id=model_id)
