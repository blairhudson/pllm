"""Public model source declarations."""

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pllm.configuration import ConfigurationError, Model


@runtime_checkable
class ModelSource(Protocol):
    source: str
    kind: str
    model_id: str | None

    def to_spec(self) -> dict[str, Any]: ...


class TinyModel(Model):
    __slots__ = ()

    def __init__(self, family: str = "qwen2", *, model_id: str | None = None) -> None:
        if family != "qwen2":
            raise ConfigurationError("TinyModel currently supports only qwen2")
        super().__init__(family, kind="tiny", model_id=model_id)


class BundleModel(Model):
    __slots__ = ()

    def __init__(
        self,
        path: str,
        *,
        format: str = "huggingface",
        model_id: str | None = None,
    ) -> None:
        if format not in {
            "huggingface",
            "safetensors",
            "vllm",
            "mlx",
            "mlx-lm",
            "gguf",
            "llama.cpp",
        }:
            raise ConfigurationError(f"unsupported model bundle format {format!r}")
        super().__init__(
            path,
            kind=format,
            model_id=model_id,
            local_files_only=format in {"huggingface", "safetensors", "vllm"},
        )


def _materialize_tiny_model(path: Path) -> Path:
    import numpy as np
    from safetensors.numpy import save_file

    path.mkdir(parents=True, exist_ok=True)
    hidden, intermediate, heads, kv_heads, head_dim, vocab = 32, 64, 4, 2, 8, 258
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "name_or_path": "pllm-tiny-qwen2",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": 1024,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "attention_bias": True,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "pllm_test_tokenizer": "byte",
    }
    (path / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "model_max_length": 1024,
                "chat_template": (
                    "{% for message in messages %}{{ message['role'] }}: "
                    "{{ message['content'] }}\\n{% endfor %}assistant: "
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(17)

    def matrix(rows: int, columns: int, scale: float = 0.08) -> np.ndarray:
        return (rng.standard_normal((rows, columns), dtype=np.float32) * scale).astype(
            np.float32
        )

    prefix = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": matrix(vocab, hidden, 0.12),
        "model.norm.weight": np.ones(hidden, dtype=np.float32),
        f"{prefix}.self_attn.q_proj.weight": matrix(heads * head_dim, hidden),
        f"{prefix}.self_attn.q_proj.bias": np.zeros(heads * head_dim, dtype=np.float32),
        f"{prefix}.self_attn.k_proj.weight": matrix(kv_heads * head_dim, hidden),
        f"{prefix}.self_attn.k_proj.bias": np.zeros(kv_heads * head_dim, dtype=np.float32),
        f"{prefix}.self_attn.v_proj.weight": matrix(kv_heads * head_dim, hidden),
        f"{prefix}.self_attn.v_proj.bias": np.zeros(kv_heads * head_dim, dtype=np.float32),
        f"{prefix}.self_attn.o_proj.weight": matrix(hidden, heads * head_dim),
        f"{prefix}.mlp.gate_proj.weight": matrix(intermediate, hidden),
        f"{prefix}.mlp.up_proj.weight": matrix(intermediate, hidden),
        f"{prefix}.mlp.down_proj.weight": matrix(hidden, intermediate),
        f"{prefix}.input_layernorm.weight": np.ones(hidden, dtype=np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.ones(hidden, dtype=np.float32),
    }
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    return path


__all__ = ["BundleModel", "ModelSource", "TinyModel"]
