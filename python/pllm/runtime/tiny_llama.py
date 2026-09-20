from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def create_tiny_llama_checkpoint(
    path: str | Path,
    *,
    seed: int = 23,
    vocab_size: int = 258,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 8,
    with_qkv_bias: bool = True,
    model_type: str = "qwen2",
    qk_norm: bool = False,
) -> Path:
    """Create a deterministic Qwen/Llama-compatible checkpoint for integration tests."""

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)

    def matrix(out_features: int, in_features: int, scale: float = 0.08) -> torch.Tensor:
        return torch.randn(out_features, in_features, generator=generator) * scale

    def vector(width: int, scale: float = 0.02) -> torch.Tensor:
        return torch.randn(width, generator=generator) * scale

    architecture = "Qwen3ForCausalLM" if model_type == "qwen3" else "Qwen2ForCausalLM"
    config = {
        "architectures": [architecture],
        "model_type": model_type,
        "name_or_path": f"tiny-{model_type}-pllm",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "max_position_embeddings": 256,
        "sliding_window": None,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "rope_theta": 10000.0,
        "attention_bias": with_qkv_bias,
        "attention_dropout": 0.0,
        "rope_scaling": None,
        "use_cache": True,
        "use_sliding_window": False,
        "max_window_layers": num_hidden_layers,
        "layer_types": ["full_attention"] * num_hidden_layers,
        "pllm_test_tokenizer": "byte",
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "model_max_length": 256,
                "chat_template": "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\\n{% endfor %}assistant: ",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": matrix(vocab_size, hidden_size, 0.12),
        "model.norm.weight": torch.ones(hidden_size),
    }
    for index in range(num_hidden_layers):
        prefix = f"model.layers.{index}"
        q_width = num_attention_heads * head_dim
        kv_width = num_key_value_heads * head_dim
        tensors.update(
            {
                f"{prefix}.self_attn.q_proj.weight": matrix(q_width, hidden_size),
                f"{prefix}.self_attn.k_proj.weight": matrix(kv_width, hidden_size),
                f"{prefix}.self_attn.v_proj.weight": matrix(kv_width, hidden_size),
                f"{prefix}.self_attn.o_proj.weight": matrix(hidden_size, q_width),
                f"{prefix}.mlp.gate_proj.weight": matrix(intermediate_size, hidden_size),
                f"{prefix}.mlp.up_proj.weight": matrix(intermediate_size, hidden_size),
                f"{prefix}.mlp.down_proj.weight": matrix(hidden_size, intermediate_size),
                f"{prefix}.input_layernorm.weight": torch.ones(hidden_size),
                f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden_size),
            }
        )
        if with_qkv_bias:
            tensors.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": vector(q_width),
                    f"{prefix}.self_attn.k_proj.bias": vector(kv_width),
                    f"{prefix}.self_attn.v_proj.bias": vector(kv_width),
                }
            )
        if qk_norm:
            tensors.update(
                {
                    f"{prefix}.self_attn.q_norm.weight": torch.ones(head_dim),
                    f"{prefix}.self_attn.k_norm.weight": torch.ones(head_dim),
                }
            )
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    return root
