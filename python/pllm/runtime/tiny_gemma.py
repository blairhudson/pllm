from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def create_tiny_gemma4_checkpoint(
    path: str | Path,
    *,
    seed: int = 17,
    vocab_size: int = 258,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 8,
    ple_dim: int = 0,
) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)

    def matrix(out_features: int, in_features: int, scale: float = 0.08) -> torch.Tensor:
        return torch.randn(out_features, in_features, generator=generator) * scale

    config = {
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4_text",
        "name_or_path": "tiny-gemma4-he",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "max_position_embeddings": 256,
        "sliding_window": 64,
        "layer_types": ["sliding_attention"] * max(0, num_hidden_layers - 1) + ["full_attention"],
        "hidden_activation": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "hidden_size_per_layer_input": ple_dim,
        "vocab_size_per_layer_input": vocab_size,
        "num_kv_shared_layers": 0,
        "attention_k_eq_v": False,
        "use_double_wide_mlp": False,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        "he_test_tokenizer": "byte",
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<bos>", "eos_token": "<eos>", "model_max_length": 256,
        "chat_template": "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\\n{% endfor %}assistant: ",
    }, indent=2) + "\n", encoding="utf-8")

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": matrix(vocab_size, hidden_size, 0.12),
        # Gemma 4 inherits Gemma3n RMSNorm, which stores the full scale.
        "model.norm.weight": torch.ones(hidden_size),
    }
    if ple_dim:
        tensors.update({
            "model.embed_tokens_per_layer.weight": matrix(vocab_size, num_hidden_layers * ple_dim, 0.1),
            "model.per_layer_model_projection.weight": matrix(num_hidden_layers * ple_dim, hidden_size),
            "model.per_layer_projection_norm.weight": torch.ones(ple_dim),
        })
    for index in range(num_hidden_layers):
        prefix = f"model.layers.{index}"
        q_width = num_attention_heads * head_dim
        kv_width = num_key_value_heads * head_dim
        tensors.update({
            f"{prefix}.self_attn.q_proj.weight": matrix(q_width, hidden_size),
            f"{prefix}.self_attn.k_proj.weight": matrix(kv_width, hidden_size),
            f"{prefix}.self_attn.v_proj.weight": matrix(kv_width, hidden_size),
            f"{prefix}.self_attn.o_proj.weight": matrix(hidden_size, q_width),
            f"{prefix}.self_attn.q_norm.weight": torch.ones(head_dim),
            f"{prefix}.self_attn.k_norm.weight": torch.ones(head_dim),
            f"{prefix}.mlp.gate_proj.weight": matrix(intermediate_size, hidden_size),
            f"{prefix}.mlp.up_proj.weight": matrix(intermediate_size, hidden_size),
            f"{prefix}.mlp.down_proj.weight": matrix(hidden_size, intermediate_size),
            f"{prefix}.input_layernorm.weight": torch.ones(hidden_size),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden_size),
            f"{prefix}.pre_feedforward_layernorm.weight": torch.ones(hidden_size),
            f"{prefix}.post_feedforward_layernorm.weight": torch.ones(hidden_size),
        })
        if ple_dim:
            tensors.update({
                f"{prefix}.per_layer_input_gate.weight": matrix(ple_dim, hidden_size),
                f"{prefix}.per_layer_projection.weight": matrix(hidden_size, ple_dim),
                f"{prefix}.post_per_layer_input_norm.weight": torch.ones(hidden_size),
            })
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    return root


def create_tiny_llama_checkpoint(
    path: str | Path,
    *,
    seed: int = 29,
    vocab_size: int = 258,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 8,
) -> Path:
    """Create a tiny Llama-compatible Safetensors checkpoint for integration tests."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)

    def matrix(out_features: int, in_features: int, scale: float = 0.08) -> torch.Tensor:
        return torch.randn(out_features, in_features, generator=generator) * scale

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "name_or_path": "tiny-llama-he",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "max_position_embeddings": 256,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "rope_theta": 10000.0,
        "he_test_tokenizer": "byte",
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "model_max_length": 256,
        "chat_template": "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\\n{% endfor %}assistant: ",
    }, indent=2) + "\n", encoding="utf-8")

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": matrix(vocab_size, hidden_size, 0.12),
        "model.norm.weight": torch.ones(hidden_size),
    }
    for index in range(num_hidden_layers):
        prefix = f"model.layers.{index}"
        q_width = num_attention_heads * head_dim
        kv_width = num_key_value_heads * head_dim
        tensors.update({
            f"{prefix}.self_attn.q_proj.weight": matrix(q_width, hidden_size),
            f"{prefix}.self_attn.k_proj.weight": matrix(kv_width, hidden_size),
            f"{prefix}.self_attn.v_proj.weight": matrix(kv_width, hidden_size),
            f"{prefix}.self_attn.o_proj.weight": matrix(hidden_size, q_width),
            f"{prefix}.mlp.gate_proj.weight": matrix(intermediate_size, hidden_size),
            f"{prefix}.mlp.up_proj.weight": matrix(intermediate_size, hidden_size),
            f"{prefix}.mlp.down_proj.weight": matrix(hidden_size, intermediate_size),
            f"{prefix}.input_layernorm.weight": torch.ones(hidden_size),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden_size),
        })
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    return root
