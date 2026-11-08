"""Deterministic public checkpoint fixture with learned absolute positions.

Source generation only; numerical reference evaluation belongs outside this module.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .generated_checkpoint import write_checkpoint as _write_checkpoint


def write_checkpoint(directory: Path, config: dict, tensors: dict[str, np.ndarray]) -> Path:
    """Write source files; synchronize config special IDs before callers lower it."""
    alphabet = " abcdefgh"
    vocabulary = len(alphabet) + 2
    if config["vocab_size"] != vocabulary:
        raise ValueError("tiny checkpoint tokenizer requires vocabulary 11")
    special_ids = {"bos_token_id": len(alphabet), "eos_token_id": len(alphabet) + 1,
                   "pad_token_id": len(alphabet) + 1}
    config.update(special_ids)
    descriptor = {
        "type": "alphabet", "alphabet": alphabet, "vocab_size": vocabulary,
        **special_ids,
        "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
    }
    return _write_checkpoint(directory, config, tensors, tokenizer_descriptor=descriptor)


def tiny_checkpoint() -> tuple[dict, dict[str, np.ndarray]]:
    """Real source spelling deliberately has n_positions, not its canonical alias."""
    config = {
        "model_type": "gpt2", "architectures": ["GPT2LMHeadModel"],
        "n_embd": 4, "n_head": 2, "n_layer": 2, "n_inner": 7,
        "n_positions": 8, "vocab_size": 11, "layer_norm_epsilon": 1e-5,
        "activation_function": "gelu_new", "tie_word_embeddings": False,
        "attn_pdrop": 0.0, "embd_pdrop": 0.0, "resid_pdrop": 0.0,
    }
    weights: dict[str, np.ndarray] = {}

    def matrix(shape: tuple[int, ...], salt: int) -> np.ndarray:
        index = np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
        # Nonzero, nonsymmetric, dyadic coefficients; no random/model dependency.
        return (((index * (salt * 2 + 3) + salt * 11) % 61 - 30) / 128).astype(np.float32)

    weights["transformer.wte.weight"] = matrix((11, 4), 1)
    weights["transformer.wpe.weight"] = matrix((8, 4), 2)
    weights["lm_head.weight"] = matrix((11, 4), 3)
    for layer in range(2):
        prefix = f"transformer.h.{layer}"
        for index, norm in enumerate(("ln_1", "ln_2")):
            weights[f"{prefix}.{norm}.weight"] = np.array(
                [0.75, 1.125, 0.875, 1.25], dtype=np.float32,
            ) + np.float32((layer * 2 + index) / 32)
            weights[f"{prefix}.{norm}.bias"] = np.array(
                [-0.125, 0.0625, 0.1875, -0.03125], dtype=np.float32,
            ) + np.float32((layer + index) / 64)
        for index, (name, shape) in enumerate((
            ("attn.c_attn", (4, 12)), ("attn.c_proj", (4, 4)),
            ("mlp.c_fc", (4, 7)), ("mlp.c_proj", (7, 4)),
        )):
            weights[f"{prefix}.{name}.weight"] = matrix(shape, 4 + layer * 4 + index)
            weights[f"{prefix}.{name}.bias"] = matrix((shape[1],), 20 + layer * 4 + index)
    weights["transformer.ln_f.weight"] = np.array([1.125, 0.875, 1.25, 0.75], np.float32)
    weights["transformer.ln_f.bias"] = np.array([0.0625, -0.125, 0.03125, 0.15625], np.float32)
    return config, weights
