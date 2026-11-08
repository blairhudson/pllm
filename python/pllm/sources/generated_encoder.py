"""Deterministic FP32 BERT classification source, without runtime or oracle code."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .generated_checkpoint import write_checkpoint as _write_checkpoint


def tiny_checkpoint() -> tuple[dict, dict[str, np.ndarray]]:
    """Return genuine HF source fields and out-in linear weights for B1/L2/H8."""
    config = {
        "model_type": "bert", "architectures": ["BertForSequenceClassification"],
        "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
        "intermediate_size": 13, "vocab_size": 17, "max_position_embeddings": 8,
        "type_vocab_size": 2, "position_embedding_type": "absolute",
        "hidden_act": "gelu", "layer_norm_eps": 1e-12,
        "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
        "classifier_dropout": 0.0, "is_decoder": False, "add_cross_attention": False,
        "pad_token_id": 0, "torch_dtype": "float32", "initializer_range": 0.02,
        "num_labels": 3, "problem_type": "single_label_classification",
        "id2label": {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        "label2id": {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
    }
    tensors: dict[str, np.ndarray] = {}

    def matrix(shape: tuple[int, ...], salt: int) -> np.ndarray:
        index = np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
        return (((index * (salt * 2 + 3) + salt * 11) % 61 - 30) / 128).astype(np.float32)

    def linear(prefix: str, outputs: int, inputs: int, salt: int) -> None:
        tensors[f"{prefix}.weight"] = matrix((outputs, inputs), salt)
        # Odd dyadic numerators keep every bias nonzero.
        tensors[f"{prefix}.bias"] = matrix((outputs,), salt + 37) + np.float32(1 / 256)

    def norm(prefix: str, salt: int) -> None:
        tensors[f"{prefix}.weight"] = np.array(
            [0.75, 1.125, 0.875, 1.25, 1.0625, 0.8125, 1.1875, 0.9375],
            dtype=np.float32,
        ) + np.float32(salt / 64)
        tensors[f"{prefix}.bias"] = matrix((8,), salt + 43) + np.float32(1 / 256)

    tensors["bert.embeddings.word_embeddings.weight"] = matrix((17, 8), 1)
    tensors["bert.embeddings.position_embeddings.weight"] = matrix((8, 8), 2)
    tensors["bert.embeddings.token_type_embeddings.weight"] = matrix((2, 8), 3)
    norm("bert.embeddings.LayerNorm", 0)
    for layer in range(2):
        prefix = f"bert.encoder.layer.{layer}"
        salt = 4 + layer * 7
        for offset, name in enumerate(("query", "key", "value")):
            linear(f"{prefix}.attention.self.{name}", 8, 8, salt + offset)
        linear(f"{prefix}.attention.output.dense", 8, 8, salt + 3)
        norm(f"{prefix}.attention.output.LayerNorm", 1 + layer * 2)
        linear(f"{prefix}.intermediate.dense", 13, 8, salt + 4)
        linear(f"{prefix}.output.dense", 8, 13, salt + 5)
        norm(f"{prefix}.output.LayerNorm", 2 + layer * 2)
    linear("bert.pooler.dense", 8, 8, 19)
    linear("classifier", 3, 8, 20)
    return config, tensors


def write_checkpoint(
    root: Path,
    config: dict | None = None,
    tensors: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write supplied source unchanged, defaulting missing inputs to tiny BERT."""
    if config is None or tensors is None:
        default_config, default_tensors = tiny_checkpoint()
        if config is None:
            config = default_config
        if tensors is None:
            tensors = default_tensors
    return _write_checkpoint(root, config, tensors)
