"""Independent generated-checkpoint oracle for the existing W8A8 source runtime.

Checkpoint generation reuses production helpers; numerical evaluation remains
independent, with no pllm execution kernels, imported stage values, or actual logits.
This is a numerical source/reference oracle, not an encrypted protocol oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pllm.sources.generated_absolute import (
    tiny_checkpoint as tiny_checkpoint,
    write_checkpoint as write_checkpoint,
)


def quantized_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """codec.rs::quantize: FP32 max/127, ties-even, symmetric [-127,127]."""
    value = np.asarray(value, dtype=np.float32)
    maxima = np.max(np.abs(value), axis=1)
    scales = np.where(maxima > 0, maxima / np.float32(127), np.float32(1)).astype(np.float32)
    # NumPy rint matches the inspected Rust round_ties_even, including negative ties.
    codes = np.clip(np.rint(value / scales[:, None]), -127, 127).astype(np.int8)
    return codes, scales


def linear(value: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """Weight is output/input oriented. Integer dot precedes two FP32 multiplies."""
    codes, scales = quantized_rows(weight)
    activation, row_scales = quantized_rows(value.reshape(-1, value.shape[-1]))
    integer = activation.astype(np.int64) @ codes.astype(np.int64).T
    result = integer.astype(np.float32) * row_scales[:, None]
    result = result * scales[None, :]
    if bias is not None:
        result = result + bias
    return result.reshape(value.shape[:-1] + (weight.shape[0],))


@dataclass(frozen=True)
class OracleResult:
    logits: np.ndarray
    keys: tuple[np.ndarray, ...]  # sequence, heads, feature
    values: tuple[np.ndarray, ...]

    @property
    def selected(self) -> int:
        return int(np.argmax(self.logits))


def evaluate(weights: dict[str, np.ndarray], tokens: list[int]) -> OracleResult:
    """Replay complete causal prefix from source tensors, including all two blocks.

    Decode reference intentionally recomputes the prefix, never consumes actual KV.
    Token lookup uses the existing transposed table's per-feature quantization;
    learned positions and affine normalization vectors remain source FP32.
    """
    if not tokens or len(tokens) > 8 or any(type(t) is not int or not 0 <= t < 11 for t in tokens):
        raise ValueError("oracle IDs outside token/position domain")
    table, scales = quantized_rows(weights["transformer.wte.weight"].T)
    # Existing transposed_embedding one-hot W8A8 lookup, including operation order.
    embedded = (table[:, tokens].T.astype(np.int32) * 127).astype(np.float32)
    hidden = embedded * np.float32(1.0 / 127) * scales[None, :]
    hidden = hidden + weights["transformer.wpe.weight"][:len(tokens)]
    keys, values = [], []

    def norm(value: np.ndarray, prefix: str) -> np.ndarray:
        centered = value - value.mean(axis=-1, keepdims=True, dtype=np.float32)
        variance = (centered * centered).mean(axis=-1, keepdims=True, dtype=np.float32)
        normalized = centered / np.sqrt(variance + np.float32(1e-5))
        return normalized * weights[prefix + ".weight"] + weights[prefix + ".bias"]

    def project(value: np.ndarray, prefix: str) -> np.ndarray:
        return linear(value, weights[prefix + ".weight"].T, weights[prefix + ".bias"])

    for layer in range(2):
        prefix = f"transformer.h.{layer}"
        qkv = project(norm(hidden, prefix + ".ln_1"), prefix + ".attn.c_attn")
        query, key, value = (part.reshape(len(tokens), 2, 2) for part in np.split(qkv, 3, axis=-1))
        keys.append(key.copy())
        values.append(value.copy())
        attended = np.empty_like(query)
        for row in range(len(tokens)):
            for head in range(2):
                scores = np.sum(query[row, head] * key[:row + 1, head], axis=-1, dtype=np.float32)
                scores = scores * np.float32(1.0 / math.sqrt(2))
                mass = np.exp(scores - scores.max()).astype(np.float32)
                probability = mass / mass.sum(dtype=np.float32)
                attended[row, head] = np.sum(probability[:, None] * value[:row + 1, head], axis=0, dtype=np.float32)
        hidden = hidden + project(attended.reshape(len(tokens), 4), prefix + ".attn.c_proj")
        up = project(norm(hidden, prefix + ".ln_2"), prefix + ".mlp.c_fc")
        argument = np.float32(math.sqrt(2 / math.pi)) * (
            up + np.float32(0.044715) * up * up * up
        )
        activated = (np.float32(0.5) * up) * (np.float32(1) + np.tanh(argument))
        hidden = hidden + project(activated, prefix + ".mlp.c_proj")
    last = norm(hidden, "transformer.ln_f")[-1:]
    logits = linear(last, weights["lm_head.weight"])[0]
    return OracleResult(logits, tuple(keys), tuple(values))
