"""Client-local causal mask semantics for validated decoder attention domains."""

from __future__ import annotations

from typing import Any

import numpy as np


class SemanticAttentionError(ValueError):
    pass


_MAX_ELEMENTS = 16_777_216
_FULL_LAYOUT = "batch_kv_heads_sequence_feature"
_WINDOW_LAYOUT = "batch_kv_heads_query_window_feature"


def _grouped_heads(query: np.ndarray, kv_heads: int, group: int) -> np.ndarray:
    if (
        query.dtype != np.float32
        or query.ndim != 4
        or query.shape[0] != 1
        or min(query.shape) <= 0
        or type(group) is not int
        or group < 1
        or query.shape[1] != kv_heads * group
        or not np.all(np.isfinite(query))
    ):
        raise SemanticAttentionError("attention head grouping is invalid")
    return query.reshape(1, kv_heads, group, query.shape[2], query.shape[3])


def semantic_attention_scores(
    query: np.ndarray, keys: np.ndarray, *, group: int, layout: str
) -> np.ndarray:
    """One batch-one score tensor for full keys or a per-query key window."""
    if keys.dtype != np.float32 or keys.ndim not in {4, 5} or keys.shape[0] != 1 or min(keys.shape) <= 0:
        raise SemanticAttentionError("attention keys must be nonempty float32")
    grouped = _grouped_heads(query, keys.shape[1], group)
    if layout == _FULL_LAYOUT and keys.ndim == 4:
        width = keys.shape[2]
    elif layout == _WINDOW_LAYOUT and keys.ndim == 5 and keys.shape[2] == query.shape[2]:
        width = keys.shape[3]
    else:
        raise SemanticAttentionError("attention key layout is unsupported")
    if keys.shape[-1] != query.shape[-1] or not np.all(np.isfinite(keys)):
        raise SemanticAttentionError("attention key features are invalid")
    if query.shape[1] * query.shape[2] * width > _MAX_ELEMENTS:
        raise SemanticAttentionError("attention score tensor exceeds the local bound")
    if layout == _FULL_LAYOUT:
        scores = np.einsum("bkgqd,bktd->bkgqt", grouped, keys, optimize=False)
    else:
        scores = np.einsum("bkgqd,bkqwd->bkgqw", grouped, keys, optimize=False)
    if not np.all(np.isfinite(scores)):
        raise SemanticAttentionError("attention scores exceed the numeric domain")
    return scores.reshape(1, query.shape[1], query.shape[2], width)


def semantic_attention_values(
    probabilities: np.ndarray, values: np.ndarray, *, group: int, layout: str
) -> np.ndarray:
    """One batch-one weighted value tensor for full keys or per-query windows."""
    if values.dtype != np.float32 or values.ndim not in {4, 5} or values.shape[0] != 1 or min(values.shape) <= 0:
        raise SemanticAttentionError("attention values must be nonempty float32")
    grouped = _grouped_heads(probabilities, values.shape[1], group)
    if layout == _FULL_LAYOUT and values.ndim == 4:
        width = values.shape[2]
    elif layout == _WINDOW_LAYOUT and values.ndim == 5 and values.shape[2] == probabilities.shape[2]:
        width = values.shape[3]
    else:
        raise SemanticAttentionError("attention value layout is unsupported")
    if probabilities.shape[-1] != width or not np.all(np.isfinite(values)):
        raise SemanticAttentionError("attention value shape is invalid")
    if probabilities.shape[1] * probabilities.shape[2] * values.shape[-1] > _MAX_ELEMENTS:
        raise SemanticAttentionError("attention output tensor exceeds the local bound")
    if layout == _FULL_LAYOUT:
        output = np.einsum("bkgqt,bktd->bkgqd", grouped, values, optimize=False)
    else:
        output = np.einsum("bkgqw,bkqwd->bkgqd", grouped, values, optimize=False)
    if not np.all(np.isfinite(output)):
        raise SemanticAttentionError("attention output exceeds the numeric domain")
    return output.reshape(1, probabilities.shape[1], probabilities.shape[2], values.shape[-1])


def mask_causal_scores(
    scores: np.ndarray,
    positions: np.ndarray,
    attributes: dict[str, Any],
    *,
    padding_mask: np.ndarray | None = None,
    valid_lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Mask full-capacity or query-relative windows using absolute client positions.

    This is one numeric operator, not authorization to execute a complete state
    schedule: key/value layouts and state ownership are admitted separately.
    """
    if scores.dtype != np.float32 or scores.ndim != 4 or scores.shape[0] != 1:
        raise SemanticAttentionError("causal scores require a batch-one float32 tensor")
    if not np.all(np.isfinite(scores)):
        raise SemanticAttentionError("causal scores must be finite before masking")
    if (
        positions.dtype != np.int64
        or positions.ndim != 1
        or positions.shape[0] != scores.shape[2]
        or positions.size == 0
        or np.any(positions < 0)
        or np.any(np.diff(positions) != 1)
    ):
        raise SemanticAttentionError("causal query positions must be contiguous and nonnegative")
    indices = np.arange(scores.shape[-1], dtype=np.int64)[None, None, None, :]
    query_positions = positions[None, None, :, None]
    if not attributes:
        # The dense baseline has only an absolute-position causal contract.
        allowed = indices <= query_positions
    else:
        if (
            attributes.get("absolute_positions_input") != "input.positions"
            or attributes.get("padding_mask_input") != "input.attention_mask"
            or attributes.get("valid_lengths_input") != "input.sequence_lengths"
            or padding_mask is None
            or padding_mask.dtype != np.bool_
            or padding_mask.shape != (1, positions.size)
            or not np.all(padding_mask)
            or valid_lengths is None
            or valid_lengths.dtype != np.int64
            or valid_lengths.shape != (1,)
            or valid_lengths[0] < positions[-1] + 1
        ):
            raise SemanticAttentionError("causal mask lacks valid client-side bounds")
        if attributes.get("kind") == "full_causal":
            maximum = attributes.get("maximum_position_embeddings")
            if (
                type(maximum) is not int
                or maximum < scores.shape[-1]
                or valid_lengths[0] > scores.shape[-1]
                or attributes.get("key_domain") != "fixed_capacity"
                or attributes.get("cache_validity") != "valid_lengths_fixed_capacity"
            ):
                raise SemanticAttentionError("full attention capacity contract is unsupported")
            allowed = (indices <= query_positions) & (indices < valid_lengths[0])
        elif attributes.get("kind") == "sliding_causal":
            window = attributes.get("sliding_window")
            if (
                type(window) is not int
                or window < 1
                or window != scores.shape[-1]
                or attributes.get("left_context") != window - 1
                or attributes.get("includes_current") is not True
                or attributes.get("key_domain") != "query_relative_window"
                or attributes.get("cache_validity") != "valid_lengths_bounded_suffix"
            ):
                raise SemanticAttentionError("sliding attention window contract is unsupported")
            key_positions = query_positions - (window - 1) + indices
            allowed = (key_positions >= 0) & (key_positions <= query_positions)
            allowed &= key_positions < valid_lengths[0]
        else:
            raise SemanticAttentionError("causal attention policy is unsupported")
    return np.where(allowed, scores, np.float32(-np.inf))


__all__ = [
    "SemanticAttentionError",
    "mask_causal_scores",
    "semantic_attention_scores",
    "semantic_attention_values",
]
