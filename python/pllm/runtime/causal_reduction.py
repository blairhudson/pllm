"""Bounded canonical-prefix float32 attention numeric reference.

Only dense full causal attention. Each reduction has the same contiguous input
shape for a given absolute query position, regardless of prefill partitioning.
This reference does not authorize cache lineage changes or Pipeline execution.
"""
import numpy as np

from .semantic_attention import SemanticAttentionError


def scores(query, keys, positions, group):
    if (query.dtype != np.float32 or keys.dtype != np.float32 or query.ndim != 4
            or keys.ndim != 4 or query.shape[0] != 1 or keys.shape[0] != 1
            or query.shape[1] != keys.shape[1] * group or query.shape[-1] != keys.shape[-1]
            or positions.shape != (query.shape[2],) or positions.dtype != np.int64
            or np.any(positions < 0) or np.any(positions >= keys.shape[2])
            or query.shape[1] * query.shape[2] * keys.shape[2] > 16_777_216
            or not np.all(np.isfinite(query)) or not np.all(np.isfinite(keys))):
        raise SemanticAttentionError("canonical score contract invalid")
    result = np.zeros((1, query.shape[1], query.shape[2], keys.shape[2]), np.float32)
    for row, position in enumerate(positions):
        count = int(position) + 1
        for head in range(query.shape[1]):
            products = np.ascontiguousarray(keys[0, head // group, :count] * query[0, head, row])
            result[0, head, row, :count] = np.sum(products, axis=-1, dtype=np.float32)
    return result


def softmax(masked_scores):
    if (masked_scores.dtype != np.float32 or masked_scores.ndim != 4
            or masked_scores.size > 16_777_216 or np.any(np.isnan(masked_scores))
            or np.any(np.isposinf(masked_scores))):
        raise SemanticAttentionError("canonical softmax contract invalid")
    result = np.zeros_like(masked_scores)
    for index in np.ndindex(masked_scores.shape[:-1]):
        source = masked_scores[index]
        count = int(np.count_nonzero(np.isfinite(source)))
        if count < 1 or not np.all(np.isfinite(source[:count])) or not np.all(np.isneginf(source[count:])):
            raise SemanticAttentionError("canonical softmax requires a contiguous causal prefix")
        current = np.ascontiguousarray(source[:count])
        exponentials = np.exp(current - np.max(current)).astype(np.float32)
        result[index][:count] = exponentials / np.sum(exponentials, dtype=np.float32)
    return result


def values(probabilities, cache, positions, group):
    if (probabilities.dtype != np.float32 or cache.dtype != np.float32
            or probabilities.ndim != 4 or cache.ndim != 4 or probabilities.shape[0] != 1
            or cache.shape[0] != 1 or probabilities.shape[1] != cache.shape[1] * group
            or probabilities.shape[-1] != cache.shape[2] or positions.shape != (probabilities.shape[2],)
            or probabilities.shape[1] * probabilities.shape[2] * cache.shape[-1] > 16_777_216
            or positions.dtype != np.int64 or np.any(positions < 0) or np.any(positions >= cache.shape[2])
            or not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(cache))):
        raise SemanticAttentionError("canonical value contract invalid")
    result = np.empty((*probabilities.shape[:3], cache.shape[-1]), np.float32)
    for row, position in enumerate(positions):
        count = int(position) + 1
        if np.any(probabilities[:, :, row, count:] != 0):
            raise SemanticAttentionError("canonical value contract has unmasked future keys")
        for head in range(probabilities.shape[1]):
            products = np.ascontiguousarray(cache[0, head // group, :count]
                                            * probabilities[0, head, row, :count, None])
            result[0, head, row] = np.sum(products, axis=0, dtype=np.float32)
    return result
