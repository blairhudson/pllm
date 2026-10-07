"""Physical sparse ownership, original-position semantics and independent oracles."""

import gc
import math
import weakref

import numpy as np
import pytest

from pllm import _native
from pllm.runtime.retained_state import RetainedLayerCache, static_retention
from pllm.runtime.semantic_attention import (
    mask_retained_scores, semantic_attention_scores, semantic_attention_values,
)
from pllm.runtime.transformer_client import LayerCache


def fixture(rows=64):
    key = ((np.arange(rows * 8) % 37).astype(np.float32).reshape(rows, 2, 4) - 18) / 8
    return key, np.flip(key, axis=0).copy()


def test_owner_releases_dense_source_and_prices_reserved_decode_capacity():
    key, value = fixture()
    key[0, 0, 0] = -0.0
    expected_k, expected_v = key[[0, 17, 63]].copy(), value[[0, 17, 63]].copy()
    refs = weakref.ref(key), weakref.ref(value)
    cache = RetainedLayerCache(key, value, np.array([0, 17, 63]), capacity=5, maximum_position=66)
    assert not isinstance(cache, LayerCache)
    del key, value
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert cache.allocated_bytes == 5 * (2 * 2 * 4 * 4 + 4)
    assert cache.allocated_bytes < 64 * 2 * 2 * 4 * 4
    assert cache.position == 64 and cache.length == 3
    k, v, p = cache.gather()
    np.testing.assert_array_equal(k.view(np.uint32), expected_k.view(np.uint32))
    np.testing.assert_array_equal(v.view(np.uint32), expected_v.view(np.uint32))
    np.testing.assert_array_equal(p, [0, 17, 63])
    assert not k.flags.writeable and not v.flags.writeable and not p.flags.writeable
    tail_k, tail_v = fixture(2)
    cache.append(64, tail_k, tail_v)
    cache.retain(np.array([17, 65]), capacity=2)
    assert cache.allocated_bytes == 2 * (2 * 2 * 4 * 4 + 4)
    assert cache.position == 66
    cache.close()
    assert cache.allocated_bytes == 0
    np.testing.assert_array_equal(k.view(np.uint32), expected_k.view(np.uint32))
    with pytest.raises(ValueError):
        cache.gather()
    cache.close()


def test_static_selection_matches_independent_stable_ranking():
    scores = np.array([0.25, 0.5, 0.5, 0.125, 0.0, 0.25, 0.5, 0.0], np.float32)
    for keep in range(1, 9):
        for recent in range(keep + 1):
            ranked = sorted(range(8 - recent), key=lambda i: (-float(scores[i]), i))[:keep - recent]
            expected = sorted(ranked + list(range(8 - recent, 8)))
            np.testing.assert_array_equal(static_retention(scores, keep=keep, recent=recent), expected)


def cluster_oracle(key, query, positions, cluster_rows, keep, alpha):
    groups = {}
    for position in positions:
        groups.setdefault(int(position) // cluster_rows, []).append(int(position))
    groups = list(groups.values())
    scores = []
    for group in groups:
        score = 0.0
        for col, q in enumerate(query.flat):
            vals = [float(key[p].flat[col]) for p in group]
            lo, hi = min(vals) * float(q), max(vals) * float(q)
            if alpha is None:
                score += max(lo, hi)
            else:
                a = float(np.float32(alpha))
                # MPCache equation (3) weights KEY extrema before query
                # multiplication. It is not a weighted sign-aware upper bound.
                score += a * hi + (1 - a) * lo
        scores.append(score)
    winners = sorted(range(len(groups)), key=lambda i: (-scores[i], i))[:keep]
    return sorted(p for i in winners for p in groups[i]), scores


@pytest.mark.parametrize("alpha", [None, 0.6])
def test_cluster_selection_keeps_original_positions_after_physical_compaction(alpha):
    key, value = fixture()
    positions = np.array([0, 3, 16, 17, 31, 48, 63])
    cache = RetainedLayerCache(key, value, positions, capacity=9, maximum_position=66)
    query = np.array([[1, -2, 3, 0], [-1, 0.5, 2, 1]], np.float32)
    for size in [4, 8, 16]:
        selected, scores = cache.select_clusters(query, positions, cluster_rows=size, keep_clusters=2, alpha=alpha)
        expected, expected_scores = cluster_oracle(key, query, positions, size, 2, alpha)
        np.testing.assert_array_equal(selected, expected)
        np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=1e-12)
        got_k, got_v, got_p = cache.gather(selected)
        np.testing.assert_array_equal(got_p, expected)
        np.testing.assert_array_equal(got_k, key[expected])
        np.testing.assert_array_equal(got_v, value[expected])
    with pytest.raises(ValueError, match="evicted"):
        cache.select_clusters(query, np.array([1]), cluster_rows=8, keep_clusters=1)


@pytest.mark.parametrize("retained", [np.arange(64), np.array([2, 16, 31, 63])])
def test_two_row_decode_uses_absolute_causal_positions_and_grouped_heads(retained):
    key, value = fixture()
    tail_k, tail_v = fixture(2)
    dense_key = np.concatenate([key, tail_k])
    dense_value = np.concatenate([value, tail_v])
    cache = RetainedLayerCache(key, value, retained, capacity=len(retained) + 2, maximum_position=66)
    cache.append(64, tail_k, tail_v)
    k, v, positions = cache.gather()
    query = np.arange(32, dtype=np.float32).reshape(1, 4, 2, 4) / 64
    layout = "batch_kv_heads_sequence_feature"
    scores = semantic_attention_scores(query, k.transpose(1, 0, 2)[None], group=2, layout=layout)
    scores = mask_retained_scores(scores, np.array([64, 65], np.int64), positions, maximum_position=66)
    assert np.all(np.isneginf(scores[:, :, 0, -1]))
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    output = semantic_attention_values(probabilities, v.transpose(1, 0, 2)[None], group=2, layout=layout)
    expected = np.empty((1, 4, 2, 4), np.float64)
    for head in range(4):
        for row, query_position in enumerate([64, 65]):
            valid = [int(p) for p in positions if p <= query_position]
            products = [sum(float(query[0, head, row, col]) * float(dense_key[p, head // 2, col]) for col in range(4)) for p in valid]
            numerators = [math.exp(score - max(products)) for score in products]
            for col in range(4):
                expected[0, head, row, col] = sum(n * float(dense_value[p, head // 2, col]) for n, p in zip(numerators, valid, strict=True)) / sum(numerators)
    np.testing.assert_allclose(output, expected, atol=4e-7, rtol=2e-6)


@pytest.mark.parametrize("failure", ["rewind", "gap", "nonfinite", "shape", "capacity", "missing", "duplicate"])
def test_failure_preserves_complete_state(failure):
    key, value = fixture()
    cache = RetainedLayerCache(key, value, np.array([1, 63]), capacity=3, maximum_position=66)
    before = cache.gather()
    k, v = fixture(1)
    with pytest.raises(ValueError):
        if failure == "rewind": cache.append(63, k, v)
        elif failure == "gap": cache.append(65, k, v)
        elif failure == "nonfinite": cache.append(64, np.full_like(k, np.inf), v)
        elif failure == "shape": cache.append(64, k.reshape(1, 1, 8), v.reshape(1, 1, 8))
        elif failure == "capacity": cache.append(64, *fixture(2))
        elif failure == "missing": cache.retain(np.array([2]), capacity=1)
        else: cache.retain(np.array([1, 1]), capacity=2)
    assert cache.position == 64 and cache.length == 2 and cache.capacity == 3
    for a, b in zip(cache.gather(), before, strict=True):
        np.testing.assert_array_equal(a, b)
    cache.append(64, k, v)
    assert cache.position == 65


def test_native_frames_and_resource_bounds_reject_before_use():
    keys, values = fixture(2)
    raw, other = keys.tobytes(), values.tobytes()
    positions = np.array([0, 1], dtype="<u4").tobytes()
    for width, cap, bound in [(8192, 8192, 3), (0, 2, 3), (8, 2, 1), (8, 0, 3)]:
        with pytest.raises(ValueError):
            _native._RetainedKvCacheF32(raw, other, width, positions, cap, bound)
    owner = _native._RetainedKvCacheF32(raw, other, 8, positions, 3, 3)
    for invalid in [b"", b"\0", positions + b"\0", np.array([2], dtype="<u4").tobytes()]:
        with pytest.raises(ValueError):
            owner.export(invalid)
    assert owner.stats() == (2, 3, 8, 2, 3 * 68)
    owner.close()
    with pytest.raises(ValueError):
        owner.append(0, raw[:32], other[:32])
