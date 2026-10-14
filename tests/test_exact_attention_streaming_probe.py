"""Research-only numeric and ownership controls against installed PLLM operators."""
import importlib.util
from pathlib import Path
import weakref

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_exact_attention_streaming.py"
SPEC = importlib.util.spec_from_file_location("exact_attention_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def independent_oracle(q, k, v, positions, group, profile):
    """Independent full orchestration; never calls probe stages/stream or _local."""
    from pllm.runtime.semantic_attention import semantic_attention_scores, semantic_attention_values
    from pllm.runtime.semantic_numeric import bfloat16_softmax
    from pllm.runtime import causal_reduction
    if profile == "prefix_f32":
        scores = causal_reduction.scores(q, k, positions, group)
    elif profile != "legacy":
        scores = semantic_attention_scores(q, k, group=group, layout=probe.LAYOUT)
    else:
        grouped = q.reshape(1, k.shape[1], group, q.shape[2], q.shape[3])
        scores = np.empty((1, k.shape[1], group, q.shape[2], k.shape[2]), np.float32)
        for row in range(q.shape[2]):
            scores[0, :, :, row] = np.einsum("kgd,tkd->kgt", grouped[0, :, :, row], k[0].transpose(1, 0, 2))
        scores = scores.reshape(1, q.shape[1], q.shape[2], k.shape[2])
    import math
    scaled = scores * float(1.0 / math.sqrt(q.shape[-1]))
    masked = np.where(np.arange(k.shape[2])[None, None, None] <= positions[None, None, :, None],
                      scaled, np.float32(-np.inf))
    if profile == "prefix_f32":
        probabilities = causal_reduction.softmax(masked)
        return causal_reduction.values(probabilities, v, positions, group)
    if profile == "bf16":
        probabilities = bfloat16_softmax(masked)
    else:
        centered = masked - masked.max(axis=-1, keepdims=True)
        exponential = np.exp(centered).astype(np.float32)
        probabilities = exponential / exponential.sum(axis=-1, keepdims=True)
    if profile != "legacy":
        return semantic_attention_values(probabilities, v, group=group, layout=probe.LAYOUT)
    grouped = probabilities.reshape(1, k.shape[1], group, q.shape[2], k.shape[2])
    result = np.empty((1, k.shape[1], group, q.shape[2], v.shape[-1]), np.float32)
    for row in range(q.shape[2]):
        result[0, :, :, row] = np.einsum("kgt,tkd->kgd", grouped[0, :, :, row], v[0].transpose(1, 0, 2))
    return result.reshape(q.shape)


@pytest.mark.parametrize("profile", probe.PROFILES)
@pytest.mark.parametrize("layout", ("c", "sequence_major", "reverse", "fortran"))
@pytest.mark.parametrize("start,rows", ((0, 9), (8, 1), (5, 4)))
def test_independent_bitwise_oracle(profile, layout, start, rows):
    args = probe.fixture(layout=layout, start=start, rows=rows)
    before = [a.copy() for a in args]
    oracle = independent_oracle(*args, 2, profile)
    assert probe.bits_equal(probe.stages(*args, 2, profile), oracle)
    for qr, kh in ((1, 1), (3, 2), (7, 3)):
        assert probe.bits_equal(probe.stream(*args, 2, profile, query_rows=qr, kv_heads=kh), oracle)
    assert all(probe.bits_equal(a, b) for a, b in zip(args, before))


def test_real_counterexamples_not_tolerance():
    records = probe.counterexamples()
    assert all(record["changed_words"] > 0 for record in records)


def test_owner_dedup_and_tile_retirement(monkeypatch):
    args = probe.fixture(layout="reverse")
    q, k, v, p = args
    assert probe.owner_bytes(q, q[:, :, :1], k, v, p) == q.nbytes + k.nbytes + v.nbytes + p.nbytes
    previous = []
    original = probe.stages
    def checked(*args, **kwargs):
        assert all(ref() is None for ref in previous)
        result = original(*args, **kwargs)
        previous.append(weakref.ref(result))
        return result
    monkeypatch.setattr(probe, "stages", checked)
    probe.stream(*args, 2, "dense")
    assert all(ref() is None for ref in previous)
    narrow = probe.accounting(*args, 2, "dense", 1, 1)
    broad = probe.accounting(*args, 2, "dense", 9, 3)
    assert narrow["input_unique_owner_bytes"] == broad["input_unique_owner_bytes"]
    assert narrow["complete_output_bytes"] == broad["complete_output_bytes"]
    assert narrow["score_slab_bytes"] < broad["score_slab_bytes"]
    assert not narrow["is_whole_client_admission"]


@pytest.mark.parametrize("profile", probe.PROFILES)
def test_small_generated_attention_recurrence(profile):
    """Toy token-dependent attention loop, not generated-model/SDK evidence."""
    rng = np.random.default_rng(90)
    embedding = rng.normal(size=(11, 4, 7)).astype(np.float32)
    head = rng.normal(size=(28, 11)).astype(np.float32)
    traces = []
    for tiled in (False, True):
        keys = np.zeros((1, 2, 13, 7), np.float32)
        values = keys.copy()
        history, logits = [1, 4, 2], []
        for step in range(4):
            start = 0 if step == 0 else len(history) - 1
            ids = history if step == 0 else history[-1:]
            query = embedding[ids].transpose(1, 0, 2)[None]
            keys[:, :, start:start + len(ids)] = query[:, ::2]
            values[:, :, start:start + len(ids)] = query[:, 1::2]
            positions = np.arange(start, start + len(ids), dtype=np.int64)
            output = probe.stream(query, keys, values, positions, 2, profile) if tiled else independent_oracle(
                query, keys, values, positions, 2, profile)
            current = np.einsum("d,dt->t", output[0, :, -1].reshape(-1), head, optimize=False)
            logits.append(current.copy())
            history.append(int(np.argmax(current)))
        traces.append((history, logits, keys, values))
    assert traces[0][0] == traces[1][0]
    assert all(probe.bits_equal(a, b) for a, b in zip(traces[0][1], traces[1][1]))
    assert probe.bits_equal(traces[0][2], traces[1][2])
    assert probe.bits_equal(traces[0][3], traces[1][3])


def test_installed_identity():
    info = probe.identity()
    assert "site-packages/pllm" in info["pllm"]
    assert all("site-packages/pllm" in op["path"] for op in info["operators"].values())


def test_research_bounds():
    with pytest.raises(ValueError):
        probe.fixture(rows=385, capacity=512)
    with pytest.raises(ValueError):
        probe.stream(*probe.fixture(), 2, "dense", query_rows=0)


@pytest.mark.parametrize("profile", ("dense", "prefix_f32"))
def test_measurement_worker_uses_declared_gqa_group(profile):
    payload = {"fixture": {"rows": 17, "capacity": 31, "width": 16, "heads": 2,
                           "group": 7, "layout": "sequence_major"},
               "query_rows": 4, "kv_heads": 1, "profile": profile}
    control = probe.worker(payload | {"arm": "control"})
    candidate = probe.worker(payload | {"arm": "candidate"})
    assert control["output_sha256"] == candidate["output_sha256"]
    assert candidate["owners"]["full_score_bytes"] == 4 * 14 * 17 * 31
    assert candidate["owners"]["score_slab_bytes"] == 4 * 7 * 4 * 31


@pytest.mark.parametrize("profile", probe.PROFILES)
def test_post_qk_norm_and_rope_inputs(profile):
    q, k, v, positions = probe.fixture(width=16)
    runtime = probe.operator_runtime(profile, 0)
    runtime._weight = lambda name: np.linspace(0.5, 1.5, 16, dtype=np.float32)
    norm = {"operator": "rms_norm", "inputs": ["x"], "attributes": {
        "weight": "gamma", "epsilon": "0.00001", "weight_offset": 0}}
    q = runtime._local(norm, {"x": q}, {}, {})
    k = runtime._local(norm, {"x": k}, {}, {})
    attrs = {"rotary_dimensions": 16, "theta": 10000}
    q = runtime._rotary(q, positions, attrs)
    k[:, :, :9] = runtime._rotary(k[:, :, :9], positions, attrs)
    # Cache-compatible sequence-major views, no candidate-side normalization/copy.
    q, k, v = [np.ascontiguousarray(a.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
               for a in (q, k, v)]
    expected = independent_oracle(q, k, v, positions, 2, profile)
    assert probe.bits_equal(expected, probe.stream(q, k, v, positions, 2, profile))


@pytest.mark.parametrize("profile", probe.PROFILES)
def test_cancellation_signed_zero_and_feature_stride(profile):
    q, k, v, p = probe.fixture(seed=13, width=64, layout="reverse")
    k *= np.tile(np.array([1e6, 1e-6, -1e6, -1e-6], np.float32), 16)
    v[..., ::5] = np.float32(-0.0)
    expected = independent_oracle(q, k, v, p, 2, profile)
    assert probe.bits_equal(expected, probe.stream(q, k, v, p, 2, profile, query_rows=3, kv_heads=2))


def test_fixed_capacity_mask_query_partition():
    from pllm.runtime.semantic_attention import mask_causal_scores
    args = probe.fixture(start=5, rows=4)
    q, k, v, p = args
    rng = np.random.default_rng(11)
    scores = rng.normal(size=(1, 6, 4, 23)).astype(np.float32)
    attrs = {"absolute_positions_input": "input.positions", "padding_mask_input": "input.attention_mask",
             "valid_lengths_input": "input.sequence_lengths", "kind": "full_causal",
             "maximum_position_embeddings": 23, "key_domain": "fixed_capacity",
             "cache_validity": "valid_lengths_fixed_capacity"}
    oracle = mask_causal_scores(scores, p, attrs, padding_mask=np.ones((1, 4), np.bool_),
                                valid_lengths=np.array([9], np.int64))
    candidate = np.concatenate([mask_causal_scores(scores[:, :, row:row+1], p[row:row+1], attrs,
        padding_mask=np.ones((1, 1), np.bool_), valid_lengths=np.array([9], np.int64)) for row in range(4)], axis=2)
    assert probe.bits_equal(oracle, candidate)
