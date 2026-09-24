"""State ownership belongs to semantic operators, not output-name conventions."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pllm.modeling import lower_model
from pllm.runtime.semantic_attention import SemanticAttentionError, mask_causal_scores
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.semantic_state import SemanticStateError, WindowedLayerCache
from pllm.runtime.transformer_client import TransformerClientError


def _gemma_graph(mode: str = "prefill") -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"
    )
    plan = lower_model(json.loads(path.read_text()), batch=1, max_input_tokens=2, max_new_tokens=2)
    return plan.to_dict()[mode]


def test_shared_kv_source_updates_and_sliding_suffixes_keep_declared_state_kinds() -> None:
    graph = _gemma_graph()
    kinds = SemanticDecoderRuntime._declared_state_kinds(graph)
    outputs = {state["id"] for state in graph["state_outputs"]}
    assert "layer.0.key_append" not in outputs
    assert "layer.0.value_append" not in outputs
    assert kinds["layer.0.key_append"] == kinds["layer.0.key_suffix"] == "key"
    assert kinds["layer.0.value_append"] == kinds["layer.0.value_suffix"] == "value"
    assert any(
        operation["operator"] == "attention_scores"
        and operation["attributes"].get("key_value_source_layer") != operation["layer"]
        for operation in graph["operations"]
    ), "pinned graph must cover a consumer of another layer's KV state"


def test_state_ownership_disagrees_with_plan_fails_closed() -> None:
    graph = _gemma_graph()
    bad_output = copy.deepcopy(graph)
    bad_output["state_outputs"][0]["kind"] = "value"
    with pytest.raises(TransformerClientError, match="ownership"):
        SemanticDecoderRuntime._declared_state_kinds(bad_output)
    bad_operator = copy.deepcopy(graph)
    operation = next(
        row for row in bad_operator["operations"] if row["id"] == "layer.0.key_append"
    )
    operation["state_kind"] = "recurrent"
    with pytest.raises(TransformerClientError, match="unsupported state kind"):
        SemanticDecoderRuntime._declared_state_kinds(bad_operator)


def test_full_and_query_relative_window_masks_follow_absolute_positions() -> None:
    graph = _gemma_graph()
    policies = [
        row["attributes"] for row in graph["operations"] if row["operator"] == "causal_mask"
    ]
    sliding = next(row for row in policies if row["kind"] == "sliding_causal")
    full = next(row for row in policies if row["kind"] == "full_causal")
    positions = np.asarray([0, 1], dtype=np.int64)
    lengths = np.asarray([2], dtype=np.int64)
    padding = np.ones((1, 2), dtype=np.bool_)
    full_scores = np.ones((1, 1, 2, 4), dtype=np.float32)
    full_result = mask_causal_scores(
        full_scores, positions, full, padding_mask=padding, valid_lengths=lengths
    )
    np.testing.assert_array_equal(np.isfinite(full_result[0, 0]), [
        [True, False, False, False],
        [True, True, False, False],
    ])
    window = sliding["sliding_window"]
    window_scores = np.ones((1, 1, 2, window), dtype=np.float32)
    window_result = mask_causal_scores(
        window_scores, positions, sliding, padding_mask=padding, valid_lengths=lengths
    )
    assert window_result.dtype == np.float32
    assert np.isfinite(window_result[0, 0, 0]).sum() == 1
    assert np.isfinite(window_result[0, 0, 1]).sum() == 2
    np.testing.assert_array_equal(window_result[0, 0, :, -1], [1.0, 1.0])
    advanced = mask_causal_scores(
        window_scores[:, :, :1],
        np.asarray([window + 3], dtype=np.int64),
        sliding,
        padding_mask=padding[:, :1],
        valid_lengths=np.asarray([window + 4], dtype=np.int64),
    )
    assert np.all(np.isfinite(advanced))


def test_window_mask_rejects_forged_domain_and_invalid_lengths() -> None:
    sliding = next(
        row["attributes"]
        for row in _gemma_graph()["operations"]
        if row["operator"] == "causal_mask" and row["attributes"]["kind"] == "sliding_causal"
    )
    window = sliding["sliding_window"]
    scores = np.ones((1, 1, 1, window), dtype=np.float32)
    positions = np.asarray([4], dtype=np.int64)
    padding = np.ones((1, 1), dtype=np.bool_)
    lengths = np.asarray([5], dtype=np.int64)
    with pytest.raises(SemanticAttentionError, match="window"):
        mask_causal_scores(
            scores,
            positions,
            {**sliding, "left_context": 0},
            padding_mask=padding,
            valid_lengths=lengths,
        )
    with pytest.raises(SemanticAttentionError, match="bounds"):
        mask_causal_scores(
            scores,
            positions,
            sliding,
            padding_mask=padding,
            valid_lengths=np.asarray([4], dtype=np.int64),
        )
    with pytest.raises(SemanticAttentionError, match="bounds"):
        mask_causal_scores(
            scores,
            positions,
            sliding,
            padding_mask=np.zeros((1, 1), dtype=np.bool_),
            valid_lengths=lengths,
        )


@pytest.mark.parametrize("layout", ["full", "window"])
def test_grouped_attention_scores_and_values_match_explicit_window_reference(layout: str) -> None:
    operations = {row["id"]: row for row in _gemma_graph()["operations"]}
    attention = next(
        row
        for row in operations.values()
        if row["operator"] == "attention_scores"
        and (row["attributes"]["key_layout"] == "batch_kv_heads_sequence_feature")
        == (layout == "full")
    )
    layer = attention["layer"]
    output = operations[f"layer.{layer}.attention_values"]
    group = attention["attributes"]["group_size"]
    query = np.arange(group * 2 * 3, dtype=np.float32).reshape(1, group, 2, 3) / 23.0
    tensor = np.arange(2 * 5 * 3, dtype=np.float32).reshape(1, 1, 2, 5, 3) / 31.0
    keys = tensor[:, :, 0, :, :] if layout == "full" else tensor
    scored = object.__new__(SemanticDecoderRuntime)._local(
        attention,
        {attention["inputs"][0]: query, attention["inputs"][1]: keys},
        {},
        {},
    )
    expected = np.empty((1, group, 2, 5), dtype=np.float32)
    for head in range(group):
        for row in range(2):
            for column in range(5):
                key = keys[0, 0, column] if layout == "full" else keys[0, 0, row, column]
                expected[0, head, row, column] = np.dot(query[0, head, row], key)
    np.testing.assert_allclose(scored, expected, rtol=0, atol=1e-6)
    probabilities = np.exp(scored - scored.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    values = keys + 0.25
    weighted = object.__new__(SemanticDecoderRuntime)._local(
        output,
        {output["inputs"][0]: probabilities, output["inputs"][1]: values},
        {},
        {},
    )
    reference = np.empty((1, group, 2, 3), dtype=np.float32)
    for head in range(group):
        for row in range(2):
            reference[0, head, row] = sum(
                probabilities[0, head, row, column]
                * (values[0, 0, column] if layout == "full" else values[0, 0, row, column])
                for column in range(5)
            )
    np.testing.assert_allclose(weighted, reference, rtol=0, atol=1e-6)


def test_windowed_cache_retains_only_valid_prefix_and_copies_snapshot() -> None:
    cache = WindowedLayerCache(window=4)
    current = np.arange(2, dtype=np.float32).reshape(2, 1, 1) + 1
    keys, values = cache.append_windows(current, current + 10)
    assert keys.shape == (1, 1, 2, 4, 1)
    np.testing.assert_array_equal(keys[0, 0, :, :, 0], [[0, 0, 0, 1], [0, 0, 1, 2]])
    np.testing.assert_array_equal(values[0, 0, 1, :, 0], [0, 0, 11, 12])
    assert (cache.length, cache.position) == (2, 2)

    next_tokens = np.arange(3, 6, dtype=np.float32).reshape(3, 1, 1)
    keys, _ = cache.append_windows(next_tokens, next_tokens + 10)
    np.testing.assert_array_equal(
        keys[0, 0, :, :, 0], [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
    )
    np.testing.assert_array_equal(cache.key[:, 0, 0], [3, 4, 5])
    assert (cache.length, cache.position) == (3, 5)
    snapshot = cache.copy_active()
    assert isinstance(snapshot, WindowedLayerCache)
    np.testing.assert_array_equal(snapshot.value[:, 0, 0], [13, 14, 15])
    cache.key[0, 0, 0] = 99
    np.testing.assert_array_equal(snapshot.key[:, 0, 0], [3, 4, 5])
    suffix, _ = snapshot.append_windows(
        np.asarray([[[6]]], dtype=np.float32), np.asarray([[[16]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(suffix[0, 0, 0, :, 0], [3, 4, 5, 6])
    assert (snapshot.length, snapshot.position) == (3, 6)


def test_windowed_cache_rejects_invalid_inputs_before_state_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    import pllm.runtime.semantic_state as semantic_state

    cache = WindowedLayerCache(window=4)
    first = np.asarray([[[2]]], dtype=np.float32)
    cache.append_windows(first, first)
    state = cache.copy_active()
    for key, value in (
        (first.astype(np.float64), first),
        (np.asarray([[[np.nan]]], dtype=np.float32), first),
        (np.ones((1, 2, 1), dtype=np.float32), first),
    ):
        with pytest.raises(SemanticStateError):
            cache.append_windows(key, value)
        assert (cache.length, cache.position) == (state.length, state.position)
        np.testing.assert_array_equal(cache.key, state.key)
    monkeypatch.setattr(semantic_state, "MAX_WINDOW_VIEW_ELEMENTS", 3)
    with pytest.raises(SemanticStateError, match="resource bound"):
        cache.append_windows(first, first)
    assert (cache.length, cache.position) == (state.length, state.position)
    cache.value = None
    with pytest.raises(SemanticStateError, match="stored prefix"):
        cache.append_windows(first, first)
    cache = state.copy_active()
    cache.key[0, 0, 0] = np.nan
    with pytest.raises(SemanticStateError, match="stored prefix"):
        cache.append_windows(first, first)
    assert cache.position == state.position


def test_one_element_window_keeps_no_past_state() -> None:
    cache = WindowedLayerCache(window=1)
    for position in range(1, 4):
        row = np.asarray([[[position]]], dtype=np.float32)
        keys, _ = cache.append_windows(row, row)
        np.testing.assert_array_equal(keys[0, 0, 0, :, 0], [position])
        assert (cache.length, cache.position) == (0, position)


def test_checked_sliding_append_and_suffix_pair_preserve_semantic_state() -> None:
    graph = _gemma_graph()
    operations = {row["id"]: row for row in graph["operations"]}
    key_op = operations["layer.0.key_append"]
    value_op = operations["layer.0.value_append"]
    key_suffix = operations["layer.0.key_suffix"]
    value_suffix = operations["layer.0.value_suffix"]
    window = key_op["attributes"]["attention_domain"]["maximum_sequence"]
    runtime = object.__new__(SemanticDecoderRuntime)
    runtime._window_contracts = {0: window}
    runtime.caches = [WindowedLayerCache(window=window)]
    kinds = runtime._declared_state_kinds(graph)
    values = {
        key_op["inputs"][0]: np.asarray([[[[1], [2]]]], dtype=np.float32),
        value_op["inputs"][0]: np.asarray([[[[11], [12]]]], dtype=np.float32),
    }
    pending: dict[int, tuple[str, np.ndarray]] = {}
    values[key_op["id"]] = runtime._local(key_op, values, kinds, pending)
    values[value_op["id"]] = runtime._local(value_op, values, kinds, pending)
    assert not pending
    assert values[key_op["id"]].shape == (1, 1, 2, window, 1)
    assert values[value_op["id"]].shape == values[key_op["id"]].shape
    np.testing.assert_array_equal(values[key_op["id"]][0, 0, :, -2:, 0], [[0, 1], [1, 2]])
    np.testing.assert_array_equal(values[value_op["id"]][0, 0, :, -2:, 0], [[0, 11], [11, 12]])
    np.testing.assert_array_equal(
        runtime._local(key_suffix, values, kinds, pending)[0, 0, :, 0], [1, 2]
    )
    np.testing.assert_array_equal(
        runtime._local(value_suffix, values, kinds, pending)[0, 0, :, 0], [11, 12]
    )
    assert (runtime.caches[0].length, runtime.caches[0].position) == (2, 2)


def test_windowed_plan_preflights_aggregate_state_before_issuance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pllm.runtime.semantic_state as semantic_state

    graphs = {"prefill": _gemma_graph(), "decode": _gemma_graph("decode")}
    windows = SemanticDecoderRuntime._declared_windows(graphs)
    assert windows
    required = SemanticDecoderRuntime._window_resource_bytes(graphs, windows)
    assert required > 0
    monkeypatch.setattr(semantic_state, "MAX_WINDOW_STATE_BYTES", required - 1)
    with pytest.raises(TransformerClientError, match="client memory budget"):
        SemanticDecoderRuntime._window_resource_bytes(graphs, windows)
    monkeypatch.setattr(semantic_state, "MAX_WINDOW_STATE_BYTES", required)
    assert SemanticDecoderRuntime._window_resource_bytes(graphs, windows) == required
    state_inputs = graphs["decode"]["state_inputs"]
    state_inputs.pop(next(index for index, state in enumerate(state_inputs) if state["layer"] in windows))
    with pytest.raises(TransformerClientError, match="lacks key/value capacity"):
        SemanticDecoderRuntime._window_resource_bytes(graphs, windows)
