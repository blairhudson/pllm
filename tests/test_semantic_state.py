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
from pllm.runtime.transformer_client import TransformerClientError


def _gemma_graph() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"
    )
    plan = lower_model(json.loads(path.read_text()), batch=1, max_input_tokens=2, max_new_tokens=2)
    return plan.to_dict()["prefill"]


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


@pytest.mark.parametrize(
    "operator_id",
    [
        "layer.0.key_append",
        "layer.0.key_suffix",
    ],
)
def test_unbound_bfloat16_and_windowed_state_execution_fails_before_using_tensors(
    operator_id: str,
) -> None:
    graph = _gemma_graph()
    operation = next(row for row in graph["operations"] if row["id"] == operator_id)
    runtime = object.__new__(SemanticDecoderRuntime)
    with pytest.raises(TransformerClientError, match="not yet bound"):
        runtime._local(
            operation,
            {operation["inputs"][0]: np.ones((1, 1, 1, 1), dtype=np.float32)},
            SemanticDecoderRuntime._declared_state_kinds(graph),
            {},
        )


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
