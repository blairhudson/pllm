"""Ordered token boundary stages follow semantic artifacts, not model names."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pllm import lower_model
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.semantic_stages import _token_lookup_stage
from pllm.runtime.semantic_tensors import SemanticTensorError, preflight_semantic_checkpoint
from pllm.runtime.transformer_client import TransformerClientError


def _pinned_token_group() -> tuple[dict, dict, int]:
    source = Path("crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json")
    config = json.loads(source.read_text())
    plan = lower_model(config, batch=1, max_input_tokens=2, max_new_tokens=2)
    operations = {operation["id"]: operation for operation in plan.prefill["operations"]}
    first = operations["main_embedding"]
    second = operations["ple_token_embedding"]
    first_width = first["output_shape"][-1]
    step = {
        "operation_ids": [first["id"], second["id"]],
        "operators": ["token_lookup", "token_lookup"],
        "input_ids": ["input.tokens"],
        "layer": None,
        "weight_ids": [first["attributes"]["weight"], second["attributes"]["weight"]],
        "outputs": [
            {"operation_id": first["id"], "stage_offset": 0, "stage_width": first_width},
            {
                "operation_id": second["id"],
                "stage_offset": first_width,
                "stage_width": second["output_shape"][-1],
            },
        ],
    }
    return step, operations, config["text_config"]["vocab_size"]


def test_pinned_token_tables_form_one_ordered_boundary_stage() -> None:
    step, operations, vocabulary = _pinned_token_group()
    stage = _token_lookup_stage(step, operations, vocabulary)
    assert stage.id == "token_lookup"
    assert stage.in_features == 262_144
    assert stage.out_features == 1536 + 8960
    assert stage.metadata == {"ple_width": 8960}
    assert stage.weight_keys == tuple(step["weight_ids"])

    forged = copy.deepcopy(step)
    forged["outputs"][1]["stage_offset"] -= 1
    with pytest.raises(ValueError, match="offsets"):
        _token_lookup_stage(forged, operations, vocabulary)
    forged = copy.deepcopy(step)
    forged["weight_ids"].reverse()
    with pytest.raises(ValueError, match="weights"):
        _token_lookup_stage(forged, operations, vocabulary)


def test_pinned_token_stage_matches_real_checkpoint_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    path = os.environ.get("PLLM_GEMMA4_E2B_PATH")
    if path is None:
        pytest.skip("set PLLM_GEMMA4_E2B_PATH for the pinned real-checkpoint check")
    step, operations, vocabulary = _pinned_token_group()
    stage = _token_lookup_stage(step, operations, vocabulary)
    store = SafeTensorStore(path)
    widths = (stage.out_features - stage.metadata["ple_width"], stage.metadata["ple_width"])
    for key, width in zip(stage.weight_keys, widths, strict=True):
        assert store.tensor_shape(key) == (vocabulary, width)
        assert store.tensor_dtype(key) == "BF16"
    source = Path("crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json")
    plan = lower_model(json.loads(source.read_text()), batch=1, max_input_tokens=2, max_new_tokens=2)
    assert preflight_semantic_checkpoint(plan, store) == 540
    original = store.tensor_shape
    monkeypatch.setattr(
        store, "tensor_shape", lambda key: (vocabulary, widths[1] + 1) if key == stage.weight_keys[1] else original(key)
    )
    with pytest.raises(SemanticTensorError, match="invalid shape"):
        preflight_semantic_checkpoint(plan, store)


@pytest.mark.parametrize("selected", ["main_embedding", "ple_token_embedding"])
def test_fused_token_lookup_assigns_each_declared_numeric_output(selected: str) -> None:
    step, operations, vocabulary = _pinned_token_group()
    stage = _token_lookup_stage(step, operations, vocabulary)
    decoder = object.__new__(SemanticDecoderRuntime)
    decoder.position = 0
    decoder.caches = []
    decoder.bundle = SimpleNamespace(stages={"token_lookup": stage})
    primary_width = stage.out_features - stage.metadata["ple_width"]
    decoder._token_lookup = lambda _ids: (
        np.full((1, primary_width), 1.0, dtype=np.float32),
        np.full((1, stage.metadata["ple_width"]), 2.0, dtype=np.float32),
    )
    graph = {
        "query_sequence": 1,
        "maximum_key_sequence": 2,
        "state_inputs": [],
        "state_outputs": [],
        "operations": [
            operations["main_embedding"],
            operations["ple_token_embedding"],
            {"id": "selection", "operator": "greedy_token_selection", "inputs": [selected], "attributes": {}},
        ],
    }
    schedule = {
        "steps": [
            {"executor": "remote_stage", **step},
            {"executor": "client_local", "operation_ids": ["selection"], "input_ids": [selected]},
        ]
    }
    decoder._graphs = {"prefill": graph, "decode": graph}
    decoder._schedule = {"prefill": schedule, "decode": schedule}
    decoder._stages = {
        f"{phase}:{op_id}": "token_lookup"
        for phase in ("prefill", "decode")
        for op_id in step["operation_ids"]
    }
    actual = decoder._forward(np.asarray([3], dtype=np.int64))
    expected = 1.0 if selected == "main_embedding" else 2.0
    width = primary_width if selected == "main_embedding" else stage.metadata["ple_width"]
    np.testing.assert_array_equal(actual, np.full((1, width), expected, dtype=np.float32))
    assert decoder.position == 1

    decoder.position = 0
    decoder._token_lookup = lambda _ids: (
        np.ones((1, primary_width), dtype=np.float32), np.ones((1, 1), dtype=np.float32)
    )
    with pytest.raises(TransformerClientError, match="invalid width"):
        decoder._forward(np.asarray([3], dtype=np.int64))
    assert decoder.position == 0
