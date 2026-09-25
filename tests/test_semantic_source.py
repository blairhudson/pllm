"""Nested semantic sources must survive the provider/client bundle boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllm import lower_model
from pllm.runtime.model_binding import RuntimeBindingError, _runtime_config, _validate_runtime_semantics
from pllm.runtime.semantic_source import SemanticSourceError, semantic_source_config


_FIXTURE = Path("crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json")


def _flattened() -> tuple[dict, dict]:
    source = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return {**source["text_config"], "semantic_source_config": source}, source


def test_nested_semantic_source_reconstructs_exact_adapter_plan() -> None:
    flattened, source = _flattened()
    assert semantic_source_config(flattened) is source
    plan = lower_model(semantic_source_config(flattened), batch=1, max_input_tokens=2, max_new_tokens=2)
    assert plan.to_dict()["adapter"] == "pllm.gemma4_e2b_text.v1"
    runtime = _runtime_config(flattened, nested_source=True)
    assert runtime["sliding_window"] == 512
    assert "sliding_attention" in runtime["layer_types"]
    graph = plan.to_dict()
    phases = {
        phase: (graph[phase], {op["id"]: op for op in graph[phase]["operations"]})
        for phase in ("prefill", "decode")
    }
    _validate_runtime_semantics(runtime, phases, nested_source=True)


@pytest.mark.parametrize("key", ["hidden_size", "head_dim", "layer_types", "rope_parameters"])
def test_flattened_semantic_config_rejects_drift(key: str) -> None:
    flattened, _ = _flattened()
    flattened[key] = "out-of-band override"
    with pytest.raises(SemanticSourceError, match=key):
        semantic_source_config(flattened)


def test_semantic_source_rejects_missing_outer_or_unbound_rotary_width() -> None:
    flattened, _ = _flattened()
    flattened["semantic_source_config"] = {"text_config": {"model_type": "gemma4_text"}}
    with pytest.raises(SemanticSourceError, match="malformed"):
        semantic_source_config(flattened)

    flattened, _ = _flattened()
    del flattened["layer_types"]
    with pytest.raises(SemanticSourceError, match="layer_types"):
        semantic_source_config(flattened)

    flattened, source = _flattened()
    plan = lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=2).to_dict()
    phases = {
        phase: (plan[phase], {op["id"]: op for op in plan[phase]["operations"]})
        for phase in ("prefill", "decode")
    }
    rotary = next(
        op for op in phases["prefill"][1].values() if op["operator"] == "rotary_embedding"
    )
    rotary["attributes"]["head_dim"] += 1
    with pytest.raises(RuntimeBindingError, match="rotary width"):
        _validate_runtime_semantics(_runtime_config(flattened, nested_source=True), phases, nested_source=True)


def test_flat_semantic_source_preserves_existing_decoder_identity() -> None:
    config = {"model_type": "qwen2", "hidden_size": 32}
    assert semantic_source_config(config) is config
