from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import pllm
from pllm.components import KvCacheEviction


CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
}

GEMMA4_E4B_CONFIG = (
    Path(__file__).parents[1]
    / "crates/pllm-models/tests/fixtures/gemma-4-E4B-it-ee0ef602-config.json"
)
GEMMA4_E2B_CONFIG = (
    Path(__file__).parents[1]
    / "crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"
)
MODEL_FIXTURES = (
    (
        Path(__file__).parents[1]
        / "crates/pllm-models/tests/fixtures/mini-coder-4b-c87892d-config.json",
        "pllm.qwen3.v1",
        "qwen3",
    ),
    (
        Path(__file__).parents[1]
        / "crates/pllm-models/tests/fixtures/Phi-4-mini-instruct-cfbefac-config.json",
        "pllm.phi4_mini.v1",
        "phi4_mini",
    ),
    (
        Path(__file__).parents[1]
        / "crates/pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json",
        "pllm.qwen3_5_text.v1",
        "qwen3_5_text",
    ),
)


def test_model_lowering_is_complete_immutable_and_deterministic() -> None:
    first = pllm.lower_model(CONFIG, batch=1, max_input_tokens=128, max_new_tokens=32)
    second = pllm.lower_model(CONFIG, batch=1, max_input_tokens=128, max_new_tokens=32)

    assert first.digest == second.digest
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.to_dict()["model_family"] == "qwen2"
    assert first.to_dict()["adapter"] == "pllm.qwen2.v1"
    assert first.to_dict()["transformations"] == []
    assert first.prefill["output"] == "token_feedback"
    assert first.decode["query_sequence"] == 1
    assert len(first.prefill["state_outputs"]) == 48
    operators = {operation["operator"] for operation in first.prefill["operations"]}
    assert {"softmax", "silu", "output_head", "token_feedback"} <= operators
    coverage = first.coverage()
    assert coverage.complete is False
    levels = {row["operator"]: row["level"] for row in coverage.operators}
    assert levels["linear"] == "executable_region"
    assert levels["silu"] == "primitive"
    assert levels["rms_norm"] == "primitive"
    assert levels["softmax"] == "missing"
    assert levels["greedy_token_selection"] == "missing"
    with pytest.raises(TypeError):
        first.prefill["output"] = "changed"


def test_model_component_transforms_generic_decoder_plan_immutably() -> None:
    base = pllm.lower_model(CONFIG, batch=1, max_input_tokens=128, max_new_tokens=32)
    base_document = base.to_dict()
    optimized = base.apply(KvCacheEviction())
    document = optimized.to_dict()

    assert base.to_dict() == base_document
    assert base_document["transformations"] == []
    transformation = document["transformations"][-1]
    assert transformation["component"] == "pllm/kv-cache-eviction"
    assert transformation["implementation"] == "pllm/mpcache/v1"
    assert transformation["method_id"] == "R23"
    assert transformation["input_digest"] == base.digest
    assert optimized.digest != base.digest

    def kv_state_signature(
        source: dict[str, object], phase: str, field: str
    ) -> tuple[tuple[object, ...], ...]:
        graph = source[phase]
        assert isinstance(graph, dict)
        states = graph[field]
        assert isinstance(states, list)
        return tuple(
            (
                state["layer"],
                state["kind"],
                tuple(state["shape"]),
                state["maximum_sequence"],
            )
            for state in states
            if state["kind"] in {"key", "value"}
        )

    for phase, field in (
        ("prefill", "state_outputs"),
        ("decode", "state_inputs"),
        ("decode", "state_outputs"),
    ):
        assert kv_state_signature(document, phase, field) == kv_state_signature(
            base_document, phase, field
        )
    assert (
        document["decode"]["maximum_key_sequence"]
        == base_document["decode"]["maximum_key_sequence"]
    )
    assert sum(
        state["kind"] == "cache_indices"
        for state in document["prefill"]["state_outputs"]
    ) == CONFIG["num_hidden_layers"]
    assert sum(
        state["kind"] == "cache_indices"
        for state in document["decode"]["state_inputs"]
    ) == CONFIG["num_hidden_layers"]

    decode_operations = {
        operation["id"]: operation for operation in document["decode"]["operations"]
    }
    scores = decode_operations["layer.0.attention_scores"]
    mask = decode_operations["layer.0.causal_mask"]
    assert scores["inputs"][1] == "method.mpcache.layer.0.dynamic_key_gather"
    assert (
        scores["output_shape"][-1]
        < base_document["decode"]["maximum_key_sequence"]
    )
    assert mask["inputs"][2] == "method.mpcache.layer.0.selected_positions"

    levels = {row["operator"]: row["level"] for row in optimized.coverage().operators}
    assert levels["cache_active_indices"] == "missing"
    schema = json.loads(Path("schemas/decoder-plan.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)

    with pytest.raises(ValueError, match="unsupported model component"):
        base.apply(pllm.ComponentRef("example/unknown"))


def test_model_lowering_rejects_unsupported_or_unbounded_models() -> None:
    with pytest.raises(ValueError, match="qwen3_moe has no decoder adapter"):
        pllm.lower_model(
            {**CONFIG, "model_type": "qwen3_moe"},
            batch=1,
            max_input_tokens=128,
            max_new_tokens=32,
        )
    with pytest.raises(ValueError, match="permits"):
        pllm.lower_model(
            CONFIG,
            batch=1,
            max_input_tokens=32768,
            max_new_tokens=1,
        )


def test_lower_model_accepts_exact_gemma4_e4b_outer_config() -> None:
    source = GEMMA4_E4B_CONFIG.read_bytes()
    plan = pllm.lower_model(
        source,
        batch=2,
        max_input_tokens=1_024,
        max_new_tokens=32,
    )

    document = plan.to_dict()
    assert document["model_family"] == "gemma4_text"
    assert document["adapter"] == "pllm.gemma4_e4b_text.v1"
    assert plan.prefill["state_inputs"] == ()
    assert len(plan.prefill["state_outputs"]) == 48
    assert len(plan.decode["state_inputs"]) == 48
    missing = {
        row["operator"]
        for row in plan.coverage().to_dict()["operators"]
        if row["level"] == "missing"
    }
    assert "permute" in missing

    direct_text_config = json.dumps(json.loads(source)["text_config"])
    with pytest.raises(ValueError, match="gemma4_text has no decoder adapter"):
        pllm.lower_model(
            direct_text_config,
            batch=1,
            max_input_tokens=8,
            max_new_tokens=2,
        )


def test_lower_model_accepts_exact_gemma4_e2b_outer_config() -> None:
    plan = pllm.lower_model(
        GEMMA4_E2B_CONFIG.read_bytes(),
        batch=2,
        max_input_tokens=1_024,
        max_new_tokens=32,
    )

    document = plan.to_dict()
    assert document["model_family"] == "gemma4_text"
    assert document["adapter"] == "pllm.gemma4_e2b_text.v1"
    assert len(plan.prefill["state_outputs"]) == 30
    assert len(plan.decode["state_inputs"]) == 30
    operations = {row["id"]: row for row in plan.prefill["operations"]}
    assert operations["layer.14.gate_proj"]["output_shape"][-1] == 6_144
    assert operations["layer.15.gate_proj"]["output_shape"][-1] == 12_288
    assert operations["layer.34.attention_scores"]["inputs"][1] == "layer.14.key_append"


@pytest.mark.parametrize(("fixture", "adapter", "family"), MODEL_FIXTURES)
def test_lower_model_accepts_pinned_cross_family_configs(
    fixture: Path, adapter: str, family: str
) -> None:
    plan = pllm.lower_model(
        fixture.read_bytes(),
        batch=2,
        max_input_tokens=16,
        max_new_tokens=4,
    )
    document = plan.to_dict()
    assert document["adapter"] == adapter
    assert document["model_family"] == family
