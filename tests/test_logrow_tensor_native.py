"""End-to-end Q7 tensor parity through the opaque native research handle."""

from __future__ import annotations

import struct

import pytest

from pllm import _native, lower_model
from pllm.protocols import LogRowGarbledLookup, prepare_logrow_q7_tensor_reference

QWEN2 = {
    "model_type": "qwen2",
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 32,
    "max_position_embeddings": 32,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}
QWEN3 = {
    **QWEN2,
    "model_type": "qwen3",
    "head_dim": 4,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rope_scaling": None,
    "sliding_window": None,
    "use_sliding_window": False,
    "use_cache": True,
    "max_window_layers": 1,
    "layer_types": ["full_attention"],
}


def _profile(*, nonuniform: bool = False):
    counts = [0] * 257
    if nonuniform:
        counts[128] = 5
    return _native.fit_compact_silu_q7_reference(struct.pack("<257I", *counts), 4)


@pytest.mark.parametrize("source", [QWEN2, QWEN3], ids=["qwen2", "qwen3"])
@pytest.mark.parametrize("mode,elements", [("prefill", 32), ("decode", 16)])
def test_semantic_q7_tensor_native_parity_and_one_use(
    source: dict, mode: str, elements: int
) -> None:
    plan = lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=1)
    graph = plan.to_dict()[mode]
    operation = next(item["id"] for item in graph["operations"] if item["operator"] == "silu")
    profile = _profile()
    handle = prepare_logrow_q7_tensor_reference(
        plan,
        profile,
        mode=mode,
        operation_id=operation,
        max_elements=elements,
        max_evaluator_material_bytes=elements * 2144,
    )
    assert handle.elements == elements
    assert handle.evaluator_material_bytes == elements * 2144
    assert len(handle.binding_digest) == 64
    values = ([-128, -87, 0, 49, 128] * elements)[:elements]
    output = handle.evaluate(struct.pack(f"<{elements}h", *values))
    assert struct.unpack(f"<{elements}h", output) == tuple(
        profile.evaluate(value) for value in values
    )
    with pytest.raises(ValueError, match="already consumed"):
        handle.evaluate(struct.pack(f"<{elements}h", *values))
    with pytest.raises(ValueError, match="not yet implemented"):
        LogRowGarbledLookup()


def test_native_rejects_budget_plan_profile_and_bad_input_before_reuse() -> None:
    plan = lower_model(QWEN2, batch=1, max_input_tokens=2, max_new_tokens=1)
    operation = next(
        item["id"] for item in plan.to_dict()["prefill"]["operations"] if item["operator"] == "silu"
    )
    profile = _profile()
    with pytest.raises(ValueError, match="material limit"):
        _native.prepare_logrow_q7_tensor_reference(
            plan.canonical_bytes(), "prefill", operation, profile, 32, 68607
        )
    with pytest.raises(ValueError, match="element limit"):
        _native.prepare_logrow_q7_tensor_reference(
            plan.canonical_bytes(), "prefill", operation, profile, 31, 68608
        )
    with pytest.raises(ValueError, match="semantic operation missing is absent"):
        _native.prepare_logrow_q7_tensor_reference(
            plan.canonical_bytes(), "prefill", "missing", profile, 32, 68608
        )
    with pytest.raises(ValueError, match="decoder plan"):
        _native.prepare_logrow_q7_tensor_reference(b"{}", "prefill", operation, profile, 32, 68608)
    handle = _native.prepare_logrow_q7_tensor_reference(
        plan.canonical_bytes(), "prefill", operation, profile, 32, 68608
    )
    with pytest.raises(ValueError, match="must be bytes"):
        handle.evaluate([0] * 32)
    with pytest.raises(ValueError, match="already consumed"):
        handle.evaluate(bytes(64))
    with pytest.raises(TypeError, match="immutable ModelPlan"):
        prepare_logrow_q7_tensor_reference(
            plan.to_dict(),
            profile,
            mode="prefill",
            operation_id=operation,
            max_elements=32,
            max_evaluator_material_bytes=68608,
        )
    handle = _native.prepare_logrow_q7_tensor_reference(
        plan.canonical_bytes(),
        "prefill",
        operation,
        _profile(nonuniform=True),
        32,
        68608,
    )
    with pytest.raises(ValueError, match="domain"):
        handle.evaluate(struct.pack("<32h", *([0] * 31 + [129])))
    with pytest.raises(ValueError, match="already consumed"):
        handle.evaluate(bytes(64))
