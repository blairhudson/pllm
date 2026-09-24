"""End-to-end Q7 tensor parity through the opaque native research handle."""

from __future__ import annotations

import struct
import math

import pytest

from pllm import _native, lower_model
from pllm.protocols import (
    LogRowGarbledLookup,
    estimate_logrow_q7_session_reference,
    prepare_logrow_q7_tensor_reference,
)

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


def test_float32_bridge_is_bounded_accurate_and_burns_on_domain_failures() -> None:
    plan = lower_model(QWEN2, batch=1, max_input_tokens=2, max_new_tokens=1)
    operation = next(item["id"] for item in plan.decode["operations"] if item["operator"] == "silu")
    profile = _profile()
    values = [-1.0, -0.875, -0.00390625, 0.0, 0.00390625, 0.4999, 1.0] + [0.0] * 9
    handle = prepare_logrow_q7_tensor_reference(
        plan,
        profile,
        mode="decode",
        operation_id=operation,
        max_elements=16,
        max_evaluator_material_bytes=34304,
    )
    output = struct.unpack("<16f", handle.evaluate_float32(struct.pack("<16f", *values)))
    for value, actual in zip(values, output, strict=True):
        quantized = round(value * 128)
        assert actual == profile.evaluate(quantized) / 128
        assert abs(actual - value / (1.0 + math.exp(-value))) <= (
            1 / 256 + (profile.maximum_encoded_error + 0.5) / 128
        )
    with pytest.raises(ValueError, match="already consumed"):
        handle.evaluate(bytes(32))
    for bad in [float("nan"), float("inf"), float("-inf"), 1.001, -1.001]:
        handle = prepare_logrow_q7_tensor_reference(
            plan,
            profile,
            mode="decode",
            operation_id=operation,
            max_elements=16,
            max_evaluator_material_bytes=34304,
        )
        with pytest.raises(ValueError, match="FloatInputOutOfRange"):
            handle.evaluate_float32(struct.pack("<16f", *([0.0] * 15 + [bad])))
        with pytest.raises(ValueError, match="already consumed"):
            handle.evaluate_float32(bytes(64))
    handle = prepare_logrow_q7_tensor_reference(
        plan,
        profile,
        mode="decode",
        operation_id=operation,
        max_elements=16,
        max_evaluator_material_bytes=34304,
    )
    with pytest.raises(ValueError, match="must be bytes"):
        handle.evaluate_float32([0.0] * 16)
    with pytest.raises(ValueError, match="already consumed"):
        handle.evaluate(bytes(32))


@pytest.mark.parametrize("source", [QWEN2, QWEN3], ids=["qwen2", "qwen3"])
def test_session_estimate_admits_total_before_any_one_use_material(source: dict) -> None:
    plan = lower_model(source, batch=1, max_input_tokens=2, max_new_tokens=2)
    profile = _profile()
    estimate = estimate_logrow_q7_session_reference(
        plan,
        profile,
        max_elements=32,
        max_evaluator_material_bytes=68608,
        max_decode_steps=2,
        max_session_evaluator_material_bytes=137216,
    )
    assert estimate.plan_digest == plan.digest
    assert estimate.profile_digest == profile.digest
    assert estimate.prefill_elements == 32
    assert estimate.decode_elements_per_step == 16
    assert estimate.reserved_evaluator_material_bytes == 137216
    assert estimate.largest_tensor_material_bytes == 68608
    assert len(estimate.estimate_digest) == 64
    with pytest.raises(ValueError, match="session exceeds its material limit"):
        estimate_logrow_q7_session_reference(
            plan,
            profile,
            max_elements=32,
            max_evaluator_material_bytes=68608,
            max_decode_steps=2,
            max_session_evaluator_material_bytes=137215,
        )
    with pytest.raises(ValueError, match="semantic state bound"):
        estimate_logrow_q7_session_reference(
            plan,
            profile,
            max_elements=32,
            max_evaluator_material_bytes=68608,
            max_decode_steps=3,
            max_session_evaluator_material_bytes=200000,
        )
