"""Bounded stateful convolution versus the upstream depthwise PyTorch arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.model_binding import RuntimeBindingError, _runtime_config
from pllm.runtime.semantic_stages import scheduled_stage_specs
from pllm.state import BoundedDepthwiseCausalConvolution, HybridStateError


def _torch_reference(
    samples: np.ndarray, state: np.ndarray, weight: np.ndarray, bias: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    torch = pytest.importorskip("torch")
    from torch.nn import functional as F

    prior = torch.from_numpy(state.copy())
    inputs = torch.from_numpy(samples.transpose(0, 2, 1).copy())
    filter_weight = torch.from_numpy(weight[:, None, :].copy())
    filter_bias = None if bias is None else torch.from_numpy(bias.copy())
    joined = torch.cat((prior, inputs), dim=-1)
    output = F.silu(
        F.conv1d(joined, filter_weight, filter_bias, groups=weight.shape[0])[:, :, -samples.shape[1] :]
    ).transpose(1, 2)
    return output.numpy(), joined[:, :, -weight.shape[1] :].numpy()


@pytest.mark.parametrize("kernel_size", [1, 2, 4, 7])
@pytest.mark.parametrize("with_bias", [False, True])
def test_causal_convolution_prefill_decode_state_matches_torch(
    kernel_size: int, with_bias: bool
) -> None:
    rng = np.random.default_rng(0xAC52 + kernel_size)
    weight = rng.normal(size=(5, kernel_size)).astype(np.float32)
    bias = rng.normal(size=5).astype(np.float32) if with_bias else None
    reference_weight = weight.copy()
    reference_bias = None if bias is None else bias.copy()
    kernel = BoundedDepthwiseCausalConvolution(weight, bias=bias)
    weight[:] = 120.0  # caller cannot change the committed coefficient array
    if bias is not None:
        bias[:] = -120.0
    first = rng.normal(size=(1, 3, 5)).astype(np.float32)
    second = rng.normal(size=(1, 4, 5)).astype(np.float32)
    state = kernel.initial_state()
    original = state.copy()
    prefill, state_after_prefill = kernel.evaluate(first, state, mode="prefill")
    expected_prefill, expected_state = _torch_reference(
        first, state, reference_weight, reference_bias
    )
    np.testing.assert_allclose(prefill, expected_prefill, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(state_after_prefill, expected_state)
    np.testing.assert_array_equal(state, original)
    for token in second.transpose(1, 0, 2):
        actual, state_after_prefill = kernel.evaluate(token[:, None, :], state_after_prefill, mode="decode")
        expected, expected_state = _torch_reference(
            token[:, None, :], expected_state, reference_weight, reference_bias
        )
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(state_after_prefill, expected_state)
    full = np.concatenate((first, second), axis=1)
    full_output, full_state = kernel.evaluate(full, kernel.initial_state(), mode="prefill")
    expected_full, expected_full_state = _torch_reference(
        full, kernel.initial_state(), reference_weight, reference_bias
    )
    np.testing.assert_allclose(full_output, expected_full, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(full_state, expected_full_state)


def test_causal_convolution_bad_inputs_leave_state_and_weights_unchanged() -> None:
    kernel = BoundedDepthwiseCausalConvolution(
        np.array([[1.0, 2.0, 3.0], [-0.5, 0.25, 0.5]], dtype=np.float32)
    )
    valid = np.ones((1, 2, 2), dtype=np.float32)
    state = kernel.initial_state()
    for samples, saved, mode in [
        (valid.astype(np.float64), state, "prefill"),
        (valid, state.astype(np.float64), "prefill"),
        (valid, state, "decode"),
        (np.full_like(valid, np.nan), state, "prefill"),
        (np.ones((1, 257, 2), dtype=np.float32), state, "prefill"),
        (valid, np.full_like(state, np.nan), "prefill"),
        (valid, state, "unknown"),
    ]:
        before = state.copy()
        with pytest.raises(HybridStateError):
            kernel.evaluate(samples, saved, mode=mode)  # type: ignore[arg-type]
        np.testing.assert_array_equal(state, before)
    huge = np.full((1, 1, 2), np.finfo(np.float32).max, dtype=np.float32)
    with pytest.raises(HybridStateError, match="finite"):
        kernel.evaluate(huge, state, mode="prefill")
    np.testing.assert_array_equal(state, kernel.initial_state())

    with pytest.raises(HybridStateError):
        BoundedDepthwiseCausalConvolution(np.ones((8193, 4), dtype=np.float32))
    with pytest.raises(HybridStateError):
        BoundedDepthwiseCausalConvolution(np.ones((2, 17), dtype=np.float32))
    with pytest.raises(HybridStateError):
        BoundedDepthwiseCausalConvolution(np.ones((2, 3), dtype=np.float32), bias=np.ones(3, dtype=np.float32))


def test_causal_convolution_accepts_bounded_full_width_qwen35_state() -> None:
    channels, kernel_size = 8192, 4
    weights = np.ones((channels, kernel_size), dtype=np.float32) / kernel_size
    reference = BoundedDepthwiseCausalConvolution(weights)
    values = np.ones((1, 2, channels), dtype=np.float32)
    output, state = reference.evaluate(values, reference.initial_state(), mode="prefill")
    assert output.shape == (1, 2, channels)
    assert state.shape == (1, channels, kernel_size)
    assert np.all(np.isfinite(output))


def test_pinned_hybrid_source_has_explicit_state_and_native_schedule() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "crates/pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json"
    )
    plan = lower_model(source.read_bytes(), batch=1, max_input_tokens=5, max_new_tokens=2)
    prefill = plan.prefill
    initialized = next(row for row in prefill["operations"] if row["operator"] == "state_initialize")
    assert initialized["attributes"] == {
        "initial_value": 0,
        "dtype": "float32",
        "state_kind": "convolution",
    }
    assert initialized["id"] not in {row["id"] for row in prefill["state_inputs"]}
    schedule = plan.runtime_schedule(MaskedLinearCpu(Model("org/model"))).to_dict()
    assert schedule["prefill"]["state_inputs"] == []
    assert len(schedule["prefill"]["state_outputs"]) == 64
    assert schedule["model_plan_digest"] == plan.digest
    stages = scheduled_stage_specs(plan, MaskedLinearCpu(Model("org/model")))
    assert stages[0].role == "token_lookup" and stages[-1].role == "lm_head"
    grouped = [stage for stage in stages if stage.role == "semantic_linear" and len(stage.weight_keys) > 1]
    assert any(len(stage.weight_keys) >= 4 for stage in grouped), [
        (stage.id, stage.role, len(stage.weight_keys))
        for stage in stages if len(stage.weight_keys) > 1
    ][:8]


def test_missing_source_bos_requires_explicit_non_adding_tokenizer() -> None:
    config = json.loads((
        Path(__file__).resolve().parents[1]
        / "crates/pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json"
    ).read_text())["text_config"]
    assert "bos_token_id" not in config
    descriptor = {"bos_token_id": 2, "add_bos_token": False}
    transport = {
        **config,
        "bos_token_id": 2,
        "bos_token_policy": "nonempty_only",
        "semantic_source_config": {"text_config": config},
    }
    runtime = _runtime_config(transport, nested_source=True, tokenizer_descriptor=descriptor)
    assert runtime["bos_token_id"] == 2
    assert "bos_token_id" not in config
    for forged in (None, {**descriptor, "add_bos_token": True}, {**descriptor, "bos_token_id": -1}):
        with pytest.raises(RuntimeBindingError, match="BOS"):
            _runtime_config(transport, nested_source=True, tokenizer_descriptor=forged)
    with pytest.raises(RuntimeBindingError, match="BOS"):
        _runtime_config({**transport, "bos_token_policy": "ignored"}, nested_source=True, tokenizer_descriptor=descriptor)
