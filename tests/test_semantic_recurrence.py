"""Bounded gated-delta recurrence against independent PyTorch arithmetic."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from pllm.state import BoundedGatedDeltaRecurrence, HybridStateError, bounded_gated_delta_decay
from pllm.runtime.semantic_hybrid import execute_declared_gated_rms_norm


def _torch_recurrent(query, key, value, decay, beta, state):
    torch = pytest.importorskip("torch")
    q = torch.from_numpy(query.copy())
    k = torch.from_numpy(key.copy())
    v = torch.from_numpy(value.copy())
    g = torch.from_numpy(decay.copy())
    b = torch.from_numpy(beta.copy())
    prior = torch.from_numpy(state.copy())
    repeats = value.shape[2] // key.shape[2]
    q = q.repeat_interleave(repeats, dim=2)
    k = k.repeat_interleave(repeats, dim=2)
    q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6) / math.sqrt(query.shape[-1])
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    outputs = []
    for token in range(query.shape[1]):
        prior = prior * g[:, token].exp()[:, :, None, None]
        remember = (prior * k[:, token, :, :, None]).sum(dim=-2)
        correction = (v[:, token] - remember) * b[:, token, :, None]
        prior = prior + k[:, token, :, :, None] * correction[:, :, None, :]
        outputs.append((prior * q[:, token, :, :, None]).sum(dim=-2))
    return torch.stack(outputs, dim=1).numpy(), prior.numpy()


@pytest.mark.parametrize("key_heads,value_heads,key_dim,value_dim", [(1, 1, 4, 7), (2, 4, 8, 6), (2, 6, 16, 4)])
def test_gated_delta_prefill_decode_matches_torch_recurrent(key_heads, value_heads, key_dim, value_dim):
    rng = np.random.default_rng(0x35AC + key_dim)
    kernel = BoundedGatedDeltaRecurrence(key_heads, value_heads, key_dim, value_dim)
    query = rng.normal(size=(1, 7, key_heads, key_dim)).astype(np.float32)
    key = rng.normal(size=query.shape).astype(np.float32)
    value = rng.normal(size=(1, 7, value_heads, value_dim)).astype(np.float32)
    decay = -(rng.random(size=(1, 7, value_heads)) + 0.05).astype(np.float32)
    beta = rng.random(size=decay.shape).astype(np.float32)
    state = kernel.initial_state()
    original = state.copy()
    prefill, next_state = kernel.evaluate(
        query[:, :3], key[:, :3], value[:, :3], decay[:, :3], beta[:, :3], state,
        mode="prefill",
    )
    expected, expected_state = _torch_recurrent(
        query[:, :3], key[:, :3], value[:, :3], decay[:, :3], beta[:, :3], state,
    )
    np.testing.assert_allclose(prefill, expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(next_state, expected_state, rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(state, original)
    decoded = []
    for step in range(3, 7):
        actual, next_state = kernel.evaluate(
            query[:, step:step + 1], key[:, step:step + 1], value[:, step:step + 1],
            decay[:, step:step + 1], beta[:, step:step + 1], next_state, mode="decode",
        )
        expected, expected_state = _torch_recurrent(
            query[:, step:step + 1], key[:, step:step + 1], value[:, step:step + 1],
            decay[:, step:step + 1], beta[:, step:step + 1], expected_state,
        )
        np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=5e-6)
        np.testing.assert_allclose(next_state, expected_state, rtol=4e-5, atol=5e-6)
        decoded.append(actual)
    full, full_state = kernel.evaluate(query, key, value, decay, beta, kernel.initial_state(), mode="prefill")
    np.testing.assert_allclose(full, np.concatenate([prefill, *decoded], axis=1), rtol=4e-5, atol=5e-6)
    np.testing.assert_allclose(full_state, next_state, rtol=4e-5, atol=5e-6)


def test_gated_delta_matches_upstream_chunk_reference_at_checked_tiny_width() -> None:
    torch = pytest.importorskip("torch")
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    rng = np.random.default_rng(0x3564)
    kernel = BoundedGatedDeltaRecurrence(1, 2, 4, 3)
    q = rng.normal(size=(1, 5, 1, 4)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    v = rng.normal(size=(1, 5, 2, 3)).astype(np.float32)
    decay = -(rng.random(size=(1, 5, 2)) + 0.02).astype(np.float32)
    beta = rng.random(size=decay.shape).astype(np.float32)
    actual, state = kernel.evaluate(q, k, v, decay, beta, kernel.initial_state(), mode="prefill")
    pure_torch = inspect.unwrap(torch_chunk_gated_delta_rule)
    upstream_output, upstream_state = pure_torch(
        *[torch.from_numpy(values.copy()) for values in (q, k, v, decay, beta)],
        chunk_size=64, initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    np.testing.assert_allclose(actual, upstream_output.numpy(), rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(state, upstream_state.numpy(), rtol=2e-4, atol=2e-5)


def test_delta_decay_matches_torch_and_fail_closed() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0x3565)
    projected = rng.normal(size=(1, 5, 4)).astype(np.float32)
    a_log = np.log(rng.uniform(0.01, 16.0, size=4)).astype(np.float32)
    bias = rng.normal(size=4).astype(np.float32)
    actual = bounded_gated_delta_decay(projected, a_log, bias)
    expected = -torch.exp(torch.from_numpy(a_log)) * torch.nn.functional.softplus(
        torch.from_numpy(projected) + torch.from_numpy(bias)
    )
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-6, atol=1e-6)
    with pytest.raises(HybridStateError, match="finite"):
        bounded_gated_delta_decay(projected, np.full_like(a_log, 100.0), bias)


def test_delta_invalid_inputs_do_not_mutate_state() -> None:
    kernel = BoundedGatedDeltaRecurrence(1, 2, 4, 3)
    q = np.ones((1, 2, 1, 4), dtype=np.float32)
    v = np.ones((1, 2, 2, 3), dtype=np.float32)
    g = -np.ones((1, 2, 2), dtype=np.float32)
    beta = np.full_like(g, 0.5)
    prior = kernel.initial_state()
    for query, decay, confidence, state, mode in [
        (q, g, beta, prior, "decode"),
        (q.astype(np.float64), g, beta, prior, "prefill"),
        (q, np.abs(g), beta, prior, "prefill"),
        (q, g, beta + 1, prior, "prefill"),
        (q, g, beta, prior.astype(np.float64), "prefill"),
        (q, g, beta, np.full_like(prior, np.nan), "prefill"),
    ]:
        snapshot = prior.copy()
        with pytest.raises(HybridStateError):
            kernel.evaluate(query, q, v, decay, confidence, state, mode=mode)
        np.testing.assert_array_equal(prior, snapshot)
    with pytest.raises(HybridStateError):
        BoundedGatedDeltaRecurrence(3, 5, 4, 4)
    with pytest.raises(HybridStateError):
        BoundedGatedDeltaRecurrence(32, 64, 256, 256)


def test_full_width_bounded_recurrent_state_is_finite_and_retained() -> None:
    kernel = BoundedGatedDeltaRecurrence(16, 32, 128, 128)
    q = np.full((1, 1, 16, 128), np.float32(0.125))
    v = np.full((1, 1, 32, 128), np.float32(0.25))
    decay = np.full((1, 1, 32), np.float32(-0.25))
    beta = np.full((1, 1, 32), np.float32(0.5))
    output, state = kernel.evaluate(q, q, v, decay, beta, kernel.initial_state(), mode="decode")
    assert output.shape == (1, 1, 32, 128)
    assert state.nbytes == 2 * 1024 * 1024
    assert np.all(np.isfinite(output))


def test_gated_rms_norm_matches_upstream_float32_order_and_rejects_bad_weight() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0x3566)
    input_value = rng.normal(size=(1, 3, 2, 6)).astype(np.float32)
    gate = rng.normal(size=input_value.shape).astype(np.float32)
    weight = rng.normal(size=6).astype(np.float32)
    operation = {
        "operator": "rms_norm_gated", "inputs": ["input", "gate"],
        "output_shape": [1, 3, 2, 6],
        "attributes": {
            "epsilon": "0.000001", "weight": "norm.weight", "activation": "silu",
            "weight_offset": 0, "norm_before_gate": True,
        },
    }
    actual = execute_declared_gated_rms_norm(operation, {"input": input_value, "gate": gate}, weight)
    x = torch.from_numpy(input_value)
    reference = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6))
    reference = reference * torch.from_numpy(weight)
    reference *= torch.nn.functional.silu(torch.from_numpy(gate))
    np.testing.assert_allclose(actual, reference.numpy(), rtol=3e-6, atol=2e-6)
    with pytest.raises(HybridStateError, match="shapes"):
        execute_declared_gated_rms_norm(operation, {"input": input_value, "gate": gate}, weight[:4])
    forged = {**operation, "attributes": {**operation["attributes"], "weight_offset": 1}}
    with pytest.raises(HybridStateError, match="declaration"):
        execute_declared_gated_rms_norm(forged, {"input": input_value, "gate": gate}, weight)
