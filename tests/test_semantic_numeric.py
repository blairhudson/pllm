from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
import torch

from pllm.modeling import lower_model
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.semantic_numeric import SemanticNumericError, round_bfloat16
from pllm.runtime.transformer_client import TransformerClientError


@pytest.fixture(scope="module")
def operations() -> dict[str, dict]:
    path = (
        Path(__file__).resolve().parents[1]
        / "crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"
    )
    plan = lower_model(json.loads(path.read_text()), batch=1, max_input_tokens=2, max_new_tokens=2)
    return {operation["id"]: operation for operation in plan.to_dict()["prefill"]["operations"]}


def _local(operation: dict, value: np.ndarray) -> np.ndarray:
    runtime = object.__new__(SemanticDecoderRuntime)
    return runtime._local(operation, {operation["inputs"][0]: value}, {}, {})


@pytest.mark.parametrize(
    ("operation_id", "coefficient"),
    [
        ("main_embedding_scaled", math.sqrt(1536)),
        ("ple_context_scaled", 1.0 / math.sqrt(1536)),
        ("layer.0.attention_scale", 1.0),
    ],
)
def test_public_semantic_bfloat16_scales_match_torch(
    operations: dict[str, dict], operation_id: str, coefficient: float
) -> None:
    source = np.asarray([[-3.5, -0.123, 0.0, 0.125, 1.75, 17.0]], dtype=np.float32)
    result = _local(operations[operation_id], source)
    expected = (
        torch.from_numpy(source).to(torch.bfloat16)
        * torch.tensor(coefficient, dtype=torch.bfloat16)
    ).float().numpy()
    np.testing.assert_array_equal(result, expected)


def test_semantic_bfloat16_gelu_and_softcap_match_torch(operations: dict[str, dict]) -> None:
    source = np.asarray([[-8.0, -3.25, -1.0, -0.125, 0.0, 0.375, 2.5, 8.0]], dtype=np.float32)
    gelu = _local(operations["layer.0.gelu_tanh"], source)
    expected_gelu = torch.nn.functional.gelu(
        torch.from_numpy(source).to(torch.bfloat16), approximate="tanh"
    ).float().numpy()
    np.testing.assert_array_equal(gelu, expected_gelu)
    grid = np.linspace(-8.0, 8.0, 257, dtype=np.float32)
    np.testing.assert_allclose(
        _local(operations["layer.0.gelu_tanh"], grid),
        torch.nn.functional.gelu(
            torch.from_numpy(grid).to(torch.bfloat16), approximate="tanh"
        ).float().numpy(),
        rtol=0,
        atol=3.1e-5,
    )

    logits = np.asarray([[-500.0, -37.0, -0.125, 0.0, 4.0, 45.0, 500.0]], dtype=np.float32)
    result = _local(operations["logit_softcap"], logits)
    cap = torch.tensor(30, dtype=torch.bfloat16)
    expected = ((torch.from_numpy(logits).to(torch.bfloat16) / cap).tanh() * cap).float().numpy()
    np.testing.assert_array_equal(result, expected)


def test_semantic_nonlinear_numeric_contract_across_all_finite_bfloat16_values(
    operations: dict[str, dict]
) -> None:
    encoded = np.arange(65536, dtype=np.uint16).view(ml_dtypes.bfloat16).astype(np.float32)
    source = encoded[np.isfinite(encoded)]
    torch_input = torch.from_numpy(source).to(torch.bfloat16)
    gelu = _local(operations["layer.0.gelu_tanh"], source)
    oracle = torch.nn.functional.gelu(torch_input, approximate="tanh").float().numpy()
    np.testing.assert_allclose(gelu, oracle, rtol=0, atol=3.1e-5)
    assert np.all(np.isfinite(gelu))
    cap = torch.tensor(30, dtype=torch.bfloat16)
    np.testing.assert_array_equal(
        _local(operations["logit_softcap"], source),
        ((torch_input / cap).tanh() * cap).float().numpy(),
    )


def test_bfloat16_normalization_uses_declared_weight_offset_and_optional_scale(
    operations: dict[str, dict],
) -> None:
    source = np.asarray([[[0.125, -0.75, 2.5, -0.25], [3.0, 0.0, -1.5, 0.5]]], dtype=np.float32)
    weights = np.asarray([-0.25, 0.0, 0.25, 1.0], dtype=np.float32)
    runtime = object.__new__(SemanticDecoderRuntime)
    norm = operations["layer.0.input_norm"]
    runtime._tensors = {norm["attributes"]["weight"]: "gamma"}
    runtime.bundle = SimpleNamespace(arrays={"gamma": weights})
    expected_input = torch.from_numpy(source).to(torch.bfloat16).float()
    normalized = expected_input * torch.rsqrt(
        expected_input.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    assert norm["attributes"]["weight_offset"] == 0
    expected = (normalized * torch.from_numpy(weights)).to(torch.bfloat16).float().numpy()
    np.testing.assert_array_equal(
        runtime._local(norm, {norm["inputs"][0]: source}, {}, {}), expected
    )
    unweighted = operations["layer.0.v_norm"]
    np.testing.assert_array_equal(
        runtime._local(unweighted, {unweighted["inputs"][0]: source}, {}, {}),
        normalized.to(torch.bfloat16).float().numpy(),
    )
    bad = {**norm, "attributes": {**norm["attributes"], "with_scale": False}}
    with pytest.raises(TransformerClientError, match="inconsistent"):
        runtime._local(bad, {bad["inputs"][0]: source}, {}, {})
    runtime.bundle = SimpleNamespace(arrays={"gamma": weights[:3]})
    with pytest.raises(TransformerClientError, match="wrong shape"):
        runtime._local(norm, {norm["inputs"][0]: source}, {}, {})


def test_bfloat16_softmax_handles_masked_queries_without_nan(operations: dict[str, dict]) -> None:
    source = np.asarray(
        [[[[1.0, -np.inf, -2.0, -np.inf], [-np.inf, 0.0, -1.0, -3.0]]]],
        dtype=np.float32,
    )
    result = _local(operations["layer.0.softmax"], source)
    reference = torch.nn.functional.softmax(torch.from_numpy(source), dim=-1).to(torch.bfloat16)
    np.testing.assert_array_equal(result, reference.float().numpy())
    assert np.all(np.isfinite(result))
    with pytest.raises(TransformerClientError, match="fully masked"):
        _local(operations["layer.0.softmax"], np.full((1, 1, 1, 4), -np.inf, dtype=np.float32))


@pytest.mark.parametrize("rope_type", ["default", "proportional"])
def test_pinned_rotary_contract_matches_bfloat16_torch_steps(
    operations: dict[str, dict], rope_type: str
) -> None:
    operation = next(
        row for row in operations.values()
        if row["operator"] == "rotary_embedding" and row["attributes"]["rope_type"] == rope_type
    )
    attrs = operation["attributes"]
    head = attrs["head_dim"]
    positions = np.asarray([0, 1, 2, 127, 1023], dtype=np.int64)
    source = np.linspace(-1.5, 1.5, num=positions.size * 2 * head, dtype=np.float32).reshape(
        1, positions.size, 2, head
    )
    torch_source = torch.from_numpy(source).bfloat16()
    rounded = torch_source.float().numpy()
    actual = SemanticDecoderRuntime._rotary(rounded, positions, attrs)

    fraction = attrs["partial_rotary_factor"]
    angle_count = head * fraction["numerator"] // (2 * fraction["denominator"])
    if rope_type == "default":
        angle_count = head // 2
    power = torch.arange(0, 2 * angle_count, 2, dtype=torch.float32) / head
    inverse = 1 / (attrs["theta"] ** power)
    if inverse.numel() < head // 2:
        inverse = torch.cat((inverse, torch.zeros(head // 2 - inverse.numel())))
    angles = torch.from_numpy(positions).float()[:, None] * inverse[None, :]
    embedded = torch.cat((angles, angles), dim=-1)
    cosine = embedded.cos().bfloat16()[None, :, None, :]
    sine = embedded.sin().bfloat16()[None, :, None, :]
    rotated = torch.cat((-torch_source[..., head // 2 :], torch_source[..., : head // 2]), dim=-1)
    expected = (torch_source * cosine + rotated * sine).float().numpy()
    np.testing.assert_array_equal(actual, expected)

    with pytest.raises(TransformerClientError, match="BF16 rotary"):
        SemanticDecoderRuntime._rotary(
            rounded, positions, {**attrs, "input_layout": "batch_heads_sequence_feature"}
        )
    with pytest.raises(TransformerClientError, match="BF16 rotary"):
        SemanticDecoderRuntime._rotary(source, positions, attrs)


def test_wavelength_transition_matches_independent_torch_rotary_oracle() -> None:
    attrs = {
        "theta": 10000, "rotary_dimensions": 16, "pairing": "split_half",
        "position_policy": "sequential_absolute",
        "coefficient_profile": "pllm.numeric.rope.float32.wavelength.v1",
        "input_layout": "batch_heads_sequence_feature",
        "output_layout": "batch_heads_sequence_feature", "tail_policy": "unchanged",
        "frequency_scaling": {
            "kind": "wavelength_transition", "factor": 8.0,
            "original_max_position_embeddings": 32,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        },
    }
    positions = np.asarray([0, 1, 7, 63, 255, 8192, 131071], dtype=np.int64)
    source = np.linspace(-1.0, 1.0, positions.size * 2 * 16, dtype=np.float32).reshape(
        1, 2, positions.size, 16,
    )
    actual = SemanticDecoderRuntime._rotary(source, positions, attrs)

    # Independent HF v4.44.2 Llama 3.1 reference formula, evaluated in torch.
    frequency = 1 / (10000 ** (torch.arange(0, 16, 2).float() / 16))
    wavelength = 2 * math.pi / frequency
    assert bool(torch.any(wavelength < 8))
    assert bool(torch.any((wavelength >= 8) & (wavelength <= 32)))
    assert bool(torch.any(wavelength > 32))
    low_scaled = torch.where(wavelength > 32, frequency / 8, frequency)
    smooth = (32 / wavelength - 1) / (4 - 1)
    medium = (1 - smooth) * low_scaled / 8 + smooth * low_scaled
    frequency = torch.where((wavelength >= 8) & (wavelength <= 32), medium, low_scaled)
    angles = torch.from_numpy(positions).float()[:, None] * frequency[None, :]
    embedded = torch.cat((angles, angles), dim=-1)
    x = torch.from_numpy(source)
    rotated = torch.cat((-x[..., 8:], x[..., :8]), dim=-1)
    expected = x * embedded.cos()[None, None] + rotated * embedded.sin()[None, None]
    np.testing.assert_allclose(actual, expected.numpy(), atol=3e-5, rtol=3e-5)

    with pytest.raises(TransformerClientError, match="wavelength rotary"):
        SemanticDecoderRuntime._rotary(
            source, positions, {**attrs, "frequency_scaling": {
                **attrs["frequency_scaling"], "high_freq_factor": 1.0,
            }},
        )
    with pytest.raises(TransformerClientError, match="wavelength rotary"):
        SemanticDecoderRuntime._rotary(
            source, positions, {**attrs, "frequency_scaling": {
                **attrs["frequency_scaling"], "low_freq_factor": 1e-40,
            }},
        )
    with pytest.raises(TransformerClientError, match="wavelength rotary"):
        SemanticDecoderRuntime._rotary(source, positions, {**attrs, "coefficient_profile": "unsafe"})
    with pytest.raises(TransformerClientError, match="wavelength rotary"):
        SemanticDecoderRuntime._rotary(source, np.asarray([-1], dtype=np.int64), attrs)


def test_per_frequency_partial_rotary_matches_independent_torch_oracle() -> None:
    attrs = {
        "theta": 10000, "rotary_dimensions": 6, "rope_type": "longrope",
        "pairing": "split_half", "position_policy": "sequential_absolute",
        "coefficient_profile": "pllm.numeric.rope.float32.per_frequency.v1",
        "input_layout": "batch_heads_sequence_feature",
        "output_layout": "batch_heads_sequence_feature", "tail_policy": "unchanged",
        "frequency_scaling": {
            "kind": "per_frequency_context", "factor": 4.0,
            "original_max_position_embeddings": 16,
            "short_factor": [1.0, 1.25, 2.0], "long_factor": [4.0, 5.0, 6.0],
        },
    }
    positions = np.asarray([0, 1, 7, 15], dtype=np.int64)
    source = np.linspace(-1, 1, 2 * positions.size * 8, dtype=np.float32).reshape(
        1, 2, positions.size, 8,
    )
    actual = SemanticDecoderRuntime._rotary(source, positions, attrs)
    power = torch.arange(0, 6, 2, dtype=torch.float32) / 6
    inverse = 1 / (torch.tensor([1.0, 1.25, 2.0]) * 10000 ** power)
    angles = torch.from_numpy(positions).float()[:, None] * inverse[None, :]
    embedded = torch.cat((angles, angles), dim=-1)
    attention_factor = math.sqrt(1 + math.log(4) / math.log(16))
    cosine = embedded.cos() * attention_factor
    sine = embedded.sin() * attention_factor
    tensor = torch.from_numpy(source)
    rotated = torch.cat((-tensor[..., 3:6], tensor[..., :3]), dim=-1)
    expected = tensor.clone()
    expected[..., :6] = (
        tensor[..., :6] * cosine[None, None] + rotated * sine[None, None]
    )
    np.testing.assert_allclose(actual, expected.numpy(), rtol=0, atol=2e-6)
    np.testing.assert_array_equal(actual[..., 6:], source[..., 6:])
    for mutated in (
        {**attrs, "frequency_scaling": {**attrs["frequency_scaling"], "short_factor": [1.0]}},
        {**attrs, "frequency_scaling": {**attrs["frequency_scaling"], "long_factor": [1e-40] * 3}},
        {**attrs, "coefficient_profile": "unreviewed"},
    ):
        with pytest.raises(TransformerClientError, match="per-frequency rotary"):
            SemanticDecoderRuntime._rotary(source, positions, mutated)
    with pytest.raises(TransformerClientError, match="short context"):
        SemanticDecoderRuntime._rotary(source, np.asarray([0, 1, 7, 16]), attrs)


def test_declared_operator_outputs_round_at_local_and_remote_numeric_edges(
    operations: dict[str, dict],
) -> None:
    values = np.asarray([1.001, -1.001, 0.125], dtype=np.float32)
    reference = torch.from_numpy(values).to(torch.bfloat16).float().numpy()
    for operator_id in (
        "main_embedding",
        "ple_token_embedding",
        "layer.0.q_linear",
        "layer.0.attention_scores",
        "layer.0.attention_values",
        "layer.0.attention_residual",
        "layer.0.gated_multiply",
        "layer.0.feedforward_residual",
        "layer.0.ple_multiply",
        "layer.0.ple_residual",
        "output_head",
    ):
        operation = operations[operator_id]
        assert operation["attributes"]["output_dtype"] == "bfloat16"
        np.testing.assert_array_equal(
            SemanticDecoderRuntime._numeric_output(operation, values), reference
        )
    policy = operations["layer.0.attention_scale"]
    assert policy["attributes"]["factor"]["output_dtype"] == "bfloat16"
    np.testing.assert_array_equal(SemanticDecoderRuntime._numeric_output(policy, values), reference)
    invalid = {**operations["layer.0.q_linear"], "attributes": {"output_dtype": "float16"}}
    with pytest.raises(TransformerClientError, match="numeric representation"):
        SemanticDecoderRuntime._numeric_output(invalid, values)
    with pytest.raises(TransformerClientError, match="must be finite"):
        SemanticDecoderRuntime._numeric_output(
            operations["layer.0.q_linear"], np.asarray([np.inf], dtype=np.float32)
        )


def test_bfloat16_elementwise_edges_match_torch_operand_and_result_rounding(
    operations: dict[str, dict],
) -> None:
    left = torch.tensor([[1.001, -0.753, 3.011]], dtype=torch.bfloat16).float().numpy()
    right = torch.tensor([[0.376, 1.507, -2.126]], dtype=torch.bfloat16).float().numpy()
    runtime = object.__new__(SemanticDecoderRuntime)
    for operator_id, oracle in (
        ("layer.0.attention_residual", lambda a, b: a + b),
        ("layer.0.gated_multiply", lambda a, b: a * b),
    ):
        operation = operations[operator_id]
        values = {operation["inputs"][0]: left, operation["inputs"][1]: right}
        actual = runtime._numeric_output(operation, runtime._local(operation, values, {}, {}))
        expected = oracle(
            torch.from_numpy(left).to(torch.bfloat16),
            torch.from_numpy(right).to(torch.bfloat16),
        ).float().numpy()
        np.testing.assert_array_equal(actual, expected)


def test_semantic_permute_and_slice_follow_declared_axes(operations: dict[str, dict]) -> None:
    values = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    np.testing.assert_array_equal(
        _local(operations["layer.0.q_permute"], values), values.transpose(0, 2, 1, 3)
    )
    packed = np.arange(2 * 35 * 256, dtype=np.float32).reshape(1, 2, 35, 256)
    np.testing.assert_array_equal(
        _local(operations["layer.0.ple_slice"], packed), packed[:, :, 0, :]
    )
    # A shorter request must remain valid under the compiled sequence ceiling.
    bounded = dict(operations["layer.0.ple_slice"])
    bounded["output_shape"] = (1, 4, 256)
    np.testing.assert_array_equal(_local(bounded, packed), packed[:, :, 0, :])


def test_semantic_numeric_rejects_unsupported_or_nonfinite_contracts(
    operations: dict[str, dict]
) -> None:
    source = np.asarray([[1.0, -2.0]], dtype=np.float32)
    for operation_id, key, bad in [
        ("layer.0.gelu_tanh", "approximation", "exact"),
        ("logit_softcap", "formula", "identity"),
        ("layer.0.ple_slice", "axis", 0),
    ]:
        operation = {**operations[operation_id], "attributes": dict(operations[operation_id]["attributes"])}
        operation["attributes"][key] = bad
        with pytest.raises(TransformerClientError, match="unsupported|invalid"):
            _local(operation, source)
    with pytest.raises(SemanticNumericError, match="finite"):
        round_bfloat16(np.asarray([float("nan")], dtype=np.float32))
    with pytest.raises(SemanticNumericError, match="BF16 domain"):
        round_bfloat16(np.asarray([3.4e38], dtype=np.float32))


def test_pinned_gemma_reshapes_and_selects_last_valid_token(
    operations: dict[str, dict],
) -> None:
    runtime = object.__new__(SemanticDecoderRuntime)
    runtime.position = 0
    ple = operations["ple_token_reshape"]
    packed = np.arange(2 * 8960, dtype=np.float32).reshape(2, 8960)
    result = runtime._local(ple, {ple["inputs"][0]: packed}, {}, {})
    assert result.shape == (1, 2, 35, 256)
    np.testing.assert_array_equal(result.reshape(2, -1), packed)

    query = operations["layer.0.q_heads"]
    flattened = np.arange(2 * 2048, dtype=np.float32).reshape(2, 2048)
    heads = runtime._local(query, {query["inputs"][0]: flattened}, {}, {})
    assert heads.shape == (1, 2, 8, 256)
    merged = operations["layer.0.attention_hidden"]
    np.testing.assert_array_equal(
        runtime._local(merged, {merged["inputs"][0]: heads}, {}, {}), flattened
    )

    last = operations["last_hidden"]
    hidden = np.arange(2 * 1536, dtype=np.float32).reshape(1, 2, 1536)
    values = {last["inputs"][0]: hidden, last["inputs"][1]: np.asarray([2], dtype=np.int64)}
    np.testing.assert_array_equal(runtime._local(last, values, {}, {}), hidden[:, -1, :])
    values[last["inputs"][1]] = np.asarray([1], dtype=np.int64)
    with pytest.raises(TransformerClientError, match="last-valid"):
        runtime._local(last, values, {}, {})
