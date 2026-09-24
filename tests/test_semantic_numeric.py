from __future__ import annotations

import json
import math
from pathlib import Path

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


def test_semantic_permute_and_slice_follow_declared_axes(operations: dict[str, dict]) -> None:
    values = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    np.testing.assert_array_equal(
        _local(operations["layer.0.q_permute"], values), values.transpose(0, 2, 1, 3)
    )
    np.testing.assert_array_equal(_local(operations["layer.0.ple_slice"], values), values[:, :, 0, :])


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
        with pytest.raises(TransformerClientError, match="unsupported"):
            _local(operation, source)
    with pytest.raises(SemanticNumericError, match="finite"):
        round_bfloat16(np.asarray([float("nan")], dtype=np.float32))
    with pytest.raises(SemanticNumericError, match="BF16 domain"):
        round_bfloat16(np.asarray([3.4e38], dtype=np.float32))
