from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from pllm.modeling import lower_model
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.semantic_tensors import SemanticTensorError, required_client_tensors
from pllm.runtime.transformer_client import TransformerClientError
from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError


def _plan():
    source = Path(__file__).resolve().parents[1] / "crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"
    return lower_model(
        json.loads(source.read_text(encoding="utf-8")),
        batch=1,
        max_input_tokens=2,
        max_new_tokens=2,
    )


def test_semantic_tensor_requirements_bind_checkpoint_scalars_across_phases() -> None:
    plan = _plan()
    required = required_client_tensors(plan)
    operations = {op["id"]: op for op in plan.to_dict()["prefill"]["operations"]}
    scalars = [
        operation["attributes"]["factor"]["weight"]
        for operation in operations.values()
        if operation["operator"] == "scale"
        and operation["attributes"]["factor"]["kind"] == "checkpoint_scalar"
    ]
    assert scalars
    assert all(required[weight] == (1,) for weight in scalars)
    assert operations["layer.0.v_norm"]["attributes"]["weight"] is None
    assert None not in required

    operation = operations["layer.0.layer_scalar"]
    runtime = object.__new__(SemanticDecoderRuntime)
    runtime._tensors = {operation["attributes"]["factor"]["weight"]: "scalar"}
    runtime.bundle = SimpleNamespace(arrays={"scalar": np.asarray([1.25], dtype=np.float32)})
    source = np.asarray([[1.0, -0.75, 0.375, 2.0]], dtype=np.float32)
    actual = runtime._local(operation, {operation["inputs"][0]: source}, {}, {})
    expected = (
        torch.from_numpy(source).to(torch.bfloat16) * torch.tensor([1.25], dtype=torch.bfloat16)
    ).float().numpy()
    np.testing.assert_array_equal(actual, expected)
    runtime._tensors = {}
    with pytest.raises(TransformerClientError, match="bound local tensor"):
        runtime._local(operation, {operation["inputs"][0]: source}, {}, {})


def test_checkpoint_scalar_import_requires_exact_shape_and_finite_value(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    weight = "model.language_model.layers.0.layer_scalar"
    for vector, valid in [([1.25], True), ([1.25, 1.0], False), ([float("nan")], False)]:
        save_file({weight: np.asarray(vector, dtype=np.float32)}, path)
        store = SafeTensorStore(tmp_path)
        if valid:
            loaded = MaskedTransformerEngine._load_local_tensors(store, required={weight: (1,)})
            np.testing.assert_array_equal(loaded[weight], np.asarray(vector, dtype=np.float32))
        else:
            with pytest.raises(TransformerEngineError, match="invalid shape or values"):
                MaskedTransformerEngine._load_local_tensors(store, required={weight: (1,)})
    with pytest.raises(TransformerEngineError, match="missing"):
        MaskedTransformerEngine._load_local_tensors(store, required={"absent.scalar": (1,)})


def test_malformed_local_tensor_shape_is_not_inferred_from_name() -> None:
    plan = _plan()
    document = plan.to_dict()
    operation = next(
        row for row in document["prefill"]["operations"] if row["id"] == "layer.0.layer_scalar"
    )
    operation["attributes"]["factor"]["weight_shape"] = [2]
    # Even before a native plan could be built from a forged document, the
    # shared import contract refuses to infer the shape from the weight path.
    with pytest.raises(SemanticTensorError, match="unit shape"):
        required_client_tensors(SimpleNamespace(to_dict=lambda: document))


def test_declared_hybrid_convolution_binds_exact_weight_and_runs_local_prefill() -> None:
    source = Path(__file__).resolve().parents[1] / "crates/pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json"
    plan = lower_model(
        json.loads(source.read_text(encoding="utf-8")),
        batch=1,
        max_input_tokens=2,
        max_new_tokens=1,
    )
    required = required_client_tensors(plan)
    operations = plan.to_dict()["prefill"]["operations"]
    convolution = next(row for row in operations if row["operator"] == "causal_convolution")
    update = next(
        row for row in operations if row["operator"] == "convolution_state_update"
    )
    initializer = next(
        row for row in operations if row["operator"] == "state_initialize"
    )
    weight_id = convolution["attributes"]["weight"]
    assert required[weight_id] == (8192, 1, 4)
    assert update["attributes"] == convolution["attributes"]

    weights = np.array([[[0.25, 0.5, -0.125, 0.75]], [[0.5, 0.125, 0.25, -0.5]]], dtype=np.float32)
    runtime = object.__new__(SemanticDecoderRuntime)
    runtime._tensors = {weight_id: "conv"}
    runtime.bundle = SimpleNamespace(arrays={"conv": weights})
    initialized = runtime._local(initializer, {}, {}, {})
    assert initialized.shape == (1, 8192, 4)
    assert initialized.dtype == np.float32
    assert not np.any(initialized)
    forged_initial = dict(initializer)
    forged_initial["attributes"] = {**initializer["attributes"], "initial_value": 1}
    with pytest.raises(TransformerClientError, match="zero state"):
        runtime._local(forged_initial, {}, {}, {})
    values = {
        convolution["inputs"][0]: np.asarray([[[0.5, -0.5], [1.0, 0.25]]], dtype=np.float32),
        convolution["inputs"][1]: np.zeros((1, 2, 4), dtype=np.float32),
    }
    for operator in (update, convolution):
        operator["attributes"]["groups"] = 2
        operator["output_shape"] = [1, 2, 4] if operator is update else [1, 2, 2]
    state_after = runtime._local(update, values, {}, {})
    output = runtime._local(convolution, values, {}, {})
    expected = torch.nn.functional.silu(
        torch.nn.functional.conv1d(
            torch.from_numpy(values[convolution["inputs"][0]].transpose(0, 2, 1).copy()),
            torch.from_numpy(weights),
            padding=3,
            groups=2,
        )[..., :2]
    ).transpose(1, 2).numpy()
    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(state_after[:, :, -2:], values[convolution["inputs"][0]].transpose(0, 2, 1))
    bad = dict(convolution)
    bad["attributes"] = {**convolution["attributes"], "kernel_size": 5}
    with pytest.raises(TransformerClientError, match="shapes"):
        runtime._local(bad, values, {}, {})
    runtime._tensors = {}
    with pytest.raises(TransformerClientError, match="bound local tensor"):
        runtime._local(convolution, values, {}, {})


def test_declared_hybrid_decay_and_retained_recurrence_bind_local_coefficients() -> None:
    source = Path(__file__).resolve().parents[1] / "crates/pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json"
    plan = lower_model(json.loads(source.read_text(encoding="utf-8")), batch=1, max_input_tokens=3, max_new_tokens=1)
    required = required_client_tensors(plan)
    ops = plan.to_dict()["prefill"]["operations"]
    decay = next(row for row in ops if row["operator"] == "gated_delta_decay")
    beta = next(row for row in ops if row["operator"] == "sigmoid" and row["layer"] == decay["layer"])
    rule = next(row for row in ops if row["operator"] == "gated_delta_rule" and row["layer"] == decay["layer"])
    update = next(row for row in ops if row["operator"] == "gated_delta_state_update" and row["layer"] == decay["layer"])
    initializer = next(row for row in ops if row["operator"] == "state_initialize"
                       and row["attributes"]["state_kind"] == "recurrent")
    assert required[decay["attributes"]["a_log"]] == (32,)
    assert required[decay["attributes"]["dt_bias"]] == (32,)
    assert required[f"model.layers.{decay['layer']}.linear_attn.norm.weight"] == (128,)

    runtime = object.__new__(SemanticDecoderRuntime)
    runtime._tensors = {decay["attributes"]["a_log"]: "a", decay["attributes"]["dt_bias"]: "dt"}
    runtime.bundle = SimpleNamespace(arrays={
        "a": np.array([0.0, 0.5], dtype=np.float32),
        "dt": np.array([0.1, -0.2], dtype=np.float32),
    })
    initializer["output_shape"] = [1, 2, 4, 3]
    retained = runtime._local(initializer, {}, {}, {})
    assert retained.shape == (1, 2, 4, 3) and not retained.any()

    projected = np.array([[0.5, -0.5], [0.25, 0.1]], dtype=np.float32)
    decay["output_shape"] = [1, 2, 2]
    beta["output_shape"] = [1, 2, 2]
    gates = runtime._local(decay, {decay["inputs"][0]: projected}, {}, {})
    confidence = runtime._local(beta, {beta["inputs"][0]: projected}, {}, {})
    assert gates.shape == confidence.shape == (1, 2, 2)
    assert np.all(gates < 0) and np.all((confidence > 0) & (confidence < 1))

    for operator in (rule, update):
        operator["attributes"]["key_head_repeats"] = 2
        operator["attributes"]["query_scale"]["sqrt_denominator"] = 4
        operator["output_shape"] = [1, 2, 2, 3] if operator is rule else [1, 2, 4, 3]
    samples = {
        rule["inputs"][0]: np.ones((1, 2, 4), dtype=np.float32),
        rule["inputs"][1]: np.ones((1, 2, 4), dtype=np.float32),
        rule["inputs"][2]: np.full((1, 2, 6), 0.25, dtype=np.float32),
        rule["inputs"][3]: gates,
        rule["inputs"][4]: confidence,
        rule["inputs"][5]: retained,
    }
    next_state = runtime._local(update, samples, {}, {})
    output = runtime._local(rule, samples, {}, {})
    assert next_state.shape == (1, 2, 4, 3)
    assert output.shape == (1, 2, 2, 3)
    assert np.all(np.isfinite(output)) and np.any(next_state)
    assert not retained.any()
    forged = dict(rule)
    forged["attributes"] = {**rule["attributes"], "query_scale": {"numerator": 1, "sqrt_denominator": 5}}
    with pytest.raises(TransformerClientError, match="shapes"):
        runtime._local(forged, samples, {}, {})
    runtime._tensors = {}
    with pytest.raises(TransformerClientError, match="bound local tensor"):
        runtime._local(decay, {decay["inputs"][0]: projected}, {}, {})
