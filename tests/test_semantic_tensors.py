from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from pllm.modeling import lower_model
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.semantic_numeric import SemanticNumericError, bfloat16_scale
from pllm.runtime.semantic_hybrid import (
    HybridStateError, bounded_gated_delta_decay, execute_declared_causal_convolution,
    execute_declared_gated_delta,
)
from pllm.runtime.semantic_tensors import (
    SemanticTensorError, lookup_contract, projection_weight_shape, required_client_tensors,
)
from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError


def _plan():
    source = Path(__file__).resolve().parents[1] / "crates/pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"
    return lower_model(
        json.loads(source.read_text(encoding="utf-8")),
        batch=1,
        max_input_tokens=2,
        max_new_tokens=2,
    )


def _contract_representation(value, frozen):
    if not frozen:
        return value
    if type(value) is dict:
        return MappingProxyType({key: _contract_representation(item, True) for key, item in value.items()})
    if type(value) is list:
        return tuple(_contract_representation(item, True) for item in value)
    return value


class _HookDict(dict):
    def __iter__(self):
        raise AssertionError("custom mapping traversed")

    def get(self, *args):
        raise AssertionError("custom mapping accessed")


class _HookList(list):
    def __len__(self):
        raise AssertionError("custom shape measured")

    def __iter__(self):
        raise AssertionError("custom shape traversed")


class _HookTuple(tuple):
    def __len__(self):
        raise AssertionError("custom tuple measured")

    def __iter__(self):
        raise AssertionError("custom tuple traversed")


class _HookMapping(Mapping):
    def __len__(self):
        raise AssertionError("custom mapping measured")

    def __iter__(self):
        raise AssertionError("custom mapping traversed")

    def __getitem__(self, key):
        raise AssertionError("custom mapping accessed")


class _HookSequence(Sequence):
    def __len__(self):
        raise AssertionError("custom sequence measured")

    def __getitem__(self, key):
        raise AssertionError("custom sequence accessed")


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("field,bad", [
    ("inputs", None), ("inputs", []), ("inputs", [True]),
    ("inputs", _HookList(["input.tokens"])), ("inputs", _HookTuple(("input.tokens",))),
    ("output_shape", None), ("output_shape", [1, True, 4]),
    ("output_shape", _HookList([1, 2, 4])),
    ("attributes", None), ("attributes", _HookDict()), ("attributes", _HookMapping()),
    ("weight_shape", None), ("weight_shape", [7, True]),
    ("weight_shape", [True, 4]), ("weight_shape", [0, 4]),
    ("weight_shape", [7, 5]), ("weight_shape", _HookTuple((7, 4))),
    ("weight_shape", _HookSequence()), ("inputs", _HookSequence()),
    ("index_domain", None), ("index_domain", False), ("source", None),
    ("weight_layout", None),
])
def test_lookup_contract_rejects_malformed_json_and_frozen_declarations(frozen, field, bad):
    op = {"inputs": ["input.tokens"], "output_shape": [1, 2, 4],
          "attributes": {"index_domain": "tokens", "weight": "table", "weight_shape": [7, 4]}}
    if field in {"inputs", "output_shape", "attributes"}:
        op[field] = bad
    else:
        op["attributes"][field] = bad
    with pytest.raises(SemanticTensorError):
        lookup_contract(_contract_representation(op, frozen), legacy_vocabulary=11)


@pytest.mark.parametrize("frozen", [False, True])
def test_lookup_frozen_legacy_fallback_is_only_for_absent_domain_and_shape(frozen):
    op = {"inputs": ["input.tokens"], "output_shape": [1, 2, 4], "attributes": {"weight": "table"}}
    assert lookup_contract(_contract_representation(op, frozen), legacy_vocabulary=7) == (
        "tokens", "input.tokens", (7, 4))
    for key, value in (("index_domain", None), ("index_domain", "tokens"), ("weight_shape", None)):
        declared = {**op, "attributes": {**op["attributes"], key: value}}
        with pytest.raises(SemanticTensorError):
            lookup_contract(_contract_representation(declared, frozen), legacy_vocabulary=7)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("layout,shape", [("input_output", [3, 5]), ("output_input", [5, 3])])
def test_projection_geometry_accepts_only_validated_json_or_frozen_dimensions(frozen, layout, shape):
    op = {"output_shape": [1, 2, 5], "attributes": {
        "weight_layout": layout, "weight_shape": shape, "bias_shape": [5]}}
    assert projection_weight_shape(_contract_representation(op, frozen), 3) == tuple(shape)
    for key, bad in (("weight_shape", None), ("weight_shape", [True, 5]),
                     ("weight_shape", [3, 0]), ("weight_shape", [3, 6]),
                     ("weight_shape", _HookList(shape)), ("weight_shape", _HookTuple(shape)),
                     ("bias_shape", None), ("bias_shape", [False]), ("bias_shape", [6]),
                     ("bias_shape", _HookList([5])), ("weight_layout", None)):
        altered = {**op, "attributes": {**op["attributes"], key: bad}}
        with pytest.raises(SemanticTensorError):
            projection_weight_shape(_contract_representation(altered, frozen), 3)


@pytest.mark.parametrize("bad", [_HookDict(), _HookMapping(), _HookSequence(), _HookList(), _HookTuple(), None])
def test_tensor_contract_rejects_unknown_root_containers_without_hooks(bad):
    with pytest.raises(SemanticTensorError):
        lookup_contract(bad)
    with pytest.raises(SemanticTensorError):
        projection_weight_shape(bad, 3)


def test_native_frozen_lookup_and_projection_contracts_match_to_dict():
    from absolute_position_decoder_oracle import tiny_checkpoint
    config, _ = tiny_checkpoint()
    plan = lower_model(config, batch=1, max_input_tokens=2, max_new_tokens=2)
    for phase in ("prefill", "decode"):
        frozen = {op["id"]: op for op in getattr(plan, phase)["operations"]}
        plain = {op["id"]: op for op in plan.to_dict()[phase]["operations"]}
        for name, operation in frozen.items():
            if operation["operator"] == "token_lookup":
                assert lookup_contract(operation) == lookup_contract(plain[name])
            elif operation["operator"] in ("linear", "output_head"):
                width = frozen[operation["inputs"][0]]["output_shape"][-1]
                assert projection_weight_shape(operation, width) == projection_weight_shape(plain[name], width)


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
    # Numerical kernel only; the resized source is not a bound decoder workload.
    scalar = np.asarray([1.25], dtype=np.float32)
    source = np.asarray([[1.0, -0.75, 0.375, 2.0]], dtype=np.float32)
    actual = bfloat16_scale(source, operation["attributes"]["factor"], scalar)
    expected = (
        torch.from_numpy(source).to(torch.bfloat16) * torch.tensor([1.25], dtype=torch.bfloat16)
    ).float().numpy()
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(SemanticNumericError, match="bound BF16 value"):
        bfloat16_scale(source, operation["attributes"]["factor"])


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
    # The native declaration is checked above; resized evaluation below is a
    # numerical kernel test, not compiled/imported runtime admission.
    assert initializer["attributes"] == {"initial_value": 0, "dtype": "float32", "state_kind": "convolution"}
    initialized = np.zeros(initializer["output_shape"], dtype=np.float32)
    assert initialized.shape == (1, 8192, 4)
    assert initialized.dtype == np.float32
    assert not np.any(initialized)
    values = {
        convolution["inputs"][0]: np.asarray([[[0.5, -0.5], [1.0, 0.25]]], dtype=np.float32),
        convolution["inputs"][1]: np.zeros((1, 2, 4), dtype=np.float32),
    }
    for operator in (update, convolution):
        operator["attributes"]["groups"] = 2
        operator["output_shape"] = [1, 2, 4] if operator is update else [1, 2, 2]
    state_after = execute_declared_causal_convolution(update, values, weights)
    output = execute_declared_causal_convolution(convolution, values, weights)
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
    with pytest.raises(HybridStateError, match="shapes"):
        execute_declared_causal_convolution(bad, values, weights)
    with pytest.raises(HybridStateError, match="shapes"):
        execute_declared_causal_convolution(convolution, values, weights[:1])


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
    gated_norm = next(row for row in ops if row["operator"] == "rms_norm_gated"
                      and row["layer"] == decay["layer"])
    assert required[decay["attributes"]["a_log"]] == (32,)
    assert required[decay["attributes"]["dt_bias"]] == (32,)
    assert required[gated_norm["attributes"]["weight"]] == (128,)

    # Pure resized numerical kernels; no fabricated checkpoint owner/binding.
    a_log = np.array([0.0, 0.5], dtype=np.float32)
    dt_bias = np.array([0.1, -0.2], dtype=np.float32)
    assert initializer["attributes"] == {"initial_value": 0, "dtype": "float32", "state_kind": "recurrent"}
    initializer["output_shape"] = [1, 2, 4, 3]
    retained = np.zeros(initializer["output_shape"], dtype=np.float32)
    assert retained.shape == (1, 2, 4, 3) and not retained.any()

    projected = np.array([[0.5, -0.5], [0.25, 0.1]], dtype=np.float32)
    decay["output_shape"] = [1, 2, 2]
    beta["output_shape"] = [1, 2, 2]
    gates = bounded_gated_delta_decay(projected[None, :, :], a_log, dt_bias)
    assert beta["attributes"] == {}
    confidence = 1.0 / (1.0 + np.exp(-projected[None, :, :]))
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
    next_state = execute_declared_gated_delta(update, samples)
    output = execute_declared_gated_delta(rule, samples)
    assert next_state.shape == (1, 2, 4, 3)
    assert output.shape == (1, 2, 2, 3)
    assert np.all(np.isfinite(output)) and np.any(next_state)
    assert not retained.any()
    forged = dict(rule)
    forged["attributes"] = {**rule["attributes"], "query_scale": {"numerator": 1, "sqrt_denominator": 5}}
    with pytest.raises(HybridStateError, match="shapes"):
        execute_declared_gated_delta(forged, samples)
    with pytest.raises(HybridStateError, match="shape"):
        bounded_gated_delta_decay(projected[None, :, :], a_log[:1], dt_bias)
