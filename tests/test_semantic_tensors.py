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
