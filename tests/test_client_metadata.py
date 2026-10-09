"""Allocation attribution executes source validation and retires instrumentation."""
import tracemalloc

import pytest

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.runtime import model_binding
from pllm.runtime.benchmark_client_metadata import client_metadata_attribution
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def test_attribution_target_is_explicit_and_separate(tmp_path):
    from pllm import Experiment, Deployment, ExecutionBudget
    from pllm.metrics import BenchmarkTarget
    experiment = Experiment("metadata", MaskedLinearCpu(Model.tiny()),
        Deployment.local(root=str(tmp_path)), ExecutionBudget(1, 16, 2))
    ordinary = BenchmarkTarget.response(experiment)
    assert "client_metadata" not in ordinary.to_spec()["parameters"]
    traced = BenchmarkTarget.response(experiment, client_metadata=True)
    assert traced.to_spec()["parameters"]["client_metadata"] is True
    assert BenchmarkTarget.from_spec(traced.to_spec()).digest == traced.digest
    with pytest.raises(ValueError):
        BenchmarkTarget.response(experiment, client_metadata=1)
    with pytest.raises(ValueError):
        BenchmarkTarget.response(experiment, client_metadata=True, client_mlp={"tile_rows": 0})


def test_binding_attribution_preserves_validation_and_restores_on_failure(tmp_path):
    import asyncio
    import json
    from pllm.runtime.loaders import load_hf_directory

    path = create_tiny_llama_checkpoint(tmp_path / "model")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(path, model_id="metadata")))
    bundle = ClientBundle.unpack(engine.client_bundle("metadata"))
    plan = lower_model(json.loads((path / "config.json").read_text()), batch=1,
                       max_input_tokens=16, max_new_tokens=2)
    composition = MaskedLinearCpu(Model("metadata"))
    expected = model_binding.compile_runtime_model(plan, bundle, composition=composition)
    original = model_binding._compile_runtime_model
    runtime = model_binding.CompiledRuntimeModel.runtime
    demand = model_binding.CompiledRuntimeModel.prepared_stage_rows
    with pytest.raises(model_binding.RuntimeBindingError):
        with client_metadata_attribution() as records:
            actual = model_binding.compile_runtime_model(plan, bundle, composition=composition)
            assert actual.digest == expected.digest
            actual.prepared_stage_rows(8, 1)
            actual.runtime(lambda *_: None)
            bundle.arrays["model.layers.0.input_layernorm.weight"][0] += 1.0
            actual.validate()
    assert {row["phase"] for row in records} == {"compile_binding", "runtime_binding", "preparation_demand"}
    assert all(row["new_traced_peak_bytes"] >= row["new_traced_retained_bytes"] > 0 for row in records)
    assert all(row["public_metadata_python_bytes"] > 0 for row in records)
    assert not tracemalloc.is_tracing()
    assert model_binding._compile_runtime_model is original
    assert model_binding.CompiledRuntimeModel.runtime is runtime
    assert model_binding.CompiledRuntimeModel.prepared_stage_rows is demand


def test_attribution_preserves_existing_tracer():
    from pllm._benchmark_contract import TargetUnavailable
    tracemalloc.start()
    try:
        with pytest.raises(TargetUnavailable):
            with client_metadata_attribution():
                pass
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()
