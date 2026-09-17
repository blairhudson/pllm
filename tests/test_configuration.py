from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from pllm import Experiment, Model, Pipeline
from pllm.components import ComponentDescriptor, ComponentRef, KvCacheEviction
from pllm.config import (
    ConfigurationError,
    ExecutionBudget,
    canonical_bytes,
    configuration_digest,
    load_configuration,
    loads_configuration,
)
from pllm.deployment import Deployment
from pllm.kernels import Cpu
from pllm.preparation import ModelAwareCorrections
from pllm.protocols.masked_linear import MaskedLinear

ROOT = Path(__file__).resolve().parents[1]
YAML_EXAMPLE = ROOT / "examples/pllm.yaml"
PYTHON_EXAMPLE = ROOT / "examples/composition.py"


def example() -> Experiment:
    return Experiment(
        name="qwen-local",
        pipeline=Pipeline.from_profile(
            "baseline.masked_linear_cpu",
            model=Model("Qwen/Qwen2.5-0.5B-Instruct"),
            components={
                "linear": MaskedLinear(),
                "preparation": ModelAwareCorrections(),
                "inference": ComponentRef("pllm/inference"),
                "kernels": Cpu(threads=4),
            },
        ),
        deployment=Deployment.local(root=".pllm/qwen-local"),
        budget=ExecutionBudget(requests=1, max_input_tokens=128, max_new_tokens=32),
    )


def test_python_yaml_and_importable_example_have_schema_parity():
    expected = example()
    loaded = load_configuration(YAML_EXAMPLE)
    module_spec = importlib.util.spec_from_file_location("pllm_example_composition", PYTHON_EXAMPLE)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    assert loaded == expected == module.experiment
    assert loaded.to_spec() == expected.to_spec()


def test_canonical_bytes_and_digest_golden():
    expected = (
        b'{"budget":{"max_input_tokens":128,"max_new_tokens":32,"requests":1},'
        b'"deployment":{"kind":"local","root":".pllm/qwen-local"},"name":"qwen-local",'
        b'"pipeline":{"components":{"inference":{"component":"pllm/inference","params":{}},'
        b'"kernels":{"component":"pllm/cpu","params":{"threads":4}},'
        b'"linear":{"component":"pllm/masked-linear","params":{}},"preparation":'
        b'{"component":"pllm/model-aware-corrections","params":{}}},"model":'
        b'{"source":"Qwen/Qwen2.5-0.5B-Instruct"},"profile":"baseline.masked_linear_cpu"},'
        b'"schema":"pllm.experiment.v1"}'
    )
    assert canonical_bytes(example()) == expected
    assert example().canonical_bytes() == expected
    assert configuration_digest(example()) == (
        "863af238d286ed9970ee710a9c4694a14fb43fea2ffde883b9e43ca59f90197e"
    )
    assert (
        configuration_digest(example())
        == hashlib.sha256(b"pllm.configuration.v1\0" + expected).hexdigest()
    )
    assert example().configuration_digest() == configuration_digest(example())


def test_ordering_does_not_change_identity():
    spec = example().to_spec()
    reordered = json.loads(json.dumps(spec), object_pairs_hook=lambda pairs: dict(reversed(pairs)))
    assert canonical_bytes(spec) == canonical_bytes(reordered)
    assert configuration_digest(spec) == configuration_digest(reordered)


def test_inputs_and_nested_values_are_deeply_immutable():
    params = {"options": {"levels": [1, 2]}}
    components = {"custom": ComponentRef("example/custom", params)}
    pipeline = Pipeline.from_profile(
        "example.profile", model=Model("example/model"), components=components
    )
    params["options"]["levels"].append(3)
    components.clear()

    assert pipeline.components["custom"].to_spec()["params"] == {"options": {"levels": [1, 2]}}
    with pytest.raises(TypeError):
        pipeline.components["new"] = ComponentRef("example/new")
    with pytest.raises(TypeError):
        pipeline.components["custom"].params["new"] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        pipeline.profile = "changed"
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        pipeline.components["custom"].new_attribute = "changed"
    assert hash(pipeline) == hash(
        Pipeline.from_profile(
            "example.profile",
            model=Model("example/model"),
            components={"custom": ComponentRef("example/custom", {"options": {"levels": [1, 2]}})},
        )
    )


def test_get_params_and_with_params_are_nested_and_immutable():
    original = example()
    params = original.get_params(deep=True)
    assert params["pipeline__components__kernels__threads"] == 4
    assert params["budget__requests"] == 1

    changed = original.with_params(
        pipeline__components__kernels__threads=8,
        budget__requests=6,
    )
    assert original.pipeline.components["kernels"].params["threads"] == 4
    assert changed.pipeline.components["kernels"].params["threads"] == 8
    assert changed.budget.requests == 6
    with pytest.raises(ConfigurationError, match="unknown parameter path"):
        original.with_params(budget__unknown=2)
    with pytest.raises(ConfigurationError, match="overlapping"):
        original.with_params(budget=original.budget, budget__requests=2)


@pytest.mark.parametrize(
    "text, message",
    [
        ("schema: pllm.experiment.v1\nschema: pllm.experiment.v1\n", "duplicate"),
        ('{"schema":"pllm.experiment.v1","schema":"pllm.experiment.v1"}', "duplicate"),
        ("schema: other.v1\n", "unknown fields|unsupported schema|missing fields"),
        (YAML_EXAMPLE.read_text() + "unknown: value\n", "unknown fields"),
        (YAML_EXAMPLE.read_text().replace("requests: 1", "requests: true"), "integer"),
        (YAML_EXAMPLE.read_text().replace("threads: 4", "threads: .nan"), "finite|integer"),
        (YAML_EXAMPLE.read_text().replace("threads: 4", "threads: !!binary YQ=="), "integer"),
    ],
)
def test_strict_loader_rejects_invalid_documents(text: str, message: str):
    with pytest.raises(ConfigurationError, match=message):
        loads_configuration(text)


def test_loader_rejects_oversized_document_before_parsing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("parser called")

    monkeypatch.setattr(json, "loads", forbidden)
    with pytest.raises(ConfigurationError, match="byte limit"):
        loads_configuration("{" + " " * 1_048_576, format="json")


@pytest.mark.parametrize(
    "text, format",
    [
        ("? [invalid, key]\n: value\n", "yaml"),
        ("[" * 2_000 + "]" * 2_000, "json"),
        (
            YAML_EXAMPLE.read_text()
            .replace("component: pllm/masked-linear", "component: example/custom")
            .replace("params: {}", "params: &params\n        cycle: *params", 1),
            "yaml",
        ),
        ("\ud800", "yaml"),
    ],
)
def test_loader_normalizes_malformed_document_structures(text: str, format: str):
    with pytest.raises(ConfigurationError):
        loads_configuration(text, format=format)


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_integer_fields_reject_non_integer_values(value):
    with pytest.raises(ConfigurationError, match="integer"):
        ExecutionBudget(requests=value, max_input_tokens=1, max_new_tokens=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), b"bad", object()])
def test_component_parameters_reject_unsafe_values(value):
    with pytest.raises(ConfigurationError):
        ComponentRef("example/custom", {"value": value})


def test_constructors_and_imports_have_no_runtime_side_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("runtime side effect")

    monkeypatch.setattr("socket.socket", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    assert example().pipeline.model.source == "Qwen/Qwen2.5-0.5B-Instruct"


def test_root_exports_are_lazy_and_preparation_export_is_typed_surface():
    import pllm

    assert pllm.Model is Model
    assert pllm.create_preparation_app.__name__ == "create_preparation_app"


def test_config_facade_exposes_declarations_and_loaders() -> None:
    import pllm.config as config

    assert config.Model is Model
    assert config.ComponentRef is ComponentRef
    assert config.Pipeline is Pipeline
    assert config.Deployment is Deployment
    assert config.ExecutionBudget is ExecutionBudget
    assert config.Experiment is Experiment
    assert config.loads_configuration is loads_configuration


def test_kv_cache_eviction_clones_and_loads_as_builtin() -> None:
    component = KvCacheEviction(alpha=(2, 3), cluster_sizes=(64, 16, 8))
    cloned = component.with_params(alpha__numerator=1)

    assert isinstance(cloned, KvCacheEviction)
    assert cloned.to_spec()["params"]["alpha"] == {"numerator": 1, "denominator": 3}
    assert component.to_spec()["params"]["alpha"] == {"numerator": 2, "denominator": 3}

    spec = example().to_spec()
    spec["pipeline"]["components"]["eviction"] = component.to_spec()
    loaded = Experiment.from_spec(spec)
    assert isinstance(loaded.pipeline.components["eviction"], KvCacheEviction)
    assert loaded.pipeline.components["eviction"].to_spec() == component.to_spec()


def test_builtin_component_descriptors_are_static_and_immutable() -> None:
    from pllm.components import get_component, list_components
    from pllm.configuration import ComponentDescriptor as ConfigurationComponentDescriptor

    descriptor = KvCacheEviction.describe()

    assert ComponentDescriptor is ConfigurationComponentDescriptor
    assert isinstance(descriptor, ComponentDescriptor)
    assert KvCacheEviction().describe() is descriptor
    assert descriptor.component == "pllm/kv-cache-eviction"
    assert descriptor.lifecycle_phase == "model-lowering"
    assert descriptor.to_dict()["parameter_schema"]["type"] == "object"
    with pytest.raises(TypeError):
        descriptor.parameter_schema["changed"] = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.version = "2"
    descriptors = list_components()
    assert descriptors == tuple(sorted(descriptors, key=lambda item: item.component))
    assert get_component("pllm/kv-cache-eviction") is descriptor
    with pytest.raises(KeyError, match="built-in component not found"):
        get_component("missing")
