from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import runpy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from pllm import BundleModel, Experiment, MaskedLinearCpu, Model, Pipeline, TinyModel
from pllm.components import ComponentDescriptor, ComponentRef
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
from pllm.nonlinear import BinaryTableGatedMultiplyQ7, R03CrtGatedMultiplyQ7
from pllm.passes import KvCacheEviction
from pllm.schedulers import (
    ChunkedIndependentLanesProtectedTensorSchedule,
    IndependentLanesProtectedTensorSchedule,
    ScalarProtectedTensorSchedule,
)

ROOT = Path(__file__).resolve().parents[1]
YAML_EXAMPLE = ROOT / "examples/pllm.yaml"
PYTHON_EXAMPLE = ROOT / "examples/composition.py"
FIRST_REQUEST_EXAMPLE = ROOT / "examples/first_request.py"


def example() -> Experiment:
    return Experiment(
        name="qwen-local",
        pipeline=MaskedLinearCpu(
            Model("Qwen/Qwen2.5-0.5B-Instruct"),
            kernels=Cpu(threads=4),
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


def test_first_request_example_executes_and_resolves() -> None:
    namespace = runpy.run_path(str(FIRST_REQUEST_EXAMPLE))
    assert namespace["experiment"] == example()
    assert namespace["resolved"].configuration_digest == example().configuration_digest()


def test_model_is_one_typed_source_spec_with_legacy_canonical_default() -> None:
    default = Model("Qwen/Qwen2.5-0.5B-Instruct")
    assert default == Model.hf("Qwen/Qwen2.5-0.5B-Instruct")
    assert default.to_spec() == {"source": "Qwen/Qwen2.5-0.5B-Instruct"}
    assert default.to_runtime_spec() == {
        "kind": "huggingface",
        "source": "Qwen/Qwen2.5-0.5B-Instruct",
    }

    pinned = Model.hf(
        "Qwen/Qwen2.5-0.5B-Instruct",
        model_id="qwen-local",
        revision="7ae5576",
        local_files_only=True,
    )
    assert Model.from_spec(pinned.to_spec()) == pinned
    assert pinned.to_spec() == {
        "source": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_id": "qwen-local",
        "revision": "7ae5576",
        "local_files_only": True,
    }
    bundle = Model.path("/models/qwen", format="safetensors")
    tiny = Model.tiny()
    assert isinstance(bundle, BundleModel)
    assert bundle.kind == "safetensors"
    assert Model.path("/models/qwen.gguf", format="gguf").kind == "gguf"
    assert isinstance(tiny, TinyModel)
    assert tiny.to_spec() == {"source": "qwen2", "kind": "tiny"}
    ollama = Model.ollama("qwen2:latest")
    assert ollama.to_spec() == {
        "source": "qwen2:latest",
        "kind": "ollama",
        "endpoint": "http://127.0.0.1:11434",
    }
    model_schema = json.loads((ROOT / "schemas/model.schema.json").read_text())
    for model in (default, pinned, Model.path("/models/qwen.gguf", format="gguf"), ollama):
        Draft202012Validator(model_schema).validate(model.to_spec())

    for invalid in (
        {"source": "x", "kind": "unknown"},
        {"source": "x", "kind": "gguf", "revision": "main"},
        {"source": "x", "kind": "tiny", "local_files_only": True},
        {"source": "x", "endpoint": "https://example.invalid"},
        {"source": "x", "kind": "ollama", "endpoint": "https://user:secret@example.invalid"},
        {"source": "x", "kind": "ollama", "endpoint": "https://example.invalid/api"},
        {"kind": "huggingface"},
        {"source": "x", "extra": True},
    ):
        with pytest.raises(ConfigurationError):
            Model.from_spec(invalid)


def test_explicit_model_spec_round_trips_through_experiment_schema() -> None:
    pinned = example().with_params(
        pipeline__model=Model.hf(
            "Qwen/Qwen2.5-0.5B-Instruct",
            model_id="qwen-pinned",
            revision="7ae5576",
            local_files_only=True,
        )
    )
    restored = loads_configuration(json.dumps(pinned.to_spec()))
    assert restored == pinned
    schema = json.loads((ROOT / "schemas/experiment.schema.json").read_text())
    Draft202012Validator(schema).validate(pinned.to_spec())


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
    assert params["pipeline__kernels__threads"] == 4
    assert params["budget__requests"] == 1

    changed = original.with_params(
        pipeline__kernels__threads=8,
        budget__requests=6,
    )
    assert original.pipeline.components["kernels"].params["threads"] == 4
    assert changed.pipeline.components["kernels"].params["threads"] == 8
    assert changed.budget.requests == 6
    with pytest.raises(ConfigurationError, match="unknown parameter path"):
        original.with_params(budget__unknown=2)
    with pytest.raises(ConfigurationError, match="unknown parameter path"):
        original.with_params(pipeline__components__kernels__threads=8)
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
    from pllm.components import get, get_component, list_component_classes, list_components
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
    classes = list_component_classes()
    assert descriptors == tuple(sorted(descriptors, key=lambda item: item.component))
    assert tuple(item.describe() for item in classes) == descriptors
    assert get("pllm/kv-cache-eviction") is KvCacheEviction
    assert get_component("pllm/kv-cache-eviction") is descriptor
    with pytest.raises(KeyError, match="built-in component not found"):
        get_component("missing")


@pytest.mark.parametrize(
    "component, identity, category, params",
    [
        (
            BinaryTableGatedMultiplyQ7(),
            "pllm/binary-table/v1",
            "pllm/nonlinear-protocol",
            {},
        ),
        (
            R03CrtGatedMultiplyQ7(),
            "pllm/r03-crt/v1",
            "pllm/nonlinear-protocol",
            {},
        ),
        (
            ScalarProtectedTensorSchedule(),
            "pllm/scalar/v1",
            "pllm/protected-scheduler",
            {},
        ),
        (
            IndependentLanesProtectedTensorSchedule(),
            "pllm/independent-lanes/v1",
            "pllm/protected-scheduler",
            {"max_elements": 4},
        ),
        (
            ChunkedIndependentLanesProtectedTensorSchedule(max_elements=4096),
            "pllm/chunked-independent-lanes/v1",
            "pllm/protected-scheduler",
            {"max_elements": 4096},
        ),
    ],
)
def test_concrete_components_have_distinct_descriptors_and_roundtrip(
    component: ComponentRef,
    identity: str,
    category: str,
    params: dict[str, object],
) -> None:
    descriptor = component.describe()
    spec = example().to_spec()
    spec["pipeline"]["components"]["candidate"] = component.to_spec()

    loaded = Experiment.from_spec(spec).pipeline.components["candidate"]

    assert type(loaded) is type(component)
    assert loaded == component
    assert component.component == descriptor.component == identity
    assert component.get_params() == params
    assert component.to_spec() == {"component": identity, "params": params}
    assert descriptor.category == category
    assert descriptor.provider == descriptor.distribution == "pllm"
    assert descriptor.capabilities
    assert descriptor.parameter_schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "component",
    [
        BinaryTableGatedMultiplyQ7(),
        R03CrtGatedMultiplyQ7(),
        ScalarProtectedTensorSchedule(),
        IndependentLanesProtectedTensorSchedule(),
        ChunkedIndependentLanesProtectedTensorSchedule(max_elements=4096),
    ],
)
def test_concrete_component_identity_cannot_be_switched_by_params(component: ComponentRef) -> None:
    with pytest.raises(ConfigurationError, match="unknown parameter path"):
        component.with_params(implementation="other")


def test_independent_lanes_schedule_clones_max_elements_immutably() -> None:
    original = IndependentLanesProtectedTensorSchedule()
    changed = original.with_params(max_elements=2)

    assert isinstance(changed, IndependentLanesProtectedTensorSchedule)
    assert original.get_params() == {"max_elements": 4}
    assert changed.get_params() == {"max_elements": 2}


@pytest.mark.parametrize("value", [True, 1, 1.0, "2", 5])
def test_independent_lanes_schedule_rejects_invalid_max_elements(value: object) -> None:
    with pytest.raises(ConfigurationError, match="integer"):
        IndependentLanesProtectedTensorSchedule(max_elements=value)


@pytest.mark.parametrize("value", [True, 4, 4.0, "5", 4_000_001])
def test_chunked_schedule_rejects_invalid_max_elements(value: object) -> None:
    with pytest.raises(ConfigurationError, match="integer"):
        ChunkedIndependentLanesProtectedTensorSchedule(max_elements=value)


@pytest.mark.parametrize(
    "component, params",
    [
        ("pllm/binary-table/v1", {"implementation": "other"}),
        ("pllm/r03-crt/v1", {"unexpected": True}),
        ("pllm/scalar/v1", {"max_elements": 1}),
        ("pllm/independent-lanes/v1", {}),
        ("pllm/chunked-independent-lanes/v1", {}),
        (
            "pllm/independent-lanes/v1",
            {"max_elements": 4, "implementation": "other"},
        ),
    ],
)
def test_concrete_component_deserialization_rejects_inexact_params(
    component: str, params: dict[str, object]
) -> None:
    spec = example().to_spec()
    spec["pipeline"]["components"]["candidate"] = {
        "component": component,
        "params": params,
    }

    with pytest.raises(ConfigurationError, match="unknown fields|missing fields"):
        Experiment.from_spec(spec)


def test_component_classes_live_in_capability_families() -> None:
    import pllm
    import pllm.components as components

    for name in (
        "BinaryTableGatedMultiplyQ7",
        "R03CrtGatedMultiplyQ7",
        "ScalarProtectedTensorSchedule",
        "IndependentLanesProtectedTensorSchedule",
        "ChunkedIndependentLanesProtectedTensorSchedule",
        "GatedMultiplyQ7",
        "ProtectedTensorSchedule",
    ):
        assert not hasattr(pllm, name)
        assert not hasattr(components, name)
    assert BinaryTableGatedMultiplyQ7.__module__ == "pllm.nonlinear"
    assert R03CrtGatedMultiplyQ7.__module__ == "pllm.nonlinear"
    assert ScalarProtectedTensorSchedule.__module__ == "pllm.schedulers"
    assert IndependentLanesProtectedTensorSchedule.__module__ == "pllm.schedulers"
    assert ChunkedIndependentLanesProtectedTensorSchedule.__module__ == "pllm.schedulers"
