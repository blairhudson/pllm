"""Source-only forward contracts and frozen pre-forward serialization fixtures."""

import hashlib
import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from pllm.components import ComponentRef, create_component, get, get_component
from pllm.configuration import (
    ConfigurationError, Deployment, ExecutionBudget, Experiment, ForwardBudget,
    Model, Pipeline, loads_configuration,
)
from pllm.kernels import AppleMetal, Cpu
from pllm.numeric import SourceFloat32
from pllm.profiles import (
    ClientOnlyCpu, ForwardRuntimeComposition, resolve_decoder_runtime_composition,
    resolve_forward_runtime_composition, resolve_runtime_composition,
)
from pllm.quantization import QuantizationScheme, SymmetricPerRow
from pllm.state import ClientLocalKv


LEGACY_LOCAL = (
    b'{"budget":{"max_input_tokens":8,"max_new_tokens":2,"requests":1},'
    b'"deployment":{"kind":"local","root":".pllm/x"},"name":"x",'
    b'"pipeline":{"components":{"kernels":{"component":"pllm/cpu","params":{"threads":1}},'
    b'"linear":{"component":"pllm/cleartext-linear","params":{}},'
    b'"topology":{"component":"pllm/client-only/v1","params":{}}},'
    b'"model":{"source":"example/model"}},"schema":"pllm.experiment.v2"}'
)
LEGACY_NETWORK = (
    b'{"budget":{"max_input_tokens":8,"max_new_tokens":2,"requests":1},'
    b'"deployment":{"kind":"network","network_id":"dev","network_spec_digest":"'
    b'1111111111111111111111111111111111111111111111111111111111111111",'
    b'"snapshot_digest":null},"name":"x",'
    b'"pipeline":{"components":{"kernels":{"component":"pllm/cpu","params":{"threads":1}},'
    b'"linear":{"component":"pllm/cleartext-linear","params":{}},'
    b'"topology":{"component":"pllm/client-only/v1","params":{}}},'
    b'"model":{"source":"example/model"}},"schema":"pllm.experiment.v3"}'
)


def test_legacy_absent_numeric_preimages():
    for deployment, expected in (
        (Deployment.local(root=".pllm/x"), LEGACY_LOCAL),
        (Deployment.network(network_id="dev", network_spec_digest="1" * 64), LEGACY_NETWORK),
    ):
        experiment = Experiment("x", ClientOnlyCpu(Model("example/model")), deployment,
                                ExecutionBudget(1, 8, 2))
        assert experiment.canonical_bytes() == expected
        assert experiment.configuration_digest() == hashlib.sha256(
            b"pllm.configuration.v1\0" + expected
        ).hexdigest()
        assert "numeric" not in experiment.pipeline.to_spec()["components"]


def forward_experiment():
    return Experiment("forward", ClientOnlyCpu(Model("example/model"),
                      kernels=Cpu(threads=2), numeric=SourceFloat32()),
                      Deployment.local(root=".pllm/forward"), ForwardBudget(3, 8))


def test_source_float32_component_contract():
    numeric = SourceFloat32()
    assert isinstance(numeric, ComponentRef)
    assert not isinstance(numeric, QuantizationScheme)
    assert numeric.to_spec() == {"component": "pllm/source-float32/v1", "params": {}}
    assert numeric.get_params() == {}
    assert get(numeric.component) is SourceFloat32
    assert create_component(numeric.component, {}) == numeric
    descriptor = get_component(numeric.component)
    assert (descriptor.category, descriptor.category_version, descriptor.version) == (
        "pllm/numeric", "1", "1")
    assert (descriptor.provider, descriptor.distribution, descriptor.lifecycle_phase) == (
        "pllm", "pllm.run", "compile")
    assert descriptor.to_dict()["parameter_schema"] == {
        "type": "object", "properties": {}, "additionalProperties": False}
    for params in ({"weight_bits": 32}, {"activation_bits": 32}, {"unknown": 1}):
        with pytest.raises(ConfigurationError):
            create_component(numeric.component, params)


def test_forward_resolution_has_no_integer_decoder_fields():
    experiment = forward_experiment()
    pipeline = experiment.pipeline
    resolved = resolve_forward_runtime_composition(pipeline)
    assert resolved == ForwardRuntimeComposition(threads=2)
    assert dataclasses.asdict(resolved) == {
        "runtime_kind": "client_only_source_f32",
        "numeric_profile": "pllm.numeric.source_float32.v1", "threads": 2}
    assert not hasattr(resolved, "weight_bits")
    assert not hasattr(resolved, "activation_bits")
    assert resolve_runtime_composition(pipeline) is None
    with pytest.raises(ConfigurationError):
        resolve_decoder_runtime_composition(pipeline)
    with pytest.raises(ConfigurationError):
        experiment.resolve()
    legacy = resolve_decoder_runtime_composition(ClientOnlyCpu(Model("example/model")))
    assert legacy.weight_bits == legacy.activation_bits == 8
    assert pipeline.numeric == SourceFloat32()
    assert pipeline.with_params(kernels__threads=4).kernels == Cpu(threads=4)
    assert pipeline.with_params(numeric=None).numeric is None


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", None, 1 << 64])
@pytest.mark.parametrize("field", ["requests", "max_input_tokens"])
def test_forward_budget_requires_positive_actual_uint64(field, value):
    values = {"requests": 1, "max_input_tokens": 8, field: value}
    with pytest.raises(ConfigurationError):
        ForwardBudget(**values)


def test_forward_budget_is_immutable_and_does_not_broaden_decoder_budget():
    budget = ForwardBudget(1, 8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        budget.requests = 2
    assert budget.with_params(max_input_tokens=9) == ForwardBudget(1, 9)
    assert ForwardBudget((1 << 64) - 1, 1).requests == (1 << 64) - 1
    # Preserve historical unbounded positive Python ints in ExecutionBudget.
    assert ExecutionBudget(1 << 64, 1, 1).requests == 1 << 64


def test_forward_roundtrip_and_budget_schema_are_distinct():
    experiment = forward_experiment()
    spec = experiment.to_dict()
    assert spec["schema"] == "pllm.experiment.forward.v1"
    assert spec["budget"] == {"requests": 3, "max_input_tokens": 8}
    assert Experiment.from_dict(spec) == experiment
    assert Experiment.from_json(experiment.to_json()) == experiment
    assert loads_configuration(yaml.safe_dump(spec), format="yaml") == experiment
    assert Pipeline.from_spec(experiment.pipeline.to_spec()) == experiment.pipeline
    for schema in ("pllm.experiment.v2", "pllm.experiment.v3"):
        with pytest.raises(ConfigurationError):
            Experiment.from_dict({**spec, "schema": schema})
    with pytest.raises(ConfigurationError):
        Experiment.from_dict({**spec, "budget": {**spec["budget"], "max_new_tokens": 1}})
    with pytest.raises(ConfigurationError):
        Experiment.from_dict({**spec, "workload": "forward"})
    with pytest.raises(ConfigurationError):
        Experiment.from_json(experiment.to_json().replace('"requests":3', '"requests":true'))
    with pytest.raises(ConfigurationError):
        Experiment.from_dict({**spec, "schema": []})


def test_forward_budget_and_numeric_selection_are_both_required():
    experiment = forward_experiment()
    with pytest.raises(ConfigurationError):
        experiment.with_params(budget=ExecutionBudget(1, 8, 1))
    with pytest.raises(ConfigurationError):
        experiment.with_params(pipeline=ClientOnlyCpu(Model("example/model")))
    with pytest.raises(ConfigurationError):
        experiment.with_params(deployment=Deployment.network(
            network_id="dev", network_spec_digest="1" * 64))
    with pytest.raises(ConfigurationError):
        ClientOnlyCpu(Model("x"), numeric=SourceFloat32(), quantization=SymmetricPerRow())
    with pytest.raises(ConfigurationError):
        ClientOnlyCpu(Model("x"), numeric=SourceFloat32(), kernels=AppleMetal())
    with pytest.raises(ConfigurationError):
        ClientOnlyCpu(Model("x"), numeric=SourceFloat32(), state=ClientLocalKv())
    with pytest.raises(ConfigurationError):
        ClientOnlyCpu(Model("x"), numeric=SourceFloat32(), kernels=Cpu(threads=1 << 64))


@pytest.mark.parametrize("slot", ["quantization", "cache", "scheduler", "state", "preparation",
                                  "verification", "private", "remote", "extra"])
def test_raw_pipeline_rejects_extra_forward_components(slot):
    pipeline = forward_experiment().pipeline
    with pytest.raises(ConfigurationError):
        Pipeline(pipeline.model, {**pipeline.components, slot: ComponentRef("example/extra")})
    spec = pipeline.to_spec()
    spec["components"][slot] = {"component": "example/extra", "params": {}}
    with pytest.raises(ConfigurationError):
        Pipeline.from_spec(spec)


@pytest.mark.parametrize("slot", ["numeric", "linear", "topology", "kernels"])
def test_serialized_forward_components_reject_unknown_params(slot):
    spec = forward_experiment().to_spec()
    spec["pipeline"]["components"][slot]["params"]["unknown"] = 1
    with pytest.raises(ConfigurationError):
        Experiment.from_dict(spec)


def test_sdk_forward_lazy_factory_and_close(monkeypatch):
    calls = []
    module = types.ModuleType("pllm.runtime.forward_binding")

    class FakeCompiled:
        @classmethod
        def _from_experiment(cls, experiment, *, memory_bytes):
            calls.append((experiment, memory_bytes))
            return cls()

        def forward(self, inputs):
            calls.append(inputs)
            if inputs == "fail":
                raise ValueError("failed")
            return "result"

        def close(self):
            calls.append("closed")

    module.CompiledForwardModel = FakeCompiled
    monkeypatch.setitem(sys.modules, module.__name__, module)
    experiment = forward_experiment()
    assert experiment.forward("inputs", memory_bytes=1024) == "result"
    assert calls == [(experiment, 1024), "inputs", "closed"]
    with pytest.raises(ValueError, match="failed"):
        experiment.forward("fail", memory_bytes=1024)
    assert calls[-1] == "closed"
    for memory in (True, 0, -1, 1.0, 1 << 64):
        with pytest.raises(ConfigurationError):
            experiment.compile_forward(memory_bytes=memory)


def test_public_forward_exports_and_stubs():
    import pllm
    from pllm.config import ForwardBudget as FacadeBudget

    assert pllm.SourceFloat32 is SourceFloat32
    assert pllm.ForwardBudget is FacadeBudget is ForwardBudget
    assert {"ForwardInputs", "ForwardResult", "CompiledForwardModel"} <= set(pllm.__all__)
    root = Path(__file__).resolve().parents[1]
    for path, names in (
        ("python/pllm/configuration.pyi", ("ForwardBudget", "compile_forward", "ForwardInputs")),
        ("python/pllm/profiles/__init__.pyi", ("SourceFloat32", "ForwardRuntimeComposition",
                                               "resolve_forward_runtime_composition")),
        ("python/pllm/__init__.pyi", ("ForwardBudget", "ForwardInputs", "ForwardResult",
                                    "CompiledForwardModel", "SourceFloat32")),
    ):
        text = (root / path).read_text()
        assert all(name in text for name in names)


def test_forward_json_schema_is_separate_and_strict():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas/experiment-forward.schema.json").read_text())
    # Resolve ordinary local model schema without network access.
    schema["properties"]["pipeline"]["properties"]["model"] = json.loads(
        (root / "schemas/model.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    spec = forward_experiment().to_spec()
    validator.validate(spec)
    for mutate in (
        lambda value: value["budget"].update(max_new_tokens=1),
        lambda value: value["budget"].update(requests=1 << 64),
        lambda value: value["budget"].update(requests=True),
        lambda value: value["pipeline"]["components"]["numeric"]["params"].update(weight_bits=32),
        lambda value: value["pipeline"]["components"].update(state={"component": "x", "params": {}}),
        lambda value: value.update(schema="pllm.experiment.v2"),
    ):
        invalid = json.loads(json.dumps(spec))
        mutate(invalid)
        assert not validator.is_valid(invalid)


def test_legacy_pipeline_digest_domain_and_literal_composition():
    expected = (
        b'{"components":{"kernels":{"component":"pllm/cpu","params":{"threads":1}},'
        b'"linear":{"component":"pllm/cleartext-linear","params":{}},'
        b'"topology":{"component":"pllm/client-only/v1","params":{}}},'
        b'"model":{"source":"example/model"}}'
    )
    pipeline = ClientOnlyCpu(Model("example/model"))
    assert pipeline.canonical_bytes() == expected
    assert pipeline.digest() == hashlib.sha256(b"pllm.pipeline.v2\0" + expected).hexdigest()
