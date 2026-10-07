"""Component discovery, composition/replay and benchmark admission use one contract."""

import json
from pathlib import Path

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline, _native
from pllm.components import create_component, get, list_components
from pllm.configuration import ComponentRef, ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import MaskedWireGatedMultiplyQ12, NonlinearProtocol, SharedRescale
from pllm.profiles import MaskedLinearCpu


@pytest.mark.parametrize("component", [MaskedWireGatedMultiplyQ12(), MaskedWireGatedMultiplyQ12(layout="fused"), MaskedWireGatedMultiplyQ12(material_encoding="seeded"), SharedRescale()])
def test_components_round_trip_through_discovery_and_experiment(component):
    assert isinstance(component, NonlinearProtocol)
    assert get(component.component) is type(component)
    assert component.describe() in list_components()
    assert create_component(component.component, component.params) == component
    assert type(component).from_params(component.get_params()) == component
    assert "lanes" not in component.get_params()
    assert "samples" not in component.get_params()
    with pytest.raises(ConfigurationError):
        create_component(component.component, component.params | {"samples": 3})

    experiment = Experiment(
        name="operator-component-config",
        pipeline=Pipeline(model=Model.tiny(), components={"nonlinear": component}),
        deployment=Deployment.local(root="local://operator-config"),
        budget=ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=1),
    )
    replay = Experiment.from_spec(json.loads(experiment.canonical_bytes()))
    assert type(replay.pipeline.components["nonlinear"]) is type(component)
    assert replay.configuration_digest() == experiment.configuration_digest()
    result = benchmark_operator(replay.pipeline.components["nonlinear"], lanes=1, samples=1)
    assert result.configuration["component"] == component.to_spec()
    # Configuration discovery cannot silently activate an unbound decoder method.
    baseline = MaskedLinearCpu(model=Model.tiny())
    with pytest.raises(ConfigurationError):
        Experiment(
            name=experiment.name,
            pipeline=Pipeline(model=baseline.model, components={**baseline.components, "nonlinear": component}),
            deployment=experiment.deployment,
            budget=experiment.budget,
        ).resolve()


def test_nested_component_updates_bind_configuration_and_numeric_identity():
    component = SharedRescale(layout="unfused")
    pipeline = Pipeline(model=Model.tiny(), components={"nonlinear": component})
    fused = pipeline.with_params(components__nonlinear__layout="fused")
    assert pipeline.components["nonlinear"].params["layout"] == "unfused"
    assert fused.digest() != pipeline.digest()
    before, after = (
        benchmark_operator(p.components["nonlinear"], lanes=2, samples=1)
        for p in (pipeline, fused)
    )
    assert before.numeric_contract_digest == after.numeric_contract_digest
    assert before.configuration_digest != after.configuration_digest
    assert before.quality == after.quality
    assert before.resources.peer_frame_bytes == 2 * after.resources.peer_frame_bytes
    with pytest.raises(ConfigurationError):
        pipeline.with_params(components__nonlinear__shift=16)


def test_forged_component_or_replay_rejects_before_native_issuance(monkeypatch):
    config = benchmark_operator(SharedRescale(), lanes=1, samples=1).configuration

    def forbidden(*args):
        pytest.fail("invalid configuration reached issuance")

    monkeypatch.setattr(_native, "shared_rescale_benchmark", forbidden)
    for component in (
        ComponentRef("pllm/unknown"),
        ComponentRef(SharedRescale.descriptor.component, {**SharedRescale().params, "bits": True}),
        ComponentRef(SharedRescale.descriptor.component, {**SharedRescale().params, "rounding": "nearest"}),
        ComponentRef(MaskedWireGatedMultiplyQ12.descriptor.component, {"numeric_profile": "changed"}),
    ):
        with pytest.raises(ConfigurationError):
            benchmark_operator(component)
    for inputs in ({}, {"values": None}, {"values": [0], "extra": [0]}, {"gate_q24": [0]}):
        with pytest.raises(ConfigurationError):
            benchmark_operator(SharedRescale(), lanes=1, inputs=inputs)
    for change in (
        {"component": {"component": SharedRescale.descriptor.component}},
        {"component": {**config["component"], "extra": True}},
        {"lanes": True}, {"samples": 17}, {"unknown": 0},
    ):
        with pytest.raises(ConfigurationError):
            replay_operator_benchmark(config | change)


def test_historical_masked_wire_record_migrates_without_rewriting_evidence():
    path = Path(__file__).resolve().parents[1] / "docs/evidence/research-masked-wire-sdk.json"
    original = path.read_bytes()
    old = json.loads(original)
    result = replay_operator_benchmark(old["configuration"]).to_dict()
    assert result["configuration"]["component"] == MaskedWireGatedMultiplyQ12().to_spec()
    assert result["configuration_digest"] != old["configuration_digest"]
    # Source changes must retain a new implementation identity, while replaying
    # the historical numeric/layout contract and preserving the original file.
    assert result["numeric_contract_digest"] == old["numeric_contract_digest"]
    for key in ("resources", "quality"):
        assert all(result[key][name] == value for name, value in old[key].items())
    assert path.read_bytes() == original


def test_masked_wire_overlap_binds_dispatch_without_changing_default_identity():
    from pllm.profiles import TwoOnlineOffsetCpu

    serial = MaskedWireGatedMultiplyQ12()
    assert serial.params == {}
    assert serial.with_params(dispatch="serial") == serial
    overlap = serial.with_params(dispatch="overlap")
    assert create_component(overlap.component, overlap.params) == overlap
    pipeline = TwoOnlineOffsetCpu(Model.tiny(), nonlinear=overlap)
    experiment = Experiment(name="overlap", pipeline=pipeline, deployment=Deployment.local(root="local://overlap"),
        budget=ExecutionBudget(requests=1, max_input_tokens=4, max_new_tokens=2))
    assert Experiment.from_spec(experiment.to_spec()) == experiment
    with pytest.raises(ConfigurationError, match="compiled two-worker"):
        benchmark_operator(overlap, lanes=1, samples=1)
