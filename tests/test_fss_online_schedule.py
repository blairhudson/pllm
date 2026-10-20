"""Public online schedule binding; metadata doubles do not establish execution."""
import json

import pytest
from jsonschema import Draft202012Validator

from pllm import Model, _native
from pllm.components import create_component
from pllm.configuration import ConfigurationError, Pipeline, loads_configuration
from pllm.profiles import FuseFssSourceCpu, resolve_runtime_composition
from pllm.protocols import FuseFssSourceDecoder
from pllm.runtime import fss_source
from test_fss_source import bridge, write_checkpoint, admit, experiment, GPT, ROPE, CHECKED


@pytest.mark.parametrize("profile", [GPT, ROPE, CHECKED])
@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
@pytest.mark.parametrize("delivery", ["single", "batch-v1"])
def test_online_schedule_roundtrip_and_legacy_identity(profile, capacity, delivery):
    model = Model.path("/generated/config-only")
    serial = FuseFssSourceCpu(model, numeric_profile=profile, capacity=capacity,
                             material_delivery=delivery)
    assert serial == serial.with_params(linear__online_schedule="serial")
    assert "online_schedule" not in serial.linear.params
    wave = serial.with_params(linear__online_schedule="wavefront-v1")
    assert wave == FuseFssSourceCpu(model, linear=wave.linear)
    assert wave == Pipeline.from_spec(wave.to_spec())
    assert wave.linear == create_component(wave.linear.component, wave.linear.params)
    assert wave.linear == FuseFssSourceDecoder(**wave.linear.get_params())
    assert wave.linear.params == dict(serial.linear.params) | {"online_schedule": "wavefront-v1"}
    assert wave.digest() != serial.digest()
    assert wave.with_params(linear__online_schedule="serial") == serial
    assert resolve_runtime_composition(wave) == resolve_runtime_composition(serial)
    Draft202012Validator(wave.linear.describe().to_dict()["parameter_schema"]).validate(wave.linear.to_spec()["params"])
    exp = experiment("/generated/config-only", profile=profile, capacity=capacity,
                     material_delivery=delivery, online_schedule="wavefront-v1")
    assert loads_configuration(exp.canonical_bytes().decode()) == exp


@pytest.mark.parametrize("value", [None, True, 1, [], {}, "wavefront", "wavefront-v2", "SERIAL"])
def test_invalid_online_schedule_rejects(value):
    with pytest.raises(ConfigurationError):
        FuseFssSourceDecoder(numeric_profile=ROPE, online_schedule=value)
    with pytest.raises(ConfigurationError):
        FuseFssSourceCpu(Model.path("/generated/config-only"), numeric_profile=ROPE, online_schedule=value)


@pytest.mark.parametrize("requested,returned", [
    ("wavefront-v1", None), ("wavefront-v1", "serial"), ("serial", "wavefront-v1"),
    ("wavefront-v1", True), ("wavefront-v1", "wavefront-v2"),
])
def test_source_rejects_substituted_native_schedule_before_value_import(tmp_path, bridge, requested, returned):
    write_checkpoint(tmp_path, profile=ROPE)
    bridge.contract_mutation = (lambda value: value.pop("online_schedule", None)) if returned is None else (
        lambda value: value.update(online_schedule=returned))
    with pytest.raises(fss_source.FssSourceError, match="online.schedule"):
        admit(bridge, tmp_path, profile=ROPE, online_schedule=requested)
    assert all(owner.closed and not owner.frames for owner in bridge.owners)


def test_source_selects_wavefront_seal_and_preserves_original_ieee_frames(tmp_path, bridge):
    write_checkpoint(tmp_path, profile=ROPE)
    serial = admit(bridge, tmp_path, profile=ROPE)
    with fss_source.load_role_source(bridge.experiment, source=serial, expected_pins=serial.snapshot_pins) as loaded:
        original = loaded.native_program.builder.frames
        original_input = loaded.manifest["input_digest"]
    wave = admit(bridge, tmp_path, profile=ROPE, online_schedule="wavefront-v1")
    assert wave.snapshot_pins == serial.snapshot_pins
    with fss_source.load_role_source(bridge.experiment, source=wave, expected_pins=wave.snapshot_pins) as loaded:
        assert loaded.native_program.builder.online_schedule == "wavefront-v1"
        assert loaded.native_program.builder.frames == original
        assert loaded.manifest["input_digest"] == original_input


def test_missing_native_schedule_rejects_before_source_reads(tmp_path, bridge, monkeypatch):
    def unavailable(payload):
        assert json.loads(payload)["pipeline"]["components"]["linear"]["params"]["online_schedule"] == "wavefront-v1"
        raise ValueError("native online scheduling unavailable")
    monkeypatch.setattr(_native, "resolve_experiment", unavailable)
    monkeypatch.setattr(fss_source.io, "_read_bounded", lambda *a: pytest.fail("read before native admission"))
    with pytest.raises(ValueError, match="online scheduling unavailable"):
        fss_source.resolve_client_source(experiment(tmp_path, online_schedule="wavefront-v1"))
    assert not bridge.owners
