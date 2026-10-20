"""Material-selection/configuration/source controls with explicit native doubles.

These tests do not qualify native batching, execution parity or serving.
"""
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, _native
from pllm.components import create_component, get
from pllm.configuration import ConfigurationError, Pipeline, loads_configuration
from pllm.profiles import FuseFssSourceCpu, resolve_runtime_composition
from pllm.protocols import FuseFssSourceDecoder
from pllm.runtime import fss_source
from test_fss_source import bridge, write_checkpoint, admit, experiment, GPT, ROPE, CHECKED


@pytest.mark.parametrize("profile", [GPT, ROPE, CHECKED])
@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
def test_material_delivery_default_and_opt_in_roundtrip(profile, capacity):
    model = Model.path("/generated/config-only")
    single = FuseFssSourceCpu(model, numeric_profile=profile, capacity=capacity)
    explicit = FuseFssSourceCpu(model, numeric_profile=profile, capacity=capacity, material_delivery="single")
    assert single.canonical_bytes() == explicit.canonical_bytes()
    assert "material_delivery" not in single.linear.params
    batch = single.with_params(linear__material_delivery="batch-v1")
    assert batch == FuseFssSourceCpu(model, numeric_profile=profile, capacity=capacity, material_delivery="batch-v1")
    assert batch == FuseFssSourceCpu(model, linear=batch.linear)
    assert batch.linear.params == dict(single.linear.params) | {"material_delivery": "batch-v1"}
    assert batch.digest() != single.digest()
    assert batch.with_params(linear__material_delivery="single").canonical_bytes() == single.canonical_bytes()
    assert batch.get_params()["linear__material_delivery"] == "batch-v1"
    assert single.get_params()["linear__material_delivery"] == "single"
    assert FuseFssSourceDecoder(**batch.linear.get_params()) == batch.linear
    assert get(batch.linear.component) is FuseFssSourceDecoder
    assert create_component(batch.linear.component, batch.linear.params) == batch.linear
    Draft202012Validator(batch.linear.describe().to_dict()["parameter_schema"]).validate(batch.linear.to_spec()["params"])
    assert Pipeline.from_spec(batch.to_spec()).digest() == batch.digest()
    exp = Experiment("batch-config-only", batch, Deployment.local(root="local://batch-config-only"),
                     ExecutionBudget(requests=1, max_input_tokens=1, max_new_tokens=1))
    assert loads_configuration(exp.canonical_bytes().decode()) == exp
    # Material delivery does not create a new numerical/role/state choice.
    assert resolve_runtime_composition(single) == resolve_runtime_composition(batch)


@pytest.mark.parametrize("value", [None, True, 1, [], {}, "batch", "batch-v2", "SINGLE"])
def test_invalid_material_selection_never_becomes_a_default(value):
    with pytest.raises(ConfigurationError):
        FuseFssSourceDecoder(numeric_profile=ROPE, material_delivery=value)
    with pytest.raises(ConfigurationError):
        FuseFssSourceCpu(Model.path("/generated/config-only"), numeric_profile=ROPE, material_delivery=value)
    with pytest.raises(ConfigurationError):
        create_component(FuseFssSourceDecoder.descriptor.component,
                         {"numeric_profile": ROPE, "material_delivery": value})


def test_duplicate_profile_material_selection_is_rejected():
    method = FuseFssSourceDecoder(numeric_profile=ROPE, material_delivery="batch-v1")
    with pytest.raises(ConfigurationError, match="belongs in linear"):
        FuseFssSourceCpu(Model.path("/generated/config-only"), linear=method, material_delivery="batch-v1")


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
def test_batch_source_keeps_ieee_rotary_and_numeric_source_identity(tmp_path, bridge, capacity):
    write_checkpoint(tmp_path, profile=CHECKED)
    single = admit(bridge, tmp_path, profile=CHECKED, capacity=capacity)
    with fss_source.load_role_source(bridge.experiment, source=single, expected_pins=single.snapshot_pins) as loaded:
        frames = loaded.native_program.builder.frames
        rotary = loaded.rotary_source
        single_manifest = loaded.manifest
    batch = admit(bridge, tmp_path, profile=CHECKED, capacity=capacity, material_delivery="batch-v1")
    assert batch.material_delivery == "batch-v1" and single.material_delivery == "single"
    assert batch.snapshot_pins == single.snapshot_pins
    assert batch.contract["source_shapes"] == single.contract["source_shapes"]
    with fss_source.load_role_source(bridge.experiment, source=batch, expected_pins=batch.snapshot_pins) as loaded:
        assert loaded.native_program.builder.frames == frames
        assert loaded.rotary_source == rotary
        assert loaded.manifest["profile"] == single_manifest["profile"] == CHECKED
        assert loaded.manifest["input_digest"] == single_manifest["input_digest"]
        assert loaded.manifest["contract"] != single_manifest["contract"]
    with pytest.raises(fss_source.FssSourceError, match="exact admitted Experiment"):
        single.runtime_contract(bridge.experiment, 3, 3)


@pytest.mark.parametrize("requested,returned", [("batch-v1", None), ("batch-v1", "single"),
    ("single", "batch-v1"), ("batch-v1", True), ("batch-v1", "batch-v2")])
def test_source_contract_requires_exact_native_material_policy(tmp_path, bridge, requested, returned):
    write_checkpoint(tmp_path, profile=ROPE)
    bridge.contract_mutation = (lambda value: value.pop("material_delivery", None)) if returned is None else (
        lambda value: value.update(material_delivery=returned))
    with pytest.raises(fss_source.FssSourceError, match="material.delivery"):
        admit(bridge, tmp_path, profile=ROPE, material_delivery=requested)
    assert all(owner.closed and not owner.frames for owner in bridge.owners)


def test_unavailable_native_batch_selection_rejects_before_files(tmp_path, bridge, monkeypatch):
    def unavailable(payload):
        assert json.loads(payload)["pipeline"]["components"]["linear"]["params"]["material_delivery"] == "batch-v1"
        raise ValueError("native material batching unavailable")
    monkeypatch.setattr(_native, "resolve_experiment", unavailable)
    monkeypatch.setattr(fss_source.io, "_read_bounded", lambda *a: pytest.fail("source read before native admission"))
    with pytest.raises(ValueError, match="native material batching unavailable"):
        fss_source.resolve_client_source(experiment(tmp_path, material_delivery="batch-v1"))
    assert not bridge.owners


def test_generated_and_qwen_factories_match_source_workload_and_numeric_choice(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from examples.papers.fusefss_smoke import dense_single_smoke, dense_batch_smoke
    from examples.papers.fusefss import qwen_checked_smoke, qwen_checked_batch_smoke
    generated = dense_single_smoke(tmp_path)
    payloads = {path.name: path.read_bytes() for path in (tmp_path / "source").iterdir()}
    batch = dense_batch_smoke(tmp_path)
    assert payloads == {path.name: path.read_bytes() for path in (tmp_path / "source").iterdir()}
    assert generated.budget == ExecutionBudget(requests=1, max_input_tokens=1, max_new_tokens=2)
    for single, batched in ((generated, batch), (qwen_checked_smoke(), qwen_checked_batch_smoke())):
        assert single.name != batched.name  # Ordinary CLI comparisons require unique names.
        assert single.pipeline.model == batched.pipeline.model
        assert single.budget == batched.budget
        assert single.pipeline.linear.params["numeric_profile"] == batched.pipeline.linear.params["numeric_profile"]
        assert "material_delivery" not in single.pipeline.linear.params
        assert batched.pipeline.linear.params["material_delivery"] == "batch-v1"
        assert single.configuration_digest() != batched.configuration_digest()
        assert batched.with_params(name=single.name, pipeline__linear__material_delivery="single") == single
