"""Exact boundary delivery choices bind composition, never numeric semantics."""
import copy
from types import SimpleNamespace

import pytest

from pllm import Model, Pipeline
from pllm.components import create_component
from pllm.configuration import ComponentRef, ConfigurationError
from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition
from pllm.roles import PreparedBoundaryRoles, PrivateLookupAndFullHead
from pllm.runtime.model_binding import (
    CompiledRuntimeModel, RuntimeBindingError, _state_numeric_contract,
    _validate_private_boundary_contract,
)
from pllm.runtime.private_boundary_loader import build_private_boundary_contract
from pllm.runtime.private_boundary_protocol import BoundaryDescriptor, BoundaryError

from test_private_boundary_binding import boundary_plan, source_binding
from test_private_boundary_topology import experiment, imported_boundary


def pipeline(**params):
    return MaskedLinearCpu(Model("model.fixture"), boundary=PrivateLookupAndFullHead(**params),
                           topology=PreparedBoundaryRoles())


def test_defaults_keep_historical_component_and_pipeline_bytes():
    baseline = pipeline()
    explicit = pipeline(head_dispatch="serial", head_output_encoding="u32")
    assert PrivateLookupAndFullHead().to_spec() == {
        "component": "pllm/private-lookup-full-head/v1", "params": {}}
    assert explicit.canonical_bytes() == baseline.canonical_bytes()
    assert explicit.digest() == baseline.digest()


@pytest.mark.parametrize("dispatch", ["serial", "parallel"])
@pytest.mark.parametrize("encoding", ["u32", "uniform_residues"])
def test_choices_roundtrip_and_admit(dispatch, encoding):
    selected = pipeline(head_dispatch=dispatch, head_output_encoding=encoding)
    boundary = selected.components["boundary"]
    expected = {}
    if dispatch != "serial":
        expected["head_dispatch"] = dispatch
    if encoding != "u32":
        expected["head_output_encoding"] = encoding
    assert dict(boundary.params) == expected
    assert create_component(boundary.component, expected) == boundary
    restored = Pipeline.from_spec(selected.to_spec())
    assert restored.canonical_bytes() == selected.canonical_bytes()
    assert resolve_runtime_composition(restored).private_boundary
    assert (selected.digest() == pipeline().digest()) is (not expected)


@pytest.mark.parametrize("params", [
    {"head_dispatch": "striped"}, {"head_dispatch": True}, {"head_dispatch": None},
    {"head_output_encoding": "row_residues"}, {"head_output_encoding": 32},
    {"head_output_encoding": []}, {"unknown": "parallel"},
])
def test_invalid_choices_fail_closed(params):
    with pytest.raises((ConfigurationError, TypeError)):
        PrivateLookupAndFullHead(**params)
    with pytest.raises(ConfigurationError):
        create_component(PrivateLookupAndFullHead.descriptor.component, params)
    components = dict(pipeline().components)
    components["boundary"] = ComponentRef(PrivateLookupAndFullHead.descriptor.component, params)
    with pytest.raises(ConfigurationError):
        Pipeline(Model("model.fixture"), components)
    # Also exercise the resolver's independent gate without Pipeline normalization.
    assert resolve_runtime_composition(SimpleNamespace(components=components)) is None


@pytest.mark.parametrize("params", [
    {}, {"head_dispatch": "parallel"}, {"head_output_encoding": "uniform_residues"},
    {"head_dispatch": "parallel", "head_output_encoding": "uniform_residues"},
])
def test_native_contract_choices_require_independent_selection(boundary_plan, params):
    plan, document, contract = boundary_plan
    chosen = {**contract, **params}
    kwargs = dict(plan_digest=plan.digest, composition_digest="c" * 64,
                  vocabulary=32, hidden=8, activation_bits=8)
    _validate_private_boundary_contract(chosen, document, boundary_params=params, **kwargs)
    if params:
        with pytest.raises(RuntimeBindingError):
            _validate_private_boundary_contract(chosen, document, **kwargs)
        with pytest.raises(RuntimeBindingError):
            _validate_private_boundary_contract(contract, document, boundary_params=params, **kwargs)
    for key, value in (("weight_bits", 4), ("activation_bits", 4),
                       ("head_numeric", "approximate"), ("extra", True)):
        with pytest.raises(RuntimeBindingError):
            _validate_private_boundary_contract({**chosen, key: value}, document,
                                               boundary_params=params, **kwargs)
    biased = copy.deepcopy(document)
    next(op for op in biased["decode"]["operations"]
         if op["operator"] == "output_head")["attributes"]["bias"] = "bias.weight"
    with pytest.raises(RuntimeBindingError):
        _validate_private_boundary_contract(chosen, biased, boundary_params=params, **kwargs)


def test_delivery_choices_do_not_change_state_numeric_identity():
    def numeric(selected):
        binding = SimpleNamespace(_canonical_composition=selected.canonical_bytes(),
            model_plan_digest="a" * 64, runtime_config_digest="b" * 64, tokenizer_digest="c" * 64,
            to_spec=lambda: {"stages": [], "local_tensors": []},
            _bundle=SimpleNamespace(privacy={}))
        return _state_numeric_contract(binding)
    assert numeric(pipeline()) == numeric(pipeline(
        head_dispatch="parallel", head_output_encoding="uniform_residues"))


@pytest.mark.parametrize("params", [{}, {
    "head_dispatch": "parallel", "head_output_encoding": "uniform_residues"}])
def test_client_descriptor_ignores_untrusted_bundle_choices(boundary_plan, monkeypatch, params):
    binding, contract = source_binding(boundary_plan)
    contract.update(params)
    binding._bundle.privacy["private_boundary"].update(
        head_dispatch="forged", head_output_encoding="forged")
    monkeypatch.setattr(CompiledRuntimeModel, "_validated_schedule",
                        lambda self: {"private_boundary": contract})
    descriptor = binding.private_boundary_descriptor()
    assert descriptor.head_dispatch == params.get("head_dispatch", "serial")
    assert descriptor.head_output_encoding == params.get("head_output_encoding", "u32")


@pytest.mark.parametrize("dispatch,encoding", [
    ("serial", "u32"), ("parallel", "u32"),
    ("serial", "uniform_residues"), ("parallel", "uniform_residues"),
])
def test_loader_derives_choices_only_after_exact_native_comparison(
        experiment, imported_boundary, dispatch, encoding):
    params = dict(PrivateLookupAndFullHead(dispatch, encoding).params)
    spec = experiment.to_spec()
    spec["pipeline"]["components"]["boundary"]["params"] = params
    selected = type(experiment).from_spec(spec)
    schedule = imported_boundary.schedule
    document = copy.deepcopy(schedule.to_dict())
    document["private_boundary"].update(params)
    document["private_boundary"]["composition_digest"] = selected.pipeline.digest()
    schedule.composition_digest = selected.pipeline.digest()
    schedule.to_dict = lambda: document
    descriptor, _ = build_private_boundary_contract(imported_boundary, selected)
    assert (descriptor.head_dispatch, descriptor.head_output_encoding) == (dispatch, encoding)
    assert BoundaryDescriptor.from_dict(descriptor.to_dict()) == descriptor
    forged = {**document["private_boundary"], "head_dispatch": "parallel" if dispatch == "serial" else "serial"}
    document["private_boundary"] = forged
    with pytest.raises(BoundaryError, match="contract differs"):
        build_private_boundary_contract(imported_boundary, selected)
