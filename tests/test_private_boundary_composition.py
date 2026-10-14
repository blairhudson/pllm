"""Pure immutable selection checks; no checkpoint or native execution required."""
import dataclasses

import pytest

from pllm import Model, Pipeline
from pllm.components import create_component
from pllm.kernels import AppleMetal
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.roles import (
    OutputHeadAtInference,
    PreparedBoundaryRoles,
    PreparedProviderRoles,
    PrivateLookupAndFullHead,
    RoleGraph,
)
from pllm.roles.topology import graph_for_runtime
from pllm.verification import FreivaldsVerify


def selected(**kwargs):
    return MaskedLinearCpu(Model("model.fixture"), boundary=PrivateLookupAndFullHead(),
                           topology=PreparedBoundaryRoles(), **kwargs)


def test_private_boundary_selection_roundtrip_and_offline_preparation():
    pipeline = selected()
    options = resolve_runtime_composition(pipeline)
    assert options is not None and options.private_boundary
    assert options.requires_preparation and not options.remote_output_head
    assert options.client_runtime == "masked_transformer_v1"
    assert resolve_runtime_composition(Pipeline.from_spec(pipeline.to_spec())) == options
    graph = graph_for_runtime(options)
    assert {role.id for role in graph.roles} == {
        "client", "preparation", "inference", "boundary_a", "boundary_b"}
    assert graph.separate_operators == (("boundary_a", "boundary_b"), ("inference", "preparation"))
    assert all(channel.phase == "offline" for channel in graph.channels
               if "preparation" in (channel.source, channel.target))
    assert graph == RoleGraph.from_spec(graph.to_spec())
    assert graph.digest() == RoleGraph.from_spec(graph.to_spec()).digest()
    for cls in (PrivateLookupAndFullHead, PreparedBoundaryRoles):
        component = create_component(cls.descriptor.component, {})
        assert isinstance(component, cls)
        assert component.to_spec() == cls().to_spec()
        with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
            component.component = "foreign"


@pytest.mark.parametrize("reduction", [None, "prefix_f32"])
@pytest.mark.parametrize("encoding", ["none", "zlib", "artifacts"])
def test_private_boundary_permits_ordinary_demand_inventory_and_delivery(reduction, encoding):
    pipeline = selected(
        quantization=SymmetricPerRow(causal_reduction=reduction),
        inventory=PreparedInventory(allocation="demand", refill="on-demand"),
        linear=MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues"),
        delivery=ClientBundleTransport(encoding),
    )
    options = resolve_runtime_composition(pipeline)
    assert options is not None and options.private_boundary
    assert options.prepared_output_encoding == "row_residues"
    assert options.bundle_compression == encoding


@pytest.mark.parametrize("slot,value", [
    ("topology", None),
    ("topology", PreparedProviderRoles()),
    ("boundary", OutputHeadAtInference()),
    ("verification", FreivaldsVerify()),
    ("kernels", AppleMetal()),
    ("quantization", SymmetricPerRow(weight_bits=4)),
    ("quantization", SymmetricPerRow(activation_bits=4)),
])
def test_private_boundary_rejects_implicit_or_incompatible_selection(slot, value):
    components = dict(selected().components)
    if value is None:
        components.pop(slot)
    else:
        components[slot] = value
    assert resolve_runtime_composition(Pipeline(model=Model("model.fixture"), components=components)) is None


def test_ordinary_selection_retains_original_identity_and_head_flag():
    ordinary = MaskedLinearCpu(Model("model.fixture"))
    options = resolve_runtime_composition(ordinary)
    assert options is not None and not options.private_boundary and not options.remote_output_head
    assert "topology" not in ordinary.to_spec()["components"]
    old = resolve_runtime_composition(MaskedLinearCpu(Model("model.fixture"), boundary=OutputHeadAtInference()))
    assert old is not None and old.remote_output_head and not old.private_boundary
