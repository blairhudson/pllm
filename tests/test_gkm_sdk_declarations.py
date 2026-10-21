"""GKM declarations bind exact public contracts without bypassing native admission."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm import profiles
from pllm.components import create_component, get, list_components, planned_component
from pllm.configuration import ConfigurationError, loads_configuration
from pllm.kernels import AppleMetal, Cpu
from pllm.passes import KvCacheEviction
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import (
    MaskedLinearCpu,
    OpenWeightVerifiableCpu,
    TwoOnlineOffsetCpu,
    resolve_runtime_composition,
)
from pllm.protocols import ClientBundleTransport, MaskedLinear, OpenWeightVerifiableOutsourcing
from pllm.quantization import PublicPerChannelEqualized, SymmetricPerRow
from pllm.roles import (
    PreparedProviderRoles,
    Role,
    RoleGraph,
    TwoOnlineDelegationRoles,
    TwoOnlineOffsetRoles,
    two_online_delegation_graph,
    two_online_reference_graph,
)
from pllm.roles.topology import graph_for_runtime
from pllm.state import ClientLocalKv, ClientPrefixReuse
from pllm.verification import FreivaldsVerify, PublishedLinearVerify


def _pipeline() -> OpenWeightVerifiableCpu:
    # Declaration-only source: execution must separately admit a complete bounded fixture.
    return OpenWeightVerifiableCpu(Model("fixtures/gkm-complete", revision="source-pin"))


def _experiment(deployment: Deployment | None = None) -> Experiment:
    return Experiment(
        name="gkm-declaration",
        pipeline=_pipeline(),
        deployment=deployment or Deployment.local(root="local://gkm-declaration"),
        budget=ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=2),
    )


@pytest.mark.parametrize("component", [
    OpenWeightVerifiableOutsourcing(), OpenWeightVerifiableOutsourcing(sharing="seeded"),
    OpenWeightVerifiableOutsourcing(execution="source_stage_batch_v1"),
    PublishedLinearVerify(), TwoOnlineDelegationRoles(),
])
def test_gkm_component_discovery_schema_and_immutable_recreation(component) -> None:
    descriptor = type(component).describe()
    assert descriptor in list_components()
    assert get(component.component) is type(component)
    schema = descriptor.to_dict()["parameter_schema"]
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(component.to_spec()["params"])
    restored = create_component(component.component, component.get_params(deep=False))
    assert type(restored) is type(component)
    assert restored == component
    assert type(component)(**component.get_params(deep=False)) == component
    with pytest.raises(dataclasses.FrozenInstanceError):
        component.component = "forged/component"
    with pytest.raises(TypeError):
        component.params["secret_seed"] = "private"
    with pytest.raises(TypeError):
        descriptor.parameter_schema["additionalProperties"] = True
    copied = component.to_spec()
    copied["params"]["secret_seed"] = "private"
    assert "secret_seed" not in component.params
    with pytest.raises(ConfigurationError, match="unknown fields"):
        create_component(component.component, copied["params"])


def test_promoted_component_retires_planned_identity_without_claiming_unbounded_execution() -> None:
    plan = planned_component("open-weight-verifiable")
    assert plan.status == "implemented"
    assert OpenWeightVerifiableOutsourcing.__module__ == "pllm.protocols"
    assert "full-width admission" in plan.gate
    assert "bounded-experimental-gkm" in OpenWeightVerifiableOutsourcing.describe().capabilities
    with pytest.raises(ConfigurationError, match="retired planned identity"):
        create_component(plan.identity, {})


def test_typed_profile_and_canonical_pipeline_round_trip_preserve_contract() -> None:
    pipeline = _pipeline()
    assert pipeline.profile == "research.open_weight_verifiable_cpu"
    assert "OpenWeightVerifiableCpu" in profiles.__all__
    recreated = profiles.OpenWeightVerifiableCpu(**pipeline.get_params(deep=False))
    assert type(recreated) is OpenWeightVerifiableCpu
    assert recreated == pipeline
    assert pipeline.get_params()["linear__sharing"] == "full"
    assert pipeline.get_params()["linear__generation"] == "fixed_length_greedy"
    assert pipeline.get_params()["verification__acceptance"] == "request"
    assert pipeline.get_params()["kernels__threads"] == 1
    assert isinstance(pipeline.state, ClientLocalKv) and not pipeline.state.params
    assert pipeline.to_spec()["components"] == {
        "linear": {"component": "pllm/open-weight-verifiable-outsourcing/v1", "params": {
            "sharing": "full", "generation": "fixed_length_greedy"}},
        "verification": {"component": "pllm/published-linear-verify/v1", "params": {
            "acceptance": "request"}},
        "topology": {"component": "pllm/two-online-delegation-workers/v1", "params": {}},
        "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
        "state": {"component": "pllm/client-local-kv", "params": {}},
    }
    restored = Pipeline.from_spec(json.loads(pipeline.canonical_bytes()))
    assert type(restored) is Pipeline
    assert type(restored.components["linear"]) is OpenWeightVerifiableOutsourcing
    assert type(restored.components["verification"]) is PublishedLinearVerify
    assert type(restored.components["topology"]) is TwoOnlineDelegationRoles
    assert restored == pipeline
    assert restored.digest() == recreated.digest() == pipeline.digest()
    assert restored.canonical_bytes() == pipeline.canonical_bytes()
    with pytest.raises(TypeError):
        pipeline.components["state"] = ClientLocalKv()
    changed = pipeline.with_params(model__revision="other-source-pin")
    assert type(changed) is OpenWeightVerifiableCpu
    assert changed.digest() != pipeline.digest()
    assert pipeline.model.revision == "source-pin"


@pytest.mark.parametrize("sharing", ["full", "seeded"])
def test_explicit_source_stage_batch_selector_preserves_scalar_defaults(sharing) -> None:
    scalar = _pipeline().with_params(linear__sharing=sharing)
    original = scalar.canonical_bytes()
    assert "execution" not in scalar.linear.to_spec()["params"]
    assert scalar.with_params(linear__execution=None).canonical_bytes() == original
    selected = scalar.with_params(linear__execution="source_stage_batch_v1")
    assert selected.linear.params["execution"] == "source_stage_batch_v1"
    assert selected.digest() != scalar.digest()
    assert selected.linear.NUMERIC_PROFILE == scalar.linear.NUMERIC_PROFILE
    assert resolve_runtime_composition(selected) is not None
    restored = Pipeline.from_spec(json.loads(selected.canonical_bytes()))
    assert restored == selected
    assert restored.with_params(components__linear__execution=None).canonical_bytes() == original


@pytest.mark.parametrize("execution", [False, True, 1, "scalar", "auto", "batch", {}])
def test_source_stage_batch_rejects_unknown_selector(execution) -> None:
    with pytest.raises(ConfigurationError, match="GKM execution"):
        OpenWeightVerifiableOutsourcing(execution=execution)


@pytest.mark.parametrize("deployment", [
    Deployment.local(root="local://gkm-declaration"),
    Deployment.network(network_id="gkm", network_spec_digest="a" * 64),
])
def test_experiment_json_and_configuration_loader_round_trips(deployment) -> None:
    experiment = _experiment(deployment)
    restored = Experiment.from_spec(json.loads(experiment.canonical_bytes()))
    loaded = loads_configuration(experiment.canonical_bytes().decode())
    for copy in (restored, loaded):
        assert copy.canonical_bytes() == experiment.canonical_bytes()
        assert copy.configuration_digest() == experiment.configuration_digest()
        assert copy.pipeline.digest() == experiment.pipeline.digest()
        assert resolve_runtime_composition(copy.pipeline) == resolve_runtime_composition(experiment.pipeline)
    changed = experiment.with_params(budget__max_new_tokens=3)
    assert changed.configuration_digest() != experiment.configuration_digest()
    assert changed.pipeline.digest() == experiment.pipeline.digest()


def test_resolver_describes_source_numerics_and_final_acceptance_without_profile_label() -> None:
    pipeline = _pipeline()
    contract = resolve_runtime_composition(Pipeline.from_spec(pipeline.to_spec()))
    assert contract is not None
    assert contract.runtime_kind == "gkm"
    assert contract.privacy_mode == "gkm_public"
    assert contract.client_runtime == "compiled_gkm_v1"
    assert contract.numeric_profile == OpenWeightVerifiableOutsourcing.NUMERIC_PROFILE
    assert contract.local_numeric_profile == OpenWeightVerifiableOutsourcing.LOCAL_NUMERIC_PROFILE
    assert (contract.weight_bits, contract.activation_bits) == (16, 64)
    assert contract.sharing == "full"
    assert contract.generation_policy == "fixed_length_greedy"
    assert contract.verification_window == "request_end"
    assert contract.verification_component == PublishedLinearVerify.descriptor.component
    assert contract.setup_owner == "client"
    assert not contract.requires_preparation
    assert contract.prepared_inventory_rows == 0
    assert contract.prefix_cache_bytes == 0 and not contract.decode
    assert graph_for_runtime(contract) == two_online_delegation_graph()
    relabeled = Pipeline(model=pipeline.model, components=pipeline.components, profile="unrelated-label")
    assert relabeled.digest() == pipeline.digest()
    assert resolve_runtime_composition(relabeled) == contract


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
def test_seeded_selection_round_trips_and_binds_distinct_execution(capacity) -> None:
    full = _pipeline().with_params(linear__capacity=capacity)
    seeded = full.with_params(linear__sharing="seeded")
    restored = Pipeline.from_spec(json.loads(seeded.canonical_bytes()))
    assert restored == seeded
    assert restored.digest() != full.digest()
    selected = resolve_runtime_composition(restored)
    control = resolve_runtime_composition(full)
    assert selected.sharing == "seeded" and control.sharing == "full"
    assert selected.delegation_capacity == capacity
    assert selected.numeric_profile == control.numeric_profile
    assert selected.local_numeric_profile == control.local_numeric_profile
    assert selected.verification_window == control.verification_window == "request_end"
    assert graph_for_runtime(selected) == graph_for_runtime(control)


def test_registry_declaration_cannot_bypass_native_execution_admission(monkeypatch) -> None:
    from pllm import _native

    experiment = _experiment()
    calls = []

    def unavailable(payload):
        calls.append(payload)
        raise ValueError("native GKM capability unavailable")

    monkeypatch.setattr(_native, "resolve_experiment", unavailable)
    with pytest.raises(ConfigurationError, match="native GKM capability unavailable"):
        experiment.resolve()
    assert calls == [experiment.canonical_bytes()]


@pytest.mark.parametrize("factory,params,match", [
    (OpenWeightVerifiableOutsourcing, {"sharing": "reused_seed"}, "sharing"),
    (OpenWeightVerifiableOutsourcing, {"sharing": True}, "sharing"),
    (OpenWeightVerifiableOutsourcing, {"generation": "greedy_until_eos"}, "fixed_length_greedy"),
    (OpenWeightVerifiableOutsourcing, {"generation": "sampling"}, "fixed_length_greedy"),
    (PublishedLinearVerify, {"acceptance": "streaming"}, "acceptance must be request"),
    (PublishedLinearVerify, {"acceptance": 1}, "acceptance must be request"),
])
def test_unsupported_sharing_generation_and_verification_reject(factory, params, match) -> None:
    with pytest.raises(ConfigurationError, match=match):
        factory(**params)
    defaults = factory().get_params(deep=False)
    with pytest.raises(ConfigurationError, match=match):
        create_component(factory.descriptor.component, defaults | params)


@pytest.mark.parametrize("slot,params,match", [
    ("linear", {"sharing": "reused_seed", "generation": "fixed_length_greedy"}, "sharing"),
    ("linear", {"generation": "fixed_length_greedy"}, "missing fields"),
    ("linear", {"sharing": "full", "generation": "fixed_length_greedy", "seed": "secret"}, "unknown fields"),
    ("verification", {"acceptance": "streaming"}, "acceptance must be request"),
    ("verification", {}, "missing fields"),
    ("topology", {"publisher": "remote"}, "unknown fields"),
])
def test_tampered_serialized_contracts_reject_before_resolution(slot, params, match) -> None:
    spec = _experiment().to_spec()
    spec["pipeline"]["components"][slot]["params"] = params
    with pytest.raises(ConfigurationError, match=match):
        Experiment.from_spec(spec)


def _retained_state() -> ClientLocalKv:
    return ClientLocalKv(retention=KvCacheEviction(
        static_keep=(1, 2), dynamic_keep=(1, 1), share_adjacent_layers=False,
    ))


@pytest.mark.parametrize("slot,component", [
    ("linear", MaskedLinear()),
    ("kernels", Cpu(threads=2)),
    ("kernels", AppleMetal()),
    ("verification", FreivaldsVerify()),
    ("topology", PreparedProviderRoles()),
    ("topology", TwoOnlineOffsetRoles()),
    ("quantization", SymmetricPerRow()),
    ("quantization", PublicPerChannelEqualized(profile_digest="a" * 64)),
    ("preparation", ModelAwareCorrections()),
    ("inventory", PreparedInventory()),
    ("delivery", ClientBundleTransport()),
    ("cache", ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=2)),
    ("state", _retained_state()),
])
def test_incompatible_components_have_no_gkm_runtime_contract(slot, component) -> None:
    pipeline = _pipeline()
    with pytest.raises((ConfigurationError, TypeError)):
        OpenWeightVerifiableCpu(pipeline.model, **{slot: component})
    generic = Pipeline(model=pipeline.model, components=dict(pipeline.components) | {slot: component})
    assert generic.digest() != pipeline.digest()
    assert resolve_runtime_composition(generic) is None


@pytest.mark.parametrize("slot", OpenWeightVerifiableCpu.SLOT_NAMES)
def test_required_slots_cannot_be_omitted_from_generic_composition(slot) -> None:
    pipeline = _pipeline()
    components = {name: item for name, item in pipeline.components.items() if name != slot}
    assert resolve_runtime_composition(Pipeline(model=pipeline.model, components=components)) is None


def test_gkm_role_graph_has_two_authenticated_workers_and_client_owned_setup() -> None:
    graph = two_online_delegation_graph()
    assert graph.roles == (
        Role("client", "trusted_client_publisher"),
        Role("worker_a", "gkm_delegation_provider"),
        Role("worker_b", "gkm_delegation_provider"),
    )
    assert graph.separate_operators == (("worker_a", "worker_b"),)
    assert graph.separation_violations({"client": "customer", "worker_a": "same", "worker_b": "same"}) == graph.separate_operators
    assert graph.separation_violations({"client": "customer", "worker_a": "a", "worker_b": "b"}) == ()
    assert all(channel.payload.startswith("authenticated_") for channel in graph.channels)
    assert all("client" in (channel.source, channel.target) for channel in graph.channels)
    per_worker = [
        {(channel.source == "client", channel.phase, channel.payload) for channel in graph.channels
         if worker in (channel.source, channel.target)}
        for worker in ("worker_a", "worker_b")
    ]
    assert per_worker[0] == per_worker[1]
    assert (True, "offline", "authenticated_publication") in per_worker[0]
    assert (False, "online", "authenticated_linear_reply") in per_worker[0]
    assert (True, "online", "authenticated_window_retirement") in per_worker[0]
    assert TwoOnlineDelegationRoles.describe().artifacts[0]["role"] == "client"
    assert TwoOnlineDelegationRoles.describe().role_eligibility == ("client", "worker_a", "worker_b")
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/role-topology.schema.json").read_text())
    Draft202012Validator(schema).validate(graph.to_spec())
    assert RoleGraph.from_spec(graph.to_spec()).digest() == graph.digest()
    reordered = RoleGraph(tuple(reversed(graph.roles)), tuple(reversed(graph.channels)), (("worker_b", "worker_a"),))
    assert reordered.canonical_bytes() == graph.canonical_bytes()
    assert graph.digest() != two_online_reference_graph().digest()


def test_existing_default_composition_identities_remain_unchanged() -> None:
    model = Model("org/model")
    assert MaskedLinearCpu(model).digest() == "d7183c3584a4186d29d4f5d7d64d864d4413568811ac54a0b5c9f8c7a0392636"
    assert TwoOnlineOffsetCpu(model).digest() == "e8ba06768f2c6389c01fa3a6791c688adeb7bf475511b88675f4a398b1008287"
    for component, identity in (
        (MaskedLinear(), "pllm/masked-linear"),
        (FreivaldsVerify(), "pllm/freivalds-verify/v1"),
        (TwoOnlineOffsetRoles(), "pllm/two-online-offset-workers/v1"),
    ):
        assert component.component == identity
        assert create_component(identity, component.to_spec()["params"]) == component
    assert _pipeline().digest() not in {MaskedLinearCpu(model).digest(), TwoOnlineOffsetCpu(model).digest()}
