"""Topology inspection is bound to installed compositions, not an execution bypass."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.components import create_component, get_component
from pllm.configuration import ComponentRef, ConfigurationError
from pllm.profiles import ClientOnlyCpu, MaskedLinearCpu, VerifiedMaskedLinearCpu
from pllm.roles import (
    Channel, ClientOnlyRoles, PreparedProviderRoles, Role, RoleGraph,
    client_only_reference_graph, two_online_reference_graph,
)

ROOT = Path(__file__).resolve().parents[1]


def _resolved(pipeline: ClientOnlyCpu | MaskedLinearCpu | VerifiedMaskedLinearCpu) -> RoleGraph:
    resolved = Experiment(
        name="topology-check",
        pipeline=pipeline,
        deployment=Deployment.local(root="local://topology-check"),
        budget=ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    ).resolve()
    assert isinstance(resolved.role_graph, RoleGraph)
    return resolved.role_graph


def test_masked_runtime_has_stable_digested_topology_and_declares_separation() -> None:
    graph = _resolved(MaskedLinearCpu(Model.tiny()))
    schema = json.loads((ROOT / "schemas/role-topology.schema.json").read_text())
    Draft202012Validator(schema).validate(graph.to_spec())
    assert graph == RoleGraph.from_spec(graph.to_spec())
    assert graph.digest() == RoleGraph.from_spec(graph.to_spec()).digest()
    assert len(graph.digest()) == 64
    assert {role.id for role in graph.roles} == {"client", "preparation", "inference"}
    assert graph.separate_operators == (("inference", "preparation"),)
    assert Channel("preparation", "inference", "offline", "masked_correction") in graph.channels
    assert Channel("inference", "preparation", "offline", "correction_acknowledgement") in graph.channels
    assert Channel("client", "inference", "online", "session_reservation") in graph.channels
    assert Channel("client", "inference", "online", "masked_stage_input") in graph.channels
    assert graph.separation_violations(
        {"client": "local", "preparation": "local", "inference": "local"}
    ) == graph.separate_operators
    assert not graph.separation_violations(
        {"client": "customer", "preparation": "customer", "inference": "operator-b"}
    )


def test_client_only_comparator_has_no_channels_or_coupled_operator_requirement() -> None:
    graph = client_only_reference_graph()
    schema = json.loads((ROOT / "schemas/role-topology.schema.json").read_text())
    Draft202012Validator(schema).validate(graph.to_spec())
    assert graph == RoleGraph.from_spec(graph.to_spec())
    assert graph.roles == (Role("client", "trusted_client"),)
    assert graph.channels == graph.separate_operators == ()
    assert graph.separation_violations({"client": "local"}) == ()
    assert graph.digest() != _resolved(MaskedLinearCpu(Model.tiny())).digest()
    with pytest.raises(ConfigurationError, match="channels"):
        RoleGraph(roles=graph.roles, channels=(Channel("client", "client", "online", "input"),))
    with pytest.raises(ConfigurationError, match="channels"):
        RoleGraph(roles=(Role("client", "trusted_client"), Role("worker", "provider")), channels=())


def test_client_only_topology_resolves_from_exact_component_contract() -> None:
    pipeline = ClientOnlyCpu(Model.tiny())
    experiment = Experiment(
        name="local-only",
        pipeline=pipeline,
        deployment=Deployment.local(root="local://client-only"),
        budget=ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    )
    resolved = experiment.resolve()
    assert resolved.role_graph == client_only_reference_graph()
    assert resolved.client_runtime == "compiled_client_local_v1"
    assert not resolved.requires_preparation
    assert resolved.composition_digest == pipeline.digest()
    assert Experiment.from_spec(experiment.to_spec()).resolve().role_graph == resolved.role_graph
    assert isinstance(create_component(ClientOnlyRoles.descriptor.component, {}), ClientOnlyRoles)
    masked = MaskedLinearCpu(Model.tiny())
    with pytest.raises(ConfigurationError, match="composition|topology"):
        Experiment(
            name="invalid-local",
            pipeline=Pipeline(
                model=masked.model,
                components={**masked.components, "topology": ClientOnlyRoles()},
            ),
            deployment=experiment.deployment, budget=experiment.budget,
        ).resolve()


def test_two_online_comparator_requires_two_distinct_declared_operators() -> None:
    graph = two_online_reference_graph()
    schema = json.loads((ROOT / "schemas/role-topology.schema.json").read_text())
    Draft202012Validator(schema).validate(graph.to_spec())
    assert RoleGraph.from_spec(graph.to_spec()).digest() == graph.digest()
    assert graph.separation_violations(
        {"client": "customer", "worker_a": "same", "worker_b": "same"}
    ) == (("worker_a", "worker_b"),)
    assert graph.separation_violations(
        {"client": "customer", "worker_a": "operator-a", "worker_b": "operator-b"}
    ) == ()


def test_verified_material_changes_graph_without_parsing_profile_name() -> None:
    baseline = _resolved(MaskedLinearCpu(Model.tiny()))
    verified = _resolved(VerifiedMaskedLinearCpu(Model.tiny()))
    assert verified.digest() != baseline.digest()
    assert Channel("preparation", "client", "offline", "verification_projection") in verified.channels


def test_prepared_topology_is_an_explicit_round_trippable_experiment_option() -> None:
    default = MaskedLinearCpu(Model.tiny())
    selected = Pipeline(
        model=default.model,
        components={**default.components, "topology": PreparedProviderRoles()},
    )
    kwargs = {
        "name": "role-selection",
        "deployment": Deployment.local(root="local://role-selection"),
        "budget": ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    }
    implicit = Experiment(pipeline=default, **kwargs)
    explicit = Experiment(pipeline=selected, **kwargs)
    assert implicit.pipeline.digest() != explicit.pipeline.digest()
    assert implicit.configuration_digest() != explicit.configuration_digest()
    assert implicit.resolve().role_graph == explicit.resolve().role_graph
    assert Experiment.from_spec(explicit.to_spec()).to_spec() == explicit.to_spec()
    assert Experiment.from_spec(explicit.to_spec()).resolve().composition_digest == selected.digest()
    identity = PreparedProviderRoles.descriptor.component
    assert get_component(identity).category == "pllm/role-topology"
    assert isinstance(create_component(identity, {}), PreparedProviderRoles)


def test_explicit_prepared_roles_compose_with_verification() -> None:
    verified = VerifiedMaskedLinearCpu(Model.tiny())
    experiment = Experiment(
        name="verified-roles",
        pipeline=Pipeline(
            model=verified.model,
            components={**verified.components, "topology": PreparedProviderRoles()},
        ),
        deployment=Deployment.local(root="local://verified-roles"),
        budget=ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    )
    resolved = experiment.resolve()
    assert resolved.verification_component == "pllm/freivalds-verify/v1"
    assert Channel("preparation", "client", "offline", "verification_projection") in resolved.role_graph.channels


@pytest.mark.parametrize(
    "component",
    [
        ComponentRef("pllm/two-online-workers/v1"),
        ComponentRef(PreparedProviderRoles.descriptor.component, {"operator": "same-host"}),
    ],
)
def test_unimplemented_or_tampered_topology_cannot_activate_baseline(component: ComponentRef) -> None:
    default = MaskedLinearCpu(Model.tiny())
    with pytest.raises((ConfigurationError, ValueError), match="topology|composition|component params"):
        attempt = Experiment(
            name="unreviewed-topology",
            pipeline=Pipeline(model=default.model, components={**default.components, "topology": component}),
            deployment=Deployment.local(root="local://unreviewed-topology"),
            budget=ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
        )
        attempt.resolve()


def test_role_graph_is_deeply_immutable_and_order_independent() -> None:
    baseline = _resolved(MaskedLinearCpu(Model.tiny()))
    graph = RoleGraph(
        roles=tuple(reversed(baseline.roles)),
        channels=tuple(reversed(baseline.channels)),
        separate_operators=(("preparation", "inference"),),
    )
    assert graph == baseline
    assert graph.canonical_bytes() == baseline.canonical_bytes()
    with pytest.raises(dataclasses.FrozenInstanceError):
        graph.roles[0].id = "modified"
    with pytest.raises(dataclasses.FrozenInstanceError):
        graph.channels[0].phase = "online"


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda doc: doc.update(schema="pllm.role_topology.v2"), "unsupported"),
        (lambda doc: doc["roles"].append(doc["roles"][0]), "distinct role IDs"),
        (lambda doc: doc["channels"].append(doc["channels"][0]), "duplicate"),
        (lambda doc: doc["channels"][0].update(target="not-a-role"), "unbound"),
        (lambda doc: doc["channels"][0].update(phase="later"), "phase"),
        (lambda doc: doc["separate_operators"].append(["client", "not-a-role"]), "separation"),
        (lambda doc: doc.update(api_key="secret"), "exactly"),
    ],
)
def test_role_graph_rejects_malformed_or_unbound_records(mutation, match: str) -> None:
    record = _resolved(MaskedLinearCpu(Model.tiny())).to_spec()
    mutation(record)
    with pytest.raises(ConfigurationError, match=match):
        RoleGraph.from_spec(record)


def test_role_graph_placement_requires_exact_nonempty_operators() -> None:
    graph = _resolved(MaskedLinearCpu(Model.tiny()))
    with pytest.raises(ConfigurationError, match="exactly once"):
        graph.separation_violations({"client": "a", "inference": "b"})
    with pytest.raises(ConfigurationError, match="exactly once"):
        graph.separation_violations({"client": "a", "inference": "b", "preparation": ""})


def test_empty_or_unsupported_graph_cannot_authorize_execution() -> None:
    with pytest.raises(ConfigurationError, match="one-client topology has no channels"):
        RoleGraph(roles=(Role("client", "trusted_client"),), channels=(
            Channel("client", "inference", "online", "masked_stage_input"),
        ))
    with pytest.raises(ConfigurationError, match="unbound"):
        RoleGraph(
            roles=(Role("client", "trusted_client"), Role("inference", "provider")),
            channels=(Channel("client", "missing", "online", "masked_stage_input"),),
        )
