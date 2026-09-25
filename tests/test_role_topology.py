"""Topology inspection is bound to installed compositions, not an execution bypass."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.configuration import ConfigurationError
from pllm.profiles import MaskedLinearCpu, VerifiedMaskedLinearCpu
from pllm.roles import Channel, Role, RoleGraph

ROOT = Path(__file__).resolve().parents[1]


def _resolved(pipeline: MaskedLinearCpu | VerifiedMaskedLinearCpu) -> RoleGraph:
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


def test_verified_material_changes_graph_without_parsing_profile_name() -> None:
    baseline = _resolved(MaskedLinearCpu(Model.tiny()))
    verified = _resolved(VerifiedMaskedLinearCpu(Model.tiny()))
    assert verified.digest() != baseline.digest()
    assert Channel("preparation", "client", "offline", "verification_projection") in verified.channels


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
    with pytest.raises(ConfigurationError, match="2 to 16 roles"):
        RoleGraph(roles=(Role("client", "trusted_client"),), channels=(
            Channel("client", "inference", "online", "masked_stage_input"),
        ))
    with pytest.raises(ConfigurationError, match="unbound"):
        RoleGraph(
            roles=(Role("client", "trusted_client"), Role("inference", "provider")),
            channels=(Channel("client", "missing", "online", "masked_stage_input"),),
        )
