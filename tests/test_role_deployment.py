"""Role placement is inspectable, but is never self-authenticating evidence."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from pllm import Model
from pllm.configuration import ConfigurationError
from pllm.deployment import AttestationPolicy, RoleDeployment, RolePlacement
from pllm.roles import client_only_reference_graph, two_online_reference_graph
from pllm.roles.topology import graph_for_runtime
from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition

ROOT = Path(__file__).resolve().parents[1]


def _prepared_graph():
    composition = resolve_runtime_composition(MaskedLinearCpu(Model.tiny()))
    assert composition is not None
    return graph_for_runtime(composition)


def _placement(graph, *, inference_origin: str = "https://inference.example.invalid"):
    return RoleDeployment(
        graph_digest=graph.digest(),
        roles=(
            RolePlacement("inference", "operator-b", inference_origin),
            RolePlacement("client", "customer", None),
            RolePlacement("preparation", "customer", "https://preparation.example.invalid"),
        ),
    )


def test_prepared_role_deployment_round_trips_and_reports_only_declared_separation() -> None:
    graph = _prepared_graph()
    deployment = _placement(graph)
    schema = json.loads((ROOT / "schemas/role-deployment.schema.json").read_text())
    Draft202012Validator(schema).validate(deployment.to_spec())
    assert RoleDeployment.from_spec(deployment.to_spec()) == deployment
    assert RoleDeployment.from_file(ROOT / "examples/prepared-role-deployment.json") == deployment
    assert deployment.digest() == RoleDeployment.from_spec(deployment.to_spec()).digest()
    assert deployment.roles[0].role_id == "client"
    assessment = deployment.assess(graph)
    assert assessment.declared_separation_violations == ()
    assert assessment.roles_requiring_unverified_attestation == ()
    assert assessment.operator_independence_verified is False
    assert assessment.runtime_admission_supported is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        deployment.roles[0].operator = "forged"


def test_operator_aliases_and_shared_loopback_cannot_fake_separation() -> None:
    graph = _prepared_graph()
    collocated = RoleDeployment(
        graph_digest=graph.digest(),
        roles=(
            RolePlacement("client", "customer", None),
            RolePlacement("preparation", "same", "http://localhost:9001"),
            RolePlacement("inference", "same", "http://127.0.0.1:9002"),
        ),
    )
    assert collocated.assess(graph).declared_separation_violations == (
        ("inference", "preparation"),
    )
    different_claim_same_machine = RoleDeployment.from_spec({
        **collocated.to_spec(),
        "roles": [
            {**role, "operator": "operator-b"} if role["role_id"] == "inference" else role
            for role in collocated.to_spec()["roles"]
        ],
    })
    assert different_claim_same_machine.assess(graph).declared_separation_violations == (
        ("inference", "preparation"),
    )
    with pytest.raises(ConfigurationError, match="graph digest"):
        deployment = RoleDeployment(graph_digest="1" * 64, roles=collocated.roles)
        deployment.assess(graph)


def test_research_reference_placement_cannot_activate_a_new_topology() -> None:
    client = client_only_reference_graph()
    standalone = RoleDeployment(
        graph_digest=client.digest(), roles=(RolePlacement("client", "customer", None),),
    )
    schema = json.loads((ROOT / "schemas/role-deployment.schema.json").read_text())
    Draft202012Validator(schema).validate(standalone.to_spec())
    assert standalone.assess(client).declared_separation_violations == ()
    assert standalone.assess(client).runtime_admission_supported is False

    two_online = two_online_reference_graph()
    workers = RoleDeployment(
        graph_digest=two_online.digest(),
        roles=(
            RolePlacement("client", "customer", None),
            RolePlacement("worker_a", "operator-a", "https://a.example.invalid"),
            RolePlacement("worker_b", "operator-b", "https://b.example.invalid"),
        ),
    )
    assert workers.assess(two_online).declared_separation_violations == ()
    assert workers.assess(two_online).runtime_admission_supported is False


def test_tee_policy_is_pinned_but_remains_unverified() -> None:
    graph = _prepared_graph()
    policy = AttestationPolicy("amd-sev-snp", "a" * 64, "b" * 64, "c" * 64)
    deployment = RoleDeployment(
        graph_digest=graph.digest(),
        roles=(
            RolePlacement("client", "customer", None),
            RolePlacement("preparation", "customer", "https://prep.example.invalid"),
            RolePlacement(
                "inference", "provider", "https://infer.example.invalid",
                hardware="tee", attestation_policy=policy,
            ),
        ),
    )
    schema = json.loads((ROOT / "schemas/role-deployment.schema.json").read_text())
    Draft202012Validator(schema).validate(deployment.to_spec())
    assert RoleDeployment.from_spec(deployment.to_spec()) == deployment
    assessment = deployment.assess(graph)
    assert assessment.roles_requiring_unverified_attestation == ("inference",)
    assert assessment.attestation_verified is False
    assert assessment.runtime_admission_supported is False
    with pytest.raises(ConfigurationError, match="pinned attestation policy"):
        RolePlacement("inference", "provider", "https://infer.example.invalid", hardware="tee")
    with pytest.raises(ConfigurationError, match="standard hardware"):
        RolePlacement("inference", "provider", "https://infer.example.invalid", attestation_policy=policy)


@pytest.mark.parametrize(
    "origin",
    [
        "http://public.example.invalid", "https://user:secret@provider.invalid",
        "https://provider.invalid/path", "https://provider.invalid/?key=abc",
        "https://provider.invalid/#fragment", "https://provider.invalid:0",
        "https://[broken",
    ],
)
def test_placement_rejects_unsafe_provider_origins(origin: str) -> None:
    with pytest.raises(ConfigurationError, match="origin"):
        RolePlacement("inference", "operator", origin)


def test_role_deployment_rejects_extraneous_role_secret_and_duplicate_fields(tmp_path: Path) -> None:
    graph = _prepared_graph()
    record = _placement(graph).to_spec()
    with pytest.raises(ConfigurationError, match="unknown"):
        RoleDeployment.from_spec({**record, "api_key": "sensitive"})
    with pytest.raises(ConfigurationError, match="unknown"):
        RoleDeployment.from_spec({**record, "roles": [
            {**record["roles"][0], "credential": "sensitive"}, *record["roles"][1:],
        ]})
    with pytest.raises(ConfigurationError, match="unique"):
        RoleDeployment.from_spec({**record, "roles": [record["roles"][0]] * 3})
    with pytest.raises(ConfigurationError, match="exactly once"):
        RoleDeployment.from_spec({**record, "roles": record["roles"][:2]}).assess(graph)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate"):
        RoleDeployment.from_file(duplicate)
    excessive = tmp_path / "excessive.json"
    excessive.write_text(" " * 65_537, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="64 KiB"):
        RoleDeployment.from_file(excessive)


def test_policy_rejects_malformed_digests_and_backend() -> None:
    with pytest.raises(ConfigurationError, match="SHA-256"):
        AttestationPolicy("amd-sev-snp", "z" * 64, "b" * 64, "c" * 64)
    with pytest.raises(ConfigurationError, match="unknown TEE technology"):
        AttestationPolicy("fake-enclave", "a" * 64, "b" * 64, "c" * 64)
    with pytest.raises(ConfigurationError, match="SHA-256"):
        AttestationPolicy("intel-tdx", "0" * 64, "b" * 64, "c" * 64)
