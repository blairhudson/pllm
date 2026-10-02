"""Native placement owns topology, schedule legality, bounds and commitments."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, _native
from pllm.profiles import (
    ClientOnlyCpu,
    MaskedLinearCpu,
    TwoOnlineOffsetCpu,
    VerifiedMaskedLinearCpu,
)

pytestmark = pytest.mark.rust


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


@pytest.fixture(params=[ClientOnlyCpu, MaskedLinearCpu, TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu])
def placement(request):
    experiment = Experiment(
        name="native-network-placement",
        pipeline=request.param(Model.tiny()),
        deployment=Deployment.local(root="local://native-network-placement"),
        budget=ExecutionBudget(requests=1, max_input_tokens=3, max_new_tokens=2),
    ).resolve()
    config = {
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 32, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True,
    }
    plan = _native.lower_model(canonical(config), 1, 3, 2)
    roles = experiment.role_graph.roles
    document = {
        "schema": "pllm.network_placement.v1", "snapshot_digest": "a" * 64,
        "evaluated_at_ms": 100, "client_party_id": "client",
        "roles": [{"role_id": role.id, "party_id": role.id} for role in roles],
        "parties": [{
            "party_id": role.id, "operator_id": role.id, "instance_epoch": "b" * 64,
            "expires_at_ms": 101, "capabilities": [role.capability],
            "max_memory_bytes": 10000, "max_weight_bytes": 10000,
            "max_sessions": 1, "active_sessions": 0, "allowed_models": ["*"],
        } for role in roles],
        "required_memory_bytes": {role.id: 4000 for role in roles},
        "required_weight_bytes": {role.id: 3000 for role in roles},
    }
    return plan, experiment.canonical_composition, document, experiment.role_graph


def validate(placement, document=None):
    plan, composition, original, _ = placement
    return _native.validate_network_placement(
        plan, composition, canonical(original if document is None else document)
    )


def test_native_matches_installed_graph_and_commits_canonical_output(placement):
    plan, composition, document, graph = placement
    output = validate(placement)
    result = json.loads(output)
    assert type(output) is bytes
    assert canonical(result) == output
    assert {role["role_id"] for role in result["roles"]} == {role.id for role in graph.roles}
    assert {key: result[key] for key in document} == document
    _, schedule_digest = _native.decoder_runtime_schedule(plan, composition)
    assert result["schedule_digest"] == schedule_digest
    assert result["model_plan_digest"] == hashlib.sha256(
        b"pllm.decoder_plan.v1\0" + plan
    ).hexdigest()
    assert result["composition_digest"] == hashlib.sha256(
        b"pllm.pipeline.v2\0" + composition
    ).hexdigest()
    digest = result.pop("placement_digest")
    assert digest == hashlib.sha256(
        b"pllm.network_placement.validated.v1\0" + canonical(result)
    ).hexdigest()
    changed = copy.deepcopy(document)
    changed["roles"].reverse()
    changed["parties"].reverse()
    assert validate(placement, changed) == output
    changed["snapshot_digest"] = "c" * 64
    assert validate(placement, changed) != output


def test_native_separation_requires_declared_operator_difference(placement):
    _, _, document, graph = placement
    for left, right in graph.separate_operators:
        changed = copy.deepcopy(document)
        for party in changed["parties"]:
            if party["party_id"] in (left, right):
                party["operator_id"] = "same-operator"
        with pytest.raises(ValueError, match="separate declared operators"):
            validate(placement, changed)


@pytest.mark.parametrize("field,value", [
    ("instance_epoch", "0" * 64), ("instance_epoch", "A" * 64),
    ("expires_at_ms", 100), ("expires_at_ms", True),
    ("max_memory_bytes", 2**64), ("max_weight_bytes", -1),
    ("max_sessions", 2**32), ("active_sessions", True),
    ("capabilities", ["protected_provider"]), ("allowed_models", ["tiny"]),
    ("unexpected", 1),
])
def test_native_rejects_strict_offer_violations(placement, field, value):
    changed = copy.deepcopy(placement[2])
    changed["parties"][0][field] = value
    with pytest.raises(ValueError):
        validate(placement, changed)


def test_native_checks_exact_model_allowlist_and_capacity(placement):
    plan = json.loads(placement[0])
    changed = copy.deepcopy(placement[2])
    for party in changed["parties"]:
        party["allowed_models"] = [plan["config_digest"]]
    validate(placement, changed)
    changed["parties"][0]["allowed_models"] = ["f" * 64]
    with pytest.raises(ValueError, match="not allowed"):
        validate(placement, changed)
    for field, value in [("max_memory_bytes", 3999), ("max_weight_bytes", 2999),
                         ("active_sessions", 1)]:
        changed = copy.deepcopy(placement[2])
        changed["parties"][0][field] = value
        with pytest.raises(ValueError, match="capacity exceeded"):
            validate(placement, changed)


def test_native_rejects_role_injection_unknown_party_and_missing_resource(placement):
    for action in ("extra-role", "unknown-party", "missing-memory", "client-moved"):
        changed = copy.deepcopy(placement[2])
        if action == "extra-role":
            changed["roles"].append({"role_id": "invented", "party_id": "client"})
        elif action == "unknown-party":
            changed["client_party_id"] = "missing"
            changed["roles"][0]["party_id"] = "missing"
        elif action == "missing-memory":
            del changed["required_memory_bytes"]["client"]
        else:
            changed["client_party_id"] = "different"
        with pytest.raises(ValueError):
            validate(placement, changed)


def test_native_rejects_duplicate_keys_and_bounded_input(placement):
    plan, composition, document, _ = placement
    raw = canonical(document)
    invalid = [
        raw.replace(b'"evaluated_at_ms":100', b'"evaluated_at_ms":100,"evaluated_at_ms":100'),
        raw.replace(b'"client":4000', b'"client":4000,"client":4000'),
        raw.replace(b'"client":4000', b'"client":true'),
        b" " * (1024 * 1024 + 1),
    ]
    for candidate in invalid:
        with pytest.raises(ValueError):
            _native.validate_network_placement(plan, composition, candidate)
    for field, count in [("roles", 17), ("parties", 65)]:
        changed = copy.deepcopy(document)
        changed[field] = [changed[field][0]] * count
        with pytest.raises(ValueError, match="record bound"):
            validate(placement, changed)
    with pytest.raises(TypeError):
        _native.validate_network_placement(plan, composition, bytearray(raw))


def test_native_does_not_admit_pending_composition_or_incomplete_plan(placement):
    plan, composition, document, _ = placement
    changed = json.loads(composition)
    changed["components"]["nonlinear"] = {"component": "pllm/pending/v1", "params": {}}
    with pytest.raises(ValueError):
        _native.validate_network_placement(plan, canonical(changed), canonical(document))
    changed = json.loads(plan)
    changed["token_feedback"] = False
    with pytest.raises(ValueError, match="token feedback"):
        _native.validate_network_placement(canonical(changed), composition, canonical(document))


def test_native_owned_floors_are_exact_and_allow_exact_capacity(placement):
    _, _, document, graph = placement
    role_ids = {role.id for role in graph.roles}
    if role_ids == {"client"}:
        expected = {"client": 1480}
    elif "worker_a" in role_ids:
        expected = {"client": 296, "worker_a": 1184, "worker_b": 1184}
    else:
        expected = {"client": 296, "inference": 1184, "preparation": 1184}
    # Hand-computed fixture oracle: 1152 body matrix elements + 32 biases;
    # 40 local norm elements + one 256-element tied embedding/head artifact.
    result = json.loads(validate(placement))
    assert result["minimum_weight_bytes"] == expected
    assert result["minimum_memory_bytes"] == expected
    exact = copy.deepcopy(document)
    exact["required_weight_bytes"] = expected.copy()
    exact["required_memory_bytes"] = expected.copy()
    for offer in exact["parties"]:
        floor = expected[offer["party_id"]]
        offer["max_weight_bytes"] = floor
        offer["max_memory_bytes"] = floor
    validate(placement, exact)
    for field in ("required_weight_bytes", "required_memory_bytes"):
        for role, floor in expected.items():
            for underreported in (0, floor - 1):
                changed = copy.deepcopy(exact)
                changed[field][role] = underreported
                with pytest.raises(ValueError, match="below native minimum"):
                    validate(placement, changed)
    for field in ("max_weight_bytes", "max_memory_bytes"):
        changed = copy.deepcopy(exact)
        changed["parties"][0][field] -= 1
        with pytest.raises(ValueError, match="capacity exceeded"):
            validate(placement, changed)


def test_zero_estimates_cannot_make_large_model_fit_tiny_offers(placement):
    config = {
        "model_type": "qwen2", "hidden_size": 256, "intermediate_size": 512,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "vocab_size": 1024, "max_position_embeddings": 32, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True,
    }
    plan = _native.lower_model(canonical(config), 1, 3, 2)
    _, composition, document, _ = placement
    changed = copy.deepcopy(document)
    for field in ("required_weight_bytes", "required_memory_bytes"):
        changed[field] = dict.fromkeys(changed[field], 0)
    for offer in changed["parties"]:
        offer["max_weight_bytes"] = 1
        offer["max_memory_bytes"] = 1
    with pytest.raises(ValueError, match="below native minimum"):
        _native.validate_network_placement(plan, composition, canonical(changed))


def test_floor_fields_cannot_be_supplied_as_request_authority(placement):
    changed = copy.deepcopy(placement[2])
    changed["minimum_weight_bytes"] = dict.fromkeys(changed["required_weight_bytes"], 0)
    changed["minimum_memory_bytes"] = dict.fromkeys(changed["required_memory_bytes"], 0)
    with pytest.raises(ValueError, match="unknown field"):
        validate(placement, changed)


def test_assignment_free_requirements_match_native_validation_and_installed_graph(placement):
    plan, composition, _, graph = placement
    output = _native.network_placement_requirements(plan, composition)
    requirements = json.loads(output)
    assert type(output) is bytes
    assert canonical(requirements) == output
    assert set(requirements) == {
        "schema", "roles", "separate_operators", "minimum_weight_bytes",
        "minimum_memory_bytes", "schedule_digest", "model_plan_digest", "composition_digest",
    }
    assert requirements["schema"] == "pllm.network_placement_requirements.v1"
    assert requirements["roles"] == [
        {"role_id": role.id, "capability": role.capability} for role in graph.roles
    ]
    assert requirements["separate_operators"] == [list(pair) for pair in graph.separate_operators]
    validated = json.loads(validate(placement))
    for field in ("minimum_weight_bytes", "minimum_memory_bytes", "schedule_digest",
                  "model_plan_digest", "composition_digest"):
        assert requirements[field] == validated[field]
    assert _native.network_placement_requirements(plan, composition) == output


def test_assignment_free_requirements_reject_pending_and_invalid_models(placement):
    plan, composition, document, _ = placement
    malformed_pairs = []
    changed = json.loads(composition)
    changed["components"]["nonlinear"] = {"component": "pllm/pending/v1", "params": {}}
    malformed_pairs.append((plan, canonical(changed)))
    changed = json.loads(composition)
    changed["model"]["kind"] = "unknown"
    malformed_pairs.append((plan, canonical(changed)))
    for field, value in [("schema_version", "pllm.decoder_plan.pending"),
                         ("token_feedback", False), ("config_digest", "0" * 64)]:
        changed = json.loads(plan)
        changed[field] = value
        malformed_pairs.append((canonical(changed), composition))
    malformed_pairs.extend([(b"{}", composition), (plan, b"{}"), (b"invalid", composition)])
    for model, pipeline in malformed_pairs:
        with pytest.raises(ValueError) as inspection:
            _native.network_placement_requirements(model, pipeline)
        with pytest.raises(ValueError) as admission:
            _native.validate_network_placement(model, pipeline, canonical(document))
        assert str(inspection.value) == str(admission.value)


def test_assignment_free_requirements_keep_immutable_bytes_and_input_bounds(placement):
    plan, composition, _, _ = placement
    for model, pipeline in [(bytearray(plan), composition), (plan, bytearray(composition))]:
        with pytest.raises(TypeError):
            _native.network_placement_requirements(model, pipeline)
    with pytest.raises(ValueError, match="invalid byte length"):
        _native.network_placement_requirements(b" " * (16 * 1024 * 1024 + 1), composition)
    with pytest.raises(ValueError, match="invalid byte length"):
        _native.network_placement_requirements(plan, b" " * (1024 * 1024 + 1))
