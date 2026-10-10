"""Selected contracts hide search inputs and independently reject role substitution."""

from dataclasses import replace
import hashlib
import secrets

import pytest

from pllm.compiler import plan
from pllm.deployment.admission import ExecutionContract, RoleAdmission
from pllm.deployment.network import NetworkError, canonical
from test_network_execution import controllers, setup  # noqa: F401

pytestmark = pytest.mark.rust


def reserve(result, controller, role):
    execution = ExecutionContract.from_result(result, result.snapshot)
    admission = RoleAdmission(role, execution)
    attempt = secrets.token_hex(32)
    binding = hashlib.sha256(b"pllm.execution_attempt.v2\0" + canonical({
        "attempt_id": attempt, "execution_digest": execution.digest,
        "network_digest": controller.network.digest,
    })).hexdigest()
    return {"schema": "pllm.party_reserve.v2", "attempt_id": attempt, "ttl_seconds": 30,
        "role_id": role, "instance_epoch": controller.epoch,
        "descriptor_digest": controller.offer().descriptor_digest, "binding_digest": binding,
        "admission": admission.to_spec()}


def test_projection_excludes_private_policy_and_rejected_alternatives(controllers):
    _, _, result, clock = controllers
    public = ExecutionContract.from_result(result, result.snapshot)
    assert ExecutionContract.from_spec(public.to_spec()).digest == public.digest
    assert len(public.snapshot.offers) == 3 and not public.snapshot.links
    changed = plan(replace(result.request, policy=replace(result.request.policy,
        max_client_weight_bytes=1 << 40, buffer_reserve_bytes=1234567)), snapshot=result.snapshot)
    assert changed.status == "feasible" and changed.digest != result.digest
    assert ExecutionContract.from_result(changed, changed.snapshot).digest == public.digest
    text = public.canonical_bytes()
    assert "policy" not in public.to_spec()
    for private in (b"pllm.planning_policy", b'"cost_evidence":', b'"state_evidence":', b'"artifact_evidence":',
                    b"max_client_weight_bytes", b"1234567", b'"candidates":', b'"rejections":'):
        assert private not in text
    assert public.validate(evaluated_at_ms=clock[0])["schedule_digest"] == public.schedule_digest
    with pytest.raises(NetworkError, match="STALE"):
        public.validate(evaluated_at_ms=public.snapshot.expires_at_ms)


def test_role_contract_acknowledgements_and_cleanup(controllers, setup):
    apps, clients, result, _ = controllers
    controller = apps[0].state.controller
    role = next(row["role_id"] for row in result.native_placement["roles"] if row["party_id"] == "a")
    body = reserve(result, controller, role)
    auth = {"authorization": f"Bearer {setup[1]['PLLM_TEST_A']}"}
    response = clients[0].post("/v1/party/reserve", json=body, headers=auth)
    assert response.status_code == 200, response.text
    expected = RoleAdmission.from_spec(body["admission"]).digest
    assert response.json()["admission_digest"] == expected
    assert response.json()["schema"] == "pllm.party_capacity.v2"
    control = {key: body[key] for key in ("attempt_id", "role_id", "binding_digest", "instance_epoch")}
    armed = clients[0].post("/v1/party/arm", json=control, headers=auth)
    assert armed.status_code == 200 and armed.json()["admission_digest"] == expected
    assert clients[0].post("/v1/party/release", json=control, headers=auth).status_code == 200
    assert not controller._reservations
    assert clients[0].post("/v1/party/reserve", json=body, headers=auth).status_code == 409


@pytest.mark.parametrize("mutation", (
    lambda body: body["admission"].update(role_id="client"),
    lambda body: body["admission"]["execution"].update(source_lock_digest="e" * 64),
    lambda body: body["admission"]["execution"].update(schedule_digest="e" * 64),
    lambda body: body["admission"]["execution"].update(policy={"max_client_cpu_ns": 1}),
    lambda body: body["admission"]["execution"]["roles"].append(["worker_a", "client"]),
    lambda body: body.update(binding_digest="e" * 64),
))
def test_substituted_contract_rejects_before_capacity(controllers, setup, mutation):
    apps, clients, result, _ = controllers
    controller = apps[0].state.controller
    role = next(row["role_id"] for row in result.native_placement["roles"] if row["party_id"] == "a")
    body = reserve(result, controller, role)
    mutation(body)
    response = clients[0].post("/v1/party/reserve", json=body,
        headers={"authorization": f"Bearer {setup[1]['PLLM_TEST_A']}"})
    assert response.status_code == 409, response.text
    assert not controller._reservations and not controller._used_attempts
