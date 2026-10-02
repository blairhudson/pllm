"""Slice B: real HTTP selected execution and fail-closed capacity lifecycle.

Loopback hosts share one machine/owner. Operator declarations are not privacy
deployment evidence or physical independence attestation.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import pickle
import secrets
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from pllm import Deployment, ExecutionBudget, Experiment, Model, OpenAI
from pllm.compiler import plan
from pllm.deployment import (
    LivePartyOffer, NetworkError, NetworkSnapshot, NetworkSpec, PartyOffer, PartySpec,
    PartyTrust, async_discover, async_open_execution, discover, open_execution,
)
from pllm.deployment.network import canonical, control_request
from pllm.model_loader import resolve_model
from pllm.modeling import lower_model
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.runtime.party import create_party_app
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.search import PlanningPolicy, PlanningRequest

pytestmark = pytest.mark.rust


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.fixture
def setup(tmp_path):
    checkpoint = create_tiny_llama_checkpoint(tmp_path / "checkpoint", hidden_size=8,
        intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4)
    model = Model.path(str(checkpoint), model_id="network-tiny")
    budget = ExecutionBudget(1, 32, 2)
    installed = Experiment("installed", TwoOnlineOffsetCpu(model), Deployment.local(root="local://installed"), budget)
    source_lock = resolve_model(model).source_lock_digest
    model_plan = lower_model(json.loads((checkpoint / "config.json").read_bytes()), batch=1,
                             max_input_tokens=32, max_new_tokens=2)
    now = time.time_ns() // 1_000_000
    client = PartyOffer("client", "client", "c" * 64, now + 300_000, ("trusted_client",),
                        64 << 20, 64 << 20, 8, 0, ("*",), ("*",))
    roots = tuple(PartyTrust(party, f"operator-{party}", f"http://127.0.0.1:{free_port()}", f"PLLM_TEST_{party.upper()}")
                  for party in ("a", "b"))
    network = NetworkSpec("test", NetworkSnapshot("test", (client,), (), "local-client", now, now + 300_000),
                          "http", ("client", "operator-a", "operator-b"), roots)
    credentials = {root.credential_env: secrets.token_urlsafe(32) for root in roots}
    specs = tuple(PartySpec(root.party_id, installed, source_lock, ("worker_a", "worker_b"),
                           64 << 20, 64 << 20, peer_party_ids=tuple(r.party_id for r in roots))
                  for root in roots)
    return network, credentials, specs, model_plan, source_lock


def selected(setup, snapshot):
    network, _, specs, model_plan, lock = setup
    experiment = replace(specs[0].experiment, name="network-offset", deployment=Deployment.network(
        network_id=network.network_id, network_spec_digest=network.digest))
    request = PlanningRequest(model_plan, (experiment,), PlanningPolicy(
        "client", time.time_ns() // 1_000_000, 4, 32, ("client_weight_bytes",),
        minimum_remote_mac_fraction=0.5, max_observation_age_ms=300_000), source_lock_digest=lock)
    result = plan(request, snapshot=snapshot)
    assert result.status == "feasible", result.to_spec()["rejections"]
    return result


@contextmanager
def http_hosts(setup, tmp_path):
    network, credentials, specs, _, _ = setup
    network_path = tmp_path / "network.json"
    network_path.write_bytes(network.canonical_bytes())
    children = []
    try:
        for spec, root in zip(specs, network.parties, strict=True):
            party_path = tmp_path / f"{spec.party_id}.json"
            party_path.write_bytes(spec.canonical_bytes())
            child = subprocess.Popen([sys.executable, "-m", "pllm", "serve", "party",
                "--network", str(network_path), "--party", str(party_path), "--port", root.origin.rsplit(":", 1)[1]],
                env={**os.environ, **credentials}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            children.append(child)
        deadline = time.monotonic() + 60
        for child, root in zip(children, network.parties, strict=True):
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    pytest.fail(child.stderr.read().decode())
                try:
                    response = httpx.get(root.origin + "/healthz", timeout=0.5)
                    if response.status_code == 200 and response.json()["ready"]:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            else:
                pytest.fail("party HTTP startup timeout")
        yield network, credentials, children
    finally:
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            child.stderr.close()


def reserve_body(result, controller, role="worker_a", ttl=5, attempt=None):
    attempt = secrets.token_hex(32) if attempt is None else attempt
    binding = hashlib.sha256(b"pllm.execution_attempt.v1\0" + canonical({
        "attempt_id": attempt, "result_digest": result.digest, "network_digest": controller.network.digest,
    })).hexdigest()
    return {"schema": "pllm.party_reserve.v1", "attempt_id": attempt, "ttl_seconds": ttl,
            "role_id": role, "instance_epoch": controller.epoch,
            "descriptor_digest": controller.offer().descriptor_digest, "binding_digest": binding,
            "result": result.to_spec()}


@pytest.fixture
def controllers(setup):
    network, credentials, specs, _, _ = setup
    clock = [time.time_ns() // 1_000_000]
    apps = [create_party_app(spec, network, credentials=credentials, clock=lambda: clock[0]) for spec in specs]
    with TestClient(apps[0]) as first, TestClient(apps[1]) as second:
        offers = tuple(app.state.controller.offer() for app in apps)
        snap = NetworkSnapshot("test", (*network.snapshot.offers, *offers), (), "authenticated-test", clock[0], clock[0] + 60_000)
        result = selected(setup, snap)
        yield apps, (first, second), result, clock


def test_controller_auth_provisional_arm_release_capacity_and_fresh_attempt(controllers, setup):
    apps, clients, result, clock = controllers
    assignments = {row["role_id"]: row["party_id"] for row in result.native_placement["roles"]}
    role = next(role for role, party in assignments.items() if party == "a")
    controller, client = apps[0].state.controller, clients[0]
    auth = {"authorization": f"Bearer {setup[1]['PLLM_TEST_A']}"}
    assert client.get("/v1/party/offer").status_code == 401
    body = reserve_body(result, controller, role=role)
    response = client.post("/v1/party/reserve", json=body, headers=auth)
    assert response.status_code == 200, response.text
    token = response.json()["capacity_token"]
    role_path = f"/roles/{role}/v1/runtime/models/network-tiny"
    capacity_auth = {"authorization": f"Bearer {token}"}
    assert client.get(role_path, headers=capacity_auth).status_code == 401
    assert client.post("/v1/party/reserve", json=reserve_body(result, controller, role=role), headers=auth).status_code == 409
    control = {key: body[key] for key in ("attempt_id", "role_id", "binding_digest", "instance_epoch")}
    assert client.post("/v1/party/arm", json=control, headers=auth).status_code == 200
    assert client.get(role_path, headers=capacity_auth).status_code == 200
    assert client.get(role_path, headers={"authorization": f"Bearer {controller._static_keys[role]}"}).status_code == 401
    assert client.post("/v1/party/release", json=control, headers=auth).status_code == 200
    assert not controller._reservations
    assert client.get(role_path, headers=capacity_auth).status_code == 401
    assert client.post("/v1/party/reserve", json=body, headers=auth).status_code == 409
    fresh = reserve_body(result, controller, role=role)
    assert client.post("/v1/party/reserve", json=fresh, headers=auth).status_code == 200
    clock[0] += 5000
    client.get("/v1/party/offer", headers=auth)
    assert not controller._reservations


@pytest.mark.parametrize("mutation", [
    lambda body: body.update(instance_epoch="f" * 64),
    lambda body: body.update(descriptor_digest="f" * 64),
    lambda body: body.update(ttl_seconds=4),
    lambda body: body.update(ttl_seconds=301),
    lambda body: body.update(ttl_seconds=True),
    lambda body: body.update(role_id="inference"),
    lambda body: body["result"]["request"].update(source_lock_digest="e" * 64),
    lambda body: body["result"]["request"]["candidates"][0]["pipeline"]["components"]["kernels"].update(component="pllm/forged/v1"),
])
def test_stale_forged_source_workload_rejected_without_reservation(controllers, setup, mutation):
    apps, clients, result, _ = controllers
    controller = apps[0].state.controller
    body = reserve_body(result, controller)
    mutation(body)
    response = clients[0].post("/v1/party/reserve", json=body,
                              headers={"authorization": f"Bearer {setup[1]['PLLM_TEST_A']}"})
    assert response.status_code == 409
    assert not controller._reservations


def test_drain_rejects_new_capacity_leaves_existing_armed_work(controllers, setup):
    apps, clients, result, _ = controllers
    controller, client = apps[0].state.controller, clients[0]
    role = next(row["role_id"] for row in result.native_placement["roles"] if row["party_id"] == "a")
    auth = {"authorization": f"Bearer {setup[1]['PLLM_TEST_A']}"}
    body = reserve_body(result, controller, role=role)
    assert client.post("/v1/party/reserve", json=body, headers=auth).status_code == 200
    control = {key: body[key] for key in ("attempt_id", "role_id", "binding_digest", "instance_epoch")}
    assert client.post("/v1/party/arm", json=control, headers=auth).status_code == 200
    assert client.post("/v1/party/drain", json={}, headers=auth).status_code == 200
    assert not client.get("/v1/party/offer", headers=auth).json()["accepting"]
    assert client.post("/v1/party/leave", json={}, headers=auth).json()["status"] == "draining"
    assert client.post("/v1/party/reserve", json=reserve_body(result, controller, role=role), headers=auth).status_code == 409
    assert client.post("/v1/party/release", json=control, headers=auth).status_code == 200
    assert client.post("/v1/party/leave", json={}, headers=auth).json()["status"] == "left"


@pytest.mark.integration
def test_real_two_party_http_selected_sdk_generated_request_cancel_and_secret_isolation(setup, tmp_path):
    with http_hosts(setup, tmp_path) as (network, credentials, _):
        snapshot = discover(network, credentials=credentials)
        assert all(offer.authenticated for offer in snapshot.offers if isinstance(offer, LivePartyOffer))
        assert not any(getattr(offer, "independence_verified", False) for offer in snapshot.offers)
        result = selected(setup, snapshot)
        with open_execution(result, network=network, credentials=credentials, ttl_seconds=30) as lease:
            with pytest.raises(TypeError):
                copy.copy(lease)
            with pytest.raises(TypeError):
                pickle.dumps(lease)
            binding = json.dumps(lease.binding.to_spec())
            for secret in credentials.values():
                assert secret not in binding and secret.encode() not in result.canonical_bytes()
            with OpenAI(execution=lease) as client:
                with pytest.raises(NetworkError, match="output cap"):
                    client.responses.create(input="Hi", max_output_tokens=3)
                response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0)
                assert response.usage.output_tokens == 2
                assert client.privacy_audit.inference_stage_calls > 0
        assert lease.closed
        result = selected(setup, discover(network, credentials=credentials))
        with open_execution(result, network=network, credentials=credentials) as lease:
            with OpenAI(execution=lease) as client:
                stream = client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0, stream=True)
                next(stream)
                for event in stream:
                    if event.type == "response.output_text.delta":
                        break
                transports = tuple(client._core._active_offset_transports.values())
                assert transports
                transport = transports[0]
                sessions = transport._sessions
                stream.close()
                assert not client._core._active_offset_transports
                for index, (http, key, session) in enumerate(zip(
                    transport._clients, transport._keys, sessions, strict=True
                )):
                    response = http.post(f"/v1/offset-reference/sessions/{session}/complete",
                                         headers={"authorization": f"Bearer {key}"})
                    assert response.status_code == 409  # Existing cancel made session terminal.
        assert all(offer.active_sessions == 0 for offer in discover(network, credentials=credentials).offers)


@pytest.mark.integration
def test_partial_arm_ambiguity_releases_every_party_and_retry_fresh(setup, tmp_path, monkeypatch):
    from pllm.deployment import execution

    with http_hosts(setup, tmp_path) as (network, credentials, _):
        result = selected(setup, discover(network, credentials=credentials))
        actual = execution.control_request
        attempts = []
        def ambiguous(origin, env, path, **kwargs):
            response = actual(origin, env, path, **kwargs)
            if path.endswith("/reserve"):
                attempts.append(response["attempt_id"])
            if path.endswith("/arm") and env == "PLLM_TEST_B":
                raise NetworkError("lost arm acknowledgement")
            return response
        monkeypatch.setattr(execution, "control_request", ambiguous)
        with pytest.raises(NetworkError, match="lost arm"):
            open_execution(result, network=network, credentials=credentials)
        assert all(offer.active_sessions == 0 for offer in discover(network, credentials=credentials).offers)
        monkeypatch.setattr(execution, "control_request", actual)
        with open_execution(result, network=network, credentials=credentials) as lease:
            assert lease.binding.attempt_id not in attempts
            with OpenAI(execution=lease) as client:
                assert client.responses.create(input="Hi", temperature=0.0).usage.output_tokens == 2


def test_same_operator_and_forbidden_reassignment_rejected(controllers, setup):
    _, _, result, _ = controllers
    snapshot = result.snapshot
    same = replace(snapshot, offers=tuple(replace(offer, operator_id="shared")
        if isinstance(offer, LivePartyOffer) else offer for offer in snapshot.offers))
    rejected = plan(result.request, snapshot=same)
    assert rejected.status == "infeasible"
    reassigned = result.to_spec()
    rows = reassigned["selection"]["native_placement"]["roles"]
    for row in rows:
        if row["role_id"] == "worker_a":
            row["party_id"] = "client"
    from pllm.plan import PlanningResult
    with pytest.raises(NetworkError):
        PlanningResult.from_spec(reassigned)


def test_controller_ttl_burns_existing_session_and_revokes_bearer(controllers, setup):
    from pllm.roles import two_online_reference_graph

    apps, clients, result, clock = controllers
    controller, client = apps[0].state.controller, clients[0]
    role = next(row["role_id"] for row in result.native_placement["roles"] if row["party_id"] == "a")
    auth = {"authorization": f"Bearer {setup[1]['PLLM_TEST_A']}"}
    body = reserve_body(result, controller, role=role)
    response = client.post("/v1/party/reserve", json=body, headers=auth)
    token = response.json()["capacity_token"]
    capacity_auth = {"authorization": f"Bearer {token}"}
    control = {key: body[key] for key in ("attempt_id", "role_id", "binding_digest", "instance_epoch")}
    assert client.post("/v1/party/arm", json=control, headers=auth).status_code == 200
    base = f"/roles/{role}"
    descriptor = client.get(base + "/v1/runtime/models/network-tiny", headers=capacity_auth).json()["runtime"]
    session_body = {"schema": "pllm.offset_worker_session.v1", "model": "network-tiny", "role": role,
                    "topology_digest": two_online_reference_graph().digest(),
                    "decoder_plan": result.request.model_plan.digest,
                    "max_input_tokens": 32, "max_new_tokens": 2,
                    "body_fingerprint": descriptor["body_fingerprint"],
                    "stage_commitment": descriptor["seeded_stage_commitment"],
                    "runtime_config_digest": descriptor["runtime_config_digest"],
                    "composition_digest": result.experiment.pipeline.digest()}
    session = client.post(base + "/v1/offset-reference/sessions", json=session_body, headers=capacity_auth)
    assert session.status_code == 200, session.text
    row = next(iter(controller._reservations.values()))
    assert row.sessions == {session.json()["id"]}
    assert client.post(base + "/v1/offset-reference/sessions", json=session_body, headers=capacity_auth).status_code == 409
    assert client.post(base + "/v1/offset-reference/sessions/" + "f" * 32 + "/cancel", headers=capacity_auth).status_code == 403
    clock[0] += 5000
    assert client.get(base + "/v1/runtime/models/network-tiny", headers=capacity_auth).status_code == 401
    client.get("/v1/party/offer", headers=auth)  # Authoritative purge runs without client cooperation.
    assert row.revoked and not row.sessions and row.credential == ""
    assert not controller._reservations


def test_directory_auth_registration_renew_withdraw_and_expiry_pull_installed_offers(controllers, setup, monkeypatch):
    from pllm.runtime import directory

    apps, _, _, clock = controllers
    network, credentials, *_ = setup
    configured = replace(network, directory_origin=f"http://127.0.0.1:{free_port()}",
                         directory_credential_env="PLLM_TEST_DIRECTORY")
    credentials = {**credentials, "PLLM_TEST_DIRECTORY": secrets.token_urlsafe(32)}
    calls = []
    def pull(origin, env, path, **kwargs):
        calls.append((origin, path))
        return apps[0].state.controller.offer().to_spec()
    monkeypatch.setattr(directory, "control_request", pull)
    app = directory.create_directory_app(configured, credentials=credentials, clock=lambda: clock[0])
    with TestClient(app) as client:
        auth = {"authorization": f"Bearer {credentials['PLLM_TEST_A']}"}
        directory_auth = {"authorization": f"Bearer {credentials['PLLM_TEST_DIRECTORY']}"}
        assert client.get("/v1/directory/offers").status_code == 401
        assert client.post("/v1/directory/register", json={"party_id": "a"}).status_code == 401
        forged = {"party_id": "a", "offer": apps[0].state.controller.offer().to_spec()}
        assert client.post("/v1/directory/register", json=forged, headers=auth).status_code == 409
        assert not calls
        for verb in ("register", "renew"):
            assert client.post(f"/v1/directory/{verb}", json={"party_id": "a"}, headers=auth).status_code == 200
        assert calls == [(network.parties[0].origin, "/v1/party/offer")] * 2
        offers = client.get("/v1/directory/offers", headers=directory_auth).json()["offers"]
        assert len(offers) == 1 and offers[0]["authenticated"]
        clock[0] += 60_000
        assert client.get("/v1/directory/offers", headers=directory_auth).json()["offers"] == []
        assert client.post("/v1/directory/renew", json={"party_id": "a"}, headers=auth).status_code == 200
        assert client.post("/v1/directory/withdraw", json={"party_id": "a"}, headers=auth).status_code == 200
        assert client.get("/v1/directory/offers", headers=directory_auth).json()["offers"] == []
        monkeypatch.setattr(directory, "control_request", lambda *args, **kwargs: {**offers[0], "operator_id": "forged"})
        assert client.post("/v1/directory/register", json={"party_id": "a"}, headers=auth).status_code == 409


@pytest.mark.integration
def test_partial_reserve_conflict_releases_first_party(setup, tmp_path, monkeypatch):
    from pllm.deployment import execution

    with http_hosts(setup, tmp_path) as (network, credentials, _):
        result = selected(setup, discover(network, credentials=credentials))
        role = next(row["role_id"] for row in result.native_placement["roles"] if row["party_id"] == "b")
        root = network.parties[1]
        offer = next(offer for offer in result.snapshot.offers if offer.party_id == "b")
        attempt = secrets.token_hex(32)
        binding = hashlib.sha256(b"pllm.execution_attempt.v1\0" + canonical({
            "attempt_id": attempt, "result_digest": result.digest, "network_digest": network.digest,
        })).hexdigest()
        control = {"attempt_id": attempt, "role_id": role, "binding_digest": binding, "instance_epoch": offer.instance_epoch}
        actual = execution.control_request
        reserved = []
        def race(origin, env, path, **kwargs):
            response = actual(origin, env, path, **kwargs)
            if path.endswith("/reserve") and env == "PLLM_TEST_A":
                reserved.append(response["attempt_id"])
                control_request(root.origin, root.credential_env, "/v1/party/reserve", credentials=credentials,
                    body={**control, "schema": "pllm.party_reserve.v1", "ttl_seconds": 30,
                          "descriptor_digest": offer.descriptor_digest, "result": result.to_spec()})
            return response
        monkeypatch.setattr(execution, "control_request", race)
        # Capacity races after discovery and A's reserve. B's competitor remains
        # isolated; failed admission must release A instead of overcommitting B.
        with pytest.raises(NetworkError, match="HTTP 409"):
            open_execution(result, network=network, credentials=credentials)
        assert reserved
        assert discover(network, credentials=credentials).offers[0].active_sessions == 0
        control_request(root.origin, root.credential_env, "/v1/party/release", body=control, credentials=credentials)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_http_selected_client_and_shielded_cleanup(setup, tmp_path):
    from pllm import AsyncOpenAI

    with http_hosts(setup, tmp_path) as (network, credentials, _):
        result = selected(setup, await async_discover(network, credentials=credentials))
        lease = await async_open_execution(result, network=network, credentials=credentials)
        async with lease:
            async with AsyncOpenAI(execution=lease) as client:
                response = await client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0)
                assert response.usage.output_tokens == 2
        assert lease.closed
        assert all(offer.active_sessions == 0 for offer in discover(network, credentials=credentials).offers)


def test_cli_live_dry_run_never_discovers_reserves_or_starts_hosts(setup, tmp_path, monkeypatch, capsys):
    from pllm._cli import network as cli
    from pllm._cli.app import main

    network, credentials, specs, *_ = setup
    path, party_path = tmp_path / "network.json", tmp_path / "party.json"
    path.write_bytes(network.canonical_bytes())
    party_path.write_bytes(specs[0].canonical_bytes())
    for env, key in credentials.items():
        monkeypatch.setenv(env, key)
    monkeypatch.setattr(cli, "discover", lambda *args, **kwargs: pytest.fail("dry-run discovery"))
    monkeypatch.setattr(cli, "control_request", lambda *args, **kwargs: pytest.fail("dry-run mutation"))
    for verb in ("inspect", "parties", "snapshot", "drain", "leave"):
        args = ["network", verb, str(path), "--dry-run", "--format", "json"]
        if verb == "snapshot":
            args += ["--output", str(tmp_path / "snapshot.json")]
        if verb in {"drain", "leave"}:
            args += ["--party", "a"]
        main(args)
        assert json.loads(capsys.readouterr().out)["data"]["dry_run"]
    assert not (tmp_path / "snapshot.json").exists()
    main(["serve", "party", "--network", str(path), "--party", str(party_path), "--dry-run", "--format", "json"])
    assert json.loads(capsys.readouterr().out)["command"] == "serve.party"


def test_live_prepared_role_specs_supported_and_unknown_deployment_claims_rejected(setup):
    from pllm.profiles import MaskedLinearCpu

    network, _, specs, *_ = setup
    prepared = replace(specs[0], experiment=replace(specs[0].experiment,
                       pipeline=MaskedLinearCpu(specs[0].experiment.pipeline.model)),
                       role_ids=("inference", "preparation"))
    assert PartySpec.from_spec(prepared.to_spec()) == prepared
    with pytest.raises(NetworkError):
        NetworkSpec.from_spec({**network.to_spec(), "password": "should-never-be-config"})


@pytest.mark.asyncio
async def test_async_admission_cancel_waits_and_releases_returned_lease(monkeypatch):
    import threading
    from pllm.deployment import execution

    started, finish = threading.Event(), threading.Event()
    class Lease:
        closed = False
        def close(self):
            self.closed = True
    lease = Lease()
    def admitting(*args, **kwargs):
        started.set()
        finish.wait(timeout=10)
        return lease
    monkeypatch.setattr(execution, "open_execution", admitting)
    task = asyncio.create_task(execution.async_open_execution(None, network=None))
    await asyncio.to_thread(started.wait, 10)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lease.closed


@pytest.mark.integration
def test_controller_loss_revokes_at_ttl_without_client_and_stale_epoch_restart(setup, tmp_path):
    with http_hosts(setup, tmp_path) as (network, credentials, children):
        result = selected(setup, discover(network, credentials=credentials))
        lease = open_execution(result, network=network, credentials=credentials, ttl_seconds=5)
        children[1].terminate()
        children[1].wait(timeout=10)
        lease.close()  # A releases immediately; lost B cannot authorize any new call.
        assert lease.closed
        first = control_request(network.parties[0].origin, network.parties[0].credential_env,
                                "/v1/party/offer", credentials=credentials)
        assert first["active_sessions"] == 0
        with pytest.raises(NetworkError, match="closed"):
            OpenAI(execution=lease)
    # Every process restart changes epoch, including a process with the same spec.
    with http_hosts(setup, tmp_path) as (network, credentials, _):
        with pytest.raises(NetworkError, match="STALE_INSTANCE_EPOCH"):
            open_execution(result, network=network, credentials=credentials)


def test_authoritative_expiry_cancels_active_provider_call_before_capacity_reuse(controllers, setup):
    apps, clients, result, clock = controllers
    controller = apps[0].state.controller
    role = next(row["role_id"] for row in result.native_placement["roles"] if row["party_id"] == "a")
    auth = {"authorization": f"Bearer {setup[1]['PLLM_TEST_A']}"}
    body = reserve_body(result, controller, role=role)
    clients[0].post("/v1/party/reserve", json=body, headers=auth)
    row = next(iter(controller._reservations.values()))
    async def active_expiry():
        burned = asyncio.Event()
        async def active_call():
            try:
                await asyncio.sleep(60)
            finally:
                burned.set()
                row.calls.discard(asyncio.current_task())
        call = asyncio.create_task(active_call())
        row.calls.add(call)
        await asyncio.sleep(0)
        clock[0] += 5000
        await controller.purge()
        assert call.cancelled() and burned.is_set()
        assert row.revoked and not controller._reservations
    clients[0].portal.call(active_expiry)


@pytest.mark.integration
def test_real_directory_membership_automatic_renew_and_live_sdk(setup, tmp_path):
    network, credentials, specs, model_plan, lock = setup
    directory_port = free_port()
    network = replace(network, directory_origin=f"http://127.0.0.1:{directory_port}",
                      directory_credential_env="PLLM_TEST_DIRECTORY")
    credentials = {**credentials, "PLLM_TEST_DIRECTORY": secrets.token_urlsafe(32)}
    configured = network, credentials, specs, model_plan, lock
    network_path = tmp_path / "directory-network.json"
    network_path.write_bytes(network.canonical_bytes())
    child = subprocess.Popen([sys.executable, "-m", "pllm", "serve", "directory",
        "--network", str(network_path), "--port", str(directory_port)],
        env={**os.environ, **credentials}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if child.poll() is not None:
                pytest.fail(child.stderr.read().decode())
            try:
                if httpx.get(network.directory_origin + "/healthz", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        else:
            pytest.fail("directory HTTP startup timeout")
        with http_hosts(configured, tmp_path) as (network, credentials, _):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                snapshot = discover(network, credentials=credentials)
                if len(snapshot.offers) == 3:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("directory membership did not renew")
            result = selected(configured, snapshot)
            with open_execution(result, network=network, credentials=credentials) as lease:
                with OpenAI(execution=lease) as client:
                    assert client.responses.create(input="Hi", temperature=0.0).usage.output_tokens == 2
            for trust in network.parties:
                control_request(network.directory_origin, trust.credential_env, "/v1/directory/withdraw",
                                credentials=credentials, body={"party_id": trust.party_id})
            assert len(discover(network, credentials=credentials).offers) == 1
    finally:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        child.stderr.close()
