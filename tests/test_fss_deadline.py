"""Authenticated deadline propagation with bounded ASGI/native doubles."""
import asyncio
import json
import sys
import time
from types import SimpleNamespace

import pytest

from pllm.runtime import fss_protocol as p, fss_transport as client, fss_worker as worker
from test_fss_worker import (
    Frozen, ORIGINS, callback_metadata, client_key, cluster, configuration, contract, double, key,
)
from test_fss_http_lifecycle import client_route


@pytest.fixture
def clock(monkeypatch):
    # Only deadline reads in these two modules use logical time. Profiling,
    # asyncio, HTTP, thread joins and test timeouts retain their real clocks.
    clock = SimpleNamespace(now=1000.0, process_time_ns=time.process_time_ns,
                            perf_counter_ns=time.perf_counter_ns)
    clock.monotonic = lambda: clock.now
    monkeypatch.setattr(worker, "time", clock)
    monkeypatch.setattr(client, "time", clock)
    return clock


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder", "ttl": 86400}], indirect=True)
async def test_client_budget_covers_setup_and_imports_longer_than_300_seconds(cluster, client_route, clock, monkeypatch):
    original_contract = client_route.source.runtime_contract
    def runtime_contract(*args):
        clock.now += 100
        return original_contract(*args)
    monkeypatch.setattr(client_route.source, "runtime_contract", runtime_contract)
    peers_type = client._Peers
    class SlowChannelConstruction(peers_type):
        def start(self, *args, **kwargs):
            clock.now += 50
            super().start(*args, **kwargs)
    monkeypatch.setattr(client, "_Peers", SlowChannelConstruction)
    def setup_delay(role, action, content):
        if action == "manifest": clock.now += 25
        if action == "prepare": clock.now += 400
        return content
    client_route.transforms.append(setup_delay)
    original_load, attempts = worker.load_role_source, []
    def slow_import(*args, **kwargs):
        attempt = kwargs["cancelled"].__self__
        attempts.append(attempt)
        assert attempt.expires == 4600 and not attempt.cancelled()
        clock.now += 400  # Each import individually exceeds the old 300s TTL.
        return original_load(*args, **kwargs)
    monkeypatch.setattr(worker, "load_role_source", slow_import)
    result = await asyncio.wait_for(asyncio.to_thread(client.execute_request,
        configuration("decoder"), client_route.source, [1, 2, 3], 3,
        {r: (ORIGINS[r], client_key(r)) for r in p.ROLES}, 3600, client_route.attempt), 5)
    assert result.selected == (5, 6, 7)
    admissions = [p.unpack(raw, limit=p.ADMISSION_LIMIT) for _, action, raw in client_route.requests if action == "admit"]
    assert [a["remaining_seconds"] for a in admissions] == [3375, 2975, 2575]
    assert client_route.peers[0].deadline == 4600
    assert all(a.expires == 4600 and a.phase == "closed" for a in attempts)
    assert all(r["closed"] for r in cluster.double.storage_reservations)
    assert "input-0-after-dealer-retirement" in client_route.events


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster,remaining,effective", [
    ({"capacity": "decoder"}, 3600, 300),
    ({"capacity": "decoder", "ttl": 3600}, 600, 600),
    ({"capacity": "decoder", "ttl": 3600}, 86400, 3600),
    ({"capacity": "decoder", "ttl": 86400}, 86400, 86400),
], indirect=["cluster"])
async def test_worker_clamps_once_and_peer_traffic_cannot_refresh_expiry(cluster, remaining, effective, clock, monkeypatch):
    original_load, attempts = worker.load_role_source, []
    def load(*args, **kwargs):
        attempts.append(kwargs["cancelled"].__self__)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(worker, "load_role_source", load)
    assert (await cluster.admit(roles=("worker_a",), remaining_seconds=remaining))["worker_a"].status_code == 200
    attempt = attempts[0]
    expires = 1000 + effective
    assert attempt.expires == expires
    await cluster.bind(roles=("worker_a",))
    def material(sequence):
        return {"sender": "dealer", "sender_epoch": cluster.epochs["dealer"],
            "recipient_epoch": cluster.epochs["worker_a"], "sequence": sequence,
            "metadata": callback_metadata(cluster.request_id, "material", 0, sequence), "packet": b"packet"}
    clock.now = expires - 1
    assert (await cluster.post("worker_a", "material", material(0), credential=key("dealer", "worker_a"))).status_code == 200
    assert (await cluster.metrics("worker_a"))["active_sessions"] == 1
    assert attempt.expires == expires
    clock.now = expires + 1
    assert (await cluster.post("worker_a", "material", material(1), credential=key("dealer", "worker_a"))).status_code == 400
    assert attempt.expires == expires and attempt.phase == "closed"
    assert not attempt.disk_reservations and attempt.reservation is None
    assert (await cluster.admit(roles=("worker_a",), remaining_seconds=remaining))["worker_a"].status_code == 400
    assert (await cluster.metrics("worker_a"))["active_sessions"] == 0
    assert not cluster.double.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder", "ttl": 86400}], indirect=True)
@pytest.mark.parametrize("remaining", [None, True, False, "600", b"600", [], {}, 0, -1,
    float("nan"), float("inf"), -float("inf"), 86400.001, (1 << 64) - 1])
async def test_invalid_remaining_lifetime_rejects_before_source_or_disk(cluster, remaining):
    response = (await cluster.admit(roles=("worker_a",), remaining_seconds=remaining))["worker_a"]
    assert response.status_code == 400
    assert not cluster.double.loaded and not cluster.double.storage_reservations
    # Valid session IDs remain burned even when the authenticated duration fails.
    assert (await cluster.admit(roles=("worker_a",), remaining_seconds=600))["worker_a"].status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder", "ttl": 86400}], indirect=True)
async def test_decoder_requires_client_duration_and_client_authentication(cluster):
    await cluster.manifests()
    value = {"request": cluster.request_id, "epoch": cluster.epochs["worker_a"], "epochs": cluster.epochs,
        "rows": 3, "outputs": 3, "contract": contract(capacity="decoder")}
    assert (await cluster.clients["worker_a"].post("/v1/fss/admit", content=p.pack(value))).status_code == 400
    value["remaining_seconds"] = 600
    response = await cluster.clients["worker_a"].post("/v1/fss/admit", content=p.pack(value),
        headers={"authorization": f"Bearer {key('dealer', 'worker_a')}"})
    assert response.status_code == 401
    assert not cluster.double.loaded and not cluster.double.storage_reservations


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder", "ttl": 86400}], indirect=True)
@pytest.mark.parametrize("spent_in", ["body", "local_contract"])
async def test_time_before_import_is_not_a_fresh_budget(cluster, clock, monkeypatch, spent_in):
    if spent_in == "body":
        original = worker._body
        async def body(request, limit):
            value = await original(request, limit)
            if request.url.path.endswith("/admit"): clock.now += 601
            return value
        monkeypatch.setattr(worker, "_body", body)
    else:
        original = Frozen.runtime_contract
        def runtime_contract(self, *args):
            clock.now += 601
            return original(self, *args)
        monkeypatch.setattr(Frozen, "runtime_contract", runtime_contract)
    monkeypatch.setattr(worker, "load_role_source", lambda *_, **__: pytest.fail("expired request imported source"))
    response = (await cluster.admit(roles=("worker_a",), remaining_seconds=600))["worker_a"]
    assert response.status_code == 400
    assert not cluster.double.loaded and not cluster.double.natives
    assert all(r["closed"] for r in cluster.double.storage_reservations + cluster.double.reservations)


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder", "ttl": 86400}], indirect=True)
async def test_short_client_deadline_interrupts_import_but_drains_owners_before_refund(cluster, monkeypatch):
    cluster.double.load_wait = True
    original_load, attempts = worker.load_role_source, []
    def load(*args, **kwargs):
        attempts.append(kwargs["cancelled"].__self__)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(worker, "load_role_source", load)
    admitting = asyncio.create_task(cluster.admit(roles=("worker_a",), remaining_seconds=.1))
    cancelling = None
    try:
        assert await asyncio.to_thread(cluster.double.load_entered.wait, 2)
        attempt = attempts[0]
        expires = attempt.expires
        for _ in range(200):
            if attempt.stop.is_set(): break
            await asyncio.sleep(.005)
        assert attempt.stop.is_set() and not admitting.done()
        assert attempt.reservation is not None and attempt.disk_reservations
        assert all(not r["closed"] for r in cluster.double.storage_reservations)
        cancelling = asyncio.create_task(cluster.post("worker_a", "cancel"))
        await asyncio.sleep(.01)
        assert not cancelling.done() and attempt.expires == expires
    finally:
        cluster.double.load_release.set()
        results = await asyncio.wait_for(asyncio.gather(admitting, *([cancelling] if cancelling else [])), 3)
    assert results[0]["worker_a"].status_code == 400 and results[1].status_code == 200
    assert cluster.double.load_cancelled == [True] and cluster.double.loaded[-1].closed
    assert not attempt.disk_reservations and attempt.reservation is None and attempt.expires == expires
    assert all(r["closed"] for r in cluster.double.storage_reservations + cluster.double.reservations)
    assert (await cluster.admit(roles=("worker_a",), remaining_seconds=600))["worker_a"].status_code == 400


@pytest.mark.parametrize("capacity,maximum", [("capsule", 300), ("decoder", 86400)])
def test_cli_uses_decoder_maximum_as_cap_not_api_default(monkeypatch, capacity, maximum):
    import uvicorn
    experiment = configuration(capacity)
    monkeypatch.setattr(worker.Experiment, "from_spec", lambda _: experiment)
    monkeypatch.setattr(sys, "argv", ["fss-worker", "--role", "worker_a", "--port", "7300"])
    for name, value in {"PLLM_ROLE_EXPERIMENT_JSON": {}, "PLLM_FSS_PEERS_JSON": {},
                        "PLLM_FSS_PEER_CREDENTIALS_JSON": {}}.items():
        monkeypatch.setenv(name, json.dumps(value))
    monkeypatch.setenv("PLLM_FSS_API_KEY", client_key("worker_a"))
    monkeypatch.delenv("PLLM_FSS_SOURCE_PINS_JSON", raising=False)
    calls = []
    app = object()
    def create(value, role, api_key, **kwargs):
        assert value is experiment and role == "worker_a" and api_key == client_key(role)
        calls.append(kwargs)
        return app
    monkeypatch.setattr(worker, "create_fss_worker_app", create)
    def serve(value, **kwargs):
        assert value is app and kwargs["host"] == "127.0.0.1"
    monkeypatch.setattr(uvicorn, "run", serve)
    worker.main()
    assert calls[0]["ttl"] == maximum


@pytest.mark.asyncio
async def test_capsule_rejects_deadline_field_in_legacy_payload(cluster):
    assert (await cluster.admit(roles=("worker_a",), remaining_seconds=600))["worker_a"].status_code == 400
    assert not cluster.double.loaded
