"""Authenticated peer isolation and response-owned replicated admission."""

import asyncio

import httpx
import pytest

from pllm.runtime import replicated_worker as worker
from pllm.runtime.delegated_protocol import pack, unpack
from pllm.runtime.replicated_source import admit_source
from test_replicated_sdk import experiment_fixture


@pytest.mark.asyncio
async def test_peer_credentials_stale_sessions_and_burned_owner(tmp_path, monkeypatch):
    experiment = experiment_fixture(tmp_path)
    publisher, manifest = admit_source(experiment).publisher(experiment, (3, 4), 1, lambda: False)
    import json
    request_id = bytes(json.loads(manifest)["session"]).hex()
    client_key, incoming_a, incoming_c = "client-key-12345678", "a-to-b-key-12345678", "c-to-b-key-12345678"
    peers = {"worker_a": ("http://127.0.0.1:1", "b-to-a-key-12345678", incoming_a),
             "worker_c": ("http://127.0.0.1:2", "b-to-c-key-12345678", incoming_c)}
    reservations = []

    class Reservation:
        def __init__(self, size):
            self.size, self.closed = size, False
            reservations.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            assert not self.closed
            self.closed = True

    monkeypatch.setattr(worker, "reserve_incremental_memory", Reservation)
    app = worker.create_replicated_worker_app(experiment, "worker_b", client_key, peers)
    try:
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test",
            headers={"authorization": f"Bearer {client_key}"},
        ) as client:
            public = unpack((await client.get("/v1/replicated/manifest")).content)
            body = pack({"request": request_id, "epoch": public["epoch"], "manifest": manifest})
            assert (await client.post("/v1/replicated/admit", content=body)).status_code == 200
            held, = reservations
            assert held.size == worker.REQUEST_MEMORY and not held.closed
            assert (await client.post("/v1/replicated/admit", content=body)).status_code == 400

            # Neither a client credential on the peer endpoint nor a correctly
            # authenticated stale peer can consume the active request.
            path = f"/v1/replicated/{request_id}/peer/public"
            frame = pack({"sender": "worker_a", "payload": b"bad"})
            assert (await client.post(path, content=frame)).status_code == 401
            assert (await client.post(path.replace(request_id, "f" * 64), content=frame,
                headers={"authorization": f"Bearer {incoming_a}"})).status_code == 409
            assert (await client.get("/v1/replicated/metrics")).json()["active_sessions"] == 1
            assert not held.closed

            # An authenticated role cannot substitute the other peer's sender.
            assert (await client.post(path, content=frame,
                headers={"authorization": f"Bearer {incoming_c}"})).status_code == 400
            assert (await client.post(f"/v1/replicated/{request_id}/setup_public")).status_code == 409
            assert not held.closed
            assert (await client.post(f"/v1/replicated/{request_id}/cancel")).status_code == 200
            assert held.closed
            assert (await client.get("/v1/replicated/metrics")).json()["active_sessions"] == 0
            assert (await client.post("/v1/replicated/admit", content=body)).status_code == 400
            assert all(item.closed for item in reservations)
    finally:
        publisher.close()


@pytest.mark.asyncio
async def test_cancelled_native_operation_drains_before_owner_refund():
    import threading

    entered, release = threading.Event(), threading.Event()
    finished = []

    def operation():
        entered.set()
        assert release.wait(5)
        finished.append(True)

    task = asyncio.create_task(worker.native_call(operation))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished == [True]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
