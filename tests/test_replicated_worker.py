"""Authenticated peer isolation and response-owned replicated admission."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from pllm.runtime import replicated_worker as worker
from pllm.runtime.delegated_protocol import pack, unpack
from pllm.runtime.replicated_source import admit_source
from test_replicated_sdk import experiment_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("weights,range_policy,lookup,selection,transport,expected", [
    ("public", "checked", "per_token", "expanded", "http", "source"),
    ("private", "static", "per_token", "expanded", "http", "legacy"),
    ("public", "static", "per_token", "expanded", "http", "legacy"),
    ("public", "checked", "per_token", "expanded", "streams", "legacy"),
])
@pytest.mark.parametrize("failure", [None, "missing", "old_signature", "constructor", "source",
    "config", "binding", "runtime", "disk", "import", "client_policy"])
async def test_source_request_route_mock(tmp_path, monkeypatch, weights, range_policy,
                                         lookup, selection, transport, expected, failure):
    """Real canonical policy/admit envelope; no native call or filesystem source IO."""
    from pllm import Model, Deployment, ExecutionBudget
    from pllm.configuration import Experiment
    from pllm.profiles import PumaThreePartyCpu
    from pllm.protocols import PumaThreePartyDecoder
    from pllm.runtime.replicated_protocol import execution_binding

    experiment = Experiment("route-mock", PumaThreePartyCpu(Model.path(str(tmp_path)),
        linear=PumaThreePartyDecoder(weights=weights, range_policy=range_policy,
            lookup=lookup, selection=selection, peer_transport=transport)),
        Deployment.local(root=str(tmp_path / "roles")), ExecutionBudget(requests=8,
            max_input_tokens=8, max_new_tokens=2))
    events, held = [], []
    retained_experiment = bytes(bytearray(experiment.canonical_bytes()))
    retained_config = bytes(bytearray(b"server config"))

    class Reservation:
        def __enter__(self):
            held.append(self)
            events.append("reserved")
            return self

        def __exit__(self, *_):
            events.append("refund")

    class Native:
        def __init__(self, manifest, role):
            events.append("legacy")

        @staticmethod
        def _from_source_request(manifest, role, *, server_policy):
            events.append("source")
            assert type(manifest) is bytes and role == 1
            assert type(server_policy) is tuple and len(server_policy) == 2
            assert server_policy[0] is retained_experiment
            assert server_policy[1] is retained_config
            if failure == "client_policy":
                assert json.loads(manifest)["server_policy"] != [list(item) for item in server_policy]
            if failure == "constructor":
                raise ValueError("constructor refused")
            return SimpleNamespace(close=lambda: events.append("close"))

        def close(self):
            events.append("close")

    if failure == "missing":
        del Native._from_source_request
    if failure == "old_signature":
        def old_signature(manifest, role):
            raise AssertionError("old constructor body must not be entered")
        Native._from_source_request = staticmethod(old_signature)

    def runtime(*args):
        events.append("runtime")
        if failure == "runtime":
            raise ValueError("runtime refused")

    def populate(owner, cancelled):
        events.append("populate")
        assert not cancelled() and "refund" not in events
        if failure == "import":
            raise ValueError("source import refused")

    source = SimpleNamespace(source_lock_digest="a" * 64, config_bytes=retained_config,
        experiment_bytes=retained_experiment,
        contract={"weights": weights, "public_weight_snapshot_bytes_per_worker": 1},
        runtime_contract=runtime, populate_public_worker=populate,
        runtime_contract_digest="b" * 64, tokenizer_digest="d" * 64)
    monkeypatch.setattr(worker, "admit_source", lambda _: source)
    monkeypatch.setattr(worker._native, "ReplicatedWorker", Native)
    monkeypatch.setattr(worker, "reserve_incremental_memory", lambda _: Reservation())
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(
        free=0 if failure == "disk" else 1 << 40))
    key = "client-secret-12345678"
    peers = {"worker_a": ("http://127.0.0.1:1", "outgoing-a-12345678", "incoming-a-12345678"),
             "worker_c": ("http://127.0.0.1:2", "outgoing-c-12345678", "incoming-c-12345678")}
    app = worker.create_replicated_worker_app(experiment, "worker_b", key, peers)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test",
            headers={"authorization": f"Bearer {key}"}) as client:
        epoch = unpack((await client.get("/v1/replicated/manifest")).content)["epoch"]
        value = {"session": list(bytes.fromhex("c" * 64)),
                 "source": list(bytes.fromhex("a" * 64)),
                 "composition": list(bytes.fromhex(experiment.pipeline.digest())),
                 "config": list(source.config_bytes), "weights": weights, "rows": 2, "outputs": 2}
        if execution_binding(experiment) is not None:
            value["execution"] = execution_binding(experiment)
        if failure == "source":
            value["source"] = [0] * 32
        if failure == "config":
            value["config"] = list(b"other config")
        if failure == "binding":
            value["execution"] = {}
        if failure == "client_policy":
            # Untrusted manifest metadata cannot replace admitted server owners
            # or select the source route on a canonical legacy server policy.
            value["server_policy"] = [list(b"attacker enlarged policy"), list(b"other config")]
        response = await client.post("/v1/replicated/admit", content=pack({
            "request": "c" * 64, "epoch": epoch, "manifest": json.dumps(value).encode()}))
        rejected = (failure in {"source", "config", "binding", "runtime"}
                    or failure == "disk" and weights == "public"
                     or failure in {"missing", "old_signature", "constructor"} and expected == "source"
                    or failure == "import" and weights == "public")
        assert response.status_code == (400 if rejected else 200)
        if failure in {"source", "config", "binding", "runtime"} or failure == "disk" and weights == "public":
            assert "source" not in events and "legacy" not in events
        elif not (failure in {"missing", "old_signature"} and expected == "source"):
            assert expected in events
            assert ("legacy" if expected == "source" else "source") not in events
        else:
            assert "source" not in events and "legacy" not in events
        if not rejected:
            assert "refund" not in events
            assert events.index("runtime") < events.index(expected)
            assert ("populate" in events) == (weights == "public")
            assert (await client.post(f'/v1/replicated/{"c" * 64}/cancel')).status_code == 200
        assert events[-1] == "refund"


@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
def test_public_import_preserves_original_error_when_cleanup_fails(monkeypatch, error_type):
    from pllm.runtime.replicated_source import ReplicatedSource
    original = error_type("import failed")

    def fail(*args):
        raise original

    monkeypatch.setattr(ReplicatedSource, "_append_source", fail)
    source = SimpleNamespace(contract={"weights": "public", "source_shapes": {}}, _append_source=fail)
    owner = SimpleNamespace(append_public_source=lambda *args: None,
        close=lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")))
    with pytest.raises(error_type) as caught:
        ReplicatedSource.populate_public_worker(source, owner, lambda: False)
    assert caught.value is original
    assert "RuntimeError" in original.__notes__[0]


@pytest.mark.asyncio
async def test_cancelled_source_constructor_closes_before_reservation_refund():
    from functools import partial
    import threading
    entered, release = threading.Event(), threading.Event()
    events = []
    identities = []

    def constructor(manifest, role, *, server_policy):
        assert manifest == b"request" and role == 1
        assert tuple(map(id, server_policy)) == tuple(identities)
        entered.set()
        assert release.wait(5)
        assert tuple(map(id, server_policy)) == tuple(identities)
        assert server_policy == (b"retained server experiment", b"retained server config")
        events.append("returned")
        return SimpleNamespace(close=lambda: events.append("closed"))

    async def admission():
        source = SimpleNamespace(experiment_bytes=bytes(bytearray(b"retained server experiment")),
            config_bytes=bytes(bytearray(b"retained server config")))
        identities.extend((id(source.experiment_bytes), id(source.config_bytes)))
        bound = partial(constructor, server_policy=(source.experiment_bytes, source.config_bytes))
        source = None  # Only the bound call/native thread need retain these exact bytes.
        try:
            await worker.native_call(bound, b"request", 1, _discard=lambda owner: owner.close())
        finally:
            events.append("refund")

    task = asyncio.create_task(admission())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and events == []
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["returned", "closed", "refund"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


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
