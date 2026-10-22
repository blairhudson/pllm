"""Actual worker callback + encrypted tape, bounded native/source doubles only."""
import asyncio

import httpx
import pytest
import pytest_asyncio

from pllm.runtime import fss_worker as worker, fss_protocol as p
import test_fss_worker as helpers
from test_fss_worker import double, cluster
from test_fss_material_receive import receive_quote


@pytest_asyncio.fixture(autouse=True)
async def receive_mode(double, monkeypatch):
    loop = asyncio.get_running_loop()
    server_tasks = set()
    asgi_transport = httpx.ASGITransport

    class IsolatedASGITransport(asgi_transport):
        async def handle_async_request(self, request):
            # A network server owns its request Task independently of its client.
            # Plain ASGITransport instead lets worker retirement cancel pytest's
            # Task. Its finally then sends /cancel and joins the retirement which
            # is waiting for that same Task: a fixture-only self-wait cycle.
            task = asyncio.create_task(super().handle_async_request(request))
            server_tasks.add(task)

            def retired(task):
                server_tasks.discard(task)
                try:
                    task.result()
                except BaseException as error:
                    p.clear_failure_frames(error)  # Includes unsent response leases.

            task.add_done_callback(retired)
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    task.cancel()
                    # Server/fixture retains ownership; a disconnected client
                    # must not join remote cleanup from its own callback stack.
                    raise
                raise httpx.RemoteProtocolError("FSS handler retired before response") from None

    monkeypatch.setattr(httpx, "ASGITransport", IsolatedASGITransport)
    double.receive_exited = {role: asyncio.Event() for role in (0, 1)}
    double.receive_failed = []
    double.receive_returned = []
    double.receive_finished = []
    original_contract, original_configuration = helpers.contract, helpers.configuration
    def contract(*args, **kwargs):
        return p.canonical(p.metadata(original_contract(*args, **kwargs)) | {"online_schedule": "tensor-v1"})
    monkeypatch.setattr(helpers, "contract", contract)
    monkeypatch.setattr(helpers, "configuration", lambda *args, **kwargs:
        original_configuration(*args, **kwargs).with_params(pipeline__linear__online_schedule="tensor-v1"))
    double.kernels = 1
    double.admission_changes = {"primitive_requests_per_party": 2,
        "cumulative_packet_bytes_per_party_bound": 128, "material_receive": receive_quote()}
    double.receive_fault = None
    double.receive_calls = []
    native_class = worker._native.FssRoleRequest
    original_init = native_class.__init__
    def init(self, *args, material_window=False):
        assert material_window is True
        original_init(self, *args)
    monkeypatch.setattr(native_class, "__init__", init)
    def meta(session, role, pc):
        raw = p.metadata(helpers.callback_metadata(session, "material", role, 0))
        raw["pc"] = pc
        return p.canonical(raw)
    def run(self, frame, callbacks):
        self.callbacks, self.running = callbacks, True
        try:
            if self.role == 2:
                for holder in (0, 1):
                    for pc in (3, 4):
                        callbacks.deliver(meta(self.request_id, holder, pc), f"packet-{holder}-{pc}".encode())
                callbacks.finish()
                return None
            metadata = tuple(meta(self.request_id, self.role, pc) for pc in (3, 4))
            if self.role == 1:
                if double.receive_fault == "reordered":
                    metadata = metadata[::-1]
                elif double.receive_fault == "duplicate":
                    metadata = (metadata[0], metadata[0])
                elif double.receive_fault in ("recipient", "publicmeta"):
                    value = p.metadata(metadata[0])
                    if double.receive_fault == "recipient":
                        value["role"] = 0
                    else:
                        value["kernel"]["context"]["plan"] = "0" * 64
                    metadata = (p.canonical(value), metadata[1])
            double.receive_calls.append(self.role)
            result = callbacks.material_window(metadata)
            double.receive_returned.append(self.role)
            assert result == tuple(f"packet-{self.role}-{pc}".encode() for pc in (3, 4))
            other = callbacks.exchange(helpers.callback_metadata(self.request_id, "peer", self.role, 0),
                f"peer{self.role}-0".encode())
            assert other == f"peer{1 - self.role}-0".encode()
            callbacks.finish()
            double.receive_finished.append(self.role)
            return f"final-{self.role}".encode()
        except BaseException:
            double.receive_failed.append((self.role, callbacks.attempt.cancelled()))
            raise
        finally:
            self.running = False
            if self.role in double.receive_exited:
                loop.call_soon_threadsafe(double.receive_exited[self.role].set)
    monkeypatch.setattr(native_class, "run", run)
    original_tape_close = worker.MaterialTape.close

    def tape_close(tape):
        for native in double.natives:
            if native.callbacks is not None and native.callbacks.attempt.tape is tape:
                assert native.closed and not native.running
                assert not native.callbacks.attempt.callback_tasks
                assert not native.callbacks.attempt.callback_futures
        return original_tape_close(tape)

    monkeypatch.setattr(worker.MaterialTape, "close", tape_close)
    yield
    # Cluster lifespan retires roles before this fixture's network-owner fence.
    # This set, not the disconnected client callback, retains server requests.
    await asyncio.gather(*tuple(server_tasks), return_exceptions=True)
    assert not server_tasks


@pytest.mark.asyncio
async def test_worker_window_uses_one_callback_for_two_original_encrypted_records(cluster):
    await cluster.ready()
    await cluster.prepare()
    assert (await cluster.post("dealer", "finish")).status_code == 200
    await cluster.inputs()
    for role in p.HOLDERS:
        assert (await cluster.post(role, "start")).status_code == 200
    results = await asyncio.wait_for(asyncio.gather(*(cluster.post(role, "finish") for role in p.HOLDERS)), 5)
    assert all(result.status_code == 200 for result in results)
    assert sorted(cluster.double.receive_calls) == [0, 1]
    for response in results:
        storage = p.unpack(response.content)["storage"]
        assert storage["state"] == "finished" and storage["records_read"] == storage["records_written"] == 2
        assert storage["packet_bytes_read"] == storage["packet_bytes_written"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["reordered", "duplicate", "recipient", "publicmeta"])
async def test_worker_rejects_entire_malformed_window_and_drains_no_accepted_output(cluster, fault):
    cluster.double.receive_fault = fault
    await cluster.ready()
    await cluster.prepare()
    assert (await cluster.post("dealer", "finish")).status_code == 200
    await cluster.inputs()
    try:
        for role in p.HOLDERS:
            assert (await cluster.post(role, "start")).status_code == 200
        # Observe failing B directly: sibling-finish visibility is a separate task.
        try:
            response = await asyncio.wait_for(cluster.post("worker_b", "finish"), 5)
        except httpx.RemoteProtocolError:
            # Cancelling a real server handler closes the connection, not the
            # client/coordinator Task. Both rejection forms disclose no output.
            pass
        else:
            assert response.status_code != 200
    finally:
        await asyncio.wait_for(asyncio.gather(*(cluster.post(role, "cancel") for role in p.HOLDERS)), 5)
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in cluster.double.receive_exited.values())), 5)
    assert (1, True) in cluster.double.receive_failed  # Signalled BEFORE native unwind.
    assert 1 not in cluster.double.receive_returned and not cluster.double.receive_finished
    constructors = len(cluster.double.natives)
    for role in p.HOLDERS:
        assert (await cluster.metrics(role))["active_sessions"] == 0
        assert (await cluster.post(role, "start")).status_code == 409
    assert len(cluster.double.natives) == constructors
    assert all(native.closed and not native.running for native in cluster.double.natives)
    assert all(source.closed for source in cluster.double.loaded)
    assert all(row["closed"] for row in cluster.double.storage_reservations)
    assert all(row["closed"] for row in cluster.double.reservations)
