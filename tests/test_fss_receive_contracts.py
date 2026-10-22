"""Independent worker/tape integration controls, with no native execution.

Adapter seam is the real worker's material_window(tuple[bytes, ...]) callback,
which supplies take_window(tuple[(raw_metadata, packet_bound), ...]). Existing
ASGI/source doubles provide topology only; frozen requests below are deliberately
mocked, not evidence about native scheduling, reply leases, or packet import.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from pllm.runtime import fss_protocol as p, fss_tape as tape, fss_worker as worker
import test_fss_worker as topology
from test_fss_worker import cluster, double


def material_metadata(session, role, pc, bound):
    value = p.metadata(topology.callback_metadata(session, "material", role))
    value.update(pc=pc, packet_bytes_bound=bound)
    return p.canonical(value)


@pytest.fixture
def receive_contract(double, monkeypatch, request):
    schedule = getattr(request, "param", "tensor-v1")
    state = SimpleNamespace(
        schedule=schedule, transform=lambda values: values, constructors=[],
        calls=[], accepted=[], returned=[], callbacks={}, tapes=[], read_calls=[],
        stop_after_first=False, before_finish=lambda: None,
        sequence=tuple((100 + (index * 7) % 17, (32, 48, 64)[index % 3])
                       for index in range(17)),
    )
    original_contract = topology.contract
    original_configuration = topology.configuration

    def contract(*args, **kwargs):
        value = p.metadata(original_contract(*args, **kwargs))
        if schedule is not None:
            value["online_schedule"] = schedule
        return p.canonical(value)

    def configuration(*args, **kwargs):
        experiment = original_configuration(*args, **kwargs)
        return (experiment if schedule is None else experiment.with_params(
            pipeline__linear__online_schedule=schedule))

    monkeypatch.setattr(topology, "contract", contract)
    monkeypatch.setattr(topology, "configuration", configuration)
    double.kernels = 1
    double.admission_changes = {
        "primitive_requests_per_party": len(state.sequence), "frames_per_party": 0,
        "largest_packet_bytes": tape.MATERIAL_RECEIVE_BYTES,
        "cumulative_packet_bytes_per_party_bound": len(state.sequence) * tape.MATERIAL_RECEIVE_BYTES,
    }
    if schedule is not None:
        # Synthetic native quote, not an assertion about Rust's actual allocation.
        double.admission_changes["material_receive"] = {
            "records_bound": tape.MATERIAL_RECEIVE_RECORDS,
            "payload_bytes_bound": tape.MATERIAL_RECEIVE_BYTES,
            "native_reservation_bytes_per_holder": 2 * tape.MATERIAL_RECEIVE_BYTES,
            "python_overlap_bytes_per_holder": tape.MATERIAL_RECEIVE_PYTHON_BYTES,
        }

    native_class = worker._native.FssRoleRequest
    original_init = native_class.__init__

    def init(self, *args, **kwargs):
        state.constructors.append((args[-1], kwargs.copy()))
        assert kwargs == ({} if schedule is None else {"material_window": True})
        original_init(self, *args)

    def rows(session, role):
        return tuple((material_metadata(session, role, pc, bound),
                      f"packet-{role}-{pc}".encode()) for pc, bound in state.sequence)

    def run(self, frame, callbacks):
        self.callbacks, self.running = callbacks, True
        try:
            if self.role == 2:
                assert frame is None
                for holder in (0, 1):
                    for raw, packet in rows(self.request_id, holder):
                        callbacks.deliver(raw, packet)
                callbacks.finish()
                return None
            assert frame == f"input-{self.role}".encode()
            state.callbacks[self.role] = callbacks
            originals = rows(self.request_id, self.role)
            width = 1 if schedule is None else tape.MATERIAL_RECEIVE_RECORDS
            for offset in range(0, len(originals), width):
                batch = originals[offset:offset + width]
                metadata = tuple(raw for raw, _ in batch)
                state.calls.append((self.role, metadata))
                if schedule is None:
                    packets = (callbacks.material(metadata[0]),)
                else:
                    packets = callbacks.material_window(state.transform(metadata))
                # Check outputs, but never emulate native reply validation/import.
                assert packets == tuple(packet for _, packet in batch)
                state.returned.append(packets)
                state.accepted.extend(zip(metadata, packets))
                if state.stop_after_first:
                    break
            state.before_finish()
            callbacks.finish()
            return f"final-{self.role}".encode()
        finally:
            self.running = False

    monkeypatch.setattr(native_class, "__init__", init)
    monkeypatch.setattr(native_class, "run", run)
    original_tape_init = worker.MaterialTape.__init__

    def tape_init(self, *args, **kwargs):
        original_tape_init(self, *args, **kwargs)
        state.tapes.append(self)

    original_take = worker.MaterialTape._take_locked

    def take(self, raw, bound):
        state.read_calls.append(raw)
        return original_take(self, raw, bound)

    monkeypatch.setattr(worker.MaterialTape, "__init__", tape_init)
    monkeypatch.setattr(worker.MaterialTape, "_take_locked", take)
    return state


@pytest.fixture
def contract_cluster(receive_contract, cluster):
    cluster.receive = receive_contract
    return cluster


async def prepared(cluster):
    await cluster.ready()
    await cluster.prepare()
    assert (await cluster.post("dealer", "finish")).status_code == 200
    await cluster.inputs()


def holder_native(cluster):
    return next(native for native in cluster.double.natives if native.role == 0)


async def finish_holder(cluster):
    assert (await cluster.post("worker_a", "start")).status_code == 200
    return await asyncio.wait_for(cluster.post("worker_a", "finish"), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("receive_contract", [None, "wavefront-v1", "tensor-v1"], indirect=True)
async def test_same_material_sequence_default_scalar_or_fewer_window_callbacks(contract_cluster):
    c, state = contract_cluster, contract_cluster.receive
    await prepared(c)
    response = await finish_holder(c)
    assert response.status_code == 200
    value = p.unpack(response.content)
    assert value["frame"] == b"final-0"
    storage = value["storage"]
    assert storage["state"] == "finished"
    assert storage["records_read"] == storage["records_written"] == 17
    assert storage["packet_bytes_read"] == storage["packet_bytes_written"]
    assert [p.metadata(raw)["pc"] for raw, _ in state.accepted] == [pc for pc, _ in state.sequence]
    assert [packet for _, packet in state.accepted] == [
        f"packet-0-{pc}".encode() for pc, _ in state.sequence]
    counts = [len(metadata) for role, metadata in state.calls if role == 0]
    if state.schedule is None:
        assert counts == [1] * 17
        assert all(kwargs == {} for _, kwargs in state.constructors)
    else:
        assert counts == [8, 8, 1]
        assert len(counts) < len(state.sequence)  # Callback count only; no timing claim.
    for packets in state.returned:
        assert type(packets) is tuple and all(type(packet) is bytes for packet in packets)
    assert holder_native(c).closed
    # Python immutable outputs intentionally survive tape retirement, as documented.
    assert state.returned[0][0] == b"packet-0-100"


def mutate_second(metadata, field):
    value = p.metadata(metadata[1])
    if field == "role":
        value["role"] = 1
    elif field == "pc":
        value["pc"] = p.metadata(metadata[0])["pc"]
    elif field in ("ordinal", "plan_digest"):
        value["kernel"][field] = 1 if field == "ordinal" else "b" * 64
    elif field == "first_tensor_index":
        value["kernel"]["context"][field] = 1
    else:
        value["kernel"]["context"][field] = "b" * 64
    return (metadata[0], p.canonical(value), *metadata[2:])


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["role", "pc", "ordinal", "plan_digest", "plan", "session",
                                       "operation", "first_tensor_index"])
async def test_each_record_context_checked_before_any_window_read(contract_cluster, field):
    c, state = contract_cluster, contract_cluster.receive
    state.transform = lambda values: mutate_second(values, field)
    await prepared(c)
    response = await finish_holder(c)
    assert response.status_code != 200 and not response.content
    assert not state.read_calls and not state.returned
    assert holder_native(c).closed
    assert (await c.post("worker_a", "start")).status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["empty", "list", "nine", "not_bytes", "metadata_bytes",
                                  "packet_bound", "payload_sum"])
async def test_public_window_bounds_rejected_before_read_or_result(contract_cluster, fault):
    c, state = contract_cluster, contract_cluster.receive

    def transform(metadata):
        if fault == "empty":
            return ()
        if fault == "list":
            return list(metadata)
        if fault == "nine":
            return metadata + metadata[:1]
        if fault == "not_bytes":
            return (bytearray(metadata[0]), *metadata[1:])
        if fault == "metadata_bytes":
            return (b"x" * 1025, *metadata[1:])
        if fault == "packet_bound":
            value = p.metadata(metadata[0])
            value["packet_bytes_bound"] = tape.MATERIAL_RECEIVE_BYTES + 1
            return (p.canonical(value), *metadata[1:])
        result = []
        for raw in metadata[:2]:
            value = p.metadata(raw)
            value["packet_bytes_bound"] = tape.MATERIAL_RECEIVE_BYTES // 2 + 1
            result.append(p.canonical(value))
        return tuple(result)

    state.transform = transform
    await prepared(c)
    response = await finish_holder(c)
    assert response.status_code != 200 and not response.content
    assert not state.read_calls and not state.returned
    assert state.tapes[0].closed and holder_native(c).closed


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["inline", "external_aad"])
@pytest.mark.parametrize("fault", ["reordered", "whole_context", "second_context", "short_file"])
async def test_tape_binding_and_short_reads_burn_window_without_suffix_retry(
        contract_cluster, monkeypatch, storage, fault):
    # Exercise actual AAD storage modes at the constructor seam, without claiming
    # decoder-native admission. Inline pricing conservatively covers either mode.
    c, state = contract_cluster, contract_cluster.receive
    original_init = worker.MaterialTape.__init__

    def init(self, *args, **kwargs):
        kwargs["metadata_storage"] = storage
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(worker.MaterialTape, "__init__", init)
    await prepared(c)
    owner = state.tapes[0]
    if fault == "short_file":
        owner._file.truncate(1)  # Real short encrypted backing; no mocked decrypt.
    else:
        def transform(metadata):
            if fault == "reordered":
                return metadata[::-1]
            changed = []
            for index, raw in enumerate(metadata):
                value = p.metadata(raw)
                if fault == "whole_context" or index == 1:
                    value["kernel"]["context"]["operation"] = "b" * 64
                changed.append(p.canonical(value))
            return tuple(changed)
        state.transform = transform
    response = await finish_holder(c)
    assert response.status_code != 200 and not response.content
    assert not state.returned and owner.closed
    assert holder_native(c).closed
    raw = material_metadata(c.request_id, 0, *state.sequence[1])
    with pytest.raises(tape.MaterialTapeError):
        owner.take_window(((raw, state.sequence[1][1]),))
    with pytest.raises(ValueError):
        state.callbacks[0].material(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["second_aad", "replay", "early_finish"])
async def test_partial_or_previous_window_cannot_release_final_output(contract_cluster, fault):
    c, state = contract_cluster, contract_cluster.receive
    if fault == "second_aad":
        # Keep callback kernel/PC contract valid, but change bound in second AAD.
        def transform(metadata):
            value = p.metadata(metadata[1])
            value["packet_bytes_bound"] += 1
            return (metadata[0], p.canonical(value), *metadata[2:])
        state.transform = transform
    elif fault == "replay":
        first = []
        def transform(metadata):
            if not first:
                first.append(metadata)
            return first[0]
        state.transform = transform
    else:
        state.stop_after_first = True
    await prepared(c)
    response = await finish_holder(c)
    assert response.status_code != 200 and not response.content
    owner = state.tapes[0]
    assert owner.closed and holder_native(c).closed
    assert owner.receipt()["records_read"] == (1 if fault == "second_aad" else 8)
    assert len(state.returned) == (0 if fault == "second_aad" else 1)
    assert (await c.post("worker_a", "finish")).status_code == 409


@pytest.mark.asyncio
async def test_cancel_mid_window_drains_before_refund_and_never_retries_suffix(contract_cluster, monkeypatch):
    c, state = contract_cluster, contract_cluster.receive
    await prepared(c)
    owner = state.tapes[0]
    entered, release = threading.Event(), threading.Event()
    signalled = asyncio.Event()
    original_take, original_signal = owner._take_locked, worker._Attempt.signal

    def take(raw, bound):
        packet = original_take(raw, bound)
        if owner.receipt()["records_read"] == 1:
            entered.set()
            assert release.wait(5)
        return packet

    def signal(attempt):
        original_signal(attempt)
        if attempt.tape is owner:
            attempt.loop.call_soon_threadsafe(signalled.set)

    monkeypatch.setattr(owner, "_take_locked", take)
    monkeypatch.setattr(worker._Attempt, "signal", signal)
    assert (await c.post("worker_a", "start")).status_code == 200
    cancel = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        held = [row for row in c.double.reservations if not row["closed"]]
        cancel = asyncio.create_task(c.post("worker_a", "cancel"))
        await asyncio.wait_for(signalled.wait(), 5)
        assert not cancel.done() and not holder_native(c).closed
        assert all(not row["closed"] for row in held)
        assert not state.returned
    finally:
        release.set()
        if cancel is not None:
            assert (await asyncio.wait_for(cancel, 5)).status_code == 200
    assert owner.receipt()["records_read"] == 1
    assert owner.closed and holder_native(c).closed and not state.returned
    assert len(state.calls) == len(state.read_calls) == 1
    assert (await c.post("worker_a", "start")).status_code == 409


@pytest.mark.asyncio
async def test_only_final_body_releases_output_and_keeps_export_owner_until_send(contract_cluster):
    c, state = contract_cluster, contract_cluster.receive
    await prepared(c)
    ready, release_native = threading.Event(), threading.Event()

    def before_finish():
        ready.set()
        assert release_native.wait(5)

    state.before_finish = before_finish
    started = await c.post("worker_a", "start")
    sending, release_send = asyncio.Event(), asyncio.Event()
    messages = []
    task = None
    try:
        assert p.unpack(started.content) == {"started": True}
        assert await asyncio.to_thread(ready.wait, 5)
        assert state.tapes[0].receipt()["state"] == "sealed"
        assert state.tapes[0].receipt()["records_read"] == 17
        held = [row for row in c.double.reservations if not row["closed"]]
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": f"/v1/fss/{c.request_id}/finish",
            "raw_path": b"/", "query_string": b"", "root_path": "",
            "server": ("127.0.0.1", 7300), "client": ("127.0.0.1", 1),
            "headers": [(b"authorization", f"Bearer {topology.client_key('worker_a')}".encode()),
                        (b"content-length", b"0")]}

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body":
                sending.set()
                await release_send.wait()

        task = asyncio.create_task(c.apps["worker_a"](scope, receive, send))
        # No HTTP output can be produced while native still lacks finish callback.
        await asyncio.sleep(0)
        assert not messages and not task.done()
        release_native.set()
        await asyncio.wait_for(sending.wait(), 5)
        assert holder_native(c).closed and state.tapes[0].closed
        assert all(not row["closed"] for row in held)
        value = p.unpack(next(message["body"] for message in messages
                              if message["type"] == "http.response.body"))
        assert value["frame"] == b"final-0" and value["storage"]["state"] == "finished"
    finally:
        release_native.set()
        release_send.set()
        if task is not None:
            await asyncio.wait_for(task, 5)
    assert sum(row["closed"] for row in held) == 1
