"""Authenticated batching through real ASGI/tapes; native/source work is doubled."""
from contextlib import contextmanager
import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from pllm.profiles import FuseFssSourceCpu
from pllm.runtime import fss_protocol as p, fss_transport as transport, fss_worker as worker
from pllm.runtime.protocol import encode_length_prefixed, iter_length_prefixed, ProtocolError
import test_fss_worker as base
import test_fss_http_lifecycle as lifecycle
from test_fss_worker import double as base_double, cluster as cluster
from test_fss_http_lifecycle import client_route as client_route


def batch_admission():
    return {"material_delivery": "batch-v1", "delivery_batch": {
            "records_bound": 64, "body_bytes_bound": 1 << 20, "envelope_bytes_bound": 1024,
            "record_overhead_bytes": 8, "native_reservation_bytes": 2 << 20,
            "python_overlap_bytes": 10 * (1 << 20) + 2 * 64 * 4096 + 4096}}


@pytest.fixture
def double(base_double, monkeypatch, request):
    d = base_double
    d.delivery = getattr(request, "param", "batch-v1")
    d.batch_changes, d.batches, d.batch_jobs, d.append_threads = {}, [], [], []
    original_contract, original_configuration = base.contract, base.configuration
    def contract(*args, **kwargs):
        value = p.metadata(original_contract(*args, **kwargs))
        if d.delivery != "single": value["material_delivery"] = d.delivery
        return p.canonical(value)
    def configuration(capacity="capsule"):
        exp = original_configuration(capacity)
        return exp.with_params(pipeline=FuseFssSourceCpu(exp.pipeline.model, numeric_profile=base.PROFILE,
            capacity=capacity, material_delivery=d.delivery))
    monkeypatch.setattr(base, "contract", contract)
    monkeypatch.setattr(lifecycle, "contract", contract)
    monkeypatch.setattr(base, "configuration", configuration)
    original_init = base.Frozen.__init__
    def frozen(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if d.delivery != "single": self.contract["material_delivery"] = d.delivery
    monkeypatch.setattr(base.Frozen, "__init__", frozen)
    original_reserve = worker.reserve_incremental_memory
    @contextmanager
    def reserve(amount):
        assert amount == p.ROLE_MEMORY + (p.MATERIAL_BATCH_PYTHON_BYTES if d.delivery == "batch-v1" else 0)
        with original_reserve(p.ROLE_MEMORY):
            d.reservations[-1]["amount"] = amount
            yield
    monkeypatch.setattr(worker, "reserve_incremental_memory", reserve)
    native = worker._native.FssRoleRequest
    original_metadata, original_run = native.metadata, native.run
    def metadata(self):
        value = p.metadata(original_metadata(self))
        if d.delivery == "batch-v1": value.update(batch_admission())
        for name, replacement in d.batch_changes.items():
            if name in value: value[name] = replacement
            elif replacement is None: value["delivery_batch"].pop(name)
            else: value["delivery_batch"][name] = replacement
        return p.canonical(value)
    def run(self, frame, callbacks):
        if self.role != 2 or d.delivery == "single":
            return original_run(self, frame, callbacks)
        self.callbacks, self.running = callbacks, True
        d.native_threads.append(threading.get_ident())
        try:
            assert frame is None
            for role in (0, 1):
                for start in range(0, d.kernels, 64):
                    records = tuple((base.callback_metadata(self.request_id, "material", role, index),
                                     f"material-{role}-{index}".encode())
                                    for index in range(start, min(start + 64, d.kernels)))
                    d.batches.append((role, records))
                    callbacks.deliver_batch(role, records)
            callbacks.finish()
        finally:
            self.running = False
    monkeypatch.setattr(native, "metadata", metadata)
    monkeypatch.setattr(native, "run", run)
    original_call, original_append = worker.native_call, worker.MaterialTape.append
    async def call(function, *args, **kwargs):
        if function.__name__ == "append_batch": d.batch_jobs.append(function.__name__)
        return await original_call(function, *args, **kwargs)
    def append(owner, metadata, packet):
        d.append_threads.append(threading.get_ident())
        return original_append(owner, metadata, packet)
    monkeypatch.setattr(worker, "native_call", call)
    monkeypatch.setattr(worker.MaterialTape, "append", append)
    return d


def envelope(cluster, *, role=0, start=0, count=2, **changes):
    records = tuple((base.callback_metadata(cluster.request_id, "material", role, index),
                     f"material-{role}-{index}".encode()) for index in range(start, start + count))
    return {"sender": "dealer", "sender_epoch": cluster.epochs["dealer"],
            "recipient_epoch": cluster.epochs[p.HOLDERS[role]], "sequence": start,
            "count": count, "records": p.encode_material_batch(records), **changes}


async def send(cluster, value, *, role="worker_a", **kwargs):
    return await cluster.post(role, "material-batch", value, credential=base.key("dealer", role), **kwargs)


def dealer(cluster):
    return next(n for n in cluster.double.natives if n.role == 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "capsule"}, {"capacity": "decoder"}], indirect=True)
async def test_bounded_batches_complete_real_tapes_without_online_dealer_and_count_actual_bytes(cluster, client_route):
    cluster.double.kernels = 65  # exercises full 64-record binary envelope plus remainder
    result = await asyncio.wait_for(asyncio.to_thread(transport.execute_request,
        base.configuration(cluster.double.capacity), client_route.source, [1, 2, 3], 3,
        {role: (base.ORIGINS[role], base.client_key(role)) for role in p.ROLES},
        60, client_route.attempt), 10)
    assert result.selected == (5, 6, 7)
    assert [len(records) for _, records in cluster.double.batches] == [64, 1, 64, 1]
    assert len(cluster.double.batch_jobs) == 4
    assert len(cluster.double.append_threads) == 130
    assert threading.get_ident() not in cluster.double.append_threads
    assert "input-0-after-dealer-retirement" in client_route.events
    assert all(row["closed"] and row["amount"] == p.ROLE_MEMORY + p.MATERIAL_BATCH_PYTHON_BYTES
               for row in cluster.double.reservations)
    assert all(row["closed"] for row in cluster.double.storage_reservations)
    assert all("dealer" not in (a, b) for a, b, phase, _, _ in cluster.double.rpc_log if phase == "online")
    for sender in p.ROLES:
        for recipient in p.ROLES:
            if sender == recipient: continue
            for phase in ("setup", "online"):
                expected = sum(up for a, b, q, up, _ in cluster.double.rpc_log if (a, b, q) == (sender, recipient, phase))
                expected += sum(down for a, b, q, _, down in cluster.double.rpc_log if (a, b, q) == (recipient, sender, phase))
                assert client_route.attempt.audit[f"peer_link.{sender}.{recipient}.{phase}_bytes"] == expected
    assert (await cluster.metrics("dealer"))["phase_timing"]["preparation_calls"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [{"material_delivery": "single"}, {"records_bound": None},
    {"records_bound": True}, {"records_bound": 65}, {"body_bytes_bound": None},
    {"body_bytes_bound": (1 << 20) + 1}, {"envelope_bytes_bound": 0},
    {"native_reservation_bytes": None}, {"native_reservation_bytes": (64 << 20) + 1},
    {"native_reservation_bytes": 0}, {"record_overhead_bytes": 4},
    {"python_overlap_bytes": None}, {"python_overlap_bytes": 1 << 20}, {"delivery_batch": None}])
async def test_batch_requires_exact_native_priced_fields_before_tape(cluster, changes, monkeypatch):
    assert all(r.status_code == 200 for r in (await cluster.admit()).values())
    cluster.double.batch_changes = changes
    monkeypatch.setattr(worker.MaterialTape, "__init__", lambda *a, **k: pytest.fail("unpriced batch created tape"))
    response = await cluster.post("worker_a", "bind", {"request_frame": b"request:" + cluster.request_id.encode()})
    assert response.status_code == 400
    assert cluster.double.natives[-1].closed
    assert not cluster.double.batch_jobs


@pytest.mark.asyncio
@pytest.mark.parametrize("double", ["single"], indirect=True)
async def test_legacy_declared_single_refuses_batch_and_keeps_original_reservation(cluster):
    await cluster.ready()
    response = await send(cluster, envelope(cluster))
    assert response.status_code == 400 and not cluster.double.append_threads
    assert all(row["amount"] == p.ROLE_MEMORY for row in cluster.double.reservations)


@pytest.mark.asyncio
@pytest.mark.parametrize("double", ["single"], indirect=True)
async def test_legacy_singleton_still_completes_without_batch_callback(cluster):
    await lifecycle.complete(cluster)
    assert not cluster.double.batches and not cluster.double.batch_jobs
    assert len(cluster.double.append_threads) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("double", ["single"], indirect=True)
async def test_legacy_native_callback_cannot_enable_unpriced_batch(cluster):
    await cluster.ready()
    native = dealer(cluster)
    def run(frame, callbacks):
        callbacks.deliver_batch(0, ((base.callback_metadata(cluster.request_id, "material", 0), b"packet"),))
    native.run = run
    assert (await cluster.post("dealer", "prepare")).status_code == 400
    assert native.closed and not cluster.double.rpc_log and not cluster.double.append_threads


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["role_bool", "role_dealer", "mixed_role", "list", "count", "bytes"])
async def test_bad_native_batch_retires_dealer_before_rpc_or_encode(cluster, monkeypatch, damage):
    cluster.double.admission_changes = {"largest_packet_bytes": 2 << 20,
                                       "cumulative_packet_bytes_per_party_bound": 4 << 20}
    await cluster.ready()
    native = dealer(cluster)
    def run(frame, callbacks):
        native.callbacks = callbacks
        role, records = 0, [(base.callback_metadata(cluster.request_id, "material", 0), b"packet")]
        if damage == "role_bool": role = False
        elif damage == "role_dealer": role = 2
        elif damage == "mixed_role": records.append((base.callback_metadata(cluster.request_id, "material", 1), b"packet"))
        elif damage == "count": records *= 65
        elif damage == "bytes":
            meta = p.metadata(records[0][0]); meta["packet_bytes_bound"] = 2 << 20
            records = [(p.canonical(meta), b"p" * p.MATERIAL_BATCH_BYTES)]
        callbacks.deliver_batch(role, records if damage == "list" else tuple(records))
    native.run = run
    monkeypatch.setattr(p, "encode_material_batch", lambda *_: pytest.fail("invalid native tuple reached encoding"))
    assert (await cluster.post("dealer", "prepare")).status_code == 400
    assert native.closed and not cluster.double.rpc_log and not cluster.double.append_threads


@pytest.mark.asyncio
async def test_genuine_oversized_material_uses_original_singleton_without_python_buffering(cluster, monkeypatch):
    cluster.double.kernels = 1
    cluster.double.admission_changes = {"largest_packet_bytes": 2 << 20,
                                       "cumulative_packet_bytes_per_party_bound": 2 << 20}
    await cluster.ready()
    native, payload = dealer(cluster), b"p" * p.MATERIAL_BATCH_BYTES
    def run(frame, callbacks):
        native.callbacks = callbacks
        for role in (0, 1):
            meta = p.metadata(base.callback_metadata(cluster.request_id, "material", role))
            meta["packet_bytes_bound"] = 2 << 20
            callbacks.deliver(p.canonical(meta), payload)
        callbacks.finish()
    native.run = run
    monkeypatch.setattr(p, "encode_material_batch", lambda *_: pytest.fail("singleton was invisibly batched"))
    await cluster.prepare()
    assert cluster.double.sealed_roles == set(p.HOLDERS)
    assert not cluster.double.batch_jobs and len(cluster.double.append_threads) == 2
    assert [up > p.MATERIAL_BATCH_BYTES for _, _, _, up, _ in cluster.double.rpc_log] == [True, True, False, False]


@pytest.mark.asyncio
async def test_complete_batch_http_body_bound_rejects_before_append(cluster):
    await cluster.ready()
    value = envelope(cluster)
    value["records"] = b"p" * p.MATERIAL_BATCH_BYTES
    raw = p.pack(value, limit=p.FRAME_LIMIT)
    assert len(raw) > p.MATERIAL_BATCH_BYTES
    response = await cluster.post("worker_a", "material-batch", raw=raw,
                                  credential=base.key("dealer", "worker_a"))
    assert response.status_code == 400 and not cluster.double.append_threads


@pytest.mark.asyncio
async def test_wrong_batch_credential_does_not_consume_valid_holder(cluster):
    await cluster.ready()
    value = envelope(cluster)
    assert (await cluster.post("worker_a", "material-batch", value,
                              credential=base.client_key("worker_a"))).status_code == 401
    assert (await send(cluster, value)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["wrong_role", "wrong_session", "wrong_plan", "sender", "sender_epoch",
    "recipient_epoch", "sequence", "sequence_bool", "count_bool", "count_zero", "count_big", "count_short",
    "count_long", "truncated", "metadata_length", "packet_bound", "wrong_record_type", "extra_field"])
async def test_batch_rejects_malformed_or_wrong_context_before_append(cluster, change):
    await cluster.ready()
    value = envelope(cluster)
    if change in {"wrong_role", "wrong_session", "wrong_plan", "packet_bound"}:
        records = list(p.decode_material_batch(value["records"], 2))
        meta = p.metadata(records[-1][0])
        if change == "wrong_role": meta["role"] = 1
        elif change == "wrong_session": meta["kernel"]["context"]["session"] = "2" * 64
        elif change == "wrong_plan": meta["kernel"]["context"]["plan"] = "2" * 64
        else: meta["packet_bytes_bound"] = 1
        records[-1] = p.canonical(meta), records[-1][1]
        value["records"] = p.encode_material_batch(tuple(records))
    elif change in ("sender", "sender_epoch", "recipient_epoch"): value[change] = "other"
    elif change == "sequence": value["sequence"] = 1
    elif change == "sequence_bool": value["sequence"] = False
    elif change == "count_bool": value["count"] = True
    elif change == "count_zero": value["count"] = 0
    elif change == "count_big": value["count"] = 65
    elif change == "count_short": value["count"] = 1
    elif change == "count_long": value["count"] = 3
    elif change == "truncated": value["records"] = value["records"][:-1]
    elif change == "metadata_length": value["records"] = value["records"][:4] + b"\xff" * 4 + value["records"][8:]
    elif change == "wrong_record_type": value["records"] = []
    else: value["extra"] = 1
    assert (await send(cluster, value)).status_code == 400
    assert not cluster.double.append_threads
    assert (await cluster.metrics("worker_a"))["active_sessions"] == 0
    assert (await send(cluster, envelope(cluster))).status_code == 409


@pytest.mark.asyncio
async def test_batch_sequences_count_original_records_and_seal_requires_all(cluster):
    await cluster.ready()
    value = envelope(cluster, count=1)
    first = await send(cluster, value)
    assert p.unpack(first.content) == {"accepted": True, "sequence": 0, "count": 1}
    seal = {k: value[k] for k in ("sender", "sender_epoch", "recipient_epoch", "sequence")}
    seal["sequence"] = 1
    assert (await cluster.post("worker_a", "seal", seal, credential=base.key("dealer", "worker_a"))).status_code == 400
    assert len(cluster.double.append_threads) == 1


@pytest.mark.asyncio
async def test_replayed_batch_burns_holder_without_second_append(cluster):
    await cluster.ready()
    value = envelope(cluster, count=1)
    assert (await send(cluster, value)).status_code == 200
    assert (await send(cluster, value)).status_code == 400
    assert len(cluster.double.append_threads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "capsule"}, {"capacity": "decoder"}], indirect=True)
async def test_reordered_originals_inside_batch_fail_exact_native_take_metadata(cluster, monkeypatch):
    await cluster.ready()
    original = base.Routing.handle_async_request
    async def swapped(self, request):
        if request.url.path.endswith("/material-batch") and request.url.port == 7300:
            value = p.unpack(await request.aread(), limit=p.MATERIAL_BATCH_BYTES)
            records = p.decode_material_batch(value["records"], value["count"])
            value["records"] = p.encode_material_batch(tuple(reversed(records)))
            raw = p.pack(value, limit=p.MATERIAL_BATCH_BYTES)
            headers = dict(request.headers)
            headers["content-length"] = str(len(raw))
            request = httpx.Request(request.method, request.url, headers=headers, content=raw)
        return await original(self, request)
    monkeypatch.setattr(base.Routing, "handle_async_request", swapped)
    await cluster.prepare()
    await cluster.inputs()
    native = next(n for n in cluster.double.natives if n.role == 0)
    assert (await cluster.post("worker_a", "start")).status_code == 200
    await lifecycle.retired(cluster, roles=("worker_a",))
    assert native.closed
    assert native.callbacks.attempt.phase == "closed"
    assert not any(row[2] == "online" for row in cluster.double.rpc_log)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["count", "count_bool", "sequence_bool", "accepted_int", "lost"])
async def test_batch_ack_requires_exact_count_types_and_never_retries(cluster, monkeypatch, failure):
    if failure == "count_bool": cluster.double.kernels = 1  # True == 1 must still fail
    await cluster.ready()
    original = base.Routing.handle_async_request
    paths = []
    async def corrupt(self, request):
        response = await original(self, request)
        paths.append(request.url.path)
        if request.url.path.endswith("/material-batch"):
            if failure == "lost": raise httpx.ReadError("lost batch ACK")
            ack = p.unpack(response.content)
            if failure == "count": ack["count"] += 1
            elif failure == "count_bool": ack["count"] = True
            elif failure == "sequence_bool": ack["sequence"] = False
            else: ack["accepted"] = 1
            return httpx.Response(200, content=p.pack(ack))
        return response
    monkeypatch.setattr(base.Routing, "handle_async_request", corrupt)
    assert (await cluster.post("dealer", "prepare")).status_code == 400
    assert len(paths) == 1 and paths[0].endswith("/material-batch")
    native = dealer(cluster)
    assert native.closed
    with pytest.raises(ValueError):
        await asyncio.to_thread(native.callbacks.deliver_batch, 0, cluster.double.batches[0][1])
    assert len(paths) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
async def test_client_lost_batch_ack_drains_all_three_native_source_and_storage_owners(cluster, client_route, monkeypatch):
    original = base.Routing.handle_async_request
    sent = []
    async def lost(self, request):
        response = await original(self, request)
        if request.url.path.endswith("/material-batch"):
            sent.append(request.url.path)
            raise httpx.ReadError("lost accepted batch ACK")
        return response
    monkeypatch.setattr(base.Routing, "handle_async_request", lost)
    with pytest.raises(transport.ProtocolError):
        await asyncio.wait_for(asyncio.to_thread(transport.execute_request,
            base.configuration("decoder"), client_route.source, [1, 2, 3], 3,
            {role: (base.ORIGINS[role], base.client_key(role)) for role in p.ROLES},
            60, client_route.attempt), 10)
    assert len(sent) == 1
    assert all(native.closed for native in cluster.double.natives)
    assert all(loaded.closed for loaded in cluster.double.loaded)
    assert all(row["closed"] for row in cluster.double.reservations + cluster.double.storage_reservations)
    assert all(owner.closed for owner in client_route.peers + client_route.clients)


@pytest.mark.asyncio
async def test_partial_batch_append_failure_retires_file_and_has_no_ack_or_retry(cluster, monkeypatch):
    await cluster.ready()
    append, calls, owners = worker.MaterialTape.append, [], []
    def fail(owner, meta, packet):
        calls.append(packet)
        owners.append(owner)
        if len(calls) == 2: raise OSError("private partial write error")
        return append(owner, meta, packet)
    monkeypatch.setattr(worker.MaterialTape, "append", fail)
    assert (await send(cluster, envelope(cluster))).status_code == 400
    assert len(calls) == 2 and len(cluster.double.batch_jobs) == 1
    assert owners[0].receipt()["records_written"] == 1
    assert all(owner.closed and owner._file is None for owner in owners)
    assert (await send(cluster, envelope(cluster))).status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "capsule"}, {"capacity": "decoder"}], indirect=True)
async def test_cancel_partial_batch_append_waits_for_thread_before_refunding_memory_or_disk(cluster, monkeypatch):
    await cluster.ready()
    entered, release = threading.Event(), threading.Event()
    append, owners = worker.MaterialTape.append, []
    def blocked(owner, meta, packet):
        owners.append(owner)
        append(owner, meta, packet)
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(worker.MaterialTape, "append", blocked)
    receiving = asyncio.create_task(send(cluster, envelope(cluster)))
    cancellation = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        # Shared legacy fixture labels reservation creation with its last startup
        # role; admission order identifies the first still-live holder owner.
        held = next(row for row in cluster.double.reservations if not row["closed"])
        storage = [row for row in cluster.double.storage_reservations if row["role"] == "worker_a"]
        cancellation = asyncio.create_task(cluster.post("worker_a", "cancel"))
        await asyncio.sleep(.02)
        assert not receiving.done() and not cancellation.done()
        assert not held["closed"] and all(not row["closed"] for row in storage)
    finally:
        release.set()
    results = await asyncio.wait_for(asyncio.gather(receiving, cancellation, return_exceptions=True), 5)
    assert results[1].status_code == 200
    assert held["closed"] and all(row["closed"] for row in storage)
    assert len(owners) == 1 and owners[0].closed  # cancellation stops before record two


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
async def test_expiry_during_batch_keeps_fixed_deadline_and_retires_partial_tape(cluster, monkeypatch, count):
    clock = SimpleNamespace(now=1000, perf_counter_ns=time.perf_counter_ns, process_time_ns=time.process_time_ns)
    clock.monotonic = lambda: clock.now
    monkeypatch.setattr(worker, "time", clock)
    await cluster.ready()
    append, owners = worker.MaterialTape.append, []
    def expire(owner, meta, packet):
        append(owner, meta, packet)
        owners.append(owner)
        clock.now = 1301
    monkeypatch.setattr(worker.MaterialTape, "append", expire)
    assert (await send(cluster, envelope(cluster, count=count))).status_code == 400
    assert len(owners) == 1 and owners[0].closed
    assert owners[0].receipt()["records_written"] == 1
    assert (await cluster.metrics("worker_a"))["active_sessions"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "capsule"}, {"capacity": "decoder"}], indirect=True)
async def test_retained_batch_callback_drains_transport_payload_before_retirement(cluster, monkeypatch):
    await cluster.ready()
    original = base.Routing.handle_async_request
    entered, release = asyncio.Event(), asyncio.Event()
    async def stalled(self, request):
        response = await original(self, request)
        if request.url.path.endswith("/material-batch"):
            entered.set()
            while not release.is_set():
                try: await release.wait()
                except asyncio.CancelledError: pass
        return response
    monkeypatch.setattr(base.Routing, "handle_async_request", stalled)
    preparing = asyncio.create_task(cluster.post("dealer", "prepare"))
    cancellation = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        callbacks = dealer(cluster).callbacks
        storage = [row for row in cluster.double.storage_reservations if not row["closed"]]
        source = next(loaded for loaded in cluster.double.loaded if loaded.role == "dealer")
        cancellation = asyncio.create_task(cluster.post("dealer", "cancel"))
        await asyncio.sleep(.02)
        assert not cancellation.done() and not dealer(cluster).closed
        assert not source.closed and all(not row["closed"] for row in storage)
        assert callbacks.attempt.callback_pending
        assert any(not row["closed"] for row in cluster.double.reservations if row["role"] == "dealer")
        with pytest.raises(ValueError):
            await asyncio.to_thread(callbacks.deliver_batch, 1, cluster.double.batches[0][1])
    finally:
        release.set()
    await asyncio.wait_for(asyncio.gather(preparing, cancellation, return_exceptions=True), 5)
    assert dealer(cluster).closed and not callbacks.attempt.callback_pending
    assert source.closed
    assert sum(row["closed"] for row in storage) == (1 if cluster.double.capacity == "decoder" else 0)
    with pytest.raises(ValueError):
        await asyncio.to_thread(callbacks.deliver_batch, 0, cluster.double.batches[0][1])


def test_binary_framing_limits_preserve_original_bytes_and_do_not_relax_msgpack_arrays():
    records = tuple((b"m", b"p") for _ in range(64))
    raw = p.encode_material_batch(records)
    assert raw == (b"\0\0\0\x06\0\0\0\x01mp") * 64
    assert p.decode_material_batch(raw, 64) == records
    with pytest.raises(ValueError): p.unpack(p.pack({"records": [b"p"] * 64}))
    exact = ((b"m", b"p" * (p.MATERIAL_BATCH_RECORD_BYTES - 9)),)
    assert len(p.encode_material_batch(exact)) == p.MATERIAL_BATCH_RECORD_BYTES
    assert p.decode_material_batch(p.encode_material_batch(exact), 1) == exact
    with pytest.raises(ValueError): p.encode_material_batch(((b"m", exact[0][1] + b"p"),))
    for bad in ((), records + ((b"m", b"p"),), list(records), ((b"", b"p"),), ((b"m", b""),), ((bytearray(b"m"), b"p"),)):
        with pytest.raises(ValueError): p.encode_material_batch(bad)
    for bad in (raw[:-1], raw + b"x", b"\xff" * 4, raw[:4] + b"\xff" * 4 + raw[8:]):
        with pytest.raises((ValueError, ProtocolError)): p.decode_material_batch(bad, 64)


def test_native_batch_price_is_shared_without_changing_tape_formats_or_capsule_default():
    for capacity in ("capsule", "decoder"):
        common = {"capacity": capacity, "primitive_requests_per_party": 2, "largest_packet_bytes": 64,
                  "cumulative_packet_bytes_per_party_bound": 128, "callback_metadata_bytes_bound": 1024,
                  "guard_disk_bytes": 0, "guard_reservation_bytes": 1024, "combined_reservation_bytes": 64 << 20}
        single_contract = p.canonical({"capacity": capacity})
        batch_contract = p.canonical({"capacity": capacity, "material_delivery": "batch-v1"})
        single = transport.price_transport(common, single_contract)
        batch = transport.price_transport({**common, **batch_admission()}, batch_contract)
        assert single.material_delivery == "single" and single.python_batch_reservation_bytes == 0
        assert batch.material_delivery == "batch-v1"
        assert batch.tape == single.tape
        assert batch.transport_disk_bytes_per_holder == single.transport_disk_bytes_per_holder
        assert batch.python_batch_reservation_bytes == p.MATERIAL_BATCH_PYTHON_BYTES == 11_014_144
        assert batch.native_batch_reservation_bytes == 2 << 20
        with pytest.raises(ValueError): transport.price_transport(common, batch_contract)
        with pytest.raises(ValueError): transport.price_transport({**common, **batch_admission()}, single_contract)
        with pytest.raises(ValueError): transport.price_transport({**common, "delivery_batch": batch_admission()["delivery_batch"]}, single_contract)
    for bad in (None, True, 1, "batch", "", []):
        with pytest.raises(ValueError): transport.transport_material_delivery({"material_delivery": bad})


def test_shared_length_prefix_optional_bounds_are_incremental_and_legacy_bytes_unchanged():
    frames = [b"", b"a", b"bc"]
    raw = encode_length_prefixed(frames)
    assert raw == b"\0\0\0\0\0\0\0\x01a\0\0\0\x02bc"
    assert encode_length_prefixed(frames, max_count=3, max_total_bytes=15) == raw
    assert list(iter_length_prefixed(raw, max_count=3, max_total_bytes=15)) == frames
    assert encode_length_prefixed([], max_count=0, max_total_bytes=0) == b""
    assert list(iter_length_prefixed(b"", max_count=0, max_total_bytes=0)) == []
    for options in ({"max_count": 2}, {"max_total_bytes": 14}, {"max_count": True}, {"max_total_bytes": -1}):
        with pytest.raises(ProtocolError): encode_length_prefixed(frames, **options)
        with pytest.raises(ProtocolError): list(iter_length_prefixed(raw, **options))
    consumed = []
    def forever():
        while True:
            consumed.append(1)
            yield b"a"
    with pytest.raises(ProtocolError): encode_length_prefixed(forever(), max_count=2)
    assert len(consumed) == 3
