"""Decoder storage-policy seams; no checkpoint, long run or native qualification."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import tracemalloc

import pytest

from pllm.runtime import fss_tape as tape, fss_transport as transport, fss_worker as worker
from pllm.runtime import fss_protocol as p
from pllm.runtime.client import ProtocolError
from pllm.deployment import resources as disk_resources
from pllm.deployment.network import NetworkError
from test_fss_worker import (Loaded, ORIGINS, client_key, cluster, configuration, connections, credentials, double)
from test_fss_http_lifecycle import client_route


CONTRACT = p.canonical({"capacity": "decoder", "source_lock": "a" * 64})
REQUEST = "12" * 32
RECORDS = 3_530_000


def admission(**changes):
    return {"capacity": "decoder", "primitive_requests_per_party": RECORDS,
        "largest_packet_bytes": 512, "cumulative_packet_bytes_per_party_bound": RECORDS * 512,
        "callback_metadata_bytes_bound": 1024, "guard_disk_bytes": 256 << 20,
        "guard_reservation_bytes": 64 << 10, "combined_reservation_bytes": 64 << 20, **changes}


def test_decoder_public_full_record_price_and_separate_native_guard(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail("public pricing created a file/key")
    monkeypatch.setattr(tape.tempfile, "TemporaryFile", forbidden)
    monkeypatch.setattr(tape.AESGCM, "generate_key", forbidden)
    resources = transport.price_transport(admission(), CONTRACT)
    price = resources.tape
    assert price.records == RECORDS and price.total_packet_bytes > 1 << 30
    assert price.metadata_storage == "external_aad"
    assert price.metadata_bytes_per_record == 1024 and price.stored_metadata_bytes_per_record == 0
    assert price.max_file_bytes == RECORDS * (512 + 24 + 16)
    assert (1 << 30) < price.max_file_bytes < 32 << 30
    assert resources.guard_disk_bytes == 256 << 20
    assert resources.transport_disk_bytes_per_holder == price.max_file_bytes + (256 << 20)
    assert resources.required_disk_free_bytes_per_holder == resources.transport_disk_bytes_per_holder + tape.DISK_HEADROOM_BYTES
    small = tape.MaterialTape.price(records=1, packet_bytes=512, total_packet_bytes=512,
                                    capacity="decoder", metadata_bytes_per_record=1024)
    for field in ("max_internal_buffer_bytes", "caller_input_bytes", "max_returned_packet_bytes"):
        assert getattr(price, field) == getattr(small, field)
    with pytest.raises(tape.MaterialTapeError):
        tape.MaterialTape.price(records=RECORDS, packet_bytes=512, total_packet_bytes=RECORDS * 512)


@pytest.mark.parametrize("limits", [
    {"capacity": "other"}, {"capacity": True}, {"metadata_bytes_per_record": 4097},
    {"metadata_bytes_per_record": True}, {"metadata_bytes_per_record": 0},
    {"packet_bytes": (16 << 20) + 1}, {"total_packet_bytes": (32 << 30) + 1},
    {"records": 1 << 32}, {"records": 10**100},
    {"records": (32 << 30) // (1024 + 40 + 1) + 1,
     "packet_bytes": 1, "total_packet_bytes": (32 << 30) // (1024 + 40 + 1) + 1},
    {"records": ((32 << 30) - (2 << 30)) // (1024 + 40) + 1,
     "packet_bytes": 16 << 20, "total_packet_bytes": 2 << 30},
])
def test_decoder_hard_caps_and_overflow_before_file_key(monkeypatch, tmp_path, limits):
    def forbidden(*args, **kwargs): pytest.fail("invalid decoder price reached file/key")
    monkeypatch.setattr(tape.tempfile, "TemporaryFile", forbidden)
    monkeypatch.setattr(tape.AESGCM, "generate_key", forbidden)
    with pytest.raises(tape.MaterialTapeError):
        tape.MaterialTape(REQUEST, "worker_a", directory=tmp_path, **{
            "records": RECORDS, "packet_bytes": 512, "total_packet_bytes": RECORDS * 512,
            "capacity": "decoder", "metadata_bytes_per_record": 1024, **limits})


def test_decoder_file_ceiling_exact_boundary():
    count = (32 << 30) // (1024 + 40 + 1)
    price = tape.MaterialTape.price(records=count, packet_bytes=1, total_packet_bytes=count,
                                   capacity="decoder", metadata_bytes_per_record=1024)
    assert price.max_file_bytes <= 32 << 30 < price.max_file_bytes + 1065
    assert count < 1 << 32  # v1 binding count and zero-based ordinal both fit


def test_multi_million_record_owner_creation_and_cancel_remain_constant_space(tmp_path, monkeypatch):
    monkeypatch.setattr(tape.shutil, "disk_usage", lambda _: SimpleNamespace(free=64 << 30))
    owner = None
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        owner = tape.MaterialTape(REQUEST, "worker_a", records=RECORDS, packet_bytes=512,
            total_packet_bytes=RECORDS * 512, capacity="decoder", metadata_bytes_per_record=1024,
            directory=tmp_path)
        for _ in range(3): owner.append(b"m", b"p")
        owner.cancel()
        # Includes construction. No record-sized index, metadata vector or nonce set.
        assert tracemalloc.get_traced_memory()[1] - baseline < 128 << 10
    finally:
        tracemalloc.stop()
        if owner is not None: owner.close()
    assert owner._file is owner._aead is owner._binding is None
    assert owner.receipt()["records_written"] == 3
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(tape.MaterialTapeError): owner.seal()


def test_decoder_policy_preserves_v1_bytes_and_native_metadata_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(tape.AESGCM, "generate_key", lambda **_: b"k" * 32)
    from test_fss_tape import bytes_on_disk, overwrite
    with tape.MaterialTape(REQUEST, "worker_a", records=1, packet_bytes=1,
            total_packet_bytes=1, directory=tmp_path) as capsule:
        capsule.append(b"m", b"p"); capsule.seal()
        with tape.MaterialTape(REQUEST, "worker_a", records=1, packet_bytes=1, total_packet_bytes=1,
                capacity="decoder", metadata_bytes_per_record=1024, directory=tmp_path) as decoder:
            decoder.append(b"m", b"p"); decoder.seal()
            assert bytes_on_disk(decoder) == bytes_on_disk(capsule)
            forged = bytearray(bytes_on_disk(decoder))
            forged[:4] = (1025).to_bytes(4, "big")
            overwrite(decoder, forged)
            with pytest.raises(tape.MaterialTapeError, match="geometry"):
                decoder.take(b"m")
            assert decoder.closed
        assert capsule.take(b"m") == b"p"
        capsule.finish()
    with tape.MaterialTape(REQUEST, "worker_a", records=1, packet_bytes=1, total_packet_bytes=1,
            capacity="decoder", metadata_bytes_per_record=1024, directory=tmp_path) as decoder:
        decoder._aead = SimpleNamespace(encrypt=lambda *_: pytest.fail("metadata overflow reached encryption"))
        with pytest.raises(tape.MaterialTapeError): decoder.append(b"m" * 1025, b"p")


@pytest.mark.parametrize("field,value", [("capacity", None), ("guard_disk_bytes", None),
    ("guard_disk_bytes", True), ("guard_disk_bytes", (32 << 30) + 1),
    ("guard_reservation_bytes", None), ("guard_reservation_bytes", (64 << 20) + 1),
    ("combined_reservation_bytes", (64 << 20) + 1)])
def test_decoder_native_price_requires_explicit_capacity_and_separate_guard_bounds(field, value):
    raw = admission(**{field: value})
    if value is None: raw.pop(field)
    with pytest.raises(ValueError): transport.price_transport(raw, CONTRACT)


def test_decoder_startup_uses_shared_owner_without_caller_hook(double, monkeypatch):
    double.capacity, double.making_role = "decoder", "worker_a"
    monkeypatch.setattr(worker, "load_role_source", lambda *_, **__: pytest.fail("startup imported source"))
    worker.create_fss_worker_app(configuration("decoder"), "worker_a", client_key("worker_a"),
        peer_connections=connections("worker_a"), peer_credentials=credentials("worker_a"))
    assert not double.reservations and not double.loaded


@pytest.mark.parametrize("capacity,ttl", [("capsule", 601), ("decoder", 86401),
    ("decoder", True), ("decoder", 0), ("decoder", float("nan")), ("decoder", float("inf"))])
def test_capacity_deadline_rejection_precedes_source(double, monkeypatch, capacity, ttl):
    monkeypatch.setattr(worker, "resolve_client_source", lambda *_, **__: pytest.fail("invalid lifetime reached source"))
    with pytest.raises(ValueError, match="lifetime"):
        worker.create_fss_worker_app(configuration(capacity), "worker_a", client_key("worker_a"),
            peer_connections=connections("worker_a"), peer_credentials=credentials("worker_a"),
            ttl=ttl)


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder", "ttl": 3600}], indirect=True)
async def test_decoder_small_client_lifecycle_explicit_timeout_and_reserved_cleanup(cluster, client_route, monkeypatch):
    finished, receipt = {}, transport._receipt
    def capture(raw, role, **options):
        result = receipt(raw, role, **options)
        if role in p.HOLDERS: finished[role] = result[2]
        return result
    monkeypatch.setattr(transport, "_receipt", capture)
    result = await asyncio.wait_for(asyncio.to_thread(transport.execute_request,
        configuration("decoder"), client_route.source, [1, 2, 3], 3,
        {r: (ORIGINS[r], client_key(r)) for r in p.ROLES},
        3600, client_route.attempt), 5)
    assert result.selected == (5, 6, 7)
    assert set(finished) == set(p.HOLDERS)
    for storage in finished.values():
        assert storage["records_read"] == storage["records_written"] == 2
        assert (storage["encrypted_file_bytes_read"] == storage["encrypted_file_bytes_written"]
                == storage["sealed_file_bytes"] == storage["packet_bytes_read"] + 40 * storage["records_read"])
    assert len(cluster.double.storage_reservations) == 7  # 3 sources + 2 tapes + 2 guards
    assert all(r["closed"] for r in cluster.double.storage_reservations)
    assert "input-0-after-dealer-retirement" in client_route.events
    assert sum(r["amount"] for r in cluster.double.storage_reservations) == 3 * 4096 + 2 * (8192 + 2 * (64 + 40))


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
@pytest.mark.parametrize("changes", [{"guard_disk_bytes": 8193},
    {"guard_reservation_bytes": 4097}, {"combined_reservation_bytes": (64 << 20) - 1},
    {"primitive_requests_per_party": 3, "cumulative_packet_bytes_per_party_bound": 192},
    {"callback_metadata_bytes_bound": 1025}])
async def test_decoder_reserves_actual_native_price_after_import(cluster, changes):
    assert (await cluster.admit(roles=("worker_a",)))["worker_a"].status_code == 200
    cluster.double.admission_changes = changes
    response = await cluster.post("worker_a", "bind", {"request_frame": b"request:" + cluster.request_id.encode()})
    assert response.status_code == 200
    assert not cluster.double.native_threads
    native = cluster.double.natives[-1]
    expected = transport.price_transport(p.metadata(native.metadata()), cluster.double.loaded[-1].contract_bytes)
    assert [r["amount"] for r in cluster.double.storage_reservations] == [4096, expected.guard_disk_bytes, expected.tape.max_file_bytes]
    assert (await cluster.post("worker_a", "cancel")).status_code == 200
    assert cluster.double.natives[-1].closed and cluster.double.loaded[-1].closed
    assert cluster.double.storage_reservations[-1]["closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
async def test_remote_resource_claim_cannot_authorize_decoder_source(cluster):
    response = (await cluster.admit(roles=("worker_a",), resources=admission()))["worker_a"]
    assert response.status_code == 400
    assert not cluster.double.loaded and not cluster.double.storage_reservations


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
async def test_source_disk_pressure_rejects_before_import_or_native_files(cluster, monkeypatch):
    monkeypatch.setattr(tape.shutil, "disk_usage", lambda _: SimpleNamespace(total=1 << 30, free=0))
    monkeypatch.setattr(worker, "load_role_source", lambda *_, **__: pytest.fail("unreserved native source import"))
    response = (await cluster.admit(roles=("worker_a",)))["worker_a"]
    assert response.status_code == 400
    assert not cluster.double.loaded and not cluster.double.natives and not cluster.double.storage_reservations
    assert all(row["closed"] for row in cluster.double.reservations)


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
async def test_decoder_cancel_keeps_storage_until_native_and_source_retire(cluster):
    await cluster.ready(); await cluster.prepare(); await cluster.inputs()
    cluster.double.hold_role = "worker_a"
    assert (await cluster.post("worker_a", "start")).status_code == 200
    assert await asyncio.to_thread(cluster.double.entered.wait, 2)
    held = next(r for r in cluster.double.storage_reservations if r["role"] == "worker_a")
    cancelling = asyncio.create_task(cluster.post("worker_a", "cancel"))
    try:
        await asyncio.sleep(.02)
        assert not cancelling.done() and not held["closed"] and not held["lease"].closed
    finally:
        cluster.double.release.set()
    assert (await asyncio.wait_for(cancelling, 3)).status_code == 200
    assert held["closed"] and held["lease"].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
async def test_local_source_backing_allows_transport_but_keeps_guard_tape_pending(cluster, monkeypatch):
    ledger = disk_resources._DiskCapacity()
    monkeypatch.setattr(disk_resources, "_disk_capacity", ledger)
    headroom, guard, tape_bytes = 1 << 20, 8192, 2 * (64 + 40)
    free, events, files = [headroom + guard + tape_bytes + 4096], [], []
    block, borrowing = [False], [False]
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(disk_resources.shutil, "disk_usage",
                        lambda _: SimpleNamespace(total=64 << 20, free=free[0]))
    original_load = worker.load_role_source
    def load(*args, **kwargs):
        owner = original_load(*args, **kwargs)
        owned = tempfile.TemporaryFile(dir=tempfile.gettempdir())
        files.append(owned)
        original_close = owner.close
        def close():
            assert not borrowing[0]  # mirrors PyO3's shared/mutable borrow boundary
            owned.close()
            original_close()
        owner.close = close
        owned.write(b"x" * 4096); owned.flush(); os.fsync(owned.fileno())
        def backing():
            assert events and events[0] == "imported"
            assert not owner.closed and ledger.lock._is_owned()
            borrowing[0] = True
            try:
                if block[0]:
                    entered.set()
                    assert release.wait(5)
                info = os.fstat(owned.fileno())
                events.append("backing")
                return [{"device": info.st_dev, "file": info.st_ino, "allocated_bytes": info.st_blocks * 512}]
            finally:
                borrowing[0] = False
        owner.disk_backing = backing
        free[0] = headroom + guard + tape_bytes
        events.append("imported")
        return owner
    monkeypatch.setattr(worker, "load_role_source", load)
    assert (await cluster.admit(roles=("worker_a",)))["worker_a"].status_code == 200
    assert events[:2] == ["imported", "backing"]
    await cluster.bind(roles=("worker_a",))
    observed = ledger.observe(Path(tempfile.gettempdir()), 1)
    assert observed.backed_disk_bytes == 4096
    assert observed.pending_disk_bytes == guard + tape_bytes and observed.headroom_bytes == 0
    with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
        with disk_resources.reserve_incremental_disk(1, directory=tempfile.gettempdir()):
            pytest.fail("unwritten guard/tape reservation discarded")
    # Another local admission can sample backing while HTTP cancellation drains.
    # Source close must await that borrow without refunding any disk ownership.
    block[0] = True
    observation = asyncio.create_task(asyncio.to_thread(ledger.observe, Path(tempfile.gettempdir()), 1))
    cancellation = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        cancellation = asyncio.create_task(cluster.post("worker_a", "cancel"))
        await asyncio.sleep(.02)
        assert not cancellation.done() and borrowing[0]
        assert not cluster.double.storage_reservations[0]["lease"].closed
    finally:
        release.set()
    await asyncio.wait_for(observation, 3)
    samples = len(events)
    assert (await asyncio.wait_for(cancellation, 3)).status_code == 200
    assert len(events) == samples  # retirement must not sample a now-closed source
    assert all(f.closed for f in files) and not ledger._owners
    assert all(r["lease"].closed for r in cluster.double.storage_reservations)


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
async def test_old_core_without_backing_api_remains_fully_pending(cluster, monkeypatch):
    monkeypatch.delattr(Loaded, "disk_backing")
    assert (await cluster.admit(roles=("worker_a",)))["worker_a"].status_code == 200
    observed = disk_resources._disk_capacity.observe(Path(tempfile.gettempdir()), 1)
    assert observed.pending_disk_bytes == 4096 and observed.backed_disk_bytes == 0
    assert (await cluster.post("worker_a", "cancel")).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
@pytest.mark.parametrize("failure", ["malformed", "closed", "noncallable"])
async def test_bad_local_backing_rejects_admission_and_cleanup_does_not_sample_it(cluster, monkeypatch, failure):
    calls = []
    def backing(self):
        calls.append("sample")
        if failure == "closed": raise ValueError("native source backing unavailable")
        return [{}]
    monkeypatch.setattr(Loaded, "disk_backing", 42 if failure == "noncallable" else backing)
    assert (await cluster.admit(roles=("worker_a",)))["worker_a"].status_code == 400
    assert calls == ([] if failure == "noncallable" else ["sample"])
    assert cluster.double.loaded[-1].closed and not cluster.double.natives
    assert all(r["lease"].closed for r in cluster.double.storage_reservations)


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
@pytest.mark.parametrize("failure", ["source", "disk"])
async def test_decoder_admission_failure_drains_reserved_storage(cluster, monkeypatch, failure):
    if failure == "source": cluster.double.load_fail = True
    response = (await cluster.admit(roles=("worker_a",)))["worker_a"]
    if failure == "disk":
        assert response.status_code == 200
        monkeypatch.setattr(tape.shutil, "disk_usage", lambda _: SimpleNamespace(total=1 << 30, free=0))
        response = await cluster.post("worker_a", "bind", {"request_frame": b"request:" + cluster.request_id.encode()})
    assert response.status_code == 400
    assert cluster.double.storage_reservations[-1]["closed"]
    assert all(n.closed for n in cluster.double.loaded + cluster.double.natives)


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
async def test_decoder_cancelled_import_keeps_storage_until_late_owner_drains(cluster):
    cluster.double.load_wait = True
    task = asyncio.create_task(cluster.admit(roles=("worker_a",)))
    try:
        assert await asyncio.to_thread(cluster.double.load_entered.wait, 2)
        storage = cluster.double.storage_reservations[-1]
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done() and not storage["closed"]
    finally:
        cluster.double.load_release.set()
    with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(task, 3)
    assert storage["closed"] and cluster.double.loaded[-1].closed


@pytest.mark.asyncio
async def test_failed_native_cleanup_never_refunds_decoder_storage_or_memory():
    releases = []
    memory = SimpleNamespace(close=lambda: releases.append("memory"))
    storage = SimpleNamespace(__exit__=lambda *args: releases.append("storage"))
    attempt = worker._Attempt(memory, 300)
    attempt.disk_reservations.append(storage)
    def fail(): raise RuntimeError("cleanup failed")
    attempt.native = SimpleNamespace(cancel=lambda: None, close=fail)
    with pytest.raises(RuntimeError, match="retirement failed"):
        await attempt.close()
    assert attempt.reservation is memory and attempt.disk_reservations == [storage]
    assert releases == []


@pytest.mark.parametrize("capacity,maximum", [("capsule", 600), ("decoder", 86400)])
def test_sdk_uses_capacity_timeout_policy_without_changing_default(monkeypatch, capacity, maximum):
    from pllm.configuration import Experiment
    from pllm.runtime.fss_client import FssClient
    monkeypatch.setattr(Experiment, "resolve", lambda _: SimpleNamespace(client_runtime="compiled_fss_v1", model="unit"))
    endpoints = {role: (ORIGINS[role], client_key(role)) for role in p.ROLES}
    for options, expected in (({}, 60), ({"timeout": maximum}, maximum)):
        client = FssClient(configuration(capacity), endpoints, **options)
        assert client.timeout == expected
        client.close()
    with pytest.raises(ValueError, match="request lifetime"):
        FssClient(configuration(capacity), endpoints, timeout=maximum + 1)


def storage_reply(**changes):
    packets = RECORDS * 512
    encrypted = RECORDS * (512 + 40)
    return p.pack({"frame": b"output", "peer_bytes": p.peer_bytes(), "storage": {
        "state": "finished", "records_written": RECORDS, "records_read": RECORDS,
        "packet_bytes_written": packets, "packet_bytes_read": packets,
        "encrypted_file_bytes_written": encrypted, "encrypted_file_bytes_read": encrypted,
        "sealed_file_bytes": encrypted, "sealed_digest": "a" * 64, **changes}})


def test_decoder_large_receipt_uses_local_capacity_and_full_framing_limits():
    with pytest.raises(ProtocolError): transport._receipt(storage_reply(), "worker_a")
    _, _, storage = transport._receipt(storage_reply(), "worker_a", capacity="decoder")
    assert storage["records_read"] == RECORDS
    for encrypted in ((32 << 30) + 1, RECORDS * (512 + 40) - 1,
                      RECORDS * (512 + 40) + 1, RECORDS * (512 + 4096 + 40) + 1):
        raw = storage_reply(encrypted_file_bytes_written=encrypted,
            encrypted_file_bytes_read=encrypted, sealed_file_bytes=encrypted)
        with pytest.raises(ProtocolError): transport._receipt(raw, "worker_a", capacity="decoder")
    with pytest.raises(ProtocolError):
        transport._receipt(storage_reply(capacity="decoder"), "worker_a", capacity="decoder")


@pytest.mark.parametrize("size,accepted", [(176, True), (184, True), (185, False), (0, False)])
def test_final_frame_envelope_covers_four_tokens_plus_checked_validity(size, accepted):
    value = p.unpack(storage_reply(), limit=p.CONTROL_LIMIT)
    value["frame"] = b"x" * size
    raw = p.pack(value)
    if accepted:
        # Only envelope admission here. Native finish validates namespace,
        # count, context, token shares and the final acceptance bit.
        assert transport._receipt(raw, "worker_a", capacity="decoder")[0] == value["frame"]
    else:
        with pytest.raises(ProtocolError, match="completion frame"):
            transport._receipt(raw, "worker_a", capacity="decoder")
