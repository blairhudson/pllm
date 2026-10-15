"""Research-only delivery fixture: native full-vector parity and bounded failure cleanup."""
import asyncio
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import threading

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/probe_boundary_head_duplex.py"
SPEC = importlib.util.spec_from_file_location("boundary_head_duplex_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


@pytest.fixture
def fixture():
    return probe.Fixture(vocabulary=33, hidden=8, stripe_rows=7)


def test_native_full_output_parity_and_conserved_directed_frames(fixture):
    async def run():
        return [await probe.run_case(fixture, mode)
                for mode in probe.MODES]
    rows = asyncio.run(run())
    assert all(row["ok"] for row in rows)
    assert len({row["full_logits_sha256"] for row in rows}) == 1
    for row in rows:
        wan = row["wan"]
        assert wan["role_parties"] == {"client": "client", **dict(zip(probe.ROLES, probe.ROLES))}
        admitted = wan["directed_stream_bytes"]
        for role, ledger in row["ledger"].items():
            assert admitted[f"client->{role}"] == ledger["request_body_bytes"] + ledger["request_envelope_bytes"]
            assert admitted[f"{role}->client"] == ledger["reply_body_bytes"] + ledger["reply_envelope_bytes"]
        client = wan["parties"]["client"]
        assert client["download"]["bytes_per_second"] == 12_500_000
        assert client["download"]["admitted_bytes"] == sum(admitted[f"{r}->client"] for r in probe.ROLES)
        assert client["upload"]["admitted_bytes"] == sum(admitted[f"client->{r}"] for r in probe.ROLES)
        assert wan["total_stream_bytes"] == sum(sum(x.values()) for x in row["ledger"].values())
        assert all(owner["retired"] and not owner["busy"] for owner in row["owners"])
        assert all(o["max_pending_native_tasks"] <= 1 for o in row["owners"])
        assert all(o["peak_reserved_output_slots"] <= row["resources"]["worker_native_output_slots_bound"]
                   for o in row["owners"])
        assert all(o["peak_reserved_output_bytes"] <= row["resources"]["worker_native_output_bytes_bound"]
                   for o in row["owners"])
    assert rows[0]["wan"]["total_stream_bytes"] == rows[1]["wan"]["total_stream_bytes"]
    assert rows[2]["wan"]["total_stream_bytes"] > rows[1]["wan"]["total_stream_bytes"]
    assert [o["native_calls"] for o in rows[2]["owners"]] == [5, 5]
    assert rows[2]["wan"]["directed_stream_bytes"] == rows[3]["wan"]["directed_stream_bytes"]
    assert [o["native_calls"] for o in rows[3]["owners"]] == [5, 5]
    assert all(o["peak_reserved_output_slots"] == 2 for o in rows[3]["owners"])
    assert all(o["peak_reserved_output_slots"] == 1 for o in rows[2]["owners"])


@pytest.mark.parametrize("mode", probe.MODES)
@pytest.mark.parametrize("fault", ["fail", "truncate", "malformed", "reorder", "context"])
def test_one_worker_failure_never_delivers_and_drains(fixture, mode, fault):
    row = asyncio.run(probe.run_case(fixture, mode, fault=fault))
    assert not row["ok"]
    assert row["full_logits_sha256"] is None
    assert all(owner["retired"] and not owner["busy"] for owner in row["owners"])


@pytest.mark.parametrize("mode", probe.MODES)
def test_cancel_during_native_call_waits_before_retirement(fixture, mode):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = fixture.full if mode in ("serial", "parallel") else fixture.stripes[0][1]

    class BlockingMatrix:
        def wrap32(self, share):
            started.set()
            assert release.wait(5), "test must release native owner"
            output = original.wrap32(share)
            finished.set()
            return output

    if mode in ("serial", "parallel"):
        fixture.full = BlockingMatrix()
    else:
        fixture.stripes[0] = (0, BlockingMatrix())

    async def run():
        task = asyncio.create_task(probe.run_case(fixture, mode))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done(), "owner retired while native call still running"
        release.set()
        return await task

    row = asyncio.run(run())
    assert finished.is_set()
    assert row["error"] == "CancelledError"
    assert row["owners"][0]["drained_native_calls"] == 1
    assert all(o["retired"] and not o["busy"] for o in row["owners"])


def test_williams_orders_balance_positions_and_predecessors():
    orders = probe.arm_orders(4)
    assert len({tuple(order) for order in orders}) == 4
    for position in range(4):
        assert {order[position] for order in orders} == set(probe.MODES)
    edges = [(a, b) for order in orders for a, b in zip(order, order[1:])]
    assert len(set(edges)) == 12
    assert probe.arm_orders(1, offset=2) == [orders[2]]


@pytest.mark.parametrize("terminate", ["cancel", "upload_failure"])
def test_pipeline_overlaps_next_native_with_upload_and_drains_both(fixture, monkeypatch, terminate):
    uploading, next_started, release, next_finished = (
        threading.Event(), threading.Event(), threading.Event(), threading.Event())
    start, original_next = fixture.stripes[1]

    class BlockingNext:
        def wrap32(self, share):
            next_started.set()
            assert release.wait(5)
            output = original_next.wrap32(share)
            next_finished.set()
            return output

    fixture.stripes[1] = (start, BlockingNext())
    original_write = probe.paced_write
    trigger_failure = asyncio.Event()

    async def held_write(writer, raw, pacer, source, destination):
        if source == probe.ROLES[0] and probe.HEADER.unpack(raw[:probe.HEADER.size])[1] == 0:
            uploading.set()
            await trigger_failure.wait()
            raise ConnectionError("injected current upload failure")
        await original_write(writer, raw, pacer, source, destination)

    monkeypatch.setattr(probe, "paced_write", held_write)

    async def run():
        task = asyncio.create_task(probe.run_case(fixture, "pipelined"))
        assert await asyncio.to_thread(uploading.wait, 5)
        assert await asyncio.to_thread(next_started.wait, 5), "next stripe never overlapped upload"
        if terminate == "cancel":
            task.cancel()
        else:
            trigger_failure.set()
        await asyncio.sleep(.01)
        assert not task.done(), "retired with next native call still active"
        release.set()
        return await task

    row = asyncio.run(run())
    assert not row["ok"] and row["full_logits_sha256"] is None
    assert next_finished.is_set()
    assert row["owners"][0]["drained_native_calls"] == 1
    assert row["owners"][0]["peak_reserved_output_slots"] == 2
    assert row["owners"][0]["native_calls"] == 2  # Current plus one next; no queue.
    assert all(o["retired"] and not o["busy"] for o in row["owners"])


def test_striped_control_has_no_upload_lookahead(fixture, monkeypatch):
    next_started = threading.Event()
    start, original_next = fixture.stripes[1]

    class Next:
        def wrap32(self, share):
            next_started.set()
            return original_next.wrap32(share)

    fixture.stripes[1] = (start, Next())
    uploads = set()
    original_write = probe.paced_write

    async def held_write(writer, raw, pacer, source, destination):
        if source in probe.ROLES:
            uploads.add(source)
            await asyncio.Event().wait()
        await original_write(writer, raw, pacer, source, destination)

    monkeypatch.setattr(probe, "paced_write", held_write)

    async def run():
        task = asyncio.create_task(probe.run_case(fixture, "striped"))
        async with asyncio.timeout(5):
            while len(uploads) < 2:
                await asyncio.sleep(.001)
        assert not next_started.is_set()
        task.cancel()
        return await task

    row = asyncio.run(run())
    assert [o["native_calls"] for o in row["owners"]] == [1, 1]
    assert all(o["peak_reserved_output_slots"] == 1 for o in row["owners"])


def test_cli_records_counterbalanced_execution_and_native_binary_identity():
    completed = subprocess.run([sys.executable, str(PATH), "--vocabulary", "9", "--hidden", "3",
                                "--stripe-rows", "4", "--repeats", "4", "--order-offset", "1"],
                               capture_output=True, text=True, check=True, timeout=30)
    document = json.loads(completed.stdout)
    assert document["arm_orders"] == probe.arm_orders(4, offset=1)
    for repetition, order in enumerate(document["arm_orders"]):
        rows = document["runs"][repetition * 4:(repetition + 1) * 4]
        assert [r["mode"] for r in rows] == order
        assert [r["order_position"] for r in rows] == list(range(4))
        assert all(r["repetition"] == repetition and r["ok"] for r in rows)
    provenance = document["provenance"]
    assert provenance["native_extension_sha256"] == hashlib.sha256(
        Path(provenance["native_extension_path"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("raw", [b"", b"x", probe.HEADER.pack(b"BHD1", 0, 0, 1, 1, 0),
                                probe.HEADER.pack(b"BHD1", 0, 0, 1, 1, 0xffffffff),
                                probe.HEADER.pack(b"BHD1", 0, 0, 1, 1, 4) + b"bad!"])
def test_truncated_malformed_and_oversized_frames(raw):
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(raw)
        reader.feed_eof()
        return await probe.read_reply(reader, role=probe.ROLES[0], session="a" * 32,
                                      stripe=0, start=0, rows=1, vocabulary=1)
    with pytest.raises((probe.BoundaryError, asyncio.IncompleteReadError)):
        asyncio.run(run())


def test_geometry_is_bounded():
    with pytest.raises(ValueError):
        probe.Fixture(33, 8, 0)
    with pytest.raises(ValueError):
        probe.Fixture(262144, 4096, 64)


def test_real_paced_sockets_obey_one_shared_client_download_budget():
    fixture = probe.Fixture(8193, 8, 1024)
    row = asyncio.run(probe.run_case(fixture, "parallel", download_mbps=1., upload_mbps=2.))
    assert row["ok"]
    wan = row["wan"]
    bucket = wan["parties"]["client"]["download"]
    # Token-bucket conservation permits one aggregate burst, not one per worker.
    assert bucket["admitted_bytes"] > 4 * wan["burst_bytes"]
    assert bucket["admitted_bytes"] <= bucket["bytes_per_second"] * wan["elapsed_seconds"] + wan["burst_bytes"]
    assert row["online"]["wall_seconds"] > .2  # Real reservation waits, not analytic timing.


def test_stalled_worker_deadline_closes_peer_and_retires(fixture):
    row = asyncio.run(probe.run_case(fixture, "parallel", fault="stall", timeout_seconds=.02))
    assert row["error"] == "TimeoutError"
    assert row["full_logits_sha256"] is None
    assert all(o["retired"] and not o["busy"] for o in row["owners"])
