"""Shared access caps, opaque duplex routing, cancellation and ledger integrity."""
import asyncio
import copy
import time
from types import SimpleNamespace
import threading

import pytest

from pllm.deployment import LinkConditions, WanConditions
from pllm.metrics.wan import _check_native_sample
from pllm.runtime.native_wan import CHUNK_BYTES, READ_CHUNK_BYTES, NativePartyNetwork, SharedPacer


@pytest.mark.parametrize("direction", ("upload", "download"))
def test_concurrent_peers_share_caps_and_directions_remain_independent(direction):
    conditions = WanConditions(download_mbps=1 if direction == "download" else 8,
                               upload_mbps=1 if direction == "upload" else 8)
    pacer = SharedPacer(conditions, ("client", "worker_a", "worker_b"))
    async def transfer(source, destination):
        if direction == "download":
            source, destination = destination, source
        for _ in range(4):
            await pacer.reserve(source, destination, CHUNK_BYTES)
    async def run():
        await asyncio.gather(transfer("client", "worker_a"), transfer("client", "worker_b"),
                             transfer("worker_a", "client"))
    start = time.monotonic()
    asyncio.run(run())
    assert time.monotonic() - start >= (8 * CHUNK_BYTES - CHUNK_BYTES) / 125000
    sample = pacer.snapshot()
    assert sample["total_stream_bytes"] == 12 * CHUNK_BYTES
    assert sample["client_stream_bytes"] == 12 * CHUNK_BYTES
    _check_native_sample(sample, conditions, pacer.assignments)
    for mutate in (
        lambda x: x["parties"]["client"]["upload"].update(bytes_per_second=1e9),
        lambda x: x.update(total_stream_bytes=0),
        lambda x: x.update(client_stream_bytes=0),
        lambda x: x.update(elapsed_seconds=0),
        lambda x: x.update(conditions_digest="wrong"),
    ):
        forged = copy.deepcopy(sample)
        mutate(forged)
        with pytest.raises(ValueError):
            _check_native_sample(forged, conditions, pacer.assignments)


def test_grouped_roles_keep_intra_party_bytes_without_charging_access_twice():
    conditions = WanConditions(role_parties=(("client", "local"), ("worker_a", "local")))
    pacer = SharedPacer(conditions, ("client", "worker_a", "worker_b"))
    async def run():
        await pacer.reserve("client", "worker_a", CHUNK_BYTES)
        await pacer.reserve("worker_a", "worker_b", CHUNK_BYTES)
    asyncio.run(run())
    sample = pacer.snapshot()
    assert sample["total_stream_bytes"] == 2 * CHUNK_BYTES
    assert sample["parties"]["local"]["upload"]["admitted_bytes"] == CHUNK_BYTES
    assert sample["parties"]["local"]["download"]["admitted_bytes"] == 0
    _check_native_sample(sample, conditions, pacer.assignments)


def test_cancelled_pacing_does_not_consume_budget_or_record_unsent_chunk():
    pacer = SharedPacer(WanConditions(upload_mbps=0.1), ("client", "inference"))
    async def run():
        task = asyncio.create_task(pacer.reserve("client", "inference", CHUNK_BYTES))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert pacer.snapshot()["total_stream_bytes"] == 0


def test_five_role_boundary_access_shares_client_budget_across_all_peers():
    roles = ("client", "inference", "preparation", "boundary_a", "boundary_b")
    conditions = WanConditions()
    pacer = SharedPacer(conditions, roles)

    async def run():
        for peer in roles[1:]:
            await pacer.reserve("client", peer, READ_CHUNK_BYTES)
            await pacer.reserve(peer, "client", READ_CHUNK_BYTES)
        await pacer.reserve("preparation", "inference", READ_CHUNK_BYTES)
        await pacer.reserve("inference", "preparation", READ_CHUNK_BYTES)

    asyncio.run(run())
    sample = pacer.snapshot()
    assert sample["client_stream_bytes"] == 8 * READ_CHUNK_BYTES
    assert sample["total_stream_bytes"] == 10 * READ_CHUNK_BYTES
    assert sample["parties"]["client"]["upload"]["admitted_bytes"] == 4 * READ_CHUNK_BYTES
    assert sample["parties"]["client"]["download"]["admitted_bytes"] == 4 * READ_CHUNK_BYTES
    _check_native_sample(sample, conditions, pacer.assignments)


def test_native_relay_duplex_bytes_and_cancellation_leave_no_owned_sockets(monkeypatch):
    network = NativePartyNetwork(WanConditions(), ("client", "inference"))
    reservations = []
    reserve = network.pacer.reserve
    async def capture(source, destination, size):
        reservations.append(size)
        await reserve(source, destination, size)
    monkeypatch.setattr(network.pacer, "reserve", capture)
    async def run():
        async def echo(reader, writer):
            try:
                while value := await reader.read(65536):
                    writer.write(value)
                    await writer.drain()
            finally:
                writer.close()
        backend = await asyncio.start_server(echo, "127.0.0.1", 0)
        port = backend.sockets[0].getsockname()[1]
        network.start([("client", "inference", 0, port)])
        routed = network.servers[0].sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", routed)
        payload = bytes(range(256)) * 400
        writer.write(payload)
        await writer.drain()
        assert await asyncio.wait_for(reader.readexactly(len(payload)), 5) == payload
        sample = network.snapshot()
        assert sample["directed_stream_bytes"] == {
            "client->inference": len(payload), "inference->client": len(payload)}
        assert sample["total_stream_bytes"] == 2 * len(payload)
        assert sample["burst_bytes"] == 16384
        assert sample["read_chunk_bytes"] == READ_CHUNK_BYTES
        assert max(reservations) == READ_CHUNK_BYTES == 8192
        network.close()  # Active keepalive must be cancelled, not left hanging.
        assert not network.thread.is_alive()
        assert await asyncio.wait_for(reader.read(), 2) == b""
        writer.close()
        backend.close()
        await backend.wait_closed()
    try:
        asyncio.run(run())
    finally:
        network.close()


def test_late_wakeups_preserve_capacity_without_enlarging_burst(monkeypatch):
    from pllm.runtime import native_wan
    # Deterministic 1 ms timer granularity; no wall-clock speed assertion.
    import math
    now = [0.0]
    monkeypatch.setattr(native_wan, "time", SimpleNamespace(monotonic=lambda: now[0]))
    async def sleep(delay):
        now[0] += math.ceil(delay * 1000) / 1000
    monkeypatch.setattr(native_wan.asyncio, "sleep", sleep)
    conditions = WanConditions()
    async def transfer(size):
        now[0] = 0.0
        pacer = SharedPacer(conditions, ("client", "inference"))
        events = [(0.0, 0)]
        for _ in range((1 << 20) // size):
            await pacer.reserve("client", "inference", size)
            events.append((now[0], pacer.links["client->inference"]))
        # Every observed window respects the same rate plus 16 KiB burst.
        for start, sent in events:
            for end, total in events:
                if end >= start:
                    assert total - sent <= (end - start) * 5_000_000 + CHUNK_BYTES + 1
        _check_native_sample(pacer.snapshot(), conditions, pacer.assignments)
        return now[0]
    legacy = asyncio.run(transfer(CHUNK_BYTES))
    current = asyncio.run(transfer(READ_CHUNK_BYTES))
    assert current >= ((1 << 20) - CHUNK_BYTES) / 5_000_000
    assert current < legacy * 0.9


def test_total_memory_uses_same_window_sum_not_sum_of_independent_peaks(monkeypatch):
    from pllm.runtime import benchmark_resources as resources
    values = {1: 100, 2: 50}
    monkeypatch.setattr(resources.os, "getpid", lambda: 1)
    monkeypatch.setattr(resources.psutil, "Process", lambda pid: SimpleNamespace(
        create_time=lambda: 0, memory_info=lambda: SimpleNamespace(rss=values[pid])))
    runtime = SimpleNamespace(_topology=SimpleNamespace(
        _process_lock=threading.Lock(), _processes={"inference": SimpleNamespace(pid=2)}))
    sampler = resources.ProcessMemorySampler(runtime)
    sampler.sample()
    values.update({1: 50, 2: 100})
    sampler.sample()
    assert sampler.peak == 150
    runtime._topology._processes["inference"].pid = 3
    sampler.sample()
    assert sampler.error


def test_delayed_relay_pipelines_bulk_bytes_and_preserves_round_trip_floor():
    link = LinkConditions(latency_ms=50)
    conditions = WanConditions()
    network = NativePartyNetwork(conditions, ("client", "inference"), link_conditions=link)

    async def run():
        async def echo(reader, writer):
            try:
                while data := await reader.read(65536):
                    writer.write(data)
                    await writer.drain()
            finally:
                writer.close()
        backend = await asyncio.start_server(echo, "127.0.0.1", 0)
        try:
            network.start([("client", "inference", 0, backend.sockets[0].getsockname()[1])])
            reader, writer = await asyncio.open_connection("127.0.0.1", network.servers[0].sockets[0].getsockname()[1])
            try:
                for _ in range(2):
                    started = time.monotonic()
                    writer.write(b"q")
                    await writer.drain()
                    assert await asyncio.wait_for(reader.readexactly(1), 5) == b"q"
                    assert time.monotonic() - started >= 0.1
                payload = bytes(range(256)) * 8192
                writer.write(payload)
                await writer.drain()
                # Per-read sleeps would take >25 s for this duplex transfer.
                assert await asyncio.wait_for(reader.readexactly(len(payload)), 5) == payload
                sample = network.snapshot()
                assert sample["total_stream_bytes"] == 2 * (len(payload) + 2)
                _check_native_sample(sample, conditions, network.pacer.assignments, link_digest=link.digest)
                for mutate in (
                    lambda s: s.update(link_conditions_digest="wrong"),
                    lambda s: s["link_conditions"].update(latency_ms=0),
                    lambda s: s.update(peak_delayed_chunks=s["delay_slots"] + 1),
                    lambda s: s.update(delayed_chunks=-1),
                ):
                    forged = copy.deepcopy(sample)
                    mutate(forged)
                    with pytest.raises(ValueError):
                        _check_native_sample(forged, conditions, network.pacer.assignments, link_digest=link.digest)
                with pytest.raises(ValueError):
                    _check_native_sample(sample, conditions, network.pacer.assignments)
            finally:
                network.close()
                assert network.delayed_chunks == 0
                assert not network.connections and not network.writers
                writer.close()
                await writer.wait_closed()
        finally:
            backend.close()
            await backend.wait_closed()
    try:
        asyncio.run(run())
    finally:
        network.close()


def test_delayed_relay_cancellation_retires_all_queued_chunks():
    from pllm.runtime.native_wan import DELAY_SLOTS
    network = NativePartyNetwork(WanConditions(), ("client", "inference"),
                                 link_conditions=LinkConditions(latency_ms=500))

    async def run():
        class Origin:
            async def read(self, _):
                return b"x" * READ_CHUNK_BYTES
        class Output:
            def write(self, _):
                pytest.fail("cancel before propagation delay")
        task = asyncio.create_task(network._delayed_copy(Origin(), Output(), "client", "inference"))
        await asyncio.sleep(0.02)
        assert network.delayed_chunks > 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert network.delayed_chunks == 0
        assert network.delay_slots._value == DELAY_SLOTS
    asyncio.run(run())


def test_native_delay_rejects_unemulated_loss_conflicting_rate_and_excess_storage():
    from pllm.runtime.servers import TopologyError
    for link in (LinkConditions(loss_fraction=0.01), LinkConditions(bytes_per_second=1024)):
        with pytest.raises(ValueError):
            NativePartyNetwork(WanConditions(), ("client", "inference"), link_conditions=link)
    with pytest.raises(TopologyError, match="storage"):
        NativePartyNetwork(WanConditions(), ("client", "inference"),
                           link_conditions=LinkConditions(latency_ms=5000))


def test_intra_party_stream_bypasses_selected_delay(monkeypatch):
    conditions = WanConditions(role_parties=(("client", "local"), ("inference", "local")))
    network = NativePartyNetwork(conditions, ("client", "inference"),
                                link_conditions=LinkConditions(latency_ms=50))
    async def reject(*args):
        pytest.fail("same-party traffic entered delay queue")
    monkeypatch.setattr(network, "_delayed_copy", reject)
    async def run():
        async def echo(reader, writer):
            data = await reader.readexactly(1)
            writer.write(data)
            await writer.drain()
            writer.close()
        backend = await asyncio.start_server(echo, "127.0.0.1", 0)
        try:
            network.start([("client", "inference", 0, backend.sockets[0].getsockname()[1])])
            reader, writer = await asyncio.open_connection("127.0.0.1", network.servers[0].sockets[0].getsockname()[1])
            writer.write(b"x")
            assert await asyncio.wait_for(reader.readexactly(1), 2) == b"x"
            writer.close()
            await writer.wait_closed()
            assert network.snapshot()["peak_delayed_chunks"] == 0
        finally:
            network.close()
            backend.close()
            await backend.wait_closed()
    asyncio.run(run())
