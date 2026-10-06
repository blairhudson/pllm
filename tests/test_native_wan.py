"""Shared access caps, opaque duplex routing, cancellation and ledger integrity."""
import asyncio
import copy
import time
from types import SimpleNamespace
import threading

import pytest

from pllm.deployment import WanConditions
from pllm.metrics.wan import _check_native_sample
from pllm.runtime.native_wan import CHUNK_BYTES, NativePartyNetwork, SharedPacer


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


def test_native_relay_duplex_bytes_and_cancellation_leave_no_owned_sockets():
    network = NativePartyNetwork(WanConditions(), ("client", "inference"))
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
