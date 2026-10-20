"""Bounded native TCP-stream pacing, with shared full-duplex party budgets.

An opaque loopback relay counts each directed stream once. HTTP/WebSocket/TLS
bytes are covered; IP/TCP headers, ACKs and retransmissions are not. This is a
userspace rate/delay experiment, not kernel shaping or measured Internet latency.
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import time

from pllm.deployment import LinkConditions, WanConditions
from .servers import LocalTopology, TopologyError

CHUNK_BYTES = 16 << 10
# Leave refill headroom when the event loop wakes later than a requested sleep.
# Reserving an entire burst discards every late refill at the bucket ceiling.
# Socket buffers and the aggregate per-party burst remain independently bounded.
READ_CHUNK_BYTES = CHUNK_BYTES // 2
MAX_CONNECTIONS = 32
# Pipelined propagation, not one sleep per serialized read. Slots charge a full
# read buffer even for a one-byte chunk, bounding both payload and queue objects.
# This fits the existing 32 MiB relay allowance alongside socket/stream buffers.
DELAY_BUFFER_BYTES = 8 << 20
DELAY_SLOTS = DELAY_BUFFER_BYTES // READ_CHUNK_BYTES


def validate_native_link(link):
    if link is not None:
        if type(link) is not LinkConditions:
            raise TypeError("native WAN delay requires LinkConditions")
        if link.bytes_per_second is not None or link.loss_fraction:
            raise ValueError("native WAN supports delay only; WanConditions owns rates, packet loss requires Docker")
    return link


class SharedPacer:
    def __init__(self, conditions, roles):
        self.conditions = conditions
        self.assignments = {role: conditions.party_for(role) for role in roles}
        self.buckets = {}
        self.links = {}
        self.started = time.monotonic()
        for party in set(self.assignments.values()):
            access = conditions.access(party)
            for direction in ("upload", "download"):
                self.buckets[party, direction] = {
                    "rate": getattr(access, direction + "_mbps") * 1e6 / 8,
                    "tokens": 0.0, "time": self.started, "bytes": 0,
                }

    async def reserve(self, source, destination, size):
        if type(size) is not int or not 0 < size <= CHUNK_BYTES:
            raise ValueError("native pacing requires a bounded chunk")
        a, b = self.assignments[source], self.assignments[destination]
        buckets = [] if a == b else [self.buckets[a, "upload"], self.buckets[b, "download"]]
        while True:
            now = time.monotonic()
            wait = 0.0
            # All reservations run on one event loop, with no yield during commit.
            for bucket in buckets:
                bucket["tokens"] = min(CHUNK_BYTES, bucket["tokens"] + (now - bucket["time"]) * bucket["rate"])
                bucket["time"] = now
                wait = max(wait, (size - bucket["tokens"]) / bucket["rate"])
            if wait <= 0:
                for bucket in buckets:
                    bucket["tokens"] -= size
                    bucket["bytes"] += size
                key = source + "->" + destination
                self.links[key] = self.links.get(key, 0) + size
                return
            await asyncio.sleep(wait)

    def snapshot(self):
        return {
            "schema": "pllm.native_wan.v1", "backend": "native-shared-tcp-pacer",
            "enforced": True, "conditions": self.conditions.to_spec(),
            "conditions_digest": self.conditions.digest, "role_parties": self.assignments,
            "burst_bytes": CHUNK_BYTES, "read_chunk_bytes": READ_CHUNK_BYTES,
            "max_connections": MAX_CONNECTIONS,
            "elapsed_seconds": time.monotonic() - self.started,
            "parties": {party: {direction: {
                "bytes_per_second": self.buckets[party, direction]["rate"],
                "admitted_bytes": self.buckets[party, direction]["bytes"],
            } for direction in ("upload", "download")} for party in sorted(set(self.assignments.values()))},
            "directed_stream_bytes": dict(self.links),
            "total_stream_bytes": sum(self.links.values()),
            "client_stream_bytes": sum(value for link, value in self.links.items()
                                       if "client" in link.split("->")),
            "scope": "rate-admitted TCP payload; HTTP/WebSocket/TLS framing included; each directed stream counted once",
            "excluded": ["IP/TCP headers, ACKs and retransmissions", "telemetry", "checkpoint distribution"],
        }


class NativePartyNetwork:
    def __init__(self, conditions, roles, *, link_conditions=None):
        if type(conditions) is not WanConditions:
            raise TypeError("native WAN requires WanConditions")
        self.pacer = SharedPacer(conditions, roles)
        self.link = validate_native_link(link_conditions)
        self.delay_seconds = self.link.latency_ms / 1000 if self.link is not None else 0
        parties = set(self.pacer.assignments.values())
        rate = min(sum(getattr(conditions.access(party), direction + "_mbps") * 1e6 / 8
                       for party in parties) for direction in ("upload", "download"))
        # Include idle reads and burst overlap, not just the bandwidth-delay product.
        required = math.ceil(rate * self.delay_seconds) + 2 * MAX_CONNECTIONS * READ_CHUNK_BYTES + len(parties) * CHUNK_BYTES
        if self.delay_seconds and required > DELAY_BUFFER_BYTES:
            raise TopologyError("native WAN bandwidth-delay product exceeds bounded relay storage")
        self.delay_slots = asyncio.BoundedSemaphore(DELAY_SLOTS)
        self.delayed_chunks = 0
        self.peak_delayed_chunks = 0
        self.loop = None
        self.thread = None
        self.servers = []
        self.connections = set()
        self.writers = set()
        self.error = None
        self.closed = False
        self.lifecycle = threading.Lock()

    async def _delayed_copy(self, origin, output, source, destination):
        queue = asyncio.Queue()

        def release():
            self.delayed_chunks -= 1
            self.delay_slots.release()

        async def receive():
            held = False
            try:
                while True:
                    await self.delay_slots.acquire()
                    held = True
                    self.delayed_chunks += 1
                    self.peak_delayed_chunks = max(self.peak_delayed_chunks, self.delayed_chunks)
                    chunk = await origin.read(READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    await self.pacer.reserve(source, destination, len(chunk))
                    queue.put_nowait((time.monotonic() + self.delay_seconds, chunk))
                    chunk = None
                    held = False
            finally:
                if held:
                    release()
                queue.put_nowait(None)

        receiving = asyncio.create_task(receive())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    await receiving
                    if output.can_write_eof():
                        output.write_eof()
                    break
                due, chunk = item
                item = None
                try:
                    remaining = due - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    output.write(chunk)
                    await output.drain()
                finally:
                    chunk = None
                    release()
        finally:
            receiving.cancel()
            await asyncio.gather(receiving, return_exceptions=True)
            while not queue.empty():
                item = queue.get_nowait()
                if item is not None:
                    item = None
                    release()

    async def _connect(self, reader, writer, source, destination, target):
        task = asyncio.current_task()
        if self.closed or len(self.connections) >= MAX_CONNECTIONS:
            writer.close()
            return
        self.connections.add(task)
        self.writers.add(writer)
        peer = None
        copies = []
        try:
            remote, peer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", target, limit=2 * CHUNK_BYTES), 5)
            self.writers.add(peer)
            for stream in (writer, peer):
                stream.transport.set_write_buffer_limits(high=2 * CHUNK_BYTES, low=CHUNK_BYTES)

            async def copy(origin, output, a, b):
                if self.delay_seconds and self.pacer.assignments[a] != self.pacer.assignments[b]:
                    await self._delayed_copy(origin, output, a, b)
                    return
                while chunk := await origin.read(READ_CHUNK_BYTES):
                    await self.pacer.reserve(a, b, len(chunk))
                    output.write(chunk)
                    await output.drain()
                if output.can_write_eof():
                    output.write_eof()
            copies = [asyncio.create_task(copy(reader, peer, source, destination)),
                      asyncio.create_task(copy(remote, writer, destination, source))]
            await asyncio.gather(*copies)
        except (ConnectionError, OSError, asyncio.TimeoutError):
            pass  # Expected during role startup and normal peer disconnects.
        except Exception as error:
            self.error = type(error).__name__
        finally:
            for copy in copies:
                copy.cancel()
            if copies:
                await asyncio.gather(*copies, return_exceptions=True)
            for stream in (writer, peer):
                if stream is not None:
                    stream.close()
                    self.writers.discard(stream)
            self.connections.discard(task)

    def start(self, routes):
        ready = threading.Event()
        failure = []

        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def start():
                for source, destination, port, target in routes:
                    server = await asyncio.start_server(
                        lambda r, w, a=source, b=destination, target=target: self._connect(r, w, a, b, target),
                        "127.0.0.1", port, limit=2 * CHUNK_BYTES)
                    self.servers.append(server)
            try:
                self.loop.run_until_complete(start())
            except BaseException as error:
                failure.append(error)
            finally:
                ready.set()
            if not failure:
                self.loop.run_forever()
            self.loop.run_until_complete(self._close())
            self.loop.close()
        self.thread = threading.Thread(target=run, name="pllm-native-wan", daemon=True)
        self.thread.start()
        if not ready.wait(10):
            self.close()
            raise TopologyError("native WAN relay startup timed out")
        if failure:
            self.close()
            raise TopologyError("native WAN relay startup failed") from failure[0]

    def snapshot(self):
        if self.error:
            raise TopologyError("native WAN relay failed: " + self.error)
        def capture():
            sample = self.pacer.snapshot()
            if self.link is not None:
                sample.update(link_conditions=self.link.to_spec(), link_conditions_digest=self.link.digest,
                              delay_scope="per-inter-party-direction propagation after rate admission; no TCP handshake emulation",
                              delay_buffer_bytes=DELAY_BUFFER_BYTES, delay_slots=DELAY_SLOTS,
                              delayed_chunks=self.delayed_chunks, peak_delayed_chunks=self.peak_delayed_chunks)
            return sample
        if self.loop is None:
            return capture()  # Client-only has no channels.
        async def on_loop():
            return capture()
        return asyncio.run_coroutine_threadsafe(on_loop(), self.loop).result(timeout=5)

    async def _close(self):
        self.closed = True
        for server in self.servers:
            server.close()
        tasks = list(self.connections)
        for task in tasks:
            task.cancel()
        for writer in list(self.writers):
            writer.close()
        await asyncio.gather(*tasks, return_exceptions=True)
        for server in self.servers:
            await server.wait_closed()

    def close(self):
        with self.lifecycle:
            if self.thread is not None and self.thread.is_alive():
                if self.loop is not None:
                    self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join(timeout=10)
                if self.thread.is_alive():
                    raise TopologyError("native WAN relay failed to stop")
            self.closed = True


class NativeWanTopology(LocalTopology):
    __slots__ = ("_native_network", "_peer_inference_url", "_replicated_peer_urls")

    def __init__(self, *args, wan, link_conditions=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self._privacy_mode not in {"public", "offset_public", "client_only", "gkm_public", "replicated_public", "fss_public"}:
            raise TopologyError("native WAN requires an admitted public role topology")
        self._native_network = NativePartyNetwork(wan, ("client", *self._role_ids), link_conditions=link_conditions)
        self._peer_inference_url = None
        self._replicated_peer_urls = {}

    def _commands(self, ports):
        excluded = set(ports.values()) | set(self._reserved_ports)
        backends = {}
        for role in ports:
            backends[role] = self._free_port(excluded)
            excluded.add(backends[role])
        routes = [("client", role, ports[role], backends[role]) for role in ports]
        if self._privacy_mode in {"replicated_public", "fss_public"}:
            for source in ports:
                for destination in ports:
                    if source == destination:
                        continue
                    if self._privacy_mode == "fss_public" and destination == "dealer":
                        continue
                    peer = self._free_port(excluded)
                    excluded.add(peer)
                    routes.append((source, destination, peer, backends[destination]))
                    self._replicated_peer_urls[source, destination] = f"http://127.0.0.1:{peer}"
        if "preparation" in ports:
            peer = self._free_port(excluded)
            routes.append(("preparation", "inference", peer, backends["inference"]))
            self._peer_inference_url = f"http://127.0.0.1:{peer}"
        commands = super()._commands(backends)
        self._native_network.start(routes)
        return commands

    def _environment(self, role):
        environment = super()._environment(role)
        if self._privacy_mode in {"replicated_public", "fss_public"}:
            key = "PLLM_FSS_PEERS_JSON" if self._privacy_mode == "fss_public" else "PLLM_REPLICATED_PEERS_JSON"
            peers = json.loads(environment[key])
            for peer, values in peers.items():
                values[0] = self._replicated_peer_urls[role, peer]
            environment[key] = json.dumps(peers)
        if role == "preparation":
            environment["PLLM_INFERENCE_URL"] = self._peer_inference_url
        return environment

    def resource_samples(self):
        return {"wan_emulation": self._native_network.snapshot()}

    def close(self):
        try:
            super().close()
        finally:
            self._native_network.close()
