"""Bounded native TCP-stream pacing, with shared full-duplex party budgets.

An opaque loopback relay counts each directed stream once. HTTP/WebSocket/TLS
bytes are covered; IP/TCP headers, ACKs and retransmissions are not. This is a
userspace bandwidth experiment, not kernel shaping or measured Internet latency.
"""
from __future__ import annotations

import asyncio
import threading
import time

from pllm.deployment import WanConditions
from .servers import LocalTopology, TopologyError

CHUNK_BYTES = 16 << 10
MAX_CONNECTIONS = 32


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
            "burst_bytes": CHUNK_BYTES, "max_connections": MAX_CONNECTIONS,
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
    def __init__(self, conditions, roles):
        if type(conditions) is not WanConditions:
            raise TypeError("native WAN requires WanConditions")
        self.pacer = SharedPacer(conditions, roles)
        self.loop = None
        self.thread = None
        self.servers = []
        self.connections = set()
        self.writers = set()
        self.error = None
        self.closed = False
        self.lifecycle = threading.Lock()

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
                while chunk := await origin.read(CHUNK_BYTES):
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
        async def capture():
            return self.pacer.snapshot()
        if self.loop is None:
            return self.pacer.snapshot()  # Client-only has no channels.
        return asyncio.run_coroutine_threadsafe(capture(), self.loop).result(timeout=5)

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
    __slots__ = ("_native_network", "_peer_inference_url")

    def __init__(self, *args, wan, **kwargs):
        super().__init__(*args, **kwargs)
        if self._privacy_mode not in {"public", "offset_public", "client_only"}:
            raise TopologyError("native WAN requires an admitted public role topology")
        self._native_network = NativePartyNetwork(wan, ("client", *self._role_ids))
        self._peer_inference_url = None

    def _commands(self, ports):
        excluded = set(ports.values()) | set(self._reserved_ports)
        backends = {}
        for role in ports:
            backends[role] = self._free_port(excluded)
            excluded.add(backends[role])
        routes = [("client", role, ports[role], backends[role]) for role in ports]
        if "preparation" in ports:
            peer = self._free_port(excluded)
            routes.append(("preparation", "inference", peer, backends["inference"]))
            self._peer_inference_url = f"http://127.0.0.1:{peer}"
        commands = super()._commands(backends)
        self._native_network.start(routes)
        return commands

    def _environment(self, role):
        environment = super()._environment(role)
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
