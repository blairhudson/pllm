"""Calibrate bounded native TCP pacing against its configured shared access rate.

Fresh processes transfer public bytes through the real relay. Read size changes
only; access rates, 16 KiB burst, connection and socket-buffer bounds stay fixed.
This measures the local rate emulator, not a decoder or Internet connection.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def transfer(read_bytes, geometry, payload_bytes):
    from pllm.deployment import WanConditions
    from pllm.metrics.wan import _check_native_sample
    from pllm.runtime import native_wan

    if not hasattr(native_wan, "READ_CHUNK_BYTES"):
        raise ValueError("installed relay lacks an independently bounded read size")
    native_wan.READ_CHUNK_BYTES = read_bytes
    conditions = WanConditions()
    routes = {
        "correction": [("preparation", "inference")],
        "shared_upload": [("preparation", "inference"), ("preparation", "client")],
        "duplex": [("preparation", "inference"), ("inference", "preparation")],
    }[geometry]
    network = native_wan.NativePartyNetwork(conditions, ("client", "inference", "preparation"))
    block = bytes(range(256)) * 32
    expected = hashlib.sha256()
    for _ in range(payload_bytes // len(block)):
        expected.update(block)
    result = {}

    async def run():
        failures = []
        async def sink(reader, writer):
            try:
                digest, remaining = hashlib.sha256(), payload_bytes
                while remaining:
                    value = await reader.read(min(65536, remaining))
                    if not value:
                        raise ValueError("relay truncated public payload")
                    digest.update(value)
                    remaining -= len(value)
                writer.write(digest.digest())
                await writer.drain()
            except BaseException as error:
                failures.append(type(error).__name__)
            finally:
                writer.close()
                await writer.wait_closed()

        backend = await asyncio.start_server(sink, "127.0.0.1", 0)
        port = backend.sockets[0].getsockname()[1]
        try:
            network.start([(a, b, 0, port) for a, b in routes])
            async def send(server):
                reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
                writer.transport.set_write_buffer_limits(high=32768, low=16384)
                try:
                    for _ in range(payload_bytes // len(block)):
                        writer.write(block)
                        await writer.drain()
                    if await reader.readexactly(32) != expected.digest():
                        raise ValueError("relay changed public payload")
                finally:
                    writer.close()
                    await writer.wait_closed()

            wall, cpu = time.perf_counter(), time.process_time()
            await asyncio.wait_for(asyncio.gather(*(send(server) for server in network.servers)), 120)
            result.update(wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu)
            if failures:
                raise ValueError("relay sink failed")
            snapshot = network.snapshot()
            if snapshot.get("read_chunk_bytes") != read_bytes:
                raise ValueError("relay did not bind the selected read size")
            _check_native_sample(snapshot, conditions, network.pacer.assignments)
            expected_links = {}
            for a, b in routes:
                expected_links[a + "->" + b] = expected_links.get(a + "->" + b, 0) + payload_bytes
                expected_links[b + "->" + a] = expected_links.get(b + "->" + a, 0) + 32
            if snapshot["directed_stream_bytes"] != expected_links:
                raise ValueError("relay link accounting differs from transferred bytes")
            result["snapshot"] = snapshot
        finally:
            network.close()
            backend.close()
            await backend.wait_closed()
        if network.thread.is_alive() or network.connections or network.writers:
            raise ValueError("relay retained owned resources")

    asyncio.run(run())
    count = len(routes) * payload_bytes
    floor = payload_bytes * (2 if geometry == "shared_upload" else 1) / 5_000_000
    return {"geometry": geometry, "read_chunk_bytes": read_bytes, "burst_bytes": native_wan.CHUNK_BYTES,
        "checked_payload_bytes": count, "payload_sha256_per_stream": expected.hexdigest(),
        "access_floor_seconds": floor, "offered_capacity_utilization": floor / result["wall_seconds"],
        "combined_payload_mbps": count * 8 / result["wall_seconds"] / 1e6,
        "resources_retired": True, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("correction", "shared_upload", "duplex"))
    parser.add_argument("--read-bytes", type=int, choices=(8192, 16384), default=8192)
    parser.add_argument("--payload-mib", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.payload_mib <= 32:
        parser.error("payload bound is 1..32 MiB per stream")
    if args.child:
        print(json.dumps(transfer(args.read_bytes, args.child, args.payload_mib << 20)))
        return
    if args.output is None or args.output.exists():
        parser.error("a new output path is required")
    from pllm.runtime import native_wan
    sources = [Path(__file__), Path(native_wan.__file__)]
    records = []
    for repetition in range(3):
        for geometry in ("correction", "shared_upload", "duplex"):
            for size in ((16384, 8192) if repetition % 2 == 0 else (8192, 16384)):
                output = subprocess.check_output([sys.executable, "-m", "scripts.probe_native_pacing",
                    "--child", geometry, "--read-bytes", str(size), "--payload-mib", str(args.payload_mib)],
                    text=True, timeout=150)
                records.append({"repetition": repetition, **json.loads(output)})
    args.output.write_text(json.dumps({"schema": "pllm.native_pacing_calibration.v1", "scope": __doc__,
        "python": platform.python_version(), "platform": platform.platform(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "records": records}, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "processes": len(records)}))


if __name__ == "__main__":
    main()
