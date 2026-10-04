"""Small in-container entry point and Linux cgroup/interface sampler."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import runpy
import sys
import threading


def sample():
    root = Path("/sys/fs/cgroup")
    def scalar(name):
        try:
            value = (root / name).read_text().strip()
            return int(value) if value.isdecimal() else None
        except OSError:
            return None
    try:
        cpu = dict(line.split() for line in (root / "cpu.stat").read_text().splitlines())
        cpu_ns = int(cpu["usage_usec"]) * 1000
    except (OSError, ValueError, KeyError):
        cpu_ns = None
    interfaces = {}
    for path in Path("/sys/class/net").iterdir():
        if path.name == "lo":
            continue
        try:
            interfaces[path.name] = {direction + "_bytes": int((path / "statistics" / (direction + "_bytes")).read_text())
                                     for direction in ("rx", "tx")}
        except (OSError, ValueError):
            continue
    return {"cpu_ns": cpu_ns, "memory_current_bytes": scalar("memory.current"),
            "memory_peak_bytes": scalar("memory.peak"), "interfaces": interfaces}


async def forward(stream_reader, stream_writer, host, port):
    peer_writer = None
    try:
        peer_reader, peer_writer = await asyncio.open_connection(host, port)
        async def copy(reader, writer):
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        tasks = [asyncio.create_task(copy(stream_reader, peer_writer)),
                 asyncio.create_task(copy(peer_reader, stream_writer))]
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        stream_writer.close()
        if peer_writer is not None:
            peer_writer.close()


def main():
    if sys.argv[1:] == ["--sample"]:
        print(json.dumps(sample(), sort_keys=True))
        return
    arguments = json.loads(os.environ.pop("PLLM_DOCKER_COMMAND"))
    destinations = json.loads(os.environ.pop("PLLM_DOCKER_FORWARD"))
    if arguments[:2] != ["-m", "pllm"] and arguments[:2] != ["-m", "pllm.runtime.party"]:
        raise ValueError("unsupported Docker role command")
    ready = threading.Event()
    failure = []
    def proxies():
        async def run():
            servers = []
            try:
                for value in destinations:
                    port, target, host = value["port"], value["target_port"], value["host"]
                    if (type(port) is not int or not 1 <= port <= 65535
                            or type(target) is not int or not 1 <= target <= 65535):
                        raise ValueError("invalid Docker loopback forwarding port")
                    server = await asyncio.start_server(
                        lambda reader, writer, host=host, target=target: forward(reader, writer, host, target),
                        "127.0.0.1", port)
                    servers.append(server)
                ready.set()
                await asyncio.Event().wait()
            except BaseException as error:
                failure.append(error)
                ready.set()
                raise
        asyncio.run(run())
    if destinations:
        threading.Thread(target=proxies, daemon=True).start()
        if not ready.wait(10) or failure:
            raise RuntimeError("Docker loopback forwarding failed")
    sys.argv = [arguments[1], *arguments[2:]]
    runpy.run_module(arguments[1], run_name="__main__")


if __name__ == "__main__":
    main()
