"""Linux namespace setup for shared party access links; payloads stay opaque.

Two egress queues implement each full-duplex access port: router WAN egress
limits upload, and router LAN egress limits download. This needs no IFB kernel
module and never changes networking in the host namespace.
"""
from __future__ import annotations

import asyncio
from functools import partial
from ipaddress import IPv4Address
import json
import math
import os
from pathlib import Path
import sys

from .docker_network import command
from .docker_role import forward, sample

_CONFIG = Path("/tmp/pllm-wan-node.json")


def device_for(address):
    address = str(IPv4Address(address))
    devices = json.loads(command(["/usr/sbin/ip", "-j", "address", "show"]))
    matches = [row["ifname"] for row in devices
               if any(info.get("local") == address for info in row.get("addr_info", []))]
    if len(matches) != 1 or matches[0] == "lo":
        raise ValueError("WAN address must identify one owned interface")
    return matches[0]


def rate_parameters(mbps):
    # Bound per-namespace queue memory even for oversized experimental profiles.
    from pllm.deployment.wan import _rate
    _rate(mbps, "WAN rate")
    rate = int(mbps * 1_000_000 / 8)
    # Ten milliseconds avoids systematic underfilling on 100 Hz schedulers.
    # The read-back report retains this finite burst allowance explicitly.
    burst = max(2048, math.ceil(rate / 100))
    if burst > 4 * 1024 * 1024:
        raise ValueError("WAN rate requires an unsupported token-bucket burst")
    limit = max(burst * 2, min(4 * 1024 * 1024, math.ceil(rate / 4)))
    return {"bytes_per_second": rate, "burst_bytes": burst, "queue_bytes": limit}


def install_rate(device, mbps, conditions=None):
    params = rate_parameters(mbps)
    command(["/usr/sbin/tc", "qdisc", "replace", "dev", device, "root", "handle", "10:",
             "tbf", "rate", str(params["bytes_per_second"] * 8) + "bit",
             "burst", str(params["burst_bytes"]), "limit", str(params["queue_bytes"])])
    if conditions is not None:
        from pllm.deployment import LinkConditions
        link = LinkConditions.from_spec(conditions)
        if link.bytes_per_second is not None:
            raise ValueError("party WAN rates cannot be combined with a separate link rate")
        args = ["/usr/sbin/tc", "qdisc", "add", "dev", device, "parent", "10:1",
                "handle", "20:", "netem", "delay", f"{link.latency_ms}ms"]
        if link.loss_fraction:
            args += ["loss", "random", f"{link.loss_fraction * 100}%", "seed", str(link.seed)]
        command(args)
    return params


def checked_queue(device, expected):
    queues = json.loads(command(["/usr/sbin/tc", "-j", "-s", "qdisc", "show", "dev", device]))
    roots = [row for row in queues if row.get("root")]
    if (len(roots) != 1 or roots[0].get("kind") != "tbf"
            or roots[0].get("options", {}).get("rate") != expected["bytes_per_second"]
            or roots[0].get("options", {}).get("burst") != expected["burst_bytes"]):
        raise RuntimeError("WAN kernel queue differs from the selected access rate")
    return {"device": device, **expected, "qdiscs": queues, "verified": True}


def install(config):
    kind = config["kind"]
    own = str(IPv4Address(config["endpoint_ip"]))
    peers = config["peers"]
    if kind not in {"endpoint", "router"} or not isinstance(peers, list) or len(peers) > 128:
        raise ValueError("invalid bounded WAN namespace configuration")
    lan_address = own if kind == "endpoint" else config["lan_ip"]
    lan = device_for(lan_address)
    config["lan_device"] = lan
    wan = None
    if kind == "router":
        if Path("/proc/sys/net/ipv4/ip_forward").read_text().strip() != "1":
            raise RuntimeError("WAN router forwarding is unavailable")
        wan = device_for(config["wan_ip"])
        config["wan_device"] = wan
    for peer in peers:
        destination = str(IPv4Address(peer["endpoint_ip"]))
        if destination == own:
            continue
        gateway = str(IPv4Address(config["lan_ip"] if kind == "endpoint" else peer["wan_ip"]))
        args = ["/usr/sbin/ip", "route", "replace", destination + "/32", "via", gateway,
                "dev", lan if kind == "endpoint" else wan]
        if kind == "endpoint":
            args += ["src", own]
        command(args)
    if kind == "router":
        from pllm.deployment.wan import PartyAccess
        access = PartyAccess.from_spec(config["access"])
        config["upload"] = install_rate(wan, access.upload_mbps, config["link_conditions"])
        config["download"] = install_rate(lan, access.download_mbps)
        definitions, rules = [], []
        for index, peer in enumerate(peers):
            if peer["endpoint_ip"] == own:
                continue
            destination = str(IPv4Address(peer["endpoint_ip"]))
            definitions.append(f"counter peer_{index} {{ }}")
            rules.append(f'oifname "{wan}" ip saddr {own} ip daddr {destination} '
                         f'meta l4proto tcp counter name peer_{index}')
        command(["/usr/sbin/nft", "-f", "-"], stdin=(
            "table inet pllm_wan {\n" + "\n".join(definitions)
            + "\nchain sent { type filter hook postrouting priority 0; policy accept;\n"
            + "\n".join(rules) + "\n}\n}"))
    _CONFIG.write_text(json.dumps(config, sort_keys=True))


def snapshot():
    config = json.loads(_CONFIG.read_text())
    result = {"resources": sample(), "party_id": config["party_id"], "kind": config["kind"]}
    if config["kind"] == "router":
        result["upload"] = checked_queue(config["wan_device"], config["upload"])
        result["download"] = checked_queue(config["lan_device"], config["download"])
        rows = json.loads(command(["/usr/sbin/nft", "-j", "list", "table", "inet", "pllm_wan"]))
        counters = {entry["counter"]["name"]: entry["counter"] for entry in rows["nftables"]
                    if "counter" in entry}
        result["links"] = {}
        for index, peer in enumerate(config["peers"]):
            if peer["endpoint_ip"] == config["endpoint_ip"]:
                continue
            counter = counters[f"peer_{index}"]
            result["links"][peer["party_id"]] = {key: int(counter[key]) for key in ("bytes", "packets")}
    return result


async def _forward(reader, writer, host, port):
    try:
        await forward(reader, writer, host, port)
    except (OSError, ConnectionError):
        writer.close()


async def serve(config):
    servers = []
    try:
        for proxy in config.get("proxies", []):
            for key in ("port", "target_port"):
                if type(proxy[key]) is not int or not 1 <= proxy[key] <= 65535:
                    raise ValueError("invalid WAN proxy port")
            bind = proxy.get("bind", "0.0.0.0")
            if bind not in {"127.0.0.1", "0.0.0.0"}:
                raise ValueError("invalid WAN proxy binding")
            host = ("host.docker.internal" if bind == "127.0.0.1" and proxy["host"] == "host.docker.internal"
                    else str(IPv4Address(proxy["host"])))
            servers.append(await asyncio.start_server(
                partial(_forward, host=host, port=proxy["target_port"]), bind, proxy["port"]))
        snapshot()  # Read back kernel rates before reporting readiness.
        print("READY", flush=True)
        await asyncio.Event().wait()
    finally:
        for server in servers:
            server.close()
        await asyncio.gather(*(server.wait_closed() for server in servers))


def main():
    if sys.argv[1:] == ["--sample"]:
        print(json.dumps(snapshot(), sort_keys=True))
        return
    config = json.loads(os.environ.pop("PLLM_WAN_NODE"))
    install(config)
    asyncio.run(serve(config))


if __name__ == "__main__":
    main()
