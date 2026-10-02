"""Header-only Linux counters and egress shaping in one provider namespace."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .docker_role import sample


def command(arguments, *, stdin=None):
    result = subprocess.run(arguments, input=stdin, text=True, capture_output=True, timeout=10)
    if result.returncode:
        raise RuntimeError("Linux network measurement command failed: " + arguments[0] + ": " + result.stderr[:400])
    return result.stdout


def install(port, conditions):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid service port")
    rules = f"""table inet pllm {{
      set peers {{ type ipv4_addr; }}
      counter client_rx {{ }}
      counter client_tx {{ }}
      counter peer_rx {{ }}
      counter peer_tx {{ }}
      chain input {{ type filter hook input priority 0; policy accept;
        iifname "eth0" tcp dport {port} ip saddr != @peers counter name client_rx
        iifname "eth0" tcp dport {port} ip saddr @peers counter name peer_rx
      }}
      chain output {{ type filter hook output priority 0; policy accept;
        oifname "eth0" tcp sport {port} ip daddr != @peers counter name client_tx
        oifname "eth0" tcp sport {port} ip daddr @peers counter name peer_tx
      }}
    }}"""
    command(["/usr/sbin/nft", "-f", "-"], stdin=rules)
    if conditions is not None:
        from pllm.deployment import LinkConditions
        shape = LinkConditions.from_spec(conditions)
        arguments = ["/usr/sbin/tc", "qdisc", "replace", "dev", "eth0", "root", "netem",
                     "delay", f"{shape.latency_ms}ms"]
        if shape.bytes_per_second is not None:
            arguments += ["rate", f"{shape.bytes_per_second * 8}bit"]
        if shape.loss_fraction:
            arguments += ["loss", "random", f"{shape.loss_fraction * 100}%", "seed", str(shape.seed)]
        command(arguments)


def counters():
    value = json.loads(command(["/usr/sbin/nft", "-j", "list", "table", "inet", "pllm"]))
    result = {}
    for item in value["nftables"]:
        row = item.get("counter")
        if row is not None:
            result[row["name"]] = {key: int(row[key]) for key in ("bytes", "packets")}
    if set(result) != {"client_rx", "client_tx", "peer_rx", "peer_tx"}:
        raise RuntimeError("Linux network counters are incomplete")
    return {"counters": result, "measurement_resources": sample()}


def main():
    if sys.argv[1:] == ["--sample"]:
        print(json.dumps(counters(), sort_keys=True))
        return
    if sys.argv[1:2] == ["--peer"] and len(sys.argv) == 3:
        from ipaddress import IPv4Address
        address = str(IPv4Address(sys.argv[2]))
        command(["/usr/sbin/nft", "add", "element", "inet", "pllm", "peers", "{", address, "}"])
        return
    data = json.loads(os.environ.pop("PLLM_LINK_MEASUREMENT"))
    install(data["port"], data["conditions"])
    print("READY", flush=True)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
