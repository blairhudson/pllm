"""Owned Docker party access networks for the existing role supervisor."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from ipaddress import IPv4Network
import json
import os
import threading
import time
from urllib.parse import urlsplit

from pllm.deployment import WanConditions
from .servers import TopologyError


class DockerPartyNetwork:
    """One routed, bidirectionally shaped access port per selected party.

    Roles at a party share the endpoint namespace. A separate router namespace
    gives download its own egress queue without requiring an IFB kernel module.
    The host client reaches an opaque TCP portal over an unshaped management
    link; client/provider and provider/provider packets then use the same WAN.
    """

    def __init__(self, *, name, network, image, role_urls, conditions, link_conditions=None,
                 telemetry_port=None, stop_event=None):
        if type(conditions) is not WanConditions:
            raise TypeError("WAN emulation requires WanConditions")
        if link_conditions is not None and link_conditions.bytes_per_second is not None:
            raise ValueError("WAN emulation owns rates; Docker link rate cannot also be set")
        from .docker_wan_node import rate_parameters
        self.conditions = conditions
        self.role_parties = {role: conditions.party_for(role) for role in ("client", *role_urls)}
        self.party_ids = tuple(sorted(set(self.role_parties.values())))
        for party in self.party_ids:
            access = conditions.access(party)
            rate_parameters(access.upload_mbps)
            rate_parameters(access.download_mbps)
        self.name, self.network, self.image = name, network, image
        self.link_conditions = link_conditions
        self.telemetry_port = telemetry_port
        self.public_ports = {role: urlsplit(url).port for role, url in role_urls.items()}
        self.backend_ports = {}
        port = 10000
        for role in role_urls:
            while port in self.public_ports.values() or port == telemetry_port:
                port += 1
            self.backend_ports[role] = port
            port += 1
        self.management = name + "-management"
        self.networks = []
        self.containers = {}
        self.parties = {}
        self.started = False
        self._stopping = stop_event if stop_event is not None else threading.Event()
        self._lifecycle = threading.RLock()

    def _check_running(self):
        if self._stopping.is_set():
            raise TopologyError("WAN namespace startup was cancelled")

    @staticmethod
    def _command(*args, **kwargs):
        from .docker_roles import _docker
        return _docker(*args, **kwargs)

    def _create_network(self, name):
        self._check_running()
        # Retain ownership before an ambiguous create timeout.
        self.networks.append(name)
        self._command(["network", "create", "--label", "pllm.benchmark=true", name])
        return self._subnet(name)

    def _subnet(self, name):
        info = json.loads(self._command(["network", "inspect", name]))
        configs = info[0]["IPAM"]["Config"]
        if len(configs) != 1:
            raise TopologyError("WAN emulation requires one owned IPv4 subnet")
        network = IPv4Network(configs[0]["Subnet"])
        if network.num_addresses < 256:
            raise TopologyError("WAN subnet has insufficient bounded namespace capacity")
        return network

    def _create_node(self, name, primary_network, secondary_network, primary_ip,
                     secondary_ip, config, *, publish=False):
        self._check_running()
        from .docker_roles import _Container
        node = _Container(name, created=False)
        self.containers[name] = node
        options = ["create", "--name", name, "--label", "pllm.benchmark=true",
                   "--network", primary_network, "--cap-drop=ALL", "--cap-add=NET_ADMIN",
                   "--security-opt=no-new-privileges", "--pids-limit=64",
                   "--add-host", "host.docker.internal:host-gateway", "--env", "PLLM_WAN_NODE",
                   "--entrypoint", "/opt/pllm/.venv/bin/python"]
        if primary_ip is not None:
            options += ["--ip", primary_ip]
        if config["kind"] == "router":
            options += ["--sysctl", "net.ipv4.ip_forward=1"]
        if publish:
            for port in self.public_ports.values():
                options += ["--publish", f"127.0.0.1:{port}:{port}"]
        self._command([*options, self.image, "-m", "pllm.runtime.docker_wan_node"],
                      environment={**os.environ, "PLLM_WAN_NODE": json.dumps(config)})
        node.created = True
        self._command(["network", "connect", "--ip", secondary_ip, secondary_network, name])
        return node

    def start(self):
        with self._lifecycle:
            return self._start()

    def _start(self):
        self._check_running()
        if self.started:
            raise TopologyError("WAN emulation cannot be started twice")
        try:
            wan = self._subnet(self.network)
            self._create_network(self.management)
            for index, party in enumerate(self.party_ids):
                lan_name = f"{self.name}-lan-{index}"
                lan = self._create_network(lan_name)
                self.parties[party] = {
                    "party_id": party, "wan_ip": str(wan[16 + index]),
                    "endpoint_ip": str(lan[10]), "lan_ip": str(lan[11]),
                    "lan_network": lan_name, "endpoint": f"{self.name}-party-{index}",
                    "router": f"{self.name}-router-{index}",
                }
            peers = [{key: row[key] for key in ("party_id", "wan_ip", "endpoint_ip")}
                     for row in self.parties.values()]
            for party, row in self.parties.items():
                config = {"party_id": party, "endpoint_ip": row["endpoint_ip"],
                          "lan_ip": row["lan_ip"], "wan_ip": row["wan_ip"], "peers": peers,
                          "access": self.conditions.access(party).to_spec(),
                          "link_conditions": self.link_conditions.to_spec() if self.link_conditions else None}
                self._create_node(row["router"], self.network, row["lan_network"],
                                  row["wan_ip"], row["lan_ip"], {**config, "kind": "router"})
                client = party == self.role_parties["client"]
                proxies = [{"port": port, "host": self.address(role), "target_port": self.backend_ports[role]}
                           for role, port in self.public_ports.items()] if client else []
                if self.telemetry_port is not None:
                    proxies.append({"port": self.telemetry_port, "target_port": self.telemetry_port,
                                    "host": "host.docker.internal", "bind": "127.0.0.1"})
                self._create_node(row["endpoint"], self.management, row["lan_network"],
                                  None, row["endpoint_ip"], {**config, "kind": "endpoint", "proxies": proxies},
                                  publish=client)
            for name in self.containers:
                self._check_running()
                self._command(["start", name])
                deadline = time.monotonic() + 20
                while "READY" not in self._command(["logs", name]):
                    self._check_running()
                    if self.containers[name].poll() is not None or time.monotonic() > deadline:
                        raise TopologyError("WAN namespace/rate setup failed: " + self._command(["logs", name])[-1200:])
                    time.sleep(0.1)
            self.started = True
            return self
        except BaseException:
            self.close()
            raise

    def address(self, role):
        return self.parties[self.role_parties[role]]["endpoint_ip"]

    def namespace(self, role):
        return self.parties[self.role_parties[role]]["endpoint"]

    def same_party(self, a, b):
        return self.role_parties[a] == self.role_parties[b]

    def resource_samples(self):
        parties, helpers, links = {}, {}, {}
        for party, row in self.parties.items():
            for kind in ("endpoint", "router"):
                data = json.loads(self._command([
                    "exec", row[kind], "/opt/pllm/.venv/bin/python", "-m",
                    "pllm.runtime.docker_wan_node", "--sample"]))
                helpers[f"{party}/{kind}"] = data["resources"]
                if kind == "router":
                    parties[party] = {key: data[key] for key in ("upload", "download")}
                    for peer, counter in data["links"].items():
                        links[f"{party}->{peer}"] = counter
        return {
            "schema": "pllm.wan_emulation.v1", "backend": "linux-tbf-routed-party-ports",
            "enforced": True, "conditions": self.conditions.to_spec(),
            "conditions_digest": self.conditions.digest, "role_parties": self.role_parties,
            "scope": "shared per-party full-duplex IP access; co-located Docker namespaces",
            "excluded": ["host/portal management leg", "out-of-band telemetry", "checkpoint distribution"],
            "parties": parties, "measurement_helpers": helpers,
            "directed_ip": {"scope": "non-overlapping party-router WAN TCP postrouting hooks",
                            "links": links, "bytes": sum(row["bytes"] for row in links.values()),
                            "includes": "IP/TCP headers, ACKs, retransmissions and service controls"},
            "full_wire_bytes": None,
        }

    def close(self):
        self._stopping.set()
        with self._lifecycle:
            self._close()

    def _close(self):
        failure = None
        def remove_network(name):
            present = self._command(["network", "ls", "--filter", f"name=^{name}$", "--format", "{{.Name}}"])
            if name in present.splitlines():
                self._command(["network", "rm", name])
        # Providers have already stopped. Independent helper namespaces can be
        # retired concurrently; detach all endpoints before removing bridges.
        with ThreadPoolExecutor(max_workers=8) as pool:
            stages = ([(node.close, ()) for node in self.containers.values()],
                      [(remove_network, (name,)) for name in self.networks])
            for stage in stages:
                futures = [pool.submit(function, *arguments) for function, arguments in stage]
                for future in futures:
                    try:
                        future.result()
                    except Exception as error:
                        failure = failure or error
        if failure is not None:
            raise failure
