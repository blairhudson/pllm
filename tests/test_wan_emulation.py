"""Shared client/provider access caps, kernel evidence and actual token clocks."""
import asyncio
import copy
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

import pytest

from pllm.deployment import PartyAccess, WanConditions
from pllm.metrics import wan_readiness
from pllm.runtime.docker_wan_node import checked_queue, rate_parameters
from test_wan_readiness import edge, report


def kernel_sample(conditions):
    parties = {}
    for party in ("client", "a"):
        parties[party] = {}
        for direction in ("upload", "download"):
            rate = int(getattr(conditions.access(party), direction + "_mbps") * 1_000_000 / 8)
            parties[party][direction] = {"verified": True, "bytes_per_second": rate,
                "qdiscs": [{"root": True, "kind": "tbf", "options": {"rate": rate}}]}
    return {"wan_emulation": {"enforced": True, "conditions_digest": conditions.digest,
            "role_parties": {"client": "client", "a": "a"}, "parties": parties}}


def test_measured_throughput_requires_kernel_readback_and_uses_n_minus_one():
    conditions = WanConditions()
    data = report([edge("a", "client", 5_000_000, "bundle")], tokens=5)
    data["configuration"] = {"roles": ["client", "a"], "wan_emulation": True,
                             "wan_emulation_digest": conditions.digest}
    data["runs"][0]["durations"] = {"full_seconds": 10, "online_seconds": 8, "generation_seconds": 2}
    sample = kernel_sample(conditions)
    data["docker_accounting"] = {"samples": {"startup": sample, "warmups": [],
                                              "runs": [{"before": sample, "after": sample}]}}
    result = wan_readiness(data, conditions)
    assert result["summary"]["setup_inclusive"]["minimum_transfer_seconds"] == 1
    assert result["emulation"]["summary"] == {"end_to_end_tokens_per_second": 0.5,
        "online_tokens_per_second": 0.625, "decode_tokens_per_second": 2}
    assert result["measured_wan_latency_seconds"] is None  # Not a real Internet measurement.
    hypothetical = wan_readiness(data, WanConditions(upload_mbps=8))
    assert hypothetical["emulation"] is None
    assert hypothetical["summary"]["setup_inclusive"]["minimum_transfer_seconds"] == 5
    for bad in (None, {"wan_emulation": {"enforced": True}}, kernel_sample(conditions)):
        forged = copy.deepcopy(data)
        if bad is not None and "parties" in bad["wan_emulation"]:
            bad["wan_emulation"]["parties"]["client"]["upload"]["qdiscs"][0]["options"]["rate"] *= 2
        forged["docker_accounting"]["samples"]["runs"][0]["after"] = bad
        with pytest.raises(ValueError, match="WAN kernel"):
            wan_readiness(forged, conditions)
    data["runs"][0]["tokens"]["output_tokens"] = 1
    assert wan_readiness(data)["emulation"]["summary"]["decode_tokens_per_second"] is None


def test_kernel_rate_unit_and_forged_queue_rejection(monkeypatch):
    assert rate_parameters(100)["bytes_per_second"] == 12_500_000
    assert rate_parameters(40)["bytes_per_second"] == 5_000_000
    params = rate_parameters(40)
    monkeypatch.setattr("pllm.runtime.docker_wan_node.command", lambda *a, **kw: json.dumps([
        {"root": True, "kind": "tbf", "options": {"rate": 40_000_000, "burst": params["burst_bytes"]}}]))
    with pytest.raises(RuntimeError, match="kernel queue"):
        checked_queue("eth1", params)


def test_delay_loss_readback_rejects_changed_queue_and_profile():
    from pllm.deployment import LinkConditions
    from pllm.metrics.wan import _check_link_qdisc
    shape = LinkConditions(latency_ms=20, loss_fraction=0.01, seed=7).to_spec()
    queue = {"kind": "netem", "parent": "10:1", "options": {
        "delay": {"delay": 0.02, "jitter": 0, "correlation": 0},
        "loss-random": {"loss": 0.01, "correlation": 0}, "seed": 7}}
    _check_link_qdisc([queue], shape)
    for changed in ([], [queue, queue]):
        with pytest.raises(ValueError, match="WAN kernel"):
            _check_link_qdisc(changed, shape)
    with pytest.raises(ValueError, match="unselected"):
        _check_link_qdisc([queue], None)
    for section, key, value in (("delay", "delay", 0.01), ("delay", "jitter", 0.01),
                                ("loss-random", "loss", 0)):
        changed = copy.deepcopy(queue)
        changed["options"][section][key] = value
        with pytest.raises(ValueError, match="WAN kernel"):
            _check_link_qdisc([changed], shape)
    queue["options"]["seed"] = 8
    with pytest.raises(ValueError, match="WAN kernel"):
        _check_link_qdisc([queue], shape)


def test_retained_wan_cohort_preserves_outputs_and_kernel_bound_throughput():
    root = Path(__file__).resolve().parents[1] / "docs/evidence"
    cohort = [json.loads((root / f"wan-emulation-qwen25-{label}-2026-10-04.json").read_text())
              for label in ("uncapped", "100-40", "20-8")]
    assert len({item["runs"][0]["model_fingerprint"] for item in cohort}) == 1
    assert len({item["runs"][0]["generation"]["output_text_digest"] for item in cohort}) == 1
    for item in cohort:
        row = item["runs"][0]
        assert (row["tokens"]["input_tokens"], row["tokens"]["output_tokens"]) == (39, 8)
        assert row["privacy"]["plaintext_prompt_bytes_sent"] == 0
        assert row["privacy"]["plaintext_token_ids_sent"] == 0
        profile = WanConditions.from_spec(item["wan_readiness"]["conditions"])
        assert wan_readiness(item, profile) == item["wan_readiness"]
    assert cohort[0]["wan_readiness"]["emulation"] is None
    for item in cohort[1:]:
        measured = item["wan_readiness"]["emulation"]
        assert measured["kernel_rate_snapshots_checked"] is True
        assert measured["runs"][0]["decode_output_tokens"] == 7
        assert measured["summary"]["decode_tokens_per_second"] == 7 / item["runs"][0]["durations"]["generation_seconds"]


def test_cancelled_namespace_start_removes_only_owned_partial_state(monkeypatch):
    from pllm.runtime.docker_wan import DockerPartyNetwork
    from pllm.runtime.servers import TopologyError
    stop = threading.Event()
    networks = {"existing-uplink"}
    def command(arguments, **_kwargs):
        if arguments[:2] == ["network", "inspect"]:
            return json.dumps([{"IPAM": {"Config": [{"Subnet": "10.100.0.0/24"}]}}])
        if arguments[:2] == ["network", "create"]:
            networks.add(arguments[-1])
            stop.set()  # Cancellation races completion of the first allocation.
        elif arguments[:2] == ["network", "ls"]:
            return "\n".join(networks)
        elif arguments[:2] == ["network", "rm"]:
            networks.remove(arguments[-1])
        else:
            pytest.fail("cancelled startup allocated another resource")
        return ""
    monkeypatch.setattr(DockerPartyNetwork, "_command", staticmethod(command))
    network = DockerPartyNetwork(name="cancelled-run", network="existing-uplink", image="unused",
        role_urls={"a": "http://127.0.0.1:12345"}, conditions=WanConditions(), stop_event=stop)
    with pytest.raises(TopologyError, match="cancelled"):
        network.start()
    assert networks == {"existing-uplink"}
    assert not network.containers and not network.started


@pytest.fixture(scope="module")
def measurement_image():
    if os.environ.get("PLLM_RUN_DOCKER") != "1":
        pytest.skip("opt-in Docker WAN emulation")
    from pllm.runtime.docker_roles import ensure_image, ensure_measurement_image
    runtime, identifier = ensure_image(os.environ.get("PLLM_DOCKER_TEST_IMAGE"))
    return ensure_measurement_image(runtime, identifier)


def ports():
    sockets = []
    try:
        for _ in range(2):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


_ECHO = """import asyncio, sys
async def handle(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()
async def main():
    server = await asyncio.start_server(handle, '0.0.0.0', int(sys.argv[1]))
    async with server:
        await server.serve_forever()
asyncio.run(main())
"""


async def transfer(port, size):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    block = b"x" * 65536
    async def send():
        remaining = size
        while remaining:
            data = block[:remaining]
            writer.write(data)
            await writer.drain()
            remaining -= len(data)
    async def receive():
        remaining = size
        while remaining:
            data = await reader.read(min(65536, remaining))
            assert data and data == block[:len(data)]
            remaining -= len(data)
    try:
        await asyncio.wait_for(asyncio.gather(send(), receive()), timeout=45)
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.integration
@pytest.mark.parametrize("bottleneck", ["upload", "download", "grouped_providers"])
def test_concurrent_flows_share_one_access_budget_and_cleanup(measurement_image, bottleneck):
    from pllm.runtime.docker_roles import _docker
    from pllm.runtime.docker_wan import DockerPartyNetwork
    conditions = WanConditions(download_mbps=80, upload_mbps=80,
        parties=(PartyAccess("providers", 80, 8),) if bottleneck == "grouped_providers" else (
            PartyAccess("client", 8 if bottleneck == "download" else 80,
                        8 if bottleneck == "upload" else 80),),
        role_parties=(("a", "providers"), ("b", "providers")) if bottleneck == "grouped_providers" else ())
    name = "pllm-wan-test-" + secrets.token_hex(8)
    _docker(["network", "create", "--label", "pllm.benchmark=true", name])
    exposed = ports()
    network = DockerPartyNetwork(name=name, network=name, image=measurement_image,
        role_urls={role: f"http://127.0.0.1:{port}" for role, port in zip(("a", "b"), exposed)},
        conditions=conditions)
    try:
        network.start()
        for role in ("a", "b"):
            _docker(["exec", "--detach", network.namespace(role), "/opt/pllm/.venv/bin/python",
                     "-c", _ECHO, str(network.backend_ports[role])])
        async def wait_ready():
            for port in exposed:
                for attempt in range(30):
                    try:
                        await transfer(port, 1)
                        break
                    except (OSError, AssertionError):
                        if attempt == 29:
                            raise
                        await asyncio.sleep(0.1)
        asyncio.run(wait_ready())
        before = network.resource_samples()
        size = 4 * 1024 * 1024
        async def run():
            await asyncio.gather(*(transfer(port, size) for port in exposed))
        started = time.monotonic()
        asyncio.run(run())
        elapsed = time.monotonic() - started
        after = network.resource_samples()
        # Two independent 8 Mbps allowances would finish near 4.2 s. The shared
        # 8 Mbps access port needs at least ~8.4 s, including either direction.
        assert 2 * size / 1_000_000 * 0.95 <= elapsed < 30
        assert after["enforced"] and after["conditions_digest"] == conditions.digest
        assert after["directed_ip"]["bytes"] - before["directed_ip"]["bytes"] >= 4 * size
        assert after["parties"]["client"]["upload"]["verified"]
        print(json.dumps({"bottleneck": bottleneck, "elapsed_seconds": elapsed,
                          "two_way_payload_bytes": 4 * size}, sort_keys=True))
    finally:
        network.close()
        _docker(["network", "rm", name])
    for node in network.containers:
        assert node not in _docker(["ps", "--all", "--format", "{{.Names}}"])


@pytest.mark.integration
def test_cancelled_transfer_releases_owned_namespaces(measurement_image):
    # Exercise teardown after a partial transfer; no roles or helper queues can
    # survive a cancelled benchmark to interfere with the next experiment.
    from pllm.runtime.docker_roles import _docker
    from pllm.runtime.docker_wan import DockerPartyNetwork
    name = "pllm-wan-cancel-" + secrets.token_hex(8)
    _docker(["network", "create", "--label", "pllm.benchmark=true", name])
    port = ports()[0]
    network = DockerPartyNetwork(name=name, network=name, image=measurement_image,
        role_urls={"a": f"http://127.0.0.1:{port}"}, conditions=WanConditions(1, 1))
    try:
        network.start()
        _docker(["exec", "--detach", network.namespace("a"), "/opt/pllm/.venv/bin/python",
                 "-c", _ECHO, str(network.backend_ports["a"])])
        async def cancel():
            task = asyncio.create_task(transfer(port, 64 * 1024 * 1024))
            await asyncio.sleep(0.5)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        asyncio.run(cancel())
    finally:
        network.close()
        _docker(["network", "rm", name])
    names = _docker(["ps", "--all", "--format", "{{.Names}}"])
    assert all(node not in names for node in network.containers)


@pytest.mark.integration
def test_shared_caps_compose_with_measured_round_trip_delay(measurement_image):
    import statistics
    from pllm.deployment import LinkConditions
    from pllm.runtime.docker_roles import _docker
    from pllm.runtime.docker_wan import DockerPartyNetwork
    name = "pllm-wan-rtt-" + secrets.token_hex(8)
    _docker(["network", "create", "--label", "pllm.benchmark=true", name])
    port = ports()[0]
    delay = LinkConditions(latency_ms=20)
    network = DockerPartyNetwork(name=name, network=name, image=measurement_image,
        role_urls={"a": f"http://127.0.0.1:{port}"}, conditions=WanConditions(),
        link_conditions=delay)
    try:
        network.start()
        _docker(["exec", "--detach", network.namespace("a"), "/opt/pllm/.venv/bin/python",
                 "-c", _ECHO, str(network.backend_ports["a"])])
        async def run():
            for attempt in range(30):
                try:
                    await transfer(port, 1)
                    break
                except (OSError, AssertionError):
                    if attempt == 29:
                        raise
                    await asyncio.sleep(0.1)
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            samples = []
            try:
                for index in range(7):
                    started = time.monotonic()
                    writer.write(b"x")
                    await writer.drain()
                    assert await asyncio.wait_for(reader.readexactly(1), 3) == b"x"
                    if index:
                        samples.append(time.monotonic() - started)
            finally:
                writer.close()
                await writer.wait_closed()
            return samples
        samples = asyncio.run(run())
        measured = statistics.median(samples)
        assert 0.038 <= measured < 0.25
        snapshot = network.resource_samples()
        assert snapshot["link_conditions"] == delay.to_spec()
        print(json.dumps({"selected_egress_delay_ms": 20, "median_round_trip_seconds": measured}))
    finally:
        network.close()
        _docker(["network", "rm", name])
