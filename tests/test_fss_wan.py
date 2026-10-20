"""Ordinary FSS topology must shape dealer delivery and both online peer links."""
import json

import pytest

from pllm.deployment import LinkConditions, WanConditions
from pllm.runtime.benchmark_delegated import admit_options
from pllm.runtime.native_wan import NativeWanTopology
from pllm.runtime.servers import build_roles


def test_fss_client_dealer_and_holders_use_distinct_shared_access_routes(tmp_path, monkeypatch):
    from examples.papers.fusefss_smoke import dense_batch_smoke
    experiment = dense_batch_smoke(tmp_path)
    link = LinkConditions(latency_ms=20)
    topology = build_roles(experiment, wan=WanConditions(), docker_network=link)
    routes = []
    try:
        assert isinstance(topology, NativeWanTopology)
        monkeypatch.setattr(topology._native_network, "start", lambda values: routes.extend(values))
        ports = dict(zip(topology._role_ids, (19001, 19002, 19003), strict=True))
        commands = topology._commands(ports)
        topology._source_pins = dict.fromkeys(("source_lock", "tokenizer", "checkpoint", "config"), "a" * 64)
        topology._role_credentials = {role: "api-" + role for role in ports}
        topology._peer_credentials = {(a, b): a + "-" + b for a in ports for b in ports if a != b}
        topology._role_urls = {role: "http://127.0.0.1:" + str(port) for role, port in ports.items()}
        assert {(a, b) for a, b, _, _ in routes} == {
            ("client", "dealer"), ("client", "worker_a"), ("client", "worker_b"),
            ("dealer", "worker_a"), ("dealer", "worker_b"),
            ("worker_a", "worker_b"), ("worker_b", "worker_a"),
        }
        assert len({port for _, _, port, _ in routes}) == 7
        for a, b, port, target in routes:
            command = commands[b]
            assert target == int(command[command.index("--port") + 1])
            assert target not in ports.values()
            if a != "client":
                environment = topology._environment(a)
                assert json.loads(environment["PLLM_FSS_PEERS_JSON"])[b] == [
                    f"http://127.0.0.1:{port}", topology._peer_credentials[a, b]]
        assert topology._native_network.link == link
    finally:
        topology.close()


def test_buffered_benchmark_requires_explicit_native_delay_emulation(tmp_path):
    from examples.papers.fusefss_smoke import dense_batch_smoke
    experiment = dense_batch_smoke(tmp_path)
    options = dict(backend="native", output_tokens=2, docker_network=LinkConditions(latency_ms=20))
    with pytest.raises(ValueError):
        admit_options(experiment, **options)
    admit_options(experiment, emulate_wan=True, **options)
    for link in (LinkConditions(loss_fraction=0.01), LinkConditions(bytes_per_second=1024)):
        with pytest.raises(ValueError):
            admit_options(experiment, emulate_wan=True, **(options | {"docker_network": link}))


@pytest.mark.parametrize("flag", ["--link-conditions", "--docker-network-profile"])
def test_cli_identifies_native_delay_and_access_caps(tmp_path, monkeypatch, capsys, flag):
    from pllm.cli import main
    from pllm.runtime import benchmark_cli
    from test_fss_report_timing import report, saved_run
    link = LinkConditions(latency_ms=20)
    path = tmp_path / "link.json"
    path.write_text(json.dumps(link.to_spec()))
    result = report([saved_run()])
    result["wan_readiness"] = {"emulation": {
        "inter_party_links_present": True,
        "kernel_rate_snapshots_checked": False,
        "native_stream_rate_snapshots_checked": True,
        "summary": dict.fromkeys(("end_to_end_tokens_per_second", "online_tokens_per_second",
                                  "decode_tokens_per_second"), None),
    }}
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return result
    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", run)
    main(["benchmark", "run", "--tiny", "--backend", "native", "--wan", flag, str(path)])
    assert len(calls) == 1 and calls[0]["docker_network"] == link
    assert calls[0]["emulate_wan"] is True
    text = capsys.readouterr().out
    assert "TCP-stream pacing" in text and "one-way delay: 20 ms" in text
    assert "no inter-party links" not in text
