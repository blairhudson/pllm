"""Installed replicated Experiment through ordinary CLI and benchmark owners."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx
import psutil
import pytest

from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario, benchmark_suite
from pllm.runtime.benchmark_delegated import REPLICATED_TPS_UNAVAILABLE
from pllm.runtime.topology_accounting import REPLICATED_COVERAGE

from test_replicated_sdk import experiment_fixture, ROLES


def saved_experiment(tmp_path):
    experiment = experiment_fixture(tmp_path)
    path = tmp_path / "experiment.json"
    path.write_bytes(experiment.canonical_bytes())
    return experiment, path


@contextmanager
def gateway(path, tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = tmp_path / "gateway.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen([sys.executable, "-I", "-B", "-m", "pllm", "gateway",
            "--local", "--experiment", str(path), "--port", str(port), "--api-key", "local-puma-test"],
            stdout=log, stderr=log, start_new_session=True)
        children = []
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}",
                    headers={"Authorization": "Bearer local-puma-test"}, timeout=60, trust_env=False) as client:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    try:
                        if client.get("/healthz").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.05)
                else:
                    pytest.fail(log_path.read_text())
                children = psutil.Process(process.pid).children(recursive=True)
                assert len(children) == 3
                yield client
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            _, alive = psutil.wait_procs(children, timeout=3)
            for child in alive:
                child.kill()
            assert not alive, "gateway failed to retire its workers"


def test_cli_gateway_responses_stream_and_chat(tmp_path):
    _, path = saved_experiment(tmp_path)
    with gateway(path, tmp_path) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        model = models.json()["data"][0]["id"]
        baseline = None
        for stream in (False, True):
            response = client.post("/v1/responses", json={"model": model, "input": "hello",
                "max_output_tokens": 1, "temperature": 0, "stream": stream})
            assert response.status_code == 200, response.text
            if stream:
                events = [json.loads(line[6:]) for line in response.text.splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                assert events[-1]["type"] == "response.incomplete"
                value = events[-1]["response"]
            else:
                value = response.json()
                baseline = value["output"]
            assert value["usage"]["output_tokens"] == 1
            assert value["output"][0]["content"] == baseline[0]["content"]
        chat = client.post("/v1/chat/completions", json={"model": model,
            "messages": [{"role": "user", "content": "hello"}], "max_completion_tokens": 1,
            "temperature": 0})
        assert chat.status_code == 200, chat.text
        assert chat.json()["choices"][0]["finish_reason"] == "length"


def test_cli_benchmark_accounts_client_and_peer_links_under_shared_caps(tmp_path):
    _, path = saved_experiment(tmp_path)
    result = subprocess.run([sys.executable, "-I", "-B", "-m", "pllm", "benchmark", "run",
        "--experiment", str(path), "--prompt", "hello", "--max-output-tokens", "2",
        "--warmups", "0", "--repetitions", "1", "--backend", "native", "--wan", "--format", "json"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)["data"]["report"]
    assert report["checks"]["passed"], report["checks"]
    assert report["runtime_contract"]["output_delivery"] == "buffered_final_ids"
    assert report["summary"]["median_tokens_per_second"] is None
    assert report["runs"][0]["network"]["decode_online_body_bytes"] is None
    ledger, = report["topology_accounting"]["runs"]
    assert ledger["body_counter_coverage"] == REPLICATED_COVERAGE
    assert ledger["all_link_serialized_body_bytes"] > ledger["client_serialized_body_bytes"] > 0
    assert sum(row["serialized_body_bytes"] for row in ledger["body_bytes_by_edge"]) == ledger["all_link_serialized_body_bytes"]
    snapshot = report["native_network_accounting"]["samples"]["runs"][0]["after"]["wan_emulation"]
    for source in ROLES:
        for destination in ROLES:
            if source != destination:
                assert snapshot["directed_stream_bytes"][f"{source}->{destination}"] > 0
    assert set(report["process_cpu_accounting"]["cold_first_response_cpu_seconds_by_role"]) == {"client", *ROLES}


def test_ordinary_response_target_preserves_peer_costs_and_unknown_decode_rate(tmp_path):
    experiment, _ = saved_experiment(tmp_path)
    target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=2), backend="native")
    suite = BenchmarkSuite([BenchmarkCase("replicated-response", target)],
                          Sampling(processes=1, timeout_seconds=120, memory_bytes=3 << 30))
    report = benchmark_suite(suite, inputs={"replicated-response": {"prompt": "hello"}},
                             output=tmp_path / "benchmark.json")
    value = report.results[0].to_dict()
    assert value["status"] == "completed", value
    metrics = {entry["id"]: entry for entry in value["metrics"]}
    assert metrics["response.decode_tps"]["value"] is None
    assert metrics["response.decode_tps"]["unavailable_reason"] == REPLICATED_TPS_UNAVAILABLE
    for role in ROLES:
        assert metrics[f"{role}.cold_cpu"]["value"] > 0
        assert metrics[f"{role}.peak_rss"]["value"] > 0
    assert metrics["network.covered_body_bytes"]["value"] > 0
