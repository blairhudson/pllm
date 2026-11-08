"""Installed GKM Experiment through real CLI, gateway and benchmark processes."""

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
from fastapi.testclient import TestClient

from pllm import OpenAI
from pllm.metrics import BenchmarkCase, BenchmarkRun, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario, benchmark_suite
from pllm.runtime import delegated_transport
from pllm.runtime.sidecar import create_sidecar_app

from test_delegated_sdk import active_sessions, experiment_fixture


def saved_experiment(tmp_path, *, sharing="full", execution=None):
    experiment, _, _ = experiment_fixture(tmp_path, sharing=sharing, execution=execution)
    path = tmp_path / "experiment.json"
    path.write_bytes(experiment.canonical_bytes())
    return experiment, path


def run_cli(*args):
    result = subprocess.run([sys.executable, "-I", "-B", "-m", "pllm", *map(str, args)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@contextmanager
def gateway(path, tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    headers = {"Authorization": "Bearer local-gkm-test"}
    log_path = tmp_path / "gateway.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen([sys.executable, "-I", "-B", "-m", "pllm", "gateway",
            "--local", "--experiment", str(path), "--port", str(port),
            "--api-key", "local-gkm-test"], stdout=log, stderr=log, start_new_session=True)
        children = []
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", headers=headers,
                              timeout=20, trust_env=False) as client:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    try:
                        response = client.get("/healthz")
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.05)
                else:
                    pytest.fail(log_path.read_text())
                children = psutil.Process(process.pid).children(recursive=True)
                assert len(children) == 2
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
            assert not alive, "gateway failed to retire its provider children"


@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("execution", [None, "source_stage_batch_v1"])
def test_cli_gateway_responses_stream_and_chat(tmp_path, sharing, execution):
    experiment, path = saved_experiment(tmp_path, sharing=sharing, execution=execution)
    with gateway(path, tmp_path) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        model_id = models.json()["data"][0]["id"]
        assert models.json()["data"][0]["runtime"]["sharing"] == sharing
        assert model_id == (experiment.pipeline.model.model_id or Path(experiment.pipeline.model.source).name)
        response = client.post("/v1/responses", json={"model": model_id, "input": "hello world",
            "max_output_tokens": 2, "temperature": 0})
        assert response.status_code == 200, response.text
        value = response.json()
        assert value["status"] == "incomplete"
        assert value["usage"]["output_tokens"] == 2
        streamed = client.post("/v1/responses", json={"model": model_id, "input": "hello world",
            "max_output_tokens": 2, "temperature": 0, "stream": True})
        assert streamed.status_code == 200, streamed.text
        events = [json.loads(line[6:]) for line in streamed.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        assert events[-1]["type"] == "response.incomplete"
        assert events[-1]["response"]["output"][0]["content"] == value["output"][0]["content"]
        chat = client.post("/v1/chat/completions", json={"model": model_id,
            "messages": [{"role": "user", "content": "hello world"}],
            "max_completion_tokens": 2, "temperature": 0})
        assert chat.status_code == 200, chat.text
        assert chat.json()["choices"][0]["finish_reason"] == "length"
        assert chat.json()["usage"]["completion_tokens"] == 2


@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("execution", [None, "source_stage_batch_v1"])
def test_cli_benchmark_uses_serialized_experiment_and_buffered_metrics(tmp_path, sharing, execution):
    _, path = saved_experiment(tmp_path, sharing=sharing, execution=execution)
    output = run_cli("benchmark", "run", "--experiment", path, "--prompt", "hello world",
                     "--max-output-tokens", "2", "--warmups", "0", "--repetitions", "1",
                     "--backend", "native", "--format", "json")
    result = json.loads(output)["data"]["report"]
    assert result["checks"]["passed"], result
    assert result["runtime_contract"]["output_delivery"] == "buffered_until_verified"
    assert result["runtime_contract"]["sharing"] == sharing
    assert result["summary"]["median_tokens_per_second"] is None
    assert result["runs"][0]["tokens"]["output_tokens"] == 2
    assert set(result["process_cpu_accounting"]["cold_first_response_cpu_seconds_by_role"]) == {
        "client", "worker_a", "worker_b"}


@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("execution", [None, "source_stage_batch_v1"])
def test_ordinary_response_target_owns_workers_and_keeps_decode_rate_unknown(tmp_path, sharing, execution):
    experiment, _ = saved_experiment(tmp_path, sharing=sharing, execution=execution)
    target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=2),
                                      backend="native")
    suite = BenchmarkSuite([BenchmarkCase("delegated-response", target)],
                           Sampling(processes=1, timeout_seconds=90, memory_bytes=2 << 30))
    report = benchmark_suite(suite, inputs={"delegated-response": {"prompt": "hello world"}},
                             output=tmp_path / "benchmark.json")
    value = report.results[0].to_dict()
    assert value["status"] == "completed", value
    metrics = {entry["id"]: entry for entry in value["metrics"]}
    assert metrics["response.decode_tps"]["value"] is None
    assert metrics["response.decode_tps"]["unavailable_reason"] == "request_end_verification_buffers_all_output"
    for role in ("worker_a", "worker_b"):
        assert metrics[f"{role}.cold_cpu"]["value"] > 0
        assert metrics[f"{role}.peak_rss"]["value"] > 0
    assert Path(tmp_path / "benchmark.json").is_file()
    loaded = BenchmarkRun.load(tmp_path / "benchmark.json")
    persisted = loaded.to_dict()
    assert loaded.digest == report.digest
    assert persisted == report.to_dict()
    assert persisted["complete"] is True
    assert len(persisted["results"]) == 1
    assert persisted["results"][0] == value
    assert persisted["results"][0]["execution"]["case"] == suite.cases[0].to_spec()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("execution", [None, "source_stage_batch_v1"])
def test_gateway_verification_failure_exposes_only_error(tmp_path, monkeypatch, stream, sharing, execution):
    from pllm.runtime.delegated_source import AdmittedSource

    experiment, _ = saved_experiment(tmp_path, sharing=sharing, execution=execution)
    original = AdmittedSource.load_transport

    class Rejected:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def finish(self):
            self.native.abort()
            raise ValueError("injected request verification rejection")

    with OpenAI(experiment=experiment) as sdk:
        def rejected(source, *args, **kwargs):
            native = original(source, *args, **kwargs)
            return Rejected(native) if kwargs.get("worker") is None else native

        monkeypatch.setattr(AdmittedSource, "load_transport", rejected)
        app = create_sidecar_app(client=sdk, local_api_key="gateway-test")
        with TestClient(app, raise_server_exceptions=False) as client:
            result = client.post("/v1/responses", headers={"Authorization": "Bearer gateway-test"},
                json={"model": sdk._core.default_model, "input": "hello world",
                      "max_output_tokens": 2, "temperature": 0, "stream": stream})
        assert "error" in result.text
        assert not any(kind in result.text for kind in (
            "response.created", "response.output", "response.incomplete", "response.completed"))
        assert not sdk._core.records and sdk._core._active is None
        assert active_sessions(sdk._core) == [0, 0]


@pytest.mark.parametrize("benchmark", [False, True])
@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("execution", [None, "source_stage_batch_v1"])
def test_published_example_uses_installed_sdk(tmp_path, benchmark, sharing, execution):
    source_experiment, _ = saved_experiment(tmp_path)
    example = Path(__file__).resolve().parents[1] / "examples/delegated_sdk.py"
    saved = tmp_path / "published-experiment.json"
    command = [sys.executable, "-I", "-B", str(example), "--model-dir",
               source_experiment.pipeline.model.source, "--root", str(tmp_path / "example-roles"),
               "--max-output-tokens", "2", "--sharing", sharing, "--write-experiment", str(saved)]
    if benchmark:
        command += ["--benchmark-output", str(tmp_path / "example-benchmark.json")]
    if execution:
        command += ["--execution", execution]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert saved.is_file()
    assert json.loads(saved.read_text())["pipeline"]["components"]["linear"]["params"]["sharing"] == sharing
    assert json.loads(saved.read_text())["pipeline"]["components"]["linear"]["params"].get("execution") == execution
    assert "completed" in result.stdout if benchmark else "'output_tokens': 2" in result.stdout
