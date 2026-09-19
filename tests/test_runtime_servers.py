from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from pllm import Cpu, Experiment, Model, Pipeline
from pllm.runtime.servers import LocalTopology, TopologyError, build_roles, serve_local
from pllm.sources import TinyModel


def test_role_process_construction_has_one_implementation() -> None:
    root = Path("python/pllm/runtime")
    users = sorted(
        path.name
        for path in root.glob("*.py")
        if "subprocess.Popen" in path.read_text(encoding="utf-8")
    )
    assert users == ["servers.py"]


def test_build_roles_is_side_effect_free_and_validates_inputs(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("build_roles performed runtime work")

    monkeypatch.setattr("pllm.runtime.servers.socket.socket", forbidden)
    monkeypatch.setattr("pllm.runtime.servers.secrets.token_urlsafe", forbidden)
    monkeypatch.setattr("pllm.runtime.servers.subprocess.Popen", forbidden)
    topology = build_roles(Model.hf("org/model", revision="abc"), model_id="public")
    assert topology.model_id == "public"
    assert topology.started is False
    assert topology.closed is False
    assert [(row.role, row.pid, row.running) for row in topology.statuses] == [
        ("inference", None, False),
        ("preparation", None, False),
    ]
    with pytest.raises(TopologyError, match="has not started"):
        _ = topology.inference_url

    pipeline = Pipeline.from_profile(
        "baseline.masked_linear_cpu",
        model=Model("org/pipeline"),
        components={"kernels": Cpu(threads=3)},
    )
    from_pipeline = build_roles(pipeline)
    assert from_pipeline.model_id == "org/pipeline"
    assert from_pipeline._engine_threads == 3
    with pytest.raises(ValueError, match="conflicts"):
        build_roles(pipeline, engine_threads=2)
    experiment = Experiment.from_spec({
        "schema": "pllm.experiment.v1",
        "name": "topology",
        "pipeline": {
            "profile": "baseline.masked_linear_cpu",
            "model": {"source": "org/experiment"},
            "components": {
                "inference": {"component": "pllm/inference", "params": {}},
                "kernels": {"component": "pllm/cpu", "params": {"threads": 2}},
                "linear": {"component": "pllm/masked-linear", "params": {}},
                "preparation": {
                    "component": "pllm/model-aware-corrections",
                    "params": {},
                },
            },
        },
        "deployment": {"kind": "local", "root": "local://topology"},
        "budget": {"requests": 1, "max_input_tokens": 8, "max_new_tokens": 1},
    })
    from_experiment = build_roles(experiment)
    assert from_experiment.model_id == "org/experiment"
    assert from_experiment._experiment is experiment
    assert from_experiment._engine_threads == 2

    invalid = (
        {"model": Model.ollama("qwen")},
        {"model": Model("x"), "model_id": ""},
        {"model": Model("x"), "engine_threads": 0},
        {"model": Model("x"), "weight_bits": 7},
        {"model": Model("x"), "activation_bits": 7},
        {"model": Model("x"), "correlation_mode": "unknown"},
        {"model": Model("x"), "reserved_ports": (0,)},
        {"model": Model("x"), "reserved_ports": (8000, 8000)},
        {"model": Model("x"), "rendezvous_capacity": 0},
        {"model": Model("x"), "rendezvous_max_bytes": 0},
        {"model": Model("x"), "startup_timeout": 0},
        {"model": Model("x"), "startup_timeout": float("nan")},
        {"model": Model("x"), "credential_prefix": "not valid"},
        {"model": Model("x"), "tenseal_path": ""},
        {"model": Model("x"), "telemetry_endpoint": "http://otel"},
        {
            "model": Model("x"),
            "telemetry_endpoint": "http://user:secret@otel/path",
            "telemetry_token": "token",
        },
        {"model": Model("x"), "telemetry_token": "token"},
    )
    for kwargs in invalid:
        with pytest.raises((TypeError, ValueError)):
            build_roles(**kwargs)


def test_topology_builds_secret_free_commands_and_closes_processes(monkeypatch, tmp_path: Path) -> None:
    ports = iter((9101, 9102))
    credentials = iter(("inference", "preparation", "push"))
    spawned = []
    progress = []

    class Process:
        def __init__(self, command, kwargs):
            self.command = command
            self.kwargs = kwargs
            self.pid = 4000 + len(spawned)
            self.returncode = None
            self.terminated = False
            self.killed = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout):
            assert timeout in {5, 10}
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    def popen(command, **kwargs):
        process = Process(command, kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(
        LocalTopology,
        "_free_port",
        staticmethod(lambda excluded: next(ports)),
    )
    monkeypatch.setattr(
        "pllm.runtime.servers.secrets.token_urlsafe",
        lambda _length: next(credentials),
    )
    monkeypatch.setattr("pllm.runtime.servers.subprocess.Popen", popen)
    monkeypatch.setattr(
        "pllm.runtime.servers.httpx.get",
        lambda url, timeout: SimpleNamespace(
            status_code=200,
            json=lambda: {
                "role": "trusted-preparation" if "9102" in url else "inference"
            },
        ),
    )

    def killpg(pid, _signal):
        process = next(process for process in spawned if process.pid == pid)
        process.terminated = True
        process.returncode = 0

    monkeypatch.setattr("pllm.runtime.servers.os.killpg", killpg)

    topology = build_roles(
        Model.hf("org/model", revision="abc", local_files_only=True),
        model_id="public",
        engine_threads=3,
        correlation_mode="local-test",
        tenseal_path="/tmp/tenseal",
        hf_cache_dir="/tmp/cache",
        reserved_ports=(8080,),
        log_dir=tmp_path / "logs",
        telemetry_endpoint="http://127.0.0.1:7777",
        telemetry_token="otel-token",
        credential_prefix="test",
        progress=progress.append,
    ).start()

    assert topology.started is True
    assert topology.is_healthy() is True
    assert topology.inference_url == "http://127.0.0.1:9101"
    assert topology.preparation_url == "http://127.0.0.1:9102"
    assert progress == ["inference", "preparation"]
    assert len(spawned) == 2
    assert all(process.kwargs["start_new_session"] for process in spawned)
    arguments = [item for process in spawned for item in process.command]
    secret_values = {
        value
        for process in spawned
        for key, value in process.kwargs["env"].items()
        if key in {"PLLM_API_KEY", "PLLM_PROVIDER_PUSH_API_KEY", "PLLM_PUSH_API_KEY"}
    }
    assert secret_values == {"test_inference", "test_preparation", "test_push"}
    assert secret_values.isdisjoint(arguments)
    assert all(secret not in repr(topology.statuses) for secret in secret_values)
    assert all(secret not in repr(topology) for secret in secret_values)
    assert "--allow-insecure-local-correlations" in spawned[0].command
    assert "--revision" in spawned[0].command
    assert spawned[1].kwargs["env"]["PLLM_INFERENCE_URL"] == topology.inference_url
    assert spawned[0].kwargs["env"]["OTEL_SERVICE_NAME"] == "pllm-inference"
    assert spawned[1].kwargs["env"]["OTEL_SERVICE_NAME"] == "pllm-preparation"
    assert topology.__enter__() is topology

    topology.close()
    topology.close()
    assert topology.closed is True
    assert topology._inference_key == topology._preparation_key == topology._push_key == ""
    assert all(process.terminated for process in spawned)
    assert not any(process.killed for process in spawned)
    with pytest.raises(TopologyError, match="started again"):
        topology.start()


def test_client_and_gateway_keep_private_routing_immutable(monkeypatch) -> None:
    topology = build_roles("org/model")
    topology._started = True
    topology._inference_url = "http://127.0.0.1:9101"
    topology._preparation_url = "http://127.0.0.1:9102"
    topology._inference_key = "inference-secret"
    topology._preparation_key = "preparation-secret"
    monkeypatch.setattr(LocalTopology, "is_healthy", lambda self: True)
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

    monkeypatch.setattr("pllm.runtime.client.OpenAI", Client)
    client = topology.client(timeout=12)
    assert isinstance(client, Client)
    assert captured["client"] == {
        "base_url": "http://127.0.0.1:9101",
        "api_key": "inference-secret",
        "default_model": "org/model",
        "preparation_base_url": "http://127.0.0.1:9102",
        "preparation_api_key": "preparation-secret",
        "correlation_mode": "bfv",
        "experiment": None,
        "timeout": 12,
    }
    with pytest.raises(TopologyError, match="cannot be overridden"):
        topology.client(base_url="http://other")
    with pytest.raises(TopologyError, match="cannot be overridden"):
        topology.client(http_client=object())

    monkeypatch.setattr(
        "pllm.runtime.sidecar.create_sidecar_app",
        lambda **kwargs: captured.setdefault("gateway", kwargs),
    )
    app = topology.gateway_app(local_api_key="local-secret", timeout=9)
    assert app["remote_api_key"] == "inference-secret"
    assert app["preparation_api_key"] == "preparation-secret"
    assert app["local_api_key"] == "local-secret"
    assert app["timeout"] == 9
    with pytest.raises(TopologyError, match="cannot be overridden"):
        topology.gateway_app(local_api_key="local", remote_api_key="other")
    with pytest.raises(TopologyError, match="cannot be overridden"):
        topology.gateway_app(local_api_key="local", client=object())


def test_keyboard_interrupt_during_startup_closes_spawned_role(monkeypatch) -> None:
    ports = iter((9101, 9102))

    class Process:
        pid = 1234
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            return self.returncode

    process = Process()
    monkeypatch.setattr(LocalTopology, "_free_port", staticmethod(lambda excluded: next(ports)))
    monkeypatch.setattr("pllm.runtime.servers.subprocess.Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        "pllm.runtime.servers.os.killpg",
        lambda _pid, _signal: setattr(process, "returncode", 0),
    )
    monkeypatch.setattr(
        LocalTopology,
        "_wait",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    topology = build_roles("org/model")
    with pytest.raises(KeyboardInterrupt):
        topology.start()
    assert topology.closed is True
    assert process.poll() == 0


def test_close_during_startup_prevents_later_role_spawn(monkeypatch) -> None:
    ports = iter((9101, 9102))
    waiting = threading.Event()
    spawned = []
    failures = []

    class Process:
        pid = 1234
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0

    def popen(*_args, **_kwargs):
        process = Process()
        spawned.append(process)
        return process

    def wait(topology, role, _expected, _url):
        assert role == "inference"
        waiting.set()
        topology._stopping.wait(5)
        raise TopologyError("local topology stopped during startup")

    monkeypatch.setattr(LocalTopology, "_free_port", staticmethod(lambda excluded: next(ports)))
    monkeypatch.setattr(LocalTopology, "_wait", wait)
    monkeypatch.setattr("pllm.runtime.servers.subprocess.Popen", popen)
    monkeypatch.setattr(
        "pllm.runtime.servers.os.killpg",
        lambda _pid, _signal: setattr(spawned[0], "returncode", 0),
    )
    topology = build_roles("org/model")

    def start():
        try:
            topology.start()
        except TopologyError as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=start)
    thread.start()
    assert waiting.wait(2)
    topology.close()
    thread.join(2)
    assert thread.is_alive() is False
    assert len(spawned) == 1
    assert failures == ["local topology stopped during startup"]


def test_startup_failure_rolls_back_and_redacts_credentials(monkeypatch, tmp_path: Path) -> None:
    class Process:
        pid = 1234
        returncode = 7

        @staticmethod
        def poll():
            return 7

        @staticmethod
        def wait(timeout):
            return 7

    def popen(_command, **kwargs):
        kwargs["stdout"].write(kwargs["env"]["PLLM_API_KEY"].encode())
        kwargs["stdout"].flush()
        return Process()

    ports = iter((9101, 9102))
    monkeypatch.setattr(LocalTopology, "_free_port", staticmethod(lambda excluded: next(ports)))
    monkeypatch.setattr("pllm.runtime.servers.subprocess.Popen", popen)
    topology = build_roles(
        "org/model",
        log_dir=tmp_path,
        credential_prefix="secretprefix",
        startup_timeout=0.1,
    )
    with pytest.raises(TopologyError) as failure:
        topology.start()
    assert "secretprefix_" not in str(failure.value)
    assert "<redacted>" in str(failure.value)
    assert topology.closed is True


@pytest.mark.integration
def test_actual_tiny_local_topology_starts_and_stops(tmp_path: Path) -> None:
    topology = serve_local(
        TinyModel(model_id="topology-tiny"),
        correlation_mode="local-test",
        hf_cache_dir=str(tmp_path / "models"),
        log_dir=tmp_path / "logs",
        startup_timeout=60,
    )
    client = None
    try:
        assert topology.is_healthy() is True
        assert all(status.running for status in topology.statuses)
        client = topology.client(bundle_cache_mode="off", timeout=30)
        models = client.models.list()["data"]
        assert any(row["id"] == "topology-tiny" for row in models)
    finally:
        if client is not None:
            client.close()
        topology.close()
    assert all(not status.running for status in topology.statuses)
