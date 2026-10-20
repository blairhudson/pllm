"""LocalTopology FSS launcher ownership controls, with explicit bounded doubles.

No subprocess/native execution or cryptographic/source-origin qualification in
this module. test_fss_sdk.py covers generated-source actual role children.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from pllm import Model
from pllm.runtime import servers
from pllm.runtime.fss_protocol import ROLES, HOLDERS


PINS = {"source_lock": "a"*64, "tokenizer": "b"*64, "checkpoint": "c"*64, "config": "d"*64}


def topology(tmp_path):
    declaration = SimpleNamespace(canonical_bytes=lambda: b'{"fixture":"supervision-only"}',
        budget=SimpleNamespace(max_input_tokens=1, max_new_tokens=2), pipeline=object())
    return servers.LocalTopology(Model.path(str(tmp_path / "source")), model_id="generated-fss-supervision",
        experiment=declaration, engine_threads=1, weight_bits=16, activation_bits=16,
        correlation_mode="local-test", privacy_mode="fss_public", proprietary_protocol="guarded",
        role_ids=ROLES, requires_preparation=False, guard_max_rows_per_request=1,
        guard_max_rows_per_owner_stage=1, guard_max_requests_per_minute=1, guard_output_dither=0,
        verification_component=None, verification_target_failure_bits=40, public_equalization_digest=None,
        remote_output_head=False, client_prefix_layers=0, client_linear_roles=(), client_linear_layer_roles=(),
        tenseal_path=None, hf_cache_dir=None, reserved_ports=(), rendezvous_capacity=1,
        rendezvous_max_bytes=1024, log_dir=tmp_path / "logs", telemetry_endpoint=None,
        telemetry_token=None, credential_prefix="supervision-secret", startup_timeout=0.01, progress=None)


class Process:
    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        return self.returncode


@pytest.fixture
def launch(tmp_path, monkeypatch):
    from pllm.runtime import fss_source, benchmark_memory
    value = topology(tmp_path)
    state = SimpleNamespace(topology=value, events=[], children=[], environments={}, commands={},
                            stopped=[], failure=None, public_secrets=[])

    def admit(experiment):
        assert experiment is value._experiment
        state.events.append("pins")
        if state.failure == "pins":
            raise ValueError("source pins changed")
        return SimpleNamespace(config_bytes=b'{"model_type":"gpt2","n_embd":2}', snapshot_pins=dict(PINS))

    def estimate(*args, **kwargs):
        state.events.append("price")
        return object()

    def spawn(command, *, environment, stdout, stderr):
        role = command[command.index("--role") + 1]
        assert state.events[:2] == ["pins", "price"]
        assert value._source_pins == PINS
        assert json.loads(environment["PLLM_FSS_SOURCE_PINS_JSON"]) == PINS
        state.events.append("spawn:" + role)
        if state.failure == "spawn" and role == "worker_b":
            raise OSError("injected spawn failure")
        state.environments[role], state.commands[role] = environment, command
        child = Process(1_000_000 + len(state.children))
        state.children.append(child)
        if stdout is not None:
            stdout.write(("ready " + role + "\n").encode())
            stdout.flush()
        return child

    def wait(self, role, expected_role, url):
        assert expected_role == role  # FSS health is actual role, not generic worker.
        assert self._role_urls[role] == url
        if state.failure == "health" and role == "worker_b":
            raise servers.TopologyError("injected unhealthy role")

    def stop(processes):
        children = list(processes)
        state.stopped.append(children)
        for child in children:
            child.returncode = 0

    monkeypatch.setattr(fss_source, "admit_source", admit)
    monkeypatch.setattr(benchmark_memory, "estimate_memory", estimate)
    monkeypatch.setattr(benchmark_memory, "host_memory", lambda: object())
    monkeypatch.setattr(benchmark_memory, "admit_memory", lambda *args, **kwargs: {"admitted": state.failure != "memory"})
    monkeypatch.setattr(servers, "_spawn_role_process", spawn)
    monkeypatch.setattr(servers.LocalTopology, "_wait", wait)
    monkeypatch.setattr(servers.LocalTopology, "_stop_all", staticmethod(stop))
    yield state
    value.close()


def test_exact_pins_before_spawn_and_only_appropriate_directed_credentials(launch, monkeypatch):
    monkeypatch.setenv("PLLM_FSS_API_KEY", "operator-secret-must-not-inherit")
    monkeypatch.setenv("PLLM_FSS_PEERS_JSON", '{"dealer":["https://evil.invalid","operator-peer-secret"]}')
    monkeypatch.setenv("PYTHONPATH", "/operator/injection")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/operator/injection")
    t = launch.topology.start()
    all_keys = {*t._role_credentials.values(), *t._peer_credentials.values()}
    for role in ROLES:
        env, command = launch.environments[role], launch.commands[role]
        assert command[1:4] == ["-I", "-m", "pllm.runtime.fss_worker"]
        assert command[command.index("--role") + 1] == role
        outgoing = json.loads(env["PLLM_FSS_PEERS_JSON"])
        incoming = json.loads(env["PLLM_FSS_PEER_CREDENTIALS_JSON"])
        expected_out = set(HOLDERS) if role == "dealer" else set(HOLDERS) - {role}
        expected_in = set() if role == "dealer" else set(ROLES) - {role}
        assert set(outgoing) == expected_out and set(incoming) == expected_in
        own_keys = {t._role_credentials[role]}
        for peer, pair in outgoing.items():
            assert pair == [t._role_urls[peer], t._peer_credentials[(role, peer)]]
            own_keys.add(pair[1])
        for peer, key in incoming.items():
            assert key == t._peer_credentials[(peer, role)]
            own_keys.add(key)
        encoded = json.dumps(env)
        assert all(key in encoded for key in own_keys)
        assert all(key not in encoded for key in all_keys - own_keys)
        assert env["PLLM_FSS_API_KEY"] == t._role_credentials[role]
        assert "PYTHONPATH" not in env and "DYLD_LIBRARY_PATH" not in env
        assert "operator-secret" not in encoded and "operator-peer-secret" not in encoded
        assert env["PLLM_ROLE_EXPERIMENT_JSON"].encode() == t._experiment.canonical_bytes()
        assert not any(key in " ".join(command) for key in all_keys)
    public = repr(t.statuses) + "".join(path.read_text() for path in t._log_dir.glob("*.log"))
    assert all(secret not in public for secret in all_keys)
    assert len({s.pid for s in t.statuses}) == 3
    t.close()
    assert t.closed and all(not s.running for s in t.statuses)
    assert not t._role_credentials and not t._peer_credentials and t._source_pins is None
    assert all(log.closed for log in t._logs.values())


@pytest.mark.parametrize("failure", ["pins", "memory", "spawn", "health"])
def test_startup_failure_rolls_back_no_secret_or_child_reuse(launch, failure):
    launch.failure = failure
    t = launch.topology
    with pytest.raises(servers.TopologyError):
        t.start()
    assert t.closed and not t.started
    assert not t._role_credentials and not t._peer_credentials and t._source_pins is None
    assert all(child.poll() is not None for child in launch.children)
    assert all(log.closed for log in t._logs.values())
    if failure in {"pins", "memory"}:
        assert not launch.children and not launch.environments
    if launch.children:
        assert launch.stopped == [list(reversed(launch.children))]
    with pytest.raises(servers.TopologyError):
        t.start()


def test_environment_cannot_be_exported_before_source_is_pinned(tmp_path):
    t = topology(tmp_path)
    try:
        with pytest.raises(servers.TopologyError, match="pinned"):
            t._environment("worker_a")
    finally:
        t.close()


def test_concurrent_close_reaps_once_and_clears_all_credentials(launch):
    t = launch.topology.start()
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: t.close(), range(3)))
    assert len(launch.stopped) == 1
    assert launch.stopped[0] == list(reversed(launch.children))
    assert all(child.poll() is not None for child in launch.children)
    assert not t._role_credentials and not t._peer_credentials and t._source_pins is None


def test_startup_diagnostics_redact_any_role_and_directed_secret_echo(launch):
    t = launch.topology.start()
    secrets = [*t._role_credentials.values(), *t._peer_credentials.values()]
    path = t._log_dir / "worker_a.log"
    path.write_text("\n".join(secrets))  # Deliberate synthetic child echo, not production logging.
    tail = t._redacted_tail("worker_a")
    assert "<redacted>" in tail and all(key not in tail for key in secrets)


def test_health_requires_actual_role_identity(tmp_path, monkeypatch):
    t = topology(tmp_path)
    child = Process(1_000_000)
    t._processes["worker_a"] = child
    monkeypatch.setattr(servers.LocalTopology, "_stop_all", staticmethod(lambda children: [setattr(c, "returncode", 0) for c in children]))
    monkeypatch.setattr(servers.time, "sleep", lambda _: None)
    monkeypatch.setattr(servers.httpx, "get", lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: {"role": "worker_b"}))
    try:
        with pytest.raises(servers.TopologyError, match="ready"):
            t._wait("worker_a", "worker_a", "http://127.0.0.1:1234")
        monkeypatch.setattr(servers.httpx, "get", lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: {"role": "worker_a"}))
        t._wait("worker_a", "worker_a", "http://127.0.0.1:1234")
    finally:
        t.close()


def test_common_process_boundary_places_credentials_only_in_environment(monkeypatch):
    captured = {}
    command = [servers.sys.executable, "-I", "-m", "pllm.runtime.fss_worker", "--role", "dealer", "--port", "1234"]
    environment = {"PLLM_FSS_API_KEY": "fixture-directed-secret-not-argv"}

    def popen(args, **kwargs):
        captured.update(command=args, **kwargs)
        return Process(1_000_000)

    monkeypatch.setattr(servers.subprocess, "Popen", popen)
    servers._spawn_role_process(command, environment=environment)
    assert captured["command"] == command and captured["env"] is environment
    assert captured["stdin"] == servers.subprocess.DEVNULL
    assert captured["start_new_session"] is True
    assert environment["PLLM_FSS_API_KEY"] not in repr(captured["command"])
