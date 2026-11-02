from __future__ import annotations

import threading
from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from pllm import (
    Cpu,
    DirectFHEProfile,
    Experiment,
    MaskedLinearCpu,
    Model,
    ProprietaryBlinded,
    ProprietaryGuarded,
)
from pllm.protocols import GuardedLinear
from pllm.runtime import servers
from pllm.runtime.servers import LocalTopology, TopologyError, build_roles, serve_local
from pllm.sources import TinyModel


def test_role_process_construction_has_one_implementation() -> None:
    root = Path("python/pllm/runtime")
    users = sorted(
        path.name
        for path in root.glob("*.py")
        # The stage-only stripe fixture owns a push/ack measurement peer, not a
        # serving role. Every serving graph still uses this shared supervisor.
        if path.name != "benchmark_correction_stripes.py"
        and "subprocess.Popen" in path.read_text(encoding="utf-8")
    )
    assert users == ["servers.py"]


# Controlled Popen-boundary choreography only. These doubles do not establish
# OS process-group containment, native retirement, or permission to launch tests.
@pytest.fixture
def supervised_boundary(monkeypatch):
    state = SimpleNamespace(events=[], processes=[], clock=0.0, fail_wait=False,
                            term_exit=False, init_error=None, topology=None)
    monkeypatch.setenv("PLLM_SUPERVISED_ROLE_GROUP", "1")
    monkeypatch.setattr(servers, "provider_popen_factory", lambda _: servers.UsagePopen)

    def initialize(process, command, **options):
        topology = state.topology
        owner = next(owner for owner in topology._role_process_owners.values()
                     if owner.process is process)
        assert owner.supervised is True
        assert owner.role not in topology._ready_role_owners
        assert type(topology) is LocalTopology
        assert type(process) is servers.UsagePopen  # Exact metric/reaper owner preserved.
        state.processes.append(process)
        process._child_created = False  # No OS child exists at this mocked boundary.
        if state.init_error is not None:
            raise state.init_error
        process.pid = 9000 + len(state.processes)
        process.returncode = None
        process.args = command
        process.options = options

    def terminate(process):
        state.events.append(("terminate", process))
        if state.term_exit:
            process.returncode = 0

    def kill(process):
        state.events.append(("kill", process))

    def wait(process, timeout):
        state.events.append(("wait", process, timeout))
        if process.returncode is not None:
            return process.returncode
        state.clock += min(1.0, timeout)
        if state.fail_wait or not any(event[:2] == ("kill", process) for event in state.events):
            raise servers.subprocess.TimeoutExpired(process.args, timeout)
        process.returncode = -9
        return process.returncode

    def forbidden_group(*_args):
        raise AssertionError("supervised cleanup must not signal any process group")

    monkeypatch.setattr(servers.UsagePopen, "__init__", initialize)
    monkeypatch.setattr(servers.UsagePopen, "poll", lambda process: process.returncode)
    monkeypatch.setattr(servers.UsagePopen, "terminate", terminate)
    monkeypatch.setattr(servers.UsagePopen, "kill", kill)
    monkeypatch.setattr(servers.UsagePopen, "wait", wait)
    monkeypatch.setattr(servers.os, "killpg", forbidden_group)
    monkeypatch.setattr(servers.time, "monotonic", lambda: state.clock)
    state.topology = build_roles("org/model")
    return state


@pytest.fixture
def topology_reservation(monkeypatch):
    from pllm.deployment import resources

    # Real ordinary reservation/ExitStack, isolated admission observation only.
    capacity = resources._ClientCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 30))
    reservation = resources.reserve_incremental_memory(64 << 10)
    leases = ExitStack()
    leases.enter_context(reservation)
    return SimpleNamespace(capacity=capacity, reservation=reservation, leases=leases,
                           token=reservation._token)


@pytest.mark.parametrize("value", ["", "true", "false", "01", "2", " 1", "1 ", "\n1", "0\n"])
def test_supervised_role_group_invalid_before_acquisition_or_spawn(monkeypatch, value):
    monkeypatch.setenv("PLLM_SUPERVISED_ROLE_GROUP", value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid mode reached startup acquisition/spawn")

    monkeypatch.setattr(servers.subprocess, "Popen", forbidden)
    monkeypatch.setattr(servers, "provider_popen_factory", forbidden)
    monkeypatch.setattr(servers.socket, "socket", forbidden)
    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        build_roles("org/model")
    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        servers._spawn_role_process(["unused"], environment={})


@pytest.mark.parametrize("value", [None, "0"])
def test_supervised_role_group_ordinary_foreign_factory_unchanged(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PLLM_SUPERVISED_ROLE_GROUP", raising=False)
    else:
        monkeypatch.setenv("PLLM_SUPERVISED_ROLE_GROUP", value)
    calls = []
    process = SimpleNamespace(pid=9000, returncode=None)

    def foreign(command, **options):
        calls.append((command, options))
        return process

    monkeypatch.setattr(servers.subprocess, "Popen", foreign)
    result = servers._spawn_role_process(["unused"], environment={"KEY": "secret"},
                                         discard_output=True, pass_fds=(7,))
    assert result is process
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["env"] == {"KEY": "secret"}
    assert calls[0][1]["stdout"] == calls[0][1]["stderr"] == servers.subprocess.DEVNULL
    assert calls[0][1]["pass_fds"] == (7,)


@pytest.mark.parametrize("value", [None, "0"])
def test_supervised_role_group_ordinary_cleanup_ignores_later_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PLLM_SUPERVISED_ROLE_GROUP", raising=False)
    else:
        monkeypatch.setenv("PLLM_SUPERVISED_ROLE_GROUP", value)
    calls = []

    class Process:
        pid = 9000
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert self.returncode is not None
            return self.returncode

        def terminate(self):
            raise AssertionError("ordinary POSIX role must retain group signalling")

        kill = terminate

    process = Process()

    def foreign(command, **options):
        assert options["start_new_session"] is True
        return process

    def killpg(pid, sig):
        calls.append((pid, sig))
        process.returncode = 0

    monkeypatch.setattr(servers.subprocess, "Popen", foreign)
    monkeypatch.setattr(servers.os, "killpg", killpg)
    topology = build_roles("org/model")
    assert topology._spawn("inference", ["unused"]) is process
    monkeypatch.setenv("PLLM_SUPERVISED_ROLE_GROUP", "1")
    topology.close()
    assert calls == [(process.pid, servers.signal.SIGTERM)]


def test_supervised_role_group_foreign_or_unsupported_refuses_prespawn(monkeypatch):
    monkeypatch.setenv("PLLM_SUPERVISED_ROLE_GROUP", "1")
    calls = []

    def foreign(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("foreign factory invoked")

    monkeypatch.setattr(servers.subprocess, "Popen", foreign)
    with pytest.raises(TopologyError, match="exact UsagePopen"):
        build_roles("org/model")
    with pytest.raises(TopologyError, match="exact UsagePopen"):
        servers._spawn_role_process(["unused"], environment={})
    assert not calls


def test_supervised_role_group_direct_helper_requires_owner(supervised_boundary):
    with pytest.raises(TopologyError, match="prepublished topology owner"):
        servers._spawn_role_process(["unused"], environment={})
    assert not supervised_boundary.processes


def test_supervised_role_group_spawn_identity_and_env_capture(supervised_boundary, monkeypatch):
    state = supervised_boundary
    topology = state.topology
    # Mode captured at topology construction, not from scrubbed child env or
    # mutable parent environment at spawn/cleanup time.
    monkeypatch.setenv("PLLM_SUPERVISED_ROLE_GROUP", "invalid-after-capture")
    process = topology._spawn("inference", ["unused"])
    owner = topology._role_process_owners["inference"]
    assert owner.process is topology._processes["inference"] is process
    assert type(process) is servers.UsagePopen
    assert process.options["start_new_session"] is False
    assert process.options["stdin"] == servers.subprocess.DEVNULL
    with pytest.raises(FrozenInstanceError):
        owner.supervised = False
    topology.close()
    assert [(event[0], event[1]) for event in state.events if event[0] != "wait"] == [
        ("terminate", process), ("kill", process)]
    assert topology.closed and not topology.is_healthy()


def test_supervised_role_group_already_reaped_not_signalled(supervised_boundary):
    state = supervised_boundary
    process = state.topology._spawn("inference", ["unused"])
    process.returncode = 0
    state.topology.close()
    assert not [event for event in state.events if event[0] != "wait"]


def test_supervised_role_group_term_reap_skips_kill(supervised_boundary):
    state = supervised_boundary
    state.term_exit = True
    process = state.topology._spawn("inference", ["unused"])
    state.topology.close()
    assert [event for event in state.events if event[0] != "wait"] == [("terminate", process)]


@pytest.mark.parametrize("fail_wait", [False, True])
def test_supervised_role_group_failed_reap_retains_custody(
    supervised_boundary, topology_reservation, fail_wait,
):
    state, charge = supervised_boundary, topology_reservation
    topology = state.topology
    topology._capacity_reservations = charge.leases
    state.fail_wait = fail_wait
    first = topology._spawn("inference", ["unused-a"])
    second = topology._spawn("preparation", ["unused-b"])
    owners = tuple(topology._role_process_owners.values())
    if fail_wait:
        with pytest.raises(TopologyError, match="did not stop"):
            topology.close()
        assert topology._capacity_reservations is charge.leases
        assert charge.capacity._owners == {charge.token: 64 << 10}
        assert not charge.reservation._exited
        assert tuple(topology._role_process_owners.values()) == owners
        assert topology._processes == {"inference": first, "preparation": second}
        assert all(owner.process in (first, second) for owner in owners)
        assert not topology.is_healthy()
        # Already-closed path must not turn an unreaped child into a refund.
        with pytest.raises(TopologyError, match="did not stop"):
            topology.close()
        assert topology._capacity_reservations is charge.leases
        assert charge.capacity._owners == {charge.token: 64 << 10}
    else:
        topology.close()
        assert topology._capacity_reservations is None
        assert charge.token not in charge.capacity._owners
        assert charge.reservation._exited
    assert first.options["start_new_session"] is second.options["start_new_session"] is False
    # One reverse-role signal pass, then shared waits. No per-role timeout reset.
    assert state.events[:2] == [("terminate", second), ("terminate", first)]
    waits = [event[2] for event in state.events if event[0] == "wait"]
    assert waits[:4] == [5.0, 4.0, 2.0, 1.0]
    assert state.events[4:6] == [("kill", second), ("kill", first)]
    if fail_wait:
        # Controlled boundary hygiene: a later known reap may retire an ordinary
        # failed wait, unlike permanently ambiguous initializer failure below.
        state.fail_wait = False
        topology.close()
        assert charge.token not in charge.capacity._owners


@pytest.mark.parametrize("failure", ["publication", "reader", "diagnostic"])
def test_supervised_role_group_post_init_failure_keeps_actual_owner(
    supervised_boundary, topology_reservation, monkeypatch, failure,
):
    state, charge = supervised_boundary, topology_reservation
    topology = state.topology
    topology._capacity_reservations = charge.leases
    error = RuntimeError("publication interrupted")

    class RejectPublication(dict):
        def __setitem__(self, key, value):
            raise error

    class RejectBreadcrumb(list):
        def append(self, value):
            raise error

    if failure == "publication":
        topology._processes = RejectPublication()
    elif failure == "diagnostic":
        topology._startup_breadcrumbs["spawned_roles"] = RejectBreadcrumb()
    else:
        from pllm.runtime import fss_child_status

        class Reader:
            child = SimpleNamespace(fileno=lambda: 7)

            def __init__(self, *args):
                pass

            def start(self):
                pass

            def spawned(self):
                raise error

            def close(self):
                pass

        topology._privacy_mode = "fss_public"
        monkeypatch.setattr(LocalTopology, "_environment", lambda self, role: {})
        monkeypatch.setattr(fss_child_status, "StatusReader", Reader)
    role = "worker_a" if failure == "reader" else "inference"
    if failure == "reader":
        topology._role_ids = (role,)
    with pytest.raises(RuntimeError) as raised:
        topology._spawn(role, ["unused"])
    assert raised.value is error
    owner = topology._role_process_owners[role]
    assert owner.process is state.processes[0]
    assert type(owner.process) is servers.UsagePopen
    assert role in topology._ready_role_owners
    assert charge.capacity._owners == {charge.token: 64 << 10}
    topology.close()
    assert [event[0] for event in state.events if event[0] != "wait"] == ["terminate", "kill"]
    assert charge.token not in charge.capacity._owners


@pytest.mark.parametrize("error", [RuntimeError("partial init"), KeyboardInterrupt()])
def test_supervised_role_group_partial_init_stays_unknown(
    supervised_boundary, topology_reservation, monkeypatch, error,
):
    state, charge = supervised_boundary, topology_reservation
    topology = state.topology
    topology._capacity_reservations = charge.leases
    state.init_error = error
    with pytest.raises(type(error)) as raised:
        topology._spawn("inference", ["unused"])
    assert raised.value is error
    owner = topology._role_process_owners["inference"]
    assert owner.process is state.processes[0]
    assert not topology._processes and not topology._ready_role_owners

    def forbidden(*args, **kwargs):
        raise AssertionError("partial raw owner must never be polled, signalled or waited")

    for method in ("poll", "wait", "terminate", "kill"):
        monkeypatch.setattr(servers.UsagePopen, method, forbidden)
    for _ in range(2):
        with pytest.raises(TopologyError, match="construction unresolved"):
            topology.close()
        assert topology._capacity_reservations is charge.leases
        assert topology._role_process_owners["inference"] is owner
        assert charge.capacity._owners == {charge.token: 64 << 10}
        assert not charge.reservation._exited
        # Apparently dead fields cannot qualify the unknown original generation.
        owner.process.pid = 9999
        owner.process.returncode = 0
        owner.process._child_created = False
    assert topology.closed and not topology.is_healthy()
    assert not state.events
    # Preserve synthetic unresolved custody through generator finalization too;
    # no drain, retry, ledger deletion or claimed refund.
    charge.reservation.retain_failed((topology,))
    assert charge.reservation.pending_retirement


def test_supervised_role_group_factory_resolved_once_before_allocation(
    supervised_boundary, monkeypatch,
):
    state = supervised_boundary

    def forbidden(*args):
        raise AssertionError("captured trusted factory must not be selected again")

    monkeypatch.setattr(servers, "provider_popen_factory", forbidden)
    process = state.topology._spawn("inference", ["unused"])
    assert type(process) is servers.UsagePopen
    state.topology.close()


def test_supervised_role_group_custom_allocator_refuses_before_allocation(
    supervised_boundary, monkeypatch,
):
    state = supervised_boundary
    calls = []

    def foreign_new(cls):
        calls.append(cls)
        raise AssertionError("custom allocator invoked")

    monkeypatch.setattr(servers.UsagePopen, "__new__", staticmethod(foreign_new))
    with pytest.raises(TopologyError, match="exact UsagePopen"):
        state.topology._spawn("inference", ["unused"])
    assert not calls and not state.processes and not state.topology._role_process_owners


def test_supervised_role_group_first_signal_failure_keeps_lease(
    supervised_boundary, topology_reservation, monkeypatch,
):
    state, charge = supervised_boundary, topology_reservation
    topology = state.topology
    topology._capacity_reservations = charge.leases
    process = topology._spawn("inference", ["unused"])
    error = RuntimeError("first signal failed")

    def terminate(raw):
        assert raw is process
        raise error

    monkeypatch.setattr(servers.UsagePopen, "terminate", terminate)
    with pytest.raises(RuntimeError) as raised:
        topology.close()
    assert raised.value is error
    assert process.returncode == -9
    assert topology._capacity_reservations is charge.leases
    assert charge.capacity._owners == {charge.token: 64 << 10}
    topology.close()  # Existing known-reap fast path, never unknown construction.
    assert charge.token not in charge.capacity._owners


def test_supervised_role_group_startup_interrupt_rolls_back_owned_child(
    supervised_boundary, topology_reservation, monkeypatch,
):
    state, charge = supervised_boundary, topology_reservation
    topology = state.topology
    topology._capacity_reservations = charge.leases
    ports = iter((9101, 9102))
    monkeypatch.setattr(LocalTopology, "_free_port", staticmethod(lambda excluded: next(ports)))

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(LocalTopology, "_wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        topology.start()
    assert len(state.processes) == 1
    assert topology._processes["inference"] is state.processes[0]
    assert state.processes[0].returncode == -9
    assert charge.token not in charge.capacity._owners
    assert topology.closed and not topology.started


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

    pipeline = MaskedLinearCpu(
        Model("org/pipeline"),
        kernels=Cpu(threads=3),
    )
    from_pipeline = build_roles(pipeline)
    assert from_pipeline.model_id == "org/pipeline"
    assert from_pipeline._engine_threads == 3
    with pytest.raises(ValueError, match="conflicts"):
        build_roles(pipeline, engine_threads=2)
    experiment = Experiment.from_spec({
        "schema": "pllm.experiment.v2",
        "name": "topology",
        "pipeline": {
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


@pytest.mark.parametrize(
    ("pipeline", "protocol"),
    [
        (ProprietaryGuarded(Model("org/model")), "guarded"),
        (ProprietaryBlinded(Model("org/model")), "blinded"),
        (DirectFHEProfile(Model("org/model")), "direct"),
    ],
)
def test_proprietary_profiles_build_one_role_commands(pipeline, protocol: str) -> None:
    topology = build_roles(pipeline)
    commands = topology._commands({"inference": 9101})
    assert set(commands) == {"inference"}
    command = commands["inference"]
    assert command[1:5] == ["-m", "pllm", "serve", "inference"]
    assert command[command.index("--privacy-mode") + 1] == "proprietary"
    assert command[command.index("--protocol") + 1] == protocol
    assert "preparation" not in command
    with pytest.raises(TopologyError, match="no preparation role"):
        _ = topology.preparation_url


def test_guarded_profile_maps_exact_policy_flags() -> None:
    pipeline = ProprietaryGuarded(
        Model("org/model"),
        linear=GuardedLinear(
            max_rows_per_request=11,
            max_rows_per_owner_stage=22,
            max_requests_per_minute=33,
            output_dither_bound=4,
        ),
    )
    command = build_roles(pipeline)._commands({"inference": 9101})["inference"]
    for flag, value in (
        ("--guard-max-rows-per-request", "11"),
        ("--guard-max-rows-per-stage", "22"),
        ("--guard-max-requests-per-minute", "33"),
        ("--guard-output-dither", "4"),
    ):
        assert command[command.index(flag) + 1] == value


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
            assert 0 <= timeout <= 5
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
    assert spawned[0].command[1:5] == ["-m", "pllm", "serve", "inference"]
    assert spawned[1].command[1:5] == ["-m", "pllm", "serve", "preparation"]
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


def test_one_role_topology_lifecycle_and_routing(monkeypatch, tmp_path: Path) -> None:
    spawned = []
    credentials = []

    class Process:
        pid = 5100
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            return 0

    def popen(command, **kwargs):
        process = Process()
        process.command = command
        process.kwargs = kwargs
        spawned.append(process)
        return process

    def credential(_length):
        credentials.append("guarded-inference")
        return "guarded-inference"

    monkeypatch.setattr(LocalTopology, "_free_port", staticmethod(lambda excluded: 9101))
    monkeypatch.setattr("pllm.runtime.servers.secrets.token_urlsafe", credential)
    monkeypatch.setattr("pllm.runtime.servers.subprocess.Popen", popen)
    monkeypatch.setattr(
        "pllm.runtime.servers.httpx.get",
        lambda url, timeout: SimpleNamespace(
            status_code=200,
            json=lambda: {"role": "inference"},
        ),
    )
    monkeypatch.setattr(
        "pllm.runtime.servers.os.killpg",
        lambda pid, signal: setattr(spawned[0], "returncode", 0),
    )
    topology = build_roles(
        ProprietaryGuarded(Model.path(str(tmp_path), model_id="guarded")),
        log_dir=tmp_path / "logs",
        progress=lambda role: None,
    ).start()
    assert len(spawned) == 1
    assert len(credentials) == 1
    assert len(topology.statuses) == 1
    assert topology.statuses[0].role == "inference"
    assert topology.is_healthy() is True
    environment = spawned[0].kwargs["env"]
    assert environment["PLLM_API_KEY"] == "local_guarded-inference"
    assert "PLLM_PROVIDER_PUSH_API_KEY" not in environment
    assert "PLLM_PUSH_API_KEY" not in environment
    assert topology._preparation_key == topology._push_key == ""

    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

    monkeypatch.setattr("pllm.runtime.client.OpenAI", Client)
    topology.client(timeout=3)
    assert captured["client"]["base_url"] == topology.inference_url
    assert "preparation_base_url" not in captured["client"]
    assert "preparation_api_key" not in captured["client"]

    monkeypatch.setattr(
        "pllm.runtime.sidecar.create_sidecar_app",
        lambda **kwargs: captured.setdefault("gateway", kwargs),
    )
    topology.gateway_app(local_api_key="local")
    assert "preparation_base_url" not in captured["gateway"]
    assert "preparation_api_key" not in captured["gateway"]
    topology.close()
    assert topology.closed is True
    assert topology._inference_key == ""


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
        "_automatic": False,
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
def test_actual_tiny_guarded_topology_starts_one_role(tmp_path: Path) -> None:
    topology = serve_local(
        ProprietaryGuarded(TinyModel(model_id="guarded-tiny")),
        hf_cache_dir=str(tmp_path / "models"),
        log_dir=tmp_path / "guarded-logs",
        startup_timeout=60,
    )
    try:
        assert topology.is_healthy()
        assert tuple(status.role for status in topology.statuses) == ("inference",)
        with pytest.raises(TopologyError, match="no preparation role"):
            _ = topology.preparation_url
    finally:
        topology.close()
    assert topology.closed


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
