"""Installed GKM Experiment through real CLI, gateway and benchmark processes."""

from contextlib import contextmanager
from dataclasses import replace
import copy
import gc
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
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


@pytest.mark.parametrize("stored_dtype", ["F32", "F16", "BF16"])
def test_generated_sidecar_private_audit_matches_stored_dtype_oracle(
        tmp_path, stored_dtype, record_property):
    from test_gkm_checkpoint_conformance import (
        _generated_accepted_result_matches_bounded_checkpoint_reader,
    )

    # Only missing HTTP adapter; do not repeat SDK sharing/execution matrices.
    _generated_accepted_result_matches_bounded_checkpoint_reader(
        tmp_path, "full", None, stored_dtype, record_property, adapter="sidecar")


@pytest.mark.parametrize("stream", [False, True])
def test_sidecar_cancellation_discards_output_and_retires_roles(tmp_path, monkeypatch, stream):
    from pllm.runtime import benchmark_response_audit as audit

    experiment, _, _ = experiment_fixture(tmp_path, requests=1)
    original = delegated_transport._Peers.post
    cancelled, accepted = [], []
    with audit.response_state_audit(os.urandom(32), runtime_kind="gkm") as records:
        owner = audit._GKM_OWNER
        with OpenAI(experiment=experiment) as sdk:
            topology = sdk._owned_topology
            sdk._core._result_hook = accepted.append

            def cancel_after_consumption(peers, role, action, *args, **kwargs):
                value = original(peers, role, action, *args, **kwargs)
                if not cancelled and role == 0 and action == "step":
                    attempt = sdk._core._active
                    assert attempt is not None
                    attempt.cancel()
                    assert attempt.cancelled.is_set()
                    cancelled.append(True)
                return value

            monkeypatch.setattr(delegated_transport._Peers, "post", cancel_after_consumption)
            app = create_sidecar_app(client=sdk, local_api_key="cancel-test")
            with TestClient(app, raise_server_exceptions=False) as http:
                response = http.post("/v1/responses",
                    headers={"Authorization": "Bearer cancel-test"},
                    json={"model": sdk._core.default_model, "input": "hello world",
                          "max_output_tokens": 2, "temperature": 0, "stream": stream})
            assert cancelled == [True]
            assert response.status_code != 200 and "error" in response.text
            assert not any(kind in response.text for kind in (
                "response.created", "response.output", "response.incomplete", "response.completed"))
            assert accepted == records == [] and owner.accepted is None
            assert not sdk._core.records and sdk._core._active is None
            assert active_sessions(sdk._core) == [0, 0]
        assert all(not status.running for status in topology.statuses)
    assert records == [] and owner.closed and audit._GKM_OWNER is None


@contextmanager
def _combined_controller_watchdog(monkeypatch, combined_report, stopped, deadline):
    """Keep the runner's abort/drain authority; tighten only its host floor."""
    from pllm.runtime import benchmark_memory as memory

    original = memory.MemoryWatchdog

    class CombinedControllerWatchdog(original):
        def __init__(self, controller_report, abort):
            super().__init__(combined_report, abort)

        def check(self):
            if stopped.is_set():
                raise memory.BenchmarkMemoryError("corrected suite outer watchdog cancelled")
            deadline.remaining()
            super().check()

    with monkeypatch.context() as scoped:
        scoped.setattr(memory, "MemoryWatchdog", CombinedControllerWatchdog)
        yield


def _corrected_prompt_sampling(sampling, deadline):
    return replace(sampling, timeout_seconds=min(sampling.timeout_seconds, deadline.remaining()))


@pytest.mark.parametrize("failure_phase", ["start", "communicate"])
def test_corrected_suite_combined_watchdog_cancels_and_drains(tmp_path, monkeypatch, failure_phase):
    from test_gkm_checkpoint_conformance import _CorrectedPromptDeadline

    _corrected_controller_regression(tmp_path, monkeypatch, failure_phase,
                                    _CorrectedPromptDeadline())


def _corrected_controller_regression(tmp_path, monkeypatch, failure_phase, deadline, *, expire=None):
    from types import SimpleNamespace
    from pllm._benchmark_worker import PROTOCOL
    from pllm.metrics import benchmark_runner as runner
    from pllm.runtime import benchmark_memory as memory, client_process_usage as usage

    original_watchdog = memory.MemoryWatchdog
    expected_timeout = min(90, deadline.remaining())
    stopped = threading.Event()
    combined_report = {"host": {"reserve_bytes": 1 << 30, "swap_used": 0,
                                 "available": 6 << 30}, "runtime_floor_bytes": 3 << 30}
    ordinary_floor, low_available = 1 << 30, 2 << 30
    assert ordinary_floor < low_available < combined_report["runtime_floor_bytes"]
    events, monitors, samples, finalized = [], [], [], []
    phase = "start"

    def host():
        available = (low_available if expire is None and samples and phase == failure_phase else 6 << 30)
        samples.append(available)
        return SimpleNamespace(reserve=1 << 30, swap_used=0, available=available, pressure=None)

    class Event:
        def __init__(self):
            self.stopped = False

        def set(self):
            self.stopped = True

        def is_set(self):
            return self.stopped

        def wait(self, timeout):
            if timeout == 0.02:
                # One deterministic real tracker iteration captures descendants.
                self.set()
            else:
                assert timeout == 0.25
            return self.stopped

    class Thread:
        def __init__(self, *, target, daemon, name=None):
            assert daemon is True
            self.target, self.name, self.ident = target, name, None
            if name == "pllm-memory-guard":
                monitors.append(target.__self__)

        def start(self):
            self.ident = 1
            events.append("guard-start" if self.name else "tracker-start")
            if self.name is None:
                self.target()

        def join(self, timeout=None):
            assert timeout == (2 if self.name else None)
            events.append("guard-join" if self.name else "tracker-join")

    class Descendant:
        pid = 2147483645

        def create_time(self):
            return 123.0

        def kill(self):
            events.append("descendant-kill")

        def wait(self, timeout):
            assert timeout == 5
            events.append("descendant-wait")

    descendant = Descendant()

    class Process:
        pid = 2147483646
        returncode = None

        def __init__(self, command, **options):
            assert command[1:] == ["-I", "-m", "pllm._benchmark_worker"]
            assert options["start_new_session"] is True
            events.append("spawn")
            if failure_phase == "start" and expire is not None:
                expire()

        def communicate(self, body, timeout):
            nonlocal phase
            assert json.loads(body)["protocol"] == PROTOCOL and timeout == expected_timeout
            assert json.loads(body)["sampling"]["timeout_seconds"] == expected_timeout
            events.append("communicate")
            phase = "communicate"
            if expire is not None:
                expire()
            # Execute the inherited watchdog loop synchronously, not a fake
            # abort/result path. A valid completed child must still be rejected.
            monitors[0]._run()
            self.returncode = 0
            return json.dumps({"protocol": PROTOCOL, "status": "completed"}).encode(), b""

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            self.returncode = -signal.SIGTERM if self.returncode is None else self.returncode

    def process_identity(pid):
        if pid == Process.pid:
            return SimpleNamespace(children=lambda recursive: [descendant])
        assert pid == descendant.pid
        return descendant

    def forbidden(*args, **kwargs):
        pytest.fail("uncontrolled process, thread or host probe")

    monkeypatch.setattr(memory, "host_memory", host)
    monkeypatch.setattr(memory, "threading", SimpleNamespace(
        Event=Event, Thread=Thread, current_thread=lambda: None))
    monkeypatch.setattr(runner, "threading", SimpleNamespace(Event=Event, Thread=Thread))
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(**(vars(subprocess) | {
        "Popen": Process, "run": forbidden})))
    monkeypatch.setattr(runner, "os", SimpleNamespace(**(vars(os) | {
        "killpg": lambda pid, sig: events.append(("signal", pid, sig))})))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=process_identity, Error=RuntimeError))
    monkeypatch.setattr(usage, "client_popen_factory", lambda factory: factory)
    monkeypatch.setattr(usage, "finalize_client_payload", lambda payload, process: finalized.append(payload))
    monkeypatch.setattr(runner, "_interpreter", lambda python: "/mock/python")
    monkeypatch.setattr(runner, "_environment", lambda threads: {})
    case = BenchmarkCase("combined-watchdog", BenchmarkTarget.runtime())
    sampling = Sampling(processes=1, timeout_seconds=90)
    bounded = _corrected_prompt_sampling(sampling, deadline)
    assert sampling.timeout_seconds == 90 and bounded is not sampling
    result = None
    if failure_phase == "start":
        with pytest.raises(memory.BenchmarkMemoryError,
                           match="deadline expired" if expire else "below runtime reserve"):
            with _combined_controller_watchdog(monkeypatch, combined_report, stopped, deadline):
                result = runner._execute(case, bounded, {}, b"s" * 32, tmp_path)
    else:
        with _combined_controller_watchdog(monkeypatch, combined_report, stopped, deadline):
            result = runner._execute(case, bounded, {}, b"s" * 32, tmp_path)
        assert result["status"] == "blocked" and result["failure_code"] == "host_pressure"
        diagnostics = result["details"]["controller_memory_watchdog"]
        assert diagnostics["error_class"] == "BenchmarkMemoryError"
        if expire is None:
            assert diagnostics["failure_observation"] == {
                "pressure": None, "available_bytes": low_available, "swap_used_bytes": 0}
            assert diagnostics["minimum_available_bytes"] == low_available
        else:
            assert "deadline expired" in diagnostics["error_message"]
            assert diagnostics["failure_observation"] is None
            assert diagnostics["minimum_available_bytes"] == 6 << 30
    monitor, = monitors
    assert monitor.report is combined_report
    assert type(monitor)._run is original_watchdog._run
    assert type(monitor).start is original_watchdog.start
    assert type(monitor).close is original_watchdog.close
    assert monitor._stop.is_set()
    assert monitor.abort.__qualname__.endswith("_execute.<locals>.abort")
    assert memory.MemoryWatchdog is original_watchdog
    assert finalized == [None] and (result is None or result["status"] != "completed")
    prefix = ["spawn", "tracker-start"]
    if failure_phase == "communicate":
        prefix += ["guard-start", "communicate", ("signal", Process.pid, signal.SIGTERM), "guard-join"]
    else:
        # Synchronous start.check raises before _run/abort or guard thread start.
        assert monitor.error is None and monitor.error_class is None
        assert monitor._thread.ident is None
        prefix += [("signal", Process.pid, signal.SIGTERM), ("wait", 10)]
    assert events == prefix + [("signal", Process.pid, signal.SIGKILL), ("wait", None),
                               "tracker-join", "descendant-kill", "descendant-wait"]
    stopped.set()
    before = list(samples)
    with pytest.raises(memory.BenchmarkMemoryError, match="outer watchdog cancelled"):
        monitor.check()
    assert samples == before  # Latched outer cancellation needs no new telemetry.


@pytest.mark.parametrize("failure_phase", ["preflight", "oracle", "sdk", "sidecar",
                                         "execution_start", "execution_communicate"])
def test_corrected_prompt_deadline_expiry(tmp_path, monkeypatch, failure_phase):
    import weakref
    from types import SimpleNamespace
    import fastapi.testclient as http_boundary
    import test_gkm_checkpoint_conformance as conformance
    from pllm.deployment import resources
    from pllm.runtime import benchmark_memory as memory, delegated_source, sidecar, tokenizer_admission
    from pllm.runtime.delegated_protocol import clear_failure_frames

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(conformance.time, "monotonic", lambda: clock.now)
    deadline = conformance._CorrectedPromptDeadline()
    clock.now = 410.0
    assert deadline.remaining() == 40 and deadline.expires == 450
    events, live, references, clients, captured, retired_roles = [], [], [], [], [], []

    def expire():
        clock.now = 451.0

    def forbidden(*args, **kwargs):
        pytest.fail("deadline test crossed uncontrolled source/host/thread/process boundary")

    monkeypatch.setattr(memory, "benchmark_memory", forbidden)
    monkeypatch.setattr(delegated_source, "admit_source", forbidden)
    monkeypatch.setattr(tokenizer_admission, "_read_named", forbidden)
    monkeypatch.setattr(delegated_source, "_locked", forbidden)
    monkeypatch.setattr(delegated_source, "_source_float_frame", forbidden)
    capacity = resources._ClientCapacity()
    observations, reservations, exits = [], [], []
    original_factory = resources.reserve_incremental_memory
    original_exit = resources._MemoryReservation.__exit__

    def observe():
        # Closed-world telemetry only. The actual capacity still owns every
        # debit/refund, and outstanding real tokens remain in this snapshot.
        with capacity.lock:
            snapshot = resources.ClientResources(
                observed_at_ms=0, available_memory_bytes=6 * memory.GiB,
                process_memory_bytes=64 * memory.MiB, reserve_bytes=memory.GiB,
                reserved_memory_bytes=sum(capacity._owners.values()),
                container_headroom_bytes=5 * memory.GiB)
            observations.append(snapshot.reserved_memory_bytes)
            return snapshot

    def observe_exit(reservation, *exc):
        try:
            assert reservations == [reservation] and exits == []
            assert reservation._capacity is capacity and reservation._acquired
            assert reservation._amount == 512 * memory.MiB
            with capacity.lock:
                assert capacity._owners == {reservation._token: reservation._amount}
            assert capacity.observe().reserved_memory_bytes == reservation._amount
            assert live == clients == captured == []
            assert all(reference() is None for reference in references)
            assert events[-1] == "payloads-source-exception-frames-retired"
            events.append("retired-before-refund")
        finally:
            # Forward the same exception type/value/traceback exactly once;
            # only the genuine reservation/hold_bytes machinery can refund.
            result = original_exit(reservation, *exc)
        with capacity.lock:
            assert reservation._token not in capacity._owners
        assert reservation._exited and capacity.observe().reserved_memory_bytes == 0
        exits.append((reservation._token, exc[0], result))
        events.append("refund-raw-512")
        return result

    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", observe)
    monkeypatch.setattr(resources._MemoryReservation, "__exit__", observe_exit)
    if failure_phase == "preflight":
        monkeypatch.setattr(tokenizer_admission, "reserve_decoder_metadata", forbidden)
        expire()
        experiment = SimpleNamespace(budget=SimpleNamespace(
            requests=1, max_input_tokens=30, max_new_tokens=2))
        with pytest.raises(memory.BenchmarkMemoryError, match="deadline expired"):
            conformance._corrected_prompt_memory_admission(
                experiment, object(), check=deadline.remaining)
        assert events == live == clients == captured == []
        assert capacity._owners == {} and observations == reservations == exits == []
        assert resources.reserve_incremental_memory is original_factory
        return  # No metadata or raw-oracle reservation was acquired.

    class Payload:
        pass

    class Event:
        def __init__(self):
            self.stopped = False

        def set(self):
            self.stopped = True

        def is_set(self):
            return self.stopped

        def wait(self, timeout):
            assert timeout == 0.25
            return self.stopped

    class Thread:
        def __init__(self, *, target, name, daemon):
            assert daemon and name == "pllm-memory-guard"
            self.ident = None

        def start(self):
            self.ident = 1
            events.append("guard-start")

        def join(self, timeout):
            assert timeout == 2
            events.append("guard-join")

    def cancel():
        assert clients[0]._core._active is not None
        clients[0]._core._active = None
        clients[0].sessions[:] = [0, 0]
        events.append("active-attempt-cancelled")

    class Client:
        def __init__(self):
            self._core = SimpleNamespace(_active=None, timeout=999, default_model="mock", payload=live[1])
            self.responses = SimpleNamespace(create=dispatch)
            self.sessions = [0, 0]
            self.roles = [SimpleNamespace(running=True), SimpleNamespace(running=True)]
            self.closed = False

        def close(self):
            assert self._core._active is None and self.sessions == [0, 0]
            for role in self.roles:
                role.running = False
            retired_roles.extend(self.roles)
            self._core.payload = None
            self.closed = True
            events.append("mock-client-closed-roles-drained")

    class Response:
        status_code, text = 200, "syntactically completed mock"

        def to_dict(self):
            return {"status": "completed"}

        json = to_dict

    def dispatch(**kwargs):
        assert kwargs == {"input": "Hello", "max_output_tokens": 2}
        assert clients[0]._core.timeout == 40
        clients[0]._core._active = SimpleNamespace(cancel=cancel)
        clients[0].sessions[:] = [1, 1]
        events.append("active-attempt-started")
        expire()
        guard._run()  # Real inherited loop calls the shared trained SDK abort.
        assert stopped.is_set() and clients[0]._core._active is None
        assert guard.error_class == "BenchmarkMemoryError"
        return Response()  # Must not become accepted output after cancellation.

    def app(*, client, local_api_key):
        assert client is clients[0] and local_api_key == "corrected-audit"
        events.append("sidecar-app")
        return client

    class HTTP:
        def __init__(self, selected):
            assert selected is clients[0]

        def __enter__(self):
            events.append("http-enter")
            return self

        def __exit__(self, *error):
            events.append("http-close")

        def post(self, path, *, headers, json):
            assert path == "/v1/responses"
            assert headers == {"Authorization": "Bearer corrected-audit"}
            assert json == {"model": "mock", "input": "Hello", "max_output_tokens": 2,
                            "temperature": 0}
            events.append("http-post")
            return clients[0].responses.create(input=json["input"], max_output_tokens=json["max_output_tokens"])

    stopped, guard = Event(), None

    def protected_work():
        nonlocal guard
        source = tensors = client = response = value = None
        with conformance._corrected_prompt_extra_allowance(512 * memory.MiB) as reservation:
            assert type(reservation) is resources._MemoryReservation
            assert reservation._capacity is capacity and reservation._acquired
            assert reservation._amount == 512 * memory.MiB  # Raw, never 640 MiB charge.
            reservations.append(reservation)
            assert capacity.observe().reserved_memory_bytes == reservation._amount
            events.append("reserve-raw-512")
            live.extend([Payload(), Payload()])
            references.extend(weakref.ref(payload) for payload in live)
            try:
                if failure_phase.startswith("execution_"):
                    _corrected_controller_regression(
                        tmp_path, monkeypatch, failure_phase.removeprefix("execution_"),
                        deadline, expire=expire)
                else:
                    monkeypatch.setattr(memory, "threading", SimpleNamespace(
                        Event=Event, Thread=Thread, current_thread=lambda: None))
                    monkeypatch.setattr(memory, "host_memory", lambda: SimpleNamespace(
                        available=6 << 30, swap_used=0, pressure=None))
                    report = {"host": {"available": 6 << 30, "swap_used": 0, "reserve_bytes": 1 << 30}}
                    guard = conformance._corrected_prompt_watchdog(
                        report, conformance._corrected_prompt_abort(stopped, clients), deadline)
                    guard.start()
                    if failure_phase == "oracle":
                        source = SimpleNamespace(source_shapes={}, _tensor_specs={}, _identities={},
                                                 payload=live[0])
                        tensors = conformance.AdmittedTensors(source, check=guard.check)
                        expire()
                        tensors.verify_lock()  # Actual oracle callback precedes source access.
                    else:
                        client = Client()
                        clients.append(client)
                        monkeypatch.setattr(sidecar, "create_sidecar_app", app)
                        monkeypatch.setattr(http_boundary, "TestClient", HTTP)
                        response, value = conformance._corrected_prompt_dispatch(client, failure_phase, deadline)
                        captured.append(value)
            except BaseException as error:
                clear_failure_frames(error)
                raise
            finally:
                if guard is not None:
                    guard.close()  # No deadline check or new work budget in cleanup.
                if clients:
                    clients[0].close()
                    assert clients[0].closed and all(not role.running for role in clients[0].roles)
                clients.clear()
                captured.clear()
                live.clear()
                source = tensors = client = response = value = None
                gc.collect()
                events.append("payloads-source-exception-frames-retired")

    if failure_phase.startswith("execution_"):
        protected_work()  # Shared regression checks refusal/blocked result plus real drain.
    else:
        with pytest.raises(memory.BenchmarkMemoryError, match="deadline expired"):
            protected_work()
        assert guard._stop.is_set()
        assert type(guard)._run is memory.MemoryWatchdog._run
        assert type(guard).start is memory.MemoryWatchdog.start
        assert type(guard).close is memory.MemoryWatchdog.close
    assert capacity._owners == {} and captured == []
    reservation, = reservations
    expected_error = None if failure_phase.startswith("execution_") else memory.BenchmarkMemoryError
    assert exits == [(reservation._token, expected_error, False)]
    assert observations == [0, 512 * memory.MiB, 512 * memory.MiB, 0]
    assert resources.reserve_incremental_memory is original_factory
    assert events[-3:] == ["payloads-source-exception-frames-retired", "retired-before-refund", "refund-raw-512"]
    assert all(reference() is None for reference in references)
    if failure_phase in ("sdk", "sidecar"):
        assert events.index("active-attempt-started") < events.index("active-attempt-cancelled")
        assert events.index("active-attempt-cancelled") < events.index("mock-client-closed-roles-drained")
        assert len(retired_roles) == 2 and all(not role.running for role in retired_roles)
        assert events.count("http-post") == (1 if failure_phase == "sidecar" else 0)
        if failure_phase == "sidecar":
            assert events.index("http-close") < events.index("mock-client-closed-roles-drained")
    assert deadline.expires == 450 and clock.now == 451  # No stage/adapter reset.


def test_corrected_prompt_suite_builder_is_inert(monkeypatch):
    from examples.benchmarks.gkm_corrected_prompt import experiment, suite
    from pllm.metrics import benchmark_runner
    from pllm.runtime import benchmark_memory, delegated_source

    def forbidden(*args, **kwargs):
        pytest.fail("suite builder executed benchmark or source admission")

    monkeypatch.setattr(benchmark_runner, "_execute", forbidden)
    monkeypatch.setattr(benchmark_memory, "benchmark_memory", forbidden)
    monkeypatch.setattr(delegated_source, "admit_source", forbidden)
    supplied_budget = 704 << 20  # Configuration round-trip only, not fit authority.
    spec = suite(memory_bytes=supplied_budget).to_spec()
    case, = spec["cases"]
    target = case["target"]
    assert target["kind"] == "response"
    parameters = target["parameters"]
    assert parameters["backend"] == "native" and parameters["state_audit"] is True
    assert parameters["scenario"]["kind"] == "cold"
    assert parameters["scenario"]["count"] == 1
    assert parameters["scenario"]["max_output_tokens"] == 2
    assert parameters["model"]["experiment"] == experiment.to_spec()
    assert (experiment.budget.requests, experiment.budget.max_input_tokens,
            experiment.budget.max_new_tokens) == (1, 30, 2)
    sampling = spec["sampling"]
    assert (sampling["processes"], sampling["repetitions"], sampling["warmups"]) == (1, 1, 0)
    assert sampling["memory_bytes"] == supplied_budget
    assert sampling["timeout_seconds"] == 450


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("PLLM_RUN_GKM_CORRECTED_PROMPT") != "1",
                    reason="opt-in pinned offline GKM corrected Hello suite persistence")
def test_pinned_qwen_corrected_prompt_suite_persists_independent_audit(
        tmp_path, monkeypatch, record_property):
    from test_gkm_checkpoint_conformance import _CorrectedPromptDeadline

    deadline = _CorrectedPromptDeadline()
    from huggingface_hub.constants import HF_HUB_CACHE
    from pllm import Deployment
    from pllm.metrics import benchmark_runner
    from pllm.runtime import benchmark_memory as memory, delegated_source
    from pllm.runtime.delegated_protocol import clear_failure_frames
    from pllm.runtime.safetensors_store import SafeTensorStore
    from examples.benchmarks.gkm_corrected_prompt import experiment as saved, suite
    from test_gkm_checkpoint_conformance import (
        PIN, OUTPUTS, AdmittedTensors, SelectionCommitments, clear_decoder,
        _corrected_prompt_extra_allowance, _corrected_prompt_memory_admission,
        _independent_pinned_qwen_hello, _corrected_prompt_watchdog, CORRECTED_MEMORY_CAP,
    )

    cache = Path(HF_HUB_CACHE).expanduser().resolve()
    snapshot = cache / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / PIN
    if not snapshot.is_dir():
        pytest.skip(f"missing pinned offline checkpoint cache: {snapshot}; no download/fallback")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    experiment = saved.with_params(deployment=Deployment.local(root=str(tmp_path / "roles")))

    def forbidden(*args, **kwargs):
        raise AssertionError("corrected suite checkpoint values loaded before memory admission")

    with monkeypatch.context() as early:
        early.setattr(delegated_source, "_source_float_frame", forbidden)
        early.setattr(delegated_source.AdmittedSource, "load_transport", forbidden)
        early.setattr(SafeTensorStore, "get", forbidden)
        early.setattr(delegated_source._AdmittedTensorStore, "get", forbidden)
        source, estimate, combined_report, workspace, extra = _corrected_prompt_memory_admission(
            experiment, snapshot, audit_bytes=32 * memory.MiB, check=deadline.remaining)
    selected = suite(memory_bytes=CORRECTED_MEMORY_CAP, selected=experiment)
    assert (selected.sampling.processes, selected.sampling.repetitions,
            selected.sampling.warmups) == (1, 1, 0)
    expected_records, calls = [], []
    stopped = threading.Event()
    guard = _corrected_prompt_watchdog(combined_report, lambda reason: stopped.set(), deadline)

    def check():
        if stopped.is_set():
            raise memory.BenchmarkMemoryError(guard.error or "corrected suite cancelled")
        guard.check()

    original = benchmark_runner._execute

    def independently_checked_execute(case, sampling, inputs, salt, directory):
        assert calls == [] and inputs == {"prompt": "Hello"}
        calls.append(case.name)
        # Observe the existing private seam; never replace salt or RNG. Oracle
        # arrays retire BEFORE measured child starts; only commitments survive.
        tensors = expected = observer = None
        try:
            check()
            ids = _independent_pinned_qwen_hello(source)
            check()
            gc.collect()
            tensors = AdmittedTensors(source, check=check)
            tensors.verify_lock()
            observer = SelectionCommitments(salt, max_selections=OUTPUTS)
            expected = clear_decoder(source.config, tensors, ids, OUTPUTS - 1,
                                     check=check, selection_observer=observer)
            tensors.verify_lock()
            expected_records.append(observer.expected_record(source.contract))
        except BaseException as error:
            clear_failure_frames(error)
            raise
        finally:
            tensors = expected = observer = None
            gc.collect()
        check()
        with _combined_controller_watchdog(monkeypatch, combined_report, stopped, deadline):
            return original(case, _corrected_prompt_sampling(sampling, deadline),
                            inputs, salt, directory)

    monkeypatch.setattr(benchmark_runner, "_execute", independently_checked_execute)
    output = tmp_path / "corrected-benchmark.json"
    with _corrected_prompt_extra_allowance(extra):
        try:
            guard.start()
            report = benchmark_suite(selected, inputs={"gkm-corrected-prompt": {"prompt": "Hello"}},
                                     output=output)
            value, = [result.to_dict() for result in report.results]
            assert calls == ["gkm-corrected-prompt"]
            assert value["status"] == "completed", value
            check()
            assert value["comparison"]["source_digest"] == source.source_lock_digest
            assert value["details"]["runtime_contract"]["output_delivery"] == "buffered_until_verified"
            binding = value["details"]["response_state_audit"]
            assert binding["schema"] == "pllm.response_state_audit.gkm.v1"
            assert binding["bindings"] == expected_records
            assert [row["position"] for row in expected_records[0]["selections"]] == [30, 31]
            metrics = {entry["id"]: entry for entry in value["metrics"]}
            assert metrics["response.decode_tps"]["value"] is None
            assert metrics["response.decode_tps"]["unavailable_reason"] == (
                "request_end_verification_buffers_all_output")
            for role in ("worker_a", "worker_b"):
                assert metrics[f"{role}.cold_cpu"]["value"] > 0
                assert metrics[f"{role}.peak_rss"]["value"] > 0
            loaded = BenchmarkRun.load(output)
            assert loaded.digest == report.digest and loaded.to_dict() == report.to_dict()
            persisted = loaded.to_dict()
            assert persisted["complete"] is True and len(persisted["results"]) == 1
            assert persisted["results"][0] == value
            assert value["execution"]["case"] == selected.cases[0].to_spec()
            record_property("trained_requests", 1)
            record_property("source_lock_digest", source.source_lock_digest)
            record_property("checkpoint_digest", source.checkpoint_digest)
            record_property("tokenizer_digest", source.tokenizer_digest)
            record_property("oracle_workspace_estimate", workspace)
            record_property("oracle_and_audit_allowance_bytes", extra)
        except BaseException as error:
            clear_failure_frames(error)
            raise
        finally:
            try:
                guard.close()
            finally:
                source = None
                expected_records.clear()
                gc.collect()
    check()


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
