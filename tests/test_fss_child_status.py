"""Local diagnostics tests, ASGI doubles and one stdlib-only Python exec.

No native FSS work, trained imports, checkpoints or SDK execution.
"""
import asyncio
from contextlib import contextmanager
import os
import socket
import sys
import threading
from types import SimpleNamespace as NS
import weakref

import httpx
import pytest

from pllm.runtime import benchmark_failure as diagnostics
from pllm.runtime import benchmark_memory as memory
from pllm.runtime import fss_child_status as status
from pllm.runtime import fss_worker as worker
from pllm.runtime import servers
from test_fss_worker import (client_key, configuration, connections, contract, credentials,
                             double as _worker_double)
from test_fss_benchmark import contract as _price_contract, decoder_contract

worker_double = _worker_double
price_contract = _price_contract
SECRET = "private/prompt/path/token/source-payload"


def topology():
    value = servers.LocalTopology.__new__(servers.LocalTopology)
    value._process_lock = threading.Lock()
    value._stopping = threading.Event()
    value._startup_breadcrumbs = {"phase": "ready", "role": None}
    value._child_source_status = {}
    value._failure_observer = None
    return value


def collector_for(value):
    runtime = NS(_topology=value, _lock=threading.Lock(), _state={"phase": "online"},
                 _active_run=object(), _has_started_run=True, _memory_guard=None)
    return diagnostics.FailureDiagnostics(runtime)


@contextmanager
def channel(role="worker_a"):
    owner = topology()
    collector = collector_for(owner)
    observed = threading.Event()

    def failure():
        assert owner._stopping.is_set()
        collector.capture()
        observed.set()

    owner._failure_observer = failure
    reader = status.StatusReader(role, owner.request_stop, owner._observe_failure)
    owner._child_source_status[role] = reader
    # dup models the explicit descriptor inherited by exec; no child is launched.
    writer = status.StatusWriter(reader.child.dup())
    reader.spawned()
    try:
        yield owner, collector, reader, writer, observed
    finally:
        writer.close()
        reader.close()


def packet(generation=1, sequence=0, kind=status._UPDATE, phase="source_weights"):
    return status._RECORD.pack(status._MAGIC, generation, sequence, kind,
                               status.SOURCE_IMPORT_PHASES.index(phase))


def test_cached_child_import_precedes_request_and_completion_cannot_revive():
    with channel() as (_, collector, reader, _, _):
        assert reader.accept(packet())
        assert collector.observe()["phase"] == "source_weights"
        assert collector.observe()["role"] == "worker_a"
        assert reader.accept(packet(sequence=1, kind=status._END))
        assert collector.observe()["phase"] == "request"
        assert reader.accept(packet(sequence=2))  # Completed scope stays closed.
        assert collector.observe()["phase"] == "request"
        assert reader.accept(packet(generation=2, phase="source_headers"))
        assert reader.accept(packet(generation=1, sequence=3))
        assert collector.observe()["phase"] == "source_headers"


@pytest.mark.parametrize("raw", [
    b"", b"x" * 15, b"x" * 17, b"payload" * 20,
    status._RECORD.pack(b"PATH", 1, 0, 1, 0),
    status._RECORD.pack(status._MAGIC, 0, 0, 1, 0),
    status._RECORD.pack(status._MAGIC, 66, 0, 1, 0),
    status._RECORD.pack(status._MAGIC, 1, 32, 1, 0),
    status._RECORD.pack(status._MAGIC, 1, 0, 9, 0),
    status._RECORD.pack(status._MAGIC, 1, 0, 1, 255),
    packet(sequence=1), packet(kind=status._FAIL), packet(generation=2),
])
def test_invalid_child_records_disable_channel_without_payload_retention(raw):
    with channel() as (_, _, reader, _, observed):
        assert reader.accept(raw) is False
        assert reader.snapshot() is None
        assert not observed.is_set()


def test_fixed_total_record_bound_and_no_event_history():
    with channel() as (_, _, reader, _, _):
        assert reader.accept(packet())
        for _ in range(status._MAX_RECORDS - 1):
            assert reader.accept(packet())  # Replays count against the bound too.
        assert not reader.accept(packet())
        assert set(reader.snapshot()) == {"active", "phase", "role"}


def test_multiple_imports_are_ambiguous_but_explicit_failure_identifies_its_role():
    with channel() as (owner, collector, first, _, observed):
        second = status.StatusReader("worker_b", owner.request_stop, owner._observe_failure)
        owner._child_source_status["worker_b"] = second
        try:
            first.accept(packet())
            second.accept(packet(phase="source_headers"))
            assert collector.observe()["role"] is None
            first.accept(packet(sequence=1, kind=status._FAIL))
            assert observed.is_set()
            assert collector.capture()["role"] == "worker_a"
        finally:
            second.close()


def test_stop_precedes_status_lock_and_failure_snapshot_is_first_and_frozen():
    with channel() as (owner, collector, reader, _, observed):
        assert reader.accept(packet())
        stopped = threading.Event()
        original_stop = reader.stop
        reader.stop = lambda: (original_stop(), stopped.set())
        with reader.lock:
            thread = threading.Thread(target=lambda: reader.accept(
                packet(sequence=1, kind=status._FAIL)))
            thread.start()
            assert stopped.wait(1)
            assert owner._stopping.is_set()
            assert collector.observe()["phase"] == "request"  # Contended => unavailable.
        thread.join(timeout=1)
        assert not thread.is_alive() and observed.is_set()
        assert collector.capture()["phase"] == "source_weights"
        assert reader.snapshot() is None
        reader.accept(packet(generation=2, phase="source_tokenizer"))
        assert collector.capture()["phase"] == "source_weights"
        assert SECRET not in str(collector.capture())


def test_writer_stop_precedes_its_lock_and_never_waits_on_contention():
    with channel() as (_, _, _, writer, _):
        stopped = threading.Event()
        scope = writer.begin(stopped.set)
        with writer.lock:
            scope.fail()
            assert stopped.is_set() and writer.closed


@pytest.mark.parametrize("terminal", ["finish", "fail"])
def test_completed_status_scope_does_not_retain_attempt_or_lease_owners(terminal):
    class Owner:
        def signal(self):
            pass

    with channel() as (_, _, _, writer, _):
        owner = Owner()
        reference = weakref.ref(owner)
        scope = writer.begin(owner.signal)
        del owner
        assert reference() is not None
        getattr(scope, terminal)()
        assert scope.signal is None and writer.current is None
        assert reference() is None


def test_body_error_survives_signal_error_and_fail_is_still_published():
    stopped = threading.Event()

    class Owner:
        def signal(self):
            stopped.set()
            raise RuntimeError(SECRET)

    with channel() as (_, collector, reader, writer, observed):
        owner = Owner()
        reference = weakref.ref(owner)
        signal = owner.signal
        del owner
        body_error = ValueError("original source failure")
        with pytest.raises(ValueError) as caught:
            with status.source_status_scope(writer, signal=signal) as scope:
                scope.phase("source_weights")
                raise body_error
        del signal
        assert caught.value is body_error and stopped.is_set()
        assert scope.signal is None and writer.current is None and reference() is None
        writer.close()
        reader.start()
        reader.thread.join(timeout=1)
        assert not reader.thread.is_alive() and observed.is_set()
        assert collector.capture()["phase"] == "source_weights"


def test_reader_callback_errors_are_contained_and_closed_reader_releases_owners():
    stopped = threading.Event()
    calls = []

    class Owner:
        def stop(self):
            stopped.set()
            calls.append("stop")
            raise RuntimeError(SECRET)

        def fail(self):
            assert stopped.is_set()
            calls.append("failure")
            raise RuntimeError(SECRET)

    owner = Owner()
    reference = weakref.ref(owner)
    reader = status.StatusReader("worker_a", owner.stop, owner.fail)
    del owner
    try:
        reader.accept(packet())
        assert reader.accept(packet(sequence=1, kind=status._FAIL))
        assert calls == ["stop", "failure"]
        assert reader.snapshot() is None
    finally:
        reader.close()
    assert reader.stop is reader.failure is None and reference() is None


@pytest.mark.parametrize("mode", ["blocked", "partial"])
def test_writer_full_or_partial_send_disables_without_retry_or_blocking(mode):
    class Full:
        calls = 0
        def send(self, packet):
            self.calls += 1
            if mode == "blocked":
                raise BlockingIOError
            return len(packet) - 1
        def close(self):
            pass

    with channel() as (_, _, _, writer, _):
        writer.channel.close()
        writer.channel = Full()
        with status.source_status_scope(writer) as scope:
            scope.phase("source_weights")
        assert writer.closed and writer.channel.calls == 1


def test_delayed_writer_scope_and_cancelled_failure_do_not_emit_late_updates():
    with channel() as (_, _, reader, writer, _):
        first = writer.begin()
        second = writer.begin()
        first.phase("source_weights")
        assert first.sequence == 1
        second.finish()
        second.phase("source_manifest")
        assert second.sequence == 2
        with pytest.raises(asyncio.CancelledError):
            with status.source_status_scope(writer) as scope:
                scope.phase("source_headers")
                raise asyncio.CancelledError
        assert scope.ended
        # Drain queued data after the producer has already unwound. Its failure
        # phase was frozen before unwind, and END cannot replace that record.
        writer.close()
        reader.start()
        reader.thread.join(timeout=1)
        assert not reader.thread.is_alive()


def test_eof_truncation_and_close_leave_no_active_scope_or_reader_thread():
    with channel() as (_, collector, reader, writer, _):
        reader.start()
        writer.channel.send(packet()[:7])
        writer.close()
        reader.thread.join(timeout=1)
        assert not reader.thread.is_alive()
        assert reader.accept(packet()) is False  # Disconnected scopes cannot revive.
        assert reader.snapshot() is None
        assert collector.observe()["phase"] == "request"


def test_fragmented_valid_records_are_reassembled_without_retaining_history():
    with channel() as (_, collector, reader, _, _):
        packets = [packet(), packet(sequence=1, phase="source_rotary"),
                   packet(sequence=2, kind=status._END)]
        fragments = iter([part for value in packets
                          for part in (value[:1], value[1:7], value[7:])])
        original_accept = reader.accept
        parsed = []

        def accept(value):
            parsed.append(value)
            return original_accept(value)

        class Fragmented:
            def recv(self, bound):
                value = next(fragments, b"")
                assert len(value) <= bound
                return value
            def close(self):
                pass
            def shutdown(self, how):
                pass

        reader.accept = accept
        reader.channel.close()
        reader.channel = Fragmented()
        reader._run()
        assert parsed == packets and reader.count == 3
        assert reader.closed and reader.snapshot() is None
        assert collector.observe()["phase"] == "request"


def test_malformed_record_after_active_update_clears_cache_and_role_attribution():
    with channel() as (_, collector, reader, writer, _):
        writer.channel.sendall(packet() + b"x" * status._RECORD.size)
        reader.start()
        reader.thread.join(timeout=1)
        assert not reader.thread.is_alive()
        assert reader.snapshot() is None
        assert collector.observe()["role"] is None


def test_contended_close_and_snapshots_do_not_wait_for_status_lock():
    with channel() as (_, collector, reader, _, _):
        reader.accept(packet())
        with reader.lock:
            assert reader.snapshot() is None
            reader.close()
            assert reader.snapshot() is None
            assert collector.observe()["phase"] == "request"


def test_environment_is_consumed_and_only_inherited_local_socket_is_used(monkeypatch):
    left, right = socket.socketpair()
    descriptor = right.detach()
    monkeypatch.setenv(status.ENV, str(descriptor))
    writer = status.StatusWriter.from_environment()
    try:
        assert writer is not None
        assert status.ENV not in os.environ
        assert writer.channel.get_inheritable() is False
    finally:
        left.close()
        writer.close()
    for raw in (SECRET, "0", "-1", "999999999999999999999"):
        monkeypatch.setenv(status.ENV, raw)
        assert status.StatusWriter.from_environment() is None


def test_spawn_passes_only_bound_role_descriptor_and_failure_closes_it(monkeypatch):
    owner = topology()
    owner._privacy_mode = "fss_public"
    owner._log_dir = None
    owner._processes = {}
    owner._startup_breadcrumbs["spawned_roles"] = []
    monkeypatch.setattr(servers.LocalTopology, "_environment", lambda *a: {})
    received = []

    def spawn(command, **kwargs):
        descriptor, = kwargs["pass_fds"]
        assert kwargs["env"][status.ENV] == str(descriptor)
        assert kwargs["start_new_session"] is True
        received.append(owner._child_source_status["worker_a"])
        raise OSError(SECRET)

    monkeypatch.setattr(servers.subprocess, "Popen", spawn)
    with pytest.raises(OSError):
        owner._spawn("worker_a", [])
    assert not owner._child_source_status and not owner._processes
    assert received[0].closed and received[0].child.fileno() == -1
    assert not received[0].thread.is_alive()


@pytest.mark.parametrize("failure", ["socketpair", "setsockopt", "thread_start", "stack_size"])
def test_optional_status_setup_failure_cleans_up_and_does_not_prevent_spawn(monkeypatch, failure):
    owner = topology()
    owner._privacy_mode = "fss_public"
    owner._log_dir = None
    owner._processes = {}
    owner._startup_breadcrumbs["spawned_roles"] = []
    monkeypatch.setattr(servers.LocalTopology, "_environment", lambda *a: {})
    endpoints = []
    original_pair = socket.socketpair

    class BadBuffer:
        closed = False
        def setsockopt(self, *args):
            raise OSError(SECRET)
        def close(self):
            self.closed = True
        def fileno(self):
            return -1 if self.closed else 10

    def pair():
        if failure == "socketpair":
            raise OSError(SECRET)
        values = (BadBuffer(), BadBuffer()) if failure == "setsockopt" else original_pair()
        endpoints.extend(values)
        return values

    def start(self):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(status.socket, "socketpair", pair)
    if failure == "thread_start":
        monkeypatch.setattr(status.threading.Thread, "start", start)
    elif failure == "stack_size":
        monkeypatch.setattr(status.threading, "stack_size", lambda: 16 << 20)
    calls = []

    def spawn(command, **kwargs):
        calls.append(kwargs)
        assert "pass_fds" not in kwargs and status.ENV not in kwargs["env"]
        return NS(pid=1)

    monkeypatch.setattr(servers.subprocess, "Popen", spawn)
    assert owner._spawn("worker_a", []).pid == 1
    assert len(calls) == 1 and owner._child_source_status == {}
    assert all(value.fileno() == -1 for value in endpoints)


@pytest.mark.skipif(os.name != "posix", reason="inherited Unix status descriptor")
def test_real_stdlib_python_exec_inherits_only_explicit_status_fd_and_tears_down(monkeypatch):
    owner = topology()
    owner._privacy_mode = "fss_public"
    owner._log_dir = None
    owner._processes = {}
    owner._logs = {}
    owner._closed = False
    owner._close_lock = threading.Lock()
    owner._role_ids = ("worker_a",)
    owner._local_engine = None
    owner._role_credentials = {}
    owner._peer_credentials = {}
    owner._startup_breadcrumbs["spawned_roles"] = []
    released = []
    owner._capacity_reservations = NS(close=lambda: released.append(True))
    monkeypatch.setattr(servers.LocalTopology, "_environment", lambda *a: {})
    parsed = []
    original_accept = status.StatusReader.accept

    def accept(self, value):
        parsed.append(value)
        return original_accept(self, value)

    monkeypatch.setattr(status.StatusReader, "accept", accept)
    spare_read, spare_write = os.pipe()
    os.set_inheritable(spare_read, True)
    os.set_inheritable(spare_write, True)
    # No pllm import in child: only Python stdlib and the private diagnostic FD.
    script = """
import os, sys
for other in map(int, sys.argv[3:]):
    try:
        os.fstat(other)
    except OSError:
        pass
    else:
        raise AssertionError('unpassed descriptor inherited')
import socket, struct
assert 'pllm' not in sys.modules
fd = int(os.environ.pop('PLLM_FSS_STATUS_FD'))
stream = socket.socket(fileno=fd)
assert stream.family == socket.AF_UNIX
for sequence, kind in ((0, 1), (1, 1), (2, 2)):
    value = struct.pack('<4sQHBB', bytes.fromhex(sys.argv[1]), 1, sequence, kind, int(sys.argv[2]))
    for part in (value[:1], value[1:7], value[7:]):
        stream.sendall(part)
stream.close()
"""
    process = None
    try:
        process = owner._spawn("worker_a", [sys.executable, "-I", "-S", "-c", script,
            status._MAGIC.hex(), str(status.SOURCE_IMPORT_PHASES.index("source_weights")),
            str(spare_read), str(spare_write)])
        reader = owner._child_source_status["worker_a"]
        assert reader.child.fileno() == -1
        assert process.wait(timeout=5) == 0
        reader.thread.join(timeout=1)
        assert not reader.thread.is_alive()
        assert parsed == [packet(sequence=0), packet(sequence=1),
                          packet(sequence=2, kind=status._END)]
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        owner.close()
        os.close(spare_read)
        os.close(spare_write)
    assert owner._stopping.is_set() and not owner._child_source_status
    assert reader.closed and reader.channel.fileno() == -1
    assert reader.stop is reader.failure is None and released == [True]


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
@pytest.mark.parametrize("margin", [0, 25])
def test_status_process_overhead_is_additive_without_cap_or_relay_credit(price_contract, capacity, margin):
    raw = decoder_contract(price_contract) if capacity == "decoder" else price_contract
    estimate = memory._fss_memory({}, configuration(capacity).pipeline, 2, 3, 0, None, 0,
                                  contract=raw, memory_safety_margin_percent=margin)
    owners = estimate["allocation_owners"]
    status_owner = owners["local_source_status"]
    assert status_owner["supervisor_bytes"] == 26 << 20
    assert status_owner["per_role_bytes"] == 1 << 20
    assert 4 * len(status.ROLES) * status._SOCKET_BUFFER_MAX_BYTES <= 1 << 20
    preflight = owners["source_metadata_preflight_bytes_per_process"]
    base_client = 3 * memory._PROCESS_BYTES + max(sum(owners["client"].values()), preflight) + (32 << 20)
    base_role = memory._PROCESS_BYTES + max(sum(owners["each_role"].values()), preflight)
    progress = owners["execution_progress"]
    assert progress["python_bytes_per_holder"] == progress["supervisor_bytes_per_holder"] == 16 << 10
    supervisor_progress = 2 * progress["supervisor_bytes_per_holder"]
    assert estimate["client_peak_bytes"] == memory._slack(base_client + (26 << 20) + supervisor_progress, margin)
    holder_extra = max(sum(owners["each_role"].values()) + progress["python_bytes_per_holder"], preflight) - max(sum(owners["each_role"].values()), preflight)
    assert all(value == memory._slack(base_role + (1 << 20)
               + (holder_extra if role in ("worker_a", "worker_b") else 0), margin)
               for role, value in estimate["provider_peak_bytes"].items())
    baseline = memory._slack(base_client, margin) + 3 * memory._slack(base_role, margin)
    assert estimate["native_total_peak_bytes"] - baseline == memory._slack(
        (29 << 20) + supervisor_progress + 2 * holder_extra, margin)
    assert owners["relay_workspace_bytes"] == 32 << 20
    assert owners["each_role"]["native_kernel_owner"] == 64 << 20
    assert estimate["incremental_owner_bounds"]["per_role_execution"] == worker.p.ROLE_MEMORY


def test_worker_main_consumes_inherited_descriptor_and_closes_after_server(monkeypatch):
    with channel() as (_, _, reader, old_writer, _):
        descriptor = os.dup(old_writer.channel.fileno())
        old_writer.close()
        monkeypatch.setenv(status.ENV, str(descriptor))
        monkeypatch.setenv("PLLM_ROLE_EXPERIMENT_JSON", configuration().canonical_bytes().decode())
        monkeypatch.setenv("PLLM_FSS_API_KEY", SECRET)
        monkeypatch.setenv("PLLM_FSS_PEERS_JSON", "{}")
        monkeypatch.setenv("PLLM_FSS_PEER_CREDENTIALS_JSON", "{}")
        monkeypatch.setattr(sys, "argv", ["worker", "--role", "worker_a", "--port", "1"])
        captured = []

        def create(*args, source_status, **kwargs):
            captured.append(source_status)
            with status.source_status_scope(source_status) as scope:
                scope.phase("source_config")
            return object()

        monkeypatch.setattr(worker, "create_fss_worker_app", create)
        monkeypatch.setitem(sys.modules, "uvicorn", NS(run=lambda *a, **k: None))
        worker.main()
        assert captured[0].closed
        reader.start()
        reader.thread.join(timeout=1)
        assert not reader.thread.is_alive() and reader.count == 3
        assert reader.snapshot() is None


def test_worker_startup_failure_freezes_metadata_phase_before_cleanup(worker_double, monkeypatch):
    cleanup = []

    def resolve(*args, lifecycle, on_failure, **kwargs):
        lifecycle("source_tokenizer")
        on_failure()
        cleanup.append(True)
        raise ValueError(SECRET)

    monkeypatch.setattr(worker, "resolve_client_source", resolve)
    with channel("dealer") as (_, collector, reader, writer, observed):
        with pytest.raises(ValueError, match=SECRET):
            worker.create_fss_worker_app(configuration(), "dealer", client_key("dealer"),
                peer_connections=connections("dealer"), peer_credentials=credentials("dealer"),
                source_status=writer)
        assert cleanup == [True]
        # Deliberately receive only after cleanup: the child's frozen failure
        # survives unwind, without claiming synchronous supervisor delivery.
        writer.close()
        reader.start()
        reader.thread.join(timeout=1)
        assert not reader.thread.is_alive() and observed.is_set()
        assert collector.capture()["phase"] == "source_tokenizer"
        assert collector.capture()["role"] == "dealer"


@pytest.mark.asyncio
async def test_real_worker_admit_forwards_phase_and_signals_before_partial_cleanup(worker_double, monkeypatch):
    original = worker.resolve_client_source
    cleaned = []

    def resolve(experiment, *, expected_pins, lifecycle, on_failure):
        lifecycle("source_tokenizer")
        return original(experiment, expected_pins=expected_pins)

    def load(*args, lifecycle, on_failure, cancelled, **kwargs):
        lifecycle("source_weights")
        on_failure()
        cleaned.append(cancelled())  # Source owner cleanup starts AFTER signal/snapshot send.
        raise ValueError(SECRET)

    monkeypatch.setattr(worker, "resolve_client_source", resolve)
    monkeypatch.setattr(worker, "load_role_source", load)
    worker_double.making_role = "worker_a"
    with channel() as (owner, collector, reader, writer, observed):
        reader.start()
        app = worker.create_fss_worker_app(configuration(), "worker_a", client_key("worker_a"),
            peer_connections=connections("worker_a"), peer_credentials=credentials("worker_a"),
            source_status=writer)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                    base_url="http://127.0.0.1", headers={
                        "authorization": f"Bearer {client_key('worker_a')}"}) as client:
                manifest = worker.p.unpack((await client.get("/v1/fss/manifest")).content)
                epochs = {role: str(index + 2) * 64 for index, role in enumerate(worker.p.ROLES)}
                epochs["worker_a"] = manifest["epoch"]
                result = await client.post("/v1/fss/admit", content=worker.p.pack({
                    "request": "1" * 64, "epoch": manifest["epoch"], "epochs": epochs,
                    "rows": 3, "outputs": 3, "contract": contract()}, limit=worker.p.ADMISSION_LIMIT))
                assert result.status_code != 200
        assert observed.wait(1) and owner._stopping.is_set()
        assert cleaned == [True]
        assert collector.capture()["phase"] == "source_weights"
        assert collector.capture()["role"] == "worker_a"
