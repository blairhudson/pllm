"""Real Python exec/status-FD smoke tests, not native role qualification.

Use the existing isolated interpreter, launcher and inherited Unix socket
convention. No role server, source import, HTTP or native FSS work runs here.
"""
from contextlib import contextmanager
import os
import select
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from pllm.runtime import fss_child_status as status
from pllm.runtime.servers import _spawn_role_process


pytestmark = pytest.mark.skipif(os.name != "posix", reason="inherited Unix status FD")

_CHILD = """
import os, sys
from pllm.runtime import fss_child_status as status

writer = status.StatusWriter.from_environment()
assert writer is not None
assert status.ENV not in os.environ
assert status.StatusWriter.from_environment() is None
assert not writer.channel.get_inheritable()
assert not writer.channel.getblocking()
"""


def _packet(sequence=0, *, kind=status._UPDATE, phase="source_weights"):
    return status._RECORD.pack(status._MAGIC, 1, sequence, kind,
                               status.SOURCE_IMPORT_PHASES.index(phase))


@contextmanager
def _exec_child(script, *, role="worker_a", arguments=()):
    """Test-local ownership only; keep actual Popen, sockets and parser intact."""
    events, failures, records = [], [], []
    updated = threading.Event()

    def stopped():
        events.append("stop")

    def failed():
        assert events == ["stop"]
        failures.append(reader.snapshot())
        events.append("failure")

    reader = status.StatusReader(role, stopped, failed)
    accept = reader.accept

    def observe(packet):
        records.append(packet)
        accepted = accept(packet)
        if accepted:
            updated.set()
        return accepted

    reader.accept = observe
    process = None
    try:
        reader.start()
        environment = dict(os.environ)
        descriptor = reader.child.fileno()
        environment[status.ENV] = str(descriptor)
        process = _spawn_role_process(
            [sys.executable, "-I", "-c", _CHILD + script, *arguments],
            environment=environment, pass_fds=(descriptor,),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        reader.spawned()
        assert reader.child.fileno() == -1
        yield SimpleNamespace(process=process, reader=reader, records=records,
                              events=events, failures=failures, updated=updated)
    finally:
        reader.close()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=3)
        reader.thread.join(timeout=1)
        assert not reader.thread.is_alive()
        assert reader.closed and reader.snapshot() is None
        assert reader.child.fileno() == reader.channel.fileno() == -1
        assert reader.stop is reader.failure is None


def _finished(child):
    output, error = child.process.communicate(timeout=5)
    # Natural EOF must stop the reader without needing explicit reader.close().
    child.reader.thread.join(timeout=1)
    assert not child.reader.thread.is_alive()
    assert child.reader.closed and child.reader.snapshot() is None
    assert child.reader.channel.fileno() == -1
    assert child.reader.stop is child.reader.failure is None
    return output, error


def _ready(child):
    # A fixed stdout marker coordinates only these test children, not status IPC.
    ready, _, _ = select.select([child.process.stdout], [], [], 5)
    assert ready, "exec child did not reach the test barrier"
    assert os.read(child.process.stdout.fileno(), 32) == b"ready\n"


@pytest.mark.parametrize("role", status.ROLES)
def test_exec_writer_sends_only_phase_enums_and_success_eof(role):
    with _exec_child("""
try:
    with status.source_status_scope(writer) as scope:
        for phase in status.SOURCE_IMPORT_PHASES:
            scope.phase(phase)
    assert scope.signal is None and writer.current is None
finally:
    writer.close()
assert writer.channel.fileno() == -1
""", role=role) as child:
        output, error = _finished(child)
        assert child.process.returncode == 0, error.decode()
        assert output == error == b""
        expected = [_packet(phase=status.SOURCE_IMPORT_PHASES[0])]
        expected.extend(_packet(index, phase=phase) for index, phase in
                        enumerate(status.SOURCE_IMPORT_PHASES, start=1))
        expected.append(_packet(len(status.SOURCE_IMPORT_PHASES) + 1, kind=status._END,
                                phase=status.SOURCE_IMPORT_PHASES[-1]))
        assert child.records == expected
        assert all(len(packet) == status._RECORD.size for packet in child.records)
        assert child.reader.count == len(expected)
        assert not child.events and not child.failures


@pytest.mark.parametrize("signal_raises", [False, True])
def test_exec_original_exception_survives_failure_signal_and_eof(signal_raises):
    with _exec_child("""
original = ValueError("original exec source failure")
signals = []
def stop():
    signals.append(True)
    if sys.argv[1] == "raise":
        raise RuntimeError("diagnostic callback failure")
try:
    try:
        with status.source_status_scope(writer, signal=stop) as scope:
            scope.phase("source_weights")
            raise original
    except ValueError as caught:
        assert caught is original
        assert signals == [True]
        assert scope.signal is None and writer.current is None
        raise
finally:
    writer.close()
""", role="dealer", arguments=("raise" if signal_raises else "return",)) as child:
        output, error = _finished(child)
        assert child.process.returncode == 1
        assert output == b""
        assert error.rstrip().endswith(b"ValueError: original exec source failure")
        assert b"RuntimeError: diagnostic callback failure" not in error
        assert b"AssertionError" not in error
        assert child.records == [_packet(phase=status.SOURCE_IMPORT_PHASES[0]),
                                 _packet(1), _packet(2, kind=status._FAIL)]
        assert child.events == ["stop", "failure"]
        assert child.failures == [{"active": True, "phase": "source_weights",
                                   "role": "dealer", "failed": True}]


def test_exec_non_enum_phase_disables_writer_without_sending_text():
    with _exec_child("""
try:
    with status.source_status_scope(writer) as scope:
        scope.phase("private-path-not-a-phase")
        assert writer.closed
        scope.phase("source_weights")
finally:
    writer.close()
""") as child:
        output, error = _finished(child)
        assert child.process.returncode == 0, error.decode()
        assert output == error == b""
        assert child.records == [_packet(phase=status.SOURCE_IMPORT_PHASES[0])]
        assert not child.events and not child.failures


def test_exec_writer_capability_is_not_inherited_by_another_exec():
    with _exec_child('''
import subprocess
try:
    result = subprocess.run([sys.executable, "-I", "-c", """
import errno, os, sys
assert 'PLLM_FSS_STATUS_FD' not in os.environ
try:
    os.fstat(int(sys.argv[1]))
except OSError as error:
    assert error.errno == errno.EBADF
else:
    raise AssertionError("status capability survived another exec")
""", str(writer.channel.fileno())], close_fds=False, capture_output=True, timeout=3)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == result.stderr == b""
    with status.source_status_scope(writer) as scope:
        scope.phase("source_weights")
finally:
    writer.close()
''') as child:
        output, error = _finished(child)
        assert child.process.returncode == 0, error.decode()
        assert output == error == b""
        assert child.records == [_packet(phase=status.SOURCE_IMPORT_PHASES[0]),
                                 _packet(1), _packet(2, kind=status._END)]
        assert not child.events and not child.failures


def test_exec_fragmented_writes_reassemble_complete_records():
    packets = [_packet(), _packet(1, phase="source_rotary"),
               _packet(2, kind=status._END, phase="source_rotary")]
    with _exec_child("""
try:
    for encoded in sys.argv[1:]:
        packet = bytes.fromhex(encoded)
        for part in (packet[:1], packet[1:7], packet[7:]):
            writer.channel.sendall(part)
finally:
    writer.close()
""", arguments=tuple(packet.hex() for packet in packets)) as child:
        output, error = _finished(child)
        assert child.process.returncode == 0, error.decode()
        assert output == error == b""
        assert child.records == packets and child.reader.count == 3
        assert not child.events and not child.failures


@pytest.mark.parametrize("invalid", [
    pytest.param(b"PATH" + _packet(1)[4:], id="magic"),
    pytest.param(status._RECORD.pack(status._MAGIC, 1, 1, status._UPDATE, 255), id="phase"),
    pytest.param(status._RECORD.pack(status._MAGIC, 1, 1, 9, 0), id="kind"),
    pytest.param(_packet(2), id="sequence-gap"),
    pytest.param(status._RECORD.pack(status._MAGIC, 3, 0, status._UPDATE, 0), id="generation-gap"),
])
def test_exec_malformed_record_closes_reader_and_clears_active_phase(invalid):
    with _exec_child("""
try:
    writer.channel.sendall(bytes.fromhex(sys.argv[1]))
finally:
    writer.close()
""", arguments=((_packet() + invalid + _packet(2)).hex(),)) as child:
        output, error = _finished(child)
        assert child.process.returncode == 0, error.decode()
        assert output == error == b""
        assert child.records == [_packet(), invalid]  # No parsing after rejection.
        assert child.reader.count == 2
        assert not child.events and not child.failures


def test_exec_partial_record_at_eof_is_not_published():
    with _exec_child("""
try:
    writer.channel.sendall(bytes.fromhex(sys.argv[1]))
finally:
    writer.close()
""", arguments=((_packet() + _packet(1, kind=status._FAIL)[:7]).hex(),)) as child:
        output, error = _finished(child)
        assert child.process.returncode == 0, error.decode()
        assert output == error == b""
        assert child.records == [_packet()] and child.reader.count == 1
        assert not child.events and not child.failures


@pytest.mark.parametrize("partial", [False, True])
def test_close_unblocks_reader_with_live_exec_child(partial):
    with _exec_child("""
import signal
if sys.argv[1] == "partial":
    writer.channel.sendall(bytes.fromhex(sys.argv[2])[:7])
else:
    scope = writer.begin()
    scope.phase("source_weights")
print("ready", flush=True)
signal.pause()
""", arguments=("partial" if partial else "complete", _packet().hex())) as child:
        _ready(child)
        if not partial:
            assert child.updated.wait(1)
        assert child.process.poll() is None
        started = time.monotonic()
        child.reader.close()
        assert time.monotonic() - started < 2
        child.reader.thread.join(timeout=1)
        assert not child.reader.thread.is_alive()
        assert child.reader.snapshot() is None
        assert child.process.poll() is None  # Diagnostic closure does not kill work.
        child.process.terminate()
        output, error = child.process.communicate(timeout=3)
        assert child.process.returncode == -signal.SIGTERM
        assert output == error == b""
        assert not child.events and not child.failures


def test_closed_supervisor_socket_cannot_replace_child_body_exception():
    with _exec_child("""
import signal, threading
proceed = threading.Event()
signal.signal(signal.SIGUSR1, lambda *_: proceed.set())
print("ready", flush=True)
assert proceed.wait(5)
try:
    with status.source_status_scope(writer) as scope:
        assert writer.closed  # Real peer closure disables the nonblocking sender.
        raise ValueError("original after peer closure")
finally:
    writer.close()
""") as child:
        _ready(child)
        child.reader.close()
        child.process.send_signal(signal.SIGUSR1)
        output, error = _finished(child)
        assert child.process.returncode == 1
        assert output == b""
        assert error.rstrip().endswith(b"ValueError: original after peer closure")
        assert b"BrokenPipeError" not in error and b"AssertionError" not in error
        assert child.records == []
        assert not child.events and not child.failures
