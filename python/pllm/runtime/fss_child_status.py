"""Bounded local diagnostic side channel, never an MPC or source-data channel.

Only the supervisor's inherited AF_UNIX socket grants write access. Role identity
comes from that socket, not the child record. Delivery is asynchronous and may be
unavailable; diagnostics never wait/retry/acknowledge or grant admission authority.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import socket
import struct
import threading

from .benchmark_failure import SOURCE_IMPORT_PHASES


ENV = "PLLM_FSS_STATUS_FD"
ROLES = ("worker_a", "worker_b", "dealer")
_RECORD = struct.Struct("<4sQHBB")
_MAGIC = b"FSD1"
_UPDATE, _END, _FAIL = 1, 2, 3
_SCOPES, _STEPS = 65, 32  # Startup plus existing <=64 requests; no event history.
_MAX_RECORDS = _SCOPES * _STEPS
# Separate process overhead, not credit against relay/native/client workspaces.
# Follow the existing conservative 8 MiB stack convention; disable optional
# readers if an explicitly configured Python stack exceeds that allowance.
_THREAD_STACK_BYTES = 8 << 20
_SOCKET_BUFFER_MAX_BYTES = 64 << 10
# At most 768 KiB socket buffers (3 pairs, 4 buffers/pair), rounded to
# 1 MiB kernel overhead, plus 1 MiB Python reader/cache/control metadata.
SUPERVISOR_MEMORY_BYTES = len(ROLES) * _THREAD_STACK_BYTES + (2 << 20)
WORKER_MEMORY_BYTES = 1 << 20


class StatusWriter:
    def __init__(self, channel):
        self.channel = channel
        channel.setblocking(False)
        os.set_inheritable(channel.fileno(), False)
        self.lock = threading.Lock()
        self.generation = 0
        self.current = None
        self.closed = False

    @classmethod
    def from_environment(cls):
        raw = os.environ.pop(ENV, None)
        if raw is None:
            return None
        channel = None
        try:
            if not 1 <= len(raw) <= 10 or not raw.isdecimal() or not 3 <= int(raw) <= (1 << 31) - 1:
                return None
            channel = socket.socket(fileno=int(raw))
            if channel.family != socket.AF_UNIX or channel.type != socket.SOCK_STREAM:
                channel.close()
                return None
            return cls(channel)
        except (OSError, ValueError):
            if channel is not None:
                channel.close()
            return None

    def close(self):
        self.closed = True
        self.current = None
        try:
            self.channel.close()
        except OSError:
            pass

    def _send(self, scope, kind):
        if self.closed or scope.sequence >= _STEPS:
            self.close()
            return
        packet = _RECORD.pack(_MAGIC, scope.generation, scope.sequence, kind, scope.phase_index)
        try:
            # A partial/blocked send disables diagnostics, never waits or retries.
            if self.channel.send(packet) != len(packet):
                self.close()
        except OSError:
            self.close()
        scope.sequence += 1

    def begin(self, signal=None):
        scope = _Scope(self, signal)
        if not self.lock.acquire(blocking=False):
            self.close()
            return scope
        try:
            if self.closed or self.generation >= _SCOPES:
                self.close()
                return scope
            self.generation += 1
            scope.generation = self.generation
            self.current = scope
            self._send(scope, _UPDATE)
        finally:
            self.lock.release()
        return scope


class _Scope:
    def __init__(self, writer=None, signal=None):
        self.writer, self.signal = writer, signal
        self.generation = self.sequence = self.phase_index = 0
        self.ended = False

    @property
    def callbacks(self):
        return {} if self.writer is None else {"lifecycle": self.phase, "on_failure": self.fail}

    def _publish(self, phase=None, kind=_UPDATE):
        writer = self.writer
        if writer is None or self.ended:
            return
        if not writer.lock.acquire(blocking=False):
            writer.close()
            return
        try:
            if writer.current is not self or self.ended:
                return
            if phase is not None:
                if type(phase) is not str or phase not in SOURCE_IMPORT_PHASES:
                    writer.close()
                    return
                self.phase_index = SOURCE_IMPORT_PHASES.index(phase)
            if kind != _UPDATE:
                self.ended = True
            writer._send(self, kind)
            if kind != _UPDATE and writer.current is self:
                writer.current = None
        finally:
            writer.lock.release()

    def phase(self, phase):
        self._publish(phase)

    def fail(self):
        if self.ended:
            return
        try:
            if self.writer is not None:
                if self.signal is not None:
                    try:
                        self.signal()  # Stop BEFORE any diagnostic lock.
                    except Exception:
                        pass  # Preserve source error; still attempt frozen FAIL.
                self._publish(kind=_FAIL)
        finally:
            self.ended = True
            self.signal = None  # Never retain the attempt/source/leases in diagnostics.

    def finish(self):
        try:
            self._publish(kind=_END)
        finally:
            self.ended = True
            self.signal = None


@contextmanager
def source_status_scope(writer, signal=None):
    scope = _Scope() if writer is None else writer.begin(signal)
    try:
        yield scope
    except BaseException:
        scope.fail()  # Freeze child phase before caller/native owner cleanup.
        raise
    finally:
        scope.finish()  # Failed scopes are already ended; no overwriting clear.


class StatusReader:
    """One bounded event-driven reader per existing FSS role process.

    Failure-time callers use snapshot() only. It never reads the socket, joins a
    thread, probes a process, or waits for a lock. No child-supplied strings exist.
    """
    def __init__(self, role, stop, failure):
        if role not in ROLES:
            raise ValueError("invalid local source status role")
        self.role, self.stop, self.failure = role, stop, failure
        self.channel, self.child = socket.socketpair()
        try:
            for channel in (self.channel, self.child):
                for option in (socket.SO_SNDBUF, socket.SO_RCVBUF):
                    channel.setsockopt(socket.SOL_SOCKET, option, 4096)
                    if channel.getsockopt(socket.SOL_SOCKET, option) > _SOCKET_BUFFER_MAX_BYTES:
                        raise OSError("local source status buffer exceeds bound")
        except OSError:
            self.channel.close()
            self.child.close()
            raise
        self.lock = threading.Lock()
        self.latest = None
        self.generation = self.sequence = self.count = 0
        self.ended = True
        self.thread = None
        self.closed = False

    def start(self):
        if threading.stack_size() > _THREAD_STACK_BYTES:
            raise RuntimeError("local source status thread exceeds priced stack")
        self.thread = threading.Thread(target=self._run, name="pllm-fss-status", daemon=True)
        self.thread.start()

    def spawned(self):
        self.child.close()  # Only the exec'd child retains the write endpoint.

    def snapshot(self):
        if self.closed:
            return None
        if not self.lock.acquire(blocking=False):
            return None
        try:
            return None if self.closed or self.latest is None else dict(self.latest)
        finally:
            self.lock.release()

    def _clear(self):
        with self.lock:
            self.latest = None

    def accept(self, packet):
        """Reader-thread-only fixed-record parser; false disables this channel."""
        if self.closed:
            return False
        self.count += 1
        if len(packet) != _RECORD.size or self.count > _MAX_RECORDS:
            return False
        magic, generation, sequence, kind, phase = _RECORD.unpack(packet)
        if (magic != _MAGIC or not 1 <= generation <= _SCOPES or sequence >= _STEPS
                or kind not in (_UPDATE, _END, _FAIL) or phase >= len(SOURCE_IMPORT_PHASES)):
            return False
        if generation < self.generation:
            return True  # Delayed old scope cannot revive an import.
        if generation == self.generation:
            if self.ended or sequence <= self.sequence:
                return True
            if sequence != self.sequence + 1:
                return False
        elif generation != self.generation + 1 or sequence != 0 or kind != _UPDATE:
            return False
        self.generation, self.sequence = generation, sequence
        self.ended = kind != _UPDATE
        if kind == _FAIL:
            stop = self.stop
            if stop is not None:
                try:
                    stop()  # Publish supervisor stop BEFORE the diagnostic lock.
                except Exception:
                    pass  # A callback error cannot discard the frozen phase.
        with self.lock:
            if self.closed:
                return False
            self.latest = {"active": kind != _END, "phase": SOURCE_IMPORT_PHASES[phase],
                           "role": self.role}
            if kind == _FAIL:
                self.latest["failed"] = True
        if kind == _FAIL:
            try:
                failure = self.failure
                if not self.closed and failure is not None:
                    try:
                        failure()  # First-failure collector reads the frozen cache.
                    except Exception:
                        pass  # Optional observations cannot break the reader.
            finally:
                self._clear()
        return True

    def _run(self):
        pending = b""
        try:
            while not self.closed:
                block = self.channel.recv(_RECORD.size - len(pending))
                if not block:
                    break
                pending += block
                if len(pending) == _RECORD.size:
                    if not self.accept(pending):
                        break
                    pending = b""
        except (OSError, ValueError, struct.error):
            pass  # Diagnostic transport cannot change native/HTTP failure paths.
        finally:
            self.closed = True
            self.stop = self.failure = None
            self._clear()
            self.channel.close()

    def close(self):
        self.closed = True
        self.stop = self.failure = None
        self.child.close()
        try:
            self.channel.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if (self.thread is not None and self.thread.ident is not None
                and self.thread is not threading.current_thread()):
            self.thread.join(timeout=.5)
        self.channel.close()
        if self.lock.acquire(blocking=False):
            try:
                self.latest = None
            finally:
                self.lock.release()
