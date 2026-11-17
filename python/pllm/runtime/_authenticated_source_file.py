"""Dormant POSIX reader for one already selected, SDK-admission-relative file.

The enclosing trusted code must select a local regular file under trusted
ancestors, supply its resolved path and admitted row, and hold one owner slot
through confirmed retirement. This module does not issue or approve rows,
resolve HF aliases, establish a secure directory walk, or reserve resources.
The envelope bounds logical payload and attempted reads, not RSS, kernel work,
or caller-held results. Metadata, one descriptor slot, synchronization, hash
contexts and other bounded control objects require separate provision.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass

_LEAF_BYTES = 64 << 10
_HASH_BYTES = 32
_MAX_POSITION = min(sys.maxsize - 1, (1 << 63) - 2)
_Cancel = Callable[[], bool]


class _SourceFileError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _SelectedFile:
    path: str
    basename: str
    size: int
    # dev, ino, mode, size, mtime_ns, ctime_ns of the admitted regular file.
    identity: tuple[int, int, int, int, int, int]
    sha256: bytes


@dataclass(frozen=True, slots=True)
class _FileBounds:
    file_bytes: int
    table_bytes: int
    return_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class _ReadAllowance:
    calls: int
    leaf_bytes: int


@dataclass(frozen=True, slots=True)
class _Requirements:
    leaves: int
    retained_bytes: int
    seal_payload_bytes: int
    read_payload_bytes: int
    seal_requested_bytes: int


@dataclass(frozen=True, slots=True)
class _Failure:
    phase: str
    kind: str
    detail: str | int | None


# Pre-existing bounded records: failure transitions never construct diagnostics.
_SEAL_FAILED = _Failure("seal", "unreported", None)
_READ_FAILED = _Failure("read", "unreported", None)
_CLOSE_FAILED = _Failure("close", "unreported", None)


def _check_uint(value: int, limit: int, name: str) -> None:
    if type(value) is not int or not 0 <= value <= limit:
        raise ValueError(f"invalid {name}")


def _requirements(size: int, return_bytes: int) -> _Requirements:
    """Size only; no stat, source reads, allocation of payload, or attestation."""
    _check_uint(size, _MAX_POSITION, "file size")
    _check_uint(return_bytes, _LEAF_BYTES, "per-call return limit")
    leaves, tail = divmod(size, _LEAF_BYTES)
    leaves += bool(tail)
    table = leaves * _HASH_BYTES
    block = min(size, _LEAF_BYTES)
    result = min(size, return_bytes)
    # Builder, immutable table and single pread buffer (or 1-byte EOF probe).
    seal = max(2 * table, table + max(1, block))
    # Table, one full leaf and assembly, or table plus assembly/result overlap.
    read = table + max(block + result if result else 0, 2 * result)
    _check_uint(max(seal, read), sys.maxsize, "payload geometry")
    return _Requirements(leaves, table, seal, read, size + 1)


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _validate_row(row: _SelectedFile) -> None:
    if type(row) is not _SelectedFile:
        raise ValueError("expected one trusted selected file row")
    _check_uint(row.size, _MAX_POSITION, "file size")
    path, name, identity = row.path, row.basename, row.identity
    if (type(path) is not str or not 1 < len(path) <= 4096 or not path.startswith("/")
            or "\0" in path or any(part in {"", ".", ".."} for part in path[1:].split("/"))
            or type(name) is not str or "\\" in name or path.rsplit("/", 1)[1] != name):
        raise ValueError("expected resolved absolute path and exact admitted basename")
    if (type(identity) is not tuple or len(identity) != 6
            or any(type(value) is not int for value in identity)
            or not 0 <= identity[2] <= 0xFFFF or not stat.S_ISREG(identity[2])
            or identity[3] != row.size):
        raise ValueError("expected exact admitted regular file identity")
    if type(row.sha256) is not bytes or len(row.sha256) != _HASH_BYTES:
        raise ValueError("expected an admitted 32-byte SHA-256")


def _check_cancel(cancelled: _Cancel | None) -> None:
    if cancelled is not None and cancelled() is not False:
        raise _SourceFileError("source operation cancelled")


def _failure(phase: str, exc: BaseException) -> _Failure:
    # Never retain a traceback, custom exception class, or unbounded attribute.
    # No attribute access on arbitrary exception types.
    kind = type(exc)
    detail = None
    if kind is _SourceFileError:
        if exc.args and type(exc.args[0]) is str:
            detail = exc.args[0][:160]
        return _Failure(phase, "source", detail)
    if kind is OSError or kind is InterruptedError:
        number = exc.errno
        if type(number) is int and -sys.maxsize <= number <= sys.maxsize:
            detail = number
        return _Failure(phase, "interrupted" if kind is InterruptedError else "os", detail)
    return _Failure(phase, "other", None)


class _AuthenticatedSourceFile:
    """One staged slot, one sealing attempt, one active operation at a time.

    Publish this inert owner in the enclosing attempt before calling ``seal``.
    Budget valid read attempts up front (including zero-length reads and failed
    I/O), without refunds; invalid/busy/closed calls start no source work.
    Cancellation is cooperative: trusted hooks return exact False to continue.
    Do not inject asynchronous exceptions into the owner or transfer the fd.

    ``close()`` returns False while settlement is pending, True only on confirmed
    retirement. A synchronous close failure is raised unchanged to its caller;
    deferred close never replaces the operation's exception. Once close fails,
    every later close reports terminal ambiguity without retrying the syscall.
    Keep the slot and its conservative charge in that case; the fd number must
    not be used even if the kernel reused it.
    No implicit cleanup, funding, or refund mechanism is supplied here.
    """

    __slots__ = ("_row", "_needs", "_max_return", "_calls", "_leaf_bytes", "_lock", "_state",
                 "_active", "_fd", "_table", "_primary_failure", "_close_failure")

    def __init__(self, row: _SelectedFile, bounds: _FileBounds, allowance: _ReadAllowance):
        _validate_row(row)
        if type(bounds) is not _FileBounds or type(allowance) is not _ReadAllowance:
            raise ValueError("expected explicit file bounds and finite read allowance")
        _check_uint(bounds.file_bytes, _MAX_POSITION, "file byte limit")
        _check_uint(bounds.table_bytes, sys.maxsize, "table byte limit")
        _check_uint(bounds.payload_bytes, sys.maxsize, "payload byte limit")
        _check_uint(allowance.calls, sys.maxsize, "read call allowance")
        _check_uint(allowance.leaf_bytes, sys.maxsize, "read leaf byte allowance")
        needs = _requirements(row.size, bounds.return_bytes)
        if (row.size > bounds.file_bytes or needs.retained_bytes > bounds.table_bytes
                or max(needs.seal_payload_bytes, needs.read_payload_bytes) > bounds.payload_bytes):
            raise ValueError("source file exceeds staged bounds")
        self._row, self._needs, self._max_return = row, needs, bounds.return_bytes
        self._calls, self._leaf_bytes = allowance.calls, allowance.leaf_bytes
        self._lock = threading.Lock()
        self._state, self._active = "staged", False
        self._fd = self._table = None
        self._primary_failure = self._close_failure = None

    @property
    def primary_failure(self) -> _Failure | None:
        with self._lock:
            return self._primary_failure

    @property
    def close_failure(self) -> _Failure | None:
        with self._lock:
            return self._close_failure

    def _checkpoint(self, cancelled: _Cancel | None) -> None:
        with self._lock:
            if self._state == "closing":
                raise _SourceFileError("source owner is closing")
        _check_cancel(cancelled)
        with self._lock:
            if self._state == "closing":
                raise _SourceFileError("source owner is closing")

    def _check_admitted(self) -> None:
        if (_identity(os.fstat(self._fd)) != self._row.identity
                or _identity(os.lstat(self._row.path)) != self._row.identity):
            raise _SourceFileError("source identity or size differs from admission")

    def _check_pinned(self) -> None:
        info = os.fstat(self._fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_size != self._row.size
                or (info.st_dev, info.st_ino) != self._row.identity[:2]):
            raise _SourceFileError("pinned source identity or size changed")
        # Timestamps are not authenticity; same-size in-place edits must reach
        # the digest comparison. Never consult/reopen the pathname on read.

    def _read_leaf(self, offset: int, length: int) -> bytes:
        block = os.pread(self._fd, length, offset)
        if type(block) is not bytes or len(block) != length:
            block = None
            raise _SourceFileError("source leaf short or invalid read")
        return block

    def _build_table(self, cancelled: _Cancel | None) -> None:
        path = self._row.path
        if (os.name != "posix" or sys.platform not in ("linux", "darwin")
                or not callable(getattr(os, "pread", None))
                or not all(hasattr(os, name) for name in
                           ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"))):
            raise _SourceFileError("unsupported positioned source access")
        self._checkpoint(cancelled)
        # Refuse even stable aliases; this is a check, not an alternate path to
        # open, nor a proof against hostile changes to ancestor directories.
        if os.path.realpath(path, strict=True) != path:
            raise _SourceFileError("source path is not resolved")
        if _identity(os.lstat(path)) != self._row.identity:
            raise _SourceFileError("source identity or size differs from admission")
        self._fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        self._check_admitted()
        self._checkpoint(cancelled)
        whole = hashlib.sha256()
        packed = bytearray(self._needs.retained_bytes)
        block = None
        try:
            for ordinal in range(self._needs.leaves):
                self._checkpoint(cancelled)
                offset = ordinal * _LEAF_BYTES
                block = self._read_leaf(offset, min(_LEAF_BYTES, self._row.size - offset))
                whole.update(block)
                slot = ordinal * _HASH_BYTES
                packed[slot:slot + _HASH_BYTES] = hashlib.sha256(block).digest()
                block = None
            self._checkpoint(cancelled)
            eof = os.pread(self._fd, 1, self._row.size)
            if type(eof) is not bytes or eof:
                eof = None
                raise _SourceFileError("source has data past admitted EOF")
            if whole.digest() != self._row.sha256:
                raise _SourceFileError("source whole SHA-256 mismatch")
            self._table = bytes(packed)
        finally:
            packed = block = None

    def seal(self, *, cancelled: _Cancel | None = None) -> None:
        with self._lock:
            if self._state != "staged" or self._active:
                raise _SourceFileError("source owner cannot start sealing")
            self._active = True
        try:
            self._build_table(cancelled)
            self._checkpoint(cancelled)
            self._check_admitted()
            self._handoff()
        except BaseException as exc:
            self._poison(_SEAL_FAILED, exc)
            raise
        finally:
            self._finish()

    def _assemble(self, offset: int, count: int, first: int, end: int,
                  cancelled: _Cancel | None) -> bytes:
        self._checkpoint(cancelled)
        self._check_pinned()
        assembly = bytearray(count)
        block = None
        try:
            for ordinal in range(first, end):
                self._checkpoint(cancelled)
                start = ordinal * _LEAF_BYTES
                block = self._read_leaf(start, min(_LEAF_BYTES, self._row.size - start))
                slot = ordinal * _HASH_BYTES
                if hashlib.sha256(block).digest() != self._table[slot:slot + _HASH_BYTES]:
                    raise _SourceFileError("source leaf SHA-256 mismatch")
                lo = max(offset, start)
                hi = min(offset + count, start + len(block))
                # Memoryviews avoid an extra bytes copy or bytearray slice-assignment
                # converting a view into a temporary bytearray of the overlap.
                with memoryview(assembly) as dst, memoryview(block) as src:
                    dst[lo - offset:hi - offset] = src[lo - start:hi - start]
                block = None
            return bytes(assembly)
        finally:
            assembly = block = None

    def read_verified(self, offset: int, count: int, *, cancelled: _Cancel | None = None) -> bytes:
        _check_uint(offset, self._row.size, "read offset")
        _check_uint(count, min(self._row.size - offset, self._max_return), "read count")
        first = offset // _LEAF_BYTES
        end = (offset + count - 1) // _LEAF_BYTES + 1 if count else first
        work = min(self._row.size, end * _LEAF_BYTES) - first * _LEAF_BYTES
        with self._lock:
            if self._state != "ready" or self._active:
                raise _SourceFileError("source owner is not ready for a read")
            if not self._calls or work > self._leaf_bytes:
                raise _SourceFileError("source read allowance exhausted")
            # Compute both debits before committing either counter or the loan.
            calls_left = self._calls - 1
            bytes_left = self._leaf_bytes - work
            self._calls = calls_left
            self._leaf_bytes = bytes_left
            self._active = True
        result = None
        try:
            result = self._assemble(offset, count, first, end, cancelled)
            self._checkpoint(cancelled)
            self._check_pinned()
            self._handoff()
            return result
        except BaseException as exc:
            result = None
            self._poison(_READ_FAILED, exc)
            raise
        finally:
            self._finish()

    def _handoff(self) -> None:
        # Linearization point: pending close wins before this point, not after.
        with self._lock:
            if self._state == "closing":
                raise _SourceFileError("source owner is closing")
            self._state = "ready"

    def _poison(self, fallback: _Failure, exc: BaseException) -> None:
        with self._lock:
            self._primary_failure = fallback
            self._state = "closing"
        try:
            reason = _failure(fallback.phase, exc)
            with self._lock:
                self._primary_failure = reason
        except BaseException:
            # Optional diagnostics cannot replace the original exception.
            pass

    def _claim_retirement(self) -> int | None:
        # Under the state lock, only when no operation can still use the fd.
        fd = self._fd
        self._fd = None
        self._table = None
        self._state = "closed" if fd is None else "retiring"
        return fd

    def _retire(self, fd: int, *, propagate: bool) -> None:
        try:
            os.close(fd)
        except BaseException as exc:
            # A raised close cannot establish whether the descriptor was
            # consumed. Retrying might close an unrelated reused descriptor.
            with self._lock:
                self._close_failure = _CLOSE_FAILED
                self._state = "close-unknown"
            try:
                reason = _failure(_CLOSE_FAILED.phase, exc)
                with self._lock:
                    self._close_failure = reason
            except BaseException:
                pass
            if propagate:
                raise
        else:
            with self._lock:
                self._state = "closed"

    def _finish(self) -> None:
        with self._lock:
            self._active = False
            fd = self._claim_retirement() if self._state == "closing" else None
        if fd is not None:
            self._retire(fd, propagate=False)

    def close(self) -> bool:
        """Request retirement without waiting for a loan or retrying a close."""
        with self._lock:
            if self._state in ("closed", "close-unknown", "retiring"):
                fd = None
            else:
                self._state = "closing"
                if self._active:
                    return False
                fd = self._claim_retirement()
        if fd is not None:
            self._retire(fd, propagate=True)
        with self._lock:
            if self._state == "close-unknown":
                raise _SourceFileError("source descriptor retirement is unconfirmed; do not retry")
            return self._state == "closed"
