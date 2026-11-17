"""Mechanical single-file proof and custody; fixture rows are not SDK authority."""

import errno
import os
import sys
from dataclasses import replace
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from pllm.runtime import _authenticated_source_file as source

pytestmark = pytest.mark.skipif(
    sys.platform not in ("linux", "darwin"), reason="local Linux/macOS positioned reads"
)

C = 65536
DATA = (b"A<12" + b"01xy!@az" * 8191 + b">3a?"
        + b"B[45" + b"89MN#%qr" * 8191 + b"]6b~" + b"t!9z?")
# Fixed with shasum on independently generated fixture bytes, not the reader.
DIGEST = bytes.fromhex("e4df1f66958f65ee8e9a82f2aa59da77d91eeac4b7c39e1475f8076ed2358fce")
EMPTY_DIGEST = bytes.fromhex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
Error = source._SourceFileError
Bounds = source._FileBounds
Allowance = source._ReadAllowance


def _row(path, digest=DIGEST):
    info = path.lstat()
    identity = (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)
    return source._SelectedFile(str(path), path.name, info.st_size, identity, digest)


@pytest.fixture
def selected(tmp_path):
    path = tmp_path.resolve() / "selected.bin"
    path.write_bytes(DATA)
    return path, _row(path)


@pytest.fixture
def stage():
    owners = []

    def stage(row, *, bounds=None, allowance=None):
        if bounds is None:
            bounds = Bounds(len(DATA), 96, C, 3 * C)
        if allowance is None:
            allowance = Allowance(16, 8 * C)
        owner = source._AuthenticatedSourceFile(row, bounds, allowance)
        owners.append(owner)
        return owner

    yield stage
    for owner in owners:
        if owner.close_failure is None:
            assert owner.close()


def _spy_reads(monkeypatch):
    calls = []
    pread = os.pread

    def read(fd, count, offset):
        calls.append((offset, count))
        return pread(fd, count, offset)

    monkeypatch.setattr(source.os, "pread", read)
    return calls


def _patch_os(monkeypatch, **changes):
    # Patch only the primitive's view of os, not unrelated descriptors.
    class OS:
        def __getattr__(self, name):
            return getattr(os, name)

    proxy = OS()
    for name, value in changes.items():
        setattr(proxy, name, value)
    monkeypatch.setattr(source, "os", proxy)


def _patched_close(monkeypatch, *, failure=None):
    calls = []
    close = os.close

    def retire(fd):
        close(fd)
        calls.append(fd)
        if failure is not None:
            raise failure

    _patch_os(monkeypatch, close=retire)
    return calls


def test_literal_reads_two_independent_positions_and_exact_full_leaf_work(selected, stage, monkeypatch):
    _, row = selected
    calls = _spy_reads(monkeypatch)
    owner = stage(row, allowance=Allowance(5, 4 * C + 5))
    with pytest.raises(Error, match="not ready"):
        owner.read_verified(0, 1)
    owner.seal()
    assert calls == [(0, C), (C, C), (2 * C, 5), (len(DATA), 1)]
    calls.clear()

    # Two independently positioned consumers, no shared file position.
    a, b = C - 2, 2 * C + 1
    assert owner.read_verified(a, 5) == b"a?B[4"
    assert owner.read_verified(b, 3) == b"!9z"
    first = owner.read_verified(4 + 8 * 371 + 2, 4)
    assert type(first) is bytes and first == b"xy!@"
    assert owner.read_verified(C + 4 + 8 * 113 + 3, 3) == b"N#%"
    assert owner.read_verified(len(DATA), 0) == b""
    assert calls == [(0, C), (C, C), (2 * C, 5), (0, C), (C, C)]
    with pytest.raises(Error, match="allowance"):
        owner.read_verified(0, 0)
    assert owner.close()
    with pytest.raises(Error, match="not ready"):
        owner.read_verified(0, 0)


def test_empty_seal_only_probes_eof(tmp_path, stage, monkeypatch):
    path = tmp_path.resolve() / "empty"
    path.touch()
    calls = _spy_reads(monkeypatch)
    owner = stage(_row(path, EMPTY_DIGEST), bounds=Bounds(0, 0, 0, 1), allowance=Allowance(1, 0))
    owner.seal()
    assert owner.read_verified(0, 0) == b""
    assert calls == [(0, 1)]


def test_crossing_into_short_tail_charges_exact_work(selected, stage, monkeypatch):
    _, row = selected
    owner = stage(row, allowance=Allowance(2, C + 5))
    owner.seal()
    calls = _spy_reads(monkeypatch)
    assert owner.read_verified(2 * C - 1, 3) == b"~t!"
    with pytest.raises(Error, match="allowance"):
        owner.read_verified(2 * C, 1)
    assert calls == [(C, C), (2 * C, 5)]


def test_pure_sizing_and_resource_refusals_before_fs_access(selected, stage, monkeypatch):
    _, row = selected

    class NoFilesystem:
        def __getattr__(self, name):
            pytest.fail(f"sizing touched os.{name}")

    monkeypatch.setattr(source, "os", NoFilesystem())
    need = source._requirements(row.size, C)
    assert need == source._Requirements(3, 96, C + 96, 2 * C + 96, row.size + 1)
    table_overlap = source._requirements(2049 * C, 0)
    assert table_overlap.seal_payload_bytes == 2 * table_overlap.retained_bytes
    for bounds in (Bounds(row.size - 1, 96, C, 3 * C), Bounds(row.size, 95, C, 3 * C),
                   Bounds(row.size, 96, C, 2 * C + 95), Bounds(row.size, 96, C + 1, 3 * C),
                   Bounds(row.size, True, C, 3 * C), Bounds(row.size, 96, True, 3 * C)):
        with pytest.raises(ValueError):
            stage(row, bounds=bounds)
    for allowance in (Allowance(True, C), Allowance(1, False), Allowance(1, sys.maxsize + 1)):
        with pytest.raises(ValueError):
            stage(row, allowance=allowance)
    with pytest.raises(ValueError):
        stage(replace(row, size=True))
    with pytest.raises(ValueError):
        stage(replace(row, identity=(True, *row.identity[1:])))
    with pytest.raises(ValueError):
        stage(replace(row, basename="other"))
    with pytest.raises(ValueError):
        stage(replace(row, sha256=bytearray(DIGEST)))
    with pytest.raises(TypeError, match="leaf_hashes"):
        source._AuthenticatedSourceFile(row, Bounds(row.size, 96, C, 3 * C), Allowance(0, 0),
                                        leaf_hashes=b"\0" * 96)
    assert stage(row).close()


def test_bad_ranges_and_work_refused_without_read_or_callback(selected, stage, monkeypatch):
    _, row = selected
    calls = _spy_reads(monkeypatch)
    owner = stage(row, allowance=Allowance(4, C + 5))
    owner.seal()
    calls.clear()

    def forbidden():
        pytest.fail("callback on a refused request")

    for offset, count in ((-1, 1), (0, -1), (True, 1), (0, False), (len(DATA) + 1, 0),
                          (len(DATA), 1), (0, C + 1), (0, sys.maxsize + 1)):
        with pytest.raises(ValueError):
            owner.read_verified(offset, count, cancelled=forbidden)
    with pytest.raises(Error, match="allowance"):
        owner.read_verified(C - 1, 2, cancelled=forbidden)  # 2 full leaves, not 2 bytes.
    assert calls == []
    assert owner.read_verified(0, 1) == b"A"
    with pytest.raises(Error, match="allowance"):
        owner.read_verified(0, 1, cancelled=forbidden)  # Revisit charges the full leaf again.
    assert owner.read_verified(2 * C + 4, 1) == b"?"  # Charge exact short tail.
    assert owner.read_verified(0, 0) == b""
    assert owner.read_verified(2 * C, 0) == b""
    with pytest.raises(Error, match="allowance"):
        owner.read_verified(0, 0, cancelled=forbidden)
    assert calls == [(0, C), (2 * C, 5)]


@pytest.mark.parametrize("fault", ["digest", "size", "replacement", "symlink", "ancestor_alias"])
def test_wrong_admitted_row_or_unsupported_path_never_ready(selected, stage, fault):
    path, row = selected
    message = "identity or size"
    if fault == "digest":
        row = replace(row, sha256=b"\0" * 32)
        message = "whole SHA-256"
    elif fault == "size":
        row = replace(row, size=row.size - 1, identity=(*row.identity[:3], row.size - 1, *row.identity[4:]))
    elif fault == "replacement":
        replacement = path.with_name("substitute")
        replacement.write_bytes(b"Z" * len(DATA))
        replacement.replace(path)
    elif fault == "symlink":
        original = path.rename(path.with_name("original"))
        path.symlink_to(original)
        message = "not resolved"
    else:
        alias = path.parent / "alias"
        alias.symlink_to(path.parent, target_is_directory=True)
        row = replace(row, path=str(alias / path.name))
        message = "not resolved"
    owner = stage(row)
    with pytest.raises(Error, match=message):
        owner.seal()
    assert owner.primary_failure.phase == "seal"
    with pytest.raises(Error, match="not ready"):
        owner.read_verified(0, 0)
    with pytest.raises(Error, match="cannot start sealing"):
        owner.seal()
    assert owner.close()


def test_unsupported_positioned_access_never_opens(selected, stage, monkeypatch):
    _, row = selected
    owner = stage(row)
    monkeypatch.delattr(source.os, "pread")
    with pytest.raises(Error, match="unsupported positioned"):
        owner.seal()
    assert owner.close()


def test_cancellation_before_acquisition_never_opens(selected, stage, monkeypatch):
    _, row = selected
    owner = stage(row)

    def forbidden(*args, **kwargs):
        pytest.fail("cancelled seal opened a descriptor")

    with monkeypatch.context() as patch:
        patch.setattr(source.os, "open", forbidden)
        with pytest.raises(Error, match="cancelled"):
            owner.seal(cancelled=lambda: True)
    assert owner.close()


@pytest.mark.parametrize("fault", ["post_open_stat", "short", "growth", "mid_table_cancel", "ready_cancel"])
def test_failed_seal_retires_fd_and_never_discloses(selected, stage, monkeypatch, fault):
    path, row = selected
    closed = _patched_close(monkeypatch)
    pread, fstat = os.pread, os.fstat
    reads = []

    def faulty_read(fd, count, offset):
        reads.append((offset, count))
        if fault == "growth" and offset == row.size:
            with path.open("ab") as writer:
                writer.write(b"!")
        return pread(fd, count - (fault == "short"), offset)

    def faulty_stat(fd):
        if fault == "post_open_stat":
            raise OSError(errno.EIO, "post-open fstat")
        return fstat(fd)

    monkeypatch.setattr(source.os, "pread", faulty_read)
    monkeypatch.setattr(source.os, "fstat", faulty_stat)
    owner = stage(row)

    def cancelled():
        with pytest.raises(Error, match="not ready"):
            owner.read_verified(0, 0)
        return ((fault == "mid_table_cancel" and len(reads) == 1)
                or (fault == "ready_cancel" and len(reads) == 4))

    with pytest.raises((Error, OSError), match={
        "post_open_stat": "post-open fstat", "short": "short", "growth": "past admitted EOF",
        "mid_table_cancel": "cancelled", "ready_cancel": "cancelled",
    }[fault]):
        owner.seal(cancelled=cancelled)
    assert len(closed) == 1
    assert owner.primary_failure.phase == "seal"
    assert owner.close() and len(closed) == 1
    with pytest.raises(Error, match="not ready"):
        owner.read_verified(0, 1)


@pytest.mark.parametrize("target", ["admitted_fd", "admitted_path", "pinned_fd"])
def test_final_identity_check_refuses_after_complete_proof(selected, stage, monkeypatch, target):
    _, row = selected
    owner = stage(row)
    if target == "pinned_fd":
        owner.seal()
    probe_count = 2 if target == "pinned_fd" else 4
    pread, fstat, lstat = os.pread, os.fstat, os.lstat
    reads, mismatches = [], []

    def read(fd, count, offset):
        reads.append((offset, count))
        return pread(fd, count, offset)

    def altered(info):
        mismatches.append(len(reads))
        return SimpleNamespace(st_dev=info.st_dev, st_ino=info.st_ino + 1, st_mode=info.st_mode,
                               st_size=info.st_size, st_mtime_ns=info.st_mtime_ns,
                               st_ctime_ns=info.st_ctime_ns)

    def fd_stat(fd):
        info = fstat(fd)
        return altered(info) if len(reads) == probe_count and target != "admitted_path" else info

    def path_stat(path):
        info = lstat(path)
        return altered(info) if len(reads) == probe_count and target == "admitted_path" else info

    _patch_os(monkeypatch, pread=read, fstat=fd_stat, lstat=path_stat)
    observed = []
    if target == "pinned_fd":
        with pytest.raises(Error, match="pinned source identity"):
            observed.append(owner.read_verified(C - 1, 2))
        assert reads == [(0, C), (C, C)]
    else:
        with pytest.raises(Error, match="identity or size differs from admission"):
            owner.seal()
        assert reads == [(0, C), (C, C), (2 * C, 5), (row.size, 1)]
    assert not observed and mismatches == [probe_count]
    with pytest.raises(Error, match="not ready"):
        owner.read_verified(0, 0)
    assert owner.close()


@pytest.mark.parametrize("phase", ["seal", "read"])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_report_fault_preserves_original_post_open_failure(
        selected, stage, monkeypatch, phase, ambiguous):
    _, row = selected
    owner = stage(row)
    if phase == "read":
        owner.seal()
    close_error = InterruptedError(errno.EINTR, "ambiguous deferred close") if ambiguous else None
    closed = _patched_close(monkeypatch, failure=close_error)
    pread = os.pread
    original = OSError(errno.EIO, "original positioned read failure")
    reports = []

    def fault(fd, count, offset):
        if offset == C:
            raise original
        return pread(fd, count, offset)

    def report(which, exc):
        reports.append((which, exc, owner.primary_failure, owner.close_failure))
        raise MemoryError("simulated report allocation failure")

    monkeypatch.setattr(source.os, "pread", fault)
    monkeypatch.setattr(source, "_failure", report)
    with pytest.raises(OSError) as caught:
        if phase == "seal":
            owner.seal()
        else:
            owner.read_verified(C - 1, 2)
    fallback = source._Failure(phase, "unreported", None)
    assert caught.value is original and reports[0] == (phase, original, fallback, None)
    assert owner.primary_failure == fallback
    with pytest.raises(Error, match="not ready"):
        owner.read_verified(0, 0)
    if ambiguous:
        assert reports[1:] == [("close", close_error, fallback, source._CLOSE_FAILED)]
        assert owner.close_failure == source._CLOSE_FAILED
        with pytest.raises(Error, match="unconfirmed; do not retry"):
            owner.close()
    else:
        assert len(reports) == 1 and owner.close()
    assert len(closed) == 1


def test_same_inode_write_outside_slice_hits_digest(selected, stage):
    path, row = selected
    owner = stage(row)
    owner.seal()
    with path.open("r+b") as writer:
        writer.seek(C - 1)
        writer.write(b"!")
    os.utime(path, ns=(path.stat().st_atime_ns, row.identity[4]))
    assert (path.stat().st_dev, path.stat().st_ino) == row.identity[:2]
    with pytest.raises(Error, match="leaf SHA-256"):
        owner.read_verified(0, 1)
    with pytest.raises(Error, match="not ready"):
        owner.read_verified(C, 1)
    assert owner.close()


@pytest.mark.parametrize("fault", ["swapped", "last_leaf", "short", "cancel_between", "cancel_at_handoff"])
def test_crossing_read_is_all_or_nothing(selected, stage, monkeypatch, fault):
    _, row = selected
    owner = stage(row, allowance=Allowance(3, 3 * C))
    owner.seal()
    calls = []
    pread = os.pread

    def faulty(fd, count, offset):
        calls.append((offset, count))
        if fault == "swapped":
            return pread(fd, count, C)
        if offset == C and fault == "last_leaf":
            return pread(fd, count, offset)[:-1] + b"!"  # Only first byte requested.
        if offset == C and fault == "short":
            return pread(fd, count - 1, offset)
        return pread(fd, count, offset)

    monkeypatch.setattr(source.os, "pread", faulty)
    observed = []

    def cancelled():
        return ((fault == "cancel_between" and len(calls) == 1)
                or (fault == "cancel_at_handoff" and len(calls) == 2))

    with pytest.raises(Error, match={
        "swapped": "leaf SHA-256", "last_leaf": "leaf SHA-256", "short": "short",
        "cancel_between": "cancelled", "cancel_at_handoff": "cancelled",
    }[fault]):
        observed.append(owner.read_verified(C - 1, 2, cancelled=cancelled))
    assert observed == []
    assert calls == ([(0, C)] if fault in {"swapped", "cancel_between"} else [(0, C), (C, C)])
    assert owner.primary_failure.phase == "read"
    assert (owner._calls, owner._leaf_bytes) == (2, C)  # No refund for an interrupted attempt.
    with pytest.raises(Error, match="not ready"):
        owner.read_verified(0, 0)
    assert owner.close()


def test_replacement_after_seal_never_reopens_path(selected, stage, monkeypatch):
    path, row = selected
    owner = stage(row)
    owner.seal()
    replacement = path.with_name("replacement")
    replacement.write_bytes(b"Z" * len(DATA))
    replacement.replace(path)

    def forbidden(*args, **kwargs):
        pytest.fail("post-seal pathname access")

    with monkeypatch.context() as patch:
        patch.setattr(source.os, "open", forbidden)
        patch.setattr(source.os, "lstat", forbidden)
        assert owner.read_verified(C - 1, 2) == b"?B"


def test_edit_after_proof_returns_verified_copy_not_second_read(selected, stage, monkeypatch):
    path, row = selected
    owner = stage(row)
    owner.seal()
    calls = _spy_reads(monkeypatch)
    changed = False

    def change_at_handoff():
        nonlocal changed
        if len(calls) == 2 and not changed:
            with path.open("r+b") as writer:
                writer.seek(C - 1)
                writer.write(b"!!")
            changed = True
        return False

    assert owner.read_verified(C - 1, 2, cancelled=change_at_handoff) == b"?B"
    assert changed and calls == [(0, C), (C, C)]
    with pytest.raises(Error, match="leaf SHA-256"):
        owner.read_verified(C - 1, 2)


@pytest.mark.parametrize("operation", ["seal", "read"])
def test_close_during_active_loan_defers_retirement(selected, stage, monkeypatch, operation):
    _, row = selected
    entered, release, done = Event(), Event(), Queue()
    closed = _patched_close(monkeypatch)
    reads = _spy_reads(monkeypatch)
    owner = stage(row)
    if operation == "read":
        owner.seal()
        reads.clear()

    def cancel():
        if len(reads) == 1:
            entered.set()
            assert release.wait(5), "active loan never released"
        return False

    def active():
        try:
            if operation == "seal":
                result = owner.seal(cancelled=cancel)
            else:
                result = owner.read_verified(C - 1, 2, cancelled=cancel)
        except BaseException as exc:
            result = exc
        done.put(result)

    thread = Thread(target=active, daemon=True)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(Error, match="not ready"):
            owner.read_verified(0, 0)
        with pytest.raises(Error, match="cannot start sealing"):
            owner.seal()
        assert owner.close() is False
        assert closed == []
        with pytest.raises(Error, match="not ready"):
            owner.read_verified(0, 0)
    finally:
        release.set()
        thread.join(5)
    exc = done.get(timeout=5)
    assert type(exc) is Error and "closing" in str(exc)
    assert owner.close() and len(closed) == 1
    assert len(reads) == 1


def test_hook_can_reenter_close_and_primary_survives_terminal_close_fault(selected, stage, monkeypatch):
    _, row = selected
    closed = _patched_close(monkeypatch, failure=InterruptedError(errno.EINTR, "ambiguous close"))
    owner = stage(row)
    owner.seal()
    failure = LookupError("cancel hook fault")

    def fault():
        assert owner.close() is False  # Reentrant: the hook must not hold the state lock.
        assert closed == []
        raise failure

    done = Queue()

    def run():
        try:
            done.put(owner.read_verified(0, 1, cancelled=fault))
        except BaseException as exc:
            done.put(exc)

    thread = Thread(target=run, daemon=True)
    thread.start()
    try:
        assert done.get(timeout=5) is failure
        assert owner.primary_failure == source._Failure("read", "other", None)
        assert owner.close_failure == source._Failure("close", "interrupted", errno.EINTR)
        for _ in range(2):
            with pytest.raises(Error, match="unconfirmed; do not retry"):
                owner.close()
        assert len(closed) == 1
        with pytest.raises(Error, match="not ready"):
            owner.read_verified(0, 0)
    finally:
        thread.join(1)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_close_pending_in_syscall_then_ambiguity_committed_before_reporting(
        selected, stage, monkeypatch, ambiguous):
    _, row = selected
    owner = stage(row)
    owner.seal()
    closing, release_close, reporting, release_report = Event(), Event(), Event(), Event()
    done, calls = Queue(), []
    real_close = os.close
    original = InterruptedError(errno.EINTR, "ambiguous close after descriptor consumption")

    def close(fd):
        calls.append(fd)
        closing.set()
        released = release_close.wait(5)
        real_close(fd)
        assert released, "close syscall never released"
        if ambiguous:
            raise original

    def report(phase, exc):
        if phase == "close" and exc is original:
            reporting.set()
            release_report.wait(5)
        raise MemoryError("simulated close report allocation failure")

    def run_close():
        try:
            done.put(owner.close())
        except BaseException as exc:
            done.put(exc)

    _patch_os(monkeypatch, close=close)
    monkeypatch.setattr(source, "_failure", report)
    thread = Thread(target=run_close, daemon=True)
    thread.start()
    try:
        assert closing.wait(5)
        assert owner.close() is False
        assert owner.close_failure is None
        with pytest.raises(Error, match="not ready"):
            owner.read_verified(0, 0)
        release_close.set()
        if ambiguous:
            assert reporting.wait(5)
            assert owner.close_failure == source._CLOSE_FAILED
            with pytest.raises(Error, match="unconfirmed; do not retry"):
                owner.close()
            with pytest.raises(Error, match="not ready"):
                owner.read_verified(0, 0)
            assert len(calls) == 1
            release_report.set()
            assert done.get(timeout=5) is original
            assert owner.close_failure == source._CLOSE_FAILED
            with pytest.raises(Error, match="unconfirmed; do not retry"):
                owner.close()
        else:
            assert done.get(timeout=5) is True and owner.close()
        assert len(calls) == 1
    finally:
        release_close.set()
        release_report.set()
        thread.join(5)
