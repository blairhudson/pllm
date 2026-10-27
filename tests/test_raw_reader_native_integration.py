"""Isolated synthetic D55 seam; not checkpoint trust or a Python peak proof.

Use distinct installed feature/production wheels. Genuine-reader tests use no
fake reader counter, native owner, DiskLease, retirement method or cleanup result.
Python CM instrumentation observes then executes the REAL reserved CM exit.
Authored helper-failure tests prove exception/cleanup choreography only, not
native refusal, reader capacity, or original generation-ledger identity.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pllm import _native
from pllm.deployment import resources
from pllm.runtime import replicated_source


HOOKS = (
    "_test_raw_prepare_fixture", "_test_hold_reader", "_test_release_reader",
    "_test_raw_snapshot", "_test_raw_inventory", "_test_raw_chunk",
)
FEATURE = hasattr(_native.ReplicatedPublisher, HOOKS[0])
feature_only = pytest.mark.skipif(not FEATURE, reason="requires isolated lifecycle-test-support wheel")


def _assert_installed_origins():
    dist = importlib.metadata.distribution("pllm.run")
    recorded = {Path(dist.locate_file(file)).resolve() for file in dist.files or ()}
    for module in (_native, resources, replicated_source):
        origin = Path(module.__file__).resolve()
        assert origin.is_relative_to(Path(sys.prefix).resolve()), origin
        assert origin in recorded, origin


def _snapshot(native):
    # stage, original used, collection charge, reader charge, backing count,
    # reader live, reserved, attempted, retained root, descriptor count.
    result = tuple(native._test_raw_snapshot())
    assert len(result) == 10 and all(type(x) is int and x >= 0 for x in result)
    return result


def _directories(parent):
    # External test discovery only. No new path/label authority enters native.
    entries = list(itertools.islice(parent.glob("pllm-source-*"), 4097))
    assert len(entries) <= 4096
    return set(entries)


def _prepare():
    _assert_installed_origins()
    assert all(hasattr(_native.ReplicatedPublisher, name) for name in HOOKS)
    scope = resources.reserve_cleanup_slot()
    holder = replicated_source._RawPublisherOwner(scope)
    try:
        try:
            native = _native.ReplicatedPublisher._test_raw_prepare_fixture()
            holder._retain(native)
            parent = Path(native._raw_disk_directory)
            before = _directories(parent)
            # Record independently frozen NATIVE expected inventory before ingestion.
            inventory = tuple(native._test_raw_inventory(i) for i in range(3))
            assert all(0 < size <= 1 << 20 and len(digest) == 32 and 0 < count <= 64
                       for size, digest, count in inventory)
            assert len({count for _, _, count in inventory}) == 1
            # Ordinary future pytest -s output records exact frozen native values, not a
            # receipt archive, fixture execution now, or a caller-supplied expectation.
            print("D55 native inventory", [(size, digest.hex(), count) for size, digest, count in inventory], file=sys.stderr)
            assert native._raw_disk_bytes == sum(size for size, _, _ in inventory)
            holder._admit_disk_inflight()
        finally:
            # Only this initial preparation gate belongs to this helper. append,
            # finish and retirement release their own gates; never release those
            # again from the outer error path (which could race a newer request).
            holder._end_operation()
        created = _directories(parent) - before
        assert len(created) == 1
        directory = created.pop()
        assert type(holder._lease) is resources.DiskLease
        assert scope.owns(holder)
        assert holder._lease._raw_holder is holder
        assert holder._lease._backing is None  # full reservation, never invented credit
        for file, (size, expected, _) in enumerate(inventory):
            digest = hashlib.sha256()
            for offset in range(0, size, 17):
                chunk = native._test_raw_chunk(file, offset)
                assert type(chunk) is bytes and len(chunk) == min(17, size - offset)
                digest.update(chunk)
                holder.append(file, offset, chunk)
            assert digest.digest() == expected  # comparison only; never Python repinning
        holder.finish()
        assert tuple(native._test_raw_inventory(i) for i in range(3)) == inventory
        sealed = _snapshot(native)
        assert sealed[0] == 2 and sealed[4] == 3 and not sealed[5]
        return holder, native, directory
    except BaseException as error:
        # Same captured holder/slot, one ordinary cleanup attempt. Production
        # preserves error identity and roots any sticky retirement failure.
        holder._close_preserving(error)
        raise


def _capture_prepare_holders(monkeypatch):
    """Observe actual production holders; no substitute custody or ledger."""
    holders = []
    real_init = replicated_source._RawPublisherOwner.__init__

    def observed_init(holder, scope):
        real_init(holder, scope)
        holders.append(holder)

    monkeypatch.setattr(replicated_source._RawPublisherOwner, "__init__", observed_init)
    return holders


@feature_only
def test_prepare_authored_factory_refusal_releases_empty_holder(monkeypatch):
    """Helper choreography only: authored refusal, NOT native capacity evidence."""
    holders = _capture_prepare_holders(monkeypatch)
    failure = RuntimeError("authored pre-fixture refusal")
    calls = []

    class RefusingFactory:
        @staticmethod
        def _test_raw_prepare_fixture():
            calls.append(None)
            assert len(holders) == 1
            holder = holders[0]
            assert holder._busy and holder._cleanup_scope.owns(holder)
            assert holder._owner is None
            raise failure

    # Only the factory is substituted. No fake native result reaches _retain.
    for name in HOOKS[1:]:
        monkeypatch.setattr(RefusingFactory, name, None, raising=False)
    monkeypatch.setattr(_native, "ReplicatedPublisher", RefusingFactory)
    with pytest.raises(RuntimeError) as failed:
        _prepare()
    assert failed.value is failure and calls == [None]
    assert len(holders) == 1
    holder = holders[0]
    assert holder._closed and not holder._busy
    assert holder._cleanup_scope.closed and not holder._cleanup_scope.owns(holder)
    assert holder._cleanup_failure is None and holder._owner is None


@feature_only
@pytest.mark.parametrize("stage", ["discovery", "inventory_output", "append", "hash", "finish"])
def test_prepare_authored_boundary_failure_cleans_same_holder(monkeypatch, stage):
    """Authored Python boundary failures, not native append/finish failure proof."""
    holders = _capture_prepare_holders(monkeypatch)
    failure = RuntimeError(f"authored {stage} failure")
    calls = []

    def refuse(*args, **kwargs):
        calls.append(None)
        raise failure

    if stage == "discovery":
        monkeypatch.setitem(globals(), "_directories", refuse)
    elif stage == "inventory_output":
        # Inventory has been read/asserted; this still precedes disk admission.
        monkeypatch.setitem(globals(), "print", refuse)
    elif stage == "hash":
        real_sha256 = hashlib.sha256

        class RefusingDigest:
            def __init__(self):
                self.real = real_sha256()

            def update(self, chunk):
                return self.real.update(chunk)

            def digest(self):
                # Real first-file hashing/append completed; authored hash-result
                # refusal supplies no replacement digest or native expectation.
                return refuse()

        monkeypatch.setattr(hashlib, "sha256", RefusingDigest)
    else:
        real_operation = getattr(replicated_source._RawPublisherOwner, stage)

        def fail_after_real_operation(holder, *args):
            result = real_operation(holder, *args)
            calls.append(holder)
            # Append fails partway through ingestion, after TWO real appends;
            # finish fails on return from real finish, not inside native finish.
            if stage == "finish" or len(calls) == 2:
                raise failure
            return result

        monkeypatch.setattr(replicated_source._RawPublisherOwner, stage, fail_after_real_operation)
    with pytest.raises(RuntimeError) as failed:
        _prepare()
    assert failed.value is failure
    assert len(calls) == (2 if stage == "append" else 1)
    assert len(holders) == 1  # no fresh fixture/root retry to hide missing funding
    holder = holders[0]
    assert holder._closed and not holder._busy and holder._cleanup_scope.closed
    assert not holder._cleanup_scope.owns(holder)
    assert holder._owner is holder._lease is holder._disk_cm is None
    assert holder._cleanup_failure is None
    assert not getattr(failure, "__notes__", ())


@feature_only
def test_prepare_authored_failure_does_not_release_newer_operation(monkeypatch):
    """Synchronous gate choreography only; no concurrent native-reader claim."""
    holders = _capture_prepare_holders(monkeypatch)
    real_append = replicated_source._RawPublisherOwner.append
    failure = RuntimeError("authored failure after newer gate acquisition")
    newer = []

    def fail_with_newer_operation(holder, *args):
        real_append(holder, *args)
        holder._begin_operation()  # authored stand-in for a newer gate owner
        newer.append(holder)
        raise failure

    monkeypatch.setattr(replicated_source._RawPublisherOwner, "append", fail_with_newer_operation)
    try:
        with pytest.raises(RuntimeError) as failed:
            _prepare()
        assert failed.value is failure and len(holders) == 1
        holder = holders[0]
        assert holder._busy and not holder._closed
        assert holder._cleanup_scope.owns(holder) and not holder._cleanup_scope.closed
        assert holder._cleanup_failure is None  # Busy is not sticky retirement
        assert holder._owner is not None and holder._lease._raw_holder is holder
        assert failure.__notes__ == ["raw host cleanup failed: ValueError"]
    finally:
        # The authored newer operation releases ONLY its own gate, then the
        # test explicitly drains the same holder. _prepare must not do this.
        for holder in newer:
            holder._end_operation()
            holder.close()


@feature_only
def test_genuine_reader_busy_drop_retry(monkeypatch):
    real_reserve = replicated_source.reserve_incremental_disk
    exits = []
    observations = {}

    class ObserveRealExit:
        """Test instrumentation only; yielded lease and exit both remain real."""
        def __init__(self, cm):
            self.cm = cm
            self.lease = None

        def __enter__(self):
            self.lease = self.cm.__enter__()
            assert type(self.lease) is resources.DiskLease
            return self.lease

        def __exit__(self, *args):
            native, directory = observations[self.lease._token]
            entry = _snapshot(native)
            assert entry[0] == 4 and entry[1] > 0 and entry[2] > 0
            assert not directory.exists() and not self.lease.closed
            # First owner held a real reader: its charge survives reader Drop.
            if self.lease is first_lease:
                assert entry[2] == held[2] > 0 and entry[3] == held[3] > 0
            result = self.cm.__exit__(*args)  # NEVER replace the real exit
            assert self.lease.closed
            assert _snapshot(native)[0] == 4  # final release follows CM return
            exits.append(self.lease)
            return result

    def observed_reserve(*args, **kwargs):
        return ObserveRealExit(real_reserve(*args, **kwargs))

    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", observed_reserve)
    first, native, directory = _prepare()
    first_lease = first._lease
    first_cm = first._disk_cm
    observations[first_lease._token] = native, directory
    native._test_hold_reader()  # actual original RawHost::claim_reader()
    held = _snapshot(native)
    assert held[5] == 1 and held[3] > 0
    with pytest.raises(ValueError, match="already held"):
        native._test_hold_reader()
    assert _snapshot(native) == held
    filesystem = (directory.stat().st_dev, directory.stat().st_ino, tuple(directory.iterdir()))
    backing = first_lease._backing
    with pytest.raises(ValueError, match="raw host readers in flight"):
        first.close()  # existing ordinary holder/cleanup-slot/native Busy path
    assert _snapshot(native) == held
    assert (directory.stat().st_dev, directory.stat().st_ino, tuple(directory.iterdir())) == filesystem
    assert first._lease is first_lease and first._disk_cm is first_cm
    assert not first_lease.closed and first_lease._backing is backing
    assert first_lease._capacity._owners[first_lease._token] is first_lease
    assert first_lease._raw_holder is first
    assert first._cleanup_scope.owns(first) and first._cleanup_failure is None
    assert not first._busy and not first._closed and exits == []

    # Retiring a different genuine publisher cannot release this original reader.
    second, other, other_directory = _prepare()
    second_lease = second._lease
    observations[second_lease._token] = other, other_directory
    second.close()
    assert _snapshot(other) == (5, 0, 0, 0, 0, 0, 0, 1, 0, 0)
    assert _snapshot(native) == held
    with pytest.raises(ValueError, match="raw host readers in flight"):
        first.close()
    assert first._cleanup_failure is None and _snapshot(native) == held

    native._test_release_reader()  # takes/drops LAST actual native reader
    released_reader = _snapshot(native)
    assert released_reader[5] == 0 and released_reader[1:5] == held[1:5]
    with pytest.raises(ValueError, match="not held"):
        native._test_release_reader()
    with pytest.raises(ValueError):
        native._test_hold_reader()  # original consuming claim never resets
    first.close()  # explicit retry reaches real Ready, retirement, exit, release
    assert exits == [second_lease, first_lease]
    assert _snapshot(native) == (5, 0, 0, 0, 0, 0, 0, 1, 0, 0)
    assert first_lease.closed and not directory.exists()
    assert first._cleanup_scope.closed and first._closed
    assert first._owner is first._lease is first._disk_cm is None


def _sticky_failure_child(*, during_prepare=False):
    # Intentional admission-rooted retention ONLY inside a bounded subprocess.
    if during_prepare:
        # Authored Python finish-return error plus REAL native removal failure.
        # Observe discovery without substituting filesystem/native ownership.
        real_directories = _directories
        real_finish = replicated_source._RawPublisherOwner.finish
        discoveries, captured = [], []
        preparation_error = RuntimeError("authored prepare finish-return failure")

        def observed_directories(parent):
            result = real_directories(parent)
            discoveries.append(result)
            return result

        def fail_after_finish(holder):
            real_finish(holder)
            assert len(discoveries) == 2
            created = discoveries[1] - discoveries[0]
            assert len(created) == 1
            directory = created.pop()
            native = holder._owner
            native._test_hold_reader()
            native._test_release_reader()
            charged = _snapshot(native)
            (directory / "bounded-test-blocker").write_bytes(b"x")
            captured.append((holder, native, directory, charged,
                             holder._lease, holder._disk_cm, holder._cleanup_scope))
            raise preparation_error

        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(globals(), "_directories", observed_directories)
            patch.setattr(replicated_source._RawPublisherOwner, "finish", fail_after_finish)
            with pytest.raises(RuntimeError) as failed:
                _prepare()
        assert failed.value is preparation_error and len(captured) == 1
        holder, native, directory, charged, lease, cm, scope = captured[0]
        failure = holder._cleanup_failure
        assert type(failure) is ValueError and failure is not preparation_error
        assert preparation_error.__notes__ == ["raw host cleanup failed: ValueError"]
        assert not holder._busy
        blocker = directory / "bounded-test-blocker"
    else:
        holder, native, directory = _prepare()
        native._test_hold_reader()
        native._test_release_reader()
        lease, cm, scope = holder._lease, holder._disk_cm, holder._cleanup_scope
        charged = _snapshot(native)
        blocker = directory / "bounded-test-blocker"
        blocker.write_bytes(b"x")
        with pytest.raises(ValueError) as failed:
            holder.close()  # real nonrecursive directory removal fails
        failure = failed.value
    assert holder._cleanup_failure is failure
    retained = _snapshot(native)
    assert retained[0] == 3 and retained[7] == retained[8] == 1
    assert retained[2:4] == charged[2:4] and retained[1] > 0
    blocker.unlink()
    with pytest.raises(ValueError) as again:
        holder.close()
    assert again.value is failure  # removing blocker never permits retry/release
    assert _snapshot(native) == retained and directory.exists()
    assert holder._owner is native and holder._lease is lease and holder._disk_cm is cm
    assert not lease.closed and lease._capacity._owners[lease._token] is lease
    assert lease._raw_holder is holder and scope.owns(holder)
    assert not scope.closed and not holder._closed
    print(json.dumps({"directory": str(directory), "parent": native._raw_disk_directory}))


@feature_only
@pytest.mark.parametrize("during_prepare", [False, True])
def test_genuine_cleanup_failure_is_sticky_in_subprocess(during_prepare):
    script = (
        "import runpy; "
        f"runpy.run_path({str(Path(__file__).resolve())!r})"
        f"['_sticky_failure_child'](during_prepare={during_prepare!r})"
    )
    result = subprocess.run([sys.executable, "-I", "-c", script],
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = json.loads(result.stdout.strip())
    directory = Path(paths["directory"])
    assert directory.name.startswith("pllm-source-")
    assert directory.parent == Path(paths["parent"])
    # Child exited, OS closed its handles. External test housekeeping is NOT a
    # successful native retry, final-charge release, or candidate qualification.
    directory.rmdir()


@pytest.mark.skipif(FEATURE, reason="requires separate ordinary production wheel")
def test_production_wheel_has_no_lifecycle_test_hooks():
    _assert_installed_origins()
    assert not any(hasattr(_native.ReplicatedPublisher, name) for name in HOOKS)
    assert not any("lifecycle" in name.lower() or "rawreader" in name.lower()
                   for name in dir(_native))
