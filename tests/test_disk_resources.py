"""Shared process-local disk ownership, independent of protocol/model policy."""
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

import pytest

from pllm.deployment import resources
from pllm.deployment.network import NetworkError


@pytest.fixture
def disk(monkeypatch):
    capacity = resources._DiskCapacity()
    monkeypatch.setattr(resources, "_disk_capacity", capacity)
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(total=100, free=100))
    return capacity


@pytest.mark.parametrize("amount,headroom", [(None, 1), (True, 1), (0, 1), (-1, 1),
    (1.5, 1), (1 << 64, 1), (1, 0), (1, True), (1, 1 << 64)])
def test_disk_limits_reject_before_observation(disk, monkeypatch, tmp_path, amount, headroom):
    monkeypatch.setattr(disk, "observe", lambda *_: pytest.fail("invalid bytes reached disk observation"))
    with pytest.raises(NetworkError):
        with resources.reserve_incremental_disk(amount, directory=tmp_path, minimum_free_bytes=headroom):
            pytest.fail("invalid reservation admitted")
    assert not disk._owners


def test_directory_aliases_share_device_and_preserve_existing_headroom(disk, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    with resources.reserve_incremental_disk(70, directory=first, minimum_free_bytes=20) as lease:
        observed = lease.observation
        assert observed.available_disk_bytes == 100 and observed.headroom_bytes == 80
        for directory in (second, alias):
            with pytest.raises(NetworkError, match="DISK_CAPACITY_CHANGED"):
                with resources.reserve_incremental_disk(11, directory=directory, minimum_free_bytes=1):
                    pytest.fail("same device or prior minimum reserve was ignored")
        with resources.reserve_incremental_disk(10, directory=alias, minimum_free_bytes=1) as second_lease:
            current = second_lease.observation
            assert current.device == observed.device and current.reserved_disk_bytes == 70
            assert current.reserve_bytes == 20 and current.headroom_bytes == 10
    assert not disk._owners


def test_different_devices_have_independent_charges(disk, tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    original = Path.stat
    def stat(path, **kwargs):
        info = original(path, **kwargs)
        if path == first or path == second:
            return SimpleNamespace(st_mode=info.st_mode, st_ino=info.st_ino,
                                   st_dev=1 if path == first else 2)
        return info
    monkeypatch.setattr(Path, "stat", stat)
    with resources.reserve_incremental_disk(80, directory=first, minimum_free_bytes=20):
        with resources.reserve_incremental_disk(80, directory=second, minimum_free_bytes=20):
            assert len(disk._owners) == 2
    assert not disk._owners


def test_disk_rechecks_current_free_space_before_each_owner(disk, tmp_path, monkeypatch):
    with resources.reserve_incremental_disk(40, directory=tmp_path, minimum_free_bytes=10):
        monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(total=100, free=50))
        with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
            with resources.reserve_incremental_disk(1, directory=tmp_path, minimum_free_bytes=10):
                pytest.fail("external disk writer ignored")
    assert not disk._owners


def test_disk_concurrency_failure_release_and_bounded_ledger(disk, tmp_path):
    entered, release = threading.Event(), threading.Event()
    errors = []
    def hold():
        try:
            with resources.reserve_incremental_disk(80, directory=tmp_path, minimum_free_bytes=20):
                entered.set()
                assert release.wait(5)
                raise RuntimeError("expected body failure")
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
            with resources.reserve_incremental_disk(1, directory=tmp_path, minimum_free_bytes=1):
                pytest.fail("concurrent owner double spent disk")
    finally:
        release.set(); thread.join(5)
    assert not thread.is_alive() and errors == ["expected body failure"] and not disk._owners
    disk.MAX_OWNERS = 2
    with ExitStack() as stack:
        for _ in range(2):
            stack.enter_context(resources.reserve_incremental_disk(1, directory=tmp_path, minimum_free_bytes=1))
        with pytest.raises(NetworkError, match="RESERVATION_CAPACITY_EXHAUSTED"):
            with resources.reserve_incremental_disk(1, directory=tmp_path, minimum_free_bytes=1):
                pytest.fail("unbounded logical owner ledger")
    assert not disk._owners


@pytest.mark.parametrize("free,total", [(True, 100), (-1, 100), (101, 100), (0, 0), (1.0, 100)])
def test_invalid_disk_observation_never_leaves_owner(disk, tmp_path, monkeypatch, free, total):
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=free, total=total))
    with pytest.raises(NetworkError, match="OBSERVATION_UNAVAILABLE"):
        with resources.reserve_incremental_disk(1, directory=tmp_path, minimum_free_bytes=1):
            pytest.fail("invalid live observation admitted")
    assert not disk._owners


def test_unavailable_directory_never_leaves_owner(disk, tmp_path):
    with pytest.raises(NetworkError, match="OBSERVATION_UNAVAILABLE"):
        with resources.reserve_incremental_disk(1, directory=tmp_path / "missing", minimum_free_bytes=1):
            pytest.fail("missing filesystem admitted")
    assert not disk._owners


def backing(directory, amount, inode=1):
    return {"device": directory.stat().st_dev, "file": inode, "allocated_bytes": amount}


def test_22_gib_source_already_written_leaves_room_for_pending_second_tape(disk, tmp_path, monkeypatch):
    gib, free = 1 << 30, [22 << 30]
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(total=22*gib, free=free[0]))
    # One worker's ledger. Other processes' files affect live free space only:
    # 3 sources * 4GiB + first tape * 4GiB written, second tape still pending.
    with resources.reserve_incremental_disk(4*gib, directory=tmp_path, minimum_free_bytes=2*gib) as source:
        free[0] = 6*gib
        with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
            with resources.reserve_incremental_disk(4*gib, directory=tmp_path, minimum_free_bytes=1):
                pytest.fail("unbound source was speculatively discounted")
        source.bind_backing(lambda: [backing(tmp_path, 4*gib)])
        with resources.reserve_incremental_disk(4*gib, directory=tmp_path, minimum_free_bytes=1):
            observed = disk.observe(tmp_path, 1)
            assert (observed.reserved_disk_bytes, observed.backed_disk_bytes,
                    observed.pending_disk_bytes, observed.headroom_bytes) == (8*gib, 4*gib, 4*gib, 0)
            with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
                with resources.reserve_incremental_disk(1, directory=tmp_path, minimum_free_bytes=1):
                    pytest.fail("unallocated second tape was discarded")
        assert not source.closed and len(disk._owners) == 1
    assert source.closed and not disk._owners


def test_real_owned_backing_and_sparse_extent_never_credit_logical_length(disk, tmp_path, monkeypatch):
    budget = 128 << 10
    free = [2 << 20]
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(total=2 << 20, free=free[0]))
    with resources.reserve_incremental_disk(budget, directory=tmp_path, minimum_free_bytes=4096) as lease:
        with tempfile.TemporaryFile(dir=tmp_path) as owned:
            owned.truncate(budget)
            os.fsync(owned.fileno())
            def sample():
                assert disk.lock._is_owned()
                info = os.fstat(owned.fileno())
                return [{"device": info.st_dev, "file": info.st_ino,
                         "allocated_bytes": info.st_blocks * 512}]
            if not hasattr(os.fstat(owned.fileno()), "st_blocks"):
                pytest.skip("physical allocation observation unavailable")
            sparse = os.fstat(owned.fileno())
            free[0] = (2 << 20) - sparse.st_blocks * 512
            lease.bind_backing(sample)
            observed = disk.observe(tmp_path, 1)
            assert observed.pending_disk_bytes == budget - min(budget, sparse.st_blocks * 512)
            # Bounded physical writes into the sparse extent, followed by fresh
            # fstat (not truncate's logical length) and a corresponding free drop.
            owned.write(b"x" * 16384)
            owned.flush(); os.fsync(owned.fileno())
            allocated = os.fstat(owned.fileno()).st_blocks * 512
            assert allocated > 0
            free[0] = (2 << 20) - allocated
            observed = disk.observe(tmp_path, 1)
            assert observed.backed_disk_bytes == min(budget, allocated)
            assert observed.headroom_bytes == free[0] - observed.pending_disk_bytes - 4096
        # A closed provider cannot reuse the earlier observation's credit.
        with pytest.raises(NetworkError, match="DISK_BACKING_UNAVAILABLE"):
            disk.observe(tmp_path, 1)
        assert not lease.closed and len(disk._owners) == 1
    # Cleanup removes the lease without calling the now-closed provider.
    assert lease.closed and not disk._owners


def test_unique_inode_credit_is_shared_capped_and_resampled(disk, tmp_path):
    records = [backing(tmp_path, 80)] * 2
    other_records = [backing(tmp_path, 80)]
    with resources.reserve_incremental_disk(20, directory=tmp_path, minimum_free_bytes=10) as first:
        with resources.reserve_incremental_disk(70, directory=tmp_path, minimum_free_bytes=10) as second:
            first.bind_backing(lambda: records)
            second.bind_backing(lambda: other_records)
            current = disk.observe(tmp_path, 1)
            assert current.backed_disk_bytes == 80 and current.pending_disk_bytes == 10
            # Duplicate samples never add, and differing samples use the smaller
            # observed allocation. Credit can fall; no cached optimistic value.
            records[0] = backing(tmp_path, 30)
            current = disk.observe(tmp_path, 1)
            assert current.backed_disk_bytes == 30 and current.pending_disk_bytes == 60
            records[:] = [backing(tmp_path, (1 << 64) - 1)]
            other_records.clear()  # no additional physical backing claimed
            current = disk.observe(tmp_path, 1)
            assert current.backed_disk_bytes == 20 and current.pending_disk_bytes == 70


@pytest.mark.parametrize("field,value", [(field, value)
    for field in ("device", "file", "allocated_bytes")
    for value in (True, -1, 1 << 64, 1.5, float("nan"), float("inf"), "1")])
def test_malformed_backing_fields_leave_full_pending_bound(disk, tmp_path, field, value):
    with resources.reserve_incremental_disk(40, directory=tmp_path, minimum_free_bytes=10) as lease:
        with pytest.raises(NetworkError, match="BACKING_UNAVAILABLE"):
            lease.bind_backing(lambda: [{**backing(tmp_path, 40), field: value}])
        current = disk.observe(tmp_path, 1)
        assert current.pending_disk_bytes == 40 and current.backed_disk_bytes == 0


@pytest.mark.parametrize("records", [None, {}, (), [None], [{}],
    [{"device": 1, "file": 1, "allocated_bytes": 1, "path": "/untrusted"}]])
def test_malformed_backing_container_leaves_pending_bound(disk, tmp_path, records):
    with resources.reserve_incremental_disk(40, directory=tmp_path, minimum_free_bytes=10) as lease:
        with pytest.raises(NetworkError, match="BACKING_UNAVAILABLE"):
            lease.bind_backing(lambda: records)
        assert disk.observe(tmp_path, 1).pending_disk_bytes == 40


def test_empty_backing_and_rejected_bindings_preserve_lifetime_and_owner_limit(disk, tmp_path):
    disk.MAX_OWNERS = 1
    with resources.reserve_incremental_disk(40, directory=tmp_path, minimum_free_bytes=10) as lease:
        with pytest.raises(NetworkError, match="BINDING_INVALID"):
            lease.bind_backing([backing(tmp_path, 40)])
        lease.bind_backing(lambda: [])  # unsupported platform / no allocated files
        with pytest.raises(NetworkError, match="BINDING_INVALID"):
            lease.bind_backing(lambda: [backing(tmp_path, 40)])
        assert disk.observe(tmp_path, 1).pending_disk_bytes == 40
        with pytest.raises(NetworkError, match="CAPACITY_EXHAUSTED"):
            with resources.reserve_incremental_disk(1, directory=tmp_path, minimum_free_bytes=1):
                pytest.fail("live lease was dropped")
    with pytest.raises(NetworkError, match="BINDING_INVALID"):
        lease.bind_backing(lambda: [backing(tmp_path, 40)])


def test_backing_withdrawal_keeps_full_bound_and_does_not_sample_closed_owner(disk, tmp_path):
    alive, samples = [True], []
    def sample():
        assert alive[0]
        samples.append(1)
        return [backing(tmp_path, 40)]
    with resources.reserve_incremental_disk(40, directory=tmp_path, minimum_free_bytes=10) as lease:
        lease.bind_backing(sample)
        assert disk.observe(tmp_path, 1).pending_disk_bytes == 0
        lease.clear_backing()
        alive[0] = False
        count = len(samples)
        assert disk.observe(tmp_path, 1).pending_disk_bytes == 40
        assert len(samples) == count and not lease.closed
    lease.clear_backing()  # idempotent retirement, no callback or new reservation
    assert not disk._owners and len(samples) == count


def test_backing_records_have_combined_bound_and_do_not_consume_iterators(disk, tmp_path):
    disk.MAX_BACKING_FILES = 2
    def unbounded():
        pytest.fail("backing iterator consumed")
        yield  # only plain bounded lists are accepted
    with resources.reserve_incremental_disk(40, directory=tmp_path, minimum_free_bytes=10) as first:
        with pytest.raises(NetworkError, match="BACKING_UNAVAILABLE"):
            first.bind_backing(unbounded)
        first.bind_backing(lambda: [backing(tmp_path, 10, 1), backing(tmp_path, 10, 2)])
        with resources.reserve_incremental_disk(40, directory=tmp_path, minimum_free_bytes=10) as second:
            with pytest.raises(NetworkError, match="BACKING_UNAVAILABLE"):
                second.bind_backing(lambda: [backing(tmp_path, 10, 3)])
            observed = disk.observe(tmp_path, 1)
            assert observed.backed_disk_bytes == 20 and observed.pending_disk_bytes == 60


def test_backing_on_other_device_cannot_credit_either_ledger(disk, tmp_path, monkeypatch):
    first, second = tmp_path / "native", tmp_path / "tape"
    first.mkdir(); second.mkdir()
    original = Path.stat
    def stat(path, **kwargs):
        info = original(path, **kwargs)
        if path in (first, second):
            return SimpleNamespace(st_mode=info.st_mode, st_ino=info.st_ino,
                                   st_dev=1 if path == first else 2)
        return info
    monkeypatch.setattr(Path, "stat", stat)
    with resources.reserve_incremental_disk(70, directory=first, minimum_free_bytes=20) as native:
        with resources.reserve_incremental_disk(70, directory=second, minimum_free_bytes=20):
            with pytest.raises(NetworkError, match="BACKING_UNAVAILABLE"):
                native.bind_backing(lambda: [backing(second, 70)])
            native.bind_backing(lambda: [backing(first, 70)])
            assert disk.observe(first, 1).pending_disk_bytes == 0
            assert disk.observe(second, 1).pending_disk_bytes == 70


def test_callbacks_precede_fresh_free_and_cannot_discard_concurrent_pending(disk, tmp_path, monkeypatch):
    free, calls = [100], []
    entered, sample_release, owner_release = (threading.Event() for _ in range(3))
    second_started, second_finished = threading.Event(), threading.Event()
    block, outcomes = [False], []
    def usage(_):
        calls.append("free")
        return SimpleNamespace(total=100, free=free[0])
    monkeypatch.setattr(resources.shutil, "disk_usage", usage)
    def sample():
        assert disk.lock._is_owned()
        calls.append("backing")
        if block[0]:
            block[0] = False
            entered.set()
            assert sample_release.wait(5)
        return [backing(tmp_path, 40)]
    def hold(second=False):
        if second: second_started.set()
        try:
            with resources.reserve_incremental_disk(30, directory=tmp_path, minimum_free_bytes=20):
                outcomes.append("held")
                assert owner_release.wait(5)
        except NetworkError as error:
            outcomes.append(str(error))
        finally:
            if second: second_finished.set()
    with resources.reserve_incremental_disk(40, directory=tmp_path, minimum_free_bytes=20) as source:
        free[0] = 60
        source.bind_backing(sample)
        calls.clear(); block[0] = True
        first_thread, second_thread = threading.Thread(target=hold), threading.Thread(target=hold, args=(True,))
        first_thread.start()
        try:
            assert entered.wait(5)
            second_thread.start()
            assert second_started.wait(5)
            assert not second_finished.wait(.02) and calls == ["backing"]
            sample_release.set()
            assert second_finished.wait(5)
            assert outcomes == ["held", "DISK_CAPACITY_CHANGED"]
            assert calls == ["backing", "free", "backing", "free"]
        finally:
            sample_release.set(); owner_release.set()
            first_thread.join(5)
            if second_thread.ident is not None: second_thread.join(5)
        assert not first_thread.is_alive() and not second_thread.is_alive()
    assert not disk._owners
