"""Darwin swap counters must not be substituted with file pageout activity."""

import ctypes
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from scripts import host_swap


def test_darwin_pageouts_are_diagnostic_not_swap(monkeypatch):
    monkeypatch.setattr(host_swap.sys, "platform", "darwin")
    monkeypatch.setattr(host_swap.os, "sysconf", lambda _: 16384)
    monkeypatch.setattr(host_swap, "_darwin_kernel_page_size", lambda: 16384)
    monkeypatch.setattr(psutil, "swap_memory", lambda: SimpleNamespace(used=100, sout=900 * 16384))
    stats = SimpleNamespace(pageouts=900, swapouts=12)
    monkeypatch.setattr(host_swap, "_darwin_vm_statistics", lambda: stats)
    before = host_swap.swap_snapshot()
    stats.pageouts += 7
    after = host_swap.swap_snapshot()
    assert before["swapout_bytes"] == after["swapout_bytes"] == 12 * 16384
    assert after["pageout_bytes_diagnostic"] - before["pageout_bytes_diagnostic"] == 7 * 16384
    stats.swapouts += 1
    assert host_swap.swap_snapshot()["swapout_bytes"] - before["swapout_bytes"] == 16384
    from scripts.probe_gkm_checkpoint_bounds import resource_reason
    common = {"total_bytes": 32 << 30, "available_bytes": 10 << 30, "rss_bytes": 1 << 20}
    assert resource_reason(common | after, common | before, 2 << 20, elapsed=1, cpu=1) is None
    assert resource_reason(common | host_swap.swap_snapshot(), common | before,
                           2 << 20, elapsed=1, cpu=1) == "new_swapout"
    assert resource_reason(common | after | {"swap_used_bytes": 101}, common | before,
                           2 << 20, elapsed=1, cpu=1) == "new_swap_usage"


@pytest.mark.parametrize("status,count,port,released", [(1, 38, 42, [42]), (0, 24, 42, [42]), (0, 38, 0, [])])
def test_mach_failures_reject_and_release_port(monkeypatch, status, count, port, released):
    ports = []

    def read(host, flavor, data, words):
        ctypes.cast(words, ctypes.POINTER(ctypes.c_uint32)).contents.value = count
        return status

    def release(task, host):
        ports.append(host)
        return 0

    library = SimpleNamespace(mach_host_self=lambda: port, host_statistics64=read, mach_port_deallocate=release)
    monkeypatch.setattr(host_swap, "_libsystem", lambda: (library, 7))
    with pytest.raises(RuntimeError):
        host_swap._darwin_vm_statistics()
    assert ports == released


@pytest.mark.parametrize("change", [{"swapout_bytes": 99}, {"counter_source": "other"}, {"page_bytes": 4096}])
def test_counter_reset_and_identity_changes_fail_closed(change):
    before = {"schema": "pllm.host_swap_observation.v1", "swapout_bytes": 100, "swap_used_bytes": 0,
              "counter_source": "darwin.HOST_VM_INFO64.rev1.swapouts", "page_bytes": 16384}
    with pytest.raises(RuntimeError):
        host_swap.validate_counter_identity(before | change, before)


@pytest.fixture
def darwin_sample(monkeypatch):
    monkeypatch.setattr(host_swap.sys, "platform", "darwin")
    # Deliberately disagree: process pages must never convert Mach counters.
    monkeypatch.setattr(host_swap.os, "sysconf", lambda _: 4096)
    monkeypatch.setattr(host_swap, "_darwin_kernel_page_size", lambda: 16384)
    monkeypatch.setattr(psutil, "swap_memory", lambda: SimpleNamespace(used=100, sout=900))
    monkeypatch.setattr(host_swap, "_darwin_physical_total_bytes", lambda: 32 << 30)
    stats = SimpleNamespace(free_count=5228, speculative_count=1000, swapouts=12, pageouts=900,
                            inactive_count=100000, purgeable_count=10**12, external_page_count=10**12)
    calls = []
    def read():
        calls.append(True)
        return stats
    monkeypatch.setattr(host_swap, "_darwin_vm_statistics", read)
    return stats, calls


@pytest.mark.parametrize("page", [4096, 16384, 65536])
def test_host_snapshot_kernel_pages_and_one_mach_sample(monkeypatch, darwin_sample, page):
    stats, calls = darwin_sample
    monkeypatch.setattr(host_swap, "_darwin_kernel_page_size", lambda: page)
    row = host_swap.host_snapshot()
    assert calls == [True]
    assert row["free_bytes"] == 4228 * page
    if page == 16384:
        assert row["free_bytes"] == 69271552
    assert row["swapout_bytes"] == stats.swapouts * page
    assert row["swap_used_bytes"] == 100
    assert row["page_bytes"] == page
    assert row["free_count_including_speculative"] == 5228
    assert row["speculative_count"] == 1000
    assert row["inactive_count"] == 100000
    assert row["available_estimate_bytes"] == (5228 + 100000) * page
    assert row["available_estimate_source"] == host_swap.AVAILABLE_SOURCE
    assert row["page_size_source"] == "darwin.host_page_size"


@pytest.mark.parametrize("field", ["free_count", "speculative_count", "inactive_count", "swapouts", "pageouts"])
@pytest.mark.parametrize("value", [None, -1, True, False, 1.5, "1", "missing", 1 << 64])
def test_host_snapshot_invalid_or_missing_counters(darwin_sample, field, value):
    stats, _ = darwin_sample
    if value == "missing":
        delattr(stats, field)
    else:
        setattr(stats, field, value)
    with pytest.raises(RuntimeError):
        host_swap.host_snapshot()


def test_host_snapshot_speculative_exceeds_free(darwin_sample):
    stats, _ = darwin_sample
    stats.speculative_count = stats.free_count + 1
    with pytest.raises(RuntimeError, match="inconsistent"):
        host_swap.host_snapshot()


@pytest.mark.parametrize("page", [None, True, False, -4096, 0, 8192, 16384.0, "16384"])
def test_host_snapshot_invalid_kernel_page_size(monkeypatch, darwin_sample, page):
    monkeypatch.setattr(host_swap, "_darwin_kernel_page_size", lambda: page)
    with pytest.raises(RuntimeError, match="page size"):
        host_swap.host_snapshot()


@pytest.mark.parametrize("status,page,port,release_status,valid", [
    (0, 4096, 42, 0, True), (0, 16384, 42, 0, True), (0, 65536, 42, 0, True),
    (1, 16384, 42, 0, False), (0, 0, 42, 0, False), (0, 8192, 42, 0, False),
    (0, 16384, 0, 0, False), (0, 16384, 42, 1, False),
])
def test_kernel_page_size_checked_and_port_released(monkeypatch, status, page, port, release_status, valid):
    ports = []
    def read(host, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint32)).contents.value = page
        return status
    def release(task, host):
        ports.append(host)
        return release_status
    library = SimpleNamespace(mach_host_self=lambda: port, host_page_size=read,
                              mach_port_deallocate=release)
    monkeypatch.setattr(host_swap, "_libsystem", lambda: (library, 7))
    if valid:
        assert host_swap._darwin_kernel_page_size() == page
    else:
        with pytest.raises(RuntimeError):
            host_swap._darwin_kernel_page_size()
    assert ports == ([42] if port else [])


def test_linux_swap_path_unchanged(monkeypatch):
    monkeypatch.setattr(host_swap.sys, "platform", "linux")
    monkeypatch.setattr(host_swap.os, "sysconf", lambda _: 4096)
    monkeypatch.setattr(psutil, "swap_memory", lambda: SimpleNamespace(used=100, sout=900))
    row = host_swap.swap_snapshot()
    assert row["swapout_bytes"] == 900 and row["swap_used_bytes"] == 100
    assert row["page_bytes"] == 4096
    assert row["counter_source"] == "linux.psutil.proc_vmstat.pswpout"


@pytest.mark.parametrize("status,size,total,valid", [
    (0, 8, 32 << 30, True), (1, 8, 32 << 30, False),
    (0, 0, 32 << 30, False), (0, 4, 32 << 30, False),
    (0, 16, 32 << 30, False), (0, 8, 0, False),
])
def test_hw_memsize_checked_status_size_value_and_cache(monkeypatch, status, size, total, valid):
    calls = []
    def read(name, output, length, new, new_length):
        calls.append(name)
        assert name == b"hw.memsize" and new is None and new_length == 0
        assert ctypes.cast(length, ctypes.POINTER(ctypes.c_size_t)).contents.value == 8
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint64)).contents.value = total
        ctypes.cast(length, ctypes.POINTER(ctypes.c_size_t)).contents.value = size
        return status
    monkeypatch.setattr(host_swap, "_libsystem", lambda: (SimpleNamespace(sysctlbyname=read), 7))
    host_swap._darwin_physical_total_bytes.cache_clear()
    try:
        if valid:
            assert host_swap._darwin_physical_total_bytes() == total
            assert host_swap._darwin_physical_total_bytes() == total
            assert calls == [b"hw.memsize"]
        else:
            for _ in range(2):
                with pytest.raises(RuntimeError):
                    host_swap._darwin_physical_total_bytes()
            assert len(calls) == 2  # Failures cannot populate cache.
    finally:
        host_swap._darwin_physical_total_bytes.cache_clear()


@pytest.mark.parametrize("total", [None, True, False, -1, 0, 1.5, "34359738368", 1 << 64, 1])
def test_host_snapshot_invalid_or_impossible_physical_total(monkeypatch, darwin_sample, total):
    monkeypatch.setattr(host_swap, "_darwin_physical_total_bytes", lambda: total)
    with pytest.raises(RuntimeError):
        host_swap.host_snapshot()


@pytest.mark.parametrize("field", ["free_count", "speculative_count", "inactive_count"])
def test_host_snapshot_uint32_counter_overflow(darwin_sample, field):
    stats, _ = darwin_sample
    setattr(stats, field, 1 << 32)
    with pytest.raises(RuntimeError):
        host_swap.host_snapshot()


def test_inactive_used_but_purgeable_filebacked_not_added(darwin_sample):
    stats, _ = darwin_sample
    before = host_swap.host_snapshot()
    stats.inactive_count += 1
    stats.purgeable_count *= 2
    stats.external_page_count *= 2
    after = host_swap.host_snapshot()
    assert after["available_estimate_bytes"] - before["available_estimate_bytes"] == 16384
    assert after["free_bytes"] == before["free_bytes"]


def test_swap_snapshot_compatible_without_headroom_counters(monkeypatch, darwin_sample):
    stats, _ = darwin_sample
    del stats.inactive_count
    del stats.free_count
    del stats.speculative_count
    def forbidden():
        raise AssertionError("swap-only reader must not request physical memory")
    monkeypatch.setattr(host_swap, "_darwin_physical_total_bytes", forbidden)
    row = host_swap.swap_snapshot()
    assert row["swapout_bytes"] == 12 * 16384
    assert row["swap_used_bytes"] == 100
    assert "available_estimate_bytes" not in row and "free_bytes" not in row


@pytest.mark.parametrize("missing_symbol", [False, True])
def test_hw_memsize_missing_symbol_or_unwritten_output_fails_closed(monkeypatch, missing_symbol):
    library = SimpleNamespace()
    if not missing_symbol:
        library.sysctlbyname = lambda *args: 0  # Missing value stays zero.
    monkeypatch.setattr(host_swap, "_libsystem", lambda: (library, 7))
    host_swap._darwin_physical_total_bytes.cache_clear()
    try:
        with pytest.raises((RuntimeError, AttributeError)):
            host_swap._darwin_physical_total_bytes()
    finally:
        host_swap._darwin_physical_total_bytes.cache_clear()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin independent OS reader")
def test_actual_counter_matches_independent_vm_stat():
    def counters():
        output = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=5)
        return {line.split(":", 1)[0]: int(line.split(":", 1)[1].strip().rstrip("."))
                for line in output.splitlines()[1:] if ":" in line}

    before = counters()
    actual = host_swap.swap_snapshot()
    after = counters()
    assert before["Swapouts"] <= actual["swapout_pages"] <= after["Swapouts"]
    assert actual["swapout_bytes"] == actual["swapout_pages"] * actual["page_bytes"]
    assert actual["counter_source"] == "darwin.HOST_VM_INFO64.rev1.swapouts"
    assert ctypes.sizeof(host_swap._VmStatisticsRev1) == 152
    assert host_swap._VmStatisticsRev1.swapouts.offset == 120
