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
