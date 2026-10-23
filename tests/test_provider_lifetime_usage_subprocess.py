"""Real bounded POSIX execs through the existing sole UsagePopen reaper.

No model/native imports in children. Each workload touches 8 MiB and loops for
25 ms of process CPU; all pipes contain at most a ready byte or five OS scalars.
Raw wait4 evidence is retained in one fixed slot at the SAME owner syscall, never
collected by a second watcher. The deliberate external-reap test is the exception
that must produce unknown usage, not promote its outside observation into Popen.
"""
from __future__ import annotations

import json
import math
import os
import select
import signal
import subprocess
import sys
import threading
import time

import pytest

from pllm.runtime import process_usage as usage
from pllm.runtime.servers import LocalTopology


pytestmark = pytest.mark.skipif(
    os.name != "posix" or sys.platform not in {"darwin", "linux"} or not hasattr(os, "wait4"),
    reason="real Darwin/Linux wait4 ownership required",
)

_WORK = """
import atexit, os, time
def work():
    data = bytearray(8 * 1024 * 1024)
    for i in range(0, len(data), 4096):
        data[i] = 1
    until = time.process_time() + .025
    counter = 0
    while time.process_time() < until:
        counter = (counter + 1) & 65535
"""
_EXIT_WORK = _WORK + "\natexit.register(work)\n"
_GATED_EXIT_WORK = _EXIT_WORK + "\nos.write(1, b'R')\nos.read(0, 1)\n"


@pytest.fixture
def children():
    owned = []  # At most one direct child per test; never a process history.

    def spawn(code=_EXIT_WORK, *, gate=False, single_pipe=False):
        assert not owned
        process = usage.provider_popen_factory(subprocess.Popen)(
            [sys.executable, "-I", "-c", code],
            stdin=subprocess.PIPE if gate else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL if single_pipe else subprocess.PIPE,
            start_new_session=True,
        )
        assert type(process) is usage.UsagePopen
        owned.append(process)
        observed = {"terminal": None, "terminal_count": 0, "calls": 0}
        syscall = process._usage_wait4

        def capture(pid, flags):
            assert pid == process.pid and process._waitpid_lock.locked()
            observed["calls"] += 1
            result = syscall(pid, flags)
            if result[0] == pid:
                observed["terminal_count"] += 1
                observed["terminal"] = result
            return result

        process._usage_wait4 = capture
        return process, observed

    yield spawn
    for process in owned:
        try:
            if process.poll() is None:
                # Session is test-owned. Kill its complete bounded helper group
                # on failure before asking the SAME Popen owner to reap.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=5)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        # Leak assertion only AFTER owner retirement, never a competing reap.
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)


def ready(process):
    readable, _, _ = select.select([process.stdout], [], [], 5)
    assert readable and os.read(process.stdout.fileno(), 1) == b"R"


def assert_exit_usage(process, observed, code=0):
    raw = observed["terminal"]
    assert observed["terminal_count"] == 1 and raw[0] == process.pid
    assert os.WIFEXITED(raw[1]) or os.WIFSIGNALED(raw[1])
    record = process.final_usage()
    assert record["coverage"] == "os_process_exit"
    assert record["source"] == "wait4" and record["exit_code"] == code
    assert record["normal_exit"] is (code == 0)
    # Native OS scalar comparison, not a sampled-RSS stand-in or fixed RSS claim.
    factor = 1 if sys.platform == "darwin" else 1024
    assert process._usage_scale == factor
    assert record["user_cpu_seconds"] == raw[2].ru_utime
    assert record["system_cpu_seconds"] == raw[2].ru_stime
    assert record["cpu_seconds"] == raw[2].ru_utime + raw[2].ru_stime
    assert math.isfinite(record["cpu_seconds"]) and record["cpu_seconds"] >= 0
    assert record["peak_rss_bytes"] == raw[2].ru_maxrss * factor
    assert record["peak_rss_bytes"] > 0
    calls = observed["calls"]
    assert process.wait(timeout=1) == process.poll() == code
    assert process.final_usage() == record and observed["calls"] == calls
    return record


@pytest.mark.parametrize("path", ["wait", "timed_wait", "poll", "communicate", "timed_communicate",
                                  "fast_communicate"])
def test_real_exec_wait_poll_and_communicate_capture_final_cleanup(children, path):
    process, observed = children(single_pipe=path == "fast_communicate")
    assert process.final_usage()["coverage"] == "unknown" and not observed["calls"]
    if path == "wait":
        assert process.wait() == 0
    elif path == "timed_wait":
        assert process.wait(timeout=5) == 0
    elif path == "poll":
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(.001)
        assert process.returncode == 0
    else:
        assert process.communicate(**({"timeout": 5} if path == "timed_communicate" else {})) == (
            b"", None if path == "fast_communicate" else b"")
    assert_exit_usage(process, observed)
    if "communicate" not in path:
        assert process.stdout.read(1) == process.stderr.read(1) == b""


def test_real_concurrent_poll_and_blocking_wait_use_one_reaper(children):
    process, observed = children(_GATED_EXIT_WORK, gate=True)
    ready(process)
    waiting, polled, release = threading.Event(), threading.Event(), threading.Event()
    wait4 = process._usage_wait4

    def capture(pid, flags):
        if flags == 0:
            waiting.set()  # Original Popen lock already held before real wait4.
        return wait4(pid, flags)

    process._usage_wait4 = capture
    results, errors = [None, None], [None, None]

    def waiter():
        try:
            results[0] = process.wait()
        except BaseException as error:
            errors[0] = error

    def poller():
        try:
            assert waiting.wait(5)
            assert process.poll() is None
            assert process.final_usage()["reason"] == "reap_in_progress"
            polled.set()
            assert release.wait(5)
            deadline = time.monotonic() + 5
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(.001)
            results[1] = process.returncode
        except BaseException as error:
            errors[1] = error

    threads = [threading.Thread(target=waiter), threading.Thread(target=poller)]
    for thread in threads:
        thread.start()
    try:
        assert polled.wait(5)
        process.stdin.write(b"x")
        process.stdin.flush()
    finally:
        release.set()
        if not polled.is_set():
            os.killpg(process.pid, signal.SIGKILL)
        for thread in threads:
            thread.join(6)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == [None, None] and results == [0, 0]
    assert_exit_usage(process, observed)
    assert process.stdout.read(1) == process.stderr.read(1) == b""


@pytest.mark.parametrize("first", ["wait", "communicate"])
def test_real_timeout_keeps_unknown_then_same_owner_finishes(children, first):
    process, observed = children(_GATED_EXIT_WORK, gate=True)
    ready(process)
    with pytest.raises(subprocess.TimeoutExpired):
        getattr(process, first)(timeout=.01)
    assert process.returncode is None and observed["terminal"] is None
    assert process.final_usage()["peak_rss_bytes"] is None
    # communicate(None) closes stdin, releasing the gate; the 25 ms exit work
    # still exceeds its 10 ms timeout. wait() leaves the explicit gate untouched.
    if first == "wait":
        assert process.communicate(input=b"x", timeout=5) == (b"", b"")
    else:
        assert process.communicate(timeout=5) == (b"", b"")
    assert_exit_usage(process, observed)


@pytest.mark.parametrize("path", ["poll", "wait", "communicate"])
def test_real_external_reap_cannot_mint_owner_usage(children, path):
    process, observed = children(_EXIT_WORK + "\nraise SystemExit(7)\n")
    # Deliberate external ownership violation, serialized BEFORE any Popen poll.
    pid, status, outside = os.wait4(process.pid, 0)
    assert pid == process.pid and os.WEXITSTATUS(status) == 7
    assert outside.ru_maxrss > 0 and observed["terminal"] is None
    if path == "communicate":
        assert process.communicate(timeout=5) == (b"", b"")
    elif path == "wait":
        process.wait(timeout=5)
    else:
        process.poll()
    record = process.final_usage()
    assert record["coverage"] == "unknown" and record["reason"] == "exit_usage_unavailable"
    assert record["cpu_seconds"] is record["peak_rss_bytes"] is record["exit_code"] is None
    assert observed["terminal_count"] == 0


def test_real_nonzero_exit_retains_usage_without_clean_exit_claim(children):
    process, observed = children(_EXIT_WORK + "\nraise SystemExit(7)\n")
    assert process.communicate(timeout=5) == (b"", b"")
    assert process.returncode == 7
    assert_exit_usage(process, observed, 7)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_real_signal_exit_retains_usage_without_clean_exit_claim(children, sig):
    process, observed = children(_WORK + "\nwork()\nos.write(1,b'R')\nos.read(0,1)\n", gate=True)
    ready(process)
    process.send_signal(sig)  # Popen's pre-signal poll uses the same wait4 owner.
    assert process.communicate(timeout=5) == (b"", b"")
    assert process.returncode == -sig
    assert_exit_usage(process, observed, -sig)


def test_real_reaped_descendant_cpu_and_nonaggregate_hwm_contract(children):
    descendant = _WORK + "\nwork()\n"
    code = """
import json, os, resource, signal, sys
# Keep the direct child's 8 MiB live while its one 8 MiB helper runs.
data = bytearray(8 * 1024 * 1024)
for i in range(0, len(data), 4096): data[i] = 1
pid = os.fork()
if pid == 0:
    signal.alarm(5)
    os.execl(sys.executable, sys.executable, '-I', '-c', CHILD)
reaped = False
try:
    waited, status, r = os.wait4(pid, 0)
    reaped = True
    assert waited == pid and os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    # This SELF CPU snapshot is only a pre-exit lower-bound diagnostic. Final
    # CPU/HWM authority remains the outer UsagePopen's successful post-exit wait4.
    own = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps([r.ru_utime, r.ru_stime, r.ru_maxrss, own.ru_utime, own.ru_stime]), flush=True)
finally:
    # Only this direct-fork owner waits its helper. No Popen or async watcher.
    if not reaped:
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass
        os.waitpid(pid, 0)
""".replace("CHILD", repr(descendant))
    process, observed = children(code)
    out, err = process.communicate(timeout=5)
    assert err == b"" and len(out) < 256
    helper_user, helper_system, helper_hwm, own_user, own_system = json.loads(out)
    assert helper_hwm > 0 and helper_user + helper_system > 0
    record = assert_exit_usage(process, observed)
    assert record["cpu_seconds"] >= helper_user + helper_system
    # Permit one microsecond of timeval/float rounding, not a model-error budget.
    assert record["user_cpu_seconds"] + 1e-6 >= helper_user + own_user
    assert record["system_cpu_seconds"] + 1e-6 >= helper_system + own_system
    t = LocalTopology.__new__(LocalTopology)
    t._process_lock, t._processes = threading.Lock(), {"worker_a": process}
    report = usage.provider_lifetime_usage(t, ("worker_a",),
        local_cpu_loopback=True, source_present=True)
    assert report["complete"] is True  # OS exit coverage only, not a tree-peak claim.
    assert "including reaped descendants" in report["cpu_scope"]
    assert "not simultaneous provider-tree peak" in report["peak_rss_scope"]
    assert report["aggregate_peak_rss_bytes"] is None
    assert len(json.dumps(report).encode()) < 8192
    calls = observed["calls"]
    for local, source in ((False, True), (True, False)):
        unknown = usage.provider_lifetime_usage(t, ("worker_a",),
            local_cpu_loopback=local, source_present=source)
        assert not unknown["complete"]
        assert unknown["roles"]["worker_a"]["cpu_seconds"] is None
        assert unknown["roles"]["worker_a"]["peak_rss_bytes"] is None
    assert observed["calls"] == calls  # Scope checking cannot become a sampler.


def test_unqualified_platform_factory_does_not_promote_real_exec_usage(monkeypatch):
    # Exercise the unqualified-platform selection, not pretend to run another OS.
    # Restore native sys.platform BEFORE standard subprocess execution.
    with monkeypatch.context() as context:
        context.setattr(usage.sys, "platform", "unqualified-posix")
        factory = usage.provider_popen_factory(subprocess.Popen)
        assert factory is subprocess.Popen
    with factory([sys.executable, "-I", "-c", _EXIT_WORK], stdin=subprocess.DEVNULL,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            assert process.communicate(timeout=5) == (b"", b"")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
    t = LocalTopology.__new__(LocalTopology)
    t._process_lock, t._processes = threading.Lock(), {"worker_a": process}
    value = usage.provider_lifetime_usage(t, ("worker_a",),
        local_cpu_loopback=True, source_present=True)
    assert not value["complete"]
    assert value["roles"]["worker_a"]["coverage"] == "unknown"
    assert value["roles"]["worker_a"]["peak_rss_bytes"] is None
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)
