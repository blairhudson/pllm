"""Independent CPython 3.11--3.13 sole-reaper compatibility matrix.

Only exec/reap boundaries are mocked. Both owners run real Popen.__init__ and
keep its actual _waitpid_lock, wait/communicate/timeout/destructor implementations.
No child, native extension, model, server, or agent is spawned/imported here.

Isolated lightweight invocation (avoids unrelated repository conftest imports):
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=python .venv/bin/python -m pytest \
        --noconftest -q tests/test_provider_usage_reaper_regressions.py
"""
from __future__ import annotations

import errno
from functools import partial
import inspect
import os
import platform
import subprocess
import sys
import threading
from types import SimpleNamespace
import warnings

import pytest

from pllm.runtime import process_usage as usage


pytestmark = pytest.mark.skipif(
    platform.python_implementation() != "CPython"
    or not (3, 11) <= sys.version_info[:2] < (3, 14)
    or sys.platform not in {"darwin", "linux"}
    or not hasattr(os, "wait4"),
    reason="qualified CPython 3.11--3.13 Darwin/Linux Popen implementation required",
)


class ReapBoundary:
    """Finite mock syscall responses, never an alternative production reaper."""

    def __init__(self, process, event):
        self.process = process
        self.original_lock = process._waitpid_lock
        self.event = event
        self.calls = []
        self.entered = None
        self.release = None
        self.in_callback = False

    def syscall(self, pid, flags):
        assert pid == self.process.pid
        assert self.process._waitpid_lock is self.original_lock
        assert self.original_lock.locked()  # Never acquire it again inside reaping.
        assert not self.in_callback, "competing reap callback"
        self.in_callback = True
        self.calls.append((pid, flags))
        try:
            if self.entered is not None:
                self.entered.set()
                assert self.release.wait(2), "test callback gate was not released"
            if self.event == "echild":
                raise ChildProcessError(errno.ECHILD, "audit: no child status")
            if self.event == "eio":
                raise OSError(errno.EIO, "audit: source error")
            if self.event == "live":
                assert flags == os.WNOHANG
                result = (0, 0, None)
            else:
                assert self.event == "exit7"
                result = (pid, 7 << 8, SimpleNamespace(
                    ru_utime=.75, ru_stime=.25, ru_maxrss=16))
            return result if type(self.process) is usage.UsagePopen else result[:2]
        finally:
            self.in_callback = False


def bind_poll_boundary(process, boundary, monkeypatch):
    # Popen captures this syscall in finalizer-safe defaults. Merely patching
    # os.waitpid would accidentally exercise the REAL syscall in poll/__del__.
    method = subprocess.Popen._internal_poll.__get__(process)
    parameters = inspect.signature(method).parameters
    if "_waitpid" in parameters:  # Actual 3.11 stdlib source.
        replacement = partial(method, _waitpid=boundary.syscall)
    else:  # Actual 3.13 stdlib source, retaining the original method body.
        assert "_del_safe" in parameters
        replacement = partial(method, _del_safe=SimpleNamespace(
            waitpid=boundary.syscall, WNOHANG=os.WNOHANG, ECHILD=errno.ECHILD))
    monkeypatch.setattr(process, "_internal_poll", replacement)


@pytest.fixture
def owners(monkeypatch):
    owned = []
    ordinary = {}
    # Avoid touching somebody else's Popen finalizer entries during __init__.
    if getattr(subprocess, "_active", ()):
        pytest.skip("isolated mock-constructor test requires no pending Popen owners")

    def exec_boundary(process, *args, **kwargs):
        # Real __init__ creates its lock, communication state, and normal fields.
        # No PIPE/DEVNULL handles or actual _execute_child operation are involved.
        assert process.stdin is process.stdout is process.stderr is None
        process.pid = 12345 + len(owned)
        process._child_created = True  # The mocked exec boundary owns this marker.

    def waitpid_boundary(pid, flags):
        assert pid in ordinary, "UsagePopen must never fall back to old waitpid"
        return ordinary[pid].syscall(pid, flags)

    monkeypatch.setattr(subprocess.Popen, "_execute_child", exec_boundary)
    monkeypatch.setattr(subprocess.os, "waitpid", waitpid_boundary)

    def make(cls, event="echild"):
        process = cls(["not-executed-provider-audit"])
        owned.append(process)
        boundary = ReapBoundary(process, event)
        if cls is usage.UsagePopen:
            monkeypatch.setattr(process, "_usage_wait4", boundary.syscall)
        else:
            ordinary[process.pid] = boundary
            bind_poll_boundary(process, boundary, monkeypatch)
        return boundary

    yield make
    # No OS child exists. Disable implicit finalization before monkeypatch undo;
    # tests invoke the genuine destructor explicitly when auditing that path.
    for process in owned:
        process._child_created = False


def invoke(boundary, path):
    process = boundary.process
    if path == "poll":
        return process.poll()
    if path == "gc_poll":
        return process._internal_poll(_deadstate=sys.maxsize)
    if path == "wait":
        return process.wait()
    if path == "timed_wait":
        return process.wait(timeout=1)
    assert path in {"communicate", "timed_communicate"}
    assert process.communicate(**({"timeout": 1} if path == "timed_communicate" else {})) == (None, None)
    return process.returncode


def outcome(boundary, path):
    try:
        return ("return", invoke(boundary, path))
    except OSError as error:
        return ("raise", type(error), error.errno)


def assert_unknown(process, reason=None):
    record = process.final_usage()  # Metadata access must not invoke a reap.
    assert record["coverage"] == "unknown"
    if reason is not None:
        assert record["reason"] == reason
    for field in ("exit_code", "user_cpu_seconds", "system_cpu_seconds",
                  "cpu_seconds", "peak_rss_bytes", "source", "normal_exit"):
        assert record[field] is None
    return record


@pytest.mark.parametrize("deadstate", [0, 91, sys.maxsize])
def test_gc_poll_preserves_popen_deadstate_after_echild(owners, deadstate):
    ordinary = owners(subprocess.Popen)
    observed = owners(usage.UsagePopen)
    assert ordinary.process._internal_poll(_deadstate=deadstate) == deadstate
    assert observed.process._internal_poll(_deadstate=deadstate) == deadstate
    assert ordinary.calls == [(ordinary.process.pid, os.WNOHANG)]
    assert observed.calls == [(observed.process.pid, os.WNOHANG)]
    assert not observed.original_lock.locked()
    assert_unknown(observed.process, "exit_usage_unavailable")


def test_inherited_destructor_preserves_popen_missing_status_sentinel(owners):
    ordinary = owners(subprocess.Popen)
    observed = owners(usage.UsagePopen)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        ordinary.process.__del__()
        observed.process.__del__()
    assert ordinary.process.returncode == observed.process.returncode == sys.maxsize
    assert len(ordinary.calls) == len(observed.calls) == 1
    assert_unknown(observed.process, "exit_usage_unavailable")


@pytest.mark.parametrize("path", ["poll", "wait", "timed_wait", "communicate", "timed_communicate"])
def test_ordinary_poll_and_wait_keep_existing_zero_status_unknown_usage(owners, path):
    ordinary = owners(subprocess.Popen)
    observed = owners(usage.UsagePopen)
    assert invoke(ordinary, path) == invoke(observed, path) == 0
    assert len(ordinary.calls) == len(observed.calls) == 1
    assert_unknown(observed.process, "exit_usage_unavailable")


@pytest.mark.parametrize("path", ["poll", "gc_poll", "wait", "communicate", "timed_communicate"])
def test_deadstate_does_not_override_a_genuinely_observed_nonzero_exit(owners, path):
    ordinary = owners(subprocess.Popen, "exit7")
    observed = owners(usage.UsagePopen, "exit7")
    assert invoke(ordinary, path) == invoke(observed, path) == 7
    record = observed.process.final_usage()
    assert record["coverage"] == "os_process_exit"
    assert record["exit_code"] == 7 and record["normal_exit"] is False
    assert record["cpu_seconds"] == 1.0
    # Native-host unit contract only. Darwin execution does NOT validate Linux.
    assert record["peak_rss_bytes"] == 16 * (1 if sys.platform == "darwin" else 1024)
    assert len(ordinary.calls) == len(observed.calls) == 1


@pytest.mark.parametrize("event", ["echild", "eio", "exit7"])
@pytest.mark.parametrize("paths", [
    ("poll", "wait"), ("wait", "poll"), ("poll", "communicate"),
    ("communicate", "poll"), ("gc_poll", "wait"), ("wait", "gc_poll"),
    ("timed_communicate", "wait"),
])
def test_ordered_reap_matrix_preserves_literal_stdlib_status_and_cache(owners, event, paths):
    ordinary = owners(subprocess.Popen, event)
    observed = owners(usage.UsagePopen, event)
    expected = [outcome(ordinary, path) for path in paths]
    actual = [outcome(observed, path) for path in paths]
    if event == "exit7":
        assert expected == [("return", 7)] * 2
    elif event == "echild":
        code = sys.maxsize if paths[0] == "gc_poll" else 0
        assert expected == [("return", code)] * 2
    elif paths[0] == "gc_poll":
        assert expected == [("return", sys.maxsize)] * 2
    else:
        literal = {"poll": ("return", None), "gc_poll": ("return", sys.maxsize)}
        assert expected == [literal.get(path, ("raise", OSError, errno.EIO))
                            for path in paths]
    assert actual == expected
    assert ordinary.process.returncode == observed.process.returncode
    assert [flags for _, flags in ordinary.calls] == [flags for _, flags in observed.calls]
    if event != "exit7":
        assert_unknown(observed.process)
    else:
        assert observed.process.final_usage()["exit_code"] == 7
    count = len(observed.calls)
    observed.process.final_usage()
    assert len(observed.calls) == count  # Reports cannot become additional reapers.


@pytest.mark.parametrize("path", ["poll", "gc_poll", "wait", "timed_wait",
                                 "communicate", "timed_communicate"])
def test_source_error_never_mints_exit_usage_or_forces_zero_returncode(owners, path):
    ordinary = owners(subprocess.Popen, "eio")
    observed = owners(usage.UsagePopen, "eio")
    expected = outcome(ordinary, path)
    assert expected == (("return", sys.maxsize) if path == "gc_poll"
                        else ("return", None) if path == "poll"
                        else ("raise", OSError, errno.EIO))
    assert outcome(observed, path) == expected
    assert observed.process.returncode == ordinary.process.returncode
    assert_unknown(observed.process)
    assert not observed.original_lock.locked()


@pytest.mark.parametrize("path", ["wait", "communicate"])
@pytest.mark.parametrize("terminal", ["exit7", "echild"])
def test_timeout_then_same_owner_resume_preserves_unknown_until_terminal(owners, path, terminal):
    ordinary = owners(subprocess.Popen, "live")
    observed = owners(usage.UsagePopen, "live")
    for boundary in (ordinary, observed):
        with pytest.raises(subprocess.TimeoutExpired):
            getattr(boundary.process, path)(timeout=0)
        assert boundary.process.returncode is None
        assert not boundary.original_lock.locked()
    assert [flags for _, flags in ordinary.calls] == [flags for _, flags in observed.calls]
    assert_unknown(observed.process, "not_reaped")
    ordinary.event = observed.event = terminal
    expected = 7 if terminal == "exit7" else 0
    assert invoke(ordinary, "timed_communicate") == invoke(observed, "timed_communicate") == expected
    count = len(observed.calls)
    assert observed.process.wait(timeout=0) == observed.process.poll() == expected
    assert len(observed.calls) == count
    if terminal == "echild":
        assert_unknown(observed.process, "exit_usage_unavailable")
    else:
        assert observed.process.final_usage()["exit_code"] == 7


@pytest.mark.parametrize("waiter", ["wait", "timed_wait", "communicate", "timed_communicate"])
@pytest.mark.parametrize("event", ["exit7", "echild", "eio"])
def test_concurrent_callback_poll_timeout_and_gc_never_compete_with_owner(owners, waiter, event):
    results = []
    for cls in (subprocess.Popen, usage.UsagePopen):
        boundary = owners(cls, event)
        boundary.entered, boundary.release = threading.Event(), threading.Event()
        captured = []

        def run_owner():
            captured.append(outcome(boundary, waiter))

        thread = threading.Thread(target=run_owner, daemon=True)
        thread.start()
        try:
            assert boundary.entered.wait(2), "owner failed to reach its locked syscall"
            assert boundary.process.poll() is None
            assert boundary.process._internal_poll(_deadstate=sys.maxsize) is None
            with pytest.raises(subprocess.TimeoutExpired):
                boundary.process.wait(timeout=0)
            assert len(boundary.calls) == 1
            if cls is usage.UsagePopen:
                assert_unknown(boundary.process, "reap_in_progress")
                assert len(boundary.calls) == 1
        finally:
            boundary.release.set()
            thread.join(3)
        assert not thread.is_alive()
        assert len(captured) == 1
        assert not boundary.original_lock.locked()
        results.append(captured[0])
        if event == "eio":
            assert boundary.process.returncode is None
            if cls is usage.UsagePopen:
                assert_unknown(boundary.process, "not_reaped")
            # A source error is not a terminal observation. A later genuine reap
            # by this SAME owner may legitimately publish usage, without zeroing.
            boundary.entered = boundary.release = None
            boundary.event = "exit7"
            assert invoke(boundary, "timed_wait") == 7
        else:
            code = 7 if event == "exit7" else 0
            assert invoke(boundary, "gc_poll") == invoke(boundary, "timed_wait") == code
            assert len(boundary.calls) == 1
            if cls is usage.UsagePopen and event == "echild":
                assert_unknown(boundary.process, "exit_usage_unavailable")
    literal = ("raise", OSError, errno.EIO) if event == "eio" else ("return", 7 if event == "exit7" else 0)
    assert results == [literal, literal]


@pytest.mark.parametrize("event", ["exit7", "echild"])
def test_resource_metric_labels_keep_reaped_cpu_and_nonaggregate_hwm_scope(owners, event):
    observed = owners(usage.UsagePopen, event)
    invoke(observed, "wait")
    record = observed.process.final_usage()
    report = {"configuration": {"roles": ["worker_a"]}, "provider_lifetime_usage": {
        "schema": "pllm.provider_lifetime_usage.v1", "roles": {"worker_a": record}}}
    metrics = []
    sink = SimpleNamespace(metric=lambda name, value, **labels: metrics.append((name, value, labels)))
    usage.provider_lifetime_metrics(report, sink)
    assert len(metrics) == 2
    assert metrics[0][0] == "worker_a.lifetime_cpu"
    assert metrics[0][2]["scope"] == "wait4_user_system_including_reaped_descendants"
    assert metrics[1][0] == "worker_a.lifetime_max_individual_rss"
    assert metrics[1][2]["scope"] == "os_maximum_individual_rss_not_simultaneous_provider_tree_peak"
    for _, _, labels in metrics:
        assert labels["window"] == "role_birth_through_exit"
    assert [entry[1] for entry in metrics] == (
        [record["cpu_seconds"], record["peak_rss_bytes"]] if event == "exit7" else [None, None])
    assert len(observed.calls) == 1
