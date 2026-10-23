"""Final OS usage from the existing local provider's sole Popen reaper.

Never attach this owner to multiprocessing or asyncio children. Their process
owners/watchers already reap them. There is no wait-any, watcher, sample history,
HTTP request, child-side snapshot, or new execution entry point here.

wait4 user/system CPU includes descendants reaped into the child's usage. Its
ru_maxrss is an OS-reported maximum individual resident set, not the sum of
simultaneously live helpers. Neither field establishes complete provider-tree
coverage: unawaited descendants and independently hosted roles remain unknown.
"""
from __future__ import annotations

import errno
import math
import os
import subprocess
import sys


_POPEN = subprocess.Popen
_PROVIDERS = ("inference", "preparation", "worker_a", "worker_b", "worker_c",
              "dealer", "boundary_a", "boundary_b")


class UsagePopen(_POPEN):
    """POSIX Popen with wait4 in BOTH of its existing locked wait paths.

    CPython 3.11–3.13's _wait uses _try_wait; poll/send_signal/__del__ instead
    use _internal_poll. Both must share Popen's _waitpid_lock, otherwise an early
    health poll could discard usage or race the shutdown waiter. No extra reaper.
    Only the ordinary raw-Popen local role launcher constructs this subclass.
    """

    def __init__(self, *args, **kwargs):
        # Bind the syscall for Popen finalization as well as ordinary shutdown.
        self._usage_wait4 = os.wait4
        self._usage_scale = {"darwin": 1, "linux": 1024}.get(sys.platform)
        self._usage_final = None
        self._usage_unavailable = False
        super().__init__(*args, **kwargs)

    def _try_wait(self, wait_flags, _is_exit=getattr(os, "WIFEXITED", None),
                  _is_signaled=getattr(os, "WIFSIGNALED", None), *, _deadstate=None):
        """Called only with Popen's original wait lock held."""
        try:
            pid, status, usage = self._usage_wait4(self.pid, wait_flags)
        except ChildProcessError:
            # Match Popen's missing-status behavior, NOT a successful observation.
            # SIGCHLD=SIG_IGN or an outside reaper makes final usage unknowable.
            self._usage_unavailable = True
            if _deadstate is not None:
                # __del__ supplies Popen's GC sentinel, unlike ordinary wait/poll.
                # Let _internal_poll's existing OSError path preserve that status.
                raise
            return self.pid, 0
        if pid == self.pid:
            if self._usage_scale is None:
                # wait4 exists elsewhere, but its ru_maxrss units are not qualified.
                self._usage_unavailable = True
            elif _is_exit(status) or _is_signaled(status):
                # Retain one fixed record, including zero observations; no PID/argv.
                self._usage_final = (usage.ru_utime, usage.ru_stime,
                                     usage.ru_maxrss * self._usage_scale)
            else:
                # A traced/stopped child is not a post-exit observation, even if
                # the underlying Popen exposes a returncode for that status.
                self._usage_unavailable = True
        return pid, status

    def _internal_poll(self, _deadstate=None, _nohang=getattr(os, "WNOHANG", 1),
                       _echild=errno.ECHILD):
        # Preserve Popen's nonblocking poll and destructor semantics. _wait uses
        # the same lock and _try_wait, so status and usage become visible together.
        if self.returncode is None:
            if not self._waitpid_lock.acquire(False):
                return None
            try:
                if self.returncode is not None:
                    return self.returncode
                pid, status = self._try_wait(_nohang, _deadstate=_deadstate)
                if pid == self.pid:
                    self._handle_exitstatus(status)
            except OSError as error:
                if _deadstate is not None:
                    self._usage_unavailable = True
                    self.returncode = _deadstate
                elif error.errno == _echild:
                    self._usage_unavailable = True
                    self.returncode = 0
            finally:
                self._waitpid_lock.release()
        return self.returncode

    def final_usage(self):
        """Read cached OS exit usage only; never poll/reap or observe a live PID."""
        if not self._waitpid_lock.acquire(False):
            return _unknown("reap_in_progress")
        try:
            code, values = self.returncode, self._usage_final
            missing = self._usage_unavailable
        finally:
            self._waitpid_lock.release()
        if self._usage_scale is None:
            return _unknown("unsupported_platform")
        if code is None:
            return _unknown("not_reaped")
        if missing or values is None:
            return _unknown("exit_usage_unavailable")
        user, system, peak = values
        if (any(type(n) not in (int, float) or not math.isfinite(n) or n < 0
                for n in (user, system)) or type(peak) is not int or peak < 0):
            return _unknown("invalid_os_usage")
        return {"coverage": "os_process_exit", "reason": None, "exit_code": code,
                "user_cpu_seconds": user, "system_cpu_seconds": system,
                "cpu_seconds": user + system, "peak_rss_bytes": peak,
                "source": "wait4", "normal_exit": code == 0}


def _unknown(reason):
    return {"coverage": "unknown", "reason": reason, "exit_code": None,
            "user_cpu_seconds": None, "system_cpu_seconds": None,
            "cpu_seconds": None, "peak_rss_bytes": None,
            "source": None, "normal_exit": None}


def provider_popen_factory(factory):
    """Extend only our raw synchronous Popen owner; foreign launchers stay foreign."""
    return (UsagePopen if factory is _POPEN and os.name == "posix"
            and sys.platform in {"darwin", "linux"}
            and hasattr(os, "wait4") else factory)


def provider_lifetime_metrics(report, measurement):
    """Optional new metrics; never overwrite cold-first or sampled-RSS identities.

    Historical reports without this section emit exactly their original metrics.
    Availability is independent of protocol success: ordinary response checks must
    still pass before the SDK's response executor invokes its metric adapter.
    At most sixteen fixed metric rows share the existing coarse client-process
    allowance; they grant no budget credit or new exact allocator bound.
    """
    section = report.get("provider_lifetime_usage")
    if not isinstance(section, dict) or section.get("schema") != "pllm.provider_lifetime_usage.v1":
        return
    for role in _PROVIDERS:
        if role not in report["configuration"]["roles"]:
            continue
        record = section["roles"].get(role, _unknown("owned_exit_usage_unavailable"))
        observed = record["coverage"] == "os_process_exit" and record["source"] == "wait4"
        for suffix, field, kind, scope in (
            ("lifetime_cpu", "cpu_seconds", "cpu", "wait4_user_system_including_reaped_descendants"),
            ("lifetime_max_individual_rss", "peak_rss_bytes", "memory",
             "os_maximum_individual_rss_not_simultaneous_provider_tree_peak"),
        ):
            measurement.metric(f"{role}.{suffix}", record[field] if observed else None,
                kind=kind, role=role, window="role_birth_through_exit", scope=scope,
                statistic="measurement", reason=record["reason"])


def provider_lifetime_usage(topology, roles, *, local_cpu_loopback, source_present):
    """Post-teardown report; no process observation or monitoring on failure paths.

    Fixed metadata: at most eight per-Popen tuples, eight report records and sixteen
    optional SDK metric rows, no payload/history. Charge these to the existing
    coarse 256 MiB Python process
    allowances in benchmark_memory, without shrinking any owner/reserve or claiming
    a new exact Python/allocator bound. They persist through reap/report creation.
    """
    if (type(roles) not in (tuple, list) or len(roles) > 9
            or any(type(role) is not str or role not in ("client", *_PROVIDERS)
                   for role in roles) or len(set(roles)) != len(roles)):
        raise ValueError("unsupported bounded provider role set")
    expected = tuple(role for role in _PROVIDERS if role in roles)
    processes = {}
    # A foreign/network/container topology cannot supply authority via PID tags.
    if local_cpu_loopback and source_present and topology is not None:
        from .servers import LocalTopology
        if type(topology) is LocalTopology:
            with topology._process_lock:
                processes = {role: topology._processes.get(role) for role in expected}
    records = {}
    for role in expected:
        process = processes.get(role)
        if not local_cpu_loopback:
            record = _unknown("outside_local_cpu_loopback")
        elif not source_present:
            record = _unknown("source_identity_unavailable")
        elif type(process) is UsagePopen:
            record = process.final_usage()
        else:
            record = _unknown("owned_exit_usage_unavailable")
        records[role] = record
    return {"schema": "pllm.provider_lifetime_usage.v1", "roles": records,
            "complete": bool(expected) and all(
                record["coverage"] == "os_process_exit" for record in records.values()),
            "scope": "source-present local CPU loopback provider OS process birth through exit",
            "cpu_scope": "wait4 user+system CPU including reaped descendants",
            "peak_rss_scope": "OS maximum individual RSS; not simultaneous provider-tree peak",
            "aggregate_peak_rss_bytes": None,
            "limitations": [
                "OS exit covers executed cleanup, not proof of successful native/source/job retirement.",
                "Forced termination is final OS usage, not clean protocol completion.",
                "Per-process OS HWM is not simultaneous topology peak; shared pages may count twice.",
                "CPU includes reaped descendants; RSS is an individual maximum, not their simultaneous sum.",
                "Unreaped descendants remain outside this observation; no complete provider-tree coverage.",
                "No client, remote, container, GPU or OS-file-cache coverage.",
                "Failed benchmark invocations do not publish this normal-report section.",
                "Fixed metadata uses existing coarse process allowances; no cap reduction or exact heap proof.",
            ]}
