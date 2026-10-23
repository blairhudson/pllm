"""Final usage at the isolated client's existing process-owner wait site.

The multiprocessing subclass changes only spawn Popen.poll (wait/join/cleanup
already call it). No global syscall patch, new waiter, watcher, PID traversal or
exit-time RSS sampling. CPython's spawn/fd finalizers and timeout behavior remain
inherited. Darwin qualification only; other platforms/implementations stay unknown.

wait4 CPU includes reaped descendants. HWM is a maximum individual in that reaped
family, NOT self-only, simultaneously-live RSS SUM, physical RAM or private RAM.
Complete client-role membership permits individual prefix HWM as a lower bound
and sum of final family maxima as an upper bound. Stronger closed-family ownership
permits max(family maxima) as lower bound too. Neither assumes self-only wait4.
"""
from __future__ import annotations

import errno
import math
import multiprocessing.context as _context
import os
import sys
import threading

from .process_usage import UsagePopen, _POPEN

if os.name == "posix":
    from multiprocessing.popen_fork import Popen as _ForkPopen
    from multiprocessing.popen_spawn_posix import Popen as _SpawnPopen
else:
    _ForkPopen = _SpawnPopen = object

SCHEMA = "pllm.client_process_exit_usage.v1"
WINDOW = "isolated_client_group_birth_through_complete_retirement"
MAX_MEMBERS = 256
_POLL = getattr(_ForkPopen, "poll", None)
_WAIT = getattr(_ForkPopen, "wait", None)


def _slot(name):
    return name == "client" or any(name == f"client_helper_{i}" for i in range(1, MAX_MEMBERS))


def _normal_family_exit(record):
    return (type(record) is dict and record.get("source") == "wait4"
        and record.get("coverage") == "os_process_exit" and record.get("normal_exit") is True
        and type(record.get("exit_code")) is int and record["exit_code"] == 0
        and type(record.get("peak_rss_bytes")) is int and 0 <= record["peak_rss_bytes"] <= (1 << 63) - 1
        and record.get("window") == "process_birth_through_exit" and record.get("raw_hwm_unit") == "bytes"
        and record.get("hwm_scope") == "maximum_individual_in_reaped_family_not_self_or_group_sum")


def _birth_identity(pid):
    # One creation-owner identity read, never an exit sampler/discovery pass.
    import psutil
    try:
        return pid, psutil.Process(pid).create_time()
    except (psutil.Error, OSError, ValueError):
        return None


class ClientUsagePopen(UsagePopen):
    """SDK's original locked raw reaper, plus one private creation birth record."""
    def __init__(self, *args, **kwargs):
        if not supported():
            raise RuntimeError("unqualified client usage owner")
        super().__init__(*args, **kwargs)
        self._usage_identity = _birth_identity(self.pid)


def supported():
    # These minor versions share the inspected single waitpid poll site. Never
    # silently adopt another runtime's reaping/GC contract or another HWM unit.
    return (sys.implementation.name == "cpython" and (3, 11) <= sys.version_info[:2] <= (3, 13)
        and sys.platform == "darwin" and os.name == "posix" and hasattr(os, "wait4")
        and _SpawnPopen.poll is _POLL and _SpawnPopen.wait is _WAIT
        and getattr(_POLL, "__module__", None) == "multiprocessing.popen_fork")


def unknown(reason):
    return {"schema": SCHEMA, "coverage": "unknown", "reason": reason,
        "source": None, "exit_code": None, "normal_exit": None,
        "user_cpu_seconds": None, "system_cpu_seconds": None, "cpu_seconds": None,
        "peak_rss_bytes": None, "cpu_scope": "including_reaped_descendants",
        "hwm_scope": "maximum_individual_in_reaped_family_not_self_or_group_sum",
        "window": "process_birth_through_exit", "raw_hwm_unit": None}


def _observed(code, values):
    if values is None:
        return unknown("exit_usage_unavailable")
    user, system, peak = values
    if (type(code) is not int
            or any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in (user, system))
            or not math.isfinite(user + system)
            or type(peak) is not int or not 0 <= peak <= (1 << 63) - 1):
        return unknown("invalid_os_usage")
    return {**unknown(None), "coverage": "os_process_exit", "source": "wait4",
        "exit_code": code, "normal_exit": code == 0, "user_cpu_seconds": user,
        "system_cpu_seconds": system, "cpu_seconds": user + system,
        "peak_rss_bytes": peak, "raw_hwm_unit": "bytes"}


class UsageSpawnPopen(_SpawnPopen):
    """Same multiprocessing owner, one locked poll syscall, inherited wait/finalize.

    Unlike subprocess.Popen, multiprocessing preserves returncode=None on ECHILD
    and all other OSErrors, including cleanup/GC paths. No fabricated zero/sentinel.
    Nonblocking poll/readers never compete with a blocking join's reap.
    """

    def __init__(self, process_obj):
        if not supported():
            raise RuntimeError("unqualified multiprocessing usage owner")
        self._usage_lock = threading.Lock()
        self._usage_wait4 = os.wait4
        self._usage_values = None
        self._usage_reason = "not_reaped"
        super().__init__(process_obj)
        self._usage_identity = _birth_identity(self.pid)

    def poll(self, flag=getattr(os, "WNOHANG", 1)):
        if self.returncode is not None:
            return self.returncode
        if not self._usage_lock.acquire(blocking=not bool(flag & os.WNOHANG)):
            return self.returncode
        try:
            if self.returncode is None:
                try:
                    pid, status, usage = self._usage_wait4(self.pid, flag)
                except OSError as error:
                    self._usage_reason = ("external_reap_or_sigchld_ignored" if error.errno == errno.ECHILD
                                          else "wait4_os_error")
                    return None  # Literal multiprocessing behavior, not Popen's zero.
                if pid == self.pid:
                    # Match original conversion (including ValueError for a
                    # stopped/traced status); do not cache that status as final.
                    code = os.waitstatus_to_exitcode(status)
                    if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                        self._usage_values = (usage.ru_utime, usage.ru_stime, usage.ru_maxrss)
                        self._usage_reason = None
                    self.returncode = code
            return self.returncode
        finally:
            self._usage_lock.release()

    def final_usage(self):
        """Cache snapshot only: no poll, wait, Process.exitcode or live PID read."""
        if not self._usage_lock.acquire(False):
            return unknown("reap_in_progress")
        try:
            if self.returncode is None:
                return unknown(self._usage_reason)
            return _observed(self.returncode, self._usage_values)
        finally:
            self._usage_lock.release()


class UsageSpawnProcess(_context.SpawnProcess):
    @staticmethod
    def _Popen(process_obj):
        return UsageSpawnPopen(process_obj)


def client_process_factory(factory):
    """Change only this launcher's exact spawn owner; foreign factories stay foreign."""
    return UsageSpawnProcess if supported() and factory is _context.SpawnProcess else factory


def final_client_usage(process):
    owner = getattr(process, "_popen", None)
    return owner.final_usage() if type(owner) is UsageSpawnPopen else unknown("unqualified_process_owner")


def client_popen_factory(factory):
    """SDK worker already has a raw Popen owner; reuse its existing locked reaper."""
    return (ClientUsagePopen if supported() and factory is _POPEN
            and getattr(factory, "__module__", None) == "subprocess" else factory)


def final_member_usage(process):
    if type(process) in (UsagePopen, ClientUsagePopen) and supported():
        cached = process.final_usage()
        return (_observed(cached["exit_code"], (cached["user_cpu_seconds"], cached["system_cpu_seconds"],
                cached["peak_rss_bytes"])) if cached["coverage"] == "os_process_exit"
                else unknown(cached["reason"]))
    return final_client_usage(process)


def finalize_client_report(report, process, *, forced=False, workload_completed=True):
    """After supervisor/source teardown AND original parent join; no new observation.

    Adds optional fields only, never rewrites legacy sample/process/CPU metrics or
    makes failed/cancelled work valid. A normal response can retain unknown cost.
    Fixed <=256 member records use existing coarse process metadata allowances;
    no admission credit, cap increase, history, payload capture or exact heap bound.
    """
    usage = final_member_usage(process)
    report["client_process_exit_usage"] = usage
    scope = report.get("client_owned_memory")
    if type(scope) is not dict:
        return report
    manifest = scope.get("lifetime_owner", {})
    bounds = {"lower_bound_bytes": None, "upper_bound_bytes": None,
        "reason": "complete_isolated_lifetime_owner_coverage_unavailable"}
    scope["lifetime_peak_rss_sum_interval"] = bounds
    if type(manifest) is dict:
        # Consume transient root identity, do not retain PID/birth history in the
        # saved report. Same pipe/worker result is tied to its actual owner here.
        identity = manifest.pop("_root_identity", None)
        owner = process if type(process) is ClientUsagePopen else getattr(process, "_popen", None)
        captured = getattr(owner, "_usage_identity", None)
        manifest["root_birth_bound"] = (captured is not None and type(identity) in (tuple, list)
            and tuple(identity) == captured and captured[0] == getattr(owner, "pid", None))
    if forced or not workload_completed or usage["normal_exit"] is not True:
        bounds["reason"] = ("client_forced_before_full_retirement" if forced else
            "workload_not_completed" if not workload_completed else "client_exit_not_normal_or_unobserved")
        return report
    if type(manifest) is not dict:
        return report
    members = manifest.get("members")
    if (report.get("checks", {}).get("passed") is not True
            or manifest.get("schema") != "pllm.client_lifetime_owner.v1"
            or manifest.get("membership_complete") is not True
            or manifest.get("window") != WINDOW or manifest.get("root_birth_bound") is not True
            or type(members) is not dict or not 1 <= len(members) <= MAX_MEMBERS or "client" not in members
            or any(not _slot(role) for role in members)):
        return report
    maxima = {"client": usage["peak_rss_bytes"]}
    members["client"] = dict(usage)
    for role, record in members.items():
        if role == "client":
            continue
        if not _normal_family_exit(record):
            bounds["reason"] = "member_final_usage_unavailable"
            return report
        maxima[role] = record["peak_rss_bytes"]
    closed_family = manifest.get("closed_client_owned_family") is True
    prefix = scope.get("individual_client_hwm", {})
    if not closed_family and not _valid_prefix(prefix, usage["peak_rss_bytes"]):
        bounds["reason"] = "individual_prefix_lower_bound_unavailable"
        return report
    lower = max(maxima.values()) if closed_family else prefix["bytes"]
    bounds.update(lower_bound_bytes=lower, upper_bound_bytes=sum(maxima.values()),
        reason=None, source=("complete_owned_member_wait4_family_maxima" if closed_family
            else "owner_bound_individual_prefix_lower_and_wait4_family_upper"), unit="bytes",
        metric="client_owned_rss_sum_peak", window=WINDOW, isolated_lifetime=True,
        membership_complete=True, finalized_after_full_member_retirement=True,
        hwm_coverage="maximum_individual_in_each_complete_owned_family_not_self_only",
        member_family_max_hwm_bytes=maxima)
    if not closed_family:
        bounds.update(lower_bound_source="root_RUSAGE_SELF_prefix_within_isolated_lifetime",
            lower_observation_window=prefix["window"])
    return report


def _valid_prefix(prefix, upper):
    return (type(prefix) is dict and type(prefix.get("bytes")) is int and 0 <= prefix["bytes"] <= upper
        and prefix.get("source") == "resource.getrusage(RUSAGE_SELF)"
        and prefix.get("window") == "client_process_birth_through_report_endpoint"
        and prefix.get("coverage") == "individual_process_only_not_client_group")


def owned_family_interval(scope):
    """Read the NEW family-max contract; old final-individual-HWM contract is separate."""
    manifest = scope.get("lifetime_owner", {})
    bounds = scope.get("lifetime_peak_rss_sum_interval", {})
    members, maxima = manifest.get("members"), bounds.get("member_family_max_hwm_bytes")
    closed_family = bounds.get("source") == "complete_owned_member_wait4_family_maxima"
    if (manifest.get("schema") != "pllm.client_lifetime_owner.v1"
            or manifest.get("membership_complete") is not True or manifest.get("root_birth_bound") is not True
            or (closed_family and manifest.get("closed_client_owned_family") is not True)
            or manifest.get("reason") is not None
            or manifest.get("window") != WINDOW or type(members) is not dict
            or not 1 <= len(members) <= MAX_MEMBERS or "client" not in members
            or any(not _slot(name) for name in members)
            or bounds.get("source") not in {"complete_owned_member_wait4_family_maxima",
                "owner_bound_individual_prefix_lower_and_wait4_family_upper"}
            or bounds.get("window") != WINDOW or bounds.get("metric") != "client_owned_rss_sum_peak"
            or bounds.get("unit") != "bytes" or bounds.get("reason") is not None
            or bounds.get("hwm_coverage") != "maximum_individual_in_each_complete_owned_family_not_self_only"
            or bounds.get("isolated_lifetime") is not True or bounds.get("membership_complete") is not True
            or bounds.get("finalized_after_full_member_retirement") is not True
            or type(maxima) is not dict or set(maxima) != set(members)):
        return None
    for name, value in maxima.items():
        record = members[name]
        if type(value) is not int or not _normal_family_exit(record) or record.get("peak_rss_bytes") != value:
            return None
    prefix = scope.get("individual_client_hwm", {})
    if not closed_family and (not _valid_prefix(prefix, maxima["client"])
            or bounds.get("lower_bound_source") != "root_RUSAGE_SELF_prefix_within_isolated_lifetime"
            or bounds.get("lower_observation_window") != prefix["window"]):
        return None
    lower = max(maxima.values()) if closed_family else prefix["bytes"]
    if bounds.get("lower_bound_bytes") != lower or bounds.get("upper_bound_bytes") != sum(maxima.values()):
        return None
    return {"lower": bounds["lower_bound_bytes"], "upper": bounds["upper_bound_bytes"]}


def finalize_client_payload(payload, process):
    """SDK worker JSON is received before reap; mutate optional report only after it."""
    report = payload.get("details", {}).get("report") if type(payload) is dict else None
    if type(report) is dict:
        scope = report.get("client_owned_memory", {})
        manifest = scope.get("lifetime_owner", {}) if type(scope) is dict else None
        if type(manifest) is dict:
            # SDK main probes hardware after the response/report endpoint. Those
            # bookkeeping subprocesses are outside the client-role declaration;
            # never treat its whole wait4 family maximum as a client lower bound.
            manifest["closed_client_owned_family"] = False
        finalize_client_report(report, process, workload_completed=payload.get("status") == "completed")
