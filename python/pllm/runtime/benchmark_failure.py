"""Bounded public-only failure observations, never execution or cause evidence."""
from __future__ import annotations

from copy import deepcopy
import math
import threading
import time


SCHEMA = "pllm.benchmark_failure_diagnostics.v1"
ROLES = ("client", "inference", "preparation", "worker_a", "worker_b", "worker_c",
         "dealer", "boundary_a", "boundary_b")
SOURCE_IMPORT_PHASES = (
    "qualification_metadata", "qualification_import", "qualification_seal",
    "source_resolution", "source_config", "source_geometry", "source_inventory",
    "source_lock", "source_contract", "source_headers", "source_tokenizer",
    "source_weights", "source_rotary", "source_recheck", "source_manifest",
)
PHASES = ("unknown", "dashboard_startup", "memory_preflight", "source_inspection",
          "source_qualification", "qualification_pricing", "qualification_retirement",
          "graph_admission", "role_startup", "client_startup",
          "ready", "request", *SOURCE_IMPORT_PHASES)
_OBSERVE = object()


def _count(value):
    return value if type(value) is int and 0 <= value <= (1 << 63) - 1 else None


def _seconds(value):
    return (value if type(value) in (int, float) and 0 <= value <= 31_536_000
            and math.isfinite(value) else None)


def _tag(value, allowed, default=None):
    return value if type(value) is str and value in allowed else default


def _roles(value):
    # Never walk an unbounded caller-supplied list or serialize unknown role names.
    if type(value) not in (list, tuple) or len(value) > len(ROLES):
        return []
    return [role for role in ROLES if role in value]


def sanitize_failure_diagnostics(value):
    """Rebuild a fixed, <4 KiB record at each exception/IPC/output boundary."""
    if type(value) is not dict or value.get("schema_version") != SCHEMA:
        return None
    host = value.get("host")
    host = host if type(host) is dict else {}
    rss = value.get("process_rss")
    rss = rss if type(rss) is dict else {}
    owners = rss.get("roles")
    owners = owners if type(owners) is dict else {}
    qualification = value.get("qualification")
    qualification = qualification if type(qualification) is dict else {}
    role_rss = {}
    for role in ROLES:
        record = owners.get(role)
        if type(record) is dict:
            role_rss[role] = {key: _count(record.get(key)) for key in
                              ("latest_bytes", "peak_bytes")}
    return {
        "schema_version": SCHEMA,
        "phase": _tag(value.get("phase"), PHASES, "unknown"),
        "role": _tag(value.get("role"), ROLES),
        "elapsed_seconds": _seconds(value.get("elapsed_seconds")),
        "qualification": {key: qualification.get(key) if type(qualification.get(key)) is bool else None
                          for key in ("priced", "completed", "retired")},
        "spawned_roles": _roles(value.get("spawned_roles")),
        "ready_roles": _roles(value.get("ready_roles")),
        "host": {"pressure": _tag(host.get("pressure"), ("normal", "warning", "critical")),
                 "available_bytes": _count(host.get("available_bytes")),
                 "swap_used_bytes": _count(host.get("swap_used_bytes"))},
        "process_rss": {
            "latest_total_bytes": _count(rss.get("latest_total_bytes")),
            "peak_total_bytes": _count(rss.get("peak_total_bytes")),
            "sample_age_seconds": _seconds(rss.get("sample_age_seconds")),
            "sampling_incomplete": rss.get("sampling_incomplete") is not False,
            "roles": role_rss,
        },
    }


class FailureDiagnostics:
    """One invocation, first failure wins. No new memory/process observation calls."""

    def __init__(self, runtime, sampler=None):
        self.runtime, self.sampler = runtime, sampler
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._snapshot = None

    def observe(self, *, phase=None):
        """Nonblocking, unfrozen observation for a possible final-check failure.

        Contended state is unavailable, not evidence that no roles were launched.
        This does not reserve the first-failure slot or perform any new sampling.
        """
        if self._snapshot is not None:
            return deepcopy(self._snapshot)
        runtime = self.runtime
        topology = runtime._topology
        breadcrumb = getattr(topology, "startup_breadcrumbs", {})
        source_import = breadcrumb.get("source_import")
        source_import = source_import if type(source_import) is dict else {}
        importing = (source_import.get("active") is True
                     and type(source_import.get("phase")) is str
                     and source_import.get("phase") in SOURCE_IMPORT_PHASES
                     and type(source_import.get("role")) is str
                     and source_import.get("role") in ("client", "worker_a", "worker_b", "dealer"))
        current_role = breadcrumb.get("role")
        if not runtime._lock.acquire(blocking=False):
            current_phase = "unknown"
        else:
            try:
                state = runtime._state
                if importing:
                    current_phase = source_import["phase"]
                    current_role = source_import["role"]
                elif runtime._active_run is not None or state.get("phase") == "online":
                    current_phase = "request"
                elif state.get("phase") == "ready":
                    current_phase = "ready"
                elif state.get("phase") == "error" and runtime._has_started_run:
                    current_phase = "request"
                elif breadcrumb.get("phase") in PHASES:
                    current_phase = breadcrumb["phase"]
                elif state.get("startup_step") == "memory preflight":
                    current_phase = "memory_preflight"
                else:
                    current_phase = "dashboard_startup" if topology is None else "unknown"
            finally:
                runtime._lock.release()
        if phase in PHASES:
            current_phase = phase
        guard = runtime._memory_guard
        return sanitize_failure_diagnostics({
            "schema_version": SCHEMA, "phase": current_phase,
            "role": current_role if current_phase not in ("unknown", "request", "ready") else None,
            "elapsed_seconds": time.monotonic() - self.started,
            "qualification": breadcrumb.get("qualification"),
            "spawned_roles": breadcrumb.get("spawned_roles", []),
            "ready_roles": breadcrumb.get("ready_roles", []),
            "host": getattr(guard, "failure_observation", None),
            "process_rss": self.sampler.failure_snapshot() if self.sampler is not None else None,
        })

    def capture(self, *, phase=None, observation=_OBSERVE):
        # A concurrent capture must not delay cancellation or cleanup either.
        if not self._lock.acquire(blocking=False):
            return deepcopy(self._snapshot)
        try:
            if self._snapshot is None:
                self._snapshot = (self.observe(phase=phase) if observation is _OBSERVE
                                  else sanitize_failure_diagnostics(observation))
            return deepcopy(self._snapshot)
        finally:
            self._lock.release()


def capture_failure(runtime, *, phase=None, observation=_OBSERVE):
    """Observe before abort cleanup; diagnostics must never interfere with guards."""
    collector = getattr(runtime, "_benchmark_failure_diagnostics", None)
    if collector is not None:
        try:
            return collector.capture(phase=phase, observation=observation)
        except Exception:
            return None
    return None


def observe_failure(runtime):
    """Retain a fallback without claiming a failure; never interfere with cleanup."""
    collector = getattr(runtime, "_benchmark_failure_diagnostics", None)
    if collector is not None:
        try:
            return collector.observe()
        except Exception:
            return None
    return None
