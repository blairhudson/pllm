"""Bounded public-only failure observations, never execution or cause evidence."""
from __future__ import annotations

from copy import deepcopy
import math
import threading
import time
import weakref


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
REQUEST_FAILURE_STAGES = ("input_count", "prepare_prompt", "validate_source", "bind_encode", "transport", "accept_result")
_REQUEST_FAILURE_FIELDS = ("stage", "role", "code", "http_status")


def sanitize_request_failure_context(value):
    """Only fixed public request labels; no exception text or inferred HTTP status."""
    if (type(value) is not dict or len(value) != len(_REQUEST_FAILURE_FIELDS)
            or any(type(key) is not str or key not in _REQUEST_FAILURE_FIELDS for key in value)):
        return None
    stage, role, code, status = (value[key] for key in _REQUEST_FAILURE_FIELDS)
    if (type(stage) is not str or stage not in REQUEST_FAILURE_STAGES
            or type(role) is not str or role not in ("client", "worker_a", "worker_b")
            or type(code) is not str or code not in (
                "gkm_stage_failed", "worker_http_rejected", "count_owner_busy", "input_count_failed")
            or (status is not None and (type(status) is not int or not 100 <= status <= 599))):
        return None
    if code == "worker_http_rejected":
        if stage != "transport" or role == "client" or status is None:
            return None
    elif role != "client" or status is not None:
        return None
    elif (stage == "input_count") != (code in ("count_owner_busy", "input_count_failed")):
        return None
    return dict(zip(_REQUEST_FAILURE_FIELDS, (stage, role, code, status)))


def merge_failure_diagnostics(primary, fallback):
    """Keep first observations, filling only absent bounded optional context."""
    primary = sanitize_failure_diagnostics(primary)
    fallback = sanitize_failure_diagnostics(fallback)
    if primary is None:
        return fallback
    if fallback is not None:
        for key in ("request_failure_context", "fss_control_boundary", "fss_execution_progress"):
            if key not in primary and key in fallback:
                primary[key] = fallback[key]
    return primary


def _request_failure_context(error):
    if error is None:
        return None
    from .client import ProtocolError

    if isinstance(error, ProtocolError):
        return sanitize_request_failure_context(getattr(error, "request_failure_context", None))
    return None


# One immutable tuple, not an event history. Intended charging is the existing
# client HTTP control overlap (8 * 16 KiB control slots beyond admission bodies)
# and supervisor's 1 MiB reader/cache/control allowance. These are coarse shared
# allowances, not a proved aggregate Python/allocator/RSS peak or free credits.
# Transport creates markers inside its existing client lease; no body, thread
# or queue is added. Neither object-size tests nor the serialization bound grant
# native Budget authority, relay credit, a new reservation or a changed cap.
FSS_CONTROL_ACTIONS = {
    "client": ("contract", "channels", "client_request", "input_share", "reconstruct", "retire"),
    "worker_a": ("manifest", "admit", "bind", "input", "start", "finish"),
    "worker_b": ("manifest", "admit", "bind", "input", "start", "finish"),
    "dealer": ("manifest", "admit", "bind", "prepare", "finish"),
}
FSS_CONTROL_STATUSES = ("entered", "validated", "failed")
_FSS_CONTROL_FIELDS = ("role", "action", "status", "elapsed_seconds")


def sanitize_fss_control_boundary(value):
    """Bounded exact builtins only; never stringify or compare hostile tags."""
    if type(value) is tuple and len(value) == 4:
        role, action, status, elapsed = value
    elif type(value) is dict and len(value) == 4:
        # Validate four keys before lookup: a colliding custom key must not get
        # an opportunity to execute __eq__ during dict.get().
        if any(type(key) is not str or len(key) > 15 or key not in _FSS_CONTROL_FIELDS
               for key in value):
            return None
        role, action, status, elapsed = (value.get(key) for key in _FSS_CONTROL_FIELDS)
    else:
        return None
    if (type(role) is not str or len(role) > 8 or role not in FSS_CONTROL_ACTIONS
            or type(action) is not str or len(action) > 14 or action not in FSS_CONTROL_ACTIONS[role]
            or type(status) is not str or len(status) > 9 or status not in FSS_CONTROL_STATUSES
            or _seconds(elapsed) is None):
        return None
    return {"role": role, "action": action, "status": status, "elapsed_seconds": elapsed}


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


def sanitize_execution_progress(value):
    """Two latest holder gauges, fixed public scalars; never stringify tags."""
    if type(value) is not dict or len(value) > 2:
        return None
    fields = {"stage", "kernel", "pc", "frame", "materials_returned", "peer_acked",
              "peer_received", "elapsed_seconds", "changed_seconds", "cpu_ns", "sample_age_seconds"}
    if any(type(k) is not str or k not in ("worker_a", "worker_b") for k in value):
        return None
    result = {}
    for role, record in value.items():
        if (type(record) is not dict or len(record) != len(fields)
                or any(type(k) is not str or k not in fields for k in record)):
            continue
        # Local import avoids the source-status -> failure-schema import cycle.
        from .fss_child_status import PROGRESS_STAGES
        stage = _tag(record["stage"], PROGRESS_STAGES)
        if stage is None:
            continue
        result[role] = {key: (_seconds(record[key]) if key.endswith("seconds") else _count(record[key]))
                        for key in fields - {"stage"}}
        result[role]["stage"] = stage
    return result or None


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
    result = {
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
    boundary = sanitize_fss_control_boundary(value.get("fss_control_boundary"))
    if boundary is not None:
        result["fss_control_boundary"] = boundary
    progress = sanitize_execution_progress(value.get("fss_execution_progress"))
    if progress is not None:
        result["fss_execution_progress"] = progress
    context = sanitize_request_failure_context(value.get("request_failure_context"))
    if context is not None:
        result["request_failure_context"] = context
    return result


class FailureDiagnostics:
    """One invocation, first failure wins. No new memory/process observation calls."""

    def __init__(self, runtime, sampler=None):
        self.runtime, self.sampler = runtime, sampler
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._snapshot = None

    def bind_fss_client(self, core):
        """Post-construction optional hook; do not make client own this runtime."""
        if getattr(core, "runtime_id", None) != "compiled_fss_v1":
            return
        reference = weakref.ref(self)

        def failure(marker):
            collector = reference()
            if collector is not None:
                collector.capture(phase="request", control_boundary=marker)

        core._fss_failure_observer = failure

    def observe(self, *, phase=None, control_boundary=_OBSERVE):
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
        active_request = False
        client = None
        if not runtime._lock.acquire(blocking=False):
            current_phase = "unknown"
        else:
            try:
                state = runtime._state
                active_request = runtime._active_run is not None
                client = getattr(runtime, "_client", None)
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
        if control_boundary is _OBSERVE:
            control_boundary = None
            if active_request:
                # Runtime lock is already released. Never acquire the iterator's
                # execution lock, which is held across the whole request.
                try:
                    core = getattr(client, "_core", None)
                    if getattr(core, "runtime_id", None) == "compiled_fss_v1":
                        control_boundary = core._fss_control_snapshot()
                except BaseException:
                    pass  # Optional cached observation cannot change failure.
        guard = runtime._memory_guard
        # The current client marker is attempt-owned. Before holder start there
        # may still be a failed previous role generation in the reader cache.
        # Every path reaching holder start has completed all new source imports,
        # whose generation-begin records clear both old holder gauges.
        fresh_boundary = sanitize_fss_control_boundary(control_boundary)
        progress_eligible = (fresh_boundary is not None
            and fresh_boundary["role"] in ("worker_a", "worker_b")
            and fresh_boundary["action"] in ("start", "finish"))
        return sanitize_failure_diagnostics({
            "schema_version": SCHEMA, "phase": current_phase,
            "role": current_role if current_phase not in ("unknown", "request", "ready") else None,
            "elapsed_seconds": time.monotonic() - self.started,
            "qualification": breadcrumb.get("qualification"),
            "spawned_roles": breadcrumb.get("spawned_roles", []),
            "ready_roles": breadcrumb.get("ready_roles", []),
            "host": getattr(guard, "failure_observation", None),
            "process_rss": self.sampler.failure_snapshot() if self.sampler is not None else None,
            "fss_control_boundary": control_boundary,
            # Cached child data only. Startup/success cannot inherit an older
            # request's gauge; role reader also clears at each source generation.
            "fss_execution_progress": (breadcrumb.get("fss_execution_progress")
                                       if current_phase == "request" and progress_eligible else None),
        })

    def capture(self, *, phase=None, observation=_OBSERVE, control_boundary=_OBSERVE,
                request_failure_context=None):
        # A concurrent capture must not delay cancellation or cleanup either.
        if not self._lock.acquire(blocking=False):
            return deepcopy(self._snapshot)
        try:
            if self._snapshot is None:
                self._snapshot = (self.observe(phase=phase, control_boundary=control_boundary)
                                  if observation is _OBSERVE
                                  else sanitize_failure_diagnostics(observation))
                context = sanitize_request_failure_context(request_failure_context)
                if self._snapshot is not None and context is not None:
                    self._snapshot.setdefault("request_failure_context", context)
            return deepcopy(self._snapshot)
        finally:
            self._lock.release()


def capture_failure(runtime, *, phase=None, observation=_OBSERVE, error=None):
    """Observe before abort cleanup; diagnostics must never interfere with guards."""
    try:
        context = _request_failure_context(error)
        collector = getattr(runtime, "_benchmark_failure_diagnostics", None)
        if collector is not None:
            return collector.capture(phase=phase, observation=observation,
                                     request_failure_context=context)
        if context is not None:
            return sanitize_failure_diagnostics({"schema_version": SCHEMA, "phase": phase,
                                                 "request_failure_context": context})
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
