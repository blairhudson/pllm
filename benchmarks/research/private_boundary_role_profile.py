"""Bounded, research-only role bootstrap. Never changes a protocol response."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from unittest.mock import patch

from . import private_boundary_decode_profile as client

IDENTITY = "pllm.private_boundary_role_profile.parallel_residues.v1"
NAME = "coupled-parallel-uniform_residues"
ROLES = ("boundary_a", "boundary_b")
MAX_RECORDS = 32
MAX_RECORD_BYTES = 4096
MAX_FILE_BYTES = MAX_RECORDS * MAX_RECORD_BYTES
EXTRA_MEMORY = 16 << 20
AUDIT_MEMORY = 32 << 20
BASELINE_KEY = "private_boundary_role_profile_ordinary_estimate"
ORDER_KEY = "private_boundary_role_profile_admission_order"
OWNERS = {"client_and_collection_bytes": 8 << 20,
    "boundary_a_bytes": 4 << 20, "boundary_b_bytes": 4 << 20}
LABELS = ("native_head_dispatch", "reply_codec", "reply_frame")
_REQUEST = ContextVar("research_role_request", default=None)


def _engine():
    """Private module namespace reuses frozen auditor without changing its globals."""
    spec = importlib.util.spec_from_file_location(
        "benchmarks.research._role_profile_client_engine", client.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.IDENTITY = IDENTITY
    module.NAMES = (NAME,)
    module.EXTRA_MEMORY = EXTRA_MEMORY
    module.matched_suite = matched_suite
    module.priced_admission = priced_admission
    return module


def matched_suite(**kwargs):
    from pllm.metrics import BenchmarkCase, BenchmarkSuite
    from .private_boundary_decode import matched_suite as ordinary
    base = ordinary(**kwargs)
    case, = [c for c in base.cases if c.name == NAME]
    return BenchmarkSuite([BenchmarkCase(f"{mode}--{NAME}", case.target, case.build)
        for mode in ("control", "probe")], base.sampling, IDENTITY)


def preflight(suite):
    from copy import deepcopy
    from pllm import Experiment
    from pllm.metrics import benchmark_memory
    from pllm.runtime.benchmark_response_audit import state_audit_admission
    rows = []
    for case in suite.cases:
        exp = Experiment.from_spec(case.target.to_spec()["parameters"]["model"]["experiment"])
        ordinary = benchmark_memory(exp, backend="native", enforced_wan=True,
            memory_budget_bytes=suite.sampling.memory_bytes)
        baseline = deepcopy(ordinary["estimate"])
        audited = state_audit_admission(ordinary, suite.sampling.memory_bytes, enforced_wan=True)
        charged = priced_admission(audited, suite.sampling.memory_bytes)
        if (charged["estimate"][BASELINE_KEY] != baseline
                or charged["estimate"] != _charged_estimate(baseline, True, "audit_then_research")):
            raise ValueError("preflight charges differ from original ordinary admission")
        rows.append({"case": case.name, "admission": charged})
    return rows


def priced_admission(report, memory_bytes):
    from copy import deepcopy
    from pllm.runtime.benchmark_memory import admit_memory, host_memory
    baseline = deepcopy(report["estimate"])
    has_audit = "response_state_audit_bytes" in baseline["components"]
    if has_audit:
        if baseline["components"].pop("response_state_audit_bytes") != AUDIT_MEMORY:
            raise ValueError("unexpected existing state audit allowance")
        baseline["client_peak_bytes"] -= AUDIT_MEMORY
        baseline["native_total_peak_bytes"] -= AUDIT_MEMORY
    _ordinary_estimate(baseline)
    estimate = _charged_estimate(baseline, has_audit,
        "audit_then_research" if has_audit else "research_then_audit")
    return admit_memory(estimate, host_memory(), backend="native", enforced_wan=True,
        memory_budget_bytes=memory_bytes)


def _ordinary_estimate(estimate):
    """Structural/numerical replay only; never invokes model/native planning."""
    components = estimate["components"]
    if (any(key in estimate for key in (BASELINE_KEY, ORDER_KEY))
            or any(key in components for key in ("response_state_audit_bytes",
                "private_boundary_profile_bytes", "private_boundary_role_profile_owners"))):
        raise ValueError("baseline already contains audit or research charges")
    _peak_estimate(estimate)


def _peak_estimate(estimate):
    providers = estimate["provider_peak_bytes"]
    if set(providers) != {"inference", "preparation", *ROLES}:
        raise ValueError("ordinary provider owners differ")
    values = [estimate["client_peak_bytes"], estimate["native_total_peak_bytes"], *providers.values()]
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("ordinary memory estimates must be positive integer bytes")
    if estimate["native_total_peak_bytes"] != estimate["client_peak_bytes"] + sum(providers.values()):
        raise ValueError("ordinary native estimate does not reconcile")


def _charged_estimate(baseline, audited, order):
    from copy import deepcopy
    _ordinary_estimate(baseline)
    estimate = deepcopy(baseline)
    estimate[BASELINE_KEY] = deepcopy(baseline)
    estimate[ORDER_KEY] = order
    estimate["client_peak_bytes"] += (8 << 20) + (AUDIT_MEMORY if audited else 0)
    for role in ROLES:
        estimate["provider_peak_bytes"][role] += 4 << 20
    estimate["native_total_peak_bytes"] += EXTRA_MEMORY + (AUDIT_MEMORY if audited else 0)
    estimate["components"].update(private_boundary_profile_bytes=EXTRA_MEMORY,
        private_boundary_role_profile_owners=dict(OWNERS))
    if audited:
        estimate["components"]["response_state_audit_bytes"] = AUDIT_MEMORY
    return estimate


def _native_admission(report):
    """Check recorded admission arithmetic, including separate 32 MiB WAN relay."""
    estimate = report["estimate"]
    native = report["candidates"]["native"]
    if (report["schema"] != "pllm.benchmark_memory.v1" or report["admitted"] is not True
            or report["requested_backend"] != "native" or report["selected_backend"] != "native"
            or report["enforced_wan"] is not True or native["admitted"] is not True
            or native["reasons"] != [] or type(native["required_host_bytes"]) is not int
            or native["required_host_bytes"] != estimate["native_total_peak_bytes"] + (32 << 20)
            or type(report["host"]["admission_budget_bytes"]) is not int
            or not 0 < report["host"]["admission_budget_bytes"] <= (8 << 30)
            or native["required_host_bytes"] > report["host"]["admission_budget_bytes"]):
        raise ValueError("native admission numerical contract differs")


def validate_admission(details):
    admission = details["topology_admission"]
    estimate = admission["estimate"]
    _peak_estimate(estimate)
    baseline = estimate[BASELINE_KEY]
    # Runtime must have priced research before ordinary target adds its audit.
    if estimate != _charged_estimate(baseline, True, "research_then_audit"):
        raise ValueError("runtime owner charges differ from retained ordinary baseline")
    raw = details["report"]["memory_preflight"]
    _ordinary_estimate(raw["estimate"])
    if raw["estimate"] != baseline:
        raise ValueError("retained baseline differs from raw ordinary dashboard admission")
    _native_admission(raw)
    _native_admission(admission)


class RequestRecord:
    def __init__(self):
        self.start = time.monotonic_ns()
        self.cpu = time.process_time_ns()
        self.thread_cpu = time.thread_time_ns()
        self.spans = []
        self.lock = threading.Lock()
        self.failed = False
        self.closed = False

    def wrap(self, label, function):
        def measured(*args, **kwargs):
            start, cpu = time.monotonic_ns(), time.thread_time_ns()
            ok = False
            try:
                result = function(*args, **kwargs)
                ok = True
                return result
            finally:
                end, used = time.monotonic_ns(), time.thread_time_ns() - cpu
                with self.lock:
                    if self.closed or len(self.spans) >= 8:
                        self.failed = True
                    else:
                        self.spans.append({"label": label, "start_ns": start - self.start,
                            "end_ns": end - self.start, "calling_thread_cpu_ns": used})
                    self.failed |= not ok
        return measured

    def finish(self, ok):
        with self.lock:
            self.closed = True
            end = time.monotonic_ns() - self.start
            return {"elapsed_ns": end, "process_cpu_ns": time.process_time_ns() - self.cpu,
                "handler_thread_cpu_ns": time.thread_time_ns() - self.thread_cpu,
                "completed": bool(ok and not self.failed), "spans": list(self.spans)}


def _safe_open(path, flags):
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > MAX_FILE_BYTES):
        os.close(fd)
        raise ValueError("unsafe role telemetry owner")
    return fd


class Sink:
    def __init__(self, path):
        self.fd = _safe_open(path, os.O_WRONLY | os.O_APPEND)
        self.count = 0

    def emit(self, row):
        if self.count >= MAX_RECORDS:
            raise RuntimeError("role telemetry bound exceeded")
        raw = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(raw) > MAX_RECORD_BYTES or os.fstat(self.fd).st_size + len(raw) > MAX_FILE_BYTES:
            raise RuntimeError("role telemetry byte bound exceeded")
        if os.write(self.fd, raw) != len(raw):
            raise RuntimeError("incomplete role telemetry write")
        self.count += 1

    def close(self):
        os.close(self.fd)


class Handling:
    """Pure ASGI wrapper: no body/header inspection, buffering or extra tasks."""
    def __init__(self, app, sink):
        self.app, self.sink = app, sink

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").endswith("/execute"):
            return await self.app(scope, receive, send)
        record = RequestRecord()
        token = _REQUEST.set(record)
        ok = False
        try:
            result = await self.app(scope, receive, send)
            ok = True
            return result
        finally:
            _REQUEST.reset(token)
            row = record.finish(ok)
            if any(s["label"] in {"native_head_dispatch", "reply_codec"} for s in row["spans"]):
                self.sink.emit(row)


@contextmanager
def role_instrument(sink):
    from pllm.runtime import private_boundary_worker as worker
    from pllm.runtime.native import CompiledMatrix
    from pllm.runtime.private_boundary_head_codec import HeadOutputCodec

    def wrapped(label, function):
        def call(*args, **kwargs):
            record = _REQUEST.get()
            if record is None:
                return function(*args, **kwargs)
            return record.wrap(label, function)(*args, **kwargs)
        return call

    with ExitStack() as stack:
        for owner, name, label in ((CompiledMatrix, "_execute", "native_head_dispatch"),
                (HeadOutputCodec, "encode", "reply_codec"), (worker, "pack_frame", "reply_frame")):
            stack.enter_context(patch.object(owner, name, wrapped(label, getattr(owner, name))))
        # Loader sets app.state before returning; wrap only when standalone CLI gets app.
        from pllm.runtime import private_boundary_role as role
        original_role_create = role.create_private_boundary_role_app
        def create_role(*args, **kwargs):
            return Handling(original_role_create(*args, **kwargs), sink)
        stack.enter_context(patch.object(role, "create_private_boundary_role_app", create_role))
        yield


def summarize_role(row):
    if set(row) != {"elapsed_ns", "process_cpu_ns", "handler_thread_cpu_ns", "completed", "spans"}:
        raise ValueError("role record schema differs")
    if row["completed"] is not True or any(type(row[k]) is not int or row[k] < 0
            for k in ("elapsed_ns", "process_cpu_ns", "handler_thread_cpu_ns")):
        raise ValueError("incomplete role handling")
    spans = row["spans"]
    if not isinstance(spans, list) or len(spans) != 3:
        raise ValueError("role head owners incomplete")
    ordered = sorted(spans, key=lambda s: s["start_ns"])
    if [s["label"] for s in ordered] != list(LABELS):
        raise ValueError("role head owner sequence differs")
    previous = 0
    result = {}
    for span in ordered:
        if (set(span) != {"label", "start_ns", "end_ns", "calling_thread_cpu_ns"}
                or any(type(span[k]) is not int for k in ("start_ns", "end_ns", "calling_thread_cpu_ns"))
                or not previous <= span["start_ns"] < span["end_ns"] <= row["elapsed_ns"]
                or span["calling_thread_cpu_ns"] < 0):
            raise ValueError("invalid role head interval")
        previous = span["end_ns"]
        result[span["label"]] = span["end_ns"] - span["start_ns"]
    result["handling_unclassified"] = row["elapsed_ns"] - sum(result.values())
    return {"exclusive_wall_ns": result, "handler_process_cpu_ns": row["process_cpu_ns"],
        "handler_thread_cpu_ns": row["handler_thread_cpu_ns"],
        "owner_calling_thread_cpu_ns": {s["label"]: s["calling_thread_cpu_ns"] for s in spans}}


def collect(path, outputs):
    fd = _safe_open(path, os.O_RDONLY)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES or not raw.endswith(b"\n"):
        raise ValueError("incomplete role telemetry")
    lines = raw.splitlines()
    if len(lines) != outputs or len(lines) > MAX_RECORDS or any(len(line) > MAX_RECORD_BYTES for line in lines):
        raise ValueError("role head count differs")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate telemetry field")
            result[key] = value
        return result
    rows = [json.loads(line, object_pairs_hook=unique) for line in lines]
    for row in rows:
        summarize_role(row)
    return rows


@contextmanager
def role_children(script, outputs, enabled):
    """Redirect only exact installed loopback role command; keep supervisor cleanup."""
    import subprocess
    original = subprocess.Popen
    with tempfile.TemporaryDirectory(prefix="pllm-role-profile-") as directory:
        paths = {role: Path(directory) / role for role in ROLES}
        for path in paths.values():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        seen = set()
        def popen(command, *args, **kwargs):
            if (isinstance(command, list) and command[1:3] == ["-m", "pllm.runtime.private_boundary_role"]):
                if (len(command) != 9 or command[3] != "--role" or command[4] not in ROLES
                        or command[5:8] != ["--host", "127.0.0.1", "--port"]
                        or not command[8].isdigit() or not 1 <= int(command[8]) <= 65535
                        or command[4] in seen):
                    raise ValueError("role child command not allowlisted")
                role = command[4]
                seen.add(role)
                command = [command[0], "-I", str(script), "--role-worker",
                    str(paths[role]), "probe" if enabled else "control", *command[3:]]
            return original(command, *args, **kwargs)
        records = {}
        with patch.object(subprocess, "Popen", popen):
            yield records
        if seen != set(ROLES):
            raise ValueError("separate role ownership incomplete")
        if enabled:
            records.update({role: collect(paths[role], outputs) for role in ROLES})
        elif any(path.stat().st_size for path in paths.values()):
            raise ValueError("control role instrumented")


@contextmanager
def worker_bootstrap(mode, source_identity, script):
    from pllm.runtime import benchmark_targets
    from pllm._benchmark_contract import digest
    original = benchmark_targets.execute_target
    engine = _engine()
    def execute(target, measurement):
        outputs = target.to_spec()["parameters"]["scenario"]["max_output_tokens"]
        with role_children(script, outputs, mode == "probe") as records:
            original(target, measurement)
        if mode == "probe":
            measurement.details["private_boundary_role_profile"] = {
                "schema": IDENTITY, "clock_alignment": "role_local_only", "roles": records}
    with patch.object(benchmark_targets, "execute_target", execute):
        with engine.worker_bootstrap(mode, source_identity):
            measured = benchmark_targets.execute_target
            def bound(target, measurement):
                measured(target, measurement)
                validate_admission(measurement.details)
                commitment = telemetry_digest(measurement.details)
                measurement.details["research_profile_identity"]["telemetry_digest"] = commitment
                measurement.comparison["measurement_digest"] = digest({
                    "ordinary_research": measurement.comparison["measurement_digest"],
                    "role_telemetry": commitment})
            with patch.object(benchmark_targets, "execute_target", bound):
                yield


def telemetry_digest(details):
    from pllm._benchmark_contract import digest
    return digest({"client": details.get("private_boundary_decode_profile"),
        "roles": details.get("private_boundary_role_profile"),
        "topology_admission": details.get("topology_admission"),
        "ordinary_admission": details.get("report", {}).get("memory_preflight")})


def research_workers(script, source_identity):
    return _engine().research_workers(script, source_identity)


def paired_summary(role_profile, profile, outputs):
    if (set(role_profile) != {"schema", "clock_alignment", "roles"}
            or role_profile["schema"] != IDENTITY or role_profile["clock_alignment"] != "role_local_only"
            or set(role_profile["roles"]) != set(ROLES)):
        raise ValueError("role profile schema differs")
    pairs = sorted((s for s in profile["spans"] if s["label"] == "paired_exchange"), key=lambda s: s["start_ns"])
    if len(pairs) != outputs:
        raise ValueError("head pairs incomplete")
    result = {}
    for role, label in zip(ROLES, ("http_exchange_a", "http_exchange_b")):
        rows = role_profile["roles"][role]
        if len(rows) != outputs:
            raise ValueError("role head count differs")
        items = []
        for ordinal, (pair, row) in enumerate(zip(pairs, rows)):
            http, = [s for s in profile["spans"] if s["label"] == label
                and pair["start_ns"] <= s["start_ns"] < s["end_ns"] <= pair["end_ns"]]
            elapsed = http["end_ns"] - http["start_ns"]
            items.append({"head_ordinal": ordinal, "role_local": summarize_role(row),
                "client_http_wall_ns": elapsed,
                "unaligned_http_minus_handling_ns": elapsed - row["elapsed_ns"]})
        result[role] = {"heads": items, "decode_head_ordinals": list(range(1, outputs)),
            "decode_denominator": outputs - 1}
    return result


def audit(document, *, source_identity, tiny=False):
    from copy import deepcopy
    from pllm._benchmark_contract import digest
    from pllm.metrics import BenchmarkSuite
    from .sota_combinations import Rejected
    try:
        # Validate role envelope before passing unchanged ordinary commitments to
        # frozen auditor. This is an integrity binding, not a signature/MAC.
        normalized = deepcopy(document)
        suite = BenchmarkSuite.from_spec(document["suite"])
        for row in normalized["results"]:
            execution = row["execution"]
            details = execution["details"]
            validate_admission(details)
            identity = details["research_profile_identity"]
            commitment = telemetry_digest(details)
            target, = [c.target for c in suite.cases if c.name == execution["case"]["name"]]
            ordinary = client.ordinary_comparison(target, details)["measurement_digest"]
            base = digest({"ordinary": ordinary, "research": IDENTITY,
                "mode": identity["mode"], "sources": source_identity})
            if (identity.get("telemetry_digest") != commitment
                    or execution["comparison"]["measurement_digest"] != digest({
                        "ordinary_research": base, "role_telemetry": commitment})):
                raise ValueError("role telemetry integrity binding differs")
            execution["comparison"]["measurement_digest"] = base
        summary = _engine().audit(normalized, source_identity=source_identity, tiny=tiny)
        profiles = []
        for row in document["results"]:
            execution = row["execution"]
            details = execution["details"]
            role = details.get("private_boundary_role_profile")
            if execution["case"]["name"].startswith("probe--"):
                profiles.append({"replicate": execution["replicate"], "paired_roles": paired_summary(
                    role, details["private_boundary_decode_profile"], 2 if tiny else 8)})
            elif role is not None:
                raise ValueError("control role telemetry present")
        summary["role_profiles"] = profiles
        summary["admission_contract"] = {"ordinary_baseline": "exact raw dashboard memory_preflight estimate",
            "research_client_bytes": 8 << 20, "research_per_boundary_bytes": 4 << 20,
            "research_native_total_bytes": EXTRA_MEMORY, "state_audit_client_and_native_bytes": AUDIT_MEMORY,
            "relay_required_host_bytes_outside_estimate": 32 << 20,
            "runtime_order": "research_then_audit", "all_numerical_owner_charges_reconciled": True}
        summary["role_scope"] = ("Role-local durations only; no shared monotonic epoch asserted. "
            "Native dispatch excludes input/output conversion. Handling includes ASGI send, not socket drain. "
            "HTTP-minus-handling is an unaligned diagnostic including transport, pacing, queueing and client work; "
            "not pure network time. Parallel role spans and CPUs must not be added to client wall time. "
            "Decode role calls 1..N-1 are whole calls, not clipped shared-clock window attribution.")
        return summary
    except Rejected:
        raise
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        raise Rejected("role_profile_invalid_evidence") from exc
