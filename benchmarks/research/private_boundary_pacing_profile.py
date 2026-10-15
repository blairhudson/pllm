"""Read-only wrappers around installed HTTP and WAN pacing; no transport replacement."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from copy import deepcopy
import importlib.util
import math
from statistics import mean, median
import threading
import time
from unittest.mock import patch

from . import private_boundary_role_profile as roles

IDENTITY = "pllm.private_boundary_pacing_profile.parallel_residues.v1"
NAME = roles.NAME
ROLES = roles.ROLES
EXTRA_TRACE_MEMORY = 32 << 20
EXTRA_MEMORY = roles.EXTRA_MEMORY + EXTRA_TRACE_MEMORY
MAX_HEADS = 8
MAX_CHUNKS = 2048
MAX_SLEEPS = 4096
MAX_HTTP_SPANS = 16
HTTP_REQUIRED = ("http11.send_request_headers", "http11.send_request_body",
    "http11.receive_response_headers", "http11.receive_response_body", "http11.response_closed")
HTTP_ALLOWED = (*HTTP_REQUIRED, "connection.connect_tcp", "connection.start_tls")
TRACE_KEY = "private_boundary_pacing_profile"
_HTTP = ContextVar("research_head_http", default=None)
_RESERVATION = ContextVar("research_head_reservation", default=None)
_LOCK = threading.Lock()


def _engine():
    """Specialize a private namespace; historical readers and evidence stay frozen."""
    spec = importlib.util.spec_from_file_location(
        "benchmarks.research._pacing_role_engine", roles.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.IDENTITY = IDENTITY
    module.EXTRA_MEMORY = EXTRA_MEMORY
    module.OWNERS = {**roles.OWNERS,
        "client_and_collection_bytes": roles.OWNERS["client_and_collection_bytes"] + EXTRA_TRACE_MEMORY}
    module.matched_suite = matched_suite
    ordinary = module._ordinary_estimate
    charged = module._charged_estimate

    def ordinary_estimate(estimate):
        if "private_boundary_pacing_trace_bytes" in estimate["components"]:
            raise ValueError("ordinary baseline already charges pacing traces")
        ordinary(estimate)

    def charged_estimate(baseline, audited, order):
        # Original helper reads this module's larger aggregate/owner constants.
        # Its literal client increment needs the separately priced trace owner.
        result = charged(baseline, audited, order)
        result["client_peak_bytes"] += EXTRA_TRACE_MEMORY
        result["components"]["private_boundary_pacing_trace_bytes"] = EXTRA_TRACE_MEMORY
        return result

    module._ordinary_estimate = ordinary_estimate
    module._charged_estimate = charged_estimate
    return module


def matched_suite(**kwargs):
    from pllm.metrics import BenchmarkSuite
    base = roles.matched_suite(**kwargs)
    return BenchmarkSuite(base.cases, base.sampling, IDENTITY)


def preflight(suite):
    return _engine().preflight(suite)


def research_workers(script, source_identity):
    return _engine().research_workers(script, source_identity)


def _edge(source, destination):
    if source == "client" and destination in ROLES:
        return destination, "upload"
    if destination == "client" and source in ROLES:
        return source, "download"
    return None


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.heads = {role: [] for role in ROLES}
        self.active = {}
        self.pacer = None
        self.totals = {f"{a}->{b}": 0 for role in ROLES
            for a, b in (("client", role), (role, "client"))}
        self.chunks = self.sleeps = 0
        self.started = None
        self.failed = False

    def begin(self, role, size):
        with self.lock:
            if role in self.active or len(self.heads[role]) >= MAX_HEADS:
                self.failed = True
                raise RuntimeError("bounded sequential head ownership required")
            row = {"start_ns": time.monotonic_ns(), "end_ns": None, "completed": False,
                "request_body_bytes": size, "response_body_bytes": None, "http_spans": [], "chunks": []}
            self.heads[role].append(row)
            self.active[role] = row
            return row

    def finish(self, role, row, result):
        with self.lock:
            row["end_ns"] = time.monotonic_ns()
            row["completed"] = result is not None and result[0] == 200
            if result is not None:
                row["response_body_bytes"] = len(result[1])
            self.failed |= not row["completed"]
            if self.active.pop(role, None) is not row:
                self.failed = True

    def chunk(self, pacer, role, direction, size):
        with self.lock:
            if self.pacer is None:
                self.pacer = pacer
            if self.pacer is not pacer:
                self.failed = True
                raise RuntimeError("multiple relay owners")
            head = self.active.get(role)
            if head is None:
                return None
            if self.chunks >= MAX_CHUNKS:
                self.failed = True
                raise RuntimeError("pacing chunk trace bound exceeded")
            self.chunks += 1
            row = {"direction": direction, "bytes": size, "start_ns": time.monotonic_ns(),
                "end_ns": None, "counter_after": None, "completed": False, "sleeps": []}
            head["chunks"].append(row)
            return row

    def sleep(self, row, delay, source, destination):
        with self.lock:
            if self.sleeps >= MAX_SLEEPS:
                self.failed = True
                raise RuntimeError("pacing sleep trace bound exceeded")
            self.sleeps += 1
            item = {"start_ns": time.monotonic_ns(), "end_ns": None, "completed": False,
                "requested_seconds": delay, "source_tokens": source["tokens"],
                "destination_tokens": destination["tokens"],
                "source_rate": source["rate"], "destination_rate": destination["rate"]}
            row["sleeps"].append(item)
            return item

    def trace(self, row):
        pending = {}
        def callback(event, info):
            # Never inspect/store info: it can contain request headers or sockets.
            name, state = event.rsplit(".", 1)
            if name not in HTTP_ALLOWED or state not in {"started", "complete", "failed"}:
                self.failed = True
                raise RuntimeError("unrecognized head HTTP trace")
            if state == "started":
                if name in pending or len(row["http_spans"]) >= MAX_HTTP_SPANS:
                    self.failed = True
                    raise RuntimeError("HTTP trace owner bound exceeded")
                item = {"label": name, "start_ns": time.monotonic_ns(),
                    "end_ns": None, "completed": False}
                pending[name] = item
                row["http_spans"].append(item)
            else:
                item = pending.pop(name, None)
                if item is None:
                    self.failed = True
                    raise RuntimeError("HTTP trace completion without owner")
                item.update(end_ns=time.monotonic_ns(), completed=state == "complete")
                self.failed |= state != "complete"
        return callback

    def report(self):
        if self.failed or self.active or self.started is None or self.pacer is None:
            raise RuntimeError("incomplete pacing profile")
        # Ordinary target has stopped/joined the relay before this read.
        snapshot = self.pacer.snapshot()
        if any(snapshot["directed_stream_bytes"].get(edge, 0) != value for edge, value in self.totals.items()):
            raise RuntimeError("pacing wrapper missed boundary stream bytes")
        heads = deepcopy(self.heads)
        for rows in heads.values():
            for row in rows:
                intervals = [row, *row["http_spans"], *row["chunks"],
                    *(sleep for chunk in row["chunks"] for sleep in chunk["sleeps"])]
                for interval in intervals:
                    for field in ("start_ns", "end_ns"):
                        if type(interval[field]) is not int:
                            raise RuntimeError("incomplete pacing interval")
                        interval[field] -= self.started
        return {"schema": IDENTITY, "clock": "same_client_process_monotonic_ns",
            "heads": heads, "boundary_stream_bytes": dict(self.totals), "final_relay": snapshot}


@contextmanager
def instrument():
    import httpx
    from pllm.runtime import native_wan as wan
    from pllm.runtime.dashboard import DashboardRuntime
    from pllm.runtime.private_boundary_transport import PrivateBoundaryTransport
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("concurrent pacing profile")
    record = Recorder()
    reserve, post = wan.SharedPacer.reserve, PrivateBoundaryTransport._post
    stream, finish = httpx.Client.stream, DashboardRuntime._finish_run
    asynchronous = wan.asyncio

    async def measured_reserve(pacer, source, destination, size):
        edge = _edge(source, destination)
        if edge is None:
            return await reserve(pacer, source, destination, size)
        row = record.chunk(pacer, *edge, size)
        token = _RESERVATION.set((row, pacer, source, destination) if row else None)
        try:
            value = await reserve(pacer, source, destination, size)
            key = source + "->" + destination
            record.totals[key] += size
            if row is not None:
                row.update(end_ns=time.monotonic_ns(), completed=True, counter_after=pacer.links[key])
            return value
        finally:
            _RESERVATION.reset(token)
            if row is not None and not row["completed"]:
                row["end_ns"] = time.monotonic_ns()
                record.failed = True

    class AsyncioView:
        def __getattr__(self, name):
            return getattr(asynchronous, name)

        async def sleep(self, delay):
            context = _RESERVATION.get()
            if context is None:
                return await asynchronous.sleep(delay)
            row, pacer, source, destination = context
            a, b = pacer.assignments[source], pacer.assignments[destination]
            item = record.sleep(row, delay, pacer.buckets[a, "upload"], pacer.buckets[b, "download"])
            try:
                value = await asynchronous.sleep(delay)
                item["completed"] = True
                return value
            finally:
                item["end_ns"] = time.monotonic_ns()
                record.failed |= not item["completed"]

    def measured_post(owner, index, path, payload, phase, *, limit, operation=None):
        if operation != "head":
            return post(owner, index, path, payload, phase, limit=limit, operation=operation)
        role = ROLES[index]
        row = record.begin(role, len(payload))
        token = _HTTP.set(row)
        result = None
        try:
            result = post(owner, index, path, payload, phase, limit=limit, operation=operation)
            return result
        finally:
            _HTTP.reset(token)
            record.finish(role, row, result)

    def measured_stream(owner, *args, **kwargs):
        row = _HTTP.get()
        if row is not None:
            if kwargs.get("extensions"):
                raise RuntimeError("existing HTTP extensions not admitted in profile")
            kwargs["extensions"] = {"trace": record.trace(row)}
        return stream(owner, *args, **kwargs)

    def measured_finish(runtime, capture, *args, **kwargs):
        if record.started is not None or type(capture.started_monotonic_ns) is not int:
            raise RuntimeError("pacing profile requires one request clock")
        record.started = capture.started_monotonic_ns
        return finish(runtime, capture, *args, **kwargs)

    try:
        with ExitStack() as stack:
            for owner, name, value in ((wan.SharedPacer, "reserve", measured_reserve),
                    (wan, "asyncio", AsyncioView()), (PrivateBoundaryTransport, "_post", measured_post),
                    (httpx.Client, "stream", measured_stream), (DashboardRuntime, "_finish_run", measured_finish)):
                stack.enter_context(patch.object(owner, name, value))
            yield record
    finally:
        _LOCK.release()


@contextmanager
def worker_bootstrap(mode, source_identity, script):
    from pllm.runtime import benchmark_targets
    from pllm._benchmark_contract import digest
    original = benchmark_targets.execute_target
    engine = _engine()

    def execute(target, measurement):
        if mode == "probe":
            with instrument() as record:
                original(target, measurement)
            measurement.details[TRACE_KEY] = record.report()
        else:
            original(target, measurement)

    with patch.object(benchmark_targets, "execute_target", execute):
        with engine.worker_bootstrap(mode, source_identity, script):
            measured = benchmark_targets.execute_target
            def bound(target, measurement):
                measured(target, measurement)
                commitment = digest(measurement.details.get(TRACE_KEY))
                measurement.details["research_profile_identity"]["pacing_digest"] = commitment
                measurement.comparison["measurement_digest"] = digest({
                    "role_research": measurement.comparison["measurement_digest"], "pacing": commitment})
            with patch.object(benchmark_targets, "execute_target", bound):
                yield


def _interval(row, start, end):
    if (row.get("completed") is not True or any(type(row.get(key)) is not int for key in ("start_ns", "end_ns"))
            or not start <= row["start_ns"] < row["end_ns"] <= end):
        raise ValueError("incomplete or unowned interval")
    return row["end_ns"] - row["start_ns"]


def _sleep_owner(sleep, size):
    from pllm.runtime.native_wan import CHUNK_BYTES
    fields = {"start_ns", "end_ns", "completed", "requested_seconds", "source_tokens",
        "destination_tokens", "source_rate", "destination_rate"}
    if set(sleep) != fields:
        raise ValueError("sleep schema differs")
    if (sleep["source_rate"] != 5_000_000 or sleep["destination_rate"] != 12_500_000
            or any(type(sleep[k]) not in (int, float) or not math.isfinite(sleep[k])
                for k in ("source_tokens", "destination_tokens", "requested_seconds"))
            or any(not 0 <= sleep[k] <= CHUNK_BYTES for k in ("source_tokens", "destination_tokens"))):
        raise ValueError("pacing observation differs from original buckets")
    deficits = {"source_upload": max(0, (size - sleep["source_tokens"]) / sleep["source_rate"]),
        "destination_download": max(0, (size - sleep["destination_tokens"]) / sleep["destination_rate"])}
    delay = max(deficits.values())
    if delay <= 0 or not math.isclose(sleep["requested_seconds"], delay, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("requested sleep does not match limiting deficit")
    if math.isclose(*deficits.values(), rel_tol=0, abs_tol=1e-12):
        return "both"
    return max(deficits, key=deficits.get)


def summarize_trace(trace, client_profile, report, outputs):
    """Replay only public intervals/counters. No source loading or live admission."""
    from pllm.deployment import WanConditions
    from pllm.metrics.wan import _check_native_sample
    from pllm.runtime.native_wan import CHUNK_BYTES, READ_CHUNK_BYTES
    from .private_boundary_decode_profile import union_ns
    if (set(trace) != {"schema", "clock", "heads", "boundary_stream_bytes", "final_relay"}
            or trace["schema"] != IDENTITY or trace["clock"] != "same_client_process_monotonic_ns"
            or set(trace["heads"]) != set(ROLES)):
        raise ValueError("pacing profile identity differs")
    relay = trace["final_relay"]
    parties = {name: name for name in ("client", "inference", "preparation", *ROLES)}
    _check_native_sample(relay, WanConditions(), parties)
    if relay["read_chunk_bytes"] != READ_CHUNK_BYTES or relay["burst_bytes"] != CHUNK_BYTES:
        raise ValueError("pacing geometry changed")
    expected_edges = {f"{a}->{b}" for role in ROLES for a, b in (("client", role), (role, "client"))}
    totals = trace["boundary_stream_bytes"]
    after = report["native_network_accounting"]["samples"]["runs"][0]["after"]["wan_emulation"]
    if (set(totals) != expected_edges or any(type(value) is not int or value <= 0
            or value != relay["directed_stream_bytes"].get(edge)
            or value < after["directed_stream_bytes"][edge] for edge, value in totals.items())):
        raise ValueError("full boundary relay accounting differs")
    request = client_profile["windows_relative_ns"]["request"]
    paired = sorted((s for s in client_profile["spans"] if s["label"] == "paired_exchange"), key=lambda s: s["start_ns"])
    result = {role: [] for role in ROLES}
    all_sleeps = [[] for _ in range(outputs)]
    head_bytes = {edge: 0 for edge in expected_edges}
    body_up = body_down = chunk_count = sleep_count = 0
    for role, label in zip(ROLES, ("http_exchange_a", "http_exchange_b")):
        heads = trace["heads"][role]
        if len(heads) != outputs or len(paired) != outputs:
            raise ValueError("head coverage incomplete")
        previous = 0
        for ordinal, (head, pair) in enumerate(zip(heads, paired)):
            if set(head) != {"start_ns", "end_ns", "completed", "request_body_bytes", "response_body_bytes", "http_spans", "chunks"}:
                raise ValueError("head trace schema differs")
            http, = [s for s in client_profile["spans"] if s["label"] == label
                and pair["start_ns"] <= s["start_ns"] < s["end_ns"] <= pair["end_ns"]]
            # Pacing's _post wrapper encloses the original client-profiler hook.
            # Both clocks belong to this client; the paired exchange owns both.
            elapsed = _interval(head, max(previous, pair["start_ns"]), min(request[1], pair["end_ns"]))
            if not head["start_ns"] <= http["start_ns"] < http["end_ns"] <= head["end_ns"]:
                raise ValueError("original HTTP interval not enclosed by pacing owner")
            previous = head["end_ns"]
            if any(type(head[k]) is not int or head[k] <= 0 for k in ("request_body_bytes", "response_body_bytes")):
                raise ValueError("head application body missing")
            body_up += head["request_body_bytes"]
            body_down += head["response_body_bytes"]
            http_spans = head["http_spans"]
            labels = [s["label"] for s in http_spans]
            if (not len(HTTP_REQUIRED) <= len(http_spans) <= MAX_HTTP_SPANS or len(set(labels)) != len(labels)
                    or any(label not in HTTP_ALLOWED for label in labels)
                    or [label for label in labels if label in HTTP_REQUIRED] != list(HTTP_REQUIRED)):
                raise ValueError("HTTP transport coverage incomplete")
            http_durations = {}
            end = head["start_ns"]
            for span in http_spans:
                if set(span) != {"label", "start_ns", "end_ns", "completed"}:
                    raise ValueError("HTTP trace fields differ")
                http_durations[span["label"]] = _interval(span, end, head["end_ns"])
                end = span["end_ns"]
            links = {}
            for direction in ("upload", "download"):
                chunks = [c for c in head["chunks"] if c["direction"] == direction]
                if not chunks:
                    raise ValueError("missing head relay direction")
                end, counter = head["start_ns"], None
                reserve_ns = requested = count = 0
                sleep_bins = {key: 0 for key in ("source_upload", "destination_download", "both")}
                for chunk in chunks:
                    chunk_count += 1
                    if (set(chunk) != {"direction", "bytes", "start_ns", "end_ns", "counter_after", "completed", "sleeps"}
                            or type(chunk["bytes"]) is not int or not 0 < chunk["bytes"] <= READ_CHUNK_BYTES
                            or type(chunk["counter_after"]) is not int or chunk["counter_after"] < chunk["bytes"]
                            or counter is not None and chunk["counter_after"] != counter + chunk["bytes"]):
                        raise ValueError("chunk bytes or sequential link coverage differ")
                    reserve_ns += _interval(chunk, end, head["end_ns"])
                    end, counter = chunk["end_ns"], chunk["counter_after"]
                    count += chunk["bytes"]
                    sleep_end = chunk["start_ns"]
                    for sleep in chunk["sleeps"]:
                        sleep_count += 1
                        owner = _sleep_owner(sleep, chunk["bytes"])
                        sleep_bins[owner] += _interval(sleep, sleep_end, chunk["end_ns"])
                        sleep_end = sleep["end_ns"]
                        requested += sleep["requested_seconds"]
                        all_sleeps[ordinal].append((sleep["start_ns"], sleep["end_ns"]))
                body = head["request_body_bytes" if direction == "upload" else "response_body_bytes"]
                # One non-chunked HTTP request/reply; bounded framing stays visible.
                if not body < count <= body + 4096:
                    raise ValueError("TCP head bytes do not cover bounded HTTP framing")
                if count > elapsed / 1e9 * 5_000_000 + CHUNK_BYTES + 1:
                    raise ValueError("head bytes exceed source upload allowance")
                edge = f"client->{role}" if direction == "upload" else f"{role}->client"
                head_bytes[edge] += count
                if counter > totals[edge] or sum(sleep_bins.values()) > reserve_ns:
                    raise ValueError("pacing counter or elapsed conservation failure")
                links[direction] = {"stream_bytes": count, "application_body_bytes": body,
                    "http_framing_bytes": count - body, "reservation_ns": reserve_ns,
                    "sleep_ns_by_limiting_bucket": sleep_bins, "requested_sleep_seconds_sum": requested,
                    "reservation_non_sleep_ns": reserve_ns - sum(sleep_bins.values()),
                    "burst_adjusted_source_upload_floor_ns": max(0, count - CHUNK_BYTES) / 5_000_000 * 1e9,
                    "first_chunk_ns": chunks[0]["start_ns"] - head["start_ns"],
                    "last_chunk_ns": chunks[-1]["end_ns"] - head["start_ns"]}
            if len(head["chunks"]) != sum(1 for c in head["chunks"] if c["direction"] in links):
                raise ValueError("unknown relay direction")
            reserved = sum(link["reservation_ns"] for link in links.values())
            # Request and response reservation loops can overlap; retain their union.
            covered = union_ns((c["start_ns"], c["end_ns"]) for c in head["chunks"])
            result[role].append({"ordinal": ordinal, "http_ns": elapsed, "http_owners_ns": http_durations,
                "links": links, "reservation_union_ns": covered,
                "reservation_sum_nonadditive_ns": reserved, "http_outside_reservation_ns": elapsed - covered})
    if chunk_count > MAX_CHUNKS or sleep_count > MAX_SLEEPS:
        raise ValueError("trace allocation bound exceeded")
    privacy = report["runs"][0]["privacy"]
    if (body_up != privacy["private_boundary_head_upload_bytes"]
            or body_down != privacy["private_boundary_head_download_bytes"]
            or any(head_bytes[edge] > totals[edge] for edge in head_bytes)):
        raise ValueError("head bodies do not reconcile with original SDK counters")
    return {"roles": result, "decode_head_ordinals": list(range(1, outputs)),
        "paired_sleep_union_ns": [union_ns(intervals) for intervals in all_sleeps],
        "paired_exchange_ns": [row["end_ns"] - row["start_ns"] for row in paired],
        "head_stream_bytes": head_bytes,
        "non_head_boundary_stream_bytes": {edge: totals[edge] - head_bytes[edge] for edge in totals},
        "post_scenario_boundary_stream_bytes": {edge: totals[edge] - after["directed_stream_bytes"][edge] for edge in totals}}


def decode_means(trace):
    """One mean per client, then cohort medians; never pool role clocks/replicates."""
    ordinals = trace["decode_head_ordinals"]
    result = {"paired_exchange_ns": mean(trace["paired_exchange_ns"][i] for i in ordinals),
        "paired_sleep_union_ns": mean(trace["paired_sleep_union_ns"][i] for i in ordinals)}
    for role in ROLES:
        heads = [trace["roles"][role][i] for i in ordinals]
        for field in ("http_ns", "reservation_union_ns", "http_outside_reservation_ns"):
            result[f"{role}.{field}"] = mean(head[field] for head in heads)
        for label in HTTP_REQUIRED:
            result[f"{role}.{label}_ns"] = mean(head["http_owners_ns"][label] for head in heads)
        for direction in ("upload", "download"):
            links = [head["links"][direction] for head in heads]
            for field in ("stream_bytes", "application_body_bytes", "http_framing_bytes", "reservation_ns",
                    "requested_sleep_seconds_sum", "reservation_non_sleep_ns",
                    "burst_adjusted_source_upload_floor_ns", "first_chunk_ns", "last_chunk_ns"):
                result[f"{role}.{direction}.{field}"] = mean(link[field] for link in links)
            for owner in ("source_upload", "destination_download", "both"):
                result[f"{role}.{direction}.{owner}_sleep_ns"] = mean(
                    link["sleep_ns_by_limiting_bucket"][owner] for link in links)
    return result


def audit(document, *, source_identity, tiny=False):
    from pllm._benchmark_contract import digest
    from .sota_combinations import Rejected
    try:
        engine = _engine()
        normalized = deepcopy(document)
        for row in normalized["results"]:
            execution = row["execution"]
            details = execution["details"]
            commitment = digest(details.get(TRACE_KEY))
            identity = details["research_profile_identity"]
            target = engine.client.ordinary_comparison(
                _target(execution), details)["measurement_digest"]
            base = digest({"ordinary": target, "research": IDENTITY,
                "mode": identity["mode"], "sources": source_identity})
            role_bound = digest({"ordinary_research": base, "role_telemetry": engine.telemetry_digest(details)})
            if (identity.get("pacing_digest") != commitment or execution["comparison"]["measurement_digest"]
                    != digest({"role_research": role_bound, "pacing": commitment})):
                raise ValueError("pacing integrity binding differs")
            execution["comparison"]["measurement_digest"] = role_bound
        summary = engine.audit(normalized, source_identity=source_identity, tiny=tiny)
        summaries = []
        for row in document["results"]:
            execution = row["execution"]
            details = execution["details"]
            trace = details.get(TRACE_KEY)
            if execution["case"]["name"].startswith("probe--"):
                summaries.append({"replicate": execution["replicate"], "trace": summarize_trace(trace,
                    details["private_boundary_decode_profile"], details["report"], 2 if tiny else 8)})
            elif trace is not None:
                raise ValueError("control pacing instrumentation present")
        summary["pacing_profiles"] = summaries
        means = [decode_means(row["trace"]) for row in summaries]
        summary["pacing_decode_client_means"] = means
        summary["pacing_decode_medians"] = {key: median(row[key] for row in means) for key in means[0]}
        summary["admission_contract"].update(research_client_bytes=(8 << 20) + EXTRA_TRACE_MEMORY,
            research_native_total_bytes=EXTRA_MEMORY, pacing_trace_client_bytes=EXTRA_TRACE_MEMORY)
        summary["pacing_scope"] = ("Same-client-clock HTTP and relay intervals. Sleep wall includes event-loop wakeup; "
            "limiting bucket is checked at each original sleep decision, not inferred from residuals. "
            "Parallel spans use unions, never sums as wall time. HTTP outside reservations includes role work, "
            "read/write/drain, client work and scheduling; role clocks remain separate. "
            "Rate floors are analytical; instrumented control/probe costs stay charged.")
        return summary
    except Rejected:
        raise
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        raise Rejected("pacing_profile_invalid_evidence") from exc


def _target(execution):
    from pllm.metrics import BenchmarkTarget
    return BenchmarkTarget.from_spec(execution["case"]["target"])
