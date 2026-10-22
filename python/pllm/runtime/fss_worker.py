"""Authenticated, one-attempt FuseFSS roles with strictly offline material tapes.

The client must await dealer prepare (both seal acknowledgements) before sending
either input, and upload both inputs before starting either holder. Each holder
independently enforces its local seal/input barrier. No dealer participates in
online execution. Native admission precedes tape creation; disk pricing is not
RSS, native-cap expansion, authentication of source publishers, or qualification.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import resource
import secrets
import threading
import tempfile
import time

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from pllm import _native
from pllm.configuration import Experiment
from pllm.deployment.network import NetworkError
from pllm.deployment.resources import reserve_incremental_memory, reserve_incremental_disk
from .delegated_transport import _connections
from .delegated_worker import _Reservation, _PayloadResponse, _body
from .replicated_worker import native_call
from .fss_source import resolve_client_source, load_role_source
from .fss_tape import MaterialTape
from .fss_transport import price_transport, timeout_limit, transport_capacity, transport_material_delivery
from .benchmark_delegated import FSS_PHASE_COUNTERS
from .delegated_source import _DECODER_SOURCE_BYTES
from . import fss_protocol as p


_NATIVE_COUNT_MAX = (1 << 64) - 1  # native usize counts / u64 callback encodings
_METRICS_MEMORY_ERRORS = frozenset({
    "CLIENT_MEMORY_OBSERVATION_UNAVAILABLE", "CLIENT_MEMORY_PRESSURE",
    "CLIENT_MEMORY_CAPACITY_CHANGED",
})


def _key(value):
    if (type(value) is not str or not 16 <= len(value) <= 512
            or any(not 33 <= ord(c) <= 126 for c in value)):
        raise ValueError("FuseFSS requires bounded distinct ASCII credentials")
    return value


def _hex(value):
    return p.digest(value)


def _integer(value, maximum, *, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("FuseFSS integer exceeds admission")
    return value


def _channels(role, api_key, connections, credentials):
    if role not in p.ROLES:
        raise ValueError("FuseFSS role differs")
    _key(api_key)
    outgoing_roles = p.HOLDERS if role == "dealer" else tuple(r for r in p.HOLDERS if r != role)
    incoming_roles = () if role == "dealer" else ("dealer", *outgoing_roles)
    outgoing = dict(zip(outgoing_roles, _connections(connections, outgoing_roles)))
    if type(credentials) is not dict or set(credentials) != set(incoming_roles):
        raise ValueError("FuseFSS incoming credential directions differ")
    incoming = {r: _key(credentials[r]) for r in incoming_roles}
    keys = [api_key, *incoming.values(), *(key for _, key in outgoing.values())]
    if len(set(keys)) != len(keys):
        raise ValueError("FuseFSS client and directed peer credentials must differ")
    return outgoing, incoming


class _Attempt:
    """Single retirement owner; leases outlive handlers and exported HTTP bodies."""

    def __init__(self, reservation, ttl):
        self.reservation = reservation
        self.expires = time.monotonic() + ttl
        self.loop = asyncio.get_running_loop()
        self.request_id = self.epochs = self.contract = None
        self.loaded = self.native = self.tape = self.admission = None
        self.transport_resources = None
        self.disk_reservations = []  # at most source, tape and native guard owners
        self.source_disk_lease = None
        self.input = self.output = None
        self.phase = "admitting"
        self.stop = threading.Event()
        self.callback_lock = threading.Lock()
        self.callback_pending = False
        self.callback_futures = set()
        self.callback_tasks = set()
        self.handlers = set()
        self.clients = {}
        self.job = self.timer = self.close_task = None
        self.control_busy = self.peer_busy = False
        self.retiring = False
        self.finished_callback = False
        self.mailbox = None
        self.mail_available = asyncio.Event()
        self.mail_empty = asyncio.Event()
        self.mail_empty.set()
        self.rpc_lock = asyncio.Lock()
        self.sent = {r: 0 for r in p.ROLES}
        self.material_count = self.peer_received = self.peer_sent = 0
        self.peer_bytes = p.peer_bytes()
        self.native_run_ns = None
        self.delivery_callback_ns = self.seal_callback_ns = 0

    def cancelled(self):
        return self.stop.is_set() or time.monotonic() >= self.expires

    def reserve_disk(self, amount, directory):
        if amount:
            manager = reserve_incremental_disk(amount, directory=directory)
            lease = manager.__enter__()
            self.disk_reservations.append(manager)
            return lease

    def signal(self):
        """Safe inside native cancel callbacks: never join or call native again."""
        if self.stop.is_set():
            return
        self.stop.set()
        with self.callback_lock:
            for future in tuple(self.callback_futures):
                future.cancel()
        self.loop.call_soon_threadsafe(self.mail_available.set)
        self.loop.call_soon_threadsafe(self.mail_empty.set)

    async def close(self):
        if self.close_task is None:
            self.retiring = True
            self.signal()
            self.close_task = asyncio.create_task(self._close())
        interrupted = False
        while not self.close_task.done():
            try:
                await asyncio.shield(self.close_task)
            except asyncio.CancelledError:
                interrupted = True
        self.close_task.result()
        if interrupted:
            raise asyncio.CancelledError

    async def _close(self):
        # Callers remove themselves from handlers before joining this owner.
        errors = []
        async def drain(function, *args):
            try:
                await native_call(function, *args)
            except BaseException as error:
                p.clear_failure_frames(error)
                errors.append(True)

        if self.timer is not None:
            self.timer.cancel()
        tasks = set(self.handlers) | set(self.callback_tasks)
        for task in tasks:
            task.cancel()
        # Closing sockets unblocks outstanding callback RPCs before joining native.
        results = await asyncio.gather(*(c.aclose() for c in self.clients.values()), return_exceptions=True)
        for error in results:
            if isinstance(error, BaseException):
                p.clear_failure_frames(error)
                errors.append(True)
        results = None
        if self.native is not None:
            await drain(self.native.cancel)
        if self.job is not None:
            self.job.cancel()
            await asyncio.gather(self.job, return_exceptions=True)
        await asyncio.gather(*tasks, return_exceptions=True)
        # Flush thread-safe callback scheduling, including cancelled-before-start.
        await asyncio.sleep(0)
        while self.callback_tasks:
            pending = tuple(self.callback_tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if self.timer is not None:
            await asyncio.gather(self.timer, return_exceptions=True)
        if self.native is not None:
            await drain(self.native.close)
        if self.tape is not None:
            await drain(self.tape.close)
        if self.loaded is not None:
            def close_source():
                if self.source_disk_lease is not None:
                    # The native fstat getter can release the GIL while borrowing
                    # its program. Drain that borrow before mutable native close.
                    self.source_disk_lease.clear_backing()
                self.loaded.close()
            await drain(close_source)
        # Source files, native exact-nonce index and tape may overlap. Shared
        # process-local leases outlive every owner/callback, including failure.
        # A failed owner cleanup must not refund potentially live disk either.
        while not errors and self.disk_reservations:
            await drain(self.disk_reservations[-1].__exit__, None, None, None)
            if not errors:
                self.disk_reservations.pop()
        self.clients.clear()
        self.mailbox = self.input = self.output = self.contract = None
        self.native = self.tape = self.loaded = self.job = self.timer = None
        self.transport_resources = None
        if not errors:
            self.source_disk_lease = None
        self.phase = "closed"
        reservation, self.reservation = self.reservation, None
        # A failed native cleanup must not silently refund potentially live owners.
        if errors:
            self.reservation = reservation
            raise RuntimeError("FuseFSS retirement failed") from None
        reservation.close()


def create_fss_worker_app(experiment, role, api_key, *, peer_connections,
                          peer_credentials, source_pins=None, ttl=300,
                          tape_directory=None):
    """One independent dealer or holder; no reusable requests/reconnects/retries.

    Source disk is reserved from the locally recomputed native source contract
    before import. Transport disk is priced after native role admission and held
    before tape/guard creation. Shared reservations recheck device free space;
    they are process-local logical charges, not cross-process filesystem quotas.
    Decoder admission requires the client's remaining_seconds; ttl is its local
    maximum. The admission-time clamp cannot be refreshed by subsequent traffic.
    """
    configured_capacity = experiment.pipeline.components["linear"].params.get("capacity", "capsule")
    configured_delivery = p.material_delivery(experiment.pipeline.components["linear"].params.get("material_delivery", "single"))
    if (type(ttl) not in (int, float) or not math.isfinite(ttl)
            or not 0 < ttl <= timeout_limit(configured_capacity)):
        raise ValueError("FuseFSS lifetime differs")
    if tape_directory is not None and not isinstance(tape_directory, Path):
        raise ValueError("FuseFSS tape directory differs")
    native_directory = Path(tempfile.gettempdir())  # native tempfile owners use the OS temporary directory
    tape_directory = tape_directory if tape_directory is not None else native_directory
    outgoing, incoming = _channels(role, api_key, peer_connections, peer_credentials)
    if not callable(getattr(_native, "FssRoleRequest", None)):
        raise ValueError("installed native core lacks FssRoleRequest; rebuild required")
    source = resolve_client_source(experiment, expected_pins=source_pins)
    capacity = transport_capacity(source.contract)
    delivery = transport_material_delivery(source.contract)
    if capacity != configured_capacity or delivery != configured_delivery:
        raise ValueError("FuseFSS native source policy differs from configuration")
    # One shared/refcounted owner covers admission, native callbacks, incoming
    # bodies and exported ACKs. Batch Python overlap is additional to both the
    # legacy role owner and native's separately checked queue reservation.
    role_memory = p.ROLE_MEMORY + (p.MATERIAL_BATCH_PYTHON_BYTES if delivery == "batch-v1" else 0)
    configuration = experiment.configuration_digest()
    epoch = secrets.token_hex(32)
    active = None
    closing = False
    seen, seen_frames = set(), set()
    # These are epoch-local *session* ledgers, not primitive/frame ledgers:
    # at most min(64, budget.requests) IDs and one bind hash per burned ID.
    # Per-attempt replay/order uses exact sequences and native-bound metadata,
    # with only one incoming mailbox and one live outgoing callback.
    handlers, retirements = set(), set()
    admission_lock = asyncio.Lock()
    retired_sessions = 0
    phase_timing = dict.fromkeys(FSS_PHASE_COUNTERS, 0)

    def authorized(request, key):
        values = request.headers.getlist("authorization")
        return (len(values) == 1 and values[0].isascii()
                and hmac.compare_digest(values[0], f"Bearer {key}"))

    def reply(value, reservation, limit=p.CONTROL_LIMIT):
        payload = p.pack(value, limit=limit)
        return _PayloadResponse(payload, reservation, limit=limit, media_type="application/octet-stream")

    async def retire(attempt):
        nonlocal active, retired_sessions
        try:
            await attempt.close()
        finally:
            if active is attempt and attempt.reservation is None:
                active = None
                retired_sessions += 1

    def schedule_retire(attempt):
        attempt.retiring = True
        attempt.signal()
        task = asyncio.create_task(retire(attempt))
        retirements.add(task)
        def done(task):
            retirements.discard(task)
            if not task.cancelled() and task.exception() is not None:
                p.clear_failure_frames(task.exception())
        task.add_done_callback(done)

    async def callback_job(attempt, function, args):
        task = asyncio.current_task()
        attempt.callback_tasks.add(task)
        try:
            if attempt.cancelled():
                raise ValueError("FuseFSS callback retired")
            async with asyncio.timeout(max(0, attempt.expires - time.monotonic())):
                return await function(attempt, *args)
        except BaseException as error:
            p.clear_failure_frames(error)
            raise
        finally:
            args = function = None
            attempt.callback_tasks.discard(task)
            with attempt.callback_lock:
                attempt.callback_pending = False

    def bridge(attempt, function, *args):
        with attempt.callback_lock:
            if attempt.cancelled() or attempt.callback_pending or attempt.callback_futures:
                raise ValueError("FuseFSS callback retired")
            # A cancelled concurrent Future may finish before its coroutine has
            # drained. Keep the single payload slot until callback_job retires;
            # otherwise a caught callback failure could enqueue retained bodies.
            attempt.callback_pending = True
            job = callback_job(attempt, function, args)
            try:
                future = asyncio.run_coroutine_threadsafe(job, attempt.loop)
            except BaseException:
                job.close()
                attempt.callback_pending = False
                raise
            attempt.callback_futures.add(future)
        try:
            return future.result(timeout=max(.001, attempt.expires - time.monotonic()))
        finally:
            args = function = None
            future.cancel()
            with attempt.callback_lock:
                attempt.callback_futures.discard(future)

    def callback_meta(attempt, raw, kind, expected_role):
        if type(raw) is not bytes or len(raw) > attempt.admission["callback_metadata_bytes_bound"]:
            raise ValueError("FuseFSS callback metadata exceeds admission")
        meta = p.metadata(raw)
        fields = {"schema", "kind", "kernel", "role"} | (
            {"pc", "packet_bytes_bound"} if kind == "material" else {"frame", "frame_bytes"})
        if (set(meta) != fields or meta["schema"] != "pllm.fss.callback.v1"
                or meta["kind"] != kind or type(meta["role"]) is not int or meta["role"] != expected_role):
            raise ValueError("FuseFSS callback identity differs")
        kernel = meta["kernel"]
        if type(kernel) is not dict or set(kernel) != {"context", "plan_digest", "ordinal"}:
            raise ValueError("FuseFSS kernel differs")
        _hex(kernel["plan_digest"])
        _integer(kernel["ordinal"], attempt.admission["kernels"] - 1)
        context = kernel["context"]
        if type(context) is not dict or set(context) != {"plan", "session", "operation", "first_tensor_index"}:
            raise ValueError("FuseFSS callback context differs")
        for name in ("plan", "session", "operation"):
            _hex(context[name])
        _integer(context["first_tensor_index"], (1 << 64)-1)
        if (context["session"] != attempt.request_id
                or context["plan"] != attempt.admission["context"]["plan"]):
            raise ValueError("FuseFSS callback session/program differs")
        if kind == "material":
            _integer(meta["pc"], (1 << 64)-1)
            _integer(meta["packet_bytes_bound"], attempt.admission["largest_packet_bytes"], minimum=1)
        else:
            # Frame is kernel-local, but can never reach the admitted whole-run
            # count. Its exact expected index/context is matched to native below.
            _integer(meta["frame"], attempt.admission["frames_per_party"] - 1)
            _integer(meta["frame_bytes"], attempt.admission["peer_frame_bytes_bound"])
        return meta

    async def rpc(attempt, peer, action, value, phase):
        async with attempt.rpc_lock:  # exactly one outgoing request, no retry/pool queue
            if attempt.cancelled():
                raise ValueError("FuseFSS peer attempt retired")
            origin, key = outgoing[peer]
            if peer not in attempt.clients:
                attempt.clients[peer] = httpx.AsyncClient(trust_env=False, follow_redirects=False,
                    limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                    timeout=httpx.Timeout(max(.001, attempt.expires-time.monotonic()), connect=10, pool=10))
            limit = p.MATERIAL_BATCH_BYTES if action == "material-batch" else p.FRAME_LIMIT
            raw = p.pack(value, limit=limit)
            if action == "material-batch" and len(raw) - len(value["records"]) > p.MATERIAL_BATCH_ENVELOPE_BYTES:
                raise ValueError("FuseFSS material batch envelope exceeds bound")
            counts = attempt.peer_bytes[peer]
            counts[f"{phase}_upload_bytes"] += len(raw)
            result_bytes = bytearray()
            block = None
            try:
                async with attempt.clients[peer].stream("POST", f"{origin}/v1/fss/{attempt.request_id}/{action}",
                    content=raw, headers={"authorization": f"Bearer {key}",
                                          "content-type": "application/octet-stream"}) as result:
                    lengths = result.headers.get_list("content-length")
                    if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                            or len(lengths[0]) > 5 or not 0 < int(lengths[0]) <= p.CONTROL_LIMIT
                            or result.headers.get("content-encoding", "identity") != "identity"):
                        raise ValueError("FuseFSS peer ACK declaration differs")
                    async for block in result.aiter_raw(chunk_size=p.CONTROL_LIMIT):
                        counts[f"{phase}_download_bytes"] += len(block)
                        if len(result_bytes)+len(block) > int(lengths[0]):
                            raise ValueError("FuseFSS peer ACK exceeds bound")
                        result_bytes.extend(block)
                    expected = {"accepted": True, "sequence": value["sequence"]}
                    if action == "material-batch":
                        expected["count"] = value["count"]
                    if (result.status_code != 200 or len(result_bytes) != int(lengths[0])
                            or p.pack(p.unpack(bytes(result_bytes), limit=p.CONTROL_LIMIT)) != p.pack(expected)):
                        raise ValueError("FuseFSS peer ACK differs")
            finally:
                raw = value = result_bytes = block = None

    async def deliver(attempt, raw, packet):
        if role != "dealer" or attempt.phase != "preparing":
            raise ValueError("FuseFSS dealer is offline only")
        value = p.metadata(raw)
        recipient_index = _integer(value.get("role"), 1)
        meta = callback_meta(attempt, raw, "material", recipient_index)
        if type(packet) is not bytes or not 0 < len(packet) <= meta["packet_bytes_bound"]:
            raise ValueError("FuseFSS material size differs")
        peer = p.HOLDERS[recipient_index]
        sequence = attempt.sent[peer]
        if sequence >= attempt.admission["primitive_requests_per_party"]:
            raise ValueError("FuseFSS material count exceeds admission")
        await rpc(attempt, peer, "material", {"sender": role, "sender_epoch": epoch,
            "recipient_epoch": attempt.epochs[peer], "sequence": sequence,
            "metadata": raw, "packet": packet}, "setup")
        attempt.sent[peer] += 1


    def require_batch(attempt):
        if (delivery != "batch-v1" or attempt.transport_resources is None
                or attempt.transport_resources.material_delivery != "batch-v1"):
            raise ValueError("FuseFSS material batch was not admitted")

    async def deliver_batch(attempt, recipient, records):
        require_batch(attempt)
        if role != "dealer" or attempt.phase != "preparing":
            raise ValueError("FuseFSS dealer is offline only")
        recipient = _integer(recipient, 1)
        p.material_batch_size(records)
        peer, count = p.HOLDERS[recipient], len(records)
        sequence = attempt.sent[peer]
        if count > attempt.admission["primitive_requests_per_party"] - sequence:
            raise ValueError("FuseFSS material batch count exceeds admission")
        for raw, packet in records:
            meta = callback_meta(attempt, raw, "material", recipient)
            if len(packet) > meta["packet_bytes_bound"]:
                raise ValueError("FuseFSS material size differs")
        encoded = p.encode_material_batch(records)
        await rpc(attempt, peer, "material-batch", {"sender": role, "sender_epoch": epoch,
            "recipient_epoch": attempt.epochs[peer], "sequence": sequence, "count": count,
            "records": encoded}, "setup")
        if attempt.cancelled():
            raise ValueError("FuseFSS material batch retired")
        attempt.sent[peer] += count

    async def exchange(attempt, raw, frame):
        if role not in p.HOLDERS or attempt.phase != "running":
            raise ValueError("FuseFSS online role differs")
        meta = callback_meta(attempt, raw, "peer", p.HOLDERS.index(role))
        if type(frame) is not bytes or len(frame) != meta["frame_bytes"]:
            raise ValueError("FuseFSS peer frame size differs")
        if attempt.peer_sent >= attempt.admission["frames_per_party"]:
            raise ValueError("FuseFSS frame count exceeds admission")
        peer = next(r for r in p.HOLDERS if r != role)
        await rpc(attempt, peer, "peer", {"sender": role, "epochs": attempt.epochs,
            "sequence": attempt.peer_sent, "metadata": raw, "frame": frame}, "online")
        attempt.peer_sent += 1
        while attempt.mailbox is None:
            if attempt.cancelled():
                raise ValueError("FuseFSS mailbox retired")
            attempt.mail_available.clear()
            await attempt.mail_available.wait()
        incoming, payload = attempt.mailbox
        p.matching_peer_metadata(incoming, raw)
        attempt.mailbox = None
        attempt.mail_empty.set()
        return payload

    async def finish_callback(attempt):
        if attempt.finished_callback:
            raise ValueError("FuseFSS repeated native finish")
        if role == "dealer":
            for peer in p.HOLDERS:
                if attempt.sent[peer] != attempt.admission["primitive_requests_per_party"]:
                    raise ValueError("FuseFSS incomplete dealer material")
                await rpc(attempt, peer, "seal", {"sender": role, "sender_epoch": epoch,
                    "recipient_epoch": attempt.epochs[peer], "sequence": attempt.sent[peer]}, "setup")
                attempt.sent[peer] += 1
        else:
            if (attempt.mailbox is not None or attempt.peer_sent != attempt.admission["frames_per_party"]
                    or attempt.peer_received != attempt.peer_sent):
                raise ValueError("FuseFSS unconsumed peer frames")
        attempt.finished_callback = True

    class Callbacks:
        def __init__(self, attempt):
            self.attempt = attempt

        def deliver(self, meta, packet):
            started = time.perf_counter_ns()
            try:
                return bridge(self.attempt, deliver, meta, packet)
            finally:
                self.attempt.delivery_callback_ns += time.perf_counter_ns() - started

        def deliver_batch(self, role, records):
            started = time.perf_counter_ns()
            try:
                return bridge(self.attempt, deliver_batch, role, records)
            except BaseException as error:
                self.attempt.retiring = True
                self.attempt.signal()
                p.clear_failure_frames(error)
                raise
            finally:
                self.attempt.delivery_callback_ns += time.perf_counter_ns() - started

        def material(self, meta):
            attempt = self.attempt
            if role not in p.HOLDERS or attempt.cancelled() or attempt.phase != "running":
                raise ValueError("FuseFSS material is sealed offline only")
            callback_meta(attempt, meta, "material", p.HOLDERS.index(role))
            return attempt.tape.take(meta)  # native-owned thread; never a dealer RPC

        def exchange(self, meta, frame):
            return bridge(self.attempt, exchange, meta, frame)

        def finish(self):
            if self.attempt.cancelled() or self.attempt.phase != ("preparing" if role == "dealer" else "running"):
                raise ValueError("FuseFSS callback retired")
            if role in p.HOLDERS:
                self.attempt.tape.finish()  # native-owned thread; validates complete consumption
                return bridge(self.attempt, finish_callback)
            started = time.perf_counter_ns()
            try:
                return bridge(self.attempt, finish_callback)
            finally:
                self.attempt.seal_callback_ns += time.perf_counter_ns() - started

        def cancel(self):
            self.attempt.signal()

    async def run(attempt, input_frame):
        def measured_run():
            # Clock the actual native call on its owning thread, excluding the
            # executor queue and later retirement. Native planning, callbacks,
            # packet/frame encoding and Python/native bridge overhead remain in it.
            started = time.perf_counter_ns()
            try:
                return attempt.native.run(input_frame, Callbacks(attempt))
            finally:
                attempt.native_run_ns = time.perf_counter_ns() - started
        try:
            result = await native_call(measured_run, _cancel=attempt.signal)
            if attempt.cancelled() or not attempt.finished_callback:
                raise ValueError("FuseFSS native run incomplete")
            if role == "dealer":
                if result is not None:
                    raise ValueError("FuseFSS dealer returned private output")
                attempt.output = b""
            else:
                if type(result) is not bytes or not 0 < len(result) <= attempt.admission["output_frame_bytes"]:
                    raise ValueError("FuseFSS native output exceeds admission")
                attempt.output = result
            attempt.phase = "complete"
        except BaseException as error:
            result = input_frame = None
            p.clear_failure_frames(error)
            attempt.retiring = True
            attempt.signal()
            # prepare/finish await this job. Let their wrapper report the bounded
            # rejection and join retirement after removing its handler ownership.
            # A detached holder run still needs automatic retirement.
            if not attempt.control_busy and attempt.close_task is None:
                schedule_retire(attempt)
            raise
        finally:
            # native_call drains cancellation before this point. Publish once on
            # the event-loop thread, retaining aggregate failed-call work after
            # the attempt/source/tape retire, without keeping request identifiers.
            if attempt.native_run_ns is not None:
                phase = "preparation" if role == "dealer" else "online"
                phase_timing[f"{phase}_calls"] += 1
                phase_timing[f"{phase}_ns"] += attempt.native_run_ns
                if role == "dealer":
                    phase_timing["delivery_callback_ns"] += attempt.delivery_callback_ns
                    phase_timing["seal_callback_ns"] += attempt.seal_callback_ns
            input_frame = None

    def validate_admission(attempt, raw):
        a = p.metadata(raw)
        if (a.get("schema") != "pllm.fss.role_admission.v1" or type(a.get("role")) is not int
                or a["role"] != p.ROLES.index(role) or a.get("capacity", "capsule") != capacity):
            raise ValueError("FuseFSS native role admission differs")
        context = a.get("context")
        if (type(context) is not dict or set(context) != {"plan", "session", "operation", "first_tensor_index"}
                or context["session"] != attempt.request_id):
            raise ValueError("FuseFSS native session differs")
        for name in ("plan", "session", "operation"):
            _hex(context[name])
        _integer(context["first_tensor_index"], (1 << 64)-1)
        for name in ("kernels", "primitive_requests_per_party", "frames_per_party", "guard_capacity"):
            _integer(a.get(name), _NATIVE_COUNT_MAX)
        _integer(a.get("combined_reservation_bytes"), 64 << 20)
        _integer(a.get("peer_frame_bytes_bound"), p.PACKET_LIMIT)
        _integer(a.get("callback_metadata_bytes_bound"), p.METADATA_LIMIT, minimum=1)
        for name in ("input_frame_bytes", "output_frame_bytes"):
            _integer(a.get(name), 272, minimum=1)
        # Native admits every primitive/guard/work requirement. Tape adds tighter
        # storage limits, never authorizes a request rejected by native admission.
        resources = price_transport(a, attempt.contract)
        manifest = p.metadata(attempt.loaded.manifest_bytes)
        if context["plan"] != manifest.get("program"):
            raise ValueError("FuseFSS admitted program differs from loaded source")
        return a, resources

    async def manifest(request, reservation):
        return reply({"schema": "pllm.fss.worker.v1", "role": role, "epoch": epoch,
            "configuration": configuration, "source_lock": source.source_lock_digest,
            "tokenizer": source.tokenizer_digest, "transport": p.TRANSPORT}, reservation)

    async def metrics(request, reservation):
        value = {"schema": "pllm.fss.worker.metrics.v1", "role": role,
            "cpu_ns": time.process_time_ns(), "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                * (1 if os.sys.platform == "darwin" else 1024),
             "active_sessions": int(active is not None), "seen_requests": len(seen),
            "retired_sessions": retired_sessions, "serving_qualified": False,
            "phase_timing": dict(phase_timing)}
        return _PayloadResponse(json.dumps(value).encode(), reservation,
                                limit=p.CONTROL_LIMIT, media_type="application/json")

    async def admit(request, reservation):
        nonlocal active
        started = time.monotonic()
        attempt = _Attempt(reservation, ttl)
        active = attempt
        request.state.attempt = attempt
        attempt.handlers.add(asyncio.current_task())
        value = p.unpack(await _body(request, p.ADMISSION_LIMIT), limit=p.ADMISSION_LIMIT)
        fields = {"request", "epoch", "epochs", "rows", "outputs", "contract"}
        if capacity == "decoder":
            fields.add("remaining_seconds")
        if set(value) != fields:
            raise ValueError("FuseFSS admission fields differ")
        request_id = _hex(value["request"])
        if request_id in seen or request_id == "0"*64:
            raise ValueError("FuseFSS session replay")
        seen.add(request_id)  # before source/native work, including failed admission
        attempt.request_id = request_id
        if capacity == "decoder":
            remaining = value["remaining_seconds"]
            if (type(remaining) not in (int, float) or not 0 < remaining <= timeout_limit(capacity)
                    or not math.isfinite(remaining)):
                raise ValueError("FuseFSS remaining request lifetime differs")
            # The only clamp: charge body reading/validation from admission
            # entry, before any source/native work. Expiry stays fixed thereafter.
            attempt.expires = min(attempt.expires, started + remaining)
        epochs = value["epochs"]
        if (value["epoch"] != epoch or type(epochs) is not dict or set(epochs) != set(p.ROLES)
                or any(_hex(e) == "0"*64 for e in epochs.values())
                or epochs[role] != epoch or len(set(epochs.values())) != 3):
            raise ValueError("FuseFSS role epochs differ")
        rows = _integer(value["rows"], experiment.budget.max_input_tokens, minimum=1)
        outputs = _integer(value["outputs"], min(4, experiment.budget.max_new_tokens), minimum=1)
        if rows+outputs-1 > 16 or type(value["contract"]) is not bytes:
            raise ValueError("FuseFSS request geometry/contract differs")
        attempt.epochs, attempt.contract = epochs, value["contract"]
        def load():
            source_lease = None
            if attempt.cancelled():
                raise ValueError("FuseFSS source admission retired")
            local_contract = source.runtime_contract(experiment, rows, outputs)
            if (local_contract != attempt.contract or transport_capacity(local_contract) != capacity
                    or transport_material_delivery(local_contract) != delivery):
                raise ValueError("FuseFSS source runtime contract differs")
            if capacity == "decoder":
                source_disk = _integer(json.loads(local_contract).get("source_disk_bytes"), 2 * _DECODER_SOURCE_BYTES)
                source_lease = attempt.reserve_disk(source_disk, native_directory)
                attempt.source_disk_lease = source_lease
            if attempt.cancelled():
                raise ValueError("FuseFSS source admission retired")
            attempt.loaded = load_role_source(experiment, expected_pins=source.snapshot_pins,
                input_rows=rows, outputs=outputs, cancelled=attempt.cancelled, source=source)
            if attempt.loaded.contract_bytes != attempt.contract:
                raise ValueError("FuseFSS loaded source contract differs")
            if source_lease is not None and hasattr(attempt.loaded.native_program, "disk_backing"):
                # Only the locally imported native owner can establish backing.
                # Older cores without this API retain the full pending charge.
                source_lease.bind_backing(attempt.loaded.native_program.disk_backing)
        # Admission's outer timeout bounds the unread body by the configured
        # maximum. Its authenticated, possibly shorter budget must also cover
        # import; the idle expiry timer is installed only after this job drains.
        async with asyncio.timeout(max(0, attempt.expires-time.monotonic())):
            attempt.job = asyncio.create_task(native_call(load, _cancel=attempt.signal))
            await asyncio.shield(attempt.job)
        attempt.job = None
        if attempt.cancelled():
            raise ValueError("FuseFSS source admission retired")
        attempt.phase = "admitted"
        async def expire():
            await asyncio.sleep(max(0, attempt.expires-time.monotonic()))
            attempt.timer = None
            schedule_retire(attempt)
        attempt.timer = asyncio.create_task(expire())
        return reply({"manifest": attempt.loaded.manifest_bytes}, reservation)

    async def operation(request, reservation):
        attempt = request.state.attempt
        action = request.path_params["action"]
        if action == "cancel":
            if await _body(request, 0):
                raise ValueError("FuseFSS cancel must be empty")
            attempt.retiring = True
            attempt.signal()
            return reply({"retired": True}, reservation)
        if attempt.control_busy:
            raise ValueError("FuseFSS overlapping client control")
        attempt.control_busy = True
        try:
            if attempt.cancelled():
                raise ValueError("FuseFSS attempt expired")
            if action == "bind":
                if attempt.phase != "admitted":
                    raise ValueError("FuseFSS bind is one-use")
                value = p.unpack(await _body(request, p.CONTROL_LIMIT), limit=p.CONTROL_LIMIT)
                if set(value) != {"request_frame"} or type(value["request_frame"]) is not bytes:
                    raise ValueError("FuseFSS native request differs")
                identity = hashlib.sha256(value["request_frame"]).digest()
                if identity in seen_frames:
                    raise ValueError("FuseFSS native request replay")
                seen_frames.add(identity)
                def bind():
                    attempt.native = _native.FssRoleRequest(attempt.loaded.native_program,
                        attempt.contract, value["request_frame"], p.ROLES.index(role))
                    attempt.admission, resources = validate_admission(attempt, attempt.native.metadata())
                    if resources.python_batch_reservation_bytes != role_memory - p.ROLE_MEMORY:
                        raise ValueError("FuseFSS Python batch reservation differs")
                    attempt.transport_resources = resources
                    if role in p.HOLDERS:
                        price = resources.tape
                        # Native role construction is metadata-only; the native
                        # file guard is allocated by run, after these leases.
                        attempt.reserve_disk(resources.guard_disk_bytes, native_directory)
                        attempt.reserve_disk(price.max_file_bytes, tape_directory)
                        attempt.tape = MaterialTape(attempt.request_id, role, records=price.records,
                            packet_bytes=price.packet_bytes, total_packet_bytes=price.total_packet_bytes,
                             capacity=price.capacity, metadata_bytes_per_record=price.metadata_bytes_per_record,
                             metadata_storage=price.metadata_storage,
                             directory=tape_directory)
                attempt.job = asyncio.create_task(native_call(bind, _cancel=attempt.signal))
                await asyncio.shield(attempt.job)
                attempt.job = None
                if attempt.cancelled():
                    raise ValueError("FuseFSS bind retired")
                attempt.phase = "bound"
                return reply({"accepted": True}, reservation)
            if action == "input":
                if role not in p.HOLDERS or attempt.phase != "sealed":
                    raise ValueError("FuseFSS input requires a sealed holder tape")
                frame = await _body(request, min(272, attempt.admission["input_frame_bytes"]))
                if not frame:
                    raise ValueError("FuseFSS input frame empty")
                attempt.input, attempt.phase = frame, "ready"
                return reply({"accepted": True}, reservation)
            if await _body(request, 0):
                raise ValueError("FuseFSS action must be empty")
            if action == "prepare":
                if role != "dealer" or attempt.phase != "bound":
                    raise ValueError("FuseFSS prepare is dealer-only and one-use")
                attempt.phase = "preparing"
                attempt.job = asyncio.create_task(run(attempt, None))
                await asyncio.shield(attempt.job)
                return reply({"prepared": True}, reservation)
            if action == "start":
                if role not in p.HOLDERS or attempt.phase != "ready":
                    raise ValueError("FuseFSS start requires a sealed tape and input")
                frame, attempt.input = attempt.input, None
                attempt.phase = "running"
                attempt.job = asyncio.create_task(run(attempt, frame))
                return reply({"started": True}, reservation)
            if action == "finish":
                if attempt.job is None or attempt.phase not in {"preparing", "running", "complete"}:
                    raise ValueError("FuseFSS finish requires started execution")
                await asyncio.shield(attempt.job)
                if attempt.phase != "complete" or attempt.mailbox is not None:
                    raise ValueError("FuseFSS execution incomplete")
                storage = {} if attempt.tape is None else attempt.tape.receipt()
                response = reply({"frame": attempt.output, "peer_bytes": attempt.peer_bytes,
                                  "storage": storage}, reservation)
                attempt.retiring = True
                return response
            raise ValueError("FuseFSS action differs")
        finally:
            attempt.control_busy = False

    async def peer(request, reservation):
        attempt, sender = request.state.attempt, request.state.sender
        action = request.path_params["action"]
        if attempt.peer_busy:
            raise ValueError("FuseFSS overlapping incoming peer")
        attempt.peer_busy = True
        try:
            if attempt.cancelled():
                raise ValueError("FuseFSS peer retired")
            if action in {"material", "material-batch", "seal"}:
                if sender != "dealer" or role not in p.HOLDERS or attempt.phase != "bound":
                    raise ValueError("FuseFSS material is offline only")
                if action == "material-batch":
                    require_batch(attempt)
                limit = p.MATERIAL_BATCH_BYTES if action == "material-batch" else p.FRAME_LIMIT
                raw = await _body(request, limit)
                value = p.unpack(raw, limit=limit)
                body_bytes, raw = len(raw), None
                fields = {"sender", "sender_epoch", "recipient_epoch", "sequence"}
                if action == "material":
                    fields |= {"metadata", "packet"}
                elif action == "material-batch":
                    fields |= {"records", "count"}
                if (set(value) != fields or value["sender"] != sender
                        or value["sender_epoch"] != attempt.epochs[sender]
                        or value["recipient_epoch"] != epoch
                        or type(value["sequence"]) is not int or value["sequence"] != attempt.material_count):
                    raise ValueError("FuseFSS material epoch/sequence differs")
                if action == "material-batch":
                    count = _integer(value["count"], p.MATERIAL_BATCH_RECORDS, minimum=1)
                    if (count > attempt.admission["primitive_requests_per_party"] - attempt.material_count
                            or type(value["records"]) is not bytes
                            or body_bytes - len(value["records"]) > p.MATERIAL_BATCH_ENVELOPE_BYTES):
                        raise ValueError("FuseFSS material batch envelope differs")
                    records = p.decode_material_batch(value["records"], count)
                    for metadata, packet in records:
                        meta = callback_meta(attempt, metadata, "material", p.HOLDERS.index(role))
                        if len(packet) > meta["packet_bytes_bound"]:
                            raise ValueError("FuseFSS material packet differs")
                    def append_batch():
                        for metadata, packet in records:
                            if attempt.cancelled():
                                raise ValueError("FuseFSS material batch retired")
                            attempt.tape.append(metadata, packet)
                    # No executor job per record, and no partial-batch ACK. A
                    # cancelled/failing append drains before the attempt retires.
                    await native_call(append_batch, _cancel=attempt.signal)
                    if attempt.cancelled():
                        raise ValueError("FuseFSS material batch retired")
                    attempt.material_count += count
                elif action == "material":
                    meta = callback_meta(attempt, value["metadata"], "material", p.HOLDERS.index(role))
                    if (attempt.material_count >= attempt.admission["primitive_requests_per_party"]
                            or type(value["packet"]) is not bytes
                            or not 0 < len(value["packet"]) <= meta["packet_bytes_bound"]):
                        raise ValueError("FuseFSS material packet differs")
                    await native_call(attempt.tape.append, value["metadata"], value["packet"], _cancel=attempt.signal)
                    attempt.material_count += 1
                else:
                    if attempt.material_count != attempt.admission["primitive_requests_per_party"]:
                        raise ValueError("FuseFSS seal before complete material")
                    await native_call(attempt.tape.seal, _cancel=attempt.signal)
                    attempt.phase = "sealed"
            else:
                if (sender not in p.HOLDERS or sender == role or role not in p.HOLDERS
                        or attempt.phase not in {"ready", "running"}):
                    raise ValueError("FuseFSS online peer requires sealed input-ready holder")
                # Backpressure BEFORE reading another private body. One waiting
                # headers-only handler is allowed, never a second payload queue.
                await attempt.mail_empty.wait()
                if attempt.cancelled():
                    raise ValueError("FuseFSS mailbox retired")
                value = p.unpack(await _body(request, p.FRAME_LIMIT), limit=p.FRAME_LIMIT)
                if (set(value) != {"sender", "epochs", "sequence", "metadata", "frame"}
                        or value["sender"] != sender or value["epochs"] != attempt.epochs
                        or type(value["sequence"]) is not int or value["sequence"] != attempt.peer_received
                        or attempt.peer_received >= attempt.admission["frames_per_party"]):
                    raise ValueError("FuseFSS peer epoch/sequence differs")
                meta = callback_meta(attempt, value["metadata"], "peer", p.HOLDERS.index(sender))
                if type(value["frame"]) is not bytes or len(value["frame"]) != meta["frame_bytes"]:
                    raise ValueError("FuseFSS peer payload differs")
                attempt.mailbox = value["metadata"], value["frame"]
                attempt.mail_empty.clear()
                attempt.mail_available.set()
                attempt.peer_received += 1
            ack = {"accepted": True, "sequence": value["sequence"]}
            if action == "material-batch":
                ack["count"] = value["count"]
            return reply(ack, reservation)
        finally:
            attempt.peer_busy = False

    def wrapped(function):
        async def endpoint(request):
            nonlocal active
            if closing:
                return Response(status_code=503)
            action = request.path_params.get("action")
            is_peer = action in {"material", "material-batch", "seal", "peer"}
            sender = next((r for r, key in incoming.items() if authorized(request, key)), None) if is_peer else None
            if (is_peer and sender is None) or (not is_peer and not authorized(request, api_key)):
                return Response(status_code=401)
            if len(handlers) >= 8:
                return Response(status_code=429)
            attempt = None
            temporary = lease = reservation = None
            task = asyncio.current_task()
            handlers.add(task)
            admission_owned = False
            response = None
            try:
                # Idempotent client retirement also joins cleanup already started
                # by a failing callback/TTL. Never register that joiner among the
                # handlers which cleanup must itself join.
                if function is operation and action == "cancel":
                    candidate = active
                    request_id = request.path_params["request_id"]
                    if candidate is None or candidate.request_id != request_id or candidate.retiring:
                        if request_id not in seen:
                            return Response(status_code=409)
                        if (candidate is not None and candidate.request_id == request_id
                                and candidate.reservation is not None and candidate.phase != "closed"):
                            reservation = candidate.reservation
                        else:
                            temporary = reservation = _Reservation(reserve_incremental_memory(role_memory))
                        lease = reservation.lease()
                        # Acquire the exported-body lease before cleanup can close
                        # its root; an existing handler lease cannot mint a new
                        # response lease after that root retires.
                        response = reply({"retired": True}, reservation)
                        await _body(request, 0)
                        if candidate is not None and candidate.request_id == request_id:
                            await retire(candidate)
                        return response
                if function is admit:
                    if admission_lock.locked() or active is not None or len(seen) >= min(64, experiment.budget.requests):
                        return Response(status_code=409)
                    await admission_lock.acquire()
                    admission_owned = True
                    reservation = _Reservation(reserve_incremental_memory(role_memory))
                elif function in (manifest, metrics):
                    if active is not None and not active.retiring:
                        reservation = active.reservation
                    else:
                        temporary = reservation = _Reservation(reserve_incremental_memory(role_memory))
                else:
                    attempt = active
                    if (attempt is None or attempt.retiring
                            or attempt.request_id != request.path_params["request_id"]):
                        return Response(status_code=409)
                    request.state.attempt = attempt
                    request.state.sender = sender
                    reservation = attempt.reservation
                    attempt.handlers.add(task)
                lease = reservation.lease()
                remaining = ttl if attempt is None else max(0, attempt.expires-time.monotonic())
                async with asyncio.timeout(remaining):
                    response = await (peer(request, reservation) if is_peer else function(request, reservation))
                return response
            except BaseException as error:
                attempt = getattr(request.state, "attempt", attempt)
                if attempt is not None:
                    attempt.retiring = True
                    attempt.signal()
                p.clear_failure_frames(error)
                if response is not None and hasattr(response, "close"):
                    response.close()
                if not isinstance(error, Exception):
                    raise
                if function is metrics:
                    # Authenticated public diagnostics only. Keep the empty body,
                    # status and reservation/retirement guards; never export an
                    # arbitrary exception message or its private cause.
                    code = "FSS_METRICS_FAILED"
                    if (isinstance(error, NetworkError) and len(error.args) == 1
                            and type(error.args[0]) is str
                            and error.args[0] in _METRICS_MEMORY_ERRORS):
                        code = error.args[0]
                    return Response(status_code=400, headers={"x-pllm-error-code": code})
                return Response(status_code=400)
            finally:
                attempt = getattr(request.state, "attempt", attempt)
                handlers.discard(task)
                if attempt is not None:
                    attempt.handlers.discard(task)
                if admission_owned:
                    admission_lock.release()
                # Remove this handler before retirement joins handlers/native.
                # Keep its lease until retirement has cleared captured payloads.
                try:
                    if attempt is not None and attempt.retiring and attempt.close_task is None:
                        await retire(attempt)
                    elif function is admit and attempt is None and reservation is not None:
                        reservation.close()
                finally:
                    if lease is not None:
                        lease.close()
                    if temporary is not None:
                        temporary.close()
        return endpoint

    @asynccontextmanager
    async def lifespan(app):
        nonlocal closing
        try:
            yield
        finally:
            closing = True
            if active is not None:
                await retire(active)
            tasks = tuple(handlers)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(*tuple(retirements), return_exceptions=True)

    async def healthz(request):
        return JSONResponse({"status": "healthy", "role": role})

    return Starlette(routes=[
        Route("/healthz", healthz),
        Route("/v1/fss/manifest", wrapped(manifest), methods=["GET"]),
        Route("/v1/fss/metrics", wrapped(metrics), methods=["GET"]),
        Route("/v1/fss/admit", wrapped(admit), methods=["POST"]),
        Route("/v1/fss/{request_id}/{action}", wrapped(operation), methods=["POST"]),
    ], lifespan=lifespan)


def main():
    """Credentials are environment-only; local IP binding, no proxy/header trust."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=p.ROLES, default=os.environ.get("PLLM_FSS_ROLE"))
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.role not in p.ROLES or not 1 <= args.port <= 65535:
        parser.error("valid role and port required")
    experiment = Experiment.from_spec(json.loads(os.environ["PLLM_ROLE_EXPERIMENT_JSON"]))
    capacity = experiment.pipeline.components["linear"].params.get("capacity", "capsule")
    app = create_fss_worker_app(experiment, args.role, os.environ["PLLM_FSS_API_KEY"],
        peer_connections=json.loads(os.environ["PLLM_FSS_PEERS_JSON"]),
        peer_credentials=json.loads(os.environ["PLLM_FSS_PEER_CREDENTIALS_JSON"]),
        source_pins=json.loads(os.environ["PLLM_FSS_SOURCE_PINS_JSON"]) if "PLLM_FSS_SOURCE_PINS_JSON" in os.environ else None,
        ttl=timeout_limit(capacity) if capacity == "decoder" else 300)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False, log_level="warning",
                limit_concurrency=8, h11_max_incomplete_event_size=16 << 10,
                timeout_keep_alive=5, proxy_headers=False)


if __name__ == "__main__":
    main()
