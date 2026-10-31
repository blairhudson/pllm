"""One authenticated native GKM worker; it never receives the other role's share."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing, asynccontextmanager
import hashlib
import json
import os
import secrets
import time

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from pllm import _native
from pllm.configuration import Experiment
from pllm.deployment.resources import reserve_incremental_memory

from .delegated_protocol import ROLES, digest, limits, pack, unpack, unpack_admission
from .delegated_protocol import (BATCH_EXECUTION, BATCH_HTTP_BACKEND, BATCH_HTTP_CHUNK_BYTES,
                                 BATCH_HTTP_SERVER_BODY_BYTES, batch_contract)
from .delegated_protocol import clear_failure_frames as _clear_failure_frames
from .delegated_source import admit_source
from .delegated_transport import request_memory_bytes


async def _body(request, limit, *, exact=None):
    lengths = request.headers.getlist("content-length")
    if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
            or len(lengths[0]) > 10 or int(lengths[0]) > limit
            or request.headers.get("content-encoding", "identity") != "identity"):
        raise ValueError("GKM invalid bounded body declaration")
    expected = int(lengths[0])
    if exact is not None and expected != exact:
        raise ValueError("GKM batch body length differs from trusted descriptor")
    raw = bytearray(expected)
    cursor = 0
    block = None
    try:
        # A rejected block can still be held by the suspended stream generator.
        # Close it before returning control to the reservation owner.
        async with aclosing(request.stream()) as stream:
            async for block in stream:
                if exact is not None and len(block) > BATCH_HTTP_SERVER_BODY_BYTES:
                    raise ValueError("GKM batch HTTP backend block exceeds its bound")
                if len(block) > expected - cursor:
                    raise ValueError("GKM body exceeds declared length")
                raw[cursor:cursor + len(block)] = block
                cursor += len(block)
        if cursor != expected:
            raise ValueError("GKM body length mismatch")
        return bytes(raw)
    finally:
        # Cancellation/timeout tracebacks must not retain partial input buffers.
        raw = block = None


class _Reservation:
    """One host charge shared by the native root and its outstanding replies."""

    def __init__(self, manager):
        self._closed = True
        self._manager = None
        self._references = 0
        manager.__enter__()
        self._manager = manager
        self._references = 1
        self._closed = False

    def lease(self):
        if self._closed:
            raise ValueError("GKM reservation root is retired")
        return _ReplyLease(self)

    def _release(self):
        self._references -= 1
        if self._references == 0:
            manager, self._manager = self._manager, None
            manager.__exit__(None, None, None)

    def close(self):
        if not self._closed:
            self._closed = True
            self._release()

    def __del__(self):
        self.close()


class _ReplyLease:
    def __init__(self, reservation):
        self._reservation = reservation
        reservation._references += 1

    def close(self):
        reservation, self._reservation = self._reservation, None
        if reservation is not None:
            reservation._release()

    def __del__(self):
        self.close()


class _PayloadResponse(Response):
    """Keep the original reply charged until send retirement or unsent destruction.

    Server concurrency bounds live replies. Transport/caller copies have their
    own admission; no registry or detached response queue is retained here.
    """

    def __init__(self, content, reservation, *, limit, operation_slot=None, **kwargs):
        self._lease = None
        self._operation_slot = operation_slot
        self._sending = False
        self.body = b""
        try:
            self._lease = reservation.lease()
            if not 0 < len(content) <= limit:
                raise ValueError("GKM reply exceeds its bounded owner")
            super().__init__(content, **kwargs)
        except BaseException as error:
            content = None
            self._clear_failure(error)
            self.close()
            raise

    def _clear_failure(self, error):
        root = self._lease._reservation if self._lease is not None else None
        retirement = root._manager if root is not None else None
        if not isinstance(retirement, _WorkerRetirement):
            try:
                _clear_failure_frames(error)
            except BaseException:
                pass  # Preserve established primary; no worker refund authority.
            return
        previous = retirement.uncertain
        terminal = retirement.terminal
        retirement.terminal = retirement.uncertain = True
        try:
            _clear_failure_frames(error)
        except BaseException:
            return  # Recording only; original charge remains nonrefundable.
        retirement.uncertain = previous
        if not previous:
            retirement.terminal = terminal

    def close(self):
        # An explicit close cannot refund a body still borrowed by an active send.
        # __call__ always completes this cleanup after that send has unwound.
        if self._sending:
            return
        self.body = b""
        lease, self._lease = self._lease, None
        slot, self._operation_slot = self._operation_slot, None
        try:
            if lease is not None:
                lease.close()
        finally:
            if slot is not None:
                slot.close()

    async def __call__(self, scope, receive, send):
        if self._lease is None or self._sending:
            raise RuntimeError("GKM reply is retired or already sending")
        self._sending = True
        try:
            await super().__call__(scope, receive, send)
        except BaseException as error:
            # Failed send frames can retain the message and its original body.
            # They must drain before close releases the last reservation claim.
            self._clear_failure(error)
            raise
        finally:
            self._sending = False
            self.close()

    def __del__(self):
        self.close()


class _OperationSlot:
    """One batch operation/reply, including retained unsent responses."""

    def __init__(self, owner):
        if owner.batch_operation_outstanding:
            raise ValueError("GKM batch operation reply is still outstanding")
        self.owner = owner
        owner.batch_operation_outstanding = True

    def close(self):
        owner, self.owner = self.owner, None
        if owner is not None:
            owner.batch_operation_outstanding = False


def _retirement_error(primary, error, phase):
    """First established failure wins; no metadata hooks or added annotations."""
    return error if primary is None else primary


class _WorkerRetirement:
    """One prearmed worker root; failed physical retirement is never retried.

    Fixed controls use the existing native envelope plus 64MiB allowance (and
    reservation control allowance >=64KiB), not a physical-byte qualification.
    No destructor, exception cache, second token, payload copy or retry task.
    Opaque resources from a constructor which never returns remain excluded.
    """

    __slots__ = ("manager", "root", "native", "loop", "lock", "custody",
                 "armed", "terminal", "abort_ok", "root_closed", "exit_seen",
                 "busy", "done", "started", "uncertain")

    def __init__(self, manager, loop, lock):
        self.manager, self.loop, self.lock = manager, loop, lock
        self.root = self.native = None
        self.custody = (self,)  # Allocate before acquiring the original charge.
        self.armed = self.terminal = self.abort_ok = False
        self.root_closed = self.exit_seen = self.busy = self.done = False
        self.started = self.uncertain = False

    def _on_loop(self):
        try:
            return asyncio.get_running_loop() is self.loop
        except RuntimeError:
            return False

    def _published(self):
        manager = self.manager
        with manager._capacity.lock:
            return (manager._capacity._owners.get(manager._token) == manager._amount
                    and manager._token.retained is manager
                    and manager._failed is self.custody)

    def __enter__(self):
        self.manager.__enter__()
        try:
            self.manager.retain_failed(self.custody)
        finally:
            # Publication can succeed before an injected failure. Inspect only;
            # never replace foreign custody or manufacture ledger removal.
            self.armed = self._published()
        if not self.armed:
            raise RuntimeError("GKM worker custody was not published")

    def validate(self):
        if not self.armed or not self._published():
            raise RuntimeError("GKM worker custody identity changed")

    def __exit__(self, *exc):
        self.exit_seen = True
        # Final reply release (including send-finally and __del__) cannot call
        # a fallible backend. Keep full charge until locked proof consumption.
        # Shared reply counters retain their existing single-loop contract.
        return False

    def close(self):
        """Called by manager.drain_failed: proof check ONLY, never native abort."""
        if (not self._on_loop() or self.uncertain or not self.terminal or not self.abort_ok
                or self.native is not None
                or not self.root_closed or not self.exit_seen
                or (self.root is not None and (not self.root._closed
                    or self.root._references != 0 or self.root._manager is not None))):
            raise RuntimeError("GKM worker retirement is not refundable")
        with self.manager._capacity.lock:
            completed = (self.manager._exited
                         and self.manager._token not in self.manager._capacity._owners
                         and self.manager._token.retained is None
                         and not self.manager._failed)
        if self.manager._drain_finish_state is not None and not completed:
            raise RuntimeError("GKM worker retirement finalization is ambiguous")
        if self.armed and not completed:
            self.validate()

    def _finalize(self):
        if self.uncertain:
            raise RuntimeError("GKM worker retirement is not refundable")
        if self.done:
            return
        if self.busy:
            raise RuntimeError("GKM worker reservation exit is ambiguous")
        self.close()
        with self.manager._capacity.lock:
            if (self.manager._exited
                    and self.manager._token not in self.manager._capacity._owners):
                self.done = True  # A direct proof-only manager drain completed.
                return
        self.busy = True
        # Leave busy latched after ambiguous exit: a consumed generator is not
        # refund evidence, and no later explicit drain may manufacture proof.
        if self.armed:
            self.manager.drain_failed()
        else:
            self.manager.__exit__(None, None, None)
        with self.manager._capacity.lock:
            if (not self.manager._exited
                    or self.manager._token in self.manager._capacity._owners):
                raise RuntimeError("GKM worker reservation exit is ambiguous")
        self.done = True
        self.busy = False

    def completion(self):
        """Inspect original proof only; never abort or retry a backend."""
        if not self._on_loop():
            return "FAILED/UNKNOWN"
        if self.uncertain or self.busy:
            return "FAILED/UNKNOWN"
        if not self.terminal:
            return "NOTREADY"
        if not self.abort_ok or self.native is not None:
            return "FAILED/UNKNOWN"
        if (not self.root_closed or not self.exit_seen
                or (self.root is not None and self.root._references != 0)):
            return "NOTREADY"
        if self.root is not None and (not self.root._closed or self.root._manager is not None):
            return "FAILED/UNKNOWN"
        with self.manager._capacity.lock:
            completed = (self.manager._exited
                         and self.manager._token not in self.manager._capacity._owners
                         and self.manager._token.retained is None
                         and not self.manager._failed)
        if completed:
            return "PROVEN"
        if self.done:
            return "FAILED/UNKNOWN"
        if self.manager._drain_finish_state is not None:
            return "FAILED/UNKNOWN"
        if not self.armed or not self._published():
            return "FAILED/UNKNOWN"
        return "PROVEN"

    def retire(self):
        if not self._on_loop():
            raise RuntimeError("GKM worker retirement requires its owning loop")
        if self.started or self.uncertain:
            return  # No automatic pending-abort retry, including shutdown.
        self.terminal = True  # Fence BEFORE calling possibly failing native code.
        self.started = True
        if not self.abort_ok:
            try:
                if self.native is not None:
                    self.native.abort()
            except BaseException as error:
                _retirement_error(error, error, "shutdown")
                raise
            self.abort_ok = True  # Witness precedes dropping native/root.
            self.native = None
        if not self.root_closed:
            self.root_closed = True  # Original root close attempted ONCE.
            if self.root is None:
                # Acquisition/prearm failure: no native was imported, no leases.
                self.exit_seen = True
            else:
                self.root.close()
        if self.exit_seen and not self.done:
            self._finalize()


class _Owner:
    def __init__(self, experiment, role, api_key, *, ttl=300.0):
        if (role not in ROLES or type(api_key) is not str or not 16 <= len(api_key) <= 256
                or not api_key.isascii() or not 0 < ttl <= 600):
            raise ValueError("GKM invalid role, credential or lifetime")
        profile = experiment.resolve()
        if profile.client_runtime != "compiled_gkm_v1":
            raise ValueError("GKM worker requires its exact Experiment")
        self.experiment, self.role, self.key, self.ttl = experiment, role, api_key, ttl
        self.source = admit_source(experiment)
        self.frame_limit, self.admission_limit = limits(
            experiment.pipeline.components["linear"].params.get("capacity", "capsule"))
        self.epoch = secrets.token_hex(32)
        self.lock = asyncio.Lock()
        self.active = None
        self.request = None
        self.deadline = 0
        self.seen = set()
        self.closed = False
        self.reservation = None
        self.retirement = None
        self.batch_selected = experiment.pipeline.components["linear"].params.get("execution") == BATCH_EXECUTION
        self.batch = None
        self.batch_index = 0
        self.batch_operation_outstanding = False

    def fence(self):
        retirement = self.retirement
        if retirement is None:
            return
        if not retirement._on_loop():
            raise RuntimeError("GKM worker retirement requires its owning loop")
        # No route may execute this request again, even if abort throws.
        self.active = None
        self.request = None
        self.deadline = 0
        self.batch = None
        self.batch_index = 0
        retirement.terminal = True

    def clear_failure(self, error):
        # Fence routing before fallible frame metadata access. Unproved clearing
        # stays sticky on this original owner, including direct manager drains.
        self.fence()
        retirement = self.retirement
        previous = retirement.uncertain if retirement is not None else False
        if retirement is not None:
            retirement.uncertain = True
        try:
            _clear_failure_frames(error)
        except BaseException:
            return False
        if retirement is not None:
            retirement.uncertain = previous
        return not previous

    def retire(self):
        retirement = self.retirement
        if retirement is None:
            return
        self.fence()
        try:
            retirement.retire()
        finally:
            if retirement.root_closed:
                self.reservation = None

    @property
    def pending_retirement(self):
        return (self.retirement is not None and self.retirement.terminal
                and not self.retirement.done)

    async def _drain_retirement(self):
        retirement = self.retirement
        if retirement is None:
            return
        if not retirement._on_loop():
            raise RuntimeError("GKM worker drain requires its owning loop")
        async with self.lock:
            if not retirement.terminal:
                raise RuntimeError("GKM worker request is not terminal")
            if self._completion_state() == "PROVEN" and not retirement.done:
                # Administrative drain preserves the first backend failure.
                retirement._finalize()
            if not retirement.done:
                raise RuntimeError("GKM worker retirement incomplete or ambiguous")

    def _completion_state(self):
        """Shared proof inspection, without backend work or error conversion."""
        retirement = self.retirement
        if retirement is None:
            return "NOTREADY"
        if not retirement._on_loop() or not self.lock.locked():
            raise RuntimeError("GKM worker completion requires its locked owning loop")
        return retirement.completion()

    def _consume_completion(self):
        """Ingress classifies failure; explicit drain propagates its first error."""
        state = self._completion_state()
        retirement = self.retirement
        if state == "PROVEN" and not retirement.done:
            try:
                retirement._finalize()
            except AssertionError:
                raise  # Never hide regression-oracle/test failures.
            except BaseException:
                return "FAILED/UNKNOWN"  # busy stays latched; no backend retry.
        return state

    def expire(self):
        if self.active is not None and time.monotonic() >= self.deadline:
            self.retire()

    def authenticated(self, request):
        values = request.headers.getlist("authorization")
        return (len(values) == 1 and values[0].isascii()
                and secrets.compare_digest(values[0], f"Bearer {self.key}"))

    def manifest(self):
        return {"schema": "pllm.gkm.worker.v1", "role": self.role, "epoch": self.epoch,
                "configuration": self.experiment.configuration_digest(),
                "source_lock": self.source.source_lock_digest,
                "tokenizer": self.source.tokenizer_digest, "frame_limit": self.frame_limit}


def create_delegated_worker_app(experiment: Experiment, role: str, api_key: str, *, ttl=300.0):
    owner = _Owner(experiment, role, api_key, ttl=ttl)

    @asynccontextmanager
    async def lifespan(app):
        async def expire():
            while True:
                await asyncio.sleep(min(1.0, owner.ttl))
                async with owner.lock:
                    owner.expire()
        task = asyncio.create_task(expire())
        primary = None
        try:
            yield
        except BaseException as error:
            primary = _retirement_error(error, error, "shutdown")
        finally:
            try:
                task.cancel()
                external_seen = False
                while not task.done():
                    try:
                        # Same existing task; wait never forwards parent cancel.
                        # No new task, shield, timeout or bounded latency claim.
                        await asyncio.wait({task})
                    except asyncio.CancelledError as error:
                        if not external_seen:
                            primary = _retirement_error(primary, error, "teardown")
                            external_seen = True
                        else:
                            _retirement_error(error, error, "teardown")
                if not task.cancelled():
                    try:
                        task.result()  # Retrieve terminal expiry failure ONCE.
                    except BaseException as error:
                        primary = _retirement_error(primary, error, "expiry")
            finally:
                task = None  # Do not retain a failed expiry task via primary TB.
                # An expiry task which already failed cannot skip shutdown.
                # retire() fences but NEVER retries an already terminal abort.
                try:
                    async with owner.lock:
                        owner.closed = True
                        owner.retire()
                except BaseException as error:
                    primary = _retirement_error(primary, error, "shutdown")
                else:
                    try:
                        retirement = owner.retirement
                        if retirement is not None and not retirement.done:
                            if (not retirement.terminal or not retirement.abort_ok
                                    or not retirement.root_closed or not retirement.exit_seen
                                    or (retirement.root is not None
                                        and retirement.root._references != 0)):
                                raise RuntimeError("GKM worker retirement incomplete")
                            # RELEASE owner.lock before drain reacquires it. A
                            # terminal shortcut cannot silently leave full charge.
                            await owner._drain_retirement()
                            if not retirement.done:
                                raise RuntimeError("GKM worker retirement incomplete")
                    except BaseException as error:
                        primary = _retirement_error(primary, error, "shutdown")
            if primary is not None:
                raise primary

    async def health(request):
        return JSONResponse({"role": "delegation-worker", "worker": role})

    async def manifest(request):
        if not owner.authenticated(request):
            return Response(status_code=401)
        return Response(pack(owner.manifest()), media_type="application/msgpack")

    async def metrics(request):
        if not owner.authenticated(request):
            return Response(status_code=401)
        from .benchmark_memory import process_memory

        return JSONResponse({"schema": "pllm.delegation_worker_process_metrics.v1",
                             "cpu_ns": time.process_time_ns(),
                             "peak_rss_bytes": process_memory()["lifetime_peak_rss_bytes"],
                             "active_sessions": int(owner.active is not None)})

    async def admit(request: Request):
        if not owner.authenticated(request):
            return Response(status_code=401)
        async with owner.lock:
            completion = owner._consume_completion()
            if completion == "FAILED/UNKNOWN" or owner.pending_retirement:
                return Response(status_code=409)
            owner.expire()
            if owner.pending_retirement:
                return Response(status_code=409)
            # This admission owns no existing window. A busy/replayed caller
            # cannot cancel an unrelated request merely by attempting to start.
            if owner.closed or owner.active is not None or owner.batch_operation_outstanding:
                return Response(status_code=409)
            if owner.batch_selected and request.scope.get("pllm.batch_http_backend") != BATCH_HTTP_BACKEND:
                return Response(status_code=503)
            body_reservation = None
            raw = value = None
            failed = False
            try:
                if getattr(owner.source, "capacity", "capsule") == "decoder":
                    reservation = reserve_incremental_memory(4 * owner.admission_limit)
                    reservation.__enter__()
                    body_reservation = reservation
                raw = await asyncio.wait_for(_body(request, owner.admission_limit), timeout=owner.ttl)
                value = unpack_admission(raw, getattr(owner.source, "capacity", "capsule"))
                if set(value) != {"schema", "request", "epoch", "configuration", "source_lock",
                                  "tokenizer", "input_rows", "output_tokens", "publication", "pin", "start"}:
                    raise ValueError("GKM invalid admission fields")
                rid = digest(value["request"])
                if rid in owner.seen or len(owner.seen) >= experiment.budget.requests:
                    raise ValueError("GKM request replay or exhausted attempt budget")
                owner.seen.add(rid)
                if (value["schema"] != "pllm.gkm.admit.v1" or value["epoch"] != owner.epoch
                        or value["configuration"] != experiment.configuration_digest()
                        or value["source_lock"] != owner.source.source_lock_digest
                        or value["tokenizer"] != owner.source.tokenizer_digest
                        or type(value["input_rows"]) is not int or type(value["output_tokens"]) is not int
                        or any(type(value[name]) is not bytes for name in ("publication", "pin", "start"))):
                    raise ValueError("GKM admission identity differs")
                rows, outputs = value["input_rows"], value["output_tokens"]
                # Complete Experiment envelope and actual query must admit BEFORE
                # loading source values or allocating any publication/session.
                contract = owner.source.runtime_contract(experiment, rows, outputs)
                batch = batch_contract(json.loads(contract)) if owner.batch_selected else None
                if owner.batch_selected and batch is None:
                    raise ValueError("GKM native batch execution metadata is missing")
                retirement = _WorkerRetirement(
                    reserve_incremental_memory(request_memory_bytes(owner.source, worker=True,
                        **({"role": role} if batch is not None else {}))),
                    asyncio.get_running_loop(), owner.lock)
                owner.retirement = retirement
                root = _Reservation(retirement)
                retirement.root = owner.reservation = root
                retirement.validate()  # Exact token/custody BEFORE native import.
                retirement.native = owner.source.load_transport(experiment, rows, outputs, worker=(
                    value["publication"], value["pin"], ROLES.index(role), value["start"],
                ))
                native = retirement.native
                owner.active, owner.request = native, rid
                owner.batch, owner.batch_index = batch, 0
                if native.request_id().hex() != rid:
                    raise ValueError("GKM request differs from its native start frame")
                owner.deadline = time.monotonic() + owner.ttl
                if await request.is_disconnected():
                    raise ValueError("GKM admission disconnected")
                return _PayloadResponse(pack({"schema": "pllm.gkm.ready.v1", "request": rid, "role": role,
                    "contract_sha256": hashlib.sha256(contract).hexdigest(), "epoch": owner.epoch}),
                    owner.reservation, limit=16 << 10, media_type="application/msgpack")
            except BaseException as error:
                failed = True
                # Reraised cancellation can retain source/import arguments in
                # its traceback after this handler returns. Keep stack metadata,
                # but release inactive frame locals while reservations still hold.
                if owner.clear_failure(error):
                    try:
                        owner.retire()
                    except BaseException as cleanup:
                        owner.clear_failure(cleanup)
                        # First primary wins, including cleanup interrupts.
                if not isinstance(error, Exception):
                    raise
                return Response(status_code=400)
            finally:
                # Retire both received and decoded publication copies before
                # releasing their temporary reservation, including failed imports.
                raw = value = None
                if body_reservation is not None:
                    try:
                        body_reservation.__exit__(None, None, None)
                    except BaseException as cleanup:
                        owner.clear_failure(cleanup)
                        if not failed:
                            raise

    async def operation(request: Request):
        if not owner.authenticated(request):
            return Response(status_code=401)
        async with owner.lock:
            completion = owner._consume_completion()
            if completion == "FAILED/UNKNOWN" or owner.pending_retirement:
                return Response(status_code=409)
            owner.expire()
            if owner.pending_retirement:
                return Response(status_code=409)
            try:
                rid = digest(request.path_params["request"])
            except ValueError:
                return Response(status_code=400)
            if owner.closed or owner.active is None or rid != owner.request:
                return Response(status_code=404)
            if owner.batch is not None and request.scope.get("pllm.batch_http_backend") != BATCH_HTTP_BACKEND:
                return Response(status_code=503)
            if owner.batch_operation_outstanding and request.path_params["action"] != "cancel":
                return Response(status_code=409)
            raw = result = receipt = response = None
            slot = None
            try:
                action = request.path_params["action"]
                if owner.batch is not None and not owner.batch_operation_outstanding:
                    slot = _OperationSlot(owner)
                if action == "step":
                    descriptor = None
                    expected = reply_size = owner.frame_limit
                    if owner.batch is not None:
                        if owner.batch_index >= len(owner.batch["descriptors"]):
                            raise ValueError("GKM batch descriptor exhausted")
                        descriptor = owner.batch["descriptors"][owner.batch_index]
                        expected = descriptor["request_bytes"][ROLES.index(role)]
                        reply_size = descriptor["reply_bytes"][ROLES.index(role)]
                    raw = await asyncio.wait_for(_body(request, expected,
                        **({"exact": expected} if descriptor is not None else {})),
                                                 timeout=max(0, owner.deadline - time.monotonic()))
                    started = time.perf_counter_ns()
                    # The native start selects scalar versus aggregate execution;
                    # the same consuming worker method verifies either wire form.
                    result = owner.active.execute(raw)
                    elapsed = time.perf_counter_ns() - started
                    if descriptor is not None:
                        if type(result) is not bytes or len(result) != reply_size:
                            raise ValueError("GKM native batch reply length differs")
                        owner.batch_index += 1
                    if time.monotonic() >= owner.deadline or await request.is_disconnected():
                        raise ValueError("GKM step disconnected")
                    response = _PayloadResponse(result, owner.reservation, limit=reply_size,
                                      operation_slot=slot,
                                      media_type="application/octet-stream",
                                      headers={"x-pllm-server-ns": str(elapsed)})
                    slot = None  # Response now owns it through send/unsent close.
                    return response
                raw = await asyncio.wait_for(_body(request, 0),
                                             timeout=max(0, owner.deadline - time.monotonic()))
                if raw or action not in {"finish", "cancel"}:
                    raise ValueError("GKM invalid lifecycle action")
                if action == "finish":
                    receipt = owner.active.finish()  # Completion only; client verifies later.
                    response = _PayloadResponse(receipt, owner.reservation, limit=16 << 10,
                                                operation_slot=slot, media_type="application/json")
                    slot = None
                    owner.retire()
                    return response
                response = _PayloadResponse(pack({"request": rid, "role": role, "action": action}),
                                            owner.reservation, limit=1024,
                                            operation_slot=slot,
                                            media_type="application/msgpack")
                slot = None
                owner.retire()
                return response
            except BaseException as error:
                # This active frame cannot be cleared by traceback.clear_frames.
                # Drain its payloads, then inactive body/native/response callees,
                # before retire releases the request's reservation.
                raw = result = receipt = response = None
                if owner.clear_failure(error):
                    try:
                        owner.retire()
                    except BaseException as cleanup:
                        owner.clear_failure(cleanup)
                if not isinstance(error, Exception):
                    raise
                return Response(status_code=400)
            finally:
                raw = result = receipt = response = None
                if slot is not None:
                    slot.close()

    app = Starlette(routes=[Route("/healthz", health), Route("/v1/delegation/manifest", manifest),
        Route("/v1/delegation/metrics", metrics), Route("/v1/delegation/admit", admit, methods=["POST"]),
        Route("/v1/delegation/{request}/{action}", operation, methods=["POST"])], lifespan=lifespan)
    app.state.delegated_owner = owner
    return app


def _reject_batch_ssl(config):
    if config.is_ssl or getattr(config, "ssl", None) is not None:
        raise ValueError("GKM batch TLS is unsupported; bounded loopback HTTP only")


def _batch_h11_protocol():
    """Pinned batch hosting contract; direct ASGI callers must supply its proof.

    asyncio's selector transport reads at most max_size per callback. H11 pauses
    when its body queue crosses HIGH_WATER_LIMIT; one callback can overshoot by
    at most max_size. Reject unsupported transports instead of assuming uvloop,
    TLS or another ASGI server has the same buffering/lifetime contract.
    """
    from asyncio.selector_events import _SelectorSocketTransport
    from uvicorn.protocols.http.h11_impl import H11Protocol, HIGH_WATER_LIMIT

    if HIGH_WATER_LIMIT != BATCH_HTTP_CHUNK_BYTES:
        raise ValueError("GKM batch H11 flow-control bound is unsupported")

    class BoundedH11Protocol(H11Protocol):
        def __init__(self, config, *args, **kwargs):
            # H11Protocol can load an unloaded Config; reject before that path
            # could create an SSLContext or publish a plaintext proof marker.
            _reject_batch_ssl(config)
            super().__init__(config, *args, **kwargs)

        def connection_made(self, transport):
            self.transport = transport
            self._batch_rejected = len(self.connections) >= 4
            if self._batch_rejected:
                # Uvicorn's ASGI limit alone does not bound accepted sockets and
                # their pre-endpoint body queues. Refuse extras before reading.
                transport.close()
                return
            if not isinstance(transport, _SelectorSocketTransport):
                transport.close()
                raise ValueError("GKM batch requires bounded asyncio selector hosting")
            transport.max_size = BATCH_HTTP_CHUNK_BYTES
            self.ws_protocol_class = None
            super().connection_made(transport)

        def data_received(self, data):
            if self._batch_rejected or len(data) > BATCH_HTTP_CHUNK_BYTES:
                self.transport.close()
                return
            super().data_received(data)
            # H11 schedules the ASGI task while handling this callback; it cannot
            # run until this synchronous callback returns with the proof attached.
            if self.scope is not None:
                self.scope["pllm.batch_http_backend"] = BATCH_HTTP_BACKEND

    return BoundedH11Protocol


def _batch_worker_config(app, **options):
    import uvicorn

    class PlaintextConfig(uvicorn.Config):
        def load(self):
            # Server normally loads config before constructing its protocol.
            # Gate here as well, before Uvicorn's SSL context allocator runs.
            _reject_batch_ssl(self)
            super().load()

    return PlaintextConfig(app, loop="asyncio", http=_batch_h11_protocol(), ws="none", **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("invalid loopback port")
    raw = os.environ.get("PLLM_ROLE_EXPERIMENT_JSON", "")
    if not 0 < len(raw) <= 32768:
        parser.error("bounded Experiment required")
    experiment = Experiment.from_spec(json.loads(raw))
    app = create_delegated_worker_app(experiment, args.role, os.environ["PLLM_GKM_API_KEY"])
    import uvicorn

    options = {"host": "127.0.0.1", "port": args.port, "access_log": False,
               "limit_concurrency": 4, "h11_max_incomplete_event_size": 16 << 10}
    if app.state.delegated_owner.batch_selected:
        uvicorn.Server(_batch_worker_config(app, **options)).run()
    else:
        uvicorn.run(app, **options)


if __name__ == "__main__":
    main()
