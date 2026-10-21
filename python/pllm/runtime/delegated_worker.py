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
from .delegated_protocol import clear_failure_frames as _clear_failure_frames
from .delegated_source import admit_source
from .delegated_transport import request_memory_bytes


async def _body(request, limit):
    lengths = request.headers.getlist("content-length")
    if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
            or len(lengths[0]) > 10 or int(lengths[0]) > limit
            or request.headers.get("content-encoding", "identity") != "identity"):
        raise ValueError("GKM invalid bounded body declaration")
    expected = int(lengths[0])
    raw = bytearray(expected)
    cursor = 0
    block = None
    try:
        # A rejected block can still be held by the suspended stream generator.
        # Close it before returning control to the reservation owner.
        async with aclosing(request.stream()) as stream:
            async for block in stream:
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

    def __init__(self, content, reservation, *, limit, **kwargs):
        self._lease = None
        self._sending = False
        self.body = b""
        try:
            self._lease = reservation.lease()
            if not 0 < len(content) <= limit:
                raise ValueError("GKM reply exceeds its bounded owner")
            super().__init__(content, **kwargs)
        except BaseException as error:
            content = None
            _clear_failure_frames(error)
            self.close()
            raise

    def close(self):
        # An explicit close cannot refund a body still borrowed by an active send.
        # __call__ always completes this cleanup after that send has unwound.
        if self._sending:
            return
        self.body = b""
        lease, self._lease = self._lease, None
        if lease is not None:
            lease.close()

    async def __call__(self, scope, receive, send):
        if self._lease is None or self._sending:
            raise RuntimeError("GKM reply is retired or already sending")
        self._sending = True
        try:
            await super().__call__(scope, receive, send)
        except BaseException as error:
            # Failed send frames can retain the message and its original body.
            # They must drain before close releases the last reservation claim.
            _clear_failure_frames(error)
            raise
        finally:
            self._sending = False
            self.close()

    def __del__(self):
        self.close()


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

    def retire(self):
        try:
            if self.active is not None:
                self.active.abort()
        except BaseException as error:
            _clear_failure_frames(error)
            raise
        finally:
            self.active = None
            self.request = None
            self.deadline = 0
            if self.reservation is not None:
                reservation, self.reservation = self.reservation, None
                reservation.close()

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
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            async with owner.lock:
                owner.closed = True
                owner.retire()

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
            owner.expire()
            # This admission owns no existing window. A busy/replayed caller
            # cannot cancel an unrelated request merely by attempting to start.
            if owner.closed or owner.active is not None:
                return Response(status_code=409)
            body_reservation = None
            raw = value = None
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
                owner.reservation = _Reservation(
                    reserve_incremental_memory(request_memory_bytes(owner.source, worker=True)))
                native = owner.source.load_transport(experiment, rows, outputs, worker=(
                    value["publication"], value["pin"], ROLES.index(role), value["start"],
                ))
                owner.active, owner.request = native, rid
                if native.request_id().hex() != rid:
                    raise ValueError("GKM request differs from its native start frame")
                owner.deadline = time.monotonic() + owner.ttl
                if await request.is_disconnected():
                    raise ValueError("GKM admission disconnected")
                return _PayloadResponse(pack({"schema": "pllm.gkm.ready.v1", "request": rid, "role": role,
                    "contract_sha256": hashlib.sha256(contract).hexdigest(), "epoch": owner.epoch}),
                    owner.reservation, limit=16 << 10, media_type="application/msgpack")
            except BaseException as error:
                # Reraised cancellation can retain source/import arguments in
                # its traceback after this handler returns. Keep stack metadata,
                # but release inactive frame locals while reservations still hold.
                _clear_failure_frames(error)
                owner.retire()
                if not isinstance(error, Exception):
                    raise
                return Response(status_code=400)
            finally:
                # Retire both received and decoded publication copies before
                # releasing their temporary reservation, including failed imports.
                raw = value = None
                if body_reservation is not None:
                    body_reservation.__exit__(None, None, None)

    async def operation(request: Request):
        if not owner.authenticated(request):
            return Response(status_code=401)
        async with owner.lock:
            owner.expire()
            try:
                rid = digest(request.path_params["request"])
            except ValueError:
                return Response(status_code=400)
            if owner.closed or owner.active is None or rid != owner.request:
                return Response(status_code=404)
            raw = result = receipt = response = None
            try:
                action = request.path_params["action"]
                if action == "step":
                    raw = await asyncio.wait_for(_body(request, owner.frame_limit),
                                                 timeout=max(0, owner.deadline - time.monotonic()))
                    started = time.perf_counter_ns()
                    result = owner.active.execute(raw)
                    elapsed = time.perf_counter_ns() - started
                    if time.monotonic() >= owner.deadline or await request.is_disconnected():
                        raise ValueError("GKM step disconnected")
                    return _PayloadResponse(result, owner.reservation, limit=owner.frame_limit,
                                     media_type="application/octet-stream",
                                     headers={"x-pllm-server-ns": str(elapsed)})
                raw = await asyncio.wait_for(_body(request, 0),
                                             timeout=max(0, owner.deadline - time.monotonic()))
                if raw or action not in {"finish", "cancel"}:
                    raise ValueError("GKM invalid lifecycle action")
                if action == "finish":
                    receipt = owner.active.finish()  # Completion only; client verifies later.
                    response = _PayloadResponse(receipt, owner.reservation, limit=16 << 10,
                                                media_type="application/json")
                    owner.retire()
                    return response
                response = _PayloadResponse(pack({"request": rid, "role": role, "action": action}),
                                            owner.reservation, limit=1024,
                                            media_type="application/msgpack")
                owner.retire()
                return response
            except BaseException as error:
                # This active frame cannot be cleared by traceback.clear_frames.
                # Drain its payloads, then inactive body/native/response callees,
                # before retire releases the request's reservation.
                raw = result = receipt = response = None
                _clear_failure_frames(error)
                owner.retire()
                if not isinstance(error, Exception):
                    raise
                return Response(status_code=400)
            finally:
                raw = result = receipt = response = None

    app = Starlette(routes=[Route("/healthz", health), Route("/v1/delegation/manifest", manifest),
        Route("/v1/delegation/metrics", metrics), Route("/v1/delegation/admit", admit, methods=["POST"]),
        Route("/v1/delegation/{request}/{action}", operation, methods=["POST"])], lifespan=lifespan)
    app.state.delegated_owner = owner
    return app


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

    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False,
                limit_concurrency=4, h11_max_incomplete_event_size=16 << 10)


if __name__ == "__main__":
    main()
