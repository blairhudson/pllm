"""One authenticated native GKM worker; it never receives the other role's share."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
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

from .delegated_protocol import ADMISSION_BYTES, FRAME_BYTES, ROLES, digest, pack, unpack
from .delegated_source import admit_source
from .delegated_transport import request_memory_bytes


async def _body(request, limit):
    lengths = request.headers.getlist("content-length")
    if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
            or len(lengths[0]) > 10 or int(lengths[0]) > limit
            or request.headers.get("content-encoding", "identity") != "identity"):
        raise ValueError("GKM invalid bounded body declaration")
    expected = int(lengths[0])
    raw = bytearray()
    async for block in request.stream():
        if len(raw) + len(block) > expected:
            raise ValueError("GKM body exceeds declared length")
        raw.extend(block)
    if len(raw) != expected:
        raise ValueError("GKM body length mismatch")
    return bytes(raw)


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
        finally:
            self.active = None
            self.request = None
            self.deadline = 0
            if self.reservation is not None:
                reservation, self.reservation = self.reservation, None
                reservation.__exit__(None, None, None)

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
                "tokenizer": self.source.tokenizer_digest, "frame_limit": FRAME_BYTES}


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
            try:
                raw = await asyncio.wait_for(_body(request, ADMISSION_BYTES), timeout=owner.ttl)
                value = unpack(raw)
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
                reservation = reserve_incremental_memory(request_memory_bytes(owner.source))
                reservation.__enter__()
                owner.reservation = reservation
                native = owner.source.load_transport(experiment, rows, outputs, worker=(
                    value["publication"], value["pin"], ROLES.index(role), value["start"],
                ))
                owner.active, owner.request = native, rid
                if native.request_id().hex() != rid:
                    raise ValueError("GKM request differs from its native start frame")
                owner.deadline = time.monotonic() + owner.ttl
                if await request.is_disconnected():
                    raise ValueError("GKM admission disconnected")
                return Response(pack({"schema": "pllm.gkm.ready.v1", "request": rid, "role": role,
                    "contract_sha256": hashlib.sha256(contract).hexdigest(), "epoch": owner.epoch}),
                    media_type="application/msgpack")
            except BaseException as error:
                owner.retire()
                if not isinstance(error, Exception):
                    raise
                return Response(status_code=400)

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
            try:
                action = request.path_params["action"]
                if action == "step":
                    raw = await asyncio.wait_for(_body(request, FRAME_BYTES),
                                                 timeout=max(0, owner.deadline - time.monotonic()))
                    started = time.perf_counter_ns()
                    result = owner.active.execute(raw)
                    elapsed = time.perf_counter_ns() - started
                    if time.monotonic() >= owner.deadline or await request.is_disconnected():
                        raise ValueError("GKM step disconnected")
                    return Response(result, media_type="application/octet-stream",
                                    headers={"x-pllm-server-ns": str(elapsed)})
                raw = await asyncio.wait_for(_body(request, 0),
                                             timeout=max(0, owner.deadline - time.monotonic()))
                if raw or action not in {"finish", "cancel"}:
                    raise ValueError("GKM invalid lifecycle action")
                if action == "finish":
                    receipt = owner.active.finish()  # Completion only; client verifies later.
                    owner.retire()
                    return Response(receipt, media_type="application/json")
                owner.retire()
                return Response(pack({"request": rid, "role": role, "action": action}),
                                media_type="application/msgpack")
            except BaseException as error:
                owner.retire()
                if not isinstance(error, Exception):
                    raise
                return Response(status_code=400)

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
