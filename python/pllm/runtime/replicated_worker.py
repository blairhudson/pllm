"""Authenticated party-local execution of the compiled replicated decoder.

Peer payloads travel directly between their two owners. Worker A coordinates
public quiescence barriers; it never relays another pair's payloads.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import hmac
import json
import os
import secrets
import shutil
import tempfile
import threading
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from pllm import _native
from pllm.configuration import Experiment
from pllm.deployment.resources import reserve_incremental_memory
from .delegated_protocol import clear_failure_frames, digest, pack, unpack
from .delegated_transport import _connections
from .delegated_worker import _Reservation, _PayloadResponse, _body
from .replicated_protocol import ROLES, FRAME_LIMIT, ADMISSION_LIMIT, REQUEST_MEMORY, input_frame
from .replicated_source import admit_source

async def native_call(function, *args, _cancel=None):
    """Drain a cancelled native call before its caller can release admission."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if _cancel is not None:
            _cancel()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            task.result()
        except BaseException as error:
            clear_failure_frames(error)
        raise
    finally:
        function = args = task = None


class _Attempt:
    def __init__(self, native, reservation, request_id):
        self.native, self.reservation, self.request_id = native, reservation, request_id
        self.lock = asyncio.Lock()
        self.pending = deque()
        self.outgoing = ()
        self.task = None
        self.started = False
        self.phase = "setup"
        self.setup_step = 0
        self.setup_frame = None
        self.closed = False
        self.complete = False
        self.expires = time.monotonic() + 1200
        self.peer_bytes = {role: {f"{phase}_{direction}_bytes": 0
            for phase in ("setup", "online") for direction in ("upload", "download")}
            for role in ROLES}

    def enqueue(self, frames):
        if (len(self.pending) + len(frames) > 16
                or any(type(to) is not int or not 0 <= to < 3 or type(frame) is not bytes
                       or len(frame) > FRAME_LIMIT for to, frame in frames)):
            raise ValueError("replicated peer queue exceeds admission")
        self.pending.extend(frames)

    async def close(self):
        self.closed = True
        task, self.task = self.task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        task = None
        async with self.lock:
            self.pending.clear()
            self.outgoing = ()
            self.setup_frame = None
            native, self.native = self.native, None
            if native is not None:
                native.close()
            native = None
            reservation, self.reservation = self.reservation, None
            if reservation is not None:
                reservation.close()


def create_replicated_worker_app(experiment, role, api_key, peers):
    if (role not in ROLES or type(api_key) is not str or not 16 <= len(api_key) <= 256
            or not api_key.isascii()):
        raise ValueError("replicated worker identity is invalid")
    others = tuple(item for item in ROLES if item != role)
    if type(peers) is not dict or set(peers) != set(others):
        raise ValueError("replicated worker needs its two authenticated peers")
    outgoing = _connections({name: peers[name][:2] for name in others}, others)
    incoming = {name: peers[name][2] for name in others}
    if any(type(key) is not str or not 16 <= len(key) <= 512 or not key.isascii()
           for key in incoming.values()):
        raise ValueError("invalid replicated peer credential")
    keys = [api_key, *incoming.values(), *(key for _, key in outgoing)]
    if len(set(keys)) != len(keys):
        raise ValueError("replicated client and directed peer credentials must differ")
    source = admit_source(experiment)
    epoch = secrets.token_hex(32)
    configuration = experiment.configuration_digest()
    active = None
    seen = set()
    clients = {}
    admission_lock = asyncio.Lock()

    async def retire(attempt):
        nonlocal active
        if attempt is not None:
            await attempt.close()
            if active is attempt:
                active = None

    async def body(request, limit):
        # Shared fixed-length reader closes failed streams and drops traceback
        # buffers before their request lease can refund capacity.
        return await _body(request, limit)

    async def rpc(attempt, peer, action, payload=b""):
        phase = "setup" if action in {"public", "receive_seed"} else "online"
        encoded = pack({"sender": role, "payload": payload}, limit=FRAME_LIMIT)
        async with clients[peer].stream("POST", f"/v1/replicated/{attempt.request_id}/peer/{action}",
                                        content=encoded) as response:
            attempt.peer_bytes[peer][f"{phase}_upload_bytes"] += len(encoded)
            received = bytearray()
            async for chunk in response.aiter_raw():
                if len(received) + len(chunk) > FRAME_LIMIT:
                    raise ValueError("replicated peer response exceeds admission")
                received.extend(chunk)
            # Count each directed request and reverse ACK once at its initiator.
            attempt.peer_bytes[peer][f"{phase}_download_bytes"] += len(received)
            if response.status_code != 200 or response.headers.get("content-encoding", "identity") != "identity":
                raise ValueError("replicated peer rejected its one-use frame")
            return unpack(bytes(received), limit=FRAME_LIMIT)

    async def control(attempt, action):
        if action == "prepare_dispatch":
            async with attempt.lock:
                if attempt.closed or attempt.phase != "run" or attempt.outgoing:
                    raise ValueError("replicated peer wave outside admitted phase")
                attempt.outgoing = tuple(attempt.pending)
                attempt.pending.clear()
                return {"sent": len(attempt.outgoing)}
        if action == "dispatch":
            async with attempt.lock:
                batch, attempt.outgoing = attempt.outgoing, ()
            for recipient, frame in batch:
                if ROLES[recipient] == role:
                    raise ValueError("replicated peer frame has local destination")
                if await rpc(attempt, ROLES[recipient], "receive", frame) != {"accepted": True}:
                    raise ValueError("replicated peer acknowledgement differs")
            return {"sent": len(batch)}
        if action == "advance":
            async with attempt.lock:
                if attempt.pending or attempt.outgoing:
                    raise ValueError("replicated barrier has undrained frames")
                if attempt.closed or attempt.phase not in {"import", "run"}:
                    raise ValueError("replicated execution outside admitted phase")
                attempt.phase = "run"
                complete, frames = await native_call(attempt.native.call, "advance", b"", ROLES.index(role))
                attempt.complete = complete
                attempt.enqueue(frames)
            return {"complete": complete}
        raise ValueError("unknown replicated control")

    async def run(attempt):
        async def all_roles(action):
            tasks = [asyncio.create_task(control(attempt, action) if peer == role
                     else rpc(attempt, peer, action)) for peer in ROLES]
            try:
                return await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        async with asyncio.timeout(max(0.0, attempt.expires - time.monotonic())):
            for _ in range(1 << 20):
                states = await all_roles("advance")
                if any(set(state) != {"complete"} or type(state["complete"]) is not bool for state in states):
                    raise ValueError("replicated barrier acknowledgement differs")
                if all(state["complete"] for state in states):
                    return
                if any(state["complete"] for state in states):
                    raise ValueError("replicated schedules diverged")
                for _ in range(1024):
                    # Freeze every sender's wave before any send. Replies from a
                    # fast peer belong to the next wave; they cannot overtake a
                    # slow peer's prerequisite rotation/opening.
                    sent = await all_roles("prepare_dispatch")
                    if any(set(state) != {"sent"} or type(state["sent"]) is not int
                           or not 0 <= state["sent"] <= 16 for state in sent):
                        raise ValueError("replicated dispatch acknowledgement differs")
                    if not any(state["sent"] for state in sent):
                        break
                    delivered = await all_roles("dispatch")
                    if delivered != sent:
                        raise ValueError("replicated peer wave count differs")
                else:
                    raise ValueError("replicated peer round bound exceeded")
            raise ValueError("replicated barrier budget exhausted")

    @asynccontextmanager
    async def lifespan(app):
        for name, (origin, key) in zip(others, outgoing):
            clients[name] = httpx.AsyncClient(base_url=origin, timeout=60, trust_env=False,
                follow_redirects=False, headers={"authorization": f"Bearer {key}", "accept-encoding": "identity"},
                limits=httpx.Limits(max_connections=2, max_keepalive_connections=2))
        async def expiry():
            while True:
                await asyncio.sleep(0.25)
                if active is not None and time.monotonic() >= active.expires:
                    await retire(active)
        watcher = asyncio.create_task(expiry())
        try:
            yield
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            await retire(active)
            await asyncio.gather(*(client.aclose() for client in clients.values()))

    app = FastAPI(lifespan=lifespan)

    def authorized(request, key):
        values = request.headers.getlist("authorization")
        return len(values) == 1 and values[0].isascii() and hmac.compare_digest(values[0], f"Bearer {key}")

    @app.get("/healthz")
    async def health():
        return {"status": "healthy", "role": "replicated-worker"}

    @app.get("/v1/replicated/manifest")
    async def manifest(request: Request):
        if not authorized(request, api_key):
            return Response(status_code=401)
        return Response(pack({"schema": "pllm.replicated.worker.v1", "role": role,
            "configuration": configuration, "source_lock": source.source_lock_digest,
            "tokenizer": source.tokenizer_digest, "epoch": epoch}), media_type="application/octet-stream")

    @app.get("/v1/replicated/metrics")
    async def metrics(request: Request):
        if not authorized(request, api_key):
            return Response(status_code=401)
        from .benchmark_memory import process_memory
        return JSONResponse({"schema": "pllm.replicated_worker_process_metrics.v1",
            "cpu_ns": time.process_time_ns(), "peak_rss_bytes": process_memory()["lifetime_peak_rss_bytes"],
            "active_sessions": int(active is not None), "role": role})

    @app.post("/v1/replicated/admit")
    async def admit(request: Request):
        nonlocal active
        if not authorized(request, api_key):
            return Response(status_code=401)
        reservation = None
        attempt = None
        raw = value = public = native = None
        try:
            async with admission_lock:
                if active is not None or len(seen) >= min(4096, experiment.budget.requests):
                    raise ValueError("replicated worker capacity exhausted")
                reservation = _Reservation(reserve_incremental_memory(REQUEST_MEMORY))
                raw = await asyncio.wait_for(body(request, ADMISSION_LIMIT), timeout=60)
                value = unpack(raw, limit=ADMISSION_LIMIT)
                if (set(value) != {"request", "epoch", "manifest"} or value["epoch"] != epoch
                        or type(value["manifest"]) is not bytes):
                    raise ValueError("replicated admission envelope differs")
                request_id = digest(value["request"])
                if request_id in seen:
                    raise ValueError("replicated request replay")
                seen.add(request_id)
                public = json.loads(value["manifest"])
                if (bytes(public["session"]).hex() != request_id
                        or bytes(public["source"]).hex() != source.source_lock_digest
                        or bytes(public["composition"]).hex() != experiment.pipeline.digest()
                        or bytes(public["config"]) != source.config_bytes
                        or public.get("weights", "private") != source.contract.get("weights", "private")):
                    raise ValueError("replicated publication differs from source")
                source.runtime_contract(experiment, public["rows"], public["outputs"])
                if source.contract.get("weights") == "public":
                    required = source.contract["public_weight_snapshot_bytes_per_worker"]
                    if required + (64 << 20) > shutil.disk_usage(tempfile.gettempdir()).free:
                        raise ValueError("replicated public snapshot disk headroom exhausted")
                native = await native_call(_native.ReplicatedWorker, value["manifest"], ROLES.index(role))
                attempt = _Attempt(native, reservation, request_id)
                reservation = None
                active = attempt
                if source.contract.get("weights") == "public":
                    stop = threading.Event()
                    async with attempt.lock:
                        await native_call(source.populate_public_worker, native,
                            lambda: stop.is_set() or attempt.closed or time.monotonic() >= attempt.expires,
                            _cancel=stop.set)
                if await request.is_disconnected():
                    raise ValueError("replicated admission disconnected")
                return _PayloadResponse(pack({"accepted": True}), attempt.reservation,
                    limit=4096, media_type="application/octet-stream")
        except BaseException as error:
            raw = value = public = native = None
            clear_failure_frames(error)
            await retire(attempt)
            if reservation is not None:
                reservation.close()
            if not isinstance(error, Exception):
                raise
            return Response(status_code=400)
        finally:
            raw = value = public = native = None

    @app.post("/v1/replicated/{request_id}/peer/{action}")
    async def peer(request_id: str, action: str, request: Request):
        sender = next((name for name, key in incoming.items() if authorized(request, key)), None)
        if sender is None:
            return Response(status_code=401)
        attempt = active
        if attempt is None or request_id != attempt.request_id or attempt.closed:
            return Response(status_code=409)
        reservation = attempt.reservation
        lease = reservation.lease()
        value = result = frames = None
        try:
            value = unpack(await asyncio.wait_for(body(request, FRAME_LIMIT),
                timeout=max(0.0, attempt.expires - time.monotonic())), limit=FRAME_LIMIT)
            if (set(value) != {"sender", "payload"} or value["sender"] != sender
                    or type(value["payload"]) is not bytes):
                raise ValueError("replicated peer sender differs")
            if action in {"advance", "prepare_dispatch", "dispatch"}:
                if sender != ROLES[0] or value["payload"] != b"":
                    raise ValueError("replicated coordinator identity differs")
                result = await control(attempt, action)
            else:
                async with attempt.lock:
                    if (attempt.phase == "setup" and
                            (action == "public" and attempt.setup_step in {1, 2}
                             or action == "receive_seed" and attempt.setup_step in {3, 4})):
                        await native_call(attempt.native.call, action, value["payload"], ROLES.index(sender))
                    elif action == "receive" and attempt.phase == "run":
                        complete, frames = await native_call(attempt.native.call, "peer", value["payload"], ROLES.index(sender))
                        attempt.complete = complete
                        attempt.enqueue(frames)
                    else:
                        raise ValueError("unknown replicated peer action")
                result = {"accepted": True}
            return _PayloadResponse(pack(result), reservation, limit=FRAME_LIMIT,
                                    media_type="application/octet-stream")
        except BaseException as error:
            value = result = frames = None
            clear_failure_frames(error)
            # The coordinator's outstanding HTTP call must finish before its
            # run task can drain; close the local native owner here directly.
            async with attempt.lock:
                attempt.pending.clear()
                attempt.closed = True
                if attempt.native is not None:
                    attempt.native.close()
                    attempt.native = None
            if not isinstance(error, Exception):
                raise
            return Response(status_code=400)
        finally:
            value = result = frames = None
            lease.close()

    @app.post("/v1/replicated/{request_id}/{action}")
    async def operation(request_id: str, action: str, request: Request):
        if not authorized(request, api_key):
            return Response(status_code=401)
        attempt = active
        if attempt is None or request_id != attempt.request_id or (attempt.closed and action != "cancel"):
            return Response(status_code=409)
        reservation = attempt.reservation
        lease = reservation.lease()
        raw = result = value = frame = frames = payload = response = None
        started = time.perf_counter_ns()
        try:
            raw = await asyncio.wait_for(body(request, FRAME_LIMIT),
                timeout=max(0.0, attempt.expires - time.monotonic()))
            if action == "cancel":
                if raw:
                    raise ValueError("replicated cancellation body is nonempty")
                response = _PayloadResponse(pack({"retired": True}), reservation, limit=4096)
                await retire(attempt)
                return response
            if action == "execute":
                if role != ROLES[0] or raw or attempt.started:
                    raise ValueError("replicated execution is one-use")
                attempt.started = True
                attempt.task = asyncio.create_task(run(attempt))
                await attempt.task
                result = {"complete": True}
            elif action in {"setup_public", "setup_seed"}:
                if (raw or attempt.phase != "setup"
                        or attempt.setup_step != (0 if action == "setup_public" else 2)):
                    raise ValueError("replicated setup body is nonempty")
                async with attempt.lock:
                    complete, frames = await native_call(attempt.native.call,
                        "offer" if action == "setup_public" else "seed", b"", ROLES.index(role))
                if complete or len(frames) != 1:
                    raise ValueError("invalid replicated native setup")
                recipient, frame = frames[0]
                attempt.setup_frame = (recipient, frame)
                attempt.setup_step += 1
                result = {"accepted": True}
            elif action == "setup_dispatch":
                if raw or attempt.phase != "setup" or attempt.setup_step not in {1, 3}:
                    raise ValueError("replicated setup dispatch outside admitted phase")
                recipient, frame = attempt.setup_frame
                attempt.setup_frame = None
                if attempt.setup_step == 1:
                    if recipient != ROLES.index(role):
                        raise ValueError("replicated public setup identity differs")
                    for peer in others:
                        if await rpc(attempt, peer, "public", frame) != {"accepted": True}:
                            raise ValueError("replicated setup acknowledgement differs")
                    result = {"accepted": True}
                else:
                    if recipient != (ROLES.index(role) + 2) % 3:
                        raise ValueError("replicated seed recipient differs")
                    result = await rpc(attempt, ROLES[recipient], "receive_seed", frame)
                attempt.setup_step += 1
            else:
                async with attempt.lock:
                    if (action == "activate" and not raw and attempt.phase == "setup"
                            and attempt.setup_step == 4):
                        await native_call(attempt.native.activate)
                        attempt.phase = "import"
                        result = {"accepted": True}
                    elif action == "input" and attempt.phase == "import":
                        value = unpack(raw, limit=FRAME_LIMIT)
                        await native_call(attempt.native.call, "import", input_frame(value), ROLES.index(role))
                        result = {"accepted": True}
                    elif action == "finish" and not raw and attempt.complete:
                        frame = await native_call(attempt.native.finish)
                        result = {"frame": frame, "peer_bytes": attempt.peer_bytes}
                    else:
                        raise ValueError("unknown replicated client action")
            payload = pack(result, limit=FRAME_LIMIT)
            response = _PayloadResponse(payload, reservation, limit=FRAME_LIMIT, media_type="application/octet-stream",
                headers={"x-pllm-server-ns": str(time.perf_counter_ns() - started)})
            if action == "finish":
                # The response retains the host reservation until its final send.
                await retire(attempt)
            return response
        except BaseException as error:
            raw = result = value = frame = frames = payload = response = None
            clear_failure_frames(error)
            await retire(attempt)
            if not isinstance(error, Exception):
                raise
            return Response(status_code=400)
        finally:
            raw = result = value = frame = frames = payload = response = None
            lease.close()

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    experiment = Experiment.from_spec(json.loads(os.environ["PLLM_ROLE_EXPERIMENT_JSON"]))
    app = create_replicated_worker_app(experiment, args.role, os.environ["PLLM_REPLICATED_API_KEY"],
                                       json.loads(os.environ["PLLM_REPLICATED_PEERS_JSON"]))
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False, log_level="warning",
                limit_concurrency=16, h11_max_incomplete_event_size=16 << 10)


if __name__ == "__main__":
    main()
