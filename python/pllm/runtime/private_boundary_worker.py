"""Authenticated boundary-only worker; no prepared-body or sampling execution."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response

from .native import CompiledMatrix
from .private_boundary_head_codec import HeadOutputCodec
from .private_boundary_protocol import (
    BoundaryAdmission, BoundaryDescriptor, BoundaryError, MAX_BODY_BYTES,
    MAX_CONTROL_BYTES, PACKED_QUERY_MAGIC, ROLES, SCHEMA, canonical, context_bytes,
    contextual_lookup_digest, integer, object_fields, pack_frame, parse_control, public_lookup_digest,
    session_id, unpack_frame, digest,
)


@dataclass(slots=True)
class _Session:
    admission: BoundaryAdmission
    epoch: int
    lookup: object | None
    last_active: float
    sequence: int = 0
    lookup_rows: int = 0
    heads: int = 0
    busy: bool = False
    terminal: bool = False

    def burn(self):
        self.terminal = True
        self.lookup = None


def create_private_boundary_worker_app(
    *, descriptor: BoundaryDescriptor, head: CompiledMatrix, head_scales: np.ndarray,
    role_id: str, api_key: str, peer_credential_digest: str,
    admit: Callable[[BoundaryDescriptor, BoundaryAdmission], BoundaryAdmission],
    lookup_root=None, max_sessions: int = 8, idle_seconds: float = 90.0,
    head_certificate: dict | None = None,
) -> FastAPI:
    """Bind actual native weights/table to mandatory trusted decoder admission.

    ``admit(actual_descriptor, requested_admission)`` must validate the native
    compiled plan/schedule and return its independently derived exact contract.
    Returning request fields without validation violates this API's contract.
    The actual descriptor must come from the head-only loader's validated source
    lock/checkpoint and native re-lowered config/composition/schedule. This
    service never derives prepared-body identity or receives the peer's secret.
    ``lookup_root`` permits an already verified native table owner; production
    default uses ``PrivatePageServer.from_matrix`` without duplicating weights.
    A missing context API always fails closed.
    """
    from pllm import _native

    if type(descriptor) is not BoundaryDescriptor or role_id not in ROLES or not callable(admit):
        raise BoundaryError("invalid boundary worker configuration")
    if (type(api_key) is not str or not 16 <= len(api_key) <= 512
            or not api_key.isascii() or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in api_key)):
        raise BoundaryError("boundary own credential is invalid")
    own_credential_digest = hashlib.sha256(api_key.encode()).hexdigest()
    digest(peer_credential_digest, "peer credential digest")
    if own_credential_digest == peer_credential_digest:
        raise BoundaryError("boundary credential digests must be distinct")
    credential_digests = {role_id: own_credential_digest,
                          ROLES[1 - ROLES.index(role_id)]: peer_credential_digest}
    key = api_key
    integer(max_sessions, 1, 8, "session capacity")
    if isinstance(idle_seconds, bool) or not isinstance(idle_seconds, (float, int)) or not np.isfinite(idle_seconds) or idle_seconds <= 0:
        raise BoundaryError("invalid idle timeout")
    if type(head) is not CompiledMatrix or head._matrix is None or head.shape != (descriptor.vocabulary, descriptor.hidden):
        raise BoundaryError("boundary requires an immutable native compiled head")
    if descriptor.head_kernel == "byte_dot" and not callable(getattr(head.owner._executor, "wrap32_byte_dot", None)):
        raise BoundaryError("native byte-dot head API unavailable")
    # Hash the immutable native view, without serializing a second matrix.
    weights = head.weight_view()
    if hashlib.sha256(memoryview(weights).cast("B")).hexdigest() != descriptor.head_weight_digest:
        raise BoundaryError("actual head weight identity differs")
    scales = np.asarray(head_scales)
    if scales.dtype != np.float32 or scales.shape != (descriptor.vocabulary,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise BoundaryError("actual head scales are invalid")
    scale_bytes = scales.astype("<f4", copy=False).tobytes()
    if hashlib.sha256(scale_bytes).hexdigest() != descriptor.head_scales_digest:
        raise BoundaryError("actual head scales identity differs")
    lookup_digest = public_lookup_digest(
        ((memoryview(row).cast("B"), memoryview(scale_bytes)[i * 4:i * 4 + 4])
         for i, row in enumerate(weights)), vocabulary=descriptor.vocabulary, hidden=descriptor.hidden)
    if lookup_digest != descriptor.lookup_digest:
        raise BoundaryError("lookup identity is not the actual tied head")
    row_l1 = 0
    for start in range(0, len(weights), 64):
        chunk = weights[start:start + 64].astype(np.int16)
        row_l1 = max(row_l1, int(np.max(np.sum(np.abs(chunk), axis=1, dtype=np.int64))))
    if row_l1 * ((1 << (descriptor.activation_bits - 1)) - 1) > descriptor.signed_output_bound:
        raise BoundaryError("actual head exceeds committed signed bound")
    head_codec = HeadOutputCodec(descriptor, certificate=head_certificate, weights=weights)
    if lookup_root is None:
        constructor = getattr(_native.PrivatePageServer, "from_matrix", None)
        if constructor is None:
            raise BoundaryError("native shared-matrix private lookup API unavailable")
        lookup_root = constructor(head._matrix, scale_bytes, ROLES.index(role_id))
    if type(lookup_root) is not _native.PrivatePageServer:
        raise BoundaryError("boundary requires a native private lookup root")
    if lookup_root.descriptor() != (bytes.fromhex(descriptor.lookup_digest), descriptor.vocabulary, descriptor.record_bytes):
        raise BoundaryError("actual lookup descriptor differs")
    # Applications supplying a prebuilt root must validate its party ownership.
    # The native query evaluator independently rejects the wrong party.
    sessions: dict[str, _Session] = {}
    used_contexts: set[bytes] = set()
    lock = threading.Lock()
    root_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_):
        try:
            yield
        finally:
            with lock:
                for slot in sessions.values():
                    slot.burn()

    app = FastAPI(title="PLLM private boundary", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)

    def reject(message, status=409):
        return HTTPException(status_code=status, detail={"error": {"message": message}})

    def authenticate(authorization):
        if (not isinstance(authorization, str) or not authorization.startswith("Bearer ")
                or not authorization.isascii() or not secrets.compare_digest(authorization[7:], key)):
            raise reject("boundary credential is invalid", 401)

    async def body(request, limit):
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise reject("boundary body exceeds limit", 413)
            chunks.append(chunk)
        return b"".join(chunks)

    def active(sid):
        slot = sessions.get(sid)
        if slot is None or slot.terminal:
            raise BoundaryError("boundary session is terminal or absent")
        if time.monotonic() - slot.last_active > idle_seconds:
            slot.burn()
            raise BoundaryError("boundary session expired")
        slot.last_active = time.monotonic()
        return slot

    async def native_job(slot, function):
        # Cancelling an ASGI await does not stop the native thread. Keep its
        # capacity slot until completion; never recycle a still-running owner.
        task = asyncio.create_task(asyncio.to_thread(function))
        try:
            return await asyncio.shield(task)
        except BaseException:
            def retire(finished):
                try:
                    finished.result()
                except BaseException:
                    pass
                with lock:
                    slot.busy = False
                    slot.burn()
            task.add_done_callback(retire)
            raise

    @app.get("/health")
    async def health():
        return {"status": "ok", "role": role_id}

    @app.post("/v1/private-boundary/sessions")
    async def open_session(request: Request, authorization: str | None = Header(default=None)):
        authenticate(authorization)
        sid = None
        slot = None
        try:
            row = object_fields(parse_control(await body(request, MAX_CONTROL_BYTES)),
                                {"schema", "role", "session", "epoch", "admission", "credential_digests"})
            if row["schema"] != SCHEMA + ".open" or row["role"] != role_id:
                raise BoundaryError("boundary admission role/schema differs")
            if row["credential_digests"] != credential_digests:
                raise BoundaryError("boundary peer credential pairing differs")
            sid = session_id(row["session"])
            epoch = integer(row["epoch"], 0, (1 << 32) - 1, "epoch")
            requested = BoundaryAdmission.from_dict(row["admission"])
            if requested.descriptor_digest != descriptor.digest:
                raise BoundaryError("boundary admission descriptor differs")
            admitted = await asyncio.to_thread(admit, descriptor, requested)
            if type(admitted) is not BoundaryAdmission or admitted != requested:
                raise BoundaryError("compiled boundary admission differs")
            context = context_bytes(descriptor, admitted, sid, epoch)
            with lock:
                now = time.monotonic()
                for old_id, old in list(sessions.items()):
                    if old.terminal or now - old.last_active > idle_seconds:
                        old.burn()
                        if not old.busy:
                            del sessions[old_id]
                if sid in sessions or context in used_contexts or len(used_contexts) >= 4096:
                    raise BoundaryError("boundary session/context replayed or exhausted")
                if len(sessions) >= max_sessions:
                    raise BoundaryError("boundary session capacity exceeded")
                used_contexts.add(context)  # Never refund a reserved context.
                slot = _Session(admitted, epoch, None, now, busy=True)
                sessions[sid] = slot

            def bind():
                with root_lock:
                    binder = getattr(lookup_root, "bind_context", None)
                    if binder is None:
                        raise BoundaryError("native context-bound lookup API unavailable")
                    child = binder(context)
                    if child.descriptor() != (contextual_lookup_digest(descriptor, context), descriptor.vocabulary, descriptor.record_bytes):
                        raise BoundaryError("native context-bound descriptor differs")
                    return child

            child = await native_job(slot, bind)
            with lock:
                slot.busy = False
                if slot.terminal:
                    raise BoundaryError("boundary admission was cancelled")
                slot.lookup = child
            return {"schema": SCHEMA + ".opened", "role": role_id, "session": sid,
                    "epoch": epoch, "admission": admitted.to_dict(),
                    "credential_digests": dict(credential_digests),
                    "lookup_digest": contextual_lookup_digest(descriptor, context).hex()}
        except BaseException as exc:
            with lock:
                if slot is not None:
                    slot.burn()
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, (ValueError, TypeError, KeyError)):
                raise reject("boundary admission rejected", 400) from exc
            raise

    @app.post("/v1/private-boundary/sessions/{sid}/execute")
    async def execute(sid: str, request: Request, authorization: str | None = Header(default=None)):
        authenticate(authorization)
        slot = None
        try:
            # Find the owner before parsing, so malformed/oversized frames burn it.
            with lock:
                slot = active(sid)
            row = unpack_frame(await body(request, MAX_BODY_BYTES))
            with lock:
                active(sid)
                if (slot.busy or row["session"] != sid or row["role"] != role_id
                        or row["epoch"] != slot.epoch or row["sequence"] != slot.sequence):
                    raise BoundaryError("boundary operation ownership/sequence differs")
                count = len(row["frames"])
                if row["operation"] == "lookup":
                    if any(not frame.startswith(PACKED_QUERY_MAGIC) for frame in row["frames"]):
                        raise BoundaryError("boundary requires packed128 native query frames")
                    if slot.lookup_rows + count > slot.admission.lookup_rows:
                        raise BoundaryError("boundary lookup budget exhausted")
                    slot.lookup_rows += count
                else:
                    if slot.heads >= slot.admission.head_calls or len(row["frames"][0]) != descriptor.hidden * 4:
                        raise BoundaryError("boundary head geometry/budget differs")
                    slot.heads += 1
                slot.sequence += 1  # Consume before native work; no refund.
                slot.busy = True
                child = slot.lookup

            def evaluate():
                if row["operation"] == "lookup":
                    return child.evaluate_batch(row["frames"])
                share = np.frombuffer(row["frames"][0], "<u4").reshape(1, descriptor.hidden)
                result = (head.wrap32_byte_dot(share) if descriptor.head_kernel == "byte_dot"
                          else head.wrap32(share))
                if result.shape != (1, descriptor.vocabulary) or result.dtype != np.uint32:
                    raise BoundaryError("native head result geometry differs")
                return [head_codec.encode(result.astype("<u4", copy=False).tobytes(),
                    admission=slot.admission, role=role_id, paired_session=sid,
                    epoch=slot.epoch, sequence=row["sequence"])]

            frames = await native_job(slot, evaluate)
            with lock:
                slot.busy = False
                if slot.terminal:
                    raise BoundaryError("boundary cancelled during evaluation")
                slot.last_active = time.monotonic()
            result = pack_frame(role=role_id, paired_session=sid, epoch=slot.epoch,
                                sequence=row["sequence"], operation=row["operation"],
                                frames=frames, response=True)
            return Response(result, media_type="application/octet-stream")
        except BaseException as exc:
            with lock:
                if slot is not None:
                    slot.burn()
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, (ValueError, TypeError, KeyError)):
                raise reject("boundary operation rejected", 400) from exc
            raise

    @app.post("/v1/private-boundary/sessions/{sid}/{action}")
    async def finish(sid: str, action: str, request: Request,
                     authorization: str | None = Header(default=None)):
        authenticate(authorization)
        slot = None
        try:
            with lock:
                slot = active(sid)
            row = object_fields(parse_control(await body(request, MAX_CONTROL_BYTES)),
                                {"schema", "role", "session", "epoch"})
            if (action not in ("complete", "cancel") or canonical(row) != canonical({
                "schema": SCHEMA + "." + action, "role": role_id,
                "session": sid, "epoch": slot.epoch,
            })):
                raise BoundaryError("boundary finish ownership differs")
            with lock:
                if action == "complete" and (slot.busy or slot.terminal):
                    raise BoundaryError("boundary has pending work")
                slot.burn()
            return {"schema": SCHEMA + ".finished", "role": role_id, "session": sid,
                    "epoch": slot.epoch, "status": action}
        except BaseException as exc:
            with lock:
                if slot is not None:
                    slot.burn()
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, (ValueError, TypeError)):
                raise reject("boundary finish rejected", 400) from exc
            raise

    return app
