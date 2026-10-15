"""Client-owned two-party boundary sessions, logits and exact reconstruction."""

from __future__ import annotations

import hashlib
import ipaddress
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import httpx
import numpy as np

from .http_response import owned_response
from .private_boundary_head_codec import HeadOutputCodec
from .private_boundary_protocol import (
    BoundaryAdmission, BoundaryDescriptor, BoundaryError, MAX_BODY_BYTES,
    MAX_CONTROL_BYTES, ROLES, SCHEMA, canonical, context_bytes,
    contextual_lookup_digest, integer, object_fields, pack_frame, parse_control,
    unpack_frame,
)
from .quantization import dequantize_matmul, quantize_activation_per_row


def _origin(value: str) -> tuple:
    url = urlsplit(value)
    try:
        loopback = ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        loopback = url.hostname == "localhost"
    if (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
            or url.password is not None or url.query or url.fragment or url.path not in ("", "/")
            or (url.scheme == "http" and not loopback)):
        raise BoundaryError("boundary URL requires HTTPS outside loopback and a bare origin")
    return url.scheme, url.hostname, url.port or (443 if url.scheme == "https" else 80)


class PrivateBoundaryTransport:
    """Explicitly opened response-owned boundary, separate from prepared body.

    HTTP clients passed by callers remain caller-owned. URL arguments allocate
    owned clients. No token IDs, activation scales, clear activations or full
    reconstructed logits appear in worker requests. Version one is bias-free.
    """

    def __init__(
        self, *, worker_a: httpx.Client | str, worker_b: httpx.Client | str,
        api_key_a: str, api_key_b: str, descriptor: BoundaryDescriptor,
        head_scales: np.ndarray, bias: np.ndarray | None = None, timeout: float = 30.0,
    ):
        if type(descriptor) is not BoundaryDescriptor:
            raise BoundaryError("boundary requires an immutable descriptor")
        if any(type(k) is not str or not 16 <= len(k) <= 512 or not k.isascii()
               or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in k)
               for k in (api_key_a, api_key_b)) or api_key_a == api_key_b:
            raise BoundaryError("boundary credentials must be distinct")
        if bias is not None:
            raise BoundaryError("boundary v1 requires a bias-free head")
        if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not np.isfinite(timeout) or timeout <= 0:
            raise BoundaryError("invalid boundary timeout")
        scales = np.asarray(head_scales)
        if scales.dtype != np.float32 or scales.shape != (descriptor.vocabulary,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise BoundaryError("invalid public head scales")
        raw_scales = scales.astype("<f4", copy=False).tobytes()
        if hashlib.sha256(raw_scales).hexdigest() != descriptor.head_scales_digest:
            raise BoundaryError("public head scales commitment differs")
        arguments = (worker_a, worker_b)
        if any(type(c) not in (httpx.Client, str) for c in arguments) or worker_a is worker_b:
            raise BoundaryError("boundary requires distinct clients/origins")
        origins = [_origin(str(c.base_url) if type(c) is httpx.Client else c) for c in arguments]
        if origins[0] == origins[1]:
            raise BoundaryError("boundary requires distinct origins")
        self.descriptor = descriptor
        self._head_codec = (HeadOutputCodec(descriptor)
                            if descriptor.head_output_encoding == "uniform_residues" else None)
        self._scales = np.frombuffer(raw_scales, "<f4")  # immutable private copy
        self._owned = tuple(type(c) is str for c in arguments)
        clients = []
        try:
            for argument in arguments:
                clients.append(argument if type(argument) is httpx.Client else
                               httpx.Client(base_url=argument, timeout=timeout))
        except BaseException:
            for client, owned in zip(clients, self._owned):
                if owned:
                    client.close()
            raise
        self._clients = tuple(clients)
        self._keys = (api_key_a, api_key_b)
        self._credential_digests = {role: hashlib.sha256(key.encode()).hexdigest()
                                    for role, key in zip(ROLES, self._keys, strict=True)}
        self._timeout = timeout
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._finish_lock = threading.Lock()
        self._state = "new"
        self._retired = False
        self._completed = False
        self._attempted: set[int] = set()
        self._admission = None
        self._session = None
        self._epoch = 0
        self._sequence = 0
        self._lookup_rows = self._heads = 0
        self._lookup_digest = None
        self._bodies = {role: {f"{phase}_{direction}_bytes": 0
                              for phase in ("setup", "online", "teardown")
                              for direction in ("upload", "download")} for role in ROLES}
        self._coverage_complete = True
        self._diagnostics = {op: {"calls": 0, "rows": 0, "upload_bytes": 0, "download_bytes": 0}
                             for op in ("lookup", "head")}

    @property
    def session_id(self) -> str | None:
        """Shared opaque client session, assigned only by explicit open()."""
        with self._state_lock:
            return self._session

    @property
    def admission(self) -> BoundaryAdmission | None:
        with self._state_lock:
            return self._admission

    @property
    def role_link_bodies(self) -> dict[str, dict[str, int]]:
        with self._state_lock:
            return {role: dict(values) for role, values in self._bodies.items()}

    @property
    def body_coverage_complete(self) -> bool:
        with self._state_lock:
            return self._coverage_complete

    @property
    def operation_counters(self) -> dict[str, dict[str, int]]:
        with self._state_lock:
            return {op: dict(values) for op, values in self._diagnostics.items()}

    def _post(self, index, path, payload, phase, *, limit, operation=None):
        role = ROLES[index]
        with self._state_lock:
            self._bodies[role][phase + "_upload_bytes"] += len(payload)
            if operation is not None:
                self._diagnostics[operation]["upload_bytes"] += len(payload)
        size = 0
        chunks = []
        try:
            with self._clients[index].stream("POST", path, content=payload, timeout=self._timeout,
                    follow_redirects=False,
                    headers={"authorization": "Bearer " + self._keys[index],
                             "accept-encoding": "identity", "content-type": "application/octet-stream"}) as response, owned_response(response):
                if response.history or response.headers.get("content-encoding", "identity") != "identity":
                    raise BoundaryError("boundary redirects/compressed replies are not admitted")
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    with self._state_lock:
                        self._bodies[role][phase + "_download_bytes"] += len(chunk)
                        if operation is not None:
                            self._diagnostics[operation]["download_bytes"] += len(chunk)
                    if size > limit:
                        raise BoundaryError("boundary response exceeds limit")
                    chunks.append(chunk)
                return response.status_code, b"".join(chunks)
        except BaseException:
            # Attempted uploads and observed partial chunks stay charged. Delivery
            # and unobserved tails are unknown, never promoted to a complete ledger.
            with self._state_lock:
                self._coverage_complete = False
            raise

    def open(self, admission: BoundaryAdmission, *, epoch: int = 0) -> PrivateBoundaryTransport:
        """Open both sessions; rollback both attempted roles on any partial failure."""
        if type(admission) is not BoundaryAdmission:
            raise BoundaryError("boundary requires an immutable admission")
        integer(epoch, 0, (1 << 32) - 1, "epoch")
        if admission.descriptor_digest != self.descriptor.digest:
            raise BoundaryError("admission descriptor differs")
        with self._lock:
            with self._state_lock:
                if self._state != "new":
                    raise BoundaryError("boundary session cannot reopen")
                self._state = "opening"
                self._admission, self._epoch = admission, epoch
                self._session = secrets.token_hex(16)
            context = context_bytes(self.descriptor, admission, self._session, epoch)
            self._lookup_digest = contextual_lookup_digest(self.descriptor, context)
            try:
                for index, role in enumerate(ROLES):
                    with self._state_lock:
                        if self._state != "opening":
                            raise BoundaryError("boundary admission cancelled")
                    self._attempted.add(index)
                    status, raw = self._post(index, "/v1/private-boundary/sessions", canonical({
                        "schema": SCHEMA + ".open", "role": role, "session": self._session,
                        "epoch": epoch, "admission": admission.to_dict(),
                        "credential_digests": dict(self._credential_digests),
                    }), "setup", limit=MAX_CONTROL_BYTES)
                    expected = {"schema": SCHEMA + ".opened", "role": role, "session": self._session,
                                "epoch": epoch, "admission": admission.to_dict(),
                                "credential_digests": dict(self._credential_digests),
                                "lookup_digest": self._lookup_digest.hex()}
                    if status != 200 or canonical(parse_control(raw)) != canonical(expected):
                        raise BoundaryError("boundary worker admission differs")
                with self._state_lock:
                    if self._state != "opening":
                        raise BoundaryError("boundary admission cancelled")
                    self._state = "active"
                return self
            except BaseException:
                self.abort()
                raise

    def _active(self):
        with self._state_lock:
            if self._state != "active":
                raise BoundaryError("boundary session is not active")

    def _exchange(self, operation, frames_a, frames_b):
        self._active()
        sequence = self._sequence
        self._sequence += 1
        with self._state_lock:
            self._diagnostics[operation]["calls"] += 1
            self._diagnostics[operation]["rows"] += len(frames_a)

        def exchange_one(index, frames):
            payload = pack_frame(role=ROLES[index], paired_session=self._session, epoch=self._epoch,
                                 sequence=sequence, operation=operation, frames=frames)
            status, raw = self._post(index, f"/v1/private-boundary/sessions/{self._session}/execute",
                                     payload, "online", limit=MAX_BODY_BYTES, operation=operation)
            if status != 200:
                raise BoundaryError("boundary worker refused operation")
            row = unpack_frame(raw, response=True)
            if (row["role"] != ROLES[index] or row["session"] != self._session
                    or row["epoch"] != self._epoch or row["sequence"] != sequence
                    or row["operation"] != operation or len(row["frames"]) != len(frames)):
                raise BoundaryError("boundary reply ownership differs")
            return row["frames"]

        if operation == "head" and self.descriptor.head_dispatch == "parallel":
            # One admitted current-step pair, never speculative work. The caller
            # holds _lock until both futures drain, including submission failure,
            # malformed replies and BaseException. Consequently abort/close cannot
            # retire sessions or close owned clients beneath a pending request.
            # Per-pair ownership also leaves no idle executor after an exchange.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pllm-boundary-head") as owner:
                pending = [owner.submit(exchange_one, index, frames)
                           for index, frames in enumerate((frames_a, frames_b))]
                replies = [future.result() for future in pending]
        else:
            replies = []
            for index, frames in enumerate((frames_a, frames_b)):
                self._active()
                replies.append(exchange_one(index, frames))
        self._active()  # Cancellation during a blocking exchange cannot deliver.
        return replies

    def token_lookup(self, ids) -> np.ndarray:
        from pllm import _native

        with self._lock:
            try:
                self._active()
                indices = np.asarray(ids)
                if indices.ndim != 1 or indices.dtype.kind not in ("i", "u") or not len(indices):
                    raise BoundaryError("invalid token lookup indices")
                if np.any(indices < 0) or np.any(indices >= self.descriptor.vocabulary):
                    raise BoundaryError("token ID outside vocabulary")
                if self._lookup_rows + len(indices) > self._admission.lookup_rows:
                    raise BoundaryError("boundary lookup budget exhausted")
                self._lookup_rows += len(indices)
                result = np.empty((len(indices), self.descriptor.hidden), np.float32)
                for start in range(0, len(indices), 64):
                    queries = []
                    try:
                        for index in indices[start:start + 64]:
                            queries.append(_native.private_page_issue(self._lookup_digest,
                                self.descriptor.vocabulary, self.descriptor.record_bytes, int(index), leaf_bits=128))
                        replies = self._exchange("lookup", [q[0] for q in queries], [q[1] for q in queries])
                        for offset, query in enumerate(queries):
                            raw = query[2].decode(replies[0][offset], replies[1][offset])
                            if len(raw) != self.descriptor.record_bytes:
                                raise BoundaryError("lookup record width differs")
                            scale = np.frombuffer(raw, "<f4", count=1, offset=self.descriptor.hidden)[0]
                            if not np.isfinite(scale) or scale <= 0 or scale.tobytes() != self._scales[int(indices[start + offset])].tobytes():
                                raise BoundaryError("lookup record scale differs")
                            # Preserve tied local_token_lookup float32 operation ordering.
                            result[start + offset] = np.frombuffer(raw, np.int8, count=self.descriptor.hidden).astype(np.float32) * scale
                    finally:
                        for query in queries:
                            query[2].cancel()
                self._active()
                return result
            except BaseException:
                self.abort()
                raise

    def linear(self, stage_id: str, x: np.ndarray) -> np.ndarray:
        with self._lock:
            left = right = None
            try:
                self._active()
                value = np.asarray(x)
                if (stage_id != self.descriptor.head_stage_id or value.dtype != np.float32
                        or value.ndim not in (1, 2, 3) or value.shape[-1] != self.descriptor.hidden
                        or value.size != self.descriptor.hidden or not np.all(np.isfinite(value))):
                    raise BoundaryError("boundary head requires one finite float32 row")
                if self._heads >= self._admission.head_calls:
                    raise BoundaryError("boundary head budget exhausted")
                self._heads += 1
                quantized = quantize_activation_per_row(value, bits=self.descriptor.activation_bits)
                clear = quantized.values.astype(np.int64) % (1 << 32)
                right = np.frombuffer(bytearray(secrets.token_bytes(self.descriptor.hidden * 4)), "<u4").reshape(1, -1)
                left = np.asarray((clear - right.astype(np.int64)) % (1 << 32), dtype="<u4")
                sequence = self._sequence
                replies = self._exchange("head", [left.tobytes()], [right.tobytes()])
                if self._head_codec is not None:
                    centered = self._head_codec.decode(replies[0][0], replies[1][0],
                        admission=self._admission, paired_session=self._session,
                        epoch=self._epoch, sequence=sequence)
                else:
                    if any(len(r[0]) != self.descriptor.vocabulary * 4 for r in replies):
                        raise BoundaryError("boundary head reply width differs")
                    a, b = (np.frombuffer(r[0], "<u4").astype(np.int64).reshape(1, -1) for r in replies)
                    combined = (a + b) % (1 << 32)
                    centered = np.where(combined >= 1 << 31, combined - (1 << 32), combined)
                if np.any(np.abs(centered) > self.descriptor.signed_output_bound):
                    raise BoundaryError("boundary head exceeds signed output bound")
                result = dequantize_matmul(centered, quantized.scales, self._scales,
                                          output_shape=quantized.original_shape[:-1] + (self.descriptor.vocabulary,))
                if not np.all(np.isfinite(result)):
                    raise BoundaryError("boundary head produced nonfinite logits")
                self._active()
                return np.ascontiguousarray(result, dtype=np.float32)
            except BaseException:
                self.abort()
                raise
            finally:
                if left is not None:
                    left.fill(0)
                if right is not None:
                    right.fill(0)

    def _finish(self, action):
        with self._state_lock:
            if self._retired:
                if action == "complete" and not self._completed:
                    raise BoundaryError("boundary completion requires an active session")
                return
            if action == "complete" and self._state != "active":
                raise BoundaryError("boundary completion requires an active session")
            self._state = "closed"  # Signal cancellation before draining active call.
        with self._lock, self._finish_lock:
            if self._retired:
                return
            failures = False
            for index in sorted(self._attempted):
                role = ROLES[index]
                payload = canonical({"schema": SCHEMA + "." + action, "role": role,
                                     "session": self._session, "epoch": self._epoch})
                try:
                    status, raw = self._post(index, f"/v1/private-boundary/sessions/{self._session}/{action}",
                                             payload, "teardown", limit=MAX_CONTROL_BYTES)
                    expected = {"schema": SCHEMA + ".finished", "role": role,
                                "session": self._session, "epoch": self._epoch, "status": action}
                    if status != 200 or canonical(parse_control(raw)) != canonical(expected):
                        failures = True
                except Exception:
                    failures = True
            with self._state_lock:
                self._completed = action == "complete" and not failures
                self._retired = True
            if failures and action == "complete":
                raise BoundaryError("boundary completion failed")

    def complete(self):
        # Completion cannot race an operation and turn its cancellation into a
        # successful acknowledgement. Abort deliberately signals before draining.
        if not self._lock.acquire(blocking=False):
            raise BoundaryError("boundary completion has pending work")
        try:
            self._finish("complete")
        finally:
            self._lock.release()

    def abort(self):
        self._finish("cancel")

    def close(self):
        try:
            self.abort()
        finally:
            for client, owned in zip(self._clients, self._owned, strict=True):
                if owned:
                    client.close()
            self._head_codec = None

    def __enter__(self):
        self._active()
        return self

    def __exit__(self, *_):
        self.close()
