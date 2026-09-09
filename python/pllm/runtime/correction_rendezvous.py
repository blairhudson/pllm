from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass

from .preparation_protocol import CorrectionPush, SessionAuthorization, validate_attempt_id
from .protocol import ProtocolError
from .stage_protocol import MaskedStageRequest


class RendezvousError(ProtocolError):
    pass


@dataclass(slots=True)
class _Entry:
    created: float
    correction_future: concurrent.futures.Future[CorrectionPush]
    correction: CorrectionPush | None = None
    correction_bytes: int = 0
    activated: bool = False
    activation_metadata: tuple[object, ...] | None = None


@dataclass(slots=True)
class _Session:
    expected: SessionAuthorization
    burned: set[bytes]
    last_active: float
    authorized: bool = False


class CorrectionRendezvous:
    """Bounded authorized-session correction rendezvous."""

    def __init__(
        self,
        *,
        timeout: float,
        capacity: int,
        max_bytes: int,
        session_capacity: int | None = None,
        session_idle_timeout: float = 300.0,
    ) -> None:
        session_capacity = capacity if session_capacity is None else session_capacity
        if (
            timeout <= 0
            or capacity <= 0
            or max_bytes <= 0
            or session_capacity <= 0
            or session_idle_timeout <= 0
        ):
            raise ValueError("rendezvous limits must be positive")
        self.timeout = float(timeout)
        self.capacity = int(capacity)
        self.max_bytes = int(max_bytes)
        self.session_capacity = int(session_capacity)
        self.session_idle_timeout = float(session_idle_timeout)
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, bytes], _Entry] = {}
        self._sessions: dict[str, _Session] = {}
        self._bytes = 0
        self.authorizations = 0
        self.authorization_bytes = 0
        self.attempts = 0
        self.consumed = 0
        self.failures = 0

    @staticmethod
    def _key(session_id: str, attempt_id: str) -> tuple[str, bytes]:
        validate_attempt_id(attempt_id)
        return session_id, bytes.fromhex(attempt_id)

    def register_session(self, authorization: SessionAuthorization) -> None:
        authorization._validate()
        with self._lock:
            now = time.monotonic()
            self._cleanup_locked(now)
            if authorization.session_id in self._sessions:
                raise RendezvousError("correction session is already registered")
            if len(self._sessions) >= self.session_capacity:
                raise RendezvousError("correction session capacity exceeded")
            self._sessions[authorization.session_id] = _Session(
                authorization, set(), now
            )

    def authorize_session(
        self,
        authorization: SessionAuthorization,
        payload_bytes: int,
    ) -> None:
        authorization._validate()
        with self._lock:
            now = time.monotonic()
            self._cleanup_locked(now)
            session = self._sessions.get(authorization.session_id)
            if session is None:
                raise RendezvousError("correction session is not active")
            if authorization != session.expected:
                raise RendezvousError("session authorization metadata mismatch")
            if session.authorized:
                raise RendezvousError("session authorization was already consumed")
            if payload_bytes <= 0:
                raise RendezvousError("session authorization payload is empty")
            session.authorized = True
            session.last_active = now
            self.authorizations += 1
            self.authorization_bytes += payload_bytes

    def _cancel_entry_locked(self, entry: _Entry) -> None:
        self._bytes -= entry.correction_bytes
        entry.correction_future.cancel()

    def _burn_locked(self, key: tuple[str, bytes], *, failure: bool = True) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._cancel_entry_locked(entry)
        session = self._sessions.get(key[0])
        if session is not None:
            session.burned.add(key[1])
        if failure:
            self.failures += 1

    def _cleanup_locked(self, now: float) -> None:
        cutoff = now - self.timeout
        for key, entry in list(self._entries.items()):
            if entry.created <= cutoff:
                self._burn_locked(key)
        session_cutoff = now - self.session_idle_timeout
        for session_id, session in list(self._sessions.items()):
            if session.last_active <= session_cutoff:
                self._terminal_locked(session_id)

    def _entry_locked(self, key: tuple[str, bytes], now: float) -> _Entry:
        self._cleanup_locked(now)
        session = self._sessions.get(key[0])
        if session is None:
            raise RendezvousError("correction session is not active")
        if not session.authorized:
            raise RendezvousError("correction session is not authorized")
        session.last_active = now
        if key[1] in session.burned:
            raise RendezvousError("attempt was already consumed or burned")
        entry = self._entries.get(key)
        if entry is None:
            active_for_session = sum(item[0] == key[0] for item in self._entries)
            if len(self._entries) >= self.capacity:
                self.failures += 1
                raise RendezvousError("correction rendezvous capacity exceeded")
            if len(session.burned) + active_for_session >= session.expected.max_attempts:
                self.failures += 1
                raise RendezvousError("session attempt capacity exceeded")
            entry = _Entry(
                now,
                concurrent.futures.Future(),
            )
            self._entries[key] = entry
            self.attempts += 1
        return entry

    def push(self, correction: CorrectionPush, payload_bytes: int) -> None:
        key = self._key(correction.session_id, correction.attempt_id)
        now = time.monotonic()
        with self._lock:
            entry = self._entry_locked(key, now)
            if entry.activated and correction.metadata() != entry.activation_metadata:
                self._burn_locked(key)
                raise RendezvousError("correction and activation metadata mismatch")
            if entry.correction is not None:
                self._burn_locked(key)
                raise RendezvousError("duplicate correction push")
            if payload_bytes <= 0 or self._bytes + payload_bytes > self.max_bytes:
                self._burn_locked(key)
                raise RendezvousError("correction rendezvous byte capacity exceeded")
            entry.correction = correction
            entry.correction_bytes = payload_bytes
            self._bytes += payload_bytes
            try:
                entry.correction_future.set_result(correction)
            except concurrent.futures.InvalidStateError as exc:
                self._burn_locked(key)
                raise RendezvousError("correction arrived after attempt cancellation") from exc

    @staticmethod
    def _request_metadata(request: MaskedStageRequest) -> tuple[object, ...]:
        rows = 1 if request.masked_input.ndim == 1 else request.masked_input.shape[0]
        return (
            request.session_id, request.model, request.body_fingerprint,
            request.correlation_id, request.stage_id, request.weight_digest,
            rows, request.masked_input.shape[-1], request.out_features,
            request.weight_bits, request.activation_bits,
            request.signed_output_bound, request.ring, request.modulus,
            request.wire_bits,
        )

    def activate(self, request: MaskedStageRequest) -> _Entry:
        key = self._key(request.session_id, request.correlation_id)
        now = time.monotonic()
        metadata = self._request_metadata(request)
        with self._lock:
            entry = self._entry_locked(key, now)
            session = self._sessions[request.session_id]
            if (
                request.model != session.expected.model
                or request.body_fingerprint != session.expected.body_fingerprint
            ):
                self._burn_locked(key)
                raise RendezvousError("activation and session authorization metadata mismatch")
            if entry.activated:
                self._burn_locked(key)
                raise RendezvousError("duplicate activation attempt")
            entry.activated = True
            entry.activation_metadata = metadata
            if entry.correction is not None and entry.correction.metadata() != metadata:
                self._burn_locked(key)
                raise RendezvousError("correction and activation metadata mismatch")
        return entry

    async def correction(self, request: MaskedStageRequest, entry: _Entry) -> CorrectionPush:
        key = self._key(request.session_id, request.correlation_id)
        with self._lock:
            if self._entries.get(key) is not entry:
                raise RendezvousError("attempt was concurrently burned")
            future = entry.correction_future
            remaining = max(0.0, self.timeout - (time.monotonic() - entry.created))
        try:
            correction = await asyncio.wait_for(asyncio.wrap_future(future), remaining)
            if correction.metadata() != self._request_metadata(request):
                raise RendezvousError("correction and activation metadata mismatch")
        except asyncio.TimeoutError as exc:
            self.abort(request, entry)
            raise RendezvousError("correction rendezvous timed out") from exc
        except asyncio.CancelledError as exc:
            self.abort(request, entry)
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            raise RendezvousError("activation attempt cancelled") from exc
        except RendezvousError:
            self.abort(request, entry)
            raise
        with self._lock:
            if self._entries.get(key) is not entry:
                raise RendezvousError("attempt was concurrently burned")
            self._burn_locked(key, failure=False)
            self.consumed += 1
        return correction

    def abort(self, request: MaskedStageRequest, entry: _Entry) -> None:
        key = self._key(request.session_id, request.correlation_id)
        with self._lock:
            if self._entries.get(key) is entry:
                self._burn_locked(key)

    def terminal(self, session_id: str) -> None:
        with self._lock:
            self._terminal_locked(session_id)

    def _terminal_locked(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        for key, entry in list(self._entries.items()):
            if key[0] == session_id:
                self._entries.pop(key)
                self._cancel_entry_locked(entry)
                self.failures += 1

    def close(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                self._cancel_entry_locked(entry)
            self._entries.clear()
            self._sessions.clear()
            self._bytes = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._cleanup_locked(time.monotonic())
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "sessions": len(self._sessions),
                "authorized_sessions": sum(
                    session.authorized for session in self._sessions.values()
                ),
                "authorization_bytes": self.authorization_bytes,
                "authorizations": self.authorizations,
                "burned": sum(len(session.burned) for session in self._sessions.values()),
                "attempts": self.attempts,
                "consumed": self.consumed,
                "failures": self.failures,
            }
