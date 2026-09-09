from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Callable, Protocol

import msgpack

from .shared_mpc import SharedMPCError


class SharedSessionError(SharedMPCError):
    """Raised when a shared-transformer session violates its commitment."""


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ABORTED = "aborted"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class SharedSessionCommitment:
    session_id: str
    model_id: str
    model_fingerprint: str
    graph_fingerprint: str
    scale_fingerprint: str
    party_fingerprints: tuple[str, str]
    maximum_operations: int

    def __post_init__(self) -> None:
        if not self.session_id or not self.model_id:
            raise SharedSessionError("shared session identity is invalid")
        if not isinstance(self.party_fingerprints, tuple) or len(self.party_fingerprints) != 2:
            raise SharedSessionError("shared session party fingerprints are invalid")
        if self.party_fingerprints[0] == self.party_fingerprints[1]:
            raise SharedSessionError("shared session party fingerprints must be distinct")
        for value in (
            self.model_fingerprint,
            self.graph_fingerprint,
            self.scale_fingerprint,
            *self.party_fingerprints,
        ):
            if len(value) != 64:
                raise SharedSessionError("shared session fingerprint is invalid")
        if type(self.maximum_operations) is not int or self.maximum_operations <= 0:
            raise SharedSessionError("shared session operation budget is invalid")

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "v": 1,
                "s": self.session_id,
                "m": self.model_id,
                "f": bytes.fromhex(self.model_fingerprint),
                "g": bytes.fromhex(self.graph_fingerprint),
                "c": bytes.fromhex(self.scale_fingerprint),
                "p": tuple(bytes.fromhex(value) for value in self.party_fingerprints),
                "n": self.maximum_operations,
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "SharedSessionCommitment":
        if not payload or len(payload) > 4096:
            raise SharedSessionError("shared session commitment is empty or too large")
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=True)
            if not isinstance(value, dict) or set(value) != {
                "v",
                "s",
                "m",
                "f",
                "g",
                "c",
                "p",
                "n",
            }:
                raise SharedSessionError("shared session commitment fields are invalid")
            if type(value["v"]) is not int or value["v"] != 1:
                raise SharedSessionError("shared session commitment version is unsupported")
            fingerprints = []
            for key in ("f", "g", "c"):
                raw = value[key]
                if type(raw) is not bytes or len(raw) != 32:
                    raise SharedSessionError("shared session fingerprint is invalid")
                fingerprints.append(raw.hex())
            parties = value["p"]
            if (
                not isinstance(parties, (list, tuple))
                or len(parties) != 2
                or any(type(raw) is not bytes or len(raw) != 32 for raw in parties)
            ):
                raise SharedSessionError("shared session party fingerprints are invalid")
            if type(value["s"]) is not str or type(value["m"]) is not str:
                raise SharedSessionError("shared session identity is invalid")
            if type(value["n"]) is not int:
                raise SharedSessionError("shared session operation budget is invalid")
            return cls(
                session_id=value["s"],
                model_id=value["m"],
                model_fingerprint=fingerprints[0],
                graph_fingerprint=fingerprints[1],
                scale_fingerprint=fingerprints[2],
                party_fingerprints=(parties[0].hex(), parties[1].hex()),
                maximum_operations=value["n"],
            )
        except (KeyError, TypeError, ValueError, msgpack.ExtraData, msgpack.FormatError) as exc:
            raise SharedSessionError("shared session commitment is malformed") from exc

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.pack()).hexdigest()


@dataclass(frozen=True, slots=True)
class PeerFrame:
    session_id: str
    channel_id: bytes
    sender_party: int
    sequence: int
    kind: str
    operation_id: str
    payload: bytes

    def __post_init__(self) -> None:
        if not self.session_id or not self.kind or not self.operation_id:
            raise SharedSessionError("peer frame identity is invalid")
        if len(self.session_id.encode()) > 128:
            raise SharedSessionError("peer frame session identifier is invalid")
        if len(self.kind.encode()) > 64:
            raise SharedSessionError("peer frame kind is invalid")
        if len(self.operation_id.encode()) > 128:
            raise SharedSessionError("peer frame operation identifier is invalid")
        if type(self.channel_id) is not bytes or len(self.channel_id) != 32:
            raise SharedSessionError("peer frame channel binding is invalid")
        if type(self.sender_party) is not int or self.sender_party not in (0, 1):
            raise SharedSessionError("peer frame sender is invalid")
        if type(self.sequence) is not int or self.sequence < 0 or self.sequence >= 2**64:
            raise SharedSessionError("peer frame sequence is invalid")
        if not isinstance(self.payload, bytes):
            raise SharedSessionError("peer frame payload must be bytes")

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "v": 1,
                "s": self.session_id,
                "c": self.channel_id,
                "r": self.sender_party,
                "q": self.sequence,
                "k": self.kind,
                "o": self.operation_id,
                "p": self.payload,
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes, *, max_payload_bytes: int) -> "PeerFrame":
        if not payload or len(payload) > max_payload_bytes + 4096:
            raise SharedSessionError("peer frame is empty or too large")
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=True)
            if not isinstance(value, dict) or set(value) != {
                "v",
                "s",
                "c",
                "r",
                "q",
                "k",
                "o",
                "p",
            }:
                raise SharedSessionError("peer frame fields are invalid")
            if type(value["v"]) is not int or value["v"] != 1:
                raise SharedSessionError("peer frame version is unsupported")
            if (
                type(value["s"]) is not str
                or type(value["c"]) is not bytes
                or type(value["r"]) is not int
                or type(value["q"]) is not int
                or type(value["k"]) is not str
                or type(value["o"]) is not str
                or type(value["p"]) is not bytes
            ):
                raise SharedSessionError("peer frame field types are invalid")
            frame = cls(
                session_id=value["s"],
                channel_id=value["c"],
                sender_party=value["r"],
                sequence=value["q"],
                kind=value["k"],
                operation_id=value["o"],
                payload=value["p"],
            )
            if len(frame.payload) > max_payload_bytes:
                raise SharedSessionError("peer frame payload is too large")
            return frame
        except (KeyError, TypeError, ValueError, msgpack.ExtraData, msgpack.FormatError) as exc:
            raise SharedSessionError("peer frame is malformed") from exc


class SharedSessionState:
    def __init__(
        self,
        commitment: SharedSessionCommitment,
        *,
        party: int,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if party not in (0, 1):
            raise ValueError("shared session party must be zero or one")
        self.commitment = commitment
        self.party = party
        self.status = SessionStatus.ACTIVE
        self.abort_reason: str | None = None
        self._now = now
        self.last_activity = now()
        self._next_outbound = 0
        self._next_inbound = 0
        self._operations: set[str] = set()
        self._channel_id: bytes | None = None
        self._lock = Lock()

    def establish(
        self,
        peer_commitment: SharedSessionCommitment,
        *,
        local_nonce: bytes,
        peer_nonce: bytes,
        authenticated_peer_fingerprint: str,
    ) -> None:
        with self._lock:
            self._require_active()
            if self._channel_id is not None:
                self._abort_locked("shared session handshake was repeated")
            elif peer_commitment != self.commitment:
                self._abort_locked("peer shared session commitment does not match")
            elif (
                authenticated_peer_fingerprint != self.commitment.party_fingerprints[1 - self.party]
            ):
                self._abort_locked("authenticated peer fingerprint does not match commitment")
            elif len(local_nonce) != 32 or len(peer_nonce) != 32 or local_nonce == peer_nonce:
                self._abort_locked("shared session handshake nonce is invalid")
            if self.status is not SessionStatus.ACTIVE:
                raise SharedSessionError(self.abort_reason or "shared session aborted")
            nonces = (local_nonce, peer_nonce) if self.party == 0 else (peer_nonce, local_nonce)
            self._channel_id = hashlib.sha256(
                b"pllm/shared-peer-channel/v1\x00"
                + bytes.fromhex(self.commitment.digest)
                + nonces[0]
                + nonces[1]
            ).digest()
            self.last_activity = self._now()

    def outbound(self, kind: str, operation_id: str, payload: bytes) -> PeerFrame:
        with self._lock:
            self._require_active()
            if self._channel_id is None:
                self._abort_locked("shared session handshake is incomplete")
                raise SharedSessionError(self.abort_reason or "shared session aborted")
            if operation_id in self._operations:
                self._abort_locked("shared operation identifier was reused")
                raise SharedSessionError(self.abort_reason or "shared session aborted")
            if len(self._operations) >= self.commitment.maximum_operations:
                self._abort_locked("shared session operation budget exhausted")
                raise SharedSessionError(self.abort_reason or "shared session aborted")
            self._operations.add(operation_id)
            frame = PeerFrame(
                self.commitment.session_id,
                self._channel_id,
                self.party,
                self._next_outbound,
                kind,
                operation_id,
                payload,
            )
            self._next_outbound += 1
            self.last_activity = self._now()
            return frame

    def accept(self, frame: PeerFrame, *, expected_kind: str | None = None) -> None:
        with self._lock:
            self._require_active()
            if frame.session_id != self.commitment.session_id:
                self._abort_locked("peer frame belongs to another session")
            elif self._channel_id is None or frame.channel_id != self._channel_id:
                self._abort_locked("peer frame belongs to another authenticated channel")
            elif frame.sender_party != 1 - self.party:
                self._abort_locked("peer frame sender identity is invalid")
            elif frame.sequence != self._next_inbound:
                self._abort_locked("peer frame sequence is replayed or out of order")
            elif expected_kind is not None and frame.kind != expected_kind:
                self._abort_locked("peer frame kind is unexpected")
            if self.status is not SessionStatus.ACTIVE:
                raise SharedSessionError(self.abort_reason or "shared session aborted")
            self._next_inbound += 1
            self.last_activity = self._now()

    def close(self) -> None:
        with self._lock:
            self._require_active()
            self.status = SessionStatus.CLOSED
            self.last_activity = self._now()

    @property
    def channel_id(self) -> bytes:
        with self._lock:
            self._require_active()
            if self._channel_id is None:
                raise SharedSessionError("shared session handshake is incomplete")
            return self._channel_id

    def abort(self, reason: str) -> None:
        with self._lock:
            if self.status is SessionStatus.ACTIVE:
                self._abort_locked(reason)

    def _abort_locked(self, reason: str) -> None:
        self.status = SessionStatus.ABORTED
        self.abort_reason = reason
        self.last_activity = self._now()

    def _require_active(self) -> None:
        if self.status is not SessionStatus.ACTIVE:
            raise SharedSessionError(self.abort_reason or "shared session is not active")


class SharedSessionRegistry:
    def __init__(
        self,
        *,
        capacity: int,
        idle_seconds: float,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0 or idle_seconds <= 0:
            raise ValueError("shared session registry limits must be positive")
        self.capacity = capacity
        self.idle_seconds = idle_seconds
        self._now = now
        self._sessions: dict[str, SharedSessionState] = {}
        self._lock = Lock()

    def create(self, commitment: SharedSessionCommitment, *, party: int) -> SharedSessionState:
        with self._lock:
            self._expire_locked()
            if commitment.session_id in self._sessions:
                raise SharedSessionError("shared session identifier already exists")
            if len(self._sessions) >= self.capacity:
                raise SharedSessionError("shared session capacity exhausted")
            state = SharedSessionState(commitment, party=party, now=self._now)
            self._sessions[commitment.session_id] = state
            return state

    def get(self, session_id: str) -> SharedSessionState:
        with self._lock:
            self._expire_locked()
            state = self._sessions.get(session_id)
            if state is None:
                raise SharedSessionError("shared session is unknown or expired")
            return state

    def _expire_locked(self) -> None:
        cutoff = self._now() - self.idle_seconds
        expired = [key for key, value in self._sessions.items() if value.last_activity < cutoff]
        for key in expired:
            self._sessions[key].abort("shared session expired")
            del self._sessions[key]


class BinaryPeerSocket(Protocol):
    async def send(self, payload: bytes) -> None: ...

    async def recv(self, maximum_bytes: int) -> bytes: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class SharedPeerStats:
    sent_frames: int = 0
    received_frames: int = 0
    uploaded_bytes: int = 0
    downloaded_bytes: int = 0


class SharedPeerConnection:
    """Full-duplex session over an externally authenticated binary transport."""

    def __init__(
        self,
        state: SharedSessionState,
        socket: BinaryPeerSocket,
        *,
        max_payload_bytes: int,
        authenticated_peer_fingerprint: str,
        timeout_seconds: float,
    ) -> None:
        if max_payload_bytes <= 0:
            raise ValueError("peer payload limit must be positive")
        if timeout_seconds <= 0:
            raise ValueError("peer timeout must be positive")
        self.state = state
        self.socket = socket
        self.max_payload_bytes = max_payload_bytes
        self.authenticated_peer_fingerprint = authenticated_peer_fingerprint
        self.timeout_seconds = timeout_seconds
        self.stats = SharedPeerStats()
        self._send_lock = asyncio.Lock()
        self._receive_lock = asyncio.Lock()
        self._handshake_lock = asyncio.Lock()
        self._handshake_complete = False
        self._inbox: dict[tuple[str, str], bytes] = {}

    async def handshake(self) -> None:
        if self._handshake_complete:
            return
        async with self._handshake_lock:
            if self._handshake_complete:
                return
            local_nonce = secrets.token_bytes(32)
            payload = msgpack.packb(
                {
                    "v": 1,
                    "c": self.state.commitment.pack(),
                    "p": self.state.party,
                    "n": local_nonce,
                },
                use_bin_type=True,
            )
            try:
                async with self._send_lock:
                    await asyncio.wait_for(self.socket.send(payload), timeout=self.timeout_seconds)
                async with self._receive_lock:
                    raw = await asyncio.wait_for(
                        self.socket.recv(8192), timeout=self.timeout_seconds
                    )
                if not isinstance(raw, bytes) or not raw or len(raw) > 8192:
                    raise SharedSessionError("peer handshake is invalid")
                item = msgpack.unpackb(raw, raw=False, strict_map_key=True)
                if (
                    not isinstance(item, dict)
                    or set(item) != {"v", "c", "p", "n"}
                    or type(item["v"]) is not int
                    or item["v"] != 1
                    or type(item["c"]) is not bytes
                    or type(item["p"]) is not int
                    or item["p"] != 1 - self.state.party
                    or type(item["n"]) is not bytes
                ):
                    raise SharedSessionError("peer handshake fields are invalid")
                peer_commitment = SharedSessionCommitment.unpack(item["c"])
                peer_nonce = item["n"]
                self.state.establish(
                    peer_commitment,
                    local_nonce=local_nonce,
                    peer_nonce=peer_nonce,
                    authenticated_peer_fingerprint=self.authenticated_peer_fingerprint,
                )
            except BaseException as exc:
                self.state.abort(f"peer handshake failed: {type(exc).__name__}")
                if isinstance(exc, TimeoutError):
                    await self.socket.close()
                    raise SharedSessionError("shared peer handshake timed out") from exc
                raise
            self._handshake_complete = True

    async def send(self, kind: str, operation_id: str, payload: bytes) -> None:
        await self.handshake()
        if len(payload) > self.max_payload_bytes:
            self.state.abort("outbound peer payload is too large")
            raise SharedSessionError("outbound peer payload is too large")
        packed = self.state.outbound(kind, operation_id, payload).pack()
        try:
            async with self._send_lock:
                await asyncio.wait_for(self.socket.send(packed), timeout=self.timeout_seconds)
        except BaseException as exc:
            self.state.abort(f"peer send failed: {type(exc).__name__}")
            if isinstance(exc, TimeoutError):
                await self.socket.close()
                raise SharedSessionError("shared peer send timed out") from exc
            raise
        self.stats.sent_frames += 1
        self.stats.uploaded_bytes += len(packed)

    async def receive(self, *, kind: str, operation_id: str) -> bytes:
        await self.handshake()
        expected = (kind, operation_id)
        try:
            async with self._receive_lock:
                queued = self._inbox.pop(expected, None)
                if queued is not None:
                    if self.state.status is not SessionStatus.ACTIVE:
                        raise SharedSessionError("shared session is not active")
                    return queued
                while True:
                    raw = await asyncio.wait_for(
                        self.socket.recv(self.max_payload_bytes + 1024),
                        timeout=self.timeout_seconds,
                    )
                    if not isinstance(raw, bytes):
                        raise SharedSessionError("peer transport returned non-binary data")
                    frame = PeerFrame.unpack(raw, max_payload_bytes=self.max_payload_bytes)
                    self.state.accept(frame)
                    self.stats.received_frames += 1
                    self.stats.downloaded_bytes += len(raw)
                    key = (frame.kind, frame.operation_id)
                    if key == expected:
                        return frame.payload
                    if (
                        key in self._inbox
                        or len(self._inbox) >= self.state.commitment.maximum_operations
                    ):
                        raise SharedSessionError(
                            "peer response operation is duplicated or excessive"
                        )
                    self._inbox[key] = frame.payload
        except BaseException as exc:
            self.state.abort(f"peer receive failed: {type(exc).__name__}")
            if isinstance(exc, TimeoutError):
                await self.socket.close()
                raise SharedSessionError("shared peer receive timed out") from exc
            raise

    async def exchange(self, kind: str, operation_id: str, payload: bytes) -> bytes:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                await self.send(kind, operation_id, payload)
                return await self.receive(kind=kind, operation_id=operation_id)
        except TimeoutError as exc:
            self.state.abort("shared peer exchange timed out")
            await self.socket.close()
            raise SharedSessionError("shared peer exchange timed out") from exc

    async def close(self) -> None:
        try:
            if self.state.status is SessionStatus.ACTIVE:
                self.state.close()
        finally:
            await self.socket.close()
