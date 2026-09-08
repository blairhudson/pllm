from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator

import msgpack


PROTOCOL_VERSION = "he-responses/1"
BINARY_MEDIA_TYPE = "application/vnd.openai.he+msgpack"
JSON_MEDIA_TYPE = "application/json"
FRAME_HEADER = struct.Struct("!I")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HEEnvelope:
    request_id: str
    session_id: str
    model: str
    kind: str
    sequence: int
    payload: bytes
    nonce: bytes
    created_at: float
    metadata: dict[str, Any]
    mac: bytes = b""

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "v": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "model": self.model,
            "kind": self.kind,
            "sequence": self.sequence,
            "payload": self.payload,
            "nonce": self.nonce,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "mac": self.mac}

    def sign(self, key: bytes) -> "HEEnvelope":
        digest = hmac.new(key, canonical_pack(self.unsigned_dict()), hashlib.sha256).digest()
        return replace(self, mac=digest)

    def verify(self, key: bytes, *, max_age_seconds: float = 300.0) -> None:
        expected = hmac.new(key, canonical_pack(self.unsigned_dict()), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, self.mac):
            raise ProtocolError("invalid envelope MAC")
        age = abs(time.time() - self.created_at)
        if age > max_age_seconds:
            raise ProtocolError("stale envelope")

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        session_id: str,
        model: str,
        kind: str,
        sequence: int,
        payload: bytes,
        metadata: dict[str, Any] | None = None,
        key: bytes,
    ) -> "HEEnvelope":
        envelope = cls(
            request_id=request_id,
            session_id=session_id,
            model=model,
            kind=kind,
            sequence=sequence,
            payload=payload,
            nonce=os.urandom(16),
            created_at=time.time(),
            metadata=metadata or {},
        )
        return envelope.sign(key)


class ReplayWindow:
    """Bounded nonce/sequence replay guard for one HE session."""

    def __init__(self, max_entries: int = 4096) -> None:
        self.max_entries = max_entries
        self._seen: dict[bytes, int] = {}
        self._highest_sequence = -1

    def accept(self, envelope: HEEnvelope) -> None:
        if envelope.nonce in self._seen:
            raise ProtocolError("replayed nonce")
        # Allow modest reordering for multiplexed requests, but not sequence reuse.
        if envelope.sequence in self._seen.values():
            raise ProtocolError("replayed sequence")
        self._seen[envelope.nonce] = envelope.sequence
        self._highest_sequence = max(self._highest_sequence, envelope.sequence)
        while len(self._seen) > self.max_entries:
            oldest = min(self._seen, key=self._seen.get)
            del self._seen[oldest]


def canonical_pack(value: Any) -> bytes:
    return msgpack.packb(_sort_mapping(value), use_bin_type=True, strict_types=True)


def pack_envelope(envelope: HEEnvelope) -> bytes:
    return canonical_pack(envelope.to_dict())


def unpack_envelope(data: bytes) -> HEEnvelope:
    try:
        value = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except Exception as exc:  # pragma: no cover - precise backend exception is not API surface
        raise ProtocolError("invalid MessagePack envelope") from exc
    if not isinstance(value, dict) or value.get("v") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported HE protocol")
    required = {
        "request_id",
        "session_id",
        "model",
        "kind",
        "sequence",
        "payload",
        "nonce",
        "created_at",
        "metadata",
        "mac",
    }
    missing = required - value.keys()
    if missing:
        raise ProtocolError(f"missing envelope fields: {sorted(missing)}")
    return HEEnvelope(
        request_id=str(value["request_id"]),
        session_id=str(value["session_id"]),
        model=str(value["model"]),
        kind=str(value["kind"]),
        sequence=int(value["sequence"]),
        payload=bytes(value["payload"]),
        nonce=bytes(value["nonce"]),
        created_at=float(value["created_at"]),
        metadata=dict(value["metadata"]),
        mac=bytes(value["mac"]),
    )


def envelope_to_json(envelope: HEEnvelope) -> bytes:
    value = envelope.to_dict()
    for key in ("payload", "nonce", "mac"):
        value[key] = base64.b64encode(value[key]).decode("ascii")
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def envelope_from_json(data: bytes) -> HEEnvelope:
    try:
        value = json.loads(data)
        for key in ("payload", "nonce", "mac"):
            value[key] = base64.b64decode(value[key], validate=True)
    except Exception as exc:
        raise ProtocolError("invalid JSON HE envelope") from exc
    return unpack_envelope(msgpack.packb(value, use_bin_type=True))


def encode_length_prefixed(frames: Iterable[bytes]) -> bytes:
    out = bytearray()
    for frame in frames:
        out.extend(FRAME_HEADER.pack(len(frame)))
        out.extend(frame)
    return bytes(out)


def iter_length_prefixed(data: bytes) -> Iterator[bytes]:
    offset = 0
    length = len(data)
    while offset < length:
        if length - offset < FRAME_HEADER.size:
            raise ProtocolError("truncated frame header")
        (size,) = FRAME_HEADER.unpack_from(data, offset)
        offset += FRAME_HEADER.size
        if size < 0 or length - offset < size:
            raise ProtocolError("truncated frame")
        yield data[offset : offset + size]
        offset += size


def sse_event(value: dict[str, Any]) -> bytes:
    event_type = value.get("type", "message")
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8")


def _sort_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_mapping(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_mapping(item) for item in value]
    if isinstance(value, tuple):
        return [_sort_mapping(item) for item in value]
    return value
