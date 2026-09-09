from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

import msgpack
import numpy as np

from .shared_mpc import SharedMPCError, SharedTensor


def _ring_array(values: np.ndarray) -> np.ndarray:
    signed = np.ascontiguousarray(values, dtype=np.int64)
    return signed.view(np.uint64)


@dataclass(frozen=True, slots=True)
class PrivateLookupKey:
    session_id: str
    query_id: str
    commitment_digest: bytes
    channel_id: bytes
    table_fingerprint: bytes
    party: int
    shares: np.ndarray

    def __post_init__(self) -> None:
        shares = np.ascontiguousarray(self.shares, dtype=np.uint64)
        if not self.session_id or not self.query_id:
            raise SharedMPCError("private lookup identifiers cannot be empty")
        if (
            len(self.commitment_digest) != 32
            or len(self.channel_id) != 32
            or len(self.table_fingerprint) != 32
            or self.party not in (0, 1)
        ):
            raise SharedMPCError("private lookup binding is invalid")
        if shares.ndim != 2 or not 0 < shares.shape[1] <= 1_000_000:
            raise SharedMPCError("private lookup shares have invalid dimensions")
        object.__setattr__(self, "shares", shares)

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                b"v": 1,
                b"s": self.session_id,
                b"q": self.query_id,
                b"c": self.commitment_digest,
                b"h": self.channel_id,
                b"f": self.table_fingerprint,
                b"p": self.party,
                b"n": self.shares.shape,
                b"x": self.shares.astype("<u8", copy=False).tobytes(),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes, *, maximum_bytes: int) -> "PrivateLookupKey":
        if not payload or len(payload) > maximum_bytes:
            raise SharedMPCError("private lookup key exceeds limit")
        try:
            item = msgpack.unpackb(
                payload,
                raw=True,
                strict_map_key=False,
                max_bin_len=maximum_bytes,
                max_array_len=2,
                max_map_len=9,
            )
            if not isinstance(item, dict) or set(item) != {
                b"v",
                b"s",
                b"q",
                b"c",
                b"h",
                b"f",
                b"p",
                b"n",
                b"x",
            }:
                raise ValueError("invalid fields")
            if type(item[b"v"]) is not int or item[b"v"] != 1:
                raise ValueError("unsupported version")
            raw_shape = item[b"n"]
            if (
                not isinstance(raw_shape, (list, tuple))
                or len(raw_shape) != 2
                or any(type(value) is not int or value <= 0 for value in raw_shape)
            ):
                raise ValueError("shape mismatch")
            shape = (raw_shape[0], raw_shape[1])
            raw = item[b"x"]
            if type(raw) is not bytes or shape[0] > maximum_bytes // 8 // shape[1]:
                raise ValueError("shape mismatch")
            if len(raw) != shape[0] * shape[1] * 8:
                raise ValueError("shape mismatch")
            if (
                type(item[b"s"]) is not bytes
                or type(item[b"q"]) is not bytes
                or type(item[b"c"]) is not bytes
                or type(item[b"h"]) is not bytes
                or type(item[b"f"]) is not bytes
                or type(item[b"p"]) is not int
            ):
                raise ValueError("invalid field types")
            shares = np.frombuffer(raw, dtype="<u8").reshape(shape).astype(np.uint64, copy=True)
            return cls(
                item[b"s"].decode("utf-8"),
                item[b"q"].decode("utf-8"),
                item[b"c"],
                item[b"h"],
                item[b"f"],
                item[b"p"],
                shares,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            msgpack.ExtraData,
            msgpack.FormatError,
        ) as exc:
            raise SharedMPCError("invalid private lookup key") from exc


class DenseReferenceLookupEncoder:
    """Information-theoretic reference encoder; production must use DPF."""

    @staticmethod
    def generate(
        session_id: str,
        query_id: str,
        token_ids: np.ndarray,
        *,
        vocabulary_size: int,
        commitment_digest: bytes,
        channel_id: bytes,
        table_fingerprint: bytes,
        maximum_tokens: int = 4096,
        maximum_bytes: int = 64 * 1024 * 1024,
    ) -> tuple[PrivateLookupKey, PrivateLookupKey]:
        tokens = np.asarray(token_ids, dtype=np.int64)
        if tokens.ndim != 1 or not 0 < tokens.size <= maximum_tokens:
            raise SharedMPCError("private lookup tokens must be a non-empty vector")
        if vocabulary_size <= 0 or np.any(tokens < 0) or np.any(tokens >= vocabulary_size):
            raise SharedMPCError("private lookup token is outside vocabulary")
        byte_count = int(tokens.size) * vocabulary_size * 8
        if byte_count > maximum_bytes:
            raise SharedMPCError("dense reference lookup exceeds memory limit")
        first = np.frombuffer(secrets.token_bytes(byte_count), dtype="<u8").reshape(
            tokens.size, vocabulary_size
        )
        second = np.uint64(0) - first
        second[np.arange(tokens.size), tokens] += np.uint64(1)
        return (
            PrivateLookupKey(
                session_id, query_id, commitment_digest, channel_id, table_fingerprint, 0, first
            ),
            PrivateLookupKey(
                session_id, query_id, commitment_digest, channel_id, table_fingerprint, 1, second
            ),
        )


class SharedEmbeddingParty:
    def __init__(
        self,
        *,
        session_id: str,
        commitment_digest: bytes,
        channel_id: bytes,
        party: int,
        table: np.ndarray,
        scale: int,
        consumed_query_ids: set[str],
    ) -> None:
        table = np.array(table, dtype=np.int64, order="C", copy=True)
        if party not in (0, 1) or table.ndim != 2 or table.size == 0 or scale <= 0:
            raise SharedMPCError("shared embedding configuration is invalid")
        self.session_id = session_id
        if len(commitment_digest) != 32 or len(channel_id) != 32:
            raise SharedMPCError("shared embedding commitment is invalid")
        self.commitment_digest = commitment_digest
        self.channel_id = channel_id
        self.party = party
        self.table = table
        self.table_ring = _ring_array(table)
        self.scale = scale
        self.table_fingerprint = hashlib.sha256(table.astype("<i8", copy=False).tobytes()).digest()
        self._used_queries = consumed_query_ids

    def lookup(self, key: PrivateLookupKey) -> SharedTensor:
        if (
            key.session_id != self.session_id
            or key.commitment_digest != self.commitment_digest
            or key.channel_id != self.channel_id
            or key.party != self.party
        ):
            raise SharedMPCError("private lookup key belongs to another party, session, or channel")
        if key.table_fingerprint != self.table_fingerprint:
            raise SharedMPCError("private lookup key targets another table")
        if key.shares.shape[1] != self.table.shape[0]:
            raise SharedMPCError("private lookup vocabulary mismatch")
        if key.query_id in self._used_queries:
            raise SharedMPCError("private lookup key was already consumed")
        self._used_queries.add(key.query_id)
        output = np.matmul(key.shares, self.table_ring, dtype=np.uint64)
        return SharedTensor(
            self.session_id, f"lookup.{key.query_id}", self.party, output, self.scale
        )
