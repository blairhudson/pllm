"""Shared output projection and secure-selection boundary."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Protocol

import msgpack
import numpy as np

from .shared_mpc import PartyRuntime, SharedMPCError, SharedTensor, reconstruct


class SecureSelectionBackend(Protocol):
    """Production implementation must compare shares without opening logits."""

    async def select(
        self,
        logits: SharedTensor,
        *,
        generation_id: str,
        signed_bound: int,
    ) -> "TokenShare": ...


@dataclass(frozen=True, slots=True)
class TokenShare:
    session_id: str
    generation_id: str
    commitment_digest: bytes
    channel_id: bytes
    party: int
    value: int

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or not self.generation_id
            or len(self.commitment_digest) != 32
            or len(self.channel_id) != 32
        ):
            raise SharedMPCError("token share identifiers must not be empty")
        if (
            type(self.party) is not int
            or self.party not in (0, 1)
            or type(self.value) is not int
            or not 0 <= self.value < 1 << 64
        ):
            raise SharedMPCError("token share is outside the ring")

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                b"w": 1,
                b"s": self.session_id,
                b"g": self.generation_id,
                b"c": self.commitment_digest,
                b"h": self.channel_id,
                b"p": self.party,
                b"v": self.value,
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "TokenShare":
        if not payload or len(payload) > 4096:
            raise SharedMPCError("token share payload is invalid")
        try:
            item = msgpack.unpackb(payload, raw=True, strict_map_key=False)
            if not isinstance(item, dict) or set(item) != {
                b"w",
                b"s",
                b"g",
                b"c",
                b"h",
                b"p",
                b"v",
            }:
                raise ValueError("invalid fields")
            if type(item[b"w"]) is not int or item[b"w"] != 1:
                raise ValueError("unsupported version")
            if (
                type(item[b"s"]) is not bytes
                or type(item[b"g"]) is not bytes
                or type(item[b"c"]) is not bytes
                or type(item[b"h"]) is not bytes
                or type(item[b"p"]) is not int
                or type(item[b"v"]) is not int
            ):
                raise ValueError("invalid field types")
            return cls(
                item[b"s"].decode("utf-8"),
                item[b"g"].decode("utf-8"),
                item[b"c"],
                item[b"h"],
                item[b"p"],
                item[b"v"],
            )
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, msgpack.ExtraData) as exc:
            raise SharedMPCError("token share payload is malformed") from exc


class SharedOutputHeadParty:
    def __init__(self, runtime: PartyRuntime, weight: np.ndarray) -> None:
        raw = np.asarray(weight)
        if raw.ndim != 2 or raw.size == 0:
            raise SharedMPCError("output-head weight must be a non-empty matrix")
        if not np.issubdtype(raw.dtype, np.integer):
            raise SharedMPCError("output-head weight must contain integers")
        if int(raw.min()) < -128 or int(raw.max()) > 127:
            raise SharedMPCError("output-head weight must fit signed int8")
        self.runtime = runtime
        self.weight = np.ascontiguousarray(raw, dtype=np.int8)
        self.weight.setflags(write=False)

    def project(
        self, hidden: SharedTensor, generation_id: str, *, signed_bound: int
    ) -> SharedTensor:
        return self.runtime.linear_public(
            hidden, self.weight, f"{generation_id}.logits", signed_bound=signed_bound
        )


class ReferenceJointSelector:
    """Ideal test functionality; never instantiate in a service process."""

    @staticmethod
    def select_pair(
        first: SharedTensor,
        second: SharedTensor,
        *,
        generation_id: str,
        commitment_digest: bytes,
        channel_id: bytes,
    ) -> tuple[TokenShare, TokenShare]:
        clear = reconstruct(first, second).view(np.int64)
        if clear.ndim != 2 or clear.shape[0] != 1:
            raise SharedMPCError("reference selector expects one logits row")
        token = int(np.argmax(clear[0]))
        first_value = int.from_bytes(secrets.token_bytes(8), "little")
        second_value = (token - first_value) % (1 << 64)
        return (
            TokenShare(
                first.session_id, generation_id, commitment_digest, channel_id, 0, first_value
            ),
            TokenShare(
                first.session_id, generation_id, commitment_digest, channel_id, 1, second_value
            ),
        )


def reconstruct_token(
    first: TokenShare,
    second: TokenShare,
    *,
    vocabulary_size: int,
    expected_commitment_digest: bytes,
    expected_channel_id: bytes,
    expected_generation_id: str,
    consumed_generation_ids: set[str],
) -> int:
    if (
        first.session_id != second.session_id
        or first.generation_id != second.generation_id
        or first.commitment_digest != second.commitment_digest
        or first.commitment_digest != expected_commitment_digest
        or first.channel_id != second.channel_id
        or first.channel_id != expected_channel_id
        or first.party == second.party
    ):
        raise SharedMPCError("token shares do not belong together")
    if first.generation_id != expected_generation_id:
        raise SharedMPCError("token shares belong to an unexpected generation")
    if first.generation_id in consumed_generation_ids:
        raise SharedMPCError("token shares were already consumed")
    consumed_generation_ids.add(first.generation_id)
    token = (first.value + second.value) % (1 << 64)
    if token >= vocabulary_size:
        raise SharedMPCError("reconstructed token is outside vocabulary")
    return token
