"""Bounded, in-process BFV-to-shares reference for one quadratic island.

Party A holds the BFV secret key and an additive input share. Party B holds
the other input share, a public evaluation context and a fresh output mask.
Only encrypted input/output envelopes cross the reference role boundary.
This is an honest-execution numeric/cost experiment, not an MPC transport.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np

from .tiled_bfv import import_tenseal


_MODULUS = 65_537
_POLY_DEGREE = 4_096
_SCHEMA = "pllm.he_quadratic_share.v1"
_MAX_ENVELOPE_BYTES = 1 << 20


class HEQuadraticShareError(ValueError):
    """Invalid, mismatched or consumed reference material."""


def _shape(rows: int, width: int) -> tuple[int, int]:
    if type(rows) is not int or type(width) is not int or not 1 <= rows <= 64 or not 1 <= width <= 64:
        raise HEQuadraticShareError("rows and width must be bounded positive integers <= 64")
    if rows * width > _POLY_DEGREE:
        raise HEQuadraticShareError("island exceeds the BFV packing bound")
    return rows, width


def _share(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu":
        raise HEQuadraticShareError(f"share must be an integer matrix of shape {shape}")
    if np.any(array < 0) or np.any(array >= _MODULUS):
        raise HEQuadraticShareError("share residues must be in [0, 65537)")
    return array.astype(np.int64, copy=True)


def _pack(kind: str, session_id: bytes, context_digest: str, shape: tuple[int, int], ciphertext: bytes) -> bytes:
    return msgpack.packb({
        "schema": _SCHEMA,
        "kind": kind,
        "session_id": session_id,
        "context_digest": context_digest,
        "rows": shape[0],
        "width": shape[1],
        "ciphertext": ciphertext,
    }, use_bin_type=True)


def _unpack(payload: bytes, kind: str, digest: str, shape: tuple[int, int]) -> dict[str, Any]:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_ENVELOPE_BYTES:
        raise HEQuadraticShareError("encrypted island envelope exceeds the byte bound")
    try:
        value = msgpack.unpackb(payload, raw=False, strict_map_key=True)
    except Exception as exc:
        raise HEQuadraticShareError("invalid encrypted island envelope") from exc
    if type(value) is not dict or set(value) != {
        "schema", "kind", "session_id", "context_digest", "rows", "width", "ciphertext",
    }:
        raise HEQuadraticShareError("invalid encrypted island envelope fields")
    if (
        value["schema"] != _SCHEMA
        or value["kind"] != kind
        or value["context_digest"] != digest
        or type(value["rows"]) is not int
        or type(value["width"]) is not int
        or (value["rows"], value["width"]) != shape
        or type(value["session_id"]) is not bytes
        or len(value["session_id"]) != 16
        or type(value["ciphertext"]) is not bytes
        or not value["ciphertext"]
    ):
        raise HEQuadraticShareError("encrypted island binding mismatch")
    return value


@dataclass(frozen=True, slots=True)
class HEQuadraticShareEvaluation:
    response: bytes
    worker_b_share: np.ndarray


class HEQuadraticShareClient:
    """Party A: one outstanding issue; secret key never enters public context."""

    def __init__(self, rows: int, width: int) -> None:
        self.shape = _shape(rows, width)
        import_tenseal()
        import tenseal as ts

        self.ts = ts
        self.context = ts.context(ts.SCHEME_TYPE.BFV, _POLY_DEGREE, _MODULUS, n_threads=1)
        self.context.generate_relin_keys()
        self.public_context = self.context.serialize(
            save_public_key=True, save_secret_key=False,
            save_galois_keys=False, save_relin_keys=True,
        )
        self.context_digest = hashlib.sha256(self.public_context).hexdigest()
        self._pending: bytes | None = None

    def issue(self, worker_a_share: np.ndarray) -> bytes:
        if self._pending is not None:
            raise HEQuadraticShareError("one-use client island is already pending")
        share = _share(worker_a_share, self.shape)
        session_id = secrets.token_bytes(16)
        encrypted = self.ts.bfv_vector(self.context, share.reshape(-1).tolist()).serialize()
        self._pending = session_id
        return _pack("request", session_id, self.context_digest, self.shape, encrypted)

    def finish(self, response: bytes) -> np.ndarray:
        session_id, self._pending = self._pending, None  # burn even on invalid response
        if session_id is None:
            raise HEQuadraticShareError("no outstanding one-use client island")
        value = _unpack(response, "response", self.context_digest, self.shape)
        if value["session_id"] != session_id:
            raise HEQuadraticShareError("encrypted island session mismatch")
        try:
            decoded = self.ts.bfv_vector_from(self.context, value["ciphertext"]).decrypt()
        except Exception as exc:
            raise HEQuadraticShareError("invalid encrypted island result") from exc
        if len(decoded) != self.shape[0] * self.shape[1]:
            raise HEQuadraticShareError("encrypted island output shape mismatch")
        return (np.asarray(decoded, dtype=np.int64) % _MODULUS).reshape(self.shape)

    def cancel(self) -> None:
        self._pending = None


class HEQuadraticShareEvaluator:
    """Party B: public BFV context and own share only; no private key."""

    def __init__(self, public_context: bytes, rows: int, width: int) -> None:
        self.shape = _shape(rows, width)
        if type(public_context) is not bytes or not 0 < len(public_context) <= _MAX_ENVELOPE_BYTES:
            raise HEQuadraticShareError("public BFV context exceeds the byte bound")
        import_tenseal()
        import tenseal as ts

        self.ts = ts
        try:
            self.context = ts.context_from(public_context, n_threads=1)
        except Exception as exc:
            raise HEQuadraticShareError("invalid public BFV context") from exc
        if self.context.has_secret_key():
            raise HEQuadraticShareError("evaluator must not hold BFV secret key")
        self.context_digest = hashlib.sha256(public_context).hexdigest()
        self._consumed: set[bytes] = set()

    def evaluate(self, request: bytes, worker_b_share: np.ndarray) -> HEQuadraticShareEvaluation:
        share = _share(worker_b_share, self.shape)
        value = _unpack(request, "request", self.context_digest, self.shape)
        session_id = value["session_id"]
        if session_id in self._consumed or len(self._consumed) >= 64:
            raise HEQuadraticShareError("encrypted island replay or capacity exceeded")
        self._consumed.add(session_id)  # burn before ciphertext loading and evaluation
        try:
            encrypted = self.ts.bfv_vector_from(self.context, value["ciphertext"])
            mask = np.asarray(
                [secrets.randbelow(_MODULUS) for _ in range(share.size)], dtype=np.int64,
            ).reshape(self.shape)
            combined = encrypted + share.reshape(-1).tolist()
            # Exact polynomial over F_65537; no SiLU/rounding or Qwen range claim.
            result = combined * combined + combined * 3 + 7 - mask.reshape(-1).tolist()
            response = _pack(
                "response", session_id, self.context_digest, self.shape, result.serialize(),
            )
        except Exception as exc:
            raise HEQuadraticShareError("encrypted island evaluation failed") from exc
        if len(response) > _MAX_ENVELOPE_BYTES:
            raise HEQuadraticShareError("encrypted island response exceeds the byte bound")
        return HEQuadraticShareEvaluation(response, mask)
