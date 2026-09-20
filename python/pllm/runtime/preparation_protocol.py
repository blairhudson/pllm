from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np

from .protocol import ProtocolError
from .stage_protocol import RingKind, pack_residues, unpack_residues


PREPARATION_PROTOCOL_VERSION = 3
PREPARATION_SEED_BYTES = 32
_ATTEMPT_RE = re.compile(r"[0-9a-f]{32}")
PREPARATION_MAX_IDENTIFIER_BYTES = 512


@dataclass(frozen=True, slots=True)
class SeededRingProfile:
    signed_output_bound: int
    ring: RingKind
    modulus: int
    wire_bits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "signed_output_bound": self.signed_output_bound,
            "ring": self.ring,
            "modulus": self.modulus,
            "wire_bits": self.wire_bits,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SeededRingProfile":
        if not isinstance(value, dict) or set(value) != {
            "signed_output_bound",
            "ring",
            "modulus",
            "wire_bits",
        }:
            raise ProtocolError("invalid seeded ring profile")
        expected = seeded_ring_profile(int(value["signed_output_bound"]))
        actual = cls(
            signed_output_bound=int(value["signed_output_bound"]),
            ring=str(value["ring"]),  # type: ignore[arg-type]
            modulus=int(value["modulus"]),
            wire_bits=int(value["wire_bits"]),
        )
        if actual != expected:
            raise ProtocolError("seeded ring profile is insufficient or inconsistent")
        return actual


def seeded_ring_profile(signed_output_bound: int) -> SeededRingProfile:
    bound = int(signed_output_bound)
    if bound < 0:
        raise ProtocolError("signed output bound must be non-negative")
    for bits in (16, 24, 32):
        if bound < 1 << (bits - 1):
            return SeededRingProfile(bound, f"u{bits}", 1 << bits, bits)  # type: ignore[arg-type]
    raise ProtocolError("stage signed output exceeds exact ring range")


def validate_attempt_id(value: str) -> None:
    if _ATTEMPT_RE.fullmatch(value) is None:
        raise ProtocolError("attempt id must encode at least 128 random bits")


def _validate_identifiers(*values: str, kind: str) -> None:
    if not all(values):
        raise ProtocolError(f"{kind} identifiers and commitments must be non-empty")
    if any(len(value.encode("utf-8")) > PREPARATION_MAX_IDENTIFIER_BYTES for value in values):
        raise ProtocolError(f"{kind} identifier is too large")


def _validate_shape(
    rows: int,
    in_features: int,
    out_features: int,
    *,
    kind: str,
    max_rows: int | None = None,
    max_tensor_elements: int | None = None,
) -> None:
    if rows <= 0 or in_features <= 0 or out_features <= 0:
        raise ProtocolError(f"{kind} shape must be positive")
    if max_rows is not None and rows > max_rows:
        raise ProtocolError(f"{kind} row count exceeds model context")
    if (
        max_tensor_elements is not None
        and max(rows * in_features, rows * out_features) > max_tensor_elements
    ):
        raise ProtocolError(f"{kind} tensor allocation is too large")


@dataclass(frozen=True, slots=True)
class PreparationRequestContext:
    stage_id: str
    model: str
    body_fingerprint: str
    weight_digest: str
    in_features: int
    out_features: int
    weight_bits: int
    activation_bits: int
    signed_output_bound: int
    ring: RingKind
    modulus: int
    wire_bits: int


@dataclass(frozen=True, slots=True)
class PreparationRequest:
    attempt_id: str
    session_id: str
    model: str
    body_fingerprint: str
    stage_id: str
    weight_digest: str
    rows: int
    in_features: int
    out_features: int
    weight_bits: int
    activation_bits: int
    signed_output_bound: int
    ring: RingKind
    modulus: int
    wire_bits: int
    seed: bytes

    @property
    def request_id(self) -> str:
        return self.attempt_id

    @property
    def profile(self) -> SeededRingProfile:
        return SeededRingProfile(self.signed_output_bound, self.ring, self.modulus, self.wire_bits)

    @property
    def context(self) -> PreparationRequestContext:
        return PreparationRequestContext(
            stage_id=self.stage_id,
            model=self.model,
            body_fingerprint=self.body_fingerprint,
            weight_digest=self.weight_digest,
            in_features=self.in_features,
            out_features=self.out_features,
            weight_bits=self.weight_bits,
            activation_bits=self.activation_bits,
            signed_output_bound=self.signed_output_bound,
            ring=self.ring,
            modulus=self.modulus,
            wire_bits=self.wire_bits,
        )

    def _validate(
        self,
        *,
        max_rows: int | None = None,
        max_tensor_elements: int | None = None,
    ) -> None:
        validate_attempt_id(self.attempt_id)
        if len(self.seed) != PREPARATION_SEED_BYTES:
            raise ProtocolError("preparation seed must be 32 bytes")
        _validate_identifiers(
            self.session_id,
            self.model,
            self.body_fingerprint,
            self.stage_id,
            self.weight_digest,
            kind="preparation",
        )
        _validate_shape(
            self.rows,
            self.in_features,
            self.out_features,
            kind="preparation",
            max_rows=max_rows,
            max_tensor_elements=max_tensor_elements,
        )
        if self.weight_bits not in {4, 8} or self.activation_bits not in {4, 8}:
            raise ProtocolError("unsupported preparation quantization")
        if self.profile != seeded_ring_profile(self.signed_output_bound):
            raise ProtocolError("seeded ring profile is insufficient or inconsistent")

    def pack(self) -> bytes:
        self._validate()
        return msgpack.packb(
            {
                "v": PREPARATION_PROTOCOL_VERSION,
                "a": bytes.fromhex(self.attempt_id),
                "h": self.session_id,
                "r": self.rows,
                "z": self.seed,
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(
        cls,
        payload: bytes,
        *,
        context: PreparationRequestContext,
        max_rows: int | None = None,
        max_tensor_elements: int | None = None,
    ) -> "PreparationRequest":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise ProtocolError("invalid preparation request") from exc
        required = {"v", "a", "h", "r", "z"}
        if not isinstance(value, dict) or set(value) != required:
            raise ProtocolError("invalid preparation request schema")
        if int(value["v"]) != PREPARATION_PROTOCOL_VERSION:
            raise ProtocolError("unsupported preparation protocol")
        seed = value["z"]
        if not isinstance(seed, (bytes, bytearray)):
            raise ProtocolError("preparation seed must be 32 bytes")
        attempt = value["a"]
        if not isinstance(attempt, (bytes, bytearray)) or len(attempt) != 16:
            raise ProtocolError("attempt ID must contain 128 random bits")
        request = cls(
            attempt_id=bytes(attempt).hex(),
            session_id=str(value["h"]),
            model=context.model,
            body_fingerprint=context.body_fingerprint,
            stage_id=context.stage_id,
            weight_digest=context.weight_digest,
            rows=int(value["r"]),
            in_features=context.in_features,
            out_features=context.out_features,
            weight_bits=context.weight_bits,
            activation_bits=context.activation_bits,
            signed_output_bound=context.signed_output_bound,
            ring=context.ring,
            modulus=context.modulus,
            wire_bits=context.wire_bits,
            seed=bytes(seed),
        )
        request._validate(max_rows=max_rows, max_tensor_elements=max_tensor_elements)
        return request

    def metadata(self) -> tuple[Any, ...]:
        return (
            self.session_id,
            self.model,
            self.body_fingerprint,
            self.attempt_id,
            self.stage_id,
            self.weight_digest,
            self.rows,
            self.in_features,
            self.out_features,
            self.weight_bits,
            self.activation_bits,
            self.signed_output_bound,
            self.ring,
            self.modulus,
            self.wire_bits,
        )


@dataclass(frozen=True, slots=True)
class SessionAuthorization:
    session_id: str
    model: str
    body_fingerprint: str
    stage_commitment: str
    weight_bits: int
    activation_bits: int
    max_attempts: int
    rows: int
    stage_ids: tuple[str, ...]
    verification_component: str = "none"
    verification_target_failure_bits: int = 0

    def metadata(self) -> tuple[Any, ...]:
        return (
            self.session_id,
            self.model,
            self.body_fingerprint,
            self.stage_commitment,
            self.weight_bits,
            self.activation_bits,
            self.max_attempts,
            self.rows,
            self.stage_ids,
            self.verification_component,
            self.verification_target_failure_bits,
        )

    def _validate(self) -> None:
        _validate_identifiers(
            self.session_id,
            self.model,
            self.body_fingerprint,
            self.stage_commitment,
            kind="session authorization",
        )
        if self.weight_bits not in {4, 8} or self.activation_bits not in {4, 8}:
            raise ProtocolError("unsupported session authorization quantization")
        if self.max_attempts <= 0:
            raise ProtocolError("session authorization attempt budget must be positive")
        _validate_identifiers(*self.stage_ids, kind="session authorization stage")
        if self.verification_component == "none":
            if self.verification_target_failure_bits != 0:
                raise ProtocolError("disabled verification requires a zero failure target")
        elif self.verification_component == "pllm/freivalds-verify/v1":
            if not 1 <= self.verification_target_failure_bits <= 80:
                raise ProtocolError("invalid Freivalds failure target")
        else:
            raise ProtocolError("unsupported verification component")
        if self.rows <= 0 or not self.stage_ids or len(self.stage_ids) != len(set(self.stage_ids)):
            raise ProtocolError("invalid session authorization inventory shape")
        if self.max_attempts != self.rows * len(self.stage_ids):
            raise ProtocolError("session authorization attempt budget mismatch")

    def pack(self) -> bytes:
        self._validate()
        return msgpack.packb(
            {
                "v": PREPARATION_PROTOCOL_VERSION,
                "h": self.session_id,
                "m": self.model,
                "b": self.body_fingerprint,
                "t": self.stage_commitment,
                "wb": self.weight_bits,
                "ab": self.activation_bits,
                "a": self.max_attempts,
                "r": self.rows,
                "s": list(self.stage_ids),
                "vc": self.verification_component,
                "vf": self.verification_target_failure_bits,
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "SessionAuthorization":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise ProtocolError("invalid session authorization") from exc
        required = {"v", "h", "m", "b", "t", "wb", "ab", "a", "r", "s", "vc", "vf"}
        if not isinstance(value, dict) or set(value) != required:
            raise ProtocolError("invalid session authorization schema")
        if int(value["v"]) != PREPARATION_PROTOCOL_VERSION:
            raise ProtocolError("unsupported preparation protocol")
        result = cls(
            session_id=str(value["h"]),
            model=str(value["m"]),
            body_fingerprint=str(value["b"]),
            stage_commitment=str(value["t"]),
            weight_bits=int(value["wb"]),
            activation_bits=int(value["ab"]),
            max_attempts=int(value["a"]),
            rows=int(value["r"]),
            stage_ids=tuple(str(item) for item in value["s"]),
            verification_component=str(value["vc"]),
            verification_target_failure_bits=int(value["vf"]),
        )
        result._validate()
        return result


def freivalds_session_id(authorization: SessionAuthorization) -> bytes:
    authorization._validate()
    return hashlib.sha256(b"pllm.freivalds.session.v1\x00" + authorization.pack()).digest()


def freivalds_material_id(
    authorization: SessionAuthorization, request: PreparationRequest
) -> bytes:
    domain = msgpack.packb(
        ["pllm.freivalds.material.v1", authorization.metadata(), request.metadata()],
        use_bin_type=True,
    )
    return hashlib.sha256(domain).digest()


def freivalds_binding(authorization: SessionAuthorization, request: PreparationRequest) -> bytes:
    value = msgpack.packb(
        [
            "pllm.freivalds.runtime_binding.v1",
            authorization.metadata(),
            request.metadata(),
        ],
        use_bin_type=True,
    )
    if len(value) > 4_096:
        raise ProtocolError("Freivalds binding exceeds its fixed limit")
    return value


def _mask_domain(request: PreparationRequest, purpose: str) -> bytes:
    return msgpack.packb(
        [
            "pllm-prepared-correction-v2",
            purpose,
            request.session_id,
            request.model,
            request.body_fingerprint,
            request.attempt_id,
            request.stage_id,
            request.weight_digest,
            request.rows,
            request.in_features,
            request.out_features,
            request.weight_bits,
            request.activation_bits,
            request.ring,
            request.modulus,
            request.wire_bits,
            request.signed_output_bound,
        ],
        use_bin_type=True,
    )


def expand_preparation_mask(request: PreparationRequest, purpose: str = "input-r") -> np.ndarray:
    request._validate()
    if purpose not in {"input-r", "output-s"}:
        raise ProtocolError("unsupported preparation mask domain")
    width = request.in_features if purpose == "input-r" else request.out_features
    shake = hashlib.shake_256()
    shake.update(_mask_domain(request, purpose))
    shake.update(request.seed)
    payload = shake.digest(request.rows * width * (request.wire_bits // 8))
    return unpack_residues(payload, (request.rows, width), request.wire_bits)


def expand_output_mask(request: PreparationRequest) -> np.ndarray:
    return expand_preparation_mask(request, "output-s")


@dataclass(frozen=True, slots=True)
class CorrectionPush:
    attempt_id: str
    session_id: str
    model: str
    body_fingerprint: str
    stage_id: str
    weight_digest: str
    rows: int
    in_features: int
    out_features: int
    weight_bits: int
    activation_bits: int
    signed_output_bound: int
    ring: RingKind
    modulus: int
    wire_bits: int
    correction: np.ndarray
    server_ns: int = 0

    @property
    def profile(self) -> SeededRingProfile:
        return SeededRingProfile(self.signed_output_bound, self.ring, self.modulus, self.wire_bits)

    def metadata(self) -> tuple[Any, ...]:
        return (
            self.session_id,
            self.model,
            self.body_fingerprint,
            self.attempt_id,
            self.stage_id,
            self.weight_digest,
            self.rows,
            self.in_features,
            self.out_features,
            self.weight_bits,
            self.activation_bits,
            self.signed_output_bound,
            self.ring,
            self.modulus,
            self.wire_bits,
        )

    def _validate(
        self,
        *,
        max_rows: int | None = None,
        max_tensor_elements: int | None = None,
    ) -> None:
        validate_attempt_id(self.attempt_id)
        _validate_identifiers(
            self.session_id,
            self.model,
            self.body_fingerprint,
            self.stage_id,
            self.weight_digest,
            kind="correction",
        )
        _validate_shape(
            self.rows,
            self.in_features,
            self.out_features,
            kind="correction",
            max_rows=max_rows,
            max_tensor_elements=max_tensor_elements,
        )
        if self.weight_bits not in {4, 8} or self.activation_bits not in {4, 8}:
            raise ProtocolError("unsupported correction quantization")
        if self.profile != seeded_ring_profile(self.signed_output_bound):
            raise ProtocolError("invalid correction ring profile")

    def pack(self) -> bytes:
        self._validate()
        value = np.asarray(self.correction, dtype=np.uint32)
        if value.shape != (self.rows, self.out_features):
            raise ProtocolError("correction shape mismatch")
        return msgpack.packb(
            {
                "v": PREPARATION_PROTOCOL_VERSION,
                "a": self.attempt_id,
                "h": self.session_id,
                "m": self.model,
                "b": self.body_fingerprint,
                "t": self.stage_id,
                "w": self.weight_digest,
                "r": self.rows,
                "n": self.in_features,
                "o": self.out_features,
                "wb": self.weight_bits,
                "ab": self.activation_bits,
                "q": self.signed_output_bound,
                "k": self.ring,
                "p": self.modulus,
                "x": self.wire_bits,
                "e": int(self.server_ns),
                "d": pack_residues(value, self.wire_bits),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(
        cls,
        payload: bytes | bytearray | memoryview,
        *,
        max_rows: int | None = None,
        max_tensor_elements: int | None = None,
    ) -> "CorrectionPush":
        try:
            value = msgpack.unpackb(
                payload,
                raw=False,
                strict_map_key=False,
                max_str_len=PREPARATION_MAX_IDENTIFIER_BYTES,
                max_bin_len=len(payload),
                max_array_len=0,
                max_map_len=18,
                max_ext_len=0,
            )
        except Exception as exc:
            raise ProtocolError("invalid correction push") from exc
        required = {
            "v",
            "a",
            "h",
            "m",
            "b",
            "t",
            "w",
            "r",
            "n",
            "o",
            "wb",
            "ab",
            "q",
            "k",
            "p",
            "x",
            "e",
            "d",
        }
        if (
            not isinstance(value, dict)
            or set(value) != required
            or int(value["v"]) != PREPARATION_PROTOCOL_VERSION
        ):
            raise ProtocolError("invalid correction push schema")
        rows, in_features, out_features = int(value["r"]), int(value["n"]), int(value["o"])
        _validate_shape(
            rows,
            in_features,
            out_features,
            kind="correction",
            max_rows=max_rows,
            max_tensor_elements=max_tensor_elements,
        )
        data = value["d"]
        wire_bits = int(value["x"])
        if not isinstance(data, (bytes, bytearray)) or len(data) != (
            rows * out_features * (wire_bits // 8)
        ):
            raise ProtocolError("correction payload length mismatch")
        result = cls(
            attempt_id=str(value["a"]),
            session_id=str(value["h"]),
            model=str(value["m"]),
            body_fingerprint=str(value["b"]),
            stage_id=str(value["t"]),
            weight_digest=str(value["w"]),
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            weight_bits=int(value["wb"]),
            activation_bits=int(value["ab"]),
            signed_output_bound=int(value["q"]),
            ring=str(value["k"]),
            modulus=int(value["p"]),
            wire_bits=wire_bits,  # type: ignore[arg-type]
            correction=unpack_residues(bytes(data), (rows, out_features), wire_bits),
            server_ns=int(value["e"]),
        )
        result._validate(max_rows=max_rows, max_tensor_elements=max_tensor_elements)
        return result


def derive_online_attempt_id(
    request: PreparationRequest | CorrectionPush,
    row_index: int,
) -> str:
    """Derive one online attempt ID from a bulk preparation attempt row."""
    request._validate()
    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise ProtocolError("bulk preparation row index must be an integer")
    if row_index < 0 or row_index >= request.rows:
        raise ProtocolError("bulk preparation row index is out of bounds")
    domain = msgpack.packb(
        [
            request.session_id,
            request.model,
            request.body_fingerprint,
            request.attempt_id,
            request.stage_id,
            request.weight_digest,
            request.rows,
            request.in_features,
            request.out_features,
            request.weight_bits,
            request.activation_bits,
            request.signed_output_bound,
            request.ring,
            request.modulus,
            request.wire_bits,
            row_index,
        ],
        use_bin_type=True,
    )
    digest = hashlib.sha256(b"pllm-online-attempt-id-v1\x00" + domain).digest()
    return digest[:16].hex()


@dataclass(frozen=True, slots=True)
class SessionAuthorizationAck:
    session_id: str

    def pack(self) -> bytes:
        _validate_identifiers(self.session_id, kind="session authorization acknowledgement")
        return msgpack.packb(
            {"v": PREPARATION_PROTOCOL_VERSION, "h": self.session_id},
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "SessionAuthorizationAck":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise ProtocolError("invalid session authorization acknowledgement") from exc
        if not isinstance(value, dict) or set(value) != {"v", "h"}:
            raise ProtocolError("invalid session authorization acknowledgement schema")
        result = cls(str(value["h"]))
        _validate_identifiers(result.session_id, kind="session authorization acknowledgement")
        return result


@dataclass(frozen=True, slots=True)
class PreparationAck:
    attempt_id: str
    stage_id: str
    correction_bytes: int
    server_ns: int
    push_ns: int = 0
    verification_component: str = "none"
    verification_checks: int = 0
    verification_max_row_l1: int = 0
    verification_material_id: bytes = b""
    verification_tag: bytes = b""
    verification_payload: bytes = b""

    def _validate_verification(self) -> None:
        fields = (
            self.verification_checks,
            self.verification_max_row_l1,
            self.verification_material_id,
            self.verification_tag,
            self.verification_payload,
        )
        if self.verification_component == "none":
            if any(fields):
                raise ProtocolError("disabled verification carries material")
            return
        if self.verification_component != "pllm/freivalds-verify/v1":
            raise ProtocolError("unsupported verification acknowledgement")
        if not 1 <= self.verification_checks <= 8 or self.verification_max_row_l1 <= 0:
            raise ProtocolError("invalid Freivalds verification metadata")
        if len(self.verification_material_id) != 32 or len(self.verification_tag) != 16:
            raise ProtocolError("invalid Freivalds material identity")
        if not self.verification_payload or len(self.verification_payload) % 4:
            raise ProtocolError("invalid Freivalds projection payload")

    def pack(self) -> bytes:
        validate_attempt_id(self.attempt_id)
        self._validate_verification()
        value = {
            "v": PREPARATION_PROTOCOL_VERSION,
            "a": self.attempt_id,
            "t": self.stage_id,
            "c": int(self.correction_bytes),
            "e": int(self.server_ns),
            "p": int(self.push_ns),
        }
        if self.verification_component != "none":
            value.update(
                {
                    "vc": self.verification_component,
                    "vk": self.verification_checks,
                    "vl": self.verification_max_row_l1,
                    "vi": self.verification_material_id,
                    "vt": self.verification_tag,
                    "vp": self.verification_payload,
                }
            )
        return msgpack.packb(value, use_bin_type=True)

    @classmethod
    def unpack(cls, payload: bytes) -> "PreparationAck":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise ProtocolError("invalid preparation acknowledgement") from exc
        base_keys = {"v", "a", "t", "c", "e", "p"}
        verification_keys = {"vc", "vk", "vl", "vi", "vt", "vp"}
        if (
            not isinstance(value, dict)
            or set(value) not in {frozenset(base_keys), frozenset(base_keys | verification_keys)}
            or int(value["v"]) != PREPARATION_PROTOCOL_VERSION
        ):
            raise ProtocolError("invalid preparation acknowledgement schema")
        result = cls(
            str(value["a"]),
            str(value["t"]),
            int(value["c"]),
            int(value["e"]),
            int(value["p"]),
            str(value.get("vc", "none")),
            int(value.get("vk", 0)),
            int(value.get("vl", 0)),
            bytes(value.get("vi", b"")),
            bytes(value.get("vt", b"")),
            bytes(value.get("vp", b"")),
        )
        validate_attempt_id(result.attempt_id)
        if result.correction_bytes <= 0 or result.server_ns < 0 or result.push_ns < 0:
            raise ProtocolError("invalid preparation acknowledgement")
        result._validate_verification()
        return result
