from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import msgpack
import numpy as np

from .protocol import ProtocolError
from .quantization import centered_residues, positive_residues

STAGE_PROTOCOL_VERSION = 1
RingKind = Literal["u16", "u32", "prime"]


def _ring_from(modulus: int, wire_bits: int, ring: str | None) -> RingKind:
    if ring is not None:
        if ring not in {"u16", "u32", "prime"}:
            raise ProtocolError("unsupported ring kind")
        return ring  # type: ignore[return-value]
    if modulus == 1 << 16 and wire_bits == 16:
        return "u16"
    if modulus == 1 << 32 and wire_bits == 32:
        return "u32"
    return "prime"


def pack_residues(values: np.ndarray, wire_bits: int) -> bytes:
    from .compact import pack_unsigned, CompactCodecError
    if wire_bits not in {16, 24, 32}:
        raise ProtocolError("wire_bits must be 16, 24, or 32")
    try:
        return pack_unsigned(values, width=wire_bits // 8)
    except (ValueError, CompactCodecError) as exc:
        raise ProtocolError(str(exc)) from exc


def unpack_residues(payload: bytes, shape: tuple[int, ...], wire_bits: int) -> np.ndarray:
    from .compact import unpack_unsigned, CompactCodecError
    import math
    if wire_bits not in {16, 24, 32} or any(d < 0 for d in shape):
        raise ProtocolError("invalid residue shape or wire width")
    try:
        return unpack_unsigned(payload, width=wire_bits // 8, count=math.prod(shape)).reshape(shape)
    except (ValueError, CompactCodecError) as exc:
        raise ProtocolError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class MaskedStageRequest:
    model: str
    stage_id: str
    correlation_id: str
    masked_input: np.ndarray
    activation_scales: np.ndarray | float
    modulus: int
    wire_bits: int
    ring: RingKind | None = None

    @property
    def stage(self) -> str:
        return self.stage_id

    @property
    def activation_scale(self) -> float:
        return float(np.asarray(self.activation_scales).reshape(-1)[0])

    def pack(self) -> bytes:
        value = np.asarray(self.masked_input, dtype=np.uint32)
        if value.ndim not in {1, 2} or value.size == 0:
            raise ProtocolError("stage input must be a non-empty vector or matrix")
        scales = np.asarray(self.activation_scales, dtype=np.float32).reshape(-1)
        rows = 1 if value.ndim == 1 else value.shape[0]
        if scales.size not in {1, rows}:
            raise ProtocolError("activation scale count does not match stage rows")
        ring = _ring_from(self.modulus, self.wire_bits, self.ring)
        return msgpack.packb(
            {
                "v": STAGE_PROTOCOL_VERSION,
                "model": self.model,
                "stage_id": self.stage_id,
                "correlation_id": self.correlation_id,
                "ring": ring,
                "modulus": int(self.modulus),
                "wire_bits": int(self.wire_bits),
                "shape": list(value.shape),
                # The true activation scales stay on the client. This v1 field
                # is retained only as a public constant for wire compatibility.
                "scales": np.ones(rows, dtype="<f4").tobytes(),
                "data": pack_residues(value, self.wire_bits),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "MaskedStageRequest":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise ProtocolError("invalid masked stage request") from exc
        required = {
            "v", "model", "stage_id", "correlation_id", "ring", "modulus",
            "wire_bits", "shape", "scales", "data",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ProtocolError("invalid masked stage request schema")
        if int(value["v"]) != STAGE_PROTOCOL_VERSION:
            raise ProtocolError("unsupported masked stage protocol")
        shape = tuple(int(item) for item in value["shape"])
        if len(shape) not in {1, 2} or any(item <= 0 for item in shape):
            raise ProtocolError("invalid masked stage input shape")
        wire_bits = int(value["wire_bits"])
        scales = np.frombuffer(value["scales"], dtype="<f4").copy()
        rows = 1 if len(shape) == 1 else shape[0]
        if scales.size not in {1, rows}:
            raise ProtocolError("activation scale count does not match stage rows")
        ring = _ring_from(int(value["modulus"]), wire_bits, str(value["ring"]))
        return cls(
            model=str(value["model"]),
            stage_id=str(value["stage_id"]),
            correlation_id=str(value["correlation_id"]),
            masked_input=unpack_residues(value["data"], shape, wire_bits),
            activation_scales=scales,
            modulus=int(value["modulus"]),
            wire_bits=wire_bits,
            ring=ring,
        )


@dataclass(frozen=True, slots=True)
class MaskedStageResponse:
    correlation_id: str
    masked_output: np.ndarray
    modulus: int
    wire_bits: int
    server_ns: int = 0
    stage_id: str = ""
    ring: RingKind | None = None

    def pack(self) -> bytes:
        value = np.asarray(self.masked_output, dtype=np.uint32)
        if value.ndim not in {1, 2} or value.size == 0:
            raise ProtocolError("stage output must be a non-empty vector or matrix")
        ring = _ring_from(self.modulus, self.wire_bits, self.ring)
        return msgpack.packb(
            {
                "v": STAGE_PROTOCOL_VERSION,
                "stage_id": self.stage_id,
                "correlation_id": self.correlation_id,
                "ring": ring,
                "modulus": int(self.modulus),
                "wire_bits": int(self.wire_bits),
                "server_ns": int(self.server_ns),
                "shape": list(value.shape),
                "data": pack_residues(value, self.wire_bits),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "MaskedStageResponse":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise ProtocolError("invalid masked stage response") from exc
        required = {
            "v", "stage_id", "correlation_id", "ring", "modulus", "wire_bits",
            "server_ns", "shape", "data",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ProtocolError("invalid masked stage response schema")
        if int(value["v"]) != STAGE_PROTOCOL_VERSION:
            raise ProtocolError("unsupported masked stage protocol")
        shape = tuple(int(item) for item in value["shape"])
        if len(shape) not in {1, 2} or any(item <= 0 for item in shape):
            raise ProtocolError("invalid stage output shape")
        wire_bits = int(value["wire_bits"])
        ring = _ring_from(int(value["modulus"]), wire_bits, str(value["ring"]))
        return cls(
            correlation_id=str(value["correlation_id"]),
            masked_output=unpack_residues(value["data"], shape, wire_bits),
            modulus=int(value["modulus"]),
            wire_bits=wire_bits,
            server_ns=int(value["server_ns"]),
            stage_id=str(value["stage_id"]),
            ring=ring,
        )


@dataclass(slots=True)
class StageCorrelation:
    id: str
    stage_id: str
    mask: np.ndarray
    transformed_mask: np.ndarray
    modulus: int
    ring: RingKind | None = None
    consumed: bool = False


def unmask_stage_output(response: MaskedStageResponse, correlation: StageCorrelation) -> np.ndarray:
    if response.correlation_id != correlation.id:
        raise ProtocolError("stage result correlation id mismatch")
    if response.modulus != correlation.modulus:
        raise ProtocolError("stage result modulus mismatch")
    value = np.asarray(response.masked_output, dtype=np.uint32)
    transformed = np.asarray(correlation.transformed_mask, dtype=np.uint32)
    if value.shape != transformed.shape:
        raise ProtocolError("stage result and transformed mask shapes differ")
    ring = _ring_from(response.modulus, response.wire_bits, response.ring or correlation.ring)
    if ring == "u16":
        raw = (value.astype(np.uint16) - transformed.astype(np.uint16)).astype(np.uint16)
        return raw.view(np.int16).astype(np.int64)
    if ring == "u32":
        raw = (value.astype(np.uint32) - transformed.astype(np.uint32)).astype(np.uint32)
        return raw.view(np.int32).astype(np.int64)
    from ._native_support import extension
    native = extension()
    if native is not None:
        data = native.unmask(value.astype("<u4", copy=False).tobytes(), transformed.astype("<u4", copy=False).tobytes(), response.modulus)
        return np.frombuffer(data, dtype="<i8").reshape(value.shape)
    return centered_residues(value.astype(np.int64) - transformed.astype(np.int64), response.modulus)


def encode_signed_with_mask(values: np.ndarray, correlation: StageCorrelation) -> np.ndarray:
    signed = np.asarray(values, dtype=np.int64)
    mask = np.asarray(correlation.mask)
    if signed.shape != mask.shape:
        raise ProtocolError("activation and correlation shapes differ")
    ring = correlation.ring or "prime"
    if ring == "u16":
        return (signed.astype(np.uint16) + mask.astype(np.uint16)).astype(np.uint16)
    if ring == "u32":
        return (signed.astype(np.uint32) + mask.astype(np.uint32)).astype(np.uint32)
    from ._native_support import extension
    native = extension()
    if native is not None and (not signed.size or (signed.min() >= -128 and signed.max() <= 127)):
        data = native.mask(signed.astype(np.int8).tobytes(), mask.astype("<u4", copy=False).tobytes(), correlation.modulus)
        return np.frombuffer(data, dtype="<u4").reshape(signed.shape)
    return positive_residues(signed + mask.astype(np.int64), correlation.modulus)


def decode_unmasked(values: np.ndarray, correlation: StageCorrelation) -> np.ndarray:
    response = MaskedStageResponse(
        correlation_id=correlation.id,
        masked_output=np.asarray(values),
        modulus=correlation.modulus,
        wire_bits=16 if correlation.ring == "u16" else 32,
        ring=correlation.ring or "prime",
    )
    return unmask_stage_output(response, correlation)


def correlation_to_wire(item: StageCorrelation) -> dict[str, Any]:
    ring = item.ring or "prime"
    wire_bits = 16 if ring == "u16" else 32
    return {
        "id": item.id,
        "stage_id": item.stage_id,
        "ring": ring,
        "modulus": int(item.modulus),
        "wire_bits": wire_bits,
        "input_shape": list(item.mask.shape),
        "output_shape": list(item.transformed_mask.shape),
        "mask": pack_residues(np.asarray(item.mask, dtype=np.uint32), wire_bits),
        "transformed": pack_residues(np.asarray(item.transformed_mask, dtype=np.uint32), wire_bits),
    }


def correlation_from_wire(value: dict[str, Any]) -> StageCorrelation:
    required = {
        "id", "stage_id", "ring", "modulus", "wire_bits", "input_shape",
        "output_shape", "mask", "transformed",
    }
    if set(value) != required:
        raise ProtocolError("invalid stage correlation schema")
    ring = _ring_from(int(value["modulus"]), int(value["wire_bits"]), str(value["ring"]))
    return StageCorrelation(
        id=str(value["id"]),
        stage_id=str(value["stage_id"]),
        mask=unpack_residues(value["mask"], value["input_shape"], int(value["wire_bits"])),
        transformed_mask=unpack_residues(
            value["transformed"], value["output_shape"], int(value["wire_bits"])
        ),
        modulus=int(value["modulus"]),
        ring=ring,
    )


DIRECT_FHE_STAGE_VERSION = 1


@dataclass(frozen=True, slots=True)
class DirectFHEStageRequest:
    """Encrypted activation rows for the proprietary-weight mode."""

    model: str
    stage_id: str
    context_id: str
    input_shape: tuple[int, ...]
    activation_scales: np.ndarray
    encrypted_rows: tuple[bytes, ...]

    def pack(self) -> bytes:
        shape = tuple(int(item) for item in self.input_shape)
        if len(shape) < 1 or any(item <= 0 for item in shape):
            raise ProtocolError("invalid direct-FHE input shape")
        rows = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
        scales = np.asarray(self.activation_scales, dtype=np.float32).reshape(-1)
        if scales.shape != (rows,):
            raise ProtocolError("direct-FHE activation scale count mismatch")
        if len(self.encrypted_rows) != rows or any(not item for item in self.encrypted_rows):
            raise ProtocolError("direct-FHE encrypted row count mismatch")
        return msgpack.packb(
            {
                "v": DIRECT_FHE_STAGE_VERSION,
                "model": self.model,
                "stage_id": self.stage_id,
                "context_id": self.context_id,
                "input_shape": list(shape),
                # The true activation scales stay on the client. This v1 field
                # is retained only as a public constant for wire compatibility.
                "scales": np.ones(rows, dtype="<f4").tobytes(),
                "encrypted_rows": list(self.encrypted_rows),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "DirectFHEStageRequest":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise ProtocolError("invalid direct-FHE stage request") from exc
        required = {
            "v", "model", "stage_id", "context_id", "input_shape", "scales", "encrypted_rows",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ProtocolError("invalid direct-FHE stage request schema")
        if int(value["v"]) != DIRECT_FHE_STAGE_VERSION:
            raise ProtocolError("unsupported direct-FHE stage protocol")
        shape = tuple(int(item) for item in value["input_shape"])
        if len(shape) < 1 or any(item <= 0 for item in shape):
            raise ProtocolError("invalid direct-FHE input shape")
        rows = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
        scales = np.frombuffer(value["scales"], dtype="<f4").copy()
        encrypted_rows = tuple(bytes(item) for item in value["encrypted_rows"])
        if scales.shape != (rows,) or len(encrypted_rows) != rows:
            raise ProtocolError("direct-FHE row metadata mismatch")
        return cls(
            model=str(value["model"]),
            stage_id=str(value["stage_id"]),
            context_id=str(value["context_id"]),
            input_shape=shape,
            activation_scales=scales,
            encrypted_rows=encrypted_rows,
        )


@dataclass(frozen=True, slots=True)
class DirectFHEStageResponse:
    stage_id: str
    output_rows: tuple[bytes, ...]
    server_ns: int = 0

    def pack(self) -> bytes:
        if not self.output_rows or any(not item for item in self.output_rows):
            raise ProtocolError("direct-FHE stage response is empty")
        return msgpack.packb(
            {
                "v": DIRECT_FHE_STAGE_VERSION,
                "stage_id": self.stage_id,
                "server_ns": int(self.server_ns),
                "output_rows": list(self.output_rows),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "DirectFHEStageResponse":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise ProtocolError("invalid direct-FHE stage response") from exc
        if not isinstance(value, dict) or set(value) != {"v", "stage_id", "server_ns", "output_rows"}:
            raise ProtocolError("invalid direct-FHE stage response schema")
        if int(value["v"]) != DIRECT_FHE_STAGE_VERSION:
            raise ProtocolError("unsupported direct-FHE stage protocol")
        rows = tuple(bytes(item) for item in value["output_rows"])
        if not rows:
            raise ProtocolError("direct-FHE stage response is empty")
        return cls(
            stage_id=str(value["stage_id"]),
            output_rows=rows,
            server_ns=int(value["server_ns"]),
        )

BLINDED_STAGE_VERSION = 1


@dataclass(slots=True)
class BlindedStageCorrelation:
    """Client half of a model-private output-blinded OLE correlation.

    The client knows ``r`` and ``W r + s``. The server alone retains the
    uniformly random output mask ``s``. The pair can be consumed exactly once.
    """

    id: str
    stage_id: str
    owner_id: str
    mask: np.ndarray
    blinded_transformed_mask: np.ndarray
    modulus: int
    ring: RingKind | None = None
    consumed: bool = False


def blinded_correlation_to_wire(item: BlindedStageCorrelation) -> dict[str, Any]:
    ring = item.ring or "prime"
    wire_bits = 16 if ring == "u16" else 32
    return {
        "id": item.id,
        "stage_id": item.stage_id,
        "owner_id": item.owner_id,
        "ring": ring,
        "modulus": int(item.modulus),
        "wire_bits": wire_bits,
        "input_shape": list(item.mask.shape),
        "output_shape": list(item.blinded_transformed_mask.shape),
        "mask": pack_residues(np.asarray(item.mask, dtype=np.uint32), wire_bits),
        "blinded_transformed": pack_residues(
            np.asarray(item.blinded_transformed_mask, dtype=np.uint32), wire_bits
        ),
    }


def blinded_correlation_from_wire(value: dict[str, Any]) -> BlindedStageCorrelation:
    required = {
        "id", "stage_id", "owner_id", "ring", "modulus", "wire_bits", "input_shape",
        "output_shape", "mask", "blinded_transformed",
    }
    if set(value) != required:
        raise ProtocolError("invalid blinded stage correlation schema")
    ring = _ring_from(int(value["modulus"]), int(value["wire_bits"]), str(value["ring"]))
    return BlindedStageCorrelation(
        id=str(value["id"]),
        stage_id=str(value["stage_id"]),
        owner_id=str(value["owner_id"]),
        mask=unpack_residues(value["mask"], value["input_shape"], int(value["wire_bits"])),
        blinded_transformed_mask=unpack_residues(
            value["blinded_transformed"], value["output_shape"], int(value["wire_bits"])
        ),
        modulus=int(value["modulus"]),
        ring=ring,
    )


@dataclass(frozen=True, slots=True)
class BlindedStageRequest:
    """Fast proprietary-weight online stage request.

    Each row is ``x-r``. The server consumes the matching output mask ``s`` and
    returns ``W(x-r)-s``. The client adds its preprocessed ``Wr+s`` value to
    reconstruct ``Wx`` without ever receiving an unblinded ``Wr`` equation.
    """

    model: str
    stage_id: str
    owner_id: str
    correlation_ids: tuple[str, ...]
    masked_input: np.ndarray
    activation_scales: np.ndarray | float
    modulus: int
    wire_bits: int
    ring: RingKind | None = None

    def pack(self) -> bytes:
        value = np.asarray(self.masked_input, dtype=np.uint32)
        if value.ndim not in {1, 2} or value.size == 0:
            raise ProtocolError("blinded stage input must be a non-empty vector or matrix")
        rows = 1 if value.ndim == 1 else value.shape[0]
        if len(self.correlation_ids) != rows or len(set(self.correlation_ids)) != rows:
            raise ProtocolError("blinded stage correlation ids must be unique per row")
        scales = np.asarray(self.activation_scales, dtype=np.float32).reshape(-1)
        if scales.size not in {1, rows}:
            raise ProtocolError("activation scale count does not match blinded stage rows")
        ring = _ring_from(self.modulus, self.wire_bits, self.ring)
        return msgpack.packb(
            {
                "v": BLINDED_STAGE_VERSION,
                "model": self.model,
                "stage_id": self.stage_id,
                "owner_id": self.owner_id,
                "correlation_ids": list(self.correlation_ids),
                "ring": ring,
                "modulus": int(self.modulus),
                "wire_bits": int(self.wire_bits),
                "shape": list(value.shape),
                # The true activation scales stay on the client. This v1 field
                # is retained only as a public constant for wire compatibility.
                "scales": np.ones(rows, dtype="<f4").tobytes(),
                "data": pack_residues(value, self.wire_bits),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "BlindedStageRequest":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise ProtocolError("invalid blinded stage request") from exc
        required = {
            "v", "model", "stage_id", "owner_id", "correlation_ids", "ring",
            "modulus", "wire_bits", "shape", "scales", "data",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ProtocolError("invalid blinded stage request schema")
        if int(value["v"]) != BLINDED_STAGE_VERSION:
            raise ProtocolError("unsupported blinded stage protocol")
        shape = tuple(int(item) for item in value["shape"])
        if len(shape) not in {1, 2} or any(item <= 0 for item in shape):
            raise ProtocolError("invalid blinded stage input shape")
        rows = 1 if len(shape) == 1 else shape[0]
        ids = tuple(str(item) for item in value["correlation_ids"])
        if len(ids) != rows or len(set(ids)) != rows:
            raise ProtocolError("invalid blinded stage correlation ids")
        wire_bits = int(value["wire_bits"])
        scales = np.frombuffer(value["scales"], dtype="<f4").copy()
        if scales.size not in {1, rows}:
            raise ProtocolError("activation scale count does not match blinded stage rows")
        ring = _ring_from(int(value["modulus"]), wire_bits, str(value["ring"]))
        return cls(
            model=str(value["model"]),
            stage_id=str(value["stage_id"]),
            owner_id=str(value["owner_id"]),
            correlation_ids=ids,
            masked_input=unpack_residues(value["data"], shape, wire_bits),
            activation_scales=scales,
            modulus=int(value["modulus"]),
            wire_bits=wire_bits,
            ring=ring,
        )


@dataclass(frozen=True, slots=True)
class BlindedStageResponse:
    stage_id: str
    correlation_ids: tuple[str, ...]
    masked_output: np.ndarray
    modulus: int
    wire_bits: int
    server_ns: int = 0
    ring: RingKind | None = None

    def pack(self) -> bytes:
        value = np.asarray(self.masked_output, dtype=np.uint32)
        if value.ndim not in {1, 2} or value.size == 0:
            raise ProtocolError("blinded stage output must be a non-empty vector or matrix")
        rows = 1 if value.ndim == 1 else value.shape[0]
        if len(self.correlation_ids) != rows:
            raise ProtocolError("blinded stage response correlation count mismatch")
        ring = _ring_from(self.modulus, self.wire_bits, self.ring)
        return msgpack.packb(
            {
                "v": BLINDED_STAGE_VERSION,
                "stage_id": self.stage_id,
                "correlation_ids": list(self.correlation_ids),
                "ring": ring,
                "modulus": int(self.modulus),
                "wire_bits": int(self.wire_bits),
                "server_ns": int(self.server_ns),
                "shape": list(value.shape),
                "data": pack_residues(value, self.wire_bits),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "BlindedStageResponse":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise ProtocolError("invalid blinded stage response") from exc
        required = {
            "v", "stage_id", "correlation_ids", "ring", "modulus", "wire_bits",
            "server_ns", "shape", "data",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ProtocolError("invalid blinded stage response schema")
        if int(value["v"]) != BLINDED_STAGE_VERSION:
            raise ProtocolError("unsupported blinded stage protocol")
        shape = tuple(int(item) for item in value["shape"])
        if len(shape) not in {1, 2} or any(item <= 0 for item in shape):
            raise ProtocolError("invalid blinded stage output shape")
        rows = 1 if len(shape) == 1 else shape[0]
        ids = tuple(str(item) for item in value["correlation_ids"])
        if len(ids) != rows:
            raise ProtocolError("invalid blinded stage response correlation count")
        wire_bits = int(value["wire_bits"])
        ring = _ring_from(int(value["modulus"]), wire_bits, str(value["ring"]))
        return cls(
            stage_id=str(value["stage_id"]),
            correlation_ids=ids,
            masked_output=unpack_residues(value["data"], shape, wire_bits),
            modulus=int(value["modulus"]),
            wire_bits=wire_bits,
            server_ns=int(value["server_ns"]),
            ring=ring,
        )
