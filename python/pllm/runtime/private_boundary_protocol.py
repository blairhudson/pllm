"""Strict, public commitments and bounded frames for a coupled private boundary.

This protocol does not admit a decoder. Applications must supply a trusted
compiled-contract admission callback to the worker factory. Preparation and the
prepared decoder body are deliberately outside this protocol.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any

import msgpack

ROLES = ("boundary_a", "boundary_b")
SCHEMA = "pllm.private_boundary.v1"
MAX_LOOKUP_ROWS = 4096
MAX_HEADS = 4096
MAX_LOOKUP_BATCH = 64
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_CONTROL_BYTES = 16384
PACKED_QUERY_MAGIC = b"PLLMPQ02"


class BoundaryError(ValueError):
    """An untrusted frame or compiled boundary differs from its contract."""


def digest(value: Any, name: str = "digest") -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise BoundaryError(f"invalid {name}")
    return value


def integer(value: Any, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise BoundaryError(f"invalid {name}")
    return value


def session_id(value: Any) -> str:
    if type(value) is not str or len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise BoundaryError("invalid paired session ID")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise BoundaryError("duplicate or non-string field")
        result[key] = value
    return result


def object_fields(value: Any, keys: set[str]) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise BoundaryError("boundary fields differ")
    return value


def parse_control(raw: bytes) -> dict:
    if type(raw) is not bytes or len(raw) > MAX_CONTROL_BYTES:
        raise BoundaryError("boundary control exceeds limit")
    try:
        value = json.loads(raw, object_pairs_hook=strict_object)
    except (UnicodeError, ValueError) as exc:
        raise BoundaryError("invalid boundary control") from exc
    if type(value) is not dict:
        raise BoundaryError("boundary control is not an object")
    return value


@dataclass(frozen=True, slots=True)
class BoundaryDescriptor:
    """Identity supplied from validated compiled stages, never from a request.

    The first version supports a tied W8 table with records ``i8[hidden], f32``
    and an exact u32 wrapping head. Both A4 and A8 activations are admitted.
    Bias is absent; all head scales and full logits remain client-owned.

    ``source_digest`` is source_lock_digest, ``checkpoint_digest`` is the source
    checkpoint_digest, and ``model_config_digest`` is the re-lowered plan's
    config_digest. Head-only loaders derive these independently. Prepared body
    fingerprint/runtime-config commitments remain with ordinary Inference.
    """

    source_digest: str
    checkpoint_digest: str
    model_config_digest: str
    composition_digest: str
    schedule_digest: str
    head_weight_digest: str
    head_scales_digest: str
    lookup_digest: str
    vocabulary: int
    hidden: int
    signed_output_bound: int
    head_stage_id: str = "lm_head"
    activation_bits: int = 8
    weight_bits: int = 8
    ring: str = "u32"
    wire_bits: int = 32
    modulus: int = 1 << 32
    leaf_bits: int = 128
    head_dispatch: str = "serial"
    head_output_encoding: str = "u32"
    head_kernel: str = "existing"

    def __post_init__(self):
        if type(self.head_kernel) is not str or self.head_kernel not in ("existing", "byte_dot"):
            raise BoundaryError("invalid head kernel")
        if type(self.head_dispatch) is not str or self.head_dispatch not in ("serial", "parallel"):
            raise BoundaryError("invalid head dispatch")
        if type(self.head_output_encoding) is not str or self.head_output_encoding not in ("u32", "uniform_residues"):
            raise BoundaryError("invalid head output encoding")
        for field in fields(self):
            if field.name.endswith("digest"):
                digest(getattr(self, field.name), field.name)
        integer(self.vocabulary, 1, 262144, "vocabulary")
        integer(self.hidden, 1, 65532, "hidden")
        if self.head_kernel == "byte_dot" and self.vocabulary * self.hidden > 256 * 1024 * 1024:
            raise BoundaryError("byte-dot matrix exceeds native bound")
        if self.vocabulary * (self.hidden + 4) > 512 * 1024 * 1024:
            raise BoundaryError("boundary exceeds native private table capacity")
        if self.vocabulary * 4 + 8192 > MAX_BODY_BYTES or 64 * (self.hidden + 4 + 256) > MAX_BODY_BYTES:
            raise BoundaryError("boundary geometry exceeds frame capacity")
        integer(self.activation_bits, 4, 8, "activation bits")
        if self.activation_bits not in (4, 8) or type(self.weight_bits) is not int or self.weight_bits != 8:
            raise BoundaryError("boundary requires W8 and A4/A8")
        if (self.ring, self.wire_bits, self.modulus, self.leaf_bits) != ("u32", 32, 1 << 32, 128):
            raise BoundaryError("boundary requires exact wrap32 and packed128")
        if any(type(x) is not int for x in (self.wire_bits, self.modulus, self.leaf_bits)):
            raise BoundaryError("invalid boundary numeric profile")
        integer(self.signed_output_bound, 0, (1 << 31) - 1, "signed output bound")
        if self.signed_output_bound > self.hidden * 128 * ((1 << (self.activation_bits - 1)) - 1):
            raise BoundaryError("signed bound exceeds W8 activation geometry")
        if type(self.head_stage_id) is not str or not 1 <= len(self.head_stage_id) <= 128 or any(
            not (c.isascii() and (c.isalnum() or c in "_.-")) for c in self.head_stage_id
        ):
            raise BoundaryError("invalid head stage ID")

    def to_dict(self) -> dict:
        row = {"schema": SCHEMA + ".descriptor", **asdict(self)}
        for name, default in (("head_dispatch", "serial"), ("head_output_encoding", "u32"),
                              ("head_kernel", "existing")):
            if row[name] == default:
                del row[name]
        return row

    @classmethod
    def from_dict(cls, value: Any):
        optional = {"head_dispatch", "head_output_encoding", "head_kernel"}
        required = {"schema", *(f.name for f in fields(cls))} - optional
        if type(value) is not dict or not required <= set(value) or set(value) - required - optional:
            raise BoundaryError("boundary fields differ")
        row = value
        if row["schema"] != SCHEMA + ".descriptor":
            raise BoundaryError("boundary descriptor schema differs")
        return cls(**{k: v for k, v in row.items() if k != "schema"})

    @property
    def digest(self) -> str:
        return hashlib.sha256(b"pllm/private-boundary-descriptor/v1\0" + canonical(self.to_dict())).hexdigest()

    @property
    def record_bytes(self) -> int:
        return self.hidden + 4


@dataclass(frozen=True, slots=True)
class BoundaryAdmission:
    """Bounded result of trusted application/native decoder admission.

    The callback must reconstruct/validate the compiled plan and its schedule,
    rather than accepting client-declared digests as validation evidence.
    """

    descriptor_digest: str
    plan_digest: str
    lookup_rows: int
    head_calls: int

    def __post_init__(self):
        digest(self.descriptor_digest)
        digest(self.plan_digest)
        integer(self.lookup_rows, 1, MAX_LOOKUP_ROWS, "lookup rows")
        integer(self.head_calls, 1, MAX_HEADS, "head calls")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**object_fields(value, {f.name for f in fields(cls)}))


def context_bytes(descriptor: BoundaryDescriptor, admission: BoundaryAdmission, paired_session: str, epoch: int) -> bytes:
    session_id(paired_session)
    integer(epoch, 0, (1 << 32) - 1, "epoch")
    if admission.descriptor_digest != descriptor.digest:
        raise BoundaryError("admission descriptor differs")
    return hashlib.sha256(b"pllm/private-boundary-session/v1\0" + canonical({
        "descriptor": descriptor.digest, "admission": admission.to_dict(),
        "session": paired_session, "epoch": epoch, "roles": list(ROLES),
    })).digest()


def contextual_lookup_digest(descriptor: BoundaryDescriptor, context: bytes) -> bytes:
    if type(context) is not bytes or len(context) != 32:
        raise BoundaryError("invalid lookup context")
    return hashlib.sha256(b"pllm/private-page-context/v1\0" + bytes.fromhex(descriptor.lookup_digest) + context).digest()


def public_lookup_digest(rows, *, vocabulary: int, hidden: int) -> str:
    """Hash original native table identity from bounded (i8-row, LE-f32) chunks."""
    integer(vocabulary, 1, 1_048_576, "vocabulary")
    integer(hidden, 1, 1_048_576, "hidden")
    value = hashlib.sha256(b"pllm/public-page-table/v1\0")
    value.update(vocabulary.to_bytes(8, "little"))
    value.update((hidden + 4).to_bytes(8, "little"))
    count = 0
    for weight, scale in rows:
        if count >= vocabulary or len(weight) != hidden or len(scale) != 4:
            raise BoundaryError("public lookup row geometry differs")
        value.update(weight)
        value.update(scale)
        count += 1
    if count != vocabulary:
        raise BoundaryError("public lookup row count differs")
    return value.hexdigest()


def pack_frame(*, role: str, paired_session: str, epoch: int, sequence: int,
               operation: str, frames: list[bytes], response: bool = False) -> bytes:
    row = {"schema": SCHEMA + (".reply" if response else ".frame"), "role": role,
           "session": paired_session, "epoch": epoch, "sequence": sequence,
           "operation": operation, "frames": frames}
    validate_frame(row, response=response)
    raw = msgpack.packb(row, use_bin_type=True)
    if len(raw) > MAX_BODY_BYTES:
        raise BoundaryError("boundary frame exceeds limit")
    return raw


def validate_frame(row, *, response=False):
    object_fields(row, {"schema", "role", "session", "epoch", "sequence", "operation", "frames"})
    if row["schema"] != SCHEMA + (".reply" if response else ".frame") or row["role"] not in ROLES:
        raise BoundaryError("boundary frame role/schema differs")
    session_id(row["session"])
    integer(row["epoch"], 0, (1 << 32) - 1, "epoch")
    integer(row["sequence"], 0, MAX_LOOKUP_ROWS + MAX_HEADS - 1, "operation sequence")
    if row["operation"] not in ("lookup", "head"):
        raise BoundaryError("unknown boundary operation")
    values = row["frames"]
    if type(values) is not list or not 1 <= len(values) <= (64 if row["operation"] == "lookup" else 1):
        raise BoundaryError("invalid boundary batch")
    if any(type(value) is not bytes or not value for value in values):
        raise BoundaryError("invalid boundary binary frame")
    return row


def unpack_frame(raw: bytes, *, response=False) -> dict:
    if type(raw) is not bytes or len(raw) > MAX_BODY_BYTES:
        raise BoundaryError("boundary frame exceeds limit")
    try:
        row = msgpack.unpackb(raw, raw=False, strict_map_key=True, object_pairs_hook=strict_object,
                              max_bin_len=MAX_BODY_BYTES, max_str_len=256, max_array_len=64, max_map_len=8)
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise BoundaryError("invalid boundary binary document") from exc
    return validate_frame(row, response=response)
