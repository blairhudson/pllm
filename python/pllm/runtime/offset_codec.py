"""Bounded, composition-bound share encodings for the two-worker protocol."""

from __future__ import annotations

import hashlib
import json

import msgpack
import numpy as np

from pllm import _native
from .stage_protocol import MaskedStageRequest, ProtocolError
from .residue_codec import (
    ROW_MAGIC as ROW_MAGIC,
    compiled_row_layout as compiled_row_layout,
    pack_row_response as pack_row_response,
    row_layout_digest as row_layout_digest,
    unpack_row_response as unpack_row_response,
)

SEED_MAGIC = b"PLLMOS01"


def seed_header(
    *, model, stage, body, weight, plan, composition, session, ticket, rows, columns, bits
) -> dict:
    return dict(
        model=model,
        stage=stage,
        body=body,
        weight=weight,
        plan=plan,
        composition=composition,
        session=session,
        ticket=ticket,
        rows=rows,
        columns=columns,
        bits=bits,
    )


def _validate(header: dict, *, max_rows: int) -> None:
    if type(header) is not dict or set(header) != {
        "model",
        "stage",
        "body",
        "weight",
        "plan",
        "composition",
        "session",
        "ticket",
        "rows",
        "columns",
        "bits",
    }:
        raise ProtocolError("invalid seeded offset header")
    for field in ("body", "weight", "plan", "composition", "session", "ticket"):
        value = header[field]
        size = 32 if field in {"session", "ticket"} else 64
        if (
            type(value) is not str
            or len(value) != size
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ProtocolError("invalid seeded offset commitment")
    if any(
        type(header[k]) is not str or not 1 <= len(header[k].encode()) <= 512
        for k in ("model", "stage")
    ):
        raise ProtocolError("invalid seeded offset identity")
    if (
        type(header["rows"]) is not int
        or not 1 <= header["rows"] <= max_rows
        or type(header["columns"]) is not int
        or not 1 <= header["columns"] <= 4_000_000
        or header["rows"] * header["columns"] > 4_000_000
        or type(header["bits"]) is not int
        or header["bits"] not in (16, 24, 32)
    ):
        raise ProtocolError("seeded offset dimensions exceed bounds")


def context(header: dict) -> bytes:
    _validate(header, max_rows=4096)
    return hashlib.sha256(
        b"pllm/offset-seed-context/v1\0"
        + json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).digest()


def pack_seed(header: dict, seed: bytes) -> bytes:
    context(header)
    if type(seed) is not bytes or len(seed) != 32:
        raise ProtocolError("offset seed must be 32 bytes")
    return SEED_MAGIC + msgpack.packb([header, seed], use_bin_type=True)


def unpack_seed(payload: bytes, *, max_rows: int) -> tuple[dict, bytes]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate seed header field")
            result[key] = value
        return result

    if not payload.startswith(SEED_MAGIC) or len(payload) > 8192:
        raise ProtocolError("invalid seeded offset envelope")
    try:
        record = msgpack.unpackb(payload[len(SEED_MAGIC) :], raw=False, object_pairs_hook=unique)
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise ProtocolError("invalid seeded offset envelope") from exc
    if (
        type(record) is not list
        or len(record) != 2
        or type(record[1]) is not bytes
        or len(record[1]) != 32
    ):
        raise ProtocolError("invalid seeded offset seed")
    _validate(record[0], max_rows=max_rows)
    return record[0], record[1]


def expand_request(header: dict, seed: bytes, entry) -> MaskedStageRequest:
    """Called only after the worker validates all bound header fields."""
    count = header["rows"] * header["columns"]
    share = np.frombuffer(
        _native.offset_seeded_share(seed, context(header), count, header["bits"]), dtype="<u4"
    ).reshape(header["rows"], header["columns"])
    profile = entry.seeded_profile
    return MaskedStageRequest(
        model=header["model"],
        stage_id=header["stage"],
        correlation_id=header["ticket"],
        masked_input=share,
        activation_scales=1.0,
        modulus=profile.modulus,
        wire_bits=profile.wire_bits,
        ring=profile.ring,
        body_fingerprint=header["body"],
        weight_digest=header["weight"],
        weight_bits=entry.spec.weight_bits,
        activation_bits=entry.spec.activation_bits,
        session_id=header["session"],
        out_features=entry.spec.out_features,
        signed_output_bound=profile.signed_output_bound,
    )
