"""Public, commitment-bound output layouts shared by linear role protocols."""
from __future__ import annotations

import hashlib
import base64
import zlib
import json

import msgpack

from pllm import _native
from .stage_protocol import ProtocolError

ROW_MAGIC = b"PLLMOR01"
PREPARED_ROW_MAGIC = b"PLLMPR01"
MAX_VALUES = 4_000_000
MAX_ROWS = 4096


def encode_layout(widths: bytes) -> dict:
    """Keep public layouts inline instead of fetching an HTTP artifact per stage."""
    if type(widths) is not bytes or not 0 < len(widths) <= MAX_VALUES or min(widths) < 1 or max(widths) > 32:
        raise ProtocolError("invalid public residue layout")
    return {"schema": "pllm.residue_layout.v1", "count": len(widths),
            "sha256": hashlib.sha256(widths).hexdigest(),
            "zlib_base64": base64.b64encode(zlib.compress(widths, 1)).decode("ascii")}


def decode_layout(record, expected_count: int) -> bytes:
    if (type(record) is not dict or set(record) != {"schema", "count", "sha256", "zlib_base64"}
        or record["schema"] != "pllm.residue_layout.v1"
        or type(record["count"]) is not int or record["count"] != expected_count
        or not 0 < expected_count <= MAX_VALUES or type(record["zlib_base64"]) not in (str, bytes)
        or len(record["zlib_base64"]) > 2 * expected_count + 256):
        raise ProtocolError("invalid public residue layout record")
    try:
        packed = base64.b64decode(record["zlib_base64"], validate=True)
        decoder = zlib.decompressobj()
        widths = decoder.decompress(packed, expected_count + 1)
    except (ValueError, zlib.error) as exc:
        raise ProtocolError("invalid compressed public residue layout") from exc
    if (len(widths) != expected_count or not decoder.eof or decoder.unconsumed_tail
        or decoder.unused_data or hashlib.sha256(widths).hexdigest() != record["sha256"]
        or min(widths) < 1 or max(widths) > 32):
        raise ProtocolError("compressed public residue layout differs from commitment")
    return widths


def _magic(namespace: str) -> bytes:
    if namespace not in {"offset", "prepared"}:
        raise ProtocolError("unknown residue-codec namespace")
    return ROW_MAGIC if namespace == "offset" else PREPARED_ROW_MAGIC


def row_layout_digest(records, *, namespace: str = "offset") -> str:
    _magic(namespace)
    return hashlib.sha256(f"pllm/{namespace}-row-layout/v1\0".encode()
        + json.dumps(sorted(records), separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def compiled_row_layout(compiled, *, namespace: str = "offset", packed_input: bool = False) -> str:
    records = []
    prefill_rows = compiled._plan.prefill["query_sequence"] if namespace == "prepared" else 1
    for binding in compiled._stages:
        if binding.client_weight_layout is not None:
            continue
        stage = compiled._bundle.stages[binding.stage_id]
        widths = stage.output_residue_bits
        if (type(widths) is not bytes or len(widths) != stage.out_features
            or stage.seeded_profile is None
            or any(not 1 <= width <= stage.seeded_profile.wire_bits for width in widths)):
            raise ProtocolError("compiled stage lacks an exact residue layout")
        maximum_rows = 1 if stage.op == "lm_head" else prefill_rows
        if namespace == "prepared" and (maximum_rows > MAX_ROWS or maximum_rows * stage.out_features > MAX_VALUES):
            raise ProtocolError("compiled residue output exceeds bounded codec capacity")
        if packed_input and maximum_rows * stage.in_features > MAX_VALUES:
            raise ProtocolError("compiled residue input exceeds bounded codec capacity")
        records.append((binding.stage_id, stage.weight_digest, stage.in_features,
                        stage.out_features, stage.activation_bits, widths.hex()))
    digest = row_layout_digest(records, namespace=namespace)
    if not records or digest != compiled._bundle.manifest["metadata"].get(f"{namespace}_residue_layout_digest"):
        raise ProtocolError("compiled residue layout differs from bundle")
    return digest


def pack_row_response(response, widths: bytes, *, namespace: str = "offset") -> bytes:
    rows, columns = response.masked_output.shape
    if columns != len(widths):
        raise ProtocolError("row-residue response width differs")
    encoded = _native.offset_pack_rows(response.masked_output.astype("<u4", copy=False).tobytes(), widths, rows)
    return _magic(namespace) + msgpack.packb([
        response.correlation_id, response.stage_id, rows, response.wire_bits,
        hashlib.sha256(widths).digest(), response.server_ns, encoded,
    ], use_bin_type=True)


def unpack_row_response(payload: bytes, *, ticket: str, stage: str, widths: bytes,
                        rows: int, bits: int, namespace: str = "offset") -> tuple[bytes, int]:
    magic = _magic(namespace)
    if not payload.startswith(magic) or len(payload) > 16 * 1024 * 1024 + 8192:
        raise ProtocolError("row response encoding differs")
    try:
        record = msgpack.unpackb(payload[len(magic):], raw=False, max_array_len=7,
                                max_map_len=0, max_str_len=512, max_ext_len=0)
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise ProtocolError("invalid row response") from exc
    if (type(record) is not list or len(record) != 7
        or record[:5] != [ticket, stage, rows, bits, hashlib.sha256(widths).digest()]
        or type(record[2]) is not int or type(record[3]) is not int
        or type(record[5]) is not int or not 0 <= record[5] < 1 << 63
        or type(record[6]) is not bytes):
        raise ProtocolError("row response differs from its stage contract")
    return record[6], record[5]
