"""Public envelopes for the offline-dealer, two-holder FSS transaction."""
from __future__ import annotations

import json

from .delegated_protocol import clear_failure_frames, digest, pack, unpack
from .protocol import FRAME_HEADER, encode_length_prefixed, iter_length_prefixed

HOLDERS = ("worker_a", "worker_b")
ROLES = (*HOLDERS, "dealer")
TRANSPORT = "fss-offline-tape-v1"
ADMISSION_LIMIT = 2 << 20
CONTROL_LIMIT = 16 << 10
METADATA_LIMIT = 4096
PACKET_LIMIT = 16 << 20
FRAME_LIMIT = PACKET_LIMIT + (16 << 10)
# Native header, up to four token shares and one checked-validity share, checksum.
# Native admission still checks the exact namespace/count for each request.
FINAL_FRAME_LIMIT = 112 + 5 * 8 + 32
CLIENT_MEMORY = 64 << 20
ROLE_MEMORY = 256 << 20
MATERIAL_BATCH_RECORDS = 64
MATERIAL_BATCH_BYTES = 1 << 20  # complete MessagePack HTTP body, not just records
MATERIAL_BATCH_ENVELOPE_BYTES = 1024
MATERIAL_BATCH_RECORD_BYTES = MATERIAL_BATCH_BYTES - MATERIAL_BATCH_ENVELOPE_BYTES
# Additional Python ownership, not native's queue/six-slot reservation: original
# PyO3 records, encoded frames/bytearray/bytes, MessagePack body/staging, HTTP body
# and decode copies can overlap. Tuple/maps and bounded metadata staging have a
# separate allowance. This is a payload-owner bound, not an allocator/RSS claim.
MATERIAL_BATCH_PYTHON_BYTES = 10 * MATERIAL_BATCH_BYTES + 2 * MATERIAL_BATCH_RECORDS * METADATA_LIMIT + 4096
PEER_COUNTERS = frozenset(
    f"peer_link.{source}.{destination}.{phase}_bytes"
    for source in ROLES for destination in ROLES if source != destination
    for phase in ("setup", "online")
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def metadata(raw):
    """No type-coercing equality for native metadata, and no duplicate keys."""
    if type(raw) is not bytes or not 0 < len(raw) <= METADATA_LIMIT:
        raise ValueError("FuseFSS metadata exceeds its bound")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("FuseFSS duplicate metadata key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    if type(value) is not dict or canonical(value) != raw:
        raise ValueError("FuseFSS metadata is not canonical")
    return value


def material_delivery(value="single"):
    if type(value) is not str or value not in ("single", "batch-v1"):
        raise ValueError("FuseFSS material delivery differs")
    return value


def material_batch_size(records):
    """Validate originals before allocating an encoded copy."""
    if type(records) is not tuple or not 1 <= len(records) <= MATERIAL_BATCH_RECORDS:
        raise ValueError("FuseFSS material batch count differs")
    total = 0
    for record in records:
        if (type(record) is not tuple or len(record) != 2
                or type(record[0]) is not bytes or not 0 < len(record[0]) <= METADATA_LIMIT
                or type(record[1]) is not bytes or not 0 < len(record[1]) <= PACKET_LIMIT):
            raise ValueError("FuseFSS material batch record differs")
        total += 8 + len(record[0]) + len(record[1])
        if total > MATERIAL_BATCH_RECORD_BYTES:
            raise ValueError("FuseFSS material batch bytes exceed bound")
    return total


def encode_material_batch(records):
    """Original immutable records; no queuing/coalescing of singleton delivery."""
    material_batch_size(records)
    return encode_length_prefixed(
        (FRAME_HEADER.pack(len(meta)) + meta + packet for meta, packet in records),
        max_count=MATERIAL_BATCH_RECORDS, max_total_bytes=MATERIAL_BATCH_RECORD_BYTES)


def decode_material_batch(raw, count):
    """Validate complete framing before callers can append any of its records."""
    if (type(raw) is not bytes or type(count) is not int
            or not 1 <= count <= MATERIAL_BATCH_RECORDS):
        raise ValueError("FuseFSS material batch count differs")
    records = []
    for frame in iter_length_prefixed(raw, max_count=count, max_total_bytes=MATERIAL_BATCH_RECORD_BYTES):
        if len(frame) < 6:
            raise ValueError("FuseFSS material batch record truncated")
        size, = FRAME_HEADER.unpack_from(frame)
        if not 0 < size <= METADATA_LIMIT or not 0 < len(frame) - 4 - size <= PACKET_LIMIT:
            raise ValueError("FuseFSS material batch record differs")
        records.append((frame[4:4 + size], frame[4 + size:]))
    if len(records) != count:
        raise ValueError("FuseFSS material batch count differs")
    return tuple(records)


def matching_peer_metadata(incoming, expected):
    own, other = metadata(expected), metadata(incoming)
    if (type(own.get("role")) is not int or own["role"] not in (0, 1)
            or type(other.get("role")) is not int or other["role"] != 1 - own["role"]):
        raise ValueError("FuseFSS peer role differs")
    other["role"] = own["role"]
    if canonical(other) != expected:
        raise ValueError("FuseFSS peer kernel, context or frame differs")


def peer_bytes():
    return {role: {f"{phase}_{direction}_bytes": 0
                   for phase in ("setup", "online")
                   for direction in ("upload", "download")}
            for role in ROLES}


def collect_peer_bytes(value, role):
    """Initiator-only counters count each directed request and ACK exactly once."""
    if type(value) is not dict or set(value) != set(ROLES) or role not in ROLES:
        raise ValueError("FuseFSS peer accounting differs")
    result = dict.fromkeys(PEER_COUNTERS, 0)
    keys = {f"{phase}_{direction}_bytes" for phase in ("setup", "online")
            for direction in ("upload", "download")}
    for peer, counters in value.items():
        if (type(counters) is not dict or set(counters) != keys
                or any(type(v) is not int or not 0 <= v < 1 << 63 for v in counters.values())
                or (peer == role and any(counters.values()))
                or (role in HOLDERS and peer == "dealer" and any(counters.values()))
                or (role == "dealer" and any(counters[f"online_{d}_bytes"]
                                             for d in ("upload", "download")))):
            raise ValueError("FuseFSS invalid directed peer accounting")
        if peer != role:
            for phase in ("setup", "online"):
                result[f"peer_link.{role}.{peer}.{phase}_bytes"] += counters[f"{phase}_upload_bytes"]
                result[f"peer_link.{peer}.{role}.{phase}_bytes"] += counters[f"{phase}_download_bytes"]
    return result


__all__ = ["HOLDERS", "ROLES", "TRANSPORT", "ADMISSION_LIMIT", "CONTROL_LIMIT",
           "METADATA_LIMIT", "PACKET_LIMIT", "FRAME_LIMIT", "CLIENT_MEMORY", "ROLE_MEMORY",
           "PEER_COUNTERS", "canonical", "metadata", "matching_peer_metadata", "peer_bytes",
           "collect_peer_bytes", "clear_failure_frames", "digest", "pack", "unpack",
           "material_delivery", "material_batch_size", "encode_material_batch", "decode_material_batch",
           "MATERIAL_BATCH_RECORDS", "MATERIAL_BATCH_BYTES", "MATERIAL_BATCH_ENVELOPE_BYTES",
           "MATERIAL_BATCH_RECORD_BYTES", "MATERIAL_BATCH_PYTHON_BYTES"]
