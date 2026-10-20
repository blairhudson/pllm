"""Public envelopes for the offline-dealer, two-holder FSS transaction."""
from __future__ import annotations

import json

from .delegated_protocol import clear_failure_frames, digest, pack, unpack

HOLDERS = ("worker_a", "worker_b")
ROLES = (*HOLDERS, "dealer")
TRANSPORT = "fss-offline-tape-v1"
ADMISSION_LIMIT = 2 << 20
CONTROL_LIMIT = 16 << 10
METADATA_LIMIT = 4096
PACKET_LIMIT = 16 << 20
FRAME_LIMIT = PACKET_LIMIT + (16 << 10)
CLIENT_MEMORY = 64 << 20
ROLE_MEMORY = 256 << 20
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
           "collect_peer_bytes", "clear_failure_frames", "digest", "pack", "unpack"]
