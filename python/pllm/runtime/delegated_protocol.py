"""Bounded outer framing for native, full-share verified delegation."""

import re
import json

import msgpack

ROLES = ("worker_a", "worker_b")
FRAME_BYTES = 8192
PUBLICATION_BYTES = 512 << 10
ADMISSION_BYTES = PUBLICATION_BYTES + (32 << 10)
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def pack(value):
    raw = msgpack.packb(value, use_bin_type=True)
    if len(raw) > ADMISSION_BYTES:
        raise ValueError("GKM outer frame exceeds admission bound")
    return raw


def unpack(raw, *, limit=ADMISSION_BYTES):
    if type(raw) is not bytes or len(raw) > limit:
        raise ValueError("GKM outer frame exceeds admission bound")

    def pairs(items):
        result = {}
        for key, value in items:
            if type(key) is not str or key in result:
                raise ValueError("GKM outer frame has invalid or duplicate keys")
            result[key] = value
        return result

    value = msgpack.unpackb(raw, raw=False, strict_map_key=True, object_pairs_hook=pairs,
                           max_str_len=1024, max_bin_len=PUBLICATION_BYTES,
                           max_array_len=32, max_map_len=32, max_ext_len=0)
    if type(value) is not dict:
        raise ValueError("GKM outer frame must be an object")
    return value


def digest(value):
    if type(value) is not str or DIGEST.fullmatch(value) is None:
        raise ValueError("GKM invalid public digest")
    return value


def read_json(raw):
    """Bounded native completion receipts; preserve bytes for digest authority."""
    if type(raw) is not bytes or not 0 < len(raw) <= 16 << 10:
        raise ValueError("GKM invalid receipt length")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("GKM duplicate receipt key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    if type(value) is not dict or len(value) > 32:
        raise ValueError("GKM invalid receipt object")
    return value
