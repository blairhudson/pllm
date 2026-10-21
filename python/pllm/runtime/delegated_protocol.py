"""Bounded outer framing for native verified delegation."""

import re
import json
import traceback

import msgpack

ROLES = ("worker_a", "worker_b")
FRAME_BYTES = 8192
PUBLICATION_BYTES = 512 << 10
ADMISSION_BYTES = PUBLICATION_BYTES + (32 << 10)
DECODER_FRAME_BYTES = 4 << 20
DECODER_PUBLICATION_BYTES = 32 << 20
DECODER_ADMISSION_BYTES = DECODER_PUBLICATION_BYTES + DECODER_FRAME_BYTES + (32 << 10)
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def clear_failure_frames(error):
    """Release inactive payload frames before their admission owner retires."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(item for item in (current.__cause__, current.__context__)
                       if item is not None)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def limits(capacity):
    if capacity == "capsule":
        return FRAME_BYTES, ADMISSION_BYTES
    if capacity == "decoder":
        return DECODER_FRAME_BYTES, DECODER_ADMISSION_BYTES
    raise ValueError("GKM invalid transport capacity")


def pack(value, *, limit=ADMISSION_BYTES):
    raw = msgpack.packb(value, use_bin_type=True)
    if len(raw) > limit:
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
                           max_str_len=1024, max_bin_len=limit,
                           max_array_len=32, max_map_len=32, max_ext_len=0)
    if type(value) is not dict:
        raise ValueError("GKM outer frame must be an object")
    return value


def digest(value):
    if type(value) is not str or DIGEST.fullmatch(value) is None:
        raise ValueError("GKM invalid public digest")
    return value


def unpack_admission(raw, capacity="capsule"):
    """Scan the complete shallow envelope before allocating decoded values.

    Only eleven bounded keys/spans are visited. Opaque native BIN payloads are
    skipped by offset, never copied or inspected here. Legal nonminimal headers
    remain accepted; this is admission, not a canonical wire encoding.
    """
    frame_limit, admission_limit = limits(capacity)
    if type(raw) is not bytes or len(raw) > admission_limit:
        raise ValueError("GKM outer frame exceeds admission bound")
    offset = 0

    def number(width):
        nonlocal offset
        if width > len(raw) - offset:
            raise ValueError("GKM truncated admission header")
        value = 0
        for _ in range(width):
            value = (value << 8) | raw[offset]
            offset += 1
        return value

    def string_length(code):
        if 0xA0 <= code <= 0xBF:
            return code & 31
        if code in (0xD9, 0xDA, 0xDB):
            return number({0xD9: 1, 0xDA: 2, 0xDB: 4}[code])
        raise ValueError("GKM admission requires STR")

    code = number(1)
    if 0x80 <= code <= 0x8F:
        count = code & 15
    elif code in (0xDE, 0xDF):
        count = number(2 if code == 0xDE else 4)
    else:
        raise ValueError("GKM admission requires root map")
    fields = (b"schema", b"request", b"epoch", b"configuration", b"source_lock",
              b"tokenizer", b"input_rows", b"output_tokens", b"publication", b"start", b"pin")
    if count != len(fields):
        raise ValueError("GKM invalid admission fields")
    seen = 0
    publication_limit = PUBLICATION_BYTES if capacity == "capsule" else DECODER_PUBLICATION_BYTES
    for _ in fields:
        size = string_length(number(1))
        if size > 13 or size > len(raw) - offset:
            raise ValueError("GKM invalid admission key span")
        key = raw[offset:offset + size]
        offset += size
        try:
            index = fields.index(key)
        except ValueError:
            raise ValueError("GKM unknown admission key") from None
        bit = 1 << index
        if seen & bit:
            raise ValueError("GKM duplicate admission key")
        seen |= bit
        code = number(1)
        # Classify before reading any container/ext length or children.
        if index <= 5:
            size = string_length(code)
            expected = len(b"pllm.gkm.admit.v1") if index == 0 else 64
            if size != expected:
                raise ValueError("GKM invalid admission identity length")
        elif index <= 7:
            if code <= 0x7F or code >= 0xE0:
                size = 0
            elif code in (0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1, 0xD2, 0xD3):
                size = {0xCC: 1, 0xCD: 2, 0xCE: 4, 0xCF: 8,
                        0xD0: 1, 0xD1: 2, 0xD2: 4, 0xD3: 8}[code]
            else:
                raise ValueError("GKM admission counter requires integer")
        else:
            if code not in (0xC4, 0xC5, 0xC6):
                raise ValueError("GKM admission requires BIN")
            size = number({0xC4: 1, 0xC5: 2, 0xC6: 4}[code])
            maximum = publication_limit if index == 8 else frame_limit if index == 9 else 32
            if not 0 < size <= maximum or index == 10 and size != 32:
                raise ValueError("GKM invalid admission binary length")
        if size > len(raw) - offset:
            raise ValueError("GKM truncated admission value")
        if index == 0 and raw[offset:offset + size] != b"pllm.gkm.admit.v1":
            raise ValueError("GKM invalid admission schema")
        offset += size
    if offset != len(raw):
        raise ValueError("GKM admission has trailing content")
    return unpack(raw, limit=admission_limit)


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
