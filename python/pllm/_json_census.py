"""Constant-state lexical refusal, extracted from SourceReadPolicy.

No decoder or package imports. This is deliberately NOT a JSON grammar or a
parse-memory quote. A readonly view prevents writes through that view, not writes
through aliases: callers must own mutable backing exclusively during this call.
"""


# Borrow the builtin neutral exception; no helper-owned exception class graph.
# Fixed reason in args[0]; never contains input bytes or scalar contents.
JSONCensusError = ValueError


def json_census(raw, *, byte_limit=16_777_216, node_limit=65_536,
                depth_limit=17, token_limit=1024, ascii_only=True):
    """Return (bytes, nodes, strings, tokens, containers, depth, longest).

    Counts occurrences, including duplicate keys. Escapes only affect quote
    tracking. Final depth, unfinished strings, bracket kinds, numeric validity,
    duplicates and all other grammar remain decoder responsibilities. None for
    token_limit and False for ascii_only retain the source SDK's legacy census.
    No callbacks, copies, arbitrary iterables or borrowed mutable-view writes.
    """
    if (type(byte_limit) is not int or not 0 < byte_limit <= 16_777_216
            or type(node_limit) is not int or not 0 < node_limit <= 65_536
            or type(depth_limit) is not int or not 0 < depth_limit <= 17
            or (token_limit is not None and
                (type(token_limit) is not int or not 0 < token_limit <= 1024))
            or type(ascii_only) is not bool):
        raise JSONCensusError("policy")
    if type(raw) is not bytes:
        if type(raw) is not memoryview:
            raise JSONCensusError("bytes")
        try:
            supported = (raw.readonly and raw.ndim == 1 and raw.c_contiguous
                         and raw.format == "B" and raw.itemsize == 1
                         and type(raw.obj) in (bytes, bytearray))
        except ValueError:
            raise JSONCensusError("bytes") from None
        if not supported:
            raise JSONCensusError("bytes")
    if not 0 < len(raw) <= byte_limit:
        raise JSONCensusError("bytes")
    nodes = depth = maximum = strings = tokens = containers = 0
    token_length = longest = 0
    quoted = escaped = token = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
            continue
        if ascii_only and byte >= 128:
            raise JSONCensusError("ascii")
        if byte in (32, 9, 10, 13, 44, 58):
            token = False
            token_length = 0
            continue
        if byte in (93, 125):
            depth -= 1
            if depth < 0:
                raise JSONCensusError("nesting")
            token = False
            token_length = 0
            continue
        if byte in (91, 123):
            depth += 1
            maximum = max(maximum, depth)
            containers += 1
            nodes += 1
            token = False
            token_length = 0
        elif byte == 34:
            strings += 1
            nodes += 1
            quoted = True
            token = False
            token_length = 0
        else:
            if not token:
                tokens += 1
                nodes += 1
                token = True
            # Preserve first byte, minus, decimal/exponent and malformed tokens.
            token_length += 1
            longest = max(longest, token_length)
            if token_limit is not None and token_length > token_limit:
                raise JSONCensusError("scalar")
        if nodes > node_limit or depth > depth_limit:
            raise JSONCensusError("structure")
    return (len(raw), nodes, strings, tokens, containers, maximum, longest)
