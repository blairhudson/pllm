"""Public envelopes shared by the replicated client and authenticated workers."""

ROLES = ("worker_a", "worker_b", "worker_c")
FRAME_LIMIT = 128 << 10
ADMISSION_LIMIT = 1 << 20
REQUEST_MEMORY = 192 << 20
PEER_COUNTERS = frozenset(
    f"peer_link.{source}.{destination}.{phase}_bytes"
    for source in ROLES for destination in ROLES if source != destination
    for phase in ("setup", "online")
)


def input_frame(value):
    """Encode the native ordered-input header after bounded outer validation."""
    if (set(value) != {"key", "offset", "frame"} or type(value["key"]) is not str
            or type(value["offset"]) is not int or not 0 <= value["offset"] <= 1 << 20
            or type(value["frame"]) is not bytes or not 0 < len(value["frame"]) <= 64 << 10
            or len(value["frame"]) % 16):
        raise ValueError("invalid replicated source frame")
    key = value["key"].encode("utf-8")
    if not 0 < len(key) <= 512:
        raise ValueError("invalid replicated source key")
    return (len(key).to_bytes(2, "little") + key + value["offset"].to_bytes(4, "little")
            + (len(value["frame"]) // 16).to_bytes(4, "little") + value["frame"])
