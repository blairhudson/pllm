"""Bounded raw-stage accounting diagnostic; no model, provider or crypto setup.

Do not compare masked arithmetic values across different quantizers. Compare
their public geometry, wire representation and payload lengths, preserving exact
whole-frame accounting while identifying only server_ns value-width separately.
"""
from __future__ import annotations

import hashlib
import math


def account_response(payload, *, stage_id):
    import msgpack
    from pllm.runtime.stage_protocol import MaskedStageResponse, PreparedStageBatchResponse

    if len(payload) > 1 << 20:
        raise ValueError("diagnostic response exceeds 1 MiB bound")
    record = msgpack.unpackb(payload, raw=False, max_map_len=16, max_array_len=2, max_ext_len=0)
    if type(record) is not dict or type(record.get("server_ns")) is not int or not 0 <= record["server_ns"] < 1 << 63:
        raise ValueError("raw stage response/timing schema differs")
    shape = record.get("shape")
    if type(shape) is not list or not 1 <= len(shape) <= 2 or any(type(n) is not int or not 1 <= n <= 256 for n in shape) or math.prod(shape) > 65536:
        raise ValueError("diagnostic geometry exceeds bounded stage shape")
    if "batch_id" in record:
        decoded = PreparedStageBatchResponse.unpack(payload, max_rows=256, max_tensor_elements=65536)
        schema = "prepared-batch"
    else:
        decoded = MaskedStageResponse.unpack(payload)
        schema = "masked-stage"
        if decoded.stage_id != stage_id:
            raise ValueError("stage ID differs")
    data = record["data"]
    if type(data) is not bytes or len(data) != math.prod(shape) * (decoded.wire_bits // 8):
        raise ValueError("arithmetic byte count differs from public shape/wire width")
    clock_bytes = len(msgpack.packb(record["server_ns"], use_bin_type=True))
    normalized = dict(record, server_ns=0)
    normalized_bytes = len(msgpack.packb(normalized, use_bin_type=True))
    if len(payload) != normalized_bytes + clock_bytes - 1:
        raise ValueError("response has additional noncanonical framing differences")
    return {
        "payload_sha256": hashlib.sha256(payload).hexdigest(), "full_frame_bytes": len(payload),
        "server_ns": record["server_ns"], "server_ns_value_bytes": clock_bytes,
        "static_contract": {"stage_id": stage_id, "schema": schema, "wire_bits": decoded.wire_bits,
                            "shape": shape, "arithmetic_payload_bytes": len(data),
                            "ring": record.get("ring"), "modulus": record.get("modulus"),
                            "frame_bytes_with_zero_server_ns": normalized_bytes},
    }


def assert_equal_static_contract(left, right):
    if len(left) != len(right) or not left:
        raise AssertionError("stage response counts differ or are empty")
    for first, second in zip(left, right, strict=True):
        if first["static_contract"] != second["static_contract"]:
            raise AssertionError("public arithmetic representation or non-timing framing differs")
    full_delta = sum(x["full_frame_bytes"] for x in left) - sum(x["full_frame_bytes"] for x in right)
    timing_delta = sum(x["server_ns_value_bytes"] for x in left) - sum(x["server_ns_value_bytes"] for x in right)
    if full_delta != timing_delta:
        raise AssertionError("whole-frame delta is not explained exactly by timing value widths")
