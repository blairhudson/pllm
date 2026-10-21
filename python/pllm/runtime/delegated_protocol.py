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
BATCH_EXECUTION = "source_stage_batch_v1"
BATCH_HTTP_CHUNK_BYTES = 64 << 10
BATCH_HTTP_SERVER_BODY_BYTES = 2 * BATCH_HTTP_CHUNK_BYTES
BATCH_HTTP_CLIENT_BACKEND_BYTES = 8 * BATCH_HTTP_CHUNK_BYTES
BATCH_HTTP_WORKER_BACKEND_BYTES = 16 * BATCH_HTTP_CHUNK_BYTES
BATCH_HTTP_BACKEND = "pllm.gkm.asyncio-h11-bounded.v1"


def batch_contract(runtime):
    """Validate trusted native metadata, never infer lengths from peer frames.

    ``descriptors`` is the compiler handshake seam. Missing exact native
    per-role lengths fail closed. A/R/S are
    native owners, not a substitute for Python copy or runtime backing prices.
    """
    batch = runtime.get("batch_execution")
    if batch is None:
        return None
    if (type(batch) is not dict or batch.get("execution") != BATCH_EXECUTION
            or batch.get("schema_version") != "pllm.gkm.source-stage-batch-contract.v1"):
        raise ValueError("GKM invalid batch execution metadata")
    digest(batch.get("batch_plan"))
    resources = batch.get("resources")
    names = ("max_aggregate_bytes", "retained_frame_bytes", "transport_scratch_bytes",
             "extra_native_owner_bytes", "extra_wire_bytes")
    if (type(resources) is not dict or any(type(resources.get(k)) is not int
            or resources[k] < 0 for k in names) or resources["max_aggregate_bytes"] <= 0):
        raise ValueError("GKM missing native batch resource quote")
    descriptors = batch.get("descriptors")
    if type(descriptors) is not list or not descriptors:
        raise ValueError("GKM batch needs trusted per-descriptor role lengths")
    claim = 0
    previous = (-1, -1)
    for batch_index, item in enumerate(descriptors):
        if type(item) is not dict:
            raise ValueError("GKM invalid batch descriptor")
        for name in ("batch_index", "phase_index", "operation_index", "first_claim", "rows", "columns", "output_width"):
            if type(item.get(name)) is not int or item[name] < 0:
                raise ValueError("GKM invalid batch descriptor integer")
        boundary = (item["phase_index"], item["operation_index"])
        if (item["batch_index"] != batch_index or item["first_claim"] != claim or item["rows"] == 0 or boundary <= previous
                or item.get("kind") not in ("lookup", "linear")
                or item["output_width"] == 0
                or item["kind"] == "lookup" and (item["columns"] != 0 or item.get("stage") is not None)
                or item["kind"] == "linear" and (item["columns"] == 0 or type(item.get("stage")) is not int or item["stage"] < 0)):
            raise ValueError("GKM invalid batch descriptor schedule")
        for name in ("request_bytes", "reply_bytes"):
            lengths = item.get(name)
            if (type(lengths) not in (list, tuple) or len(lengths) != 2
                    or any(type(size) is not int or not 0 < size <= resources["max_aggregate_bytes"]
                           for size in lengths)):
                raise ValueError("GKM invalid batch role frame lengths")
        claim += item["rows"]
        previous = boundary
    # Compiler encoding overlaps the core's independently owned i64 input copy.
    # Neither serialized Python f32 input nor graph/dtype owners fund this Vec.
    encoded = max((8 * item["rows"] * item["columns"] for item in descriptors
                   if item["kind"] == "linear"), default=0)
    if (type(batch.get("compiler_encoded_input_peak_bytes")) is not int
            or batch["compiler_encoded_input_peak_bytes"] != encoded):
        raise ValueError("GKM batch compiler input quote differs")
    return batch


def batch_copy_prices(runtime):
    """Price distinct Python copies and the new runtime serialization owner.

    Existing decoder_tensor_bytes prices complete grouped graph roots, alias
    backing and dtype coercion: its eight-byte element allowance covers float32
    reconstruction plus biased-member copies. No additional persistent O is
    charged. C-order serialized I remains live in the runtime caller through the
    HTTP/accept call; reserve max(I) separately, without assuming a scalar-row
    serialization allowance can be subtracted. Both request frames survive
    collection; neither survives into acceptance. Role-one collection also
    retains role-zero's returned bytes. The binding's lookup Vec<usize> (<=8B)
    overlaps frame creation and is not a core/native R/S owner.
    """
    batch = batch_contract(runtime)
    if batch is None:
        return {}
    client = 0
    workers = [0, 0]
    backing = binding_tokens = 0
    for item in batch["descriptors"]:
        i, o = batch_input_bytes(item), batch_output_bytes(item)
        q0, q1 = item["request_bytes"]
        d0, d1 = item["reply_bytes"]
        tokens = 8 * item["rows"] if item["kind"] == "lookup" else 0
        # Plaintext HTTP/1 network read, h11 conversion and socket buffers are
        # additional owners. The batch backend pins selector reads and streams
        # uploads in 64 KiB views with zero-high-water write draining. No whole-Q
        # socket write is issued. Eight chunks price these backend copies apart
        # from the collector's yielded chunk, native scratch and process slack.
        # TLS is unsupported: handshake/OpenSSL/BIO allocation is not proved.
        backend = BATCH_HTTP_CLIENT_BACKEND_BYTES
        # Keep this descriptor's backend ownership in its phase maximum, not
        # accumulated once per descriptor across the request.
        descriptor_peak = max(q0 + q1 + tokens,
                              q0 + q1 + 2 * d0 + BATCH_HTTP_CHUNK_BYTES,
                              q0 + q1 + d0 + 2 * d1 + BATCH_HTTP_CHUNK_BYTES,
                              d0 + d1 + o) + backend
        client = max(client, descriptor_peak)
        for role, (q, d) in enumerate(zip((q0, q1), (d0, d1), strict=True)):
            # The pinned H11 server pauses at 64 KiB and admits at most one
            # further 64 KiB socket read. Count backend queue/conversion buffers
            # separately, including up to four bounded hosting connections even
            # when their endpoint rejects busy/unauthorized work. The operation
            # slot, not this hosting cap, bounds live D. Send may copy all of D.
            workers[role] = max(workers[role], max(2 * q + min(q, BATCH_HTTP_SERVER_BODY_BYTES),
                                                  q + d, 2 * d) + BATCH_HTTP_WORKER_BACKEND_BYTES)
        backing = max(backing, i)
        binding_tokens = max(binding_tokens, tokens)
    return {"batch_python_client_peak_bytes": client,
            "batch_python_worker_peak_bytes": dict(zip(ROLES, workers, strict=True)),
            "batch_runtime_backing_delta_bytes": backing,
            "batch_binding_token_peak_bytes": binding_tokens}


def batch_input_bytes(descriptor):
    return 4 * descriptor["rows"] * descriptor["columns"]


def batch_output_bytes(descriptor):
    return 4 * descriptor["rows"] * descriptor["output_width"]


def batch_memory_bytes(runtime, capacity):
    """One shared source/watchdog/request price; original payload caps apply."""
    prices = batch_copy_prices(runtime)
    if not prices:
        return {}
    batch, decoder = runtime["batch_execution"], runtime["decoder"]
    delta = batch["resources"]["extra_native_owner_bytes"] + batch["resources"]["extra_wire_bytes"]
    encoded = batch["compiler_encoded_input_peak_bytes"]
    scalar_client = decoder["execution_payload_estimate_bytes"]
    scalar_worker = (decoder["decoder_resources"]["worker_payload_bytes"]
                     if capacity == "decoder" else scalar_client)
    if (type(batch.get("client_payload_bytes")) is not int
            or type(batch.get("worker_payload_bytes")) is not int
            or batch["client_payload_bytes"] != scalar_client + delta + encoded
            or batch["worker_payload_bytes"] != scalar_worker + delta):
        raise ValueError("GKM batch native payload quote differs")
    client = batch["client_payload_bytes"] + prices["batch_python_client_peak_bytes"] + prices["batch_runtime_backing_delta_bytes"]
    workers = {role: batch["worker_payload_bytes"] + size
               for role, size in prices["batch_python_worker_peak_bytes"].items()}
    cap = 4 << 30 if capacity == "decoder" else 64 << 20
    if any(size > cap for size in (client, *workers.values())):
        raise ValueError("GKM batch Python/backing owners exceed execution payload capacity")
    return {**prices, "compiler_encoded_input_peak_bytes": encoded,
            "client_payload_bytes": client, "worker_payload_bytes": workers}


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
