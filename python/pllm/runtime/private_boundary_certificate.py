"""Offline publisher certificates; authenticity is the caller-approved byte pin.

This is a trusted publisher assertion, not a cryptographic proof of quantization.
Never obtain a pin from the worker supplying the artifact. No source matrix is
downloaded by the importer. Workers independently check the claimed layout.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import zlib
from collections.abc import Mapping
from contextlib import ExitStack

from .private_boundary_protocol import BoundaryError, canonical, digest, integer, strict_object

SCHEMA = "pllm.private_boundary.row_certificate.v1"
MAX_BYTES = (1 << 20)
HEADER = struct.Struct("<8sI")
MAGIC = b"PLLMBRC1"
IDENTITY = ("source_digest", "checkpoint_digest", "model_config_digest",
            "head_weight_digest", "head_scales_digest", "lookup_digest",
            "vocabulary", "hidden", "activation_bits", "weight_bits", "signed_output_bound")
NUMERIC = "signed-i8-row-major-bias-free/symmetric-qmax127/int-wrap32/f32-separate-multiplies/v1"


def _metadata(raw):
    # Schema has exactly three shallow objects and no arrays. Bound parser
    # containers before allocation, including authenticated malformed artifacts.
    depth = objects = 0
    quoted = escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 93):
            raise BoundaryError("row certificate metadata arrays are not admitted")
        elif byte == 123:
            depth += 1
            objects += 1
            if depth > 2 or objects > 3:
                raise BoundaryError("row certificate metadata nesting exceeds bound")
        elif byte == 125:
            depth -= 1
    return json.loads(raw, object_pairs_hook=strict_object)


def validate_pin(pin):
    if not isinstance(pin, Mapping) or set(pin) != {"path", "digest", "size_bytes"}:
        raise BoundaryError("row certificate requires explicit caller-approved artifact pin")
    if type(pin["path"]) is not str or not 1 <= len(pin["path"]) <= 4096 or not Path(pin["path"]).is_absolute():
        raise BoundaryError("row certificate requires absolute local artifact path")
    digest(pin["digest"], "certificate digest")
    integer(pin["size_bytes"], HEADER.size + 1, MAX_BYTES, "certificate size")
    return dict(pin)


def source_identity(descriptor):
    if descriptor.activation_bits != 8 or descriptor.weight_bits != 8:
        raise BoundaryError("source-certified head requires W8A8")
    return {name: getattr(descriptor, name) for name in IDENTITY}


def matrix_layout(weights, descriptor):
    """Bounded scan of immutable native rows; includes -128 and zero rows."""
    import numpy as np
    if weights.dtype != np.int8 or weights.shape != (descriptor.vocabulary, descriptor.hidden):
        raise BoundaryError("certificate matrix geometry differs")
    widths = bytearray()
    bounds = hashlib.sha256()
    # <= 1 MiB row chunks, no second full matrix owner.
    chunk_rows = min(64, max(1, (1 << 20) // (2 * descriptor.hidden)))
    for start in range(0, len(weights), chunk_rows):
        chunk = weights[start:start + chunk_rows].astype(np.int16)
        for value in np.sum(np.abs(chunk), axis=1, dtype=np.int64):
            bound = int(value) * 127
            if bound >= 1 << 31 or bound > descriptor.signed_output_bound:
                raise BoundaryError("certificate row exceeds signed bound")
            bounds.update(struct.pack("<I", bound))
            widths.append(max(1, bound.bit_length() + 1))
    return bytes(widths), bounds.hexdigest()


def publisher_implementation():
    """Retain implementation provenance under the independently trusted pin."""
    from pllm import _native
    from . import private_boundary_loader, quantization
    modules = (_native, private_boundary_loader, quantization)
    paths = {m.__name__: m.__file__ for m in modules} | {__name__: __file__}
    result = {}
    for name, path in paths.items():
        with Path(path).open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def publication_memory(source, spec, store):
    """Head-only incremental allocation price from metadata, never tensor values.

    Full source/F32 head allowances deliberately dominate current streamed input.
    No decoder weights, KV, inventory or fake placement/planning record is used.
    """
    vocabulary = integer(spec.out_features, 1, 262144, "publication vocabulary")
    hidden = integer(spec.in_features, 1, 65532, "publication hidden")
    if len(spec.weight_keys) != 1 or spec.bias_keys or spec.transpose_weight:
        raise BoundaryError("publication requires one canonical bias-free row-major head")
    key = store.resolve(spec.weight_keys[0])
    if store.tensor_shape(key) != (vocabulary, hidden):
        raise BoundaryError("publication source head shape differs")
    dtype = store.tensor_dtype(key)
    if dtype not in {"F32", "F16", "BF16"}:
        raise BoundaryError("publication source head dtype unsupported")
    if store.contains(key.removesuffix(".weight") + ".bias"):
        raise BoundaryError("publication source head contains undeclared bias")
    elements = vocabulary * hidden
    if elements + 4 * vocabulary > 512 << 20:
        raise BoundaryError("publication source head exceeds native capacity")
    tokenizer = 0
    for name in ("pllm_tokenizer.json", "tokenizer_config.json", "tokenizer.model", "tokenizer.json"):
        path = source.path / name
        if path.exists():
            size = path.stat().st_size
            if not path.is_file() or not 0 <= size <= 64 << 20:
                raise BoundaryError("publication tokenizer exceeds bounded source capacity")
            tokenizer += size
    chunk = min(vocabulary, 64, max(1, (1 << 20) // hidden)) * hidden
    costs = {
        "source_head_bytes": elements * (4 if dtype == "F32" else 2),
        "source_f32_head_allowance_bytes": 4 * elements,
        "quantized_and_native_copy_bytes": 4 * elements,
        "scales_and_row_metadata_bytes": 32 * vocabulary,
        "source_quantizer_chunk_work_bytes": 32 * chunk,
        "tokenizer_work_bytes": 32 * tokenizer,
        "row_scan_and_artifact_work_bytes": (2 << 20) + 8 * vocabulary + 4 * MAX_BYTES,
        "importer_metadata_and_native_runtime_bytes": 128 << 20,
    }
    return {**costs, "incremental_memory_bytes": sum(costs.values())}


def publish_from_source(experiment, path):
    """Keep live same-process capacity reserved before the first tensor read."""
    from pllm.deployment.resources import reserve_incremental_memory
    from .private_boundary_loader import load_private_boundary

    boundary = experiment.pipeline.components.get("boundary")
    if boundary is not None and boundary.params.get("head_output_encoding") == "row_residues":
        raise BoundaryError("publish using a source recipe without a certificate")
    with ExitStack() as owners:
        def admit(source, spec, store):
            price = publication_memory(source, spec, store)
            owners.enter_context(reserve_incremental_memory(price["incremental_memory_bytes"]))

        loaded = load_private_boundary(experiment, _admit_import=admit)
        try:
            return publish(loaded, experiment, path)
        finally:
            del loaded


def publish(loaded, experiment, path):
    """Publish from ordinary locked head-only import, using a legacy head recipe.

    Return a pin for independent publisher approval/distribution. Publication does
    not itself authenticate that pin to another party. Source import is charged
    separately; this call scans owned native weights, never checkpoint tensors.
    """
    from .private_boundary_loader import build_private_boundary_contract
    descriptor, _ = build_private_boundary_contract(loaded, experiment)
    if descriptor.head_output_encoding == "row_residues":
        raise BoundaryError("publish using a source recipe without a certificate")
    widths, bounds = matrix_layout(loaded.head.compiled_weight.weight_view(), descriptor)
    metadata = canonical({"schema": SCHEMA, "source": source_identity(descriptor),
        "numeric": NUMERIC, "layout": "zlib-u8-minimal-signed-widths-v1",
        "widths_digest": hashlib.sha256(widths).hexdigest(), "bounds_digest": bounds,
        "implementation": publisher_implementation()})
    payload = HEADER.pack(MAGIC, len(metadata)) + metadata + zlib.compress(widths, 9)
    if len(metadata) > 16384 or len(payload) > MAX_BYTES:
        raise BoundaryError("certificate publication exceeds capacity")
    path = Path(path).resolve()
    pin = {"path": str(path), "digest": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    # Link a complete, fsynced sibling into place without overwriting an existing
    # artifact. An interrupted/failed write cannot leave a truncated public file.
    fd, temporary = tempfile.mkstemp(prefix=".pllm-head-certificate-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if stream.write(payload) != len(payload):
                raise OSError("incomplete row certificate publication")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return pin


def import_certificate(pin, descriptor, *, weights=None, with_metadata=False):
    """Authenticate once; optionally return validated metadata from that parse.

    Metadata adds local ``artifact_digest`` and ``artifact_size_bytes`` from the
    authenticated pin, not from publisher assertions. Default remains width bytes.
    """
    if type(with_metadata) is not bool:
        raise BoundaryError("invalid certificate metadata selection")
    pin = validate_pin(pin)
    if descriptor.head_certificate_digest != pin["digest"]:
        raise BoundaryError("certificate pin differs from compiled source schedule")
    try:
        with Path(pin["path"]).open("rb") as stream:
            payload = stream.read(pin["size_bytes"] + 1)
    except OSError as exc:
        raise BoundaryError("row certificate unavailable") from exc
    if len(payload) != pin["size_bytes"] or hashlib.sha256(payload).hexdigest() != pin["digest"]:
        raise BoundaryError("row certificate authentication failed")
    magic, size = HEADER.unpack_from(payload)
    if magic != MAGIC or not 1 <= size <= 16384 or HEADER.size + size >= len(payload):
        raise BoundaryError("invalid row certificate envelope")
    try:
        metadata = _metadata(payload[HEADER.size:HEADER.size + size])
    except (ValueError, UnicodeError) as exc:
        raise BoundaryError("invalid row certificate metadata") from exc
    if (type(metadata) is not dict or set(metadata) != {"schema", "source", "numeric", "layout",
            "widths_digest", "bounds_digest", "implementation"}
            or metadata["schema"] != SCHEMA or metadata["numeric"] != NUMERIC
            or metadata["layout"] != "zlib-u8-minimal-signed-widths-v1"
            or canonical(metadata["source"]) != canonical(source_identity(descriptor))):
        raise BoundaryError("row certificate source or numeric contract differs")
    for key in ("widths_digest", "bounds_digest"):
        digest(metadata[key], key)
    implementation = metadata["implementation"]
    if type(implementation) is not dict or set(implementation) != {
        "pllm._native", "pllm.runtime.private_boundary_loader", "pllm.runtime.quantization",
        "pllm.runtime.private_boundary_certificate"}:
        raise BoundaryError("row certificate implementation provenance absent")
    for value in implementation.values():
        digest(value, "publisher implementation digest")
    decoder = zlib.decompressobj()
    try:
        widths = decoder.decompress(payload[HEADER.size + size:], descriptor.vocabulary + 1)
    except zlib.error as exc:
        raise BoundaryError("invalid compressed row layout") from exc
    if (len(widths) != descriptor.vocabulary or not decoder.eof or decoder.unused_data
            or decoder.unconsumed_tail or any(not 1 <= w <= 32 for w in widths)
            or hashlib.sha256(widths).hexdigest() != metadata["widths_digest"]):
        raise BoundaryError("row certificate layout differs or exceeds bound")
    # Geometry is an additional independent ceiling, never authentication.
    ceiling = (2 * descriptor.hidden * 128 * 127).bit_length()
    if max(widths) > ceiling:
        raise BoundaryError("row certificate width exceeds geometry")
    if weights is not None:
        actual, bounds = matrix_layout(weights, descriptor)
        if actual != widths or bounds != metadata["bounds_digest"]:
            raise BoundaryError("certificate differs from actual immutable native rows")
    if with_metadata:
        return widths, {**metadata, "artifact_digest": pin["digest"],
                        "artifact_size_bytes": pin["size_bytes"]}
    return widths
