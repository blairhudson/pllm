#!/usr/bin/env python3
"""Bounded public-head certificate scanner. Real sources run only in guarded workers.

No downloads, SDK changes, builds, serving measurements, or private data logging.
Raw mode requires an independently authenticated pin; checkpoint mode requires an
independently retained checkpoint lock and uses the ordinary head-only importer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_ROWS = 262144
MAX_WEIGHTS = 256 << 20
MAX_CHUNK = 4 << 20
REPLY_HEADER = struct.Struct("<8sBB32s32s")
SETUP_HEADER = struct.Struct("<8sIIQ32s32s")
QUANTIZER = "symmetric-per-row-i8-qmax127-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def geometry(rows, hidden, chunk_rows):
    if (any(type(x) is not int for x in (rows, hidden, chunk_rows))
            or not 1 <= rows <= MAX_ROWS or not 1 <= hidden <= 65532
            or rows * hidden > MAX_WEIGHTS or hidden * 128 * 127 >= 1 << 31
            or not 1 <= chunk_rows <= 256 or min(rows, chunk_rows) * hidden > MAX_CHUNK):
        raise ValueError("public scan geometry exceeds bounded domain")


def validate_pin(pin):
    keys = {"source_lock_sha256", "checkpoint_sha256", "model_config_sha256",
            "descriptor_sha256", "weight_sha256", "scales_sha256", "vocabulary",
            "hidden", "qmax", "weight_dtype", "activation_quantizer"}
    if type(pin) is not dict or set(pin) != keys:
        raise ValueError("complete independent source pin required")
    geometry(pin["vocabulary"], pin["hidden"], 1)
    if (pin["qmax"] != 127 or type(pin["qmax"]) is not int
            or pin["weight_dtype"] != "i8" or pin["activation_quantizer"] != QUANTIZER):
        raise ValueError("signed-i8/symmetric-input contract differs")
    for key in keys - {"vocabulary", "hidden", "qmax", "weight_dtype", "activation_quantizer"}:
        value = pin[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid source digest")


def scan(read_rows, read_scales, pin, *, chunk_rows=64, native_widths=None):
    """Two full bounded passes; independent scalar second pass verifies every row.

    Readers return only requested public rows. Hashes of BOTH passes must equal
    independent pin. No full matrix or scales copy is required by this scanner.
    Bounds are tight on the declared input box: choose x_j=127*sign(w_j).
    """
    validate_pin(pin)
    rows, hidden = pin["vocabulary"], pin["hidden"]
    geometry(rows, hidden, chunk_rows)
    widths = bytearray(rows)
    bounds = bytearray(rows * 8)
    ranges = []
    hashes = []
    for verify in (False, True):
        wh, sh = hashlib.sha256(), hashlib.sha256()
        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            w = np.asarray(read_rows(start, stop))
            s = np.asarray(read_scales(start, stop))
            if w.dtype != np.int8 or w.shape != (stop - start, hidden):
                raise ValueError("actual weight dtype/shape differs")
            if s.dtype != np.dtype("<f4") or s.shape != (stop - start,):
                raise ValueError("actual scale dtype/shape differs")
            if np.any(s <= 0) or not np.all(np.isfinite(s)):
                raise ValueError("invalid public scales")
            raw = w.tobytes(order="C")
            wh.update(raw)
            sh.update(s.astype("<f4", copy=False).tobytes())
            if not verify:
                l1 = np.abs(w.astype(np.int16)).sum(axis=1, dtype=np.uint64)
                for index, value in enumerate(l1, start):
                    bound = 127 * int(value)
                    widths[index] = max(1, bound.bit_length() + 1)
                    struct.pack_into("<Q", bounds, index * 8, bound)
            else:
                # Python integers and explicit two's-complement decoding: no
                # shared np.abs/i8 overflow or reduction implementation.
                oracle = bytearray()
                for index in range(stop - start):
                    row = raw[index * hidden:(index + 1) * hidden]
                    l1 = sum(abs(x if x < 128 else x - 256) for x in row)
                    bound = 127 * l1
                    width = max(1, (2 * bound).bit_length())
                    target = start + index
                    if (width != widths[target]
                            or bound != struct.unpack_from("<Q", bounds, target * 8)[0]):
                        raise ValueError("independent per-row certificate differs")
                    oracle.append(width)
                if native_widths is not None and native_widths(raw, hidden, 127) != bytes(oracle):
                    raise ValueError("installed native certificate differs")
            if not verify:
                ranges.append((float(s.min()), float(s.max())))
        hashes.append((wh.hexdigest(), sh.hexdigest()))
    expected = (pin["weight_sha256"], pin["scales_sha256"])
    if any(pair != expected for pair in hashes):
        raise ValueError("actual weights/scales differ from independent pin")
    metadata = canonical({"schema": "pllm.compact-private-head-certificate.v1", **pin,
        "bound": "127*sum(abs(signed-i8-row))", "input_box": [-127, 127],
        "bias": False, "rows_per_head": 1, "width_sha256": digest(widths),
        "bounds_sha256": digest(bounds), "decode": "add-low-residues-then-signed-center",
        "zero_rows_use_native_one_bit": True})
    return {"metadata": metadata, "widths": bytes(widths), "bounds": bytes(bounds),
            "public_scale_min": min(x[0] for x in ranges),
            "public_scale_max": max(x[1] for x in ranges),
            "native_checked": native_widths is not None}


def layout_encodings(widths):
    """All encodings public, lossless; byte counts include their identifying tag."""
    if not widths or any(not 1 <= x <= 32 for x in widths):
        raise ValueError("invalid native widths")
    rle = bytearray(b"R")
    start = 0
    while start < len(widths):
        stop = start + 1
        while stop < len(widths) and widths[stop] == widths[start]:
            stop += 1
        rle.extend(struct.pack("<BI", widths[start], stop - start))
        start = stop
    buffer = used = 0
    packed = bytearray(b"5")
    for width in widths:
        buffer |= (width - 1) << used
        used += 5
        while used >= 8:
            packed.append(buffer & 255)
            buffer >>= 8
            used -= 8
    if used:
        packed.append(buffer)
    candidates = {"raw": b"B" + widths, "five_bit": bytes(packed),
                  "rle_u32": bytes(rle), "zlib9": b"Z" + zlib.compress(widths, 9)}
    if len(set(widths)) == 1:
        candidates["uniform"] = b"U" + widths[:1]
    return candidates


def decode_layout(encoded, rows):
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS or not encoded:
        raise ValueError("invalid layout admission")
    tag, payload = encoded[:1], encoded[1:]
    if tag == b"B":
        result = payload
    elif tag == b"U" and len(payload) == 1:
        result = payload * rows
    elif tag == b"Z":
        decoder = zlib.decompressobj()
        result = decoder.decompress(payload, rows + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("compressed layout exceeds exact admission")
    elif tag == b"R" and len(payload) % 5 == 0:
        result = bytearray()
        for offset in range(0, len(payload), 5):
            width, count = struct.unpack_from("<BI", payload, offset)
            if count == 0 or len(result) + count > rows:
                raise ValueError("invalid run length")
            result.extend(bytes([width]) * count)
        result = bytes(result)
    elif tag == b"5" and len(payload) == (rows * 5 + 7) // 8:
        result = bytes(((int.from_bytes(payload[(i * 5) // 8:(i * 5) // 8 + 2], "little")
                         >> ((i * 5) % 8)) & 31) + 1 for i in range(rows))
        if rows * 5 % 8 and payload[-1] >> (rows * 5 % 8):
            raise ValueError("noncanonical layout padding")
    else:
        raise ValueError("invalid layout encoding")
    if len(result) != rows or any(not 1 <= x <= 32 for x in result):
        raise ValueError("expanded layout differs")
    return result


def costs(cert, hidden, *, calls=8, outer_body_bytes=142, http_bytes=137):
    """Installed baseline overhead 74+142+137=353 per role; not a wire trace.

    142/137 observed in pinned pacing cohort. Counterfactual keeps these fixed;
    different msgpack length prefixes, HTTP/TLS/TCP segmentation remain unmeasured.
    """
    if (type(calls) is not int or not 1 <= calls <= 4096
            or any(type(x) is not int or x < 0 for x in (outer_body_bytes, http_bytes))):
        raise ValueError("invalid cost admission")
    widths, metadata = cert["widths"], cert["metadata"]
    rows = len(widths)
    uniform_width = (2 * hidden * 128 * 127).bit_length()
    base_payload = (rows * uniform_width + 7) // 8
    payload = (sum(widths) + 7) // 8
    # Candidate binds a layout hash in addition to existing role/context/checksum.
    overhead = REPLY_HEADER.size + 32 + outer_body_bytes + http_bytes
    baseline_framed = 2 * (base_payload + REPLY_HEADER.size + outer_body_bytes + http_bytes)
    framed = 2 * (payload + overhead)
    saved = baseline_framed - framed
    encodings = layout_encodings(widths)
    for value in encodings.values():
        if decode_layout(value, rows) != widths:
            raise AssertionError("public layout roundtrip differs")
    codec = min(encodings, key=lambda name: (len(encodings[name]), name))
    meta_encodings = {"raw": b"B" + metadata, "zlib9": b"Z" + zlib.compress(metadata, 9)}
    meta_codec = min(meta_encodings, key=lambda name: (len(meta_encodings[name]), name))
    setup = SETUP_HEADER.size + len(meta_encodings[meta_codec]) + len(encodings[codec])
    # All three recipients receive full certificate. Each envelope additionally
    # uses supplied body/HTTP allowance; actual deployment must trace these.
    cold = 3 * (setup + outer_body_bytes + http_bytes)
    head_source = rows * (hidden + 4)
    extra_source = rows * hidden  # Client already owns row scales.
    local_saving = baseline_framed + 2 * hidden * 4
    return {"vocabulary": rows, "hidden": hidden, "head_calls": calls,
        "width_histogram": dict(sorted(Counter(widths).items())),
        "width_bit_sum_per_role": sum(widths), "zero_bound_rows": sum(
            struct.unpack_from("<Q", cert["bounds"], i * 8)[0] == 0 for i in range(rows)),
        "baseline_uniform_width": uniform_width,
        "baseline_reply_payload_bytes_both": 2 * base_payload,
        "reply_payload_bytes_per_role": payload, "reply_payload_bytes_both": 2 * payload,
        "candidate_reply_context_layout_checksum_bytes_per_role": REPLY_HEADER.size + 32,
        "outer_body_overhead_bytes_per_role_assumption": outer_body_bytes,
        "http_overhead_bytes_per_role_assumption": http_bytes,
        "baseline_reply_framed_bytes_both_estimate": baseline_framed,
        "reply_framed_bytes_both_estimate": framed,
        "saved_reply_bytes_both_per_head_estimate": saved,
        "layout_encoding_bytes": {name: len(value) for name, value in encodings.items()},
        "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        "selected_layout_encoding": codec, "metadata_bytes": len(metadata),
        "metadata_encoding_bytes": {name: len(value) for name, value in meta_encodings.items()},
        "selected_metadata_encoding": meta_codec,
        "setup_envelope_header_bytes": SETUP_HEADER.size,
        "cold_layout_metadata_delivery_bytes_three_recipients_estimate": cold,
        "cold_break_even_heads_vs_installed_uniform_estimate": math.ceil(cold / saved) if saved > 0 else None,
        "net_saved_bytes_at_head_calls_estimate": calls * saved - cold,
        "worker_upload_40Mbps_saved_seconds_per_head_floor_difference": saved / 2 * 8 / 40e6,
        "client_download_100Mbps_saved_seconds_per_head_floor_difference": saved * 8 / 100e6,
        "unchanged_request_u32_payload_bytes_both": 2 * hidden * 4,
        "client_width_bounds_metadata_retained_payload_bytes": rows * 9 + len(metadata),
        "client_public_weight_verification_delivery_bytes_if_not_preinstalled": head_source,
        "client_public_weight_verification_cold_break_even_heads_estimate":
            math.ceil((head_source + cold) / saved) if saved > 0 else None,
        "client_codec_arrays_floor_bytes": 4 * payload + 25 * rows + 2 * overhead,
        "worker_codec_arrays_floor_bytes": 9 * rows + 2 * payload + overhead,
        "local_full_head_alternative": {
            "cold_public_weights_and_scales_bytes": head_source,
            "incremental_cold_weights_bytes_when_scales_already_owned": extra_source,
            "cold_header_and_http_allowance_bytes": outer_body_bytes + http_bytes,
            "additional_stored_weight_bytes": rows * hidden,
            "new_head_protocol_messages_and_one_use_material_bytes": 0,
            "removed_head_request_and_reply_bytes_per_call_estimate": local_saving,
            "cold_break_even_heads_estimate": math.ceil((extra_source + outer_body_bytes + http_bytes) / local_saving),
            "integer_macs_per_head": rows * hidden, "f32_casts_per_head": rows,
            "f32_multiplies_per_head": 2 * rows,
            "full_f32_logits_bytes": 4 * rows,
            "chunked_int64_head_scratch_payload_allowance_bytes": 8 * min(64, rows) * hidden + 8 * hidden + 32 * min(64, rows),
            "top_p_sampler_payload_array_conservative_allowance_bytes": 128 * rows,
            "tier_admitted": False, "whole_request_improvement_measured": False},
        "scope": "counterfactual bodies/framing; TLS, packets, setup auth, allocator and timing unmeasured"}


def local_head(read_rows, read_scales, rows, hidden, query, activation_scale, *, chunk_rows=64):
    """Complete local private full-vocabulary head, not private-data server selection.

    Public head must be fully downloaded independently of query. Exact integers
    followed by original separate float32 cast/mul/mul. Public fixed scan order.
    """
    geometry(rows, hidden, chunk_rows)
    x = np.asarray(query)
    a = np.float32(activation_scale)
    if (x.dtype != np.int8 or x.shape != (hidden,) or np.any(x < -127)
            or not np.isfinite(a) or a <= 0):
        raise ValueError("symmetric signed-i8 query/finite scale required")
    output = np.empty(rows, np.float32)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        w = read_rows(start, stop)
        s = read_scales(start, stop)
        if (w.dtype != np.int8 or w.shape != (stop - start, hidden)
                or s.dtype != np.float32 or s.shape != (stop - start,)
                or not np.all(np.isfinite(s)) or np.any(s <= 0)):
            raise ValueError("local public source differs")
        dots = w.astype(np.int64) @ x.astype(np.int64)
        output[start:stop] = (dots.astype(np.float32) * a) * s
    if not np.all(np.isfinite(output)):
        raise ValueError("local dequantization nonfinite")
    return output


def private_selection_gate(widths):
    """Reject this explicit half-gates composition before unpriced float work.

    Ripple modular adder uses 1+2*(b-2) AND gates, then a free final XOR.
    b=1 is XOR only. This is NOT a lower bound on all secure selection designs.
    """
    add_ands = sum(max(0, 2 * b - 3) for b in widths)
    return {"construction": "two-boundary Yao/free-XOR/128-bit-half-gates",
        "modular_reconstruction_adder_AND_gates": add_ands,
        "adder_only_garbled_table_bytes": 32 * add_ands,
        "garbler_input_selected_label_bytes": 16 * sum(widths),
        "evaluator_input_OT_count": sum(widths),
        "reusable_garbled_tables": False,
        "exact_float32_dequantization_circuit_available": False,
        "exact_numpy_float64_exp_argsort_choice_RNG_circuit_available": False,
        "full_vocabulary_sampler_gate": "FAIL: no bit-exact pinned sampler circuit; adder traffic alone exceeds compact replies",
        "greedy_gate": "FAIL for exact f32 contract: missing cast/mul/mul/first-tie circuit and full costs",
        "unpriced_required_costs": ["OT base/setup/extension, fresh randomness and all input labels",
            "cast and two IEEE float32 multiplies per row; secret activation scale input",
            "float compare and first-index tie logic; full-vocab selection",
            "float64 division/exp/reductions, exact NumPy sort and choice for top-p",
            "client-private RNG input and updated state output; output labels to client",
            "malicious consistency if required, authentication, framing, all peer links",
            "garbler/evaluator CPU, streamed memory, cancellation/one-use retirement"],
        "security": "half-gates semi-honest assumptions do not establish malicious-worker security"}


def source_locks():
    paths = [Path(__file__), ROOT / "scripts/frontier_probe_support.py"]
    from pllm import _native
    import pllm.runtime.private_boundary_loader as loader
    import pllm.runtime.private_boundary_head_codec as codec
    import pllm.runtime.quantization as quantization
    import pllm.runtime.transformer_client as client
    import pllm.runtime.transformer_engine as engine
    import pllm.model_loader as model_loader
    import numpy.random._generator as generator
    import numpy.random._pcg64 as pcg64
    paths += [Path(m.__file__) for m in (_native, loader, codec, quantization, client,
                                       engine, model_loader, np, np._core._multiarray_umath,
                                       generator, pcg64)]
    paths += [ROOT / "crates/pllm-core/src/offset_transport.rs"]
    return {str(p): file_digest(p) for p in paths}


def worker(payload):
    from pllm import _native
    before = source_locks()
    chunk_rows = payload.get("chunk_rows", 64)
    if payload.get("experiment"):
        from pllm import Experiment
        from pllm.model_loader import resolve_model
        from pllm.runtime.private_boundary_loader import load_private_boundary, build_private_boundary_contract
        lock = json.loads(Path(payload["checkpoint_lock"]).read_text())
        if set(lock.get("sha256", {})) != {"config.json", "model.safetensors"}:
            raise ValueError("checkpoint lock must independently pin config and single safetensors checkpoint")
        experiment = Experiment.from_spec(json.loads(Path(payload["experiment"]).read_text()))
        resolved = resolve_model(experiment.pipeline.model)
        for name, expected in lock["sha256"].items():
            if file_digest(resolved.path / name) != expected:
                raise ValueError("original checkpoint differs from retained source lock")
        config = json.loads((resolved.path / "config.json").read_text())
        config = config.get("text_config", config)
        geometry(config["vocab_size"], config["hidden_size"], chunk_rows)
        importer_allowance = 4 * config["vocab_size"] * config["hidden_size"] + (512 << 20)
        if importer_allowance > payload["admission_mib"] << 20:
            raise ValueError("modeled importer allowance exceeds reserved admission")
        loaded = load_private_boundary(experiment)
        descriptor, _ = build_private_boundary_contract(loaded, experiment)
        # View borrows sole importer owner; scanner makes only bounded chunks.
        weights = loaded.head.compiled_weight.weight_view()
        scales = loaded.head.weight_scales
        pin = {"source_lock_sha256": descriptor.source_digest,
               "checkpoint_sha256": descriptor.checkpoint_digest,
               "model_config_sha256": descriptor.model_config_digest,
               "descriptor_sha256": descriptor.digest,
               "weight_sha256": descriptor.head_weight_digest,
               "scales_sha256": descriptor.head_scales_digest,
               "vocabulary": descriptor.vocabulary, "hidden": descriptor.hidden,
               "qmax": 127, "weight_dtype": "i8", "activation_quantizer": QUANTIZER}
        origin = {"kind": "ordinary pinned checkpoint/head-only importer",
                  "checkpoint_lock_sha256": file_digest(payload["checkpoint_lock"]),
                  "experiment_sha256": file_digest(payload["experiment"]),
                  "descriptor": descriptor.to_dict() if hasattr(descriptor, "to_dict") else descriptor.digest}
    else:
        manifest = json.loads(Path(payload["raw_manifest"]).read_text())
        if set(manifest) != {"pin", "weights", "scales"}:
            raise ValueError("raw manifest requires pin, weights, scales only")
        pin = manifest["pin"]
        validate_pin(pin)
        rows, hidden = pin["vocabulary"], pin["hidden"]
        geometry(rows, hidden, chunk_rows)
        if rows * hidden + 64 * rows + (128 << 20) > payload["admission_mib"] << 20:
            raise ValueError("modeled mapped-source allowance exceeds reserved admission")
        for key, size in (("weights", rows * hidden), ("scales", rows * 4)):
            if Path(manifest[key]).stat().st_size != size:
                raise ValueError("raw public artifact length differs")
        weights = np.memmap(manifest["weights"], mode="r", dtype=np.int8, shape=(rows, hidden))
        scales = np.memmap(manifest["scales"], mode="r", dtype="<f4", shape=(rows,))
        origin = {"kind": "externally authenticated raw public head pin",
                  "manifest_sha256": file_digest(payload["raw_manifest"]),
                  "provenance_warning": "self-declared manifest is not source authentication"}
    cert = scan(lambda a, b: weights[a:b], lambda a, b: scales[a:b], pin,
                chunk_rows=chunk_rows, native_widths=_native.offset_row_bits)
    if source_locks() != before:
        raise RuntimeError("scanner/importer/native source changed during scan")
    rows, hidden = pin["vocabulary"], pin["hidden"]
    return {"schema": "pllm.compact-private-head-scan.v1", "source": origin,
        "source_locks": before, "pin": pin, "certificate": json.loads(cert["metadata"]),
        "certificate_sha256": digest(cert["metadata"]),
        "widths_hex": cert["widths"].hex(), "bounds_le_u64_hex": cert["bounds"].hex(),
        "public_scale_range": [cert["public_scale_min"], cert["public_scale_max"]],
        "verification": {"full_passes": 2, "every_row_scalar_verified": True,
                         "installed_native_every_row_verified": cert["native_checked"]},
        "costs": costs(cert, hidden, calls=payload.get("head_calls", 8)),
        "private_selection": private_selection_gate(cert["widths"]),
        "resources": {"admitted_bytes": payload["admission_mib"] << 20,
            "chunk_rows": chunk_rows, "public_weight_bytes": rows * hidden,
            "scanner_numeric_scratch_upper_payload_bytes": 8 * min(rows, chunk_rows) * hidden + rows * 12,
            "certificate_scan_weight_coordinate_visits": 2 * rows * hidden,
            "source_import_additional_owners": "ordinary importer, checkpoint buffers/native compile; admitted separately",
            "unmeasured": "whole RSS/CPU/delivery; report hex/JSON and allocator add storage; no serving claim"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--raw-manifest", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--checkpoint-lock", type=str)
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--head-calls", type=int, default=8)
    parser.add_argument("--admission-mib", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False))
        return
    if (not args.output or bool(args.raw_manifest) == bool(args.experiment)
            or bool(args.checkpoint_lock) != bool(args.experiment)):
        parser.error("require --output and either --raw-manifest or --experiment with --checkpoint-lock")
    if not 128 <= args.admission_mib <= 8192 or not 1 <= args.chunk_rows <= 256 or not 1 <= args.head_calls <= 4096:
        parser.error("bounded admission/chunk/head count exceeded")
    # Reserve output before doing expensive work; failed attempts remain explicit.
    with args.output.open("x") as output:
        output.write(json.dumps({"status": "incomplete", "reason": "scan not yet accepted"}) + "\n")
    from frontier_probe_support import fresh_workers
    with fresh_workers(Path(__file__).resolve(), memory_bytes=args.admission_mib << 20,
                       timeout=240) as (run, guard, host):
        report = run(vars(args) | {"output": str(args.output)})
        report["installed_build"] = run.build
        report["host_admission"] = {"reserve_bytes": host.reserve, "guard_error": guard.error}
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
