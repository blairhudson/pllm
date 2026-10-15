#!/usr/bin/env python3
"""Bounded, synthetic-only full-head residue experiment; not a runtime protocol.

Run with the installed SDK interpreter. Public fixture bytes are reproducible;
private additive masks deliberately are not. No downloads or model loading.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import secrets
import struct
import time

import numpy as np
from pllm import _native
from pllm.runtime.quantization import dequantize_matmul

ROOT = Path(__file__).resolve().parents[1]
ENCODING = "u32-low-residues-lsb-first-v1"
DOMAIN = b"pllm/probe/private-boundary-head-residues/v1\0"
HEADER = struct.Struct("<8sBBH32s32sIIQQ")
MAX_WEIGHT_BYTES = 160 * 1024 * 1024
MAX_ROWS = 262144
L1_PROFILE = "minimal-row-l1-i8-v1"
UNIFORM_PROFILE = "uniform-public-geometry-i8-v1"
QUANTIZER = "symmetric-per-row-i8-qmax127-v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def geometry(vocabulary: int, hidden: int) -> None:
    # Admission precedes RNG, array creation, or native calls.
    if (type(vocabulary) is not int or type(hidden) is not int
            or not 1 <= vocabulary <= MAX_ROWS or not 1 <= hidden <= 65532
            or vocabulary * hidden > MAX_WEIGHT_BYTES
            or hidden * 128 * 127 >= 1 << 31):
        raise ValueError("head geometry exceeds static probe geometry limit")


def oracle_layout(weights: bytes, hidden: int, qmax: int = 127) -> tuple[bytes, bytes]:
    if (type(hidden) is not int or hidden <= 0 or type(qmax) is not int
            or qmax != 127 or not weights or len(weights) % hidden):
        raise ValueError("requires public W8A8 weights and qmax=127")
    geometry(len(weights) // hidden, hidden)
    widths, bounds = bytearray(), bytearray()
    for start in range(0, len(weights), hidden):
        bound = qmax * sum(abs(x if x < 128 else x - 256)
                           for x in weights[start:start + hidden])
        width = max(1, bound.bit_length() + 1)
        if width > 32:
            raise ValueError("signed dot exceeds wrap32")
        widths.append(width)
        bounds.extend(struct.pack("<Q", bound))
    return bytes(widths), bytes(bounds)


@dataclass(frozen=True)
class Certificate:
    metadata: bytes
    widths: bytes
    transmitted_layout: bytes | None = None

    @property
    def digest(self) -> bytes:
        layout = self.widths if self.transmitted_layout is None else self.transmitted_layout
        return hashlib.sha256(DOMAIN + self.metadata + layout).digest()


def validate_pin(pin: dict) -> None:
    """Validate shape of externally authenticated admission, not its provenance."""
    expected_keys = {"source_lock_sha256", "checkpoint_sha256", "model_config_sha256",
                     "descriptor_sha256", "weight_sha256", "scales_sha256",
                     "vocabulary", "hidden", "qmax", "weight_dtype", "activation_quantizer"}
    if type(pin) is not dict or set(pin) != expected_keys:
        raise ValueError("independent source pin required")
    geometry(pin["vocabulary"], pin["hidden"])
    if (pin["weight_dtype"] != "i8" or pin["activation_quantizer"] != QUANTIZER
            or type(pin["qmax"]) is not int or pin["qmax"] != 127):
        raise ValueError("trusted dtype/quantizer differs from W8A8 contract")
    for key in expected_keys - {"vocabulary", "hidden", "qmax", "weight_dtype", "activation_quantizer"}:
        value = pin[key]
        if (type(value) is not str or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("invalid independent digest")


def certificate(weights: bytes, scales: bytes, pin: dict) -> Certificate:
    """Minimal L1 profile: actual pinned weights are mandatory."""
    validate_pin(pin)
    if (len(weights) != pin["vocabulary"] * pin["hidden"]
            or sha(weights) != pin["weight_sha256"]
            or len(scales) != 4 * pin["vocabulary"]
            or sha(scales) != pin["scales_sha256"]):
        raise ValueError("actual weight/scales commitment differs")
    scale_values = np.frombuffer(scales, "<f4")
    if not np.all(np.isfinite(scale_values)) or np.any(scale_values <= 0):
        raise ValueError("invalid public scales")
    widths, bounds = oracle_layout(weights, pin["hidden"], pin["qmax"])
    native_widths = _native.offset_row_bits(weights, pin["hidden"], pin["qmax"])
    if native_widths != widths:
        raise AssertionError("native public bound differs from integer oracle")
    metadata = canonical({"schema": "pllm.probe.boundary-head-certificate.v1",
                          "profile": L1_PROFILE,
                          "encoding": ENCODING, "bound_method": "qmax-times-row-l1-i8-v1",
                          "bound_sha256": sha(bounds), "width_sha256": sha(widths),
                          "rows_per_call": 1, "bias": False, **pin})
    if len(metadata) > 4096:
        raise ValueError("certificate metadata exceeds limit")
    return Certificate(metadata, widths)


def verify_certificate(candidate: Certificate, weights: bytes, scales: bytes,
                       trusted_pin: dict) -> Certificate:
    # Never accept an unsigned, self-declared bound, digest, qmax, or source ID.
    expected = certificate(weights, scales, trusted_pin)
    if candidate != expected:
        raise ValueError("certificate/layout differs from recomputed pinned weights")
    return expected


def geometry_certificate(trusted_pin: dict) -> Certificate:
    """Distinct source-free-client path; assumes already-authenticated W8A8 pin.

    It authenticates no opaque pin itself and deliberately takes no weights.
    """
    validate_pin(trusted_pin)
    bound = trusted_pin["hidden"] * 128 * 127
    width = bound.bit_length() + 1
    layout = bytes([width])
    metadata = canonical({"schema": "pllm.probe.boundary-head-certificate.v1",
                          "profile": UNIFORM_PROFILE, "encoding": ENCODING,
                          "bound_method": "hidden-times-i8-abs128-times-qmax127-v1",
                          "uniform_bound": bound, "uniform_width": width,
                          "width_sha256": sha(layout), "rows_per_call": 1,
                          "bias": False, **trusted_pin})
    return Certificate(metadata, layout * trusted_pin["vocabulary"], layout)


def verify_geometry_certificate(candidate: Certificate, trusted_pin: dict) -> Certificate:
    expected = geometry_certificate(trusted_pin)
    if candidate != expected:
        raise ValueError("uniform profile differs from independently admitted public geometry")
    return expected


def pack_oracle(values: list[int], widths: bytes) -> bytes:
    result = bytearray()
    buffer = used = 0
    for value, width in zip(values, widths, strict=True):
        buffer |= (value & ((1 << width) - 1)) << used
        used += width
        while used >= 8:
            result.append(buffer & 255)
            buffer >>= 8
            used -= 8
    if used:
        result.append(buffer)
    return bytes(result)


def reconstruct_oracle(a: bytes, b: bytes, widths: bytes) -> list[int]:
    result, position = [], 0
    for width in widths:
        index, shift = divmod(position, 8)
        mask = (1 << width) - 1
        x = (int.from_bytes(a[index:index + 5], "little") >> shift) & mask
        y = (int.from_bytes(b[index:index + 5], "little") >> shift) & mask
        value = (x + y) & mask
        result.append(value - (1 << width) if value >= 1 << (width - 1) else value)
        position += width
    return result


def frame(payload: bytes, cert: Certificate, context: bytes, role: int,
          sequence: int = 0) -> bytes:
    if role not in (0, 1) or len(context) != 32 or not 0 <= sequence < 1 << 64:
        raise ValueError("invalid frame context")
    return HEADER.pack(b"PLLMHR01", 1, role, 0, cert.digest, context, 1,
                       len(cert.widths), sequence, len(payload)) + payload


def unframe(raw: bytes, cert: Certificate, context: bytes, role: int,
            sequence: int = 0) -> bytes:
    size = (sum(cert.widths) + 7) // 8
    if len(raw) != HEADER.size + size:
        raise ValueError("frame length differs")
    expected = HEADER.pack(b"PLLMHR01", 1, role, 0, cert.digest, context, 1,
                           len(cert.widths), sequence, size)
    if raw[:HEADER.size] != expected:
        raise ValueError("encoding/context/role/sequence/layout differs")
    payload = raw[HEADER.size:]
    bits = sum(cert.widths)
    if bits % 8 and payload[-1] >> (bits % 8):
        raise ValueError("nonzero terminal padding")
    return payload


def reconstruct_checked(a: bytes, b: bytes, cert: Certificate, bounds: bytes | None = None) -> bytes:
    """Use only a previously verified certificate and locally recomputed bounds."""
    metadata = json.loads(cert.metadata)
    if metadata["profile"] == UNIFORM_PROFILE:
        if bounds is not None:
            raise ValueError("uniform profile uses scalar public geometry bound")
        bound = metadata["hidden"] * 128 * 127
        if metadata["uniform_bound"] != bound:
            raise ValueError("uniform bound differs")
    elif metadata["profile"] == L1_PROFILE:
        if (bounds is None or len(bounds) != len(cert.widths) * 8
                or sha(bounds) != metadata["bound_sha256"]):
            raise ValueError("local public bounds differ from certificate")
        bound = np.frombuffer(bounds, "<u8")
    else:
        raise ValueError("unknown certificate profile")
    output = _native.offset_reconstruct_rows(a, b, cert.widths, 1)
    values = np.frombuffer(output, "<i8")
    if np.any(np.abs(values) > bound):
        raise ValueError("reconstructed head exceeds certified public bounds")
    return output


def costs(widths: bytes, certificate_bytes: int, hidden: int, calls: int,
          profile: str = L1_PROFILE) -> dict:
    v, bits = len(widths), sum(widths)
    packed = (bits + 7) // 8
    # Eight-byte setup length prefix, exact metadata and one byte per public row.
    uniform = profile == UNIFORM_PROFILE
    layout_bytes = 1 if uniform else v
    delivery = 8 + certificate_bytes + layout_bytes
    raw = 2 * v * 4
    body = 2 * (HEADER.size + packed)
    return {
        "profile": profile, "vocabulary": v, "hidden": hidden, "head_calls": calls,
        "raw_u32_two_party_payload_bytes_per_call": raw,
        "packed_bytes_per_party": packed, "packed_two_party_payload_bytes_per_call": 2 * packed,
        "terminal_padding_bits_per_party": packed * 8 - bits,
        "header_bytes_per_party": HEADER.size,
        "two_party_framed_reply_bytes_per_call": body,
        "public_layout_bytes_per_recipient": layout_bytes,
        "client_expanded_native_width_array_bytes": v,
        "certificate_metadata_bytes_per_recipient": certificate_bytes,
        "setup_length_prefix_bytes_per_recipient": 8,
        "delivery_recipients": ["boundary_a", "boundary_b", "client"],
        "cold_certificate_layout_delivery_bytes": 3 * delivery,
        "raw_payload_minus_new_frames_and_cold_delivery_bytes": calls * (raw - body) - 3 * delivery,
        "unchanged_two_party_u32_request_payload_bytes_per_call": 2 * hidden * 4,
        "client_packed_plus_i64_plus_f32_array_floor_bytes": 2 * packed + v * 12,
        "client_frame_plus_slice_copy_peak_floor_bytes": 4 * packed + 2 * HEADER.size + v * 12,
        "client_owned_arrays_with_f32_cast_and_two_multiply_outputs_bytes":
            4 * packed + 2 * HEADER.size + v * 20,
        "client_retained_public_bounds_bytes_if_postchecking": 0 if uniform else v * 8,
        "client_public_verification_weight_and_scales_bytes": 0 if uniform else v * (hidden + 4),
        "public_verification_bounds_and_widths_bytes": v if uniform else v * 9,
        "per_worker_raw_u32_plus_packed_array_floor_bytes": v * 4 + packed,
        "http_tls_tcp_and_python_allocator_bytes": "not measured; not included",
        "cold_delivery_break_even_calls_vs_headerless_raw":
            (3 * delivery + raw - body - 1) // (raw - body) if raw > body else None,
    }


def source_lock() -> dict:
    paths = [Path(__file__), ROOT / "crates/pllm-core/src/offset_transport.rs",
             ROOT / "python/pllm/runtime/private_boundary_transport.py",
             Path(dequantize_matmul.__code__.co_filename),
             Path(_native.__file__)]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            sha(p.read_bytes()) for p in paths}


def baseline_center(raw_a: bytes, raw_b: bytes) -> np.ndarray:
    a, b = (np.frombuffer(raw, "<u4").astype(np.int64) for raw in (raw_a, raw_b))
    combined = (a + b) % (1 << 32)
    return np.where(combined >= 1 << 31, combined - (1 << 32), combined)


def allocation_preflight(vocabulary: int, hidden: int, budget_mib: int) -> dict:
    """Conservative modeled envelope, NOT an OS free-memory/headroom query."""
    geometry(vocabulary, hidden)
    if type(budget_mib) is not int or not 64 <= budget_mib <= 4096:
        raise ValueError("modeled memory budget must be 64..4096 MiB")
    # Includes weights ndarray + bytes and conservative copy/allocator allowance;
    # 1024V prices Python scalar lists/tuples, two codec paths, frames, arrays,
    # oracle results, native outputs, bounds, scale fixtures and timing temporaries.
    envelope = 4 * vocabulary * hidden + 1024 * vocabulary + 32 * hidden + 64 * 1024**2
    if envelope > budget_mib * 1024**2:
        raise ValueError("modeled temporary envelope exceeds explicit memory budget")
    return {"modeled_envelope_bytes": envelope, "budget_bytes": budget_mib * 1024**2,
            "method": "4*V*H + 1024*V + 32*H + 64MiB; modeled allowance, not measured RSS",
            "available_host_headroom_queried": False,
            "generation_chunk_rows": 64,
            "generation_chunk_int16_plus_i8_max_bytes": 64 * hidden * 3,
            "dot_oracle_live_python_weight_rows": 1}


def run(vocabulary: int = 257, hidden: int = 32, calls: int = 8,
        repetitions: int = 3, oracle_mac_limit: int = 2_000_000,
        memory_budget_mib: int = 512) -> dict:
    geometry(vocabulary, hidden)
    if not 1 <= repetitions <= 100 or not 1 <= calls <= 4096:
        raise ValueError("repetitions/calls exceed guard")
    if vocabulary * hidden > oracle_mac_limit:
        raise ValueError("independent integer dot oracle exceeds MAC guard")
    preflight = allocation_preflight(vocabulary, hidden, memory_budget_mib)
    rng = np.random.default_rng(6103)
    weights_array = np.empty((vocabulary, hidden), dtype=np.int8)
    for start in range(0, vocabulary, 64):
        count = min(64, vocabulary - start)
        weights_array[start:start + count] = rng.integers(
            -128, 128, (count, hidden), dtype=np.int16).astype(np.int8)
    # Include zero, small, -128, and opposite-sign boundary rows.
    for row, value in enumerate((0, 1, -128, 127)):
        if row < vocabulary:
            weights_array[row].fill(value)
    weights = weights_array.tobytes()
    scales = rng.uniform(0.001, 0.02, vocabulary).astype("<f4").tobytes()
    pin = {name: sha(("synthetic-only/" + name).encode()) for name in (
        "source_lock_sha256", "checkpoint_sha256", "model_config_sha256", "descriptor_sha256")}
    pin.update(weight_sha256=sha(weights), scales_sha256=sha(scales),
               vocabulary=vocabulary, hidden=hidden, qmax=127,
               weight_dtype="i8", activation_quantizer=QUANTIZER)
    t = time.process_time_ns()
    cert = certificate(weights, scales, pin)
    verify_certificate(cert, weights, scales, pin)
    certificate_cpu = time.process_time_ns() - t
    t = time.process_time_ns()
    uniform = geometry_certificate(pin)
    verify_geometry_certificate(uniform, pin)  # No weights/scales argument or read.
    uniform_certificate_cpu = time.process_time_ns() - t
    q = [127] * hidden  # Public test fixture; never used to select widths.
    dots = [sum(int(w) * x for w, x in zip(row, q, strict=True)) for row in weights_array]
    # Fresh uniform wrap32 masks on every invocation, never seeded fixture masks.
    b = list(struct.unpack("<" + "I" * vocabulary, secrets.token_bytes(vocabulary * 4)))
    a = [(dot - mask) & 0xffffffff for dot, mask in zip(dots, b, strict=True)]
    raw_a, raw_b = struct.pack("<" + "I" * vocabulary, *a), struct.pack("<" + "I" * vocabulary, *b)
    packed_a = _native.offset_pack_rows(raw_a, cert.widths, 1)
    packed_b = _native.offset_pack_rows(raw_b, cert.widths, 1)
    assert packed_a == pack_oracle(a, cert.widths)
    assert packed_b == pack_oracle(b, cert.widths)
    context = hashlib.sha256(b"synthetic session/epoch/admission/head/sequence=0").digest()
    frames = [frame(p, cert, context, role) for role, p in enumerate((packed_a, packed_b))]
    unpacked = [unframe(f, cert, context, role) for role, f in enumerate(frames)]
    _, bounds = oracle_layout(weights, hidden)
    recovered = reconstruct_checked(*unpacked, cert, bounds)
    integers = np.frombuffer(recovered, "<i8").reshape(1, vocabulary)
    assert integers.tolist()[0] == dots == reconstruct_oracle(*unpacked, cert.widths)
    # Compare original wrap32 reconstruction, not only direct dots.
    original = np.array([((x + y + (1 << 31)) & 0xffffffff) - (1 << 31)
                         for x, y in zip(a, b, strict=True)], dtype=np.int64).reshape(1, -1)
    activation_scale = np.array([0.03125], dtype=np.float32)
    weight_scales = np.frombuffer(scales, "<f4")
    baseline = dequantize_matmul(original, activation_scale, weight_scales)
    candidate = dequantize_matmul(integers, activation_scale, weight_scales)
    assert baseline.tobytes() == candidate.tobytes()
    uniform_packed = [_native.offset_pack_rows(raw, uniform.widths, 1) for raw in (raw_a, raw_b)]
    assert uniform_packed == [pack_oracle(values, uniform.widths) for values in (a, b)]
    uniform_frames = [frame(p, uniform, context, r) for r, p in enumerate(uniform_packed)]
    uniform_unpacked = [unframe(f, uniform, context, r) for r, f in enumerate(uniform_frames)]
    uniform_output = reconstruct_checked(*uniform_unpacked, uniform)
    uniform_integers = np.frombuffer(uniform_output, "<i8").reshape(1, vocabulary)
    assert uniform_integers.tolist()[0] == dots == reconstruct_oracle(*uniform_unpacked, uniform.widths)
    assert baseline.tobytes() == dequantize_matmul(uniform_integers, activation_scale, weight_scales).tobytes()
    timings = {}
    for name, operation in (
        ("worker_a_pack", lambda: _native.offset_pack_rows(raw_a, cert.widths, 1)),
        ("worker_b_pack", lambda: _native.offset_pack_rows(raw_b, cert.widths, 1)),
        ("client_reconstruct", lambda: _native.offset_reconstruct_rows(packed_a, packed_b, cert.widths, 1)),
        ("client_reconstruct_with_public_bound_check", lambda: reconstruct_checked(
            packed_a, packed_b, cert, bounds)),
        ("client_dequantize", lambda: dequantize_matmul(integers, activation_scale, weight_scales)),
        ("client_frame_validation", lambda: [unframe(f, cert, context, r) for r, f in enumerate(frames)]),
        ("baseline_client_u32_center", lambda: baseline_center(raw_a, raw_b)),
        ("uniform_worker_a_pack", lambda: _native.offset_pack_rows(raw_a, uniform.widths, 1)),
        ("uniform_worker_b_pack", lambda: _native.offset_pack_rows(raw_b, uniform.widths, 1)),
        ("uniform_client_reconstruct_checked", lambda: reconstruct_checked(*uniform_packed, uniform)),
        ("uniform_client_frame_validation", lambda: [unframe(f, uniform, context, r)
                                                    for r, f in enumerate(uniform_frames)]),
    ):
        start = time.process_time_ns()
        for _ in range(repetitions):
            operation()
        timings[name + "_process_cpu_ns_per_call"] = (time.process_time_ns() - start) // repetitions
    return {
        "schema": "pllm.probe.boundary-head-residues.v1", "evidence": "synthetic-only",
        "source_lock": source_lock(), "pin": pin, "certificate": json.loads(cert.metadata),
        "certificate_sha256": cert.digest.hex(), "width_histogram": dict(sorted(Counter(cert.widths).items())),
        "correctness": {"native_vs_python_packing": True, "native_vs_integer_dot": True,
                        "negative_wrap": min(dots) < 0, "full_float32_logits_byte_equal": True,
                        "fresh_private_masks": True},
        "costs": costs(cert.widths, len(cert.metadata), hidden, calls),
        "uniform_geometry_profile": {
            "certificate": json.loads(uniform.metadata), "certificate_sha256": uniform.digest.hex(),
            "client_verification_reads_weight_bytes": False,
            "transmitted_width_hex": uniform.transmitted_layout.hex(),
            "native_vs_python_packing_and_integer_dot": True,
            "full_float32_logits_byte_equal": True,
            "costs": costs(uniform.widths, len(uniform.metadata), hidden, calls, UNIFORM_PROFILE)},
        "qwen_l1_payload_ceiling_only": costs(bytes([25]) * 151936, len(canonical({
            **json.loads(cert.metadata), "vocabulary": 151936, "hidden": 896})), 896, calls),
        "qwen_uniform_geometry_only": costs(bytes([25]) * 151936, len(geometry_certificate({
            **pin, "vocabulary": 151936, "hidden": 896}).metadata), 896, calls, UNIFORM_PROFILE),
        "qwen_certificate_size_note": "exact fixed-schema metadata length at Qwen geometry, all digests fixed length; not actual Qwen certificate or histogram",
        "cpu": {"certificate_build_and_second_recomputation_process_cpu_ns": certificate_cpu,
                "uniform_certificate_build_and_source_free_verification_process_cpu_ns": uniform_certificate_cpu,
                "repetitions": repetitions, **timings},
        "limits": {"weight_bytes": MAX_WEIGHT_BYTES, "oracle_macs": oracle_mac_limit,
                   "memory_preflight": preflight, "geometry_limit_is_host_headroom_check": False},
        "scope": "codec-only; no whole-response, security, RSS, network, or speed promotion",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=int, default=257)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--head-calls", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--oracle-mac-limit", type=int, default=2_000_000)
    parser.add_argument("--memory-budget-mib", type=int, default=512,
                        help="Modeled temporary-envelope budget, not available host RAM")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.vocabulary, args.hidden, args.head_calls, args.repetitions,
                 args.oracle_mac_limit, args.memory_budget_mib)
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
